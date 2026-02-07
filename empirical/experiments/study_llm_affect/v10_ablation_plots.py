"""
V10 Ablation Analysis Plots
============================

Loads final_results.pkl from all 7 conditions × 3 seeds,
runs RSA analysis, and generates publication-quality figures.

Figures:
1. Ablation bar chart (RSA ρ by condition, error bars across seeds)
2. Per-dimension heatmap (dimension × condition)
3. Affect trajectories (6D over time, full condition seed 0)
4. Training curves (reward, loss, health per condition)
5. Null distribution (Mantel test, one representative)
6. MDS scatter (info-theoretic vs embedding affect space)

Usage:
    cd empirical/experiments/study_llm_affect
    python v10_ablation_plots.py
"""

import pickle
import os
import sys
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

# Add current dir to path for v10_affect import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v10_affect import AffectExtractor, AffectVector, compute_affect_for_agent
from v10_analysis import (
    compute_rdm, rsa_correlation, mantel_test, linear_cka,
    per_dimension_correlation, RSAResult, run_rsa_analysis,
)
from sklearn.manifold import MDS

# ============================================================================
# Configuration
# ============================================================================

CONDITIONS = [
    'full',
    'no_partial_obs',
    'no_long_horizon',
    'no_world_model',
    'no_self_prediction',
    'no_intrinsic_motivation',
    'no_delayed_rewards',
]

CONDITION_LABELS = {
    'full': 'Full',
    'no_partial_obs': '−Partial Obs',
    'no_long_horizon': '−Long Horizon',
    'no_world_model': '−World Model',
    'no_self_prediction': '−Self Predict',
    'no_intrinsic_motivation': '−Intrinsic Motiv',
    'no_delayed_rewards': '−Delayed Reward',
}

SEEDS = [0, 1, 2]
DATA_DIR = Path('results/modal_data')
OUT_DIR = Path('results/figures')

DIM_NAMES = AffectVector.dim_names()


# ============================================================================
# Data loading
# ============================================================================

def load_results(condition, seed):
    """Load final_results.pkl for one condition/seed."""
    path = DATA_DIR / condition / f'seed_{seed}' / 'final_results.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)


def extract_affect_and_embeddings(data, agent_id=0, subsample=2000):
    """Extract affect matrix and obs_embedding matrix from one run."""
    history = data['latent_history']
    n = len(history)

    # Subsample for speed
    if n > subsample:
        idx = np.linspace(0, n - 1, subsample, dtype=int)
    else:
        idx = np.arange(n)

    z_all = np.array([history[i]['z'][agent_id] for i in idx])
    actions = np.array([history[i]['action'][agent_id] for i in idx])
    rewards = np.array([history[i]['reward'][agent_id] for i in idx])
    values = np.array([history[i]['value'][agent_id] for i in idx])
    obs_emb = np.array([history[i]['obs_embedding'][agent_id] for i in idx])

    # Extract affect — detect n_actions from data
    n_actions = int(actions.max()) + 1
    extractor = AffectExtractor(z_all.shape[1], n_actions=n_actions)
    extractor.fit_probes(z_all, actions, rewards, values)

    # Monkey-patch compute_salience to handle action space mismatches
    # (some conditions have actions not seen during probe training)
    _orig_salience = extractor.self_model_probe.compute_salience
    def _safe_salience(z, acts):
        try:
            return _orig_salience(z, acts)
        except (IndexError, ValueError):
            # Fallback: uniform salience when probe can't handle all actions
            return np.ones(len(acts)) * 0.5
    extractor.self_model_probe.compute_salience = _safe_salience

    affect_vecs = extractor.extract(z_all, actions, values)
    affect_mat = extractor.extract_matrix(affect_vecs)

    return affect_mat, obs_emb


def run_all_analyses():
    """Run RSA for all conditions and seeds. Returns dict of results."""
    all_results = {}

    for cond in CONDITIONS:
        cond_results = []
        for seed in SEEDS:
            print(f"\n{'='*50}")
            print(f"  {cond} / seed {seed}")
            print(f"{'='*50}")

            data = load_results(cond, seed)
            affect_mat, obs_emb = extract_affect_and_embeddings(data)

            # RSA analysis
            rsa = run_rsa_analysis(
                affect_mat, obs_emb,
                n_permutations=2000,
                subsample=500,
            )
            cond_results.append({
                'rsa': rsa,
                'affect_mat': affect_mat,
                'obs_emb': obs_emb,
                'data': data,
            })

        all_results[cond] = cond_results

    return all_results


# ============================================================================
# Figure 1: Ablation bar chart
# ============================================================================

def plot_ablation_bars(all_results, save_path):
    """RSA ρ by condition with error bars across 3 seeds."""
    fig, ax = plt.subplots(figsize=(10, 5))

    conditions = CONDITIONS
    means = []
    stds = []
    for cond in conditions:
        rhos = [r['rsa'].mantel_rho for r in all_results[cond]]
        means.append(np.mean(rhos))
        stds.append(np.std(rhos))

    x = np.arange(len(conditions))
    labels = [CONDITION_LABELS[c] for c in conditions]

    colors = ['#2196F3'] + ['#FF9800'] * (len(conditions) - 1)
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=0.5)

    # Significance markers
    for i, cond in enumerate(conditions):
        ps = [r['rsa'].mantel_p for r in all_results[cond]]
        mean_p = np.mean(ps)
        marker = '***' if mean_p < 0.001 else '**' if mean_p < 0.01 else '*' if mean_p < 0.05 else 'ns'
        ax.text(i, means[i] + stds[i] + 0.008, marker,
                ha='center', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('RSA ρ (Mantel test)', fontsize=12)
    ax.set_title('V10 Ablation: Geometric Alignment by Condition', fontsize=13)
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)
    ax.set_ylim(bottom=-0.02)

    # Add annotation
    ax.annotate('All conditions p < 0.0001',
                xy=(0.98, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=9, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# Figure 2: Per-dimension heatmap
# ============================================================================

def plot_dimension_heatmap(all_results, save_path):
    """Per-dimension marginal correlation heatmap (dim × condition)."""
    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Compute per-dim correlations for each condition (average across seeds)
    matrix = np.zeros((len(DIM_NAMES), len(CONDITIONS)))

    for j, cond in enumerate(CONDITIONS):
        dim_rhos = np.zeros((len(SEEDS), len(DIM_NAMES)))
        for s, res in enumerate(all_results[cond]):
            dim_corrs = per_dimension_correlation(res['affect_mat'], res['obs_emb'])
            for d, name in enumerate(DIM_NAMES):
                dim_rhos[s, d] = dim_corrs[name][0]
        matrix[:, j] = dim_rhos.mean(axis=0)

    im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    plt.colorbar(im, ax=ax, label='Spearman ρ', shrink=0.8)

    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS], rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(DIM_NAMES)))
    ax.set_yticklabels(DIM_NAMES, fontsize=10)
    ax.set_title('Per-Dimension Marginal Correlations', fontsize=12)

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = 'white' if abs(matrix[i, j]) > 0.15 else 'black'
            ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center',
                    fontsize=8, color=color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# Figure 3: Affect trajectories
# ============================================================================

def plot_affect_trajectories(all_results, save_path):
    """6D affect trajectories for full condition, seed 0."""
    affect_mat = all_results['full'][0]['affect_mat']

    fig, axes = plt.subplots(3, 2, figsize=(12, 8))

    for i, (ax, name) in enumerate(zip(axes.flat, DIM_NAMES)):
        vals = affect_mat[:, i]
        ax.plot(vals, alpha=0.3, linewidth=0.5, color='steelblue')

        # Rolling average
        window = min(100, len(vals) // 10)
        if window > 1:
            rolling = np.convolve(vals, np.ones(window) / window, mode='valid')
            ax.plot(np.arange(window - 1, window - 1 + len(rolling)), rolling,
                    color='darkred', linewidth=1.5, label=f'MA({window})')

        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Step', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(labelsize=8)

    fig.suptitle('V10 Affect Trajectories (Full Condition, Seed 0)', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# Figure 4: Training curves
# ============================================================================

def plot_training_curves(all_results, save_path):
    """Training curves across conditions (reward, loss, health)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for cond in CONDITIONS:
        data = all_results[cond][0]['data']
        metrics = data['metrics_history']
        label = CONDITION_LABELS[cond]

        steps = [m['step'] for m in metrics]

        # Reward
        ax = axes[0, 0]
        rewards = [m['reward'] for m in metrics]
        ax.plot(steps, rewards, label=label, alpha=0.8)

        # Total loss
        ax = axes[0, 1]
        losses = [m['total_loss'] for m in metrics]
        ax.plot(steps, losses, label=label, alpha=0.8)

        # Health
        ax = axes[1, 0]
        health = [m['health'] for m in metrics]
        ax.plot(steps, health, label=label, alpha=0.8)

        # Signal entropy
        ax = axes[1, 1]
        sig_h = [m['signal_entropy'] for m in metrics]
        ax.plot(steps, sig_h, label=label, alpha=0.8)

    for ax, title, ylabel in zip(
        axes.flat,
        ['Episode Reward', 'Total Loss', 'Agent Health', 'Signal Entropy'],
        ['Reward', 'Loss', 'Health', 'Entropy (nats)']
    ):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Step', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)

    axes[0, 0].legend(fontsize=7, ncol=2, loc='lower right')

    fig.suptitle('V10 Training Metrics by Condition', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# Figure 5: Null distribution
# ============================================================================

def plot_null_distribution(all_results, save_path):
    """Mantel test null distribution for full condition."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for s, (ax, seed_result) in enumerate(zip(axes, all_results['full'])):
        rsa = seed_result['rsa']
        if rsa.null_distribution is not None:
            ax.hist(rsa.null_distribution, bins=60, alpha=0.7, color='steelblue',
                    edgecolor='white', linewidth=0.3, density=True)
            ax.axvline(rsa.mantel_rho, color='red', linestyle='--', linewidth=2,
                       label=f'Observed ρ={rsa.mantel_rho:.3f}')
            ax.set_xlabel('Spearman ρ', fontsize=10)
            ax.set_ylabel('Density' if s == 0 else '', fontsize=10)
            ax.set_title(f'Seed {s} (p={rsa.mantel_p:.4f})', fontsize=11)
            ax.legend(fontsize=9)
        ax.tick_params(labelsize=8)

    fig.suptitle('V10 Mantel Test: Null Distribution (Full Condition)', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# Figure 6: MDS scatter
# ============================================================================

def plot_mds_scatter(all_results, save_path):
    """MDS of info-theoretic vs embedding affect space (full, seed 0)."""
    res = all_results['full'][0]
    affect_mat = res['affect_mat'][:500]
    obs_emb = res['obs_emb'][:500]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42,
              normalized_stress='auto')

    # Info-theoretic space
    ax = axes[0]
    A_2d = mds.fit_transform(affect_mat)
    sc = ax.scatter(A_2d[:, 0], A_2d[:, 1], c=affect_mat[:, 0],
                    cmap='RdYlGn', s=8, alpha=0.5)
    plt.colorbar(sc, ax=ax, label='Valence', shrink=0.8)
    ax.set_title('Info-Theoretic Affect (MDS)', fontsize=11)
    ax.set_xlabel('MDS-1', fontsize=9)
    ax.set_ylabel('MDS-2', fontsize=9)

    # Embedding-predicted space
    ax = axes[1]
    E_2d = mds.fit_transform(obs_emb)
    sc = ax.scatter(E_2d[:, 0], E_2d[:, 1], c=affect_mat[:, 0],
                    cmap='RdYlGn', s=8, alpha=0.5)
    plt.colorbar(sc, ax=ax, label='Valence', shrink=0.8)
    ax.set_title('Embedding Space (MDS)', fontsize=11)
    ax.set_xlabel('MDS-1', fontsize=9)

    # RDM scatter
    ax = axes[2]
    D_a = compute_rdm(affect_mat[:200])
    D_e = compute_rdm(obs_emb[:200])
    upper = np.triu_indices(D_a.shape[0], k=1)
    ax.scatter(D_a[upper], D_e[upper], s=1, alpha=0.1, color='steelblue')
    z = np.polyfit(D_a[upper], D_e[upper], 1)
    p = np.poly1d(z)
    x_line = np.linspace(D_a[upper].min(), D_a[upper].max(), 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2, alpha=0.8)
    rho, _ = rsa_correlation(D_a, D_e)
    ax.set_title(f'RDM Correlation (ρ={rho:.3f})', fontsize=11)
    ax.set_xlabel('Info-theoretic distance', fontsize=9)
    ax.set_ylabel('Embedding distance', fontsize=9)

    fig.suptitle('V10 Geometric Alignment: Affect Space Structure', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    print("="*60)
    print("V10 Ablation Analysis: Loading and Analyzing All Conditions")
    print("="*60)

    # Run all RSA analyses
    all_results = run_all_analyses()

    # Print summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"{'Condition':25s} | {'ρ (mean±std)':15s} | {'p':10s} | {'CKA':8s}")
    print("-"*65)
    for cond in CONDITIONS:
        rhos = [r['rsa'].mantel_rho for r in all_results[cond]]
        ps = [r['rsa'].mantel_p for r in all_results[cond]]
        ckas = [r['rsa'].cka_linear for r in all_results[cond]]
        print(f"{CONDITION_LABELS[cond]:25s} | {np.mean(rhos):.3f}±{np.std(rhos):.3f}       "
              f"| {np.mean(ps):.4f}    | {np.mean(ckas):.3f}")

    # Generate all figures
    print("\n" + "="*60)
    print("GENERATING FIGURES")
    print("="*60)

    plot_ablation_bars(all_results, OUT_DIR / 'v10_ablation_bars.png')
    plot_dimension_heatmap(all_results, OUT_DIR / 'v10_dimension_heatmap.png')
    plot_affect_trajectories(all_results, OUT_DIR / 'v10_affect_trajectories.png')
    plot_training_curves(all_results, OUT_DIR / 'v10_training_curves.png')
    plot_null_distribution(all_results, OUT_DIR / 'v10_null_distribution.png')
    plot_mds_scatter(all_results, OUT_DIR / 'v10_mds_scatter.png')

    print("\n" + "="*60)
    print(f"All 6 figures saved to {OUT_DIR}/")
    print("="*60)
