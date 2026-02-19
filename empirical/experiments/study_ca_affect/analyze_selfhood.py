"""Analyze V24 seed 7 for proto-self signatures.

V24 seed 7 had Phi=0.130 (highest in any prediction experiment).
Look for: detachment, mode switching, affect motif clustering,
internal mixing, semantic grounding.

This is an analysis of EXISTING data, not a new experiment.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json
import os


def analyze_snapshot(snap_path, label=""):
    """Analyze a single snapshot for self-emergence signatures."""
    data = np.load(snap_path, allow_pickle=True)

    hidden = data['hidden']       # (M, H)
    positions = data['positions']  # (M, 2)
    energy = data['energy']       # (M,)
    alive = data['alive']         # (M,)
    resources = data['resources']  # (N, N)

    alive_idx = np.where(alive)[0]
    if len(alive_idx) < 10:
        return None

    h = hidden[alive_idx]  # (n_alive, H)
    pos = positions[alive_idx]
    eng = energy[alive_idx]

    # Filter out NaN/Inf hidden states
    valid = np.all(np.isfinite(h), axis=1)
    h = h[valid]
    pos = pos[valid]
    eng = eng[valid]

    if len(h) < 10:
        return None

    results = {'label': label, 'n_alive': len(h)}

    # 1. Hidden state effective dimensionality
    # PCA on hidden states — how many dimensions are actually used?
    if h.shape[0] > h.shape[1]:
        pca = PCA()
        pca.fit(h)
        explained = pca.explained_variance_ratio_
        cumulative = np.cumsum(explained)

        # Effective rank
        evals = pca.explained_variance_
        evals_norm = evals / (np.sum(evals) + 1e-10)
        ent = -np.sum(evals_norm * np.log(evals_norm + 1e-10))
        eff_rank = np.exp(ent)

        results['effective_rank'] = float(eff_rank)
        results['dims_for_90pct'] = int(np.searchsorted(cumulative, 0.9) + 1)
        results['explained_variance'] = explained.tolist()

    # 2. Affect motif clustering
    # K-means on hidden states — are there distinct modes?
    best_k = 2
    best_sil = -1
    for k in range(2, min(8, max(len(h) // 5, 3))):
        try:
            km = KMeans(n_clusters=k, n_init=5, random_state=42)
            labels = km.fit_predict(h)
            if len(set(labels)) < 2:
                continue
            sil = silhouette_score(h, labels)
            if sil > best_sil:
                best_sil = sil
                best_k = k
        except Exception:
            continue

    results['best_k'] = best_k
    results['silhouette'] = float(best_sil)
    # silhouette > 0.5 = reasonable clustering, > 0.7 = strong

    # 3. Hidden state diversity
    # How different are agents' hidden states from each other?
    h_centered = h - h.mean(axis=0)
    pairwise_dist = np.sqrt(np.sum(
        (h_centered[:, None, :] - h_centered[None, :, :]) ** 2, axis=2
    ))
    mean_dist = float(np.mean(pairwise_dist[np.triu_indices(len(h), k=1)]))
    results['mean_pairwise_distance'] = mean_dist

    # 4. Energy-hidden correlation
    # Can we decode energy from hidden state? (basic semantic grounding)
    from numpy.linalg import lstsq
    X = np.column_stack([h, np.ones(len(h))])
    coefs, residuals, _, _ = lstsq(X, eng, rcond=None)
    pred_eng = X @ coefs
    ss_res = np.sum((eng - pred_eng) ** 2)
    ss_tot = np.sum((eng - eng.mean()) ** 2)
    r2_energy = 1 - ss_res / (ss_tot + 1e-10) if ss_tot > 0 else 0
    results['r2_energy'] = float(r2_energy)

    # 5. Position-hidden correlation
    # Can we decode position from hidden state? (spatial grounding)
    for dim, name in [(0, 'x'), (1, 'y')]:
        target = pos[:, dim].astype(float)
        coefs, _, _, _ = lstsq(X, target, rcond=None)
        pred = X @ coefs
        ss_res = np.sum((target - pred) ** 2)
        ss_tot = np.sum((target - target.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10) if ss_tot > 0 else 0
        results['r2_pos_%s' % name] = float(r2)

    # 6. Local resource grounding
    # Can we decode local resource level from hidden state?
    N = resources.shape[0]
    local_res = np.array([
        np.mean(resources[
            max(0,p[0]-2):min(N,p[0]+3),
            max(0,p[1]-2):min(N,p[1]+3)
        ]) for p in pos
    ])
    coefs, _, _, _ = lstsq(X, local_res, rcond=None)
    pred = X @ coefs
    ss_res = np.sum((local_res - pred) ** 2)
    ss_tot = np.sum((local_res - local_res.mean()) ** 2)
    r2_resource = 1 - ss_res / (ss_tot + 1e-10) if ss_tot > 0 else 0
    results['r2_local_resource'] = float(r2_resource)

    # 7. Hidden state norm distribution
    # How spread are hidden state magnitudes? (activity levels)
    h_norms = np.sqrt(np.sum(h ** 2, axis=1))
    results['h_norm_mean'] = float(np.mean(h_norms))
    results['h_norm_std'] = float(np.std(h_norms))
    results['h_norm_cv'] = float(np.std(h_norms) / (np.mean(h_norms) + 1e-10))

    return results


def compare_versions(base_dir, versions_seeds, cycles=None):
    """Compare self-emergence signatures across versions and seeds."""
    if cycles is None:
        cycles = [0, 5, 10, 15, 20, 25, 29]

    all_results = []

    for version, seed in versions_seeds:
        for cycle in cycles:
            snap_path = os.path.join(
                base_dir, '%s_s%d' % (version, seed),
                'snapshot_c%02d.npz' % cycle
            )
            if not os.path.exists(snap_path):
                continue

            label = '%s_s%d_c%02d' % (version, seed, cycle)
            result = analyze_snapshot(snap_path, label)
            if result:
                result['version'] = version
                result['seed'] = seed
                result['cycle'] = cycle
                all_results.append(result)

    return all_results


def plot_self_emergence(results, out_path=None):
    """Plot self-emergence signatures across cycles."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Group by version+seed
    groups = {}
    for r in results:
        key = '%s_s%d' % (r['version'], r['seed'])
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    colors = {
        'v22_s42': '#e41a1c', 'v22_s123': '#ff7f7f', 'v22_s7': '#ffa0a0',
        'v23_s42': '#377eb8', 'v23_s123': '#7fb0e8', 'v23_s7': '#a0c0f0',
        'v24_s42': '#4daf4a', 'v24_s123': '#7fcf7f', 'v24_s7': '#00aa00',
    }

    metrics = [
        ('effective_rank', 'Effective Rank (hidden state)'),
        ('silhouette', 'Affect Motif Clustering (silhouette)'),
        ('r2_energy', 'Energy Decoding R² (semantic grounding)'),
        ('r2_local_resource', 'Resource Decoding R² (spatial grounding)'),
        ('mean_pairwise_distance', 'Hidden State Diversity'),
        ('h_norm_cv', 'Activity Level Variation (CV)'),
    ]

    for ax_idx, (metric, title) in enumerate(metrics):
        ax = axes[ax_idx // 3, ax_idx % 3]
        for key, group in sorted(groups.items()):
            x = [r['cycle'] for r in group]
            y = [r.get(metric, 0) for r in group]
            color = colors.get(key, 'gray')
            lw = 2.5 if 'v24_s7' in key else 1.0
            ax.plot(x, y, color=color, label=key, linewidth=lw, alpha=0.8)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Cycle')
        ax.grid(alpha=0.2)
        if ax_idx == 0:
            ax.legend(fontsize=6, ncol=3)

    fig.suptitle('Proto-Self Signatures Across V22-V24', fontsize=14, y=1.02)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print('Saved: %s' % out_path)
    plt.close()


if __name__ == '__main__':
    base_dir = 'results'
    out_dir = 'results/figures'
    os.makedirs(out_dir, exist_ok=True)

    versions_seeds = [
        ('v22', 42), ('v22', 123), ('v22', 7),
        ('v23', 42), ('v23', 123), ('v23', 7),
        ('v24', 42), ('v24', 123), ('v24', 7),
    ]

    print("Analyzing self-emergence signatures...")
    results = compare_versions(base_dir, versions_seeds)

    # Print summary
    print("\n=== SELF-EMERGENCE ANALYSIS ===")
    print("%-20s | rank | sil  | R²_E | R²_R | dist | CV" % "Snapshot")
    print("-" * 75)
    for r in sorted(results, key=lambda x: (x['version'], x['seed'], x['cycle'])):
        print("%-20s | %.1f  | %.2f | %.2f | %.2f | %.2f | %.2f" % (
            r['label'],
            r.get('effective_rank', 0),
            r.get('silhouette', 0),
            r.get('r2_energy', 0),
            r.get('r2_local_resource', 0),
            r.get('mean_pairwise_distance', 0),
            r.get('h_norm_cv', 0),
        ))

    # Special focus on V24 seed 7 (highest Phi)
    v24_s7 = [r for r in results if r['version'] == 'v24' and r['seed'] == 7]
    if v24_s7:
        print("\n=== V24 SEED 7 (Phi=0.130) — DETAILED ===")
        for r in v24_s7:
            print("\nCycle %d:" % r['cycle'])
            for k, v in sorted(r.items()):
                if k not in ['label', 'version', 'seed', 'cycle', 'explained_variance']:
                    print("  %s: %s" % (k, v))

    # Plot
    plot_self_emergence(results, out_path=os.path.join(out_dir, 'self_emergence_signatures.png'))

    # Save JSON
    with open(os.path.join(out_dir, 'self_emergence_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\nAnalysis saved.")
