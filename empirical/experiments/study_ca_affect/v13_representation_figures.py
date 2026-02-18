"""Experiment 3: Internal Representation Structure — Visualization.

Figures:
1. d_eff and Abstraction level across evolutionary cycles
2. Disentanglement score trajectory
3. Eigenspectrum comparison: early vs late
4. Summary card
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Dict, List

matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['savefig.dpi'] = 200
matplotlib.rcParams['savefig.bbox'] = 'tight'

SEED_COLORS = {123: '#E74C3C', 42: '#3498DB', 7: '#2ECC71'}
SEED_LABELS = {123: 'Seed 123', 42: 'Seed 42', 7: 'Seed 7'}


def plot_representation_trajectory(out_dir: Path, trajectories: Dict, seeds: list):
    """d_eff, A, D, K_comp across cycles."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    metrics = [
        ('mean_d_eff', 'd_eff (Effective Dimensionality)', 'o'),
        ('mean_A', 'Abstraction Level (A)', 's'),
        ('mean_D', 'Disentanglement (D)', '^'),
        ('mean_K_comp', 'Compositionality Error (K_comp)', 'D'),
    ]

    for ax, (key, ylabel, marker) in zip(axes.flat, metrics):
        for seed in seeds:
            if seed not in trajectories:
                continue
            traj = trajectories[seed]
            cycles = [t['cycle'] for t in traj]
            vals = [t[key] for t in traj]
            color = SEED_COLORS.get(seed, '#999')
            ax.plot(cycles, vals, f'{marker}-', color=color,
                    label=SEED_LABELS.get(seed, f's={seed}'),
                    markersize=5, linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Evolutionary Cycle')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0, 0].set_title('(a) Effective Dimensionality')
    axes[0, 1].set_title('(b) Abstraction')
    axes[1, 0].set_title('(c) Disentanglement')
    axes[1, 1].set_title('(d) Compositionality Error')

    plt.suptitle('Experiment 3: Representation Structure Over Evolution',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / 'rep_trajectory.png')
    plt.close()
    print(f"  Saved: {out_dir / 'rep_trajectory.png'}")


def plot_eigenspectrum(out_dir: Path, seeds: list):
    """Eigenspectrum of internal state: early vs late evolution."""
    fig, axes = plt.subplots(1, len(seeds), figsize=(5 * len(seeds), 4.5),
                             squeeze=False)

    for col, seed in enumerate(seeds):
        ax = axes[0, col]
        rep_dir = out_dir.parent / f'rep_s{seed}'
        if not rep_dir.exists():
            ax.set_title(f'Seed {seed}: no data')
            continue

        cycle_files = sorted(rep_dir.glob('rep_cycle_*.json'))
        if len(cycle_files) < 2:
            ax.set_title(f'Seed {seed}: insufficient data')
            continue

        for label, fpath, color, ls in [
            ('Early', cycle_files[0], '#95A5A6', '--'),
            ('Late', cycle_files[-1], SEED_COLORS.get(seed, '#E74C3C'), '-'),
        ]:
            with open(fpath) as f:
                data = json.load(f)
            cycle = data.get('cycle', '?')

            # Aggregate eigenspectra across patterns
            all_eig = []
            for pat in data.get('patterns', []):
                eig = pat.get('eigenspectrum', [])
                if eig:
                    all_eig.append(eig)

            if all_eig:
                max_len = max(len(e) for e in all_eig)
                padded = [e + [0] * (max_len - len(e)) for e in all_eig]
                mean_eig = np.mean(padded, axis=0)
                # Normalize
                total = mean_eig.sum()
                if total > 0:
                    mean_eig = mean_eig / total
                dims = np.arange(1, len(mean_eig) + 1)
                ax.plot(dims, mean_eig, f'o{ls}', color=color,
                        label=f'{label} (cycle {cycle})',
                        markersize=4, linewidth=1.5)

        ax.set_xlabel('PCA Dimension')
        ax.set_ylabel('Variance Fraction')
        ax.set_title(f'Seed {seed}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    plt.suptitle('Eigenspectrum of Internal State: Early vs Late', y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / 'rep_eigenspectrum.png')
    plt.close()
    print(f"  Saved: {out_dir / 'rep_eigenspectrum.png'}")


def plot_summary_card(out_dir: Path, trajectories: Dict, seeds: list):
    """Summary card."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (0,0) d_eff trajectory
    ax = axes[0, 0]
    for seed in seeds:
        if seed not in trajectories:
            continue
        traj = trajectories[seed]
        cycles = [t['cycle'] for t in traj]
        vals = [t['mean_d_eff'] for t in traj]
        ax.plot(cycles, vals, 'o-', color=SEED_COLORS.get(seed, '#999'),
                label=SEED_LABELS.get(seed), markersize=4, linewidth=1.5)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('d_eff')
    ax.set_title('(a) Effective Dimensionality')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (0,1) Abstraction + Disentanglement
    ax = axes[0, 1]
    for seed in seeds:
        if seed not in trajectories:
            continue
        traj = trajectories[seed]
        cycles = [t['cycle'] for t in traj]
        ax.plot(cycles, [t['mean_A'] for t in traj], 'o-',
                color=SEED_COLORS.get(seed, '#999'),
                label=f'A ({SEED_LABELS.get(seed)})',
                markersize=3, linewidth=1.5, alpha=0.7)
        ax.plot(cycles, [t['mean_D'] for t in traj], 's--',
                color=SEED_COLORS.get(seed, '#999'),
                label=f'D ({SEED_LABELS.get(seed)})',
                markersize=3, linewidth=1, alpha=0.5)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Score')
    ax.set_title('(b) Abstraction (A) & Disentanglement (D)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # (1,0) K_comp
    ax = axes[1, 0]
    for seed in seeds:
        if seed not in trajectories:
            continue
        traj = trajectories[seed]
        cycles = [t['cycle'] for t in traj]
        ax.plot(cycles, [t['mean_K_comp'] for t in traj], 'D-',
                color=SEED_COLORS.get(seed, '#999'),
                label=SEED_LABELS.get(seed),
                markersize=4, linewidth=1.5)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('K_comp')
    ax.set_title('(c) Compositionality Error (lower = better)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (1,1) Stats text
    ax = axes[1, 1]
    ax.axis('off')
    lines = ['Experiment 3: Representation Structure\n']
    for seed in seeds:
        if seed not in trajectories or not trajectories[seed]:
            continue
        last = trajectories[seed][-1]
        lines.append(f'Seed {seed} (final):')
        lines.append(f'  d_eff:   {last["mean_d_eff"]:.1f}')
        lines.append(f'  A:       {last["mean_A"]:.3f}')
        lines.append(f'  D:       {last["mean_D"]:.3f}')
        lines.append(f'  K_comp:  {last["mean_K_comp"]:.3f}')
        lines.append('')
    ax.text(0.1, 0.9, '\n'.join(lines), transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Experiment 3: Internal Representation — Summary',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / 'rep_summary_card.png')
    plt.close()
    print(f"  Saved: {out_dir / 'rep_summary_card.png'}")


def plot_all(out_dir: Path, trajectories: Dict, seeds: list):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating representation figures...")
    plot_representation_trajectory(out_dir, trajectories, seeds)
    plot_eigenspectrum(out_dir, seeds)
    plot_summary_card(out_dir, trajectories, seeds)
    print("All figures complete.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[123, 42, 7])
    parser.add_argument('--results-dir', type=str,
                        default=str(Path(__file__).parent / 'results'))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    analysis_dir = results_dir / 'rep_analysis'

    cross_seed_path = analysis_dir / 'rep_cross_seed.json'
    if cross_seed_path.exists():
        with open(cross_seed_path) as f:
            data = json.load(f)
        trajectories = {int(k): v for k, v in data['trajectories'].items()}
    else:
        print(f"No data at {cross_seed_path}. Run 'full' first.")
        trajectories = {}

    plot_all(analysis_dir, trajectories, args.seeds)
