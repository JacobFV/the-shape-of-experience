"""Experiment 2: Emergent World Model — Visualization.

Generates three key figures:
1. C_wm trajectory across evolutionary generations (the money plot)
2. W(τ) curves: early vs late evolution
3. C_wm vs pattern lifetime correlation
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Dict, List, Optional

matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['savefig.dpi'] = 200
matplotlib.rcParams['savefig.bbox'] = 'tight'

SEED_COLORS = {123: '#E74C3C', 42: '#3498DB', 7: '#2ECC71'}
SEED_LABELS = {123: 'Seed 123', 42: 'Seed 42', 7: 'Seed 7'}


def plot_cwm_trajectory(out_dir: Path, trajectories: Dict, seeds: list):
    """Figure 1: C_wm across evolutionary cycles.

    This is the key figure — does world model capacity increase with evolution?
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: C_wm trajectory
    ax = axes[0]
    for seed in seeds:
        if seed not in trajectories:
            continue
        traj = trajectories[seed]
        cycles = [t['cycle'] for t in traj]
        c_wm = [t['mean_C_wm'] for t in traj]
        color = SEED_COLORS.get(seed, '#999')
        ax.plot(cycles, c_wm, 'o-', color=color, label=SEED_LABELS.get(seed, f's={seed}'),
                markersize=4, linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Evolutionary Cycle')
    ax.set_ylabel('World Model Capacity (C_wm)')
    ax.set_title('Emergent World Model: Capacity Over Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Fraction of patterns with world model
    ax = axes[1]
    for seed in seeds:
        if seed not in trajectories:
            continue
        traj = trajectories[seed]
        cycles = [t['cycle'] for t in traj]
        frac = [t['frac_with_wm'] for t in traj]
        color = SEED_COLORS.get(seed, '#999')
        ax.plot(cycles, frac, 's-', color=color, label=SEED_LABELS.get(seed, f's={seed}'),
                markersize=4, linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Evolutionary Cycle')
    ax.set_ylabel('Fraction with World Model')
    ax.set_title('Prevalence of World Models')
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / 'wm_capacity_trajectory.png')
    plt.close()
    print(f"  Saved: {out_dir / 'wm_capacity_trajectory.png'}")


def plot_w_tau_curves(out_dir: Path, seeds: list):
    """Figure 2: W(τ) curves for early vs late evolution.

    Shows how prediction gap changes with horizon at different stages.
    """
    fig, axes = plt.subplots(1, len(seeds), figsize=(5 * len(seeds), 5),
                             squeeze=False)

    for col, seed in enumerate(seeds):
        ax = axes[0, col]
        wm_dir = out_dir.parent / f'wm_s{seed}'
        if not wm_dir.exists():
            ax.set_title(f'Seed {seed}: no data')
            continue

        # Find early and late cycle files
        cycle_files = sorted(wm_dir.glob('wm_cycle_*.json'))
        if len(cycle_files) < 2:
            ax.set_title(f'Seed {seed}: insufficient cycles')
            continue

        # Early = first cycle, Late = last cycle
        early_file = cycle_files[0]
        late_file = cycle_files[-1]

        for label, fpath, color, marker in [
            ('Early', early_file, '#95A5A6', 'o'),
            ('Late', late_file, SEED_COLORS.get(seed, '#E74C3C'), 's'),
        ]:
            with open(fpath) as f:
                data = json.load(f)
            cycle = data.get('cycle', '?')
            w_by_tau = data['summary'].get('mean_W_by_tau', {})
            if not w_by_tau:
                continue
            taus = sorted([int(t) for t in w_by_tau.keys()])
            ws = [w_by_tau[str(t)] for t in taus]
            ax.plot(taus, ws, f'{marker}-', color=color, label=f'{label} (cycle {cycle})',
                    markersize=6, linewidth=2)

        ax.set_xlabel('Prediction Horizon (τ)')
        ax.set_ylabel('Prediction Gap W(τ)')
        ax.set_title(f'Seed {seed}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log') if all(
            w > 0 for w in ws) else None

    plt.suptitle('Prediction Gap vs Horizon: Early vs Late Evolution', y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / 'wm_w_tau_curves.png')
    plt.close()
    print(f"  Saved: {out_dir / 'wm_w_tau_curves.png'}")


def plot_cwm_vs_lifetime(out_dir: Path, seeds: list):
    """Figure 3: C_wm vs pattern lifetime.

    Tests whether longer-lived patterns develop stronger world models.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    all_lifetimes = []
    all_cwm = []
    all_seed_colors = []

    for seed in seeds:
        wm_dir = out_dir.parent / f'wm_s{seed}'
        if not wm_dir.exists():
            continue

        # Collect per-pattern data from all cycles
        cycle_files = sorted(wm_dir.glob('wm_cycle_*.json'))
        for fpath in cycle_files:
            with open(fpath) as f:
                data = json.load(f)
            for pat in data.get('patterns', []):
                lt = pat.get('lifetime', 0)
                cw = pat.get('C_wm', 0)
                if lt > 0:
                    all_lifetimes.append(lt)
                    all_cwm.append(cw)
                    all_seed_colors.append(SEED_COLORS.get(seed, '#999'))

    if not all_lifetimes:
        ax.set_title('No per-pattern data available')
        plt.tight_layout()
        fig.savefig(out_dir / 'wm_cwm_vs_lifetime.png')
        plt.close()
        return

    ax.scatter(all_lifetimes, all_cwm, c=all_seed_colors, alpha=0.4, s=20,
               edgecolors='none')

    # Add trend line
    lifetimes_arr = np.array(all_lifetimes)
    cwm_arr = np.array(all_cwm)
    if len(lifetimes_arr) > 5:
        z = np.polyfit(lifetimes_arr, cwm_arr, 1)
        p = np.poly1d(z)
        x_line = np.linspace(lifetimes_arr.min(), lifetimes_arr.max(), 100)
        ax.plot(x_line, p(x_line), '--', color='black', alpha=0.7,
                label=f'Trend (slope={z[0]:.4f})')

        # Correlation
        corr = np.corrcoef(lifetimes_arr, cwm_arr)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Pattern Lifetime (recording steps)')
    ax.set_ylabel('World Model Capacity (C_wm)')
    ax.set_title('World Model vs Pattern Longevity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Legend for seeds
    for seed in seeds:
        color = SEED_COLORS.get(seed, '#999')
        ax.scatter([], [], c=color, s=40, label=SEED_LABELS.get(seed, f's={seed}'))
    ax.legend(loc='upper right')

    plt.tight_layout()
    fig.savefig(out_dir / 'wm_cwm_vs_lifetime.png')
    plt.close()
    print(f"  Saved: {out_dir / 'wm_cwm_vs_lifetime.png'}")


def plot_summary_card(out_dir: Path, trajectories: Dict, seeds: list):
    """Summary card combining key metrics for the book/appendix."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (0,0) C_wm trajectory
    ax = axes[0, 0]
    for seed in seeds:
        if seed not in trajectories:
            continue
        traj = trajectories[seed]
        cycles = [t['cycle'] for t in traj]
        c_wm = [t['mean_C_wm'] for t in traj]
        color = SEED_COLORS.get(seed, '#999')
        ax.plot(cycles, c_wm, 'o-', color=color,
                label=SEED_LABELS.get(seed, f's={seed}'),
                markersize=3, linewidth=1.5)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('C_wm')
    ax.set_title('(a) World Model Capacity')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (0,1) H_wm trajectory
    ax = axes[0, 1]
    for seed in seeds:
        if seed not in trajectories:
            continue
        traj = trajectories[seed]
        cycles = [t['cycle'] for t in traj]
        h_wm = [t['mean_H_wm'] for t in traj]
        color = SEED_COLORS.get(seed, '#999')
        ax.plot(cycles, h_wm, 's-', color=color,
                label=SEED_LABELS.get(seed, f's={seed}'),
                markersize=3, linewidth=1.5)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('H_wm')
    ax.set_title('(b) World Model Horizon')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (1,0) W(τ) curves — late evolution only, all seeds
    ax = axes[1, 0]
    for seed in seeds:
        wm_dir = out_dir.parent / f'wm_s{seed}'
        if not wm_dir.exists():
            continue
        cycle_files = sorted(wm_dir.glob('wm_cycle_*.json'))
        if not cycle_files:
            continue
        with open(cycle_files[-1]) as f:
            data = json.load(f)
        w_by_tau = data['summary'].get('mean_W_by_tau', {})
        if not w_by_tau:
            continue
        taus = sorted([int(t) for t in w_by_tau.keys()])
        ws = [w_by_tau[str(t)] for t in taus]
        cycle = data.get('cycle', '?')
        color = SEED_COLORS.get(seed, '#999')
        ax.plot(taus, ws, 'o-', color=color,
                label=f'Seed {seed} (cycle {cycle})',
                markersize=5, linewidth=2)
    ax.set_xlabel('τ (recording steps)')
    ax.set_ylabel('W(τ)')
    ax.set_title('(c) Prediction Gap (Late Evolution)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (1,1) Stats text
    ax = axes[1, 1]
    ax.axis('off')
    lines = ['Experiment 2: Emergent World Model\n']
    for seed in seeds:
        if seed not in trajectories:
            continue
        traj = trajectories[seed]
        if not traj:
            continue
        last = traj[-1]
        lines.append(f'Seed {seed}:')
        lines.append(f'  C_wm (final): {last["mean_C_wm"]:.4f}')
        lines.append(f'  H_wm (final): {last["mean_H_wm"]:.1f}')
        lines.append(f'  % with WM:    {last["frac_with_wm"]:.0%}')
        lines.append('')
    ax.text(0.1, 0.9, '\n'.join(lines), transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Experiment 2: Emergent World Model — Summary', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / 'wm_summary_card.png')
    plt.close()
    print(f"  Saved: {out_dir / 'wm_summary_card.png'}")


def plot_all(out_dir: Path, trajectories: Dict, seeds: list):
    """Generate all figures."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating world model figures...")
    plot_cwm_trajectory(out_dir, trajectories, seeds)
    plot_w_tau_curves(out_dir, seeds)
    plot_cwm_vs_lifetime(out_dir, seeds)
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
    analysis_dir = results_dir / 'wm_analysis'

    # Load trajectories
    cross_seed_path = analysis_dir / 'wm_cross_seed.json'
    if cross_seed_path.exists():
        with open(cross_seed_path) as f:
            data = json.load(f)
        trajectories = {int(k): v for k, v in data['trajectories'].items()}
    else:
        print(f"No cross-seed data at {cross_seed_path}. Run 'analyze' first.")
        trajectories = {}

    plot_all(analysis_dir, trajectories, args.seeds)
