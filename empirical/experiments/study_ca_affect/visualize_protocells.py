"""Visualize protocell agent snapshots as images and animations.

Creates:
  1. Single-frame images showing agent positions, resources, energy
  2. Multi-cycle comparison strips
  3. Robustness/Phi trajectory plots across V22-V24
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import json
import os
import sys
import glob


def render_snapshot(snap_path, out_path=None, title=None, figsize=(8, 8)):
    """Render a single snapshot as a 2D image."""
    data = np.load(snap_path, allow_pickle=True)

    resources = data['resources']
    positions = data['positions']
    energy = data['energy']
    alive = data['alive']

    N = resources.shape[0]
    n_alive = int(alive.sum())
    cycle = int(data['cycle']) if 'cycle' in data else -1

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Resource heatmap
    ax.imshow(resources.T, origin='lower', cmap='YlGn', alpha=0.6,
              extent=[0, N, 0, N], vmin=0, vmax=1)

    # Alive agents
    alive_idx = np.where(alive)[0]
    if len(alive_idx) > 0:
        pos = positions[alive_idx]
        eng = energy[alive_idx]

        # Color by energy
        scatter = ax.scatter(pos[:, 0], pos[:, 1],
                           c=eng, cmap='coolwarm',
                           s=20, alpha=0.8,
                           vmin=0, vmax=2.0,
                           edgecolors='black', linewidths=0.3)
        plt.colorbar(scatter, ax=ax, label='Energy', shrink=0.6)

    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_aspect('equal')

    if title:
        ax.set_title(title, fontsize=12)
    else:
        ax.set_title('Cycle %d | %d agents alive' % (cycle, n_alive), fontsize=12)

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print('Saved: %s' % out_path)
    plt.close()
    return fig


def render_cycle_strip(snap_dir, version, seed, out_path=None, cycles=None):
    """Render a strip of snapshots across cycles."""
    if cycles is None:
        cycles = [0, 5, 10, 15, 20, 25, 29]

    # Find available snapshots
    available = []
    for c in cycles:
        path = os.path.join(snap_dir, 'snapshot_c%02d.npz' % c)
        if os.path.exists(path):
            available.append((c, path))

    if not available:
        print('No snapshots found in %s' % snap_dir)
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]

    for i, (cycle, path) in enumerate(available):
        data = np.load(path, allow_pickle=True)
        resources = data['resources']
        positions = data['positions']
        energy = data['energy']
        alive = data['alive']
        N = resources.shape[0]
        n_alive = int(alive.sum())

        ax = axes[i]
        ax.imshow(resources.T, origin='lower', cmap='YlGn', alpha=0.6,
                  extent=[0, N, 0, N], vmin=0, vmax=1)

        alive_idx = np.where(alive)[0]
        if len(alive_idx) > 0:
            pos = positions[alive_idx]
            eng = energy[alive_idx]
            ax.scatter(pos[:, 0], pos[:, 1],
                      c=eng, cmap='coolwarm', s=8, alpha=0.8,
                      vmin=0, vmax=2.0, edgecolors='black', linewidths=0.2)

        ax.set_xlim(0, N)
        ax.set_ylim(0, N)
        ax.set_aspect('equal')
        ax.set_title('C%d (%d)' % (cycle, n_alive), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle('%s seed %d — Agent Evolution' % (version.upper(), seed), fontsize=14)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print('Saved: %s' % out_path)
    plt.close()


def plot_trajectories(results_dir, version, seeds, out_path=None):
    """Plot robustness, Phi, and learning metrics across cycles for multiple seeds."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {42: '#e41a1c', 123: '#377eb8', 7: '#4daf4a'}

    for seed in seeds:
        result_path = os.path.join(results_dir, '%s_s%d' % (version, seed),
                                   '%s_s%d_results.json' % (version, seed))
        if not os.path.exists(result_path):
            continue

        with open(result_path) as f:
            data = json.load(f)

        cycles_data = data['cycles']
        x = [c['cycle'] for c in cycles_data]
        color = colors.get(seed, 'gray')
        label = 'seed %d' % seed

        # Robustness
        rob = [c['robustness'] for c in cycles_data]
        axes[0, 0].plot(x, rob, color=color, label=label, alpha=0.8)
        axes[0, 0].axhline(1.0, color='gray', linestyle='--', alpha=0.3)
        axes[0, 0].set_ylabel('Robustness')
        axes[0, 0].set_title('Robustness (Phi_stress / Phi_base)')

        # Phi
        phi = [c['mean_phi'] for c in cycles_data]
        axes[0, 1].plot(x, phi, color=color, label=label, alpha=0.8)
        axes[0, 1].set_ylabel('Mean Phi')
        axes[0, 1].set_title('Integration (Phi)')

        # Population
        pop = [c['n_alive_end'] for c in cycles_data]
        axes[1, 0].plot(x, pop, color=color, label=label, alpha=0.8)
        axes[1, 0].set_ylabel('Population')
        axes[1, 0].set_title('Population')
        axes[1, 0].set_xlabel('Cycle')

        # Version-specific metric
        if 'mean_td_error' in cycles_data[0]:
            metric = [c['mean_td_error'] for c in cycles_data]
            axes[1, 1].plot(x, metric, color=color, label=label, alpha=0.8)
            axes[1, 1].set_ylabel('TD Error')
            axes[1, 1].set_title('TD Error')
        elif 'mean_pred_mse_total' in cycles_data[0]:
            metric = [c['mean_pred_mse_total'] for c in cycles_data]
            axes[1, 1].plot(x, metric, color=color, label=label, alpha=0.8)
            axes[1, 1].set_ylabel('Pred MSE')
            axes[1, 1].set_title('Prediction MSE')
        elif 'mean_pred_mse' in cycles_data[0]:
            metric = [c['mean_pred_mse'] for c in cycles_data]
            axes[1, 1].plot(x, metric, color=color, label=label, alpha=0.8)
            axes[1, 1].set_ylabel('Pred MSE')
            axes[1, 1].set_title('Prediction MSE')

        axes[1, 1].set_xlabel('Cycle')

    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    fig.suptitle('%s — Trajectories' % version.upper(), fontsize=14, y=1.02)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print('Saved: %s' % out_path)
    plt.close()


def plot_cross_version_comparison(base_dir, versions, seeds, out_path=None):
    """Compare robustness and Phi across V22, V23, V24."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    version_colors = {'v22': '#e41a1c', 'v23': '#377eb8', 'v24': '#4daf4a'}

    for vi, version in enumerate(versions):
        robs = []
        phis = []
        for seed in seeds:
            result_path = os.path.join(base_dir, '%s_s%d' % (version, seed),
                                       '%s_s%d_results.json' % (version, seed))
            if not os.path.exists(result_path):
                continue
            with open(result_path) as f:
                data = json.load(f)
            robs.append(data['summary']['mean_robustness'])
            phis.append(data['summary']['mean_phi'])

        if robs:
            color = version_colors.get(version, 'gray')
            x = [vi] * len(robs)
            axes[0].scatter(x, robs, color=color, s=80, zorder=5, alpha=0.8)
            axes[0].bar(vi, np.mean(robs), color=color, alpha=0.3, width=0.6)

            axes[1].scatter(x, phis, color=color, s=80, zorder=5, alpha=0.8)
            axes[1].bar(vi, np.mean(phis), color=color, alpha=0.3, width=0.6)

    axes[0].axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xticks(range(len(versions)))
    axes[0].set_xticklabels([v.upper() for v in versions])
    axes[0].set_ylabel('Mean Robustness')
    axes[0].set_title('Robustness Across Prediction Experiments')

    axes[1].set_xticks(range(len(versions)))
    axes[1].set_xticklabels([v.upper() for v in versions])
    axes[1].set_ylabel('Mean Phi')
    axes[1].set_title('Integration (Phi) Across Prediction Experiments')

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print('Saved: %s' % out_path)
    plt.close()


if __name__ == '__main__':
    base_dir = 'results'
    out_dir = 'results/figures'
    os.makedirs(out_dir, exist_ok=True)

    seeds = [42, 123, 7]
    versions = ['v22', 'v23', 'v24']

    # Generate all visualizations
    for version in versions:
        for seed in seeds:
            snap_dir = os.path.join(base_dir, '%s_s%d' % (version, seed))
            if not os.path.exists(snap_dir):
                continue

            # Cycle strip
            render_cycle_strip(
                snap_dir, version, seed,
                out_path=os.path.join(out_dir, '%s_s%d_strip.png' % (version, seed))
            )

            # Individual snapshots (start, middle, end)
            for c in [0, 15, 29]:
                snap_path = os.path.join(snap_dir, 'snapshot_c%02d.npz' % c)
                if os.path.exists(snap_path):
                    render_snapshot(
                        snap_path,
                        out_path=os.path.join(out_dir, '%s_s%d_c%02d.png' % (version, seed, c)),
                        title='%s seed %d cycle %d' % (version.upper(), seed, c)
                    )

        # Trajectory plots
        plot_trajectories(
            base_dir, version, seeds,
            out_path=os.path.join(out_dir, '%s_trajectories.png' % version)
        )

    # Cross-version comparison
    plot_cross_version_comparison(
        base_dir, versions, seeds,
        out_path=os.path.join(out_dir, 'v22_v23_v24_comparison.png')
    )

    print("\nAll figures saved to %s" % out_dir)
