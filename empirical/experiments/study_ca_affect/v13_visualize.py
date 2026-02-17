"""V13 Visualization: Generate figures from GPU run results.

Produces publication-quality figures for the book:
1. Evolution trajectory (Phi, robustness, mortality over cycles)
2. Grid snapshots (channel-mean heatmaps at key cycles)
3. Affect space scatter (arousal vs Phi, color=stress phase)
4. Tau/beta parameter drift
5. Stress response comparison (early vs late evolution)
"""
import json
import os
import sys
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    print("Install matplotlib: pip install matplotlib")
    sys.exit(1)


# Custom colormap: black → blue → cyan → white (for grid heatmaps)
LENIA_CMAP = LinearSegmentedColormap.from_list(
    'lenia', ['#000000', '#001133', '#0055aa', '#00aadd', '#66ddff', '#ffffff'])


def load_results(result_dir):
    """Load JSON stats and return cycle_stats list."""
    final_path = os.path.join(result_dir, 'v13_final.json')
    progress_path = os.path.join(result_dir, 'v13_progress.json')

    path = final_path if os.path.exists(final_path) else progress_path
    with open(path) as f:
        data = json.load(f)
    return data


def load_snapshot(result_dir, name):
    """Load a grid snapshot .npz file."""
    path = os.path.join(result_dir, 'snapshots', f'{name}.npz')
    if os.path.exists(path):
        return np.load(path)
    return None


def fig_evolution_trajectory(data, output_dir):
    """Figure 1: Evolution trajectory — the key result figure."""
    stats = data['cycle_stats']
    cycles = [s['cycle'] for s in stats]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('V13 Content-Based Coupling: Evolution Trajectory',
                 fontsize=14, fontweight='bold')

    # 1a: Phi base vs stress
    ax = axes[0, 0]
    ax.plot(cycles, [s['mean_phi_base'] for s in stats],
            'b-o', markersize=3, label='Baseline Φ', linewidth=1.5)
    ax.plot(cycles, [s['mean_phi_stress'] for s in stats],
            'r-s', markersize=3, label='Stress Φ', linewidth=1.5)
    ax.set_ylabel('Mean Integration (Φ)')
    ax.set_xlabel('Evolution Cycle')
    ax.legend(framealpha=0.9)
    ax.set_title('Integration Under Stress')
    ax.grid(True, alpha=0.3)

    # 1b: Robustness + % increase
    ax = axes[0, 1]
    rob = [s['mean_robustness'] for s in stats]
    up = [s['phi_increase_frac'] * 100 for s in stats]
    ax.plot(cycles, rob, 'g-o', markersize=3, label='Robustness (Φ_s/Φ_b)', linewidth=1.5)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Threshold (1.0)')
    ax.set_ylabel('Robustness', color='g')
    ax.tick_params(axis='y', labelcolor='g')
    ax2 = ax.twinx()
    ax2.plot(cycles, up, 'm-^', markersize=3, alpha=0.7, label='% Φ increase', linewidth=1)
    ax2.set_ylabel('% Patterns with Φ↑', color='m')
    ax2.tick_params(axis='y', labelcolor='m')
    ax.set_xlabel('Evolution Cycle')
    ax.set_title('Stress Robustness')
    ax.legend(loc='upper left', framealpha=0.9)
    ax2.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # 1c: Population + Mortality
    ax = axes[1, 0]
    ax.plot(cycles, [s['n_patterns'] for s in stats],
            'k-o', markersize=3, label='Patterns', linewidth=1.5)
    ax3 = ax.twinx()
    ax3.fill_between(cycles, [s['mortality'] * 100 for s in stats],
                     alpha=0.3, color='red', label='Mortality %')
    ax3.plot(cycles, [s['mortality'] * 100 for s in stats],
             'r-', alpha=0.5, linewidth=1)
    ax.set_ylabel('# Patterns', color='k')
    ax3.set_ylabel('Mortality %', color='r')
    ax3.tick_params(axis='y', labelcolor='r')
    ax.set_xlabel('Evolution Cycle')
    ax.set_title('Population Dynamics')
    ax.legend(loc='upper left', framealpha=0.9)
    ax3.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # 1d: Parameter drift (tau, beta)
    ax = axes[1, 1]
    ax.plot(cycles, [s['tau'] for s in stats],
            'b-o', markersize=3, label='τ (similarity threshold)', linewidth=1.5)
    ax4 = ax.twinx()
    ax4.plot(cycles, [s['gate_beta'] for s in stats],
             'orange', marker='s', markersize=3, label='β (gate steepness)', linewidth=1.5)
    ax.set_ylabel('τ', color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ax4.set_ylabel('β', color='orange')
    ax4.tick_params(axis='y', labelcolor='orange')
    ax.set_xlabel('Evolution Cycle')
    ax.set_title('Content-Coupling Parameter Drift')
    ax.legend(loc='upper left', framealpha=0.9)
    ax4.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Add stress schedule as background shading
    if 'stress_regen' in stats[0]:
        for ax_row in axes:
            for ax_i in ax_row:
                stress_vals = [s['stress_regen'] for s in stats]
                for i in range(len(cycles) - 1):
                    intensity = 1.0 - stress_vals[i]  # lower regen = more stress
                    ax_i.axvspan(cycles[i], cycles[i+1],
                               alpha=intensity * 0.1, color='red')

    plt.tight_layout()
    path = os.path.join(output_dir, 'v13_evolution_trajectory.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def fig_grid_snapshots(data, result_dir, output_dir):
    """Figure 2: Grid state at key moments."""
    snapshot_dir = os.path.join(result_dir, 'snapshots')
    if not os.path.exists(snapshot_dir):
        print("  No snapshots directory found")
        return None

    # Find available snapshots
    files = sorted(os.listdir(snapshot_dir))
    npz_files = [f for f in files if f.endswith('.npz')]

    if len(npz_files) < 2:
        print(f"  Only {len(npz_files)} snapshots, need at least 2")
        return None

    # Select up to 6 snapshots evenly spaced
    n_show = min(6, len(npz_files))
    indices = np.linspace(0, len(npz_files) - 1, n_show, dtype=int)
    selected = [npz_files[i] for i in indices]

    fig, axes = plt.subplots(2, n_show, figsize=(3.5 * n_show, 7))
    fig.suptitle('V13 Grid Evolution: Channel Mean & Resources',
                 fontsize=13, fontweight='bold')

    if n_show == 1:
        axes = axes.reshape(2, 1)

    for col, fname in enumerate(selected):
        snap = np.load(os.path.join(snapshot_dir, fname))
        grid = snap['grid']  # (C, N, N)
        resource = snap['resource']  # (N, N)

        # Channel mean
        mean_ch = grid.mean(axis=0) if grid.ndim == 3 else grid

        label = fname.replace('.npz', '').replace('cycle_', 'Cycle ')

        ax_top = axes[0, col]
        im1 = ax_top.imshow(mean_ch, cmap=LENIA_CMAP, vmin=0, vmax=0.5)
        ax_top.set_title(label, fontsize=10)
        ax_top.axis('off')

        ax_bot = axes[1, col]
        im2 = ax_bot.imshow(resource, cmap='YlOrRd_r', vmin=0, vmax=1.0)
        ax_bot.set_title('Resources', fontsize=9)
        ax_bot.axis('off')

    # Colorbars
    fig.colorbar(im1, ax=axes[0, :], shrink=0.6, label='Channel Mean Activity')
    fig.colorbar(im2, ax=axes[1, :], shrink=0.6, label='Resource Level')

    plt.tight_layout()
    path = os.path.join(output_dir, 'v13_grid_snapshots.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def fig_arousal_effrank(data, output_dir):
    """Figure 3: Arousal and Effective Rank response to stress."""
    stats = data['cycle_stats']
    cycles = [s['cycle'] for s in stats]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('V13: Arousal & Effective Rank Under Stress',
                 fontsize=13, fontweight='bold')

    # Arousal
    ab = [s.get('mean_arousal_base', 0) for s in stats]
    a_s = [s.get('mean_arousal_stress', 0) for s in stats]
    ax1.plot(cycles, ab, 'b-o', markersize=3, label='Baseline', linewidth=1.5)
    ax1.plot(cycles, a_s, 'r-s', markersize=3, label='Stress', linewidth=1.5)
    ax1.set_xlabel('Evolution Cycle')
    ax1.set_ylabel('Mean Arousal')
    ax1.set_title('Arousal Response')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Effective Rank
    rb = [s.get('mean_effrank_base', 0) for s in stats]
    rs = [s.get('mean_effrank_stress', 0) for s in stats]
    ax2.plot(cycles, rb, 'b-o', markersize=3, label='Baseline', linewidth=1.5)
    ax2.plot(cycles, rs, 'r-s', markersize=3, label='Stress', linewidth=1.5)
    ax2.set_xlabel('Evolution Cycle')
    ax2.set_ylabel('Mean Effective Rank')
    ax2.set_title('Representational Dimensionality')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'v13_arousal_effrank.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def fig_stress_comparison(data, output_dir):
    """Figure 4: Early vs Late stress response bar chart."""
    stats = data['cycle_stats']
    if len(stats) < 6:
        print("  Need at least 6 cycles for early/late comparison")
        return None

    n = len(stats)
    early = stats[:3]
    late = stats[-3:]

    metrics = {
        'Φ Base': ('mean_phi_base', 'b'),
        'Φ Stress': ('mean_phi_stress', 'r'),
        'Robustness': ('mean_robustness', 'g'),
        'Mortality': ('mortality', 'orange'),
        '% Φ↑': ('phi_increase_frac', 'm'),
    }

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(metrics))
    width = 0.35

    early_vals = []
    late_vals = []
    for name, (key, _) in metrics.items():
        early_vals.append(np.mean([s[key] for s in early]))
        late_vals.append(np.mean([s[key] for s in late]))

    bars1 = ax.bar(x - width/2, early_vals, width, label=f'Early (cycles 0-2)',
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, late_vals, width, label=f'Late (cycles {n-3}-{n-1})',
                   color='coral', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics.keys())
    ax.legend()
    ax.set_title('V13 Content-Based Coupling: Early vs Late Evolution',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f'{h:.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, 'v13_early_vs_late.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def fig_summary_card(data, output_dir):
    """Figure 5: Single-panel summary card for the book."""
    stats = data['cycle_stats']
    if len(stats) < 2:
        return None

    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    cycles = [s['cycle'] for s in stats]

    # Main: Robustness trajectory
    ax_main = fig.add_subplot(gs[0, :2])
    rob = [s['mean_robustness'] for s in stats]
    ax_main.plot(cycles, rob, 'g-o', markersize=4, linewidth=2)
    ax_main.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax_main.fill_between(cycles, rob, 1.0, where=[r > 1.0 for r in rob],
                        alpha=0.2, color='green', label='Φ increases')
    ax_main.fill_between(cycles, rob, 1.0, where=[r <= 1.0 for r in rob],
                        alpha=0.2, color='red', label='Φ decreases')
    ax_main.set_xlabel('Evolution Cycle')
    ax_main.set_ylabel('Robustness (Φ_stress / Φ_base)')
    ax_main.set_title('V13: Integration Robustness Under Selection',
                      fontsize=12, fontweight='bold')
    ax_main.legend(fontsize=8)
    ax_main.grid(True, alpha=0.3)

    # Stats box
    ax_stats = fig.add_subplot(gs[0, 2])
    ax_stats.axis('off')
    first = stats[0]
    last = stats[-1]
    text = (
        f"Experiment: V13\n"
        f"Substrate: Content-coupled Lenia\n"
        f"Channels: {data.get('config', {}).get('n_channels', '?')}\n"
        f"Grid: {data.get('config', {}).get('grid_size', '?')}²\n"
        f"Cycles: {len(stats)}\n"
        f"Seed: {data.get('seed', '?')}\n"
        f"\n"
        f"Φ base: {first['mean_phi_base']:.3f} → {last['mean_phi_base']:.3f}\n"
        f"Φ stress: {first['mean_phi_stress']:.3f} → {last['mean_phi_stress']:.3f}\n"
        f"Robustness: {first['mean_robustness']:.3f} → {last['mean_robustness']:.3f}\n"
        f"τ: {first['tau']:.2f} → {last['tau']:.2f}\n"
        f"β: {first['gate_beta']:.1f} → {last['gate_beta']:.1f}\n"
    )
    ax_stats.text(0.05, 0.95, text, transform=ax_stats.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Bottom: Phi trajectories
    ax_phi = fig.add_subplot(gs[1, :2])
    ax_phi.plot(cycles, [s['mean_phi_base'] for s in stats],
               'b-', linewidth=1.5, label='Φ baseline')
    ax_phi.plot(cycles, [s['mean_phi_stress'] for s in stats],
               'r-', linewidth=1.5, label='Φ stress')
    ax_phi.set_xlabel('Cycle')
    ax_phi.set_ylabel('Mean Φ')
    ax_phi.legend(fontsize=8)
    ax_phi.grid(True, alpha=0.3)

    # Bottom right: tau drift
    ax_tau = fig.add_subplot(gs[1, 2])
    ax_tau.plot(cycles, [s['tau'] for s in stats], 'purple', linewidth=1.5)
    ax_tau.set_xlabel('Cycle')
    ax_tau.set_ylabel('τ (similarity threshold)')
    ax_tau.grid(True, alpha=0.3)

    path = os.path.join(output_dir, 'v13_summary_card.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def generate_all(result_dir, output_dir=None):
    """Generate all figures from a V13 results directory."""
    if output_dir is None:
        output_dir = os.path.join(result_dir, 'figures')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading results from: {result_dir}")
    data = load_results(result_dir)
    stats = data['cycle_stats']
    print(f"  {len(stats)} cycles loaded (status: {data.get('status', 'unknown')})")

    if not stats:
        print("No cycle data found!")
        return

    print(f"\nGenerating figures in: {output_dir}")

    fig_evolution_trajectory(data, output_dir)
    fig_grid_snapshots(data, result_dir, output_dir)
    fig_arousal_effrank(data, output_dir)
    fig_stress_comparison(data, output_dir)
    fig_summary_card(data, output_dir)

    print(f"\nDone! {len(os.listdir(output_dir))} figures generated.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python v13_visualize.py <result_dir> [output_dir]")
        print("Example: python v13_visualize.py results/v13_s42")
        sys.exit(1)

    result_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    generate_all(result_dir, output_dir)
