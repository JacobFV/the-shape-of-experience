"""V13 Cross-Seed Aggregation: Combine results from multiple seeds.

Produces:
1. Mean robustness trajectory with error bands
2. Cross-seed summary statistics
3. Population-robustness correlation plot
4. Comparison with V11/V12 baselines
"""
import json
import os
import sys
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("pip install matplotlib")
    sys.exit(1)


def load_seed(result_dir):
    """Load a single seed's results."""
    final = os.path.join(result_dir, 'v13_final.json')
    progress = os.path.join(result_dir, 'v13_progress.json')
    path = final if os.path.exists(final) else progress
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def aggregate(result_dirs, labels=None):
    """Load and aggregate multiple seeds."""
    seeds = []
    for d in result_dirs:
        data = load_seed(d)
        if data and data['cycle_stats']:
            seeds.append(data)

    if not seeds:
        print("No valid results found!")
        return None

    print(f"Loaded {len(seeds)} seeds")
    for i, s in enumerate(seeds):
        stats = s['cycle_stats']
        label = labels[i] if labels else f"seed={s.get('seed', '?')}"
        alive = sum(1 for c in stats if c['n_patterns'] > 0)
        print(f"  {label}: {len(stats)} cycles, {alive} alive, "
              f"final rob={stats[-1]['mean_robustness']:.3f}")

    return seeds


def fig_robustness_comparison(seeds, labels, output_dir):
    """Cross-seed robustness trajectories."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle('V13 Content-Based Coupling: Cross-Seed Robustness',
                 fontsize=14, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Left: individual trajectories
    all_rob = []
    for i, data in enumerate(seeds):
        stats = data['cycle_stats']
        cycles = [s['cycle'] for s in stats]
        rob = [s['mean_robustness'] for s in stats]
        # Only plot cycles where patterns exist
        alive_mask = [s['n_patterns'] > 0 for s in stats]
        c_alive = [c for c, a in zip(cycles, alive_mask) if a]
        r_alive = [r for r, a in zip(rob, alive_mask) if a]
        ax1.plot(c_alive, r_alive, '-o', color=colors[i % len(colors)],
                 markersize=3, linewidth=1.5, label=labels[i], alpha=0.7)
        all_rob.append((cycles, rob, alive_mask))

    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax1.set_xlabel('Evolution Cycle')
    ax1.set_ylabel('Robustness (Φ_stress / Φ_base)')
    ax1.set_title('Individual Seed Trajectories')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: mean ± std across seeds (aligned by cycle)
    max_cycles = max(len(d['cycle_stats']) for d in seeds)
    rob_matrix = np.full((len(seeds), max_cycles), np.nan)
    for i, (cycles, rob, alive) in enumerate(all_rob):
        for j, (c, r, a) in enumerate(zip(cycles, rob, alive)):
            if a and r > 0:
                rob_matrix[i, j] = r

    mean_rob = np.nanmean(rob_matrix, axis=0)
    std_rob = np.nanstd(rob_matrix, axis=0)
    n_valid = np.sum(~np.isnan(rob_matrix), axis=0)
    cycle_range = np.arange(max_cycles)

    # Only plot where we have at least 2 seeds
    mask = n_valid >= 2
    ax2.plot(cycle_range[mask], mean_rob[mask], 'k-o', markersize=3,
             linewidth=2, label=f'Mean (n={len(seeds)} seeds)')
    ax2.fill_between(cycle_range[mask],
                     (mean_rob - std_rob)[mask],
                     (mean_rob + std_rob)[mask],
                     alpha=0.2, color='blue', label='±1 SD')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=2)

    # Add historical baselines
    ax2.axhline(y=0.938, color='red', linestyle=':', alpha=0.6,
                label='V11.0 baseline (-6.2%)')
    ax2.axhline(y=0.962, color='orange', linestyle=':', alpha=0.6,
                label='V11.2 hetero (-3.8%)')

    ax2.set_xlabel('Evolution Cycle')
    ax2.set_ylabel('Mean Robustness')
    ax2.set_title('Mean Trajectory with Historical Baselines')
    ax2.legend(fontsize=8, loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'v13_cross_seed_robustness.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def fig_population_robustness(seeds, labels, output_dir):
    """Scatter plot: population size vs robustness (key finding)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    all_pop = []
    all_rob = []

    for i, data in enumerate(seeds):
        stats = data['cycle_stats']
        pop = [s['n_patterns'] for s in stats if s['n_patterns'] > 0 and s['mean_robustness'] > 0]
        rob = [s['mean_robustness'] for s in stats if s['n_patterns'] > 0 and s['mean_robustness'] > 0]
        ax.scatter(pop, rob, color=colors[i % len(colors)], alpha=0.5,
                   s=30, label=labels[i])
        all_pop.extend(pop)
        all_rob.extend(rob)

    # Fit trend line
    if len(all_pop) > 5:
        z = np.polyfit(all_pop, all_rob, 1)
        p = np.poly1d(z)
        x_range = np.linspace(min(all_pop), max(all_pop), 100)
        ax.plot(x_range, p(x_range), 'k--', alpha=0.5,
                label=f'Trend (slope={z[0]:.4f})')

        # Correlation
        r = np.corrcoef(all_pop, all_rob)[0, 1]
        ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Population Size (# patterns)', fontsize=11)
    ax.set_ylabel('Robustness (Φ_stress / Φ_base)', fontsize=11)
    ax.set_title('V13: Population Size vs Integration Robustness',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'v13_population_robustness.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def fig_summary_table(seeds, labels, output_dir):
    """Summary statistics as a figure (for the book)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    headers = ['Seed', 'Cycles', 'Alive', 'Pop (mean)', 'Rob (mean)',
               'Rob (max)', '% Φ↑', 'τ final', 'β final']
    rows = []

    for i, data in enumerate(seeds):
        stats = data['cycle_stats']
        alive_stats = [s for s in stats if s['n_patterns'] > 0]
        rows.append([
            labels[i],
            str(len(stats)),
            str(len(alive_stats)),
            f"{np.mean([s['n_patterns'] for s in alive_stats]):.0f}" if alive_stats else "—",
            f"{np.mean([s['mean_robustness'] for s in alive_stats]):.3f}" if alive_stats else "—",
            f"{max(s['mean_robustness'] for s in alive_stats):.3f}" if alive_stats else "—",
            f"{np.mean([s['phi_increase_frac'] for s in alive_stats]):.0%}" if alive_stats else "—",
            f"{stats[-1]['tau']:.2f}",
            f"{stats[-1]['gate_beta']:.1f}",
        ])

    # Cross-seed mean
    all_alive = []
    for data in seeds:
        all_alive.extend([s for s in data['cycle_stats'] if s['n_patterns'] > 0])
    if all_alive:
        rows.append([
            'MEAN',
            '—',
            str(len(all_alive)),
            f"{np.mean([s['n_patterns'] for s in all_alive]):.0f}",
            f"{np.mean([s['mean_robustness'] for s in all_alive]):.3f}",
            f"{max(s['mean_robustness'] for s in all_alive):.3f}",
            f"{np.mean([s['phi_increase_frac'] for s in all_alive]):.0%}",
            '—',
            '—',
        ])

    table = ax.table(cellText=rows, colLabels=headers,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Style header
    for j in range(len(headers)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Style mean row
    if rows:
        for j in range(len(headers)):
            table[len(rows), j].set_facecolor('#D6E4F0')
            table[len(rows), j].set_text_props(fontweight='bold')

    ax.set_title('V13 Content-Based Coupling: Cross-Seed Summary',
                 fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    path = os.path.join(output_dir, 'v13_summary_table.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def main(result_dirs, labels=None, output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(result_dirs[0])
        output_dir = os.path.join(output_dir, 'aggregate')
    os.makedirs(output_dir, exist_ok=True)

    if labels is None:
        labels = [os.path.basename(d) for d in result_dirs]

    seeds = aggregate(result_dirs, labels)
    if not seeds:
        return

    print(f"\nGenerating aggregate figures in: {output_dir}")
    fig_robustness_comparison(seeds, labels, output_dir)
    fig_population_robustness(seeds, labels, output_dir)
    fig_summary_table(seeds, labels, output_dir)
    print("\nDone!")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python v13_aggregate.py <dir1> <dir2> ... [--output <dir>]")
        print("Example: python v13_aggregate.py results/v13_s42/v13_s42 results/v13_s123/v13_s123")
        sys.exit(1)

    dirs = []
    output = None
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--output':
            output = sys.argv[i + 1]
            i += 2
        else:
            dirs.append(sys.argv[i])
            i += 1

    main(dirs, output_dir=output)
