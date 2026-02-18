"""Visualize V18 insulation membrane structure.

Creates a multi-panel figure showing:
1. Pattern activity (mean across channels)
2. Insulation field (boundary=blue, interior=red)
3. Composite overlay (patterns colored by external/internal signal dominance)
4. Evolution of internal_gain across cycles

Usage:
    python v18_visualize_membrane.py [--seed 42] [--cycle 20]
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
import argparse
from pathlib import Path

# Insulation computation (pure numpy, no JAX needed)
MAX_EROSION_DIST = 8

def erode_mask(mask):
    """Single erosion step: min over 3x3 neighborhood."""
    result = mask.copy()
    for di in range(-1, 2):
        for dj in range(-1, 2):
            if di == 0 and dj == 0:
                continue
            result = np.minimum(result, np.roll(np.roll(mask, di, axis=0), dj, axis=1))
    return result


def compute_insulation_field(grid, activity_threshold, insulation_beta, boundary_width):
    """Compute insulation field from grid state (numpy version)."""
    activity = np.mean(grid, axis=0)  # (N, N)
    pattern_mask = (activity > activity_threshold).astype(np.float32)

    # Distance to edge via iterated erosion
    current = pattern_mask.copy()
    dist_to_edge = np.zeros_like(pattern_mask)
    for _ in range(MAX_EROSION_DIST):
        current = erode_mask(current)
        dist_to_edge += current

    # Sigmoid
    insulation = 1.0 / (1.0 + np.exp(-insulation_beta * (dist_to_edge - boundary_width)))
    return insulation, activity, pattern_mask, dist_to_edge


def make_membrane_figure(grid, resource, config, seed, cycle, save_path):
    """Create the multi-panel membrane visualization."""

    activity_threshold = config.get('activity_threshold', 0.05)
    insulation_beta = config.get('insulation_beta', 5.0)
    boundary_width = config.get('boundary_width', 1.0)
    internal_gain = config.get('internal_gain', 1.0)

    insulation, activity, pattern_mask, dist_to_edge = compute_insulation_field(
        grid, activity_threshold, insulation_beta, boundary_width
    )

    # Custom colormaps
    membrane_cmap = LinearSegmentedColormap.from_list(
        'membrane', [
            (0.0, '#1a1a2e'),   # deep background
            (0.15, '#16213e'),  # near-background
            (0.3, '#0f3460'),   # boundary region (blue)
            (0.5, '#e94560'),   # transition (magenta)
            (0.7, '#ff6b35'),   # interior transition (orange)
            (1.0, '#ffb627'),   # deep interior (gold)
        ]
    )

    activity_cmap = LinearSegmentedColormap.from_list(
        'activity', [
            (0.0, '#0d1117'),   # background
            (0.1, '#161b22'),
            (0.3, '#21262d'),
            (0.5, '#30d5c8'),   # low activity (teal)
            (0.7, '#58a6ff'),   # medium (blue)
            (0.85, '#d2a8ff'),  # high (purple)
            (1.0, '#ffffff'),   # peak (white)
        ]
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    fig.patch.set_facecolor('#0d1117')

    # Panel 1: Pattern activity
    ax = axes[0, 0]
    ax.set_facecolor('#0d1117')
    im = ax.imshow(activity, cmap=activity_cmap, vmin=0, vmax=activity.max() * 1.1,
                   interpolation='nearest')
    ax.set_title(f'Pattern Activity (mean of {grid.shape[0]} channels)',
                 color='white', fontsize=12, pad=10)
    ax.axis('off')
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color='white')
    cb.outline.set_edgecolor('#30363d')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white', fontsize=8)

    # Panel 2: Insulation field (THE MEMBRANE)
    ax = axes[0, 1]
    ax.set_facecolor('#0d1117')
    im = ax.imshow(insulation, cmap=membrane_cmap, vmin=0, vmax=1,
                   interpolation='nearest')
    ax.set_title('Insulation Field — The Membrane',
                 color='white', fontsize=12, pad=10)
    ax.axis('off')
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color='white')
    cb.outline.set_edgecolor('#30363d')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white', fontsize=8)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(['Boundary\n(external)', '', 'Transition', '', 'Interior\n(internal)'])

    # Panel 3: Composite — signal dominance overlay
    ax = axes[1, 0]
    ax.set_facecolor('#0d1117')

    # Create RGB composite: blue=external, red=internal, brightness=activity
    brightness = np.clip(activity / (activity.max() + 1e-8), 0, 1)
    # Where insulation is low (boundary), show blue/cyan
    # Where insulation is high (interior), show red/orange
    r = insulation * brightness
    g = 0.3 * brightness * (1 - insulation)  # slight green at boundary
    b = (1 - insulation) * brightness

    composite = np.stack([r, g, b], axis=-1)
    # Boost saturation for visibility
    composite = np.clip(composite * 2.5, 0, 1)
    # Add subtle boundary contour
    boundary_zone = (insulation > 0.1) & (insulation < 0.9)
    composite[boundary_zone] = np.clip(
        composite[boundary_zone] + np.array([0.2, 0.0, 0.2]), 0, 1
    )

    ax.imshow(composite, interpolation='nearest')
    ax.set_title('Signal Dominance: Blue=External, Red=Internal',
                 color='white', fontsize=12, pad=10)
    ax.axis('off')

    # Panel 4: Resource field + pattern outlines
    ax = axes[1, 1]
    ax.set_facecolor('#0d1117')

    resource_cmap = LinearSegmentedColormap.from_list(
        'resource', [
            (0.0, '#0d1117'),
            (0.3, '#1a472a'),
            (0.5, '#2ea043'),
            (0.7, '#56d364'),
            (1.0, '#aff5b4'),
        ]
    )
    im = ax.imshow(resource, cmap=resource_cmap, vmin=0, vmax=1,
                   interpolation='nearest')
    # Overlay pattern boundaries
    from scipy import ndimage
    if pattern_mask.any():
        edges = ndimage.binary_dilation(pattern_mask.astype(bool)) ^ pattern_mask.astype(bool)
        edge_overlay = np.zeros((*resource.shape, 4))
        edge_overlay[edges] = [1, 0.4, 0.4, 0.8]  # semi-transparent red
        ax.imshow(edge_overlay, interpolation='nearest')
    ax.set_title('Resource Field + Pattern Boundaries',
                 color='white', fontsize=12, pad=10)
    ax.axis('off')
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color='white')
    cb.outline.set_edgecolor('#30363d')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white', fontsize=8)

    # Super title
    interior_frac = np.mean(insulation > 0.5) * 100
    boundary_frac = np.mean((insulation > 0.1) & (insulation < 0.9)) * 100
    fig.suptitle(
        f'V18 Boundary-Dependent Lenia — Seed {seed}, Cycle {cycle}\n'
        f'internal_gain={internal_gain:.2f}  boundary_width={boundary_width:.2f}  '
        f'insulation_beta={insulation_beta:.1f}  '
        f'interior={interior_frac:.1f}%  boundary={boundary_frac:.1f}%',
        color='white', fontsize=14, y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def make_gain_evolution_figure(seeds, results_dir, save_path):
    """Show how internal_gain and insulation evolve across cycles for all seeds."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#0d1117')

    colors = {'42': '#58a6ff', '123': '#f78166', '7': '#56d364'}
    seed_labels = {'42': 'Seed 42', '123': 'Seed 123', '7': 'Seed 7'}

    for seed in seeds:
        seed_str = str(seed)
        prog_path = results_dir / f'v18_s{seed}' / 'v18_progress.json'
        if not prog_path.exists():
            continue
        with open(prog_path) as f:
            prog = json.load(f)

        cycles = [cs['cycle'] for cs in prog['cycle_stats']]
        gains = [cs.get('internal_gain', cs.get('mean_internal_gain', None)) for cs in prog['cycle_stats']]
        bws = [cs.get('boundary_width', cs.get('mean_boundary_width', None)) for cs in prog['cycle_stats']]
        ins_means = [cs.get('insulation_mean', None) for cs in prog['cycle_stats']]

        c = colors[seed_str]
        label = seed_labels[seed_str]

        # Panel 1: internal_gain
        ax = axes[0]
        if gains[0] is not None:
            ax.plot(cycles, gains, color=c, linewidth=2, label=label, marker='o', markersize=3)
        ax.set_title('Internal Gain Evolution', color='white', fontsize=12)
        ax.set_xlabel('Cycle', color='#8b949e')
        ax.set_ylabel('internal_gain', color='#8b949e')

        # Panel 2: boundary_width
        ax = axes[1]
        if bws[0] is not None:
            ax.plot(cycles, bws, color=c, linewidth=2, label=label, marker='o', markersize=3)
        ax.set_title('Boundary Width Evolution', color='white', fontsize=12)
        ax.set_xlabel('Cycle', color='#8b949e')
        ax.set_ylabel('boundary_width', color='#8b949e')

        # Panel 3: insulation_mean
        ax = axes[2]
        if ins_means[0] is not None:
            ax.plot(cycles, ins_means, color=c, linewidth=2, label=label, marker='o', markersize=3)
        ax.set_title('Mean Insulation Field', color='white', fontsize=12)
        ax.set_xlabel('Cycle', color='#8b949e')
        ax.set_ylabel('insulation_mean', color='#8b949e')

    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')
        ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='white', fontsize=9)
        ax.grid(True, alpha=0.15, color='#30363d')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def make_early_vs_late_comparison(grid_early, grid_late, config_early, config_late,
                                   seed, cycle_early, cycle_late, save_path):
    """Side-by-side comparison of membrane at early vs late cycle."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor('#0d1117')

    membrane_cmap = LinearSegmentedColormap.from_list(
        'membrane', [
            (0.0, '#1a1a2e'),
            (0.15, '#16213e'),
            (0.3, '#0f3460'),
            (0.5, '#e94560'),
            (0.7, '#ff6b35'),
            (1.0, '#ffb627'),
        ]
    )

    activity_cmap = LinearSegmentedColormap.from_list(
        'activity', [
            (0.0, '#0d1117'),
            (0.3, '#21262d'),
            (0.5, '#30d5c8'),
            (0.7, '#58a6ff'),
            (0.85, '#d2a8ff'),
            (1.0, '#ffffff'),
        ]
    )

    for row, (grid, config, cycle) in enumerate([
        (grid_early, config_early, cycle_early),
        (grid_late, config_late, cycle_late),
    ]):
        at = config.get('activity_threshold', 0.05)
        ib = config.get('insulation_beta', 5.0)
        bw = config.get('boundary_width', 1.0)
        gain = config.get('internal_gain', 1.0)

        ins, act, mask, dist = compute_insulation_field(grid, at, ib, bw)

        # Activity
        ax = axes[row, 0]
        ax.set_facecolor('#0d1117')
        ax.imshow(act, cmap=activity_cmap, vmin=0, vmax=0.5, interpolation='nearest')
        ax.set_title(f'Cycle {cycle} — Activity', color='white', fontsize=11)
        ax.axis('off')

        # Insulation
        ax = axes[row, 1]
        ax.set_facecolor('#0d1117')
        ax.imshow(ins, cmap=membrane_cmap, vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(f'Cycle {cycle} — Membrane (gain={gain:.2f}, bw={bw:.2f})',
                     color='white', fontsize=11)
        ax.axis('off')

        # Signal dominance composite
        ax = axes[row, 2]
        ax.set_facecolor('#0d1117')
        brightness = np.clip(act / (act.max() + 1e-8), 0, 1)
        r = ins * brightness
        g = 0.3 * brightness * (1 - ins)
        b = (1 - ins) * brightness
        composite = np.clip(np.stack([r, g, b], axis=-1) * 2.5, 0, 1)
        ax.imshow(composite, interpolation='nearest')
        int_frac = np.mean(ins > 0.5) * 100
        ax.set_title(f'Cycle {cycle} — Signal (interior={int_frac:.1f}%)',
                     color='white', fontsize=11)
        ax.axis('off')

    fig.suptitle(
        f'V18 Membrane Evolution — Seed {seed}\n'
        f'Early (Cycle {cycle_early}) vs Late (Cycle {cycle_late})',
        color='white', fontsize=14, y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cycle', type=int, default=20)
    parser.add_argument('--results-dir', type=str,
                        default='results')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    seed_dir = results_dir / f'v18_s{args.seed}'
    out_dir = results_dir / 'v18_figures'
    out_dir.mkdir(exist_ok=True)

    # Load snapshot
    snap = np.load(seed_dir / 'snapshots' / f'cycle_{args.cycle:03d}.npz')
    grid = snap['grid']
    resource = snap['resource']

    # Load config from progress JSON (cycle-specific evolved params)
    with open(seed_dir / 'v18_progress.json') as f:
        prog = json.load(f)

    # Get config for the requested cycle
    base_config = prog.get('config', {})
    cycle_config = dict(base_config)
    for cs in prog['cycle_stats']:
        if cs.get('cycle') == args.cycle:
            for k in ['internal_gain', 'boundary_width', 'insulation_beta',
                       'activity_threshold']:
                if k in cs:
                    cycle_config[k] = cs[k]
            break

    # Figure 1: Single-cycle membrane visualization
    make_membrane_figure(
        grid, resource, cycle_config, args.seed, args.cycle,
        out_dir / f'v18_membrane_s{args.seed}_c{args.cycle:03d}.png'
    )

    # Figure 2: Gain evolution across all seeds
    make_gain_evolution_figure(
        [42, 123, 7], results_dir,
        out_dir / 'v18_gain_evolution.png'
    )

    # Figure 3: Early vs late comparison
    early_cycle = 5
    late_cycle = args.cycle
    snap_early = np.load(seed_dir / 'snapshots' / f'cycle_{early_cycle:03d}.npz')
    early_config = dict(base_config)
    for cs in prog['cycle_stats']:
        if cs.get('cycle') == early_cycle:
            for k in ['internal_gain', 'boundary_width', 'insulation_beta',
                       'activity_threshold']:
                if k in cs:
                    early_config[k] = cs[k]
            break

    make_early_vs_late_comparison(
        snap_early['grid'], grid,
        early_config, cycle_config,
        args.seed, early_cycle, late_cycle,
        out_dir / f'v18_early_vs_late_s{args.seed}.png'
    )
