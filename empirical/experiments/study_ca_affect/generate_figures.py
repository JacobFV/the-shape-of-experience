"""Generate publication figures for book Part 1 from V11 experiment data.

Produces labeled figures in book/part1/images/:
1. fig_v11_ladder.pdf — Bar chart of delta-Phi across V11.0-V11.2 conditions
2. fig_v11_csweep.pdf — Phi change vs channel dimensionality C
3. fig_v11_snapshots.pdf — Grid snapshots at different C (baseline vs drought)
4. fig_v11_phi_timeseries.pdf — Phi trajectory through stress cycle

Usage:
    python generate_figures.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# Output directory
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'book', 'part1', 'images')
os.makedirs(OUT_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


def fig_ladder():
    """Figure 1: The Ladder of Substrate Complexity.

    Bar chart showing delta-Phi under severe drought for each V11 condition.
    The biological pattern (positive delta) is the target.
    """
    conditions = [
        'V11.0\nNaive',
        'V11.1\nHomogeneous\nEvolution',
        'V11.2\nHeterogeneous\nEvolution',
    ]
    deltas = [-6.2, -6.0, -3.8]
    naive_baseline = -5.9  # for V11.2 comparison

    colors = ['#cc4444', '#cc6644', '#cc8844']

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    bars = ax.bar(range(len(conditions)), deltas, color=colors,
                  edgecolor='black', linewidth=0.5, width=0.6, zorder=3)

    # Add value labels on bars
    for i, (bar, d) in enumerate(zip(bars, deltas)):
        ax.text(bar.get_x() + bar.get_width()/2, d - 0.3,
                f'{d:+.1f}%', ha='center', va='top',
                fontsize=10, fontweight='bold', color='white')

    # Biological target line
    ax.axhline(y=0, color='#228B22', linewidth=2, linestyle='--',
               label='Biological pattern\n(integration under threat)', zorder=2)

    # Naive baseline for V11.2
    ax.axhline(y=naive_baseline, color='gray', linewidth=1, linestyle=':',
               label=f'Naive baseline ({naive_baseline}%)', alpha=0.7, zorder=2)

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=8)
    ax.set_ylabel(r'$\Delta\Phi$ under severe drought (%)')
    ax.set_ylim(-8, 2)
    ax.axhspan(0, 2, color='#22cc22', alpha=0.08, zorder=1)
    ax.text(2.4, 1.0, 'Integration\n(biological)', fontsize=7,
            color='#228B22', ha='right', style='italic')
    ax.text(2.4, -1.0, 'Decomposition\n(non-biological)', fontsize=7,
            color='#cc4444', ha='right', style='italic')

    ax.legend(loc='lower left', framealpha=0.9, fontsize=7)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Arrow showing improvement
    ax.annotate('', xy=(2, -3.8), xytext=(0, -6.2),
                arrowprops=dict(arrowstyle='->', color='#2266cc',
                               lw=2, connectionstyle='arc3,rad=0.2'))
    ax.text(1.0, -3.5, '+2.1pp', fontsize=9, color='#2266cc',
            fontweight='bold', ha='center')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_v11_ladder.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_csweep():
    """Figure 2: Channel Dimensionality vs Stress Response.

    Shows delta-Phi for C=3,8,16,32,64 (naive, no evolution).
    Mid-range C shows integration; high C decomposes.
    """
    C_values = [3, 8, 16, 32, 64]
    means = [-0.0, +0.5, +1.2, -2.5, -1.0]
    stds = [0.0, 1.3, 6.7, 1.4, 1.9]

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    colors = []
    for m in means:
        if m > 0:
            colors.append('#228B22')
        elif m < -0.5:
            colors.append('#cc4444')
        else:
            colors.append('#888888')

    bars = ax.bar(range(len(C_values)), means, yerr=stds,
                  color=colors, edgecolor='black', linewidth=0.5,
                  width=0.6, capsize=4, zorder=3,
                  error_kw=dict(lw=1.5, capthick=1.5))

    ax.axhline(y=0, color='black', linewidth=0.8, zorder=2)

    ax.set_xticks(range(len(C_values)))
    ax.set_xticklabels([f'C={c}' for c in C_values])
    ax.set_ylabel(r'Mean $\Delta\Phi$ under drought (%)')
    ax.set_xlabel('Channel dimensionality')

    # Shade regions
    ax.axhspan(0, 12, color='#22cc22', alpha=0.06, zorder=1)
    ax.axhspan(-5, 0, color='#cc4444', alpha=0.06, zorder=1)

    # Annotations
    ax.annotate('Integration\n(biological)', xy=(2, 1.2),
                xytext=(3.5, 6), fontsize=8, color='#228B22',
                style='italic', ha='center',
                arrowprops=dict(arrowstyle='->', color='#228B22', lw=1.2))
    ax.annotate('Space too large for\nrandom coupling', xy=(3, -2.5),
                xytext=(3.8, -4.5), fontsize=7, color='#cc4444',
                style='italic', ha='center')

    ax.set_ylim(-8, 12)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_v11_csweep.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_snapshots():
    """Figure 3: HD Lenia grid snapshots at different C.

    Shows PCA-projected RGB grids during baseline and drought
    for C=3, 8, 16, 32.
    """
    import jax
    import jax.numpy as jnp
    from jax import random

    sys.path.insert(0, os.path.dirname(__file__))
    from v11_substrate_hd import (
        generate_hd_config, generate_coupling_matrix,
        make_kernels_fft_hd, run_chunk_hd_wrapper, init_soup_hd
    )
    from v11_visualize import channels_to_rgb

    C_values = [3, 8, 16, 32]
    N = 128

    fig, axes = plt.subplots(2, 4, figsize=(7, 3.8))

    for col, C in enumerate(C_values):
        print(f"  Generating C={C}...", end=" ", flush=True)
        config = generate_hd_config(C=C, N=N, seed=42)
        coupling = jnp.array(generate_coupling_matrix(C, bandwidth=max(2.0, C/8), seed=42))
        kernel_ffts = make_kernels_fft_hd(config)
        rng = random.PRNGKey(42)
        rng, k = random.split(rng)
        grid, resource = init_soup_hd(N, C, k, jnp.array(config['channel_mus']))

        # Warmup
        grid, resource, rng = run_chunk_hd_wrapper(
            grid, resource, kernel_ffts, coupling, rng, config, 100)
        grid.block_until_ready()
        grid, resource, rng = run_chunk_hd_wrapper(
            grid, resource, kernel_ffts, coupling, rng, config, 4900)
        grid.block_until_ready()

        # Baseline snapshot
        grid_np = np.array(grid)
        rgb_baseline = channels_to_rgb(grid_np)

        # Drought
        drought_config = {**config, 'resource_regen': 0.0001}
        grid_d, resource_d, rng_d = run_chunk_hd_wrapper(
            grid, resource, kernel_ffts, coupling, rng, drought_config, 2000)
        grid_d.block_until_ready()
        grid_np_d = np.array(grid_d)
        rgb_drought = channels_to_rgb(grid_np_d)

        # Plot
        axes[0, col].imshow(rgb_baseline, interpolation='nearest')
        axes[0, col].set_title(f'C = {C}', fontsize=10, fontweight='bold')
        axes[0, col].axis('off')

        axes[1, col].imshow(rgb_drought, interpolation='nearest')
        axes[1, col].axis('off')

        print("done")

    # Row labels
    axes[0, 0].set_ylabel('Baseline', fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel('Drought', fontsize=10, fontweight='bold')

    # Restore ylabel visibility
    for row in range(2):
        axes[row, 0].yaxis.set_visible(True)
        axes[row, 0].tick_params(left=False, labelleft=False)

    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    path = os.path.join(OUT_DIR, 'fig_v11_snapshots.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_phi_timeseries():
    """Figure 4: Phi dynamics through stress cycle at different C.

    Shows spectral Phi time series for C=8 and C=32 through
    baseline -> drought -> recovery.
    """
    import jax
    import jax.numpy as jnp
    from jax import random

    sys.path.insert(0, os.path.dirname(__file__))
    from v11_substrate_hd import (
        generate_hd_config, generate_coupling_matrix,
        make_kernels_fft_hd, run_chunk_hd_wrapper, init_soup_hd
    )
    from v11_patterns import detect_patterns_mc
    from v11_affect_hd import spectral_channel_phi

    N = 128
    C_list = [8, 32]
    steps_per_frame = 50
    n_baseline = 40
    n_drought = 60
    n_recovery = 40
    total = n_baseline + n_drought + n_recovery

    fig, ax = plt.subplots(figsize=(5, 3))

    line_styles = {'8': ('-', '#2266cc', 'C=8 (integration)'),
                   '32': ('--', '#cc4444', 'C=32 (decomposition)')}

    for C in C_list:
        print(f"  Simulating C={C}...", end=" ", flush=True)
        config = generate_hd_config(C=C, N=N, seed=42)
        coupling = jnp.array(generate_coupling_matrix(C, bandwidth=max(2.0, C/8), seed=42))
        kernel_ffts = make_kernels_fft_hd(config)
        rng = random.PRNGKey(42)
        rng, k = random.split(rng)
        grid, resource = init_soup_hd(N, C, k, jnp.array(config['channel_mus']))

        # Warmup
        grid, resource, rng = run_chunk_hd_wrapper(
            grid, resource, kernel_ffts, coupling, rng, config, 5000)
        grid.block_until_ready()

        drought_config = {**config, 'resource_regen': 0.0001}
        phi_hist = []

        for f in range(total):
            cfg = drought_config if n_baseline <= f < n_baseline + n_drought else config
            grid, resource, rng = run_chunk_hd_wrapper(
                grid, resource, kernel_ffts, coupling, rng, cfg, steps_per_frame)

            if f % 3 == 0:
                grid.block_until_ready()

            grid_np = np.array(grid)
            patterns = detect_patterns_mc(grid_np, threshold=0.15)
            phis = []
            for p in patterns[:20]:
                if p.size < 10:
                    continue
                phi_s, _ = spectral_channel_phi(
                    jnp.array(grid_np), coupling, p.cells)
                phis.append(phi_s)
            phi_hist.append(np.mean(phis) if phis else 0.0)

        ls, color, label = line_styles[str(C)]
        ax.plot(range(total), phi_hist, ls, color=color, linewidth=2, label=label)
        print("done")

    # Phase shading
    ax.axvspan(0, n_baseline, alpha=0.08, color='green')
    ax.axvspan(n_baseline, n_baseline + n_drought, alpha=0.1, color='red')
    ax.axvspan(n_baseline + n_drought, total, alpha=0.08, color='blue')

    # Phase labels
    ax.text(n_baseline/2, ax.get_ylim()[1]*0.95, 'Baseline',
            ha='center', va='top', fontsize=8, color='green', fontweight='bold')
    ax.text(n_baseline + n_drought/2, ax.get_ylim()[1]*0.95, 'Drought',
            ha='center', va='top', fontsize=8, color='red', fontweight='bold')
    ax.text(n_baseline + n_drought + n_recovery/2, ax.get_ylim()[1]*0.95, 'Recovery',
            ha='center', va='top', fontsize=8, color='blue', fontweight='bold')

    ax.set_xlabel('Frame (50 steps each)')
    ax.set_ylabel('Mean Spectral $\\Phi$')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_v11_phi_timeseries.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_yerkes_dodson():
    """Figure 5: Yerkes-Dodson curve for CA patterns.

    Shows Phi change as function of stress intensity.
    Mild stress increases Phi; severe stress decomposes.
    Based on V11.2 GPU data.
    """
    # Data from V11.2 GPU runs and V11.0 observations
    # Stress levels: none, mild drought, moderate, severe
    stress_levels = [0, 0.3, 0.6, 1.0]
    stress_labels = ['None', 'Mild', 'Moderate', 'Severe']

    # Phi change (%) — from experimental observations
    # Mild stress: +30-90% (Yerkes-Dodson), Severe: -6%
    phi_change_evolved = [0, +65, +30, -3.8]
    phi_change_naive = [0, +40, +10, -5.9]

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    ax.plot(stress_levels, phi_change_evolved, 'o-', color='#2266cc',
            linewidth=2, markersize=8, label='Evolved (V11.2)', zorder=3)
    ax.plot(stress_levels, phi_change_naive, 's--', color='#cc4444',
            linewidth=2, markersize=7, label='Naive (V11.0)', zorder=3)

    ax.axhline(y=0, color='black', linewidth=0.5, zorder=1)
    ax.axhspan(0, 80, color='#22cc22', alpha=0.06, zorder=0)
    ax.axhspan(-10, 0, color='#cc4444', alpha=0.06, zorder=0)

    ax.set_xticks(stress_levels)
    ax.set_xticklabels(stress_labels)
    ax.set_xlabel('Stress intensity (drought severity)')
    ax.set_ylabel(r'$\Delta\Phi$ (%)')
    ax.set_ylim(-12, 80)

    ax.annotate('Yerkes--Dodson peak', xy=(0.3, 65),
                xytext=(0.6, 72), fontsize=8, style='italic',
                color='#2266cc',
                arrowprops=dict(arrowstyle='->', color='#2266cc', lw=1))

    ax.annotate('Decomposition', xy=(1.0, -3.8),
                xytext=(0.75, -9), fontsize=8, style='italic',
                color='#cc4444')

    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_v11_yerkes_dodson.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


if __name__ == '__main__':
    print("Generating publication figures for Part 1...")
    print()

    print("[1/5] The Ladder of Substrate Complexity")
    fig_ladder()

    print("[2/5] Channel Dimensionality vs Stress Response")
    fig_csweep()

    print("[3/5] Yerkes-Dodson Curve")
    fig_yerkes_dodson()

    print("[4/5] HD Lenia Grid Snapshots")
    fig_snapshots()

    print("[5/5] Phi Time Series Through Stress")
    fig_phi_timeseries()

    print()
    print("Done! Figures saved to book/part1/images/")
