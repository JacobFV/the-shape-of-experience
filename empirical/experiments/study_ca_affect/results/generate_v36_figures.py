#!/usr/bin/env python3
"""
V36 Egocentric Trajectory — Figure Generation
==============================================
Seed 23 (highest-Phi from V32, max Phi=0.473) across 30 evolutionary cycles.
16 recorded cycles, 5 droughts (cycles 5, 10, 15, 20, 25).

Generates 6 publication-quality figures from trajectory data.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

# Paths
DATA_DIR = Path("/Users/jacob/fun/shape-of-experience/empirical/experiments/study_ca_affect/results/v36_trajectory")
OUT_DIR = Path("/Users/jacob/fun/shape-of-experience/empirical/experiments/study_ca_affect/results/v36_figures")
OUT_DIR.mkdir(exist_ok=True)

# Style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colors
C_DROUGHT = '#E8524A'       # coral red
C_DROUGHT_BG = '#FDE8E6'    # light red background
C_NORMAL = '#2E8B8B'        # teal
C_NORMAL_LIGHT = '#B0D8D8'  # light teal
C_PHI = '#4A6FE8'           # blue for phi
C_VALENCE = '#2E8B57'       # sea green
C_AROUSAL = '#E8A24A'       # amber
C_ERANK = '#8B4AE8'         # purple
C_MORTALITY = '#E8524A'     # red
C_ENERGY = '#E8A24A'        # amber
C_ACCENT = '#333333'        # near-black for text


def load_summary():
    with open(DATA_DIR / "summary.json") as f:
        return json.load(f)


def load_cycle_meta(cycle_num):
    """Load just the meta field from a cycle file (memory efficient)."""
    fname = DATA_DIR / f"cycle_{cycle_num:03d}.json"
    with open(fname) as f:
        data = json.load(f)
    return data['meta']


def load_focal_timeline(cycle_num):
    """Load focal_timeline from a cycle file. ~20MB per file, returns list of dicts."""
    fname = DATA_DIR / f"cycle_{cycle_num:03d}.json"
    with open(fname) as f:
        data = json.load(f)
    return data['focal_timeline']


def add_drought_bands(ax, cycles, drought_cycles, alpha=0.15):
    """Add red background bands for drought cycles."""
    for dc in drought_cycles:
        ax.axvspan(dc - 0.5, dc + 0.5, color=C_DROUGHT, alpha=alpha, zorder=0)


# ============================================================
# Figure 1: Phi Trajectory
# ============================================================
def fig1_phi_trajectory(summary):
    print("  Generating v36_phi_trajectory.png ...")
    cycles_data = summary['cycles']
    cycle_nums = [c['cycle'] for c in cycles_data]
    phis = [c['phi'] for c in cycles_data]
    is_drought = [c['is_drought'] for c in cycles_data]
    drought_cycles = [c['cycle'] for c in cycles_data if c['is_drought']]

    fig, ax = plt.subplots(figsize=(12, 5))

    # Drought bands
    add_drought_bands(ax, cycle_nums, drought_cycles, alpha=0.12)

    # Line connecting all points
    ax.plot(cycle_nums, phis, color='#888888', linewidth=1.0, alpha=0.5, zorder=1)

    # Scatter: color-coded
    for i, (cn, phi, dr) in enumerate(zip(cycle_nums, phis, is_drought)):
        color = C_DROUGHT if dr else C_PHI
        marker = 'D' if dr else 'o'
        size = 80 if dr else 60
        ax.scatter(cn, phi, c=color, s=size, marker=marker, zorder=3, edgecolors='white', linewidth=0.8)

    # Trend line
    z = np.polyfit(cycle_nums, phis, 1)
    trend_x = np.linspace(min(cycle_nums), max(cycle_nums), 100)
    trend_y = np.polyval(z, trend_x)
    ax.plot(trend_x, trend_y, '--', color=C_ACCENT, linewidth=1.0, alpha=0.5, label=f'Trend: {z[0]:+.4f}/cycle')

    # Annotations
    max_phi_idx = np.argmax(phis)
    ax.annotate(f'Phi={phis[max_phi_idx]:.3f}',
                xy=(cycle_nums[max_phi_idx], phis[max_phi_idx]),
                xytext=(cycle_nums[max_phi_idx]+1.5, phis[max_phi_idx]+0.008),
                fontsize=9, color=C_ACCENT,
                arrowprops=dict(arrowstyle='->', color=C_ACCENT, lw=0.8))

    min_phi_idx = np.argmin(phis)
    ax.annotate(f'Phi={phis[min_phi_idx]:.3f}',
                xy=(cycle_nums[min_phi_idx], phis[min_phi_idx]),
                xytext=(cycle_nums[min_phi_idx]+1.5, phis[min_phi_idx]-0.008),
                fontsize=9, color=C_ACCENT,
                arrowprops=dict(arrowstyle='->', color=C_ACCENT, lw=0.8))

    ax.set_xlabel('Evolutionary Cycle')
    ax.set_ylabel(r'$\Phi$ (Integration)')
    ax.set_title(r'V36: Integration Trajectory — Seed 23 Across 30 Evolutionary Cycles')

    # Legend
    legend_elements = [
        plt.scatter([], [], c=C_PHI, s=60, marker='o', edgecolors='white', label='Normal cycle'),
        plt.scatter([], [], c=C_DROUGHT, s=80, marker='D', edgecolors='white', label='Drought cycle'),
        plt.Line2D([0], [0], linestyle='--', color=C_ACCENT, alpha=0.5, label=f'Trend: {z[0]:+.4f}/cycle'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

    ax.set_xlim(-1, 30)
    ax.set_xticks(range(0, 30, 2))
    ax.set_ylim(0, max(phis) * 1.15)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "v36_phi_trajectory.png")
    plt.close(fig)
    print("    Done.")


# ============================================================
# Figure 2: Affect Dimensions (4 panels)
# ============================================================
def fig2_affect_dimensions(summary):
    print("  Generating v36_affect_dimensions.png ...")
    cycles_data = summary['cycles']
    cycle_nums = [c['cycle'] for c in cycles_data]
    drought_cycles = [c['cycle'] for c in cycles_data if c['is_drought']]

    valences = [c['segment_affect']['valence'] for c in cycles_data]
    arousals = [c['segment_affect']['arousal'] for c in cycles_data]
    eranks = [c['segment_affect']['effective_rank'] for c in cycles_data]
    mortalities = [c['mortality'] for c in cycles_data]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    panels = [
        (axes[0], valences, 'Valence', C_VALENCE, 'Valence'),
        (axes[1], arousals, 'Arousal', C_AROUSAL, 'Arousal'),
        (axes[2], eranks, 'Effective Rank', C_ERANK, 'Eff. Rank'),
        (axes[3], mortalities, 'Mortality', C_MORTALITY, 'Mortality'),
    ]

    for ax, values, title, color, ylabel in panels:
        add_drought_bands(ax, cycle_nums, drought_cycles, alpha=0.12)

        is_drought = [c['is_drought'] for c in cycles_data]
        ax.plot(cycle_nums, values, color='#AAAAAA', linewidth=1.0, alpha=0.5, zorder=1)

        for cn, val, dr in zip(cycle_nums, values, is_drought):
            c = C_DROUGHT if dr else color
            m = 'D' if dr else 'o'
            s = 70 if dr else 50
            ax.scatter(cn, val, c=c, s=s, marker=m, zorder=3, edgecolors='white', linewidth=0.6)

        ax.set_ylabel(ylabel)
        ax.set_title(title, loc='left', fontweight='bold', fontsize=11)

    axes[3].set_xlabel('Evolutionary Cycle')
    axes[3].set_xlim(-1, 30)
    axes[3].set_xticks(range(0, 30, 2))

    # Mortality: format as percentage
    axes[3].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))

    fig.suptitle(r'V36: Affect Dimensions Across Evolution — Seed 23', fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "v36_affect_dimensions.png")
    plt.close(fig)
    print("    Done.")


# ============================================================
# Figure 3: Drought Portraits (5 rows)
# ============================================================
def fig3_drought_portraits(summary):
    print("  Generating v36_drought_portraits.png ...")
    cycles_data = summary['cycles']
    drought_data = [c for c in cycles_data if c['is_drought']]

    fig, axes = plt.subplots(5, 1, figsize=(10, 12))

    metrics = [
        ('n_alive_end', 'Survivors', '#4A6FE8', '{:.0f}'),
        ('mortality', 'Mortality %', C_DROUGHT, '{:.1%}'),
        ('phi', r'$\Phi$ (Integration)', C_PHI, '{:.4f}'),
        ('mean_energy', 'Mean Energy', C_ENERGY, '{:.3f}'),
        ('energy_std', 'Energy Std Dev', '#8B4AE8', '{:.3f}'),
    ]

    drought_labels = [f'Cycle {d["cycle"]}' for d in drought_data]
    x = np.arange(len(drought_data))

    for ax_idx, (key, label, color, fmt) in enumerate(metrics):
        ax = axes[ax_idx]
        if key in ('mean_energy', 'energy_std'):
            values = [d['segment_affect'][key] for d in drought_data]
        else:
            values = [d[key] for d in drought_data]

        bars = ax.bar(x, values, color=color, alpha=0.85, edgecolor='white', linewidth=1.5, width=0.6)

        # Value labels on bars
        for bar_obj, val in zip(bars, values):
            y_pos = bar_obj.get_height()
            text = fmt.format(val)
            ax.text(bar_obj.get_x() + bar_obj.get_width()/2, y_pos,
                    text, ha='center', va='bottom', fontsize=10, fontweight='bold', color=C_ACCENT)

        ax.set_ylabel(label)
        ax.set_title(label, loc='left', fontweight='bold', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(drought_labels)

        # Clean y-axis
        if key == 'mortality':
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v*100:.0f}%'))

    fig.suptitle('V36: Five Drought Portraits — Seed 23', fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "v36_drought_portraits.png")
    plt.close(fig)
    print("    Done.")


# ============================================================
# Figure 4: Hidden State Evolution (Effective Rank over cycles)
# ============================================================
def fig4_hidden_state_evolution(summary):
    print("  Generating v36_hidden_state_evolution.png ...")
    cycles_data = summary['cycles']
    cycle_nums = [c['cycle'] for c in cycles_data]
    drought_cycles = [c['cycle'] for c in cycles_data if c['is_drought']]

    eff_ranks = []
    mean_norms = []
    max_norms = []

    for c_info in cycles_data:
        cn = c_info['cycle']
        print(f"    Loading focal_timeline for cycle {cn} ...")
        timeline = load_focal_timeline(cn)

        # Extract hidden states: (T, 16) matrix
        hidden_states = np.array([t['hidden_state'] for t in timeline], dtype=np.float32)

        # Compute effective rank of hidden state trajectory
        # Effective rank = exp(entropy of normalized singular values)
        # Center the data first
        hs_centered = hidden_states - hidden_states.mean(axis=0, keepdims=True)

        # SVD
        U, S, Vt = np.linalg.svd(hs_centered, full_matrices=False)
        # Normalized singular values (as probability distribution)
        S_norm = S / S.sum() if S.sum() > 0 else S
        # Filter out zeros for log
        S_pos = S_norm[S_norm > 1e-10]
        entropy = -np.sum(S_pos * np.log(S_pos))
        eff_rank = np.exp(entropy)
        eff_ranks.append(eff_rank)

        # Also track norm stats
        norms = np.linalg.norm(hidden_states, axis=1)
        mean_norms.append(norms.mean())
        max_norms.append(norms.max())

        del timeline, hidden_states  # Free memory

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                     gridspec_kw={'height_ratios': [2, 1]})

    # Panel 1: Effective rank
    add_drought_bands(ax1, cycle_nums, drought_cycles, alpha=0.12)
    ax1.plot(cycle_nums, eff_ranks, color='#AAAAAA', linewidth=1.0, alpha=0.5, zorder=1)

    is_drought = [c['is_drought'] for c in cycles_data]
    for cn, er, dr in zip(cycle_nums, eff_ranks, is_drought):
        c = C_DROUGHT if dr else C_ERANK
        m = 'D' if dr else 'o'
        s = 80 if dr else 60
        ax1.scatter(cn, er, c=c, s=s, marker=m, zorder=3, edgecolors='white', linewidth=0.8)

    ax1.set_ylabel('Effective Rank of Hidden State Trajectory')
    ax1.set_title('V36: Hidden State Representational Complexity — Seed 23', fontweight='bold')
    ax1.axhline(y=16, color='gray', linestyle=':', alpha=0.3, label='Max (dim=16)')
    ax1.axhline(y=1, color='gray', linestyle=':', alpha=0.3, label='Min (1D)')
    ax1.legend(loc='upper right', framealpha=0.9)

    # Panel 2: Hidden state norms
    add_drought_bands(ax2, cycle_nums, drought_cycles, alpha=0.12)
    ax2.plot(cycle_nums, mean_norms, 'o-', color=C_NORMAL, markersize=5, label='Mean ||h||', zorder=2)
    ax2.fill_between(cycle_nums, mean_norms, max_norms, alpha=0.15, color=C_NORMAL)
    ax2.plot(cycle_nums, max_norms, 's--', color=C_NORMAL, markersize=4, alpha=0.5, label='Max ||h||', zorder=2)

    ax2.set_xlabel('Evolutionary Cycle')
    ax2.set_ylabel('Hidden State Norm')
    ax2.legend(loc='upper right', framealpha=0.9)

    ax2.set_xlim(-1, 30)
    ax2.set_xticks(range(0, 30, 2))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "v36_hidden_state_evolution.png")
    plt.close(fig)
    print("    Done.")

    return dict(zip(cycle_nums, eff_ranks))


# ============================================================
# Figure 5: Energy Trajectories (4 selected cycles)
# ============================================================
def fig5_energy_trajectories(summary):
    print("  Generating v36_energy_trajectories.png ...")

    # Select 4 representative cycles:
    # Early normal (0), first drought (5), late normal (16), late drought (25)
    selected = [
        (0, 'Cycle 0 (Early Normal)', C_NORMAL, '-'),
        (5, 'Cycle 5 (1st Drought)', C_DROUGHT, '-'),
        (16, 'Cycle 16 (Mid Normal)', '#4A6FE8', '-'),
        (25, 'Cycle 25 (5th Drought)', '#B8292A', '--'),
    ]

    fig, ax = plt.subplots(figsize=(12, 6))

    for cn, label, color, ls in selected:
        print(f"    Loading focal_timeline for cycle {cn} ...")
        timeline = load_focal_timeline(cn)
        energies = [t['energy'] for t in timeline]
        steps = list(range(len(energies)))

        # Downsample for smoother plot (every 10th point for raw, rolling mean for smooth)
        energies_arr = np.array(energies)
        # Rolling mean (window=50)
        window = 50
        if len(energies_arr) >= window:
            kernel = np.ones(window) / window
            smoothed = np.convolve(energies_arr, kernel, mode='valid')
            smooth_steps = np.arange(window//2, window//2 + len(smoothed))
        else:
            smoothed = energies_arr
            smooth_steps = np.arange(len(smoothed))

        # Light raw trace
        ax.plot(steps[::5], energies_arr[::5], color=color, alpha=0.12, linewidth=0.5)
        # Bold smoothed trace
        ax.plot(smooth_steps, smoothed, color=color, linewidth=2.0, linestyle=ls, label=label, alpha=0.9)

        del timeline  # Free memory

    ax.set_xlabel('Timestep (within cycle)')
    ax.set_ylabel('Focal Agent Energy')
    ax.set_title('V36: Focal Agent Energy Trajectories — Seed 23', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(0, 5000)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "v36_energy_trajectories.png")
    plt.close(fig)
    print("    Done.")


# ============================================================
# Figure 6: Summary Card
# ============================================================
def fig6_summary_card(summary, hidden_eff_ranks=None):
    print("  Generating v36_summary_card.png ...")
    cycles_data = summary['cycles']

    # Compute summary stats
    all_phis = [c['phi'] for c in cycles_data]
    drought_data = [c for c in cycles_data if c['is_drought']]
    normal_data = [c for c in cycles_data if not c['is_drought']]

    mean_phi = np.mean(all_phis)
    max_phi = np.max(all_phis)
    min_phi = np.min(all_phis)
    std_phi = np.std(all_phis)

    # Phi trend
    cycle_nums = [c['cycle'] for c in cycles_data]
    z = np.polyfit(cycle_nums, all_phis, 1)
    phi_slope = z[0]

    # Drought stats
    mean_drought_mortality = np.mean([d['mortality'] for d in drought_data])
    mean_drought_phi = np.mean([d['phi'] for d in drought_data])
    mean_normal_phi = np.mean([d['phi'] for d in normal_data])

    # Recovery: phi in cycle after drought vs phi in drought
    recovery_pairs = []
    for d in drought_data:
        # Find the next recorded cycle
        next_cycle = None
        for c in cycles_data:
            if c['cycle'] > d['cycle'] and not c['is_drought']:
                next_cycle = c
                break
        if next_cycle:
            recovery_pairs.append((d['phi'], next_cycle['phi'], next_cycle['phi'] - d['phi']))

    mean_recovery_bounce = np.mean([r[2] for r in recovery_pairs]) if recovery_pairs else 0

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Title bar
    fig.suptitle('V36: Egocentric Trajectory — Summary Card', fontsize=16, fontweight='bold', y=0.98)

    # ---- Mini Phi trajectory (top left, spanning 2 cols) ----
    ax_phi = fig.add_subplot(gs[0, :2])
    drought_cycs = [c['cycle'] for c in drought_data]
    add_drought_bands(ax_phi, cycle_nums, drought_cycs, alpha=0.12)
    ax_phi.plot(cycle_nums, all_phis, 'o-', color=C_PHI, markersize=5, linewidth=1.5)
    for cn, phi, c_data in zip(cycle_nums, all_phis, cycles_data):
        if c_data['is_drought']:
            ax_phi.scatter(cn, phi, c=C_DROUGHT, s=50, marker='D', zorder=3, edgecolors='white')
    trend_x = np.linspace(min(cycle_nums), max(cycle_nums), 50)
    ax_phi.plot(trend_x, np.polyval(z, trend_x), '--', color=C_ACCENT, alpha=0.4, linewidth=1)
    ax_phi.set_title(r'$\Phi$ Trajectory', loc='left', fontweight='bold', fontsize=11)
    ax_phi.set_xlabel('Cycle')
    ax_phi.set_ylabel(r'$\Phi$')
    ax_phi.set_xlim(-1, 30)

    # ---- Key stats box (top right) ----
    ax_stats = fig.add_subplot(gs[0, 2])
    ax_stats.axis('off')
    stats_text = (
        f"Seed: 23\n"
        f"Cycles recorded: 16 / 30\n"
        f"Droughts: 5\n"
        f"{'─' * 25}\n"
        f"Mean Phi: {mean_phi:.4f}\n"
        f"Max Phi:  {max_phi:.4f}\n"
        f"Min Phi:  {min_phi:.4f}\n"
        f"Phi Std:  {std_phi:.4f}\n"
        f"Phi Slope: {phi_slope:+.5f}/cyc\n"
        f"{'─' * 25}\n"
        f"Mean drought mortality: {mean_drought_mortality:.1%}\n"
        f"Mean drought Phi: {mean_drought_phi:.4f}\n"
        f"Mean normal Phi:  {mean_normal_phi:.4f}\n"
        f"Mean bounce: {mean_recovery_bounce:+.4f}"
    )
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                  fontsize=10, fontfamily='monospace', verticalalignment='top',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5', edgecolor='#CCCCCC'))

    # ---- Drought comparison (middle left) ----
    ax_dr = fig.add_subplot(gs[1, 0])
    d_labels = [f'C{d["cycle"]}' for d in drought_data]
    d_phis = [d['phi'] for d in drought_data]
    x = np.arange(len(drought_data))
    ax_dr.bar(x, d_phis, color=C_DROUGHT, alpha=0.8, width=0.6)
    ax_dr.set_xticks(x)
    ax_dr.set_xticklabels(d_labels)
    ax_dr.set_ylabel(r'$\Phi$')
    ax_dr.set_title(r'Drought $\Phi$', loc='left', fontweight='bold', fontsize=11)

    # ---- Mortality across droughts (middle center) ----
    ax_mort = fig.add_subplot(gs[1, 1])
    d_mort = [d['mortality'] for d in drought_data]
    ax_mort.bar(x, [m*100 for m in d_mort], color=C_MORTALITY, alpha=0.8, width=0.6)
    ax_mort.set_xticks(x)
    ax_mort.set_xticklabels(d_labels)
    ax_mort.set_ylabel('Mortality %')
    ax_mort.set_title('Drought Mortality', loc='left', fontweight='bold', fontsize=11)
    ax_mort.set_ylim(0, 105)

    # ---- Survivors (middle right) ----
    ax_surv = fig.add_subplot(gs[1, 2])
    d_surv = [d['n_alive_end'] for d in drought_data]
    ax_surv.bar(x, d_surv, color='#4A6FE8', alpha=0.8, width=0.6)
    ax_surv.set_xticks(x)
    ax_surv.set_xticklabels(d_labels)
    ax_surv.set_ylabel('Survivors')
    ax_surv.set_title('Post-Drought Survivors', loc='left', fontweight='bold', fontsize=11)

    # ---- Effective rank over evolution (bottom left, 2 cols) ----
    ax_er = fig.add_subplot(gs[2, :2])
    eranks = [c['segment_affect']['effective_rank'] for c in cycles_data]
    add_drought_bands(ax_er, cycle_nums, drought_cycs, alpha=0.12)
    ax_er.plot(cycle_nums, eranks, 'o-', color=C_ERANK, markersize=5, linewidth=1.5)
    for cn, er, c_data in zip(cycle_nums, eranks, cycles_data):
        if c_data['is_drought']:
            ax_er.scatter(cn, er, c=C_DROUGHT, s=50, marker='D', zorder=3, edgecolors='white')
    ax_er.set_title('Population Effective Rank', loc='left', fontweight='bold', fontsize=11)
    ax_er.set_xlabel('Cycle')
    ax_er.set_ylabel('Eff. Rank')
    ax_er.set_xlim(-1, 30)

    # ---- Recovery pattern (bottom right) ----
    ax_rec = fig.add_subplot(gs[2, 2])
    if recovery_pairs:
        bounce_vals = [r[2] for r in recovery_pairs]
        bar_x = np.arange(len(recovery_pairs))
        bar_labels = [f'D{i+1}' for i in range(len(recovery_pairs))]
        colors = [C_NORMAL if b > 0 else C_DROUGHT for b in bounce_vals]
        ax_rec.bar(bar_x, bounce_vals, color=colors, alpha=0.8, width=0.6)
        ax_rec.axhline(0, color='gray', linewidth=0.5)
        ax_rec.set_xticks(bar_x)
        ax_rec.set_xticklabels(bar_labels)
        ax_rec.set_ylabel(r'$\Delta\Phi$')
        ax_rec.set_title(r'Post-Drought $\Phi$ Bounce', loc='left', fontweight='bold', fontsize=11)

    fig.savefig(OUT_DIR / "v36_summary_card.png")
    plt.close(fig)
    print("    Done.")


# ============================================================
# Main
# ============================================================
def main():
    print("V36 Figure Generation")
    print("=" * 50)

    summary = load_summary()
    print(f"Loaded summary: {len(summary['cycles'])} cycles, seed {summary['seed']}")

    fig1_phi_trajectory(summary)
    fig2_affect_dimensions(summary)
    fig3_drought_portraits(summary)
    hidden_eff_ranks = fig4_hidden_state_evolution(summary)
    fig5_energy_trajectories(summary)
    fig6_summary_card(summary, hidden_eff_ranks)

    print()
    print("=" * 50)
    print(f"All figures saved to: {OUT_DIR}")
    print("Files generated:")
    for f in sorted(OUT_DIR.glob("v36_*.png")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
