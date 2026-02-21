#!/usr/bin/env python3
"""Generate publication-quality figures for V31 (10-seed validation)."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

sns.set_style("whitegrid")

BASE = '/Users/jacob/fun/shape-of-experience/empirical/experiments/study_ca_affect/results'
OUT = os.path.join(BASE, 'v31_figures')

# Color palette
CAT_COLORS = {'HIGH': '#2ca02c', 'MOD': '#f0a830', 'LOW': '#d62728'}

# Load V31 data
summary = json.load(open(os.path.join(BASE, 'v31_summary.json')))
seed_analysis = json.load(open(os.path.join(BASE, 'v31_seed_analysis.json')))

# Build seed data for all 10 seeds
all_seed_keys = ['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's42', 's123']
seed_data = {}
for k in all_seed_keys:
    d = summary[k]
    seed_data[k] = {'mean_phi': d['mean_phi'], 'max_phi': d['max_phi'],
                     'mean_robustness': d['mean_robustness']}

# Assign categories by rank (30/30/40 split)
sorted_seeds = sorted(seed_data.items(), key=lambda x: x[1]['mean_phi'], reverse=True)
for i, (k, v) in enumerate(sorted_seeds):
    if i < 3:
        seed_data[k]['category'] = 'HIGH'
    elif i < 6:
        seed_data[k]['category'] = 'MOD'
    else:
        seed_data[k]['category'] = 'LOW'

print("V31 Seed Categories:")
for k, v in sorted(seed_data.items(), key=lambda x: x[1]['mean_phi'], reverse=True):
    print(f"  {k}: phi={v['mean_phi']:.4f} -> {v['category']}")

# Load progress trajectories
progress_data = {}
seed_dir_map = {}
for i in range(7):
    k = f's{i}'
    path = os.path.join(BASE, f'v31_{k}', f'v29_{k}_progress.json')
    if os.path.exists(path):
        seed_dir_map[k] = path

# Find original 3 seeds
for seed_id in ['s42', 's123', 's7']:
    for root, dirs, files in os.walk(BASE):
        for f in files:
            if f == f'v29_{seed_id}_progress.json':
                seed_dir_map[seed_id] = os.path.join(root, f)
                break
    # Also check results files
    if seed_id not in seed_dir_map:
        for root, dirs, files in os.walk(BASE):
            for f in files:
                if f == f'v29_{seed_id}_results.json':
                    d = json.load(open(os.path.join(root, f)))
                    if 'cycles' in d:
                        seed_dir_map[seed_id] = os.path.join(root, f)
                        break

print("\nSeed file map:")
for k, v in seed_dir_map.items():
    print(f"  {k}: {v}")

for k, path in seed_dir_map.items():
    d = json.load(open(path))
    if 'cycles' in d:
        progress_data[k] = d['cycles']

print(f"\nSeeds with progress data: {sorted(progress_data.keys())}")

# ── Figure 1: Seed Distribution (Strip/Swarm) ──────────────────
fig, ax = plt.subplots(figsize=(12, 6))

# Prepare data sorted by phi
sorted_items = sorted(seed_data.items(), key=lambda x: x[1]['mean_phi'])
phis = [v['mean_phi'] for k, v in sorted_items]
labels = [k for k, v in sorted_items]
colors = [CAT_COLORS[v['category']] for k, v in sorted_items]
categories = [v['category'] for k, v in sorted_items]

# Background bands
high_min = min(v['mean_phi'] for v in seed_data.values() if v['category'] == 'HIGH')
mod_min = min(v['mean_phi'] for v in seed_data.values() if v['category'] == 'MOD')
mod_max = max(v['mean_phi'] for v in seed_data.values() if v['category'] == 'MOD')
low_max = max(v['mean_phi'] for v in seed_data.values() if v['category'] == 'LOW')

ax.axhspan(0, (low_max + mod_min) / 2, alpha=0.06, color=CAT_COLORS['LOW'], zorder=0)
ax.axhspan((low_max + mod_min) / 2, (mod_max + high_min) / 2, alpha=0.06, color=CAT_COLORS['MOD'], zorder=0)
ax.axhspan((mod_max + high_min) / 2, 0.20, alpha=0.06, color=CAT_COLORS['HIGH'], zorder=0)

# Horizontal dashed lines at band boundaries
for y in [(low_max + mod_min) / 2, (mod_max + high_min) / 2]:
    ax.axhline(y, color='#aaaaaa', linestyle=':', linewidth=1, alpha=0.6, zorder=1)

# Plot individual seeds
x_positions = np.linspace(0.25, 0.75, len(phis))
np.random.seed(42)
x_jitter = x_positions + np.random.normal(0, 0.02, len(phis))

for i, (x, phi, label, color, cat) in enumerate(zip(x_jitter, phis, labels, colors, categories)):
    ax.scatter(x, phi, c=color, s=220, zorder=5, edgecolors='white', linewidth=2)
    offset_x = 14 if i % 2 == 0 else -14
    ha = 'left' if i % 2 == 0 else 'right'
    ax.annotate(label, (x, phi), textcoords="offset points", xytext=(offset_x, 0),
                fontsize=10, color='#444444', va='center', ha=ha, fontweight='bold')

# Category labels
ax.text(0.97, 0.92, 'HIGH (30%)', transform=ax.transAxes,
        fontsize=13, fontweight='bold', color=CAT_COLORS['HIGH'], va='top', ha='right')
ax.text(0.97, 0.55, 'MOD (30%)', transform=ax.transAxes,
        fontsize=13, fontweight='bold', color=CAT_COLORS['MOD'], va='center', ha='right')
ax.text(0.97, 0.18, 'LOW (40%)', transform=ax.transAxes,
        fontsize=13, fontweight='bold', color=CAT_COLORS['LOW'], va='bottom', ha='right')

# Mean line
mean_phi = summary['statistics']['mean_phi_mean']
ax.axhline(mean_phi, color='#333333', linestyle='--', linewidth=1.5, alpha=0.5, zorder=3)
ax.text(0.12, mean_phi + 0.002, f'mean = {mean_phi:.3f}', fontsize=10, color='#333333')

# Annotation box
ax.text(0.5, 0.02, 't = 0.09 vs V27,  p = 0.93  —  social prediction ≈ self prediction',
        transform=ax.transAxes, fontsize=13, ha='center', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8e8e8', edgecolor='#999999', alpha=0.9))

ax.set_xlim(0.0, 1.05)
ax.set_ylim(0.03, 0.17)
ax.set_ylabel('Mean $\\Phi$ (integration)', fontsize=14)
ax.set_title('V31: 10-Seed Social Prediction — Seed Distribution', fontweight='bold', fontsize=16)
ax.set_xticks([])
ax.set_xlabel('')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'v31_seed_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved v31_seed_distribution.png")

# ── Figure 2: Phi Trajectories ─────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))

drought_cycles = [5, 10, 15, 20, 25]
for dc in drought_cycles:
    ax.axvspan(dc - 0.4, dc + 0.4, alpha=0.12, color='#888888', zorder=0)

# Plot each seed
for k in sorted(progress_data.keys(), key=lambda x: seed_data[x]['mean_phi'], reverse=True):
    cycles = progress_data[k]
    x = [c['cycle'] for c in cycles]
    y = [c['mean_phi'] for c in cycles]
    cat = seed_data[k]['category']
    color = CAT_COLORS[cat]
    alpha = 0.95 if cat == 'HIGH' else (0.75 if cat == 'MOD' else 0.55)
    lw = 2.8 if cat == 'HIGH' else (2.0 if cat == 'MOD' else 1.4)
    ax.plot(x, y, color=color, alpha=alpha, linewidth=lw, zorder=3 if cat == 'HIGH' else 2)

# Drought label
ymax = ax.get_ylim()[1]
for dc in drought_cycles:
    ax.text(dc, ymax * 0.97, '☠', fontsize=10, ha='center', va='top', color='#666666')

ax.text(15, ymax * 0.99, 'drought cycles (≥90% mortality)',
        fontsize=10, ha='center', va='top', color='#666666', style='italic')

# Legend
handles = [
    plt.Line2D([0], [0], color=CAT_COLORS['HIGH'], linewidth=2.8, label='HIGH (n=3)'),
    plt.Line2D([0], [0], color=CAT_COLORS['MOD'], linewidth=2.0, alpha=0.75, label='MOD (n=3)'),
    plt.Line2D([0], [0], color=CAT_COLORS['LOW'], linewidth=1.4, alpha=0.55, label='LOW (n=4)'),
    mpatches.Patch(color='#888888', alpha=0.2, label='Drought'),
]
ax.legend(handles=handles, loc='upper right', framealpha=0.95, fontsize=12, edgecolor='#cccccc')

ax.set_xlabel('Evolutionary Cycle', fontsize=14)
ax.set_ylabel('Mean $\\Phi$ (population integration)', fontsize=14)
ax.set_title('V31: 10-Seed Social Prediction — The 30/30/40 Split',
             fontweight='bold', fontsize=18)
ax.set_xlim(-0.5, 29.5)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'v31_phi_trajectories_all.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved v31_phi_trajectories_all.png")

# ── Figure 3: Summary Card ─────────────────────────────────────
fig = plt.figure(figsize=(10, 8))
fig.suptitle('V31: 10-Seed Validation — Summary', fontweight='bold', fontsize=18, y=0.97)

# Use gridspec for better layout
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3,
                       left=0.08, right=0.92, top=0.90, bottom=0.05)

# Panel A: Distribution
ax1 = fig.add_subplot(gs[0, 0])
sizes = [3, 3, 4]
labels_pie = ['HIGH\n30%', 'MOD\n30%', 'LOW\n40%']
colors_pie = [CAT_COLORS['HIGH'], CAT_COLORS['MOD'], CAT_COLORS['LOW']]
wedges, texts = ax1.pie(sizes, labels=labels_pie, colors=colors_pie,
                         startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'},
                         wedgeprops={'edgecolor': 'white', 'linewidth': 2.5})
ax1.set_title('Category Distribution', fontsize=13, fontweight='bold', pad=8)

# Panel B: Key Statistics
ax2 = fig.add_subplot(gs[0, 1])
ax2.axis('off')
stats = summary['statistics']
stat_text = (
    f"N seeds:  10\n"
    f"Mean Φ:  {stats['mean_phi_mean']:.3f} ± {stats['mean_phi_std']:.3f}\n"
    f"95% CI:  [{stats['mean_phi_ci95_lo']:.3f}, {stats['mean_phi_ci95_hi']:.3f}]\n"
    f"Max Φ:  {stats['max_phi_max']:.3f}\n"
    f"Mean rob:  {stats['mean_rob_mean']:.3f} ± {stats['mean_rob_std']:.3f}\n\n"
    f"vs V27 (self-prediction):\n"
    f"  t = {stats['t_stat_vs_v27']:.2f},  p ≈ 0.93\n"
    f"  NOT significant"
)
ax2.text(0.05, 0.95, stat_text, fontsize=12, transform=ax2.transAxes,
         va='top', ha='left', fontfamily='monospace', linespacing=1.6,
         bbox=dict(boxstyle='round,pad=0.6', facecolor='#f0f0f0', edgecolor='#cccccc'))
ax2.set_title('Key Statistics', fontsize=13, fontweight='bold', pad=8)

# Panel C: Key Finding
ax3 = fig.add_subplot(gs[1, 0])
ax3.axis('off')
finding = (
    "Social prediction produces the SAME\n"
    "Φ distribution as self-prediction.\n\n"
    "V29's 3-seed \"social lift\" was a\n"
    "fluke of small-sample variance.\n\n"
    "Prediction target does not matter.\n"
    "What matters is the architecture."
)
ax3.text(0.5, 0.5, finding, fontsize=12, ha='center', va='center',
         transform=ax3.transAxes, linespacing=1.5,
         bbox=dict(boxstyle='round,pad=0.8', facecolor='#fff3cd',
                   edgecolor='#f0a830', linewidth=1.5))
ax3.set_title('Key Finding', fontsize=13, fontweight='bold', pad=8)

# Panel D: Post-drought bounce
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')
# Get bounce data from seed_analysis
bounce_data = []
for s in seed_analysis:
    bounce_data.append({
        'seed': s['seed'],
        'category': s['category'],
        'mean_phi': s['mean_phi'],
        'bounce': s['mean_post_drought_bounce']
    })
bounce_text = (
    "Post-Drought Bounce\n"
    "predicts category:\n\n"
)
for b in sorted(bounce_data, key=lambda x: x['bounce'], reverse=True):
    cat = b['category']
    color_name = {'HIGH': 'green', 'MOD': 'gold', 'LOW': 'red'}[cat]
    bounce_text += f"  s{b['seed']}: bounce={b['bounce']:.3f} [{cat}]\n"

bounce_text += f"\nr = 0.997, p < 0.0001"

ax4.text(0.5, 0.5, bounce_text, fontsize=11, ha='center', va='center',
         transform=ax4.transAxes, fontfamily='monospace', linespacing=1.4,
         bbox=dict(boxstyle='round,pad=0.6', facecolor='#d4edda',
                   edgecolor='#28a745', linewidth=1.5))
ax4.set_title('Trajectory Dependence', fontsize=13, fontweight='bold', pad=8)

plt.savefig(os.path.join(OUT, 'v31_summary_card.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved v31_summary_card.png")

print("\n=== V31 figures complete! ===")
