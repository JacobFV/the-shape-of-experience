#!/usr/bin/env python3
"""Generate publication-quality figures for V32 (50-seed drought autopsy)."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats as sp_stats

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
V32_DIR = os.path.join(BASE, 'v32_drought_autopsy')
OUT = os.path.join(BASE, 'v32_figures')

CAT_COLORS = {'HIGH': '#2ca02c', 'MOD': '#f0a830', 'LOW': '#d62728'}

# ── Load all V32 data ──────────────────────────────────────────
seeds = []
progress_all = {}
for i in range(50):
    rpath = os.path.join(V32_DIR, f'v32_s{i}_results.json')
    ppath = os.path.join(V32_DIR, f'v32_s{i}_progress.json')
    if not os.path.exists(rpath):
        continue
    r = json.load(open(rpath))
    s = r['summary']
    s['seed'] = i
    s['drought_events'] = r.get('drought_events', [])
    seeds.append(s)
    if os.path.exists(ppath):
        p = json.load(open(ppath))
        progress_all[i] = p['cycles']

print(f"Loaded {len(seeds)} V32 seeds, {len(progress_all)} with progress data")

# Category counts
cat_counts = {}
for s in seeds:
    cat_counts.setdefault(s['category'], []).append(s['seed'])
for c in ['HIGH', 'MOD', 'LOW']:
    n = len(cat_counts.get(c, []))
    print(f"  {c}: {n} ({100*n/len(seeds):.0f}%)")

# Compute phi_slope for each seed (linear regression over cycles)
phi_slopes = {}
for s in seeds:
    sid = s['seed']
    if sid in progress_all:
        cycles = progress_all[sid]
        x = np.array([c['cycle'] for c in cycles])
        y = np.array([c['mean_phi'] for c in cycles])
        if len(x) > 2:
            slope, intercept, r_val, p_val, stderr = sp_stats.linregress(x, y)
            phi_slopes[sid] = slope
    s['phi_slope'] = phi_slopes.get(sid, None)

# ── Figure 1: Seed Distribution (Swarm) ────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

# Sort by late_mean_phi
sorted_seeds = sorted(seeds, key=lambda x: x['late_mean_phi'])

# Swarm-like strip plot using jittered x positions
np.random.seed(42)
x_base = np.linspace(0.1, 0.9, len(sorted_seeds))
x_jitter = x_base + np.random.normal(0, 0.008, len(sorted_seeds))

for i, (x, s) in enumerate(zip(x_jitter, sorted_seeds)):
    color = CAT_COLORS[s['category']]
    ax.scatter(x, s['late_mean_phi'], c=color, s=100, zorder=5,
              edgecolors='white', linewidth=1, alpha=0.85)

# Background bands
high_phis = [s['late_mean_phi'] for s in seeds if s['category'] == 'HIGH']
mod_phis = [s['late_mean_phi'] for s in seeds if s['category'] == 'MOD']
low_phis = [s['late_mean_phi'] for s in seeds if s['category'] == 'LOW']

boundary_low_mod = (max(low_phis) + min(mod_phis)) / 2
boundary_mod_high = (max(mod_phis) + min(high_phis)) / 2

ax.axhspan(-0.05, boundary_low_mod, alpha=0.05, color=CAT_COLORS['LOW'], zorder=0)
ax.axhspan(boundary_low_mod, boundary_mod_high, alpha=0.05, color=CAT_COLORS['MOD'], zorder=0)
ax.axhspan(boundary_mod_high, 0.35, alpha=0.05, color=CAT_COLORS['HIGH'], zorder=0)

for y in [boundary_low_mod, boundary_mod_high]:
    ax.axhline(y, color='#aaaaaa', linestyle=':', linewidth=1, alpha=0.5, zorder=1)

# Category annotations
n_high = len(cat_counts.get('HIGH', []))
n_mod = len(cat_counts.get('MOD', []))
n_low = len(cat_counts.get('LOW', []))
ax.text(0.97, 0.92, f'HIGH ({n_high}/{len(seeds)} = {100*n_high/len(seeds):.0f}%)',
        transform=ax.transAxes, fontsize=13, fontweight='bold',
        color=CAT_COLORS['HIGH'], va='top', ha='right')
ax.text(0.97, 0.55, f'MOD ({n_mod}/{len(seeds)} = {100*n_mod/len(seeds):.0f}%)',
        transform=ax.transAxes, fontsize=13, fontweight='bold',
        color=CAT_COLORS['MOD'], va='center', ha='right')
ax.text(0.97, 0.12, f'LOW ({n_low}/{len(seeds)} = {100*n_low/len(seeds):.0f}%)',
        transform=ax.transAxes, fontsize=13, fontweight='bold',
        color=CAT_COLORS['LOW'], va='bottom', ha='right')

# Mean line
mean_late = np.mean([s['late_mean_phi'] for s in seeds])
ax.axhline(mean_late, color='#333333', linestyle='--', linewidth=1.5, alpha=0.5, zorder=3)
ax.text(0.02, mean_late + 0.005, f'mean = {mean_late:.3f}', fontsize=10, color='#333333',
        transform=ax.get_yaxis_transform())

ax.set_xlim(-0.02, 1.05)
ax.set_ylim(-0.01, max(s['late_mean_phi'] for s in seeds) * 1.15)
ax.set_ylabel('Late Mean $\\Phi$ (last 10 cycles)', fontsize=14)
ax.set_title(f'V32: 50-Seed Drought Autopsy — Seed Distribution ({n_high}/{n_mod}/{n_low} split)',
             fontweight='bold', fontsize=16)
ax.set_xticks([])
ax.set_xlabel('Seeds (sorted by late mean $\\Phi$)')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'v32_seed_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved v32_seed_distribution.png")

# ── Figure 2: Bounce vs Phi Scatter ────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))

bounces = np.array([s['mean_bounce'] for s in seeds])
late_phis = np.array([s['late_mean_phi'] for s in seeds])
cats = [s['category'] for s in seeds]
seed_ids = [s['seed'] for s in seeds]

# Scatter
for i, (b, p, c, sid) in enumerate(zip(bounces, late_phis, cats, seed_ids)):
    ax.scatter(b, p, c=CAT_COLORS[c], s=120, zorder=5,
              edgecolors='white', linewidth=1.2, alpha=0.85)

# Regression line
mask = np.isfinite(bounces) & np.isfinite(late_phis)
slope, intercept, r_val, p_val, stderr = sp_stats.linregress(bounces[mask], late_phis[mask])
x_line = np.linspace(bounces.min(), bounces.max(), 100)
y_line = slope * x_line + intercept
ax.plot(x_line, y_line, color='#333333', linewidth=2, linestyle='--', alpha=0.7, zorder=4)

# Confidence band
n = mask.sum()
x_mean = bounces[mask].mean()
se_line = stderr * np.sqrt(1/n + (x_line - x_mean)**2 / np.sum((bounces[mask] - x_mean)**2))
ax.fill_between(x_line, y_line - 1.96*se_line, y_line + 1.96*se_line,
                alpha=0.1, color='#333333', zorder=2)

# Correlation annotation
ax.text(0.05, 0.95, f'r = {r_val:.3f}\np < {p_val:.1e}' if p_val < 0.001 else f'r = {r_val:.3f}\np = {p_val:.4f}',
        transform=ax.transAxes, fontsize=14, fontweight='bold',
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#cccccc', alpha=0.9))

# Legend
handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=CAT_COLORS['HIGH'],
               markersize=10, label=f'HIGH (n={n_high})'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=CAT_COLORS['MOD'],
               markersize=10, label=f'MOD (n={n_mod})'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=CAT_COLORS['LOW'],
               markersize=10, label=f'LOW (n={n_low})'),
]
ax.legend(handles=handles, loc='lower right', framealpha=0.95, fontsize=12)

ax.set_xlabel('Mean Post-Drought Bounce Ratio ($\\Phi_{after}$ / $\\Phi_{before}$)', fontsize=14)
ax.set_ylabel('Late Mean $\\Phi$ (last 10 cycles)', fontsize=14)
ax.set_title(f'Post-Drought Bounce Predicts Integration (r = {r_val:.2f})',
             fontweight='bold', fontsize=16)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'v32_bounce_vs_phi.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved v32_bounce_vs_phi.png")

# ── Figure 3: Phi Slope ANOVA ──────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))

# Gather phi_slopes by category
slope_data = {'HIGH': [], 'MOD': [], 'LOW': []}
for s in seeds:
    if s['phi_slope'] is not None:
        slope_data[s['category']].append(s['phi_slope'])

# Box plots
cat_order = ['HIGH', 'MOD', 'LOW']
box_data = [slope_data[c] for c in cat_order]
bp = ax.boxplot(box_data, positions=[1, 2, 3], widths=0.6,
                patch_artist=True, showfliers=True, zorder=3)

for patch, cat in zip(bp['boxes'], cat_order):
    patch.set_facecolor(CAT_COLORS[cat])
    patch.set_alpha(0.5)
    patch.set_edgecolor(CAT_COLORS[cat])
    patch.set_linewidth(2)

for element in ['whiskers', 'caps']:
    for item, cat in zip(zip(bp[element][::2], bp[element][1::2]), cat_order):
        for line in item:
            line.set_color(CAT_COLORS[cat])
            line.set_linewidth(1.5)

for median in bp['medians']:
    median.set_color('#333333')
    median.set_linewidth(2)

# Overlay individual points
for i, (cat, data) in enumerate(zip(cat_order, box_data)):
    np.random.seed(i + 42)
    x = np.full(len(data), i + 1) + np.random.normal(0, 0.06, len(data))
    ax.scatter(x, data, c=CAT_COLORS[cat], s=50, alpha=0.6, zorder=4,
              edgecolors='white', linewidth=0.5)

# ANOVA
groups = [np.array(slope_data[c]) for c in cat_order if len(slope_data[c]) > 0]
if len(groups) >= 2:
    f_stat, p_val = sp_stats.f_oneway(*groups)
    annotation = f'One-way ANOVA: F = {f_stat:.2f}, p = {p_val:.1e}' if p_val < 0.001 else f'One-way ANOVA: F = {f_stat:.2f}, p = {p_val:.4f}'
    
    # Also compute effect size (eta-squared)
    all_slopes = np.concatenate(groups)
    grand_mean = all_slopes.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    ss_total = np.sum((all_slopes - grand_mean)**2)
    eta_sq = ss_between / ss_total if ss_total > 0 else 0
    annotation += f'\n$\\eta^2$ = {eta_sq:.3f}'
else:
    annotation = 'Insufficient data for ANOVA'

ax.text(0.5, 0.97, annotation, transform=ax.transAxes,
        fontsize=14, fontweight='bold', ha='center', va='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='#cccccc', alpha=0.9))

# Zero line
ax.axhline(0, color='#333333', linestyle='-', linewidth=1, alpha=0.3, zorder=1)

ax.set_xticks([1, 2, 3])
ax.set_xticklabels([f'HIGH\n(n={len(slope_data["HIGH"])})',
                     f'MOD\n(n={len(slope_data["MOD"])})',
                     f'LOW\n(n={len(slope_data["LOW"])})'],
                    fontsize=13, fontweight='bold')
ax.set_ylabel('$\\Phi$ Slope (per cycle)', fontsize=14)
ax.set_title('V32: Integration Trajectory by Category', fontweight='bold', fontsize=16)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'v32_phi_slope_anova.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved v32_phi_slope_anova.png")

# ── Figure 4: Summary Card ─────────────────────────────────────
fig = plt.figure(figsize=(10, 8))
fig.suptitle('V32: 50-Seed Drought Autopsy — Summary', fontweight='bold', fontsize=18, y=0.97)

gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3,
                       left=0.08, right=0.92, top=0.90, bottom=0.05)

# Panel A: Distribution
ax1 = fig.add_subplot(gs[0, 0])
sizes = [n_high, n_mod, n_low]
labels_pie = [f'HIGH\n{n_high} ({100*n_high/50:.0f}%)',
              f'MOD\n{n_mod} ({100*n_mod/50:.0f}%)',
              f'LOW\n{n_low} ({100*n_low/50:.0f}%)']
colors_pie = [CAT_COLORS['HIGH'], CAT_COLORS['MOD'], CAT_COLORS['LOW']]
wedges, texts = ax1.pie(sizes, labels=labels_pie, colors=colors_pie,
                         startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'},
                         wedgeprops={'edgecolor': 'white', 'linewidth': 2.5})
ax1.set_title('Category Distribution', fontsize=13, fontweight='bold', pad=8)

# Panel B: Key Statistics
ax2 = fig.add_subplot(gs[0, 1])
ax2.axis('off')

mean_phi_all = np.mean([s['mean_phi'] for s in seeds])
std_phi_all = np.std([s['mean_phi'] for s in seeds])
mean_late_all = np.mean([s['late_mean_phi'] for s in seeds])
mean_bounce_all = np.mean([s['mean_bounce'] for s in seeds])
std_bounce_all = np.std([s['mean_bounce'] for s in seeds])
mean_rob_all = np.mean([s['mean_robustness'] for s in seeds])

stat_text = (
    f"N seeds:  50\n"
    f"Mean Φ:  {mean_phi_all:.3f} ± {std_phi_all:.3f}\n"
    f"Late Φ:  {mean_late_all:.3f}\n"
    f"Mean rob:  {mean_rob_all:.3f}\n"
    f"Mean bounce:  {mean_bounce_all:.3f} ± {std_bounce_all:.3f}\n\n"
    f"Bounce-Φ correlation:\n"
    f"  r = {r_val:.3f}, p < {p_val:.1e}"
)
ax2.text(0.05, 0.95, stat_text, fontsize=11, transform=ax2.transAxes,
         va='top', ha='left', fontfamily='monospace', linespacing=1.6,
         bbox=dict(boxstyle='round,pad=0.6', facecolor='#f0f0f0', edgecolor='#cccccc'))
ax2.set_title('Key Statistics', fontsize=13, fontweight='bold', pad=8)

# Panel C: Key Finding
ax3 = fig.add_subplot(gs[1, 0])
ax3.axis('off')
finding = (
    "50 seeds confirm the pattern:\n\n"
    f"  {100*n_high/50:.0f}% HIGH / {100*n_mod/50:.0f}% MOD / {100*n_low/50:.0f}% LOW\n\n"
    "Post-drought bounce predicts\n"
    "final integration level.\n\n"
    "Integration = biography,\n"
    "not initial conditions."
)
ax3.text(0.5, 0.5, finding, fontsize=12, ha='center', va='center',
         transform=ax3.transAxes, linespacing=1.5,
         bbox=dict(boxstyle='round,pad=0.8', facecolor='#fff3cd',
                   edgecolor='#f0a830', linewidth=1.5))
ax3.set_title('Key Finding', fontsize=13, fontweight='bold', pad=8)

# Panel D: Drought mechanics
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

# Compute drought stats
all_survivals = []
for s in seeds:
    for de in s['drought_events']:
        if 'survivor_stats' in de:
            all_survivals.append(de['survivor_stats']['survival_rate'])
mean_survival = np.mean(all_survivals) if all_survivals else 0

drought_text = (
    "Drought Mechanics:\n\n"
    f"  Mean survival: {100*mean_survival:.1f}%\n"
    f"  Droughts per run: 5\n"
    f"  Total events: {len(all_survivals)}\n\n"
    "The furnace forges:\n"
    "repeated near-extinction\n"
    "is the mechanism, not\n"
    "the obstacle."
)
ax4.text(0.5, 0.5, drought_text, fontsize=11, ha='center', va='center',
         transform=ax4.transAxes, linespacing=1.5,
         bbox=dict(boxstyle='round,pad=0.6', facecolor='#d4edda',
                   edgecolor='#28a745', linewidth=1.5))
ax4.set_title('Bottleneck Furnace', fontsize=13, fontweight='bold', pad=8)

plt.savefig(os.path.join(OUT, 'v32_summary_card.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved v32_summary_card.png")

print("\n=== V32 figures complete! ===")
