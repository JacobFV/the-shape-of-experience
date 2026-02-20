#!/usr/bin/env python3
"""
Generate multi-panel figure: Integration (Phi) across experiment conditions V22-V35.

Panel 1: Bar chart of mean late-phase Phi with error bars.
Panel 2: Seed distribution (HIGH/MOD/LOW) as stacked percentage bars.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ---------- style ----------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# ---------- data — Panel 1 ----------
labels = [
    "V22\nScalar pred.",
    "V23\nMulti-target",
    "V24\nTD value",
    "V27\nMLP grad.\ncoupling",
    "V28\nlinear w8",
    "V28\ntanh w16",
    "V29\nSocial (3s)",
    "V31\nSocial (10s)",
    "V33\nContrastive",
    "V34\n\u03a6-inclusive\nfitness",
    "V35\nLanguage",
]
means = [0.097, 0.079, 0.085, 0.090, 0.074, 0.084, 0.104, 0.091, 0.054, 0.079, 0.074]
errs  = [0.020, 0.020, 0.030, 0.028, 0.000, 0.000, 0.040, 0.028, 0.015, 0.036, 0.013]

# Color coding
colors = [
    "#4A90D9",  # V22 blue
    "#4A90D9",  # V23 blue
    "#4A90D9",  # V24 blue
    "#2ECC71",  # V27 green (baseline MLP)
    "#999999",  # V28 linear gray
    "#999999",  # V28 tanh gray
    "#4A90D9",  # V29 blue
    "#4A90D9",  # V31 blue
    "#E74C3C",  # V33 red (negative)
    "#E67E22",  # V34 orange (mixed)
    "#9B59B6",  # V35 purple (language)
]

edge_colors = [c if c != "#999999" else "#666666" for c in colors]

# ---------- data — Panel 2 ----------
dist_labels = ["V27\n(3 seeds)", "V31\n(10 seeds)", "V33\n(10 seeds)", "V34\n(10 seeds)", "V35\n(10 seeds)"]
high_pct = [33, 30,  0, 20,  0]
mod_pct  = [33, 30, 30, 30, 70]
low_pct  = [34, 40, 70, 50, 30]

# ---------- figure ----------
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(16, 8),
    gridspec_kw={"height_ratios": [3, 2], "hspace": 0.38},
)

# ===== Panel 1 =====
x = np.arange(len(labels))
bars = ax1.bar(
    x, means, yerr=errs,
    color=colors, edgecolor=edge_colors, linewidth=0.8,
    capsize=4, error_kw={"linewidth": 1.2, "capthick": 1.2},
    width=0.65, zorder=3,
)

# V27 baseline dashed line
ax1.axhline(y=0.091, color="#2ECC71", linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)
ax1.text(
    len(labels) - 0.3, 0.0925, "V27 baseline (\u03a6 = 0.091)",
    fontsize=9, color="#2ECC71", ha="right", va="bottom", fontstyle="italic",
)

# Value annotations on bars
for i, (m, e) in enumerate(zip(means, errs)):
    y_top = m + e + 0.003 if e > 0 else m + 0.003
    ax1.text(i, y_top, f"{m:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel("Mean Late-Phase \u03a6 (Integration)")
ax1.set_ylim(0, 0.175)
ax1.set_title(
    "Integration (\u03a6) Across Experiment Conditions \u2014 V22\u2013V35",
    fontsize=15, fontweight="bold", pad=12,
)
ax1.text(
    0.5, 1.02,
    "Architecture matters.  Loss engineering and direct selection do not.",
    transform=ax1.transAxes, ha="center", va="bottom",
    fontsize=11, fontstyle="italic", color="#555555",
)

# Legend for Panel 1
legend_handles = [
    Patch(facecolor="#2ECC71", edgecolor="#2ECC71", label="MLP gradient coupling (baseline)"),
    Patch(facecolor="#4A90D9", edgecolor="#4A90D9", label="Prediction variants"),
    Patch(facecolor="#999999", edgecolor="#666666", label="V28 architecture controls"),
    Patch(facecolor="#E74C3C", edgecolor="#E74C3C", label="Negative result"),
    Patch(facecolor="#E67E22", edgecolor="#E67E22", label="Mixed result"),
    Patch(facecolor="#9B59B6", edgecolor="#9B59B6", label="Language channel"),
]
ax1.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.9)

# ===== Panel 2 =====
x2 = np.arange(len(dist_labels))
bar_w = 0.55

p_low  = ax2.bar(x2, low_pct,  width=bar_w, color="#E74C3C", edgecolor="white", linewidth=0.5, label="LOW  (\u03a6 < 0.07)", zorder=3)
p_mod  = ax2.bar(x2, mod_pct,  width=bar_w, bottom=low_pct, color="#F39C12", edgecolor="white", linewidth=0.5, label="MOD  (0.07 \u2264 \u03a6 < 0.12)", zorder=3)
p_high = ax2.bar(x2, high_pct, width=bar_w, bottom=[l + m for l, m in zip(low_pct, mod_pct)], color="#2ECC71", edgecolor="white", linewidth=0.5, label="HIGH (\u03a6 \u2265 0.12)", zorder=3)

# Percentage annotations
for i in range(len(dist_labels)):
    if low_pct[i] > 5:
        ax2.text(i, low_pct[i] / 2, f"{low_pct[i]}%", ha="center", va="center", fontsize=11, fontweight="bold", color="white")
    if mod_pct[i] > 5:
        ax2.text(i, low_pct[i] + mod_pct[i] / 2, f"{mod_pct[i]}%", ha="center", va="center", fontsize=11, fontweight="bold", color="white")
    if high_pct[i] > 5:
        ax2.text(i, low_pct[i] + mod_pct[i] + high_pct[i] / 2, f"{high_pct[i]}%", ha="center", va="center", fontsize=11, fontweight="bold", color="white")

ax2.set_xticks(x2)
ax2.set_xticklabels(dist_labels, fontsize=10)
ax2.set_ylabel("Seed Distribution (%)")
ax2.set_ylim(0, 110)
ax2.set_yticks([0, 25, 50, 75, 100])
ax2.set_title("Seed Outcome Distribution: HIGH / MOD / LOW Integration", fontsize=13, fontweight="bold", pad=10)
ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)

# ---------- save ----------
out = "/Users/jacob/fun/shape-of-experience/empirical/experiments/study_ca_affect/phi_landscape_v22_v35.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved -> {out}")
