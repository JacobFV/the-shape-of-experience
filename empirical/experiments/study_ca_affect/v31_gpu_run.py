"""V31 GPU Runner — V29 Social Prediction × 10 Seeds

Statistical validation of V29's social prediction effect.
V29 showed mean Φ 0.104 across 3 seeds (vs V27's 0.090).
V31 runs 10 seeds to compute confidence intervals and test significance.

Uses V29 substrate/evolution unchanged — only more seeds.

Usage:
    python v31_gpu_run.py run [seed]

Runs 10 seeds by default, or a specific seed if provided.
"""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v29_evolution import run_v29
from v29_substrate import generate_v29_config


def run_full(seed_filter=None):
    # 10 seeds: the original 3 + 7 new
    seeds = [42, 123, 7, 0, 1, 2, 3, 4, 5, 6]
    if seed_filter:
        seeds = [int(seed_filter)]

    cfg = generate_v29_config()
    print(f"V31: V29 Social Prediction × {len(seeds)} seeds")
    print(f"  {cfg['n_params']} params")
    print(f"  Baseline V27 self: mean_phi=0.090 (3 seeds)")
    print(f"  Baseline V29 social: mean_phi=0.104 (3 seeds)")

    all_results = {}
    t_total = time.time()

    for seed in seeds:
        output_dir = f'results/v31_s{seed}'
        results = run_v29(seed, cfg, output_dir)
        all_results[f's{seed}'] = results['summary']

    # Write combined summary
    summary = {}
    mean_phis = []
    max_phis = []
    mean_robs = []

    for key, val in all_results.items():
        summary[key] = {
            'mean_phi': val['mean_phi'],
            'max_phi': val['max_phi'],
            'mean_robustness': val['mean_robustness'],
            'max_robustness': val['max_robustness'],
            'mean_pred_mse': val['mean_pred_mse'],
            'final_pop': val['final_pop'],
        }
        mean_phis.append(val['mean_phi'])
        max_phis.append(val['max_phi'])
        mean_robs.append(val['mean_robustness'])

    # Statistics
    mean_phis = np.array(mean_phis)
    max_phis = np.array(max_phis)
    mean_robs = np.array(mean_robs)

    stats = {
        'n_seeds': len(seeds),
        'mean_phi_mean': float(np.mean(mean_phis)),
        'mean_phi_std': float(np.std(mean_phis)),
        'mean_phi_sem': float(np.std(mean_phis) / np.sqrt(len(seeds))),
        'mean_phi_ci95_lo': float(np.mean(mean_phis) - 1.96 * np.std(mean_phis) / np.sqrt(len(seeds))),
        'mean_phi_ci95_hi': float(np.mean(mean_phis) + 1.96 * np.std(mean_phis) / np.sqrt(len(seeds))),
        'max_phi_mean': float(np.mean(max_phis)),
        'max_phi_max': float(np.max(max_phis)),
        'mean_rob_mean': float(np.mean(mean_robs)),
        'mean_rob_std': float(np.std(mean_robs)),
        # Simple t-test against V27 baseline of 0.090
        'v27_baseline_mean_phi': 0.090,
        't_stat_vs_v27': float((np.mean(mean_phis) - 0.090) / (np.std(mean_phis) / np.sqrt(len(seeds)))),
    }

    summary['statistics'] = stats

    with open('results/v31_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"V31 ALL {len(seeds)} SEEDS COMPLETE — {elapsed:.0f}s total")
    print(f"{'='*60}")
    for key, val in summary.items():
        if key == 'statistics':
            continue
        print(f"  {key}: mean_phi={val['mean_phi']:.3f}, max_phi={val['max_phi']:.3f}")

    print(f"\nStatistics:")
    print(f"  Mean Φ: {stats['mean_phi_mean']:.4f} ± {stats['mean_phi_std']:.4f}")
    print(f"  95% CI: [{stats['mean_phi_ci95_lo']:.4f}, {stats['mean_phi_ci95_hi']:.4f}]")
    print(f"  SEM: {stats['mean_phi_sem']:.4f}")
    print(f"  t-stat vs V27 (0.090): {stats['t_stat_vs_v27']:.2f}")
    print(f"  Max Φ across all seeds: {stats['max_phi_max']:.3f}")
    print(f"  Mean robustness: {stats['mean_rob_mean']:.4f} ± {stats['mean_rob_std']:.4f}")


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'run':
        seed_filter = sys.argv[2] if len(sys.argv) > 2 else None
        run_full(seed_filter)
    else:
        print("Usage: python v31_gpu_run.py run [seed]")
