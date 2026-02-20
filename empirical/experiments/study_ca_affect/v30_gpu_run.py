"""V30 GPU Runner — Dual Prediction (Self + Social)

Usage:
    python v30_gpu_run.py run [seed]

Runs 3 seeds by default (42, 123, 7), or a specific seed if provided.
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v30_evolution import run_v30
from v30_substrate import generate_v30_config


def run_full(seed_filter=None):
    seeds = [42, 123, 7]
    if seed_filter:
        seeds = [int(seed_filter)]

    cfg = generate_v30_config()
    print(f"V30 Dual Prediction: {cfg['n_params']} params")
    print(f"  Baselines — V27 self: mean_phi=0.090, max=0.245")
    print(f"  Baselines — V29 social: mean_phi=0.104, max=0.243")

    all_results = {}
    t_total = time.time()

    for seed in seeds:
        output_dir = f'results/v30_s{seed}'
        results = run_v30(seed, cfg, output_dir)
        all_results[f's{seed}'] = results['summary']

    # Write combined summary
    summary = {}
    for key, val in all_results.items():
        summary[key] = {
            'mean_phi': val['mean_phi'],
            'max_phi': val['max_phi'],
            'mean_robustness': val['mean_robustness'],
            'max_robustness': val['max_robustness'],
            'mean_pred_mse': val['mean_pred_mse'],
            'mean_pred_mse_self': val['mean_pred_mse_self'],
            'mean_pred_mse_social': val['mean_pred_mse_social'],
            'final_pop': val['final_pop'],
        }

    with open('results/v30_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"V30 ALL SEEDS COMPLETE — {time.time()-t_total:.0f}s total")
    print(f"{'='*60}")
    for key, val in summary.items():
        print(f"  {key}: mean_phi={val['mean_phi']:.3f}, max_phi={val['max_phi']:.3f}, "
              f"rob={val['mean_robustness']:.3f}")
    print(f"\nBaseline comparison:")
    print(f"  V27 self:   mean=0.090, max=0.245 (seed 7)")
    print(f"  V29 social: mean=0.104, max=0.243 (seed 42)")


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'run':
        seed_filter = sys.argv[2] if len(sys.argv) > 2 else None
        run_full(seed_filter)
    else:
        print("Usage: python v30_gpu_run.py run [seed]")
