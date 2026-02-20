"""V33 GPU Runner — Contrastive Self-Prediction

Tests whether counterfactual reasoning (rung 8) is the missing ingredient
for pushing the HIGH fraction above 30%.

Usage:
    python v33_gpu_run.py smoke               # Quick test
    python v33_gpu_run.py run --seed 42       # Single seed
    python v33_gpu_run.py all                 # All 10 seeds
"""

import sys
import os
import argparse
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v33_substrate import generate_v33_config
from v33_evolution import run_v33


def run_smoke():
    print("V33 SMOKE TEST (N=32, M=32, K=4, 3 cycles)")
    print("=" * 70)
    cfg = generate_v33_config(
        N=32, M_max=32,
        K_max=4, predict_hidden=4,
        steps_per_cycle=200, n_cycles=3,
        chunk_size=50,
        activate_offspring=True,
        drought_every=0,
    )
    print(f"  n_params: {cfg['n_params']}")
    result = run_v33(seed=42, cfg=cfg, output_dir='/tmp/v33_smoke')
    print(f"\nSmoke test complete!")
    return result


def run_single(seed, output_base='results', n_cycles=30, steps_per_cycle=5000):
    output_dir = f'{output_base}/v33_s{seed}'
    cfg = generate_v33_config(
        N=128, M_max=256, K_max=8, predict_hidden=8,
        n_cycles=n_cycles, steps_per_cycle=steps_per_cycle,
        chunk_size=50, activate_offspring=True, drought_every=5,
    )
    print(f"\n{'='*70}")
    print(f"V33 CONTRASTIVE SEED {seed}: {n_cycles} cycles x {steps_per_cycle} steps")
    print(f"  n_params: {cfg['n_params']}")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}\n")
    return run_v33(seed=seed, cfg=cfg, output_dir=output_dir)


def run_all(output_base='results', **kwargs):
    """Run 10 seeds for statistical comparison with V27/V31."""
    seeds = [42, 123, 7, 0, 1, 2, 3, 4, 5, 6]
    results = {}
    t_total = time.time()

    for seed in seeds:
        results[seed] = run_single(seed, output_base=output_base, **kwargs)

    # Summary statistics
    mean_phis = [r['summary']['mean_phi'] for r in results.values()]
    late_phis = [r['summary']['late_mean_phi'] for r in results.values()]
    categories = [r['summary']['category'] for r in results.values()]

    n_high = categories.count('HIGH')
    n_mod = categories.count('MOD')
    n_low = categories.count('LOW')
    n = len(seeds)

    elapsed = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"V33 ALL {n} SEEDS COMPLETE — {elapsed:.0f}s")
    print(f"{'='*70}")
    print(f"  HIGH: {n_high}/{n} ({n_high/n:.0%})  [V31 baseline: 30%]")
    print(f"  MOD:  {n_mod}/{n}")
    print(f"  LOW:  {n_low}/{n}")
    print(f"  Mean Φ: {np.mean(mean_phis):.4f} ± {np.std(mean_phis):.4f}")
    print(f"  Late Φ: {np.mean(late_phis):.4f} ± {np.std(late_phis):.4f}")

    # t-test vs V27 baseline
    v27_baseline = 0.090
    t_stat = (np.mean(mean_phis) - v27_baseline) / (
        np.std(mean_phis) / np.sqrt(n))
    print(f"  t-stat vs V27 (0.090): {t_stat:.2f}")

    summary = {
        'per_seed': {f's{s}': r['summary'] for s, r in results.items()},
        'distribution': {'HIGH': n_high, 'MOD': n_mod, 'LOW': n_low},
        'statistics': {
            'mean_phi': float(np.mean(mean_phis)),
            'mean_phi_std': float(np.std(mean_phis)),
            'late_phi': float(np.mean(late_phis)),
            't_stat_vs_v27': float(t_stat),
        },
        'elapsed_s': elapsed,
    }
    with open(f'{output_base}/v33_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='V33 Contrastive Self-Prediction')
    parser.add_argument('command', nargs='?', default='run',
                        choices=['smoke', 'run', 'all'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cycles', type=int, default=30)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--output', default='results')
    args = parser.parse_args()

    if args.command == 'smoke':
        run_smoke()
    elif args.command == 'all':
        run_all(output_base=args.output, n_cycles=args.cycles,
                steps_per_cycle=args.steps)
    else:
        run_single(seed=args.seed, output_base=args.output,
                   n_cycles=args.cycles, steps_per_cycle=args.steps)
