"""V35 GPU Runner — Language Emergence under Cooperative POMDP

Tests whether discrete communication emerges under partial observability
with cooperative pressure.

Usage:
    python v35_gpu_run.py smoke               # Quick test
    python v35_gpu_run.py run --seed 42       # Single seed
    python v35_gpu_run.py all                 # All 10 seeds
"""

import sys
import os
import argparse
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v35_substrate import generate_v35_config
from v35_evolution import run_v35


def run_smoke():
    print("V35 SMOKE TEST (N=32, M=32, K=4, 3 cycles)")
    print("=" * 70)
    cfg = generate_v35_config(
        N=32, M_max=32,
        K_max=4, predict_hidden=4,
        steps_per_cycle=200, n_cycles=3,
        chunk_size=50,
        activate_offspring=True,
        drought_every=0,
        obs_radius=1,
        K_sym=8,
        comm_radius=5,
        coop_bonus=0.5,
    )
    print(f"  n_params: {cfg['n_params']}")
    print(f"  obs_flat: {cfg['obs_flat']} (3×3 patch + energy + sym_hist)")
    print(f"  n_actions: {cfg['n_actions']} (5 move + 1 consume + 8 symbols)")
    result = run_v35(seed=42, cfg=cfg, output_dir='/tmp/v35_smoke')
    print(f"\nSmoke test complete!")
    return result


def run_single(seed, output_base='results', n_cycles=30, steps_per_cycle=5000):
    output_dir = f'{output_base}/v35_s{seed}'
    cfg = generate_v35_config(
        N=128, M_max=256, K_max=8, predict_hidden=8,
        n_cycles=n_cycles, steps_per_cycle=steps_per_cycle,
        chunk_size=50, activate_offspring=True, drought_every=5,
        obs_radius=1,      # 3×3 obs (partial observability)
        K_sym=8,            # 8 discrete symbols
        comm_radius=5,      # Hear further than see
        coop_bonus=0.5,     # Cooperative consumption bonus
    )
    print(f"\n{'='*70}")
    print(f"V35 LANGUAGE EMERGENCE SEED {seed}")
    print(f"  obs: {cfg['obs_side']}×{cfg['obs_side']} (radius={cfg['obs_radius']})")
    print(f"  comm_radius: {cfg['comm_radius']} (hear > see)")
    print(f"  K_sym: {cfg['K_sym']} discrete symbols")
    print(f"  coop_bonus: {cfg['coop_bonus']}")
    print(f"  n_params: {cfg['n_params']}")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}\n")
    return run_v35(seed=seed, cfg=cfg, output_dir=output_dir)


def run_all(output_base='results', **kwargs):
    """Run 10 seeds for statistical comparison."""
    seeds = [42, 123, 7, 0, 1, 2, 3, 4, 5, 6]
    results = {}
    t_total = time.time()

    for seed in seeds:
        results[seed] = run_single(seed, output_base=output_base, **kwargs)

    # Summary statistics
    mean_phis = [r['summary']['mean_phi'] for r in results.values()]
    late_phis = [r['summary']['late_mean_phi'] for r in results.values()]
    categories = [r['summary']['category'] for r in results.values()]
    lang_statuses = [r['summary']['lang_status'] for r in results.values()]
    entropies = [r['summary']['mean_sym_entropy'] for r in results.values()]
    mi_proxies = [r['summary']['mean_sym_resource_mi'] for r in results.values()]
    comm_lifts = [r['summary']['comm_phi_lift'] for r in results.values()]

    n_high = categories.count('HIGH')
    n_ref = lang_statuses.count('REFERENTIAL')
    n = len(seeds)

    elapsed = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"V35 ALL {n} SEEDS COMPLETE — {elapsed:.0f}s")
    print(f"{'='*70}")
    print(f"  Integration:")
    print(f"    HIGH: {n_high}/{n} ({n_high/n:.0%})  [V31 baseline: 30%]")
    print(f"    Mean Φ: {np.mean(mean_phis):.4f} ± {np.std(mean_phis):.4f}")
    print(f"    Late Φ: {np.mean(late_phis):.4f} ± {np.std(late_phis):.4f}")
    print(f"  Language:")
    print(f"    REFERENTIAL: {n_ref}/{n}")
    print(f"    Mean entropy: {np.mean(entropies):.2f} ± {np.std(entropies):.2f} bits")
    print(f"    Mean MI proxy: {np.mean(mi_proxies):.4f}")
    print(f"    Mean comm Φ lift: {np.mean(comm_lifts):.4f}")

    # t-test vs V27 baseline
    v27_baseline = 0.090
    t_stat = (np.mean(mean_phis) - v27_baseline) / (
        np.std(mean_phis) / np.sqrt(n))
    print(f"  t-stat vs V27 (0.090): {t_stat:.2f}")

    summary = {
        'per_seed': {f's{s}': r['summary'] for s, r in results.items()},
        'distribution': {
            'HIGH': n_high,
            'MOD': categories.count('MOD'),
            'LOW': categories.count('LOW'),
        },
        'language': {
            'REFERENTIAL': n_ref,
            'DIVERSE_NOISE': lang_statuses.count('DIVERSE_NOISE'),
            'COLLAPSED': lang_statuses.count('COLLAPSED'),
        },
        'statistics': {
            'mean_phi': float(np.mean(mean_phis)),
            'mean_phi_std': float(np.std(mean_phis)),
            'late_phi': float(np.mean(late_phis)),
            'mean_entropy': float(np.mean(entropies)),
            'mean_mi_proxy': float(np.mean(mi_proxies)),
            'mean_comm_lift': float(np.mean(comm_lifts)),
            't_stat_vs_v27': float(t_stat),
        },
        'elapsed_s': elapsed,
    }
    with open(f'{output_base}/v35_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='V35 Language Emergence')
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
