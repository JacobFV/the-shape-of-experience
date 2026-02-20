"""V32 GPU Runner — Drought Autopsy at Scale (50 Seeds)

The 30/70 question: why do ~30% of seeds develop high integration
while ~70% don't, given identical architecture and initial conditions?

V31 showed post-drought bounce predicts final Φ (r=0.997).
V32 asks: what determines whether a seed bounces?

Usage:
    python v32_gpu_run.py smoke             # Quick test (3 seeds, tiny grid)
    python v32_gpu_run.py run               # Full 50-seed run
    python v32_gpu_run.py run --seed 42     # Single seed
    python v32_gpu_run.py run --seeds 10    # First 10 seeds
"""

import sys
import os
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v27_substrate import generate_v27_config
from v32_evolution import run_v32


def run_smoke():
    """Quick smoke test: tiny grid, 3 seeds, 6 cycles (1 drought)."""
    print("V32 SMOKE TEST (N=32, M=32, K=4, 6 cycles, 3 seeds)")
    print("=" * 70)

    cfg = generate_v27_config(
        N=32, M_max=32,
        K_max=4,
        predict_hidden=4,
        steps_per_cycle=200,
        n_cycles=6,
        chunk_size=50,
        activate_offspring=True,
        drought_every=3,  # drought at cycle 3
    )
    print(f"  n_params: {cfg['n_params']}")

    for seed in [0, 1, 2]:
        result = run_v32(seed, cfg, f'/tmp/v32_smoke_s{seed}')
        s = result['summary']
        print(f"  Seed {seed}: phi={s['mean_phi']:.3f}, cat={s['category']}, "
              f"bounce={s['first_bounce']:.3f}")

    print("\nSmoke test complete!")


def run_full(seed_filter=None, n_seeds=50, output_base=None):
    """Full run: n_seeds with V27 architecture."""
    if output_base is None:
        output_base = 'results'

    if seed_filter is not None:
        seeds = [int(seed_filter)]
    else:
        seeds = list(range(n_seeds))

    cfg = generate_v27_config(
        N=128,
        M_max=256,
        K_max=8,
        predict_hidden=8,
        n_cycles=30,
        steps_per_cycle=5000,
        chunk_size=50,
        activate_offspring=True,
        drought_every=5,
    )

    print(f"\n{'='*70}")
    print(f"V32: DROUGHT AUTOPSY — {len(seeds)} seeds")
    print(f"  Architecture: V27 (GRU + 8 ticks + MLP self-prediction)")
    print(f"  Grid: {cfg['N']}x{cfg['N']}, max pop: {cfg['M_max']}")
    print(f"  Params per agent: {cfg['n_params']}")
    print(f"  Droughts: every {cfg['drought_every']} cycles")
    print(f"  Output: {output_base}/v32_s*/")
    print(f"{'='*70}\n")

    all_results = {}
    t_total = time.time()

    for seed in seeds:
        output_dir = os.path.join(output_base, f'v32_s{seed}')
        results = run_v32(seed, cfg, output_dir)
        all_results[f's{seed}'] = results['summary']

    # === CROSS-SEED ANALYSIS ===
    if len(seeds) >= 5:
        _write_summary(all_results, seeds, output_base, time.time() - t_total)


def _write_summary(all_results, seeds, output_base, elapsed):
    """Write cross-seed summary and basic classification."""
    categories = {'HIGH': [], 'MOD': [], 'LOW': []}
    mean_phis = []
    late_phis = []
    first_bounces = []
    mean_bounces = []
    eff_ranks = []
    weight_divs = []

    for seed in seeds:
        key = f's{seed}'
        if key not in all_results:
            continue
        s = all_results[key]
        categories[s['category']].append(seed)
        mean_phis.append(s['mean_phi'])
        late_phis.append(s['late_mean_phi'])
        first_bounces.append(s['first_bounce'])
        mean_bounces.append(s['mean_bounce'])
        eff_ranks.append(s['final_eff_rank'])
        weight_divs.append(s['final_weight_diversity'])

    mean_phis = np.array(mean_phis)
    late_phis = np.array(late_phis)
    first_bounces = np.array(first_bounces)
    mean_bounces = np.array(mean_bounces)
    eff_ranks = np.array(eff_ranks)
    weight_divs = np.array(weight_divs)

    n = len(mean_phis)

    # Correlations: what predicts late_mean_phi?
    correlations = {}
    for name, arr in [
        ('first_bounce', first_bounces),
        ('mean_bounce', mean_bounces),
        ('final_eff_rank', eff_ranks),
        ('final_weight_diversity', weight_divs),
    ]:
        if len(arr) > 2 and np.std(arr) > 1e-10 and np.std(late_phis) > 1e-10:
            r = float(np.corrcoef(arr, late_phis)[0, 1])
            correlations[name] = r
        else:
            correlations[name] = None

    summary = {
        'n_seeds': n,
        'elapsed_s': elapsed,
        'per_seed': {f's{seeds[i]}': all_results[f's{seeds[i]}']
                     for i in range(len(seeds)) if f's{seeds[i]}' in all_results},
        'distribution': {
            'HIGH': len(categories['HIGH']),
            'MOD': len(categories['MOD']),
            'LOW': len(categories['LOW']),
            'HIGH_seeds': categories['HIGH'],
            'MOD_seeds': categories['MOD'],
            'LOW_seeds': categories['LOW'],
            'HIGH_fraction': len(categories['HIGH']) / max(n, 1),
            'MOD_fraction': len(categories['MOD']) / max(n, 1),
            'LOW_fraction': len(categories['LOW']) / max(n, 1),
        },
        'statistics': {
            'mean_phi_mean': float(np.mean(mean_phis)),
            'mean_phi_std': float(np.std(mean_phis)),
            'mean_phi_sem': float(np.std(mean_phis) / np.sqrt(n)),
            'late_phi_mean': float(np.mean(late_phis)),
            'late_phi_std': float(np.std(late_phis)),
            'mean_rob_mean': float(np.mean([
                all_results[f's{s}']['mean_robustness'] for s in seeds
                if f's{s}' in all_results
            ])),
            'first_bounce_mean': float(np.mean(first_bounces)),
            'first_bounce_std': float(np.std(first_bounces)),
            'mean_bounce_mean': float(np.mean(mean_bounces)),
            'eff_rank_mean': float(np.mean(eff_ranks)),
        },
        'predictors_of_late_phi': correlations,
    }

    summary_path = os.path.join(output_base, 'v32_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"V32 ALL {n} SEEDS COMPLETE — {elapsed:.0f}s total")
    print(f"{'='*70}")
    print(f"\nDistribution:")
    print(f"  HIGH (Φ>0.10): {len(categories['HIGH'])}/{n} "
          f"({len(categories['HIGH'])/max(n,1):.0%}) — seeds {categories['HIGH'][:10]}")
    print(f"  MOD  (0.07-0.10): {len(categories['MOD'])}/{n} "
          f"({len(categories['MOD'])/max(n,1):.0%})")
    print(f"  LOW  (<0.07): {len(categories['LOW'])}/{n} "
          f"({len(categories['LOW'])/max(n,1):.0%})")
    print(f"\nStatistics:")
    print(f"  Mean Φ: {summary['statistics']['mean_phi_mean']:.4f} "
          f"± {summary['statistics']['mean_phi_std']:.4f}")
    print(f"  Late Φ: {summary['statistics']['late_phi_mean']:.4f} "
          f"± {summary['statistics']['late_phi_std']:.4f}")
    print(f"  First bounce: {summary['statistics']['first_bounce_mean']:.3f} "
          f"± {summary['statistics']['first_bounce_std']:.3f}")
    print(f"\nPredictors of late Φ:")
    for name, r in correlations.items():
        if r is not None:
            strength = "***" if abs(r) > 0.5 else "**" if abs(r) > 0.3 else "*" if abs(r) > 0.15 else ""
            print(f"  {name}: r = {r:.3f} {strength}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='V32 Drought Autopsy')
    parser.add_argument('command', nargs='?', default='run',
                        choices=['smoke', 'run'])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--seeds', type=int, default=50,
                        help='Number of seeds (default 50)')
    parser.add_argument('--output', default='results')
    args = parser.parse_args()

    if args.command == 'smoke':
        run_smoke()
    else:
        run_full(seed_filter=args.seed, n_seeds=args.seeds,
                 output_base=args.output)
