"""V26 GPU Runner — POMDP with partial observability.

Usage:
  python v26_gpu_run.py              # All 3 seeds
  python v26_gpu_run.py --seed 42    # Single seed
  python v26_gpu_run.py --smoke      # Quick test (5 cycles)
"""

import argparse
import os
import time

from v26_substrate import generate_v26_config
from v26_evolution import run_v26


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()

    overrides = {}
    if args.smoke:
        overrides = {
            'n_cycles': 5,
            'steps_per_cycle': 1000,
            'M_max': 128,
        }

    cfg = generate_v26_config(**overrides)

    print("=" * 60)
    print("V26: POMDP — Partial Observability Forces Internal Representation")
    print("=" * 60)
    print(f"  Grid: {cfg['N']}×{cfg['N']}")
    print(f"  Agents: {cfg['M_max']} ({cfg['n_prey']} prey + {cfg['n_pred']} pred)")
    print(f"  Obs: 1×1 + compass = {cfg['obs_flat']} dims")
    print(f"  Hidden: {cfg['hidden_dim']}")
    print(f"  Params per agent: {cfg['n_params']}")
    print(f"  Compass noise: σ={cfg['compass_noise_std']}")
    print(f"  Cycles: {cfg['n_cycles']}, Steps/cycle: {cfg['steps_per_cycle']}")
    print()

    seeds = [args.seed] if args.seed else [42, 123, 7]
    base = os.path.join(os.path.dirname(__file__), 'results')

    total_start = time.time()
    for seed in seeds:
        output_dir = os.path.join(base, f'v26_s{seed}')
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")
        run_v26(seed, cfg, output_dir)

    total = time.time() - total_start
    print(f"\nAll seeds done in {total:.0f}s ({total/60:.1f}min)")


if __name__ == '__main__':
    main()
