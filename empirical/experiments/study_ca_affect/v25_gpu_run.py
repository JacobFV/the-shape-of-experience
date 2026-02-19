"""V25 GPU Runner — Predator-Prey on Structured Landscape

Run 3 seeds on Lambda GPU. Expected runtime: ~20-30 min on A10
(N=256 is 4× the grid area of V20b's N=128, but same chunk structure).

Usage:
  python v25_gpu_run.py               # All 3 seeds
  python v25_gpu_run.py --seed 42     # Single seed
  python v25_gpu_run.py --smoke       # Smoke test (5 cycles, small)
"""

import argparse
import os
import time

os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=false'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()

    from v25_substrate import generate_v25_config
    from v25_evolution import run_v25

    if args.smoke:
        cfg = generate_v25_config(
            N=128,
            M_max=128,
            n_patches=6,
            patch_radius=15,
            patch_min_spacing=40,
            steps_per_cycle=1000,
            n_cycles=5,
            drought_every=3,
        )
        seeds = [42]
        result_base = 'results/v25_smoke'
    elif args.seed is not None:
        cfg = generate_v25_config(
            drought_every=5,
            activate_offspring=True,
        )
        seeds = [args.seed]
        result_base = f'results/v25_s{args.seed}'
    else:
        cfg = generate_v25_config(
            drought_every=5,
            activate_offspring=True,
        )
        seeds = [42, 123, 7]
        result_base = None

    print(f"V25 Configuration:")
    print(f"  Grid: {cfg['N']}x{cfg['N']}")
    print(f"  Population: {cfg['M_max']} ({cfg['n_prey']} prey + {cfg['n_pred']} pred)")
    print(f"  Patches: {cfg['n_patches']} (radius {cfg['patch_radius']})")
    print(f"  Hidden dim: {cfg['hidden_dim']}")
    print(f"  Obs dim: {cfg['obs_flat']}")
    print(f"  Params/agent: {cfg['n_params']}")
    print(f"  Cycles: {cfg['n_cycles']}")
    print(f"  Steps/cycle: {cfg['steps_per_cycle']}")
    print(f"  Drought every: {cfg.get('drought_every', 0)} cycles")
    print()

    t_total = time.time()

    for seed in seeds:
        out_dir = result_base if result_base else f'results/v25_s{seed}'
        print(f"{'='*60}")
        print(f"Running seed {seed} → {out_dir}")
        print(f"{'='*60}")

        t_seed = time.time()
        results = run_v25(seed, cfg, out_dir)
        print(f"Seed {seed} done in {time.time()-t_seed:.0f}s\n")

    print(f"\nTotal time: {time.time()-t_total:.0f}s")


if __name__ == '__main__':
    main()
