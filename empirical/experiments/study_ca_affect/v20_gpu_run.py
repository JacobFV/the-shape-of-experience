"""V20 GPU Run: Protocell Agency on Lambda Labs.

The Necessity Chain Experiment — tests whether genuine action-observation
loops produce the world model → self-model → affect geometry cascade.

Usage:
    python v20_gpu_run.py smoke                   # Quick CPU test (N=32)
    python v20_gpu_run.py run --seed 42            # Single seed, full run
    python v20_gpu_run.py all                      # All 3 seeds sequentially
    python v20_gpu_run.py chain --seed 42          # Chain test on saved snapshots
"""

import sys
import os
import argparse
import json
import jax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v20_substrate import generate_v20_config
from v20_evolution import run_v20
from v20_experiments import run_all_chain_tests


def run_smoke():
    """Quick smoke test: tiny grid, few cycles, verify correctness."""
    print("V20 SMOKE TEST (N=32, M_max=32, 3 cycles)")
    print("=" * 70)
    cfg = generate_v20_config(
        N=32, M_max=32,
        steps_per_cycle=200,
        n_cycles=3,
        chunk_size=50,
    )
    print(f"  n_params per agent: {cfg['n_params']}")
    print(f"  obs_flat dim:       {cfg['obs_flat']}")

    result = run_v20(seed=42, cfg=cfg, output_dir='/tmp/v20_smoke')

    print(f"\nSmoke test complete!")
    print(f"  Mean robustness: {result['summary']['mean_robustness']:.3f}")
    print(f"  Max robustness:  {result['summary']['max_robustness']:.3f}")
    return result


def run_single(seed, output_base='/home/ubuntu/results',
               n_cycles=30, steps_per_cycle=5000, activate_offspring=False,
               drought_every=0):
    """Full single-seed run."""
    variant = 'b' if activate_offspring else ''
    output_dir = f'{output_base}/v20{variant}_s{seed}'
    cfg = generate_v20_config(
        N=128,
        M_max=256,
        n_cycles=n_cycles,
        steps_per_cycle=steps_per_cycle,
        chunk_size=50,
        activate_offspring=activate_offspring,
        drought_every=drought_every,
    )

    print(f"\n{'=' * 70}")
    print(f"V20 SEED {seed}: {n_cycles} cycles × {steps_per_cycle} steps")
    print(f"  n_params per agent: {cfg['n_params']}")
    print(f"  Grid: {cfg['N']}×{cfg['N']}, max pop: {cfg['M_max']}")
    print(f"  Output: {output_dir}")
    print(f"  JAX devices: {jax.devices()}")
    print(f"{'=' * 70}\n")

    return run_v20(seed=seed, cfg=cfg, output_dir=output_dir)


def run_all(output_base='/home/ubuntu/results', activate_offspring=False,
            drought_every=0, **kwargs):
    """Run all 3 seeds sequentially."""
    results = {}
    for seed in [42, 123, 7]:
        results[seed] = run_single(seed, output_base=output_base,
                                   activate_offspring=activate_offspring,
                                   drought_every=drought_every, **kwargs)

    print("\n" + "=" * 70)
    print("ALL SEEDS COMPLETE")
    print("=" * 70)
    for seed, r in results.items():
        s = r['summary']
        print(f"  Seed {seed}: mean_rob={s['mean_robustness']:.3f}, "
              f"max_rob={s['max_robustness']:.3f}, "
              f"mean_phi={s['mean_phi']:.3f}")

    return results


def run_chain(seed, output_base='/home/ubuntu/results'):
    """Run chain test on saved evolution snapshots."""
    results_dir = f'{output_base}/v20_s{seed}'
    cfg = generate_v20_config(N=128, M_max=256, steps_per_cycle=5000, chunk_size=50)
    key = jax.random.PRNGKey(seed + 100)
    return run_all_chain_tests(results_dir, cfg, seed, key)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='V20 Protocell Agency Experiment')
    parser.add_argument('command', nargs='?', default='run',
                        choices=['smoke', 'run', 'all', 'chain'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cycles', type=int, default=30)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--output', default='/home/ubuntu/results')
    parser.add_argument('--v20b', action='store_true',
                        help='Activate offspring after tournament (fixes V20 mort=0% bug)')
    parser.add_argument('--drought', type=int, default=0,
                        help='Apply drought every N cycles (0=off). E.g. --drought 5')
    args = parser.parse_args()

    if args.command == 'smoke':
        run_smoke()
    elif args.command == 'all':
        run_all(output_base=args.output, n_cycles=args.cycles,
                steps_per_cycle=args.steps, activate_offspring=args.v20b,
                drought_every=args.drought)
    elif args.command == 'chain':
        run_chain(args.seed, output_base=args.output)
    else:
        run_single(seed=args.seed, output_base=args.output,
                   n_cycles=args.cycles, steps_per_cycle=args.steps,
                   activate_offspring=args.v20b, drought_every=args.drought)
