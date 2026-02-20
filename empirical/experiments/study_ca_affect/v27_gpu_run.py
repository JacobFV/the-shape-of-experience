"""V27 GPU Run: Nonlinear MLP Prediction Head on Lambda Labs.

Tests whether replacing the linear prediction head with a 2-layer MLP
creates the gradient coupling needed for integration (Φ increase).

Usage:
    python v27_gpu_run.py smoke                   # Quick CPU test
    python v27_gpu_run.py run --seed 42           # Single seed, full run
    python v27_gpu_run.py all                     # All 3 seeds sequentially
"""

import sys
import os
import argparse
import json
import time
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v27_substrate import generate_v27_config, init_v27, make_v27_chunk_runner
from v27_evolution import run_v27


def run_smoke():
    """Quick smoke test: tiny grid, few cycles."""
    print("V27 SMOKE TEST (N=32, M_max=32, K_max=4, 3 cycles)")
    print("=" * 70)
    cfg = generate_v27_config(
        N=32, M_max=32,
        K_max=4,
        predict_hidden=4,
        steps_per_cycle=200,
        n_cycles=3,
        chunk_size=50,
        activate_offspring=True,
        drought_every=0,
    )
    print(f"  n_params per agent: {cfg['n_params']}")
    print(f"  obs_flat dim:       {cfg['obs_flat']}")
    print(f"  K_max:              {cfg['K_max']}")
    print(f"  predict_hidden:     {cfg['predict_hidden']}")
    print(f"  JAX devices:        {jax.devices()}")

    result = run_v27(seed=42, cfg=cfg, output_dir='/tmp/v27_smoke')

    print(f"\nSmoke test complete!")
    print(f"  Mean robustness:   {result['summary']['mean_robustness']:.3f}")
    print(f"  Mean pred MSE:     {result['summary']['mean_pred_mse']:.4f}")
    print(f"  MSE improvement:   {result['summary']['pred_mse_improvement']:.4f}")
    print(f"  Final LR:          {result['summary']['final_lr']:.6f}")
    return result


def run_single(seed, output_base='/home/ubuntu/results',
               n_cycles=30, steps_per_cycle=5000):
    """Full single-seed run."""
    output_dir = f'{output_base}/v27_s{seed}'
    cfg = generate_v27_config(
        N=128,
        M_max=256,
        K_max=8,
        predict_hidden=8,  # H // 2
        n_cycles=n_cycles,
        steps_per_cycle=steps_per_cycle,
        chunk_size=50,
        activate_offspring=True,
        drought_every=5,
    )

    print(f"\n{'=' * 70}")
    print(f"V27 SEED {seed}: {n_cycles} cycles x {steps_per_cycle} steps, K_max=8")
    print(f"  n_params per agent: {cfg['n_params']}")
    print(f"  predict_hidden:     {cfg['predict_hidden']}")
    print(f"  Grid: {cfg['N']}x{cfg['N']}, max pop: {cfg['M_max']}")
    print(f"  Output: {output_dir}")
    print(f"  JAX devices: {jax.devices()}")
    print(f"{'=' * 70}\n")

    return run_v27(seed=seed, cfg=cfg, output_dir=output_dir)


def run_all(output_base='/home/ubuntu/results', **kwargs):
    """Run all 3 seeds sequentially."""
    results = {}
    for seed in [42, 123, 7]:
        results[seed] = run_single(seed, output_base=output_base, **kwargs)

    print("\n" + "=" * 70)
    print("ALL SEEDS COMPLETE")
    print("=" * 70)
    for seed, r in results.items():
        s = r['summary']
        print(f"  Seed {seed}: mean_rob={s['mean_robustness']:.3f}, "
              f"max_rob={s['max_robustness']:.3f}, "
              f"mean_phi={s['mean_phi']:.3f}, "
              f"mse={s['mean_pred_mse']:.4f}, "
              f"lr={s['final_lr']:.6f}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='V27 Nonlinear MLP Prediction Head')
    parser.add_argument('command', nargs='?', default='run',
                        choices=['smoke', 'run', 'all'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cycles', type=int, default=30)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--output', default='/home/ubuntu/results')
    args = parser.parse_args()

    if args.command == 'smoke':
        run_smoke()
    elif args.command == 'all':
        run_all(output_base=args.output, n_cycles=args.cycles,
                steps_per_cycle=args.steps)
    else:
        run_single(seed=args.seed, output_base=args.output,
                   n_cycles=args.cycles, steps_per_cycle=args.steps)
