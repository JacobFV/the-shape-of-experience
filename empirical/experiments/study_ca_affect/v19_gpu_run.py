"""V19 GPU Run: Bottleneck Furnace Mechanism Experiment on Lambda Labs.

Selection vs. Creation: Does the Bottleneck Furnace select pre-existing
high-Phi variants, or does it create novel-stress generalization capacity?

Usage:
    python v19_gpu_run.py smoke                    # Quick CPU test (C=8, N=64)
    python v19_gpu_run.py run --seed 42            # Single seed, full experiment
    python v19_gpu_run.py all                      # All 3 seeds sequentially
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v19_experiment import run_v19


def run_smoke():
    print("V19 SMOKE TEST (C=8, N=64, 2+2+2 cycles)")
    print("=" * 70)
    result = run_v19(
        seed=42, C=8, N=64,
        steps_per_cycle=500,
        n_phase1=2, n_phase2=2, n_phase3=2,
        output_dir='/tmp/v19_smoke',
    )
    print("\nSmoke test complete!")
    if 'analysis' in result:
        print(f"Verdict: {result['analysis'].get('verdict', 'N/A')}")
    return result


def run_single(seed, output_base='/home/ubuntu/results',
               n_phase1=10, n_phase2=10, n_phase3=5, steps_per_cycle=5000):
    output_dir = f'{output_base}/v19_s{seed}'
    print(f"\n{'='*70}")
    print(f"V19 SEED {seed}: {n_phase1}+{n_phase2}+{n_phase3} cycles")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    return run_v19(
        seed=seed, C=16, N=128,
        steps_per_cycle=steps_per_cycle,
        n_phase1=n_phase1,
        n_phase2=n_phase2,
        n_phase3=n_phase3,
        output_dir=output_dir,
    )


def run_all(output_base='/home/ubuntu/results', **kwargs):
    for seed in [42, 123, 7]:
        run_single(seed, output_base=output_base, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='V19 Bottleneck Furnace Mechanism Experiment')
    parser.add_argument('command', nargs='?', default='run',
                        choices=['smoke', 'run', 'all'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--phase1', type=int, default=10,
                        help='Cycles in Phase 1 (shared)')
    parser.add_argument('--phase2', type=int, default=10,
                        help='Cycles in Phase 2 (per condition)')
    parser.add_argument('--phase3', type=int, default=5,
                        help='Cycles in Phase 3 (novel stress, frozen)')
    parser.add_argument('--steps', type=int, default=5000,
                        help='Steps per cycle')
    parser.add_argument('--output', default='/home/ubuntu/results')
    args = parser.parse_args()

    if args.command == 'smoke':
        run_smoke()
    elif args.command == 'all':
        run_all(
            output_base=args.output,
            n_phase1=args.phase1,
            n_phase2=args.phase2,
            n_phase3=args.phase3,
            steps_per_cycle=args.steps,
        )
    else:
        run_single(
            seed=args.seed,
            output_base=args.output,
            n_phase1=args.phase1,
            n_phase2=args.phase2,
            n_phase3=args.phase3,
            steps_per_cycle=args.steps,
        )
