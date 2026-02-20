"""V28: GPU Runner — Bottleneck Width Sweep

Runs 3 conditions × 3 seeds = 9 runs total:
  A) linear_w8:  predict_activation='linear', predict_hidden=8
  B) tanh_w4:    predict_activation='tanh',   predict_hidden=4
  C) tanh_w16:   predict_activation='tanh',   predict_hidden=16

Comparison baselines:
  V22: linear 1-layer (pred = h @ W + b) → mean Φ ~0.097
  V27: tanh 2-layer w=8 → seed 7 Φ=0.245, others 0.071-0.079

Usage:
  python v28_gpu_run.py smoke        # Quick test (1 cycle, 500 steps)
  python v28_gpu_run.py run          # Full sweep (30 cycles, 5000 steps)
  python v28_gpu_run.py run A 7      # Single condition + seed
"""

import sys
import os
import json
import time

from v28_substrate import generate_v28_config
from v28_evolution import run_v28


# Condition definitions
CONDITIONS = {
    'A': {'name': 'linear_w8',  'predict_activation': 'linear', 'predict_hidden': 8},
    'B': {'name': 'tanh_w4',    'predict_activation': 'tanh',   'predict_hidden': 4},
    'C': {'name': 'tanh_w16',   'predict_activation': 'tanh',   'predict_hidden': 16},
}

SEEDS = [42, 123, 7]


def run_smoke():
    """Quick smoke test — 1 cycle per condition, seed 42 only."""
    print("=" * 70)
    print("V28 SMOKE TEST")
    print("=" * 70)

    base = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base, 'results', 'v28_smoke')
    os.makedirs(results_dir, exist_ok=True)

    for cond_key, cond in CONDITIONS.items():
        cfg = generate_v28_config(
            predict_activation=cond['predict_activation'],
            predict_hidden=cond['predict_hidden'],
            n_cycles=1,
            steps_per_cycle=500,
            chunk_size=50,
        )
        print(f"\nCondition {cond_key} ({cond['name']}): "
              f"activation={cond['predict_activation']}, "
              f"predict_hidden={cond['predict_hidden']}, "
              f"n_params={cfg['n_params']}")

        out_dir = os.path.join(results_dir, cond['name'])
        run_v28(42, cfg, out_dir, condition_name=cond['name'])

    print("\n" + "=" * 70)
    print("SMOKE TEST COMPLETE")
    print("=" * 70)


def run_full(condition_filter=None, seed_filter=None):
    """Full 3-condition × 3-seed sweep."""
    print("=" * 70)
    print("V28 BOTTLENECK WIDTH SWEEP")
    print("=" * 70)

    base = os.path.dirname(os.path.abspath(__file__))

    all_results = {}
    total_start = time.time()

    for cond_key, cond in CONDITIONS.items():
        if condition_filter and cond_key != condition_filter:
            continue

        for seed in SEEDS:
            if seed_filter and seed != seed_filter:
                continue

            label = cond['name']
            cfg = generate_v28_config(
                predict_activation=cond['predict_activation'],
                predict_hidden=cond['predict_hidden'],
            )

            out_dir = os.path.join(base, 'results', f'v28_{label}_s{seed}')
            results = run_v28(seed, cfg, out_dir, condition_name=label)
            all_results[f'{label}_s{seed}'] = results['summary']

    # Print comparative summary
    print("\n" + "=" * 70)
    print("COMPARATIVE SUMMARY")
    print("=" * 70)
    print(f"{'Condition':<20} {'Seed':>5} {'Mean Φ':>8} {'Max Φ':>8} "
          f"{'Mean Rob':>9} {'MSE':>8} {'Pop':>5}")
    print("-" * 65)
    for key, summary in sorted(all_results.items()):
        parts = key.rsplit('_s', 1)
        cond_name = parts[0]
        seed = parts[1]
        print(f"  {cond_name:<18} {seed:>5} {summary['mean_phi']:8.3f} "
              f"{summary['max_phi']:8.3f} {summary['mean_robustness']:9.3f} "
              f"{summary['mean_pred_mse']:8.4f} {summary['final_pop']:5d}")

    total_elapsed = time.time() - total_start
    print(f"\nTotal time: {total_elapsed/60:.1f} min")

    # Save summary
    summary_path = os.path.join(base, 'results', 'v28_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python v28_gpu_run.py {smoke|run} [condition] [seed]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == 'smoke':
        run_smoke()
    elif cmd == 'run':
        condition_filter = sys.argv[2] if len(sys.argv) > 2 else None
        seed_filter = int(sys.argv[3]) if len(sys.argv) > 3 else None
        run_full(condition_filter, seed_filter)
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
