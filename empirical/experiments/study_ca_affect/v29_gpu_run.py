"""V29: GPU Runner — Social Prediction

Usage:
  python v29_gpu_run.py smoke    # Quick test (1 cycle, 500 steps)
  python v29_gpu_run.py run      # Full 3-seed run
"""

import sys
import os
import json
import time

from v29_substrate import generate_v29_config
from v29_evolution import run_v29

SEEDS = [42, 123, 7]


def run_smoke():
    print("=" * 60)
    print("V29 SMOKE TEST — Social Prediction")
    print("=" * 60)

    base = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(base, 'results', 'v29_smoke')

    cfg = generate_v29_config(
        n_cycles=1,
        steps_per_cycle=500,
        chunk_size=50,
    )
    print(f"n_params={cfg['n_params']}")
    run_v29(42, cfg, out)
    print("\nSMOKE TEST COMPLETE")


def run_full():
    print("=" * 60)
    print("V29 SOCIAL PREDICTION — Full 3-seed run")
    print("=" * 60)

    base = os.path.dirname(os.path.abspath(__file__))
    all_results = {}
    t_total = time.time()

    for seed in SEEDS:
        cfg = generate_v29_config()
        out = os.path.join(base, 'results', f'v29_s{seed}')
        results = run_v29(seed, cfg, out)
        all_results[f's{seed}'] = results['summary']

    # Summary
    print("\n" + "=" * 60)
    print("COMPARATIVE SUMMARY")
    print("=" * 60)
    print(f"{'Seed':>6} {'Mean Φ':>8} {'Max Φ':>8} {'Mean Rob':>9} {'MSE':>8}")
    print("-" * 45)
    for key, s in sorted(all_results.items()):
        print(f"  {key:>4} {s['mean_phi']:8.3f} {s['max_phi']:8.3f} "
              f"{s['mean_robustness']:9.3f} {s['mean_pred_mse']:8.4f}")

    # Baselines
    print("\nBaselines:")
    print("  V22 (1-layer self):  mean Φ ≈ 0.097")
    print("  V27 (tanh w=8 self): mean Φ = 0.090, max = 0.245 (seed 7)")

    total_elapsed = time.time() - t_total
    print(f"\nTotal time: {total_elapsed/60:.1f} min")

    summary_path = os.path.join(base, 'results', 'v29_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved to {summary_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python v29_gpu_run.py {smoke|run}")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == 'smoke':
        run_smoke()
    elif cmd == 'run':
        run_full()
    else:
        print(f"Unknown: {cmd}")
        sys.exit(1)
