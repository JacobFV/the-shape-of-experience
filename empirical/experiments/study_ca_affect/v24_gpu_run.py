"""V24: GPU Runner — TD Value Learning

Usage:
  python v24_gpu_run.py smoke
  python v24_gpu_run.py single --seed 42
  python v24_gpu_run.py all
  python v24_gpu_run.py predict --seed 42
"""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v24_substrate import generate_v24_config
from v24_evolution import run_v24


def run_smoke():
    print("=" * 60)
    print("V24 SMOKE TEST")
    print("=" * 60)

    cfg = generate_v24_config(
        N=32, M_max=32, K_max=4,
        chunk_size=25, steps_per_cycle=100,
        n_cycles=3, drought_every=2,
    )
    print("Params: %d (H=%d, K=%d)" % (cfg['n_params'], cfg['hidden_dim'], cfg['K_max']))

    out_dir = 'results/v24_smoke'
    result = run_v24(seed=0, cfg=cfg, output_dir=out_dir)

    s = result['summary']
    print("\n--- Smoke test results ---")
    print("Mean robustness: %.3f" % s['mean_robustness'])
    print("Mean Phi: %.3f" % s['mean_phi'])
    print("Mean TD error: %.4f" % s['mean_td_error'])
    print("Mean value: %.3f" % s['mean_value'])
    print("Final LR: %.6f" % s['final_lr'])
    print("Final gamma: %.3f" % s['final_gamma'])
    print("SMOKE TEST PASSED")


def run_single(seed):
    cfg = generate_v24_config()
    print("=" * 60)
    print("V24 FULL RUN: seed=%d" % seed)
    print("N=%d, M=%d, K=%d, params=%d" % (
        cfg['N'], cfg['M_max'], cfg['K_max'], cfg['n_params']))
    print("=" * 60)

    out_dir = 'results/v24_s%d' % seed
    result = run_v24(seed=seed, cfg=cfg, output_dir=out_dir)

    predictions = evaluate_predictions(result)
    pred_path = os.path.join(out_dir, 'v24_s%d_predictions.json' % seed)
    with open(pred_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    print("\nPredictions saved to %s" % pred_path)

    return result


def run_all():
    seeds = [42, 123, 7]
    results = {}
    for seed in seeds:
        print("\n" + "=" * 70)
        print("SEED %d" % seed)
        print("=" * 70)
        results[seed] = run_single(seed)

    print("\n" + "=" * 70)
    print("CROSS-SEED SUMMARY")
    print("=" * 70)
    for seed in seeds:
        s = results[seed]['summary']
        print("Seed %3d: rob=%.3f (max %.3f) phi=%.3f "
              "td=%.4f V=%.3f lr=%.5f g=%.3f" % (
            seed, s['mean_robustness'], s['max_robustness'], s['mean_phi'],
            s['mean_td_error'], s['mean_value'], s['final_lr'], s['final_gamma'],
        ))

    mean_rob = np.mean([results[s]['summary']['mean_robustness'] for s in seeds])
    mean_phi = np.mean([results[s]['summary']['mean_phi'] for s in seeds])
    mean_gamma = np.mean([results[s]['summary']['final_gamma'] for s in seeds])
    print("\nAggregate: mean_rob=%.3f mean_phi=%.3f mean_gamma=%.3f" % (
        mean_rob, mean_phi, mean_gamma))


def evaluate_predictions(result):
    cycles = result['cycles']

    # P1: TD error decreases within lifetime (learning works)
    td_improving = 0
    for c in cycles:
        if c['td_error_early'] > c['td_error_late']:
            td_improving += 1
    n = len(cycles)
    p1 = {
        'supported': td_improving > n * 0.5,
        'td_improving_frac': td_improving / n,
        'mean_td_error': result['summary']['mean_td_error'],
    }

    # P2: Mean Phi > 0.11 (better than V22's ~0.10, V23's ~0.08)
    mean_phi = result['summary']['mean_phi']
    p2 = {
        'supported': mean_phi > 0.11,
        'mean_phi': mean_phi,
    }

    # P3: Robustness > 1.0
    mean_rob = result['summary']['mean_robustness']
    p3 = {
        'supported': mean_rob > 1.0,
        'mean_robustness': mean_rob,
    }

    # P4: gamma does NOT evolve to 0 (stays > 0.55)
    final_gamma = result['summary']['final_gamma']
    p4 = {
        'supported': final_gamma > 0.55,
        'final_gamma': final_gamma,
        'gamma_suppressed': result['summary']['gamma_suppressed'],
    }

    # P5: V(s) shows drought/normal differentiation
    # Check if value is lower in drought cycles vs normal
    drought_values = []
    normal_values = []
    for c in cycles:
        if c['cycle'] > 0 and c['cycle'] % 5 == 0:
            drought_values.append(c['mean_value'])
        else:
            normal_values.append(c['mean_value'])
    if drought_values and normal_values:
        drought_mean = np.mean(drought_values)
        normal_mean = np.mean(normal_values)
        p5 = {
            'supported': drought_mean < normal_mean,
            'drought_value_mean': float(drought_mean),
            'normal_value_mean': float(normal_mean),
            'value_difference': float(normal_mean - drought_mean),
        }
    else:
        p5 = {'supported': False, 'note': 'no drought/normal split available'}

    drift = {
        'mean_drift_overall': float(np.mean([c['phenotype_drift']['mean_drift'] for c in cycles])),
        'max_drift_overall': float(np.max([c['phenotype_drift']['max_drift'] for c in cycles])),
    }

    return {
        'prediction_1_td_learning': p1,
        'prediction_2_phi_increases': p2,
        'prediction_3_robustness_increases': p3,
        'prediction_4_gamma_maintained': p4,
        'prediction_5_drought_differentiation': p5,
        'drift': drift,
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python v24_gpu_run.py {smoke|single|all|predict}")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == 'smoke':
        run_smoke()
    elif cmd == 'single':
        seed = 42
        if '--seed' in sys.argv:
            idx = sys.argv.index('--seed')
            seed = int(sys.argv[idx + 1])
        run_single(seed)
    elif cmd == 'all':
        run_all()
    elif cmd == 'predict':
        seed = 42
        if '--seed' in sys.argv:
            idx = sys.argv.index('--seed')
            seed = int(sys.argv[idx + 1])
        result_path = 'results/v24_s%d/v24_s%d_results.json' % (seed, seed)
        if not os.path.exists(result_path):
            print("Results not found: %s" % result_path)
            sys.exit(1)
        with open(result_path) as f:
            result = json.load(f)
        preds = evaluate_predictions(result)
        print(json.dumps(preds, indent=2))
    else:
        print("Unknown command: %s" % cmd)
        sys.exit(1)
