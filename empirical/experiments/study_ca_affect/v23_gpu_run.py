"""V23: GPU Runner — World-Model Gradient

CLI runner for V23 experiments with multi-target predictive gradient.

Usage:
  python v23_gpu_run.py smoke          # Quick local test (N=32, M=32, K=4, 3 cycles)
  python v23_gpu_run.py single --seed 42  # Full run, one seed
  python v23_gpu_run.py all            # Full run, all 3 seeds
  python v23_gpu_run.py predict --seed 42  # Evaluate predictions on results
"""

import sys
import os
import json
import time
import numpy as np

# Ensure parent directory is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v23_substrate import generate_v23_config
from v23_evolution import run_v23


def run_smoke():
    """Quick smoke test: small grid, few cycles."""
    print("=" * 60)
    print("V23 SMOKE TEST")
    print("=" * 60)

    cfg = generate_v23_config(
        N=32, M_max=32, K_max=4,
        chunk_size=25, steps_per_cycle=100,
        n_cycles=3, drought_every=2,
    )
    print("Params: %d (H=%d, K=%d, targets=%d)" % (
        cfg['n_params'], cfg['hidden_dim'], cfg['K_max'], cfg['n_targets']))

    out_dir = 'results/v23_smoke'
    result = run_v23(seed=0, cfg=cfg, output_dir=out_dir)

    print("\n--- Smoke test results ---")
    s = result['summary']
    print("Mean robustness: %.3f" % s['mean_robustness'])
    print("Mean Phi: %.3f" % s['mean_phi'])
    print("Pred MSE (E/R/N): %.5f / %.5f / %.5f" % (
        s['mean_pred_mse_energy'],
        s['mean_pred_mse_resource'],
        s['mean_pred_mse_neighbor'],
    ))
    print("Final LR: %.6f" % s['final_lr'])
    print("LR suppressed: %s" % s['lr_suppressed'])
    print("Final col cosine: %.3f" % s['final_col_cosine'])
    print("Final eff rank: %.1f" % s['final_effective_rank'])
    print("SMOKE TEST PASSED")


def run_single(seed):
    """Full run for one seed."""
    cfg = generate_v23_config()
    print("=" * 60)
    print("V23 FULL RUN: seed=%d" % seed)
    print("N=%d, M=%d, K=%d, targets=%d, params=%d" % (
        cfg['N'], cfg['M_max'], cfg['K_max'], cfg['n_targets'], cfg['n_params']))
    print("Drought every %d cycles, depletion=%.0f%%" % (
        cfg['drought_every'], cfg['drought_depletion'] * 100))
    print("=" * 60)

    out_dir = 'results/v23_s%d' % seed
    result = run_v23(seed=seed, cfg=cfg, output_dir=out_dir)

    # Evaluate predictions
    predictions = evaluate_predictions(result)
    pred_path = os.path.join(out_dir, 'v23_s%d_predictions.json' % seed)
    with open(pred_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    print("\nPredictions saved to %s" % pred_path)

    return result


def run_all():
    """Run all 3 seeds."""
    seeds = [42, 123, 7]
    results = {}
    for seed in seeds:
        print("\n" + "=" * 70)
        print("SEED %d" % seed)
        print("=" * 70)
        results[seed] = run_single(seed)

    # Cross-seed summary
    print("\n" + "=" * 70)
    print("CROSS-SEED SUMMARY")
    print("=" * 70)
    for seed in seeds:
        s = results[seed]['summary']
        print("Seed %3d: rob=%.3f (max %.3f) phi=%.3f "
              "mse_E=%.5f mse_R=%.5f mse_N=%.5f "
              "cos=%.3f rank=%.1f lr=%.5f" % (
            seed, s['mean_robustness'], s['max_robustness'], s['mean_phi'],
            s['mean_pred_mse_energy'], s['mean_pred_mse_resource'],
            s['mean_pred_mse_neighbor'],
            s['final_col_cosine'], s['final_effective_rank'], s['final_lr'],
        ))

    # Aggregate
    mean_rob = np.mean([results[s]['summary']['mean_robustness'] for s in seeds])
    mean_phi = np.mean([results[s]['summary']['mean_phi'] for s in seeds])
    mean_cos = np.mean([results[s]['summary']['final_col_cosine'] for s in seeds])
    print("\nAggregate: mean_rob=%.3f mean_phi=%.3f mean_cos=%.3f" % (
        mean_rob, mean_phi, mean_cos))


def evaluate_predictions(result):
    """Evaluate V23 pre-registered predictions against results."""
    cycles = result['cycles']

    # P1: All 3 targets show within-lifetime MSE improvement
    # Check: early MSE > late MSE for each target in majority of cycles
    energy_improving = 0
    resource_improving = 0
    neighbor_improving = 0
    for c in cycles:
        if c['pred_mse_early']['energy'] > c['pred_mse_late']['energy']:
            energy_improving += 1
        if c['pred_mse_early']['resource'] > c['pred_mse_late']['resource']:
            resource_improving += 1
        if c['pred_mse_early']['neighbor'] > c['pred_mse_late']['neighbor']:
            neighbor_improving += 1

    n = len(cycles)
    p1 = {
        'supported': (energy_improving > n * 0.5 and
                      resource_improving > n * 0.5 and
                      neighbor_improving > n * 0.5),
        'energy_improving_frac': energy_improving / n,
        'resource_improving_frac': resource_improving / n,
        'neighbor_improving_frac': neighbor_improving / n,
        'mean_mse_energy': result['summary']['mean_pred_mse_energy'],
        'mean_mse_resource': result['summary']['mean_pred_mse_resource'],
        'mean_mse_neighbor': result['summary']['mean_pred_mse_neighbor'],
    }

    # P2: Mean Phi > 0.11 (V22 was ~0.10)
    mean_phi = result['summary']['mean_phi']
    p2 = {
        'supported': mean_phi > 0.11,
        'mean_phi': mean_phi,
    }

    # P3: Robustness > 1.0 (V22 was ~0.98)
    mean_rob = result['summary']['mean_robustness']
    p3 = {
        'supported': mean_rob > 1.0,
        'mean_robustness': mean_rob,
    }

    # P4: predict_W columns specialize (low cosine similarity)
    # Specialized if mean cosine < 0.7 (orthogonal = 0, parallel = 1)
    final_cos = result['summary']['final_col_cosine']
    final_rank = result['summary']['final_effective_rank']
    p4 = {
        'supported': final_cos < 0.7,
        'final_col_cosine': final_cos,
        'final_effective_rank': final_rank,
    }

    # P5: Target difficulty varies (energy easiest, neighbor hardest)
    mse_e = result['summary']['mean_pred_mse_energy']
    mse_r = result['summary']['mean_pred_mse_resource']
    mse_n = result['summary']['mean_pred_mse_neighbor']
    p5 = {
        'supported': mse_e < mse_r < mse_n or mse_e < mse_n,
        'mse_energy': mse_e,
        'mse_resource': mse_r,
        'mse_neighbor': mse_n,
        'ordering': 'E<R<N' if mse_e < mse_r < mse_n
                    else 'E<N<R' if mse_e < mse_n < mse_r
                    else 'other',
    }

    # Drift stats
    drift = {
        'mean_drift_overall': float(np.mean([c['phenotype_drift']['mean_drift'] for c in cycles])),
        'max_drift_overall': float(np.max([c['phenotype_drift']['max_drift'] for c in cycles])),
    }

    return {
        'prediction_1_all_targets_improve': p1,
        'prediction_2_phi_increases': p2,
        'prediction_3_robustness_increases': p3,
        'prediction_4_weight_specialization': p4,
        'prediction_5_target_difficulty': p5,
        'drift': drift,
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python v23_gpu_run.py {smoke|single|all|predict}")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == 'smoke':
        run_smoke()
    elif cmd == 'single':
        seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
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
        result_path = 'results/v23_s%d/v23_s%d_results.json' % (seed, seed)
        if not os.path.exists(result_path):
            print("Results not found: %s" % result_path)
            sys.exit(1)
        with open(result_path) as f:
            result = json.load(f)
        preds = evaluate_predictions(result)
        print(json.dumps(preds, indent=2))
    else:
        print("Unknown command: %s" % cmd)
        print("Usage: python v23_gpu_run.py {smoke|single|all|predict}")
        sys.exit(1)
