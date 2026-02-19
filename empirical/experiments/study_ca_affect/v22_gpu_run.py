"""V22 GPU Run: Intrinsic Predictive Gradient on Lambda Labs.

Within-lifetime SGD from energy delta prediction — tests whether dense
gradient signal through internal ticks creates adaptive deliberation
that evolution alone (V21) could not.

Usage:
    python v22_gpu_run.py smoke                   # Quick CPU test (N=32, K_max=4)
    python v22_gpu_run.py run --seed 42           # Single seed, full run
    python v22_gpu_run.py all                     # All 3 seeds sequentially
    python v22_gpu_run.py predict --seed 42       # Evaluate predictions from saved results
"""

import sys
import os
import argparse
import json
import time
import glob
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v22_substrate import (
    generate_v22_config, init_v22, make_v22_chunk_runner,
    compute_phi_hidden, compute_phi_sync,
    extract_tick_weights_np, extract_sync_decay_np, extract_lr_np,
)
from v22_evolution import run_v22


# ---------------------------------------------------------------------------
# V22-specific analysis functions
# ---------------------------------------------------------------------------

def measure_pred_mse_trajectory(results_path):
    """Analyze prediction MSE evolution across cycles.

    Key question: does MSE decrease within each lifetime (early→late)?
    """
    with open(results_path) as f:
        results = json.load(f)

    cycles = []
    for c in results['cycles']:
        cycles.append({
            'cycle': c['cycle'],
            'mean_pred_mse': c.get('mean_pred_mse', 0),
            'pred_mse_early': c.get('pred_mse_early', 0),
            'pred_mse_late': c.get('pred_mse_late', 0),
            'improvement': c.get('pred_mse_early', 0) - c.get('pred_mse_late', 0),
            'is_drought': c.get('mortality', 0) > 0.5,
        })

    improvements = [c['improvement'] for c in cycles]
    learning_cycles = sum(1 for imp in improvements if imp > 0)

    return {
        'cycles': cycles,
        'mean_improvement': float(np.mean(improvements)),
        'learning_cycles_fraction': learning_cycles / max(len(cycles), 1),
        'first_cycle_mse': cycles[0]['mean_pred_mse'] if cycles else 0,
        'last_cycle_mse': cycles[-1]['mean_pred_mse'] if cycles else 0,
        'mse_trend': 'decreasing' if cycles[-1]['mean_pred_mse'] < cycles[0]['mean_pred_mse'] else 'increasing',
    }


def measure_lr_trajectory(results_path):
    """Analyze learning rate evolution across cycles.

    Key question: does evolution suppress lr (→0) or maintain it?
    """
    with open(results_path) as f:
        results = json.load(f)

    lrs = []
    for c in results['cycles']:
        lr = c.get('lr_stats', {})
        lrs.append({
            'cycle': c['cycle'],
            'mean_lr': lr.get('mean_lr', 0),
            'std_lr': lr.get('std_lr', 0),
            'min_lr': lr.get('min_lr', 0),
            'max_lr': lr.get('max_lr', 0),
        })

    return {
        'cycles': lrs,
        'initial_lr': lrs[0]['mean_lr'] if lrs else 0,
        'final_lr': lrs[-1]['mean_lr'] if lrs else 0,
        'lr_suppressed': lrs[-1]['mean_lr'] < 1e-4 if lrs else True,
        'lr_trend': 'decreasing' if lrs[-1]['mean_lr'] < lrs[0]['mean_lr'] else 'increasing',
    }


def measure_drift_trajectory(results_path):
    """Analyze phenotype drift across cycles."""
    with open(results_path) as f:
        results = json.load(f)

    drifts = []
    for c in results['cycles']:
        d = c.get('phenotype_drift', {})
        drifts.append({
            'cycle': c['cycle'],
            'mean_drift': d.get('mean_drift', 0),
            'max_drift': d.get('max_drift', 0),
        })

    return {
        'cycles': drifts,
        'mean_drift_overall': float(np.mean([d['mean_drift'] for d in drifts])) if drifts else 0,
        'max_drift_overall': max(d['max_drift'] for d in drifts) if drifts else 0,
    }


def measure_tick_usage_trajectory(results_path):
    """Analyze tick usage evolution (same as V21 for comparison)."""
    with open(results_path) as f:
        results = json.load(f)

    cycles = []
    for c in results['cycles']:
        tu = c.get('tick_usage', {})
        cycles.append({
            'cycle': c['cycle'],
            'effective_K': tu.get('mean_effective_K', 1.0),
            'entropy': tu.get('mean_entropy', 0.0),
            'tick_0_weight': tu.get('tick_0_weight_mean', 1.0),
            'tick_0_collapsed': tu.get('tick_0_collapsed', 1.0),
            'is_drought': c.get('mortality', 0) > 0.5,
        })

    drought_cycles = [c for c in cycles if c['is_drought']]
    normal_cycles = [c for c in cycles if not c['is_drought']]

    return {
        'cycles': cycles,
        'mean_effective_K_drought': float(np.mean([c['effective_K'] for c in drought_cycles])) if drought_cycles else 0.0,
        'mean_effective_K_normal': float(np.mean([c['effective_K'] for c in normal_cycles])) if normal_cycles else 0.0,
        'tick_0_collapsed_final': cycles[-1]['tick_0_collapsed'] if cycles else 1.0,
        'effective_K_trend': 'increasing' if cycles[-1]['effective_K'] > cycles[0]['effective_K'] else 'stable_or_decreasing',
    }


# ---------------------------------------------------------------------------
# Evaluate pre-registered predictions
# ---------------------------------------------------------------------------

def evaluate_predictions(results_path):
    """Evaluate V22 pre-registered predictions from saved results."""
    print("\n=== V22 Pre-registered Prediction Evaluation ===\n")

    pred_mse = measure_pred_mse_trajectory(results_path)
    lr_traj = measure_lr_trajectory(results_path)
    drift_traj = measure_drift_trajectory(results_path)
    tick_traj = measure_tick_usage_trajectory(results_path)

    with open(results_path) as f:
        results = json.load(f)

    # P1: Prediction MSE decreases within lifetime
    pred1 = pred_mse['mean_improvement'] > 0 and pred_mse['learning_cycles_fraction'] > 0.5
    print(f"P1: Prediction MSE decreases within lifetime?")
    print(f"    Mean improvement (early-late): {pred_mse['mean_improvement']:.6f}")
    print(f"    Learning cycles: {pred_mse['learning_cycles_fraction']:.0%}")
    print(f"    -> {'SUPPORTED' if pred1 else 'NOT SUPPORTED'}\n")

    # P2: Agents do NOT evolve lr → 0
    pred2 = not lr_traj['lr_suppressed']
    print(f"P2: Learning rate NOT suppressed?")
    print(f"    Initial LR: {lr_traj['initial_lr']:.6f}")
    print(f"    Final LR:   {lr_traj['final_lr']:.6f}")
    print(f"    Trend:      {lr_traj['lr_trend']}")
    print(f"    -> {'SUPPORTED' if pred2 else 'NOT SUPPORTED'}\n")

    # P3: Tick-0 doesn't collapse (same as V21 P3)
    collapsed = tick_traj['tick_0_collapsed_final']
    pred3 = collapsed < 0.67
    print(f"P3: tick_weights NOT collapsed to tick-0?")
    print(f"    Fraction collapsed: {collapsed:.0%}")
    print(f"    -> {'SUPPORTED' if pred3 else 'NOT SUPPORTED'}\n")

    # P4: Robustness better than V21 baseline (~1.0)
    mean_rob = results['summary']['mean_robustness']
    pred4 = mean_rob > 1.0
    print(f"P4: Better drought survival (robustness > 1.0)?")
    print(f"    Mean robustness: {mean_rob:.3f}")
    print(f"    -> {'SUPPORTED' if pred4 else 'NOT SUPPORTED'}\n")

    # P5: Effective K increases (structured tick usage)
    pred5 = tick_traj['effective_K_trend'] == 'increasing'
    print(f"P5: Effective K increases over evolution?")
    if tick_traj['cycles']:
        print(f"    K_initial: {tick_traj['cycles'][0]['effective_K']:.2f}")
        print(f"    K_final:   {tick_traj['cycles'][-1]['effective_K']:.2f}")
    print(f"    Trend: {tick_traj['effective_K_trend']}")
    print(f"    -> {'SUPPORTED' if pred5 else 'NOT SUPPORTED'}\n")

    # Summary
    print(f"Phenotype drift: mean={drift_traj['mean_drift_overall']:.2f}, "
          f"max={drift_traj['max_drift_overall']:.2f}")
    print(f"Pred MSE trend: {pred_mse['first_cycle_mse']:.4f} → {pred_mse['last_cycle_mse']:.4f} "
          f"({pred_mse['mse_trend']})")

    return {
        'prediction_1': {'supported': pred1, **{k: v for k, v in pred_mse.items() if k != 'cycles'}},
        'prediction_2': {'supported': pred2, **{k: v for k, v in lr_traj.items() if k != 'cycles'}},
        'prediction_3': {'supported': pred3, 'collapsed_fraction': collapsed},
        'prediction_4': {'supported': pred4, 'mean_robustness': mean_rob},
        'prediction_5': {'supported': pred5, **{k: v for k, v in tick_traj.items() if k != 'cycles'}},
        'drift': {k: v for k, v in drift_traj.items() if k != 'cycles'},
    }


# ---------------------------------------------------------------------------
# Animation frame capture
# ---------------------------------------------------------------------------

def save_animation_frames(state, chunk_runner, cfg, output_dir, n_frames=50):
    """Save per-step snapshots for animation (user requested visualization).

    Saves: agent positions, energy, resources, alive mask, predictions
    as lightweight npz files. One per frame.
    """
    frames_dir = os.path.join(output_dir, 'animation_frames')
    os.makedirs(frames_dir, exist_ok=True)

    # Run a short rollout collecting frames
    for frame_i in range(n_frames):
        state, metrics = chunk_runner(state)

        if frame_i % 5 == 0:  # Save every 5th chunk = every 250 steps
            np.savez_compressed(
                os.path.join(frames_dir, f'frame_{frame_i:04d}.npz'),
                positions=np.array(state['positions']),
                energy=np.array(state['energy']),
                alive=np.array(state['alive']),
                resources=np.array(state['resources']),
                signals=np.array(state['signals']),
            )

    print(f"  Saved {n_frames // 5} animation frames to {frames_dir}/")
    return state


# ---------------------------------------------------------------------------
# CLI runners
# ---------------------------------------------------------------------------

def run_smoke():
    """Quick smoke test: tiny grid, few cycles, K_max=4."""
    print("V22 SMOKE TEST (N=32, M_max=32, K_max=4, 3 cycles)")
    print("=" * 70)
    cfg = generate_v22_config(
        N=32, M_max=32,
        K_max=4,
        steps_per_cycle=200,
        n_cycles=3,
        chunk_size=50,
        activate_offspring=True,
        drought_every=0,
    )
    print(f"  n_params per agent: {cfg['n_params']}")
    print(f"  obs_flat dim:       {cfg['obs_flat']}")
    print(f"  K_max:              {cfg['K_max']}")
    print(f"  lamarckian:         {cfg['lamarckian']}")

    result = run_v22(seed=42, cfg=cfg, output_dir='/tmp/v22_smoke')

    print(f"\nSmoke test complete!")
    print(f"  Mean robustness:   {result['summary']['mean_robustness']:.3f}")
    print(f"  Mean pred MSE:     {result['summary']['mean_pred_mse']:.4f}")
    print(f"  MSE improvement:   {result['summary']['pred_mse_improvement']:.4f}")
    print(f"  Final LR:          {result['summary']['final_lr']:.6f}")
    print(f"  LR suppressed:     {result['summary']['lr_suppressed']}")
    return result


def run_single(seed, output_base='/home/ubuntu/results',
               n_cycles=30, steps_per_cycle=5000):
    """Full single-seed run."""
    output_dir = f'{output_base}/v22_s{seed}'
    cfg = generate_v22_config(
        N=128,
        M_max=256,
        K_max=8,
        n_cycles=n_cycles,
        steps_per_cycle=steps_per_cycle,
        chunk_size=50,
        activate_offspring=True,
        drought_every=5,
    )

    print(f"\n{'=' * 70}")
    print(f"V22 SEED {seed}: {n_cycles} cycles x {steps_per_cycle} steps, K_max=8")
    print(f"  n_params per agent: {cfg['n_params']}")
    print(f"  Grid: {cfg['N']}x{cfg['N']}, max pop: {cfg['M_max']}")
    print(f"  Lamarckian: {cfg['lamarckian']}")
    print(f"  Output: {output_dir}")
    print(f"  JAX devices: {jax.devices()}")
    print(f"{'=' * 70}\n")

    result = run_v22(seed=seed, cfg=cfg, output_dir=output_dir)

    # Run prediction evaluation
    results_path = os.path.join(output_dir, f'v22_s{seed}_results.json')
    preds = evaluate_predictions(results_path)
    pred_path = os.path.join(output_dir, f'v22_s{seed}_predictions.json')
    with open(pred_path, 'w') as f:
        json.dump(preds, f, indent=2)

    return result


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
              f"lr={s['final_lr']:.6f}, "
              f"lr_suppressed={s['lr_suppressed']}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='V22 Intrinsic Predictive Gradient')
    parser.add_argument('command', nargs='?', default='run',
                        choices=['smoke', 'run', 'all', 'predict'])
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
    elif args.command == 'predict':
        results_path = f'{args.output}/v22_s{args.seed}/v22_s{args.seed}_results.json'
        evaluate_predictions(results_path)
    else:
        run_single(seed=args.seed, output_base=args.output,
                   n_cycles=args.cycles, steps_per_cycle=args.steps)
