"""V21 GPU Run: CTM-Inspired Protocell Agency on Lambda Labs.

Internal ticks + sync matrices — tests whether giving agents internal
processing time enables imagination, counterfactual simulation, and
deliberation under survival pressure.

Usage:
    python v21_gpu_run.py smoke                   # Quick CPU test (N=32, K_max=4)
    python v21_gpu_run.py run --seed 42           # Single seed, full run
    python v21_gpu_run.py all                     # All 3 seeds sequentially
    python v21_gpu_run.py chain --seed 42         # Chain test on saved snapshots
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

from v21_substrate import (
    generate_v21_config, init_v21, make_chunk_runner,
    compute_phi_hidden, compute_phi_sync,
    extract_tick_weights_np, extract_sync_decay_np,
)
from v21_evolution import run_v21


# ---------------------------------------------------------------------------
# V21-specific measurement functions (post-hoc analysis)
# ---------------------------------------------------------------------------

def measure_tick_usage_trajectory(results_path):
    """Analyze tick usage evolution across cycles from saved results.

    Returns dict with per-cycle effective K and entropy.
    """
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

    # Does effective K covary with drought?
    drought_cycles = [c for c in cycles if c['is_drought']]
    normal_cycles = [c for c in cycles if not c['is_drought']]

    return {
        'cycles': cycles,
        'mean_effective_K_drought': float(np.mean([c['effective_K'] for c in drought_cycles])) if drought_cycles else 0.0,
        'mean_effective_K_normal': float(np.mean([c['effective_K'] for c in normal_cycles])) if normal_cycles else 0.0,
        'tick_0_collapsed_final': cycles[-1]['tick_0_collapsed'] if cycles else 1.0,
    }


def measure_sync_decay_trajectory(results_path):
    """Analyze sync decay evolution across cycles from saved results."""
    with open(results_path) as f:
        results = json.load(f)

    decays = []
    for c in results['cycles']:
        sd = c.get('sync_decay', {})
        decays.append({
            'cycle': c['cycle'],
            'mean_decay': sd.get('mean_sync_decay', 0.75),
            'std_decay': sd.get('std_sync_decay', 0.0),
        })

    return {
        'cycles': decays,
        'initial_decay': decays[0]['mean_decay'] if decays else 0.75,
        'final_decay': decays[-1]['mean_decay'] if decays else 0.75,
        'evolved_longer': decays[-1]['mean_decay'] > decays[0]['mean_decay'] if len(decays) > 1 else False,
    }


def measure_imagination_index(results_path):
    """Compute imagination index from divergence and performance data.

    I_img: correlation between intra-step divergence and subsequent
    energy gain (across cycles). If positive, "thinking more" helps.
    """
    with open(results_path) as f:
        results = json.load(f)

    divergences = []
    energy_gains = []
    for i, c in enumerate(results['cycles']):
        div = c.get('mean_divergence', 0)
        # Energy gain proxy: mean alive population (higher = better harvesting)
        alive_ratio = c.get('n_alive_end', 0) / max(c.get('n_alive_start', 1), 1)
        divergences.append(div)
        energy_gains.append(alive_ratio)

    if len(divergences) < 3:
        return {'I_img': 0.0, 'n_points': 0}

    from scipy import stats
    rho, pval = stats.spearmanr(divergences, energy_gains)

    return {
        'I_img': float(rho),
        'I_img_p': float(pval),
        'n_points': len(divergences),
        'mean_divergence': float(np.mean(divergences)),
    }


# ---------------------------------------------------------------------------
# Chain test: reuse V20 measurement functions with V21 chunk runner
# ---------------------------------------------------------------------------

def run_v21_chain_test(snapshot_path, cfg, key, output_path=None):
    """Run chain measurements (C_wm, rho_sync, SM_sal, RSA) on a V21 snapshot.

    Uses V20's measurement functions with V21's chunk runner.
    """
    from v20_experiments import (
        measure_world_model, measure_rho_sync,
        measure_self_model, measure_affect_geometry,
    )

    snap = np.load(snapshot_path, allow_pickle=True)
    H = cfg['hidden_dim']

    # Reconstruct V21 state (sync_matrices initialized to zeros)
    state = {
        'resources': jnp.zeros((cfg['N'], cfg['N'])),
        'signals': jnp.zeros((cfg['N'], cfg['N'])),
        'positions': jnp.array(snap['positions']),
        'hidden': jnp.array(snap['hidden']),
        'energy': jnp.array(snap['energy']),
        'alive': jnp.array(snap['alive']),
        'params': jnp.array(snap['params']),
        'sync_matrices': jnp.zeros((cfg['M_max'], H, H)),
        'regen_rate': jnp.array(cfg['resource_regen']),
        'step_count': jnp.array(0),
    }

    # Add random resources
    key, k1 = jax.random.split(key)
    state['resources'] = jax.random.uniform(k1, (cfg['N'], cfg['N'])) * 0.5

    chunk_runner = make_chunk_runner(cfg)

    # Warmup
    state, _ = chunk_runner(state)

    cycle = int(snap['cycle']) if 'cycle' in snap else -1
    print(f"  Running chain test on cycle {cycle} snapshot...")

    t0 = time.time()
    wm = measure_world_model(state, chunk_runner, cfg)
    print(f"    C_wm={wm['c_wm']:.4f}, dC_wm={wm['delta_c_wm']:.4f} ({time.time()-t0:.0f}s)")

    t0 = time.time()
    rs = measure_rho_sync(state, chunk_runner, cfg, key)
    print(f"    rho_sync={rs['rho_sync']:.4f} ({time.time()-t0:.0f}s)")

    t0 = time.time()
    sm = measure_self_model(state, chunk_runner, cfg)
    print(f"    SM_sal={sm['sm_sal']:.4f} ({time.time()-t0:.0f}s)")

    t0 = time.time()
    ag = measure_affect_geometry(state, chunk_runner, cfg)
    print(f"    RSA={ag['rsa']:.4f} (p={ag['rsa_p']:.4f}) ({time.time()-t0:.0f}s)")

    # V21-specific: Phi_sync
    phi_sync = float(compute_phi_sync(state['sync_matrices'], state['alive']))

    results = {
        'cycle': cycle,
        'world_model': wm,
        'rho_sync': rs,
        'self_model': sm,
        'affect_geometry': ag,
        'phi_sync': phi_sync,
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

    return results


def run_v21_chain(seed, output_base='/home/ubuntu/results'):
    """Run chain test on all V21 snapshots for a seed."""
    results_dir = f'{output_base}/v21_s{seed}'
    cfg = generate_v21_config(N=128, M_max=256, steps_per_cycle=5000, chunk_size=50)
    key = jax.random.PRNGKey(seed + 100)

    snapshot_files = sorted(glob.glob(os.path.join(results_dir, 'snapshot_c*.npz')))
    print(f"\n=== V21 Chain Test: seed {seed}, {len(snapshot_files)} snapshots ===")

    all_results = []
    for snap_path in snapshot_files:
        cycle_str = os.path.basename(snap_path).replace('snapshot_c', '').replace('.npz', '')
        out_path = os.path.join(results_dir, f'chain_c{cycle_str}.json')
        key, subkey = jax.random.split(key)
        result = run_v21_chain_test(snap_path, cfg, subkey, output_path=out_path)
        all_results.append(result)

    # Summary
    if all_results:
        c_wm_vals = [r['world_model']['c_wm'] for r in all_results]
        rho_vals = [r['rho_sync']['rho_sync'] for r in all_results]
        sm_vals = [r['self_model']['sm_sal'] for r in all_results]
        rsa_vals = [r['affect_geometry']['rsa'] for r in all_results]
        phi_sync_vals = [r['phi_sync'] for r in all_results]

        summary = {
            'seed': seed,
            'n_snapshots': len(all_results),
            'c_wm_last': c_wm_vals[-1],
            'rho_sync_max': max(rho_vals),
            'sm_sal_last': sm_vals[-1],
            'rsa_last': rsa_vals[-1],
            'phi_sync_last': phi_sync_vals[-1],
            'wall_broken': max(rho_vals) > 0.1,
            'results': all_results,
        }

        print(f"\n=== V21 Chain Summary (seed {seed}) ===")
        print(f"  C_wm:      {c_wm_vals[-1]:.4f} (final)")
        print(f"  rho_sync:  {max(rho_vals):.4f} (max)")
        print(f"  SM_sal:    {sm_vals[-1]:.4f} (final)")
        print(f"  RSA:       {rsa_vals[-1]:.4f} (final)")
        print(f"  Phi_sync:  {phi_sync_vals[-1]:.4f} (final)")

        out_path = os.path.join(results_dir, f'v21_s{seed}_chain_summary.json')
        with open(out_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return summary

    return {}


# ---------------------------------------------------------------------------
# Post-run analysis: evaluate pre-registered predictions
# ---------------------------------------------------------------------------

def evaluate_predictions(results_path):
    """Evaluate V21 pre-registered predictions from saved results."""
    print("\n=== V21 Pre-registered Prediction Evaluation ===\n")

    tick_traj = measure_tick_usage_trajectory(results_path)
    sync_traj = measure_sync_decay_trajectory(results_path)
    I_img = measure_imagination_index(results_path)

    # Prediction 1: Effective K covaries with drought
    eff_K_drought = tick_traj['mean_effective_K_drought']
    eff_K_normal = tick_traj['mean_effective_K_normal']
    pred1 = eff_K_drought > eff_K_normal
    print(f"P1: Effective K higher under drought?")
    print(f"    K_drought={eff_K_drought:.2f}, K_normal={eff_K_normal:.2f}")
    print(f"    -> {'SUPPORTED' if pred1 else 'NOT SUPPORTED'}\n")

    # Prediction 2: Divergence correlates with survival (I_img > 0)
    pred2 = I_img['I_img'] > 0 and I_img.get('I_img_p', 1.0) < 0.1
    print(f"P2: Intra-step divergence correlates with action quality?")
    print(f"    I_img={I_img['I_img']:.3f} (p={I_img.get('I_img_p', 1.0):.4f})")
    print(f"    -> {'SUPPORTED' if pred2 else 'NOT SUPPORTED'}\n")

    # Prediction 3: tick_weights don't collapse to tick-0
    collapsed = tick_traj['tick_0_collapsed_final']
    pred3 = collapsed < 0.67  # Less than 2/3 of agents collapsed
    print(f"P3: tick_weights NOT collapsed to tick-0?")
    print(f"    Fraction collapsed: {collapsed:.0%}")
    print(f"    -> {'SUPPORTED' if pred3 else 'NOT SUPPORTED'}\n")

    # Sync decay evolution
    print(f"Sync decay: {sync_traj['initial_decay']:.4f} -> {sync_traj['final_decay']:.4f}")
    print(f"  Evolved {'longer' if sync_traj['evolved_longer'] else 'shorter'} memory\n")

    return {
        'prediction_1': {'supported': pred1, 'K_drought': eff_K_drought, 'K_normal': eff_K_normal},
        'prediction_2': {'supported': pred2, **I_img},
        'prediction_3': {'supported': pred3, 'collapsed_fraction': collapsed},
        'sync_decay': sync_traj,
    }


# ---------------------------------------------------------------------------
# CLI runners
# ---------------------------------------------------------------------------

def run_smoke():
    """Quick smoke test: tiny grid, few cycles, K_max=4."""
    print("V21 SMOKE TEST (N=32, M_max=32, K_max=4, 3 cycles)")
    print("=" * 70)
    cfg = generate_v21_config(
        N=32, M_max=32,
        K_max=4,
        steps_per_cycle=200,
        n_cycles=3,
        chunk_size=50,
        activate_offspring=True,
        drought_every=0,  # No drought in smoke test
    )
    print(f"  n_params per agent: {cfg['n_params']}")
    print(f"  obs_flat dim:       {cfg['obs_flat']}")
    print(f"  K_max:              {cfg['K_max']}")

    result = run_v21(seed=42, cfg=cfg, output_dir='/tmp/v21_smoke')

    print(f"\nSmoke test complete!")
    print(f"  Mean robustness:   {result['summary']['mean_robustness']:.3f}")
    print(f"  Max robustness:    {result['summary']['max_robustness']:.3f}")
    print(f"  Final effective K: {result['summary']['final_effective_K']:.2f}")
    return result


def run_single(seed, output_base='/home/ubuntu/results',
               n_cycles=30, steps_per_cycle=5000):
    """Full single-seed run."""
    output_dir = f'{output_base}/v21_s{seed}'
    cfg = generate_v21_config(
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
    print(f"V21 SEED {seed}: {n_cycles} cycles x {steps_per_cycle} steps, K_max=8")
    print(f"  n_params per agent: {cfg['n_params']}")
    print(f"  Grid: {cfg['N']}x{cfg['N']}, max pop: {cfg['M_max']}")
    print(f"  Output: {output_dir}")
    print(f"  JAX devices: {jax.devices()}")
    print(f"{'=' * 70}\n")

    result = run_v21(seed=seed, cfg=cfg, output_dir=output_dir)

    # Run prediction evaluation
    results_path = os.path.join(output_dir, f'v21_s{seed}_results.json')
    preds = evaluate_predictions(results_path)
    pred_path = os.path.join(output_dir, f'v21_s{seed}_predictions.json')
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
              f"eff_K={s['final_effective_K']:.2f}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='V21 CTM-Inspired Protocell Agency')
    parser.add_argument('command', nargs='?', default='run',
                        choices=['smoke', 'run', 'all', 'chain', 'predict'])
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
    elif args.command == 'chain':
        run_v21_chain(args.seed, output_base=args.output)
    elif args.command == 'predict':
        results_path = f'{args.output}/v21_s{args.seed}/v21_s{args.seed}_results.json'
        evaluate_predictions(results_path)
    else:
        run_single(seed=args.seed, output_base=args.output,
                   n_cycles=args.cycles, steps_per_cycle=args.steps)
