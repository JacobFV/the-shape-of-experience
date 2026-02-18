"""V15 Experiment Runner — Re-run measurement experiments on V15 substrate.

V13 experiments hit the "sensory-motor coupling wall": patterns were always
internally driven (ρ_sync ≈ 0, ρ_self ≈ 0). V15 adds chemotaxis (motor
channels) and temporal memory, which creates explicit sensory-motor coupling
and temporal integration.

Priority experiments for V15 re-run:
  - Exp 2: World Model (memory channels → stronger C_wm prediction)
  - Exp 3: Representation (memory channels → different structure)
  - Exp 5: Counterfactual Detachment (motor channels → reactive start)
  - Exp 6: Self-Model (motor channels → ρ_self > 0)
  - Exp 7: Affect Geometry (already positive on V13, should strengthen)
  - Exp 8: ι (participatory default confirmed on V13, check stability)

Non-priority (likely same result, run last):
  - Exp 4: Communication (no new signaling mechanism)
  - Exp 9: Normativity (still no directed action choices)
  - Exp 10: Social Φ (still no specialization)

Architecture: Each experiment's core measurement code (feature extraction,
prediction gap, etc.) is in the v13_* modules and is substrate-agnostic.
This module provides V15 substrate setup + wrappers that feed V15's
run_chunk_fn into the existing measurement pipelines.

Usage:
    python v15_experiments.py smoke          # Quick test (CPU, ~2min)
    python v15_experiments.py run --exp 2    # Single experiment
    python v15_experiments.py all            # All experiments, all seeds
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS_BASE = Path(__file__).parent / 'results'

# V15 snapshot directories
V15_SEED_DIRS = {
    42: RESULTS_BASE / 'v15_s42',
    123: RESULTS_BASE / 'v15_s123',
    7: RESULTS_BASE / 'v15_s7',
}


def find_v15_snapshots(seed: int) -> list:
    """Find all V15 cycle snapshot files for a seed."""
    seed_dir = V15_SEED_DIRS.get(seed)
    if seed_dir is None or not seed_dir.exists():
        print(f"No V15 results directory for seed={seed}")
        return []
    snap_dir = seed_dir / 'snapshots'
    if not snap_dir.exists():
        print(f"No snapshots directory: {snap_dir}")
        return []
    files = sorted(snap_dir.glob('cycle_*.npz'))
    return files


def get_cycle_number(path: Path) -> int:
    """Extract cycle number from filename like cycle_010.npz."""
    return int(path.stem.split('_')[1])


# ============================================================================
# V15 Substrate Setup
# ============================================================================

def setup_v15_substrate(snapshot_path: str, seed: int, config_overrides: dict = None):
    """Set up V15 substrate from a snapshot.

    Returns:
        (grid_np, resource_np, run_chunk_fn, rng, config, metadata)
    where run_chunk_fn(grid, resource, n_steps, rng) -> (grid, resource, rng)
    """
    import jax.numpy as jnp
    from jax import random
    from v15_substrate import (
        generate_v15_config, init_v15, run_v15_chunk,
        generate_resource_patches, compute_patch_regen_mask,
    )

    # Load snapshot
    snap = np.load(snapshot_path)
    grid_np = snap['grid']
    resource_np = snap['resource']
    C, N = grid_np.shape[0], grid_np.shape[1]

    # Generate config (must match GPU run calibration)
    config = generate_v15_config(C=C, N=N, seed=seed)
    config['maintenance_rate'] = 0.003
    config['resource_consume'] = 0.003
    config['resource_regen'] = 0.01
    if config_overrides:
        config.update(config_overrides)

    # Initialize substrate components
    _, _, h_embed, kernel_ffts, coupling, coupling_row_sums, box_fft = \
        init_v15(config, seed=seed)

    # Resource patches (static mask for measurement — no oscillation)
    patches = generate_resource_patches(N, config['n_resource_patches'], seed=seed)
    regen_mask = compute_patch_regen_mask(N, patches, step=0,
                                          shift_period=config['patch_shift_period'])

    rng = random.PRNGKey(seed + 9999)

    # Wrapper that matches recording episode interface:
    # run_chunk_fn(grid, resource, n_steps, rng) -> (grid, resource, rng)
    def run_chunk_fn(grid, resource, n_steps, rng, **kwargs):
        return run_v15_chunk(
            grid, resource, h_embed, kernel_ffts, config,
            coupling, coupling_row_sums, rng, n_steps=n_steps,
            box_fft=box_fft, regen_mask=regen_mask)

    metadata = {
        'substrate': 'v15',
        'seed': seed,
        'C': C,
        'N': N,
        'snapshot_path': str(snapshot_path),
    }

    return grid_np, resource_np, run_chunk_fn, rng, config, metadata


# ============================================================================
# Experiment 2: World Model
# ============================================================================

def run_world_model(snapshot_path: str, seed: int,
                    n_recording_steps: int = 50,
                    substrate_steps_per_record: int = 10,
                    tau_values: tuple = (1, 2, 5, 10, 20),
                    max_patterns: int = 20):
    """Run Experiment 2 on a V15 snapshot."""
    from v13_world_model import (
        run_recording_episode, compute_prediction_gap,
        WorldModelResult, results_to_dict,
    )

    grid_np, resource_np, run_chunk_fn, rng, config, meta = \
        setup_v15_substrate(snapshot_path, seed)

    features = run_recording_episode(
        grid=grid_np,
        resource=resource_np,
        run_chunk_fn=run_chunk_fn,
        chunk_kwargs={'rng': rng},
        n_recording_steps=n_recording_steps,
        substrate_steps_per_record=substrate_steps_per_record,
        tau_values=tau_values,
    )

    results = []
    sorted_pids = sorted(features.keys(),
                         key=lambda pid: len(features[pid]['features']),
                         reverse=True)

    for pid in sorted_pids[:max_patterns]:
        feat_list = features[pid]['features']
        wm_result = compute_prediction_gap(feat_list, tau_values=tau_values)
        if wm_result is not None:
            wm_result.pattern_id = pid
            results.append(wm_result)

    meta['experiment'] = 'world_model'
    meta['n_recording_steps'] = n_recording_steps
    meta['tau_values'] = list(tau_values)
    return results, meta


# ============================================================================
# Experiment 3: Representation
# ============================================================================

def run_representation(snapshot_path: str, seed: int,
                       n_recording_steps: int = 50,
                       substrate_steps_per_record: int = 10):
    """Run Experiment 3 on a V15 snapshot."""
    from v13_representation import analyze_representation, results_to_dict
    from v13_world_model import run_recording_episode

    grid_np, resource_np, run_chunk_fn, rng, config, meta = \
        setup_v15_substrate(snapshot_path, seed)

    features = run_recording_episode(
        grid=grid_np,
        resource=resource_np,
        run_chunk_fn=run_chunk_fn,
        chunk_kwargs={'rng': rng},
        n_recording_steps=n_recording_steps,
        substrate_steps_per_record=substrate_steps_per_record,
        tau_values=(1,),
    )

    results = []
    sorted_pids = sorted(features.keys(),
                         key=lambda pid: len(features[pid]['features']),
                         reverse=True)

    for pid in sorted_pids[:20]:
        feat_list = features[pid]['features']
        if len(feat_list) < 10:
            continue
        repr_result = analyze_representation(feat_list)
        if repr_result is not None:
            repr_result.pattern_id = pid
            results.append(repr_result)

    meta['experiment'] = 'representation'
    return results, meta


# ============================================================================
# Experiment 5: Counterfactual Detachment
# ============================================================================

def run_counterfactual(snapshot_path: str, seed: int,
                       n_recording_steps: int = 50,
                       substrate_steps_per_record: int = 10):
    """Run Experiment 5 on a V15 snapshot."""
    from v13_counterfactual import analyze_counterfactual, results_to_dict
    from v13_world_model import run_recording_episode

    grid_np, resource_np, run_chunk_fn, rng, config, meta = \
        setup_v15_substrate(snapshot_path, seed)

    features = run_recording_episode(
        grid=grid_np,
        resource=resource_np,
        run_chunk_fn=run_chunk_fn,
        chunk_kwargs={'rng': rng},
        n_recording_steps=n_recording_steps,
        substrate_steps_per_record=substrate_steps_per_record,
        tau_values=(1, 5, 10),
    )

    results = []
    sorted_pids = sorted(features.keys(),
                         key=lambda pid: len(features[pid]['features']),
                         reverse=True)

    for pid in sorted_pids[:20]:
        feat_list = features[pid]['features']
        if len(feat_list) < 15:
            continue
        cf_result = analyze_counterfactual(feat_list)
        if cf_result is not None:
            cf_result.pattern_id = pid
            results.append(cf_result)

    meta['experiment'] = 'counterfactual'
    return results, meta


# ============================================================================
# Experiment 6: Self-Model
# ============================================================================

def run_self_model(snapshot_path: str, seed: int,
                   n_recording_steps: int = 50,
                   substrate_steps_per_record: int = 10):
    """Run Experiment 6 on a V15 snapshot."""
    from v13_self_model import analyze_self_model, results_to_dict
    from v13_world_model import run_recording_episode

    grid_np, resource_np, run_chunk_fn, rng, config, meta = \
        setup_v15_substrate(snapshot_path, seed)

    features = run_recording_episode(
        grid=grid_np,
        resource=resource_np,
        run_chunk_fn=run_chunk_fn,
        chunk_kwargs={'rng': rng},
        n_recording_steps=n_recording_steps,
        substrate_steps_per_record=substrate_steps_per_record,
        tau_values=(1, 2, 5, 10),
    )

    results = []
    sorted_pids = sorted(features.keys(),
                         key=lambda pid: len(features[pid]['features']),
                         reverse=True)

    for pid in sorted_pids[:20]:
        feat_list = features[pid]['features']
        if len(feat_list) < 10:
            continue
        sm_result = analyze_self_model(feat_list)
        if sm_result is not None:
            sm_result.pattern_id = pid
            results.append(sm_result)

    meta['experiment'] = 'self_model'
    return results, meta


# ============================================================================
# Experiment 7: Affect Geometry
# ============================================================================

def run_affect_geometry(snapshot_path: str, seed: int,
                        n_recording_steps: int = 50,
                        substrate_steps_per_record: int = 10):
    """Run Experiment 7 on a V15 snapshot.

    Affect geometry needs its own forward simulation because it
    collects both grids AND resources at each step, and tracks
    pattern centers and sizes for Space C extraction.
    """
    import jax.numpy as jnp
    from v13_affect_geometry import (
        extract_space_A, extract_space_C, compute_rsa,
        AffectGeometryResult, result_to_dict,
    )
    from v13_world_model import (
        detect_patterns_for_wm, extract_internal_state,
        extract_boundary_obs,
    )

    grid_np, resource_np, run_chunk_fn, rng, config, meta = \
        setup_v15_substrate(snapshot_path, seed)
    N = grid_np.shape[1]

    # Run forward collecting grids AND resources
    grids = [grid_np.copy()]
    resources = [resource_np.copy()]
    g = jnp.array(grid_np)
    r = jnp.array(resource_np)

    for _ in range(n_recording_steps):
        g, r, rng = run_chunk_fn(g, r, n_steps=substrate_steps_per_record, rng=rng)
        grids.append(np.array(g))
        resources.append(np.array(r))

    # Track patterns
    initial_pats = detect_patterns_for_wm(grids[0], threshold=0.15)
    if not initial_pats:
        meta['experiment'] = 'affect_geometry'
        return [], meta

    all_features = {}
    for pid, p in enumerate(initial_pats[:20]):
        all_features[pid] = {
            'center_history': [p['center'].copy()],
            'size_history': [p['size']],
            'features': [],
        }

    for t in range(n_recording_steps):
        grid_t = grids[t]
        pats_t = detect_patterns_for_wm(grid_t, threshold=0.15)
        if not pats_t:
            continue

        tracked_ids = list(all_features.keys())
        matched_tracked = set()
        matched_detected = set()
        costs = []
        for j, p in enumerate(pats_t):
            for pid in tracked_ids:
                last_center = all_features[pid]['center_history'][-1]
                dr = abs(p['center'][0] - last_center[0])
                dc = abs(p['center'][1] - last_center[1])
                dr = min(dr, N - dr)
                dc = min(dc, N - dc)
                dist = np.sqrt(dr ** 2 + dc ** 2)
                if dist < 40.0:
                    costs.append((dist, pid, j))
        costs.sort()
        for dist, pid, j in costs:
            if pid in matched_tracked or j in matched_detected:
                continue
            matched_tracked.add(pid)
            matched_detected.add(j)
            p = pats_t[j]
            all_features[pid]['center_history'].append(p['center'].copy())
            all_features[pid]['size_history'].append(p['size'])
            s_B = extract_internal_state(grid_t, p['cells'])
            s_dB = extract_boundary_obs(grid_t, p['cells'], N)
            all_features[pid]['features'].append({
                's_B': s_B, 's_dB': s_dB,
            })

    # Compute Space A and Space C
    dim_A = ['valence', 'arousal', 'integration', 'd_eff', 'cf_weight', 'sm_sal']
    dim_C = ['approach_avoid', 'activity', 'growth', 'stability']
    A_list, C_list, per_pattern = [], [], []

    sorted_pids = sorted(all_features.keys(),
                         key=lambda pid: len(all_features[pid]['features']),
                         reverse=True)

    for pid in sorted_pids[:20]:
        data = all_features[pid]
        feat_list = data['features']
        centers = data['center_history']
        sizes = data['size_history']
        a = extract_space_A(feat_list, resources, centers, N)
        c = extract_space_C(feat_list, centers, sizes, resources, N)
        if a is not None and c is not None:
            A_list.append(a)
            C_list.append(c)
            per_pattern.append({
                'pattern_id': pid,
                'space_A': {dim_A[i]: float(a[i]) for i in range(len(dim_A))},
                'space_C': {dim_C[i]: float(c[i]) for i in range(len(dim_C))},
                'n_timesteps': len(feat_list),
            })

    if len(A_list) < 4:
        meta['experiment'] = 'affect_geometry'
        return [], meta

    space_A = np.array(A_list)
    space_C = np.array(C_list)
    rsa_rho, rsa_p = compute_rsa(space_A, space_C)

    result = AffectGeometryResult(
        n_patterns=len(A_list),
        rsa_rho=rsa_rho,
        rsa_p=rsa_p,
        space_A=space_A,
        space_C=space_C,
        dim_labels_A=dim_A,
        dim_labels_C=dim_C,
        per_pattern=per_pattern,
    )

    meta['experiment'] = 'affect_geometry'
    return [result], meta


# ============================================================================
# Unified Runner
# ============================================================================

EXPERIMENT_MAP = {
    2: ('world_model', run_world_model),
    3: ('representation', run_representation),
    5: ('counterfactual', run_counterfactual),
    6: ('self_model', run_self_model),
    7: ('affect_geometry', run_affect_geometry),
}


def run_experiment_on_seed(exp_num: int, seed: int,
                           n_recording_steps: int = 50,
                           output_dir: Optional[Path] = None):
    """Run a single experiment on all snapshots for a seed."""
    exp_name, exp_fn = EXPERIMENT_MAP[exp_num]
    snapshots = find_v15_snapshots(seed)
    if not snapshots:
        return []

    if output_dir is None:
        output_dir = RESULTS_BASE / f'v15_{exp_name}_s{seed}'
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for snap_path in snapshots:
        cycle = get_cycle_number(snap_path)
        print(f"  Cycle {cycle:03d}...", end=" ", flush=True)
        t0 = time.time()

        try:
            results, meta = exp_fn(
                str(snap_path), seed,
                n_recording_steps=n_recording_steps,
            )
            elapsed = time.time() - t0

            # Summarize
            if exp_num == 2:  # World model
                from v13_world_model import results_to_dict
                output = results_to_dict(results, meta)
                c_wm = output['summary']['mean_C_wm']
                h_wm = output['summary']['mean_H_wm']
                n_wm = output['summary']['n_with_world_model']
                n_total = len(results)
                print(f"C_wm={c_wm:.5f} H_wm={h_wm:.1f} "
                      f"WM={n_wm}/{n_total} [{elapsed:.0f}s]")
            elif exp_num == 5:  # Counterfactual
                if results:
                    rho_syncs = [r.mean_rho_sync for r in results
                                 if hasattr(r, 'mean_rho_sync')]
                    mean_rho = np.mean(rho_syncs) if rho_syncs else 0
                    det_fracs = [r.detachment_fraction for r in results
                                 if hasattr(r, 'detachment_fraction')]
                    mean_det = np.mean(det_fracs) if det_fracs else 0
                    print(f"n={len(results)} ρ_sync={mean_rho:.4f} "
                          f"det_frac={mean_det:.2f} [{elapsed:.0f}s]")
                else:
                    print(f"n=0 [{elapsed:.0f}s]")
            elif exp_num == 6:  # Self-model
                if results:
                    rho_selfs = [r.rho_self for r in results
                                 if hasattr(r, 'rho_self')]
                    mean_rho = np.mean(rho_selfs) if rho_selfs else 0
                    sm_sals = [r.SM_sal for r in results
                               if hasattr(r, 'SM_sal')]
                    mean_sal = np.mean(sm_sals) if sm_sals else 0
                    print(f"n={len(results)} ρ_self={mean_rho:.4f} "
                          f"SM_sal={mean_sal:.4f} [{elapsed:.0f}s]")
                else:
                    print(f"n=0 [{elapsed:.0f}s]")
            elif exp_num == 7:  # Affect geometry
                if results and hasattr(results[0], 'rsa_rho'):
                    r = results[0]
                    print(f"n_pat={r.n_patterns} RSA_ρ={r.rsa_rho:.4f} "
                          f"p={r.rsa_p:.4f} [{elapsed:.0f}s]")
                else:
                    print(f"n=0 [{elapsed:.0f}s]")
            else:
                print(f"n={len(results)} [{elapsed:.0f}s]")

            # Save per-cycle results
            out_path = output_dir / f'cycle_{cycle:03d}.json'
            if exp_num == 2:
                with open(out_path, 'w') as f:
                    json.dump(output, f, indent=2, default=str)
            else:
                # Generic serialization
                out_data = {
                    'metadata': meta,
                    'n_results': len(results),
                    'cycle': cycle,
                }
                if results and hasattr(results[0], '__dict__'):
                    out_data['results'] = []
                    for r in results:
                        d = {}
                        for k, v in r.__dict__.items():
                            if isinstance(v, np.ndarray):
                                d[k] = v.tolist()
                            elif isinstance(v, (np.floating, np.integer)):
                                d[k] = float(v)
                            elif isinstance(v, dict):
                                d[k] = {str(kk): float(vv) if isinstance(vv, (float, np.floating)) else vv
                                         for kk, vv in v.items()}
                            else:
                                d[k] = v
                        out_data['results'].append(d)
                with open(out_path, 'w') as f:
                    json.dump(out_data, f, indent=2, default=str)

            all_results.append({
                'cycle': cycle,
                'n_results': len(results),
                'elapsed': elapsed,
            })

        except Exception as e:
            elapsed = time.time() - t0
            print(f"ERROR: {e} [{elapsed:.0f}s]")
            import traceback
            traceback.print_exc()

    return all_results


# ============================================================================
# CLI
# ============================================================================

def cmd_smoke(args):
    """Quick smoke test — one snapshot, one experiment."""
    print("=" * 60)
    print("V15 Experiments — Smoke Test")
    print("=" * 60)

    seed = 42
    snapshots = find_v15_snapshots(seed)
    if not snapshots:
        print("No V15 snapshots found!")
        return

    snap_path = str(snapshots[min(2, len(snapshots) - 1)])  # cycle_010
    cycle = get_cycle_number(Path(snap_path))
    print(f"Snapshot: {snap_path} (cycle {cycle})")

    # Test substrate setup
    print("\n--- Substrate Setup ---")
    t0 = time.time()
    grid_np, resource_np, run_chunk_fn, rng, config, meta = \
        setup_v15_substrate(snap_path, seed)
    print(f"Grid: {grid_np.shape}, Resource: {resource_np.shape}")
    print(f"Config keys: {len(config)} entries")
    print(f"Setup time: {time.time()-t0:.1f}s")

    # Test forward simulation
    print("\n--- Forward Simulation ---")
    import jax.numpy as jnp
    t0 = time.time()
    g, r, rng_out = run_chunk_fn(jnp.array(grid_np), jnp.array(resource_np),
                                  n_steps=10, rng=rng)
    print(f"10 steps: grid alive={float(jnp.mean(g > 0.01)):.3f} [{time.time()-t0:.1f}s]")

    # Test Experiment 2 (World Model)
    print("\n--- Experiment 2: World Model ---")
    t0 = time.time()
    results, exp_meta = run_world_model(
        snap_path, seed,
        n_recording_steps=15,
        tau_values=(1, 2, 5),
        max_patterns=5,
    )
    elapsed = time.time() - t0
    print(f"Patterns analyzed: {len(results)}")
    if results:
        c_wms = [r.C_wm for r in results]
        print(f"C_wm: mean={np.mean(c_wms):.6f}, max={np.max(c_wms):.6f}")
        print(f"H_wm: {np.mean([r.H_wm for r in results]):.1f}")
    print(f"Elapsed: {elapsed:.1f}s")

    print(f"\n{'='*60}")
    if results:
        print("SMOKE TEST PASSED")
    else:
        print("SMOKE TEST FAILED (no results)")


def cmd_run(args):
    """Run a specific experiment on specified seeds."""
    exp_num = args.exp
    seeds = args.seeds
    n_rec = args.recording_steps

    if exp_num not in EXPERIMENT_MAP:
        print(f"Unknown experiment {exp_num}. Available: {list(EXPERIMENT_MAP.keys())}")
        return

    exp_name = EXPERIMENT_MAP[exp_num][0]
    print("=" * 60)
    print(f"V15 Experiment {exp_num}: {exp_name}")
    print(f"Seeds: {seeds}, Recording steps: {n_rec}")
    print("=" * 60)

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        run_experiment_on_seed(exp_num, seed, n_recording_steps=n_rec)


def cmd_all(args):
    """Run all priority experiments on all seeds."""
    seeds = args.seeds
    n_rec = args.recording_steps
    priority_exps = [2, 5, 6, 7]  # World model, counterfactual, self-model, affect geometry

    print("=" * 60)
    print("V15 Experiments — Full Run (Priority)")
    print(f"Experiments: {priority_exps}")
    print(f"Seeds: {seeds}, Recording steps: {n_rec}")
    print("=" * 60)

    for exp_num in priority_exps:
        exp_name = EXPERIMENT_MAP[exp_num][0]
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {exp_num}: {exp_name}")
        print(f"{'='*60}")
        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            run_experiment_on_seed(exp_num, seed, n_recording_steps=n_rec)


def main():
    parser = argparse.ArgumentParser(description='V15 Experiment Runner')
    sub = parser.add_subparsers(dest='command')

    # Smoke test
    sub.add_parser('smoke', help='Quick smoke test')

    # Single experiment
    p_run = sub.add_parser('run', help='Run single experiment')
    p_run.add_argument('--exp', type=int, required=True,
                       help='Experiment number (2,3,5,6,7)')
    p_run.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 7])
    p_run.add_argument('--recording-steps', type=int, default=50)

    # All experiments
    p_all = sub.add_parser('all', help='Run all priority experiments')
    p_all.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 7])
    p_all.add_argument('--recording-steps', type=int, default=50)

    args = parser.parse_args()
    if args.command == 'smoke':
        cmd_smoke(args)
    elif args.command == 'run':
        cmd_run(args)
    elif args.command == 'all':
        cmd_all(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
