"""V18 Experiment Runner — Measurement experiments on V18 boundary-dependent substrate.

V18 introduces an insulation field that creates genuine sensory-motor boundaries.
The key question: does this break the wall that made Exps 5, 6, 9 null on V13/V15?

Priority experiments:
  - Exp 5: Counterfactual Detachment (ρ_sync > 0? Reactive → autonomous transition?)
  - Exp 6: Self-Model (ρ_self > 0? SM_sal > 1?)
  - Exp 2: World Model (C_wm improved via boundary gating?)
  - Exp 7: Affect Geometry (RSA maintained or improved?)

Usage:
    python v18_experiments.py smoke          # Quick test (CPU, ~2min)
    python v18_experiments.py run --exp 5    # Single experiment
    python v18_experiments.py all            # All priority experiments
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS_BASE = Path(__file__).parent / 'results'

V18_SEED_DIRS = {
    42: RESULTS_BASE / 'v18_s42',
    123: RESULTS_BASE / 'v18_s123',
    7: RESULTS_BASE / 'v18_s7',
}


def find_v18_snapshots(seed: int) -> list:
    """Find all V18 cycle snapshot files for a seed."""
    seed_dir = V18_SEED_DIRS.get(seed)
    if seed_dir is None or not seed_dir.exists():
        print(f"No V18 results directory for seed={seed}")
        return []
    snap_dir = seed_dir / 'snapshots'
    if not snap_dir.exists():
        print(f"No snapshots directory: {snap_dir}")
        return []
    files = sorted(snap_dir.glob('cycle_*.npz'))
    return files


def get_cycle_number(path: Path) -> int:
    return int(path.stem.split('_')[1])


# ============================================================================
# V18 Substrate Setup
# ============================================================================

def setup_v18_substrate(snapshot_path: str, seed: int, config_overrides: dict = None):
    """Set up V18 substrate from a snapshot.

    Returns:
        (grid_np, resource_np, run_chunk_fn, rng, config, metadata)
    where run_chunk_fn(grid, resource, n_steps, rng) -> (grid, resource, rng)
    """
    import jax.numpy as jnp
    from jax import random
    from v18_substrate import (
        generate_v18_config, init_v18, run_v18_chunk,
    )
    from v15_substrate import (
        generate_resource_patches, compute_patch_regen_mask,
    )

    snap = np.load(snapshot_path)
    grid_np = snap['grid']
    resource_np = snap['resource']
    C, N = grid_np.shape[0], grid_np.shape[1]

    config = generate_v18_config(C=C, N=N, seed=seed)
    config['maintenance_rate'] = 0.003
    config['resource_consume'] = 0.003
    config['resource_regen'] = 0.01
    if config_overrides:
        config.update(config_overrides)

    (_, _, h_embed, kernel_ffts, internal_kernel_ffts,
     coupling, coupling_row_sums, recurrence_coupling, recurrence_crs,
     box_fft) = init_v18(config, seed=seed)

    patches = generate_resource_patches(N, config['n_resource_patches'], seed=seed)
    regen_mask = compute_patch_regen_mask(N, patches, step=0,
                                          shift_period=config['patch_shift_period'])

    rng = random.PRNGKey(seed + 9999)

    def run_chunk_fn(grid, resource, n_steps, rng, **kwargs):
        return run_v18_chunk(
            grid, resource, h_embed, kernel_ffts, internal_kernel_ffts,
            config, coupling, coupling_row_sums,
            recurrence_coupling, recurrence_crs,
            rng, n_steps=n_steps,
            box_fft=box_fft, regen_mask=jnp.array(regen_mask))

    metadata = {
        'substrate': 'v18',
        'seed': seed,
        'C': C,
        'N': N,
        'snapshot_path': str(snapshot_path),
    }

    return grid_np, resource_np, run_chunk_fn, rng, config, metadata


# ============================================================================
# Experiment wrappers (delegate to substrate-agnostic measurement code)
# ============================================================================

def run_world_model(snapshot_path, seed, n_recording_steps=50,
                    substrate_steps_per_record=10, tau_values=(1, 2, 5, 10, 20),
                    max_patterns=20):
    from v13_world_model import run_recording_episode, compute_prediction_gap
    grid_np, resource_np, run_chunk_fn, rng, config, meta = \
        setup_v18_substrate(snapshot_path, seed)
    features = run_recording_episode(
        grid=grid_np, resource=resource_np, run_chunk_fn=run_chunk_fn,
        chunk_kwargs={'rng': rng}, n_recording_steps=n_recording_steps,
        substrate_steps_per_record=substrate_steps_per_record,
        tau_values=tau_values)
    results = []
    for pid in sorted(features.keys(),
                      key=lambda p: len(features[p]['features']), reverse=True)[:max_patterns]:
        wm = compute_prediction_gap(features[pid]['features'], tau_values=tau_values)
        if wm is not None:
            wm.pattern_id = pid
            results.append(wm)
    meta['experiment'] = 'world_model'
    meta['tau_values'] = list(tau_values)
    return results, meta


def run_counterfactual(snapshot_path, seed, n_recording_steps=50,
                       substrate_steps_per_record=10):
    from v13_counterfactual import analyze_counterfactual
    from v13_world_model import run_recording_episode
    grid_np, resource_np, run_chunk_fn, rng, config, meta = \
        setup_v18_substrate(snapshot_path, seed)
    features = run_recording_episode(
        grid=grid_np, resource=resource_np, run_chunk_fn=run_chunk_fn,
        chunk_kwargs={'rng': rng}, n_recording_steps=n_recording_steps,
        substrate_steps_per_record=substrate_steps_per_record,
        tau_values=(1, 5, 10))
    results = []
    for pid in sorted(features.keys(),
                      key=lambda p: len(features[p]['features']), reverse=True)[:20]:
        feat_list = features[pid]['features']
        if len(feat_list) < 15:
            continue
        cf = analyze_counterfactual(feat_list)
        if cf is not None:
            cf.pattern_id = pid
            results.append(cf)
    meta['experiment'] = 'counterfactual'
    return results, meta


def run_self_model(snapshot_path, seed, n_recording_steps=50,
                   substrate_steps_per_record=10):
    from v13_self_model import analyze_self_model
    from v13_world_model import run_recording_episode
    grid_np, resource_np, run_chunk_fn, rng, config, meta = \
        setup_v18_substrate(snapshot_path, seed)
    features = run_recording_episode(
        grid=grid_np, resource=resource_np, run_chunk_fn=run_chunk_fn,
        chunk_kwargs={'rng': rng}, n_recording_steps=n_recording_steps,
        substrate_steps_per_record=substrate_steps_per_record,
        tau_values=(1, 2, 5, 10))
    results = []
    for pid in sorted(features.keys(),
                      key=lambda p: len(features[p]['features']), reverse=True)[:20]:
        feat_list = features[pid]['features']
        if len(feat_list) < 10:
            continue
        sm = analyze_self_model(feat_list)
        if sm is not None:
            sm.pattern_id = pid
            results.append(sm)
    meta['experiment'] = 'self_model'
    return results, meta


def run_affect_geometry(snapshot_path, seed, n_recording_steps=50,
                        substrate_steps_per_record=10):
    import jax.numpy as jnp
    from v13_affect_geometry import (
        extract_space_A, extract_space_C, compute_rsa, AffectGeometryResult,
    )
    from v13_world_model import detect_patterns_for_wm, extract_internal_state, extract_boundary_obs

    grid_np, resource_np, run_chunk_fn, rng, config, meta = \
        setup_v18_substrate(snapshot_path, seed)
    N = grid_np.shape[1]

    grids = [grid_np.copy()]
    resources = [resource_np.copy()]
    g, r = jnp.array(grid_np), jnp.array(resource_np)
    for _ in range(n_recording_steps):
        g, r, rng = run_chunk_fn(g, r, n_steps=substrate_steps_per_record, rng=rng)
        grids.append(np.array(g))
        resources.append(np.array(r))

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
        matched_tracked, matched_detected = set(), set()
        costs = []
        for j, p in enumerate(pats_t):
            for pid in tracked_ids:
                last_c = all_features[pid]['center_history'][-1]
                dr = min(abs(p['center'][0] - last_c[0]), N - abs(p['center'][0] - last_c[0]))
                dc = min(abs(p['center'][1] - last_c[1]), N - abs(p['center'][1] - last_c[1]))
                dist = np.sqrt(dr**2 + dc**2)
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
            all_features[pid]['features'].append({'s_B': s_B, 's_dB': s_dB})

    dim_A = ['valence', 'arousal', 'integration', 'd_eff', 'cf_weight', 'sm_sal']
    dim_C = ['approach_avoid', 'activity', 'growth', 'stability']
    A_list, C_list, per_pattern = [], [], []

    for pid in sorted(all_features.keys(),
                      key=lambda p: len(all_features[p]['features']), reverse=True)[:20]:
        data = all_features[pid]
        a = extract_space_A(data['features'], resources, data['center_history'], N)
        c = extract_space_C(data['features'], data['center_history'],
                           data['size_history'], resources, N)
        if a is not None and c is not None:
            A_list.append(a)
            C_list.append(c)
            per_pattern.append({
                'pattern_id': pid,
                'space_A': {dim_A[i]: float(a[i]) for i in range(len(dim_A))},
                'space_C': {dim_C[i]: float(c[i]) for i in range(len(dim_C))},
                'n_timesteps': len(data['features']),
            })

    if len(A_list) < 4:
        meta['experiment'] = 'affect_geometry'
        return [], meta

    space_A, space_C = np.array(A_list), np.array(C_list)
    rsa_rho, rsa_p = compute_rsa(space_A, space_C)

    result = AffectGeometryResult(
        n_patterns=len(A_list), rsa_rho=rsa_rho, rsa_p=rsa_p,
        space_A=space_A, space_C=space_C,
        dim_labels_A=dim_A, dim_labels_C=dim_C, per_pattern=per_pattern)
    meta['experiment'] = 'affect_geometry'
    return [result], meta


# ============================================================================
# Unified Runner
# ============================================================================

EXPERIMENT_MAP = {
    2: ('world_model', run_world_model),
    5: ('counterfactual', run_counterfactual),
    6: ('self_model', run_self_model),
    7: ('affect_geometry', run_affect_geometry),
}


def run_experiment_on_seed(exp_num, seed, n_recording_steps=50, output_dir=None):
    exp_name, exp_fn = EXPERIMENT_MAP[exp_num]
    snapshots = find_v18_snapshots(seed)
    if not snapshots:
        return []

    if output_dir is None:
        output_dir = RESULTS_BASE / f'v18_{exp_name}_s{seed}'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for snap_path in snapshots:
        cycle = get_cycle_number(snap_path)
        print(f"  Cycle {cycle:03d}...", end=" ", flush=True)
        t0 = time.time()

        try:
            results, meta = exp_fn(str(snap_path), seed,
                                   n_recording_steps=n_recording_steps)
            elapsed = time.time() - t0

            if exp_num == 2:
                from v13_world_model import results_to_dict
                output = results_to_dict(results, meta)
                c_wm = output['summary']['mean_C_wm']
                n_wm = output['summary']['n_with_world_model']
                print(f"C_wm={c_wm:.5f} WM={n_wm}/{len(results)} [{elapsed:.0f}s]")
            elif exp_num == 5:
                if results:
                    rho_syncs = [r.mean_rho_sync for r in results if hasattr(r, 'mean_rho_sync')]
                    mean_rho = np.mean(rho_syncs) if rho_syncs else 0
                    det_fracs = [r.detachment_fraction for r in results if hasattr(r, 'detachment_fraction')]
                    mean_det = np.mean(det_fracs) if det_fracs else 0
                    print(f"n={len(results)} ρ_sync={mean_rho:.4f} det={mean_det:.2f} [{elapsed:.0f}s]")
                else:
                    print(f"n=0 [{elapsed:.0f}s]")
            elif exp_num == 6:
                if results:
                    rho_selfs = [r.rho_self for r in results if hasattr(r, 'rho_self')]
                    mean_rho = np.mean(rho_selfs) if rho_selfs else 0
                    sm_sals = [r.SM_sal for r in results if hasattr(r, 'SM_sal')]
                    mean_sal = np.mean(sm_sals) if sm_sals else 0
                    print(f"n={len(results)} ρ_self={mean_rho:.4f} SM_sal={mean_sal:.4f} [{elapsed:.0f}s]")
                else:
                    print(f"n=0 [{elapsed:.0f}s]")
            elif exp_num == 7:
                if results and hasattr(results[0], 'rsa_rho'):
                    r = results[0]
                    print(f"n_pat={r.n_patterns} RSA_ρ={r.rsa_rho:.4f} p={r.rsa_p:.4f} [{elapsed:.0f}s]")
                else:
                    print(f"n=0 [{elapsed:.0f}s]")
            else:
                print(f"n={len(results)} [{elapsed:.0f}s]")

            out_path = output_dir / f'cycle_{cycle:03d}.json'
            if exp_num == 2:
                from v13_world_model import results_to_dict
                with open(out_path, 'w') as f:
                    json.dump(results_to_dict(results, meta), f, indent=2, default=str)
            else:
                out_data = {'metadata': meta, 'n_results': len(results), 'cycle': cycle}
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
                                d[k] = {str(kk): (float(vv) if isinstance(vv, (float, np.floating)) else vv)
                                         for kk, vv in v.items()}
                            else:
                                d[k] = v
                        out_data['results'].append(d)
                with open(out_path, 'w') as f:
                    json.dump(out_data, f, indent=2, default=str)

            all_results.append({'cycle': cycle, 'n_results': len(results), 'elapsed': elapsed})

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
    print("=" * 60)
    print("V18 Experiments — Smoke Test")
    print("=" * 60)

    seed = 42
    snapshots = find_v18_snapshots(seed)
    if not snapshots:
        print("No V18 snapshots found!")
        return

    snap_path = str(snapshots[min(2, len(snapshots) - 1)])
    cycle = get_cycle_number(Path(snap_path))
    print(f"Snapshot: {snap_path} (cycle {cycle})")

    print("\n--- Substrate Setup ---")
    t0 = time.time()
    grid_np, resource_np, run_chunk_fn, rng, config, meta = \
        setup_v18_substrate(snap_path, seed)
    print(f"Grid: {grid_np.shape}, Resource: {resource_np.shape}")
    print(f"Setup time: {time.time()-t0:.1f}s")

    print("\n--- Forward Simulation ---")
    import jax.numpy as jnp
    t0 = time.time()
    g, r, rng_out = run_chunk_fn(jnp.array(grid_np), jnp.array(resource_np),
                                  n_steps=10, rng=rng)
    print(f"10 steps: grid alive={float(jnp.mean(g > 0.01)):.3f} [{time.time()-t0:.1f}s]")

    # Quick Exp 5 (counterfactual) — the key test
    print("\n--- Experiment 5: Counterfactual ---")
    t0 = time.time()
    results, exp_meta = run_counterfactual(snap_path, seed, n_recording_steps=15)
    elapsed = time.time() - t0
    if results:
        rho_syncs = [r.mean_rho_sync for r in results if hasattr(r, 'mean_rho_sync')]
        mean_rho = np.mean(rho_syncs) if rho_syncs else 0
        print(f"n={len(results)} ρ_sync={mean_rho:.4f} [{elapsed:.1f}s]")
        print(f"  >>> {'WALL BROKEN' if mean_rho > 0.05 else 'WALL PERSISTS'} <<<")
    else:
        print(f"n=0 [{elapsed:.1f}s]")

    print(f"\n{'='*60}")
    print("SMOKE TEST COMPLETE")


def cmd_run(args):
    exp_num = args.exp
    if exp_num not in EXPERIMENT_MAP:
        print(f"Unknown experiment {exp_num}. Available: {list(EXPERIMENT_MAP.keys())}")
        return
    exp_name = EXPERIMENT_MAP[exp_num][0]
    print("=" * 60)
    print(f"V18 Experiment {exp_num}: {exp_name}")
    print(f"Seeds: {args.seeds}")
    print("=" * 60)
    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        run_experiment_on_seed(exp_num, seed, n_recording_steps=args.recording_steps)


def cmd_all(args):
    priority_exps = [5, 6, 2, 7]  # Wall-breaking first, then others
    print("=" * 60)
    print("V18 Experiments — Full Run")
    print(f"Experiments: {priority_exps}")
    print(f"Seeds: {args.seeds}")
    print("=" * 60)
    for exp_num in priority_exps:
        exp_name = EXPERIMENT_MAP[exp_num][0]
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {exp_num}: {exp_name}")
        print(f"{'='*60}")
        for seed in args.seeds:
            print(f"\n--- Seed {seed} ---")
            run_experiment_on_seed(exp_num, seed, n_recording_steps=args.recording_steps)


def main():
    parser = argparse.ArgumentParser(description='V18 Experiment Runner')
    sub = parser.add_subparsers(dest='command')
    sub.add_parser('smoke', help='Quick smoke test')
    p_run = sub.add_parser('run', help='Run single experiment')
    p_run.add_argument('--exp', type=int, required=True)
    p_run.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 7])
    p_run.add_argument('--recording-steps', type=int, default=50)
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
