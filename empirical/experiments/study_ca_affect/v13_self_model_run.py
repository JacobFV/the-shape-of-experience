"""Experiment 6: Self-Model Emergence — CLI Runner.

Usage:
    python v13_self_model_run.py smoke
    python v13_self_model_run.py full --seeds 123 42 7
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS_BASE = Path(__file__).parent / 'results'
SEED_DIRS = {
    123: RESULTS_BASE / 'v13_s123' / 'v13_s123',
    42: RESULTS_BASE / 'v13_s42_v2' / 'v13_s42_v2',
    7: RESULTS_BASE / 'v13_s7' / 'v13_s7',
}


def find_snapshots(seed):
    d = SEED_DIRS.get(seed)
    if d is None or not d.exists():
        return []
    s = d / 'snapshots'
    return sorted(s.glob('cycle_*.npz')) if s.exists() else []


def get_cycle_number(p):
    return int(p.stem.split('_')[1])


def cmd_smoke(args):
    from v13_self_model import measure_self_model_from_snapshot, results_to_dict

    seed = args.seed
    snaps = find_snapshots(seed) or find_snapshots(123)
    seed = seed if find_snapshots(seed) else 123
    if not snaps:
        print("ERROR: No snapshots.")
        return

    snap_idx = min(2, len(snaps) - 1)
    snap_path = str(snaps[snap_idx])
    cycle = get_cycle_number(snaps[snap_idx])

    print("=" * 60)
    print("SMOKE TEST: Experiment 6 — Self-Model Emergence")
    print(f"Snapshot: {snap_path}")
    print(f"Seed: {seed}, Cycle: {cycle}")
    print("=" * 60)

    t0 = time.time()
    results, meta = measure_self_model_from_snapshot(
        snap_path, seed, n_recording_steps=20, max_patterns=5)
    elapsed = time.time() - t0

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Patterns analyzed: {len(results)}")

    if results:
        out = results_to_dict(results, meta, cycle=cycle)
        s = out['summary']
        print(f"\nSummary:")
        print(f"  Mean ρ_self:       {s['mean_rho_self']:.4f}")
        print(f"  Mean SM_capacity:  {s['mean_SM_capacity']:.4f}")
        print(f"  Mean SM_sal:       {s['mean_SM_sal']:.4f}")
        print(f"  With self-model:   {s['n_with_self_model']}/{len(results)}")
        print(f"  With high ρ_self:  {s['n_with_high_rho']}/{len(results)}")
        print(f"  With salience>1:   {s['n_with_salience']}/{len(results)}")

        print(f"\n  SM by τ: {s['mean_SM_by_tau']}")

        print(f"\n  Per-pattern:")
        for r in results:
            sm_str = ", ".join(f"τ={k}:{v:.4f}" for k, v in sorted(r.SM.items()))
            print(f"    P{r.pattern_id}: ρ_self={r.rho_self:.4f} "
                  f"SM_cap={r.SM_capacity:.4f} "
                  f"SM_sal={r.SM_sal:.4f}")
            print(f"      SM: {sm_str}")

        out_path = RESULTS_BASE / 'sm_smoke.json'
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nSaved: {out_path}")

    print(f"\n--- Validation ---")
    ok = (all(np.isfinite(r.rho_self) for r in results)
          and elapsed < 120 and len(results) > 0)
    print(f"  All finite: {all(np.isfinite(r.rho_self) for r in results)}")
    print(f"  Elapsed < 120s: {elapsed < 120}")
    print(f"  {'SMOKE TEST PASSED' if ok else 'SMOKE TEST FAILED'}")


def cmd_full(args):
    from v13_self_model import measure_self_model_from_snapshot, results_to_dict

    seeds = args.seeds
    all_trajectories = {}

    for seed in seeds:
        snaps = find_snapshots(seed)
        if not snaps:
            continue

        out_dir = RESULTS_BASE / f'sm_s{seed}'
        out_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print(f"SELF-MODEL ANALYSIS: seed={seed}, {len(snaps)} snapshots")
        print("=" * 60)

        cycle_results = []
        for snap_path in snaps:
            cycle = get_cycle_number(snap_path)
            print(f"  Cycle {cycle:03d}...", end=" ", flush=True)
            t0 = time.time()

            try:
                results, meta = measure_self_model_from_snapshot(
                    str(snap_path), seed,
                    n_recording_steps=args.recording_steps,
                    max_patterns=args.max_patterns)
                elapsed = time.time() - t0
                meta['cycle'] = cycle

                out = results_to_dict(results, meta, cycle=cycle)

                with open(out_dir / f'sm_cycle_{cycle:03d}.json', 'w') as f:
                    json.dump(out, f, indent=2, default=str)
                cycle_results.append(out)

                s = out['summary']
                print(f"{len(results)} pat, ρ_self={s['mean_rho_self']:.4f} "
                      f"SM_cap={s['mean_SM_capacity']:.4f} "
                      f"SM_sal={s['mean_SM_sal']:.4f} ({elapsed:.0f}s)")

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()

        if cycle_results:
            trajectory = [
                {
                    'cycle': r['cycle'],
                    'mean_rho_self': r['summary']['mean_rho_self'],
                    'mean_SM_capacity': r['summary']['mean_SM_capacity'],
                    'mean_SM_sal': r['summary']['mean_SM_sal'],
                    'n_with_self_model': r['summary']['n_with_self_model'],
                    'n_with_high_rho': r['summary']['n_with_high_rho'],
                    'n_with_salience': r['summary']['n_with_salience'],
                    'mean_SM_by_tau': r['summary']['mean_SM_by_tau'],
                }
                for r in cycle_results
            ]
            all_trajectories[seed] = trajectory

            with open(out_dir / 'sm_aggregate.json', 'w') as f:
                json.dump({'seed': seed, 'trajectory': trajectory,
                           'cycles': cycle_results}, f, indent=2, default=str)
            print(f"Saved: {out_dir / 'sm_aggregate.json'}")

    if all_trajectories:
        analysis_dir = RESULTS_BASE / 'sm_analysis'
        analysis_dir.mkdir(parents=True, exist_ok=True)
        with open(analysis_dir / 'sm_cross_seed.json', 'w') as f:
            json.dump({'seeds': seeds,
                       'trajectories': {str(s): t for s, t in all_trajectories.items()}
                       }, f, indent=2, default=str)

        print(f"\n--- Cross-Seed Summary ---")
        for seed, traj in all_trajectories.items():
            if traj:
                first, last = traj[0], traj[-1]
                print(f"  Seed {seed}:")
                print(f"    ρ_self:     {first['mean_rho_self']:.4f} → {last['mean_rho_self']:.4f}")
                print(f"    SM_cap:     {first['mean_SM_capacity']:.4f} → {last['mean_SM_capacity']:.4f}")
                print(f"    SM_sal:     {first['mean_SM_sal']:.4f} → {last['mean_SM_sal']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Experiment 6: Self-Model Emergence')
    sub = parser.add_subparsers(dest='command')

    p = sub.add_parser('smoke')
    p.add_argument('--seed', type=int, default=123)

    p = sub.add_parser('full')
    p.add_argument('--seeds', type=int, nargs='+', default=[123, 42, 7])
    p.add_argument('--recording-steps', type=int, default=50)
    p.add_argument('--max-patterns', type=int, default=20)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    {'smoke': cmd_smoke, 'full': cmd_full}[args.command](args)


if __name__ == '__main__':
    main()
