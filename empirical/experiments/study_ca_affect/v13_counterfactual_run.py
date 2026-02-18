"""Experiment 5: Counterfactual Detachment — CLI Runner.

Usage:
    python v13_counterfactual_run.py smoke
    python v13_counterfactual_run.py full --seeds 123 42 7
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
    from v13_counterfactual import measure_counterfactual_from_snapshot, results_to_dict

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
    print("SMOKE TEST: Experiment 5 — Counterfactual Detachment")
    print(f"Snapshot: {snap_path}")
    print(f"Seed: {seed}, Cycle: {cycle}")
    print("=" * 60)

    t0 = time.time()
    results, meta = measure_counterfactual_from_snapshot(
        snap_path, seed, n_recording_steps=20, max_patterns=5)
    elapsed = time.time() - t0

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Patterns analyzed: {len(results)}")

    if results:
        out = results_to_dict(results, meta)
        s = out['summary']
        print(f"\nSummary:")
        print(f"  Mean ρ_sync:       {s['mean_rho_sync']:.3f}")
        print(f"  Mean detach frac:  {s['mean_detach_frac']:.3f}")
        print(f"  Mean I_img:        {s['mean_I_img']:.4f}")
        print(f"  Mean H_branch:     {s['mean_H_branch']:.3f}")
        print(f"  With detachment:   {s['n_with_detachment']}/{len(results)}")
        print(f"  With imagination:  {s['n_with_imagination']}/{len(results)}")

        print(f"\n  Per-pattern:")
        for r in results:
            print(f"    P{r.pattern_id}: ρ={r.mean_rho_sync:.3f} "
                  f"det={r.detachment_fraction:.2f} "
                  f"events={r.n_detachment_events} "
                  f"I_img={r.I_img:.4f} H={r.H_branch:.2f}")

        out_path = RESULTS_BASE / 'cf_smoke.json'
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nSaved: {out_path}")

    print(f"\n--- Validation ---")
    ok = all(np.isfinite(r.mean_rho_sync) for r in results) and elapsed < 120
    print(f"  All finite: {all(np.isfinite(r.mean_rho_sync) for r in results)}")
    print(f"  Elapsed < 120s: {elapsed < 120}")
    print(f"  {'SMOKE TEST PASSED' if ok and results else 'SMOKE TEST FAILED'}")


def cmd_full(args):
    from v13_counterfactual import measure_counterfactual_from_snapshot, results_to_dict

    seeds = args.seeds
    all_trajectories = {}

    for seed in seeds:
        snaps = find_snapshots(seed)
        if not snaps:
            continue

        out_dir = RESULTS_BASE / f'cf_s{seed}'
        out_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print(f"COUNTERFACTUAL ANALYSIS: seed={seed}, {len(snaps)} snapshots")
        print("=" * 60)

        cycle_results = []
        for snap_path in snaps:
            cycle = get_cycle_number(snap_path)
            print(f"  Cycle {cycle:03d}...", end=" ", flush=True)
            t0 = time.time()

            try:
                results, meta = measure_counterfactual_from_snapshot(
                    str(snap_path), seed,
                    n_recording_steps=args.recording_steps,
                    max_patterns=args.max_patterns)
                elapsed = time.time() - t0
                meta['cycle'] = cycle

                out = results_to_dict(results, meta)
                out['cycle'] = cycle

                with open(out_dir / f'cf_cycle_{cycle:03d}.json', 'w') as f:
                    json.dump(out, f, indent=2, default=str)
                cycle_results.append(out)

                s = out['summary']
                print(f"{len(results)} pat, ρ={s['mean_rho_sync']:.3f} "
                      f"det={s['mean_detach_frac']:.2f} "
                      f"I_img={s['mean_I_img']:.4f} ({elapsed:.0f}s)")

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()

        if cycle_results:
            trajectory = [
                {
                    'cycle': r['cycle'],
                    'mean_rho_sync': r['summary']['mean_rho_sync'],
                    'mean_detach_frac': r['summary']['mean_detach_frac'],
                    'mean_I_img': r['summary']['mean_I_img'],
                    'mean_H_branch': r['summary']['mean_H_branch'],
                    'n_with_detachment': r['summary']['n_with_detachment'],
                    'n_with_imagination': r['summary']['n_with_imagination'],
                }
                for r in cycle_results
            ]
            all_trajectories[seed] = trajectory

            with open(out_dir / 'cf_aggregate.json', 'w') as f:
                json.dump({'seed': seed, 'trajectory': trajectory,
                           'cycles': cycle_results}, f, indent=2, default=str)
            print(f"Saved: {out_dir / 'cf_aggregate.json'}")

    if all_trajectories:
        analysis_dir = RESULTS_BASE / 'cf_analysis'
        analysis_dir.mkdir(parents=True, exist_ok=True)
        with open(analysis_dir / 'cf_cross_seed.json', 'w') as f:
            json.dump({'seeds': seeds,
                       'trajectories': {str(s): t for s, t in all_trajectories.items()}
                       }, f, indent=2, default=str)

        print(f"\n--- Cross-Seed Summary ---")
        for seed, traj in all_trajectories.items():
            if traj:
                first, last = traj[0], traj[-1]
                print(f"  Seed {seed}:")
                print(f"    ρ_sync:     {first['mean_rho_sync']:.3f} → {last['mean_rho_sync']:.3f}")
                print(f"    det_frac:   {first['mean_detach_frac']:.3f} → {last['mean_detach_frac']:.3f}")
                print(f"    I_img:      {first['mean_I_img']:.4f} → {last['mean_I_img']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Experiment 5: Counterfactual Detachment')
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
