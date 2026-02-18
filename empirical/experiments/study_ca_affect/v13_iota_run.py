"""Experiment 8 (adapted): Inhibition Coefficient (ι) — CLI Runner.

Usage:
    python v13_iota_run.py smoke
    python v13_iota_run.py full --seeds 123 42 7
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
    from v13_iota import measure_iota_from_snapshot, result_to_dict

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
    print("SMOKE TEST: Experiment 8 — Inhibition Coefficient (ι)")
    print(f"Snapshot: {snap_path}")
    print(f"Seed: {seed}, Cycle: {cycle}")
    print("=" * 60)

    t0 = time.time()
    result, meta = measure_iota_from_snapshot(
        snap_path, seed, n_recording_steps=20, max_patterns=10)
    elapsed = time.time() - t0

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Patterns analyzed: {meta.get('n_patterns_analyzed', 0)}")

    if result is not None:
        print(f"\n  ι (mean):         {result.mean_iota:.4f} ± {result.std_iota:.4f}")
        print(f"  ι (range):        [{result.iota_range[0]:.4f}, {result.iota_range[1]:.4f}]")
        print(f"  MI_social:        {result.mean_MI_social:.4f}")
        print(f"  MI_trajectory:    {result.mean_MI_trajectory:.4f}")
        print(f"  MI_resource:      {result.mean_MI_resource:.4f}")
        print(f"  Animism score:    {result.animism_score:.4f}")

        out_path = RESULTS_BASE / 'iota_smoke.json'
        out = result_to_dict(result, meta, cycle=cycle)
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nSaved: {out_path}")
    else:
        print("  Insufficient patterns.")

    print(f"\n--- Validation ---")
    ok = result is not None and elapsed < 120
    print(f"  Result computed: {result is not None}")
    print(f"  Elapsed < 120s: {elapsed < 120}")
    print(f"  {'SMOKE TEST PASSED' if ok else 'SMOKE TEST FAILED'}")


def cmd_full(args):
    from v13_iota import measure_iota_from_snapshot, result_to_dict

    seeds = args.seeds
    all_trajectories = {}

    for seed in seeds:
        snaps = find_snapshots(seed)
        if not snaps:
            continue

        out_dir = RESULTS_BASE / f'iota_s{seed}'
        out_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print(f"IOTA ANALYSIS: seed={seed}, {len(snaps)} snapshots")
        print("=" * 60)

        cycle_results = []
        for snap_path in snaps:
            cycle = get_cycle_number(snap_path)
            print(f"  Cycle {cycle:03d}...", end=" ", flush=True)
            t0 = time.time()

            try:
                result, meta = measure_iota_from_snapshot(
                    str(snap_path), seed,
                    n_recording_steps=args.recording_steps,
                    max_patterns=args.max_patterns)
                elapsed = time.time() - t0
                meta['cycle'] = cycle

                out = result_to_dict(result, meta, cycle=cycle)

                with open(out_dir / f'iota_cycle_{cycle:03d}.json', 'w') as f:
                    json.dump(out, f, indent=2, default=str)
                cycle_results.append(out)

                if result is not None:
                    print(f"{result.n_patterns} pat, "
                          f"ι={result.mean_iota:.3f}±{result.std_iota:.3f} "
                          f"MI_s={result.mean_MI_social:.4f} "
                          f"MI_t={result.mean_MI_trajectory:.4f} "
                          f"MI_r={result.mean_MI_resource:.4f} "
                          f"anim={result.animism_score:.3f} ({elapsed:.0f}s)")
                else:
                    print(f"<3 patterns, skipped ({elapsed:.0f}s)")

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()

        if cycle_results:
            trajectory = [
                {
                    'cycle': r['cycle'],
                    'mean_iota': r['mean_iota'],
                    'std_iota': r['std_iota'],
                    'mean_MI_social': r['mean_MI_social'],
                    'mean_MI_trajectory': r['mean_MI_trajectory'],
                    'mean_MI_resource': r['mean_MI_resource'],
                    'animism_score': r['animism_score'],
                    'n_patterns': r['n_patterns'],
                }
                for r in cycle_results
            ]
            all_trajectories[seed] = trajectory

            with open(out_dir / 'iota_aggregate.json', 'w') as f:
                json.dump({'seed': seed, 'trajectory': trajectory,
                           'cycles': cycle_results}, f, indent=2, default=str)
            print(f"Saved: {out_dir / 'iota_aggregate.json'}")

    if all_trajectories:
        analysis_dir = RESULTS_BASE / 'iota_analysis'
        analysis_dir.mkdir(parents=True, exist_ok=True)
        with open(analysis_dir / 'iota_cross_seed.json', 'w') as f:
            json.dump({'seeds': seeds,
                       'trajectories': {str(s): t for s, t in all_trajectories.items()}
                       }, f, indent=2, default=str)

        print(f"\n--- Cross-Seed Summary ---")
        for seed, traj in all_trajectories.items():
            valid = [t for t in traj if t['mean_iota'] is not None]
            if valid:
                iotas = [t['mean_iota'] for t in valid]
                animisms = [t['animism_score'] for t in valid]
                print(f"  Seed {seed}:")
                print(f"    ι range:       {min(iotas):.3f} to {max(iotas):.3f}")
                print(f"    ι trajectory:  {valid[0]['mean_iota']:.3f} → {valid[-1]['mean_iota']:.3f}")
                print(f"    Animism range: {min(animisms):.3f} to {max(animisms):.3f}")
                print(f"    MI_social:     {valid[0]['mean_MI_social']:.4f} → {valid[-1]['mean_MI_social']:.4f}")
                print(f"    MI_trajectory: {valid[0]['mean_MI_trajectory']:.4f} → {valid[-1]['mean_MI_trajectory']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Experiment 8: Inhibition Coefficient (ι)')
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
