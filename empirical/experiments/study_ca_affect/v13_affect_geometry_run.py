"""Experiment 7 (partial): Affect Geometry Verification — CLI Runner.

Tests RSA between structural affect (Space A) and behavioral affect (Space C).

Usage:
    python v13_affect_geometry_run.py smoke
    python v13_affect_geometry_run.py full --seeds 123 42 7
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
    from v13_affect_geometry import measure_affect_geometry_from_snapshot, result_to_dict

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
    print("SMOKE TEST: Experiment 7 — Affect Geometry (A↔C)")
    print(f"Snapshot: {snap_path}")
    print(f"Seed: {seed}, Cycle: {cycle}")
    print("=" * 60)

    t0 = time.time()
    result, meta = measure_affect_geometry_from_snapshot(
        snap_path, seed, n_recording_steps=20, max_patterns=10)
    elapsed = time.time() - t0

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Patterns analyzed: {meta.get('n_patterns_analyzed', 0)}")

    if result is not None:
        out = result_to_dict(result, meta, cycle=cycle)
        print(f"\n  RSA ρ(A,C) = {result.rsa_rho:.4f}  (p = {result.rsa_p:.4f})")
        print(f"  n_patterns = {result.n_patterns}")

        print(f"\n  Space A means: {out['space_A_means']}")
        print(f"  Space C means: {out['space_C_means']}")

        print(f"\n  Per-pattern (first 5):")
        for pp in result.per_pattern[:5]:
            a = pp['space_A']
            c = pp['space_C']
            print(f"    P{pp['pattern_id']}: "
                  f"V={a['valence']:.4f} A={a['arousal']:.2f} "
                  f"Φ={a['integration']:.3f} d={a['d_eff']:.1f} | "
                  f"app={c['approach_avoid']:.4f} act={c['activity']:.2f} "
                  f"grw={c['growth']:.3f} stb={c['stability']:.3f}")

        out_path = RESULTS_BASE / 'ag_smoke.json'
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nSaved: {out_path}")
    else:
        print("  Insufficient patterns for RSA.")

    print(f"\n--- Validation ---")
    ok = result is not None and elapsed < 120
    print(f"  Result computed: {result is not None}")
    print(f"  Elapsed < 120s: {elapsed < 120}")
    print(f"  {'SMOKE TEST PASSED' if ok else 'SMOKE TEST FAILED'}")


def cmd_full(args):
    from v13_affect_geometry import measure_affect_geometry_from_snapshot, result_to_dict

    seeds = args.seeds
    all_trajectories = {}

    for seed in seeds:
        snaps = find_snapshots(seed)
        if not snaps:
            continue

        out_dir = RESULTS_BASE / f'ag_s{seed}'
        out_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print(f"AFFECT GEOMETRY: seed={seed}, {len(snaps)} snapshots")
        print("=" * 60)

        cycle_results = []
        for snap_path in snaps:
            cycle = get_cycle_number(snap_path)
            print(f"  Cycle {cycle:03d}...", end=" ", flush=True)
            t0 = time.time()

            try:
                result, meta = measure_affect_geometry_from_snapshot(
                    str(snap_path), seed,
                    n_recording_steps=args.recording_steps,
                    max_patterns=args.max_patterns)
                elapsed = time.time() - t0
                meta['cycle'] = cycle

                out = result_to_dict(result, meta, cycle=cycle)

                with open(out_dir / f'ag_cycle_{cycle:03d}.json', 'w') as f:
                    json.dump(out, f, indent=2, default=str)
                cycle_results.append(out)

                if result is not None:
                    sig = "*" if result.rsa_p < 0.05 else ""
                    print(f"{result.n_patterns} pat, "
                          f"ρ(A,C)={result.rsa_rho:.4f}{sig} "
                          f"p={result.rsa_p:.4f} ({elapsed:.0f}s)")
                else:
                    print(f"<4 patterns, skipped ({elapsed:.0f}s)")

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()

        if cycle_results:
            trajectory = [
                {
                    'cycle': r['cycle'],
                    'rsa_rho': r['rsa_rho'],
                    'rsa_p': r['rsa_p'],
                    'n_patterns': r['n_patterns'],
                }
                for r in cycle_results
            ]
            all_trajectories[seed] = trajectory

            with open(out_dir / 'ag_aggregate.json', 'w') as f:
                json.dump({'seed': seed, 'trajectory': trajectory,
                           'cycles': cycle_results}, f, indent=2, default=str)
            print(f"Saved: {out_dir / 'ag_aggregate.json'}")

    if all_trajectories:
        analysis_dir = RESULTS_BASE / 'ag_analysis'
        analysis_dir.mkdir(parents=True, exist_ok=True)
        with open(analysis_dir / 'ag_cross_seed.json', 'w') as f:
            json.dump({'seeds': seeds,
                       'trajectories': {str(s): t for s, t in all_trajectories.items()}
                       }, f, indent=2, default=str)

        print(f"\n--- Cross-Seed Summary ---")
        for seed, traj in all_trajectories.items():
            if traj:
                rhos = [t['rsa_rho'] for t in traj if t['rsa_rho'] is not None]
                ps = [t['rsa_p'] for t in traj if t['rsa_p'] is not None]
                sig_count = sum(1 for p in ps if p < 0.05)
                print(f"  Seed {seed}:")
                print(f"    ρ(A,C) range: {min(rhos):.4f} to {max(rhos):.4f}")
                print(f"    Significant (p<0.05): {sig_count}/{len(ps)} snapshots")
                if traj:
                    first_valid = next((t for t in traj if t['rsa_rho'] is not None), None)
                    last_valid = next((t for t in reversed(traj) if t['rsa_rho'] is not None), None)
                    if first_valid and last_valid:
                        print(f"    Trajectory: {first_valid['rsa_rho']:.4f} → {last_valid['rsa_rho']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Experiment 7: Affect Geometry (A↔C)')
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
