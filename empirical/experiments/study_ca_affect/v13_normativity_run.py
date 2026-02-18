"""Experiment 9 (adapted): Proto-Normativity — CLI Runner.

Usage:
    python v13_normativity_run.py smoke
    python v13_normativity_run.py full --seeds 123 42 7
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
    from v13_normativity import measure_normativity_from_snapshot, result_to_dict

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
    print("SMOKE TEST: Experiment 9 — Proto-Normativity")
    print(f"Snapshot: {snap_path}")
    print(f"Seed: {seed}, Cycle: {cycle}")
    print("=" * 60)

    t0 = time.time()
    result, meta = measure_normativity_from_snapshot(
        snap_path, seed, n_recording_steps=20, max_patterns=10)
    elapsed = time.time() - t0

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Patterns analyzed: {meta.get('n_patterns_analyzed', 0)}")

    if result is not None:
        phi_sig = "*" if result.delta_phi_p < 0.05 else ""
        val_sig = "*" if result.delta_valence_p < 0.05 else ""
        aro_sig = "*" if result.delta_arousal_p < 0.05 else ""
        print(f"\n  Cooperative events:  {result.n_cooperative}")
        print(f"  Competitive events:  {result.n_competitive}")
        print(f"  Isolated events:     {result.n_isolated_steps}")
        print(f"\n  ΔΦ (coop-comp):      {result.delta_phi:+.6f}{phi_sig} (p={result.delta_phi_p:.4f})")
        print(f"    Φ_coop:            {result.phi_cooperative:.6f}")
        print(f"    Φ_comp:            {result.phi_competitive:.6f}")
        print(f"    Φ_iso:             {result.phi_isolated:.6f}")
        print(f"\n  ΔV (coop-comp):      {result.delta_valence:+.6f}{val_sig} (p={result.delta_valence_p:.4f})")
        print(f"  ΔA (coop-comp):      {result.delta_arousal:+.6f}{aro_sig} (p={result.delta_arousal_p:.4f})")

        out_path = RESULTS_BASE / 'norm_smoke.json'
        out = result_to_dict(result, meta, cycle=cycle)
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nSaved: {out_path}")
    else:
        print("  Insufficient data (need ≥10 cooperative AND ≥10 competitive events).")

    print(f"\n--- Validation ---")
    ok = result is not None and elapsed < 120
    print(f"  Result computed: {result is not None}")
    print(f"  Elapsed < 120s: {elapsed < 120}")
    print(f"  {'SMOKE TEST PASSED' if ok else 'SMOKE TEST FAILED'}")


def cmd_full(args):
    from v13_normativity import measure_normativity_from_snapshot, result_to_dict

    seeds = args.seeds
    all_trajectories = {}

    for seed in seeds:
        snaps = find_snapshots(seed)
        if not snaps:
            continue

        out_dir = RESULTS_BASE / f'norm_s{seed}'
        out_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print(f"NORMATIVITY ANALYSIS: seed={seed}, {len(snaps)} snapshots")
        print("=" * 60)

        cycle_results = []
        for snap_path in snaps:
            cycle = get_cycle_number(snap_path)
            print(f"  Cycle {cycle:03d}...", end=" ", flush=True)
            t0 = time.time()

            try:
                result, meta = measure_normativity_from_snapshot(
                    str(snap_path), seed,
                    n_recording_steps=args.recording_steps,
                    max_patterns=args.max_patterns)
                elapsed = time.time() - t0
                meta['cycle'] = cycle

                out = result_to_dict(result, meta, cycle=cycle)

                with open(out_dir / f'norm_cycle_{cycle:03d}.json', 'w') as f:
                    json.dump(out, f, indent=2, default=str)
                cycle_results.append(out)

                if result is not None:
                    phi_sig = "*" if result.delta_phi_p < 0.05 else ""
                    val_sig = "*" if result.delta_valence_p < 0.05 else ""
                    print(f"{result.n_patterns} pat, "
                          f"coop={result.n_cooperative} comp={result.n_competitive} "
                          f"ΔΦ={result.delta_phi:+.4f}{phi_sig} "
                          f"ΔV={result.delta_valence:+.4f}{val_sig} ({elapsed:.0f}s)")
                else:
                    print(f"insufficient data ({elapsed:.0f}s)")

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()

        if cycle_results:
            trajectory = [
                {
                    'cycle': r['cycle'],
                    'n_cooperative': r['n_cooperative'],
                    'n_competitive': r['n_competitive'],
                    'delta_phi': r['delta_phi'],
                    'delta_phi_p': r['delta_phi_p'],
                    'delta_valence': r['delta_valence'],
                    'delta_valence_p': r['delta_valence_p'],
                    'delta_arousal': r['delta_arousal'],
                    'delta_arousal_p': r['delta_arousal_p'],
                    'n_patterns': r['n_patterns'],
                }
                for r in cycle_results
            ]
            all_trajectories[seed] = trajectory

            with open(out_dir / 'norm_aggregate.json', 'w') as f:
                json.dump({'seed': seed, 'trajectory': trajectory,
                           'cycles': cycle_results}, f, indent=2, default=str)
            print(f"Saved: {out_dir / 'norm_aggregate.json'}")

    if all_trajectories:
        analysis_dir = RESULTS_BASE / 'norm_analysis'
        analysis_dir.mkdir(parents=True, exist_ok=True)
        with open(analysis_dir / 'norm_cross_seed.json', 'w') as f:
            json.dump({'seeds': seeds,
                       'trajectories': {str(s): t for s, t in all_trajectories.items()}
                       }, f, indent=2, default=str)

        print(f"\n--- Cross-Seed Summary ---")
        for seed, traj in all_trajectories.items():
            valid = [t for t in traj if t['delta_phi'] is not None]
            if valid:
                phi_sig_count = sum(1 for t in valid if t['delta_phi_p'] < 0.05)
                val_sig_count = sum(1 for t in valid if t['delta_valence_p'] < 0.05)
                dphis = [t['delta_phi'] for t in valid]
                dvals = [t['delta_valence'] for t in valid]
                print(f"  Seed {seed}:")
                print(f"    ΔΦ range:     {min(dphis):+.4f} to {max(dphis):+.4f}")
                print(f"    ΔΦ sig:       {phi_sig_count}/{len(valid)}")
                print(f"    ΔV range:     {min(dvals):+.4f} to {max(dvals):+.4f}")
                print(f"    ΔV sig:       {val_sig_count}/{len(valid)}")
                print(f"    Mean ΔΦ:      {np.mean(dphis):+.4f}")
                print(f"    Mean ΔV:      {np.mean(dvals):+.4f}")


def main():
    parser = argparse.ArgumentParser(description='Experiment 9: Proto-Normativity')
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
