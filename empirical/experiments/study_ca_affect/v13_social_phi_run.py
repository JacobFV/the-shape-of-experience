"""Experiment 10 (adapted): Social-Scale Integration — CLI Runner.

Usage:
    python v13_social_phi_run.py smoke
    python v13_social_phi_run.py full --seeds 123 42 7
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
    from v13_social_phi import measure_social_phi_from_snapshot, result_to_dict

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
    print("SMOKE TEST: Experiment 10 — Social-Scale Integration")
    print(f"Snapshot: {snap_path}")
    print(f"Seed: {seed}, Cycle: {cycle}")
    print("=" * 60)

    t0 = time.time()
    result, meta = measure_social_phi_from_snapshot(
        snap_path, seed, n_recording_steps=20, max_patterns=10)
    elapsed = time.time() - t0

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Patterns analyzed: {meta.get('n_patterns_analyzed', 0)}")

    if result is not None:
        super_mark = ">" if result.superorganism_ratio > 1 else "<"
        print(f"\n  Individual Φ (mean):  {result.mean_phi_individual:.6f}")
        print(f"  Individual Φ (sum):   {result.sum_phi_individual:.6f}")
        print(f"  Group Φ:              {result.phi_group:.6f}")
        print(f"  Superorganism ratio:  {result.superorganism_ratio:.4f} ({super_mark}1)")
        print(f"  Partition loss:       {result.partition_loss:.6f}")
        print(f"  Pairwise MI (mean):   {result.mean_pairwise_MI:.4f}")
        print(f"  Group coherence:      {result.group_coherence:.3f}")

        out_path = RESULTS_BASE / 'social_phi_smoke.json'
        out = result_to_dict(result, meta, cycle=cycle)
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nSaved: {out_path}")
    else:
        print("  Insufficient patterns (<4).")

    print(f"\n--- Validation ---")
    ok = result is not None and elapsed < 120
    print(f"  Result computed: {result is not None}")
    print(f"  Elapsed < 120s: {elapsed < 120}")
    print(f"  {'SMOKE TEST PASSED' if ok else 'SMOKE TEST FAILED'}")


def cmd_full(args):
    from v13_social_phi import measure_social_phi_from_snapshot, result_to_dict

    seeds = args.seeds
    all_trajectories = {}

    for seed in seeds:
        snaps = find_snapshots(seed)
        if not snaps:
            continue

        out_dir = RESULTS_BASE / f'social_phi_s{seed}'
        out_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print(f"SOCIAL PHI ANALYSIS: seed={seed}, {len(snaps)} snapshots")
        print("=" * 60)

        cycle_results = []
        for snap_path in snaps:
            cycle = get_cycle_number(snap_path)
            print(f"  Cycle {cycle:03d}...", end=" ", flush=True)
            t0 = time.time()

            try:
                result, meta = measure_social_phi_from_snapshot(
                    str(snap_path), seed,
                    n_recording_steps=args.recording_steps,
                    max_patterns=args.max_patterns)
                elapsed = time.time() - t0
                meta['cycle'] = cycle

                out = result_to_dict(result, meta, cycle=cycle)

                with open(out_dir / f'social_phi_cycle_{cycle:03d}.json', 'w') as f:
                    json.dump(out, f, indent=2, default=str)
                cycle_results.append(out)

                if result is not None:
                    super_mark = ">" if result.superorganism_ratio > 1 else "<"
                    print(f"{result.n_patterns} pat, "
                          f"Φ_G={result.phi_group:.4f} "
                          f"ΣΦᵢ={result.sum_phi_individual:.4f} "
                          f"ratio={result.superorganism_ratio:.3f}{super_mark}1 "
                          f"MI={result.mean_pairwise_MI:.4f} ({elapsed:.0f}s)")
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
                    'phi_group': r['phi_group'],
                    'sum_phi_individual': r['sum_phi_individual'],
                    'superorganism_ratio': r['superorganism_ratio'],
                    'partition_loss': r['partition_loss'],
                    'mean_pairwise_MI': r['mean_pairwise_MI'],
                    'group_coherence': r['group_coherence'],
                    'n_patterns': r['n_patterns'],
                }
                for r in cycle_results
            ]
            all_trajectories[seed] = trajectory

            with open(out_dir / 'social_phi_aggregate.json', 'w') as f:
                json.dump({'seed': seed, 'trajectory': trajectory,
                           'cycles': cycle_results}, f, indent=2, default=str)
            print(f"Saved: {out_dir / 'social_phi_aggregate.json'}")

    if all_trajectories:
        analysis_dir = RESULTS_BASE / 'social_phi_analysis'
        analysis_dir.mkdir(parents=True, exist_ok=True)
        with open(analysis_dir / 'social_phi_cross_seed.json', 'w') as f:
            json.dump({'seeds': seeds,
                       'trajectories': {str(s): t for s, t in all_trajectories.items()}
                       }, f, indent=2, default=str)

        print(f"\n--- Cross-Seed Summary ---")
        for seed, traj in all_trajectories.items():
            valid = [t for t in traj if t['phi_group'] is not None]
            if valid:
                ratios = [t['superorganism_ratio'] for t in valid]
                losses = [t['partition_loss'] for t in valid]
                n_super = sum(1 for r in ratios if r > 1.0)
                print(f"  Seed {seed}:")
                print(f"    Super ratio range: {min(ratios):.3f} to {max(ratios):.3f}")
                print(f"    Superorganism:     {n_super}/{len(valid)}")
                print(f"    Partition loss:    {np.mean(losses):+.4f}")
                print(f"    Φ_G trajectory:    {valid[0]['phi_group']:.4f} → {valid[-1]['phi_group']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Experiment 10: Social-Scale Integration')
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
