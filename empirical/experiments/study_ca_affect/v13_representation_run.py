"""Experiment 3: Internal Representation Structure — CLI Runner.

Subcommands:
    smoke   — Quick test on one snapshot
    full    — Complete pipeline: all seeds, all snapshots, figures

Usage:
    python v13_representation_run.py smoke
    python v13_representation_run.py full --seeds 123 42 7
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


def find_snapshots(seed: int) -> list:
    seed_dir = SEED_DIRS.get(seed)
    if seed_dir is None or not seed_dir.exists():
        return []
    snap_dir = seed_dir / 'snapshots'
    if not snap_dir.exists():
        return []
    return sorted(snap_dir.glob('cycle_*.npz'))


def get_cycle_number(path: Path) -> int:
    return int(path.stem.split('_')[1])


def cmd_smoke(args):
    """Quick smoke test."""
    from v13_representation import measure_representation_from_snapshot, results_to_dict

    seed = args.seed
    snapshots = find_snapshots(seed)
    if not snapshots:
        seed = 123
        snapshots = find_snapshots(seed)
    if not snapshots:
        print("ERROR: No snapshots available.")
        return

    snap_idx = min(2, len(snapshots) - 1)
    snap_path = str(snapshots[snap_idx])
    cycle = get_cycle_number(snapshots[snap_idx])

    print("=" * 60)
    print("SMOKE TEST: Experiment 3 — Internal Representation Structure")
    print(f"Snapshot: {snap_path}")
    print(f"Seed: {seed}, Cycle: {cycle}")
    print("=" * 60)

    t0 = time.time()
    results, metadata = measure_representation_from_snapshot(
        snapshot_path=snap_path,
        seed=seed,
        n_recording_steps=20,
        max_patterns=5,
    )
    elapsed = time.time() - t0

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Patterns analyzed: {len(results)}")

    if results:
        output = results_to_dict(results, metadata)
        s = output['summary']
        print(f"\nSummary:")
        print(f"  Mean d_eff:   {s['mean_d_eff']:.1f} / {results[0].d_raw}")
        print(f"  Mean A:       {s['mean_A']:.3f}")
        print(f"  Mean D:       {s['mean_D']:.3f}")
        print(f"  Mean K_comp:  {s['mean_K_comp']:.3f}")
        print(f"  Compressed (A>0.5):    {s['n_compressed']}/{len(results)}")
        print(f"  Disentangled (D>0.3):  {s['n_disentangled']}/{len(results)}")

        print(f"\n  Per-pattern:")
        for r in results:
            print(f"    P{r.pattern_id}: d_eff={r.d_eff:.1f} A={r.A:.3f} "
                  f"D={r.D:.3f} K={r.K_comp:.3f} ({r.n_samples} samples)")

        out_path = RESULTS_BASE / 'rep_smoke.json'
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nSaved: {out_path}")

    print(f"\n--- Validation ---")
    all_finite = all(np.isfinite(r.d_eff) and np.isfinite(r.D)
                     for r in results)
    print(f"  All metrics finite: {all_finite}")
    print(f"  Elapsed < 120s:     {elapsed < 120}")
    if all_finite and elapsed < 120 and results:
        print("  SMOKE TEST PASSED")
    else:
        print("  SMOKE TEST FAILED")


def cmd_full(args):
    """Full pipeline: all seeds, all snapshots."""
    from v13_representation import measure_representation_from_snapshot, results_to_dict

    seeds = args.seeds
    all_trajectories = {}

    for seed in seeds:
        snapshots = find_snapshots(seed)
        if not snapshots:
            print(f"No snapshots for seed={seed}")
            continue

        out_dir = RESULTS_BASE / f'rep_s{seed}'
        out_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print(f"REPRESENTATION ANALYSIS: seed={seed}, {len(snapshots)} snapshots")
        print("=" * 60)

        cycle_results = []

        for snap_path in snapshots:
            cycle = get_cycle_number(snap_path)
            print(f"  Cycle {cycle:03d}...", end=" ", flush=True)
            t0 = time.time()

            try:
                results, metadata = measure_representation_from_snapshot(
                    snapshot_path=str(snap_path),
                    seed=seed,
                    n_recording_steps=args.recording_steps,
                    max_patterns=args.max_patterns,
                )
                elapsed = time.time() - t0
                metadata['cycle'] = cycle

                output = results_to_dict(results, metadata)
                output['cycle'] = cycle

                cycle_path = out_dir / f'rep_cycle_{cycle:03d}.json'
                with open(cycle_path, 'w') as f:
                    json.dump(output, f, indent=2, default=str)

                cycle_results.append(output)

                s = output['summary']
                print(f"{len(results)} patterns, d_eff={s['mean_d_eff']:.1f} "
                      f"A={s['mean_A']:.3f} D={s['mean_D']:.3f} "
                      f"K={s['mean_K_comp']:.3f} ({elapsed:.0f}s)")

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()

        # Aggregate
        if cycle_results:
            trajectory = [
                {
                    'cycle': r['cycle'],
                    'mean_d_eff': r['summary']['mean_d_eff'],
                    'mean_A': r['summary']['mean_A'],
                    'mean_D': r['summary']['mean_D'],
                    'mean_K_comp': r['summary']['mean_K_comp'],
                    'n_compressed': r['summary']['n_compressed'],
                    'n_disentangled': r['summary']['n_disentangled'],
                }
                for r in cycle_results
            ]
            all_trajectories[seed] = trajectory

            agg_path = out_dir / 'rep_aggregate.json'
            with open(agg_path, 'w') as f:
                json.dump({
                    'seed': seed,
                    'trajectory': trajectory,
                    'cycles': cycle_results,
                }, f, indent=2, default=str)
            print(f"Aggregate saved: {agg_path}")

    # Cross-seed analysis
    if all_trajectories:
        analysis_dir = RESULTS_BASE / 'rep_analysis'
        analysis_dir.mkdir(parents=True, exist_ok=True)

        with open(analysis_dir / 'rep_cross_seed.json', 'w') as f:
            json.dump({
                'seeds': seeds,
                'trajectories': {str(s): t for s, t in all_trajectories.items()},
            }, f, indent=2, default=str)

        print(f"\n--- Cross-Seed Summary ---")
        for seed, traj in all_trajectories.items():
            if traj:
                first, last = traj[0], traj[-1]
                print(f"  Seed {seed}:")
                print(f"    d_eff:  {first['mean_d_eff']:.1f} → {last['mean_d_eff']:.1f}")
                print(f"    A:      {first['mean_A']:.3f} → {last['mean_A']:.3f}")
                print(f"    D:      {first['mean_D']:.3f} → {last['mean_D']:.3f}")
                print(f"    K_comp: {first['mean_K_comp']:.3f} → {last['mean_K_comp']:.3f}")

        # Figures
        try:
            from v13_representation_figures import plot_all
            plot_all(analysis_dir, all_trajectories, seeds)
        except Exception as e:
            print(f"Figure generation failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='Experiment 3: Internal Representation Structure')
    sub = parser.add_subparsers(dest='command')

    p_smoke = sub.add_parser('smoke')
    p_smoke.add_argument('--seed', type=int, default=123)

    p_full = sub.add_parser('full')
    p_full.add_argument('--seeds', type=int, nargs='+', default=[123, 42, 7])
    p_full.add_argument('--recording-steps', type=int, default=50)
    p_full.add_argument('--max-patterns', type=int, default=20)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    {'smoke': cmd_smoke, 'full': cmd_full}[args.command](args)


if __name__ == '__main__':
    main()
