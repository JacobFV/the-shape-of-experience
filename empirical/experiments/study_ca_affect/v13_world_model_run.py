"""Experiment 2: Emergent World Model — CLI Runner.

Subcommands:
    smoke   — Quick test on one snapshot (CPU, <60s)
    record  — Run recording episodes on all snapshots for a seed
    analyze — Compute prediction gaps from recorded data
    full    — Complete pipeline: record + analyze + figures for all seeds

Usage:
    python v13_world_model_run.py smoke
    python v13_world_model_run.py full --seeds 123 42 7
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

# Snapshot directory layout: results/v13_s{seed}/v13_s{seed}/snapshots/
SEED_DIRS = {
    123: RESULTS_BASE / 'v13_s123' / 'v13_s123',
    42: RESULTS_BASE / 'v13_s42_v2' / 'v13_s42_v2',
    7: RESULTS_BASE / 'v13_s7' / 'v13_s7',
}


def find_snapshots(seed: int) -> list:
    """Find all cycle snapshot files for a seed, sorted by cycle number."""
    seed_dir = SEED_DIRS.get(seed)
    if seed_dir is None or not seed_dir.exists():
        print(f"No results directory for seed={seed}")
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
# Smoke Test
# ============================================================================

def cmd_smoke(args):
    """Quick smoke test on a single snapshot."""
    from v13_world_model import measure_world_model_from_snapshot, results_to_dict

    seed = args.seed
    snapshots = find_snapshots(seed)
    if not snapshots:
        print("No snapshots found. Trying seed=123...")
        seed = 123
        snapshots = find_snapshots(seed)
    if not snapshots:
        print("ERROR: No V13 snapshots available. Run V13 evolution first.")
        return

    # Pick a mid-evolution snapshot
    snap_idx = min(2, len(snapshots) - 1)  # cycle_010 if available
    snap_path = str(snapshots[snap_idx])
    cycle = get_cycle_number(snapshots[snap_idx])

    print("=" * 60)
    print(f"SMOKE TEST: Experiment 2 — Emergent World Model")
    print(f"Snapshot: {snap_path}")
    print(f"Seed: {seed}, Cycle: {cycle}")
    print("=" * 60)

    t0 = time.time()
    results, metadata = measure_world_model_from_snapshot(
        snapshot_path=snap_path,
        seed=seed,
        n_recording_steps=20,          # short for smoke test
        substrate_steps_per_record=10,
        tau_values=(1, 2, 5, 10),      # skip τ=20 for speed
        max_patterns=5,                # only top 5
    )
    elapsed = time.time() - t0

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Patterns analyzed: {len(results)}")

    if results:
        output = results_to_dict(results, metadata)
        print(f"\nSummary:")
        print(f"  Mean C_wm:  {output['summary']['mean_C_wm']:.6f}")
        print(f"  Max C_wm:   {output['summary']['max_C_wm']:.6f}")
        print(f"  Mean H_wm:  {output['summary']['mean_H_wm']:.1f}")
        print(f"  Patterns with world model: {output['summary']['n_with_world_model']}/{len(results)}")
        print(f"\n  W(τ) by horizon:")
        for tau_str, w in sorted(output['summary']['mean_W_by_tau'].items(),
                                  key=lambda x: int(x[0])):
            print(f"    τ={tau_str:>2s}: W = {w:.6f}")

        print(f"\n  Per-pattern details:")
        for r in results:
            w_str = ", ".join(f"τ={t}:{r.W.get(t,0):.4f}"
                              for t in sorted(r.W.keys()))
            print(f"    Pattern {r.pattern_id}: C_wm={r.C_wm:.4f} "
                  f"H_wm={r.H_wm:.0f} ({r.n_samples} samples) [{w_str}]")

        # Save smoke test results
        out_path = RESULTS_BASE / 'wm_smoke.json'
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nSaved: {out_path}")
    else:
        print("No patterns had enough data for analysis.")

    # Validation checks
    print(f"\n--- Validation ---")
    all_finite = all(
        np.isfinite(v) for r in results
        for v in r.W.values() if not np.isnan(v))
    print(f"  All W(τ) finite: {all_finite}")
    print(f"  Elapsed < 120s:  {elapsed < 120}")
    if all_finite and elapsed < 120:
        print("  SMOKE TEST PASSED")
    else:
        print("  SMOKE TEST FAILED")


# ============================================================================
# Record
# ============================================================================

def cmd_record(args):
    """Run recording episodes on all snapshots for given seeds."""
    from v13_world_model import measure_world_model_from_snapshot, results_to_dict

    seeds = args.seeds
    tau_values = tuple(args.tau)
    n_rec = args.recording_steps
    sub_steps = args.substrate_steps

    for seed in seeds:
        snapshots = find_snapshots(seed)
        if not snapshots:
            continue

        out_dir = RESULTS_BASE / f'wm_s{seed}'
        out_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print(f"RECORDING: seed={seed}, {len(snapshots)} snapshots")
        print(f"  τ values: {tau_values}")
        print(f"  Recording steps: {n_rec}, Substrate steps/record: {sub_steps}")
        print(f"  Output: {out_dir}")
        print("=" * 60)

        all_results = []

        for snap_path in snapshots:
            cycle = get_cycle_number(snap_path)
            print(f"\n  Cycle {cycle:03d}...", end=" ", flush=True)
            t0 = time.time()

            try:
                results, metadata = measure_world_model_from_snapshot(
                    snapshot_path=str(snap_path),
                    seed=seed,
                    n_recording_steps=n_rec,
                    substrate_steps_per_record=sub_steps,
                    tau_values=tau_values,
                    max_patterns=args.max_patterns,
                )
                elapsed = time.time() - t0
                metadata['cycle'] = cycle
                metadata['elapsed_s'] = elapsed

                output = results_to_dict(results, metadata)
                output['cycle'] = cycle

                # Save per-cycle results
                cycle_path = out_dir / f'wm_cycle_{cycle:03d}.json'
                with open(cycle_path, 'w') as f:
                    json.dump(output, f, indent=2, default=str)

                all_results.append(output)

                n_wm = output['summary']['n_with_world_model']
                mean_c = output['summary']['mean_C_wm']
                print(f"{len(results)} patterns, {n_wm} with WM, "
                      f"C_wm={mean_c:.4f} ({elapsed:.0f}s)")

            except Exception as e:
                elapsed = time.time() - t0
                print(f"ERROR: {e} ({elapsed:.0f}s)")
                import traceback
                traceback.print_exc()

        # Save aggregate
        if all_results:
            agg_path = out_dir / 'wm_aggregate.json'
            with open(agg_path, 'w') as f:
                json.dump({
                    'seed': seed,
                    'n_cycles': len(all_results),
                    'cycles': all_results,
                    'trajectory': [
                        {
                            'cycle': r['cycle'],
                            'mean_C_wm': r['summary']['mean_C_wm'],
                            'max_C_wm': r['summary']['max_C_wm'],
                            'mean_H_wm': r['summary']['mean_H_wm'],
                            'n_with_wm': r['summary']['n_with_world_model'],
                            'frac_with_wm': r['summary']['frac_with_world_model'],
                        }
                        for r in all_results
                    ],
                }, f, indent=2, default=str)
            print(f"\nAggregate saved: {agg_path}")


# ============================================================================
# Analyze
# ============================================================================

def cmd_analyze(args):
    """Analyze recorded world model data across seeds."""
    seeds = args.seeds

    print("=" * 60)
    print("ANALYSIS: Emergent World Model")
    print("=" * 60)

    all_trajectories = {}

    for seed in seeds:
        agg_path = RESULTS_BASE / f'wm_s{seed}' / 'wm_aggregate.json'
        if not agg_path.exists():
            print(f"  No data for seed={seed}. Run 'record' first.")
            continue

        with open(agg_path) as f:
            data = json.load(f)

        traj = data['trajectory']
        all_trajectories[seed] = traj

        print(f"\n  Seed {seed}: {len(traj)} cycles")
        if traj:
            first = traj[0]
            last = traj[-1]
            print(f"    C_wm:  {first['mean_C_wm']:.4f} → {last['mean_C_wm']:.4f}")
            print(f"    H_wm:  {first['mean_H_wm']:.1f} → {last['mean_H_wm']:.1f}")
            print(f"    %WM:   {first['frac_with_wm']:.0%} → {last['frac_with_wm']:.0%}")

    if not all_trajectories:
        print("\nNo data to analyze.")
        return

    # Cross-seed summary
    print(f"\n--- Cross-Seed Summary ---")
    for seed, traj in all_trajectories.items():
        c_wm_vals = [t['mean_C_wm'] for t in traj]
        print(f"  Seed {seed}: C_wm range [{min(c_wm_vals):.4f}, "
              f"{max(c_wm_vals):.4f}], mean={np.mean(c_wm_vals):.4f}")

    # Save cross-seed analysis
    analysis_dir = RESULTS_BASE / 'wm_analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)

    analysis = {
        'seeds': seeds,
        'trajectories': {str(s): t for s, t in all_trajectories.items()},
    }

    with open(analysis_dir / 'wm_cross_seed.json', 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\nCross-seed analysis saved: {analysis_dir / 'wm_cross_seed.json'}")

    # Generate figures
    try:
        from v13_world_model_figures import plot_all
        plot_all(analysis_dir, all_trajectories, seeds)
        print("Figures generated.")
    except Exception as e:
        print(f"Figure generation failed: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# Full Pipeline
# ============================================================================

def cmd_full(args):
    """Full pipeline: record + analyze + figures."""
    print("FULL PIPELINE: record → analyze → figures\n")

    # Record
    cmd_record(args)

    # Analyze
    cmd_analyze(args)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Experiment 2: Emergent World Model')
    sub = parser.add_subparsers(dest='command')

    # smoke
    p_smoke = sub.add_parser('smoke', help='Quick smoke test')
    p_smoke.add_argument('--seed', type=int, default=123)

    # record
    p_record = sub.add_parser('record', help='Run recording episodes')
    p_record.add_argument('--seeds', type=int, nargs='+', default=[123, 42, 7])
    p_record.add_argument('--recording-steps', type=int, default=50)
    p_record.add_argument('--substrate-steps', type=int, default=10)
    p_record.add_argument('--tau', type=int, nargs='+', default=[1, 2, 5, 10, 20])
    p_record.add_argument('--max-patterns', type=int, default=20)

    # analyze
    p_analyze = sub.add_parser('analyze', help='Analyze recorded data')
    p_analyze.add_argument('--seeds', type=int, nargs='+', default=[123, 42, 7])

    # full
    p_full = sub.add_parser('full', help='Full pipeline')
    p_full.add_argument('--seeds', type=int, nargs='+', default=[123, 42, 7])
    p_full.add_argument('--recording-steps', type=int, default=50)
    p_full.add_argument('--substrate-steps', type=int, default=10)
    p_full.add_argument('--tau', type=int, nargs='+', default=[1, 2, 5, 10, 20])
    p_full.add_argument('--max-patterns', type=int, default=20)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    cmds = {
        'smoke': cmd_smoke,
        'record': cmd_record,
        'analyze': cmd_analyze,
        'full': cmd_full,
    }
    cmds[args.command](args)


if __name__ == '__main__':
    main()
