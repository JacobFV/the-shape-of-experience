"""V13 GPU Run: Content-based coupling evolution with content output.

Saves grid snapshots, Phi trajectories, pattern maps — not just JSON stats.
Everything needed for visualizations, figures, and the book.
"""
import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v13_substrate import generate_v13_config, init_v13, run_v13_chunk
from v11_patterns import detect_patterns_mc, PatternTracker
from v11_affect_hd import measure_all_hd
from v11_evolution import score_fitness_functional
from v11_substrate import perturb_resource_bloom
from v11_substrate_hd import generate_coupling_matrix, init_soup_hd

import jax
import jax.numpy as jnp
from jax import random


def serialize(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64, np.float16)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64, np.int16)):
        return int(obj)
    if hasattr(obj, 'item'):
        return obj.item()
    return str(obj)


def run_v13_gpu(n_cycles=30, C=16, N=128, sim_radius=5, seed=42,
                steps_per_cycle=5000, output_dir='/home/ubuntu/results'):
    """Full V13 evolution with content output."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/snapshots', exist_ok=True)

    print("=" * 70)
    print(f"V13 GPU RUN: C={C}, N={N}, {n_cycles} cycles, seed={seed}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    config = generate_v13_config(C=C, N=N, seed=seed, similarity_radius=sim_radius)
    # Calibrated for C=16, N=128: stable baseline, lethal drought at regen*0.30
    config['maintenance_rate'] = 0.003
    config['resource_consume'] = 0.003
    config['resource_regen'] = 0.01
    grid, resource, h_embed, kernel_ffts, coupling, coupling_row_sums, box_fft = \
        init_v13(config, seed=seed)

    rng = random.PRNGKey(seed)
    rng_np = np.random.RandomState(seed + 1000)

    tau = float(config['tau'])
    gate_beta = float(config['gate_beta'])

    # Stress schedule
    base_schedule = np.linspace(0.50, 0.30, n_cycles)  # calibrated: 0.50=easy, 0.30=lethal
    noise = 1.0 + 0.3 * rng_np.randn(n_cycles)
    stress_schedule = np.clip(base_schedule * noise, 0.01, 0.8)
    duration_schedule = rng_np.randint(500, 2001, size=n_cycles)

    # JIT warmup
    print("JIT compiling...", end=" ", flush=True)
    t0 = time.time()
    grid, resource, rng = run_v13_chunk(
        grid, resource, h_embed, kernel_ffts, config,
        coupling, coupling_row_sums, rng, n_steps=10, box_fft=box_fft)
    grid.block_until_ready()
    print(f"done ({time.time()-t0:.1f}s)")

    # Warmup
    warmup_steps = min(4990, max(500, N * 10))
    print(f"Warmup ({warmup_steps} steps)...", end=" ", flush=True)
    t0 = time.time()
    grid, resource, rng = run_v13_chunk(
        grid, resource, h_embed, kernel_ffts, config,
        coupling, coupling_row_sums, rng, n_steps=warmup_steps, box_fft=box_fft)
    grid.block_until_ready()
    print(f"done ({time.time()-t0:.1f}s)")

    # Save initial snapshot
    g_np = np.array(grid)
    np.savez_compressed(f'{output_dir}/snapshots/init.npz',
                        grid=g_np, resource=np.array(resource))
    pats = detect_patterns_mc(g_np, threshold=0.15)
    print(f"  {len(pats)} patterns after warmup")

    tracker = PatternTracker()
    prev_masses = {}
    prev_values = {}
    cycle_stats = []
    all_phi_trajectories = []  # For visualization

    chunk = 50

    for cycle in range(n_cycles):
        t0 = time.time()

        cycle_regen = float(stress_schedule[cycle]) * config['resource_regen']
        cycle_drought_steps = int(duration_schedule[cycle])
        cycle_baseline_steps = steps_per_cycle - cycle_drought_steps

        stress_config = {**config}
        stress_config['resource_regen'] = cycle_regen
        stress_config['resource_consume'] = config['resource_consume'] * 1.3

        step_base = cycle * steps_per_cycle
        measure_every = max(200, cycle_baseline_steps // 8)

        # ---- BASELINE PHASE ----
        baseline_affects = {}
        baseline_survival = {}

        step = 0
        while step < cycle_baseline_steps:
            grid, resource, rng = run_v13_chunk(
                grid, resource, h_embed, kernel_ffts, config,
                coupling, coupling_row_sums, rng, n_steps=chunk, box_fft=box_fft)
            step += chunk

            if step % measure_every < chunk:
                grid_np = np.array(grid)
                patterns = detect_patterns_mc(grid_np, threshold=0.15)
                tracker.update(patterns, step=step_base + step)

                for p in tracker.active.values():
                    pid = p.id
                    if pid not in baseline_affects:
                        baseline_affects[pid] = []
                    baseline_survival[pid] = step

                    hist = tracker.history.get(pid, [])
                    pm = prev_masses.get(pid)
                    pv = prev_values.get(pid)

                    affect, phi_spec, eff_rank, phi_spat, _ = measure_all_hd(
                        p, pm, pv, hist,
                        jnp.array(grid_np), kernel_ffts, coupling,
                        config, N, step_num=step_base + step,
                        fast=True,
                    )
                    baseline_affects[pid].append(affect)
                    prev_masses[pid] = p.mass
                    prev_values[pid] = p.values.copy()

        # ---- STRESS PHASE ----
        stress_affects = {}
        measure_every_stress = max(100, cycle_drought_steps // 8)

        step = 0
        while step < cycle_drought_steps:
            grid, resource, rng = run_v13_chunk(
                grid, resource, h_embed, kernel_ffts, stress_config,
                coupling, coupling_row_sums, rng, n_steps=chunk,
                box_fft=box_fft)
            step += chunk

            if step % measure_every_stress < chunk:
                grid_np = np.array(grid)
                patterns = detect_patterns_mc(grid_np, threshold=0.15)
                tracker.update(patterns,
                               step=step_base + cycle_baseline_steps + step)

                for p in tracker.active.values():
                    pid = p.id
                    if pid not in stress_affects:
                        stress_affects[pid] = []
                    if pid in baseline_survival:
                        baseline_survival[pid] = cycle_baseline_steps + step

                    hist = tracker.history.get(pid, [])
                    pm = prev_masses.get(pid)
                    pv = prev_values.get(pid)

                    affect, _, _, _, _ = measure_all_hd(
                        p, pm, pv, hist,
                        jnp.array(grid_np), kernel_ffts, coupling,
                        config, N, step_num=step_base + cycle_baseline_steps + step,
                        fast=True,
                    )
                    stress_affects[pid].append(affect)
                    prev_masses[pid] = p.mass
                    prev_values[pid] = p.values.copy()

        # ---- SCORING ----
        grid_np = np.array(grid)
        final_patterns = detect_patterns_mc(grid_np, threshold=0.15)
        tracker.update(final_patterns, step=step_base + steps_per_cycle)

        n_baseline = len(baseline_affects)
        n_survived = len(stress_affects)
        mortality = 1.0 - (n_survived / max(n_baseline, 1))

        fitness_scores = {}
        for pid in set(list(baseline_affects.keys()) + list(stress_affects.keys())):
            b_aff = baseline_affects.get(pid, [])
            s_aff = stress_affects.get(pid, [])
            surv = baseline_survival.get(pid, 0)
            fitness_scores[pid] = score_fitness_functional(
                b_aff, s_aff, surv, steps_per_cycle)

        all_phi_base = []
        all_phi_stress = []
        all_robustness = []
        all_arousal_base = []
        all_arousal_stress = []
        all_effrank_base = []
        all_effrank_stress = []

        for pid in baseline_affects:
            b = baseline_affects[pid]
            s = stress_affects.get(pid, [])
            b_phis = [a.integration for a in b if a.integration > 0]
            s_phis = [a.integration for a in s if a.integration > 0]
            b_arousals = [a.arousal for a in b]
            s_arousals = [a.arousal for a in s]
            b_ranks = [a.effective_rank for a in b]
            s_ranks = [a.effective_rank for a in s]

            if b_phis: all_phi_base.append(np.mean(b_phis))
            if s_phis: all_phi_stress.append(np.mean(s_phis))
            if b_phis and s_phis:
                all_robustness.append(np.mean(s_phis) / np.mean(b_phis))
            if b_arousals: all_arousal_base.append(np.mean(b_arousals))
            if s_arousals: all_arousal_stress.append(np.mean(s_arousals))
            if b_ranks: all_effrank_base.append(np.mean(b_ranks))
            if s_ranks: all_effrank_stress.append(np.mean(s_ranks))

        mean_phi_base = np.mean(all_phi_base) if all_phi_base else 0.0
        mean_phi_stress = np.mean(all_phi_stress) if all_phi_stress else 0.0
        mean_robustness = np.mean(all_robustness) if all_robustness else 0.0
        phi_increase_frac = np.mean([r > 1.0 for r in all_robustness]) if all_robustness else 0.0

        # ---- SELECTION ----
        if fitness_scores:
            sorted_pids = sorted(fitness_scores, key=fitness_scores.get, reverse=True)
            n_cull = int(len(sorted_pids) * 0.3)
            cull_pids = set(sorted_pids[-n_cull:]) if n_cull > 0 else set()

            for pid in cull_pids:
                if pid in tracker.active:
                    p = tracker.active[pid]
                    cells = p.cells
                    if len(cells) > 0:
                        grid = grid.at[:, cells[:, 0], cells[:, 1]].set(0.0)

            for pid in sorted_pids[:5]:
                if pid in tracker.active:
                    p = tracker.active[pid]
                    cells = p.cells
                    if len(cells) > 0:
                        rng, k_boost = random.split(rng)
                        boost = 0.05 * random.uniform(k_boost, grid[:, cells[:, 0], cells[:, 1]].shape)
                        grid = grid.at[:, cells[:, 0], cells[:, 1]].add(boost)
                        grid = jnp.clip(grid, 0.0, 1.0)

            # Mutate params
            rng, k1, k2 = random.split(rng, 3)
            tau = tau + 0.1 * float(random.normal(k1, ()))
            tau = float(np.clip(tau, -2.0, 2.0))
            gate_beta = gate_beta + 0.2 * float(random.normal(k2, ()))
            gate_beta = float(np.clip(gate_beta, 0.5, 10.0))
            config['tau'] = tau
            config['gate_beta'] = gate_beta

        # Resource bloom
        N_grid = config['grid_size']
        rng, k_bloom, k_bloom2 = random.split(rng, 3)
        cx = int(random.randint(k_bloom, (), 10, N_grid - 10))
        cy = int(random.randint(k_bloom2, (), 10, N_grid - 10))
        resource = perturb_resource_bloom(resource, (cy, cx), radius=N_grid // 3, intensity=0.8)

        elapsed = time.time() - t0

        # ---- SAVE STATS ----
        stats = {
            'cycle': cycle,
            'n_patterns': len(final_patterns),
            'n_baseline': n_baseline,
            'n_survived': n_survived,
            'mortality': float(mortality),
            'mean_phi_base': float(mean_phi_base),
            'mean_phi_stress': float(mean_phi_stress),
            'mean_robustness': float(mean_robustness),
            'phi_increase_frac': float(phi_increase_frac),
            'mean_arousal_base': float(np.mean(all_arousal_base)) if all_arousal_base else 0.0,
            'mean_arousal_stress': float(np.mean(all_arousal_stress)) if all_arousal_stress else 0.0,
            'mean_effrank_base': float(np.mean(all_effrank_base)) if all_effrank_base else 0.0,
            'mean_effrank_stress': float(np.mean(all_effrank_stress)) if all_effrank_stress else 0.0,
            'tau': float(tau),
            'gate_beta': float(gate_beta),
            'stress_regen': float(stress_schedule[cycle]),
            'drought_steps': int(cycle_drought_steps),
            'elapsed_s': float(elapsed),
            'resource_mean': float(np.array(resource).mean()),
        }
        cycle_stats.append(stats)

        # Phi trajectory point
        all_phi_trajectories.append({
            'cycle': cycle,
            'phi_base': float(mean_phi_base),
            'phi_stress': float(mean_phi_stress),
            'robustness': float(mean_robustness),
            'n_patterns': len(final_patterns),
            'mortality': float(mortality),
        })

        print(f"Cycle {cycle:3d} | pat={len(final_patterns):4d} | mort={mortality:.0%} | "
              f"Phi_b={mean_phi_base:.3f} Phi_s={mean_phi_stress:.3f} "
              f"rob={mean_robustness:.3f} up={phi_increase_frac:.0%} | "
              f"tau={tau:.2f} beta={gate_beta:.1f} | {elapsed:.0f}s")

        # Save snapshot every 5 cycles
        if cycle % 5 == 0 or cycle == n_cycles - 1:
            g_np = np.array(grid)
            np.savez_compressed(
                f'{output_dir}/snapshots/cycle_{cycle:03d}.npz',
                grid=g_np,
                resource=np.array(resource),
                # Save mean channel map for quick visualization
                mean_channels=g_np.mean(axis=0),
            )

        # Save progress JSON every cycle
        with open(f'{output_dir}/v13_progress.json', 'w') as f:
            json.dump({
                'cycle_stats': cycle_stats,
                'phi_trajectory': all_phi_trajectories,
                'config': {k: v for k, v in config.items()
                          if not isinstance(v, np.ndarray)},
                'seed': seed,
                'status': 'in_progress',
            }, f, indent=2, default=serialize)

    # ---- FINAL SAVE ----
    with open(f'{output_dir}/v13_final.json', 'w') as f:
        json.dump({
            'cycle_stats': cycle_stats,
            'phi_trajectory': all_phi_trajectories,
            'tau': float(tau),
            'gate_beta': float(gate_beta),
            'config': {k: v for k, v in config.items()
                      if not isinstance(v, np.ndarray)},
            'seed': seed,
            'status': 'complete',
        }, f, indent=2, default=serialize)

    # Save final grid
    g_np = np.array(grid)
    np.savez_compressed(f'{output_dir}/snapshots/final.npz',
                        grid=g_np, resource=np.array(resource))

    print()
    print("=" * 70)
    print("V13 EVOLUTION SUMMARY")
    print("=" * 70)
    if cycle_stats:
        first = cycle_stats[0]
        last = cycle_stats[-1]
        print(f"  Patterns:     {first['n_patterns']:4d} -> {last['n_patterns']:4d}")
        print(f"  Mortality:    {first['mortality']:.0%} -> {last['mortality']:.0%}")
        print(f"  Phi (base):   {first['mean_phi_base']:.4f} -> {last['mean_phi_base']:.4f}")
        print(f"  Phi (stress): {first['mean_phi_stress']:.4f} -> {last['mean_phi_stress']:.4f}")
        print(f"  Robustness:   {first['mean_robustness']:.3f} -> {last['mean_robustness']:.3f}")
        print(f"  Tau:          {first['tau']:.2f} -> {last['tau']:.2f}")
        print(f"  Gate beta:    {first['gate_beta']:.1f} -> {last['gate_beta']:.1f}")

    return cycle_stats


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cycles', type=int, default=30)
    parser.add_argument('--channels', type=int, default=16)
    parser.add_argument('--grid', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='/home/ubuntu/results')
    args = parser.parse_args()

    run_v13_gpu(
        n_cycles=args.cycles,
        C=args.channels,
        N=args.grid,
        seed=args.seed,
        output_dir=args.output,
    )
