"""V18 GPU Run: Boundary-Dependent Lenia evolution on Lambda Labs.

Insulation field creates genuine sensory-motor boundaries. Tracks insulation
evolution (boundary_width, internal_gain), all V15 metrics (memory, foraging,
Phi, robustness), and the cognitive dictionary progress.

Usage:
    python v18_gpu_run.py smoke                    # Quick CPU test
    python v18_gpu_run.py run --seed 42            # Single seed
    python v18_gpu_run.py all                      # All 3 seeds
    python v18_gpu_run.py run --seed 42 --output /path/to/results
"""
import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v18_substrate import (
    generate_v18_config, init_v18, run_v18_chunk,
    compute_insulation_metrics,
)
from v18_evolution import mutate_v18_params
from v14_evolution import compute_foraging_metrics
from v13_substrate import make_box_kernel_fft
from v15_substrate import (
    generate_resource_patches, compute_patch_regen_mask,
)
from v11_patterns import detect_patterns_mc, PatternTracker
from v11_affect_hd import measure_all_hd
from v11_evolution import score_fitness_functional
from v11_substrate import perturb_resource_bloom
from v11_substrate_hd import init_soup_hd

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


def run_v18_gpu(n_cycles=30, C=16, N=128, sim_radius=5, seed=42,
                steps_per_cycle=5000, chemotaxis_strength=0.5,
                memory_channels=2, n_resource_patches=4,
                patch_shift_period=500,
                output_dir='/home/ubuntu/results'):
    """Full V18 evolution with insulation tracking."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/snapshots', exist_ok=True)

    print("=" * 70)
    print(f"V18 GPU RUN: C={C}, N={N}, {n_cycles} cycles, seed={seed}")
    print(f"  Memory: {memory_channels} channels")
    print(f"  Insulation: boundary-dependent dual-signal")
    print(f"  Chemotaxis: eta={chemotaxis_strength}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    config = generate_v18_config(
        C=C, N=N, seed=seed,
        similarity_radius=sim_radius,
        chemotaxis_strength=chemotaxis_strength,
        memory_channels=memory_channels,
        n_resource_patches=n_resource_patches,
        patch_shift_period=patch_shift_period,
    )
    config['maintenance_rate'] = 0.003
    config['resource_consume'] = 0.003
    config['resource_regen'] = 0.01

    (grid, resource, h_embed, kernel_ffts, internal_kernel_ffts,
     coupling, coupling_row_sums, recurrence_coupling, recurrence_crs,
     box_fft) = init_v18(config, seed=seed)

    rng = random.PRNGKey(seed)
    rng_np = np.random.RandomState(seed + 1000)

    # Evolvable params (V15)
    tau = float(config['tau'])
    gate_beta = float(config['gate_beta'])
    eta = float(config['chemotaxis_strength'])
    motor_sensitivity = float(config['motor_sensitivity'])
    motor_threshold = float(config['motor_threshold'])
    memory_lambdas = list(config['memory_lambdas'])

    # Evolvable params (V18)
    boundary_width = float(config['boundary_width'])
    insulation_beta = float(config['insulation_beta'])
    internal_gain = float(config['internal_gain'])
    activity_threshold = float(config['activity_threshold'])

    patches = generate_resource_patches(N, n_resource_patches, seed=seed)

    # Stress schedule
    base_schedule = np.linspace(0.60, 0.40, n_cycles)
    noise = 1.0 + 0.1 * rng_np.randn(n_cycles)
    stress_schedule = np.clip(base_schedule * noise, 0.25, 0.8)
    duration_schedule = rng_np.randint(500, 1501, size=n_cycles)

    chunk = 50
    total_step = 0

    def _make_run_config():
        cfg = {**config}
        cfg['tau'] = tau
        cfg['gate_beta'] = gate_beta
        cfg['chemotaxis_strength'] = eta
        cfg['motor_sensitivity'] = motor_sensitivity
        cfg['motor_threshold'] = motor_threshold
        cfg['memory_lambdas'] = memory_lambdas
        cfg['boundary_width'] = boundary_width
        cfg['insulation_beta'] = insulation_beta
        cfg['internal_gain'] = internal_gain
        cfg['activity_threshold'] = activity_threshold
        return cfg

    regen_mask = compute_patch_regen_mask(
        N, patches, step=0, shift_period=patch_shift_period)
    regen_mask = jnp.array(regen_mask)

    # JIT warmup
    run_config = _make_run_config()
    print("JIT compiling...", end=" ", flush=True)
    t0 = time.time()
    grid, resource, rng = run_v18_chunk(
        grid, resource, h_embed, kernel_ffts, internal_kernel_ffts,
        run_config, coupling, coupling_row_sums,
        recurrence_coupling, recurrence_crs,
        rng, n_steps=chunk, box_fft=box_fft, regen_mask=regen_mask)
    jnp.array(grid).block_until_ready()
    print(f"done ({time.time()-t0:.1f}s)")

    # Warmup
    warmup_steps = min(4990, max(500, N * 10))
    warmup_chunks = warmup_steps // chunk
    print(f"Warmup ({warmup_chunks * chunk} steps)...", end=" ", flush=True)
    t0 = time.time()
    for _ in range(warmup_chunks):
        grid, resource, rng = run_v18_chunk(
            grid, resource, h_embed, kernel_ffts, internal_kernel_ffts,
            run_config, coupling, coupling_row_sums,
            recurrence_coupling, recurrence_crs,
            rng, n_steps=chunk, box_fft=box_fft, regen_mask=regen_mask)
    jnp.array(grid).block_until_ready()
    total_step += warmup_chunks * chunk
    print(f"done ({time.time()-t0:.1f}s)")

    # Save initial snapshot
    g_np = np.array(grid)
    np.savez_compressed(f'{output_dir}/snapshots/init.npz',
                        grid=g_np, resource=np.array(resource),
                        regen_mask=np.array(regen_mask))
    pats = detect_patterns_mc(g_np, threshold=0.15)
    ins_metrics = compute_insulation_metrics(grid, run_config)
    print(f"  {len(pats)} patterns after warmup")
    print(f"  Insulation: mean={ins_metrics['insulation_mean']:.4f}, "
          f"interior={ins_metrics['interior_fraction']:.1%}")

    tracker = PatternTracker()
    prev_masses = {}
    prev_values = {}
    cycle_stats = []
    all_phi_trajectories = []

    for cycle in range(n_cycles):
        t0 = time.time()
        run_config = _make_run_config()

        cycle_regen = float(stress_schedule[cycle]) * config['resource_regen']
        cycle_drought_steps = int(duration_schedule[cycle])
        cycle_baseline_steps = steps_per_cycle - cycle_drought_steps

        stress_config = {**run_config}
        stress_config['resource_regen'] = cycle_regen
        stress_config['resource_consume'] = config['resource_consume'] * 1.3

        step_base = cycle * steps_per_cycle
        measure_every = max(500, cycle_baseline_steps // 4)
        max_measure_patterns = 30  # Cap to control measurement overhead

        # ---- BASELINE ----
        baseline_affects = {}
        baseline_survival = {}
        patterns_at_start = None

        step = 0
        while step < cycle_baseline_steps:
            regen_mask = compute_patch_regen_mask(
                N, patches, step=total_step,
                shift_period=patch_shift_period)
            regen_mask = jnp.array(regen_mask)

            grid, resource, rng = run_v18_chunk(
                grid, resource, h_embed, kernel_ffts, internal_kernel_ffts,
                run_config, coupling, coupling_row_sums,
                recurrence_coupling, recurrence_crs,
                rng, n_steps=chunk, box_fft=box_fft, regen_mask=regen_mask)
            step += chunk
            total_step += chunk

            if step % measure_every < chunk:
                grid_np = np.array(grid)
                patterns = detect_patterns_mc(grid_np, threshold=0.15)
                tracker.update(patterns, step=step_base + step)

                if patterns_at_start is None:
                    patterns_at_start = patterns

                # Sort by mass, measure top N to control overhead
                active_sorted = sorted(
                    tracker.active.values(),
                    key=lambda p: p.mass, reverse=True)
                for p in active_sorted[:max_measure_patterns]:
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
                        run_config, N, step_num=step_base + step,
                        fast=True)
                    baseline_affects[pid].append(affect)
                    prev_masses[pid] = p.mass
                    prev_values[pid] = p.values.copy()
                # Track survival for all (cheap)
                for p in tracker.active.values():
                    baseline_survival[p.id] = step

        # Foraging + memory + insulation metrics
        grid_np = np.array(grid)
        patterns_at_end = detect_patterns_mc(grid_np, threshold=0.15)
        foraging = compute_foraging_metrics(
            patterns_at_start, patterns_at_end, resource)

        mem_start = config['memory_start']
        mem_end = mem_start + config['memory_channels']
        memory_mean = float(np.array(grid[mem_start:mem_end]).mean())
        memory_std = float(np.array(grid[mem_start:mem_end]).std())

        ins_metrics = compute_insulation_metrics(grid, run_config)

        # ---- STRESS ----
        stress_affects = {}
        measure_every_stress = max(250, cycle_drought_steps // 4)

        step = 0
        while step < cycle_drought_steps:
            regen_mask = compute_patch_regen_mask(
                N, patches, step=total_step,
                shift_period=patch_shift_period)
            regen_mask = jnp.array(regen_mask)

            grid, resource, rng = run_v18_chunk(
                grid, resource, h_embed, kernel_ffts, internal_kernel_ffts,
                stress_config, coupling, coupling_row_sums,
                recurrence_coupling, recurrence_crs,
                rng, n_steps=chunk, drought=True,
                box_fft=box_fft, regen_mask=regen_mask)
            step += chunk
            total_step += chunk

            if step % measure_every_stress < chunk:
                grid_np = np.array(grid)
                patterns = detect_patterns_mc(grid_np, threshold=0.15)
                tracker.update(patterns,
                               step=step_base + cycle_baseline_steps + step)

                active_sorted = sorted(
                    tracker.active.values(),
                    key=lambda p: p.mass, reverse=True)
                for p in active_sorted[:max_measure_patterns]:
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
                        run_config, N,
                        step_num=step_base + cycle_baseline_steps + step,
                        fast=True)
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
        for pid in set(list(baseline_affects.keys())
                       + list(stress_affects.keys())):
            b_aff = baseline_affects.get(pid, [])
            s_aff = stress_affects.get(pid, [])
            surv = baseline_survival.get(pid, 0)
            fitness_scores[pid] = score_fitness_functional(
                b_aff, s_aff, surv, steps_per_cycle)

        all_phi_base = []
        all_phi_stress = []
        all_robustness = []
        for pid in baseline_affects:
            b = baseline_affects[pid]
            s = stress_affects.get(pid, [])
            b_phis = [a.integration for a in b if a.integration > 0]
            s_phis = [a.integration for a in s if a.integration > 0]
            if b_phis:
                all_phi_base.append(np.mean(b_phis))
            if s_phis:
                all_phi_stress.append(np.mean(s_phis))
            if b_phis and s_phis:
                all_robustness.append(np.mean(s_phis) / np.mean(b_phis))

        mean_phi_base = np.mean(all_phi_base) if all_phi_base else 0.0
        mean_phi_stress = np.mean(all_phi_stress) if all_phi_stress else 0.0
        mean_robustness = np.mean(all_robustness) if all_robustness else 0.0
        phi_increase_frac = (np.mean([r > 1.0 for r in all_robustness])
                             if all_robustness else 0.0)

        # ---- SELECTION ----
        if fitness_scores:
            sorted_pids = sorted(fitness_scores, key=fitness_scores.get,
                                 reverse=True)
            n_cull = int(len(sorted_pids) * 0.3)
            cull_pids = set(sorted_pids[-n_cull:]) if n_cull > 0 else set()

            for pid in cull_pids:
                if pid in tracker.active:
                    p = tracker.active[pid]
                    cells = p.cells
                    if len(cells) > 0:
                        grid = grid.at[
                            :, cells[:, 0], cells[:, 1]].set(0.0)

            for pid in sorted_pids[:5]:
                if pid in tracker.active:
                    p = tracker.active[pid]
                    cells = p.cells
                    if len(cells) > 0:
                        rng, k_boost = random.split(rng)
                        boost = 0.05 * random.uniform(
                            k_boost,
                            grid[:, cells[:, 0], cells[:, 1]].shape)
                        grid = grid.at[
                            :, cells[:, 0], cells[:, 1]].add(boost)
                        grid = jnp.clip(grid, 0.0, 1.0)

            # Mutate params
            rng, k_mut = random.split(rng)
            (tau, gate_beta, coupling,
             eta, motor_sensitivity, motor_threshold,
             memory_lambdas,
             boundary_width, insulation_beta,
             internal_gain, activity_threshold,
             recurrence_coupling) = mutate_v18_params(
                tau, gate_beta, coupling,
                eta, motor_sensitivity, motor_threshold,
                memory_lambdas,
                boundary_width, insulation_beta,
                internal_gain, activity_threshold,
                recurrence_coupling, k_mut)
            coupling_row_sums = coupling.sum(axis=1)
            recurrence_crs = recurrence_coupling.sum(axis=1)

        # Resource bloom
        N_grid = config['grid_size']
        rng, k_bloom, k_bloom2 = random.split(rng, 3)
        cx = int(random.randint(k_bloom, (), 10, N_grid - 10))
        cy = int(random.randint(k_bloom2, (), 10, N_grid - 10))
        resource = perturb_resource_bloom(
            resource, (cy, cx), radius=N_grid // 3, intensity=0.8)

        # Population rescue
        if len(final_patterns) < 10:
            print(f"  [RESCUE] {len(final_patterns)} patterns — re-seeding")
            rng, k_reseed = random.split(rng)
            channel_mus = jnp.array(config['channel_mus'])
            fresh_grid, fresh_resource = init_soup_hd(
                N_grid, C, k_reseed, channel_mus)
            grid = 0.3 * grid + 0.7 * fresh_grid
            resource = jnp.maximum(resource, 0.5 * fresh_resource)
            for _ in range(500 // chunk):
                grid, resource, rng = run_v18_chunk(
                    grid, resource, h_embed, kernel_ffts,
                    internal_kernel_ffts,
                    run_config, coupling, coupling_row_sums,
                    recurrence_coupling, recurrence_crs,
                    rng, n_steps=chunk,
                    box_fft=box_fft, regen_mask=regen_mask)
            jnp.array(grid).block_until_ready()

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
            'tau': float(tau),
            'gate_beta': float(gate_beta),
            'chemotaxis_strength': float(eta),
            'motor_sensitivity': float(motor_sensitivity),
            'motor_threshold': float(motor_threshold),
            'memory_lambdas': [float(l) for l in memory_lambdas],
            'memory_mean': memory_mean,
            'memory_std': memory_std,
            # V18-specific
            'boundary_width': float(boundary_width),
            'insulation_beta': float(insulation_beta),
            'internal_gain': float(internal_gain),
            'activity_threshold': float(activity_threshold),
            **ins_metrics,
            'stress_regen': float(stress_schedule[cycle]),
            'drought_steps': int(cycle_drought_steps),
            'elapsed_s': float(elapsed),
            'resource_mean': float(np.array(resource).mean()),
            'foraging': foraging,
        }
        cycle_stats.append(stats)

        all_phi_trajectories.append({
            'cycle': cycle,
            'phi_base': float(mean_phi_base),
            'phi_stress': float(mean_phi_stress),
            'robustness': float(mean_robustness),
            'n_patterns': len(final_patterns),
            'mortality': float(mortality),
            'displacement': foraging['mean_displacement'],
            'resource_at_patterns': foraging['resource_at_patterns'],
            'memory_lambdas': [float(l) for l in memory_lambdas],
            'boundary_width': float(boundary_width),
            'insulation_beta': float(insulation_beta),
            'internal_gain': float(internal_gain),
            'activity_threshold': float(activity_threshold),
            'insulation_mean': ins_metrics['insulation_mean'],
            'interior_fraction': ins_metrics['interior_fraction'],
        })

        ins_m = ins_metrics.get('insulation_mean', 0)
        ins_i = ins_metrics.get('interior_fraction', 0)
        print(f"C{cycle:02d} | pat={len(final_patterns):4d} | "
              f"mort={mortality:.0%} | "
              f"Phi_b={mean_phi_base:.3f} Phi_s={mean_phi_stress:.3f} "
              f"rob={mean_robustness:.3f} up={phi_increase_frac:.0%} | "
              f"ins={ins_m:.3f} int={ins_i:.1%} "
              f"gain={internal_gain:.2f} bw={boundary_width:.1f} | "
              f"eta={eta:.2f} "
              f"lam=[{','.join(f'{l:.3f}' for l in memory_lambdas)}] | "
              f"{elapsed:.0f}s", flush=True)

        # Save snapshot every 5 cycles
        if cycle % 5 == 0 or cycle == n_cycles - 1:
            g_np = np.array(grid)
            np.savez_compressed(
                f'{output_dir}/snapshots/cycle_{cycle:03d}.npz',
                grid=g_np,
                resource=np.array(resource),
                mean_channels=g_np.mean(axis=0),
                regen_mask=np.array(regen_mask),
            )

        # Save progress JSON
        with open(f'{output_dir}/v18_progress.json', 'w') as f:
            json.dump({
                'cycle_stats': cycle_stats,
                'phi_trajectory': all_phi_trajectories,
                'config': {k: v for k, v in config.items()
                          if not isinstance(v, np.ndarray)},
                'seed': seed,
                'status': 'in_progress',
            }, f, indent=2, default=serialize)

    # ---- FINAL SAVE ----
    with open(f'{output_dir}/v18_final.json', 'w') as f:
        json.dump({
            'cycle_stats': cycle_stats,
            'phi_trajectory': all_phi_trajectories,
            'tau': float(tau),
            'gate_beta': float(gate_beta),
            'chemotaxis_strength': float(eta),
            'motor_sensitivity': float(motor_sensitivity),
            'motor_threshold': float(motor_threshold),
            'memory_lambdas': [float(l) for l in memory_lambdas],
            'boundary_width': float(boundary_width),
            'insulation_beta': float(insulation_beta),
            'internal_gain': float(internal_gain),
            'activity_threshold': float(activity_threshold),
            'config': {k: v for k, v in config.items()
                      if not isinstance(v, np.ndarray)},
            'seed': seed,
            'status': 'complete',
        }, f, indent=2, default=serialize)

    g_np = np.array(grid)
    np.savez_compressed(f'{output_dir}/snapshots/final.npz',
                        grid=g_np, resource=np.array(resource))

    print()
    print("=" * 70)
    print("V18 EVOLUTION SUMMARY")
    print("=" * 70)
    if cycle_stats:
        first = cycle_stats[0]
        last = cycle_stats[-1]
        print(f"  Patterns:      {first['n_patterns']:4d} -> "
              f"{last['n_patterns']:4d}")
        print(f"  Mortality:     {first['mortality']:.0%} -> "
              f"{last['mortality']:.0%}")
        print(f"  Phi (base):    {first['mean_phi_base']:.4f} -> "
              f"{last['mean_phi_base']:.4f}")
        print(f"  Phi (stress):  {first['mean_phi_stress']:.4f} -> "
              f"{last['mean_phi_stress']:.4f}")
        print(f"  Robustness:    {first['mean_robustness']:.3f} -> "
              f"{last['mean_robustness']:.3f}")
        print(f"  Internal gain: {first['internal_gain']:.3f} -> "
              f"{last['internal_gain']:.3f}")
        print(f"  Boundary w:    {first['boundary_width']:.2f} -> "
              f"{last['boundary_width']:.2f}")
        print(f"  Insulation:    {first['insulation_mean']:.4f} -> "
              f"{last['insulation_mean']:.4f}")
        print(f"  Interior %:    {first['interior_fraction']:.1%} -> "
              f"{last['interior_fraction']:.1%}")
        print(f"  Memory lam:    {first['memory_lambdas']} -> "
              f"{last['memory_lambdas']}")

    return cycle_stats


def run_smoke():
    """Quick smoke test on CPU (C=8, N=64, 2 cycles)."""
    print("V18 SMOKE TEST (C=8, N=64, 2 cycles)")
    print("=" * 70)
    stats = run_v18_gpu(
        n_cycles=2, C=8, N=64, sim_radius=5, seed=42,
        steps_per_cycle=500,
        output_dir='/tmp/v18_smoke')
    print("\nSmoke test complete!")
    return stats


def run_all_seeds(output_base='/home/ubuntu/results', n_cycles=30):
    """Run all 3 seeds sequentially."""
    for seed in [42, 123, 7]:
        output_dir = f'{output_base}/v18_s{seed}'
        print(f"\n{'='*70}")
        print(f"STARTING SEED {seed}")
        print(f"{'='*70}\n")
        run_v18_gpu(
            n_cycles=n_cycles, seed=seed,
            output_dir=output_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('command', nargs='?', default='run',
                        choices=['smoke', 'run', 'all'])
    parser.add_argument('--cycles', type=int, default=30)
    parser.add_argument('--channels', type=int, default=16)
    parser.add_argument('--grid', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eta', type=float, default=0.5)
    parser.add_argument('--mem-channels', type=int, default=2)
    parser.add_argument('--patches', type=int, default=4)
    parser.add_argument('--output', default='/home/ubuntu/results')
    args = parser.parse_args()

    if args.command == 'smoke':
        run_smoke()
    elif args.command == 'all':
        run_all_seeds(output_base=args.output, n_cycles=args.cycles)
    else:
        run_v18_gpu(
            n_cycles=args.cycles,
            C=args.channels,
            N=args.grid,
            seed=args.seed,
            chemotaxis_strength=args.eta,
            memory_channels=args.mem_channels,
            n_resource_patches=args.patches,
            output_dir=args.output,
        )
