"""V16 GPU Run: Plastic Lenia evolution on Lambda Labs.

Extends V15 with within-lifetime learning (Hebbian coupling plasticity).
Tracks coupling divergence, spatial variance, learning rate evolution.

Usage:
    python v16_gpu_run.py --seed 42 --cycles 30 --output /home/ubuntu/results
"""
import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v16_substrate import (
    generate_v16_config, init_v16, run_v16_chunk, init_coupling_field,
)
from v16_evolution import mutate_v16_params
from v14_evolution import compute_foraging_metrics
from v15_substrate import (
    generate_resource_patches, compute_patch_regen_mask,
)
from v13_substrate import make_box_kernel_fft
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


def run_v16_gpu(n_cycles=30, C=16, N=128, sim_radius=5, seed=42,
                steps_per_cycle=5000, chemotaxis_strength=0.5,
                memory_channels=2, n_resource_patches=4,
                patch_shift_period=500,
                learning_rate=0.001, decay_rate_coupling=0.0001,
                output_dir='/home/ubuntu/results'):
    """Full V16 evolution with content output."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/snapshots', exist_ok=True)

    print("=" * 70)
    print(f"V16 GPU RUN: C={C}, N={N}, {n_cycles} cycles, seed={seed}")
    print(f"  Memory: {memory_channels} channels")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Coupling decay: {decay_rate_coupling}")
    print(f"  Chemotaxis: eta={chemotaxis_strength}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    config = generate_v16_config(
        C=C, N=N, seed=seed,
        similarity_radius=sim_radius,
        chemotaxis_strength=chemotaxis_strength,
        memory_channels=memory_channels,
        n_resource_patches=n_resource_patches,
        patch_shift_period=patch_shift_period,
        learning_rate=learning_rate,
        decay_rate_coupling=decay_rate_coupling,
    )
    config['maintenance_rate'] = 0.003
    config['resource_consume'] = 0.003
    config['resource_regen'] = 0.01

    grid, resource, h_embed, kernel_ffts, coupling, coupling_row_sums, box_fft, coupling_field = \
        init_v16(config, seed=seed)

    rng = random.PRNGKey(seed)
    rng_np = np.random.RandomState(seed + 1000)

    # Evolvable params
    tau = float(config['tau'])
    gate_beta = float(config['gate_beta'])
    eta = float(config['chemotaxis_strength'])
    motor_sensitivity = float(config['motor_sensitivity'])
    motor_threshold = float(config['motor_threshold'])
    memory_lambdas = list(config['memory_lambdas'])
    lr = float(config['learning_rate'])
    dr = float(config['decay_rate_coupling'])

    patches = generate_resource_patches(N, n_resource_patches, seed=seed)

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
        cfg['learning_rate'] = lr
        cfg['decay_rate_coupling'] = dr
        return cfg

    regen_mask = compute_patch_regen_mask(
        N, patches, step=0, shift_period=patch_shift_period)
    regen_mask = jnp.array(regen_mask)

    run_config = _make_run_config()
    print("JIT compiling...", end=" ", flush=True)
    t0 = time.time()
    grid, resource, coupling_field, rng = run_v16_chunk(
        grid, resource, h_embed, kernel_ffts, run_config,
        coupling_field, rng, n_steps=chunk,
        box_fft=box_fft, regen_mask=regen_mask)
    grid.block_until_ready()
    print(f"done ({time.time()-t0:.1f}s)")

    warmup_steps = min(4990, max(500, N * 10))
    warmup_chunks = warmup_steps // chunk
    print(f"Warmup ({warmup_chunks * chunk} steps)...", end=" ", flush=True)
    t0 = time.time()
    for _ in range(warmup_chunks):
        grid, resource, coupling_field, rng = run_v16_chunk(
            grid, resource, h_embed, kernel_ffts, run_config,
            coupling_field, rng, n_steps=chunk,
            box_fft=box_fft, regen_mask=regen_mask)
    grid.block_until_ready()
    total_step += warmup_chunks * chunk
    print(f"done ({time.time()-t0:.1f}s)")

    g_np = np.array(grid)
    np.savez_compressed(f'{output_dir}/snapshots/init.npz',
                        grid=g_np, resource=np.array(resource))
    pats = detect_patterns_mc(g_np, threshold=0.15)
    print(f"  {len(pats)} patterns after warmup")

    coupling_field_init = coupling_field.copy()

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
        measure_every = max(200, cycle_baseline_steps // 8)

        baseline_affects = {}
        baseline_survival = {}
        patterns_at_start = None

        step = 0
        while step < cycle_baseline_steps:
            regen_mask = compute_patch_regen_mask(
                N, patches, step=total_step,
                shift_period=patch_shift_period)
            regen_mask = jnp.array(regen_mask)

            grid, resource, coupling_field, rng = run_v16_chunk(
                grid, resource, h_embed, kernel_ffts, run_config,
                coupling_field, rng, n_steps=chunk,
                box_fft=box_fft, regen_mask=regen_mask)
            step += chunk
            total_step += chunk

            if step % measure_every < chunk:
                grid_np = np.array(grid)
                patterns = detect_patterns_mc(grid_np, threshold=0.15)
                tracker.update(patterns, step=step_base + step)

                if patterns_at_start is None:
                    patterns_at_start = patterns

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
                        run_config, N, step_num=step_base + step,
                        fast=True)
                    baseline_affects[pid].append(affect)
                    prev_masses[pid] = p.mass
                    prev_values[pid] = p.values.copy()

        grid_np = np.array(grid)
        patterns_at_end = detect_patterns_mc(grid_np, threshold=0.15)
        foraging = compute_foraging_metrics(
            patterns_at_start, patterns_at_end, resource)

        mem_start = config['memory_start']
        mem_end = mem_start + config['memory_channels']
        memory_mean = float(np.array(grid[mem_start:mem_end]).mean())

        cf_np = np.array(coupling_field)
        coupling_divergence = float(np.abs(cf_np - np.array(coupling_field_init)).mean())
        coupling_spatial_var = float(cf_np.var(axis=(2, 3)).mean())

        stress_affects = {}
        measure_every_stress = max(100, cycle_drought_steps // 8)

        step = 0
        while step < cycle_drought_steps:
            regen_mask = compute_patch_regen_mask(
                N, patches, step=total_step,
                shift_period=patch_shift_period)
            regen_mask = jnp.array(regen_mask)

            grid, resource, coupling_field, rng = run_v16_chunk(
                grid, resource, h_embed, kernel_ffts, stress_config,
                coupling_field, rng, n_steps=chunk,
                drought=True, box_fft=box_fft, regen_mask=regen_mask)
            step += chunk
            total_step += chunk

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
                        run_config, N,
                        step_num=step_base + cycle_baseline_steps + step,
                        fast=True)
                    stress_affects[pid].append(affect)
                    prev_masses[pid] = p.mass
                    prev_values[pid] = p.values.copy()

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
        for pid in baseline_affects:
            b = baseline_affects[pid]
            s = stress_affects.get(pid, [])
            b_phis = [a.integration for a in b if a.integration > 0]
            s_phis = [a.integration for a in s if a.integration > 0]
            if b_phis: all_phi_base.append(np.mean(b_phis))
            if s_phis: all_phi_stress.append(np.mean(s_phis))
            if b_phis and s_phis:
                all_robustness.append(np.mean(s_phis) / np.mean(b_phis))

        mean_phi_base = np.mean(all_phi_base) if all_phi_base else 0.0
        mean_phi_stress = np.mean(all_phi_stress) if all_phi_stress else 0.0
        mean_robustness = np.mean(all_robustness) if all_robustness else 0.0
        phi_increase_frac = np.mean([r > 1.0 for r in all_robustness]) if all_robustness else 0.0

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
                        boost = 0.05 * random.uniform(
                            k_boost, grid[:, cells[:, 0], cells[:, 1]].shape)
                        grid = grid.at[:, cells[:, 0], cells[:, 1]].add(boost)
                        grid = jnp.clip(grid, 0.0, 1.0)

            rng, k_mut = random.split(rng)
            (tau, gate_beta, coupling,
             eta, motor_sensitivity, motor_threshold,
             memory_lambdas, lr, dr) = mutate_v16_params(
                tau, gate_beta, coupling,
                eta, motor_sensitivity, motor_threshold,
                memory_lambdas, lr, dr, k_mut)
            coupling_row_sums = coupling.sum(axis=1)

        N_grid = config['grid_size']
        rng, k_bloom, k_bloom2 = random.split(rng, 3)
        cx = int(random.randint(k_bloom, (), 10, N_grid - 10))
        cy = int(random.randint(k_bloom2, (), 10, N_grid - 10))
        resource = perturb_resource_bloom(
            resource, (cy, cx), radius=N_grid // 3, intensity=0.8)

        if len(final_patterns) < 10:
            print(f"  [RESCUE] {len(final_patterns)} patterns — re-seeding")
            rng, k_reseed = random.split(rng)
            channel_mus = jnp.array(config['channel_mus'])
            fresh_grid, fresh_resource = init_soup_hd(N_grid, C, k_reseed, channel_mus)
            grid = 0.3 * grid + 0.7 * fresh_grid
            resource = jnp.maximum(resource, 0.5 * fresh_resource)
            # Re-init coupling field from current global coupling
            coupling_field = init_coupling_field(coupling, N_grid)
            for _ in range(500 // chunk):
                grid, resource, coupling_field, rng = run_v16_chunk(
                    grid, resource, h_embed, kernel_ffts, run_config,
                    coupling_field, rng, n_steps=chunk,
                    box_fft=box_fft, regen_mask=regen_mask)
            grid.block_until_ready()

        elapsed = time.time() - t0

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
            'learning_rate': float(lr),
            'decay_rate_coupling': float(dr),
            'coupling_divergence': coupling_divergence,
            'coupling_spatial_var': coupling_spatial_var,
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
            'memory_lambdas': [float(l) for l in memory_lambdas],
            'learning_rate': float(lr),
            'coupling_divergence': coupling_divergence,
            'coupling_spatial_var': coupling_spatial_var,
        })

        print(f"C{cycle:02d} | pat={len(final_patterns):4d} | mort={mortality:.0%} | "
              f"Phi_b={mean_phi_base:.3f} Phi_s={mean_phi_stress:.3f} "
              f"rob={mean_robustness:.3f} up={phi_increase_frac:.0%} | "
              f"lr={lr:.5f} cf_div={coupling_divergence:.3f} "
              f"disp={foraging['mean_displacement']:.1f} | {elapsed:.0f}s")

        if cycle % 5 == 0 or cycle == n_cycles - 1:
            g_np = np.array(grid)
            np.savez_compressed(
                f'{output_dir}/snapshots/cycle_{cycle:03d}.npz',
                grid=g_np,
                resource=np.array(resource),
                mean_channels=g_np.mean(axis=0),
            )

        with open(f'{output_dir}/v16_progress.json', 'w') as f:
            json.dump({
                'cycle_stats': cycle_stats,
                'phi_trajectory': all_phi_trajectories,
                'config': {k: v for k, v in config.items()
                          if not isinstance(v, np.ndarray)},
                'seed': seed,
                'status': 'in_progress',
            }, f, indent=2, default=serialize)

    with open(f'{output_dir}/v16_final.json', 'w') as f:
        json.dump({
            'cycle_stats': cycle_stats,
            'phi_trajectory': all_phi_trajectories,
            'tau': float(tau),
            'gate_beta': float(gate_beta),
            'chemotaxis_strength': float(eta),
            'motor_sensitivity': float(motor_sensitivity),
            'motor_threshold': float(motor_threshold),
            'memory_lambdas': [float(l) for l in memory_lambdas],
            'learning_rate': float(lr),
            'decay_rate_coupling': float(dr),
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
    print("V16 EVOLUTION SUMMARY")
    print("=" * 70)
    if cycle_stats:
        first = cycle_stats[0]
        last = cycle_stats[-1]
        print(f"  Patterns:      {first['n_patterns']:4d} -> {last['n_patterns']:4d}")
        print(f"  Robustness:    {first['mean_robustness']:.3f} -> {last['mean_robustness']:.3f}")
        print(f"  Phi base:      {first['mean_phi_base']:.4f} -> {last['mean_phi_base']:.4f}")
        print(f"  Phi stress:    {first['mean_phi_stress']:.4f} -> {last['mean_phi_stress']:.4f}")
        print(f"  Learning rate: {first['learning_rate']:.6f} -> {last['learning_rate']:.6f}")
        print(f"  Coupling div:  {first['coupling_divergence']:.3f} -> {last['coupling_divergence']:.3f}")
        print(f"  Memory lam:    {first['memory_lambdas']} -> {last['memory_lambdas']}")

    return cycle_stats


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cycles', type=int, default=30)
    parser.add_argument('--channels', type=int, default=16)
    parser.add_argument('--grid', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eta', type=float, default=0.5)
    parser.add_argument('--mem-channels', type=int, default=2)
    parser.add_argument('--patches', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--output', default='/home/ubuntu/results')
    args = parser.parse_args()

    run_v16_gpu(
        n_cycles=args.cycles,
        C=args.channels,
        N=args.grid,
        seed=args.seed,
        chemotaxis_strength=args.eta,
        memory_channels=args.mem_channels,
        n_resource_patches=args.patches,
        learning_rate=args.lr,
        output_dir=args.output,
    )
