"""V18 Evolution: Boundary-Dependent Lenia with Insulation Field.

Extends V15 evolution with:
- Insulation field parameters (boundary_width, insulation_beta, internal_gain,
  activity_threshold) are evolvable
- Recurrence coupling matrix evolves independently from external coupling
- Insulation metrics tracked per cycle
- No signal fields (that was V17; V18 branches from V15)

Key prediction: internal_gain should evolve upward — patterns that develop
stronger internal processing will have richer world models, better predictions,
and higher fitness.
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import time

from v11_patterns import detect_patterns_mc, PatternTracker
from v11_affect_hd import measure_all_hd
from v11_evolution import score_fitness_functional
from v11_substrate import perturb_resource_bloom
from v11_substrate_hd import (
    generate_coupling_matrix, make_kernels_fft_hd,
)

from v13_substrate import make_box_kernel_fft
from v14_evolution import compute_foraging_metrics
from v15_substrate import (
    generate_resource_patches, compute_patch_regen_mask,
)
from v18_substrate import (
    generate_v18_config, init_v18, run_v18_chunk,
    compute_insulation_metrics,
)


# ============================================================================
# Parameter Mutation
# ============================================================================

def mutate_v18_params(tau, gate_beta, coupling, chemotaxis_strength,
                      motor_sensitivity, motor_threshold,
                      memory_lambdas,
                      # V18-specific
                      boundary_width, insulation_beta,
                      internal_gain, activity_threshold,
                      recurrence_coupling,
                      rng, coupling_bandwidth=3):
    """Mutate V18 parameters: V15 params + insulation params."""
    keys = random.split(rng, 14)

    # V13 params
    tau_new = tau + 0.1 * float(random.normal(keys[0], ()))
    tau_new = float(np.clip(tau_new, -2.0, 2.0))

    beta_new = gate_beta + 0.2 * float(random.normal(keys[1], ()))
    beta_new = float(np.clip(beta_new, 0.5, 10.0))

    C = coupling.shape[0]
    noise = 0.02 * random.normal(keys[2], coupling.shape)
    coupling_new = coupling + noise
    coupling_new = (coupling_new + coupling_new.T) / 2
    coupling_new = jnp.abs(coupling_new)
    coupling_new = coupling_new.at[jnp.arange(C), jnp.arange(C)].set(1.0)

    # V14 chemotaxis
    eta_new = chemotaxis_strength + 0.1 * float(random.normal(keys[3], ()))
    eta_new = float(np.clip(eta_new, 0.0, 3.0))

    sens_new = motor_sensitivity + 0.5 * float(random.normal(keys[4], ()))
    sens_new = float(np.clip(sens_new, 0.5, 20.0))

    thresh_new = motor_threshold + 0.05 * float(random.normal(keys[5], ()))
    thresh_new = float(np.clip(thresh_new, 0.01, 0.9))

    # V15 memory lambdas
    lambdas_new = []
    for i, lam in enumerate(memory_lambdas):
        log_lam = np.log(lam + 1e-6)
        log_lam_new = log_lam + 0.2 * float(random.normal(keys[6], ()))
        lam_new = float(np.clip(np.exp(log_lam_new), 0.001, 0.5))
        lambdas_new.append(lam_new)

    # V18 boundary_width
    bw_new = boundary_width + 0.2 * float(random.normal(keys[7], ()))
    bw_new = float(np.clip(bw_new, 0.5, 5.0))

    # V18 insulation_beta
    ib_new = insulation_beta + 0.3 * float(random.normal(keys[8], ()))
    ib_new = float(np.clip(ib_new, 1.0, 10.0))

    # V18 internal_gain (log-space for multiplicative evolution)
    log_ig = np.log(internal_gain + 1e-6)
    log_ig_new = log_ig + 0.15 * float(random.normal(keys[9], ()))
    ig_new = float(np.clip(np.exp(log_ig_new), 0.1, 3.0))

    # V18 activity_threshold
    at_new = activity_threshold + 0.02 * float(random.normal(keys[10], ()))
    at_new = float(np.clip(at_new, 0.03, 0.3))

    # V18 recurrence coupling (independent mutation)
    rec_noise = 0.02 * random.normal(keys[11], recurrence_coupling.shape)
    rec_new = recurrence_coupling + rec_noise
    rec_new = (rec_new + rec_new.T) / 2
    rec_new = jnp.abs(rec_new)
    rec_new = rec_new.at[jnp.arange(C), jnp.arange(C)].set(1.0)

    return (tau_new, beta_new, coupling_new,
            eta_new, sens_new, thresh_new, lambdas_new,
            bw_new, ib_new, ig_new, at_new, rec_new)


# ============================================================================
# V18 Evolution Loop
# ============================================================================

def evolve_v18(config=None, n_cycles=30, steps_per_cycle=5000,
               cull_fraction=0.3, mutate_top_n=5,
               seed=42, C=16, N=128, similarity_radius=20,
               chemotaxis_strength=0.5, memory_channels=2,
               post_cycle_callback=None):
    """V18: Evolution with boundary-dependent insulation field."""
    if config is None:
        config = generate_v18_config(
            C=C, N=N, seed=seed,
            similarity_radius=similarity_radius,
            chemotaxis_strength=chemotaxis_strength,
            memory_channels=memory_channels)

    N = config['grid_size']
    C = config['n_channels']
    rng = random.PRNGKey(seed)
    rng_np = np.random.RandomState(seed + 1000)

    (grid, resource, h_embed, kernel_ffts, internal_kernel_ffts,
     coupling, coupling_row_sums, recurrence_coupling, recurrence_crs,
     box_fft) = init_v18(config, seed=seed)

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

    # Resource patches
    patches = generate_resource_patches(N, config['n_resource_patches'],
                                         seed=seed)

    # Stress schedule
    base_schedule = np.linspace(0.60, 0.40, n_cycles)
    noise = 1.0 + 0.1 * rng_np.randn(n_cycles)
    stress_schedule = np.clip(base_schedule * noise, 0.25, 0.8)
    duration_schedule = rng_np.randint(500, 1501, size=n_cycles)

    chunk = 50
    total_step = 0

    print("=" * 60)
    print(f"V18 BOUNDARY-DEPENDENT LENIA EVOLUTION (C={C}, N={N})")
    print("=" * 60)
    print(f"  Boundary width:    {boundary_width}")
    print(f"  Insulation beta:   {insulation_beta}")
    print(f"  Internal gain:     {internal_gain}")
    print(f"  Activity thresh:   {activity_threshold}")
    print(f"  Memory channels:   {config['memory_channels']}")
    print(f"  Memory lambdas:    {memory_lambdas}")
    print(f"  Chemotaxis eta:    {eta}")
    print(f"  Cycles:            {n_cycles}")
    print()

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

    # JIT warmup
    regen_mask = compute_patch_regen_mask(
        N, patches, step=0, shift_period=config['patch_shift_period'])
    regen_mask = jnp.array(regen_mask)

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
    warmup_chunks = min(100, max(10, N * 10 // chunk))
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

    grid_np = np.array(grid)
    initial_patterns = detect_patterns_mc(grid_np, threshold=0.15)
    print(f"  {len(initial_patterns)} patterns after warmup")

    # Initial insulation check
    ins_metrics = compute_insulation_metrics(grid, run_config)
    print(f"  Insulation: mean={ins_metrics['insulation_mean']:.4f}, "
          f"interior={ins_metrics['interior_fraction']:.1%}\n")

    tracker = PatternTracker()
    prev_masses = {}
    prev_values = {}
    cycle_stats = []

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

        # ---- BASELINE ----
        baseline_affects = {}
        baseline_survival = {}
        patterns_at_start = None

        step = 0
        while step < cycle_baseline_steps:
            regen_mask = compute_patch_regen_mask(
                N, patches, step=total_step,
                shift_period=config['patch_shift_period'])
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

        # Foraging + memory + insulation metrics
        grid_np = np.array(grid)
        patterns_at_end = detect_patterns_mc(grid_np, threshold=0.15)
        foraging = compute_foraging_metrics(
            patterns_at_start, patterns_at_end, resource)

        mem_start = config['memory_start']
        mem_end = mem_start + config['memory_channels']
        memory_mean = float(np.array(grid[mem_start:mem_end]).mean())

        ins_metrics = compute_insulation_metrics(grid, run_config)

        # ---- STRESS ----
        stress_affects = {}
        measure_every_stress = max(100, cycle_drought_steps // 8)

        step = 0
        while step < cycle_drought_steps:
            regen_mask = compute_patch_regen_mask(
                N, patches, step=total_step,
                shift_period=config['patch_shift_period'])
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
            n_cull = int(len(sorted_pids) * cull_fraction)
            cull_pids = set(sorted_pids[-n_cull:]) if n_cull > 0 else set()

            for pid in cull_pids:
                if pid in tracker.active:
                    p = tracker.active[pid]
                    cells = p.cells
                    if len(cells) > 0:
                        grid = grid.at[
                            :, cells[:, 0], cells[:, 1]].set(0.0)

            for pid in sorted_pids[:mutate_top_n]:
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

        # ---- REPORT ----
        elapsed = time.time() - t0
        n_final = len(final_patterns)

        stats = {
            'cycle': cycle,
            'n_patterns': n_final,
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

        ins_m = ins_metrics.get('insulation_mean', 0)
        ins_i = ins_metrics.get('interior_fraction', 0)
        print(f"C{cycle:02d} | pat={n_final:4d} | mort={mortality:.0%} | "
              f"Phi_b={mean_phi_base:.3f} Phi_s={mean_phi_stress:.3f} "
              f"rob={mean_robustness:.3f} up={phi_increase_frac:.0%} | "
              f"ins={ins_m:.3f} int={ins_i:.1%} gain={internal_gain:.2f} "
              f"bw={boundary_width:.1f} | "
              f"eta={eta:.2f} "
              f"lam=[{','.join(f'{l:.3f}' for l in memory_lambdas)}] | "
              f"{elapsed:.0f}s")

        if post_cycle_callback:
            post_cycle_callback(cycle, stats, grid, resource, config)

    # Summary
    result = {
        'cycle_stats': cycle_stats,
        'tau': tau,
        'gate_beta': gate_beta,
        'chemotaxis_strength': eta,
        'motor_sensitivity': motor_sensitivity,
        'motor_threshold': motor_threshold,
        'memory_lambdas': memory_lambdas,
        'boundary_width': boundary_width,
        'insulation_beta': insulation_beta,
        'internal_gain': internal_gain,
        'activity_threshold': activity_threshold,
        'config': config,
        'condition': 'v18_boundary',
    }

    print()
    print("=" * 60)
    print("V18 EVOLUTION SUMMARY")
    print("=" * 60)
    if cycle_stats:
        first = cycle_stats[0]
        last = cycle_stats[-1]
        print(f"  Patterns:      {first['n_patterns']:4d} -> "
              f"{last['n_patterns']:4d}")
        print(f"  Mortality:     {first['mortality']:.0%} -> "
              f"{last['mortality']:.0%}")
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

        disps = [s['foraging']['mean_displacement'] for s in cycle_stats]
        mid = len(disps) // 2
        print(f"  Displacement:  {np.mean(disps[:mid]):.1f} -> "
              f"{np.mean(disps[mid:]):.1f}")

    return result


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == '__main__':
    print("V18 Evolution Smoke Test (C=8, N=64, 2 cycles)")
    result = evolve_v18(
        n_cycles=2, steps_per_cycle=500,
        seed=42, C=8, N=64,
        similarity_radius=5,
        memory_channels=2,
    )
    print("\nSmoke test complete!")
    for i, cs in enumerate(result['cycle_stats']):
        print(f"  Cycle {i}: rob={cs['mean_robustness']:.3f}, "
              f"gain={cs['internal_gain']:.3f}, "
              f"ins={cs['insulation_mean']:.4f}")
