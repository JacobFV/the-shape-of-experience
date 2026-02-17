"""V13 Evolution: Content-Based Coupling Lenia with Lethal Resource Dynamics.

Experiment 0 from the emergence program: can patterns persist, forage, and
evolve integration under survival pressure in a substrate with state-dependent
interaction topology?

Key differences from V12:
- Content-based coupling (cosine similarity gate) instead of learned Q-K attention
- Lethal resource dynamics (>50% naive mortality during drought)
- Simpler evolvable parameters: tau (similarity threshold), gate_beta (steepness)
- No projection matrices to mutate

The goal is NOT to reproduce V12's results but to establish the substrate
on which Experiments 1-12 will run. A successful run means:
1. Patterns survive baseline conditions (existence)
2. Patterns die under severe drought (lethality is real)
3. Evolution improves survival rate
4. Integration (Phi) is measurable and responsive to stress
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

from v13_substrate import (
    generate_v13_config, init_v13, run_v13_chunk,
    init_embedding,
)


# ============================================================================
# Parameter Mutation
# ============================================================================

def mutate_v13_params(tau, gate_beta, coupling, rng, coupling_bandwidth=3):
    """Mutate content-based coupling parameters.

    Much simpler than V12 — only 2 scalar params + coupling matrix.
    """
    k1, k2, k3 = random.split(rng, 3)

    # Mutate similarity threshold
    tau_new = tau + 0.1 * float(random.normal(k1, ()))
    tau_new = float(np.clip(tau_new, -2.0, 2.0))

    # Mutate gate steepness
    beta_new = gate_beta + 0.2 * float(random.normal(k2, ()))
    beta_new = float(np.clip(beta_new, 0.5, 10.0))

    # Small coupling perturbation
    C = coupling.shape[0]
    noise = 0.02 * random.normal(k3, coupling.shape)
    coupling_new = coupling + noise
    # Re-symmetrize and re-normalize
    coupling_new = (coupling_new + coupling_new.T) / 2
    coupling_new = jnp.abs(coupling_new)
    # Keep diagonal = 1
    coupling_new = coupling_new.at[jnp.arange(C), jnp.arange(C)].set(1.0)

    return tau_new, beta_new, coupling_new


# ============================================================================
# V13 Evolution Loop
# ============================================================================

def evolve_v13(config=None, n_cycles=30, steps_per_cycle=5000,
               cull_fraction=0.3, mutate_top_n=5,
               seed=42, C=16, N=128, similarity_radius=20,
               post_cycle_callback=None):
    """V13: Evolution with content-based coupling and lethal resources.

    Protocol:
    - Each cycle: baseline phase + stress (drought) phase
    - Graduated stress schedule (curriculum from V11.7)
    - Fitness = survival * phi_robustness * (1 + phi_base) * log(mass)
    - Bottom cull_fraction killed, top mutate_top_n boosted + params mutated
    """
    if config is None:
        config = generate_v13_config(C=C, N=N, seed=seed,
                                      similarity_radius=similarity_radius)

    N = config['grid_size']
    C = config['n_channels']
    rng = random.PRNGKey(seed)
    rng_np = np.random.RandomState(seed + 1000)

    # Initialize substrate
    grid, resource, h_embed, kernel_ffts, coupling, coupling_row_sums, box_fft = \
        init_v13(config, seed=seed)

    # Evolvable params
    tau = float(config['tau'])
    gate_beta = float(config['gate_beta'])

    # Stress schedule (graduated curriculum)
    base_schedule = np.linspace(0.5, 0.02, n_cycles)
    noise = 1.0 + 0.3 * rng_np.randn(n_cycles)
    stress_schedule = np.clip(base_schedule * noise, 0.01, 0.8)
    duration_schedule = rng_np.randint(500, 2001, size=n_cycles)

    print("=" * 60)
    print(f"V13 CONTENT-BASED COUPLING EVOLUTION (C={C}, N={N})")
    print("=" * 60)
    print(f"  Similarity R:    {config['similarity_radius']}")
    print(f"  Tau (initial):   {tau}")
    print(f"  Gate beta:       {gate_beta}")
    print(f"  Maintenance:     {config['maintenance_rate']}")
    print(f"  Cycles:          {n_cycles}")
    print(f"  Steps/cycle:     {steps_per_cycle}")
    print(f"  Cull fraction:   {cull_fraction}")
    print(f"  Stress regen:    {stress_schedule[0]:.3f} -> {stress_schedule[-1]:.3f}")
    print()

    # JIT warmup
    print("Phase 0: JIT compiling...", end=" ", flush=True)
    t0 = time.time()
    grid, resource, rng = run_v13_chunk(
        grid, resource, h_embed, kernel_ffts, config,
        coupling, coupling_row_sums, rng, n_steps=5, box_fft=box_fft)
    grid.block_until_ready()
    print(f"done ({time.time()-t0:.1f}s)")

    # Warmup run
    warmup_steps = min(4995, max(500, N * 10))
    print(f"  Running warmup ({warmup_steps} steps)...", end=" ", flush=True)
    t0 = time.time()
    grid, resource, rng = run_v13_chunk(
        grid, resource, h_embed, kernel_ffts, config,
        coupling, coupling_row_sums, rng, n_steps=warmup_steps, box_fft=box_fft)
    grid.block_until_ready()
    print(f"done ({time.time()-t0:.1f}s)")

    grid_np = np.array(grid)
    initial_patterns = detect_patterns_mc(grid_np, threshold=0.15)
    print(f"  {len(initial_patterns)} patterns after warmup\n")

    tracker = PatternTracker()
    prev_masses = {}
    prev_values = {}
    cycle_stats = []

    for cycle in range(n_cycles):
        t0 = time.time()

        # This cycle's stress parameters
        cycle_regen = float(stress_schedule[cycle]) * config['resource_regen']
        cycle_drought_steps = int(duration_schedule[cycle])
        cycle_baseline_steps = steps_per_cycle - cycle_drought_steps

        # Build stress config (reduced regen, increased consumption)
        stress_config = {**config}
        stress_config['resource_regen'] = cycle_regen
        stress_config['resource_consume'] = config['resource_consume'] * (
            1.0 + 0.5 * (1.0 - stress_schedule[cycle]))

        step_base = cycle * steps_per_cycle
        chunk = 50  # steps per chunk (more parallelizable than V12)
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
                drought=True, box_fft=box_fft)
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

        # ---- CYCLE SCORING ----
        grid_np = np.array(grid)
        final_patterns = detect_patterns_mc(grid_np, threshold=0.15)
        tracker.update(final_patterns, step=step_base + steps_per_cycle)

        # Mortality tracking
        n_baseline = len(baseline_affects)
        n_survived = len(stress_affects)
        mortality = 1.0 - (n_survived / max(n_baseline, 1))

        # Score fitness
        fitness_scores = {}
        for pid in set(list(baseline_affects.keys()) + list(stress_affects.keys())):
            b_aff = baseline_affects.get(pid, [])
            s_aff = stress_affects.get(pid, [])
            surv = baseline_survival.get(pid, 0)
            fitness_scores[pid] = score_fitness_functional(
                b_aff, s_aff, surv, steps_per_cycle)

        # Compute Phi stats
        all_phi_base = []
        all_phi_stress = []
        all_robustness = []
        for pid in baseline_affects:
            b_phis = [a.phi for a in baseline_affects[pid] if a.phi > 0]
            s_phis = [a.phi for a in stress_affects.get(pid, []) if a.phi > 0]
            if b_phis:
                all_phi_base.append(np.mean(b_phis))
            if s_phis:
                all_phi_stress.append(np.mean(s_phis))
            if b_phis and s_phis:
                all_robustness.append(np.mean(s_phis) / np.mean(b_phis))

        mean_phi_base = np.mean(all_phi_base) if all_phi_base else 0.0
        mean_phi_stress = np.mean(all_phi_stress) if all_phi_stress else 0.0
        mean_robustness = np.mean(all_robustness) if all_robustness else 0.0
        phi_increase_frac = np.mean([r > 1.0 for r in all_robustness]) if all_robustness else 0.0

        # ---- SELECTION ----
        if fitness_scores:
            sorted_pids = sorted(fitness_scores, key=fitness_scores.get, reverse=True)
            n_cull = int(len(sorted_pids) * cull_fraction)
            cull_pids = set(sorted_pids[-n_cull:]) if n_cull > 0 else set()

            # Kill bottom fraction
            for pid in cull_pids:
                if pid in tracker.active:
                    p = tracker.active[pid]
                    cells = p.cells
                    if len(cells) > 0:
                        grid = grid.at[:, cells[:, 0], cells[:, 1]].set(0.0)

            # Boost top and mutate params near winners
            for pid in sorted_pids[:mutate_top_n]:
                if pid in tracker.active:
                    p = tracker.active[pid]
                    cells = p.cells
                    if len(cells) > 0:
                        rng, k_boost = random.split(rng)
                        # Small boost to winner cells
                        boost = 0.05 * random.uniform(k_boost, grid[:, cells[:, 0], cells[:, 1]].shape)
                        grid = grid.at[:, cells[:, 0], cells[:, 1]].add(boost)
                        grid = jnp.clip(grid, 0.0, 1.0)

            # Mutate evolvable params
            rng, k_mut = random.split(rng)
            tau, gate_beta, coupling = mutate_v13_params(
                tau, gate_beta, coupling, k_mut)
            coupling_row_sums = coupling.sum(axis=1)

        # Resource bloom after stress — recover resources across the grid
        N_grid = config['grid_size']
        rng, k_bloom = random.split(rng)
        cx = int(random.randint(k_bloom, (), 10, N_grid - 10))
        rng, k_bloom2 = random.split(rng)
        cy = int(random.randint(k_bloom2, (), 10, N_grid - 10))
        resource = perturb_resource_bloom(resource, (cy, cx), radius=N_grid // 3, intensity=0.8)

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
            'stress_regen': float(stress_schedule[cycle]),
            'drought_steps': int(cycle_drought_steps),
            'elapsed_s': float(elapsed),
            'resource_mean': float(np.array(resource).mean()),
        }
        cycle_stats.append(stats)

        print(f"Cycle {cycle:3d} | pat={n_final:4d} | mort={mortality:.0%} | "
              f"Φ_b={mean_phi_base:.3f} Φ_s={mean_phi_stress:.3f} "
              f"rob={mean_robustness:.3f} ↑={phi_increase_frac:.0%} | "
              f"τ={tau:.2f} β={gate_beta:.1f} | "
              f"res={float(np.array(resource).mean()):.2f} | "
              f"{elapsed:.0f}s")

        # Callback for preemption resilience (Modal volume saves)
        if post_cycle_callback:
            post_cycle_callback(cycle, stats, grid, resource, config)

    result = {
        'cycle_stats': cycle_stats,
        'tau': tau,
        'gate_beta': gate_beta,
        'config': config,
        'condition': 'v13_content_coupling',
    }

    print()
    print("=" * 60)
    print("V13 EVOLUTION SUMMARY")
    print("=" * 60)
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

        # Key diagnostic: mortality trajectory
        mortalities = [s['mortality'] for s in cycle_stats]
        if mortalities:
            mid_idx = len(mortalities) // 2
            early_mort = np.mean(mortalities[:max(1, mid_idx)])
            late_mort = np.mean(mortalities[mid_idx:])
            print(f"\n  Mortality trend: early={early_mort:.0%} -> late={late_mort:.0%}")
            if late_mort < early_mort - 0.05:
                print("  *** EVOLUTION IS REDUCING MORTALITY ***")

    return result


# ============================================================================
# Stress Test (post-evolution)
# ============================================================================

def stress_test_v13(grid, resource, h_embed, kernel_ffts, config,
                    coupling, coupling_row_sums, tau, gate_beta,
                    rng, box_fft=None, n_trials=5, drought_steps=2000, baseline_steps=3000):
    """Post-evolution stress test: novel drought severity.

    Runs baseline->drought->recovery and measures Phi trajectory.
    Compares evolved patterns to naive (freshly initialized) patterns.
    """
    from v11_substrate_hd import init_soup_hd
    from v13_substrate import make_box_kernel_fft

    N = config['grid_size']
    C = config['n_channels']

    if box_fft is None:
        box_fft = make_box_kernel_fft(config['similarity_radius'], N)

    results = []

    for trial in range(n_trials):
        print(f"\n  Stress trial {trial+1}/{n_trials}")
        rng, k = random.split(rng)

        # Novel stress: very low regen
        stress_config = {**config}
        stress_config['resource_regen'] = 0.001  # near-zero
        stress_config['resource_consume'] = config['resource_consume'] * 1.5

        # --- Evolved substrate ---
        print("    Evolved: baseline...", end=" ", flush=True)
        g_evo, r_evo = grid, resource
        # Baseline measurement
        g_evo, r_evo, rng = run_v13_chunk(
            g_evo, r_evo, h_embed, kernel_ffts, config,
            coupling, coupling_row_sums, rng, n_steps=baseline_steps, box_fft=box_fft)
        g_evo.block_until_ready()

        g_np = np.array(g_evo)
        pats = detect_patterns_mc(g_np, threshold=0.15)
        n_evo_base = len(pats)

        phi_base_evo = []
        tracker = PatternTracker()
        tracker.update(pats, step=0)
        for p in tracker.active.values():
            affect, _, _, _, _ = measure_all_hd(
                p, None, None, [],
                jnp.array(g_np), kernel_ffts, coupling,
                config, N, step_num=0, fast=True)
            if affect.phi > 0:
                phi_base_evo.append(affect.phi)

        # Drought
        print("drought...", end=" ", flush=True)
        g_evo, r_evo, rng = run_v13_chunk(
            g_evo, r_evo, h_embed, kernel_ffts, stress_config,
            coupling, coupling_row_sums, rng, n_steps=drought_steps,
            drought=True, box_fft=box_fft)
        g_evo.block_until_ready()

        g_np = np.array(g_evo)
        pats = detect_patterns_mc(g_np, threshold=0.15)
        n_evo_stress = len(pats)

        phi_stress_evo = []
        tracker = PatternTracker()
        tracker.update(pats, step=0)
        for p in tracker.active.values():
            affect, _, _, _, _ = measure_all_hd(
                p, None, None, [],
                jnp.array(g_np), kernel_ffts, coupling,
                config, N, step_num=0, fast=True)
            if affect.phi > 0:
                phi_stress_evo.append(affect.phi)

        evo_mortality = 1.0 - (n_evo_stress / max(n_evo_base, 1))
        print(f"done. {n_evo_base}->{n_evo_stress} ({evo_mortality:.0%} mortality)")

        # --- Naive substrate ---
        print("    Naive: init...", end=" ", flush=True)
        rng, k_naive = random.split(rng)
        g_naive, r_naive = init_soup_hd(
            N, C, k_naive, jnp.array(config['channel_mus']))

        # Same warmup as evolution
        g_naive, r_naive, rng = run_v13_chunk(
            g_naive, r_naive, h_embed, kernel_ffts, config,
            coupling, coupling_row_sums, rng, n_steps=5000, box_fft=box_fft)
        g_naive.block_until_ready()

        # Baseline
        g_naive, r_naive, rng = run_v13_chunk(
            g_naive, r_naive, h_embed, kernel_ffts, config,
            coupling, coupling_row_sums, rng, n_steps=baseline_steps, box_fft=box_fft)
        g_naive.block_until_ready()

        g_np = np.array(g_naive)
        pats = detect_patterns_mc(g_np, threshold=0.15)
        n_naive_base = len(pats)

        phi_base_naive = []
        tracker = PatternTracker()
        tracker.update(pats, step=0)
        for p in tracker.active.values():
            affect, _, _, _, _ = measure_all_hd(
                p, None, None, [],
                jnp.array(g_np), kernel_ffts, coupling,
                config, N, step_num=0, fast=True)
            if affect.phi > 0:
                phi_base_naive.append(affect.phi)

        # Drought
        print("drought...", end=" ", flush=True)
        g_naive, r_naive, rng = run_v13_chunk(
            g_naive, r_naive, h_embed, kernel_ffts, stress_config,
            coupling, coupling_row_sums, rng, n_steps=drought_steps,
            drought=True, box_fft=box_fft)
        g_naive.block_until_ready()

        g_np = np.array(g_naive)
        pats = detect_patterns_mc(g_np, threshold=0.15)
        n_naive_stress = len(pats)

        phi_stress_naive = []
        tracker = PatternTracker()
        tracker.update(pats, step=0)
        for p in tracker.active.values():
            affect, _, _, _, _ = measure_all_hd(
                p, None, None, [],
                jnp.array(g_np), kernel_ffts, coupling,
                config, N, step_num=0, fast=True)
            if affect.phi > 0:
                phi_stress_naive.append(affect.phi)

        naive_mortality = 1.0 - (n_naive_stress / max(n_naive_base, 1))
        print(f"done. {n_naive_base}->{n_naive_stress} ({naive_mortality:.0%} mortality)")

        # Compute Phi deltas
        mean_phi_base_evo = np.mean(phi_base_evo) if phi_base_evo else 0
        mean_phi_stress_evo = np.mean(phi_stress_evo) if phi_stress_evo else 0
        mean_phi_base_naive = np.mean(phi_base_naive) if phi_base_naive else 0
        mean_phi_stress_naive = np.mean(phi_stress_naive) if phi_stress_naive else 0

        evo_delta = ((mean_phi_stress_evo - mean_phi_base_evo) / max(mean_phi_base_evo, 1e-6))
        naive_delta = ((mean_phi_stress_naive - mean_phi_base_naive) / max(mean_phi_base_naive, 1e-6))

        results.append({
            'trial': trial,
            'evolved': {
                'n_base': n_evo_base, 'n_stress': n_evo_stress,
                'mortality': float(evo_mortality),
                'phi_base': float(mean_phi_base_evo),
                'phi_stress': float(mean_phi_stress_evo),
                'phi_delta': float(evo_delta),
            },
            'naive': {
                'n_base': n_naive_base, 'n_stress': n_naive_stress,
                'mortality': float(naive_mortality),
                'phi_base': float(mean_phi_base_naive),
                'phi_stress': float(mean_phi_stress_naive),
                'phi_delta': float(naive_delta),
            },
        })

    # Summary
    print("\n" + "=" * 60)
    print("STRESS TEST SUMMARY")
    print("=" * 60)
    evo_morts = [r['evolved']['mortality'] for r in results]
    naive_morts = [r['naive']['mortality'] for r in results]
    evo_deltas = [r['evolved']['phi_delta'] for r in results]
    naive_deltas = [r['naive']['phi_delta'] for r in results]

    print(f"  Evolved mortality:  {np.mean(evo_morts):.0%} ± {np.std(evo_morts):.0%}")
    print(f"  Naive mortality:    {np.mean(naive_morts):.0%} ± {np.std(naive_morts):.0%}")
    print(f"  Evolved Phi delta:  {np.mean(evo_deltas):+.1%} ± {np.std(evo_deltas):.1%}")
    print(f"  Naive Phi delta:    {np.mean(naive_deltas):+.1%} ± {np.std(naive_deltas):.1%}")
    print(f"  Mortality shift:    {np.mean(naive_morts) - np.mean(evo_morts):+.1%}pp")

    if np.mean(evo_morts) < np.mean(naive_morts) - 0.05:
        print("  *** EVOLUTION REDUCES MORTALITY ***")
    if np.mean(evo_morts) > 0.3:
        print("  *** LETHAL RESOURCES CONFIRMED: significant mortality ***")

    return {
        'trials': results,
        'comparison': {
            'evolved_mortality_mean': float(np.mean(evo_morts)),
            'naive_mortality_mean': float(np.mean(naive_morts)),
            'evolved_phi_delta': float(np.mean(evo_deltas)),
            'naive_phi_delta': float(np.mean(naive_deltas)),
        }
    }


# ============================================================================
# Full Pipeline
# ============================================================================

def full_pipeline_v13(n_cycles=30, C=16, N=128, seed=42,
                      similarity_radius=20, post_cycle_callback=None):
    """Run V13 evolution + stress test."""
    config = generate_v13_config(C=C, N=N, seed=seed,
                                  similarity_radius=similarity_radius)

    rng = random.PRNGKey(seed)

    # Evolution
    evo_result = evolve_v13(
        config=config, n_cycles=n_cycles, seed=seed,
        C=C, N=N, similarity_radius=similarity_radius,
        post_cycle_callback=post_cycle_callback)

    # Re-init for stress test with evolved params
    grid, resource, h_embed, kernel_ffts, coupling_init, _, box_fft = init_v13(config, seed=seed)

    # Use evolved tau and gate_beta
    tau = evo_result['tau']
    gate_beta = evo_result['gate_beta']
    config['tau'] = tau
    config['gate_beta'] = gate_beta

    # Re-run warmup with evolved params
    coupling = generate_coupling_matrix(C, bandwidth=3, seed=seed)
    coupling = jnp.array(coupling)
    coupling_row_sums = coupling.sum(axis=1)

    # Run warmup
    grid, resource, rng = run_v13_chunk(
        grid, resource, h_embed, kernel_ffts, config,
        coupling, coupling_row_sums, rng, n_steps=5000, box_fft=box_fft)

    # Stress test
    stress_result = stress_test_v13(
        grid, resource, h_embed, kernel_ffts, config,
        coupling, coupling_row_sums, tau, gate_beta, rng, box_fft=box_fft)

    return {
        'evolution': evo_result,
        'stress_test': stress_result,
    }
