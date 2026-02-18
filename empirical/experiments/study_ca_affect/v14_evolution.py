"""V14 Evolution: Chemotactic Lenia with Directed Motion.

Extends V13 evolution with:
- Chemotaxis parameter mutation (eta, motor_sensitivity, motor_threshold)
- Foraging metrics tracking (displacement, resource correlation)
- Uses run_v14_chunk instead of run_v13_chunk

The key hypothesis: survival fitness already selects for directed motion.
Patterns that move toward resources survive drought better, so chemotaxis
emerges through the same fitness function used in V13. We track foraging
metrics to verify this emergence.
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
    generate_v13_config, init_v13, make_box_kernel_fft, init_embedding,
)
from v14_substrate import (
    generate_v14_config, init_v14, run_v14_chunk,
)


# ============================================================================
# Parameter Mutation
# ============================================================================

def mutate_v14_params(tau, gate_beta, coupling, chemotaxis_strength,
                      motor_sensitivity, motor_threshold,
                      rng, coupling_bandwidth=3):
    """Mutate V14 parameters: V13 params + chemotaxis params.

    Returns (tau, gate_beta, coupling, eta, motor_sens, motor_thresh).
    """
    k1, k2, k3, k4, k5, k6 = random.split(rng, 6)

    # ---- V13 params (unchanged) ----
    tau_new = tau + 0.1 * float(random.normal(k1, ()))
    tau_new = float(np.clip(tau_new, -2.0, 2.0))

    beta_new = gate_beta + 0.2 * float(random.normal(k2, ()))
    beta_new = float(np.clip(beta_new, 0.5, 10.0))

    C = coupling.shape[0]
    noise = 0.02 * random.normal(k3, coupling.shape)
    coupling_new = coupling + noise
    coupling_new = (coupling_new + coupling_new.T) / 2
    coupling_new = jnp.abs(coupling_new)
    coupling_new = coupling_new.at[jnp.arange(C), jnp.arange(C)].set(1.0)

    # ---- V14 chemotaxis params ----
    eta_new = chemotaxis_strength + 0.1 * float(random.normal(k4, ()))
    eta_new = float(np.clip(eta_new, 0.0, 3.0))

    sens_new = motor_sensitivity + 0.5 * float(random.normal(k5, ()))
    sens_new = float(np.clip(sens_new, 0.5, 20.0))

    thresh_new = motor_threshold + 0.05 * float(random.normal(k6, ()))
    thresh_new = float(np.clip(thresh_new, 0.01, 0.9))

    return tau_new, beta_new, coupling_new, eta_new, sens_new, thresh_new


# ============================================================================
# Foraging Metrics
# ============================================================================

def compute_foraging_metrics(patterns_before, patterns_after, resource):
    """Compute foraging metrics between two pattern snapshots.

    Returns dict with:
    - mean_displacement: average centroid displacement (pixels)
    - resource_at_patterns: mean resource at pattern locations
    - n_tracked: number of patterns tracked between snapshots
    """
    if not patterns_before or not patterns_after:
        return {
            'mean_displacement': 0.0,
            'resource_at_patterns': 0.0,
            'n_tracked': 0,
        }

    resource_np = np.array(resource)

    # Match patterns by closest centroid
    centroids_before = {i: (p.center_y, p.center_x) for i, p in enumerate(patterns_before)}
    centroids_after = {i: (p.center_y, p.center_x) for i, p in enumerate(patterns_after)}

    displacements = []
    resources_at = []

    for j, (cy_a, cx_a) in centroids_after.items():
        # Find closest pattern in before
        min_dist = float('inf')
        for i, (cy_b, cx_b) in centroids_before.items():
            # Periodic distance
            dy = min(abs(cy_a - cy_b), resource_np.shape[0] - abs(cy_a - cy_b))
            dx = min(abs(cx_a - cx_b), resource_np.shape[1] - abs(cx_a - cx_b))
            dist = np.sqrt(dy**2 + dx**2)
            if dist < min_dist:
                min_dist = dist

        if min_dist < resource_np.shape[0] / 4:  # reasonable matching distance
            displacements.append(min_dist)

        # Resource at pattern location
        iy = int(cy_a) % resource_np.shape[0]
        ix = int(cx_a) % resource_np.shape[1]
        resources_at.append(resource_np[iy, ix])

    return {
        'mean_displacement': float(np.mean(displacements)) if displacements else 0.0,
        'resource_at_patterns': float(np.mean(resources_at)) if resources_at else 0.0,
        'n_tracked': len(displacements),
    }


# ============================================================================
# V14 Evolution Loop
# ============================================================================

def evolve_v14(config=None, n_cycles=30, steps_per_cycle=5000,
               cull_fraction=0.3, mutate_top_n=5,
               seed=42, C=16, N=128, similarity_radius=20,
               chemotaxis_strength=0.5,
               post_cycle_callback=None):
    """V14: Evolution with chemotaxis and lethal resources.

    Same protocol as V13 but using V14 substrate (advection).
    Chemotaxis parameters are mutated alongside tau/gate_beta.
    Foraging metrics tracked for analysis.
    """
    if config is None:
        config = generate_v14_config(
            C=C, N=N, seed=seed,
            similarity_radius=similarity_radius,
            chemotaxis_strength=chemotaxis_strength)

    N = config['grid_size']
    C = config['n_channels']
    rng = random.PRNGKey(seed)
    rng_np = np.random.RandomState(seed + 1000)

    # Initialize substrate
    grid, resource, h_embed, kernel_ffts, coupling, coupling_row_sums, box_fft = \
        init_v14(config, seed=seed)

    # Evolvable params (V13)
    tau = float(config['tau'])
    gate_beta = float(config['gate_beta'])

    # Evolvable params (V14)
    eta = float(config['chemotaxis_strength'])
    motor_sensitivity = float(config['motor_sensitivity'])
    motor_threshold = float(config['motor_threshold'])

    # Stress schedule (graduated curriculum)
    base_schedule = np.linspace(0.5, 0.02, n_cycles)
    noise = 1.0 + 0.3 * rng_np.randn(n_cycles)
    stress_schedule = np.clip(base_schedule * noise, 0.01, 0.8)
    duration_schedule = rng_np.randint(500, 2001, size=n_cycles)

    print("=" * 60)
    print(f"V14 CHEMOTACTIC LENIA EVOLUTION (C={C}, N={N})")
    print("=" * 60)
    print(f"  Similarity R:    {config['similarity_radius']}")
    print(f"  Tau (initial):   {tau}")
    print(f"  Gate beta:       {gate_beta}")
    print(f"  Chemotaxis η:    {eta}")
    print(f"  Motor sens:      {motor_sensitivity}")
    print(f"  Motor thresh:    {motor_threshold}")
    print(f"  Motor channels:  {config['motor_channels']}")
    print(f"  Max speed:       {config['max_speed']}")
    print(f"  Maintenance:     {config['maintenance_rate']}")
    print(f"  Cycles:          {n_cycles}")
    print(f"  Steps/cycle:     {steps_per_cycle}")
    print(f"  Cull fraction:   {cull_fraction}")
    print(f"  Stress regen:    {stress_schedule[0]:.3f} -> {stress_schedule[-1]:.3f}")
    print()

    def _make_run_config(base_config):
        """Build config with current evolvable params."""
        cfg = {**base_config}
        cfg['tau'] = tau
        cfg['gate_beta'] = gate_beta
        cfg['chemotaxis_strength'] = eta
        cfg['motor_sensitivity'] = motor_sensitivity
        cfg['motor_threshold'] = motor_threshold
        return cfg

    # JIT warmup
    run_config = _make_run_config(config)
    print("Phase 0: JIT compiling...", end=" ", flush=True)
    t0 = time.time()
    grid, resource, rng = run_v14_chunk(
        grid, resource, h_embed, kernel_ffts, run_config,
        coupling, coupling_row_sums, rng, n_steps=5, box_fft=box_fft)
    grid.block_until_ready()
    print(f"done ({time.time()-t0:.1f}s)")

    # Warmup run
    warmup_steps = min(4995, max(500, N * 10))
    print(f"  Running warmup ({warmup_steps} steps)...", end=" ", flush=True)
    t0 = time.time()
    grid, resource, rng = run_v14_chunk(
        grid, resource, h_embed, kernel_ffts, run_config,
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

        # Rebuild config with current evolvable params
        run_config = _make_run_config(config)

        # This cycle's stress parameters
        cycle_regen = float(stress_schedule[cycle]) * config['resource_regen']
        cycle_drought_steps = int(duration_schedule[cycle])
        cycle_baseline_steps = steps_per_cycle - cycle_drought_steps

        # Build stress config
        stress_config = {**run_config}
        stress_config['resource_regen'] = cycle_regen
        stress_config['resource_consume'] = config['resource_consume'] * (
            1.0 + 0.5 * (1.0 - stress_schedule[cycle]))

        step_base = cycle * steps_per_cycle
        chunk = 50
        measure_every = max(200, cycle_baseline_steps // 8)

        # ---- BASELINE PHASE ----
        baseline_affects = {}
        baseline_survival = {}
        patterns_at_start = None  # for foraging metrics

        step = 0
        while step < cycle_baseline_steps:
            grid, resource, rng = run_v14_chunk(
                grid, resource, h_embed, kernel_ffts, run_config,
                coupling, coupling_row_sums, rng, n_steps=chunk, box_fft=box_fft)
            step += chunk

            if step % measure_every < chunk:
                grid_np = np.array(grid)
                patterns = detect_patterns_mc(grid_np, threshold=0.15)
                tracker.update(patterns, step=step_base + step)

                # Save first detection for foraging metrics
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
                        fast=True,
                    )
                    baseline_affects[pid].append(affect)
                    prev_masses[pid] = p.mass
                    prev_values[pid] = p.values.copy()

        # Foraging metrics (end of baseline)
        grid_np = np.array(grid)
        patterns_at_end_baseline = detect_patterns_mc(grid_np, threshold=0.15)
        foraging = compute_foraging_metrics(
            patterns_at_start, patterns_at_end_baseline, resource)

        # ---- STRESS PHASE ----
        stress_affects = {}
        measure_every_stress = max(100, cycle_drought_steps // 8)

        step = 0
        while step < cycle_drought_steps:
            grid, resource, rng = run_v14_chunk(
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
                        run_config, N,
                        step_num=step_base + cycle_baseline_steps + step,
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
                        boost = 0.05 * random.uniform(
                            k_boost, grid[:, cells[:, 0], cells[:, 1]].shape)
                        grid = grid.at[:, cells[:, 0], cells[:, 1]].add(boost)
                        grid = jnp.clip(grid, 0.0, 1.0)

            # Mutate evolvable params (V13 + V14)
            rng, k_mut = random.split(rng)
            tau, gate_beta, coupling, eta, motor_sensitivity, motor_threshold = \
                mutate_v14_params(
                    tau, gate_beta, coupling,
                    eta, motor_sensitivity, motor_threshold,
                    k_mut)
            coupling_row_sums = coupling.sum(axis=1)

        # Resource bloom after stress
        N_grid = config['grid_size']
        rng, k_bloom = random.split(rng)
        cx = int(random.randint(k_bloom, (), 10, N_grid - 10))
        rng, k_bloom2 = random.split(rng)
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
            'stress_regen': float(stress_schedule[cycle]),
            'drought_steps': int(cycle_drought_steps),
            'elapsed_s': float(elapsed),
            'resource_mean': float(np.array(resource).mean()),
            'foraging': foraging,
        }
        cycle_stats.append(stats)

        print(f"Cycle {cycle:3d} | pat={n_final:4d} | mort={mortality:.0%} | "
              f"Φ_b={mean_phi_base:.3f} Φ_s={mean_phi_stress:.3f} "
              f"rob={mean_robustness:.3f} ↑={phi_increase_frac:.0%} | "
              f"η={eta:.2f} disp={foraging['mean_displacement']:.1f} | "
              f"res={float(np.array(resource).mean()):.2f} | "
              f"{elapsed:.0f}s")

        # Callback for snapshot saves
        if post_cycle_callback:
            post_cycle_callback(cycle, stats, grid, resource, config)

    result = {
        'cycle_stats': cycle_stats,
        'tau': tau,
        'gate_beta': gate_beta,
        'chemotaxis_strength': eta,
        'motor_sensitivity': motor_sensitivity,
        'motor_threshold': motor_threshold,
        'config': config,
        'condition': 'v14_chemotactic',
    }

    print()
    print("=" * 60)
    print("V14 EVOLUTION SUMMARY")
    print("=" * 60)
    if cycle_stats:
        first = cycle_stats[0]
        last = cycle_stats[-1]
        print(f"  Patterns:     {first['n_patterns']:4d} -> {last['n_patterns']:4d}")
        print(f"  Mortality:    {first['mortality']:.0%} -> {last['mortality']:.0%}")
        print(f"  Phi (base):   {first['mean_phi_base']:.4f} -> {last['mean_phi_base']:.4f}")
        print(f"  Phi (stress): {first['mean_phi_stress']:.4f} -> {last['mean_phi_stress']:.4f}")
        print(f"  Robustness:   {first['mean_robustness']:.3f} -> {last['mean_robustness']:.3f}")
        print(f"  Chemotaxis η: {first['chemotaxis_strength']:.2f} -> {last['chemotaxis_strength']:.2f}")
        print(f"  Motor sens:   {first['motor_sensitivity']:.1f} -> {last['motor_sensitivity']:.1f}")
        print(f"  Motor thresh: {first['motor_threshold']:.2f} -> {last['motor_threshold']:.2f}")
        print(f"  Tau:          {first['tau']:.2f} -> {last['tau']:.2f}")

        # Foraging trajectory
        disps = [s['foraging']['mean_displacement'] for s in cycle_stats]
        if disps:
            mid = len(disps) // 2
            early_disp = np.mean(disps[:max(1, mid)])
            late_disp = np.mean(disps[mid:])
            print(f"\n  Displacement: early={early_disp:.1f} -> late={late_disp:.1f}")
            if late_disp > early_disp + 0.5:
                print("  *** DIRECTED MOTION EMERGING ***")

        # Mortality trajectory
        mortalities = [s['mortality'] for s in cycle_stats]
        if mortalities:
            mid = len(mortalities) // 2
            early_mort = np.mean(mortalities[:max(1, mid)])
            late_mort = np.mean(mortalities[mid:])
            print(f"  Mortality:    early={early_mort:.0%} -> late={late_mort:.0%}")
            if late_mort < early_mort - 0.05:
                print("  *** EVOLUTION IS REDUCING MORTALITY ***")

    return result


# ============================================================================
# Full Pipeline
# ============================================================================

def full_pipeline_v14(n_cycles=30, C=16, N=128, seed=42,
                      similarity_radius=20, chemotaxis_strength=0.5,
                      post_cycle_callback=None):
    """Run V14 evolution pipeline."""
    config = generate_v14_config(
        C=C, N=N, seed=seed,
        similarity_radius=similarity_radius,
        chemotaxis_strength=chemotaxis_strength)

    evo_result = evolve_v14(
        config=config, n_cycles=n_cycles, seed=seed,
        C=C, N=N, similarity_radius=similarity_radius,
        chemotaxis_strength=chemotaxis_strength,
        post_cycle_callback=post_cycle_callback)

    return {
        'evolution': evo_result,
    }
