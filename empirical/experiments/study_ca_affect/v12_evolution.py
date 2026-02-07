"""V12 Evolution: Attention-Based Lenia with Evolvable Interaction Topology.

Builds on V11.7 curriculum evolution protocol with attention-specific additions:
- Evolvable window size (w_soft) and temperature (tau)
- Attention projection matrices (W_q, W_k) mutated slowly
- Three experimental conditions:
  A: Fixed-local attention (w = R, no expansion) — free-lunch control
  B: Evolvable attention (w in [R, 32]) — main hypothesis test
  C: V11.4 convolution (FFT) — known baseline

Predictions:
- Condition A: Reproduces V11-like decomposition
- Condition B: Window expands over cycles; Phi increases under moderate stress
  (biological pattern, first time in V11+ experiments)
- Condition C: Anchors comparison (-3% to -6% Phi under stress)
"""

import jax
import jax.numpy as jnp
from jax import random, lax
import numpy as np
import time

from v11_patterns import detect_patterns_mc, PatternTracker
from v11_affect_hd import measure_all_hd
from v11_evolution import score_fitness_functional
from v11_substrate import perturb_resource_bloom

from v12_substrate_attention import (
    generate_attention_config,
    init_attention_params,
    run_chunk_attention_wrapper,
    measure_attention_entropy,
)
from v11_substrate_hd import (
    generate_hd_config, generate_coupling_matrix,
    make_kernels_fft_hd, run_chunk_hd_wrapper, init_soup_hd,
)


# ============================================================================
# Attention Parameter Mutation
# ============================================================================

def mutate_attention_params(W_q, W_k, w_soft, tau, rng,
                            w_min=5.0, w_max=16.0, fixed_window=False):
    """Mutate attention parameters for evolution.

    Args:
        W_q, W_k: projection matrices (d_attn, C)
        w_soft: current soft window radius
        tau: current temperature
        rng: JAX random key
        w_min: minimum window radius
        w_max: maximum window radius
        fixed_window: if True, don't mutate w_soft (Condition A)

    Returns:
        (W_q_new, W_k_new, w_soft_new, tau_new)
    """
    k1, k2, k3, k4 = random.split(rng, 4)

    # Mutate projection matrices (small perturbation)
    W_q_new = W_q + 0.01 * random.normal(k1, W_q.shape)
    W_k_new = W_k + 0.01 * random.normal(k2, W_k.shape)

    # Mutate window size
    if fixed_window:
        w_soft_new = w_soft
    else:
        w_soft_new = w_soft + 0.5 * float(random.normal(k3, ()))
        w_soft_new = float(np.clip(w_soft_new, w_min, w_max))

    # Mutate temperature
    tau_new = tau + 0.1 * float(random.normal(k4, ()))
    tau_new = float(np.clip(tau_new, 0.1, 5.0))

    return W_q_new, W_k_new, w_soft_new, tau_new


# ============================================================================
# V12 Evolution: Attention with Curriculum
# ============================================================================

def evolve_attention(config=None, n_cycles=30, steps_per_cycle=5000,
                     cull_fraction=0.3, mutate_top_n=5,
                     mutation_noise=0.03, seed=42,
                     C=16, N=128, bandwidth=8.0, d_attn=16,
                     w_max=16, fixed_window=False,
                     post_cycle_callback=None):
    """V12: Evolution with attention-based physics and curriculum stress.

    Uses graduated stress schedule from V11.7 as the base protocol.
    Attention parameters (W_q, W_k, w_soft, tau) are mutated alongside
    coupling bandwidth.

    Args:
        fixed_window: if True, w_soft is fixed at kernel R_max (Condition A).
                      If False, w_soft is evolvable (Condition B).
    """


    if config is None:
        config = generate_attention_config(C=C, N=N, seed=seed,
                                            w_max=w_max, d_attn=d_attn)

    N = config['grid_size']
    C = config['n_channels']
    d_attn = config['d_attn']
    rng = random.PRNGKey(seed)
    rng_np = np.random.RandomState(seed + 1000)

    # Coupling matrix (same as V11.4)
    coupling = jnp.array(generate_coupling_matrix(C, bandwidth=bandwidth, seed=seed))

    # Kernel FFTs (needed for affect measurement, not physics)
    kernel_ffts = make_kernels_fft_hd(config)

    # Attention parameters
    rng, k_attn = random.split(rng)
    attn_p = init_attention_params(C, d_attn, seed=seed)
    W_q = attn_p['W_q']
    W_k = attn_p['W_k']
    w_soft = float(config['w_soft'])
    tau = float(config['tau'])

    if fixed_window:
        # Condition A: fix window at max kernel radius
        w_soft = float(np.max(config['kernel_radii']))
        w_min = w_soft
        w_max_evo = w_soft
    else:
        # Condition B: evolvable window
        w_min = 5.0
        w_max_evo = float(w_max)

    # Build stress schedule (from V11.7 curriculum)
    base_schedule = np.linspace(0.5, 0.02, n_cycles)
    noise = 1.0 + 0.3 * rng_np.randn(n_cycles)
    stress_schedule = np.clip(base_schedule * noise, 0.01, 0.8)
    duration_schedule = rng_np.randint(500, 2001, size=n_cycles)

    condition_label = "FIXED-LOCAL" if fixed_window else "EVOLVABLE"

    print("=" * 60)
    print(f"V12 ATTENTION EVOLUTION — {condition_label} (C={C}, N={N})")
    print("=" * 60)
    print(f"  Channels:      {C}")
    print(f"  Grid:          {N}x{N}")
    print(f"  Attn dim:      {d_attn}")
    print(f"  Window:        w_soft={w_soft:.1f}, w_max={config['w_max']}")
    print(f"  Temperature:   tau={tau:.2f}")
    print(f"  Cycles:        {n_cycles}")
    print(f"  Steps/cycle:   {steps_per_cycle}")
    print(f"  Cull fraction: {cull_fraction}")
    print(f"  Bandwidth:     {bandwidth}")
    print(f"  Stress regen:  {stress_schedule[0]:.3f} -> {stress_schedule[-1]:.3f}")
    print()

    # Initialize soup (reuse V11.4 init)
    print("Phase 0: Initializing HD soup for attention physics...")
    rng, k = random.split(rng)
    grid, resource = init_soup_hd(N, C, k, jnp.array(config['channel_mus']))

    # JIT warmup
    print("  JIT compiling attention step...", end=" ", flush=True)
    t0 = time.time()
    grid, resource, rng = run_chunk_attention_wrapper(
        grid, resource, W_q, W_k, coupling, rng, config, 10,
        attn_params={'w_soft': w_soft, 'tau': tau})
    grid.block_until_ready()
    print(f"done ({time.time()-t0:.1f}s)")

    # Warmup run (shorter for small grids to keep CPU tests fast)
    warmup_steps = min(4990, max(500, N * 10))
    print(f"  Running warmup ({warmup_steps} steps)...", end=" ", flush=True)
    t0 = time.time()
    grid, resource, rng = run_chunk_attention_wrapper(
        grid, resource, W_q, W_k, coupling, rng, config, warmup_steps,
        attn_params={'w_soft': w_soft, 'tau': tau})
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

        # This cycle's stress parameters (from V11.7 curriculum)
        cycle_regen = float(stress_schedule[cycle]) * config['resource_regen']
        cycle_drought_steps = int(duration_schedule[cycle])
        cycle_baseline_steps = steps_per_cycle - cycle_drought_steps

        cycle_stress_config = {
            **config,
            'resource_regen': cycle_regen,
            'resource_consume': config['resource_consume'] * (
                1.0 + 0.5 * (1.0 - stress_schedule[cycle])),
        }

        step_base = cycle * steps_per_cycle
        chunk = 10  # smaller chunks for attention (more expensive per step)
        measure_every = max(100, cycle_baseline_steps // 10)

        # ---- BASELINE PHASE ----
        baseline_affects = {}
        baseline_survival = {}

        step = 0
        while step < cycle_baseline_steps:
            grid, resource, rng = run_chunk_attention_wrapper(
                grid, resource, W_q, W_k, coupling, rng, config, chunk,
                attn_params={'w_soft': w_soft, 'tau': tau})
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
            grid, resource, rng = run_chunk_attention_wrapper(
                grid, resource, W_q, W_k, coupling, rng,
                cycle_stress_config, chunk,
                attn_params={'w_soft': w_soft, 'tau': tau})
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
                        config, N,
                        step_num=step_base + cycle_baseline_steps + step,
                        fast=True,
                    )
                    stress_affects[pid].append(affect)
                    prev_masses[pid] = p.mass
                    prev_values[pid] = p.values.copy()

        # ---- MEASURE ATTENTION ENTROPY ----
        grid_np = np.array(grid)
        patterns = detect_patterns_mc(grid_np, threshold=0.15)
        tracker.update(patterns, step=step_base + steps_per_cycle)

        # Attention entropy for diagnostics
        attn_entropies = []
        for p in list(tracker.active.values())[:10]:
            if p.size > 10:
                ent = measure_attention_entropy(
                    grid, W_q, W_k, config, p.cells,
                    tau=tau, w_soft=w_soft)
                attn_entropies.append(ent)
        mean_attn_entropy = float(np.mean(attn_entropies)) if attn_entropies else 0.0

        # ---- SCORE ----
        scored = []
        for p in tracker.active.values():
            pid = p.id
            ba = baseline_affects.get(pid, [])
            sa = stress_affects.get(pid, [])
            surv = baseline_survival.get(pid, 0)

            fitness = score_fitness_functional(ba, sa, surv, steps_per_cycle)

            phi_base = float(np.mean([a.integration for a in ba])) if ba else 0.0
            phi_stress = float(np.mean([a.integration for a in sa])) if sa else phi_base
            robustness = phi_stress / phi_base if phi_base > 1e-6 else 1.0

            scored.append((p, fitness, phi_base, phi_stress, robustness))

        scored.sort(key=lambda x: x[1])

        if not scored:
            print(f"Cycle {cycle+1:>3d}: EXTINCTION — reseeding")
            rng, k = random.split(rng)
            grid, resource = init_soup_hd(
                N, C, k, jnp.array(config['channel_mus']))
            grid, resource, rng = run_chunk_attention_wrapper(
                grid, resource, W_q, W_k, coupling, rng, config, 3000,
                attn_params={'w_soft': w_soft, 'tau': tau})
            tracker = PatternTracker()
            prev_masses = {}
            prev_values = {}
            cycle_stats.append({
                'cycle': cycle + 1, 'n_survived': 0, 'extinction': True})
            continue

        # ---- CULL ----
        n_cull = max(1, int(len(scored) * cull_fraction))
        to_kill = scored[:n_cull]

        kill_mask = np.ones((N, N), dtype=np.float32)
        for p, _, _, _, _ in to_kill:
            kill_mask[p.cells[:, 0], p.cells[:, 1]] = 0.0
        kill_mask_j = jnp.array(kill_mask)
        grid = grid * kill_mask_j[None, :, :]

        # ---- BOOST + MUTATE (grid) ----
        top_patterns = scored[-mutate_top_n:]
        for p, _, _, _, _ in top_patterns:
            cx = int(p.center[1])
            cy = int(p.center[0])
            resource = perturb_resource_bloom(
                resource, (cx, cy), radius=20, intensity=0.3)

        for p, _, _, _, _ in top_patterns:
            rng, k1 = random.split(rng)
            r_min = max(0, p.bbox[0] - 5)
            r_max_b = min(N - 1, p.bbox[1] + 5)
            c_min = max(0, p.bbox[2] - 5)
            c_max_b = min(N - 1, p.bbox[3] + 5)
            h = r_max_b - r_min + 1
            w = c_max_b - c_min + 1
            noise_grid = mutation_noise * random.normal(k1, (C, h, w))
            region = grid[:, r_min:r_max_b+1, c_min:c_max_b+1]
            grid = grid.at[:, r_min:r_max_b+1, c_min:c_max_b+1].set(
                jnp.clip(region + noise_grid, 0.0, 1.0))

        # ---- MUTATE coupling ----
        rng, k_bw = random.split(rng)
        bandwidth = bandwidth + 0.5 * float(random.normal(k_bw, ()))
        bandwidth = max(2.0, min(C / 2, bandwidth))
        coupling = jnp.array(
            generate_coupling_matrix(C, bandwidth=bandwidth,
                                            seed=seed + cycle + 1))

        # ---- MUTATE attention parameters ----
        rng, k_attn = random.split(rng)
        W_q, W_k, w_soft, tau = mutate_attention_params(
            W_q, W_k, w_soft, tau, k_attn,
            w_min=w_min if not fixed_window else w_soft,
            w_max=w_max_evo,
            fixed_window=fixed_window)

        # ---- Restore resources ----
        resource = jnp.clip(
            resource + 0.1 * (config['resource_max'] - resource),
            0.0, config['resource_max'])

        elapsed = time.time() - t0

        all_fits = [f for _, f, _, _, _ in scored]
        all_phi_base = [pb for _, _, pb, _, _ in scored]
        all_phi_stress = [ps for _, _, _, ps, _ in scored]
        all_robust = [r for _, _, _, _, r in scored]

        stats = {
            'cycle': cycle + 1,
            'n_patterns': len(scored),
            'n_culled': n_cull,
            'mean_fitness': float(np.mean(all_fits)),
            'max_fitness': float(np.max(all_fits)),
            'mean_phi_base': float(np.mean(all_phi_base)),
            'mean_phi_stress': float(np.mean(all_phi_stress)),
            'mean_robustness': float(np.mean(all_robust)),
            'bandwidth': bandwidth,
            'w_soft': w_soft,
            'tau': tau,
            'mean_attn_entropy': mean_attn_entropy,
            'stress_regen': float(cycle_regen),
            'drought_steps': cycle_drought_steps,
            'elapsed': elapsed,
        }
        cycle_stats.append(stats)

        phi_delta = (stats['mean_phi_stress'] - stats['mean_phi_base']) / (
            stats['mean_phi_base'] + 1e-10)
        print(f"Cycle {cycle+1:>3d}/{n_cycles}: "
              f"n={len(scored):>3d} (-{n_cull}), "
              f"Phi={stats['mean_phi_base']:.4f}/{stats['mean_phi_stress']:.4f}"
              f"({phi_delta:+.0%}), "
              f"w={w_soft:.1f}, tau={tau:.2f}, "
              f"H_attn={mean_attn_entropy:.2f}, "
              f"({elapsed:.1f}s)", flush=True)

        if post_cycle_callback:
            post_cycle_callback(cycle_stats)

    print()
    print("=" * 60)
    print(f"V12 ATTENTION EVOLUTION COMPLETE — {condition_label} (C={C})")
    print("=" * 60)
    if len(cycle_stats) >= 2:
        first = next((s for s in cycle_stats if 'mean_phi_base' in s), None)
        last = next((s for s in reversed(cycle_stats) if 'mean_phi_base' in s), None)
        if first and last:
            print(f"  Phi (base):    {first['mean_phi_base']:.4f} -> {last['mean_phi_base']:.4f}")
            print(f"  Phi (stress):  {first['mean_phi_stress']:.4f} -> {last['mean_phi_stress']:.4f}")
            print(f"  Robustness:    {first['mean_robustness']:.3f} -> {last['mean_robustness']:.3f}")
            print(f"  Window:        {first['w_soft']:.1f} -> {last['w_soft']:.1f}")
            print(f"  Temperature:   {first['tau']:.2f} -> {last['tau']:.2f}")
            print(f"  Attn entropy:  {first['mean_attn_entropy']:.2f} -> {last['mean_attn_entropy']:.2f}")
            print(f"  Count:         {first['n_patterns']} -> {last['n_patterns']}")

    return {
        'cycle_stats': cycle_stats,
        'final_grid': grid,
        'final_resource': resource,
        'coupling': coupling,
        'config': config,
        'bandwidth': bandwidth,
        'W_q': np.array(W_q),
        'W_k': np.array(W_k),
        'w_soft': w_soft,
        'tau': tau,
        'stress_schedule': stress_schedule.tolist(),
        'duration_schedule': duration_schedule.tolist(),
        'condition': 'A_fixed' if fixed_window else 'B_evolvable',
    }


# ============================================================================
# Stress Test: Compare Conditions A, B, C
# ============================================================================

def stress_test_attention(evolved_grid, evolved_resource, evolved_coupling,
                          W_q, W_k, w_soft, tau,
                          config=None, seed=99, C=16, N=128,
                          bandwidth=8.0, fixed_window=False):
    """Stress test evolved attention patterns vs naive.

    Phases: baseline -> mild_drought -> severe_drought -> recovery
    """


    if config is None:
        config = generate_attention_config(C=C, N=N, seed=seed)

    N = config['grid_size']
    C = config['n_channels']
    rng = random.PRNGKey(seed)

    kernel_ffts = make_kernels_fft_hd(config)
    naive_coupling = jnp.array(
        generate_coupling_matrix(C, bandwidth=bandwidth, seed=seed + 100))

    # Naive attention params
    naive_attn = init_attention_params(C, config['d_attn'], seed=seed + 200)

    condition_label = "FIXED-LOCAL" if fixed_window else "EVOLVABLE"
    print("\n" + "=" * 60)
    print(f"V12 STRESS TEST — {condition_label} (C={C})")
    print("=" * 60)

    # Initialize naive grid
    rng, k = random.split(rng)
    naive_grid, naive_resource = init_soup_hd(
        N, C, k, jnp.array(config['channel_mus']))
    # Warmup naive
    naive_grid, naive_resource, rng = run_chunk_attention_wrapper(
        naive_grid, naive_resource,
        naive_attn['W_q'], naive_attn['W_k'],
        naive_coupling, rng, config, 5000,
        attn_params={'w_soft': float(np.max(config['kernel_radii'])),
                     'tau': 1.0})

    phases = [
        ('baseline', config, 2000),
        ('mild_drought', {**config, 'resource_regen': 0.002}, 2000),
        ('severe_drought', {**config, 'resource_regen': 0.0005}, 2000),
        ('recovery', config, 2000),
    ]

    results = {'evolved': [], 'naive': []}

    for condition_idx, (grid, res, coup, wq, wk, ws, t, label) in enumerate([
        (evolved_grid, evolved_resource, evolved_coupling,
         W_q, W_k, w_soft, tau, 'evolved'),
        (naive_grid, naive_resource, naive_coupling,
         naive_attn['W_q'], naive_attn['W_k'],
         float(np.max(config['kernel_radii'])), 1.0, 'naive'),
    ]):
        print(f"\n  Testing {label} (w={ws:.1f}, tau={t:.2f})...")
        g, r = jnp.array(grid), jnp.array(res)
        rng_test = random.PRNGKey(seed + 42 + condition_idx)

        for phase_name, phase_config, phase_steps in phases:
            chunk = 10
            step = 0
            phase_affects = []
            phase_entropies = []

            while step < phase_steps:
                g, r, rng_test = run_chunk_attention_wrapper(
                    g, r, wq, wk, coup, rng_test, phase_config, chunk,
                    attn_params={'w_soft': ws, 'tau': t})
                step += chunk

                if step % 500 < chunk:
                    g_np = np.array(g)
                    patterns = detect_patterns_mc(g_np, threshold=0.15)
                    for p in patterns[:20]:
                        if p.size < 10:
                            continue
                        affect, _, _, _, _ = measure_all_hd(
                            p, None, None, [],
                            jnp.array(g_np), kernel_ffts, coup,
                            config, N, fast=True)
                        phase_affects.append(affect)

                        ent = measure_attention_entropy(
                            g, wq, wk, config, p.cells, tau=t, w_soft=ws)
                        phase_entropies.append(ent)

            if phase_affects:
                phase_data = {
                    'phase': phase_name,
                    'mean_phi': float(np.mean([a.integration for a in phase_affects])),
                    'mean_arousal': float(np.mean([a.arousal for a in phase_affects])),
                    'mean_eff_rank': float(np.mean([a.effective_rank for a in phase_affects])),
                    'mean_attn_entropy': float(np.mean(phase_entropies)) if phase_entropies else 0.0,
                    'n_patterns': len(set(a.pattern_id for a in phase_affects)),
                }
                results[label].append(phase_data)
                print(f"    {phase_name:15s}: Phi={phase_data['mean_phi']:.4f}, "
                      f"H_attn={phase_data['mean_attn_entropy']:.2f}, "
                      f"n={phase_data['n_patterns']}")

    # Summary comparison
    print("\n  COMPARISON (Phi):")
    for phase_name in ['baseline', 'mild_drought', 'severe_drought', 'recovery']:
        evo_p = next((p for p in results['evolved'] if p['phase'] == phase_name), None)
        naive_p = next((p for p in results['naive'] if p['phase'] == phase_name), None)
        if evo_p and naive_p:
            print(f"    {phase_name:15s}: evo={evo_p['mean_phi']:.4f} "
                  f"vs naive={naive_p['mean_phi']:.4f}")

    # Compute key metrics
    evo_baseline = next((p for p in results['evolved'] if p['phase'] == 'baseline'), None)
    evo_severe = next((p for p in results['evolved'] if p['phase'] == 'severe_drought'), None)
    naive_baseline = next((p for p in results['naive'] if p['phase'] == 'baseline'), None)
    naive_severe = next((p for p in results['naive'] if p['phase'] == 'severe_drought'), None)

    comparison = {
        'condition': 'A_fixed' if fixed_window else 'B_evolvable',
    }

    if evo_baseline and evo_severe:
        evo_delta = (evo_severe['mean_phi'] - evo_baseline['mean_phi']) / (
            evo_baseline['mean_phi'] + 1e-10)
        comparison['evolved_phi_delta'] = float(evo_delta)
        comparison['evolved_phi_ratio'] = float(
            evo_severe['mean_phi'] / (evo_baseline['mean_phi'] + 1e-10))
        comparison['biological_shift'] = evo_delta > 0  # THE KEY TEST

    if naive_baseline and naive_severe:
        naive_delta = (naive_severe['mean_phi'] - naive_baseline['mean_phi']) / (
            naive_baseline['mean_phi'] + 1e-10)
        comparison['naive_phi_delta'] = float(naive_delta)

    if evo_baseline and naive_baseline and evo_severe and naive_severe:
        comparison['shift'] = float(
            (comparison.get('evolved_phi_delta', 0) -
             comparison.get('naive_phi_delta', 0)))

    print(f"\n  KEY RESULT: evolved Phi delta = "
          f"{comparison.get('evolved_phi_delta', 0):+.1%}")
    if comparison.get('biological_shift', False):
        print("  *** BIOLOGICAL PATTERN DETECTED: Phi INCREASED under stress ***")
    else:
        print("  (Still decomposition under stress)")

    return {
        'evolved': results['evolved'],
        'naive': results['naive'],
        'comparison': comparison,
    }


# ============================================================================
# Full Pipeline: Evolve -> Stress Test
# ============================================================================

def full_pipeline_attention(config=None, n_cycles=30, steps_per_cycle=5000,
                            cull_fraction=0.3, seed=42,
                            C=16, N=128, bandwidth=8.0,
                            d_attn=16, w_max=16, fixed_window=False,
                            post_cycle_callback=None):
    """V12 full pipeline: attention evolution -> stress test."""
    if config is None:
        config = generate_attention_config(C=C, N=N, seed=seed,
                                            w_max=w_max, d_attn=d_attn)

    t_start = time.time()
    condition = "A (fixed-local)" if fixed_window else "B (evolvable)"
    print()
    print("+" + "=" * 58 + "+")
    print(f"| V12 PIPELINE: Condition {condition}, C={C}, N={N}" +
          " " * max(0, 15 - len(condition) - len(str(C)) - len(str(N))) + "|")
    print("+" + "=" * 58 + "+")

    evo_result = evolve_attention(
        config=config, n_cycles=n_cycles,
        steps_per_cycle=steps_per_cycle, cull_fraction=cull_fraction,
        seed=seed, C=C, N=N, bandwidth=bandwidth,
        d_attn=d_attn, w_max=w_max, fixed_window=fixed_window,
        post_cycle_callback=post_cycle_callback)

    stress_result = stress_test_attention(
        evo_result['final_grid'], evo_result['final_resource'],
        evo_result['coupling'],
        jnp.array(evo_result['W_q']), jnp.array(evo_result['W_k']),
        evo_result['w_soft'], evo_result['tau'],
        config=config, seed=seed + 2, C=C, N=N,
        bandwidth=evo_result['bandwidth'],
        fixed_window=fixed_window)

    elapsed = time.time() - t_start
    print(f"\nTotal pipeline time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    return {'evolution': evo_result, 'stress_test': stress_result}


# ============================================================================
# V11.4 Baseline (Condition C) — wrapper for comparison
# ============================================================================

def full_pipeline_convolution_baseline(config=None, n_cycles=30,
                                       steps_per_cycle=5000,
                                       cull_fraction=0.3, seed=42,
                                       C=16, N=128, bandwidth=8.0,
                                       post_cycle_callback=None):
    """Run V11.4 convolution baseline for condition C comparison.

    Wraps the existing V11.7 curriculum pipeline.
    """
    from v11_evolution import full_pipeline_curriculum


    if config is None:
        config = generate_hd_config(C=C, N=N, seed=seed)

    print()
    print("+" + "=" * 58 + "+")
    print(f"| V12 BASELINE: Condition C (FFT convolution), C={C}, N={N}" +
          " " * max(0, 5 - len(str(C)) - len(str(N))) + "|")
    print("+" + "=" * 58 + "+")

    return full_pipeline_curriculum(
        config=config, n_cycles=n_cycles,
        steps_per_cycle=steps_per_cycle, cull_fraction=cull_fraction,
        seed=seed, C=C, bandwidth=bandwidth,
        post_cycle_callback=post_cycle_callback)
