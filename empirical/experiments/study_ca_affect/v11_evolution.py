"""V11.1 Evolution: Natural selection for integration.

V11.0 showed CA patterns decompose under threat — same zombie dynamics
as LLMs. The thesis predicts this is because neither has developmental
history where integration was selected for survival.

V11.1 tests this directly with in-situ evolution:
- Patterns live on a persistent grid (no extract/replant)
- Selection: periodically score patterns, cull low-fitness ones
- Reproduction: survivors grow into freed space (natural Lenia budding)
- Mutation: noise injection near winners creates heritable variation
- After N cycles, stress-test evolved vs naive patterns

THE prediction: evolved patterns show Phi INCREASE under threat

Tricks:
1. Solve backwards: JAX autodiff to find high-Phi initial conditions
2. Curriculum: gradually increase resource pressure across cycles
3. In-situ selection: biologically realistic, no placement artifacts
"""

import jax
import jax.numpy as jnp
from jax import random, lax
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
import time

from v11_substrate import (
    make_kernel, make_kernel_fft, growth_fn,
    run_chunk, make_params, DEFAULT_CONFIG, init_soup,
    perturb_resource_bloom,
    init_param_fields, diffuse_params, mutate_param_fields,
    MULTICHANNEL_CONFIG, make_kernels_fft, run_chunk_mc_wrapper, init_soup_mc,
)
from v11_patterns import detect_patterns, detect_patterns_mc, PatternTracker
from v11_affect import measure_all, measure_all_mc

# Lazy imports for HD (V11.4) to avoid ImportError when not using HD mode
def _import_hd():
    from v11_substrate_hd import (
        generate_hd_config, HD_CONFIG, generate_coupling_matrix,
        make_kernels_fft_hd, run_chunk_hd_wrapper, init_soup_hd,
    )
    from v11_affect_hd import measure_all_hd
    return {
        'generate_hd_config': generate_hd_config,
        'HD_CONFIG': HD_CONFIG,
        'generate_coupling_matrix': generate_coupling_matrix,
        'make_kernels_fft_hd': make_kernels_fft_hd,
        'run_chunk_hd_wrapper': run_chunk_hd_wrapper,
        'init_soup_hd': init_soup_hd,
        'measure_all_hd': measure_all_hd,
    }


# ============================================================================
# Genotype: for discrete-generation mode and seed discovery
# ============================================================================

@dataclass
class Genotype:
    """Portable representation of a pattern's structure."""
    template: np.ndarray
    fitness: float = 0.0
    generation: int = 0
    lineage_id: int = -1
    parent_id: int = -1
    survival_steps: int = 0
    mean_phi: float = 0.0
    mean_mass: float = 0.0


def extract_genotype(grid_np, pattern, margin=13):
    """Extract pattern structure from grid. Margin=kernel_radius for context."""
    cells = pattern.cells
    r_min, c_min = cells.min(axis=0)
    r_max, c_max = cells.max(axis=0)

    N = grid_np.shape[0]
    r_min = max(0, r_min - margin)
    c_min = max(0, c_min - margin)
    r_max = min(N - 1, r_max + margin)
    c_max = min(N - 1, c_max + margin)

    template = grid_np[r_min:r_max+1, c_min:c_max+1].copy()
    if template.shape[0] < 4 or template.shape[1] < 4:
        pad_r = max(0, 4 - template.shape[0])
        pad_c = max(0, 4 - template.shape[1])
        template = np.pad(template, ((0, pad_r), (0, pad_c)))

    return Genotype(template=template, mean_mass=float(pattern.mass))


def mutate(genotype, rng, noise_scale=0.02):
    """Heritable variation: noise + random transform."""
    k1, k2 = random.split(rng)
    t = jnp.array(genotype.template)
    t = t + noise_scale * random.normal(k1, t.shape)

    transform = int(random.randint(k2, (), 0, 8))
    if transform == 1:
        t = jnp.rot90(t, 1)
    elif transform == 2:
        t = jnp.rot90(t, 2)
    elif transform == 3:
        t = jnp.rot90(t, 3)
    elif transform == 4:
        t = jnp.flip(t, 0)
    elif transform == 5:
        t = jnp.flip(t, 1)

    t = jnp.clip(t, 0.0, 1.0)
    return Genotype(
        template=np.array(t),
        generation=genotype.generation + 1,
        lineage_id=genotype.lineage_id,
        parent_id=id(genotype),
    )


def place_on_grid(template, grid, r0, c0):
    """Place template onto toroidal grid via roll trick."""
    N = grid.shape[0]
    h, w = template.shape
    h, w = min(h, N), min(w, N)
    patch = jnp.zeros((N, N))
    patch = patch.at[:h, :w].set(jnp.array(template[:h, :w]))
    patch = jnp.roll(patch, (int(r0), int(c0)), axis=(0, 1))
    return jnp.maximum(grid, patch)


# ============================================================================
# Fitness
# ============================================================================

def score_fitness(survival_steps, total_steps, affect_history):
    """Fitness = survival * (1 + mean_phi) * log(mass + 1)."""
    surv = survival_steps / max(total_steps, 1)
    if not affect_history:
        return surv
    phis = [a.integration for a in affect_history]
    masses = [a.mass for a in affect_history]
    return float(surv * (1.0 + np.mean(phis)) * np.log1p(np.mean(masses)))


def score_fitness_functional(baseline_affects, stress_affects,
                             survival_steps, total_steps):
    """Fitness that selects for FUNCTIONAL integration.

    Functional = Phi that holds up (or increases) under threat.
    This directly selects for the biological pattern.

    fitness = survival * phi_robustness * (1 + phi_base) * log(mass)

    phi_robustness = phi_stress / phi_base
    > 1 means integration under threat (biological)
    < 1 means decomposition under threat (zombie)
    """
    surv = survival_steps / max(total_steps, 1)

    all_affects = baseline_affects + stress_affects
    if not all_affects:
        return surv

    masses = [a.mass for a in all_affects]

    phi_base = np.mean([a.integration for a in baseline_affects]) if baseline_affects else 0.0
    phi_stress = np.mean([a.integration for a in stress_affects]) if stress_affects else phi_base

    # Phi robustness: how well does integration hold under stress?
    if phi_base > 1e-6:
        phi_robustness = phi_stress / phi_base
    else:
        phi_robustness = 1.0

    # Clamp to avoid extreme values from noisy small patterns
    phi_robustness = np.clip(phi_robustness, 0.1, 3.0)

    return float(surv * phi_robustness * (1.0 + phi_base) * np.log1p(np.mean(masses)))


# ============================================================================
# IN-SITU EVOLUTION (primary method)
# ============================================================================

def evolve_in_situ(config=None, n_cycles=30, steps_per_cycle=5000,
                   cull_fraction=0.3, mutate_top_n=5,
                   mutation_noise=0.03, seed=42, curriculum=False):
    """In-situ evolution selecting for FUNCTIONAL integration.

    Each cycle has two phases:
    1. BASELINE phase (60% of steps, normal resources) — measure Phi_base
    2. STRESS phase (40% of steps, reduced resources) — measure Phi_stress

    Fitness = survival * phi_robustness * (1 + phi_base) * log(mass)
    where phi_robustness = phi_stress / phi_base

    This directly selects for patterns whose integration HOLDS UP
    (or increases) under stress — the biological pattern.

    Then: cull bottom fraction, boost/mutate winners, continue.
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    N = config['grid_size']
    rng = random.PRNGKey(seed)

    kernel = make_kernel(
        config['kernel_radius'], config['kernel_peak'], config['kernel_width'])
    kernel_fft = make_kernel_fft(kernel, N)
    params = make_params(config)
    stress_params = make_params({**config, 'resource_regen': 0.001})

    print("=" * 60)
    print("V11.1 IN-SITU EVOLUTION (functional selection)")
    print("=" * 60)
    print(f"  Cycles:        {n_cycles}")
    print(f"  Steps/cycle:   {steps_per_cycle}")
    print(f"  Cull fraction: {cull_fraction}")
    print(f"  Mutate top:    {mutate_top_n}")
    print(f"  Curriculum:    {curriculum}")
    print(f"  Selection:     phi_robustness (phi_stress / phi_base)")
    print()

    # Initialize from random soup + warmup
    print("Phase 0: Initializing from random soup...")
    rng, k = random.split(rng)
    grid, resource = init_soup(N, k, n_seeds=80, growth_mu=config['growth_mu'])
    grid, resource, rng = run_chunk(
        grid, resource, kernel_fft, rng, params, 5000)
    grid.block_until_ready()

    grid_np = np.array(grid)
    initial_patterns = detect_patterns(grid_np, threshold=0.1)
    print(f"  {len(initial_patterns)} patterns after warmup\n")

    tracker = PatternTracker()
    prev_masses = {}
    prev_values = {}
    cycle_stats = []

    baseline_steps = int(steps_per_cycle * 0.6)
    stress_steps = steps_per_cycle - baseline_steps

    for cycle in range(n_cycles):
        t0 = time.time()

        # Curriculum: increase stress severity over time
        cycle_stress_params = stress_params
        if curriculum:
            difficulty = cycle / max(n_cycles - 1, 1)
            stress_config = config.copy()
            stress_config['resource_regen'] = 0.001 * (1.0 - 0.8 * difficulty)
            stress_config['resource_consume'] = config['resource_consume'] * (
                1.0 + difficulty)
            cycle_stress_params = make_params(stress_config)

        step_base = cycle * steps_per_cycle
        chunk = 100
        measure_every = max(200, baseline_steps // 10)

        # ---- BASELINE PHASE ----
        baseline_affects = {}  # pid -> [AffectState]
        baseline_survival = {}

        step = 0
        while step < baseline_steps:
            grid, resource, rng = run_chunk(
                grid, resource, kernel_fft, rng, params, chunk)
            step += chunk

            if step % measure_every < chunk:
                grid_np = np.array(grid)
                patterns = detect_patterns(grid_np, threshold=0.1)
                tracker.update(patterns, step=step_base + step)

                for p in tracker.active.values():
                    pid = p.id
                    if pid not in baseline_affects:
                        baseline_affects[pid] = []
                    baseline_survival[pid] = step

                    hist = tracker.history.get(pid, [])
                    pm = prev_masses.get(pid)
                    pv = prev_values.get(pid)

                    affect = measure_all(
                        p, pm, pv, hist,
                        grid, kernel_fft,
                        config['growth_mu'], config['growth_sigma'], N,
                        step_num=step_base + step,
                    )
                    baseline_affects[pid].append(affect)
                    prev_masses[pid] = p.mass
                    prev_values[pid] = p.values.copy()

        # ---- STRESS PHASE ----
        stress_affects = {}  # pid -> [AffectState]

        measure_every_stress = max(200, stress_steps // 8)
        step = 0
        while step < stress_steps:
            grid, resource, rng = run_chunk(
                grid, resource, kernel_fft, rng, cycle_stress_params, chunk)
            step += chunk

            if step % measure_every_stress < chunk:
                grid_np = np.array(grid)
                patterns = detect_patterns(grid_np, threshold=0.1)
                tracker.update(patterns, step=step_base + baseline_steps + step)

                for p in tracker.active.values():
                    pid = p.id
                    if pid not in stress_affects:
                        stress_affects[pid] = []
                    # Track survival through stress
                    if pid in baseline_survival:
                        baseline_survival[pid] = baseline_steps + step

                    hist = tracker.history.get(pid, [])
                    pm = prev_masses.get(pid)
                    pv = prev_values.get(pid)

                    affect = measure_all(
                        p, pm, pv, hist,
                        grid, kernel_fft,
                        config['growth_mu'], config['growth_sigma'], N,
                        step_num=step_base + baseline_steps + step,
                    )
                    stress_affects[pid].append(affect)
                    prev_masses[pid] = p.mass
                    prev_values[pid] = p.values.copy()

        # ---- SCORE with functional fitness ----
        grid_np = np.array(grid)
        patterns = detect_patterns(grid_np, threshold=0.1)
        tracker.update(patterns, step=step_base + steps_per_cycle)

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

        scored.sort(key=lambda x: x[1])  # ascending fitness

        if not scored:
            print(f"Cycle {cycle+1:>3d}: EXTINCTION — reseeding")
            rng, k = random.split(rng)
            grid, resource = init_soup(
                N, k, n_seeds=80, growth_mu=config['growth_mu'])
            grid, resource, rng = run_chunk(
                grid, resource, kernel_fft, rng, params, 3000)
            tracker = PatternTracker()
            prev_masses = {}
            prev_values = {}
            cycle_stats.append({
                'cycle': cycle + 1, 'n_survived': 0, 'extinction': True})
            continue

        # ---- CULL bottom fraction ----
        n_cull = max(1, int(len(scored) * cull_fraction))
        to_kill = scored[:n_cull]

        kill_mask = np.ones((N, N), dtype=np.float32)
        for p, _, _, _, _ in to_kill:
            kill_mask[p.cells[:, 0], p.cells[:, 1]] = 0.0
        grid = grid * jnp.array(kill_mask)

        # ---- BOOST resources near top patterns ----
        top_patterns = scored[-mutate_top_n:]
        for p, _, _, _, _ in top_patterns:
            cx = int(p.center[1])
            cy = int(p.center[0])
            resource = perturb_resource_bloom(
                resource, (cx, cy), radius=20, intensity=0.3)

        # ---- MUTATE near top patterns ----
        for p, _, _, _, _ in top_patterns:
            rng, k = random.split(rng)
            r_min = max(0, p.bbox[0] - 5)
            r_max = min(N - 1, p.bbox[1] + 5)
            c_min = max(0, p.bbox[2] - 5)
            c_max = min(N - 1, p.bbox[3] + 5)
            h = r_max - r_min + 1
            w = c_max - c_min + 1
            noise = mutation_noise * random.normal(k, (h, w))
            region = grid[r_min:r_max+1, c_min:c_max+1]
            grid = grid.at[r_min:r_max+1, c_min:c_max+1].set(
                jnp.clip(region + noise, 0.0, 1.0))

        # ---- Restore resources for next cycle ----
        resource = jnp.clip(
            resource + 0.1 * (config['resource_max'] - resource),
            0.0, config['resource_max'])

        elapsed = time.time() - t0

        # Stats
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
            'elapsed': elapsed,
        }
        cycle_stats.append(stats)

        phi_delta = (stats['mean_phi_stress'] - stats['mean_phi_base']) / (
            stats['mean_phi_base'] + 1e-10)
        print(f"Cycle {cycle+1:>3d}/{n_cycles}: "
              f"n={len(scored):>3d} (-{n_cull}), "
              f"Phi_base={stats['mean_phi_base']:.4f}, "
              f"Phi_stress={stats['mean_phi_stress']:.4f} ({phi_delta:+.1%}), "
              f"robustness={stats['mean_robustness']:.3f}, "
              f"({elapsed:.1f}s)")

    print()
    print("=" * 60)
    print("IN-SITU EVOLUTION COMPLETE")
    print("=" * 60)
    if len(cycle_stats) >= 2:
        first = next((s for s in cycle_stats if 'mean_phi_base' in s), None)
        last = next((s for s in reversed(cycle_stats) if 'mean_phi_base' in s), None)
        if first and last:
            print(f"  Phi (base):   {first['mean_phi_base']:.4f} -> {last['mean_phi_base']:.4f}")
            print(f"  Phi (stress): {first['mean_phi_stress']:.4f} -> {last['mean_phi_stress']:.4f}")
            print(f"  Robustness:   {first['mean_robustness']:.3f} -> {last['mean_robustness']:.3f}")
            print(f"  Count:        {first['n_patterns']} -> {last['n_patterns']}")

    return {
        'cycle_stats': cycle_stats,
        'final_grid': grid,
        'final_resource': resource,
        'config': config,
    }


# ============================================================================
# Stress test: THE critical comparison
# ============================================================================

def stress_test(evolved_grid, evolved_resource, config=None, seed=99):
    """Compare evolved population vs naive soup under drought.

    THE thesis test: does evolutionary history change the Phi response?

    Biological: Phi increases under threat (integration serves survival)
    Zombie:     Phi decreases under threat (V11.0, LLMs)

    For in-situ evolution, we use the evolved grid directly —
    no extraction/placement needed. Naive control is fresh soup.
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    N = config['grid_size']
    rng = random.PRNGKey(seed)

    kernel = make_kernel(
        config['kernel_radius'], config['kernel_peak'], config['kernel_width'])
    kernel_fft = make_kernel_fft(kernel, N)
    params = make_params(config)
    drought_params = make_params({**config, 'resource_regen': 0.0001})

    print("\n" + "=" * 60)
    print("STRESS TEST: Evolved vs Naive Under Drought")
    print("=" * 60)

    results = {}

    for condition in ['evolved', 'naive']:
        print(f"\n  [{condition.upper()}]")
        rng, k = random.split(rng)

        if condition == 'naive':
            grid, resource = init_soup(
                N, k, n_seeds=80, growth_mu=config['growth_mu'])
            grid, resource, rng = run_chunk(
                grid, resource, kernel_fft, rng, params, 3000)
        else:
            # Use evolved grid directly — no extraction artifacts
            grid = evolved_grid
            resource = evolved_resource
            # Short equilibration to clear any mutation noise
            grid, resource, rng = run_chunk(
                grid, resource, kernel_fft, rng, params, 500)

        # Run three phases, threading state through
        phase_data = {}
        for phase_name, phase_params, phase_steps in [
            ('baseline', params, 1500),
            ('drought',  drought_params, 3000),
            ('recovery', params, 1500),
        ]:
            tracker = PatternTracker()
            prev_masses = {}
            prev_values = {}
            measurements = []

            step = 0
            chunk = 100
            while step < phase_steps:
                grid, resource, rng = run_chunk(
                    grid, resource, kernel_fft, rng, phase_params, chunk)
                step += chunk

                if step % 200 < chunk:
                    grid_np = np.array(grid)
                    patterns = detect_patterns(grid_np, threshold=0.1)
                    tracker.update(patterns, step=step)

                    phis, arousals, valences = [], [], []
                    for p in tracker.active.values():
                        hist = tracker.history.get(p.id, [])
                        pm = prev_masses.get(p.id)
                        pv = prev_values.get(p.id)
                        affect = measure_all(
                            p, pm, pv, hist,
                            grid, kernel_fft,
                            config['growth_mu'], config['growth_sigma'], N,
                            step_num=step,
                        )
                        phis.append(affect.integration)
                        arousals.append(affect.arousal)
                        valences.append(affect.valence)
                        prev_masses[p.id] = p.mass
                        prev_values[p.id] = p.values.copy()

                    if phis:
                        measurements.append({
                            'step': step,
                            'phi': float(np.mean(phis)),
                            'arousal': float(np.mean(arousals)),
                            'valence': float(np.mean(valences)),
                            'n_patterns': len(phis),
                        })

            phase_data[phase_name] = measurements
            if measurements:
                mp = np.mean([m['phi'] for m in measurements])
                ma = np.mean([m['arousal'] for m in measurements])
                mv = np.mean([m['valence'] for m in measurements])
                nn = np.mean([m['n_patterns'] for m in measurements])
                print(f"    {phase_name:>10s}: Phi={mp:.4f}, A={ma:.4f}, "
                      f"V={mv:+.4f}, n={nn:.0f}")

        results[condition] = phase_data

    # THE comparison
    def phase_phi(data, phase):
        ms = data.get(phase, [])
        return np.mean([m['phi'] for m in ms]) if ms else 0.0

    e_base = phase_phi(results.get('evolved', {}), 'baseline')
    e_drought = phase_phi(results.get('evolved', {}), 'drought')
    n_base = phase_phi(results.get('naive', {}), 'baseline')
    n_drought = phase_phi(results.get('naive', {}), 'drought')

    e_delta = (e_drought - e_base) / (e_base + 1e-10)
    n_delta = (n_drought - n_base) / (n_base + 1e-10)

    print(f"\n  {'=' * 50}")
    print(f"  CRITICAL RESULT")
    print(f"  {'=' * 50}")
    print(f"  Naive   Phi change under drought: {n_delta:+.1%}")
    print(f"  Evolved Phi change under drought: {e_delta:+.1%}")

    if e_delta > n_delta:
        print(f"  -> Evolution shifted Phi toward biological pattern!")
        if e_delta > 0:
            print(f"  -> INTEGRATION UNDER THREAT ACHIEVED")
        else:
            print(f"  -> Still decomposing, but less than naive")
    else:
        print(f"  -> Evolution did not shift Phi response")
        print(f"  -> May need: longer evolution, heterogeneous chemistry,")
        print(f"     or different selection pressure")

    results['comparison'] = {
        'naive_phi_delta': float(n_delta),
        'evolved_phi_delta': float(e_delta),
        'shift': float(e_delta - n_delta),
        'biological_shift': e_delta > n_delta,
        'integration_under_threat': e_delta > 0,
    }

    return results


# ============================================================================
# Trick: Differentiable seed discovery
# ============================================================================

def discover_seeds(config=None, n_seeds=8, opt_steps=80,
                   horizon=50, small_N=64, seed=42):
    """Solve backwards: gradient-ascend initial conditions for fitness.

    Uses JAX autodiff through differentiable Lenia to find patterns
    that are naturally high-Phi and long-surviving.

    Runs on small grid (64x64) for tractability, then templates
    can be placed on the full grid.
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    rng = random.PRNGKey(seed)
    N = small_N

    kernel = make_kernel(
        config['kernel_radius'], config['kernel_peak'], config['kernel_width'])
    kfft = make_kernel_fft(kernel, N)

    mu = jnp.float32(config['growth_mu'])
    sigma = jnp.float32(config['growth_sigma'])
    dt = jnp.float32(config['dt'])
    rc = jnp.float32(config['resource_consume'])
    rr = jnp.float32(config['resource_regen'])
    rm = jnp.float32(config['resource_max'])
    rh = jnp.float32(config['resource_half_sat'])

    print("=" * 60)
    print("DIFFERENTIABLE SEED DISCOVERY")
    print("=" * 60)
    print(f"  Grid: {N}x{N}, Horizon: {horizon} steps")
    print(f"  Seeds: {n_seeds}, Opt steps: {opt_steps}")
    print()

    patch_size = 24

    def fitness_fn(init_patch):
        """Differentiable fitness through Lenia dynamics."""
        grid = jnp.zeros((N, N))
        r0 = N // 2 - patch_size // 2
        c0 = N // 2 - patch_size // 2
        grid = lax.dynamic_update_slice(
            grid, jnp.clip(init_patch, 0.0, 1.0), (r0, c0))
        resource = jnp.full((N, N), 0.8)

        def step_fn(carry, _):
            g, r = carry
            pot = jnp.fft.irfft2(jnp.fft.rfft2(g) * kfft, s=(N, N))
            growth = growth_fn(pot, mu, sigma)
            rf = r / (r + rh)
            growth = jnp.where(growth > 0, growth * rf, growth)
            g_new = jnp.clip(g + dt * growth, 0.0, 1.0)
            r_new = jnp.clip(
                r - rc * g * r * dt + rr * (rm - r) * dt, 0.0, rm)
            return (g_new, r_new), g_new

        (final, _), traj = lax.scan(
            step_fn, (grid, resource), None, length=horizon)

        mass = jnp.sum(final)

        # Integration proxy: cross-partition influence
        pot_full = jnp.fft.irfft2(jnp.fft.rfft2(final) * kfft, s=(N, N))
        right_only = final.at[:, :N//2].set(0.0)
        pot_right = jnp.fft.irfft2(
            jnp.fft.rfft2(right_only) * kfft, s=(N, N))
        g_full = growth_fn(pot_full, mu, sigma)
        g_without_right = growth_fn(pot_full - pot_right, mu, sigma)
        phi = jnp.sum((g_full - g_without_right) ** 2)

        # Activity
        diffs = traj[1:] - traj[:-1]
        activity = jnp.mean(jnp.abs(diffs))

        return mass + 10.0 * phi + 5.0 * activity

    grad_fn = jax.grad(lambda p: -fitness_fn(p))

    print("  JIT compiling gradient...", end=" ", flush=True)
    t0 = time.time()
    _ = grad_fn(jnp.zeros((patch_size, patch_size)))
    print(f"done ({time.time()-t0:.1f}s)")

    discovered = []

    for i in range(n_seeds):
        rng, k1 = random.split(rng)

        yy, xx = jnp.mgrid[:patch_size, :patch_size]
        cx, cy = patch_size // 2, patch_size // 2
        dist = jnp.sqrt((xx - cx)**2 + (yy - cy)**2)
        patch = 0.3 * jnp.exp(-((dist - 8)**2) / (2 * 3**2))
        patch = patch + 0.05 * random.normal(k1, patch.shape)
        patch = jnp.clip(patch, 0.0, 1.0)

        lr = 0.005
        best_score = -1e10
        best_patch = patch

        for s in range(opt_steps):
            g = grad_fn(patch)
            gn = jnp.sqrt(jnp.sum(g**2) + 1e-10)
            g = jnp.where(gn > 1.0, g / gn, g)
            patch = jnp.clip(patch - lr * g, 0.0, 1.0)

            if s % 20 == 0:
                score = float(fitness_fn(patch))
                if score > best_score:
                    best_score = score
                    best_patch = patch
                print(f"    Seed {i+1}/{n_seeds}, step {s:>3d}: "
                      f"score={score:.2f}")

        final_score = float(fitness_fn(best_patch))
        print(f"  Seed {i+1} final: score={final_score:.2f}")

        discovered.append(Genotype(
            template=np.array(best_patch),
            fitness=final_score,
            lineage_id=i,
        ))

    discovered.sort(key=lambda g: g.fitness, reverse=True)
    print(f"\n  Best:  {discovered[0].fitness:.2f}")
    print(f"  Worst: {discovered[-1].fitness:.2f}")

    return discovered


# ============================================================================
# Full pipeline: evolve in-situ -> stress test
# ============================================================================

def full_pipeline(config=None, n_cycles=30, steps_per_cycle=5000,
                  cull_fraction=0.3, seed=42, curriculum=False,
                  discover=False, n_seeds=8):
    """Complete V11.1 pipeline:
    1. (Optional) Discover seeds via autodiff
    2. In-situ evolution with selection for integration
    3. Stress-test evolved vs naive patterns under drought
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    t_start = time.time()

    print()
    print("+" + "=" * 58 + "+")
    print("| V11.1 FULL PIPELINE: Evolve -> Stress Test               |")
    print("+" + "=" * 58 + "+")
    print()

    seeds = None
    if discover:
        seeds = discover_seeds(
            config=config, n_seeds=n_seeds,
            opt_steps=80, horizon=50, seed=seed,
        )

    # In-situ evolution
    evo_result = evolve_in_situ(
        config=config,
        n_cycles=n_cycles,
        steps_per_cycle=steps_per_cycle,
        cull_fraction=cull_fraction,
        seed=seed + 1,
        curriculum=curriculum,
    )

    # Stress test using the evolved grid directly
    stress_result = stress_test(
        evo_result['final_grid'],
        evo_result['final_resource'],
        config=config,
        seed=seed + 2,
    )

    elapsed = time.time() - t_start
    print(f"\nTotal pipeline time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    return {
        'seeds': seeds,
        'evolution': evo_result,
        'stress_test': stress_result,
    }


# ============================================================================
# V11.2: Heterogeneous Chemistry Evolution
# ============================================================================

def evolve_hetero(config=None, n_cycles=30, steps_per_cycle=5000,
                  cull_fraction=0.3, mutate_top_n=5,
                  mutation_noise=0.03, seed=42, curriculum=False,
                  n_zones=8, param_diffusion_rate=0.01,
                  mu_noise=0.005, sigma_noise=0.001):
    """V11.2: Evolution with heterogeneous chemistry.

    Each cell has its own growth parameters (mu, sigma), creating
    different viability manifolds across the grid. Selection can now
    produce genuinely different dynamics because patterns in different
    chemical zones face different physics.

    Same baseline+stress functional selection as evolve_in_situ, plus:
    - Parameter field initialization with spatial chemistry zones
    - Parameter diffusion between chunks (gene flow)
    - Parameter mutation near winners (heritable variation)
    - Chemistry colonization of killed regions
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    N = config['grid_size']
    rng = random.PRNGKey(seed)

    kernel = make_kernel(
        config['kernel_radius'], config['kernel_peak'], config['kernel_width'])
    kernel_fft = make_kernel_fft(kernel, N)

    # Initialize heterogeneous parameter fields
    rng, k_params = random.split(rng)
    mu_field, sigma_field = init_param_fields(
        N, k_params, base_mu=config['growth_mu'],
        base_sigma=config['growth_sigma'], n_zones=n_zones)

    # Build params with array-valued growth parameters
    params = make_params(config)
    params['growth_mu'] = mu_field
    params['growth_sigma'] = sigma_field

    stress_config = {**config, 'resource_regen': 0.001}
    stress_params = make_params(stress_config)
    stress_params['growth_mu'] = mu_field
    stress_params['growth_sigma'] = sigma_field

    print("=" * 60)
    print("V11.2 HETEROGENEOUS CHEMISTRY EVOLUTION")
    print("=" * 60)
    print(f"  Cycles:           {n_cycles}")
    print(f"  Steps/cycle:      {steps_per_cycle}")
    print(f"  Cull fraction:    {cull_fraction}")
    print(f"  Chemistry zones:  {n_zones}")
    print(f"  Param diffusion:  {param_diffusion_rate}")
    print(f"  mu range:         [{float(mu_field.min()):.3f}, {float(mu_field.max()):.3f}]")
    print(f"  sigma range:      [{float(sigma_field.min()):.3f}, {float(sigma_field.max()):.3f}]")
    print()

    # Initialize from random soup + warmup
    print("Phase 0: Initializing from random soup...")
    rng, k = random.split(rng)
    grid, resource = init_soup(N, k, n_seeds=80, growth_mu=config['growth_mu'])

    print("  JIT compiling for hetero params...", end=" ", flush=True)
    import time as _time
    _t0 = _time.time()
    grid, resource, rng = run_chunk(
        grid, resource, kernel_fft, rng, params, 5000)
    grid.block_until_ready()
    print(f"done ({_time.time()-_t0:.1f}s)")

    grid_np = np.array(grid)
    initial_patterns = detect_patterns(grid_np, threshold=0.1)
    print(f"  {len(initial_patterns)} patterns after warmup\n")

    tracker = PatternTracker()
    prev_masses = {}
    prev_values = {}
    cycle_stats = []

    baseline_steps = int(steps_per_cycle * 0.6)
    stress_steps = steps_per_cycle - baseline_steps

    for cycle in range(n_cycles):
        t0 = time.time()

        # Update params with current fields (changed by diffusion/mutation)
        params['growth_mu'] = mu_field
        params['growth_sigma'] = sigma_field

        # Curriculum: increase stress severity over time
        cycle_stress_params = stress_params
        if curriculum:
            difficulty = cycle / max(n_cycles - 1, 1)
            s_config = config.copy()
            s_config['resource_regen'] = 0.001 * (1.0 - 0.8 * difficulty)
            s_config['resource_consume'] = config['resource_consume'] * (
                1.0 + difficulty)
            cycle_stress_params = make_params(s_config)
        cycle_stress_params['growth_mu'] = mu_field
        cycle_stress_params['growth_sigma'] = sigma_field

        step_base = cycle * steps_per_cycle
        chunk = 100
        measure_every = max(200, baseline_steps // 10)

        # ---- BASELINE PHASE ----
        baseline_affects = {}  # pid -> [AffectState]
        baseline_survival = {}

        step = 0
        while step < baseline_steps:
            grid, resource, rng = run_chunk(
                grid, resource, kernel_fft, rng, params, chunk)
            step += chunk

            # Slow parameter diffusion between chunks
            mu_field, sigma_field = diffuse_params(
                mu_field, sigma_field, rate=param_diffusion_rate)
            params['growth_mu'] = mu_field
            params['growth_sigma'] = sigma_field

            if step % measure_every < chunk:
                grid_np = np.array(grid)
                patterns = detect_patterns(grid_np, threshold=0.1)
                tracker.update(patterns, step=step_base + step)

                for p in tracker.active.values():
                    pid = p.id
                    if pid not in baseline_affects:
                        baseline_affects[pid] = []
                    baseline_survival[pid] = step

                    hist = tracker.history.get(pid, [])
                    pm = prev_masses.get(pid)
                    pv = prev_values.get(pid)

                    affect = measure_all(
                        p, pm, pv, hist,
                        grid, kernel_fft,
                        mu_field, sigma_field, N,
                        step_num=step_base + step,
                    )
                    baseline_affects[pid].append(affect)
                    prev_masses[pid] = p.mass
                    prev_values[pid] = p.values.copy()

        # ---- STRESS PHASE ----
        stress_affects = {}  # pid -> [AffectState]
        measure_every_stress = max(200, stress_steps // 8)

        cycle_stress_params['growth_mu'] = mu_field
        cycle_stress_params['growth_sigma'] = sigma_field

        step = 0
        while step < stress_steps:
            grid, resource, rng = run_chunk(
                grid, resource, kernel_fft, rng, cycle_stress_params, chunk)
            step += chunk

            if step % measure_every_stress < chunk:
                grid_np = np.array(grid)
                patterns = detect_patterns(grid_np, threshold=0.1)
                tracker.update(patterns, step=step_base + baseline_steps + step)

                for p in tracker.active.values():
                    pid = p.id
                    if pid not in stress_affects:
                        stress_affects[pid] = []
                    if pid in baseline_survival:
                        baseline_survival[pid] = baseline_steps + step

                    hist = tracker.history.get(pid, [])
                    pm = prev_masses.get(pid)
                    pv = prev_values.get(pid)

                    affect = measure_all(
                        p, pm, pv, hist,
                        grid, kernel_fft,
                        mu_field, sigma_field, N,
                        step_num=step_base + baseline_steps + step,
                    )
                    stress_affects[pid].append(affect)
                    prev_masses[pid] = p.mass
                    prev_values[pid] = p.values.copy()

        # ---- SCORE with functional fitness ----
        grid_np = np.array(grid)
        patterns = detect_patterns(grid_np, threshold=0.1)
        tracker.update(patterns, step=step_base + steps_per_cycle)

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

        scored.sort(key=lambda x: x[1])  # ascending fitness

        if not scored:
            print(f"Cycle {cycle+1:>3d}: EXTINCTION — reseeding")
            rng, k, k_params = random.split(rng, 3)
            grid, resource = init_soup(
                N, k, n_seeds=80, growth_mu=config['growth_mu'])
            mu_field, sigma_field = init_param_fields(
                N, k_params, base_mu=config['growth_mu'],
                base_sigma=config['growth_sigma'], n_zones=n_zones)
            params['growth_mu'] = mu_field
            params['growth_sigma'] = sigma_field
            grid, resource, rng = run_chunk(
                grid, resource, kernel_fft, rng, params, 3000)
            tracker = PatternTracker()
            prev_masses = {}
            prev_values = {}
            cycle_stats.append({
                'cycle': cycle + 1, 'n_survived': 0, 'extinction': True})
            continue

        # ---- CULL bottom fraction ----
        n_cull = max(1, int(len(scored) * cull_fraction))
        to_kill = scored[:n_cull]

        kill_mask = np.ones((N, N), dtype=np.float32)
        for p, _, _, _, _ in to_kill:
            kill_mask[p.cells[:, 0], p.cells[:, 1]] = 0.0
        kill_mask_j = jnp.array(kill_mask)
        grid = grid * kill_mask_j

        # Colonize killed regions: replace chemistry with neighbor values
        # Surviving regions keep their chemistry; killed regions adopt neighbors'
        for _ in range(5):
            mu_avg = (jnp.roll(mu_field, 1, 0) + jnp.roll(mu_field, -1, 0) +
                      jnp.roll(mu_field, 1, 1) + jnp.roll(mu_field, -1, 1)) / 4.0
            sigma_avg = (jnp.roll(sigma_field, 1, 0) + jnp.roll(sigma_field, -1, 0) +
                         jnp.roll(sigma_field, 1, 1) + jnp.roll(sigma_field, -1, 1)) / 4.0
            mu_field = jnp.where(kill_mask_j == 0.0, mu_avg, mu_field)
            sigma_field = jnp.where(kill_mask_j == 0.0, sigma_avg, sigma_field)

        # ---- BOOST resources near top patterns ----
        top_patterns = scored[-mutate_top_n:]
        for p, _, _, _, _ in top_patterns:
            cx = int(p.center[1])
            cy = int(p.center[0])
            resource = perturb_resource_bloom(
                resource, (cx, cy), radius=20, intensity=0.3)

        # ---- MUTATE near top patterns (grid noise + chemistry) ----
        for p, _, _, _, _ in top_patterns:
            rng, k1, k2 = random.split(rng, 3)
            # Grid mutation (same as V11.1)
            r_min = max(0, p.bbox[0] - 5)
            r_max = min(N - 1, p.bbox[1] + 5)
            c_min = max(0, p.bbox[2] - 5)
            c_max = min(N - 1, p.bbox[3] + 5)
            h = r_max - r_min + 1
            w = c_max - c_min + 1
            noise = mutation_noise * random.normal(k1, (h, w))
            region = grid[r_min:r_max+1, c_min:c_max+1]
            grid = grid.at[r_min:r_max+1, c_min:c_max+1].set(
                jnp.clip(region + noise, 0.0, 1.0))

            # Chemistry mutation: perturb mu/sigma near winners
            mu_field, sigma_field = mutate_param_fields(
                mu_field, sigma_field, k2, p.cells,
                noise_mu=mu_noise, noise_sigma=sigma_noise)

        # ---- Restore resources for next cycle ----
        resource = jnp.clip(
            resource + 0.1 * (config['resource_max'] - resource),
            0.0, config['resource_max'])

        elapsed = time.time() - t0

        # Stats
        all_fits = [f for _, f, _, _, _ in scored]
        all_phi_base = [pb for _, _, pb, _, _ in scored]
        all_phi_stress = [ps for _, _, _, ps, _ in scored]
        all_robust = [r for _, _, _, _, r in scored]

        mu_std = float(jnp.std(mu_field))
        sigma_std = float(jnp.std(sigma_field))

        stats = {
            'cycle': cycle + 1,
            'n_patterns': len(scored),
            'n_culled': n_cull,
            'mean_fitness': float(np.mean(all_fits)),
            'max_fitness': float(np.max(all_fits)),
            'mean_phi_base': float(np.mean(all_phi_base)),
            'mean_phi_stress': float(np.mean(all_phi_stress)),
            'mean_robustness': float(np.mean(all_robust)),
            'mu_std': mu_std,
            'sigma_std': sigma_std,
            'mu_range': [float(mu_field.min()), float(mu_field.max())],
            'sigma_range': [float(sigma_field.min()), float(sigma_field.max())],
            'elapsed': elapsed,
        }
        cycle_stats.append(stats)

        phi_delta = (stats['mean_phi_stress'] - stats['mean_phi_base']) / (
            stats['mean_phi_base'] + 1e-10)
        print(f"Cycle {cycle+1:>3d}/{n_cycles}: "
              f"n={len(scored):>3d} (-{n_cull}), "
              f"Phi_base={stats['mean_phi_base']:.4f}, "
              f"Phi_stress={stats['mean_phi_stress']:.4f} ({phi_delta:+.1%}), "
              f"robust={stats['mean_robustness']:.3f}, "
              f"mu_std={mu_std:.4f}, "
              f"({elapsed:.1f}s)")

    print()
    print("=" * 60)
    print("V11.2 HETEROGENEOUS EVOLUTION COMPLETE")
    print("=" * 60)
    if len(cycle_stats) >= 2:
        first = next((s for s in cycle_stats if 'mean_phi_base' in s), None)
        last = next((s for s in reversed(cycle_stats) if 'mean_phi_base' in s), None)
        if first and last:
            print(f"  Phi (base):   {first['mean_phi_base']:.4f} -> {last['mean_phi_base']:.4f}")
            print(f"  Phi (stress): {first['mean_phi_stress']:.4f} -> {last['mean_phi_stress']:.4f}")
            print(f"  Robustness:   {first['mean_robustness']:.3f} -> {last['mean_robustness']:.3f}")
            print(f"  mu_std:       {first['mu_std']:.4f} -> {last['mu_std']:.4f}")
            print(f"  Count:        {first['n_patterns']} -> {last['n_patterns']}")

    return {
        'cycle_stats': cycle_stats,
        'final_grid': grid,
        'final_resource': resource,
        'mu_field': mu_field,
        'sigma_field': sigma_field,
        'config': config,
    }


def stress_test_hetero(evolved_grid, evolved_resource, mu_field, sigma_field,
                       config=None, seed=99):
    """Compare evolved-hetero vs naive-homo under drought.

    THE V11.2 test: does heterogeneous chemistry + selection produce
    patterns with better Phi robustness under stress?

    Compares:
    - evolved_hetero: evolved grid + evolved per-cell chemistry
    - naive_homo: fresh soup + uniform chemistry (V11.0 baseline)
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    N = config['grid_size']
    rng = random.PRNGKey(seed)

    kernel = make_kernel(
        config['kernel_radius'], config['kernel_peak'], config['kernel_width'])
    kernel_fft = make_kernel_fft(kernel, N)

    # Homogeneous params (scalar growth_mu/sigma)
    homo_params = make_params(config)
    homo_drought = make_params({**config, 'resource_regen': 0.0001})

    # Heterogeneous params (NxN growth fields)
    hetero_params = make_params(config)
    hetero_params['growth_mu'] = mu_field
    hetero_params['growth_sigma'] = sigma_field
    hetero_drought = make_params({**config, 'resource_regen': 0.0001})
    hetero_drought['growth_mu'] = mu_field
    hetero_drought['growth_sigma'] = sigma_field

    print("\n" + "=" * 60)
    print("STRESS TEST: Evolved-Hetero vs Naive-Homo Under Drought")
    print("=" * 60)

    results = {}

    for condition in ['evolved_hetero', 'naive_homo']:
        print(f"\n  [{condition.upper()}]")
        rng, k = random.split(rng)

        if condition == 'naive_homo':
            grid, resource = init_soup(
                N, k, n_seeds=80, growth_mu=config['growth_mu'])
            grid, resource, rng = run_chunk(
                grid, resource, kernel_fft, rng, homo_params, 3000)
            normal_p = homo_params
            drought_p = homo_drought
            gmu = config['growth_mu']
            gsigma = config['growth_sigma']
        else:
            grid = evolved_grid
            resource = evolved_resource
            grid, resource, rng = run_chunk(
                grid, resource, kernel_fft, rng, hetero_params, 500)
            normal_p = hetero_params
            drought_p = hetero_drought
            gmu = mu_field
            gsigma = sigma_field

        phase_data = {}
        for phase_name, phase_params, phase_steps in [
            ('baseline', normal_p, 1500),
            ('drought',  drought_p, 3000),
            ('recovery', normal_p, 1500),
        ]:
            tracker = PatternTracker()
            prev_masses_local = {}
            prev_values_local = {}
            measurements = []

            step = 0
            chunk = 100
            while step < phase_steps:
                grid, resource, rng = run_chunk(
                    grid, resource, kernel_fft, rng, phase_params, chunk)
                step += chunk

                if step % 200 < chunk:
                    grid_np = np.array(grid)
                    patterns = detect_patterns(grid_np, threshold=0.1)
                    tracker.update(patterns, step=step)

                    phis, arousals, valences = [], [], []
                    for p in tracker.active.values():
                        hist = tracker.history.get(p.id, [])
                        pm = prev_masses_local.get(p.id)
                        pv = prev_values_local.get(p.id)
                        affect = measure_all(
                            p, pm, pv, hist,
                            grid, kernel_fft,
                            gmu, gsigma, N,
                            step_num=step,
                        )
                        phis.append(affect.integration)
                        arousals.append(affect.arousal)
                        valences.append(affect.valence)
                        prev_masses_local[p.id] = p.mass
                        prev_values_local[p.id] = p.values.copy()

                    if phis:
                        measurements.append({
                            'step': step,
                            'phi': float(np.mean(phis)),
                            'arousal': float(np.mean(arousals)),
                            'valence': float(np.mean(valences)),
                            'n_patterns': len(phis),
                        })

            phase_data[phase_name] = measurements
            if measurements:
                mp = np.mean([m['phi'] for m in measurements])
                ma = np.mean([m['arousal'] for m in measurements])
                mv = np.mean([m['valence'] for m in measurements])
                nn = np.mean([m['n_patterns'] for m in measurements])
                print(f"    {phase_name:>10s}: Phi={mp:.4f}, A={ma:.4f}, "
                      f"V={mv:+.4f}, n={nn:.0f}")

        results[condition] = phase_data

    # THE comparison
    def phase_phi(data, phase):
        ms = data.get(phase, [])
        return np.mean([m['phi'] for m in ms]) if ms else 0.0

    e_base = phase_phi(results.get('evolved_hetero', {}), 'baseline')
    e_drought = phase_phi(results.get('evolved_hetero', {}), 'drought')
    n_base = phase_phi(results.get('naive_homo', {}), 'baseline')
    n_drought = phase_phi(results.get('naive_homo', {}), 'drought')

    e_delta = (e_drought - e_base) / (e_base + 1e-10)
    n_delta = (n_drought - n_base) / (n_base + 1e-10)

    print(f"\n  {'=' * 50}")
    print(f"  CRITICAL RESULT (V11.2)")
    print(f"  {'=' * 50}")
    print(f"  Naive (homo)     Phi change: {n_delta:+.1%}")
    print(f"  Evolved (hetero) Phi change: {e_delta:+.1%}")

    if e_delta > n_delta:
        print(f"  -> Hetero chemistry + selection shifted Phi response!")
        if e_delta > 0:
            print(f"  -> INTEGRATION UNDER THREAT ACHIEVED")
        else:
            print(f"  -> Still decomposing, but less than naive")
    else:
        print(f"  -> Hetero chemistry did not shift Phi response")
        print(f"  -> May need: more cycles, wider param ranges,")
        print(f"     or multi-channel substrate")

    results['comparison'] = {
        'naive_phi_delta': float(n_delta),
        'evolved_phi_delta': float(e_delta),
        'shift': float(e_delta - n_delta),
        'biological_shift': e_delta > n_delta,
        'integration_under_threat': e_delta > 0,
    }

    return results


def full_pipeline_hetero(config=None, n_cycles=30, steps_per_cycle=5000,
                         cull_fraction=0.3, seed=42, curriculum=False,
                         n_zones=8):
    """V11.2 pipeline: evolve with hetero chemistry -> stress test."""
    if config is None:
        config = DEFAULT_CONFIG.copy()

    t_start = time.time()

    print()
    print("+" + "=" * 58 + "+")
    print("| V11.2 PIPELINE: Hetero Evolve -> Stress Test             |")
    print("+" + "=" * 58 + "+")
    print()

    evo_result = evolve_hetero(
        config=config,
        n_cycles=n_cycles,
        steps_per_cycle=steps_per_cycle,
        cull_fraction=cull_fraction,
        seed=seed,
        curriculum=curriculum,
        n_zones=n_zones,
    )

    stress_result = stress_test_hetero(
        evo_result['final_grid'],
        evo_result['final_resource'],
        evo_result['mu_field'],
        evo_result['sigma_field'],
        config=config,
        seed=seed + 2,
    )

    elapsed = time.time() - t_start
    print(f"\nTotal pipeline time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    return {
        'evolution': evo_result,
        'stress_test': stress_result,
    }


# ============================================================================
# V11.3: Multi-Channel Lenia Evolution
# ============================================================================

def evolve_multichannel(config=None, n_cycles=30, steps_per_cycle=5000,
                        cull_fraction=0.3, mutate_top_n=5,
                        mutation_noise=0.03, seed=42, curriculum=False):
    """V11.3: Evolution with multi-channel Lenia substrate.

    3-channel Lenia with cross-channel coupling. Channels:
    - Structure (R=13): spatial pattern boundaries
    - Metabolism (R=7): internal energy processing
    - Signaling (R=20): communication/coordination

    Selection based on multi-channel Phi (spatial + channel partitions).
    Coupling weights are evolvable — mutated near winners.

    Same baseline+stress cycle structure as V11.1/V11.2.
    """
    if config is None:
        config = MULTICHANNEL_CONFIG.copy()

    N = config['grid_size']
    C = config['n_channels']
    rng = random.PRNGKey(seed)

    kernel_ffts = make_kernels_fft(config['channel_configs'], N)
    coupling = jnp.array(config['coupling_matrix'], dtype=jnp.float32)

    # Stress config: reduced resource regen
    stress_config = {**config, 'resource_regen': 0.001}

    print("=" * 60)
    print("V11.3 MULTI-CHANNEL LENIA EVOLUTION")
    print("=" * 60)
    print(f"  Channels:      {C}")
    print(f"  Cycles:        {n_cycles}")
    print(f"  Steps/cycle:   {steps_per_cycle}")
    print(f"  Cull fraction: {cull_fraction}")
    print(f"  Coupling:      {config['coupling_matrix']}")
    print(f"  Curriculum:    {curriculum}")
    print()

    # Initialize
    print("Phase 0: Initializing multi-channel soup...")
    rng, k = random.split(rng)
    grid, resource = init_soup_mc(N, C, k,
                                   channel_configs=config['channel_configs'])

    print("  JIT compiling multi-channel step...", end=" ", flush=True)
    t0 = time.time()
    grid, resource, rng = run_chunk_mc_wrapper(
        grid, resource, kernel_ffts, coupling, rng, config, 100)
    grid.block_until_ready()
    print(f"done ({time.time()-t0:.1f}s)")

    # Warmup
    grid, resource, rng = run_chunk_mc_wrapper(
        grid, resource, kernel_ffts, coupling, rng, config, 4900)
    grid.block_until_ready()

    grid_np = np.array(grid)
    initial_patterns = detect_patterns_mc(grid_np, threshold=0.1)
    print(f"  {len(initial_patterns)} patterns after warmup\n")

    tracker = PatternTracker()
    prev_masses = {}
    prev_values = {}
    cycle_stats = []

    baseline_steps = int(steps_per_cycle * 0.6)
    stress_steps = steps_per_cycle - baseline_steps

    for cycle in range(n_cycles):
        t0 = time.time()

        cycle_stress_config = stress_config
        if curriculum:
            difficulty = cycle / max(n_cycles - 1, 1)
            cycle_stress_config = {
                **config,
                'resource_regen': 0.001 * (1.0 - 0.8 * difficulty),
                'resource_consume': config['resource_consume'] * (1.0 + difficulty),
            }

        step_base = cycle * steps_per_cycle
        chunk = 100
        measure_every = max(200, baseline_steps // 10)

        # ---- BASELINE PHASE ----
        baseline_affects = {}
        baseline_survival = {}

        step = 0
        while step < baseline_steps:
            grid, resource, rng = run_chunk_mc_wrapper(
                grid, resource, kernel_ffts, coupling, rng, config, chunk)
            step += chunk

            if step % measure_every < chunk:
                grid_np = np.array(grid)
                patterns = detect_patterns_mc(grid_np, threshold=0.1)
                tracker.update(patterns, step=step_base + step)

                for p in tracker.active.values():
                    pid = p.id
                    if pid not in baseline_affects:
                        baseline_affects[pid] = []
                    baseline_survival[pid] = step

                    hist = tracker.history.get(pid, [])
                    pm = prev_masses.get(pid)
                    pv = prev_values.get(pid)

                    affect, _, _ = measure_all_mc(
                        p, pm, pv, hist,
                        jnp.array(grid_np), kernel_ffts, coupling,
                        config['channel_configs'], N,
                        step_num=step_base + step,
                    )
                    baseline_affects[pid].append(affect)
                    prev_masses[pid] = p.mass
                    prev_values[pid] = p.values.copy()

        # ---- STRESS PHASE ----
        stress_affects = {}
        measure_every_stress = max(200, stress_steps // 8)

        step = 0
        while step < stress_steps:
            grid, resource, rng = run_chunk_mc_wrapper(
                grid, resource, kernel_ffts, coupling, rng,
                cycle_stress_config, chunk)
            step += chunk

            if step % measure_every_stress < chunk:
                grid_np = np.array(grid)
                patterns = detect_patterns_mc(grid_np, threshold=0.1)
                tracker.update(patterns,
                               step=step_base + baseline_steps + step)

                for p in tracker.active.values():
                    pid = p.id
                    if pid not in stress_affects:
                        stress_affects[pid] = []
                    if pid in baseline_survival:
                        baseline_survival[pid] = baseline_steps + step

                    hist = tracker.history.get(pid, [])
                    pm = prev_masses.get(pid)
                    pv = prev_values.get(pid)

                    affect, _, _ = measure_all_mc(
                        p, pm, pv, hist,
                        jnp.array(grid_np), kernel_ffts, coupling,
                        config['channel_configs'], N,
                        step_num=step_base + baseline_steps + step,
                    )
                    stress_affects[pid].append(affect)
                    prev_masses[pid] = p.mass
                    prev_values[pid] = p.values.copy()

        # ---- SCORE ----
        grid_np = np.array(grid)
        patterns = detect_patterns_mc(grid_np, threshold=0.1)
        tracker.update(patterns, step=step_base + steps_per_cycle)

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
            grid, resource = init_soup_mc(
                N, C, k, channel_configs=config['channel_configs'])
            grid, resource, rng = run_chunk_mc_wrapper(
                grid, resource, kernel_ffts, coupling, rng, config, 3000)
            tracker = PatternTracker()
            prev_masses = {}
            prev_values = {}
            cycle_stats.append({
                'cycle': cycle + 1, 'n_survived': 0, 'extinction': True})
            continue

        # ---- CULL bottom fraction ----
        n_cull = max(1, int(len(scored) * cull_fraction))
        to_kill = scored[:n_cull]

        # Zero all channels at killed pattern cells
        kill_mask = np.ones((N, N), dtype=np.float32)
        for p, _, _, _, _ in to_kill:
            kill_mask[p.cells[:, 0], p.cells[:, 1]] = 0.0
        kill_mask_j = jnp.array(kill_mask)
        # Apply to all channels
        grid = grid * kill_mask_j[None, :, :]

        # ---- BOOST resources near top patterns ----
        top_patterns = scored[-mutate_top_n:]
        for p, _, _, _, _ in top_patterns:
            cx = int(p.center[1])
            cy = int(p.center[0])
            resource = perturb_resource_bloom(
                resource, (cx, cy), radius=20, intensity=0.3)

        # ---- MUTATE near top patterns (all channels) ----
        for p, _, _, _, _ in top_patterns:
            rng, k1 = random.split(rng)
            r_min = max(0, p.bbox[0] - 5)
            r_max = min(N - 1, p.bbox[1] + 5)
            c_min = max(0, p.bbox[2] - 5)
            c_max = min(N - 1, p.bbox[3] + 5)
            h = r_max - r_min + 1
            w = c_max - c_min + 1
            # Inject noise to all channels
            noise = mutation_noise * random.normal(k1, (C, h, w))
            region = grid[:, r_min:r_max+1, c_min:c_max+1]
            grid = grid.at[:, r_min:r_max+1, c_min:c_max+1].set(
                jnp.clip(region + noise, 0.0, 1.0))

        # ---- Mutate coupling near top patterns ----
        # Small perturbation to coupling matrix (global, affects all)
        rng, k_coup = random.split(rng)
        coupling_noise = 0.01 * random.normal(k_coup, coupling.shape)
        # Keep diagonal at 1.0, keep symmetric
        coupling_noise = coupling_noise.at[jnp.arange(C), jnp.arange(C)].set(0.0)
        coupling_noise = (coupling_noise + coupling_noise.T) / 2.0
        coupling = jnp.clip(coupling + coupling_noise, 0.0, 1.0)
        coupling = coupling.at[jnp.arange(C), jnp.arange(C)].set(1.0)

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
            'coupling_off_diag': float(
                (coupling.sum() - C) / (C * C - C)),
            'elapsed': elapsed,
        }
        cycle_stats.append(stats)

        phi_delta = (stats['mean_phi_stress'] - stats['mean_phi_base']) / (
            stats['mean_phi_base'] + 1e-10)
        print(f"Cycle {cycle+1:>3d}/{n_cycles}: "
              f"n={len(scored):>3d} (-{n_cull}), "
              f"Phi_base={stats['mean_phi_base']:.4f}, "
              f"Phi_stress={stats['mean_phi_stress']:.4f} ({phi_delta:+.1%}), "
              f"robust={stats['mean_robustness']:.3f}, "
              f"coupling={stats['coupling_off_diag']:.3f}, "
              f"({elapsed:.1f}s)")

    print()
    print("=" * 60)
    print("V11.3 MULTI-CHANNEL EVOLUTION COMPLETE")
    print("=" * 60)
    if len(cycle_stats) >= 2:
        first = next((s for s in cycle_stats if 'mean_phi_base' in s), None)
        last = next((s for s in reversed(cycle_stats) if 'mean_phi_base' in s), None)
        if first and last:
            print(f"  Phi (base):   {first['mean_phi_base']:.4f} -> {last['mean_phi_base']:.4f}")
            print(f"  Phi (stress): {first['mean_phi_stress']:.4f} -> {last['mean_phi_stress']:.4f}")
            print(f"  Robustness:   {first['mean_robustness']:.3f} -> {last['mean_robustness']:.3f}")
            print(f"  Coupling:     {first['coupling_off_diag']:.3f} -> {last['coupling_off_diag']:.3f}")
            print(f"  Count:        {first['n_patterns']} -> {last['n_patterns']}")

    return {
        'cycle_stats': cycle_stats,
        'final_grid': grid,
        'final_resource': resource,
        'coupling': coupling,
        'config': config,
    }


def stress_test_mc(evolved_grid, evolved_resource, evolved_coupling,
                   config=None, seed=99):
    """Compare evolved-multichannel vs naive-multichannel under drought.

    THE V11.3 test: does multi-channel evolution + selection produce
    patterns with better Phi robustness (especially cross-channel Phi)?
    """
    if config is None:
        config = MULTICHANNEL_CONFIG.copy()

    N = config['grid_size']
    C = config['n_channels']
    rng = random.PRNGKey(seed)

    kernel_ffts = make_kernels_fft(config['channel_configs'], N)
    naive_coupling = jnp.array(config['coupling_matrix'], dtype=jnp.float32)

    drought_config = {**config, 'resource_regen': 0.0001}

    print("\n" + "=" * 60)
    print("STRESS TEST: Evolved-MC vs Naive-MC Under Drought")
    print("=" * 60)

    results = {}

    for condition in ['evolved_mc', 'naive_mc']:
        print(f"\n  [{condition.upper()}]")
        rng, k = random.split(rng)

        if condition == 'naive_mc':
            grid, resource = init_soup_mc(
                N, C, k, channel_configs=config['channel_configs'])
            grid, resource, rng = run_chunk_mc_wrapper(
                grid, resource, kernel_ffts, naive_coupling, rng,
                config, 3000)
            cpl = naive_coupling
        else:
            grid = evolved_grid
            resource = evolved_resource
            grid, resource, rng = run_chunk_mc_wrapper(
                grid, resource, kernel_ffts, evolved_coupling, rng,
                config, 500)
            cpl = evolved_coupling

        phase_data = {}
        for phase_name, phase_cfg, phase_steps in [
            ('baseline', config, 1500),
            ('drought',  drought_config, 3000),
            ('recovery', config, 1500),
        ]:
            tracker = PatternTracker()
            prev_masses_local = {}
            prev_values_local = {}
            measurements = []

            step = 0
            chunk = 100
            while step < phase_steps:
                grid, resource, rng = run_chunk_mc_wrapper(
                    grid, resource, kernel_ffts, cpl, rng,
                    phase_cfg, chunk)
                step += chunk

                if step % 200 < chunk:
                    grid_np = np.array(grid)
                    patterns = detect_patterns_mc(grid_np, threshold=0.1)
                    tracker.update(patterns, step=step)

                    phis, arousals, valences = [], [], []
                    phi_spatials, phi_channels = [], []
                    for p in tracker.active.values():
                        hist = tracker.history.get(p.id, [])
                        pm = prev_masses_local.get(p.id)
                        pv = prev_values_local.get(p.id)
                        affect, ps, pc = measure_all_mc(
                            p, pm, pv, hist,
                            jnp.array(grid_np), kernel_ffts, cpl,
                            config['channel_configs'], N,
                            step_num=step,
                        )
                        phis.append(affect.integration)
                        arousals.append(affect.arousal)
                        valences.append(affect.valence)
                        phi_spatials.append(ps)
                        phi_channels.append(pc)
                        prev_masses_local[p.id] = p.mass
                        prev_values_local[p.id] = p.values.copy()

                    if phis:
                        measurements.append({
                            'step': step,
                            'phi': float(np.mean(phis)),
                            'phi_spatial': float(np.mean(phi_spatials)),
                            'phi_channel': float(np.mean(phi_channels)),
                            'arousal': float(np.mean(arousals)),
                            'valence': float(np.mean(valences)),
                            'n_patterns': len(phis),
                        })

            phase_data[phase_name] = measurements
            if measurements:
                mp = np.mean([m['phi'] for m in measurements])
                mps = np.mean([m['phi_spatial'] for m in measurements])
                mpc = np.mean([m['phi_channel'] for m in measurements])
                ma = np.mean([m['arousal'] for m in measurements])
                nn = np.mean([m['n_patterns'] for m in measurements])
                print(f"    {phase_name:>10s}: Phi={mp:.4f} "
                      f"(spatial={mps:.4f}, channel={mpc:.4f}), "
                      f"A={ma:.4f}, n={nn:.0f}")

        results[condition] = phase_data

    # THE comparison
    def phase_phi(data, phase):
        ms = data.get(phase, [])
        return np.mean([m['phi'] for m in ms]) if ms else 0.0

    def phase_phi_channel(data, phase):
        ms = data.get(phase, [])
        return np.mean([m['phi_channel'] for m in ms]) if ms else 0.0

    e_base = phase_phi(results.get('evolved_mc', {}), 'baseline')
    e_drought = phase_phi(results.get('evolved_mc', {}), 'drought')
    n_base = phase_phi(results.get('naive_mc', {}), 'baseline')
    n_drought = phase_phi(results.get('naive_mc', {}), 'drought')

    e_ch_base = phase_phi_channel(results.get('evolved_mc', {}), 'baseline')
    e_ch_drought = phase_phi_channel(results.get('evolved_mc', {}), 'drought')
    n_ch_base = phase_phi_channel(results.get('naive_mc', {}), 'baseline')
    n_ch_drought = phase_phi_channel(results.get('naive_mc', {}), 'drought')

    e_delta = (e_drought - e_base) / (e_base + 1e-10)
    n_delta = (n_drought - n_base) / (n_base + 1e-10)
    e_ch_delta = (e_ch_drought - e_ch_base) / (e_ch_base + 1e-10)
    n_ch_delta = (n_ch_drought - n_ch_base) / (n_ch_base + 1e-10)

    print(f"\n  {'=' * 55}")
    print(f"  CRITICAL RESULT (V11.3)")
    print(f"  {'=' * 55}")
    print(f"  Naive (mc)     Phi change: {n_delta:+.1%} "
          f"(channel: {n_ch_delta:+.1%})")
    print(f"  Evolved (mc)   Phi change: {e_delta:+.1%} "
          f"(channel: {e_ch_delta:+.1%})")

    if e_delta > n_delta:
        print(f"  -> Multi-channel evolution shifted Phi response!")
        if e_delta > 0:
            print(f"  -> INTEGRATION UNDER THREAT ACHIEVED")
        else:
            print(f"  -> Still decomposing, but less than naive")
    else:
        print(f"  -> Multi-channel evolution did not shift total Phi")

    if e_ch_delta > n_ch_delta:
        print(f"  -> Channel integration improved under selection!")

    results['comparison'] = {
        'naive_phi_delta': float(n_delta),
        'evolved_phi_delta': float(e_delta),
        'naive_ch_phi_delta': float(n_ch_delta),
        'evolved_ch_phi_delta': float(e_ch_delta),
        'shift': float(e_delta - n_delta),
        'channel_shift': float(e_ch_delta - n_ch_delta),
        'biological_shift': e_delta > n_delta,
        'integration_under_threat': e_delta > 0,
    }

    return results


def full_pipeline_mc(config=None, n_cycles=30, steps_per_cycle=5000,
                     cull_fraction=0.3, seed=42, curriculum=False):
    """V11.3 pipeline: multi-channel evolve -> stress test."""
    if config is None:
        config = MULTICHANNEL_CONFIG.copy()

    t_start = time.time()

    print()
    print("+" + "=" * 58 + "+")
    print("| V11.3 PIPELINE: Multi-Channel Evolve -> Stress Test     |")
    print("+" + "=" * 58 + "+")
    print()

    evo_result = evolve_multichannel(
        config=config,
        n_cycles=n_cycles,
        steps_per_cycle=steps_per_cycle,
        cull_fraction=cull_fraction,
        seed=seed,
        curriculum=curriculum,
    )

    stress_result = stress_test_mc(
        evo_result['final_grid'],
        evo_result['final_resource'],
        evo_result['coupling'],
        config=config,
        seed=seed + 2,
    )

    elapsed = time.time() - t_start
    print(f"\nTotal pipeline time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    return {
        'evolution': evo_result,
        'stress_test': stress_result,
    }


# ============================================================================
# V11.4: High-Dimensional Multi-Channel Lenia Evolution
# ============================================================================

def evolve_hd(config=None, n_cycles=30, steps_per_cycle=5000,
              cull_fraction=0.3, mutate_top_n=5,
              mutation_noise=0.03, seed=42, curriculum=False,
              C=64, bandwidth=8.0, post_cycle_callback=None):
    """V11.4: Evolution with high-dimensional (C=64) channel Lenia.

    Fully vectorized substrate — no Python loops in physics.
    Uses spectral Phi for per-cycle measurement (fast).
    Coupling evolution: bandwidth perturbation (2 parameters, not C^2).
    """
    hd = _import_hd()

    if config is None:
        config = hd['generate_hd_config'](C=C, N=256, seed=seed)

    N = config['grid_size']
    C = config['n_channels']
    rng = random.PRNGKey(seed)

    kernel_ffts = hd['make_kernels_fft_hd'](config)
    coupling = jnp.array(hd['generate_coupling_matrix'](C, bandwidth=bandwidth, seed=seed))

    stress_config = {**config, 'resource_regen': 0.001}

    print("=" * 60)
    print(f"V11.4 HIGH-DIMENSIONAL LENIA EVOLUTION (C={C})")
    print("=" * 60)
    print(f"  Channels:      {C}")
    print(f"  Grid:          {N}x{N}")
    print(f"  Cycles:        {n_cycles}")
    print(f"  Steps/cycle:   {steps_per_cycle}")
    print(f"  Cull fraction: {cull_fraction}")
    print(f"  Bandwidth:     {bandwidth}")
    print(f"  Curriculum:    {curriculum}")
    print()

    # Initialize
    print("Phase 0: Initializing HD soup...")
    rng, k = random.split(rng)
    grid, resource = hd['init_soup_hd'](
        N, C, k, jnp.array(config['channel_mus']))

    print("  JIT compiling HD step...", end=" ", flush=True)
    t0 = time.time()
    grid, resource, rng = hd['run_chunk_hd_wrapper'](
        grid, resource, kernel_ffts, coupling, rng, config, 100)
    grid.block_until_ready()
    print(f"done ({time.time()-t0:.1f}s)")

    # Warmup
    grid, resource, rng = hd['run_chunk_hd_wrapper'](
        grid, resource, kernel_ffts, coupling, rng, config, 4900)
    grid.block_until_ready()

    grid_np = np.array(grid)
    initial_patterns = detect_patterns_mc(grid_np, threshold=0.15)
    print(f"  {len(initial_patterns)} patterns after warmup\n")

    tracker = PatternTracker()
    prev_masses = {}
    prev_values = {}
    cycle_stats = []

    baseline_steps = int(steps_per_cycle * 0.6)
    stress_steps = steps_per_cycle - baseline_steps

    for cycle in range(n_cycles):
        t0 = time.time()

        cycle_stress_config = stress_config
        if curriculum:
            difficulty = cycle / max(n_cycles - 1, 1)
            cycle_stress_config = {
                **config,
                'resource_regen': 0.001 * (1.0 - 0.8 * difficulty),
                'resource_consume': config['resource_consume'] * (1.0 + difficulty),
            }

        step_base = cycle * steps_per_cycle
        chunk = 100
        measure_every = max(200, baseline_steps // 10)

        # ---- BASELINE PHASE ----
        baseline_affects = {}
        baseline_survival = {}

        step = 0
        while step < baseline_steps:
            grid, resource, rng = hd['run_chunk_hd_wrapper'](
                grid, resource, kernel_ffts, coupling, rng, config, chunk)
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

                    affect, phi_spec, eff_rank, phi_spat, _ = hd['measure_all_hd'](
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
        measure_every_stress = max(200, stress_steps // 8)

        step = 0
        while step < stress_steps:
            grid, resource, rng = hd['run_chunk_hd_wrapper'](
                grid, resource, kernel_ffts, coupling, rng,
                cycle_stress_config, chunk)
            step += chunk

            if step % measure_every_stress < chunk:
                grid_np = np.array(grid)
                patterns = detect_patterns_mc(grid_np, threshold=0.15)
                tracker.update(patterns,
                               step=step_base + baseline_steps + step)

                for p in tracker.active.values():
                    pid = p.id
                    if pid not in stress_affects:
                        stress_affects[pid] = []
                    if pid in baseline_survival:
                        baseline_survival[pid] = baseline_steps + step

                    hist = tracker.history.get(pid, [])
                    pm = prev_masses.get(pid)
                    pv = prev_values.get(pid)

                    affect, _, _, _, _ = hd['measure_all_hd'](
                        p, pm, pv, hist,
                        jnp.array(grid_np), kernel_ffts, coupling,
                        config, N, step_num=step_base + baseline_steps + step,
                        fast=True,
                    )
                    stress_affects[pid].append(affect)
                    prev_masses[pid] = p.mass
                    prev_values[pid] = p.values.copy()

        # ---- SCORE ----
        grid_np = np.array(grid)
        patterns = detect_patterns_mc(grid_np, threshold=0.15)
        tracker.update(patterns, step=step_base + steps_per_cycle)

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
            grid, resource = hd['init_soup_hd'](
                N, C, k, jnp.array(config['channel_mus']))
            grid, resource, rng = hd['run_chunk_hd_wrapper'](
                grid, resource, kernel_ffts, coupling, rng, config, 3000)
            tracker = PatternTracker()
            prev_masses = {}
            prev_values = {}
            cycle_stats.append({
                'cycle': cycle + 1, 'n_survived': 0, 'extinction': True})
            continue

        # ---- CULL bottom fraction ----
        n_cull = max(1, int(len(scored) * cull_fraction))
        to_kill = scored[:n_cull]

        kill_mask = np.ones((N, N), dtype=np.float32)
        for p, _, _, _, _ in to_kill:
            kill_mask[p.cells[:, 0], p.cells[:, 1]] = 0.0
        kill_mask_j = jnp.array(kill_mask)
        grid = grid * kill_mask_j[None, :, :]

        # ---- BOOST resources near top patterns ----
        top_patterns = scored[-mutate_top_n:]
        for p, _, _, _, _ in top_patterns:
            cx = int(p.center[1])
            cy = int(p.center[0])
            resource = perturb_resource_bloom(
                resource, (cx, cy), radius=20, intensity=0.3)

        # ---- MUTATE near top patterns (all channels) ----
        for p, _, _, _, _ in top_patterns:
            rng, k1 = random.split(rng)
            r_min = max(0, p.bbox[0] - 5)
            r_max = min(N - 1, p.bbox[1] + 5)
            c_min = max(0, p.bbox[2] - 5)
            c_max = min(N - 1, p.bbox[3] + 5)
            h = r_max - r_min + 1
            w = c_max - c_min + 1
            noise = mutation_noise * random.normal(k1, (C, h, w))
            region = grid[:, r_min:r_max+1, c_min:c_max+1]
            grid = grid.at[:, r_min:r_max+1, c_min:c_max+1].set(
                jnp.clip(region + noise, 0.0, 1.0))

        # ---- Mutate coupling (bandwidth perturbation) ----
        rng, k_bw = random.split(rng)
        bandwidth = bandwidth + 0.5 * float(random.normal(k_bw, ()))
        bandwidth = max(2.0, min(C / 2, bandwidth))
        coupling = jnp.array(
            hd['generate_coupling_matrix'](C, bandwidth=bandwidth, seed=seed + cycle + 1))

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
            'elapsed': elapsed,
        }
        cycle_stats.append(stats)

        phi_delta = (stats['mean_phi_stress'] - stats['mean_phi_base']) / (
            stats['mean_phi_base'] + 1e-10)
        print(f"Cycle {cycle+1:>3d}/{n_cycles}: "
              f"n={len(scored):>3d} (-{n_cull}), "
              f"Phi_base={stats['mean_phi_base']:.4f}, "
              f"Phi_stress={stats['mean_phi_stress']:.4f} ({phi_delta:+.1%}), "
              f"robust={stats['mean_robustness']:.3f}, "
              f"bw={bandwidth:.1f}, "
              f"({elapsed:.1f}s)", flush=True)

        if post_cycle_callback:
            post_cycle_callback(cycle_stats)

    print()
    print("=" * 60)
    print(f"V11.4 HD EVOLUTION COMPLETE (C={C})")
    print("=" * 60)
    if len(cycle_stats) >= 2:
        first = next((s for s in cycle_stats if 'mean_phi_base' in s), None)
        last = next((s for s in reversed(cycle_stats) if 'mean_phi_base' in s), None)
        if first and last:
            print(f"  Phi (base):   {first['mean_phi_base']:.4f} -> {last['mean_phi_base']:.4f}")
            print(f"  Phi (stress): {first['mean_phi_stress']:.4f} -> {last['mean_phi_stress']:.4f}")
            print(f"  Robustness:   {first['mean_robustness']:.3f} -> {last['mean_robustness']:.3f}")
            print(f"  Bandwidth:    {first['bandwidth']:.1f} -> {last['bandwidth']:.1f}")
            print(f"  Count:        {first['n_patterns']} -> {last['n_patterns']}")

    return {
        'cycle_stats': cycle_stats,
        'final_grid': grid,
        'final_resource': resource,
        'coupling': coupling,
        'config': config,
        'bandwidth': bandwidth,
    }


def stress_test_hd(evolved_grid, evolved_resource, evolved_coupling,
                   config=None, seed=99, C=64, bandwidth=8.0):
    """Compare evolved-HD vs naive-HD under drought."""
    hd = _import_hd()

    if config is None:
        config = hd['generate_hd_config'](C=C, N=256, seed=seed)

    N = config['grid_size']
    C = config['n_channels']
    rng = random.PRNGKey(seed)

    kernel_ffts = hd['make_kernels_fft_hd'](config)
    naive_coupling = jnp.array(
        hd['generate_coupling_matrix'](C, bandwidth=bandwidth, seed=seed))

    drought_config = {**config, 'resource_regen': 0.0001}

    print("\n" + "=" * 60)
    print(f"STRESS TEST: Evolved-HD vs Naive-HD (C={C})")
    print("=" * 60)

    results = {}

    for condition in ['evolved_hd', 'naive_hd']:
        print(f"\n  [{condition.upper()}]")
        rng, k = random.split(rng)

        if condition == 'naive_hd':
            grid, resource = hd['init_soup_hd'](
                N, C, k, jnp.array(config['channel_mus']))
            grid, resource, rng = hd['run_chunk_hd_wrapper'](
                grid, resource, kernel_ffts, naive_coupling, rng,
                config, 3000)
            cpl = naive_coupling
        else:
            grid = evolved_grid
            resource = evolved_resource
            grid, resource, rng = hd['run_chunk_hd_wrapper'](
                grid, resource, kernel_ffts, evolved_coupling, rng,
                config, 500)
            cpl = evolved_coupling

        phase_data = {}
        for phase_name, phase_cfg, phase_steps in [
            ('baseline', config, 1500),
            ('drought',  drought_config, 3000),
            ('recovery', config, 1500),
        ]:
            tracker = PatternTracker()
            prev_masses_local = {}
            prev_values_local = {}
            measurements = []

            step = 0
            chunk = 100
            while step < phase_steps:
                grid, resource, rng = hd['run_chunk_hd_wrapper'](
                    grid, resource, kernel_ffts, cpl, rng,
                    phase_cfg, chunk)
                step += chunk

                if step % 200 < chunk:
                    grid_np = np.array(grid)
                    patterns = detect_patterns_mc(grid_np, threshold=0.15)
                    tracker.update(patterns, step=step)

                    phis, arousals, valences = [], [], []
                    phi_specs, eff_ranks = [], []
                    for p in tracker.active.values():
                        hist = tracker.history.get(p.id, [])
                        pm = prev_masses_local.get(p.id)
                        pv = prev_values_local.get(p.id)
                        affect, ps, er, pspat, _ = hd['measure_all_hd'](
                            p, pm, pv, hist,
                            jnp.array(grid_np), kernel_ffts, cpl,
                            config, N, step_num=step,
                        )
                        phis.append(affect.integration)
                        arousals.append(affect.arousal)
                        valences.append(affect.valence)
                        phi_specs.append(ps)
                        eff_ranks.append(er)
                        prev_masses_local[p.id] = p.mass
                        prev_values_local[p.id] = p.values.copy()

                    if phis:
                        measurements.append({
                            'step': step,
                            'phi': float(np.mean(phis)),
                            'phi_spectral': float(np.mean(phi_specs)),
                            'eff_rank': float(np.mean(eff_ranks)),
                            'arousal': float(np.mean(arousals)),
                            'valence': float(np.mean(valences)),
                            'n_patterns': len(phis),
                        })

            phase_data[phase_name] = measurements
            if measurements:
                mp = np.mean([m['phi'] for m in measurements])
                mps = np.mean([m['phi_spectral'] for m in measurements])
                mer = np.mean([m['eff_rank'] for m in measurements])
                ma = np.mean([m['arousal'] for m in measurements])
                nn = np.mean([m['n_patterns'] for m in measurements])
                print(f"    {phase_name:>10s}: Phi={mp:.4f} "
                      f"(spectral={mps:.4f}, eff_rank={mer:.1f}), "
                      f"A={ma:.4f}, n={nn:.0f}")

        results[condition] = phase_data

    # THE comparison
    def phase_phi(data, phase):
        ms = data.get(phase, [])
        return np.mean([m['phi'] for m in ms]) if ms else 0.0

    e_base = phase_phi(results.get('evolved_hd', {}), 'baseline')
    e_drought = phase_phi(results.get('evolved_hd', {}), 'drought')
    n_base = phase_phi(results.get('naive_hd', {}), 'baseline')
    n_drought = phase_phi(results.get('naive_hd', {}), 'drought')

    e_delta = (e_drought - e_base) / (e_base + 1e-10)
    n_delta = (n_drought - n_base) / (n_base + 1e-10)

    print(f"\n  {'=' * 55}")
    print(f"  CRITICAL RESULT (V11.4, C={C})")
    print(f"  {'=' * 55}")
    print(f"  Naive (hd)     Phi change: {n_delta:+.1%}")
    print(f"  Evolved (hd)   Phi change: {e_delta:+.1%}")

    if e_delta > n_delta:
        print(f"  -> HD evolution shifted Phi toward biological pattern!")
        if e_delta > 0:
            print(f"  -> INTEGRATION UNDER THREAT ACHIEVED")
        else:
            print(f"  -> Still decomposing, but less than naive")
    else:
        print(f"  -> HD evolution did not shift total Phi")

    results['comparison'] = {
        'naive_phi_delta': float(n_delta),
        'evolved_phi_delta': float(e_delta),
        'shift': float(e_delta - n_delta),
        'biological_shift': e_delta > n_delta,
        'integration_under_threat': e_delta > 0,
        'n_channels': C,
    }

    return results


def full_pipeline_hd(config=None, n_cycles=30, steps_per_cycle=5000,
                     cull_fraction=0.3, seed=42, curriculum=False,
                     C=64, bandwidth=8.0, post_cycle_callback=None):
    """V11.4 pipeline: HD evolve -> stress test."""
    hd = _import_hd()

    if config is None:
        config = hd['generate_hd_config'](C=C, N=256, seed=seed)

    t_start = time.time()

    print()
    print("+" + "=" * 58 + "+")
    print(f"| V11.4 PIPELINE: HD Evolve (C={C}) -> Stress Test" +
          " " * (10 - len(str(C))) + "|")
    print("+" + "=" * 58 + "+")
    print()

    evo_result = evolve_hd(
        config=config,
        n_cycles=n_cycles,
        steps_per_cycle=steps_per_cycle,
        cull_fraction=cull_fraction,
        seed=seed,
        curriculum=curriculum,
        C=C,
        bandwidth=bandwidth,
        post_cycle_callback=post_cycle_callback,
    )

    stress_result = stress_test_hd(
        evo_result['final_grid'],
        evo_result['final_resource'],
        evo_result['coupling'],
        config=config,
        seed=seed + 2,
        C=C,
        bandwidth=evo_result.get('bandwidth', bandwidth),
    )

    elapsed = time.time() - t_start
    print(f"\nTotal pipeline time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    return {
        'evolution': evo_result,
        'stress_test': stress_result,
    }
