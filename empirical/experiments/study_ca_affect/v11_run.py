"""V11 Runner: Sanity check, perturbation, evolution, and full pipeline.

Usage:
    python v11_run.py sanity      # Quick test (~2 min)
    python v11_run.py perturb     # Drought/recovery experiment
    python v11_run.py experiment  # Full observation run (~hours)
    python v11_run.py ablation    # Forcing function comparison
    python v11_run.py evolve [N]  # V11.1 evolution (N cycles, default 30)
    python v11_run.py seeds [N]   # Differentiable seed discovery (N seeds, default 8)
    python v11_run.py pipeline    # Full V11.1: discover -> evolve -> stress test
    python v11_run.py hetero [N]  # V11.2 hetero chemistry evolution (N cycles)
    python v11_run.py hetero-pipeline  # Full V11.2: hetero evolve -> stress test
    python v11_run.py multichannel [N] # V11.3 multi-channel evolution (N cycles)
    python v11_run.py mc-pipeline      # Full V11.3: multi-channel evolve -> stress test
    python v11_run.py hd [N] [--channels C]  # V11.4 HD evolution (N cycles, C channels)
    python v11_run.py hd-pipeline [--channels C]  # Full V11.4: HD evolve -> stress test
"""

import sys
import os
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from v11_substrate import (
    DEFAULT_CONFIG, FORGIVING_CONFIG, NO_RESOURCE_CONFIG,
    make_kernel, make_kernel_fft, make_params,
    run_chunk, init_soup, init_orbium_seeds, growth_fn,
    perturb_resource_crash, perturb_kill_zone, perturb_noise_burst,
    perturb_resource_bloom,
)
from v11_patterns import detect_patterns, PatternTracker
from v11_affect import measure_all, AffectState


def sanity_check(config=None, n_steps=10000, chunk_size=100, measure_every=10):
    """Quick validation that the full pipeline works.

    1. Initialize substrate
    2. Run for n_steps
    3. Detect and track patterns
    4. Measure affect dimensions
    5. Print summary
    """
    if config is None:
        config = DEFAULT_CONFIG

    N = config['grid_size']
    print(f"=== V11 Substrate Sanity Check ===")
    print(f"Config: grid={N}, R={config['kernel_radius']}, "
          f"growth_mu={config['growth_mu']}, growth_sigma={config['growth_sigma']}")
    print(f"Backend: {jax.default_backend()}")
    print()

    # Setup
    rng = random.PRNGKey(42)
    k1, k2 = random.split(rng)

    print("Initializing kernel...", end=" ", flush=True)
    kernel = make_kernel(config['kernel_radius'], config['kernel_peak'],
                         config['kernel_width'])
    kernel_fft = make_kernel_fft(kernel, N)
    params = make_params(config)
    print("done")

    print("Initializing random soup...", end=" ", flush=True)
    grid, resource = init_soup(N, k1, n_seeds=50, growth_mu=config['growth_mu'])
    total_mass = float(jnp.sum(grid))
    active = float(jnp.sum(grid > 0.1))
    print(f"done (mass={total_mass:.0f}, active={active:.0f} cells)")
    print()

    # JIT warmup
    print("JIT compiling (first chunk)...", end=" ", flush=True)
    t0 = time.time()
    grid, resource, k2 = run_chunk(grid, resource, kernel_fft, k2, params, chunk_size)
    grid.block_until_ready()
    print(f"done ({time.time()-t0:.1f}s)")

    # Pattern tracking
    tracker = PatternTracker(max_match_dist=30.0)
    affect_log = []  # list of AffectState per measurement step
    prev_masses = {}  # pattern_id -> prev mass
    prev_values = {}  # pattern_id -> prev values at cells

    n_chunks = n_steps // chunk_size
    print(f"\nRunning {n_steps} steps ({n_chunks} chunks x {chunk_size})...")
    t_start = time.time()

    for chunk_idx in range(n_chunks):
        # Run physics on GPU
        grid, resource, k2 = run_chunk(grid, resource, kernel_fft, k2,
                                        params, chunk_size)

        step = (chunk_idx + 1) * chunk_size

        if chunk_idx % measure_every != 0:
            continue

        # Bring to CPU for measurement
        grid_np = np.array(grid)
        total_mass = float(grid_np.sum())
        active_frac = float((grid_np > 0.1).sum()) / (N * N)

        # Detect patterns
        patterns = detect_patterns(grid_np, threshold=0.1, min_size=8)
        patterns = tracker.update(patterns, step=step)

        # Measure affect for each pattern
        step_affects = []
        for p in patterns:
            pm = prev_masses.get(p.id)
            pv = prev_values.get(p.id)
            hist = tracker.history.get(p.id, [])

            affect = measure_all(
                p, pm, pv, hist,
                grid, kernel_fft,
                config['growth_mu'], config['growth_sigma'], N,
                step_num=step,
            )
            step_affects.append(affect)

            # Store for next measurement
            prev_masses[p.id] = p.mass
            prev_values[p.id] = p.values.copy()

        affect_log.extend(step_affects)

        # Progress
        elapsed = time.time() - t_start
        rate = step / elapsed if elapsed > 0 else 0
        print(f"  Step {step:>6d}: mass={total_mass:>8.1f}, "
              f"active={active_frac*100:>5.1f}%, "
              f"patterns={len(patterns):>3d}, "
              f"rate={rate:.0f} steps/s")

    elapsed = time.time() - t_start
    print(f"\nCompleted in {elapsed:.1f}s ({n_steps/elapsed:.0f} steps/s)")

    # ---- Results ----
    print("\n" + "="*60)
    print("SURVIVING PATTERNS")
    print("="*60)

    survivors = tracker.get_longest_lived(10)
    if not survivors:
        print("  No patterns survived!")
        print("  Try FORGIVING_CONFIG with wider growth_sigma")
        return False, affect_log, tracker

    for pid in survivors:
        hist = tracker.history[pid]
        if len(hist) < 2:
            continue
        last = hist[-1]
        # Get affect measurements for this pattern
        p_affects = [a for a in affect_log if a.pattern_id == pid]
        if not p_affects:
            continue

        mean_v = np.mean([a.valence for a in p_affects])
        mean_a = np.mean([a.arousal for a in p_affects])
        mean_phi = np.mean([a.integration for a in p_affects])
        mean_er = np.mean([a.effective_rank for a in p_affects])
        mean_sm = np.mean([a.self_model_salience for a in p_affects])

        # Velocity
        centers = tracker.get_center_trajectory(pid)
        if len(centers) > 1:
            vel = np.mean(np.linalg.norm(np.diff(centers, axis=0), axis=1))
        else:
            vel = 0.0

        print(f"\n  Pattern #{pid}: size={last['size']}, mass={last['mass']:.1f}, "
              f"age={len(hist)} measurements")
        print(f"    Velocity: {vel:.3f} cells/measurement")
        print(f"    Affect: V={mean_v:+.4f}, A={mean_a:.4f}, "
              f"Phi={mean_phi:.4f}, ER={mean_er:.1f}, SM={mean_sm:.3f}")

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    stats = tracker.survival_stats()
    print(f"  Total patterns detected: {stats['n_total']}")
    print(f"  Currently active: {stats['n_active']}")
    print(f"  Mean lifetime: {stats['mean_lifetime']:.1f} measurements")
    print(f"  Max lifetime: {stats['max_lifetime']} measurements")

    if affect_log:
        all_phi = [a.integration for a in affect_log if a.integration > 0]
        all_er = [a.effective_rank for a in affect_log]
        if all_phi:
            print(f"  Mean Phi: {np.mean(all_phi):.4f} +/- {np.std(all_phi):.4f}")
        if all_er:
            print(f"  Mean EffRank: {np.mean(all_er):.1f} +/- {np.std(all_er):.1f}")

    # Test: does Phi correlate with survival?
    if len(survivors) > 2:
        lifetimes = []
        mean_phis = []
        for pid in tracker.history:
            p_affects = [a for a in affect_log if a.pattern_id == pid]
            if len(p_affects) < 3:
                continue
            lifetimes.append(len(tracker.history[pid]))
            mean_phis.append(np.mean([a.integration for a in p_affects]))

        if len(lifetimes) > 3:
            corr = np.corrcoef(lifetimes, mean_phis)[0, 1]
            print(f"  Correlation(Phi, survival): r={corr:.3f}")

    print("\n=== VERDICT: Pipeline operational ===")
    return True, affect_log, tracker


def run_experiment(config, n_steps=100000, chunk_size=100,
                   measure_every=50, seed=42, label="default"):
    """Full experiment run with comprehensive logging."""
    N = config['grid_size']
    rng = random.PRNGKey(seed)
    k1, k2 = random.split(rng)

    kernel = make_kernel(config['kernel_radius'], config['kernel_peak'],
                         config['kernel_width'])
    kernel_fft = make_kernel_fft(kernel, N)
    params = make_params(config)

    grid, resource = init_soup(N, k1, n_seeds=50, growth_mu=config['growth_mu'])

    # JIT warmup
    grid, resource, k2 = run_chunk(grid, resource, kernel_fft, k2, params, chunk_size)
    grid.block_until_ready()

    tracker = PatternTracker()
    affect_log = []
    prev_masses = {}
    prev_values = {}
    snapshots = []  # periodic grid snapshots

    n_chunks = n_steps // chunk_size
    t_start = time.time()

    for chunk_idx in range(n_chunks):
        grid, resource, k2 = run_chunk(grid, resource, kernel_fft, k2,
                                        params, chunk_size)
        step = (chunk_idx + 1) * chunk_size

        if chunk_idx % measure_every != 0:
            continue

        grid_np = np.array(grid)
        patterns = detect_patterns(grid_np)
        patterns = tracker.update(patterns, step=step)

        for p in patterns:
            pm = prev_masses.get(p.id)
            pv = prev_values.get(p.id)
            hist = tracker.history.get(p.id, [])
            affect = measure_all(
                p, pm, pv, hist,
                grid, kernel_fft,
                config['growth_mu'], config['growth_sigma'], N,
                step_num=step,
            )
            affect_log.append(affect)
            prev_masses[p.id] = p.mass
            prev_values[p.id] = p.values.copy()

        # Periodic snapshot
        if chunk_idx % (measure_every * 10) == 0:
            snapshots.append({
                'step': step,
                'grid': grid_np.tolist(),
                'resource': np.array(resource).tolist(),
                'n_patterns': len(patterns),
                'total_mass': float(grid_np.sum()),
            })

            elapsed = time.time() - t_start
            print(f"  [{label}] Step {step}/{n_steps}: "
                  f"{len(patterns)} patterns, "
                  f"mass={grid_np.sum():.0f}, "
                  f"{step/elapsed:.0f} steps/s")

    elapsed = time.time() - t_start
    print(f"  [{label}] Completed {n_steps} steps in {elapsed:.1f}s")

    return {
        'config': config,
        'label': label,
        'n_steps': n_steps,
        'elapsed': elapsed,
        'affect_log': [(a.__dict__) for a in affect_log],
        'survival_stats': tracker.survival_stats(),
        'tracker': tracker,
    }


def forcing_function_ablation(n_steps=50000, seed=42):
    """Compare conditions with/without resource pressure.

    Tests Part 1's prediction: forcing functions increase integration.

    Conditions:
    1. Baseline: standard Lenia, no resources (uniform resource field)
    2. Resources: depletion and regeneration create viability pressure
    3. Scarce: high consumption, low regeneration (harsh environment)
    """
    conditions = {
        'no_resources': {
            **DEFAULT_CONFIG,
            'resource_consume': 0.0,
            'resource_regen': 1.0,
        },
        'resources': DEFAULT_CONFIG,
        'scarce': {
            **DEFAULT_CONFIG,
            'resource_consume': 0.06,
            'resource_regen': 0.002,
        },
    }

    results = {}
    for name, config in conditions.items():
        print(f"\n{'='*60}")
        print(f"Condition: {name}")
        print(f"{'='*60}")
        results[name] = run_experiment(
            config, n_steps=n_steps, seed=seed, label=name
        )

    # Compare
    print(f"\n{'='*60}")
    print("ABLATION COMPARISON")
    print(f"{'='*60}")
    print(f"{'Condition':<15} {'Patterns':>10} {'Mean Phi':>10} "
          f"{'Mean ER':>10} {'Survival':>10}")
    print("-" * 60)

    for name, res in results.items():
        affects = res['affect_log']
        if affects:
            mean_phi = np.mean([a['integration'] for a in affects])
            mean_er = np.mean([a['effective_rank'] for a in affects])
        else:
            mean_phi = mean_er = 0.0
        stats = res['survival_stats']
        n_active = stats.get('n_active', 0)
        mean_life = stats.get('mean_lifetime', 0)
        print(f"{name:<15} {n_active:>10} {mean_phi:>10.4f} "
              f"{mean_er:>10.1f} {mean_life:>10.1f}")

    return results


def perturbation_experiment(config=None, seed=42):
    """THE core experiment: drought/recovery cycle.

    Protocol:
    1. Equilibrate (5000 steps, normal resources)
    2. Baseline measurement (2000 steps)
    3. DROUGHT: resource regeneration stops (3000 steps)
       Patterns gradually starve as local resources deplete.
    4. RECOVERY: resources regenerate again (3000 steps)
       Surviving patterns recover; new patterns may emerge.

    Predictions (from Part 1):
    - During drought: negative valence, increasing arousal,
      integration should change (up for biological-like, down for zombie-like)
    - During recovery: positive valence, decreasing arousal
    - Survival should correlate with integration
    """
    if config is None:
        config = DEFAULT_CONFIG

    N = config['grid_size']
    rng = random.PRNGKey(seed)
    k1, k2 = random.split(rng)

    kernel = make_kernel(config['kernel_radius'], config['kernel_peak'],
                         config['kernel_width'])
    kernel_fft = make_kernel_fft(kernel, N)
    params = make_params(config)

    grid, resource = init_soup(N, k1, growth_mu=config['growth_mu'])

    print("="*60)
    print("DROUGHT / RECOVERY EXPERIMENT")
    print("="*60)

    # Normal and drought params
    normal_params = make_params(config)
    drought_config = {**config, 'resource_regen': 0.0001}  # near-zero regen
    drought_params = make_params(drought_config)

    # JIT warmup
    grid, resource, k2 = run_chunk(grid, resource, kernel_fft, k2, normal_params, 100)
    grid.block_until_ready()

    # Phase 1: Equilibration (5000 steps)
    print("\nPhase 1: Equilibration (5000 steps)...")
    for i in range(50):
        grid, resource, k2 = run_chunk(grid, resource, kernel_fft, k2, normal_params, 100)

    tracker = PatternTracker()
    prev_masses = {}
    prev_values = {}

    def measure_step(grid, step):
        grid_np = np.array(grid)
        patterns = detect_patterns(grid_np)
        patterns = tracker.update(patterns, step=step)
        affects = []
        for p in patterns:
            pm = prev_masses.get(p.id)
            pv = prev_values.get(p.id)
            hist = tracker.history.get(p.id, [])
            affect = measure_all(
                p, pm, pv, hist,
                grid, kernel_fft,
                config['growth_mu'], config['growth_sigma'], N,
                step_num=step,
            )
            affects.append(affect)
            prev_masses[p.id] = p.mass
            prev_values[p.id] = p.values.copy()
        return patterns, affects

    # Phase 2: Baseline (2000 steps, measuring every 200)
    print("Phase 2: Baseline measurement (2000 steps)...")
    timeline = []  # list of (step, phase, n_patterns, mean_V, mean_A, mean_Phi, mean_mass, mean_resource)

    for chunk in range(20):
        grid, resource, k2 = run_chunk(grid, resource, kernel_fft, k2, normal_params, 100)
        step = 5000 + (chunk + 1) * 100
        if chunk % 2 == 0:
            patterns, affects = measure_step(grid, step)
            if affects:
                entry = {
                    'step': step, 'phase': 'baseline',
                    'n_patterns': len(patterns),
                    'mass': float(np.array(grid).sum()),
                    'resource': float(np.array(resource).mean()),
                    'mean_V': float(np.mean([a.valence for a in affects])),
                    'mean_A': float(np.mean([a.arousal for a in affects])),
                    'mean_Phi': float(np.mean([a.integration for a in affects])),
                    'mean_ER': float(np.mean([a.effective_rank for a in affects])),
                }
                timeline.append(entry)

    print(f"  Baseline: {timeline[-1]['n_patterns']} patterns, "
          f"mass={timeline[-1]['mass']:.0f}")

    # Phase 3: DROUGHT (3000 steps — resources stop regenerating)
    print("\n>>> DROUGHT BEGINS: resource regeneration stopped <<<")
    for chunk in range(30):
        grid, resource, k2 = run_chunk(grid, resource, kernel_fft, k2, drought_params, 100)
        step = 7000 + (chunk + 1) * 100
        if chunk % 2 == 0:
            patterns, affects = measure_step(grid, step)
            if affects:
                entry = {
                    'step': step, 'phase': 'drought',
                    'n_patterns': len(patterns),
                    'mass': float(np.array(grid).sum()),
                    'resource': float(np.array(resource).mean()),
                    'mean_V': float(np.mean([a.valence for a in affects])),
                    'mean_A': float(np.mean([a.arousal for a in affects])),
                    'mean_Phi': float(np.mean([a.integration for a in affects])),
                    'mean_ER': float(np.mean([a.effective_rank for a in affects])),
                }
                timeline.append(entry)
                if chunk % 6 == 0:
                    print(f"  Step {step}: {entry['n_patterns']} patterns, "
                          f"mass={entry['mass']:.0f}, "
                          f"resource={entry['resource']:.3f}, "
                          f"V={entry['mean_V']:+.4f}")

    # Phase 4: RECOVERY (3000 steps — resources regenerate again)
    print("\n>>> RECOVERY: resource regeneration restored <<<")
    for chunk in range(30):
        grid, resource, k2 = run_chunk(grid, resource, kernel_fft, k2, normal_params, 100)
        step = 10000 + (chunk + 1) * 100
        if chunk % 2 == 0:
            patterns, affects = measure_step(grid, step)
            if affects:
                entry = {
                    'step': step, 'phase': 'recovery',
                    'n_patterns': len(patterns),
                    'mass': float(np.array(grid).sum()),
                    'resource': float(np.array(resource).mean()),
                    'mean_V': float(np.mean([a.valence for a in affects])),
                    'mean_A': float(np.mean([a.arousal for a in affects])),
                    'mean_Phi': float(np.mean([a.integration for a in affects])),
                    'mean_ER': float(np.mean([a.effective_rank for a in affects])),
                }
                timeline.append(entry)
                if chunk % 6 == 0:
                    print(f"  Step {step}: {entry['n_patterns']} patterns, "
                          f"mass={entry['mass']:.0f}, "
                          f"resource={entry['resource']:.3f}, "
                          f"V={entry['mean_V']:+.4f}")

    # ---- Analysis ----
    print("\n" + "="*60)
    print("RESULTS: Affect Dynamics During Drought/Recovery")
    print("="*60)

    phases = {}
    for entry in timeline:
        p = entry['phase']
        if p not in phases:
            phases[p] = []
        phases[p].append(entry)

    for phase_name in ['baseline', 'drought', 'recovery']:
        entries = phases.get(phase_name, [])
        if not entries:
            continue
        print(f"\n  {phase_name.upper()} (n={len(entries)} timepoints):")
        print(f"    Patterns:    {np.mean([e['n_patterns'] for e in entries]):.0f} "
              f"({entries[0]['n_patterns']} -> {entries[-1]['n_patterns']})")
        print(f"    Mass:        {np.mean([e['mass'] for e in entries]):.0f} "
              f"({entries[0]['mass']:.0f} -> {entries[-1]['mass']:.0f})")
        print(f"    Resource:    {np.mean([e['resource'] for e in entries]):.3f}")
        print(f"    Valence:     {np.mean([e['mean_V'] for e in entries]):+.4f}")
        print(f"    Arousal:     {np.mean([e['mean_A'] for e in entries]):.4f}")
        print(f"    Integration: {np.mean([e['mean_Phi'] for e in entries]):.4f}")
        print(f"    Eff. Rank:   {np.mean([e['mean_ER'] for e in entries]):.2f}")

    # Time series
    print("\n" + "-"*60)
    print("TIME SERIES (step | phase | patterns | V | A | Phi | resource)")
    print("-"*60)
    for e in timeline:
        print(f"  {e['step']:>6d} | {e['phase']:<8s} | {e['n_patterns']:>4d} | "
              f"{e['mean_V']:+.4f} | {e['mean_A']:.4f} | "
              f"{e['mean_Phi']:.4f} | {e['resource']:.3f}")

    return {'timeline': timeline, 'tracker': tracker}


def save_results(results, path):
    """Save experiment results to JSON."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict) and 'tracker' in v:
            v = {kk: vv for kk, vv in v.items() if kk != 'tracker'}
        serializable[k] = v

    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"Results saved to {path}")


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'sanity'

    if mode == 'sanity':
        success, affects, tracker = sanity_check()
        if not success:
            print("\nRetrying with FORGIVING_CONFIG...")
            success, affects, tracker = sanity_check(config=FORGIVING_CONFIG)

    elif mode == 'perturb':
        results = perturbation_experiment()

    elif mode == 'experiment':
        n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
        results = run_experiment(DEFAULT_CONFIG, n_steps=n_steps)
        save_results({'default': results},
                     'results/experiment_results.json')

    elif mode == 'ablation':
        n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 50000
        results = forcing_function_ablation(n_steps=n_steps)
        save_results(results, 'results/ablation_results.json')

    elif mode == 'evolve':
        from v11_evolution import evolve_in_situ, stress_test
        n_cycles = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        curriculum = '--curriculum' in sys.argv
        result = evolve_in_situ(n_cycles=n_cycles, curriculum=curriculum)
        print("\nRunning stress test...")
        stress = stress_test(
            result['final_grid'], result['final_resource'])

    elif mode == 'seeds':
        from v11_evolution import discover_seeds
        n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        seeds = discover_seeds(n_seeds=n_seeds)

    elif mode == 'pipeline':
        from v11_evolution import full_pipeline
        result = full_pipeline()

    elif mode == 'hetero':
        from v11_evolution import evolve_hetero, stress_test_hetero
        n_cycles = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        curriculum = '--curriculum' in sys.argv
        result = evolve_hetero(n_cycles=n_cycles, curriculum=curriculum)
        print("\nRunning stress test...")
        stress = stress_test_hetero(
            result['final_grid'], result['final_resource'],
            result['mu_field'], result['sigma_field'])

    elif mode == 'hetero-pipeline':
        from v11_evolution import full_pipeline_hetero
        result = full_pipeline_hetero()

    elif mode == 'multichannel':
        from v11_evolution import evolve_multichannel, stress_test_mc
        n_cycles = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        curriculum = '--curriculum' in sys.argv
        result = evolve_multichannel(n_cycles=n_cycles, curriculum=curriculum)
        print("\nRunning stress test...")
        stress = stress_test_mc(
            result['final_grid'], result['final_resource'],
            result['coupling'])

    elif mode == 'mc-pipeline':
        from v11_evolution import full_pipeline_mc
        result = full_pipeline_mc()

    elif mode == 'hd':
        from v11_evolution import evolve_hd, stress_test_hd
        n_cycles = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 30
        # Parse --channels flag
        C = 64
        for i, arg in enumerate(sys.argv):
            if arg == '--channels' and i + 1 < len(sys.argv):
                C = int(sys.argv[i + 1])
        curriculum = '--curriculum' in sys.argv
        result = evolve_hd(n_cycles=n_cycles, curriculum=curriculum, C=C)
        print("\nRunning stress test...")
        stress = stress_test_hd(
            result['final_grid'], result['final_resource'],
            result['coupling'], C=C,
            bandwidth=result.get('bandwidth', 8.0))

    elif mode == 'hd-pipeline':
        from v11_evolution import full_pipeline_hd
        C = 64
        for i, arg in enumerate(sys.argv):
            if arg == '--channels' and i + 1 < len(sys.argv):
                C = int(sys.argv[i + 1])
        result = full_pipeline_hd(C=C)

    elif mode == 'hier':
        from v11_evolution import evolve_hier
        n_cycles = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 30
        C = 64
        for i, arg in enumerate(sys.argv):
            if arg == '--channels' and i + 1 < len(sys.argv):
                C = int(sys.argv[i + 1])
        curriculum = '--curriculum' in sys.argv
        result = evolve_hier(n_cycles=n_cycles, curriculum=curriculum, C=C)

    elif mode == 'hier-pipeline':
        from v11_evolution import full_pipeline_hier
        C = 64
        for i, arg in enumerate(sys.argv):
            if arg == '--channels' and i + 1 < len(sys.argv):
                C = int(sys.argv[i + 1])
        result = full_pipeline_hier(C=C)

    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python v11_run.py "
              "[sanity|perturb|experiment|ablation|evolve|seeds|pipeline"
              "|hetero|hetero-pipeline|multichannel|mc-pipeline|hd|hd-pipeline"
              "|hier|hier-pipeline]")
