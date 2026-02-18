"""V19: Bottleneck Furnace Mechanism Experiment.

Tests whether the Bottleneck Furnace effect is:
  SELECTION: High-Phi patterns survive bottleneck; others die.
             After bottleneck, population mean Phi is higher because
             low-Phi patterns were culled. Novel-stress robustness is
             predictable from current Phi_baseline (same for all conditions).
  CREATION:  The bottleneck itself modifies survivors in a way that
             increases novel-stress generalization beyond what their
             Phi_baseline would predict.

Design:
  Phase 1 (cycles 0-9):   Standard V18 — identical for all conditions
  Phase 2 (cycles 10-19): Conditions diverge
    A (BOTTLENECK): 2 severe droughts/cycle, mortality ~90-95%
    B (GRADUAL):    Mild chronic stress, mortality <25%/cycle
    C (CONTROL):    Standard V18 stress schedule
  Phase 3 (cycles 20-24): Novel stress — params FROZEN
    Identical extreme drought applied to all 3 conditions.
    Measure Phi_baseline and Phi_stress per pattern.

Statistical test:
  novel_robustness ~ phi_baseline + is_bottleneck + is_gradual

  Significant is_bottleneck after controlling phi_baseline -> CREATION
  is_bottleneck ≈ 0 -> SELECTION
"""

import sys
import os
import json
import copy
import time
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v18_substrate import (
    generate_v18_config, init_v18, run_v18_chunk,
    compute_insulation_metrics,
)
from v18_evolution import mutate_v18_params
from v14_evolution import compute_foraging_metrics
from v15_substrate import generate_resource_patches, compute_patch_regen_mask
from v11_patterns import detect_patterns_mc, PatternTracker
from v11_affect_hd import measure_all_hd
from v11_evolution import score_fitness_functional
from v11_substrate import perturb_resource_bloom
from v11_substrate_hd import init_soup_hd

import jax
import jax.numpy as jnp
from jax import random


# ============================================================================
# State Serialisation (JAX arrays → numpy and back)
# ============================================================================

def save_state(grid, resource, rng, params, config):
    """Save evolvable state as numpy for condition forking."""
    return {
        'grid': np.array(grid),
        'resource': np.array(resource),
        'rng': np.array(rng),
        'params': {k: (np.array(v) if hasattr(v, '__jax_array__')
                       else v)
                   for k, v in params.items()},
        'config': config,
    }


def restore_state(saved):
    """Restore saved state back to JAX arrays."""
    grid = jnp.array(saved['grid'])
    resource = jnp.array(saved['resource'])
    rng = jnp.array(saved['rng'])
    params = {}
    for k, v in saved['params'].items():
        if isinstance(v, np.ndarray):
            params[k] = jnp.array(v)
        else:
            params[k] = v
    return grid, resource, rng, params


def serialize(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64, np.float16,
                        np.float_, np.floating)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64, np.int16, np.int_)):
        return int(obj)
    if hasattr(obj, 'item'):
        return obj.item()
    return str(obj)


# ============================================================================
# Single-Cycle Evolution Logic
# ============================================================================

def _make_run_config(config, params):
    cfg = {**config}
    cfg['tau'] = params['tau']
    cfg['gate_beta'] = params['gate_beta']
    cfg['chemotaxis_strength'] = params['eta']
    cfg['motor_sensitivity'] = params['motor_sensitivity']
    cfg['motor_threshold'] = params['motor_threshold']
    cfg['memory_lambdas'] = params['memory_lambdas']
    cfg['boundary_width'] = params['boundary_width']
    cfg['insulation_beta'] = params['insulation_beta']
    cfg['internal_gain'] = params['internal_gain']
    cfg['activity_threshold'] = params['activity_threshold']
    return cfg


def run_one_cycle(grid, resource, rng, params, config, patches,
                  h_embed, kernel_ffts, internal_kernel_ffts,
                  coupling, coupling_row_sums,
                  recurrence_coupling, recurrence_crs,
                  box_fft,
                  # Stress control
                  stress_mode,          # 'control', 'bottleneck', 'gradual', 'novel'
                  total_step_offset,
                  steps_per_cycle=5000,
                  chunk=50,
                  allow_mutation=True,
                  allow_culling=True,
                  tracker=None,
                  prev_masses=None,
                  prev_values=None,
                  ):
    """Run one cycle of V18 evolution under the specified stress mode.

    stress_mode:
      'control'   — standard V18 schedule (random regen fraction, random drought)
      'bottleneck'— 2 severe droughts per cycle, ~90-95% mortality
      'gradual'   — mild continuous stress, no discrete droughts, <25% mortality
      'novel'     — extreme single drought (2000 steps), params frozen
    """
    if tracker is None:
        tracker = PatternTracker()
    if prev_masses is None:
        prev_masses = {}
    if prev_values is None:
        prev_values = {}

    run_config = _make_run_config(config, params)
    N = config['grid_size']
    patch_shift_period = config['patch_shift_period']

    rng_np = np.random.RandomState(int(np.array(rng)[0]) % (2**31))

    # ---- Build stress schedule for this cycle ----
    # All durations rounded to chunk boundaries to ensure correct detection.
    if stress_mode == 'control':
        # Standard: random regen 40-60% of baseline, random drought length
        stress_regen = config['resource_regen'] * rng_np.uniform(0.40, 0.60)
        raw_drought = int(rng_np.randint(500, 1501))
        drought_dur = min((raw_drought // chunk) * chunk, steps_per_cycle - chunk)
        drought_dur = max(drought_dur, chunk)
        baseline_dur = steps_per_cycle - drought_dur
        # One drought block starting at baseline_dur
        drought_segments = [(baseline_dur, drought_dur, stress_regen)]

    elif stress_mode == 'bottleneck':
        # 2 severe droughts (regen=8% baseline), ~90-95% mortality
        # Layout: gap | drought | gap | drought | gap
        severe_regen = config['resource_regen'] * 0.08
        # Use 40% of cycle per drought, cap to steps_per_cycle
        drought_dur = max(chunk, min((steps_per_cycle * 2 // 5 // chunk) * chunk,
                                     steps_per_cycle // 3))
        gap = max(chunk, (steps_per_cycle - 2 * drought_dur) // 3 // chunk * chunk)
        d1_start = gap
        d2_start = gap + drought_dur + gap
        drought_segments = [
            (d1_start, drought_dur, severe_regen),
            (d2_start, drought_dur, severe_regen),
        ]

    elif stress_mode == 'gradual':
        # Mild continuous stress: 55% regen throughout all steps, no bursts
        mild_regen = config['resource_regen'] * 0.55
        drought_segments = [(0, steps_per_cycle, mild_regen)]

    elif stress_mode == 'novel':
        # Novel extreme: brief baseline window to measure Phi_base, then severe drought
        # Drought 4% regen — more extreme than anything in Phase 2 training
        extreme_regen = config['resource_regen'] * 0.04
        # Baseline: first 20% of cycle (at least 1 chunk)
        baseline_dur = max(chunk, (steps_per_cycle // 5 // chunk) * chunk)
        drought_dur = steps_per_cycle - baseline_dur
        drought_segments = [(baseline_dur, drought_dur, extreme_regen)]

    else:
        raise ValueError(f"Unknown stress_mode: {stress_mode}")

    # ---- Helpers: check if a step is drought / get regen ----
    def is_step_drought(step):
        for (start, dur, _) in drought_segments:
            if start <= step < start + dur:
                return True
        return False

    def get_regen_for_step(step):
        for (start, dur, regen) in drought_segments:
            if start <= step < start + dur:
                return regen
        return config['resource_regen']

    # ---- Run the cycle ----
    baseline_affects = {}
    stress_affects = {}
    baseline_survival = {}
    patterns_at_start = None

    measure_every = max(200, steps_per_cycle // 8)
    max_measure = 30

    for step in range(0, steps_per_cycle, chunk):
        total_step = total_step_offset + step

        regen_mask = compute_patch_regen_mask(
            N, patches, step=total_step,
            shift_period=patch_shift_period)
        regen_mask = jnp.array(regen_mask)

        # Current config with step-specific regen
        step_regen = get_regen_for_step(step)
        is_drought = is_step_drought(step)
        step_cfg = {**run_config}
        step_cfg['resource_regen'] = step_regen
        if is_drought:
            step_cfg['resource_consume'] = config['resource_consume'] * 1.3

        grid, resource, rng = run_v18_chunk(
            grid, resource, h_embed, kernel_ffts, internal_kernel_ffts,
            step_cfg, coupling, coupling_row_sums,
            recurrence_coupling, recurrence_crs,
            rng, n_steps=chunk, box_fft=box_fft,
            regen_mask=regen_mask, drought=is_drought)

        if step % measure_every < chunk:
            grid_np = np.array(grid)
            patterns = detect_patterns_mc(grid_np, threshold=0.15)
            tracker.update(patterns, step=total_step)

            if patterns_at_start is None:
                patterns_at_start = patterns

            active_sorted = sorted(
                tracker.active.values(),
                key=lambda p: p.mass, reverse=True)

            for p in active_sorted[:max_measure]:
                pid = p.id
                hist = tracker.history.get(pid, [])
                pm = prev_masses.get(pid)
                pv = prev_values.get(pid)

                affect, phi_spec, eff_rank, phi_spat, _ = measure_all_hd(
                    p, pm, pv, hist,
                    jnp.array(grid_np), kernel_ffts, coupling,
                    step_cfg, N, step_num=total_step, fast=True)

                if not is_drought:
                    baseline_affects.setdefault(pid, []).append(affect)
                else:
                    stress_affects.setdefault(pid, []).append(affect)
                baseline_survival[pid] = step

                prev_masses[pid] = p.mass
                prev_values[pid] = p.values.copy()

            for p in tracker.active.values():
                baseline_survival[p.id] = step

    # ---- Scoring ----
    grid_np = np.array(grid)
    final_patterns = detect_patterns_mc(grid_np, threshold=0.15)
    tracker.update(final_patterns, step=total_step_offset + steps_per_cycle)

    n_before = len(baseline_affects)
    n_after = len(stress_affects)
    mortality = 1.0 - (n_after / max(n_before, 1))

    all_phi_base, all_phi_stress, all_robustness = [], [], []
    per_pattern_data = []

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
            rob = np.mean(s_phis) / np.mean(b_phis)
            all_robustness.append(rob)
            per_pattern_data.append({
                'pid': pid,
                'phi_base': float(np.mean(b_phis)),
                'phi_stress': float(np.mean(s_phis)),
                'robustness': float(rob),
            })

    mean_phi_base = float(np.mean(all_phi_base)) if all_phi_base else 0.0
    mean_phi_stress = float(np.mean(all_phi_stress)) if all_phi_stress else 0.0
    mean_robustness = float(np.mean(all_robustness)) if all_robustness else 0.0
    phi_increase_frac = float(np.mean([r > 1.0 for r in all_robustness])) if all_robustness else 0.0

    # ---- Selection & mutation (unless frozen) ----
    fitness_scores = {}
    if allow_culling or allow_mutation:
        for pid in set(list(baseline_affects.keys()) + list(stress_affects.keys())):
            b_aff = baseline_affects.get(pid, [])
            s_aff = stress_affects.get(pid, [])
            surv = baseline_survival.get(pid, 0)
            fitness_scores[pid] = score_fitness_functional(
                b_aff, s_aff, surv, steps_per_cycle)

    if allow_culling and fitness_scores:
        sorted_pids = sorted(fitness_scores, key=fitness_scores.get, reverse=True)
        n_cull = int(len(sorted_pids) * 0.3)
        cull_pids = set(sorted_pids[-n_cull:]) if n_cull > 0 else set()

        for pid in cull_pids:
            if pid in tracker.active:
                p = tracker.active[pid]
                if len(p.cells) > 0:
                    grid = grid.at[:, p.cells[:, 0], p.cells[:, 1]].set(0.0)

        for pid in sorted_pids[:5]:
            if pid in tracker.active:
                p = tracker.active[pid]
                if len(p.cells) > 0:
                    rng, k_boost = random.split(rng)
                    boost = 0.05 * random.uniform(
                        k_boost, grid[:, p.cells[:, 0], p.cells[:, 1]].shape)
                    grid = grid.at[:, p.cells[:, 0], p.cells[:, 1]].add(boost)
                    grid = jnp.clip(grid, 0.0, 1.0)

    if allow_mutation and fitness_scores:
        rng, k_mut = random.split(rng)
        (params['tau'], params['gate_beta'], coupling,
         params['eta'], params['motor_sensitivity'], params['motor_threshold'],
         params['memory_lambdas'],
         params['boundary_width'], params['insulation_beta'],
         params['internal_gain'], params['activity_threshold'],
         recurrence_coupling) = mutate_v18_params(
            params['tau'], params['gate_beta'], coupling,
            params['eta'], params['motor_sensitivity'], params['motor_threshold'],
            params['memory_lambdas'],
            params['boundary_width'], params['insulation_beta'],
            params['internal_gain'], params['activity_threshold'],
            recurrence_coupling, k_mut)
        coupling_row_sums = coupling.sum(axis=1)
        recurrence_crs = recurrence_coupling.sum(axis=1)

    # Resource bloom
    rng, k_bloom, k_bloom2 = random.split(rng, 3)
    cx = int(random.randint(k_bloom, (), 10, N - 10))
    cy = int(random.randint(k_bloom2, (), 10, N - 10))
    resource = perturb_resource_bloom(resource, (cy, cx), radius=N // 3, intensity=0.8)

    stats_out = {
        'n_patterns': len(final_patterns),
        'n_before': n_before,
        'n_after': n_after,
        'mortality': mortality,
        'mean_phi_base': mean_phi_base,
        'mean_phi_stress': mean_phi_stress,
        'mean_robustness': mean_robustness,
        'phi_increase_frac': phi_increase_frac,
        'per_pattern': per_pattern_data,
        'stress_mode': stress_mode,
        'tau': float(params['tau']),
        'boundary_width': float(params['boundary_width']),
        'internal_gain': float(params['internal_gain']),
        'memory_lambdas': [float(l) for l in params['memory_lambdas']],
    }

    return (grid, resource, rng, params,
            coupling, coupling_row_sums,
            recurrence_coupling, recurrence_crs,
            stats_out, tracker, prev_masses, prev_values)


# ============================================================================
# Main Experiment
# ============================================================================

def run_v19(seed=42, C=16, N=128, output_dir='/home/ubuntu/results/v19',
            steps_per_cycle=5000, n_phase1=10, n_phase2=10, n_phase3=5):
    """Run the full V19 mechanism experiment for one seed.

    Runs Phase 1 (shared), then forks into 3 Phase 2 conditions (A/B/C),
    then runs Phase 3 (novel stress, params frozen) for each condition.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(f"V19 BOTTLENECK FURNACE MECHANISM EXPERIMENT")
    print(f"  Seed: {seed}, C={C}, N={N}")
    print(f"  Phase 1: {n_phase1} cycles (standard)")
    print(f"  Phase 2: {n_phase2} cycles per condition (A=bottleneck, B=gradual, C=control)")
    print(f"  Phase 3: {n_phase3} cycles novel stress (params frozen)")
    print("=" * 70)

    config = generate_v18_config(
        C=C, N=N, seed=seed,
        similarity_radius=5,
        chemotaxis_strength=0.5,
        memory_channels=2,
        n_resource_patches=4,
        patch_shift_period=500,
    )
    config['maintenance_rate'] = 0.003
    config['resource_consume'] = 0.003
    config['resource_regen'] = 0.01

    (grid, resource, h_embed, kernel_ffts, internal_kernel_ffts,
     coupling, coupling_row_sums, recurrence_coupling, recurrence_crs,
     box_fft) = init_v18(config, seed=seed)

    rng = random.PRNGKey(seed)
    patches = generate_resource_patches(N, 4, seed=seed)

    params = {
        'tau': float(config['tau']),
        'gate_beta': float(config['gate_beta']),
        'eta': float(config['chemotaxis_strength']),
        'motor_sensitivity': float(config['motor_sensitivity']),
        'motor_threshold': float(config['motor_threshold']),
        'memory_lambdas': list(config['memory_lambdas']),
        'boundary_width': float(config['boundary_width']),
        'insulation_beta': float(config['insulation_beta']),
        'internal_gain': float(config['internal_gain']),
        'activity_threshold': float(config['activity_threshold']),
    }

    chunk = 50

    # JIT warmup
    run_config = _make_run_config(config, params)
    regen_mask = compute_patch_regen_mask(N, patches, step=0,
                                          shift_period=config['patch_shift_period'])
    regen_mask = jnp.array(regen_mask)
    print("JIT compiling...", end=" ", flush=True)
    t0 = time.time()
    grid, resource, rng = run_v18_chunk(
        grid, resource, h_embed, kernel_ffts, internal_kernel_ffts,
        run_config, coupling, coupling_row_sums,
        recurrence_coupling, recurrence_crs,
        rng, n_steps=chunk, box_fft=box_fft, regen_mask=regen_mask)
    jnp.array(grid).block_until_ready()
    print(f"done ({time.time()-t0:.1f}s)")

    # Warmup (1000 steps)
    warmup_n = 1000 // chunk
    print(f"Warmup ({warmup_n * chunk} steps)...", end=" ", flush=True)
    t0 = time.time()
    for _ in range(warmup_n):
        grid, resource, rng = run_v18_chunk(
            grid, resource, h_embed, kernel_ffts, internal_kernel_ffts,
            run_config, coupling, coupling_row_sums,
            recurrence_coupling, recurrence_crs,
            rng, n_steps=chunk, box_fft=box_fft, regen_mask=regen_mask)
    jnp.array(grid).block_until_ready()
    total_step = warmup_n * chunk
    print(f"done ({time.time()-t0:.1f}s)")

    pats = detect_patterns_mc(np.array(grid), threshold=0.15)
    print(f"  {len(pats)} patterns after warmup\n")

    all_results = {'seed': seed, 'config': {k: v for k, v in config.items()
                                              if not isinstance(v, np.ndarray)}}

    # ===========================================================
    # PHASE 1: Standard evolution (10 cycles, shared)
    # ===========================================================
    print("=" * 60)
    print("PHASE 1: Standard evolution (shared)")
    print("=" * 60)

    tracker_p1 = PatternTracker()
    prev_masses_p1 = {}
    prev_values_p1 = {}
    phase1_stats = []

    for cycle in range(n_phase1):
        t0 = time.time()
        (grid, resource, rng, params,
         coupling, coupling_row_sums,
         recurrence_coupling, recurrence_crs,
         stats_out, tracker_p1, prev_masses_p1, prev_values_p1) = run_one_cycle(
            grid, resource, rng, params, config, patches,
            h_embed, kernel_ffts, internal_kernel_ffts,
            coupling, coupling_row_sums,
            recurrence_coupling, recurrence_crs,
            box_fft,
            stress_mode='control',
            total_step_offset=total_step,
            steps_per_cycle=steps_per_cycle,
            chunk=chunk,
            allow_mutation=True,
            allow_culling=True,
            tracker=tracker_p1,
            prev_masses=prev_masses_p1,
            prev_values=prev_values_p1,
        )
        total_step += steps_per_cycle
        stats_out['cycle'] = cycle
        phase1_stats.append(stats_out)
        elapsed = time.time() - t0
        print(f"P1 C{cycle:02d} | pat={stats_out['n_patterns']:4d} | "
              f"mort={stats_out['mortality']:.0%} | "
              f"rob={stats_out['mean_robustness']:.3f} | "
              f"gain={stats_out['internal_gain']:.2f} | {elapsed:.0f}s",
              flush=True)

    all_results['phase1'] = phase1_stats

    # Save Phase 1 final state as snapshot for condition forking
    phase1_snapshot = save_state(grid, resource, rng, params, config)
    phase1_snapshot['coupling'] = np.array(coupling)
    phase1_snapshot['coupling_row_sums'] = np.array(coupling_row_sums)
    phase1_snapshot['recurrence_coupling'] = np.array(recurrence_coupling)
    phase1_snapshot['recurrence_crs'] = np.array(recurrence_crs)
    phase1_snapshot['total_step'] = total_step
    print(f"\nPhase 1 complete. Snapshot saved.\n")

    # ===========================================================
    # PHASE 2: Three conditions fork from Phase 1
    # ===========================================================

    condition_final_states = {}

    for condition, stress_mode in [('A', 'bottleneck'), ('B', 'gradual'), ('C', 'control')]:
        print("=" * 60)
        print(f"PHASE 2: Condition {condition} ({stress_mode.upper()})")
        print("=" * 60)

        # Restore Phase 1 state
        grid, resource, rng, params = restore_state(phase1_snapshot)
        coupling = jnp.array(phase1_snapshot['coupling'])
        coupling_row_sums = jnp.array(phase1_snapshot['coupling_row_sums'])
        recurrence_coupling = jnp.array(phase1_snapshot['recurrence_coupling'])
        recurrence_crs = jnp.array(phase1_snapshot['recurrence_crs'])
        total_step_cond = phase1_snapshot['total_step']

        tracker_p2 = PatternTracker()
        prev_masses_p2 = {}
        prev_values_p2 = {}
        phase2_stats = []

        for cycle in range(n_phase2):
            t0 = time.time()
            (grid, resource, rng, params,
             coupling, coupling_row_sums,
             recurrence_coupling, recurrence_crs,
             stats_out, tracker_p2, prev_masses_p2, prev_values_p2) = run_one_cycle(
                grid, resource, rng, params, config, patches,
                h_embed, kernel_ffts, internal_kernel_ffts,
                coupling, coupling_row_sums,
                recurrence_coupling, recurrence_crs,
                box_fft,
                stress_mode=stress_mode,
                total_step_offset=total_step_cond,
                steps_per_cycle=steps_per_cycle,
                chunk=chunk,
                allow_mutation=True,
                allow_culling=True,
                tracker=tracker_p2,
                prev_masses=prev_masses_p2,
                prev_values=prev_values_p2,
            )
            total_step_cond += steps_per_cycle
            stats_out['cycle'] = cycle
            stats_out['condition'] = condition
            phase2_stats.append(stats_out)
            elapsed = time.time() - t0
            print(f"P2-{condition} C{cycle:02d} | pat={stats_out['n_patterns']:4d} | "
                  f"mort={stats_out['mortality']:.0%} | "
                  f"rob={stats_out['mean_robustness']:.3f} | "
                  f"gain={stats_out['internal_gain']:.2f} | {elapsed:.0f}s",
                  flush=True)

            # Rescue if near extinction
            grid_np = np.array(grid)
            pats_now = detect_patterns_mc(grid_np, threshold=0.15)
            if len(pats_now) < 5:
                print(f"  [RESCUE] {len(pats_now)} patterns — re-seeding")
                rng, k_reseed = random.split(rng)
                ch_mus = jnp.array(config['channel_mus'])
                fresh_grid, fresh_resource = init_soup_hd(N, C, k_reseed, ch_mus)
                grid = 0.3 * grid + 0.7 * fresh_grid
                resource = jnp.maximum(resource, 0.5 * fresh_resource)
                rc_cfg = _make_run_config(config, params)
                regen_mask_r = compute_patch_regen_mask(
                    N, patches, step=total_step_cond,
                    shift_period=config['patch_shift_period'])
                regen_mask_r = jnp.array(regen_mask_r)
                for _ in range(500 // chunk):
                    grid, resource, rng = run_v18_chunk(
                        grid, resource, h_embed, kernel_ffts, internal_kernel_ffts,
                        rc_cfg, coupling, coupling_row_sums,
                        recurrence_coupling, recurrence_crs,
                        rng, n_steps=chunk, box_fft=box_fft, regen_mask=regen_mask_r)

        all_results[f'phase2_{condition}'] = phase2_stats

        # Save condition's final state
        cond_state = save_state(grid, resource, rng, params, config)
        cond_state['coupling'] = np.array(coupling)
        cond_state['coupling_row_sums'] = np.array(coupling_row_sums)
        cond_state['recurrence_coupling'] = np.array(recurrence_coupling)
        cond_state['recurrence_crs'] = np.array(recurrence_crs)
        cond_state['total_step'] = total_step_cond
        condition_final_states[condition] = cond_state
        print(f"\nCondition {condition} Phase 2 complete.\n")

    # ===========================================================
    # PHASE 3: Novel stress (params frozen) — all conditions
    # ===========================================================

    print("=" * 60)
    print("PHASE 3: Novel stress test (params FROZEN)")
    print("=" * 60)

    phase3_all_patterns = []

    for condition in ['A', 'B', 'C']:
        print(f"\n--- Condition {condition} ---")
        cond_state = condition_final_states[condition]

        grid, resource, rng, params = restore_state(cond_state)
        coupling = jnp.array(cond_state['coupling'])
        coupling_row_sums = jnp.array(cond_state['coupling_row_sums'])
        recurrence_coupling = jnp.array(cond_state['recurrence_coupling'])
        recurrence_crs = jnp.array(cond_state['recurrence_crs'])
        total_step_cond = cond_state['total_step']

        tracker_p3 = PatternTracker()
        prev_masses_p3 = {}
        prev_values_p3 = {}
        phase3_stats = []

        for cycle in range(n_phase3):
            t0 = time.time()
            (grid, resource, rng, params,
             coupling, coupling_row_sums,
             recurrence_coupling, recurrence_crs,
             stats_out, tracker_p3, prev_masses_p3, prev_values_p3) = run_one_cycle(
                grid, resource, rng, params, config, patches,
                h_embed, kernel_ffts, internal_kernel_ffts,
                coupling, coupling_row_sums,
                recurrence_coupling, recurrence_crs,
                box_fft,
                stress_mode='novel',
                total_step_offset=total_step_cond,
                steps_per_cycle=steps_per_cycle,
                chunk=chunk,
                allow_mutation=False,  # FROZEN
                allow_culling=False,   # FROZEN
                tracker=tracker_p3,
                prev_masses=prev_masses_p3,
                prev_values=prev_values_p3,
            )
            total_step_cond += steps_per_cycle
            stats_out['cycle'] = cycle
            stats_out['condition'] = condition
            phase3_stats.append(stats_out)
            elapsed = time.time() - t0
            print(f"P3-{condition} C{cycle:02d} | pat={stats_out['n_patterns']:4d} | "
                  f"mort={stats_out['mortality']:.0%} | "
                  f"rob={stats_out['mean_robustness']:.3f} | {elapsed:.0f}s",
                  flush=True)

            # Collect per-pattern data tagged with condition
            for pp in stats_out['per_pattern']:
                pp['condition'] = condition
                pp['phase3_cycle'] = cycle
                phase3_all_patterns.append(pp)

        all_results[f'phase3_{condition}'] = phase3_stats

    # ===========================================================
    # STATISTICAL ANALYSIS
    # ===========================================================

    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS: Selection vs. Creation")
    print("=" * 60)

    analysis = analyze_v19(phase3_all_patterns)
    all_results['analysis'] = analysis

    print(f"\nCondition pattern counts:")
    for cond in ['A', 'B', 'C']:
        cond_data = [p for p in phase3_all_patterns if p['condition'] == cond]
        robs = [p['robustness'] for p in cond_data]
        phis = [p['phi_base'] for p in cond_data]
        print(f"  {cond}: n={len(robs)}, mean_rob={np.mean(robs):.3f}±{np.std(robs):.3f}, "
              f"mean_phi_base={np.mean(phis):.3f}")

    print(f"\nRegression: robustness ~ phi_base + is_bottleneck + is_gradual")
    reg = analysis.get('regression', {})
    print(f"  Intercept:      {reg.get('intercept', 0):.4f}")
    print(f"  phi_base coef:  {reg.get('phi_base_coef', 0):.4f}")
    print(f"  is_bottleneck:  {reg.get('bottleneck_coef', 0):.4f} "
          f"(p={reg.get('bottleneck_p', 1):.4f})")
    print(f"  is_gradual:     {reg.get('gradual_coef', 0):.4f} "
          f"(p={reg.get('gradual_p', 1):.4f})")
    print(f"  R²: {reg.get('r_squared', 0):.4f}")

    verdict = analysis.get('verdict', 'INSUFFICIENT DATA')
    print(f"\nVERDICT: {verdict}")

    # Save full results
    output_path = f'{output_dir}/v19_s{seed}_results.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=serialize)
    print(f"\nResults saved to {output_path}")

    return all_results


def analyze_v19(phase3_patterns):
    """Statistical test: selection vs. creation.

    Regression: robustness ~ phi_base + is_bottleneck + is_gradual
    Reference condition: C (control)
    """
    if len(phase3_patterns) < 10:
        return {'verdict': 'INSUFFICIENT DATA (n < 10)', 'n': len(phase3_patterns)}

    robs = np.array([p['robustness'] for p in phase3_patterns])
    phis = np.array([p['phi_base'] for p in phase3_patterns])
    is_bottleneck = np.array([1.0 if p['condition'] == 'A' else 0.0
                               for p in phase3_patterns])
    is_gradual = np.array([1.0 if p['condition'] == 'B' else 0.0
                            for p in phase3_patterns])
    conditions = [p['condition'] for p in phase3_patterns]

    # Normalize phi_base to avoid scale issues
    phi_mean = phis.mean()
    phi_std = phis.std() + 1e-10
    phis_norm = (phis - phi_mean) / phi_std

    # Multiple regression via numpy lstsq
    X = np.column_stack([np.ones(len(robs)), phis_norm, is_bottleneck, is_gradual])
    try:
        beta, residuals, rank, sv = np.linalg.lstsq(X, robs, rcond=None)
        intercept, phi_coef, bottleneck_coef, gradual_coef = beta

        # Compute R²
        y_pred = X @ beta
        ss_res = np.sum((robs - y_pred) ** 2)
        ss_tot = np.sum((robs - robs.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # p-values via t-test on coefficients
        n = len(robs)
        p = X.shape[1]
        mse = ss_res / max(n - p, 1)
        try:
            cov = mse * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(np.abs(cov)))
        except np.linalg.LinAlgError:
            se = np.ones(p) * np.nan

        def coef_pval(coef, se_val):
            if np.isnan(se_val) or se_val < 1e-10:
                return float('nan')
            t = coef / se_val
            return float(2 * (1 - stats.t.cdf(abs(t), df=n - p)))

        bottleneck_p = coef_pval(bottleneck_coef, se[2])
        gradual_p = coef_pval(gradual_coef, se[3])

        reg = {
            'intercept': float(intercept),
            'phi_base_coef': float(phi_coef),
            'bottleneck_coef': float(bottleneck_coef),
            'gradual_coef': float(gradual_coef),
            'bottleneck_p': float(bottleneck_p) if not np.isnan(bottleneck_p) else 1.0,
            'gradual_p': float(gradual_p) if not np.isnan(gradual_p) else 1.0,
            'r_squared': float(r_squared),
            'n': n,
        }

        # Verdict
        alpha = 0.05
        if reg['bottleneck_p'] < alpha and reg['bottleneck_coef'] > 0:
            verdict = (f"CREATION: Bottleneck condition shows significantly higher "
                       f"novel-stress robustness (β={bottleneck_coef:.3f}, "
                       f"p={bottleneck_p:.4f}) after controlling for Phi_baseline.")
        elif reg['bottleneck_p'] >= alpha and abs(bottleneck_coef) < 0.05:
            verdict = (f"SELECTION: Bottleneck advantage explained by higher Phi_baseline "
                       f"(bottleneck_coef={bottleneck_coef:.3f}, p={bottleneck_p:.4f} ns). "
                       f"Phi_base alone predicts robustness (phi_coef={phi_coef:.3f}).")
        else:
            verdict = (f"MIXED/INCONCLUSIVE: bottleneck_coef={bottleneck_coef:.3f} "
                       f"p={bottleneck_p:.4f}. Need more data.")

        # Summary by condition
        cond_summary = {}
        for cond in ['A', 'B', 'C']:
            cond_robs = [p['robustness'] for p in phase3_patterns if p['condition'] == cond]
            cond_phis = [p['phi_base'] for p in phase3_patterns if p['condition'] == cond]
            cond_summary[cond] = {
                'n': len(cond_robs),
                'mean_robustness': float(np.mean(cond_robs)) if cond_robs else 0.0,
                'std_robustness': float(np.std(cond_robs)) if cond_robs else 0.0,
                'mean_phi_base': float(np.mean(cond_phis)) if cond_phis else 0.0,
            }

        # Pairwise t-test: A vs B (most direct comparison)
        rob_A = [p['robustness'] for p in phase3_patterns if p['condition'] == 'A']
        rob_B = [p['robustness'] for p in phase3_patterns if p['condition'] == 'B']
        rob_C = [p['robustness'] for p in phase3_patterns if p['condition'] == 'C']

        ttest_AB = stats.ttest_ind(rob_A, rob_B) if rob_A and rob_B else None
        ttest_AC = stats.ttest_ind(rob_A, rob_C) if rob_A and rob_C else None

        return {
            'verdict': verdict,
            'regression': reg,
            'condition_summary': cond_summary,
            'ttest_A_vs_B': {
                'statistic': float(ttest_AB.statistic) if ttest_AB else None,
                'pvalue': float(ttest_AB.pvalue) if ttest_AB else None,
            },
            'ttest_A_vs_C': {
                'statistic': float(ttest_AC.statistic) if ttest_AC else None,
                'pvalue': float(ttest_AC.pvalue) if ttest_AC else None,
            },
            'n_total': n,
        }

    except Exception as e:
        return {'verdict': f'ANALYSIS ERROR: {e}', 'n': len(phase3_patterns)}


# ============================================================================
# Smoke Test
# ============================================================================

if __name__ == '__main__':
    print("V19 Smoke Test (C=8, N=64, 2+2+2 cycles)")
    result = run_v19(
        seed=42, C=8, N=64,
        steps_per_cycle=500,
        n_phase1=2, n_phase2=2, n_phase3=2,
        output_dir='/tmp/v19_smoke',
    )
    print("\nSmoke test complete!")
    if 'analysis' in result:
        print(f"Verdict: {result['analysis'].get('verdict', 'N/A')}")
