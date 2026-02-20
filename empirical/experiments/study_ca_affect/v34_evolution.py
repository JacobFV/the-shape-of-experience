"""V34: Φ-Inclusive Fitness — Does Selecting for Integration Work?

V22-V31 showed: ~30% of seeds develop high Φ regardless of prediction
target or architecture. Fitness has been pure survival. What if we
explicitly select for integration?

fitness_v34 = survival_time × (1 + α × mean_Φ)

Three questions:
  (a) Does this push 30% → 50%+?
  (b) Does it Goodhart — agents find cheap Φ hacks?
  (c) Does it produce qualitatively different dynamics?

Architecture: IDENTICAL to V27 (2-layer MLP, self-prediction).
Only change: fitness function includes a Φ component.

Pre-registered predictions:
  P1: HIGH fraction > 40% (direct selection increases success rate)
  P2: Mean Φ > 0.090 (V27 baseline)
  P3: Robustness maintained (Φ-optimized agents still survive)
  P4: No Goodhart artifacts (Φ from V34 correlates with robustness)
  P5: Qualitative behavioral difference from V27

Falsification:
  - P1 fails: selection cannot increase the rate (30% is fundamental)
  - P4 fails: Goodhart — Φ increases but robustness decreases
  - Agents develop trivially high Φ (e.g., all-same hidden states)
"""

import jax
import jax.numpy as jnp
import numpy as np
import json
import os
import time

from v27_substrate import (
    generate_v27_config, init_v27, make_v27_chunk_runner,
    compute_phi_hidden, compute_phi_sync, compute_robustness,
    extract_snapshot, extract_lr_np,
)
from v20_evolution import tournament_selection
from v27_evolution import (
    rescue_population_v27,
    compute_tick_usage,
    compute_sync_decay_stats,
    compute_lr_stats,
    compute_phenotype_drift,
    run_cycle_with_metrics,
    measure_robustness,
)


def generate_v34_config(**kwargs):
    """V34 config = V27 config + phi_fitness_alpha."""
    cfg = generate_v27_config(**kwargs)
    cfg.setdefault('phi_fitness_alpha', 2.0)  # Φ weight in fitness
    return cfg


def compute_fitness_with_phi(survival_steps, resources_gained,
                              metabolic_cost, alive, phi_per_agent, alpha):
    """Fitness = survival-based + alpha * mean_phi.

    phi_per_agent: (M,) array of per-agent Φ estimates.
    For simplicity, we use the population-level Φ for all alive agents
    (agents can't compute their own Φ individually in our framework).

    fitness = survival_time * resource_efficiency * (1 + alpha * phi_pop)
    """
    # Base fitness (from V20)
    survival = survival_steps.astype(np.float32)
    resources = resources_gained.astype(np.float32)
    cost = metabolic_cost * survival + 1e-8
    efficiency = resources / cost
    base_fitness = survival * efficiency

    # Φ bonus: population-level, applied to all alive agents equally
    alive_np = np.array(alive)
    phi_bonus = 1.0 + alpha * phi_per_agent
    phi_bonus = np.where(alive_np, phi_bonus, 0.0)

    fitness = base_fitness * phi_bonus
    return jnp.array(fitness)


def run_v34(seed, cfg, output_dir):
    """Run V34: V27 substrate with Φ-inclusive fitness.

    ONLY CHANGE from V27: fitness function includes Φ component.
    """
    os.makedirs(output_dir, exist_ok=True)

    key = jax.random.PRNGKey(seed)
    state, key = init_v27(seed, cfg)

    print(f"Warming up JIT (seed {seed})...")
    t0 = time.time()
    chunk_runner = make_v27_chunk_runner(cfg)
    state, _ = chunk_runner(state)
    print(f"  JIT warmup: {time.time()-t0:.1f}s")

    cycle_results = []
    snapshots = []
    alpha = cfg.get('phi_fitness_alpha', 2.0)

    for cycle in range(cfg['n_cycles']):
        t_start = time.time()

        state['phenotypes'] = state['genomes'].copy()
        state['hidden'] = jnp.zeros_like(state['hidden'])
        state['sync_matrices'] = jnp.zeros_like(state['sync_matrices'])
        state['energy'] = jnp.where(state['alive'], cfg['initial_energy'], 0.0)

        drought_every = cfg.get('drought_every', 0)
        is_drought = (drought_every > 0) and (cycle > 0) and (cycle % drought_every == 0)
        if is_drought:
            state['resources'] = state['resources'] * cfg.get('drought_depletion', 0.05)
            state['regen_rate'] = jnp.array(cfg.get('drought_regen', 0.00002))
            print(f"  [DROUGHT cycle {cycle}]")
        else:
            state['regen_rate'] = jnp.array(cfg['resource_regen'])

        drought_regen = cfg.get('drought_regen', 0.00002) if is_drought else None
        state, metrics, fitness_base, survival_steps = run_cycle_with_metrics(
            state, chunk_runner, cfg, regen_override=drought_regen
        )

        # === V34 KEY CHANGE: Φ-inclusive fitness ===
        # Measure population Φ at end of cycle
        cycle_phi = float(compute_phi_hidden(state['hidden'], state['alive']))

        # Create per-agent Φ proxy (population-level Φ for all alive)
        phi_per_agent = np.full(cfg['M_max'], cycle_phi) * np.array(state['alive'])

        # Recompute fitness with Φ bonus
        fitness_v34 = compute_fitness_with_phi(
            survival_steps, np.zeros_like(survival_steps),  # resources already in fitness_base
            cfg['metabolic_cost'], state['alive'], phi_per_agent, alpha
        )
        # Actually: use the base fitness but scale by phi bonus
        fitness_np = np.array(fitness_base) * (1.0 + alpha * phi_per_agent)

        # Measure robustness
        rob_metrics = measure_robustness(state, chunk_runner, cfg)

        # Standard metrics
        lr_stats = compute_lr_stats(state, cfg)
        drift_stats = compute_phenotype_drift(state, cfg)

        # Save snapshots
        if cycle % 5 == 0 or cycle == cfg['n_cycles'] - 1:
            snap = extract_snapshot(state, cycle, cfg)
            snap_path = os.path.join(output_dir, f'snapshot_c{cycle:02d}.npz')
            np.savez_compressed(snap_path, **snap)
            snapshots.append({'cycle': cycle, 'path': snap_path})

        # === Evolution with Φ-inclusive fitness ===
        key, k_sel, k_pos = jax.random.split(key, 3)
        state['genomes'] = tournament_selection(
            state['genomes'], jnp.array(fitness_np), state['alive'], k_sel, cfg
        )

        if cfg.get('activate_offspring', False):
            M = cfg['M_max']
            n_alive = int(jnp.sum(state['alive']))
            n_keep = max(1, min(n_alive, int(M * cfg['elite_fraction'])))
            fitness_masked = np.where(
                np.array(state['alive']), fitness_np, -np.inf
            )
            ranked = np.argsort(fitness_masked)[::-1]
            elite_set = set(ranked[:n_keep].tolist())

            alive_np = np.array(state['alive'])
            positions_np = np.array(state['positions'])
            energy_np = np.array(state['energy'])
            new_positions = np.array(
                jax.random.randint(k_pos, (M, 2), 0, cfg['N'])
            )
            for i in range(M):
                if i not in elite_set:
                    alive_np[i] = True
                    positions_np[i] = new_positions[i]
                    energy_np[i] = cfg['initial_energy']
            state['alive'] = jnp.array(alive_np)
            state['positions'] = jnp.array(positions_np)
            state['energy'] = jnp.array(energy_np)

        state, key = rescue_population_v27(state, key, cfg)

        elapsed = time.time() - t_start
        cycle_info = {
            'cycle': cycle,
            **metrics,
            **rob_metrics,
            'cycle_phi': cycle_phi,
            'phi_fitness_alpha': alpha,
            'lr_stats': lr_stats,
            'phenotype_drift': drift_stats,
            'elapsed_s': elapsed,
        }
        cycle_results.append(cycle_info)

        print(
            f"C{cycle:02d} | pop={metrics['n_alive_end']:3d} | "
            f"phi={metrics['mean_phi']:.3f} (cyc={cycle_phi:.3f}) | "
            f"rob={rob_metrics['robustness']:.3f} | "
            f"mse={metrics['mean_pred_mse']:.4f} | "
            f"{elapsed:.0f}s"
        )

        if cycle % 5 == 0 or cycle == cfg['n_cycles'] - 1:
            progress = {
                'seed': seed, 'cycle': cycle,
                'config': {k: v for k, v in cfg.items()
                           if isinstance(v, (int, float, str, bool))},
                'cycles': cycle_results,
                'snapshots': [{'cycle': s['cycle']} for s in snapshots],
            }
            with open(os.path.join(output_dir, f'v34_s{seed}_progress.json'), 'w') as f:
                json.dump(progress, f, indent=2)

    # Final summary
    mean_phi = float(np.mean([c['mean_phi'] for c in cycle_results]))
    max_phi = float(np.max([c['mean_phi'] for c in cycle_results]))
    late_phis = [c['mean_phi'] for c in cycle_results if c['cycle'] >= 15]
    late_mean_phi = float(np.mean(late_phis)) if late_phis else mean_phi

    if late_mean_phi > 0.10:
        category = 'HIGH'
    elif late_mean_phi > 0.07:
        category = 'MOD'
    else:
        category = 'LOW'

    # Goodhart check: correlation between cycle Phi and robustness
    cycle_phis = [c['cycle_phi'] for c in cycle_results]
    robs = [c['robustness'] for c in cycle_results]
    if len(cycle_phis) > 2:
        phi_rob_corr = float(np.corrcoef(cycle_phis, robs)[0, 1])
    else:
        phi_rob_corr = 0.0

    results = {
        'seed': seed,
        'config': {k: v for k, v in cfg.items()
                   if isinstance(v, (int, float, str, bool))},
        'cycles': cycle_results,
        'snapshots': [{'cycle': s['cycle']} for s in snapshots],
        'summary': {
            'mean_phi': mean_phi,
            'max_phi': max_phi,
            'late_mean_phi': late_mean_phi,
            'mean_robustness': float(np.mean(robs)),
            'max_robustness': float(np.max(robs)),
            'mean_pred_mse': float(np.mean([c['mean_pred_mse'] for c in cycle_results])),
            'final_lr': cycle_results[-1]['lr_stats']['mean_lr'],
            'final_pop': cycle_results[-1]['n_alive_end'],
            'phi_rob_correlation': phi_rob_corr,
            'category': category,
        },
    }

    with open(os.path.join(output_dir, f'v34_s{seed}_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSeed {seed} complete — {category}")
    print(f"  Mean Φ:          {mean_phi:.3f}")
    print(f"  Late mean Φ:     {late_mean_phi:.3f}")
    print(f"  Mean robustness: {results['summary']['mean_robustness']:.3f}")
    print(f"  Φ-Rob corr:      {phi_rob_corr:.3f} "
          f"({'OK' if phi_rob_corr > 0 else 'GOODHART WARNING'})")

    return results
