"""V21: Evolution Loop — CTM-Inspired Protocell Agency

Evolutionary loop for V21 neural agents with internal ticks and sync matrices.
Agents evolve via tournament selection on survival fitness. No gradient descent.

Key additions over V20 evolution:
  - Phi_sync tracking alongside Phi_hidden
  - Intra-step divergence tracking (from chunk runner)
  - Tick usage metrics per cycle (entropy, effective K, dominant tick)
  - Sync decay statistics per cycle
  - Sync matrices reset at cycle boundary (like hidden states)
"""

import jax
import jax.numpy as jnp
import numpy as np
import json
import os
import time

from v21_substrate import (
    generate_v21_config, init_v21, make_chunk_runner,
    compute_phi_hidden, compute_phi_sync, compute_robustness,
    extract_snapshot, extract_tick_weights_np, extract_sync_decay_np,
)

# Reuse V20 evolution machinery (fitness, selection, rescue)
from v20_evolution import (
    compute_fitness, tournament_selection, rescue_population,
)


# ---------------------------------------------------------------------------
# Tick usage metrics
# ---------------------------------------------------------------------------

def compute_tick_usage(state, cfg):
    """Compute tick usage metrics from current population's evolvable tick_weights.

    Returns dict with entropy, effective K, dominant tick distribution.
    """
    alive = np.array(state['alive'])
    alive_idx = np.where(alive)[0]

    if len(alive_idx) == 0:
        K = cfg['K_max']
        return {
            'mean_entropy': 0.0,
            'mean_effective_K': 1.0,
            'std_effective_K': 0.0,
            'dominant_tick_distribution': [0] * K,
            'tick_0_weight_mean': 1.0,
            'tick_0_collapsed': 1.0,
        }

    tw = extract_tick_weights_np(state['params'], cfg)  # (M, K)
    tw_alive = tw[alive_idx]  # (n_alive, K)

    # Entropy of tick weight distribution per agent
    entropy = -np.sum(tw_alive * np.log(tw_alive + 1e-8), axis=1)  # (n_alive,)
    effective_K = np.exp(entropy)  # (n_alive,)

    # Dominant tick per agent
    dominant = np.argmax(tw_alive, axis=1)  # (n_alive,)
    K = cfg['K_max']
    dominant_dist = [int(np.sum(dominant == k)) for k in range(K)]

    return {
        'mean_entropy': float(np.mean(entropy)),
        'mean_effective_K': float(np.mean(effective_K)),
        'std_effective_K': float(np.std(effective_K)),
        'dominant_tick_distribution': dominant_dist,
        'tick_0_weight_mean': float(np.mean(tw_alive[:, 0])),
        'tick_0_collapsed': float(np.mean(tw_alive[:, 0] > 0.9)),
    }


def compute_sync_decay_stats(state, cfg):
    """Compute sync decay statistics from current population."""
    alive = np.array(state['alive'])
    alive_idx = np.where(alive)[0]

    if len(alive_idx) == 0:
        return {'mean_sync_decay': 0.75, 'std_sync_decay': 0.0}

    sd = extract_sync_decay_np(state['params'], cfg)  # (M,)
    sd_alive = sd[alive_idx]

    return {
        'mean_sync_decay': float(np.mean(sd_alive)),
        'std_sync_decay': float(np.std(sd_alive)),
    }


# ---------------------------------------------------------------------------
# Cycle metrics
# ---------------------------------------------------------------------------

def run_cycle_with_metrics(state, chunk_runner, cfg, stress=False, regen_override=None):
    """Run one cycle (steps_per_cycle steps), collecting V21 metrics.

    Returns: (final_state, cycle_metrics dict, fitness array, survival_steps array)
    """
    n_chunks = cfg['steps_per_cycle'] // cfg['chunk_size']
    chunk_size = cfg['chunk_size']

    # Override regen rate
    if regen_override is not None:
        state = {**state, 'regen_rate': jnp.array(regen_override)}
    elif stress:
        state = {**state, 'regen_rate': jnp.array(cfg['stress_regen'])}
    else:
        state = {**state, 'regen_rate': jnp.array(cfg['resource_regen'])}

    n_alive_start = int(jnp.sum(state['alive']))

    # Track survival and resource consumption
    survival_steps = np.zeros(cfg['M_max'], dtype=np.float32)
    resources_gained = np.zeros(cfg['M_max'], dtype=np.float32)

    # Metric accumulators
    phi_samples = []
    phi_sync_samples = []
    n_alive_samples = []
    divergence_samples = []

    for chunk_i in range(n_chunks):
        prev_energy = np.array(state['energy'])
        prev_alive = np.array(state['alive'])

        state, chunk_metrics = chunk_runner(state)

        curr_alive = np.array(state['alive'])
        curr_energy = np.array(state['energy'])

        # Track survival time
        survival_steps += prev_alive.astype(np.float32) * chunk_size

        # Resource gain approximation
        energy_delta = curr_energy - prev_energy
        resources_gained += np.maximum(
            energy_delta + cfg['metabolic_cost'] * chunk_size, 0.0
        ) * prev_alive

        n_alive_samples.append(float(np.mean(chunk_metrics['n_alive'])))

        # Mean divergence from this chunk
        mean_div = float(np.mean(chunk_metrics['mean_divergence']))
        divergence_samples.append(mean_div)

        # Sample Phi and Phi_sync every 10 chunks
        if chunk_i % 10 == 0:
            phi = float(compute_phi_hidden(state['hidden'], state['alive']))
            phi_sync = float(compute_phi_sync(
                state['sync_matrices'], state['alive']
            ))
            phi_samples.append(phi)
            phi_sync_samples.append(phi_sync)

    n_alive_end = int(jnp.sum(state['alive']))
    mean_phi = float(np.mean(phi_samples)) if phi_samples else 0.0
    mean_phi_sync = float(np.mean(phi_sync_samples)) if phi_sync_samples else 0.0
    mean_alive = float(np.mean(n_alive_samples))
    mean_divergence = float(np.mean(divergence_samples)) if divergence_samples else 0.0

    # Mortality
    deaths = max(0, n_alive_start - n_alive_end)
    mortality = deaths / max(n_alive_start, 1)

    # Fitness
    fitness_np = compute_fitness(
        jnp.array(survival_steps),
        jnp.array(resources_gained),
        cfg['metabolic_cost'],
        state['alive'],
    )

    cycle_metrics = {
        'n_alive_start': n_alive_start,
        'n_alive_end': n_alive_end,
        'mean_alive': mean_alive,
        'mortality': mortality,
        'mean_phi': mean_phi,
        'mean_phi_sync': mean_phi_sync,
        'mean_divergence': mean_divergence,
        'fitness': np.array(fitness_np).tolist(),
    }

    return state, cycle_metrics, np.array(fitness_np), np.array(survival_steps)


# ---------------------------------------------------------------------------
# Robustness measurement
# ---------------------------------------------------------------------------

def measure_robustness(state, chunk_runner, cfg, n_chunks_base=10, n_chunks_stress=10):
    """Measure Phi and Phi_sync under normal and stress conditions."""
    # Base conditions
    state_base = {**state, 'regen_rate': jnp.array(cfg['resource_regen'])}
    phi_base_samples = []
    phi_sync_base_samples = []
    for _ in range(n_chunks_base):
        state_base, _ = chunk_runner(state_base)
        phi = float(compute_phi_hidden(state_base['hidden'], state_base['alive']))
        phi_sync = float(compute_phi_sync(
            state_base['sync_matrices'], state_base['alive']
        ))
        phi_base_samples.append(phi)
        phi_sync_base_samples.append(phi_sync)
    phi_base = float(np.mean(phi_base_samples))
    phi_sync_base = float(np.mean(phi_sync_base_samples))

    # Stress conditions (drought)
    state_stress = {**state, 'regen_rate': jnp.array(cfg['stress_regen'])}
    phi_stress_samples = []
    phi_sync_stress_samples = []
    for _ in range(n_chunks_stress):
        state_stress, _ = chunk_runner(state_stress)
        phi = float(compute_phi_hidden(state_stress['hidden'], state_stress['alive']))
        phi_sync = float(compute_phi_sync(
            state_stress['sync_matrices'], state_stress['alive']
        ))
        phi_stress_samples.append(phi)
        phi_sync_stress_samples.append(phi_sync)
    phi_stress = float(np.mean(phi_stress_samples))
    phi_sync_stress = float(np.mean(phi_sync_stress_samples))

    robustness = phi_stress / max(phi_base, 1e-6)

    return {
        'phi_base': phi_base,
        'phi_stress': phi_stress,
        'robustness': robustness,
        'phi_sync_base': phi_sync_base,
        'phi_sync_stress': phi_sync_stress,
        'n_alive_base': int(jnp.sum(state_base['alive'])),
        'n_alive_stress': int(jnp.sum(state_stress['alive'])),
    }


# ---------------------------------------------------------------------------
# Main evolution loop
# ---------------------------------------------------------------------------

def run_v21(seed, cfg, output_dir):
    """Run V21 evolution for one seed.

    Saves snapshots and cycle metrics to output_dir.
    Returns summary dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    key = jax.random.PRNGKey(seed)
    state, key = init_v21(seed, cfg)

    # Warmup: run 1 chunk to trigger JIT compilation
    print(f"Warming up JIT (seed {seed})...")
    t0 = time.time()
    chunk_runner = make_chunk_runner(cfg)
    state, _ = chunk_runner(state)
    print(f"  JIT warmup: {time.time()-t0:.1f}s")

    cycle_results = []
    snapshots = []

    for cycle in range(cfg['n_cycles']):
        t_start = time.time()

        # Drought schedule (from V20b)
        drought_every = cfg.get('drought_every', 0)
        is_drought = (drought_every > 0) and (cycle > 0) and (cycle % drought_every == 0)
        if is_drought:
            state['resources'] = state['resources'] * cfg.get('drought_depletion', 0.05)
            state['regen_rate'] = jnp.array(cfg.get('drought_regen', 0.00002))
            print(f"  [DROUGHT cycle {cycle}] Resources depleted to "
                  f"{cfg.get('drought_depletion', 0.05):.0%}")
        else:
            state['regen_rate'] = jnp.array(cfg['resource_regen'])

        # Run cycle
        drought_regen = cfg.get('drought_regen', 0.00002) if is_drought else None
        state, metrics, fitness, survival_steps = run_cycle_with_metrics(
            state, chunk_runner, cfg, regen_override=drought_regen
        )

        # Measure robustness
        rob_metrics = measure_robustness(state, chunk_runner, cfg,
                                          n_chunks_base=5, n_chunks_stress=5)

        # Tick usage and sync decay metrics
        tick_usage = compute_tick_usage(state, cfg)
        sync_stats = compute_sync_decay_stats(state, cfg)

        # Evolution: select and mutate
        key, k_sel, k_pos = jax.random.split(key, 3)
        state['params'] = tournament_selection(
            state['params'], jnp.array(fitness), state['alive'], k_sel, cfg
        )

        # Activate offspring (V20b fix)
        if cfg.get('activate_offspring', False):
            M = cfg['M_max']
            n_alive = int(jnp.sum(state['alive']))
            n_keep = max(1, min(n_alive, int(M * cfg['elite_fraction'])))
            fitness_masked = np.where(
                np.array(state['alive']), np.array(fitness), -np.inf
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

        # Rescue if needed
        state, key = rescue_population(state, key, cfg)

        # Reset hidden states, sync matrices, and energy for next cycle
        state['hidden'] = jnp.zeros_like(state['hidden'])
        state['sync_matrices'] = jnp.zeros_like(state['sync_matrices'])
        state['energy'] = jnp.where(
            state['alive'],
            cfg['initial_energy'],
            0.0
        )

        elapsed = time.time() - t_start
        cycle_info = {
            'cycle': cycle,
            **metrics,
            **rob_metrics,
            'tick_usage': tick_usage,
            'sync_decay': sync_stats,
            'elapsed_s': elapsed,
        }
        cycle_results.append(cycle_info)

        print(
            f"C{cycle:02d} | pop={metrics['n_alive_end']:3d} | "
            f"mort={metrics['mortality']:.0%} | "
            f"phi={metrics['mean_phi']:.3f} | "
            f"phi_sync={metrics['mean_phi_sync']:.3f} | "
            f"div={metrics['mean_divergence']:.2f} | "
            f"eff_K={tick_usage['mean_effective_K']:.1f} | "
            f"rob={rob_metrics['robustness']:.3f} | "
            f"{elapsed:.0f}s"
        )

        # Save snapshot every 5 cycles
        if cycle % 5 == 0 or cycle == cfg['n_cycles'] - 1:
            snap = extract_snapshot(state, cycle, cfg)
            snap_path = os.path.join(output_dir, f'snapshot_c{cycle:02d}.npz')
            np.savez_compressed(snap_path, **snap)
            snapshots.append({'cycle': cycle, 'path': snap_path})

        # Save progress JSON
        progress = {
            'seed': seed,
            'cycle': cycle,
            'config': {k: v for k, v in cfg.items()
                       if isinstance(v, (int, float, str, bool))},
            'cycles': cycle_results,
            'snapshots': snapshots,
        }
        with open(os.path.join(output_dir, f'v21_s{seed}_progress.json'), 'w') as f:
            json.dump(progress, f, indent=2)

    # Final results
    results = {
        'seed': seed,
        'config': {k: v for k, v in cfg.items()
                   if isinstance(v, (int, float, str, bool))},
        'cycles': cycle_results,
        'snapshots': snapshots,
        'summary': {
            'mean_robustness': float(np.mean([c['robustness'] for c in cycle_results])),
            'max_robustness': float(np.max([c['robustness'] for c in cycle_results])),
            'mean_phi': float(np.mean([c['mean_phi'] for c in cycle_results])),
            'mean_phi_sync': float(np.mean([c['mean_phi_sync'] for c in cycle_results])),
            'mean_divergence': float(np.mean([c['mean_divergence'] for c in cycle_results])),
            'final_effective_K': cycle_results[-1]['tick_usage']['mean_effective_K'],
            'tick_0_collapsed_final': cycle_results[-1]['tick_usage']['tick_0_collapsed'],
            'final_sync_decay': cycle_results[-1]['sync_decay']['mean_sync_decay'],
            'final_pop': cycle_results[-1]['n_alive_end'],
        },
    }

    with open(os.path.join(output_dir, f'v21_s{seed}_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSeed {seed} complete.")
    print(f"  Mean robustness:   {results['summary']['mean_robustness']:.3f}")
    print(f"  Max robustness:    {results['summary']['max_robustness']:.3f}")
    print(f"  Mean Phi:          {results['summary']['mean_phi']:.3f}")
    print(f"  Mean Phi_sync:     {results['summary']['mean_phi_sync']:.3f}")
    print(f"  Mean divergence:   {results['summary']['mean_divergence']:.3f}")
    print(f"  Final effective K: {results['summary']['final_effective_K']:.2f}")
    print(f"  Tick-0 collapsed:  {results['summary']['tick_0_collapsed_final']:.0%}")
    print(f"  Final sync decay:  {results['summary']['final_sync_decay']:.4f}")

    return results
