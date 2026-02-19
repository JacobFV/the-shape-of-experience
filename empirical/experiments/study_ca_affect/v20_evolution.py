"""V20: Evolution Loop — Protocell Agency

Evolutionary loop for V20 neural agents. Agents evolve via tournament
selection on survival fitness. No gradient descent — selection pressure
alone shapes the population.

Fitness = survival_time × (resources_consumed / metabolic_cost)
        = how long alive × how efficiently you harvested

The world model, self-model, and affect geometry are NOT training objectives.
They are measured post-hoc to see if they emerge from survival selection.
This is the uncontaminated emergence test.
"""

import jax
import jax.numpy as jnp
import numpy as np
import json
import os
import time
from functools import partial

from v20_substrate import (
    generate_v20_config, init_v20, make_chunk_runner,
    compute_phi_hidden, compute_robustness, extract_snapshot,
    build_agent_count_grid, build_observation_batch, agent_step_batch,
    decode_actions, MOVE_DELTAS, regen_resources, diffuse_signals,
    apply_consumption, apply_emissions,
)


# ---------------------------------------------------------------------------
# Fitness computation
# ---------------------------------------------------------------------------

def compute_fitness(survival_time, resources_consumed, metabolic_cost, alive):
    """Fitness = survival efficiency × alive bonus.

    Dead agents get 0 fitness.
    """
    efficiency = resources_consumed / (metabolic_cost * jnp.maximum(survival_time, 1.0))
    fitness = survival_time * jnp.log1p(efficiency + 1e-3)
    return fitness * alive.astype(jnp.float32)


# ---------------------------------------------------------------------------
# Tournament selection
# ---------------------------------------------------------------------------

def tournament_selection(params, fitness, alive, key, cfg):
    """Select survivors by tournament and fill population with mutations.

    Returns new params array, same shape as input.
    """
    M = params.shape[0]
    T = cfg['tournament_size']

    # Rank by fitness (alive agents only; dead get -inf)
    fitness_masked = jnp.where(alive, fitness, -jnp.inf)
    ranked = jnp.argsort(fitness_masked)[::-1]  # descending

    n_alive = int(jnp.sum(alive))
    n_keep = max(1, min(n_alive, int(M * cfg['elite_fraction'])))

    # Elite indices (top performers)
    elite_idx = np.array(ranked[:n_keep])

    new_params = np.array(params)  # copy to numpy for manipulation

    # Fill non-elite slots with mutated copies of elites
    key_np = np.array(key)  # use as seed
    rng = np.random.RandomState(int(key_np[0]) % (2**31))

    for i in range(n_keep, M):
        parent_idx = elite_idx[rng.randint(0, n_keep)]
        noise = rng.randn(*params.shape[1:]).astype(np.float32) * cfg['mutation_std']
        new_params[i] = new_params[parent_idx] + noise

    return jnp.array(new_params)


# ---------------------------------------------------------------------------
# Population rescue (if too few survivors)
# ---------------------------------------------------------------------------

def rescue_population(state, key, cfg, min_pop=10):
    """If alive population too small, reseed from survivors with mutation."""
    alive = np.array(state['alive'])
    n_alive = int(alive.sum())

    if n_alive >= min_pop:
        return state, key

    print(f"  [RESCUE] {n_alive} agents — reseeding population")

    M = cfg['M_max']
    N = cfg['N']
    H = cfg['hidden_dim']
    P = cfg['n_params']

    key, k1, k2, k3 = jax.random.split(key, 4)
    rng = np.random.RandomState(int(np.array(k1)[0]) % (2**31))

    alive_idx = np.where(alive)[0]
    params = np.array(state['params'])
    positions = np.array(state['positions'])

    # Fill up to min_pop * 8 agents from survivors
    n_fill = min(M - n_alive, min_pop * 8)
    new_alive = alive.copy()
    new_params = params.copy()
    new_positions = positions.copy()
    new_energy = np.array(state['energy'])
    new_hidden = np.array(state['hidden'])

    dead_idx = np.where(~alive)[0]
    for i in range(min(n_fill, len(dead_idx))):
        slot = dead_idx[i]
        parent = alive_idx[rng.randint(0, len(alive_idx))]
        noise = rng.randn(P).astype(np.float32) * cfg['mutation_std'] * 3.0
        new_params[slot] = params[parent] + noise
        new_positions[slot] = rng.randint(0, N, size=2)
        new_energy[slot] = cfg['initial_energy']
        new_hidden[slot] = 0.0
        new_alive[slot] = True

    new_state = dict(state)
    new_state['params'] = jnp.array(new_params)
    new_state['positions'] = jnp.array(new_positions)
    new_state['energy'] = jnp.array(new_energy)
    new_state['hidden'] = jnp.array(new_hidden)
    new_state['alive'] = jnp.array(new_alive)
    return new_state, key


# ---------------------------------------------------------------------------
# Cycle metrics
# ---------------------------------------------------------------------------

def run_cycle_with_metrics(state, chunk_runner, cfg, stress=False):
    """Run one cycle (steps_per_cycle steps), collecting metrics.

    Returns: (final_state, cycle_metrics dict)
    """
    n_chunks = cfg['steps_per_cycle'] // cfg['chunk_size']
    chunk_size = cfg['chunk_size']

    # Override regen rate if stress
    if stress:
        state = {**state, 'regen_rate': jnp.array(cfg['stress_regen'])}
    else:
        state = {**state, 'regen_rate': jnp.array(cfg['resource_regen'])}

    n_alive_start = int(jnp.sum(state['alive']))
    initial_energy = np.array(state['energy']) * np.array(state['alive'])
    total_initial_energy = float(initial_energy.sum())

    # Track survival and resource consumption
    survival_steps = np.zeros(cfg['M_max'], dtype=np.float32)
    resources_gained = np.zeros(cfg['M_max'], dtype=np.float32)

    # Run chunks
    phi_samples = []
    n_alive_samples = []

    for chunk_i in range(n_chunks):
        prev_energy = np.array(state['energy'])
        prev_alive = np.array(state['alive'])

        state, chunk_metrics = chunk_runner(state)

        curr_alive = np.array(state['alive'])
        curr_energy = np.array(state['energy'])

        # Track survival time (each step an alive agent is alive)
        survival_steps += prev_alive.astype(np.float32) * chunk_size

        # Resource gain approximation from energy delta + metabolic cost
        energy_delta = curr_energy - prev_energy
        resources_gained += np.maximum(
            energy_delta + cfg['metabolic_cost'] * chunk_size, 0.0
        ) * prev_alive

        n_alive_samples.append(float(np.mean(chunk_metrics['n_alive'])))

        # Sample Phi every 10 chunks
        if chunk_i % 10 == 0:
            phi = float(compute_phi_hidden(state['hidden'], state['alive']))
            phi_samples.append(phi)

    n_alive_end = int(jnp.sum(state['alive']))
    mean_phi = float(np.mean(phi_samples)) if phi_samples else 0.0
    mean_alive = float(np.mean(n_alive_samples))

    # Mortality
    deaths = max(0, n_alive_start - n_alive_end)
    mortality = deaths / max(n_alive_start, 1)

    # Compute fitness
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
        'fitness': np.array(fitness_np).tolist(),
    }

    return state, cycle_metrics, np.array(fitness_np), np.array(survival_steps)


# ---------------------------------------------------------------------------
# Robustness measurement (base + stress Phi)
# ---------------------------------------------------------------------------

def measure_robustness(state, chunk_runner, cfg, n_chunks_base=10, n_chunks_stress=10):
    """Measure Phi under normal and stress conditions.

    Returns: dict with phi_base, phi_stress, robustness per agent
    """
    # Base conditions
    state_base = {**state, 'regen_rate': jnp.array(cfg['resource_regen'])}
    phi_base_samples = []
    for _ in range(n_chunks_base):
        state_base, _ = chunk_runner(state_base)
        phi = float(compute_phi_hidden(state_base['hidden'], state_base['alive']))
        phi_base_samples.append(phi)
    phi_base = float(np.mean(phi_base_samples))

    # Stress conditions (drought)
    state_stress = {**state, 'regen_rate': jnp.array(cfg['stress_regen'])}
    phi_stress_samples = []
    for _ in range(n_chunks_stress):
        state_stress, _ = chunk_runner(state_stress)
        phi = float(compute_phi_hidden(state_stress['hidden'], state_stress['alive']))
        phi_stress_samples.append(phi)
    phi_stress = float(np.mean(phi_stress_samples))

    robustness = phi_stress / max(phi_base, 1e-6)

    return {
        'phi_base': phi_base,
        'phi_stress': phi_stress,
        'robustness': robustness,
        'n_alive_base': int(jnp.sum(state_base['alive'])),
        'n_alive_stress': int(jnp.sum(state_stress['alive'])),
    }


# ---------------------------------------------------------------------------
# Main evolution loop
# ---------------------------------------------------------------------------

def run_v20(seed, cfg, output_dir):
    """Run V20 evolution for one seed.

    Saves snapshots and cycle metrics to output_dir.
    Returns summary dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    key = jax.random.PRNGKey(seed)
    state, key = init_v20(seed, cfg)

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

        # Run cycle
        state, metrics, fitness, survival_steps = run_cycle_with_metrics(
            state, chunk_runner, cfg
        )

        # Measure robustness
        rob_metrics = measure_robustness(state, chunk_runner, cfg,
                                          n_chunks_base=5, n_chunks_stress=5)

        # Evolution: select and mutate
        key, k_sel = jax.random.split(key)
        state['params'] = tournament_selection(
            state['params'], jnp.array(fitness), state['alive'], k_sel, cfg
        )

        # Rescue if needed
        state, key = rescue_population(state, key, cfg)

        # Reset hidden states and energy for next cycle
        state['hidden'] = jnp.zeros_like(state['hidden'])
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
            'elapsed_s': elapsed,
        }
        cycle_results.append(cycle_info)

        print(
            f"C{cycle:02d} | pop={metrics['n_alive_end']:3d} | "
            f"mort={metrics['mortality']:.0%} | "
            f"phi={metrics['mean_phi']:.3f} | "
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
        with open(os.path.join(output_dir, f'v20_s{seed}_progress.json'), 'w') as f:
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
            'final_pop': cycle_results[-1]['n_alive_end'],
        },
    }

    with open(os.path.join(output_dir, f'v20_s{seed}_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSeed {seed} complete.")
    print(f"  Mean robustness: {results['summary']['mean_robustness']:.3f}")
    print(f"  Max robustness:  {results['summary']['max_robustness']:.3f}")
    print(f"  Mean Phi:        {results['summary']['mean_phi']:.3f}")

    return results
