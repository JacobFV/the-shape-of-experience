"""V26: Evolution Loop — POMDP with Partial Observability

Same structure as V25 evolution, but uses V26 substrate (1×1 observation,
H=32, noisy compass). The evolution loop, fitness functions, selection,
and rescue are identical.
"""

import jax
import jax.numpy as jnp
import numpy as np
import json
import os
import time

from v26_substrate import (
    generate_v26_config, init_v26, make_chunk_runner,
    compute_phi_hidden, compute_robustness, extract_snapshot,
)


# ---------------------------------------------------------------------------
# Fitness computation (same as V25)
# ---------------------------------------------------------------------------

def compute_prey_fitness(survival_time, resources_consumed, metabolic_cost, alive):
    """Prey fitness: survive long + harvest efficiently."""
    efficiency = resources_consumed / (metabolic_cost * jnp.maximum(survival_time, 1.0))
    fitness = survival_time * jnp.log1p(efficiency + 1e-3)
    return fitness * alive.astype(jnp.float32)


def compute_pred_fitness(survival_time, alive):
    """Predator fitness: pure survival."""
    return survival_time * alive.astype(jnp.float32)


# ---------------------------------------------------------------------------
# Tournament selection (per-type, same as V25)
# ---------------------------------------------------------------------------

def tournament_selection_typed(params, fitness, alive, type_mask, key, cfg):
    """Tournament selection within a type."""
    M = params.shape[0]
    P = params.shape[1]

    type_alive = type_mask & alive
    type_fitness = jnp.where(type_alive, fitness, -jnp.inf)

    type_indices = np.where(np.array(type_mask))[0]
    if len(type_indices) == 0:
        return params

    type_fit_np = np.array(type_fitness)[type_indices]
    ranked_local = np.argsort(type_fit_np)[::-1]
    ranked_global = type_indices[ranked_local]

    n_alive_type = int(np.sum(np.array(type_alive)[type_indices]))
    n_keep = max(1, min(n_alive_type, int(len(type_indices) * cfg['elite_fraction'])))

    elite_idx = ranked_global[:n_keep]

    new_params = np.array(params)
    rng = np.random.RandomState(int(np.array(key)[0]) % (2**31))

    non_elite = ranked_global[n_keep:]
    for slot in non_elite:
        parent = elite_idx[rng.randint(0, n_keep)]
        noise = rng.randn(P).astype(np.float32) * cfg['mutation_std']
        new_params[slot] = new_params[parent] + noise

    return jnp.array(new_params), elite_idx


# ---------------------------------------------------------------------------
# Cycle metrics
# ---------------------------------------------------------------------------

def run_cycle_with_metrics(state, chunk_runner, cfg, regen_override=None):
    """Run one cycle, collecting metrics."""
    n_chunks = cfg['steps_per_cycle'] // cfg['chunk_size']
    chunk_size = cfg['chunk_size']

    if regen_override is not None:
        state = {**state, 'regen_rate': jnp.array(regen_override)}
    else:
        state = {**state, 'regen_rate': jnp.array(cfg['patch_regen'])}

    n_alive_start = int(jnp.sum(state['alive']))
    is_pred = np.array(state['is_predator'])
    n_prey_start = int(np.sum(np.array(state['alive']) & ~is_pred))
    n_pred_start = int(np.sum(np.array(state['alive']) & is_pred))

    M = cfg['M_max']
    survival_steps = np.zeros(M, dtype=np.float32)
    resources_gained = np.zeros(M, dtype=np.float32)

    phi_samples = []
    n_alive_samples = []
    n_prey_samples = []
    n_pred_samples = []

    for chunk_i in range(n_chunks):
        prev_energy = np.array(state['energy'])
        prev_alive = np.array(state['alive'])

        state, chunk_metrics = chunk_runner(state)

        curr_alive = np.array(state['alive'])
        curr_energy = np.array(state['energy'])

        survival_steps += prev_alive.astype(np.float32) * chunk_size

        energy_delta = curr_energy - prev_energy
        resources_gained += np.maximum(
            energy_delta + cfg['prey_metabolic'] * chunk_size, 0.0
        ) * prev_alive * (~is_pred)

        n_alive_samples.append(float(np.mean(chunk_metrics['n_alive'])))
        n_prey_samples.append(float(np.mean(chunk_metrics['n_prey_alive'])))
        n_pred_samples.append(float(np.mean(chunk_metrics['n_pred_alive'])))

        if chunk_i % 10 == 0:
            phi = float(compute_phi_hidden(state['hidden'], state['alive']))
            phi_samples.append(phi)

    n_alive_end = int(jnp.sum(state['alive']))
    n_prey_end = int(np.sum(np.array(state['alive']) & ~is_pred))
    n_pred_end = int(np.sum(np.array(state['alive']) & is_pred))

    metrics = {
        'n_alive_start': n_alive_start,
        'n_alive_end': n_alive_end,
        'n_prey_start': n_prey_start,
        'n_prey_end': n_prey_end,
        'n_pred_start': n_pred_start,
        'n_pred_end': n_pred_end,
        'mean_alive': float(np.mean(n_alive_samples)),
        'mean_prey': float(np.mean(n_prey_samples)),
        'mean_pred': float(np.mean(n_pred_samples)),
        'mortality': max(0, n_alive_start - n_alive_end) / max(n_alive_start, 1),
        'mean_phi': float(np.mean(phi_samples)) if phi_samples else 0.0,
    }

    prey_fitness = compute_prey_fitness(
        jnp.array(survival_steps),
        jnp.array(resources_gained),
        cfg['prey_metabolic'],
        state['alive'],
    )
    pred_fitness = compute_pred_fitness(
        jnp.array(survival_steps),
        state['alive'],
    )
    fitness = jnp.where(state['is_predator'], pred_fitness, prey_fitness)

    return state, metrics, np.array(fitness), np.array(survival_steps)


# ---------------------------------------------------------------------------
# Robustness measurement
# ---------------------------------------------------------------------------

def measure_robustness(state, chunk_runner, cfg, n_chunks_base=5, n_chunks_stress=5):
    """Measure Phi under normal vs stress."""
    state_base = {**state, 'regen_rate': jnp.array(cfg['patch_regen'])}
    phi_base_samples = []
    for _ in range(n_chunks_base):
        state_base, _ = chunk_runner(state_base)
        phi = float(compute_phi_hidden(state_base['hidden'], state_base['alive']))
        phi_base_samples.append(phi)
    phi_base = float(np.mean(phi_base_samples))

    state_stress = {**state, 'regen_rate': jnp.array(cfg['stress_regen'])}
    phi_stress_samples = []
    for _ in range(n_chunks_stress):
        state_stress, _ = chunk_runner(state_stress)
        phi = float(compute_phi_hidden(state_stress['hidden'], state_stress['alive']))
        phi_stress_samples.append(phi)
    phi_stress = float(np.mean(phi_stress_samples))

    return {
        'phi_base': phi_base,
        'phi_stress': phi_stress,
        'robustness': phi_stress / max(phi_base, 1e-6),
        'n_alive_base': int(jnp.sum(state_base['alive'])),
        'n_alive_stress': int(jnp.sum(state_stress['alive'])),
    }


# ---------------------------------------------------------------------------
# Population rescue
# ---------------------------------------------------------------------------

def rescue_population(state, key, cfg, patch_centers, min_pop=10):
    """If alive population too small, reseed from survivors."""
    alive = np.array(state['alive'])
    is_pred = np.array(state['is_predator'])
    N = cfg['N']
    P = cfg['n_params']

    for type_name, type_mask in [('prey', ~is_pred), ('predators', is_pred)]:
        type_alive = alive & type_mask
        n_alive_type = int(type_alive.sum())
        type_indices = np.where(type_mask)[0]

        min_type = max(3, min_pop // 2)
        if n_alive_type >= min_type:
            continue

        print(f"  [RESCUE {type_name}] {n_alive_type} agents — reseeding")
        key, k1 = jax.random.split(key)
        rng = np.random.RandomState(int(np.array(k1)[0]) % (2**31))

        alive_type_idx = np.where(type_alive)[0]
        dead_type_idx = np.where(type_mask & ~alive)[0]

        params = np.array(state['params'])
        positions = np.array(state['positions'])
        energy = np.array(state['energy'])
        hidden = np.array(state['hidden'])

        n_fill = min(len(dead_type_idx), min_type * 4)
        for i in range(n_fill):
            slot = dead_type_idx[i]
            if len(alive_type_idx) > 0:
                parent = alive_type_idx[rng.randint(0, len(alive_type_idx))]
                noise = rng.randn(P).astype(np.float32) * cfg['mutation_std'] * 3.0
                params[slot] = params[parent] + noise
            if len(patch_centers) > 0:
                pc = patch_centers[rng.randint(0, len(patch_centers))]
                offset = rng.randint(-10, 10, size=2)
                positions[slot] = (pc + offset) % N
            else:
                positions[slot] = rng.randint(0, N, size=2)
            energy[slot] = cfg['initial_energy']
            hidden[slot] = 0.0
            alive[slot] = True

        state['params'] = jnp.array(params)
        state['positions'] = jnp.array(positions)
        state['energy'] = jnp.array(energy)
        state['hidden'] = jnp.array(hidden)
        state['alive'] = jnp.array(alive)

    return state, key


# ---------------------------------------------------------------------------
# Main evolution loop
# ---------------------------------------------------------------------------

def run_v26(seed, cfg, output_dir):
    """Run V26 evolution for one seed."""
    os.makedirs(output_dir, exist_ok=True)

    key = jax.random.PRNGKey(seed)
    state, key, patch_centers = init_v26(seed, cfg)

    # Store patch centers separately for rescue/offspring placement
    patch_centers_np = np.array(patch_centers)

    print(f"V26 POMDP — Seed {seed}")
    print(f"  Obs: 1×1 + compass = {cfg['obs_flat']} dims (vs V25: 102)")
    print(f"  Hidden: {cfg['hidden_dim']} (vs V25: 16)")
    print(f"  Params: {cfg['n_params']}")
    print(f"  Compass noise: σ={cfg['compass_noise_std']}")

    print(f"Warming up JIT (seed {seed})...")
    t0 = time.time()
    chunk_runner = make_chunk_runner(cfg)
    state, _ = chunk_runner(state)
    print(f"  JIT warmup: {time.time()-t0:.1f}s")

    cycle_results = []
    snapshots = []

    n_prey = cfg['n_prey']
    is_pred = np.array(state['is_predator'])
    prey_mask = ~is_pred
    pred_mask = is_pred

    for cycle in range(cfg['n_cycles']):
        t_start = time.time()

        # Drought schedule
        drought_every = cfg.get('drought_every', 0)
        is_drought = (drought_every > 0) and (cycle > 0) and (cycle % drought_every == 0)
        if is_drought:
            state['resources'] = state['resources'] * cfg.get('drought_depletion', 0.01)
            print(f"  [DROUGHT cycle {cycle}] Resources depleted to "
                  f"{cfg.get('drought_depletion', 0.01):.0%}")

        drought_regen = cfg.get('drought_regen', 0.0) if is_drought else None
        state, metrics, fitness, survival_steps = run_cycle_with_metrics(
            state, chunk_runner, cfg, regen_override=drought_regen
        )

        # Robustness
        rob_metrics = measure_robustness(state, chunk_runner, cfg)

        # Separate evolution for prey and predators
        key, k_prey, k_pred, k_pos = jax.random.split(key, 4)

        prey_mask_jnp = jnp.array(prey_mask)
        pred_mask_jnp = jnp.array(pred_mask)

        state['params'], prey_elite = tournament_selection_typed(
            state['params'], jnp.array(fitness), state['alive'],
            prey_mask_jnp, k_prey, cfg
        )
        state['params'], pred_elite = tournament_selection_typed(
            state['params'], jnp.array(fitness), state['alive'],
            pred_mask_jnp, k_pred, cfg
        )

        # Activate offspring
        if cfg.get('activate_offspring', False):
            M = cfg['M_max']
            N = cfg['N']
            pc_np = patch_centers_np

            alive_np = np.array(state['alive'])
            positions_np = np.array(state['positions'])
            energy_np = np.array(state['energy'])

            elite_set = set(prey_elite.tolist()) | set(pred_elite.tolist())

            rng = np.random.RandomState(int(np.array(k_pos)[0]) % (2**31))

            for i in range(M):
                if i not in elite_set:
                    alive_np[i] = True
                    pc = pc_np[rng.randint(0, len(pc_np))]
                    offset = rng.randint(-cfg['patch_radius']//2, cfg['patch_radius']//2, size=2)
                    positions_np[i] = (pc + offset) % N
                    energy_np[i] = cfg['initial_energy']

            state['alive'] = jnp.array(alive_np)
            state['positions'] = jnp.array(positions_np)
            state['energy'] = jnp.array(energy_np)

        # Rescue
        state, key = rescue_population(state, key, cfg, patch_centers_np)

        elapsed = time.time() - t_start

        cycle_info = {
            'cycle': cycle,
            **metrics,
            **rob_metrics,
            'elapsed_s': elapsed,
        }
        cycle_results.append(cycle_info)

        print(
            f"C{cycle:02d} | prey={metrics['n_prey_end']:3d} "
            f"pred={metrics['n_pred_end']:3d} | "
            f"mort={metrics['mortality']:.0%} | "
            f"phi={metrics['mean_phi']:.3f} | "
            f"rob={rob_metrics['robustness']:.3f} | "
            f"{elapsed:.0f}s"
        )

        # Save snapshot BEFORE resetting hidden states (!)
        if cycle % 5 == 0 or cycle == cfg['n_cycles'] - 1:
            snap = extract_snapshot(state, cycle, cfg)
            snap_path = os.path.join(output_dir, f'snapshot_c{cycle:02d}.npz')
            np.savez_compressed(snap_path, **snap)
            snapshots.append({'cycle': cycle, 'path': snap_path})

        # Reset hidden states and energy for next cycle
        state['hidden'] = jnp.zeros_like(state['hidden'])
        state['energy'] = jnp.where(state['alive'], cfg['initial_energy'], 0.0)

        # Save progress
        progress = {
            'seed': seed,
            'cycle': cycle,
            'config': {k: v for k, v in cfg.items()
                       if isinstance(v, (int, float, str, bool))},
            'cycles': cycle_results,
            'snapshots': snapshots,
        }
        fname = os.path.join(output_dir, f'v26_s{seed}_progress.json')
        with open(fname, 'w') as f:
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
            'final_prey': cycle_results[-1]['n_prey_end'],
            'final_pred': cycle_results[-1]['n_pred_end'],
        },
    }

    with open(os.path.join(output_dir, f'v26_s{seed}_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSeed {seed} complete.")
    print(f"  Mean robustness: {results['summary']['mean_robustness']:.3f}")
    print(f"  Max robustness:  {results['summary']['max_robustness']:.3f}")
    print(f"  Mean Phi:        {results['summary']['mean_phi']:.3f}")
    print(f"  Final prey:      {results['summary']['final_prey']}")
    print(f"  Final pred:      {results['summary']['final_pred']}")

    return results
