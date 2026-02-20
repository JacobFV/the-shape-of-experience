"""V29: Evolution Loop — Social Prediction

Fork of V27 evolution with V29 substrate (social prediction target).
Same evolution loop, different gradient signal.
"""

import jax
import jax.numpy as jnp
import numpy as np
import json
import os
import time

from v29_substrate import (
    generate_v29_config, init_v29, make_v29_chunk_runner,
    compute_phi_hidden, compute_phi_sync, compute_robustness,
    extract_snapshot, extract_tick_weights_np, extract_sync_decay_np,
    extract_lr_np,
)

from v20_evolution import (
    compute_fitness, tournament_selection,
)


def rescue_population(state, key, cfg, min_pop=10):
    alive = np.array(state['alive'])
    n_alive = int(alive.sum())
    if n_alive >= min_pop:
        return state, key

    print(f"  [RESCUE] {n_alive} agents — reseeding")
    M = cfg['M_max']
    N = cfg['N']
    P = cfg['n_params']

    key, k1 = jax.random.split(key)
    rng = np.random.RandomState(int(np.array(k1)[0]) % (2**31))

    alive_idx = np.where(alive)[0]
    genomes = np.array(state['genomes'])
    positions = np.array(state['positions'])

    n_fill = min(M - n_alive, min_pop * 8)
    new_alive = alive.copy()
    new_genomes = genomes.copy()
    new_positions = positions.copy()
    new_energy = np.array(state['energy'])
    new_hidden = np.array(state['hidden'])

    dead_idx = np.where(~alive)[0]
    for i in range(min(n_fill, len(dead_idx))):
        slot = dead_idx[i]
        parent = alive_idx[rng.randint(0, len(alive_idx))]
        noise = rng.randn(P).astype(np.float32) * cfg['mutation_std'] * 3.0
        new_genomes[slot] = genomes[parent] + noise
        new_positions[slot] = rng.randint(0, N, size=2)
        new_energy[slot] = cfg['initial_energy']
        new_hidden[slot] = 0.0
        new_alive[slot] = True

    new_state = dict(state)
    new_state['genomes'] = jnp.array(new_genomes)
    new_state['phenotypes'] = jnp.array(new_genomes)
    new_state['positions'] = jnp.array(new_positions)
    new_state['energy'] = jnp.array(new_energy)
    new_state['hidden'] = jnp.array(new_hidden)
    new_state['alive'] = jnp.array(new_alive)
    return new_state, key


def compute_tick_usage(state, cfg):
    alive = np.array(state['alive'])
    alive_idx = np.where(alive)[0]
    if len(alive_idx) == 0:
        return {'mean_entropy': 0.0, 'mean_effective_K': 1.0}
    tw = extract_tick_weights_np(state['genomes'], cfg)
    tw_alive = tw[alive_idx]
    entropy = -np.sum(tw_alive * np.log(tw_alive + 1e-8), axis=1)
    effective_K = np.exp(entropy)
    return {
        'mean_entropy': float(np.mean(entropy)),
        'mean_effective_K': float(np.mean(effective_K)),
    }


def compute_lr_stats(state, cfg):
    alive = np.array(state['alive'])
    alive_idx = np.where(alive)[0]
    if len(alive_idx) == 0:
        return {'mean_lr': 0.0}
    lr = extract_lr_np(state['genomes'], cfg)
    return {'mean_lr': float(np.mean(lr[alive_idx]))}


def compute_phenotype_drift(state, cfg):
    alive = np.array(state['alive'])
    alive_idx = np.where(alive)[0]
    if len(alive_idx) == 0:
        return {'mean_drift': 0.0}
    genomes = np.array(state['genomes'])[alive_idx]
    phenotypes = np.array(state['phenotypes'])[alive_idx]
    drift = np.sqrt(np.sum((phenotypes - genomes) ** 2, axis=1))
    return {'mean_drift': float(np.mean(drift))}


def run_cycle_with_metrics(state, chunk_runner, cfg, regen_override=None):
    n_chunks = cfg['steps_per_cycle'] // cfg['chunk_size']
    chunk_size = cfg['chunk_size']

    if regen_override is not None:
        state = {**state, 'regen_rate': jnp.array(regen_override)}
    else:
        state = {**state, 'regen_rate': jnp.array(cfg['resource_regen'])}

    n_alive_start = int(jnp.sum(state['alive']))
    survival_steps = np.zeros(cfg['M_max'], dtype=np.float32)
    resources_gained = np.zeros(cfg['M_max'], dtype=np.float32)

    phi_samples = []
    n_alive_samples = []
    pred_mse_samples = []

    for chunk_i in range(n_chunks):
        prev_energy = np.array(state['energy'])
        prev_alive = np.array(state['alive'])

        state, chunk_metrics = chunk_runner(state)

        curr_energy = np.array(state['energy'])
        survival_steps += prev_alive.astype(np.float32) * chunk_size
        energy_delta = curr_energy - prev_energy
        resources_gained += np.maximum(
            energy_delta + cfg['metabolic_cost'] * chunk_size, 0.0
        ) * prev_alive

        n_alive_samples.append(float(np.mean(chunk_metrics['n_alive'])))
        pred_mse_samples.append(float(np.mean(chunk_metrics['pred_mse'])))

        if chunk_i % 10 == 0:
            phi = float(compute_phi_hidden(state['hidden'], state['alive']))
            phi_samples.append(phi)

    n_alive_end = int(jnp.sum(state['alive']))
    mean_phi = float(np.mean(phi_samples)) if phi_samples else 0.0
    mean_pred_mse = float(np.mean(pred_mse_samples))

    n_half = len(pred_mse_samples) // 2
    pred_mse_early = float(np.mean(pred_mse_samples[:max(n_half, 1)]))
    pred_mse_late = float(np.mean(pred_mse_samples[max(n_half, 1):]))

    deaths = max(0, n_alive_start - n_alive_end)
    mortality = deaths / max(n_alive_start, 1)

    fitness_np = compute_fitness(
        jnp.array(survival_steps),
        jnp.array(resources_gained),
        cfg['metabolic_cost'],
        state['alive'],
    )

    return state, {
        'n_alive_start': n_alive_start,
        'n_alive_end': n_alive_end,
        'mortality': mortality,
        'mean_phi': mean_phi,
        'mean_pred_mse': mean_pred_mse,
        'pred_mse_early': pred_mse_early,
        'pred_mse_late': pred_mse_late,
    }, np.array(fitness_np), np.array(survival_steps)


def measure_robustness(state, chunk_runner, cfg, n_chunks=5):
    state_base = {**state, 'regen_rate': jnp.array(cfg['resource_regen'])}
    phi_base = []
    for _ in range(n_chunks):
        state_base, _ = chunk_runner(state_base)
        phi_base.append(float(compute_phi_hidden(state_base['hidden'], state_base['alive'])))

    state_stress = {**state, 'regen_rate': jnp.array(cfg['stress_regen'])}
    phi_stress = []
    for _ in range(n_chunks):
        state_stress, _ = chunk_runner(state_stress)
        phi_stress.append(float(compute_phi_hidden(state_stress['hidden'], state_stress['alive'])))

    pb = float(np.mean(phi_base))
    ps = float(np.mean(phi_stress))
    return {
        'phi_base': pb,
        'phi_stress': ps,
        'robustness': ps / max(pb, 1e-6),
    }


# ---------------------------------------------------------------------------
# Main evolution loop
# ---------------------------------------------------------------------------

def run_v29(seed, cfg, output_dir):
    """Run V29 social prediction evolution."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"V29 SOCIAL PREDICTION — seed={seed}")
    print(f"  n_params={cfg['n_params']}")
    print(f"{'='*60}")

    key = jax.random.PRNGKey(seed)
    state, key = init_v29(seed, cfg)

    print("Warming up JIT...")
    t0 = time.time()
    chunk_runner = make_v29_chunk_runner(cfg)
    state, _ = chunk_runner(state)
    print(f"  JIT warmup: {time.time()-t0:.1f}s")

    cycle_results = []
    snapshots = []

    for cycle in range(cfg['n_cycles']):
        t_start = time.time()

        # Reset
        state['phenotypes'] = state['genomes'].copy()
        state['hidden'] = jnp.zeros_like(state['hidden'])
        state['sync_matrices'] = jnp.zeros_like(state['sync_matrices'])
        state['energy'] = jnp.where(state['alive'], cfg['initial_energy'], 0.0)

        # Drought
        drought_every = cfg.get('drought_every', 0)
        is_drought = (drought_every > 0) and (cycle > 0) and (cycle % drought_every == 0)
        if is_drought:
            state['resources'] = state['resources'] * cfg.get('drought_depletion', 0.01)
            state['regen_rate'] = jnp.array(cfg.get('drought_regen', 0.0))
            print(f"  [DROUGHT cycle {cycle}]")
        else:
            state['regen_rate'] = jnp.array(cfg['resource_regen'])

        drought_regen = cfg.get('drought_regen', 0.0) if is_drought else None
        state, metrics, fitness, survival_steps = run_cycle_with_metrics(
            state, chunk_runner, cfg, regen_override=drought_regen
        )

        rob_metrics = measure_robustness(state, chunk_runner, cfg)
        tick_usage = compute_tick_usage(state, cfg)
        lr_stats = compute_lr_stats(state, cfg)
        drift_stats = compute_phenotype_drift(state, cfg)

        # Snapshot BEFORE evolution
        if cycle % 5 == 0 or cycle == cfg['n_cycles'] - 1:
            snap = extract_snapshot(state, cycle, cfg)
            snap_path = os.path.join(output_dir, f'snapshot_c{cycle:02d}.npz')
            np.savez_compressed(snap_path, **snap)
            snapshots.append({'cycle': cycle, 'path': snap_path})

        # Evolution
        key, k_sel, k_pos = jax.random.split(key, 3)
        state['genomes'] = tournament_selection(
            state['genomes'], jnp.array(fitness), state['alive'], k_sel, cfg
        )

        # Activate offspring
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

        state, key = rescue_population(state, key, cfg)

        elapsed = time.time() - t_start
        cycle_info = {
            'cycle': cycle,
            **metrics,
            **rob_metrics,
            **tick_usage,
            **lr_stats,
            **drift_stats,
            'elapsed_s': elapsed,
        }
        cycle_results.append(cycle_info)

        print(
            f"C{cycle:02d} | pop={metrics['n_alive_end']:3d} | "
            f"phi={metrics['mean_phi']:.3f} | "
            f"mse={metrics['mean_pred_mse']:.4f} "
            f"({metrics['pred_mse_early']:.4f}→{metrics['pred_mse_late']:.4f}) | "
            f"rob={rob_metrics['robustness']:.3f} | "
            f"{elapsed:.0f}s"
        )

        # Save progress
        progress = {
            'seed': seed,
            'experiment': 'v29_social_prediction',
            'cycle': cycle,
            'config': {k: v for k, v in cfg.items()
                       if isinstance(v, (int, float, str, bool))},
            'cycles': cycle_results,
            'snapshots': snapshots,
        }
        with open(os.path.join(output_dir, f'v29_s{seed}_progress.json'), 'w') as f:
            json.dump(progress, f, indent=2)

    # Summary
    results = {
        'seed': seed,
        'experiment': 'v29_social_prediction',
        'config': {k: v for k, v in cfg.items()
                   if isinstance(v, (int, float, str, bool))},
        'cycles': cycle_results,
        'snapshots': snapshots,
        'summary': {
            'mean_phi': float(np.mean([c['mean_phi'] for c in cycle_results])),
            'max_phi': float(np.max([c['mean_phi'] for c in cycle_results])),
            'mean_robustness': float(np.mean([c['robustness'] for c in cycle_results])),
            'max_robustness': float(np.max([c['robustness'] for c in cycle_results])),
            'mean_pred_mse': float(np.mean([c['mean_pred_mse'] for c in cycle_results])),
            'final_pop': cycle_results[-1]['n_alive_end'],
        },
    }

    with open(os.path.join(output_dir, f'v29_s{seed}_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  [V29 s{seed}] Done. Mean Φ={results['summary']['mean_phi']:.3f}, "
          f"Max Φ={results['summary']['max_phi']:.3f}, "
          f"Rob={results['summary']['mean_robustness']:.3f}")

    return results
