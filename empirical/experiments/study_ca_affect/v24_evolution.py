"""V24: Evolution Loop — TD Value Learning

Key differences from V22:
  - TD learning instead of next-step prediction
  - Tracks mean_value, td_error instead of pred_mse
  - Evolvable gamma (discount factor) tracked per cycle
"""

import jax
import jax.numpy as jnp
import numpy as np
import json
import os
import time

from v24_substrate import (
    generate_v24_config, init_v24, make_v24_chunk_runner,
    compute_phi_hidden, compute_phi_sync, compute_robustness,
    extract_snapshot, extract_tick_weights_np, extract_sync_decay_np,
    extract_lr_np, extract_gamma_np,
)

from v20_evolution import (
    compute_fitness, tournament_selection,
)


def rescue_population_v24(state, key, cfg, min_pop=10):
    alive = np.array(state['alive'])
    n_alive = int(alive.sum())
    if n_alive >= min_pop:
        return state, key

    print("  [RESCUE] %d agents — reseeding population" % n_alive)
    M = cfg['M_max']
    N = cfg['N']
    P = cfg['n_params']

    key, k1, k2, k3 = jax.random.split(key, 4)
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
        K = cfg['K_max']
        return {
            'mean_entropy': 0.0, 'mean_effective_K': 1.0,
            'std_effective_K': 0.0,
            'dominant_tick_distribution': [0] * K,
            'tick_0_weight_mean': 1.0, 'tick_0_collapsed': 1.0,
        }
    tw = extract_tick_weights_np(state['genomes'], cfg)
    tw_alive = tw[alive_idx]
    entropy = -np.sum(tw_alive * np.log(tw_alive + 1e-8), axis=1)
    effective_K = np.exp(entropy)
    dominant = np.argmax(tw_alive, axis=1)
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
    alive = np.array(state['alive'])
    alive_idx = np.where(alive)[0]
    if len(alive_idx) == 0:
        return {'mean_sync_decay': 0.75, 'std_sync_decay': 0.0}
    sd = extract_sync_decay_np(state['genomes'], cfg)
    sd_alive = sd[alive_idx]
    return {
        'mean_sync_decay': float(np.mean(sd_alive)),
        'std_sync_decay': float(np.std(sd_alive)),
    }


def compute_lr_stats(state, cfg):
    alive = np.array(state['alive'])
    alive_idx = np.where(alive)[0]
    if len(alive_idx) == 0:
        return {'mean_lr': 0.0, 'std_lr': 0.0, 'min_lr': 0.0, 'max_lr': 0.0}
    lr = extract_lr_np(state['genomes'], cfg)
    lr_alive = lr[alive_idx]
    return {
        'mean_lr': float(np.mean(lr_alive)),
        'std_lr': float(np.std(lr_alive)),
        'min_lr': float(np.min(lr_alive)),
        'max_lr': float(np.max(lr_alive)),
    }


def compute_gamma_stats(state, cfg):
    """Compute discount factor statistics from genomes."""
    alive = np.array(state['alive'])
    alive_idx = np.where(alive)[0]
    if len(alive_idx) == 0:
        return {'mean_gamma': 0.75, 'std_gamma': 0.0, 'min_gamma': 0.75, 'max_gamma': 0.75}
    gamma = extract_gamma_np(state['genomes'], cfg)
    gamma_alive = gamma[alive_idx]
    return {
        'mean_gamma': float(np.mean(gamma_alive)),
        'std_gamma': float(np.std(gamma_alive)),
        'min_gamma': float(np.min(gamma_alive)),
        'max_gamma': float(np.max(gamma_alive)),
    }


def compute_phenotype_drift(state, cfg):
    alive = np.array(state['alive'])
    alive_idx = np.where(alive)[0]
    if len(alive_idx) == 0:
        return {'mean_drift': 0.0, 'max_drift': 0.0, 'std_drift': 0.0}
    genomes = np.array(state['genomes'])[alive_idx]
    phenotypes = np.array(state['phenotypes'])[alive_idx]
    drift = np.sqrt(np.sum((phenotypes - genomes) ** 2, axis=1))
    return {
        'mean_drift': float(np.mean(drift)),
        'max_drift': float(np.max(drift)),
        'std_drift': float(np.std(drift)),
    }


def run_cycle_with_metrics(state, chunk_runner, cfg, stress=False, regen_override=None):
    n_chunks = cfg['steps_per_cycle'] // cfg['chunk_size']
    chunk_size = cfg['chunk_size']

    if regen_override is not None:
        state = {**state, 'regen_rate': jnp.array(regen_override)}
    elif stress:
        state = {**state, 'regen_rate': jnp.array(cfg['stress_regen'])}
    else:
        state = {**state, 'regen_rate': jnp.array(cfg['resource_regen'])}

    n_alive_start = int(jnp.sum(state['alive']))

    survival_steps = np.zeros(cfg['M_max'], dtype=np.float32)
    resources_gained = np.zeros(cfg['M_max'], dtype=np.float32)

    phi_samples = []
    phi_sync_samples = []
    n_alive_samples = []
    divergence_samples = []
    td_error_samples = []
    value_samples = []
    grad_norm_samples = []

    for chunk_i in range(n_chunks):
        prev_energy = np.array(state['energy'])
        prev_alive = np.array(state['alive'])

        state, chunk_metrics = chunk_runner(state)

        curr_alive = np.array(state['alive'])
        curr_energy = np.array(state['energy'])

        survival_steps += prev_alive.astype(np.float32) * chunk_size
        energy_delta = curr_energy - prev_energy
        resources_gained += np.maximum(
            energy_delta + cfg['metabolic_cost'] * chunk_size, 0.0
        ) * prev_alive

        n_alive_samples.append(float(np.mean(chunk_metrics['n_alive'])))
        divergence_samples.append(float(np.mean(chunk_metrics['mean_divergence'])))
        td_error_samples.append(float(np.mean(chunk_metrics['mean_td_error'])))
        value_samples.append(float(np.mean(chunk_metrics['mean_value'])))
        grad_norm_samples.append(float(np.mean(chunk_metrics['mean_grad_norm'])))

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
    mean_divergence = float(np.mean(divergence_samples))
    mean_td_error = float(np.mean(td_error_samples))
    mean_value = float(np.mean(value_samples))
    mean_grad_norm = float(np.mean(grad_norm_samples))

    # TD error trajectory (early vs late within cycle)
    n_half = len(td_error_samples) // 2
    td_error_early = float(np.mean(td_error_samples[:max(n_half, 1)]))
    td_error_late = float(np.mean(td_error_samples[max(n_half, 1):]))

    deaths = max(0, n_alive_start - n_alive_end)
    mortality = deaths / max(n_alive_start, 1)

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
        'mean_td_error': mean_td_error,
        'td_error_early': td_error_early,
        'td_error_late': td_error_late,
        'mean_value': mean_value,
        'mean_grad_norm': mean_grad_norm,
        'fitness': np.array(fitness_np).tolist(),
    }

    return state, cycle_metrics, np.array(fitness_np), np.array(survival_steps)


def measure_robustness(state, chunk_runner, cfg, n_chunks_base=5, n_chunks_stress=5):
    state_base = {**state, 'regen_rate': jnp.array(cfg['resource_regen'])}
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

    robustness = phi_stress / max(phi_base, 1e-6)
    return {
        'phi_base': phi_base, 'phi_stress': phi_stress,
        'robustness': robustness,
        'n_alive_base': int(jnp.sum(state_base['alive'])),
        'n_alive_stress': int(jnp.sum(state_stress['alive'])),
    }


def run_v24(seed, cfg, output_dir):
    """Run V24 evolution for one seed."""
    os.makedirs(output_dir, exist_ok=True)

    key = jax.random.PRNGKey(seed)
    state, key = init_v24(seed, cfg)

    print("Warming up JIT (seed %d)..." % seed)
    t0 = time.time()
    chunk_runner = make_v24_chunk_runner(cfg)
    state, _ = chunk_runner(state)
    print("  JIT warmup: %.1fs" % (time.time() - t0))

    cycle_results = []
    snapshots = []

    for cycle in range(cfg['n_cycles']):
        t_start = time.time()

        # Reset phenotype <- genome
        state['phenotypes'] = state['genomes'].copy()
        state['hidden'] = jnp.zeros_like(state['hidden'])
        state['sync_matrices'] = jnp.zeros_like(state['sync_matrices'])
        state['energy'] = jnp.where(state['alive'], cfg['initial_energy'], 0.0)

        # Drought schedule
        drought_every = cfg.get('drought_every', 0)
        is_drought = (drought_every > 0) and (cycle > 0) and (cycle % drought_every == 0)
        if is_drought:
            state['resources'] = state['resources'] * cfg.get('drought_depletion', 0.05)
            state['regen_rate'] = jnp.array(cfg.get('drought_regen', 0.00002))
            print("  [DROUGHT cycle %d] Resources depleted to %.0f%%" % (
                cycle, cfg.get('drought_depletion', 0.05) * 100))
        else:
            state['regen_rate'] = jnp.array(cfg['resource_regen'])

        drought_regen = cfg.get('drought_regen', 0.00002) if is_drought else None
        state, metrics, fitness, survival_steps = run_cycle_with_metrics(
            state, chunk_runner, cfg, regen_override=drought_regen
        )

        rob_metrics = measure_robustness(state, chunk_runner, cfg)
        tick_usage = compute_tick_usage(state, cfg)
        sync_stats = compute_sync_decay_stats(state, cfg)
        lr_stats = compute_lr_stats(state, cfg)
        gamma_stats = compute_gamma_stats(state, cfg)
        drift_stats = compute_phenotype_drift(state, cfg)

        if cfg.get('lamarckian', False):
            state['genomes'] = state['phenotypes'].copy()

        key, k_sel, k_pos = jax.random.split(key, 3)
        state['genomes'] = tournament_selection(
            state['genomes'], jnp.array(fitness), state['alive'], k_sel, cfg
        )

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

        state, key = rescue_population_v24(state, key, cfg)

        elapsed = time.time() - t_start
        cycle_info = {
            'cycle': cycle,
            **metrics,
            **rob_metrics,
            'tick_usage': tick_usage,
            'sync_decay': sync_stats,
            'lr_stats': lr_stats,
            'gamma_stats': gamma_stats,
            'phenotype_drift': drift_stats,
            'elapsed_s': elapsed,
        }
        cycle_results.append(cycle_info)

        print(
            "C%02d | pop=%3d | mort=%.0f%% | "
            "phi=%.3f | "
            "td=%.4f (%.4f->%.4f) | "
            "V=%.3f | "
            "lr=%.5f g=%.3f | "
            "drift=%.2f | "
            "rob=%.3f | %.0fs" % (
                cycle, metrics['n_alive_end'],
                metrics['mortality'] * 100,
                metrics['mean_phi'],
                metrics['mean_td_error'],
                metrics['td_error_early'],
                metrics['td_error_late'],
                metrics['mean_value'],
                lr_stats['mean_lr'],
                gamma_stats['mean_gamma'],
                drift_stats['mean_drift'],
                rob_metrics['robustness'],
                elapsed,
            )
        )

        if cycle % 5 == 0 or cycle == cfg['n_cycles'] - 1:
            snap = extract_snapshot(state, cycle, cfg)
            snap_path = os.path.join(output_dir, 'snapshot_c%02d.npz' % cycle)
            np.savez_compressed(snap_path, **snap)
            snapshots.append({'cycle': cycle, 'path': snap_path})

        progress = {
            'seed': seed, 'cycle': cycle,
            'config': {k: v for k, v in cfg.items()
                       if isinstance(v, (int, float, str, bool))},
            'cycles': cycle_results,
            'snapshots': snapshots,
        }
        with open(os.path.join(output_dir, 'v24_s%d_progress.json' % seed), 'w') as f:
            json.dump(progress, f, indent=2)

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
            'mean_td_error': float(np.mean([c['mean_td_error'] for c in cycle_results])),
            'mean_value': float(np.mean([c['mean_value'] for c in cycle_results])),
            'final_lr': cycle_results[-1]['lr_stats']['mean_lr'],
            'lr_suppressed': bool(cycle_results[-1]['lr_stats']['mean_lr'] < 1e-4),
            'final_gamma': cycle_results[-1]['gamma_stats']['mean_gamma'],
            'gamma_suppressed': bool(cycle_results[-1]['gamma_stats']['mean_gamma'] < 0.55),
            'final_drift': cycle_results[-1]['phenotype_drift']['mean_drift'],
            'final_effective_K': cycle_results[-1]['tick_usage']['mean_effective_K'],
            'tick_0_collapsed_final': cycle_results[-1]['tick_usage']['tick_0_collapsed'],
            'final_sync_decay': cycle_results[-1]['sync_decay']['mean_sync_decay'],
            'mean_divergence': float(np.mean([c['mean_divergence'] for c in cycle_results])),
            'final_pop': cycle_results[-1]['n_alive_end'],
        },
    }

    with open(os.path.join(output_dir, 'v24_s%d_results.json' % seed), 'w') as f:
        json.dump(results, f, indent=2)

    print("\nSeed %d complete." % seed)
    print("  Mean robustness:      %.3f" % results['summary']['mean_robustness'])
    print("  Max robustness:       %.3f" % results['summary']['max_robustness'])
    print("  Mean Phi:             %.3f" % results['summary']['mean_phi'])
    print("  Mean TD error:        %.4f" % results['summary']['mean_td_error'])
    print("  Mean value:           %.3f" % results['summary']['mean_value'])
    print("  Final LR:             %.6f" % results['summary']['final_lr'])
    print("  Final gamma:          %.3f" % results['summary']['final_gamma'])
    print("  Final drift:          %.2f" % results['summary']['final_drift'])
    print("  Final effective K:    %.2f" % results['summary']['final_effective_K'])

    return results
