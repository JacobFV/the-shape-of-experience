"""V33: Evolution Loop — Contrastive Self-Prediction

Minimal fork of V27 evolution adapted for V33's contrastive chunk runner.
Key difference: chunk_runner takes (state, rng) instead of just state,
because contrastive learning needs random alternative actions each step.
"""

import jax
import jax.numpy as jnp
import numpy as np
import json
import os
import time

from v33_substrate import (
    generate_v33_config, init_v33, make_v33_chunk_runner,
    extract_snapshot, extract_lr_np,
)
from v20_substrate import compute_phi_hidden, compute_robustness
from v21_substrate import compute_phi_sync
from v20_evolution import compute_fitness, tournament_selection
from v27_evolution import (
    rescue_population_v27,
    compute_tick_usage,
    compute_sync_decay_stats,
    compute_phenotype_drift,
)


def compute_lr_stats(state, cfg):
    alive = np.array(state['alive'])
    alive_idx = np.where(alive)[0]
    if len(alive_idx) == 0:
        return {'mean_lr': 0.0, 'std_lr': 0.0}
    lr = extract_lr_np(state['genomes'], cfg)
    lr_alive = lr[alive_idx]
    return {
        'mean_lr': float(np.mean(lr_alive)),
        'std_lr': float(np.std(lr_alive)),
    }


def run_cycle_with_metrics_v33(state, chunk_runner, cfg, rng,
                                stress=False, regen_override=None):
    """Run one cycle with V33's contrastive chunk runner."""
    n_chunks = cfg['steps_per_cycle'] // cfg['chunk_size']

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
    n_alive_samples = []
    pred_mse_samples = []
    action_diff_samples = []
    grad_norm_samples = []
    chunk_size = cfg['chunk_size']

    for chunk_i in range(n_chunks):
        prev_energy = np.array(state['energy'])
        prev_alive = np.array(state['alive'])

        rng, rng_chunk = jax.random.split(rng)
        state, rng_out, chunk_metrics = chunk_runner(state, rng_chunk)

        curr_alive = np.array(state['alive'])
        curr_energy = np.array(state['energy'])

        survival_steps += prev_alive.astype(np.float32) * chunk_size
        energy_delta = curr_energy - prev_energy
        resources_gained += np.maximum(
            energy_delta + cfg['metabolic_cost'] * chunk_size, 0.0
        ) * prev_alive

        n_alive_samples.append(float(np.mean(chunk_metrics['n_alive'])))
        pred_mse_samples.append(float(np.mean(chunk_metrics['pred_mse'])))
        action_diff_samples.append(float(np.mean(chunk_metrics['action_diff'])))
        grad_norm_samples.append(float(np.mean(chunk_metrics['mean_grad_norm'])))

        if chunk_i % 10 == 0:
            phi = float(compute_phi_hidden(state['hidden'], state['alive']))
            phi_samples.append(phi)

    n_alive_end = int(jnp.sum(state['alive']))
    mean_phi = float(np.mean(phi_samples)) if phi_samples else 0.0
    mean_pred_mse = float(np.mean(pred_mse_samples)) if pred_mse_samples else 0.0
    mean_action_diff = float(np.mean(action_diff_samples)) if action_diff_samples else 0.0

    n_half = len(pred_mse_samples) // 2
    pred_mse_early = float(np.mean(pred_mse_samples[:max(n_half, 1)]))
    pred_mse_late = float(np.mean(pred_mse_samples[max(n_half, 1):]))

    deaths = max(0, n_alive_start - n_alive_end)
    mortality = deaths / max(n_alive_start, 1)

    fitness_np = compute_fitness(
        jnp.array(survival_steps), jnp.array(resources_gained),
        cfg['metabolic_cost'], state['alive'],
    )

    cycle_metrics = {
        'n_alive_start': n_alive_start,
        'n_alive_end': n_alive_end,
        'mortality': mortality,
        'mean_phi': mean_phi,
        'mean_pred_mse': mean_pred_mse,
        'pred_mse_early': pred_mse_early,
        'pred_mse_late': pred_mse_late,
        'mean_action_diff': mean_action_diff,
        'mean_grad_norm': float(np.mean(grad_norm_samples)),
    }

    return state, cycle_metrics, np.array(fitness_np), rng


def measure_robustness_v33(state, chunk_runner, cfg, rng,
                            n_chunks_base=5, n_chunks_stress=5):
    """Measure Phi under normal and stress — V33 version (needs rng)."""
    state_base = {**state, 'regen_rate': jnp.array(cfg['resource_regen'])}
    phi_base_samples = []
    for _ in range(n_chunks_base):
        rng, rk = jax.random.split(rng)
        state_base, _, _ = chunk_runner(state_base, rk)
        phi = float(compute_phi_hidden(state_base['hidden'], state_base['alive']))
        phi_base_samples.append(phi)
    phi_base = float(np.mean(phi_base_samples))

    state_stress = {**state, 'regen_rate': jnp.array(cfg['stress_regen'])}
    phi_stress_samples = []
    for _ in range(n_chunks_stress):
        rng, rk = jax.random.split(rng)
        state_stress, _, _ = chunk_runner(state_stress, rk)
        phi = float(compute_phi_hidden(state_stress['hidden'], state_stress['alive']))
        phi_stress_samples.append(phi)
    phi_stress = float(np.mean(phi_stress_samples))

    robustness = phi_stress / max(phi_base, 1e-6)
    return {
        'phi_base': phi_base,
        'phi_stress': phi_stress,
        'robustness': robustness,
    }, rng


# ---------------------------------------------------------------------------
# Main evolution loop
# ---------------------------------------------------------------------------

def run_v33(seed, cfg, output_dir):
    """Run V33 contrastive self-prediction evolution."""
    os.makedirs(output_dir, exist_ok=True)

    key = jax.random.PRNGKey(seed)
    state, key = init_v33(seed, cfg)

    print(f"Warming up JIT (seed {seed})...")
    t0 = time.time()
    chunk_runner = make_v33_chunk_runner(cfg)
    key, rk = jax.random.split(key)
    state, _, _ = chunk_runner(state, rk)
    print(f"  JIT warmup: {time.time()-t0:.1f}s")

    cycle_results = []
    snapshots = []

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
        state, metrics, fitness, key = run_cycle_with_metrics_v33(
            state, chunk_runner, cfg, key, regen_override=drought_regen
        )

        rob_metrics, key = measure_robustness_v33(state, chunk_runner, cfg, key)

        lr_stats = compute_lr_stats(state, cfg)
        drift_stats = compute_phenotype_drift(state, cfg)

        # Save snapshots
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

        state, key = rescue_population_v27(state, key, cfg)

        elapsed = time.time() - t_start
        cycle_info = {
            'cycle': cycle,
            **metrics,
            **rob_metrics,
            'lr_stats': lr_stats,
            'phenotype_drift': drift_stats,
            'elapsed_s': elapsed,
        }
        cycle_results.append(cycle_info)

        print(
            f"C{cycle:02d} | pop={metrics['n_alive_end']:3d} | "
            f"phi={metrics['mean_phi']:.3f} | "
            f"mse={metrics['mean_pred_mse']:.4f} "
            f"({metrics['pred_mse_early']:.4f}→{metrics['pred_mse_late']:.4f}) | "
            f"act_diff={metrics['mean_action_diff']:.4f} | "
            f"rob={rob_metrics['robustness']:.3f} | "
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
            with open(os.path.join(output_dir, f'v33_s{seed}_progress.json'), 'w') as f:
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
            'mean_robustness': float(np.mean([c['robustness'] for c in cycle_results])),
            'max_robustness': float(np.max([c['robustness'] for c in cycle_results])),
            'mean_pred_mse': float(np.mean([c['mean_pred_mse'] for c in cycle_results])),
            'mean_action_diff': float(np.mean([c['mean_action_diff'] for c in cycle_results])),
            'final_lr': cycle_results[-1]['lr_stats']['mean_lr'],
            'final_pop': cycle_results[-1]['n_alive_end'],
            'category': category,
        },
    }

    with open(os.path.join(output_dir, f'v33_s{seed}_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSeed {seed} complete — {category}")
    print(f"  Mean Φ:          {mean_phi:.3f}")
    print(f"  Late mean Φ:     {late_mean_phi:.3f}")
    print(f"  Mean robustness: {results['summary']['mean_robustness']:.3f}")
    print(f"  Mean action diff: {results['summary']['mean_action_diff']:.4f}")

    return results
