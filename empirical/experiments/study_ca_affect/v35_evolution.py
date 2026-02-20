"""V35: Evolution Loop — Language Emergence under Cooperative POMDP

Extends V27 evolution with communication metrics tracking.
"""

import jax
import jax.numpy as jnp
import numpy as np
import json
import os
import time

from v35_substrate import (
    generate_v35_config, init_v35, make_v35_chunk_runner,
    extract_snapshot, extract_lr_np,
    compute_symbol_histograms,
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


def compute_communication_metrics(state, cfg):
    """Compute language emergence metrics from the current state.

    Returns dict with:
      - sym_entropy: Shannon entropy of symbol distribution (bits)
      - sym_distribution: histogram of symbol usage
      - sym_resource_mi_proxy: variance of mean resource per symbol
        (higher = more referential)
      - sym_unique: number of distinct symbols used
    """
    alive = np.array(state['alive'])
    alive_idx = np.where(alive)[0]
    if len(alive_idx) < 2:
        K = cfg['K_sym']
        return {
            'sym_entropy': 0.0,
            'sym_distribution': [0.0] * K,
            'sym_resource_mi_proxy': 0.0,
            'sym_unique': 0,
        }

    symbols = np.array(state['emitted_symbols'])
    K = cfg['K_sym']

    # Symbol distribution among alive agents
    sym_alive = symbols[alive_idx]
    counts = np.bincount(sym_alive, minlength=K).astype(np.float64)
    probs = counts / counts.sum()

    # Shannon entropy
    entropy = -np.sum(probs * np.log2(probs + 1e-10))

    # Unique symbols used
    n_unique = int(np.sum(counts > 0))

    # Resource MI proxy: variance of mean local resource per symbol
    resources = np.array(state['resources'])
    positions = np.array(state['positions'])
    local_res = resources[positions[alive_idx, 0], positions[alive_idx, 1]]

    mean_res_per_sym = np.zeros(K)
    for k in range(K):
        mask = sym_alive == k
        if mask.sum() > 0:
            mean_res_per_sym[k] = local_res[mask].mean()
    resource_var = float(np.var(mean_res_per_sym[counts > 0])) if n_unique > 1 else 0.0

    return {
        'sym_entropy': float(entropy),
        'sym_distribution': probs.tolist(),
        'sym_resource_mi_proxy': resource_var,
        'sym_unique': n_unique,
    }


def compute_symbol_movement_mi(state, prev_state, cfg):
    """Compute MI proxy between received symbols and movement direction.

    If agents respond to symbols by moving toward signalers,
    there should be correlation between received symbol and move direction.

    Returns: float, correlation between dominant received symbol and
    movement toward the source of that symbol.
    """
    alive = np.array(state['alive']) & np.array(prev_state['alive'])
    alive_idx = np.where(alive)[0]
    if len(alive_idx) < 4:
        return 0.0

    # Movement vectors
    pos_now = np.array(state['positions'])
    pos_before = np.array(prev_state['positions'])
    N = cfg['N']

    dx = pos_now[alive_idx, 0] - pos_before[alive_idx, 0]
    dy = pos_now[alive_idx, 1] - pos_before[alive_idx, 1]
    # Wrap around
    dx = np.where(dx > N // 2, dx - N, dx)
    dx = np.where(dx < -N // 2, dx + N, dx)
    dy = np.where(dy > N // 2, dy - N, dy)
    dy = np.where(dy < -N // 2, dy + N, dy)

    # Most common received symbol for each agent
    sym_hist = np.array(compute_symbol_histograms(
        jnp.array(prev_state['positions']),
        jnp.array(prev_state['emitted_symbols']),
        jnp.array(prev_state['alive']),
        cfg
    ))
    dominant_sym = np.argmax(sym_hist[alive_idx], axis=1)

    # For each symbol, compute mean movement vector
    K = cfg['K_sym']
    sym_move_var = 0.0
    counts = 0
    for k in range(K):
        mask = dominant_sym == k
        if mask.sum() > 2:
            mean_dx = np.mean(dx[mask])
            mean_dy = np.mean(dy[mask])
            sym_move_var += mean_dx ** 2 + mean_dy ** 2
            counts += 1

    return float(sym_move_var / max(counts, 1))


def run_cycle_with_metrics_v35(state, chunk_runner, cfg,
                                stress=False, regen_override=None):
    """Run one cycle, tracking communication metrics."""
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
    sym_entropy_samples = []
    sym_resource_var_samples = []
    n_coop_samples = []
    chunk_size = cfg['chunk_size']

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
        pred_mse_samples.append(float(np.mean(chunk_metrics['pred_mse'])))
        sym_entropy_samples.append(float(np.mean(chunk_metrics['sym_entropy'])))
        sym_resource_var_samples.append(float(np.mean(chunk_metrics['sym_resource_var'])))
        n_coop_samples.append(float(np.sum(chunk_metrics['n_coop'])))

        if chunk_i % 10 == 0:
            phi = float(compute_phi_hidden(state['hidden'], state['alive']))
            phi_samples.append(phi)

    n_alive_end = int(jnp.sum(state['alive']))
    mean_phi = float(np.mean(phi_samples)) if phi_samples else 0.0
    mean_pred_mse = float(np.mean(pred_mse_samples)) if pred_mse_samples else 0.0

    # Communication metrics
    comm_metrics = compute_communication_metrics(state, cfg)

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
        # Communication metrics
        'sym_entropy': float(np.mean(sym_entropy_samples)),
        'sym_entropy_end': comm_metrics['sym_entropy'],
        'sym_resource_mi_proxy': comm_metrics['sym_resource_mi_proxy'],
        'sym_unique': comm_metrics['sym_unique'],
        'sym_distribution': comm_metrics['sym_distribution'],
        'mean_coop_events': float(np.mean(n_coop_samples)),
    }

    return state, cycle_metrics, np.array(fitness_np), survival_steps


def measure_robustness_v35(state, chunk_runner, cfg,
                            n_chunks_base=5, n_chunks_stress=5):
    """Measure Phi under normal and stress."""
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
        'phi_base': phi_base,
        'phi_stress': phi_stress,
        'robustness': robustness,
    }


def measure_communication_ablation(state, chunk_runner, cfg, n_chunks=5):
    """Measure Φ with vs without communication channel.

    Ablation: set all symbol histograms to uniform, breaking
    the communication channel.

    This is a KEY test: if Φ drops when communication is ablated,
    integration DEPENDS on the social channel.
    """
    # Normal Φ (with communication)
    phi_with_comm = []
    state_test = {k: v for k, v in state.items()}
    state_test = {**state_test, 'regen_rate': jnp.array(cfg['resource_regen'])}
    for _ in range(n_chunks):
        state_test, _ = chunk_runner(state_test)
        phi = float(compute_phi_hidden(state_test['hidden'], state_test['alive']))
        phi_with_comm.append(phi)

    # Ablated Φ (no communication — zero out symbols)
    phi_no_comm = []
    state_ablated = {k: v for k, v in state.items()}
    state_ablated = {**state_ablated, 'regen_rate': jnp.array(cfg['resource_regen'])}
    # Set all emitted symbols to 0 (uniform-ish)
    state_ablated['emitted_symbols'] = jnp.zeros_like(state_ablated['emitted_symbols'])
    for _ in range(n_chunks):
        # Before each chunk, reset symbols to break communication
        state_ablated['emitted_symbols'] = jnp.zeros_like(state_ablated['emitted_symbols'])
        state_ablated, _ = chunk_runner(state_ablated)
        phi = float(compute_phi_hidden(state_ablated['hidden'], state_ablated['alive']))
        phi_no_comm.append(phi)

    return {
        'phi_with_comm': float(np.mean(phi_with_comm)),
        'phi_no_comm': float(np.mean(phi_no_comm)),
        'comm_phi_lift': float(np.mean(phi_with_comm) - np.mean(phi_no_comm)),
    }


# ---------------------------------------------------------------------------
# Main evolution loop
# ---------------------------------------------------------------------------

def run_v35(seed, cfg, output_dir):
    """Run V35 cooperative POMDP with discrete communication."""
    os.makedirs(output_dir, exist_ok=True)

    key = jax.random.PRNGKey(seed)
    state, key = init_v35(seed, cfg)

    print(f"Warming up JIT (seed {seed})...")
    t0 = time.time()
    chunk_runner = make_v35_chunk_runner(cfg)
    state, _ = chunk_runner(state)
    print(f"  JIT warmup: {time.time()-t0:.1f}s")

    cycle_results = []
    snapshots = []

    for cycle in range(cfg['n_cycles']):
        t_start = time.time()

        state['phenotypes'] = state['genomes'].copy()
        state['hidden'] = jnp.zeros_like(state['hidden'])
        state['sync_matrices'] = jnp.zeros_like(state['sync_matrices'])
        state['energy'] = jnp.where(state['alive'], cfg['initial_energy'], 0.0)
        state['emitted_symbols'] = jnp.zeros_like(state['emitted_symbols'])

        drought_every = cfg.get('drought_every', 0)
        is_drought = (drought_every > 0) and (cycle > 0) and (cycle % drought_every == 0)
        if is_drought:
            state['resources'] = state['resources'] * cfg.get('drought_depletion', 0.05)
            state['regen_rate'] = jnp.array(cfg.get('drought_regen', 0.00002))
            print(f"  [DROUGHT cycle {cycle}]")
        else:
            state['regen_rate'] = jnp.array(cfg['resource_regen'])

        drought_regen = cfg.get('drought_regen', 0.00002) if is_drought else None
        state, metrics, fitness, survival_steps = run_cycle_with_metrics_v35(
            state, chunk_runner, cfg, regen_override=drought_regen
        )

        # Robustness measurement
        rob_metrics = measure_robustness_v35(state, chunk_runner, cfg)

        # Communication ablation (every 10 cycles + last)
        if cycle % 10 == 0 or cycle == cfg['n_cycles'] - 1:
            comm_ablation = measure_communication_ablation(state, chunk_runner, cfg)
        else:
            comm_ablation = {'phi_with_comm': 0.0, 'phi_no_comm': 0.0, 'comm_phi_lift': 0.0}

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
                np.array(state['alive']), fitness, -np.inf
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
            **comm_ablation,
            'lr_stats': lr_stats,
            'phenotype_drift': drift_stats,
            'elapsed_s': elapsed,
        }
        cycle_results.append(cycle_info)

        # Print progress
        comm_lift_str = ''
        if cycle % 10 == 0 or cycle == cfg['n_cycles'] - 1:
            comm_lift_str = f" | lift={comm_ablation['comm_phi_lift']:.3f}"
        print(
            f"C{cycle:02d} | pop={metrics['n_alive_end']:3d} | "
            f"phi={metrics['mean_phi']:.3f} | "
            f"ent={metrics['sym_entropy']:.2f} ({metrics['sym_unique']}/{cfg['K_sym']}) | "
            f"mi_proxy={metrics['sym_resource_mi_proxy']:.4f} | "
            f"coop={metrics['mean_coop_events']:.0f} | "
            f"rob={rob_metrics['robustness']:.3f}"
            f"{comm_lift_str} | "
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
            with open(os.path.join(output_dir, f'v35_s{seed}_progress.json'), 'w') as f:
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

    # Communication summary
    late_entropies = [c['sym_entropy'] for c in cycle_results if c['cycle'] >= 15]
    late_mi_proxies = [c['sym_resource_mi_proxy'] for c in cycle_results if c['cycle'] >= 15]
    late_coop = [c['mean_coop_events'] for c in cycle_results if c['cycle'] >= 15]

    # Ablation results
    ablation_cycles = [c for c in cycle_results
                       if c.get('comm_phi_lift', 0.0) != 0.0]
    mean_comm_lift = float(np.mean([c['comm_phi_lift'] for c in ablation_cycles])) if ablation_cycles else 0.0

    # Language classification
    mean_late_entropy = float(np.mean(late_entropies)) if late_entropies else 0.0
    mean_late_mi = float(np.mean(late_mi_proxies)) if late_mi_proxies else 0.0

    if mean_late_entropy > 2.0 and mean_late_mi > 0.001:
        lang_status = 'REFERENTIAL'
    elif mean_late_entropy > 1.0:
        lang_status = 'DIVERSE_NOISE'
    else:
        lang_status = 'COLLAPSED'

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
            'final_lr': cycle_results[-1]['lr_stats']['mean_lr'],
            'final_pop': cycle_results[-1]['n_alive_end'],
            'category': category,
            # Communication summary
            'mean_sym_entropy': mean_late_entropy,
            'mean_sym_resource_mi': mean_late_mi,
            'mean_coop_events': float(np.mean(late_coop)) if late_coop else 0.0,
            'comm_phi_lift': mean_comm_lift,
            'lang_status': lang_status,
        },
    }

    with open(os.path.join(output_dir, f'v35_s{seed}_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSeed {seed} complete — {category} | Language: {lang_status}")
    print(f"  Mean Φ:           {mean_phi:.3f}")
    print(f"  Late mean Φ:      {late_mean_phi:.3f}")
    print(f"  Mean robustness:  {results['summary']['mean_robustness']:.3f}")
    print(f"  Symbol entropy:   {mean_late_entropy:.2f} bits")
    print(f"  Sym-resource MI:  {mean_late_mi:.4f}")
    print(f"  Comm Φ lift:      {mean_comm_lift:.3f}")

    return results
