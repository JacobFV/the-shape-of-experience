"""V32: Drought Autopsy at Scale — What Determines the 30/70 Split?

Fork of V27 evolution with enhanced tracking. V31 showed:
  - 30% HIGH / 30% MOD / 40% LOW integration seeds
  - Post-drought bounce r=0.997 predicts final Φ
  - But WHAT makes some seeds bounce and others not?

V32 runs 50 seeds with fine-grained drought tracking:
  - Per-cycle: eff_rank, weight diversity, hidden entropy
  - Per-drought: pre/post Φ, bounce ratio, survival rate, survivor genome stats
  - Per-drought: gradient magnitude, hidden state convergence during drought
  - Snapshot at EVERY drought cycle (not just every 5th)

Substrate is IDENTICAL to V27 (2-layer MLP, self-prediction, GRU+ticks).
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
    extract_snapshot, extract_tick_weights_np, extract_sync_decay_np,
    extract_lr_np, extract_predict_mlp_np,
)

from v20_evolution import (
    compute_fitness, tournament_selection,
)

from v27_evolution import (
    rescue_population_v27,
    compute_tick_usage,
    compute_sync_decay_stats,
    compute_lr_stats,
    compute_phenotype_drift,
    run_cycle_with_metrics,
    measure_robustness,
)


# ---------------------------------------------------------------------------
# Enhanced metrics for drought autopsy
# ---------------------------------------------------------------------------

def compute_eff_rank(hidden, alive):
    """Effective rank of alive agents' hidden states.

    Higher eff_rank = more dimensions actively used.
    V22-V24 showed 5-7 typically; V27 seed 7 reached 11.3.
    """
    alive_np = np.array(alive)
    alive_idx = np.where(alive_np)[0]
    if len(alive_idx) < 3:
        return 1.0
    H = np.array(hidden)[alive_idx]
    H_centered = H - H.mean(axis=0)
    cov = np.cov(H_centered, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0)
    total = eigvals.sum()
    if total < 1e-10:
        return 1.0
    eigvals = eigvals / total
    # Shannon entropy of normalized eigenvalues
    ent = -np.sum(eigvals * np.log(eigvals + 1e-10))
    return float(np.exp(ent))


def compute_hidden_entropy(hidden, alive):
    """Shannon entropy of hidden state activation distribution.

    Measures how uniformly the hidden state space is being used
    across the population.
    """
    alive_np = np.array(alive)
    alive_idx = np.where(alive_np)[0]
    if len(alive_idx) < 2:
        return 0.0
    H = np.array(hidden)[alive_idx]
    # Bin each hidden dimension into 10 bins, compute joint entropy
    # Simplified: compute mean activation entropy per dimension
    entropies = []
    for d in range(H.shape[1]):
        vals = H[:, d]
        hist, _ = np.histogram(vals, bins=10)
        hist = hist / (hist.sum() + 1e-10)
        ent = -np.sum(hist * np.log(hist + 1e-10))
        entropies.append(ent)
    return float(np.mean(entropies))


def compute_weight_diversity(genomes, alive):
    """Pairwise genome diversity of alive agents.

    1 - mean cosine similarity. High diversity = diverse strategies.
    Uses only alive agents and filters zero-norm vectors.
    """
    alive_np = np.array(alive)
    alive_idx = np.where(alive_np)[0]
    if len(alive_idx) < 2:
        return 0.0
    G = np.array(genomes)[alive_idx].astype(np.float64)
    norms = np.linalg.norm(G, axis=1, keepdims=True)
    # Filter out zero-norm agents
    valid = (norms.squeeze() > 1e-6)
    if valid.sum() < 2:
        return 0.0
    G = G[valid]
    norms = norms[valid]
    G_norm = G / norms
    cos_sim = G_norm @ G_norm.T
    n = len(G)
    off_diag_sum = cos_sim.sum() - np.trace(cos_sim)
    mean_sim = off_diag_sum / (n * (n - 1) + 1e-8)
    return float(np.clip(1.0 - mean_sim, 0.0, 2.0))  # diversity = 1 - similarity


def compute_hidden_convergence(hidden, alive):
    """Mean pairwise cosine similarity of hidden states.

    High convergence during drought = agents coordinating internally.
    """
    alive_np = np.array(alive)
    alive_idx = np.where(alive_np)[0]
    if len(alive_idx) < 2:
        return 0.0
    H = np.array(hidden)[alive_idx]
    norms = np.linalg.norm(H, axis=1, keepdims=True) + 1e-8
    H_norm = H / norms
    cos_sim = H_norm @ H_norm.T
    n = len(alive_idx)
    off_diag_sum = cos_sim.sum() - np.trace(cos_sim)
    mean_sim = off_diag_sum / (n * (n - 1) + 1e-8)
    return float(mean_sim)


def compute_survivor_stats(state_pre, state_post, cfg):
    """Compare survivors vs victims at a drought event.

    Returns stats on what distinguishes agents that survive drought
    from those that don't.
    """
    alive_pre = np.array(state_pre['alive'])
    alive_post = np.array(state_post['alive'])

    pre_idx = np.where(alive_pre)[0]
    if len(pre_idx) == 0:
        return {'n_pre': 0, 'n_post': 0, 'survival_rate': 0.0}

    # Who survived?
    survived = alive_pre & alive_post
    died = alive_pre & ~alive_post
    surv_idx = np.where(survived)[0]
    died_idx = np.where(died)[0]

    n_pre = len(pre_idx)
    n_surv = len(surv_idx)
    n_died = len(died_idx)

    stats = {
        'n_pre': int(n_pre),
        'n_post': int(np.sum(alive_post)),
        'n_survived': int(n_surv),
        'n_died': int(n_died),
        'survival_rate': float(n_surv / max(n_pre, 1)),
    }

    genomes = np.array(state_pre['genomes'])

    if n_surv > 0 and n_died > 0:
        # Compare genome norms
        surv_norms = np.linalg.norm(genomes[surv_idx], axis=1)
        died_norms = np.linalg.norm(genomes[died_idx], axis=1)
        stats['surv_genome_norm_mean'] = float(np.mean(surv_norms))
        stats['died_genome_norm_mean'] = float(np.mean(died_norms))

        # Compare learning rates
        lr = extract_lr_np(genomes, cfg)
        stats['surv_lr_mean'] = float(np.mean(lr[surv_idx]))
        stats['died_lr_mean'] = float(np.mean(lr[died_idx]))

        # Compare energy at drought start
        energy = np.array(state_pre['energy'])
        stats['surv_energy_mean'] = float(np.mean(energy[surv_idx]))
        stats['died_energy_mean'] = float(np.mean(energy[died_idx]))

        # Compare hidden state norms
        hidden = np.array(state_pre['hidden'])
        surv_h_norm = np.linalg.norm(hidden[surv_idx], axis=1)
        died_h_norm = np.linalg.norm(hidden[died_idx], axis=1)
        stats['surv_hidden_norm_mean'] = float(np.mean(surv_h_norm))
        stats['died_hidden_norm_mean'] = float(np.mean(died_h_norm))

        # MLP weight analysis for survivors vs victims
        try:
            W1, b1, W2, b2 = extract_predict_mlp_np(genomes, cfg)
            surv_W1_norm = np.mean(np.linalg.norm(
                W1[surv_idx].reshape(n_surv, -1), axis=1))
            died_W1_norm = np.mean(np.linalg.norm(
                W1[died_idx].reshape(n_died, -1), axis=1))
            stats['surv_mlp_W1_norm'] = float(surv_W1_norm)
            stats['died_mlp_W1_norm'] = float(died_W1_norm)
        except Exception:
            pass

    return stats


# ---------------------------------------------------------------------------
# Main evolution loop with drought tracking
# ---------------------------------------------------------------------------

def run_v32(seed, cfg, output_dir):
    """V32 evolution: V27 substrate + enhanced drought tracking.

    Saves:
      - Per-cycle: all V27 metrics + eff_rank, weight_diversity, hidden_entropy
      - Per-drought: pre/post snapshots, bounce ratio, survivor stats
      - Snapshots at every drought cycle + every 5th cycle
    """
    os.makedirs(output_dir, exist_ok=True)

    key = jax.random.PRNGKey(seed)
    state, key = init_v27(seed, cfg)

    # Warmup JIT
    print(f"Warming up JIT (seed {seed})...")
    t0 = time.time()
    chunk_runner = make_v27_chunk_runner(cfg)
    state, _ = chunk_runner(state)
    print(f"  JIT warmup: {time.time()-t0:.1f}s")

    cycle_results = []
    snapshots = []
    drought_events = []
    prev_cycle_end_phi = 0.0  # Track end-of-cycle Φ for bounce calculation

    drought_every = cfg.get('drought_every', 5)

    for cycle in range(cfg['n_cycles']):
        t_start = time.time()

        # === RESET PHENOTYPE <- GENOME at cycle start ===
        state['phenotypes'] = state['genomes'].copy()

        # Reset hidden states, sync matrices, energy
        state['hidden'] = jnp.zeros_like(state['hidden'])
        state['sync_matrices'] = jnp.zeros_like(state['sync_matrices'])
        state['energy'] = jnp.where(
            state['alive'],
            cfg['initial_energy'],
            0.0
        )

        # Drought schedule
        is_drought = (drought_every > 0) and (cycle > 0) and (cycle % drought_every == 0)

        # === PRE-DROUGHT: use PREVIOUS cycle's end-of-cycle Φ ===
        if is_drought:
            pre_drought_state = {
                'alive': np.array(state['alive']),
                'genomes': np.array(state['genomes']),
                'positions': np.array(state['positions']),
                'energy': np.array(state['energy']),
                # Hidden is zeroed at this point, so use prev cycle's measurement
                'hidden': np.zeros_like(np.array(state['hidden'])),
            }
            # Use the Φ from end of previous cycle (before reset)
            pre_drought_phi = prev_cycle_end_phi
            pre_drought_eff_rank = 0.0  # Can't measure — hidden is zeroed
            pre_drought_n_alive = int(jnp.sum(state['alive']))

            state['resources'] = state['resources'] * cfg.get('drought_depletion', 0.05)
            state['regen_rate'] = jnp.array(cfg.get('drought_regen', 0.00002))
            print(f"  [DROUGHT cycle {cycle}] Resources depleted, "
                  f"pop={pre_drought_n_alive}")
        else:
            state['regen_rate'] = jnp.array(cfg['resource_regen'])

        # Run cycle with gradient learning
        drought_regen = cfg.get('drought_regen', 0.00002) if is_drought else None
        state, metrics, fitness, survival_steps = run_cycle_with_metrics(
            state, chunk_runner, cfg, regen_override=drought_regen
        )

        # === POST-DROUGHT ANALYSIS ===
        if is_drought:
            post_drought_phi = float(compute_phi_hidden(state['hidden'], state['alive']))
            post_drought_eff_rank = compute_eff_rank(state['hidden'], state['alive'])
            post_drought_n_alive = int(jnp.sum(state['alive']))
            hidden_convergence = compute_hidden_convergence(
                state['hidden'], state['alive'])

            # Bounce ratio: post/pre Phi
            bounce = post_drought_phi / max(pre_drought_phi, 1e-6)

            # Survivor analysis: compare pre-drought alive vs post-drought alive
            # Note: pre_drought_state.hidden is zeroed (reset happened);
            # we compare based on who was alive before vs after the drought cycle
            post_drought_state = {
                'alive': np.array(state['alive']),
                'hidden': np.array(state['hidden']),
                'genomes': np.array(state['genomes']),
                'energy': np.array(state['energy']),
            }
            survivor_stats = compute_survivor_stats(
                pre_drought_state, post_drought_state, cfg)

            drought_info = {
                'cycle': cycle,
                'pre_phi': pre_drought_phi,
                'post_phi': post_drought_phi,
                'bounce': bounce,
                'pre_eff_rank': pre_drought_eff_rank,
                'post_eff_rank': post_drought_eff_rank,
                'pre_n_alive': pre_drought_n_alive,
                'post_n_alive': post_drought_n_alive,
                'hidden_convergence': hidden_convergence,
                'survivor_stats': survivor_stats,
            }
            drought_events.append(drought_info)

        # Measure robustness
        rob_metrics = measure_robustness(state, chunk_runner, cfg)

        # Save end-of-cycle Φ for next cycle's pre-drought baseline
        prev_cycle_end_phi = float(compute_phi_hidden(state['hidden'], state['alive']))

        # === ENHANCED METRICS ===
        eff_rank = compute_eff_rank(state['hidden'], state['alive'])
        weight_div = compute_weight_diversity(state['genomes'], state['alive'])
        hidden_ent = compute_hidden_entropy(state['hidden'], state['alive'])
        hidden_conv = compute_hidden_convergence(state['hidden'], state['alive'])

        # Standard V27 metrics
        tick_usage = compute_tick_usage(state, cfg)
        sync_stats = compute_sync_decay_stats(state, cfg)
        lr_stats = compute_lr_stats(state, cfg)
        drift_stats = compute_phenotype_drift(state, cfg)

        # === SAVE SNAPSHOT at drought cycles AND every 5th ===
        save_snap = (cycle % 5 == 0) or is_drought or (cycle == cfg['n_cycles'] - 1)
        if save_snap:
            snap = extract_snapshot(state, cycle, cfg)
            snap_path = os.path.join(output_dir, f'snapshot_c{cycle:02d}.npz')
            np.savez_compressed(snap_path, **snap)
            snapshots.append({'cycle': cycle, 'path': snap_path})

        # Evolution: select and mutate GENOMES
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

        # Rescue if needed
        state, key = rescue_population_v27(state, key, cfg)

        elapsed = time.time() - t_start
        cycle_info = {
            'cycle': cycle,
            'is_drought': is_drought,
            **metrics,
            **rob_metrics,
            # Enhanced metrics
            'eff_rank': eff_rank,
            'weight_diversity': weight_div,
            'hidden_entropy': hidden_ent,
            'hidden_convergence': hidden_conv,
            # Standard
            'tick_usage': tick_usage,
            'sync_decay': sync_stats,
            'lr_stats': lr_stats,
            'phenotype_drift': drift_stats,
            'elapsed_s': elapsed,
        }
        cycle_results.append(cycle_info)

        drought_marker = " [D]" if is_drought else ""
        print(
            f"C{cycle:02d}{drought_marker} | pop={metrics['n_alive_end']:3d} | "
            f"mort={metrics['mortality']:.0%} | "
            f"phi={metrics['mean_phi']:.3f} | "
            f"rob={rob_metrics['robustness']:.3f} | "
            f"eff_rank={eff_rank:.1f} | "
            f"w_div={weight_div:.3f} | "
            f"{elapsed:.0f}s"
        )

        # Save progress JSON periodically
        if cycle % 5 == 0 or cycle == cfg['n_cycles'] - 1:
            progress = {
                'seed': seed,
                'cycle': cycle,
                'config': {k: v for k, v in cfg.items()
                           if isinstance(v, (int, float, str, bool))},
                'cycles': cycle_results,
                'drought_events': drought_events,
                'snapshots': [{'cycle': s['cycle']} for s in snapshots],
            }
            with open(os.path.join(output_dir, f'v32_s{seed}_progress.json'), 'w') as f:
                json.dump(progress, f, indent=2)

    # === FINAL RESULTS ===
    mean_phi = float(np.mean([c['mean_phi'] for c in cycle_results]))
    max_phi = float(np.max([c['mean_phi'] for c in cycle_results]))
    mean_rob = float(np.mean([c['robustness'] for c in cycle_results]))
    max_rob = float(np.max([c['robustness'] for c in cycle_results]))

    # Compute bounce trajectory
    bounces = [d['bounce'] for d in drought_events]
    first_bounce = bounces[0] if bounces else 0.0
    mean_bounce = float(np.mean(bounces)) if bounces else 0.0

    # Late-phase Phi (cycles 15-29, after 3 droughts)
    late_phis = [c['mean_phi'] for c in cycle_results if c['cycle'] >= 15]
    late_mean_phi = float(np.mean(late_phis)) if late_phis else mean_phi

    # Classify seed
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
        'drought_events': drought_events,
        'snapshots': [{'cycle': s['cycle']} for s in snapshots],
        'summary': {
            'mean_phi': mean_phi,
            'max_phi': max_phi,
            'late_mean_phi': late_mean_phi,
            'mean_robustness': mean_rob,
            'max_robustness': max_rob,
            'mean_pred_mse': float(np.mean([c['mean_pred_mse'] for c in cycle_results])),
            'final_lr': cycle_results[-1]['lr_stats']['mean_lr'],
            'final_eff_rank': cycle_results[-1]['eff_rank'],
            'final_weight_diversity': cycle_results[-1]['weight_diversity'],
            'final_pop': cycle_results[-1]['n_alive_end'],
            'first_bounce': first_bounce,
            'mean_bounce': mean_bounce,
            'bounces': bounces,
            'category': category,
        },
    }

    with open(os.path.join(output_dir, f'v32_s{seed}_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSeed {seed} complete — {category}")
    print(f"  Mean Φ:           {mean_phi:.3f}")
    print(f"  Late mean Φ:      {late_mean_phi:.3f}")
    print(f"  Mean robustness:  {mean_rob:.3f}")
    print(f"  First bounce:     {first_bounce:.3f}")
    print(f"  Mean bounce:      {mean_bounce:.3f}")
    print(f"  Final eff_rank:   {results['summary']['final_eff_rank']:.1f}")

    return results
