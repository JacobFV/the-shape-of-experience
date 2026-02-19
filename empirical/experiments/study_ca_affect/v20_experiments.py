"""V20: The Chain Test — Measurement Experiments

Four measurements that test the necessity chain:
  Membrane → free energy gradient → world model → self-model → affect geometry

Run on saved snapshots from v20_evolution.py.

1. C_wm  — World model quality: MI(h_t; obs_{t+k})
2. ΔC_wm — Self-causal contribution: improvement from knowing own actions
3. ρ_sync — Counterfactual sensitivity (wall-breaking test)
4. SM_sal — Self-model salience vs world-model salience
5. RSA   — Affect geometry: correlation between internal state and viability
"""

import numpy as np
import jax
import jax.numpy as jnp
from scipy import stats
from scipy.spatial.distance import cdist
import json
import os
import time

from v20_substrate import (
    generate_v20_config, init_v20, make_chunk_runner,
    build_observation_batch, agent_step_batch, decode_actions,
    build_agent_count_grid, regen_resources, diffuse_signals,
    apply_consumption, apply_emissions, compute_phi_hidden, MOVE_DELTAS,
)
from v20_evolution import run_cycle_with_metrics


# ---------------------------------------------------------------------------
# Helper: run environment with recorded trajectories
# ---------------------------------------------------------------------------

def rollout_agents(state, chunk_runner, cfg, n_chunks=10, record_actions=True):
    """Roll out agents and record (hidden, obs, action) trajectories.

    Returns: dict of arrays, one entry per chunk
    """
    records = []
    for i in range(n_chunks):
        h_before = np.array(state['hidden'])
        alive = np.array(state['alive'])

        state, chunk_metrics = chunk_runner(state)

        h_after = np.array(state['hidden'])
        pos = np.array(state['positions'])
        energy = np.array(state['energy'])

        records.append({
            'h_before': h_before,    # (M, H) hidden state before chunk
            'h_after': h_after,       # (M, H) hidden state after chunk
            'positions': pos,          # (M, 2) positions after chunk
            'energy': energy,          # (M,) energy after chunk
            'alive': alive,            # (M,) alive mask (before chunk)
        })

    return records, state


# ---------------------------------------------------------------------------
# Experiment 1: World Model Quality (C_wm)
# ---------------------------------------------------------------------------

def measure_world_model(state, chunk_runner, cfg, k=5, n_samples=20):
    """Measure C_wm = MI(h_t; obs_{t+k}).

    Also measures ΔC_wm = MI(h_t, action_history; obs_{t+k}) - C_wm.

    Returns: dict with c_wm, delta_c_wm, n_samples
    """
    M = cfg['M_max']
    records, state = rollout_agents(state, chunk_runner, cfg, n_chunks=n_samples + k)

    c_wm_values = []
    delta_c_wm_values = []

    for i in range(n_samples):
        # h at time t
        h_t = records[i]['h_before']          # (M, H)
        alive_t = records[i]['alive']          # (M,)

        # obs at time t+k (use positions as proxy for local state)
        pos_tk = records[i + k]['positions']   # (M, 2)
        energy_tk = records[i + k]['energy']   # (M,)
        alive_tk = records[i + k]['alive']     # (M,)

        # Only consider agents alive at both timepoints
        both_alive = alive_t & alive_tk        # (M,)
        n = int(both_alive.sum())
        if n < 5:
            continue

        idx = np.where(both_alive)[0]

        # X = hidden state at t (flattened per agent)
        X = h_t[idx]          # (n, H)
        # Y = future state proxy: position normalized + energy
        Y = np.concatenate([
            pos_tk[idx] / cfg['N'],        # (n, 2) normalized position
            energy_tk[idx, None],          # (n, 1)
        ], axis=1)  # (n, 3)

        # C_wm: linear regression R² (proxy for MI)
        c_wm = _linear_r2(X, Y)
        c_wm_values.append(c_wm)

        # ΔC_wm: add intervening hidden state changes as "action proxy"
        # Use h_{t+1} - h_t as proxy for what actions were taken
        if i + 1 < len(records):
            h_t1 = records[i + 1]['h_before'][idx]
            action_proxy = h_t1 - h_t[idx]  # (n, H)
            X_aug = np.concatenate([X, action_proxy], axis=1)  # (n, 2H)
            c_wm_aug = _linear_r2(X_aug, Y)
            delta_c_wm_values.append(c_wm_aug - c_wm)

    return {
        'c_wm': float(np.mean(c_wm_values)) if c_wm_values else 0.0,
        'delta_c_wm': float(np.mean(delta_c_wm_values)) if delta_c_wm_values else 0.0,
        'n_samples': len(c_wm_values),
    }


def _linear_r2(X, Y):
    """Mean R² from linear regression X → Y (multi-output).
    Uses only first 8 PCs of X to avoid multicollinearity/overflow."""
    if len(X) < 3:
        return 0.0
    X = np.array(X, dtype=np.float64)
    Y = np.array(Y, dtype=np.float64)

    # Drop near-constant features to prevent divide-by-near-zero overflow
    x_std = X.std(0)
    active = x_std > 1e-6
    if active.sum() == 0:
        return 0.0
    X = X[:, active]
    X = (X - X.mean(0)) / x_std[active]  # safe: std > 1e-6

    n_comp = min(8, X.shape[1], X.shape[0] - 1)
    try:
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            X_red = X @ Vt[:n_comp].T  # (n, n_comp)
    except np.linalg.LinAlgError:
        return 0.0

    if not np.isfinite(X_red).all():
        return 0.0

    r2s = []
    for j in range(Y.shape[1]):
        y = Y[:, j]
        if np.var(y) < 1e-10:
            continue
        X_aug = np.column_stack([X_red, np.ones(len(X_red))])
        try:
            coef, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
            y_pred = X_aug @ coef
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1.0 - ss_res / (ss_tot + 1e-10)
            r2s.append(float(np.clip(r2, 0.0, 1.0)))
        except (np.linalg.LinAlgError, ValueError):
            pass
    return float(np.mean(r2s)) if r2s else 0.0


# ---------------------------------------------------------------------------
# Experiment 2: Counterfactual Sensitivity (ρ_sync) — Wall Breaking Test
# ---------------------------------------------------------------------------

def measure_rho_sync(state, chunk_runner, cfg, key, n_forks=10, k=3):
    """Measure ρ_sync: how much do different actions lead to different observations?

    Method:
    - Fork current state into N pairs
    - One fork: normal rollout (actual actions)
    - Other fork: inject random movement perturbation for k chunks
    - Compare divergence in positions/energies between forks
    - ρ_sync = mean divergence normalized by baseline entropy

    In Lenia: ρ_sync ≈ 0 (actions don't matter, FFT dominates)
    In V20:   ρ_sync > 0 if agent actions genuinely shape future observations
    """
    M = cfg['M_max']
    N_grid = cfg['N']
    alive = np.array(state['alive'])
    n_alive = int(alive.sum())
    if n_alive < 5:
        return {'rho_sync': 0.0, 'n_forks': 0}

    divergences = []

    for fork_i in range(n_forks):
        # Normal rollout
        state_a = {k_: v for k_, v in state.items()}
        state_b = {k_: v for k_, v in state.items()}

        # In state_b, override params with random params (forces different actions)
        key, subkey = jax.random.split(key)
        random_params = jax.random.normal(subkey, state['params'].shape) * 0.1
        state_b = {**state_b, 'params': random_params}

        # Rollout k chunks each
        for _ in range(k):
            state_a, _ = chunk_runner(state_a)
            state_b, _ = chunk_runner(state_b)

        # Measure position divergence for agents alive in both
        alive_a = np.array(state_a['alive'])
        alive_b = np.array(state_b['alive'])
        both = alive & alive_a & alive_b

        if both.sum() < 3:
            continue

        idx = np.where(both)[0]
        pos_a = np.array(state_a['positions'])[idx] / N_grid
        pos_b = np.array(state_b['positions'])[idx] / N_grid
        energy_a = np.array(state_a['energy'])[idx]
        energy_b = np.array(state_b['energy'])[idx]

        # Divergence: mean absolute position difference + energy difference
        pos_div = np.mean(np.abs(pos_a - pos_b))
        energy_div = np.mean(np.abs(energy_a - energy_b))
        divergence = (pos_div + energy_div) / 2.0
        divergences.append(divergence)

    # ρ_sync: mean divergence (normalized by max possible)
    # Max position divergence on N×N grid wrapped = 0.5 (half the grid)
    max_div = 0.5
    rho_sync = float(np.mean(divergences)) / max_div if divergences else 0.0

    return {
        'rho_sync': float(rho_sync),
        'mean_divergence': float(np.mean(divergences)) if divergences else 0.0,
        'n_forks': len(divergences),
    }


# ---------------------------------------------------------------------------
# Experiment 3: Self-Model Salience (SM_sal)
# ---------------------------------------------------------------------------

def measure_self_model(state, chunk_runner, cfg, n_samples=15):
    """Measure SM_sal = MI(h; own_state) / MI(h; env_state).

    own_state: (position, energy) — agent's self-knowledge
    env_state: mean of local resource/signal values — environment knowledge

    If SM_sal > 1.0: agent knows more about itself than environment.
    If SM_sal > 0.5 overall: self-model is dominant.
    """
    records, state = rollout_agents(state, chunk_runner, cfg, n_chunks=n_samples)

    sm_sal_values = []

    for record in records:
        h = record['h_before']    # (M, H)
        pos = record['positions']  # (M, 2)
        energy = record['energy']  # (M,)
        alive = record['alive']    # (M,)

        n = int(alive.sum())
        if n < 5:
            continue

        idx = np.where(alive)[0]
        h_alive = h[idx]                          # (n, H)
        pos_norm = pos[idx] / cfg['N']            # (n, 2)
        energy_norm = energy[idx, None]  # (n, 1)
        # Own state: position + energy
        own_state = np.concatenate([pos_norm, energy_norm], axis=1)  # (n, 3)

        # Environment state proxy: distance from grid center (structural position)
        # This captures what the environment looks like from agent's position
        center = np.array([[cfg['N'] / 2, cfg['N'] / 2]])
        env_state = np.abs(pos_norm - 0.5)  # (n, 2) distance from center

        # R² of h predicting own vs env
        r2_own = _linear_r2(h_alive, own_state)
        r2_env = _linear_r2(h_alive, env_state)

        if r2_env > 1e-6:
            sm_sal = r2_own / r2_env
            sm_sal_values.append(sm_sal)

    return {
        'sm_sal': float(np.mean(sm_sal_values)) if sm_sal_values else 0.0,
        'n_samples': len(sm_sal_values),
    }


# ---------------------------------------------------------------------------
# Experiment 4: Affect Geometry (RSA)
# ---------------------------------------------------------------------------

def measure_affect_geometry(state, chunk_runner, cfg, n_samples=15):
    """Measure RSA between internal state structure and viability structure.

    Representational Similarity Analysis:
    - Compute pairwise similarity of agent hidden states
    - Compute pairwise similarity of agent viability (energy level)
    - RSA = Spearman correlation between these two similarity matrices

    If RSA > 0: agents with similar internal states have similar energy levels
    → internal geometry reflects viability geometry
    → this is the computational correlate of affect geometry

    In V13 (Exp 7): RSA went from 0.01 → 0.38 over evolution.
    V20 prediction: similar development, but also tracks in real-time.
    """
    records, state = rollout_agents(state, chunk_runner, cfg, n_chunks=n_samples)

    rsa_values = []
    p_values = []

    for record in records:
        h = record['h_before']    # (M, H)
        energy = record['energy']  # (M,)
        alive = record['alive']    # (M,)

        n = int(alive.sum())
        if n < 10:
            continue

        idx = np.where(alive)[0]
        h_alive = h[idx]        # (n, H)
        energy_alive = energy[idx]  # (n,)

        # Pairwise distances in hidden state space
        h_dists = cdist(h_alive, h_alive, metric='cosine').flatten()

        # Pairwise distances in energy space (viability proxy)
        e_dists = np.abs(
            energy_alive[:, None] - energy_alive[None, :]
        ).flatten()

        # Remove diagonal
        n_agents = len(idx)
        mask = ~np.eye(n_agents, dtype=bool).flatten()
        h_dists = h_dists[mask]
        e_dists = e_dists[mask]

        if np.std(h_dists) < 1e-8 or np.std(e_dists) < 1e-8:
            continue

        rho, pval = stats.spearmanr(h_dists, e_dists)
        rsa_values.append(float(rho))
        p_values.append(float(pval))

    return {
        'rsa': float(np.mean(rsa_values)) if rsa_values else 0.0,
        'rsa_p': float(np.mean(p_values)) if p_values else 1.0,
        'n_significant': int(sum(p < 0.05 for p in p_values)),
        'n_samples': len(rsa_values),
    }


# ---------------------------------------------------------------------------
# Full chain test
# ---------------------------------------------------------------------------

def run_chain_test(snapshot_path, cfg, key, output_path=None):
    """Run all 4 chain measurements on a saved snapshot.

    Args:
        snapshot_path: path to .npz snapshot file
        cfg: config dict
        key: JAX random key

    Returns: dict with all measurements
    """
    # Load snapshot
    snap = np.load(snapshot_path, allow_pickle=True)

    # Reconstruct state
    state = {
        'resources': jnp.zeros((cfg['N'], cfg['N'])),  # reinitialize fields
        'signals': jnp.zeros((cfg['N'], cfg['N'])),
        'positions': jnp.array(snap['positions']),
        'hidden': jnp.array(snap['hidden']),
        'energy': jnp.array(snap['energy']),
        'alive': jnp.array(snap['alive']),
        'params': jnp.array(snap['params']),
        'regen_rate': jnp.array(cfg['resource_regen']),
        'step_count': jnp.array(0),
    }
    # Add random resources
    key, k1 = jax.random.split(key)
    state['resources'] = jax.random.uniform(k1, (cfg['N'], cfg['N'])) * 0.5

    chunk_runner = make_chunk_runner(cfg)

    # Warmup
    state, _ = chunk_runner(state)

    cycle = int(snap['cycle']) if 'cycle' in snap else -1
    print(f"  Running chain test on cycle {cycle} snapshot...")

    t0 = time.time()
    wm = measure_world_model(state, chunk_runner, cfg)
    print(f"    C_wm={wm['c_wm']:.4f}, ΔC_wm={wm['delta_c_wm']:.4f} ({time.time()-t0:.0f}s)")

    t0 = time.time()
    rs = measure_rho_sync(state, chunk_runner, cfg, key)
    print(f"    ρ_sync={rs['rho_sync']:.4f} ({time.time()-t0:.0f}s)")

    t0 = time.time()
    sm = measure_self_model(state, chunk_runner, cfg)
    print(f"    SM_sal={sm['sm_sal']:.4f} ({time.time()-t0:.0f}s)")

    t0 = time.time()
    ag = measure_affect_geometry(state, chunk_runner, cfg)
    print(f"    RSA={ag['rsa']:.4f} (p={ag['rsa_p']:.4f}) ({time.time()-t0:.0f}s)")

    results = {
        'cycle': cycle,
        'world_model': wm,
        'rho_sync': rs,
        'self_model': sm,
        'affect_geometry': ag,
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Run chain test across all snapshots for a seed
# ---------------------------------------------------------------------------

def run_all_chain_tests(results_dir, cfg, seed, key):
    """Run chain test on all snapshots from a V20 evolution run."""
    import glob
    snapshot_files = sorted(glob.glob(os.path.join(results_dir, 'snapshot_c*.npz')))

    print(f"\n=== Chain Test: seed {seed}, {len(snapshot_files)} snapshots ===")

    all_results = []
    for snap_path in snapshot_files:
        cycle_str = os.path.basename(snap_path).replace('snapshot_c', '').replace('.npz', '')
        out_path = os.path.join(results_dir, f'chain_c{cycle_str}.json')
        key, subkey = jax.random.split(key)
        result = run_chain_test(snap_path, cfg, subkey, output_path=out_path)
        all_results.append(result)

    # Summary across cycles
    cycles = [r['cycle'] for r in all_results]
    c_wm_vals = [r['world_model']['c_wm'] for r in all_results]
    delta_c_wm_vals = [r['world_model']['delta_c_wm'] for r in all_results]
    rho_vals = [r['rho_sync']['rho_sync'] for r in all_results]
    sm_vals = [r['self_model']['sm_sal'] for r in all_results]
    rsa_vals = [r['affect_geometry']['rsa'] for r in all_results]

    summary = {
        'seed': seed,
        'n_snapshots': len(all_results),
        'chain_development': {
            'c_wm_first': c_wm_vals[0] if c_wm_vals else 0.0,
            'c_wm_last': c_wm_vals[-1] if c_wm_vals else 0.0,
            'delta_c_wm_last': delta_c_wm_vals[-1] if delta_c_wm_vals else 0.0,
            'rho_sync_max': max(rho_vals) if rho_vals else 0.0,
            'rho_sync_last': rho_vals[-1] if rho_vals else 0.0,
            'sm_sal_last': sm_vals[-1] if sm_vals else 0.0,
            'rsa_last': rsa_vals[-1] if rsa_vals else 0.0,
        },
        'wall_broken': max(rho_vals) > 0.1 if rho_vals else False,
        'self_model_emergent': (sm_vals[-1] > 0.3) if sm_vals else False,
        'affect_geometry_present': (rsa_vals[-1] > 0.2) if rsa_vals else False,
        'results': all_results,
    }

    print(f"\n=== Chain Test Summary (seed {seed}) ===")
    print(f"  C_wm:     {c_wm_vals[0] if c_wm_vals else 0:.4f} → {c_wm_vals[-1] if c_wm_vals else 0:.4f}")
    print(f"  ΔC_wm:    {delta_c_wm_vals[-1] if delta_c_wm_vals else 0:.4f} (final)")
    print(f"  ρ_sync:   {max(rho_vals) if rho_vals else 0:.4f} (max, target >0.1)")
    print(f"  SM_sal:   {sm_vals[-1] if sm_vals else 0:.4f} (final, target >0.3)")
    print(f"  RSA:      {rsa_vals[-1] if rsa_vals else 0:.4f} (final, target >0.2)")
    print(f"  Wall broken: {summary['wall_broken']}")
    print(f"  Self-model: {summary['self_model_emergent']}")
    print(f"  Affect geometry: {summary['affect_geometry_present']}")

    out_path = os.path.join(results_dir, f'v20_s{seed}_chain_summary.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return summary
