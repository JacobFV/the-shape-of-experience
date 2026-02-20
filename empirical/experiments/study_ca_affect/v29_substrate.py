"""V29: Social Prediction — Neighbor Energy Prediction

V22-V28 all use SELF prediction: agents predict their own energy delta.
The gradient signal is individually-focused — each agent learns about its own
energy trajectory. High Φ appears seed-dependently (~1/3 seeds).

V29 tests SOCIAL prediction: agents predict the mean energy of their neighbors
within the observation window. This forces hidden states to encode information
about OTHER agents, not just self. The gradient signal becomes socially-coupled.

Hypothesis: Social prediction creates natural integration pressure because:
  1. To predict neighbor energy, agents must model neighbor BEHAVIOR
  2. Modeling neighbor behavior requires representing the local social dynamics
  3. Social representations are inherently multi-agent → harder to partition
  4. This should produce higher and more reliable Φ

Architecture:
  - Same V27 base (GRU + 8 inner ticks + 2-layer MLP prediction head, tanh w=8)
  - Only change: prediction target = mean energy of alive neighbors within obs_radius
  - Fallback: if no neighbors in window, target = own energy delta (V27 behavior)

Pre-registered predictions:
  P1: Mean Φ > V27's 0.090 across seeds (social coupling forces integration)
  P2: Less seed-dependent (social target is richer, more gradients to exploit)
  P3: Robustness ≥ V27 (social awareness helps coordination under stress)
  P4: Prediction MSE higher than V27 (harder target) but still decreasing within lifetime

Falsification:
  - P1 fails: social prediction doesn't create integration pressure (coupling is too weak)
  - P2 fails: same seed-dependence as V27 (social target doesn't change evolutionary landscape)
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
import numpy as np

from v20_substrate import (
    gather_patch, build_observation, build_observation_batch,
    build_agent_count_grid, apply_consumption, apply_emissions,
    diffuse_signals, regen_resources, decode_actions, MOVE_DELTAS,
    compute_phi_hidden, compute_robustness,
)

from v21_substrate import (
    compute_sync_summary, compute_phi_sync,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def generate_v29_config(**kwargs):
    """Generate V29 configuration — social prediction."""
    cfg = {
        'N': 128,
        'M_max': 256,

        'obs_radius': 2,
        'embed_dim': 24,
        'hidden_dim': 16,
        'n_actions': 7,

        'K_max': 8,

        # Same MLP head as V27
        'predict_hidden': 8,

        # Environment dynamics (same as V20b+)
        'resource_regen': 0.002,
        'signal_decay': 0.04,
        'signal_diffusion': 0.15,
        'metabolic_cost': 0.0004,
        'initial_energy': 1.0,
        'resource_value': 1.5,
        'initial_resource': 0.5,

        'stress_regen': 0.0002,

        'mutation_std': 0.03,
        'tournament_size': 4,
        'elite_fraction': 0.5,

        'activate_offspring': True,
        'drought_every': 5,
        'drought_depletion': 0.01,
        'drought_regen': 0.0,

        'lamarckian': False,
        'grad_every': 1,

        'chunk_size': 50,
        'steps_per_cycle': 5000,
        'n_cycles': 30,
    }
    cfg.update(kwargs)

    obs_side = 2 * cfg['obs_radius'] + 1
    obs_flat = obs_side * obs_side * 3 + 1
    cfg['obs_side'] = obs_side
    cfg['obs_flat'] = obs_flat

    shapes = _param_shapes(cfg)
    total = sum(int(np.prod(s)) for s in shapes.values())
    cfg['n_params'] = total

    return cfg


def _param_shapes(cfg):
    """Same param layout as V27."""
    obs_flat = cfg['obs_flat']
    E = cfg['embed_dim']
    H = cfg['hidden_dim']
    O = cfg['n_actions']
    K = cfg['K_max']
    PH = cfg['predict_hidden']
    return {
        'embed_W': (obs_flat, E),
        'embed_b': (E,),
        'gru_Wz': (E + H, H),
        'gru_bz': (H,),
        'gru_Wr': (E + H, H),
        'gru_br': (H,),
        'gru_Wh': (E + H, H),
        'gru_bh': (H,),
        'out_W': (H, O),
        'out_b': (O,),
        'internal_embed_W': (3, E),
        'internal_embed_b': (E,),
        'tick_weights': (K,),
        'sync_decay_raw': (1,),
        'predict_W1': (H, PH),
        'predict_b1': (PH,),
        'predict_W2': (PH, 1),
        'predict_b2': (1,),
        'lr_raw': (1,),
    }


def unpack_params(flat, cfg):
    shapes = _param_shapes(cfg)
    params = {}
    idx = 0
    for name, shape in shapes.items():
        size = int(np.prod(shape))
        params[name] = flat[idx:idx + size].reshape(shape)
        idx += size
    return params


# ---------------------------------------------------------------------------
# Compute neighbor mean energy (spatial aggregation)
# ---------------------------------------------------------------------------

def compute_neighbor_mean_energy(positions, energy, alive, obs_radius, N):
    """Compute mean energy of alive neighbors within obs_radius for each agent.

    Uses the energy grid + agent positions to compute local energy averages.
    For agents with no neighbors, returns own energy as fallback.

    Args:
        positions: (M, 2) int32, agent positions
        energy: (M,) float32, agent energies
        alive: (M,) bool, alive mask
        obs_radius: int, observation radius
        N: int, grid size

    Returns:
        neighbor_mean_energy: (M,) float32
    """
    M = positions.shape[0]

    # Build energy grid: accumulate alive agent energies onto grid
    energy_grid = jnp.zeros((N, N), dtype=jnp.float32)
    count_grid = jnp.zeros((N, N), dtype=jnp.float32)

    alive_f = alive.astype(jnp.float32)
    # Scatter agent energies to grid
    energy_grid = energy_grid.at[positions[:, 0], positions[:, 1]].add(
        energy * alive_f
    )
    count_grid = count_grid.at[positions[:, 0], positions[:, 1]].add(alive_f)

    # For each agent, sum energy and count in observation window
    obs_side = 2 * obs_radius + 1

    def agent_neighbor_energy(pos):
        """Sum energy and count of neighbors for one agent."""
        r, c = pos[0], pos[1]
        total_e = 0.0
        total_n = 0.0
        for dr in range(-obs_radius, obs_radius + 1):
            for dc in range(-obs_radius, obs_radius + 1):
                nr = (r + dr) % N
                nc = (c + dc) % N
                total_e = total_e + energy_grid[nr, nc]
                total_n = total_n + count_grid[nr, nc]
        return total_e, total_n

    # Vectorize over agents
    all_e, all_n = jax.vmap(agent_neighbor_energy)(positions)

    # Subtract self energy and self count
    self_e = energy * alive_f
    self_n = alive_f
    neighbor_e = all_e - self_e
    neighbor_n = all_n - self_n

    # Mean neighbor energy (fallback to own energy if no neighbors)
    has_neighbors = neighbor_n > 0.0
    mean_e = jnp.where(
        has_neighbors,
        neighbor_e / jnp.maximum(neighbor_n, 1.0),
        energy  # fallback: own energy
    )

    return mean_e


# ---------------------------------------------------------------------------
# Forward pass (same as V27)
# ---------------------------------------------------------------------------

def agent_multi_tick_v29(obs_flat, hidden, sync_matrix, params_flat, cfg):
    """Same as V27 — MLP prediction head."""
    p = unpack_params(params_flat, cfg)
    K = cfg['K_max']

    x_ext = jnp.tanh(obs_flat @ p['embed_W'] + p['embed_b'])
    sync_decay = 0.5 + 0.499 * jax.nn.sigmoid(p['sync_decay_raw'][0])

    def tick_fn(carry, tick_idx):
        h, S = carry
        sync_sum = compute_sync_summary(S)
        x_int = jnp.tanh(sync_sum @ p['internal_embed_W'] + p['internal_embed_b'])
        x = jnp.where(tick_idx == 0, x_ext, x_int)

        xh = jnp.concatenate([x, h])
        z = jax.nn.sigmoid(xh @ p['gru_Wz'] + p['gru_bz'])
        r = jax.nn.sigmoid(xh @ p['gru_Wr'] + p['gru_br'])
        h_tilde = jnp.tanh(
            jnp.concatenate([x, r * h]) @ p['gru_Wh'] + p['gru_bh']
        )
        new_h = z * h + (1.0 - z) * h_tilde
        new_S = sync_decay * S + jnp.outer(new_h, new_h)
        output = new_h @ p['out_W'] + p['out_b']
        return (new_h, new_S), output

    (final_hidden, final_sync), all_outputs = lax.scan(
        tick_fn, (hidden, sync_matrix), jnp.arange(K)
    )

    weights = jax.nn.softmax(p['tick_weights'])
    actions = jnp.einsum('k,ko->o', weights, all_outputs)

    # MLP prediction
    mlp_hidden = jnp.tanh(final_hidden @ p['predict_W1'] + p['predict_b1'])
    prediction = (mlp_hidden @ p['predict_W2']).squeeze() + p['predict_b2'][0]

    return final_hidden, final_sync, actions, prediction


agent_multi_tick_v29_batch = jax.vmap(
    agent_multi_tick_v29, in_axes=(0, 0, 0, 0, None)
)


# ---------------------------------------------------------------------------
# Prediction gradient — targets NEIGHBOR mean energy
# ---------------------------------------------------------------------------

def _prediction_loss_v29(phenotype_flat, obs_flat, hidden, sync_matrix, cfg, target):
    """Loss: (predicted - target)² where target = neighbor mean energy."""
    p = unpack_params(phenotype_flat, cfg)
    K = cfg['K_max']

    x_ext = jnp.tanh(obs_flat @ p['embed_W'] + p['embed_b'])
    sync_decay = 0.5 + 0.499 * jax.nn.sigmoid(p['sync_decay_raw'][0])

    def tick_fn(carry, tick_idx):
        h, S = carry
        sync_sum = compute_sync_summary(S)
        x_int = jnp.tanh(sync_sum @ p['internal_embed_W'] + p['internal_embed_b'])
        x = jnp.where(tick_idx == 0, x_ext, x_int)

        xh = jnp.concatenate([x, h])
        z = jax.nn.sigmoid(xh @ p['gru_Wz'] + p['gru_bz'])
        r = jax.nn.sigmoid(xh @ p['gru_Wr'] + p['gru_br'])
        h_tilde = jnp.tanh(
            jnp.concatenate([x, r * h]) @ p['gru_Wh'] + p['gru_bh']
        )
        new_h = z * h + (1.0 - z) * h_tilde
        new_S = sync_decay * S + jnp.outer(new_h, new_h)
        return (new_h, new_S), None

    (final_hidden, _), _ = lax.scan(
        tick_fn, (hidden, sync_matrix), jnp.arange(K)
    )

    mlp_hidden = jnp.tanh(final_hidden @ p['predict_W1'] + p['predict_b1'])
    pred = (mlp_hidden @ p['predict_W2']).squeeze() + p['predict_b2'][0]
    return (pred - target) ** 2


_prediction_grad_single_v29 = jax.grad(_prediction_loss_v29, argnums=0)
_prediction_grad_batch_v29 = jax.vmap(
    _prediction_grad_single_v29, in_axes=(0, 0, 0, 0, None, 0)
)


# ---------------------------------------------------------------------------
# Learning rate
# ---------------------------------------------------------------------------

def _lr_offset(cfg):
    shapes = _param_shapes(cfg)
    offset = 0
    for name, shape in shapes.items():
        if name == 'lr_raw':
            return offset
        offset += int(np.prod(shape))
    raise ValueError("lr_raw not found")


def extract_lr_jax(phenotypes, cfg):
    offset = _lr_offset(cfg)
    raw = phenotypes[:, offset]
    return 1e-5 + (1e-2 - 1e-5) * jax.nn.sigmoid(raw)


def extract_lr_np(params, cfg):
    offset = _lr_offset(cfg)
    raw = np.array(params)[:, offset]
    return 1e-5 + (1e-2 - 1e-5) / (1.0 + np.exp(-raw))


# ---------------------------------------------------------------------------
# V29 environment step
# ---------------------------------------------------------------------------

def _env_step_v29_inner(state, cfg):
    """V29 env step: same as V27 but also computes neighbor mean energy."""
    N = cfg['N']

    agent_count = build_agent_count_grid(state['positions'], state['alive'], N)
    obs = build_observation_batch(
        state['positions'], state['resources'], state['signals'],
        agent_count, state['energy'], cfg
    )

    new_hidden, new_sync, raw_actions, predictions = agent_multi_tick_v29_batch(
        obs, state['hidden'], state['sync_matrices'], state['phenotypes'], cfg
    )
    new_hidden = new_hidden * state['alive'][:, None]
    new_sync = new_sync * state['alive'][:, None, None]

    move_idx, consume, emit = decode_actions(raw_actions, cfg)
    deltas = MOVE_DELTAS[move_idx]
    new_positions = (state['positions'] + deltas) % N
    new_positions = jnp.where(state['alive'][:, None], new_positions, state['positions'])

    resources, actual_consumed = apply_consumption(
        state['resources'], new_positions, consume * state['alive'], state['alive'], N
    )
    signals = apply_emissions(state['signals'], new_positions, emit, state['alive'], N)
    signals = diffuse_signals(signals, cfg)
    resources = regen_resources(resources, state['regen_rate'])

    energy = state['energy'] - cfg['metabolic_cost'] + actual_consumed * cfg['resource_value']
    energy = jnp.clip(energy, 0.0, 2.0)
    new_alive = state['alive'] & (energy > 0.0)

    # Compute SOCIAL target: mean energy of neighbors
    neighbor_mean_e = compute_neighbor_mean_energy(
        new_positions, energy, new_alive, cfg['obs_radius'], N
    )

    new_state = {
        'resources': resources,
        'signals': signals,
        'positions': new_positions,
        'hidden': new_hidden,
        'energy': energy,
        'alive': new_alive,
        'phenotypes': state['phenotypes'],
        'genomes': state['genomes'],
        'sync_matrices': new_sync,
        'regen_rate': state['regen_rate'],
        'step_count': state['step_count'] + 1,
    }
    return new_state, obs, predictions, neighbor_mean_e


# ---------------------------------------------------------------------------
# Chunk runner with social prediction gradient
# ---------------------------------------------------------------------------

def make_v29_chunk_runner(cfg):
    """Create JIT-compiled chunk runner with SOCIAL prediction gradient."""
    def scan_body(state, step_idx):
        pre_energy = state['energy']
        pre_hidden = state['hidden']
        pre_sync = state['sync_matrices']
        pre_alive = state['alive']
        pre_phenotypes = state['phenotypes']

        new_state, obs, predictions, neighbor_mean_e = _env_step_v29_inner(state, cfg)

        # Social target: neighbor mean energy (NOT own energy delta!)
        target = neighbor_mean_e  # (M,)

        alive_f = pre_alive.astype(jnp.float32)
        n_alive = jnp.maximum(jnp.sum(alive_f), 1.0)
        pred_mse = jnp.sum((predictions - target) ** 2 * alive_f) / n_alive

        # Gradient wrt social prediction target
        grads = _prediction_grad_batch_v29(
            pre_phenotypes, obs, pre_hidden, pre_sync, cfg, target
        )

        grad_norm = jnp.sqrt(jnp.sum(grads ** 2, axis=1, keepdims=True) + 1e-8)
        max_norm = 1.0
        grads = grads * jnp.minimum(1.0, max_norm / grad_norm)

        lr = extract_lr_jax(pre_phenotypes, cfg)
        updated_phenotypes = pre_phenotypes - lr[:, None] * grads * alive_f[:, None]

        do_update = (step_idx % cfg['grad_every'] == 0)
        new_phenotypes = jnp.where(do_update, updated_phenotypes, pre_phenotypes)
        new_state = {**new_state, 'phenotypes': new_phenotypes}

        h_before_norm = jnp.sqrt(jnp.sum(pre_hidden ** 2, axis=1) + 1e-8)
        h_diff_norm = jnp.sqrt(jnp.sum((new_state['hidden'] - pre_hidden) ** 2, axis=1))
        div = h_diff_norm / h_before_norm
        mean_div = jnp.sum(div * alive_f) / n_alive

        metrics = {
            'n_alive': jnp.sum(new_state['alive']).astype(jnp.float32),
            'mean_energy': jnp.mean(new_state['energy'] * new_state['alive']),
            'mean_divergence': mean_div,
            'pred_mse': pred_mse,
            'mean_grad_norm': jnp.sum(grad_norm.squeeze() * alive_f) / n_alive,
        }
        return new_state, metrics

    @jax.jit
    def run_chunk(state):
        final_state, metrics = lax.scan(
            scan_body, state, jnp.arange(cfg['chunk_size'])
        )
        return final_state, metrics

    return run_chunk


# ---------------------------------------------------------------------------
# Utility extractors
# ---------------------------------------------------------------------------

def _param_offset(cfg, target_name):
    shapes = _param_shapes(cfg)
    offset = 0
    for name, shape in shapes.items():
        if name == target_name:
            return offset
        offset += int(np.prod(shape))
    raise ValueError(f"Unknown param: {target_name}")


def extract_tick_weights_np(params, cfg):
    offset = _param_offset(cfg, 'tick_weights')
    K = cfg['K_max']
    raw = np.array(params)[:, offset:offset + K]
    raw_shifted = raw - np.max(raw, axis=1, keepdims=True)
    exp_raw = np.exp(raw_shifted)
    return exp_raw / np.sum(exp_raw, axis=1, keepdims=True)


def extract_sync_decay_np(params, cfg):
    offset = _param_offset(cfg, 'sync_decay_raw')
    raw = np.array(params)[:, offset]
    return 0.5 + 0.499 / (1.0 + np.exp(-raw))


# ---------------------------------------------------------------------------
# Initialization + Snapshot
# ---------------------------------------------------------------------------

def init_v29(seed, cfg):
    """Initialize V29 state — same as V27."""
    key = jax.random.PRNGKey(seed)
    N = cfg['N']
    M = cfg['M_max']
    H = cfg['hidden_dim']
    P = cfg['n_params']

    key, k1, k2, k3, k4 = jax.random.split(key, 5)

    resources = jax.random.uniform(k1, (N, N)) * cfg['initial_resource']
    resources = resources + jax.random.uniform(k2, (N, N)) * 0.3
    resources = jnp.clip(resources, 0.0, 1.0)
    signals = jnp.zeros((N, N))

    n_initial = M // 4
    positions_init = jax.random.randint(k3, (n_initial, 2), 0, N)
    positions = jnp.zeros((M, 2), dtype=jnp.int32)
    positions = positions.at[:n_initial].set(positions_init)

    hidden = jnp.zeros((M, H))
    sync_matrices = jnp.zeros((M, H, H))

    energy = jnp.zeros(M)
    energy = energy.at[:n_initial].set(cfg['initial_energy'])

    alive = jnp.zeros(M, dtype=jnp.bool_)
    alive = alive.at[:n_initial].set(True)

    genomes = jax.random.normal(k4, (M, P)) * 0.1
    phenotypes = genomes.copy()

    state = {
        'resources': resources,
        'signals': signals,
        'positions': positions,
        'hidden': hidden,
        'energy': energy,
        'alive': alive,
        'genomes': genomes,
        'phenotypes': phenotypes,
        'sync_matrices': sync_matrices,
        'regen_rate': jnp.array(cfg['resource_regen']),
        'step_count': jnp.array(0),
    }
    return state, key


def extract_snapshot(state, cycle, cfg):
    alive = state['alive']
    return {
        'cycle': cycle,
        'hidden': np.array(state['hidden']),
        'positions': np.array(state['positions']),
        'energy': np.array(state['energy']),
        'alive': np.array(alive),
        'genomes': np.array(state['genomes']),
        'phenotypes': np.array(state['phenotypes']),
        'resources': np.array(state['resources']),
        'signals': np.array(state['signals']),
        'n_alive': int(jnp.sum(alive)),
    }
