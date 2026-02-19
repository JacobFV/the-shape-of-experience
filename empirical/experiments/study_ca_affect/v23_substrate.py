"""V23: World-Model Gradient — Multi-Target Predictive Learning

V22 showed: within-lifetime learning works (MSE drops 100-15,000x per
lifetime), evolution doesn't suppress it, but scalar energy-delta prediction
is orthogonal to integration under stress. "Prediction ≠ integration."

Root cause: a single scalar prediction doesn't REQUIRE cross-component
coordination. One hidden unit suffices to predict energy delta. The gradient
distributes evenly across units without forcing any to specialize or
coordinate.

V23 tests whether multi-dimensional prediction — predicting targets that
require DIFFERENT information sources — forces integration by creating
factored representations that must coexist in the same hidden state.

Three prediction targets, each drawing on different information:
  T0: energy delta (self-focused — how will my energy change?)
  T1: local resource delta (environment-focused — how will my resources change?)
  T2: local neighbor count delta (social-focused — how will my neighborhood change?)

The gradient from all three targets flows through the same 16 GRU hidden
units, forcing them to encode self-state, environmental context, AND social
context simultaneously. This is the computational analog of the understanding
vs reactivity distinction: reactivity maps present → action, understanding
maps present → possibility landscape → action. Multi-target prediction
forces the agent to model the possibility landscape.

Pre-registered predictions:
  1. All 3 targets show within-lifetime MSE improvement
  2. Phi improves over V22 (mean Phi > 0.11, V22 was ~0.10)
  3. Robustness improves over V22 (mean > 1.0, V22 was ~0.98)
  4. predict_W columns specialize (different hidden units contribute to
     different targets — measure via weight correlation)
  5. Target prediction quality varies: energy (easiest), resources, neighbors

Falsification criteria:
  - Any target MSE increases over lifetime → that target not learnable
  - Phi/robustness no better than V22 → multi-target still orthogonal
  - predict_W columns collapse to parallel → no specialization

Architecture (over V22):
  - predict_W (H, 3), predict_b (3,): multi-target prediction head
  - lr_raw (1,): evolvable learning rate (same as V22)
  - 52 new params over V21 (4,092 total vs V22's 4,058)
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
import numpy as np

# Import environment dynamics from V20 (unchanged)
from v20_substrate import (
    gather_patch, build_observation, build_observation_batch,
    build_agent_count_grid, apply_consumption, apply_emissions,
    diffuse_signals, regen_resources, decode_actions, MOVE_DELTAS,
    compute_phi_hidden, compute_robustness,
)

# Import sync operations from V21 (unchanged)
from v21_substrate import (
    compute_sync_summary, compute_phi_sync,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_TARGETS = 3  # energy delta, resource delta, neighbor delta

def generate_v23_config(**kwargs):
    """Generate V23 configuration with multi-target prediction."""
    cfg = {
        # Grid
        'N': 128,
        'M_max': 256,

        # Agent architecture
        'obs_radius': 2,
        'embed_dim': 24,
        'hidden_dim': 16,
        'n_actions': 7,

        # CTM inner ticks (from V21)
        'K_max': 8,

        # Environment dynamics (same as V20b/V21/V22)
        'resource_regen': 0.002,
        'signal_decay': 0.04,
        'signal_diffusion': 0.15,
        'metabolic_cost': 0.0004,
        'initial_energy': 1.0,
        'resource_value': 1.5,
        'initial_resource': 0.5,

        # Stress
        'stress_regen': 0.0002,

        # Evolution
        'mutation_std': 0.03,
        'tournament_size': 4,
        'elite_fraction': 0.5,

        # V20b defaults
        'activate_offspring': True,
        'drought_every': 5,
        'drought_depletion': 0.01,
        'drought_regen': 0.0,

        # V22: learning mode
        'lamarckian': False,

        # V22: gradient update frequency
        'grad_every': 1,

        # V23: prediction targets
        'n_targets': N_TARGETS,
        'neighbor_scale': 5.0,  # scale down neighbor count delta

        # Run
        'chunk_size': 50,
        'steps_per_cycle': 5000,
        'n_cycles': 30,
    }
    cfg.update(kwargs)

    # Derived dimensions
    obs_side = 2 * cfg['obs_radius'] + 1   # 5
    obs_flat = obs_side * obs_side * 3 + 1  # 76
    cfg['obs_side'] = obs_side
    cfg['obs_flat'] = obs_flat

    shapes = _param_shapes(cfg)
    total = sum(int(np.prod(s)) for s in shapes.values())
    cfg['n_params'] = total

    return cfg


def _param_shapes(cfg):
    """Ordered dict of parameter name -> shape."""
    obs_flat = cfg['obs_flat']
    E = cfg['embed_dim']
    H = cfg['hidden_dim']
    O = cfg['n_actions']
    K = cfg['K_max']
    T = cfg['n_targets']
    return {
        # V20 base params
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
        # V21 CTM params
        'internal_embed_W': (3, E),
        'internal_embed_b': (E,),
        'tick_weights': (K,),
        'sync_decay_raw': (1,),
        # V23 multi-target prediction params
        'predict_W': (H, T),
        'predict_b': (T,),
        'lr_raw': (1,),
    }


def unpack_params(flat, cfg):
    """Unpack flat parameter vector into named weight matrices."""
    shapes = _param_shapes(cfg)
    params = {}
    idx = 0
    for name, shape in shapes.items():
        size = int(np.prod(shape))
        params[name] = flat[idx:idx + size].reshape(shape)
        idx += size
    return params


# ---------------------------------------------------------------------------
# Local environment metrics (for prediction targets)
# ---------------------------------------------------------------------------

def _gather_resource_mean(resources, pos, obs_radius):
    """Mean resource in obs patch around one agent position."""
    patch = gather_patch(resources, pos, obs_radius)  # (obs_side, obs_side)
    return jnp.mean(patch)

_gather_resource_mean_batch = jax.vmap(
    _gather_resource_mean, in_axes=(None, 0, None)
)

def _gather_neighbor_count(agent_count_grid, pos, obs_radius):
    """Count of other agents in obs patch (subtract self)."""
    patch = gather_patch(agent_count_grid, pos, obs_radius)
    return jnp.sum(patch) - 1.0  # subtract self

_gather_neighbor_count_batch = jax.vmap(
    _gather_neighbor_count, in_axes=(None, 0, None)
)

def compute_local_resource_mean(resources, positions, alive, cfg):
    """Mean resource in obs_radius patch per agent. Dead agents → 0."""
    means = _gather_resource_mean_batch(resources, positions, cfg['obs_radius'])
    return means * alive.astype(jnp.float32)

def compute_local_neighbor_count(agent_count_grid, positions, alive, cfg):
    """Neighbor count in obs_radius patch per agent. Dead agents → 0."""
    counts = _gather_neighbor_count_batch(agent_count_grid, positions, cfg['obs_radius'])
    return counts * alive.astype(jnp.float32)


# ---------------------------------------------------------------------------
# Agent multi-tick forward pass with multi-target prediction
# ---------------------------------------------------------------------------

def agent_multi_tick_v23(obs_flat, hidden, sync_matrix, params_flat, cfg):
    """Run K_max GRU ticks for one agent, returning multi-target predictions.

    Returns:
        final_hidden: (H,)
        final_sync: (H, H)
        actions: (n_actions,) weighted mix of tick outputs
        predictions: (n_targets,) predicted deltas for [energy, resources, neighbors]
    """
    p = unpack_params(params_flat, cfg)
    K = cfg['K_max']

    # Pre-compute external embedding (tick 0 only)
    x_ext = jnp.tanh(obs_flat @ p['embed_W'] + p['embed_b'])

    # Sync decay: sigmoid mapped to [0.5, 0.999]
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

    # Weighted mix of tick outputs
    weights = jax.nn.softmax(p['tick_weights'])
    actions = jnp.einsum('k,ko->o', weights, all_outputs)

    # Multi-target prediction from final hidden state
    predictions = final_hidden @ p['predict_W'] + p['predict_b']  # (n_targets,)

    return final_hidden, final_sync, actions, predictions


# Vectorize over agents
agent_multi_tick_v23_batch = jax.vmap(
    agent_multi_tick_v23, in_axes=(0, 0, 0, 0, None)
)


# ---------------------------------------------------------------------------
# Multi-target prediction gradient (single agent)
# ---------------------------------------------------------------------------

def _prediction_loss_v23(phenotype_flat, obs_flat, hidden, sync_matrix, cfg, actual_targets):
    """Scalar loss: sum of (predicted[i] - actual[i])^2 across targets.

    actual_targets: (n_targets,) = [energy_delta, resource_delta, neighbor_delta_scaled]

    Gradient through all K ticks of multi-tick computation provides
    learning signal that forces factored representations.
    """
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

    predictions = final_hidden @ p['predict_W'] + p['predict_b']  # (n_targets,)
    return jnp.sum((predictions - actual_targets) ** 2)


# Gradient of prediction loss wrt phenotype params (single agent)
_prediction_grad_single = jax.grad(_prediction_loss_v23, argnums=0)

# Vectorize over agents
_prediction_grad_batch = jax.vmap(
    _prediction_grad_single, in_axes=(0, 0, 0, 0, None, 0)
)


# ---------------------------------------------------------------------------
# Learning rate extraction (same as V22)
# ---------------------------------------------------------------------------

def _lr_offset(cfg):
    """Compute offset of lr_raw in the flat parameter vector."""
    shapes = _param_shapes(cfg)
    offset = 0
    for name, shape in shapes.items():
        if name == 'lr_raw':
            return offset
        offset += int(np.prod(shape))
    raise ValueError("lr_raw not found in param shapes")


def extract_lr_jax(phenotypes, cfg):
    """Extract learning rates from phenotype vectors. Returns (M,) in [1e-5, 1e-2]."""
    offset = _lr_offset(cfg)
    raw = phenotypes[:, offset]
    return 1e-5 + (1e-2 - 1e-5) * jax.nn.sigmoid(raw)


def extract_lr_np(params, cfg):
    """Numpy version for analysis."""
    offset = _lr_offset(cfg)
    raw = np.array(params)[:, offset]
    return 1e-5 + (1e-2 - 1e-5) / (1.0 + np.exp(-raw))


# ---------------------------------------------------------------------------
# V23 environment step
# ---------------------------------------------------------------------------

def _env_step_v23_inner(state, cfg):
    """V23 env step: multi-tick with phenotypes, returns multi-target predictions."""
    N = cfg['N']

    # 1. Build agent count grid
    agent_count = build_agent_count_grid(state['positions'], state['alive'], N)

    # 2. Build observations
    obs = build_observation_batch(
        state['positions'], state['resources'], state['signals'],
        agent_count, state['energy'], cfg
    )

    # 3. Multi-tick forward pass using PHENOTYPES
    new_hidden, new_sync, raw_actions, predictions = agent_multi_tick_v23_batch(
        obs, state['hidden'], state['sync_matrices'], state['phenotypes'], cfg
    )
    new_hidden = new_hidden * state['alive'][:, None]
    new_sync = new_sync * state['alive'][:, None, None]

    # 4. Decode actions
    move_idx, consume, emit = decode_actions(raw_actions, cfg)

    # 5. Movement
    deltas = MOVE_DELTAS[move_idx]
    new_positions = (state['positions'] + deltas) % N
    new_positions = jnp.where(state['alive'][:, None], new_positions, state['positions'])

    # 6. Consumption
    resources, actual_consumed = apply_consumption(
        state['resources'], new_positions, consume * state['alive'], state['alive'], N
    )

    # 7. Emission
    signals = apply_emissions(state['signals'], new_positions, emit, state['alive'], N)

    # 8. Environment dynamics
    signals = diffuse_signals(signals, cfg)
    resources = regen_resources(resources, state['regen_rate'])

    # 9. Energy update
    energy = state['energy'] - cfg['metabolic_cost'] + actual_consumed * cfg['resource_value']
    energy = jnp.clip(energy, 0.0, 2.0)

    # 10. Kill starved agents
    new_alive = state['alive'] & (energy > 0.0)

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
    return new_state, obs, predictions, agent_count


# ---------------------------------------------------------------------------
# Chunk runner with multi-target SGD
# ---------------------------------------------------------------------------

def make_v23_chunk_runner(cfg):
    """Create JIT-compiled chunk runner with multi-target gradient learning.

    Each step:
      1. Compute pre-step local metrics (resource mean, neighbor count)
      2. Forward pass → actions + 3-target predictions
      3. Execute env dynamics
      4. Compute post-step local metrics
      5. Build 3 actual targets
      6. Gradient of multi-target loss through K ticks
      7. SGD update on phenotype
    """
    N = cfg['N']
    obs_radius = cfg['obs_radius']
    neighbor_scale = cfg.get('neighbor_scale', 5.0)

    def scan_body(state, step_idx):
        # Pre-step values
        pre_energy = state['energy']
        pre_hidden = state['hidden']
        pre_sync = state['sync_matrices']
        pre_alive = state['alive']
        pre_phenotypes = state['phenotypes']

        # Pre-step local metrics
        pre_agent_count = build_agent_count_grid(state['positions'], state['alive'], N)
        pre_resource_mean = compute_local_resource_mean(
            state['resources'], state['positions'], state['alive'], cfg
        )
        pre_neighbor_count = compute_local_neighbor_count(
            pre_agent_count, state['positions'], state['alive'], cfg
        )

        # Forward pass + env dynamics
        new_state, obs, predictions, _ = _env_step_v23_inner(state, cfg)

        # Post-step local metrics (at NEW positions)
        post_agent_count = build_agent_count_grid(
            new_state['positions'], new_state['alive'], N
        )
        post_resource_mean = compute_local_resource_mean(
            new_state['resources'], new_state['positions'], new_state['alive'], cfg
        )
        post_neighbor_count = compute_local_neighbor_count(
            post_agent_count, new_state['positions'], new_state['alive'], cfg
        )

        # Build 3 actual targets
        actual_targets = jnp.stack([
            new_state['energy'] - pre_energy,                          # T0: energy delta
            post_resource_mean - pre_resource_mean,                    # T1: resource delta
            (post_neighbor_count - pre_neighbor_count) / neighbor_scale,  # T2: neighbor delta (scaled)
        ], axis=1)  # (M, 3)

        # Per-target MSE for metrics
        alive_f = pre_alive.astype(jnp.float32)
        n_alive = jnp.maximum(jnp.sum(alive_f), 1.0)
        pred_errors = (predictions - actual_targets) ** 2  # (M, 3)
        pred_mse_per_target = jnp.sum(pred_errors * alive_f[:, None], axis=0) / n_alive  # (3,)
        pred_mse_total = jnp.sum(pred_mse_per_target)

        # Compute gradient of multi-target prediction error wrt phenotypes
        grads = _prediction_grad_batch(
            pre_phenotypes, obs, pre_hidden, pre_sync, cfg, actual_targets
        )

        # Clip gradients
        grad_norm = jnp.sqrt(jnp.sum(grads ** 2, axis=1, keepdims=True) + 1e-8)
        max_norm = 1.0
        grads = grads * jnp.minimum(1.0, max_norm / grad_norm)

        # SGD update (only for alive agents)
        lr = extract_lr_jax(pre_phenotypes, cfg)
        updated_phenotypes = pre_phenotypes - lr[:, None] * grads * alive_f[:, None]

        # Apply gradient update conditionally
        do_update = (step_idx % cfg['grad_every'] == 0)
        new_phenotypes = jnp.where(do_update, updated_phenotypes, pre_phenotypes)
        new_state = {**new_state, 'phenotypes': new_phenotypes}

        # Divergence metric
        h_before_norm = jnp.sqrt(jnp.sum(pre_hidden ** 2, axis=1) + 1e-8)
        h_diff_norm = jnp.sqrt(jnp.sum((new_state['hidden'] - pre_hidden) ** 2, axis=1))
        div = h_diff_norm / h_before_norm
        mean_div = jnp.sum(div * alive_f) / n_alive

        metrics = {
            'n_alive': jnp.sum(new_state['alive']).astype(jnp.float32),
            'mean_energy': jnp.mean(new_state['energy'] * new_state['alive']),
            'mean_divergence': mean_div,
            'pred_mse_total': pred_mse_total,
            'pred_mse_energy': pred_mse_per_target[0],
            'pred_mse_resource': pred_mse_per_target[1],
            'pred_mse_neighbor': pred_mse_per_target[2],
            'mean_grad_norm': jnp.sum(grad_norm.squeeze() * alive_f) / n_alive,
        }
        return new_state, metrics

    @jax.jit
    def run_chunk(state):
        """Run chunk_size steps with multi-target gradient learning."""
        final_state, metrics = lax.scan(
            scan_body, state, jnp.arange(cfg['chunk_size'])
        )
        return final_state, metrics

    return run_chunk


# ---------------------------------------------------------------------------
# Param extraction utilities
# ---------------------------------------------------------------------------

def _param_offset(cfg, target_name):
    """Compute byte offset for a named parameter in the flat vector."""
    shapes = _param_shapes(cfg)
    offset = 0
    for name, shape in shapes.items():
        if name == target_name:
            return offset
        offset += int(np.prod(shape))
    raise ValueError("Unknown param: %s" % target_name)


def extract_tick_weights_np(params, cfg):
    """Extract tick_weights from (M, P) params array. Returns (M, K) softmax."""
    offset = _param_offset(cfg, 'tick_weights')
    K = cfg['K_max']
    raw = np.array(params)[:, offset:offset + K]
    raw_shifted = raw - np.max(raw, axis=1, keepdims=True)
    exp_raw = np.exp(raw_shifted)
    return exp_raw / np.sum(exp_raw, axis=1, keepdims=True)


def extract_sync_decay_np(params, cfg):
    """Extract sync decay from (M, P). Returns (M,) in [0.5, 0.999]."""
    offset = _param_offset(cfg, 'sync_decay_raw')
    raw = np.array(params)[:, offset]
    return 0.5 + 0.499 / (1.0 + np.exp(-raw))


def extract_predict_weights_np(params, cfg):
    """Extract multi-target prediction head weights.
    Returns: W (M, H, n_targets), b (M, n_targets)
    """
    H = cfg['hidden_dim']
    T = cfg['n_targets']
    offset_W = _param_offset(cfg, 'predict_W')
    offset_b = _param_offset(cfg, 'predict_b')
    W = np.array(params)[:, offset_W:offset_W + H * T].reshape(-1, H, T)
    b = np.array(params)[:, offset_b:offset_b + T]
    return W, b


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_v23(seed, cfg):
    """Initialize V23 state with genome/phenotype distinction."""
    key = jax.random.PRNGKey(seed)
    N = cfg['N']
    M = cfg['M_max']
    H = cfg['hidden_dim']
    P = cfg['n_params']

    key, k1, k2, k3, k4 = jax.random.split(key, 5)

    # Environment
    resources = jax.random.uniform(k1, (N, N)) * cfg['initial_resource']
    resources = resources + jax.random.uniform(k2, (N, N)) * 0.3
    resources = jnp.clip(resources, 0.0, 1.0)
    signals = jnp.zeros((N, N))

    # Agent positions
    n_initial = M // 4
    positions_init = jax.random.randint(k3, (n_initial, 2), 0, N)
    positions = jnp.zeros((M, 2), dtype=jnp.int32)
    positions = positions.at[:n_initial].set(positions_init)

    # Hidden states
    hidden = jnp.zeros((M, H))

    # Sync matrices
    sync_matrices = jnp.zeros((M, H, H))

    # Energy
    energy = jnp.zeros(M)
    energy = energy.at[:n_initial].set(cfg['initial_energy'])

    # Alive mask
    alive = jnp.zeros(M, dtype=jnp.bool_)
    alive = alive.at[:n_initial].set(True)

    # Genome and phenotype (start identical)
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


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

def extract_snapshot(state, cycle, cfg):
    """Extract lightweight snapshot for offline measurement."""
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
