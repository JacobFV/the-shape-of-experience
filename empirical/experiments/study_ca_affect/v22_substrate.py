"""V22: Intrinsic Predictive Gradient — Protocell Agency with Within-Lifetime Learning

V21 showed: architecture for internal deliberation works (ticks don't collapse),
but evolution alone is too slow to shape tick usage in 30 generations.
Dense gradient signal is needed — each internal tick must receive feedback
about its contribution to prediction accuracy.

V22 adds within-lifetime SGD driven by intrinsic prediction error:
  1. Each agent predicts its own energy delta through its internal ticks
  2. After each step, actual energy delta is observed
  3. Prediction error → gradient through all K ticks → phenotype SGD update
  4. Genome (evolved) vs phenotype (genome + accumulated SGD updates)

This is the computational equivalent of the free energy principle: minimize
surprise about your own persistence using only thermodynamic self-interest.

The learning signal is purely "how wrong was I about my own energy change?"
— the most primitive prediction error possible. No task specification,
no affect labels, no human data. STILL UNCONTAMINATED.

Pre-registered predictions:
  1. C_wm develops faster than V21 (world model from gradient, not just evolution)
  2. Prediction MSE decreases within each lifetime (gradient is learning)
  3. Agents do NOT evolve lr → 0 (gradient signal helps fitness)
  4. Better drought survival than V21 (within-lifetime adaptation)
  5. Structured tick usage: later ticks contribute more to prediction than V21

Falsification criteria:
  - Prediction MSE does NOT decrease within lifetime → gradient not learning
  - C_wm no better than V21 → intrinsic gradient doesn't help
  - Agents evolve lr → 0 (suppress learning) → gradient hurts fitness
  - Phenotype drift destabilizes performance vs V21

Architecture (over V21):
  - predict_W (H, 1), predict_b (1,): energy delta prediction head
  - lr_raw (1,): evolvable learning rate, sigmoid → [1e-5, 1e-2]
  - genome/phenotype distinction: phenotype = genome + accumulated SGD updates
  - 19 new params (4,059 total vs V21's 4,040)
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

def generate_v22_config(**kwargs):
    """Generate V22 configuration with prediction gradient."""
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

        # Environment dynamics (same as V20b/V21)
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
        'lamarckian': False,  # If True, inherit phenotype; if False, inherit genome only

        # V22: gradient update frequency
        'grad_every': 1,  # SGD update every N env steps (1 = every step)

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
        # V22 prediction params
        'predict_W': (H, 1),
        'predict_b': (1,),
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
# Agent multi-tick forward pass with prediction (single agent)
# ---------------------------------------------------------------------------

def agent_multi_tick_v22(obs_flat, hidden, sync_matrix, params_flat, cfg):
    """Run K_max GRU ticks for one agent, returning prediction.

    Same as V21's agent_multi_tick but also returns energy delta prediction
    from the final tick's hidden state.

    Returns:
        final_hidden: (H,)
        final_sync: (H, H)
        actions: (n_actions,) weighted mix of tick outputs
        prediction: scalar, predicted energy delta
    """
    p = unpack_params(params_flat, cfg)
    K = cfg['K_max']

    # Pre-compute external embedding (tick 0 only)
    x_ext = jnp.tanh(obs_flat @ p['embed_W'] + p['embed_b'])  # (E,)

    # Sync decay: sigmoid mapped to [0.5, 0.999]
    sync_decay = 0.5 + 0.499 * jax.nn.sigmoid(p['sync_decay_raw'][0])

    def tick_fn(carry, tick_idx):
        h, S = carry

        # Internal input from sync summary (ticks 1+)
        sync_sum = compute_sync_summary(S)  # (3,)
        x_int = jnp.tanh(sync_sum @ p['internal_embed_W'] + p['internal_embed_b'])

        # Select: tick 0 = external, tick 1+ = internal
        x = jnp.where(tick_idx == 0, x_ext, x_int)

        # GRU step (shared weights across all ticks)
        xh = jnp.concatenate([x, h])
        z = jax.nn.sigmoid(xh @ p['gru_Wz'] + p['gru_bz'])
        r = jax.nn.sigmoid(xh @ p['gru_Wr'] + p['gru_br'])
        h_tilde = jnp.tanh(
            jnp.concatenate([x, r * h]) @ p['gru_Wh'] + p['gru_bh']
        )
        new_h = z * h + (1.0 - z) * h_tilde

        # Update sync matrix
        new_S = sync_decay * S + jnp.outer(new_h, new_h)

        # Output logits for this tick
        output = new_h @ p['out_W'] + p['out_b']  # (O,)

        return (new_h, new_S), output

    # Run K ticks
    (final_hidden, final_sync), all_outputs = lax.scan(
        tick_fn, (hidden, sync_matrix), jnp.arange(K)
    )

    # Weighted mix of tick outputs
    weights = jax.nn.softmax(p['tick_weights'])
    actions = jnp.einsum('k,ko->o', weights, all_outputs)

    # Energy delta prediction from final hidden state
    prediction = (final_hidden @ p['predict_W']).squeeze() + p['predict_b'][0]

    return final_hidden, final_sync, actions, prediction


# Vectorize over agents
agent_multi_tick_v22_batch = jax.vmap(
    agent_multi_tick_v22, in_axes=(0, 0, 0, 0, None)
)


# ---------------------------------------------------------------------------
# Prediction gradient (single agent)
# ---------------------------------------------------------------------------

def _prediction_loss(phenotype_flat, obs_flat, hidden, sync_matrix, cfg, actual_delta):
    """Scalar loss: (predicted_delta - actual_delta)^2.

    Gradient of this through all K ticks of the multi-tick computation
    provides the learning signal for within-lifetime SGD.
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

    pred = (final_hidden @ p['predict_W']).squeeze() + p['predict_b'][0]
    return (pred - actual_delta) ** 2


# Gradient of prediction loss wrt phenotype params (single agent)
_prediction_grad_single = jax.grad(_prediction_loss, argnums=0)

# Vectorize over agents
_prediction_grad_batch = jax.vmap(
    _prediction_grad_single, in_axes=(0, 0, 0, 0, None, 0)
)


# ---------------------------------------------------------------------------
# Learning rate extraction (JAX-compatible)
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
    """Extract learning rates from phenotype vectors.

    Returns: (M,) learning rates in [1e-5, 1e-2].
    """
    offset = _lr_offset(cfg)
    raw = phenotypes[:, offset]  # (M,)
    # sigmoid mapped to [1e-5, 1e-2]
    return 1e-5 + (1e-2 - 1e-5) * jax.nn.sigmoid(raw)


def extract_lr_np(params, cfg):
    """Numpy version for analysis."""
    offset = _lr_offset(cfg)
    raw = np.array(params)[:, offset]
    return 1e-5 + (1e-2 - 1e-5) / (1.0 + np.exp(-raw))


# ---------------------------------------------------------------------------
# V22 environment step (not JIT'd — called inside chunk runner)
# ---------------------------------------------------------------------------

def _env_step_v22_inner(state, cfg):
    """V22 env step: multi-tick with phenotypes, returns prediction.

    NOT separately JIT'd — called inside the chunk runner's lax.scan.
    """
    N = cfg['N']

    # 1. Build agent count grid
    agent_count = build_agent_count_grid(state['positions'], state['alive'], N)

    # 2. Build observations
    obs = build_observation_batch(
        state['positions'], state['resources'], state['signals'],
        agent_count, state['energy'], cfg
    )

    # 3. Multi-tick forward pass using PHENOTYPES
    new_hidden, new_sync, raw_actions, predictions = agent_multi_tick_v22_batch(
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
        'phenotypes': state['phenotypes'],  # Updated in chunk runner, not here
        'genomes': state['genomes'],
        'sync_matrices': new_sync,
        'regen_rate': state['regen_rate'],
        'step_count': state['step_count'] + 1,
    }
    return new_state, obs, predictions


# ---------------------------------------------------------------------------
# Chunk runner with SGD (lax.scan)
# ---------------------------------------------------------------------------

def make_v22_chunk_runner(cfg):
    """Create JIT-compiled chunk runner with within-lifetime gradient learning.

    Each step:
      1. Forward pass → actions + energy prediction
      2. Execute env dynamics
      3. Observe actual energy delta
      4. Gradient of prediction error through K ticks
      5. SGD update on phenotype
    """
    def scan_body(state, step_idx):
        # Save pre-step values
        pre_energy = state['energy']
        pre_hidden = state['hidden']
        pre_sync = state['sync_matrices']
        pre_alive = state['alive']
        pre_phenotypes = state['phenotypes']

        # Forward pass + env dynamics
        new_state, obs, predictions = _env_step_v22_inner(state, cfg)

        # Actual energy delta (target for prediction)
        actual_delta = new_state['energy'] - pre_energy  # (M,)

        # Prediction MSE for metrics
        alive_f = pre_alive.astype(jnp.float32)
        n_alive = jnp.maximum(jnp.sum(alive_f), 1.0)
        pred_mse = jnp.sum((predictions - actual_delta) ** 2 * alive_f) / n_alive

        # Compute gradient of prediction error wrt phenotypes
        # Uses pre-step obs, hidden, sync (same inputs as the forward pass)
        grads = _prediction_grad_batch(
            pre_phenotypes, obs, pre_hidden, pre_sync, cfg, actual_delta
        )

        # Clip gradients to prevent explosion
        grad_norm = jnp.sqrt(jnp.sum(grads ** 2, axis=1, keepdims=True) + 1e-8)
        max_norm = 1.0
        grads = grads * jnp.minimum(1.0, max_norm / grad_norm)

        # SGD update (only for alive agents)
        lr = extract_lr_jax(pre_phenotypes, cfg)  # (M,)
        updated_phenotypes = pre_phenotypes - lr[:, None] * grads * alive_f[:, None]

        # Apply gradient update conditionally based on grad_every
        do_update = (step_idx % cfg['grad_every'] == 0)
        new_phenotypes = jnp.where(do_update, updated_phenotypes, pre_phenotypes)
        new_state = {**new_state, 'phenotypes': new_phenotypes}

        # Divergence metric (same as V21)
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
        """Run chunk_size steps with gradient learning."""
        final_state, metrics = lax.scan(
            scan_body, state, jnp.arange(cfg['chunk_size'])
        )
        return final_state, metrics

    return run_chunk


# ---------------------------------------------------------------------------
# Tick weight and sync decay extraction (reuse V21 utilities)
# ---------------------------------------------------------------------------

def _param_offset(cfg, target_name):
    """Compute byte offset for a named parameter in the flat vector."""
    shapes = _param_shapes(cfg)
    offset = 0
    for name, shape in shapes.items():
        if name == target_name:
            return offset
        offset += int(np.prod(shape))
    raise ValueError(f"Unknown param: {target_name}")


def extract_tick_weights_np(params, cfg):
    """Extract tick_weights from (M, P) params array. Returns (M, K) softmax-normalized."""
    offset = _param_offset(cfg, 'tick_weights')
    K = cfg['K_max']
    raw = np.array(params)[:, offset:offset + K]
    raw_shifted = raw - np.max(raw, axis=1, keepdims=True)
    exp_raw = np.exp(raw_shifted)
    return exp_raw / np.sum(exp_raw, axis=1, keepdims=True)


def extract_sync_decay_np(params, cfg):
    """Extract sync decay from (M, P) params array. Returns (M,) in [0.5, 0.999]."""
    offset = _param_offset(cfg, 'sync_decay_raw')
    raw = np.array(params)[:, offset]
    return 0.5 + 0.499 / (1.0 + np.exp(-raw))


def extract_predict_weights_np(params, cfg):
    """Extract prediction head weights from (M, P) params array."""
    H = cfg['hidden_dim']
    offset_W = _param_offset(cfg, 'predict_W')
    offset_b = _param_offset(cfg, 'predict_b')
    W = np.array(params)[:, offset_W:offset_W + H]  # (M, H)
    b = np.array(params)[:, offset_b]  # (M,)
    return W, b


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_v22(seed, cfg):
    """Initialize V22 state with genome/phenotype distinction.

    Both genome and phenotype start as the same random initialization.
    Phenotype gets SGD updates during each cycle; genome is only
    modified by evolutionary selection.
    """
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
