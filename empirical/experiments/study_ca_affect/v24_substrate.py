"""V24: TD Value Learning — Within-Lifetime Temporal Difference

V22 showed: scalar next-step prediction is orthogonal to integration.
V23 showed: multi-target prediction creates specialization that DECREASES Phi.

Both are reactive — associations from present state, decomposable by channel.
Understanding requires associations from the possibility landscape: what could
happen over many future steps, integrated across self + environment + social.

V24 tests whether LONG-HORIZON prediction, via temporal difference learning,
forces the non-decomposable representation that creates integration.

Instead of predicting energy delta (1-step, decomposable):
  pred = h @ W + b  →  target = energy[t+1] - energy[t]

V24 predicts STATE VALUE (multi-step, non-decomposable):
  V(s_t) = h @ W + b  →  target = r_t + gamma * V(s_{t+1})

The value function integrates over ALL possible futures:
  V(s) = E[sum_t gamma^t r_t | s_0 = s]

This is inherently non-decomposable because future outcomes depend on
interactions between all state features — energy, resources, neighbors,
actions, timing. A single scalar output prevents column decomposition.
TD bootstrapping creates a temporal chain of credit assignment that
forces the hidden state to encode extended context.

The key insight (from V22+V23): the TIME HORIZON of prediction matters.
- 1-step prediction (V22): one hidden unit suffices
- Multi-target 1-step (V23): separate columns specialize
- Multi-step value (V24): must represent conjunctive features across time

Pre-registered predictions:
  1. V(s) correlates with actual survival (value function is meaningful)
  2. Phi improves over V22/V23 (long-horizon value requires non-decomposable features)
  3. Robustness > 1.0 (better value estimates improve stress survival)
  4. gamma does NOT evolve to 0 (long horizon is adaptive, not suppressed)
  5. V(s) shows distinct dynamics during drought vs normal cycles

Falsification criteria:
  - V(s) uncorrelated with survival → value function is noise
  - Phi/robustness no better than V22 → time horizon doesn't matter
  - gamma → 0 → agents reject long-horizon prediction
  - TD error doesn't decrease within lifetime → learning not working

Architecture (over V21):
  - predict_W (H, 1): state value prediction head
  - predict_b (1,): bias
  - lr_raw (1,): evolvable learning rate, sigmoid → [1e-5, 1e-2]
  - gamma_raw (1,): evolvable discount factor, sigmoid → [0.5, 0.999]
  - 20 new params (4,060 total vs V22's 4,058)
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

def generate_v24_config(**kwargs):
    """Generate V24 configuration with TD value learning."""
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

        # Environment dynamics
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

        # V24: gradient update frequency
        'grad_every': 1,

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
        # V24 TD value params
        'predict_W': (H, 1),
        'predict_b': (1,),
        'lr_raw': (1,),
        'gamma_raw': (1,),
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
# Agent multi-tick forward pass with value prediction
# ---------------------------------------------------------------------------

def agent_multi_tick_v24(obs_flat, hidden, sync_matrix, params_flat, cfg):
    """Run K_max GRU ticks, returning state value V(s).

    Returns:
        final_hidden: (H,)
        final_sync: (H, H)
        actions: (n_actions,) weighted mix of tick outputs
        value: scalar, predicted state value V(s)
    """
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

    # State value prediction
    value = (final_hidden @ p['predict_W']).squeeze() + p['predict_b'][0]

    return final_hidden, final_sync, actions, value


agent_multi_tick_v24_batch = jax.vmap(
    agent_multi_tick_v24, in_axes=(0, 0, 0, 0, None)
)


# ---------------------------------------------------------------------------
# TD loss (single agent) — for gradient computation
# ---------------------------------------------------------------------------

def _td_loss(phenotype_flat, obs_flat, hidden, sync_matrix, cfg, td_target):
    """Scalar loss: (V(s_t) - td_target)^2.

    td_target = r_t + gamma * stop_gradient(V(s_{t+1}))
    Gradient flows through V(s_t) only (semi-gradient TD).
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

    V_t = (final_hidden @ p['predict_W']).squeeze() + p['predict_b'][0]
    return (V_t - td_target) ** 2


_td_grad_single = jax.grad(_td_loss, argnums=0)
_td_grad_batch = jax.vmap(_td_grad_single, in_axes=(0, 0, 0, 0, None, 0))


# ---------------------------------------------------------------------------
# Parameter extraction utilities
# ---------------------------------------------------------------------------

def _param_offset(cfg, target_name):
    shapes = _param_shapes(cfg)
    offset = 0
    for name, shape in shapes.items():
        if name == target_name:
            return offset
        offset += int(np.prod(shape))
    raise ValueError("Unknown param: %s" % target_name)


def extract_lr_jax(phenotypes, cfg):
    """Extract learning rates. Returns (M,) in [1e-5, 1e-2]."""
    offset = _param_offset(cfg, 'lr_raw')
    raw = phenotypes[:, offset]
    return 1e-5 + (1e-2 - 1e-5) * jax.nn.sigmoid(raw)


def extract_gamma_jax(phenotypes, cfg):
    """Extract discount factors. Returns (M,) in [0.5, 0.999]."""
    offset = _param_offset(cfg, 'gamma_raw')
    raw = phenotypes[:, offset]
    return 0.5 + 0.499 * jax.nn.sigmoid(raw)


def extract_value_jax(hidden, phenotypes, cfg):
    """Compute V(s) from hidden state and phenotype. Returns (M,)."""
    H = cfg['hidden_dim']
    offset_W = _param_offset(cfg, 'predict_W')
    offset_b = _param_offset(cfg, 'predict_b')
    W = phenotypes[:, offset_W:offset_W + H]  # (M, H)
    b = phenotypes[:, offset_b]  # (M,)
    return jnp.sum(hidden * W, axis=1) + b  # (M,)


def extract_lr_np(params, cfg):
    offset = _param_offset(cfg, 'lr_raw')
    raw = np.array(params)[:, offset]
    return 1e-5 + (1e-2 - 1e-5) / (1.0 + np.exp(-raw))


def extract_gamma_np(params, cfg):
    offset = _param_offset(cfg, 'gamma_raw')
    raw = np.array(params)[:, offset]
    return 0.5 + 0.499 / (1.0 + np.exp(-raw))


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
# V24 environment step
# ---------------------------------------------------------------------------

def _env_step_v24_inner(state, cfg):
    """V24 env step: multi-tick with phenotypes, returns value prediction."""
    N = cfg['N']

    agent_count = build_agent_count_grid(state['positions'], state['alive'], N)
    obs = build_observation_batch(
        state['positions'], state['resources'], state['signals'],
        agent_count, state['energy'], cfg
    )

    new_hidden, new_sync, raw_actions, values = agent_multi_tick_v24_batch(
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
    return new_state, obs, values


# ---------------------------------------------------------------------------
# Chunk runner with TD learning
# ---------------------------------------------------------------------------

def make_v24_chunk_runner(cfg):
    """Create JIT-compiled chunk runner with TD value learning.

    Each step:
      1. Compute V(s_t) from pre-step hidden state
      2. Forward pass → actions + value prediction
      3. Execute env dynamics
      4. Compute V(s_{t+1}) from post-step hidden (stop gradient)
      5. TD target = r_t + gamma * V(s_{t+1})
      6. Gradient of TD error through K ticks
      7. SGD update on phenotype
    """
    def scan_body(state, step_idx):
        pre_energy = state['energy']
        pre_hidden = state['hidden']
        pre_sync = state['sync_matrices']
        pre_alive = state['alive']
        pre_phenotypes = state['phenotypes']

        # Forward pass + env dynamics
        new_state, obs, values_t = _env_step_v24_inner(state, cfg)

        # Reward = energy delta
        reward = new_state['energy'] - pre_energy  # (M,)

        # V(s_{t+1}) from post-step hidden state (stop gradient for semi-gradient TD)
        V_next = jax.lax.stop_gradient(
            extract_value_jax(new_state['hidden'], pre_phenotypes, cfg)
        )

        # Evolvable discount factor
        gamma = extract_gamma_jax(pre_phenotypes, cfg)  # (M,)

        # TD target
        td_target = reward + gamma * V_next * new_state['alive'].astype(jnp.float32)
        # Dead agents: V(s_{t+1}) = 0 (terminal state)

        # TD error for metrics
        alive_f = pre_alive.astype(jnp.float32)
        n_alive = jnp.maximum(jnp.sum(alive_f), 1.0)
        td_error = (values_t - td_target) ** 2
        mean_td_error = jnp.sum(td_error * alive_f) / n_alive
        mean_value = jnp.sum(values_t * alive_f) / n_alive

        # Compute gradient of TD loss w.r.t. phenotypes
        grads = _td_grad_batch(
            pre_phenotypes, obs, pre_hidden, pre_sync, cfg, td_target
        )

        # Clip gradients
        grad_norm = jnp.sqrt(jnp.sum(grads ** 2, axis=1, keepdims=True) + 1e-8)
        max_norm = 1.0
        grads = grads * jnp.minimum(1.0, max_norm / grad_norm)

        # SGD update
        lr = extract_lr_jax(pre_phenotypes, cfg)
        updated_phenotypes = pre_phenotypes - lr[:, None] * grads * alive_f[:, None]

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
            'mean_td_error': mean_td_error,
            'mean_value': mean_value,
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
# Initialization
# ---------------------------------------------------------------------------

def init_v24(seed, cfg):
    """Initialize V24 state with genome/phenotype distinction."""
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


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

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
