"""V28: Bottleneck Width Sweep — Testing the Gradient Coupling Mechanism

V27 seed comparison revealed: the MLP prediction head operates in the LINEAR
regime (zero tanh saturation, mean |z| = 0.19-0.28). The "nonlinear coupling"
hypothesis is NOT the mechanism.

New hypothesis: the mechanism is the INFORMATION BOTTLENECK.
- V22 (linear): ∂L/∂h_i = 2(pred-target) * W_i → independent per unit
- V27 (2-layer): ∂L/∂h = 2(pred-target) * W2.T @ diag(dtanh) @ W1.T
  → ALL 16 hidden units coupled through 8-dim bottleneck
  Even with dtanh ≈ 1 (linear regime), the W1 matrix creates cross-talk

V28 tests this with a 3-condition sweep:
  A) Linear activation, w=8:  Same bottleneck as V27, no tanh at all
     → If Φ matches V27: nonlinearity is irrelevant, bottleneck is the mechanism
  B) Tanh activation, w=4:   Narrower bottleneck (more coupling)
     → If Φ > V27: narrower bottleneck = more coupling = more Φ
  C) Tanh activation, w=16:  No bottleneck (width = hidden dim)
     → If Φ drops to V22: bottleneck is necessary

Pre-registered predictions:
  P1: Condition A (linear w=8) produces similar Φ to V27 (tanh w=8)
  P2: Condition B (tanh w=4) produces Φ ≥ V27 (more coupling)
  P3: Condition C (tanh w=16) produces Φ ≈ V22 baseline (~0.097)

Falsification:
  - P1 fails (linear << V27): even slight nonlinearity matters at |z|~0.2
  - P3 fails (w=16 matches V27): bottleneck is NOT the mechanism
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
import numpy as np

# Import shared environment dynamics
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

def generate_v28_config(**kwargs):
    """Generate V28 configuration with configurable prediction head."""
    cfg = {
        # Grid
        'N': 128,
        'M_max': 256,

        # Agent architecture
        'obs_radius': 2,
        'embed_dim': 24,
        'hidden_dim': 16,
        'n_actions': 7,

        # CTM inner ticks
        'K_max': 8,

        # Prediction head (configurable)
        'predict_hidden': 8,        # bottleneck width
        'predict_activation': 'tanh',  # 'tanh' or 'linear'

        # Environment dynamics (same as V20b/V21/V22/V27)
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

        # Run
        'chunk_size': 50,
        'steps_per_cycle': 5000,
        'n_cycles': 30,
    }
    cfg.update(kwargs)

    # Derived dimensions
    obs_side = 2 * cfg['obs_radius'] + 1
    obs_flat = obs_side * obs_side * 3 + 1
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
# Forward pass — parameterized activation
# ---------------------------------------------------------------------------

def _make_agent_multi_tick(cfg):
    """Create agent_multi_tick function with the configured activation.

    Returns a function suitable for vmap.
    Activation choice is baked in at function creation time (static).
    """
    use_tanh = (cfg['predict_activation'] == 'tanh')

    def agent_multi_tick(obs_flat, hidden, sync_matrix, params_flat, cfg_static):
        p = unpack_params(params_flat, cfg_static)
        K = cfg_static['K_max']

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

        # Prediction head — activation choice is static
        pre_act = final_hidden @ p['predict_W1'] + p['predict_b1']
        if use_tanh:
            mlp_hidden = jnp.tanh(pre_act)
        else:
            mlp_hidden = pre_act  # linear (identity activation)
        prediction = (mlp_hidden @ p['predict_W2']).squeeze() + p['predict_b2'][0]

        return final_hidden, final_sync, actions, prediction

    return agent_multi_tick


def _make_prediction_loss(cfg):
    """Create prediction loss function with configured activation."""
    use_tanh = (cfg['predict_activation'] == 'tanh')

    def prediction_loss(phenotype_flat, obs_flat, hidden, sync_matrix, cfg_static, actual_delta):
        p = unpack_params(phenotype_flat, cfg_static)
        K = cfg_static['K_max']

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

        # Prediction with configured activation
        pre_act = final_hidden @ p['predict_W1'] + p['predict_b1']
        if use_tanh:
            mlp_hidden = jnp.tanh(pre_act)
        else:
            mlp_hidden = pre_act
        pred = (mlp_hidden @ p['predict_W2']).squeeze() + p['predict_b2'][0]
        return (pred - actual_delta) ** 2

    return prediction_loss


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
# V28 environment step (parameterized)
# ---------------------------------------------------------------------------

def _make_env_step(cfg, agent_multi_tick_batch):
    """Create env step function using the configured agent forward pass."""
    def env_step_inner(state, cfg_static):
        N = cfg_static['N']
        agent_count = build_agent_count_grid(state['positions'], state['alive'], N)
        obs = build_observation_batch(
            state['positions'], state['resources'], state['signals'],
            agent_count, state['energy'], cfg_static
        )

        new_hidden, new_sync, raw_actions, predictions = agent_multi_tick_batch(
            obs, state['hidden'], state['sync_matrices'], state['phenotypes'], cfg_static
        )
        new_hidden = new_hidden * state['alive'][:, None]
        new_sync = new_sync * state['alive'][:, None, None]

        move_idx, consume, emit = decode_actions(raw_actions, cfg_static)
        deltas = MOVE_DELTAS[move_idx]
        new_positions = (state['positions'] + deltas) % N
        new_positions = jnp.where(state['alive'][:, None], new_positions, state['positions'])

        resources, actual_consumed = apply_consumption(
            state['resources'], new_positions, consume * state['alive'], state['alive'], N
        )
        signals = apply_emissions(state['signals'], new_positions, emit, state['alive'], N)
        signals = diffuse_signals(signals, cfg_static)
        resources = regen_resources(resources, state['regen_rate'])

        energy = state['energy'] - cfg_static['metabolic_cost'] + actual_consumed * cfg_static['resource_value']
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
        return new_state, obs, predictions
    return env_step_inner


# ---------------------------------------------------------------------------
# Chunk runner (creates JIT-compiled functions for specific config)
# ---------------------------------------------------------------------------

def make_v28_chunk_runner(cfg):
    """Create JIT-compiled chunk runner for the specified config."""
    # Build activation-specific functions
    agent_fn = _make_agent_multi_tick(cfg)
    agent_batch = jax.vmap(agent_fn, in_axes=(0, 0, 0, 0, None))
    env_step = _make_env_step(cfg, agent_batch)

    loss_fn = _make_prediction_loss(cfg)
    grad_fn = jax.grad(loss_fn, argnums=0)
    grad_batch = jax.vmap(grad_fn, in_axes=(0, 0, 0, 0, None, 0))

    def scan_body(state, step_idx):
        pre_energy = state['energy']
        pre_hidden = state['hidden']
        pre_sync = state['sync_matrices']
        pre_alive = state['alive']
        pre_phenotypes = state['phenotypes']

        new_state, obs, predictions = env_step(state, cfg)

        actual_delta = new_state['energy'] - pre_energy
        alive_f = pre_alive.astype(jnp.float32)
        n_alive = jnp.maximum(jnp.sum(alive_f), 1.0)
        pred_mse = jnp.sum((predictions - actual_delta) ** 2 * alive_f) / n_alive

        grads = grad_batch(pre_phenotypes, obs, pre_hidden, pre_sync, cfg, actual_delta)

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
# Utility extractors (numpy, for analysis)
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
# Initialization
# ---------------------------------------------------------------------------

def init_v28(seed, cfg):
    """Initialize V28 state."""
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
