"""V33: Contrastive Self-Prediction — Forcing Counterfactual Representation

V22-V31 showed: prediction target doesn't matter (self vs social → same
Φ distribution). But ALL targets were FACTUAL predictions: "what WILL
happen given what I ACTUALLY do." The gradient rewards representations
that map present-state → future-state, decomposable by channel.

V33 forces COUNTERFACTUAL prediction: "how would my outcome DIFFER if
I'd done X instead of Y?" This is inherently non-decomposable because:
  1. You must represent action-contingent futures (rung 8: CF weight)
  2. The difference between two futures requires computing both
  3. A single readout channel cannot capture the comparison

Architecture:
  - Same V27 base (GRU + 8 ticks + MLP self-prediction)
  - After the forward pass produces action a_actual:
    1. Sample alternative action a_alt uniformly
    2. Run a LIGHTWEIGHT forward model predicting:
       - energy_delta_actual (from a_actual)
       - energy_delta_alt (from a_alt)
    3. Loss = (predicted_diff - actual_diff)² where
       diff = energy_delta_actual - energy_delta_alt
  - The forward model shares the GRU hidden state but uses
    a separate MLP head that takes (hidden, action_onehot) as input

Key insight: predicting the DIFFERENCE is harder than predicting either
outcome alone. The representation must encode how the world responds
DIFFERENTLY to different actions — this is counterfactual reasoning.

Pre-registered predictions:
  P1: Mean Φ > V27's 0.090 (counterfactual forces non-decomposable repr)
  P2: HIGH fraction > 30% (if counterfactual is the missing ingredient)
  P3: Prediction accuracy improves within lifetime (gradient works)
  P4: Action-conditional hidden state divergence increases over evolution
      (agents learn that actions MATTER)
  P5: CF weight (defined as variance explained by action choice in
      hidden state) increases over evolution

Falsification:
  - P1+P2 both fail: counterfactual reasoning is not the missing ingredient
  - Agents evolve to ignore the alt action (collapse to V27): CF too hard
  - Gradient instability from dual-path computation
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

def generate_v33_config(**kwargs):
    """Generate V33 configuration with contrastive prediction."""
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

        # Contrastive prediction head
        'predict_hidden': 8,   # MLP bottleneck

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

        # Learning
        'lamarckian': False,
        'grad_every': 1,

        # Run
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
    """Ordered dict of parameter name -> shape.

    V33 adds a contrastive prediction head that takes (hidden, action_onehot)
    and predicts energy delta. The contrastive loss compares predictions
    for actual vs alternative actions.
    """
    obs_flat = cfg['obs_flat']
    E = cfg['embed_dim']
    H = cfg['hidden_dim']
    O = cfg['n_actions']
    K = cfg['K_max']
    PH = cfg['predict_hidden']
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
        # V33 contrastive prediction head
        # Input: hidden (H) + action_onehot (O) = H+O
        # Two-layer MLP: (H+O) -> PH -> 1
        'cf_W1': (H + O, PH),
        'cf_b1': (PH,),
        'cf_W2': (PH, 1),
        'cf_b2': (1,),
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
# Agent forward pass
# ---------------------------------------------------------------------------

def agent_multi_tick_v33(obs_flat, hidden, sync_matrix, params_flat, cfg):
    """Run K_max GRU ticks, return actions + contrastive prediction inputs."""
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

    return final_hidden, final_sync, actions


agent_multi_tick_v33_batch = jax.vmap(
    agent_multi_tick_v33, in_axes=(0, 0, 0, 0, None)
)


def predict_energy_for_action(hidden, action_idx, params_flat, cfg):
    """Predict energy delta for a specific action using contrastive head.

    Input: hidden (H,) + one_hot(action_idx, O) -> MLP -> scalar
    """
    p = unpack_params(params_flat, cfg)
    O = cfg['n_actions']
    action_onehot = jax.nn.one_hot(action_idx, O)
    inp = jnp.concatenate([hidden, action_onehot])  # (H+O,)
    mlp_h = jnp.tanh(inp @ p['cf_W1'] + p['cf_b1'])  # (PH,)
    pred = (mlp_h @ p['cf_W2']).squeeze() + p['cf_b2'][0]
    return pred


predict_energy_for_action_batch = jax.vmap(
    predict_energy_for_action, in_axes=(0, 0, 0, None)
)


# ---------------------------------------------------------------------------
# Contrastive prediction loss
# ---------------------------------------------------------------------------

def _contrastive_loss(phenotype_flat, obs_flat, hidden, sync_matrix, cfg,
                      actual_action_idx, alt_action_idx,
                      actual_delta, alt_delta):
    """Contrastive loss: predict the DIFFERENCE in energy delta
    between actual and alternative actions.

    loss = (predicted_diff - actual_diff)²
    where diff = energy_delta(actual) - energy_delta(alt)
    """
    p = unpack_params(phenotype_flat, cfg)
    K = cfg['K_max']

    # Re-run forward pass to get final hidden
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

    O = cfg['n_actions']

    # Predict energy delta for actual action
    actual_oh = jax.nn.one_hot(actual_action_idx, O)
    inp_actual = jnp.concatenate([final_hidden, actual_oh])
    mlp_h_actual = jnp.tanh(inp_actual @ p['cf_W1'] + p['cf_b1'])
    pred_actual = (mlp_h_actual @ p['cf_W2']).squeeze() + p['cf_b2'][0]

    # Predict energy delta for alternative action
    alt_oh = jax.nn.one_hot(alt_action_idx, O)
    inp_alt = jnp.concatenate([final_hidden, alt_oh])
    mlp_h_alt = jnp.tanh(inp_alt @ p['cf_W1'] + p['cf_b1'])
    pred_alt = (mlp_h_alt @ p['cf_W2']).squeeze() + p['cf_b2'][0]

    # Contrastive loss: predict the difference
    pred_diff = pred_actual - pred_alt
    actual_diff = actual_delta - alt_delta

    return (pred_diff - actual_diff) ** 2


_contrastive_grad_single = jax.grad(_contrastive_loss, argnums=0)

_contrastive_grad_batch = jax.vmap(
    _contrastive_grad_single,
    in_axes=(0, 0, 0, 0, None, 0, 0, 0, 0)
)


# ---------------------------------------------------------------------------
# Learning rate extraction
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
# Environment step with contrastive learning
# ---------------------------------------------------------------------------

def _env_step_v33_inner(state, cfg, rng_key):
    """V33 env step: forward pass + contrastive prediction."""
    N = cfg['N']
    M = cfg['M_max']
    O = cfg['n_actions']

    agent_count = build_agent_count_grid(state['positions'], state['alive'], N)
    obs = build_observation_batch(
        state['positions'], state['resources'], state['signals'],
        agent_count, state['energy'], cfg
    )

    # Forward pass
    new_hidden, new_sync, raw_actions = agent_multi_tick_v33_batch(
        obs, state['hidden'], state['sync_matrices'], state['phenotypes'], cfg
    )
    new_hidden = new_hidden * state['alive'][:, None]
    new_sync = new_sync * state['alive'][:, None, None]

    # Decode ACTUAL actions
    move_idx, consume, emit = decode_actions(raw_actions, cfg)

    # === SAMPLE ALTERNATIVE ACTIONS ===
    alt_move_idx = jax.random.randint(rng_key, (M,), 0, 5)  # 5 move actions

    # Execute ACTUAL actions
    deltas = MOVE_DELTAS[move_idx]
    new_positions = (state['positions'] + deltas) % N
    new_positions = jnp.where(state['alive'][:, None], new_positions, state['positions'])

    resources, actual_consumed = apply_consumption(
        state['resources'], new_positions, consume * state['alive'], state['alive'], N
    )
    signals = apply_emissions(state['signals'], new_positions, emit, state['alive'], N)
    signals = diffuse_signals(signals, cfg)
    resources = regen_resources(resources, state['regen_rate'])

    actual_energy = state['energy'] - cfg['metabolic_cost'] + actual_consumed * cfg['resource_value']
    actual_energy = jnp.clip(actual_energy, 0.0, 2.0)
    actual_delta = actual_energy - state['energy']

    # === COMPUTE ALTERNATIVE energy delta (what if they'd moved differently?) ===
    alt_deltas_arr = MOVE_DELTAS[alt_move_idx]
    alt_positions = (state['positions'] + alt_deltas_arr) % N
    alt_positions = jnp.where(state['alive'][:, None], alt_positions, state['positions'])

    # Check what resources would be at alt positions (from ORIGINAL resources)
    alt_resource_at_pos = state['resources'][alt_positions[:, 0], alt_positions[:, 1]]
    alt_consumed = jnp.minimum(consume * state['alive'], alt_resource_at_pos)
    alt_energy = state['energy'] - cfg['metabolic_cost'] + alt_consumed * cfg['resource_value']
    alt_energy = jnp.clip(alt_energy, 0.0, 2.0)
    alt_delta = alt_energy - state['energy']

    # Kill starved
    new_alive = state['alive'] & (actual_energy > 0.0)

    new_state = {
        'resources': resources,
        'signals': signals,
        'positions': new_positions,
        'hidden': new_hidden,
        'energy': actual_energy,
        'alive': new_alive,
        'phenotypes': state['phenotypes'],
        'genomes': state['genomes'],
        'sync_matrices': new_sync,
        'regen_rate': state['regen_rate'],
        'step_count': state['step_count'] + 1,
    }

    return (new_state, obs, move_idx, alt_move_idx, actual_delta, alt_delta)


# ---------------------------------------------------------------------------
# Chunk runner with contrastive SGD
# ---------------------------------------------------------------------------

def make_v33_chunk_runner(cfg):
    """Create JIT-compiled chunk runner with contrastive gradient."""

    def scan_body(carry, step_idx):
        state, rng = carry

        pre_energy = state['energy']
        pre_hidden = state['hidden']
        pre_sync = state['sync_matrices']
        pre_alive = state['alive']
        pre_phenotypes = state['phenotypes']

        rng, rng_step = jax.random.split(rng)

        (new_state, obs, actual_action, alt_action,
         actual_delta, alt_delta) = _env_step_v33_inner(state, cfg, rng_step)

        # Contrastive prediction metrics
        alive_f = pre_alive.astype(jnp.float32)
        n_alive = jnp.maximum(jnp.sum(alive_f), 1.0)

        # Compute contrastive gradient
        grads = _contrastive_grad_batch(
            pre_phenotypes, obs, pre_hidden, pre_sync, cfg,
            actual_action, alt_action, actual_delta, alt_delta
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

        # Contrastive prediction error for metrics
        # Simple MSE on actual_delta prediction (ignoring contrastive for metrics)
        cf_pred_actual = predict_energy_for_action_batch(
            pre_hidden * alive_f[:, None],  # zero out dead
            actual_action, pre_phenotypes, cfg
        )
        pred_mse = jnp.sum(
            (cf_pred_actual - actual_delta) ** 2 * alive_f) / n_alive

        # Action difference metric: how different are actual vs alt deltas?
        action_diff = jnp.sum(
            jnp.abs(actual_delta - alt_delta) * alive_f) / n_alive

        # Divergence
        h_before_norm = jnp.sqrt(jnp.sum(pre_hidden ** 2, axis=1) + 1e-8)
        h_diff_norm = jnp.sqrt(
            jnp.sum((new_state['hidden'] - pre_hidden) ** 2, axis=1))
        div = h_diff_norm / h_before_norm
        mean_div = jnp.sum(div * alive_f) / n_alive

        metrics = {
            'n_alive': jnp.sum(new_state['alive']).astype(jnp.float32),
            'mean_energy': jnp.mean(new_state['energy'] * new_state['alive']),
            'mean_divergence': mean_div,
            'pred_mse': pred_mse,
            'mean_grad_norm': jnp.sum(grad_norm.squeeze() * alive_f) / n_alive,
            'action_diff': action_diff,
        }
        return (new_state, rng), metrics

    @jax.jit
    def run_chunk(state, rng):
        (final_state, final_rng), metrics = lax.scan(
            scan_body, (state, rng), jnp.arange(cfg['chunk_size'])
        )
        return final_state, final_rng, metrics

    return run_chunk


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_v33(seed, cfg):
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
