"""V30: Dual Prediction — Self + Social through Shared Bottleneck

V27 (self-prediction): Φ=0.245 max but only 1/3 seeds (convergence path)
V29 (social prediction): Φ=0.243 max, 2/3 seeds high (richness path)
V28 showed the mechanism is gradient coupling through 2-layer architecture.

V30 tests DUAL prediction: agents predict BOTH own energy delta AND neighbor
mean energy through a SHARED MLP bottleneck. This forces the shared hidden
layer to encode both self-model and social-model information simultaneously.

Architecture:
  - Same base as V27/V29 (GRU + 8 inner ticks + 2-layer MLP)
  - predict_W2: (PH, 2) instead of (PH, 1) — two outputs
  - predict_b2: (2,) instead of (1,)
  - Output 0: self energy delta prediction
  - Output 1: neighbor mean energy prediction
  - Loss = (pred_self - actual_delta)^2 + (pred_social - neighbor_mean_e)^2

Hypothesis: Dual prediction forces richer representations than either alone:
  - Self target: encodes own energy trajectory (convergence pressure)
  - Social target: encodes neighbor states (richness pressure)
  - Shared W1 bottleneck: BOTH must coexist in the same PH-dim space
  - This should produce higher AND more reliable Phi

Pre-registered predictions:
  P1: Mean Phi > V29's 0.104 (dual targets force richer shared representation)
  P2: Less seed-dependent than V27 (social component regularizes)
  P3: Individual MSEs slightly worse than dedicated (V27/V29) predictions
  P4: Effective rank > V29 (dual demands -> more dimensions needed)

Falsification:
  - P1 fails: shared bottleneck can't serve two masters (targets compete)
  - Phi < V29: self target HURTS social encoding (interference > complementarity)
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

from v29_substrate import compute_neighbor_mean_energy


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def generate_v30_config(**kwargs):
    """Generate V30 configuration — dual prediction."""
    cfg = {
        'N': 128,
        'M_max': 256,

        'obs_radius': 2,
        'embed_dim': 24,
        'hidden_dim': 16,
        'n_actions': 7,

        'K_max': 8,

        # MLP prediction head — shared, 2 outputs
        'predict_hidden': 8,
        'predict_outputs': 2,  # [self_delta, neighbor_mean_e]

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
    """Param layout — same as V27/V29 except predict_W2 and predict_b2 have 2 outputs."""
    obs_flat = cfg['obs_flat']
    E = cfg['embed_dim']
    H = cfg['hidden_dim']
    O = cfg['n_actions']
    K = cfg['K_max']
    PH = cfg['predict_hidden']
    PO = cfg['predict_outputs']  # 2
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
        'predict_W2': (PH, PO),  # 2 outputs instead of 1
        'predict_b2': (PO,),     # 2 biases instead of 1
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
# Forward pass — dual prediction output
# ---------------------------------------------------------------------------

def agent_multi_tick_v30(obs_flat, hidden, sync_matrix, params_flat, cfg):
    """Run K_max GRU ticks, returning 2-output MLP prediction.

    Returns:
        final_hidden: (H,)
        final_sync: (H, H)
        actions: (n_actions,)
        predictions: (2,) — [self_delta_pred, neighbor_mean_e_pred]
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

    # Dual MLP prediction: shared W1, two outputs from W2
    mlp_hidden = jnp.tanh(final_hidden @ p['predict_W1'] + p['predict_b1'])  # (PH,)
    predictions = mlp_hidden @ p['predict_W2'] + p['predict_b2']  # (2,)

    return final_hidden, final_sync, actions, predictions


agent_multi_tick_v30_batch = jax.vmap(
    agent_multi_tick_v30, in_axes=(0, 0, 0, 0, None)
)


# ---------------------------------------------------------------------------
# Prediction gradient — DUAL loss
# ---------------------------------------------------------------------------

def _prediction_loss_v30(phenotype_flat, obs_flat, hidden, sync_matrix, cfg,
                         target_self, target_social):
    """Dual loss: (pred_self - target_self)^2 + (pred_social - target_social)^2."""
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
    preds = mlp_hidden @ p['predict_W2'] + p['predict_b2']  # (2,)

    loss_self = (preds[0] - target_self) ** 2
    loss_social = (preds[1] - target_social) ** 2
    return loss_self + loss_social


_prediction_grad_single_v30 = jax.grad(_prediction_loss_v30, argnums=0)
_prediction_grad_batch_v30 = jax.vmap(
    _prediction_grad_single_v30, in_axes=(0, 0, 0, 0, None, 0, 0)
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
# V30 environment step
# ---------------------------------------------------------------------------

def _env_step_v30_inner(state, cfg):
    """V30 env step: returns both energy delta and neighbor mean energy."""
    N = cfg['N']

    agent_count = build_agent_count_grid(state['positions'], state['alive'], N)
    obs = build_observation_batch(
        state['positions'], state['resources'], state['signals'],
        agent_count, state['energy'], cfg
    )

    new_hidden, new_sync, raw_actions, predictions = agent_multi_tick_v30_batch(
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

    # Social target: mean energy of neighbors
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
# Chunk runner with dual prediction gradient
# ---------------------------------------------------------------------------

def make_v30_chunk_runner(cfg):
    """Create JIT-compiled chunk runner with DUAL prediction gradient."""
    def scan_body(state, step_idx):
        pre_energy = state['energy']
        pre_hidden = state['hidden']
        pre_sync = state['sync_matrices']
        pre_alive = state['alive']
        pre_phenotypes = state['phenotypes']

        new_state, obs, predictions, neighbor_mean_e = _env_step_v30_inner(state, cfg)

        # DUAL targets
        target_self = new_state['energy'] - pre_energy    # own energy delta
        target_social = neighbor_mean_e                     # neighbor mean energy

        alive_f = pre_alive.astype(jnp.float32)
        n_alive = jnp.maximum(jnp.sum(alive_f), 1.0)

        # MSE for each target (for logging)
        pred_mse_self = jnp.sum((predictions[:, 0] - target_self) ** 2 * alive_f) / n_alive
        pred_mse_social = jnp.sum((predictions[:, 1] - target_social) ** 2 * alive_f) / n_alive
        pred_mse = pred_mse_self + pred_mse_social

        # Gradient wrt DUAL loss
        grads = _prediction_grad_batch_v30(
            pre_phenotypes, obs, pre_hidden, pre_sync, cfg,
            target_self, target_social
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
            'pred_mse_self': pred_mse_self,
            'pred_mse_social': pred_mse_social,
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

def init_v30(seed, cfg):
    """Initialize V30 state — same as V27/V29."""
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
