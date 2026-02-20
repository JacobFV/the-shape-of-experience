"""V35: Language Emergence under Cooperative POMDP

V20b showed: continuous signals → noise (z ≈ 0.5 always). No language.
V29/V31 showed: prediction target doesn't matter. Social coupling ≠ integration.

Hypothesis: Language emerges when three conditions are met simultaneously:
  1. Partial observability (information asymmetry — agents can't see everything)
  2. Discrete communication channel (forces categorical representation)
  3. Cooperative pressure (communication has survival value)

V35 tests this by adding a discrete communication channel to V27's substrate
under cooperative POMDP conditions.

Architecture (extends V27):
  - Base: GRU + 8 inner ticks + 2-layer MLP prediction head (identical V27)
  - Observation: 3×3 patch (obs_radius=1, reduced from V27's 5×5)
    → agents can see less, creating information asymmetry
  - Communication: K_sym=8 discrete symbols, one emitted per step
    → argmax of K_sym outputs appended to action head
  - Symbol reception: agents observe histogram of symbols from agents
    within comm_radius=5 (larger than obs_radius=1)
    → communication range > visual range (the key asymmetry)
  - Observation includes: 3×3 patch (R,S,N) + energy + symbol_histogram(K_sym)
  - Cooperative dynamics: agents that co-locate near resources get a
    consumption bonus (1.5× when 2+ agents consume same cell)
    → direct selective pressure for coordination

The key insight: if agents can see further by hearing than by looking,
and if cooperation at resource sites is rewarded, then agents who signal
"resource here" create value for others — and get selected because their
offspring (post-tournament) also signal.

Pre-registered predictions:
  P1: Symbol-environment MI > 0.1 bits by cycle 20
      (agents develop referential signaling about resources)
  P2: Symbol entropy > 1.0 bits (agents use >2 symbols meaningfully)
  P3: Cooperation events increase over evolution
  P4: Φ ≥ V27 baseline (0.090)
  P5: Ablating communication drops Φ (integration depends on social channel)

Falsification:
  - P1 fails: No referential signaling (like V20b, signals remain noise)
  - P2 fails: Symbol vocabulary collapses to 1-2 symbols (no language)
  - P5 fails: Φ independent of communication → language is epiphenomenal
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
import numpy as np

# Import environment dynamics from V20 (base)
from v20_substrate import (
    gather_patch,
    build_agent_count_grid, apply_emissions,
    diffuse_signals, regen_resources, MOVE_DELTAS,
    compute_phi_hidden, compute_robustness,
)

# Import sync operations from V21
from v21_substrate import (
    compute_sync_summary, compute_phi_sync,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def generate_v35_config(**kwargs):
    """Generate V35 configuration — cooperative POMDP + discrete communication."""
    cfg = {
        # Grid
        'N': 128,
        'M_max': 256,

        # Agent architecture
        'obs_radius': 1,       # REDUCED from V27's 2 → 3×3 obs (partial observability)
        'embed_dim': 24,
        'hidden_dim': 16,

        # CTM inner ticks (from V21/V27)
        'K_max': 8,

        # MLP prediction head (same as V27)
        'predict_hidden': 8,

        # Communication
        'K_sym': 8,            # Number of discrete symbols
        'comm_radius': 5,      # Communication radius (larger than obs_radius!)

        # Cooperative dynamics
        'coop_bonus': 0.5,     # Extra consumption when 2+ agents at same cell

        # Environment dynamics (same as V27)
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

        # Drought
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

    # Derived dimensions
    obs_side = 2 * cfg['obs_radius'] + 1   # 3
    K_sym = cfg['K_sym']
    # Obs: 3×3×3 (R,S,N patches) + 1 (energy) + K_sym (symbol histogram) = 28 + K_sym
    obs_flat = obs_side * obs_side * 3 + 1 + K_sym   # 28 + 8 = 36 when K_sym=8, obs_radius=1
    cfg['obs_side'] = obs_side
    cfg['obs_flat'] = obs_flat

    # Action space: 5 move + 1 consume + K_sym symbol outputs
    cfg['n_actions'] = 5 + 1 + K_sym   # 14

    shapes = _param_shapes(cfg)
    total = sum(int(np.prod(s)) for s in shapes.values())
    cfg['n_params'] = total

    return cfg


def _param_shapes(cfg):
    """Ordered dict of parameter name -> shape."""
    obs_flat = cfg['obs_flat']
    E = cfg['embed_dim']
    H = cfg['hidden_dim']
    O = cfg['n_actions']  # 14 (5 move + 1 consume + 8 symbols)
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
        # V27 MLP prediction params
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
# Observation building (V35: 3×3 patch + energy + symbol histogram)
# ---------------------------------------------------------------------------

def build_observation_v35(pos, resources, signals, agent_count_grid,
                          energy, symbol_histogram, cfg):
    """Build flat observation for one agent.

    Returns: (obs_flat,) = flattened [R_patch, S_patch, N_patch] + energy + symbol_hist
    """
    r = cfg['obs_radius']  # 1 → 3×3 patch
    R_patch = gather_patch(resources, pos, r)        # (3,3)
    S_patch = gather_patch(signals, pos, r)           # (3,3)
    N_patch = gather_patch(agent_count_grid, pos, r)  # (3,3)
    patches = jnp.stack([R_patch, S_patch, N_patch], axis=-1)  # (3,3,3)
    return jnp.concatenate([
        patches.flatten(),              # 27
        jnp.array([energy]),            # 1
        symbol_histogram,              # K_sym
    ])


build_observation_v35_batch = jax.vmap(
    build_observation_v35, in_axes=(0, None, None, None, 0, 0, None)
)


# ---------------------------------------------------------------------------
# Symbol histogram computation
# ---------------------------------------------------------------------------

def compute_symbol_histograms(positions, emitted_symbols, alive, cfg):
    """Compute per-agent histogram of symbols from agents within comm_radius.

    For each agent, count symbols emitted by other alive agents within
    comm_radius, normalize to [0, 1].

    positions: (M, 2) int32
    emitted_symbols: (M,) int32, each in [0, K_sym)
    alive: (M,) bool

    Returns: (M, K_sym) float32 normalized histograms
    """
    M = cfg['M_max']
    K_sym = cfg['K_sym']
    N = cfg['N']
    R = cfg['comm_radius']

    # Compute pairwise distances (toroidal)
    dx = jnp.abs(positions[:, None, 0] - positions[None, :, 0])  # (M, M)
    dy = jnp.abs(positions[:, None, 1] - positions[None, :, 1])  # (M, M)
    dx = jnp.minimum(dx, N - dx)
    dy = jnp.minimum(dy, N - dy)
    dist = jnp.maximum(dx, dy)  # Chebyshev distance

    # Mask: within comm_radius, alive, not self
    mask = (dist <= R) & alive[None, :] & (jnp.arange(M)[:, None] != jnp.arange(M)[None, :])
    # (M, M) bool

    # One-hot encode emitted symbols: (M, K_sym)
    sym_onehot = jax.nn.one_hot(emitted_symbols, K_sym)  # (M, K_sym)

    # Weighted sum: for each agent i, sum sym_onehot[j] where mask[i,j]
    # mask: (M, M), sym_onehot: (M, K_sym)
    histograms = jnp.einsum('ij,jk->ik', mask.astype(jnp.float32), sym_onehot)  # (M, K_sym)

    # Normalize to [0, 1]
    totals = jnp.sum(histograms, axis=1, keepdims=True) + 1e-8
    histograms = histograms / totals

    return histograms


# ---------------------------------------------------------------------------
# Action decoding (V35: move + consume + symbol)
# ---------------------------------------------------------------------------

def decode_actions_v35(raw_actions, cfg):
    """Decode raw output into actions including discrete symbol.

    Returns:
        move_idx: (M,) int32, 0-4
        consume: (M,) float, 0-1
        symbol_idx: (M,) int32, 0 to K_sym-1
    """
    # Move: argmax of first 5 outputs
    move_idx = jnp.argmax(raw_actions[:, :5], axis=-1).astype(jnp.int32)
    # Consume: sigmoid of output 5
    consume = jax.nn.sigmoid(raw_actions[:, 5])
    # Symbol: argmax of outputs 6:6+K_sym
    K_sym = cfg['K_sym']
    symbol_idx = jnp.argmax(raw_actions[:, 6:6 + K_sym], axis=-1).astype(jnp.int32)
    return move_idx, consume, symbol_idx


# ---------------------------------------------------------------------------
# Cooperative consumption
# ---------------------------------------------------------------------------

def apply_cooperative_consumption(resources, positions, consume_amounts,
                                   alive, agent_count_grid, cfg):
    """Consumption with cooperative bonus.

    When 2+ alive agents are at the same cell, each gets a bonus multiplier.
    """
    N = cfg['N']
    coop_bonus = cfg['coop_bonus']

    # Base consumption (from V20)
    cell_resources = resources[positions[:, 0], positions[:, 1]]
    actual_consumed = jnp.minimum(consume_amounts * alive, cell_resources * 0.1)

    # Cooperative bonus: count agents at each position
    counts_at_pos = agent_count_grid[positions[:, 0], positions[:, 1]]
    # Bonus when 2+ agents present: 1 + coop_bonus * (n-1)/n
    coop_multiplier = 1.0 + coop_bonus * jnp.maximum(counts_at_pos - 1, 0) / jnp.maximum(counts_at_pos, 1)
    actual_consumed = actual_consumed * coop_multiplier

    # Deplete resources (simplified — each agent takes independently)
    new_resources = resources.at[positions[:, 0], positions[:, 1]].add(
        -actual_consumed
    )
    new_resources = jnp.clip(new_resources, 0.0, 1.0)

    return new_resources, actual_consumed


# ---------------------------------------------------------------------------
# Agent multi-tick forward pass (same as V27, different obs dimension)
# ---------------------------------------------------------------------------

def agent_multi_tick_v35(obs_flat, hidden, sync_matrix, params_flat, cfg):
    """Run K_max GRU ticks — identical logic to V27, just different obs size."""
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


agent_multi_tick_v35_batch = jax.vmap(
    agent_multi_tick_v35, in_axes=(0, 0, 0, 0, None)
)


# ---------------------------------------------------------------------------
# Prediction gradient (same as V27, different obs dim)
# ---------------------------------------------------------------------------

def _prediction_loss(phenotype_flat, obs_flat, hidden, sync_matrix, cfg, actual_delta):
    """Scalar loss: (predicted_delta - actual_delta)^2."""
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
    return (pred - actual_delta) ** 2


_prediction_grad_single = jax.grad(_prediction_loss, argnums=0)
_prediction_grad_batch = jax.vmap(
    _prediction_grad_single, in_axes=(0, 0, 0, 0, None, 0)
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
# V35 environment step
# ---------------------------------------------------------------------------

def _env_step_v35_inner(state, cfg):
    """V35 env step: multi-tick with communication channel."""
    N = cfg['N']

    # 1. Build agent count grid
    agent_count = build_agent_count_grid(state['positions'], state['alive'], N)

    # 2. Compute symbol histograms from previous step's emissions
    sym_hist = compute_symbol_histograms(
        state['positions'], state['emitted_symbols'], state['alive'], cfg
    )

    # 3. Build observations (includes symbol histogram)
    obs = build_observation_v35_batch(
        state['positions'], state['resources'], state['signals'],
        agent_count, state['energy'], sym_hist, cfg
    )

    # 4. Multi-tick forward pass
    new_hidden, new_sync, raw_actions, predictions = agent_multi_tick_v35_batch(
        obs, state['hidden'], state['sync_matrices'], state['phenotypes'], cfg
    )
    new_hidden = new_hidden * state['alive'][:, None]
    new_sync = new_sync * state['alive'][:, None, None]

    # 5. Decode actions (move + consume + symbol)
    move_idx, consume, symbol_idx = decode_actions_v35(raw_actions, cfg)

    # 6. Movement
    deltas = MOVE_DELTAS[move_idx]
    new_positions = (state['positions'] + deltas) % N
    new_positions = jnp.where(state['alive'][:, None], new_positions, state['positions'])

    # 7. Update agent count for new positions
    new_agent_count = build_agent_count_grid(new_positions, state['alive'], N)

    # 8. Cooperative consumption
    resources, actual_consumed = apply_cooperative_consumption(
        state['resources'], new_positions, consume * state['alive'],
        state['alive'], new_agent_count, cfg
    )

    # 9. Signal emission (continuous, from V20 — keep for environment richness)
    # Use a fixed emission amount based on whether agent emitted a symbol
    emit_amount = state['alive'].astype(jnp.float32) * 0.05
    signals = apply_emissions(state['signals'], new_positions, emit_amount, state['alive'], N)
    signals = diffuse_signals(signals, cfg)

    # 10. Resource regeneration
    resources = regen_resources(resources, state['regen_rate'])

    # 11. Energy update
    energy = state['energy'] - cfg['metabolic_cost'] + actual_consumed * cfg['resource_value']
    energy = jnp.clip(energy, 0.0, 2.0)

    # 12. Kill starved agents
    new_alive = state['alive'] & (energy > 0.0)

    # 13. Track cooperative events: cells with 2+ agents consuming
    coop_cells = (new_agent_count >= 2).astype(jnp.float32)
    n_coop = jnp.sum(coop_cells * (resources < state['resources']))  # cells that were consumed

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
        'emitted_symbols': symbol_idx,  # Update emitted symbols
    }
    return new_state, obs, predictions, symbol_idx, n_coop


# ---------------------------------------------------------------------------
# Chunk runner with SGD
# ---------------------------------------------------------------------------

def make_v35_chunk_runner(cfg):
    """Create JIT-compiled chunk runner with communication."""
    def scan_body(state, step_idx):
        pre_energy = state['energy']
        pre_hidden = state['hidden']
        pre_sync = state['sync_matrices']
        pre_alive = state['alive']
        pre_phenotypes = state['phenotypes']

        new_state, obs, predictions, symbol_idx, n_coop = _env_step_v35_inner(state, cfg)

        actual_delta = new_state['energy'] - pre_energy
        alive_f = pre_alive.astype(jnp.float32)
        n_alive = jnp.maximum(jnp.sum(alive_f), 1.0)
        pred_mse = jnp.sum((predictions - actual_delta) ** 2 * alive_f) / n_alive

        grads = _prediction_grad_batch(
            pre_phenotypes, obs, pre_hidden, pre_sync, cfg, actual_delta
        )

        grad_norm = jnp.sqrt(jnp.sum(grads ** 2, axis=1, keepdims=True) + 1e-8)
        max_norm = 1.0
        grads = grads * jnp.minimum(1.0, max_norm / grad_norm)

        lr = extract_lr_jax(pre_phenotypes, cfg)
        updated_phenotypes = pre_phenotypes - lr[:, None] * grads * alive_f[:, None]

        do_update = (step_idx % cfg['grad_every'] == 0)
        new_phenotypes = jnp.where(do_update, updated_phenotypes, pre_phenotypes)
        new_state = {**new_state, 'phenotypes': new_phenotypes}

        # Symbol entropy (per-step)
        K_sym = cfg['K_sym']
        sym_onehot = jax.nn.one_hot(symbol_idx, K_sym)  # (M, K_sym)
        sym_counts = jnp.sum(sym_onehot * alive_f[:, None], axis=0)  # (K_sym,)
        sym_probs = sym_counts / jnp.maximum(jnp.sum(sym_counts), 1.0)
        sym_entropy = -jnp.sum(sym_probs * jnp.log2(sym_probs + 1e-10))

        # Symbol-resource MI proxy: correlation between dominant symbol and local resource
        local_resource = new_state['resources'][
            new_state['positions'][:, 0], new_state['positions'][:, 1]
        ]
        # For each symbol, compute mean local resource of agents emitting it
        sym_resource_by_type = jnp.sum(
            sym_onehot * local_resource[:, None] * alive_f[:, None], axis=0
        )
        sym_counts_safe = jnp.maximum(sym_counts, 1.0)
        mean_resource_per_sym = sym_resource_by_type / sym_counts_safe
        # Variance of mean resource across symbols (higher = more referential)
        resource_variance = jnp.var(mean_resource_per_sym)

        metrics = {
            'n_alive': jnp.sum(new_state['alive']).astype(jnp.float32),
            'mean_energy': jnp.mean(new_state['energy'] * new_state['alive']),
            'pred_mse': pred_mse,
            'mean_grad_norm': jnp.sum(grad_norm.squeeze() * alive_f) / n_alive,
            'sym_entropy': sym_entropy,
            'sym_resource_var': resource_variance,
            'n_coop': n_coop,
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

def init_v35(seed, cfg):
    """Initialize V35 state — V27 base + emitted_symbols."""
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

    # V35: initial symbols (all zeros — no communication yet)
    emitted_symbols = jnp.zeros(M, dtype=jnp.int32)

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
        'emitted_symbols': emitted_symbols,
    }
    return state, key


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

def extract_snapshot(state, cycle, cfg):
    """Extract snapshot with communication data."""
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
        'emitted_symbols': np.array(state['emitted_symbols']),
        'n_alive': int(jnp.sum(alive)),
    }
