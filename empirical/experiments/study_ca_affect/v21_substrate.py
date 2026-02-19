"""V21: CTM-Inspired Protocell Agency

Adds two Continuous Thought Machine (CTM) formalisms to the V20 protocell
substrate as ARCHITECTURAL AFFORDANCES — not training methods. Evolution
remains tournament selection + Gaussian mutation. Zero labels, zero human
data, zero gradient descent.

CTM formalisms borrowed:
  1. Internal ticks — K_max GRU steps per environment step, with evolvable
     tick_weights gating which ticks contribute to action output
  2. Sync matrix — temporal cross-unit coordination tracker on hidden states,
     a real-time proxy for integration (NOT Phi — call it Phi_sync)

V20b showed the sensory-motor wall is broken (rho_sync=0.21), but language
precursor analysis was NULL — z-gate stuck at ~0.50. Root cause: a single
GRU step per environment step provides zero internal time for decoupled
processing. V21 provides that time.

Pre-registered predictions:
  1. Effective K evolves upward under bottleneck, downward under abundance
  2. Intra-step divergence correlates with subsequent action quality (I_img > 0)
  3. tick_weights do NOT collapse to tick-0 in at least 1/3 seeds

Falsification criteria:
  - tick_weights collapse to tick-0 across all agents/seeds -> V21 negative
  - Phi_sync zero correlation with Phi_hidden -> sync matrix not measuring integration
  - Language precursors still absent -> decoupled processing alone insufficient
  - Effective K does NOT covary with resource scarcity -> no adaptive deliberation

Architecture (changes from V20b):
  - K_max=8 GRU ticks per environment step (fixed for vectorization)
  - Evolvable tick_weights (softmax gating of tick outputs)
  - Persistent sync matrix per agent: S = r*S_prev + h (x) h
  - Sync summary (3 floats) fed as input on ticks 1+ via internal_embed
  - 105 new evolvable parameters (4,040 total vs V20's 3,935)
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def generate_v21_config(**kwargs):
    """Generate V21 configuration with CTM inner ticks."""
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

        # Environment dynamics (same as V20b)
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

        # V20b defaults for V21
        'activate_offspring': True,
        'drought_every': 5,
        'drought_depletion': 0.01,
        'drought_regen': 0.0,

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
    }


def unpack_params(flat, cfg):
    """Unpack flat parameter vector into named weight matrices.
    Works inside jax.vmap (flat is shape (P,) per agent)."""
    shapes = _param_shapes(cfg)
    params = {}
    idx = 0
    for name, shape in shapes.items():
        size = int(np.prod(shape))
        params[name] = flat[idx:idx + size].reshape(shape)
        idx += size
    return params


# ---------------------------------------------------------------------------
# Sync matrix operations
# ---------------------------------------------------------------------------

def compute_sync_summary(S):
    """Extract 3-dimensional summary from H x H sync matrix.

    Returns: (3,) array of [frobenius_offdiag_norm, mean_diag, std_offdiag]
    """
    H = S.shape[0]
    diag = jnp.diag(S)
    offdiag_mask = 1.0 - jnp.eye(H)
    offdiag = S * offdiag_mask

    n_offdiag = H * (H - 1)

    # Normalized off-diagonal Frobenius norm
    frobenius_offdiag = jnp.sqrt(jnp.sum(offdiag ** 2) + 1e-8) / H
    mean_diag = jnp.mean(diag)

    # Std of off-diagonal entries (computed from sums to avoid dynamic indexing)
    offdiag_mean = jnp.sum(offdiag) / n_offdiag
    offdiag_var = jnp.sum(offdiag ** 2) / n_offdiag - offdiag_mean ** 2
    std_offdiag = jnp.sqrt(jnp.maximum(offdiag_var, 0.0) + 1e-8)

    return jnp.array([frobenius_offdiag, mean_diag, std_offdiag])


def compute_phi_sync(sync_matrices, alive):
    """Coordination metric from sync matrices.

    Measures off-diagonal Frobenius norm — how much cross-unit
    coordination exists in each agent's hidden dynamics.

    This is NOT Phi (information loss under partition). It's a
    real-time proxy measuring pairwise temporal coordination
    between hidden state dimensions.
    """
    H = sync_matrices.shape[1]
    offdiag_mask = 1.0 - jnp.eye(H)
    offdiag = sync_matrices * offdiag_mask[None, :, :]  # (M, H, H)
    phi_per = jnp.sqrt(jnp.sum(offdiag ** 2, axis=(1, 2)) + 1e-8)  # (M,)

    alive_f = alive.astype(jnp.float32)
    n_alive = jnp.maximum(jnp.sum(alive_f), 1.0)
    return jnp.sum(phi_per * alive_f) / n_alive


# ---------------------------------------------------------------------------
# Agent multi-tick forward pass (single agent)
# ---------------------------------------------------------------------------

def agent_multi_tick(obs_flat, hidden, sync_matrix, params_flat, cfg):
    """Run K_max GRU ticks for one agent per environment step.

    Tick 0:  input = embed(observation)       — external signal
    Tick 1+: input = embed(sync_summary)      — internal signal (3 -> E dims)

    All ticks share GRU weights. The hidden state differentiates behavior
    across ticks. Sync matrix accumulates h (x) h with exponential decay.

    Returns:
        final_hidden: (H,)
        final_sync: (H, H)
        actions: (n_actions,) weighted mix of tick outputs
    """
    p = unpack_params(params_flat, cfg)
    K = cfg['K_max']

    # Pre-compute external embedding (used only at tick 0)
    x_ext = jnp.tanh(obs_flat @ p['embed_W'] + p['embed_b'])  # (E,)

    # Sync decay: sigmoid mapped to [0.5, 0.999]
    sync_decay = 0.5 + 0.499 * jax.nn.sigmoid(p['sync_decay_raw'][0])

    def tick_fn(carry, tick_idx):
        h, S = carry

        # Internal input from sync summary (ticks 1+)
        sync_sum = compute_sync_summary(S)  # (3,)
        x_int = jnp.tanh(sync_sum @ p['internal_embed_W'] + p['internal_embed_b'])  # (E,)

        # Select: tick 0 = external, tick 1+ = internal
        x = jnp.where(tick_idx == 0, x_ext, x_int)

        # GRU step (shared weights across all ticks)
        xh = jnp.concatenate([x, h])  # (E+H,)
        z = jax.nn.sigmoid(xh @ p['gru_Wz'] + p['gru_bz'])
        r = jax.nn.sigmoid(xh @ p['gru_Wr'] + p['gru_br'])
        h_tilde = jnp.tanh(
            jnp.concatenate([x, r * h]) @ p['gru_Wh'] + p['gru_bh']
        )
        new_h = z * h + (1.0 - z) * h_tilde

        # Update sync matrix: S = decay * S + h (x) h
        new_S = sync_decay * S + jnp.outer(new_h, new_h)

        # Output logits for this tick
        output = new_h @ p['out_W'] + p['out_b']  # (O,)

        return (new_h, new_S), output

    # Run K ticks
    (final_hidden, final_sync), all_outputs = lax.scan(
        tick_fn, (hidden, sync_matrix), jnp.arange(K)
    )
    # all_outputs: (K, n_actions)

    # Weighted mix of tick outputs via evolvable tick_weights
    weights = jax.nn.softmax(p['tick_weights'])  # (K,)
    actions = jnp.einsum('k,ko->o', weights, all_outputs)  # (n_actions,)

    return final_hidden, final_sync, actions


# Vectorize over agents
agent_multi_tick_batch = jax.vmap(
    agent_multi_tick, in_axes=(0, 0, 0, 0, None)
)


# ---------------------------------------------------------------------------
# Single environment step (JIT-compiled)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=['cfg_tuple'])
def env_step(state, cfg_tuple):
    """One step of the V21 environment.

    Same as V20 but with multi-tick GRU and sync matrices.

    state keys (additions vs V20 marked with *):
        resources: (N, N)
        signals: (N, N)
        positions: (M, 2) int32
        hidden: (M, H)
        energy: (M,) float
        alive: (M,) bool
        params: (M, P) float
        *sync_matrices: (M, H, H) float — persistent per-agent sync
        regen_rate: float scalar
        step_count: int
    """
    cfg = dict(cfg_tuple)
    N = cfg['N']

    resources = state['resources']
    signals = state['signals']
    positions = state['positions']
    hidden = state['hidden']
    energy = state['energy']
    alive = state['alive']
    params = state['params']
    sync_matrices = state['sync_matrices']
    regen_rate = state['regen_rate']

    # 1. Build agent count grid
    agent_count = build_agent_count_grid(positions, alive, N)

    # 2. Build observations for all agents
    obs = build_observation_batch(positions, resources, signals, agent_count, energy, cfg)

    # 3. Agent multi-tick forward pass (replaces V20's single agent_step)
    new_hidden, new_sync, raw_actions = agent_multi_tick_batch(
        obs, hidden, sync_matrices, params, cfg
    )
    # Zero out dead agents
    new_hidden = new_hidden * alive[:, None]
    new_sync = new_sync * alive[:, None, None]

    # 4. Decode actions
    move_idx, consume, emit = decode_actions(raw_actions, cfg)

    # 5. Movement (only alive agents move)
    deltas = MOVE_DELTAS[move_idx]
    new_positions = (positions + deltas) % N
    new_positions = jnp.where(alive[:, None], new_positions, positions)

    # 6. Consumption
    resources, actual_consumed = apply_consumption(
        resources, new_positions, consume * alive, alive, N
    )

    # 7. Emission
    signals = apply_emissions(signals, new_positions, emit, alive, N)

    # 8. Environment dynamics
    signals = diffuse_signals(signals, cfg)
    resources = regen_resources(resources, regen_rate)

    # 9. Energy update
    energy = energy - cfg['metabolic_cost'] + actual_consumed * cfg['resource_value']
    energy = jnp.clip(energy, 0.0, 2.0)

    # 10. Kill starved agents
    new_alive = alive & (energy > 0.0)

    new_state = {
        'resources': resources,
        'signals': signals,
        'positions': new_positions,
        'hidden': new_hidden,
        'energy': energy,
        'alive': new_alive,
        'params': params,
        'sync_matrices': new_sync,
        'regen_rate': regen_rate,
        'step_count': state['step_count'] + 1,
    }
    return new_state


# ---------------------------------------------------------------------------
# Chunk runner (lax.scan over chunk_size steps)
# ---------------------------------------------------------------------------

def make_chunk_runner(cfg):
    """Create JIT-compiled chunk runner for V21.

    Returns per-step metrics including intra-step divergence.
    """
    cfg_tuple = tuple(sorted(cfg.items()))

    def scan_body(state, _):
        h_before = state['hidden']
        new_state = env_step(state, cfg_tuple)
        h_after = new_state['hidden']

        # Intra-step divergence: ||h_K - h_0|| / ||h_0||
        alive_f = state['alive'].astype(jnp.float32)
        n_alive = jnp.maximum(jnp.sum(alive_f), 1.0)
        h_norm = jnp.sqrt(jnp.sum(h_before ** 2, axis=1) + 1e-8)  # (M,)
        diff_norm = jnp.sqrt(jnp.sum((h_after - h_before) ** 2, axis=1))  # (M,)
        div = diff_norm / h_norm  # (M,)
        mean_div = jnp.sum(div * alive_f) / n_alive

        metrics = {
            'n_alive': jnp.sum(new_state['alive']).astype(jnp.float32),
            'mean_energy': jnp.mean(new_state['energy'] * new_state['alive']),
            'mean_divergence': mean_div,
        }
        return new_state, metrics

    @jax.jit
    def run_chunk(state):
        """Run chunk_size steps and return final state + metrics."""
        final_state, metrics = lax.scan(
            scan_body, state, None, length=cfg['chunk_size']
        )
        return final_state, metrics

    return run_chunk


# ---------------------------------------------------------------------------
# Utility: extract tick_weights and sync_decay from flat params (numpy)
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
    """Extract tick_weights from (M, P) params array.

    Returns: (M, K) softmax-normalized weights.
    """
    offset = _param_offset(cfg, 'tick_weights')
    K = cfg['K_max']
    raw = np.array(params)[:, offset:offset + K]
    # Stable softmax
    raw_shifted = raw - np.max(raw, axis=1, keepdims=True)
    exp_raw = np.exp(raw_shifted)
    return exp_raw / np.sum(exp_raw, axis=1, keepdims=True)


def extract_sync_decay_np(params, cfg):
    """Extract sync decay from (M, P) params array.

    Returns: (M,) decay values in [0.5, 0.999].
    """
    offset = _param_offset(cfg, 'sync_decay_raw')
    raw = np.array(params)[:, offset]
    return 0.5 + 0.499 / (1.0 + np.exp(-raw))


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_v21(seed, cfg):
    """Initialize V21 state.

    Same as V20 but with sync_matrices added.
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

    # Sync matrices (new in V21)
    sync_matrices = jnp.zeros((M, H, H))

    # Energy
    energy = jnp.zeros(M)
    energy = energy.at[:n_initial].set(cfg['initial_energy'])

    # Alive mask
    alive = jnp.zeros(M, dtype=jnp.bool_)
    alive = alive.at[:n_initial].set(True)

    # Parameters: small random initialization
    params = jax.random.normal(k4, (M, P)) * 0.1

    state = {
        'resources': resources,
        'signals': signals,
        'positions': positions,
        'hidden': hidden,
        'energy': energy,
        'alive': alive,
        'params': params,
        'sync_matrices': sync_matrices,
        'regen_rate': jnp.array(cfg['resource_regen']),
        'step_count': jnp.array(0),
    }
    return state, key


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

def extract_snapshot(state, cycle, cfg):
    """Extract lightweight snapshot for offline measurement.

    Sync matrices omitted (reset at cycle boundary; reconstructable from rollout).
    """
    alive = state['alive']
    return {
        'cycle': cycle,
        'hidden': np.array(state['hidden']),
        'positions': np.array(state['positions']),
        'energy': np.array(state['energy']),
        'alive': np.array(alive),
        'params': np.array(state['params']),
        'resources': np.array(state['resources']),
        'signals': np.array(state['signals']),
        'n_alive': int(jnp.sum(alive)),
    }
