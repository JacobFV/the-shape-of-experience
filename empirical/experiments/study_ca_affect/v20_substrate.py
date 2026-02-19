"""V20: Protocell Agency Substrate

Genuine action-observation loops in an evolved neural agent grid world.
Leaves Lenia (V13-V18) entirely. Each agent:
  - Observes a bounded 5×5 local neighborhood (not global FFT)
  - Takes discrete actions that physically change the environment
  - Whose future observations are partly caused by its own past actions

This creates the action→environment→observation causal loop that was
architecturally absent in all V13-V18 Lenia substrates. The sensory-motor
wall (ρ_sync ≈ 0) was not about signal routing or substrate complexity —
it was about the absence of self-as-cause. V20 provides it.

Architecture:
  Grid: N×N continuous resource (R) and signal (S) fields
  Agents: evolved GRU networks, bounded sensory fields, discrete actions
  Actions: move (5 dirs), consume (deplete local R), emit (write to S field)
  Uncontaminated: random init, evolved not gradient-trained, no human data
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def generate_v20_config(**kwargs):
    """Generate V20 configuration. All evolution params are scalars."""
    cfg = {
        # Grid
        'N': 128,             # Grid size (N×N)
        'M_max': 256,         # Max population

        # Agent architecture
        'obs_radius': 2,      # Observation radius → 5×5 patch
        'embed_dim': 24,      # Linear embedding dimension
        'hidden_dim': 16,     # GRU hidden state dimension

        # Action space: 5 move directions + 1 consume + 1 emit = 7
        'n_actions': 7,

        # Environment dynamics
        'resource_regen': 0.002,   # Logistic regen rate per step
        'signal_decay': 0.04,      # Signal decay per step
        'signal_diffusion': 0.15,  # Signal diffusion coefficient
        'metabolic_cost': 0.0004,  # Energy drain per step (must consume to live)
        'initial_energy': 1.0,
        'resource_value': 1.5,     # Energy per unit resource consumed
        'initial_resource': 0.5,   # Initial mean resource density

        # Stress (drought)
        'stress_regen': 0.0002,    # 10× reduction during drought

        # Evolution
        'mutation_std': 0.03,
        'tournament_size': 4,
        'elite_fraction': 0.5,

        # Run
        'chunk_size': 50,          # Steps per lax.scan segment
        'steps_per_cycle': 5000,
        'n_cycles': 30,
    }
    cfg.update(kwargs)

    # Derived dimensions
    obs_side = 2 * cfg['obs_radius'] + 1   # 5
    obs_flat = obs_side * obs_side * 3 + 1  # 5×5×(R,S,agents) + energy = 76
    H = cfg['hidden_dim']
    E = cfg['embed_dim']
    O = cfg['n_actions']

    cfg['obs_side'] = obs_side
    cfg['obs_flat'] = obs_flat

    # Parameter counts (cumulative offsets)
    shapes = _param_shapes(cfg)
    total = sum(int(np.prod(s)) for s in shapes.values())
    cfg['n_params'] = total

    return cfg


def _param_shapes(cfg):
    """Ordered dict of parameter name → shape."""
    obs_flat = cfg['obs_flat']
    E = cfg['embed_dim']
    H = cfg['hidden_dim']
    O = cfg['n_actions']
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
# Agent forward pass (single agent)
# ---------------------------------------------------------------------------

def agent_step(obs_flat, hidden, params_flat, cfg):
    """One agent GRU step.

    Args:
        obs_flat: (obs_flat,) observation vector
        hidden:   (hidden_dim,) GRU hidden state
        params_flat: (n_params,) evolved parameter vector
        cfg: config dict

    Returns:
        new_hidden: (hidden_dim,)
        actions: (n_actions,) raw logits / continuous outputs
    """
    p = unpack_params(params_flat, cfg)

    # Embed observation
    x = jnp.tanh(obs_flat @ p['embed_W'] + p['embed_b'])  # (E,)

    # GRU
    xh = jnp.concatenate([x, hidden])  # (E+H,)
    z = jax.nn.sigmoid(xh @ p['gru_Wz'] + p['gru_bz'])
    r = jax.nn.sigmoid(xh @ p['gru_Wr'] + p['gru_br'])
    h_tilde = jnp.tanh(jnp.concatenate([x, r * hidden]) @ p['gru_Wh'] + p['gru_bh'])
    new_hidden = z * hidden + (1.0 - z) * h_tilde

    # Output
    actions = new_hidden @ p['out_W'] + p['out_b']  # (O,)
    return new_hidden, actions


# Vectorize over agents
agent_step_batch = jax.vmap(agent_step, in_axes=(0, 0, 0, None))


# ---------------------------------------------------------------------------
# Local observation gathering
# ---------------------------------------------------------------------------

def gather_patch(field, pos, radius):
    """Extract (2r+1)×(2r+1) patch from field with circular wrap.

    Args:
        field: (N, N)
        pos:   (2,) integer position [row, col]
        radius: int

    Returns: (2r+1, 2r+1)
    """
    N = field.shape[0]
    offsets = jnp.arange(-radius, radius + 1)
    rows = (pos[0] + offsets) % N
    cols = (pos[1] + offsets) % N
    return field[rows[:, None], cols[None, :]]


def build_observation(pos, resources, signals, agent_count_grid, energy, cfg):
    """Build flat observation vector for one agent.

    Returns: (obs_flat,) = flattened [R_patch, S_patch, N_patch] + energy
    """
    r = cfg['obs_radius']
    R_patch = gather_patch(resources, pos, r)       # (5,5)
    S_patch = gather_patch(signals, pos, r)          # (5,5)
    N_patch = gather_patch(agent_count_grid, pos, r)  # (5,5)
    patches = jnp.stack([R_patch, S_patch, N_patch], axis=-1)  # (5,5,3)
    return jnp.concatenate([patches.flatten(), jnp.array([energy])])  # (76,)


# Vectorize over agents
build_observation_batch = jax.vmap(
    build_observation, in_axes=(0, None, None, None, 0, None)
)


# ---------------------------------------------------------------------------
# Movement deltas
# ---------------------------------------------------------------------------
# 0=stay, 1=up, 2=down, 3=left, 4=right
MOVE_DELTAS = jnp.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.int32)


def decode_actions(raw_actions, cfg):
    """Decode raw GRU output into interpretable actions.

    Returns:
        move_idx: (M,) int32, 0-4
        consume: (M,) float, 0-1
        emit: (M,) float, 0-1
    """
    # Move: argmax of first 5 outputs
    move_idx = jnp.argmax(raw_actions[:, :5], axis=-1).astype(jnp.int32)
    # Consume and emit: sigmoid of last 2 outputs
    consume = jax.nn.sigmoid(raw_actions[:, 5])
    emit = jax.nn.sigmoid(raw_actions[:, 6])
    return move_idx, consume, emit


# ---------------------------------------------------------------------------
# Environment update helpers
# ---------------------------------------------------------------------------

def build_agent_count_grid(positions, alive, N):
    """Count agents at each grid cell."""
    count = jnp.zeros((N, N), dtype=jnp.float32)
    # Add 1 for each alive agent at their position
    count = count.at[positions[:, 0], positions[:, 1]].add(alive.astype(jnp.float32))
    return count


def apply_consumption(resources, positions, consume_amounts, alive, N):
    """Agents consume from their local resource pool.

    Each alive agent consumes consume_amount * resources[pos] resources.
    Returns updated resources and actual amounts consumed per agent.
    """
    # Gather local resource values
    local_resources = resources[positions[:, 0], positions[:, 1]]  # (M,)

    # Actual consumption (bounded by availability)
    actual_consume = consume_amounts * local_resources * alive  # (M,)

    # Scatter back (subtract from grid)
    new_resources = resources.at[positions[:, 0], positions[:, 1]].add(-actual_consume)
    new_resources = jnp.clip(new_resources, 0.0, 1.0)

    return new_resources, actual_consume


def apply_emissions(signals, positions, emit_amounts, alive, N):
    """Agents emit signals at their positions."""
    new_signals = signals.at[positions[:, 0], positions[:, 1]].add(
        emit_amounts * alive * 0.1  # Scale down emissions
    )
    return jnp.clip(new_signals, 0.0, 1.0)


def diffuse_signals(signals, cfg):
    """Diffuse signals with a 3×3 Gaussian kernel and apply decay."""
    # Simple 3×3 averaging kernel (approximates diffusion)
    kernel = jnp.array([
        [0.0625, 0.125, 0.0625],
        [0.125,  0.25,  0.125],
        [0.0625, 0.125, 0.0625],
    ])
    # Pad for circular boundary
    N = signals.shape[0]
    padded = jnp.pad(signals, 1, mode='wrap')
    # Manual convolution with 3×3 kernel
    new_s = jnp.zeros_like(signals)
    for di in range(3):
        for dj in range(3):
            new_s = new_s + kernel[di, dj] * padded[di:di+N, dj:dj+N]

    # Decay
    return new_s * (1.0 - cfg['signal_decay'])


def regen_resources(resources, regen_rate):
    """Logistic resource regeneration."""
    return resources + regen_rate * (1.0 - resources)


# ---------------------------------------------------------------------------
# Single environment step (JIT-compiled)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=['cfg_tuple'])
def env_step(state, cfg_tuple):
    """One step of the V20 environment.

    state keys:
        resources: (N, N)
        signals: (N, N)
        positions: (M, 2) int32
        hidden: (M, H)
        energy: (M,) float
        alive: (M,) bool
        params: (M, P) float — FIXED, not updated here
        regen_rate: float scalar
        step_count: int
    """
    cfg = dict(cfg_tuple)
    N = cfg['N']
    M = cfg['M_max']

    resources = state['resources']
    signals = state['signals']
    positions = state['positions']
    hidden = state['hidden']
    energy = state['energy']
    alive = state['alive']
    params = state['params']
    regen_rate = state['regen_rate']

    # 1. Build agent count grid
    agent_count = build_agent_count_grid(positions, alive, N)

    # 2. Build observations for all agents
    obs = build_observation_batch(positions, resources, signals, agent_count, energy, cfg)
    # obs: (M, obs_flat)

    # 3. Agent forward pass
    new_hidden, raw_actions = agent_step_batch(obs, hidden, params, cfg)
    # Zero out dead agents
    new_hidden = new_hidden * alive[:, None]

    # 4. Decode actions
    move_idx, consume, emit = decode_actions(raw_actions, cfg)

    # 5. Movement (only alive agents move)
    deltas = MOVE_DELTAS[move_idx]  # (M, 2)
    new_positions = (positions + deltas) % N
    new_positions = jnp.where(alive[:, None], new_positions, positions)

    # 6. Consumption
    resources, actual_consumed = apply_consumption(
        resources, new_positions,
        consume * alive, alive, N
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
        'regen_rate': regen_rate,
        'step_count': state['step_count'] + 1,
    }
    return new_state


# ---------------------------------------------------------------------------
# Chunk runner (lax.scan over chunk_size steps)
# ---------------------------------------------------------------------------

def make_chunk_runner(cfg):
    """Create JIT-compiled chunk runner for the given config."""
    cfg_tuple = tuple(sorted(cfg.items()))

    def scan_body(state, _):
        new_state = env_step(state, cfg_tuple)
        # Collect per-step metrics
        metrics = {
            'n_alive': jnp.sum(new_state['alive']).astype(jnp.float32),
            'mean_energy': jnp.mean(new_state['energy'] * new_state['alive']),
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
# Phi computation on agent hidden states
# ---------------------------------------------------------------------------

def compute_phi_hidden(hidden_states, alive):
    """Compute integration (Phi) approximation on GRU hidden states.

    Uses Gaussian MI approximation between first and second halves of
    the hidden state vector, averaged over alive agents.

    Args:
        hidden_states: (M, H) hidden states
        alive: (M,) bool mask

    Returns: float, mean Phi over alive agents
    """
    H = hidden_states.shape[1]
    half = H // 2

    alive_f = alive.astype(jnp.float32)
    n_alive = jnp.maximum(jnp.sum(alive_f), 1.0)

    # Phi per agent: Gaussian MI between first and second half
    top = hidden_states[:, :half]    # (M, H/2)
    bot = hidden_states[:, half:]    # (M, H/2)

    # Variance of each half
    var_top = jnp.var(top, axis=1) + 1e-8   # (M,)
    var_bot = jnp.var(bot, axis=1) + 1e-8   # (M,)

    # Covariance between halves (simplified: mean absolute correlation)
    top_c = top - jnp.mean(top, axis=1, keepdims=True)
    bot_c = bot - jnp.mean(bot, axis=1, keepdims=True)
    cov = jnp.mean(top_c * bot_c, axis=1) ** 2  # (M,)

    # MI ≈ -0.5 * log(1 - cov^2 / (var_top * var_bot))
    corr2 = cov / (var_top * var_bot)
    corr2 = jnp.clip(corr2, 0.0, 0.999)
    phi = -0.5 * jnp.log(1.0 - corr2 + 1e-8)

    # Mask and average
    return jnp.sum(phi * alive_f) / n_alive


def compute_robustness(phi_stress, phi_base):
    """Phi_stress / Phi_base (robustness > 1 means stress increases integration)."""
    return phi_stress / jnp.maximum(phi_base, 1e-6)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_v20(seed, cfg):
    """Initialize V20 state.

    Returns:
        state: dict with all environment and agent arrays
    """
    key = jax.random.PRNGKey(seed)
    N = cfg['N']
    M = cfg['M_max']
    H = cfg['hidden_dim']
    P = cfg['n_params']

    key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)

    # Environment
    resources = jax.random.uniform(k1, (N, N)) * cfg['initial_resource']
    resources = resources + jax.random.uniform(k2, (N, N)) * 0.3  # Some rich patches
    resources = jnp.clip(resources, 0.0, 1.0)
    signals = jnp.zeros((N, N))

    # Agent initial positions (scattered randomly)
    n_initial = M // 4  # Start with 1/4 capacity
    positions_init = jax.random.randint(k3, (n_initial, 2), 0, N)
    # Pad to M_max
    positions = jnp.zeros((M, 2), dtype=jnp.int32)
    positions = positions.at[:n_initial].set(positions_init)

    # Hidden states
    hidden = jnp.zeros((M, H))

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
        'regen_rate': jnp.array(cfg['resource_regen']),
        'step_count': jnp.array(0),
    }
    return state, key


# ---------------------------------------------------------------------------
# Snapshot: save current state for measurement experiments
# ---------------------------------------------------------------------------

def extract_snapshot(state, cycle, cfg):
    """Extract a lightweight snapshot for offline measurement."""
    alive = state['alive']
    return {
        'cycle': cycle,
        'hidden': np.array(state['hidden']),       # (M, H)
        'positions': np.array(state['positions']), # (M, 2)
        'energy': np.array(state['energy']),       # (M,)
        'alive': np.array(alive),                  # (M,)
        'params': np.array(state['params']),       # (M, P)
        'n_alive': int(jnp.sum(alive)),
    }
