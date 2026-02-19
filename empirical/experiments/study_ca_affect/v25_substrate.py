"""V25: Predator-Prey on Structured Landscape

The 1D collapse finding (V20-V24 hidden states → energy counters) showed
the environment was the bottleneck, not the agent. V25 changes the WORLD:

  1. PATCHY RESOURCES: 12 circular patches on a 256×256 grid.
     Between patches: barren (no regen). Forces spatial encoding.
  2. PREDATORS: 20% of population hunts prey. Forces social modeling.
  3. NAVIGATION COST: Higher metabolic rate in barren zones.
     Forces route planning.
  4. PREY/PREDATOR CHANNELS: Observation includes agent-type counts.
     Forces type discrimination.

Same GRU architecture as V20b (H=16). The hypothesis: when the environment
demands spatial, social, and temporal richness, hidden state effective rank
should exceed 5 (vs 1-3 in V20-V24), and position/resource decoding R²
should rise well above 0.

Success criteria (vs V20-V24 baseline):
  - Effective rank > 5 (not 1-3)
  - Position decode R² > 0.3 (not ~0)
  - Resource decode R² > 0.2 (not ~0)
  - Energy decode R² < 0.8 (energy not the ONLY feature)
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def generate_v25_config(**kwargs):
    """Generate V25 configuration."""
    cfg = {
        # Grid — larger for spatial structure
        'N': 256,
        'M_max': 512,         # More agents for larger grid

        # Agent architecture (SAME as V20b)
        'obs_radius': 2,      # 5×5 patch
        'embed_dim': 24,
        'hidden_dim': 16,

        # Observation: 5×5 × 4 channels (R, signals, prey_count, pred_count) + energy + is_predator
        # = 5*5*4 + 2 = 102
        'n_actions': 7,       # Same: 5 move + consume + emit

        # Population split
        'prey_fraction': 0.8,    # 80% prey, 20% predators

        # Resource patches
        'n_patches': 12,         # Number of resource patches
        'patch_radius': 20,      # Radius of each patch (cells)
        'patch_min_spacing': 60, # Minimum distance between patch centers
        'patch_regen': 0.003,    # Regen rate WITHIN patches (slightly higher than V20b)
        'barren_regen': 0.0,     # No regen outside patches

        # Environment dynamics
        'signal_decay': 0.04,
        'signal_diffusion': 0.15,
        'initial_energy': 1.0,
        'resource_value': 1.5,
        'initial_resource': 0.6,  # Within patches

        # Metabolic costs
        'prey_metabolic': 0.0004,       # Same as V20b in patches
        'prey_metabolic_barren': 0.001, # 2.5× in barren zones
        'pred_metabolic': 0.0008,       # Predators more expensive

        # Predator-prey interaction
        'predation_gain': 0.15,   # Energy predator gains per prey on same cell
        'predation_loss': 0.2,    # Energy prey loses per predator on same cell

        # Stress
        'stress_regen': 0.0003,

        # Evolution (separate for prey and predators)
        'mutation_std': 0.03,
        'tournament_size': 4,
        'elite_fraction': 0.5,
        'activate_offspring': True,

        # Drought schedule
        'drought_every': 5,
        'drought_depletion': 0.01,
        'drought_regen': 0.0,

        # Run
        'chunk_size': 50,
        'steps_per_cycle': 5000,
        'n_cycles': 30,
    }
    cfg.update(kwargs)

    # Derived
    obs_side = 2 * cfg['obs_radius'] + 1  # 5
    # 4 channels: resources, signals, prey_count, predator_count
    obs_flat = obs_side * obs_side * 4 + 2  # +energy +is_predator = 102
    cfg['obs_side'] = obs_side
    cfg['obs_flat'] = obs_flat

    # Population split
    M = cfg['M_max']
    cfg['n_prey'] = int(M * cfg['prey_fraction'])
    cfg['n_pred'] = M - cfg['n_prey']

    # Parameter counts
    shapes = _param_shapes(cfg)
    total = sum(int(np.prod(s)) for s in shapes.values())
    cfg['n_params'] = total

    return cfg


def _param_shapes(cfg):
    """Ordered dict of parameter name → shape. Same GRU architecture."""
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
# Agent forward pass (identical GRU architecture to V20b)
# ---------------------------------------------------------------------------

def agent_step(obs_flat, hidden, params_flat, cfg):
    """One agent GRU step. Same architecture as V20b."""
    p = unpack_params(params_flat, cfg)
    x = jnp.tanh(obs_flat @ p['embed_W'] + p['embed_b'])
    xh = jnp.concatenate([x, hidden])
    z = jax.nn.sigmoid(xh @ p['gru_Wz'] + p['gru_bz'])
    r = jax.nn.sigmoid(xh @ p['gru_Wr'] + p['gru_br'])
    h_tilde = jnp.tanh(jnp.concatenate([x, r * hidden]) @ p['gru_Wh'] + p['gru_bh'])
    new_hidden = z * hidden + (1.0 - z) * h_tilde
    actions = new_hidden @ p['out_W'] + p['out_b']
    return new_hidden, actions

agent_step_batch = jax.vmap(agent_step, in_axes=(0, 0, 0, None))


# ---------------------------------------------------------------------------
# Resource patch generation
# ---------------------------------------------------------------------------

def generate_patches(key, cfg):
    """Generate circular resource patches with minimum spacing.

    Returns:
        patch_mask: (N, N) float32, 1.0 inside patches, 0.0 outside
        patch_centers: (n_patches, 2) int32
    """
    N = cfg['N']
    n_patches = cfg['n_patches']
    radius = cfg['patch_radius']
    min_spacing = cfg['patch_min_spacing']

    # Use numpy for placement (not JIT'd — done once at init)
    seed_val = int(jax.random.key_data(key).flatten()[0]) % (2**31)
    rng = np.random.RandomState(seed_val)
    centers = []
    attempts = 0
    while len(centers) < n_patches and attempts < 10000:
        c = rng.randint(0, N, size=2)
        # Check minimum spacing (toroidal distance)
        ok = True
        for existing in centers:
            d = np.abs(c - existing)
            d = np.minimum(d, N - d)
            dist = np.sqrt(np.sum(d**2))
            if dist < min_spacing:
                ok = False
                break
        if ok:
            centers.append(c)
        attempts += 1

    if len(centers) < n_patches:
        # Fallback: place remaining randomly
        while len(centers) < n_patches:
            c = rng.randint(0, N, size=2)
            centers.append(c)

    centers = np.array(centers)  # (n_patches, 2)

    # Build mask
    mask = np.zeros((N, N), dtype=np.float32)
    yy, xx = np.mgrid[0:N, 0:N]
    for cx, cy in centers:
        # Toroidal distance
        dy = np.abs(yy - cx)
        dy = np.minimum(dy, N - dy)
        dx = np.abs(xx - cy)
        dx = np.minimum(dx, N - dx)
        dist = np.sqrt(dy**2 + dx**2)
        mask[dist <= radius] = 1.0

    return jnp.array(mask), jnp.array(centers, dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Observation building (4 channels + energy + is_predator)
# ---------------------------------------------------------------------------

def gather_patch(field, pos, radius):
    """Extract (2r+1)×(2r+1) patch with circular wrap."""
    N = field.shape[0]
    offsets = jnp.arange(-radius, radius + 1)
    rows = (pos[0] + offsets) % N
    cols = (pos[1] + offsets) % N
    return field[rows[:, None], cols[None, :]]


def build_observation(pos, resources, signals, prey_count, pred_count, energy, is_pred, cfg):
    """Build observation: 4 channels × 5×5 + energy + is_predator = 102."""
    r = cfg['obs_radius']
    R_patch = gather_patch(resources, pos, r)
    S_patch = gather_patch(signals, pos, r)
    P_patch = gather_patch(prey_count, pos, r)
    D_patch = gather_patch(pred_count, pos, r)
    patches = jnp.stack([R_patch, S_patch, P_patch, D_patch], axis=-1)  # (5,5,4)
    return jnp.concatenate([patches.flatten(), jnp.array([energy, is_pred])])  # (102,)

build_observation_batch = jax.vmap(
    build_observation, in_axes=(0, None, None, None, None, 0, 0, None)
)


# ---------------------------------------------------------------------------
# Movement and actions
# ---------------------------------------------------------------------------

MOVE_DELTAS = jnp.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.int32)

def decode_actions(raw_actions, cfg):
    """Decode raw output into move/consume/emit."""
    move_idx = jnp.argmax(raw_actions[:, :5], axis=-1).astype(jnp.int32)
    consume = jax.nn.sigmoid(raw_actions[:, 5])
    emit = jax.nn.sigmoid(raw_actions[:, 6])
    return move_idx, consume, emit


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def build_type_count_grids(positions, alive, is_predator, N):
    """Build separate count grids for prey and predators."""
    is_prey = ~is_predator & alive
    is_pred_alive = is_predator & alive

    prey_count = jnp.zeros((N, N), dtype=jnp.float32)
    pred_count = jnp.zeros((N, N), dtype=jnp.float32)

    prey_count = prey_count.at[positions[:, 0], positions[:, 1]].add(
        is_prey.astype(jnp.float32))
    pred_count = pred_count.at[positions[:, 0], positions[:, 1]].add(
        is_pred_alive.astype(jnp.float32))

    return prey_count, pred_count


def apply_consumption(resources, positions, consume_amounts, alive, is_predator, N):
    """Only prey consume resources. Predators cannot eat resources."""
    local_resources = resources[positions[:, 0], positions[:, 1]]
    # Only prey consume
    can_consume = alive & (~is_predator)
    actual = consume_amounts * local_resources * can_consume
    new_resources = resources.at[positions[:, 0], positions[:, 1]].add(-actual)
    new_resources = jnp.clip(new_resources, 0.0, 1.0)
    return new_resources, actual


def apply_predation(energy, positions, alive, is_predator, cfg):
    """Predators on same cell as prey: predator gains, prey loses energy."""
    N = cfg['N']
    M = cfg['M_max']

    # Build predator and prey count grids
    is_pred_alive = is_predator & alive
    is_prey_alive = (~is_predator) & alive

    # Count prey at each cell
    prey_at = jnp.zeros((N, N), dtype=jnp.float32)
    prey_at = prey_at.at[positions[:, 0], positions[:, 1]].add(
        is_prey_alive.astype(jnp.float32))

    # Count predators at each cell
    pred_at = jnp.zeros((N, N), dtype=jnp.float32)
    pred_at = pred_at.at[positions[:, 0], positions[:, 1]].add(
        is_pred_alive.astype(jnp.float32))

    # For each agent, look up how many of the other type are at their cell
    prey_at_agent = prey_at[positions[:, 0], positions[:, 1]]  # (M,)
    pred_at_agent = pred_at[positions[:, 0], positions[:, 1]]  # (M,)

    # Predators gain energy proportional to prey present
    pred_gain = is_pred_alive * prey_at_agent * cfg['predation_gain']

    # Prey lose energy proportional to predators present
    prey_loss = is_prey_alive * pred_at_agent * cfg['predation_loss']

    energy = energy + pred_gain - prey_loss
    return jnp.clip(energy, 0.0, 2.0)


def apply_emissions(signals, positions, emit_amounts, alive, N):
    """Agents emit signals."""
    new_signals = signals.at[positions[:, 0], positions[:, 1]].add(
        emit_amounts * alive * 0.1)
    return jnp.clip(new_signals, 0.0, 1.0)


def diffuse_signals(signals, cfg):
    """3×3 Gaussian diffusion + decay."""
    kernel = jnp.array([
        [0.0625, 0.125, 0.0625],
        [0.125,  0.25,  0.125],
        [0.0625, 0.125, 0.0625],
    ])
    N = signals.shape[0]
    padded = jnp.pad(signals, 1, mode='wrap')
    new_s = jnp.zeros_like(signals)
    for di in range(3):
        for dj in range(3):
            new_s = new_s + kernel[di, dj] * padded[di:di+N, dj:dj+N]
    return new_s * (1.0 - cfg['signal_decay'])


def regen_resources(resources, patch_mask, regen_rate):
    """Logistic regen ONLY within patches."""
    regen = regen_rate * patch_mask * (1.0 - resources)
    return resources + regen


# ---------------------------------------------------------------------------
# Single environment step
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=['cfg_tuple'])
def env_step(state, cfg_tuple):
    """One step of V25 environment."""
    cfg = dict(cfg_tuple)
    N = cfg['N']

    resources = state['resources']
    signals = state['signals']
    positions = state['positions']
    hidden = state['hidden']
    energy = state['energy']
    alive = state['alive']
    is_predator = state['is_predator']
    params = state['params']
    patch_mask = state['patch_mask']
    regen_rate = state['regen_rate']

    # 1. Build type-specific count grids
    prey_count, pred_count = build_type_count_grids(positions, alive, is_predator, N)

    # 2. Observations (4 channels + energy + is_predator)
    is_pred_float = is_predator.astype(jnp.float32)
    obs = build_observation_batch(
        positions, resources, signals, prey_count, pred_count,
        energy, is_pred_float, cfg
    )

    # 3. Agent forward pass
    new_hidden, raw_actions = agent_step_batch(obs, hidden, params, cfg)
    new_hidden = new_hidden * alive[:, None]

    # 4. Decode actions
    move_idx, consume, emit = decode_actions(raw_actions, cfg)

    # 5. Movement
    deltas = MOVE_DELTAS[move_idx]
    new_positions = (positions + deltas) % N
    new_positions = jnp.where(alive[:, None], new_positions, positions)

    # 6. Prey consume resources (predators can't)
    resources, actual_consumed = apply_consumption(
        resources, new_positions, consume * alive, alive, is_predator, N
    )

    # 7. Predation: predators gain from prey on same cell
    energy_after_pred = apply_predation(
        energy, new_positions, alive, is_predator, cfg
    )

    # 8. Emissions
    signals = apply_emissions(signals, new_positions, emit, alive, N)

    # 9. Environment dynamics
    signals = diffuse_signals(signals, cfg)
    resources = regen_resources(resources, patch_mask, regen_rate)

    # 10. Energy update
    # Metabolic cost varies: prey in barren zones pay more
    on_patch = patch_mask[new_positions[:, 0], new_positions[:, 1]]  # (M,)
    prey_cost = jnp.where(
        on_patch > 0.5,
        cfg['prey_metabolic'],
        cfg['prey_metabolic_barren']
    )
    metabolic = jnp.where(is_predator, cfg['pred_metabolic'], prey_cost)

    energy = energy_after_pred - metabolic + actual_consumed * cfg['resource_value']
    energy = jnp.clip(energy, 0.0, 2.0)

    # 11. Kill starved
    new_alive = alive & (energy > 0.0)

    return {
        'resources': resources,
        'signals': signals,
        'positions': new_positions,
        'hidden': new_hidden,
        'energy': energy,
        'alive': new_alive,
        'is_predator': is_predator,
        'params': params,
        'patch_mask': patch_mask,
        'regen_rate': regen_rate,
        'step_count': state['step_count'] + 1,
    }


# ---------------------------------------------------------------------------
# Chunk runner
# ---------------------------------------------------------------------------

def make_chunk_runner(cfg):
    """Create JIT-compiled chunk runner."""
    cfg_tuple = tuple(sorted(cfg.items()))

    def scan_body(state, _):
        new_state = env_step(state, cfg_tuple)
        metrics = {
            'n_alive': jnp.sum(new_state['alive']).astype(jnp.float32),
            'n_prey_alive': jnp.sum(new_state['alive'] & (~new_state['is_predator'])).astype(jnp.float32),
            'n_pred_alive': jnp.sum(new_state['alive'] & new_state['is_predator']).astype(jnp.float32),
            'mean_energy': jnp.mean(new_state['energy'] * new_state['alive']),
        }
        return new_state, metrics

    @jax.jit
    def run_chunk(state):
        final_state, metrics = lax.scan(
            scan_body, state, None, length=cfg['chunk_size']
        )
        return final_state, metrics

    return run_chunk


# ---------------------------------------------------------------------------
# Phi computation (same as V20b)
# ---------------------------------------------------------------------------

def compute_phi_hidden(hidden_states, alive):
    """Gaussian MI approximation between hidden state halves."""
    H = hidden_states.shape[1]
    half = H // 2
    alive_f = alive.astype(jnp.float32)
    n_alive = jnp.maximum(jnp.sum(alive_f), 1.0)

    top = hidden_states[:, :half]
    bot = hidden_states[:, half:]
    var_top = jnp.var(top, axis=1) + 1e-8
    var_bot = jnp.var(bot, axis=1) + 1e-8
    top_c = top - jnp.mean(top, axis=1, keepdims=True)
    bot_c = bot - jnp.mean(bot, axis=1, keepdims=True)
    cov = jnp.mean(top_c * bot_c, axis=1) ** 2
    corr2 = cov / (var_top * var_bot)
    corr2 = jnp.clip(corr2, 0.0, 0.999)
    phi = -0.5 * jnp.log(1.0 - corr2 + 1e-8)
    return jnp.sum(phi * alive_f) / n_alive


def compute_robustness(phi_stress, phi_base):
    return phi_stress / jnp.maximum(phi_base, 1e-6)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_v25(seed, cfg):
    """Initialize V25 state with patchy resources and two agent types."""
    key = jax.random.PRNGKey(seed)
    N = cfg['N']
    M = cfg['M_max']
    H = cfg['hidden_dim']
    P = cfg['n_params']
    n_prey = cfg['n_prey']
    n_pred = cfg['n_pred']

    key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)

    # Generate resource patches
    patch_mask, patch_centers = generate_patches(k1, cfg)

    # Initialize resources: only within patches
    resources = jax.random.uniform(k2, (N, N)) * cfg['initial_resource'] * patch_mask
    resources = jnp.clip(resources, 0.0, 1.0)

    signals = jnp.zeros((N, N))

    # Agent types: first n_prey are prey, rest are predators
    is_predator = jnp.zeros(M, dtype=jnp.bool_)
    is_predator = is_predator.at[n_prey:].set(True)

    # Start 1/4 of each type alive, placed ON patches (not randomly)
    n_prey_init = n_prey // 4
    n_pred_init = n_pred // 4
    n_init = n_prey_init + n_pred_init

    # Place prey near random patches, predators near other patches
    key, kpos = jax.random.split(key)
    # Use numpy for initial placement
    rng = np.random.RandomState(seed + 1000)
    centers_np = np.array(patch_centers)
    init_positions = np.zeros((M, 2), dtype=np.int32)

    # Place prey on patches
    for i in range(n_prey_init):
        patch_idx = rng.randint(0, len(centers_np))
        offset = rng.randint(-cfg['patch_radius']//2, cfg['patch_radius']//2, size=2)
        pos = (centers_np[patch_idx] + offset) % N
        init_positions[i] = pos

    # Place predators on patches too (they hunt where prey are)
    for i in range(n_pred_init):
        patch_idx = rng.randint(0, len(centers_np))
        offset = rng.randint(-cfg['patch_radius']//2, cfg['patch_radius']//2, size=2)
        pos = (centers_np[patch_idx] + offset) % N
        init_positions[n_prey + i] = pos

    positions = jnp.array(init_positions)
    hidden = jnp.zeros((M, H))

    energy = jnp.zeros(M)
    energy = energy.at[:n_prey_init].set(cfg['initial_energy'])
    energy = energy.at[n_prey:n_prey + n_pred_init].set(cfg['initial_energy'])

    alive = jnp.zeros(M, dtype=jnp.bool_)
    alive = alive.at[:n_prey_init].set(True)
    alive = alive.at[n_prey:n_prey + n_pred_init].set(True)

    # Parameters
    params = jax.random.normal(k4, (M, P)) * 0.1

    state = {
        'resources': resources,
        'signals': signals,
        'positions': positions,
        'hidden': hidden,
        'energy': energy,
        'alive': alive,
        'is_predator': is_predator,
        'params': params,
        'patch_mask': patch_mask,
        'regen_rate': jnp.array(cfg['patch_regen']),
        'step_count': jnp.array(0),
    }
    return state, key, patch_centers


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

def extract_snapshot(state, cycle, cfg):
    """Extract snapshot for offline analysis."""
    alive = state['alive']
    return {
        'cycle': cycle,
        'hidden': np.array(state['hidden']),
        'positions': np.array(state['positions']),
        'energy': np.array(state['energy']),
        'alive': np.array(alive),
        'is_predator': np.array(state['is_predator']),
        'params': np.array(state['params']),
        'resources': np.array(state['resources']),
        'signals': np.array(state['signals']),
        'patch_mask': np.array(state['patch_mask']),
        'n_alive': int(jnp.sum(alive)),
        'n_prey_alive': int(jnp.sum(alive & (~state['is_predator']))),
        'n_pred_alive': int(jnp.sum(alive & state['is_predator'])),
    }
