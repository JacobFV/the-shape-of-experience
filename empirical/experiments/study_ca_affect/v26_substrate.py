"""V26: POMDP — Partial Observability Forces Internal Representation

V25 showed environmental complexity doesn't break the 1D collapse: agents
with 5×5 observation windows can SEE resources and predators directly, so
they never need to REMEMBER or MODEL anything.

V26 tests the hypothesis that partial observability forces richer hidden
states. Same V25 landscape (patchy resources, predator-prey), but:

  1. OBSERVATION = OWN CELL ONLY (1×1, not 5×5)
     - resource_here, signal_here, prey_count_here, pred_count_here = 4
     - energy, is_predator = 2
     - noisy_compass_to_nearest_patch (2D vector + noise) = 2
     - Total: 8 observation dimensions (vs 102 in V25)

  2. HIDDEN DIM = 32 (vs 16 in V25)
     - More capacity for belief state maintenance

  3. NOISY COMPASS: 2D unit vector toward nearest patch center, corrupted
     by Gaussian noise (σ=0.5). Gives direction but not distance. Forces
     the agent to maintain spatial belief about "where am I relative to food"
     rather than simply reacting to visible resources.

  4. EVERYTHING ELSE: Same as V25 (N=256, patchy resources, predator-prey,
     drought, type-specific evolution)

Success criteria (vs V25 baseline where ALL are 0):
  - Effective rank > 5
  - Position decode R² > 0.3
  - Resource decode R² > 0.2 (on-patch or not)
  - Energy decode R² < 0.8 (energy not the ONLY feature)

The key insight being tested: when the environment demands MEMORY for
survival (you can't see what's ahead), the hidden state must encode
more than just energy level.
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def generate_v26_config(**kwargs):
    """Generate V26 configuration."""
    cfg = {
        # Grid — same as V25
        'N': 256,
        'M_max': 512,

        # Agent architecture — CHANGED: smaller obs, bigger hidden
        'obs_radius': 0,       # 1×1 (own cell only)
        'embed_dim': 16,       # Smaller embedding (8 obs → 16)
        'hidden_dim': 32,      # DOUBLED hidden for belief state

        # Observation: own-cell (4) + energy + is_pred + compass (2) = 8
        'n_actions': 7,        # Same: 5 move + consume + emit

        # Population split
        'prey_fraction': 0.8,

        # Resource patches — same as V25
        'n_patches': 12,
        'patch_radius': 20,
        'patch_min_spacing': 60,
        'patch_regen': 0.003,
        'barren_regen': 0.0,

        # Compass noise
        'compass_noise_std': 0.5,  # Noise on the direction vector

        # Environment dynamics
        'signal_decay': 0.04,
        'signal_diffusion': 0.15,
        'initial_energy': 1.0,
        'resource_value': 1.5,
        'initial_resource': 0.6,

        # Metabolic costs
        'prey_metabolic': 0.0004,
        'prey_metabolic_barren': 0.001,
        'pred_metabolic': 0.0008,

        # Predation
        'predation_gain': 0.15,
        'predation_loss': 0.2,

        # Stress
        'stress_regen': 0.0003,

        # Evolution
        'mutation_std': 0.03,
        'tournament_size': 4,
        'elite_fraction': 0.5,
        'activate_offspring': True,

        # Drought
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
    obs_flat = 8  # resource, signal, prey_count, pred_count, energy, is_pred, compass_x, compass_y
    cfg['obs_flat'] = obs_flat

    M = cfg['M_max']
    cfg['n_prey'] = int(M * cfg['prey_fraction'])
    cfg['n_pred'] = M - cfg['n_prey']

    shapes = _param_shapes(cfg)
    total = sum(int(np.prod(s)) for s in shapes.values())
    cfg['n_params'] = total

    return cfg


def _param_shapes(cfg):
    """Parameter shapes. Same GRU architecture, different dimensions."""
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
# Agent forward pass
# ---------------------------------------------------------------------------

def agent_step(obs_flat, hidden, params_flat, cfg):
    """One agent GRU step."""
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
# Resource patch generation (same as V25)
# ---------------------------------------------------------------------------

def generate_patches(key, cfg):
    """Generate circular resource patches with minimum spacing."""
    N = cfg['N']
    n_patches = cfg['n_patches']
    radius = cfg['patch_radius']
    min_spacing = cfg['patch_min_spacing']

    seed_val = int(jax.random.key_data(key).flatten()[0]) % (2**31)
    rng = np.random.RandomState(seed_val)
    centers = []
    attempts = 0
    while len(centers) < n_patches and attempts < 10000:
        c = rng.randint(0, N, size=2)
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

    while len(centers) < n_patches:
        c = rng.randint(0, N, size=2)
        centers.append(c)

    centers = np.array(centers)

    mask = np.zeros((N, N), dtype=np.float32)
    yy, xx = np.mgrid[0:N, 0:N]
    for cx, cy in centers:
        dy = np.abs(yy - cx)
        dy = np.minimum(dy, N - dy)
        dx = np.abs(xx - cy)
        dx = np.minimum(dx, N - dx)
        dist = np.sqrt(dy**2 + dx**2)
        mask[dist <= radius] = 1.0

    return jnp.array(mask), jnp.array(centers, dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Compass computation
# ---------------------------------------------------------------------------

def compute_compass(positions, patch_centers, noise_key, cfg):
    """Compute noisy compass vector toward nearest patch for each agent.

    Returns (M, 2) unit vectors + Gaussian noise.
    """
    N = cfg['N']
    M = positions.shape[0]
    n_patches = patch_centers.shape[0]

    # Toroidal displacement from each agent to each patch center: (M, n_patches, 2)
    pos_expanded = positions[:, None, :]   # (M, 1, 2)
    pc_expanded = patch_centers[None, :, :]  # (1, n_patches, 2)

    delta = pc_expanded - pos_expanded  # (M, n_patches, 2)
    # Wrap to toroidal distance
    delta = jnp.where(delta > N // 2, delta - N, delta)
    delta = jnp.where(delta < -(N // 2), delta + N, delta)

    # Distance to each patch: (M, n_patches)
    dist = jnp.sqrt(jnp.sum(delta.astype(jnp.float32) ** 2, axis=-1) + 1e-6)

    # Index of nearest patch for each agent
    nearest_idx = jnp.argmin(dist, axis=1)  # (M,)

    # Get displacement to nearest patch
    nearest_delta = delta[jnp.arange(M), nearest_idx]  # (M, 2)

    # Normalize to unit vector
    nearest_dist = dist[jnp.arange(M), nearest_idx, None]  # (M, 1)
    compass = nearest_delta.astype(jnp.float32) / jnp.maximum(nearest_dist, 1.0)

    # Add noise
    noise = jax.random.normal(noise_key, (M, 2)) * cfg['compass_noise_std']
    compass = compass + noise

    # Clip to [-1, 1]
    return jnp.clip(compass, -2.0, 2.0)


# ---------------------------------------------------------------------------
# Observation building (1×1 + compass = 8 values)
# ---------------------------------------------------------------------------

def build_observation_v26(pos, resources, signals, prey_count, pred_count,
                          energy, is_pred, compass, cfg):
    """Build observation: own-cell (4) + energy + is_pred + compass (2) = 8."""
    r_here = resources[pos[0], pos[1]]
    s_here = signals[pos[0], pos[1]]
    p_here = prey_count[pos[0], pos[1]]
    d_here = pred_count[pos[0], pos[1]]
    return jnp.array([r_here, s_here, p_here, d_here, energy, is_pred,
                       compass[0], compass[1]])

build_observation_v26_batch = jax.vmap(
    build_observation_v26,
    in_axes=(0, None, None, None, None, 0, 0, 0, None)
)


# ---------------------------------------------------------------------------
# Movement and actions (same as V25)
# ---------------------------------------------------------------------------

MOVE_DELTAS = jnp.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.int32)

def decode_actions(raw_actions, cfg):
    """Decode raw output into move/consume/emit."""
    move_idx = jnp.argmax(raw_actions[:, :5], axis=-1).astype(jnp.int32)
    consume = jax.nn.sigmoid(raw_actions[:, 5])
    emit = jax.nn.sigmoid(raw_actions[:, 6])
    return move_idx, consume, emit


# ---------------------------------------------------------------------------
# Environment helpers (same as V25)
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
    """Only prey consume resources."""
    local_resources = resources[positions[:, 0], positions[:, 1]]
    can_consume = alive & (~is_predator)
    actual = consume_amounts * local_resources * can_consume
    new_resources = resources.at[positions[:, 0], positions[:, 1]].add(-actual)
    new_resources = jnp.clip(new_resources, 0.0, 1.0)
    return new_resources, actual


def apply_predation(energy, positions, alive, is_predator, cfg):
    """Predators on same cell as prey: predator gains, prey loses."""
    N = cfg['N']

    is_pred_alive = is_predator & alive
    is_prey_alive = (~is_predator) & alive

    prey_at = jnp.zeros((N, N), dtype=jnp.float32)
    prey_at = prey_at.at[positions[:, 0], positions[:, 1]].add(
        is_prey_alive.astype(jnp.float32))

    pred_at = jnp.zeros((N, N), dtype=jnp.float32)
    pred_at = pred_at.at[positions[:, 0], positions[:, 1]].add(
        is_pred_alive.astype(jnp.float32))

    prey_at_agent = prey_at[positions[:, 0], positions[:, 1]]
    pred_at_agent = pred_at[positions[:, 0], positions[:, 1]]

    pred_gain = is_pred_alive * prey_at_agent * cfg['predation_gain']
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
    """One step of V26 environment."""
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
    patch_centers = state['patch_centers']
    regen_rate = state['regen_rate']
    rng_key = state['rng_key']

    # Split key for compass noise
    rng_key, compass_key = jax.random.split(rng_key)

    # 1. Build type-specific count grids
    prey_count, pred_count = build_type_count_grids(positions, alive, is_predator, N)

    # 2. Compute noisy compass toward nearest patch
    compass = compute_compass(positions, patch_centers, compass_key, cfg)

    # 3. Build 1×1 observation + compass = 8 values
    is_pred_float = is_predator.astype(jnp.float32)
    obs = build_observation_v26_batch(
        positions, resources, signals, prey_count, pred_count,
        energy, is_pred_float, compass, cfg
    )

    # 4. Agent forward pass
    new_hidden, raw_actions = agent_step_batch(obs, hidden, params, cfg)
    new_hidden = new_hidden * alive[:, None]

    # 5. Decode actions
    move_idx, consume, emit = decode_actions(raw_actions, cfg)

    # 6. Movement
    deltas = MOVE_DELTAS[move_idx]
    new_positions = (positions + deltas) % N
    new_positions = jnp.where(alive[:, None], new_positions, positions)

    # 7. Prey consume resources
    resources, actual_consumed = apply_consumption(
        resources, new_positions, consume * alive, alive, is_predator, N
    )

    # 8. Predation
    energy_after_pred = apply_predation(
        energy, new_positions, alive, is_predator, cfg
    )

    # 9. Emissions
    signals = apply_emissions(signals, new_positions, emit, alive, N)

    # 10. Environment dynamics
    signals = diffuse_signals(signals, cfg)
    resources = regen_resources(resources, patch_mask, regen_rate)

    # 11. Energy update (metabolic varies by location)
    on_patch = patch_mask[new_positions[:, 0], new_positions[:, 1]]
    prey_cost = jnp.where(
        on_patch > 0.5,
        cfg['prey_metabolic'],
        cfg['prey_metabolic_barren']
    )
    metabolic = jnp.where(is_predator, cfg['pred_metabolic'], prey_cost)

    energy = energy_after_pred - metabolic + actual_consumed * cfg['resource_value']
    energy = jnp.clip(energy, 0.0, 2.0)

    # 12. Kill starved
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
        'patch_centers': patch_centers,
        'regen_rate': regen_rate,
        'rng_key': rng_key,
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
# Phi computation
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

def init_v26(seed, cfg):
    """Initialize V26 state."""
    key = jax.random.PRNGKey(seed)
    N = cfg['N']
    M = cfg['M_max']
    H = cfg['hidden_dim']
    P = cfg['n_params']
    n_prey = cfg['n_prey']
    n_pred = cfg['n_pred']

    key, k1, k2, k3, k4 = jax.random.split(key, 5)

    # Generate patches
    patch_mask, patch_centers = generate_patches(k1, cfg)

    # Resources: only within patches
    resources = jax.random.uniform(k2, (N, N)) * cfg['initial_resource'] * patch_mask
    resources = jnp.clip(resources, 0.0, 1.0)

    signals = jnp.zeros((N, N))

    # Agent types
    is_predator = jnp.zeros(M, dtype=jnp.bool_)
    is_predator = is_predator.at[n_prey:].set(True)

    # Start 1/4 of each type alive, placed on patches
    n_prey_init = n_prey // 4
    n_pred_init = n_pred // 4

    rng = np.random.RandomState(seed + 1000)
    centers_np = np.array(patch_centers)
    init_positions = np.zeros((M, 2), dtype=np.int32)

    for i in range(n_prey_init):
        patch_idx = rng.randint(0, len(centers_np))
        offset = rng.randint(-cfg['patch_radius']//2, cfg['patch_radius']//2, size=2)
        pos = (centers_np[patch_idx] + offset) % N
        init_positions[i] = pos

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
        'patch_centers': patch_centers,
        'regen_rate': jnp.array(cfg['patch_regen']),
        'rng_key': key,
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
