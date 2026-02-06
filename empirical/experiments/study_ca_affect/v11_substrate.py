"""V11 Substrate: Lenia CA with non-equilibrium resource dynamics.

This is the PHYSICS, not the architecture. We define:
- State space: continuous [0,1] on 2D toroidal grid
- Dynamics: Lenia update rules (local convolution + growth function)
- Resources: field that depletes under pattern activity and regenerates
- Noise: thermal fluctuations for genuine non-equilibrium

Patterns emerge from these dynamics or they don't. We impose nothing.
"""

import jax
import jax.numpy as jnp
from jax import random, jit, lax
import numpy as np
from functools import partial


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    'grid_size': 256,
    'dt': 0.1,

    # Kernel: Gaussian ring
    'kernel_radius': 13,
    'kernel_peak': 0.5,       # normalized radial distance of ring peak
    'kernel_width': 0.15,     # Gaussian width of ring

    # Growth function: bell curve mapping potential -> growth rate
    'growth_mu': 0.15,        # optimal neighborhood potential
    'growth_sigma': 0.035,    # tolerance width (wide enough for emergence)

    # Background decay
    'decay_rate': 0.0,        # no constant decay (organisms self-stabilize)

    # Resource dynamics
    'resource_max': 1.0,
    'resource_regen': 0.005,  # moderate regeneration
    'resource_consume': 0.03, # moderate consumption
    'resource_half_sat': 0.2, # Michaelis-Menten half-saturation

    # Non-equilibrium driving
    'noise_amp': 0.003,       # thermal noise amplitude
}

# Wider growth_sigma for easier emergence (fallback if nothing survives)
FORGIVING_CONFIG = {**DEFAULT_CONFIG, 'growth_sigma': 0.05, 'noise_amp': 0.005,
                    'resource_consume': 0.02, 'resource_regen': 0.01}

# No resources, no decay (ablation baseline: pure Lenia)
NO_RESOURCE_CONFIG = {**DEFAULT_CONFIG, 'resource_consume': 0.0, 'resource_regen': 1.0,
                      'decay_rate': 0.0}

# Harsh environment (strong selection)
HARSH_CONFIG = {**DEFAULT_CONFIG, 'resource_consume': 0.08, 'resource_regen': 0.001,
                'decay_rate': 0.08}


# ============================================================================
# Kernel
# ============================================================================

def make_kernel(R, peak, width):
    """Gaussian ring convolution kernel.

    This defines the neighborhood structure: how nearby cells influence
    each other. A ring kernel means cells care about a specific distance
    band, not just immediate neighbors.
    """
    y, x = jnp.mgrid[-R:R+1, -R:R+1]
    r = jnp.sqrt(x**2 + y**2) / R  # normalize to [0, 1]
    k = jnp.exp(-((r - peak)**2) / (2 * width**2))
    k = jnp.where(r <= 1.0, k, 0.0)
    return k / (jnp.sum(k) + 1e-10)


def make_kernel_fft(kernel, N):
    """Pre-compute kernel FFT for O(N log N) convolution."""
    kH, kW = kernel.shape
    padded = jnp.zeros((N, N))
    padded = padded.at[:kH, :kW].set(kernel)
    padded = jnp.roll(padded, (-kH // 2, -kW // 2), axis=(0, 1))
    return jnp.fft.rfft2(padded)


# ============================================================================
# Core Physics
# ============================================================================

def growth_fn(u, mu, sigma):
    """Lenia growth function: bell curve centered at mu.

    Returns values in [-1, +1]:
      +1 when u ~ mu  (optimal neighborhood density -> growth)
      -1 when u far from mu  (too sparse or dense -> decay)

    This IS the viability manifold in microcosm.
    """
    return 2.0 * jnp.exp(-((u - mu)**2) / (2 * sigma**2)) - 1.0


def _step_inner(grid, resource, kernel_fft, rng, dt, growth_mu, growth_sigma,
                noise_amp, resource_consume, resource_regen,
                resource_max, resource_half_sat, decay_rate):
    """Single timestep of substrate physics (no @jit, used inside scan)."""
    rng, k_noise = random.split(rng)

    # 1. Neighborhood potential via FFT convolution (periodic boundaries)
    potential = jnp.fft.irfft2(jnp.fft.rfft2(grid) * kernel_fft, s=grid.shape)

    # 2. Growth rate from potential
    g = growth_fn(potential, growth_mu, growth_sigma)

    # 3. Resource modulation: positive growth requires resources
    #    Decay is free (second law doesn't need fuel)
    rf = resource / (resource + resource_half_sat)
    g = jnp.where(g > 0, g * rf, g)

    # 3b. Background decay: cells die without active resource-fueled growth
    #     This is entropy: maintaining structure costs energy
    g = g - decay_rate

    # 4. State update + thermal noise
    new_grid = grid + dt * g + noise_amp * random.normal(k_noise, grid.shape)
    new_grid = jnp.clip(new_grid, 0.0, 1.0)

    # 5. Resource dynamics: consumption + logistic regeneration
    new_resource = resource \
        - resource_consume * grid * resource * dt \
        + resource_regen * (resource_max - resource) * dt
    new_resource = jnp.clip(new_resource, 0.0, resource_max)

    return new_grid, new_resource, rng


def _step_with_season(grid, resource, kernel_fft, rng, step_num,
                      dt, growth_mu, growth_sigma, noise_amp,
                      resource_consume, resource_regen,
                      resource_max, resource_half_sat, decay_rate):
    """Step + seasonal resource drift for long runs."""
    new_grid, new_resource, rng = _step_inner(
        grid, resource, kernel_fft, rng,
        dt, growth_mu, growth_sigma, noise_amp,
        resource_consume, resource_regen, resource_max, resource_half_sat,
        decay_rate
    )

    # Slowly shift resource distribution (seasonal effect)
    N = grid.shape[0]
    season_period = 5000.0
    phase = 2.0 * jnp.pi * step_num / season_period

    # Resource target: hotspots that rotate around the grid
    y_idx = jnp.arange(N)
    x_idx = jnp.arange(N)
    yy, xx = jnp.meshgrid(y_idx, x_idx, indexing='ij')

    cx = N * (0.5 + 0.25 * jnp.sin(phase))
    cy = N * (0.5 + 0.25 * jnp.cos(phase))
    dist = jnp.sqrt((xx - cx)**2 + (yy - cy)**2)
    target = 0.3 + 0.7 * jnp.exp(-(dist**2) / (2 * 40**2))

    # Gently pull resource toward seasonal target
    new_resource = new_resource + 0.0005 * (target - new_resource)
    new_resource = jnp.clip(new_resource, 0.0, resource_max)

    return new_grid, new_resource, rng


def make_params(config):
    """Convert config dict to JIT-friendly param arrays."""
    keys = ['dt', 'growth_mu', 'growth_sigma', 'noise_amp',
            'resource_consume', 'resource_regen', 'resource_max',
            'resource_half_sat', 'decay_rate']
    return {k: jnp.float32(config[k]) for k in keys}


# ============================================================================
# Perturbations: environmental disruptions for affect testing
# ============================================================================

def perturb_resource_crash(resource, center, radius, severity=0.9):
    """Crash resources in a circular region (drought/disaster).

    Patterns in this region lose access to resources.
    Should produce: negative valence, high arousal, integration spike.
    """
    N = resource.shape[0]
    yy, xx = jnp.mgrid[0:N, 0:N]
    dist = jnp.sqrt((xx - center[0])**2 + (yy - center[1])**2)
    mask = (dist < radius).astype(jnp.float32)
    return resource * (1.0 - severity * mask)


def perturb_kill_zone(grid, center, radius):
    """Kill all cells in a circular region (catastrophe).

    Should produce: mass death, negative valence for survivors nearby,
    potential positive valence during recovery.
    """
    N = grid.shape[0]
    yy, xx = jnp.mgrid[0:N, 0:N]
    dist = jnp.sqrt((xx - center[0])**2 + (yy - center[1])**2)
    mask = (dist < radius).astype(jnp.float32)
    return grid * (1.0 - mask)


def perturb_noise_burst(grid, rng, amplitude=0.3):
    """Global noise burst (environmental shock).

    Disrupts all patterns simultaneously. Should produce:
    high arousal everywhere, integration test (fragile patterns break).
    """
    return jnp.clip(grid + amplitude * random.normal(rng, grid.shape), 0.0, 1.0)


def perturb_resource_bloom(resource, center, radius, intensity=1.0):
    """Create resource bloom in a region (opportunity).

    Should produce: positive valence for nearby patterns,
    possible migration toward bloom.
    """
    N = resource.shape[0]
    yy, xx = jnp.mgrid[0:N, 0:N]
    dist = jnp.sqrt((xx - center[0])**2 + (yy - center[1])**2)
    bloom = intensity * jnp.exp(-(dist**2) / (2 * radius**2))
    return jnp.clip(resource + bloom, 0.0, 1.0)


@jit
def step(grid, resource, kernel_fft, rng, params):
    """Single timestep (JIT-compiled, for interactive use)."""
    return _step_inner(grid, resource, kernel_fft, rng, **params)


@partial(jit, static_argnums=(5,))
def run_chunk(grid, resource, kernel_fft, rng, params, n_steps):
    """Run n_steps on GPU without returning intermediates.

    Uses lax.scan for efficient compilation. Returns only final state.
    """
    def body(carry, _):
        g, r, k = carry
        g, r, k = _step_inner(g, r, kernel_fft, k, **params)
        return (g, r, k), None

    (grid, resource, rng), _ = lax.scan(
        body, (grid, resource, rng), None, length=n_steps
    )
    return grid, resource, rng


# ============================================================================
# Initialization
# ============================================================================

def init_soup(N, rng, n_seeds=50, growth_mu=0.15):
    """Random soup: maximally uncontaminated initialization.

    Scatters random circular blobs with densities centered around
    growth_mu so that some regions start in the viable band.
    Most will die immediately. What survives is genuinely emergent.
    """
    k1, k2 = random.split(rng)
    # Low background noise
    grid = 0.02 * random.uniform(k1, (N, N))

    yy, xx = np.mgrid[0:N, 0:N]
    for _ in range(n_seeds):
        ks, k2 = random.split(k2)
        keys = random.split(ks, 5)
        cx = int(random.randint(keys[0], (), 20, N - 20))
        cy = int(random.randint(keys[1], (), 20, N - 20))
        r = float(random.uniform(keys[2], (), minval=6, maxval=18))
        # Center blob values around growth_mu for viability
        center_val = growth_mu + float(random.uniform(keys[3], (),
                                        minval=-0.05, maxval=0.05))
        # Gaussian blob profile (smooth edges)
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        blob = center_val * np.exp(-(dist**2) / (2 * (r * 0.6)**2))
        # Add some asymmetry for interesting dynamics
        angle = np.arctan2(yy - cy, xx - cx)
        asym = 1.0 + 0.2 * float(random.uniform(keys[4], (),
                                   minval=-1, maxval=1)) * np.sin(angle)
        blob = blob * asym
        grid = grid + jnp.array(blob)

    grid = jnp.clip(grid, 0.0, 1.0)

    # Resource gradient: concentrated in patches, sparse elsewhere
    resource = jnp.full((N, N), 0.3)  # low baseline
    for hx, hy in [(N//3, N//3), (2*N//3, 2*N//3),
                    (N//3, 2*N//3), (2*N//3, N//3)]:
        dist_h = np.sqrt((xx - hx)**2 + (yy - hy)**2)
        resource = resource + jnp.array(0.7 * np.exp(-(dist_h**2) / (2 * 30**2)))
    resource = jnp.clip(resource, 0.0, 1.0)

    return grid, resource


def init_orbium_seeds(N):
    """Place orbium-like seeds for testing.

    Creates asymmetric ring blobs tuned for standard Lenia parameters.
    Not guaranteed to produce true orbiums, but should produce
    some kind of surviving structure.
    """
    grid = jnp.zeros((N, N))
    yy, xx = jnp.mgrid[0:N, 0:N]

    centers = [(N//4, N//4), (3*N//4, N//4), (N//2, 3*N//4),
               (N//4, 3*N//4), (3*N//4, 3*N//4)]

    for cx, cy in centers:
        dist = jnp.sqrt((xx - cx)**2 + (yy - cy)**2)
        # Gaussian ring at radius ~8, width ~3
        ring = jnp.exp(-((dist - 8)**2) / (2 * 3**2))
        # Asymmetry for directionality
        angle = jnp.arctan2(yy - cy, xx - cx)
        asym = 1.0 + 0.3 * jnp.sin(angle)
        seed = ring * asym * 0.35
        seed = jnp.where(dist < 16, seed, 0.0)
        grid = grid + seed

    grid = jnp.clip(grid, 0.0, 1.0)
    resource = jnp.full((N, N), 1.0)
    return grid, resource


# ============================================================================
# Heterogeneous Chemistry (V11.2)
# ============================================================================

def init_param_fields(N, rng, base_mu=0.15, base_sigma=0.035, n_zones=8,
                      mu_range=(0.08, 0.25), sigma_range=(0.02, 0.06)):
    """Initialize spatially diverse growth parameter fields.

    Creates n_zones circular patches with different chemistry,
    smoothly blended into the base parameters. Patterns in different
    zones have different viability manifolds — the key ingredient
    for heritable variation in integration response.

    Returns (mu_field, sigma_field), each NxN arrays.
    """
    mu_field = jnp.full((N, N), base_mu)
    sigma_field = jnp.full((N, N), base_sigma)

    yy, xx = np.mgrid[0:N, 0:N]

    for i in range(n_zones):
        rng, k1, k2, k3, k4, k5 = random.split(rng, 6)
        cx = int(random.randint(k1, (), 30, N - 30))
        cy = int(random.randint(k2, (), 30, N - 30))
        radius = float(random.uniform(k3, (), minval=20, maxval=50))

        zone_mu = float(random.uniform(
            k4, (), minval=mu_range[0], maxval=mu_range[1]))
        zone_sigma = float(random.uniform(
            k5, (), minval=sigma_range[0], maxval=sigma_range[1]))

        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        blend = jnp.array(np.exp(-(dist**2) / (2 * radius**2)))

        mu_field = mu_field * (1.0 - blend) + zone_mu * blend
        sigma_field = sigma_field * (1.0 - blend) + zone_sigma * blend

    return mu_field, sigma_field


def diffuse_params(mu_field, sigma_field, rate=0.001):
    """Local parameter diffusion via neighbor averaging (gene flow analog).

    Parameters slowly mix between adjacent cells, creating smooth
    gradients and allowing successful chemistry to spread.
    Toroidal boundaries via jnp.roll.
    """
    def neighbor_avg(f):
        return (jnp.roll(f, 1, 0) + jnp.roll(f, -1, 0) +
                jnp.roll(f, 1, 1) + jnp.roll(f, -1, 1)) / 4.0

    mu_avg = neighbor_avg(mu_field)
    sigma_avg = neighbor_avg(sigma_field)

    return (mu_field + rate * (mu_avg - mu_field),
            sigma_field + rate * (sigma_avg - sigma_field))


def mutate_param_fields(mu_field, sigma_field, rng, cells,
                        noise_mu=0.005, noise_sigma=0.001,
                        mu_range=(0.08, 0.25), sigma_range=(0.02, 0.06)):
    """Mutate growth parameters near specified pattern cells.

    Applied near winning patterns after selection — creates
    heritable variation in chemistry around successful patterns.
    """
    rng, k1, k2 = random.split(rng, 3)
    n_cells = len(cells)
    mu_noise = noise_mu * random.normal(k1, (n_cells,))
    sigma_noise = noise_sigma * random.normal(k2, (n_cells,))

    rows, cols = cells[:, 0], cells[:, 1]
    new_mu = mu_field.at[rows, cols].add(mu_noise)
    new_sigma = sigma_field.at[rows, cols].add(sigma_noise)

    new_mu = jnp.clip(new_mu, mu_range[0], mu_range[1])
    new_sigma = jnp.clip(new_sigma, sigma_range[0], sigma_range[1])

    return new_mu, new_sigma
