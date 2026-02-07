"""V11.4 High-Dimensional Multi-Channel Lenia.

Fully vectorized implementation for C=64 (or arbitrary) channels.
Unlike V11.3 which uses Python for-loops over 3 channels (fine at C=3,
explodes at C=64), this uses jax.vmap and jnp.einsum so the XLA graph
size is independent of C.

Key differences from V11.3:
- Kernel FFTs: stacked (C, N, N//2+1) array, not Python list
- Coupling: einsum for cross-channel terms
- Growth: broadcasting with (C,) parameter arrays
- No Python loops in the scan body
"""

import jax
import jax.numpy as jnp
from jax import random, jit, lax, vmap
import numpy as np
from functools import partial

from v11_substrate import make_kernel, make_kernel_fft, growth_fn


# ============================================================================
# Configuration
# ============================================================================

def generate_hd_config(C=64, N=256, seed=42):
    """Generate configuration for C-channel HD Lenia.

    Kernel radii: log-spaced from 5 to 25 across channels.
    Growth mu/sigma: sampled from Beta distributions centered on
    working values from V11.0-V11.3.

    Returns config dict with arrays instead of per-channel dicts.
    """
    rng = np.random.RandomState(seed)

    # Kernel radii: log-spaced from small (local) to large (global)
    kernel_radii = np.logspace(np.log10(5), np.log10(25), C).astype(int)
    kernel_radii = np.clip(kernel_radii, 5, 25)

    # Growth parameters: Beta-distributed around known-good values
    # mu in [0.08, 0.25], centered around 0.15
    raw_mu = rng.beta(3.0, 3.0, size=C)
    channel_mus = 0.08 + raw_mu * (0.25 - 0.08)

    # sigma in [0.02, 0.06], centered around 0.035
    raw_sigma = rng.beta(3.0, 3.0, size=C)
    channel_sigmas = 0.02 + raw_sigma * (0.06 - 0.02)

    # Kernel shape params (constant across channels for simplicity)
    kernel_peaks = np.full(C, 0.5)
    kernel_widths = np.full(C, 0.15)

    return {
        'grid_size': N,
        'n_channels': C,
        'dt': 0.1,
        'kernel_radii': kernel_radii,
        'kernel_peaks': kernel_peaks,
        'kernel_widths': kernel_widths,
        'channel_mus': channel_mus.astype(np.float32),
        'channel_sigmas': channel_sigmas.astype(np.float32),
        # Resource dynamics (shared)
        'resource_max': 1.0,
        'resource_regen': 0.005,
        'resource_consume': 0.03,
        'resource_half_sat': 0.2,
        'noise_amp': 0.003,
        'decay_rate': 0.0,
        'maintenance_rate': 0.0,  # V11.6: metabolic drain per step per unit mass
    }


HD_CONFIG = generate_hd_config(C=64, N=256, seed=42)


def generate_coupling_matrix(C, bandwidth=8.0, seed=42):
    """Banded coupling matrix with toroidal wrapping in channel space.

    W[i,j] = exp(-d(i,j)^2 / (2*bandwidth^2))
    where d(i,j) = min(|i-j|, C-|i-j|) is toroidal distance.

    Diagonal = 1.0. Off-diagonal normalized so row sums <= 3.0.

    Returns (C, C) float32 array.
    """
    idx = np.arange(C)
    # Toroidal distance matrix
    diff = np.abs(idx[:, None] - idx[None, :])
    dist = np.minimum(diff, C - diff).astype(np.float32)

    W = np.exp(-dist**2 / (2 * bandwidth**2))

    # Normalize off-diagonal so row sums are bounded
    diag = np.diag(W).copy()
    np.fill_diagonal(W, 0.0)
    row_sums = W.sum(axis=1, keepdims=True)
    # Target: off-diagonal row sum = 2.0 (so total with diag = 3.0)
    scale = np.where(row_sums > 0, 2.0 / row_sums, 0.0)
    W = W * scale
    np.fill_diagonal(W, 1.0)

    return W.astype(np.float32)


# ============================================================================
# Kernel construction (vectorized)
# ============================================================================

def make_kernels_fft_hd(config):
    """Pre-compute kernel FFTs for all C channels as a stacked array.

    Returns (C, N, N//2+1) complex64 array — not a Python list.
    """
    N = config['grid_size']
    C = config['n_channels']
    radii = config['kernel_radii']
    peaks = config['kernel_peaks']
    widths = config['kernel_widths']

    kernel_ffts = []
    for c in range(C):
        k = make_kernel(int(radii[c]), float(peaks[c]), float(widths[c]))
        kf = make_kernel_fft(k, N)
        kernel_ffts.append(kf)

    return jnp.stack(kernel_ffts, axis=0)  # (C, N, N//2+1)


# ============================================================================
# Core Physics (fully vectorized)
# ============================================================================

@partial(jit, static_argnums=(5,))
def run_chunk_hd(grid, resource, kernel_ffts, coupling, rng, n_steps,
                 dt, channel_mus, channel_sigmas, coupling_row_sums,
                 noise_amp, resource_consume, resource_regen,
                 resource_max, resource_half_sat, decay_rate,
                 maintenance_rate):
    """Run n_steps of HD multi-channel Lenia on GPU.

    ALL channel operations are vectorized — zero Python loops in body.
    XLA graph size is independent of C.

    Args:
        grid: (C, N, N) multi-channel state
        resource: (N, N) shared resource field
        kernel_ffts: (C, N, N//2+1) pre-computed kernel FFTs
        coupling: (C, C) coupling matrix
        rng: JAX random key
        n_steps: number of steps (static)
        dt: timestep
        channel_mus: (C,) growth function centers
        channel_sigmas: (C,) growth function widths
        coupling_row_sums: (C,) row sums for gate normalization
        noise_amp, resource_consume, ...: scalar params
    """
    C = grid.shape[0]
    N = grid.shape[1]

    def body(carry, _):
        g, r, k = carry
        k, k_noise = random.split(k)

        # 1. Neighborhood potential via FFT convolution (all channels at once)
        #    vmap over channel dimension: rfft2 each channel, multiply by its kernel, irfft2
        g_fft = jnp.fft.rfft2(g)  # (C, N, N//2+1)
        potentials = jnp.fft.irfft2(g_fft * kernel_ffts, s=(N, N))  # (C, N, N)

        # 2. Cross-channel coupling via einsum
        #    cross_terms[c, n, m] = sum_j coupling[c, j] * g[j, n, m]
        cross_terms = jnp.einsum('cj,jnm->cnm', coupling, g)  # (C, N, N)

        # 3. Growth from potential (vectorized over channels)
        #    growth_fn broadcasts: mus/sigmas are (C, 1, 1)
        mus = channel_mus[:, None, None]    # (C, 1, 1)
        sigs = channel_sigmas[:, None, None]  # (C, 1, 1)
        growth = 2.0 * jnp.exp(-((potentials - mus)**2) / (2 * sigs**2)) - 1.0

        # 4. Cross-channel gate (normalized for arbitrary C)
        #    Normalize cross_terms by row sums so gate operates on [0, 1] scale
        row_sums = coupling_row_sums[:, None, None]  # (C, 1, 1)
        gate = jax.nn.sigmoid(5.0 * (cross_terms / row_sums - 0.3))
        growth = growth * gate

        # 5. Resource modulation (positive growth requires resources)
        rf = r / (r + resource_half_sat)  # (N, N)
        growth = jnp.where(growth > 0, growth * rf[None, :, :], growth)

        # 6. Decay
        growth = growth - decay_rate

        # 7. Update grid + noise
        noise = noise_amp * random.normal(k_noise, g.shape)
        g_new = jnp.clip(g + dt * growth + noise, 0.0, 1.0)

        # 7b. Metabolic maintenance cost (V11.6): cells pay to exist
        g_new = jnp.maximum(g_new - maintenance_rate * g_new * dt, 0.0)

        # 8. Resource dynamics (consumption uses mean across channels)
        total_activity = jnp.mean(g, axis=0)  # (N, N)
        r_new = jnp.clip(
            r - resource_consume * total_activity * r * dt
            + resource_regen * (resource_max - r) * dt,
            0.0, resource_max)

        return (g_new, r_new, k), None

    (grid, resource, rng), _ = lax.scan(
        body, (grid, resource, rng), None, length=n_steps
    )
    return grid, resource, rng


def run_chunk_hd_wrapper(grid, resource, kernel_ffts, coupling, rng,
                         config, n_steps):
    """Convenience wrapper that unpacks config into run_chunk_hd args."""
    channel_mus = jnp.array(config['channel_mus'])
    channel_sigmas = jnp.array(config['channel_sigmas'])
    coupling_row_sums = jnp.sum(coupling, axis=1)

    return run_chunk_hd(
        grid, resource, kernel_ffts, coupling, rng, n_steps,
        jnp.float32(config['dt']),
        channel_mus, channel_sigmas, coupling_row_sums,
        jnp.float32(config['noise_amp']),
        jnp.float32(config['resource_consume']),
        jnp.float32(config['resource_regen']),
        jnp.float32(config['resource_max']),
        jnp.float32(config['resource_half_sat']),
        jnp.float32(config.get('decay_rate', 0.0)),
        jnp.float32(config.get('maintenance_rate', 0.0)),
    )


# ============================================================================
# Initialization
# ============================================================================

def init_soup_hd(N, C, rng, channel_mus):
    """Initialize high-dimensional multi-channel random soup.

    Each channel gets a few random blobs centered at its growth_mu.
    Fewer seeds per channel at high C to avoid the one-giant-blob problem.
    """
    seeds_per_channel = max(3, 50 // C)

    channels = []
    yy, xx = np.mgrid[0:N, 0:N]

    for c in range(C):
        rng, k = random.split(rng)
        mu_c = float(channel_mus[c])

        # Low background noise
        rng, k_bg = random.split(rng)
        ch_grid = 0.02 * np.array(random.uniform(k_bg, (N, N)))

        for _ in range(seeds_per_channel):
            rng, ks = random.split(rng)
            keys = random.split(ks, 5)
            cx = int(random.randint(keys[0], (), 20, N - 20))
            cy = int(random.randint(keys[1], (), 20, N - 20))
            r = float(random.uniform(keys[2], (), minval=6, maxval=18))
            center_val = mu_c + float(random.uniform(keys[3], (),
                                                      minval=-0.05, maxval=0.05))
            dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            blob = center_val * np.exp(-(dist**2) / (2 * (r * 0.6)**2))
            angle = np.arctan2(yy - cy, xx - cx)
            asym = 1.0 + 0.2 * float(random.uniform(keys[4], (),
                                                      minval=-1, maxval=1)) * np.sin(angle)
            blob = blob * asym
            ch_grid = ch_grid + blob

        channels.append(np.clip(ch_grid, 0.0, 1.0))

    grid = jnp.array(np.stack(channels, axis=0))  # (C, N, N)

    # Shared resource field with patchy distribution
    resource = jnp.full((N, N), 0.3)
    for hx, hy in [(N//3, N//3), (2*N//3, 2*N//3),
                    (N//3, 2*N//3), (2*N//3, N//3)]:
        dist_h = np.sqrt((xx - hx)**2 + (yy - hy)**2)
        resource = resource + jnp.array(0.7 * np.exp(-(dist_h**2) / (2 * 30**2)))
    resource = jnp.clip(resource, 0.0, 1.0)

    return grid, resource
