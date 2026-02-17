"""V13: Content-Based Coupling Lenia — Experiment 0 Substrate.

The V11 series established the locality ceiling: convolutional physics with
fixed interaction topology cannot produce biological-like integration under
threat. V12 showed that attention (state-dependent coupling) is necessary
but used learned Q-K projections inherited from transformer architecture.

V13 uses a simpler, more biologically grounded mechanism: content-based
coupling. Cells interact more strongly when their states are similar.

    K_i(j) = K_base(|i-j|) * sigmoid(<h(s_i), h(s_j)> - tau)

where:
- K_base is the standard Lenia kernel (Gaussian bump, distance-dependent)
- h is a fixed embedding (or identity): maps cell state to coupling space
- tau is a temperature threshold (evolvable)
- The sigmoid gate opens when states are similar

Key differences from V12:
1. No learned projections (no W_q, W_k) — coupling from raw state similarity
2. Multiplicative with base kernel (not replacing it)
3. Simpler, cheaper, more parallelizable
4. Biologically grounded: cells with similar "chemistry" interact more

Key difference from V11: the interaction graph is state-dependent.
Two cells 20 units apart CAN interact if their states are similar enough,
even though K_base is small at that distance — the similarity gate amplifies
weak long-range connections between "like" cells.

Resource dynamics: lethal depletion. Maintenance rate calibrated so >50%
of naive patterns die within drought duration.
"""

import jax
import jax.numpy as jnp
from jax import random, jit, lax
import numpy as np
from functools import partial

from v11_substrate import make_kernel, make_kernel_fft, growth_fn
from v11_substrate_hd import (
    generate_hd_config, generate_coupling_matrix, make_kernels_fft_hd,
    init_soup_hd,
)


# ============================================================================
# Configuration
# ============================================================================

def generate_v13_config(C=16, N=128, seed=42, coupling_radius=20, d_embed=None):
    """Generate configuration for content-based coupling Lenia.

    Args:
        C: number of channels
        N: grid size
        seed: random seed
        coupling_radius: max radius for content-based coupling window
        d_embed: embedding dimension for h(). None = identity (d_embed = C)
    """
    config = generate_hd_config(C=C, N=N, seed=seed)

    # Content-based coupling parameters
    config['coupling_radius'] = coupling_radius  # spatial extent of coupling
    config['tau'] = 0.0          # similarity threshold (evolvable)
    config['gate_beta'] = 2.0    # sigmoid steepness
    config['d_embed'] = d_embed or C  # embedding dimension

    # Lethal resource dynamics
    # V11 used maintenance_rate ~0.001-0.002 — too gentle.
    # We want >50% naive mortality during drought (~1000 steps)
    config['maintenance_rate'] = 0.01
    config['resource_consume'] = 0.005
    config['resource_regen'] = 0.01   # baseline regen (set to ~0 during drought)
    config['resource_max'] = 1.0
    config['resource_half_sat'] = 0.2

    return config


def init_embedding(C, d_embed, seed=42):
    """Initialize fixed embedding h: R^C -> R^d_embed.

    If d_embed == C, returns identity (no embedding).
    Otherwise returns a random orthogonal projection.
    """
    if d_embed == C:
        return jnp.eye(C, dtype=jnp.float32)
    # Random orthogonal projection via QR decomposition
    rng = np.random.RandomState(seed)
    A = rng.randn(d_embed, C).astype(np.float32)
    Q, _ = np.linalg.qr(A.T)
    return jnp.array(Q[:, :d_embed].T)  # (d_embed, C)


# ============================================================================
# Content-Based Coupling Gate
# ============================================================================

def _make_base_kernel_weights(coupling_radius, N, kernel_radii, C):
    """Pre-compute base kernel weights for the coupling window.

    Returns (C, 2*R+1, 2*R+1) array of spatial kernel weights within
    the coupling radius.
    """
    R = coupling_radius
    win = 2 * R + 1
    coords = np.arange(-R, R + 1, dtype=np.float32)
    dy, dx = np.meshgrid(coords, coords, indexing='ij')
    dist = np.sqrt(dy**2 + dx**2)  # (win, win)

    # For each channel, compute normalized Gaussian bump kernel
    weights = np.zeros((C, win, win), dtype=np.float32)
    for c in range(C):
        r = kernel_radii[c]
        # Lenia kernel: bell curve at distance r*peak with width r*width
        peak = 0.5
        width = 0.15
        d_norm = dist / max(r, 1)
        k = np.exp(-((d_norm - peak) / width)**2 / 2)
        k[dist > r] = 0.0  # zero outside kernel radius
        k_sum = k.sum()
        if k_sum > 0:
            k /= k_sum
        weights[c] = k

    return jnp.array(weights)


# ============================================================================
# Core Physics: Content-Based Coupling Step
# ============================================================================

def _compute_content_potentials(grid, h_embed, base_weights, tau, gate_beta, N, C, R):
    """Compute potentials using content-based coupling.

    For each cell i, the potential is:
        P_i = sum_j K_base(|i-j|) * sigmoid(beta * (<h(s_i), h(s_j)> - tau)) * s_j

    This replaces the FFT convolution step in V11.

    Args:
        grid: (C, N, N) current state
        h_embed: (d_embed, C) embedding matrix
        base_weights: (C, 2R+1, 2R+1) spatial kernel weights
        tau: similarity threshold
        gate_beta: sigmoid steepness
        N: grid size
        C: channels
        R: coupling radius

    Returns:
        potentials: (C, N, N) — coupled potentials
    """
    win = 2 * R + 1

    # Embed all cell states: (C, N, N) -> (d_embed, N, N)
    # h_embed is (d_embed, C), grid is (C, N, N)
    embedded = jnp.einsum('dc,cnm->dnm', h_embed, grid)  # (d_embed, N, N)

    # Normalize embeddings for cosine similarity
    embed_norm = jnp.sqrt(jnp.sum(embedded**2, axis=0, keepdims=True) + 1e-8)
    embedded_normed = embedded / embed_norm  # (d_embed, N, N)

    # Pad grid and embeddings toroidally
    grid_padded = jnp.pad(grid, ((0, 0), (R, R), (R, R)), mode='wrap')
    embed_padded = jnp.pad(embedded_normed, ((0, 0), (R, R), (R, R)), mode='wrap')

    # For each offset in the coupling window, compute:
    # 1. State similarity (dot product of normalized embeddings)
    # 2. Base kernel weight
    # 3. Gated contribution to potential

    # Accumulate potentials
    potentials = jnp.zeros((C, N, N), dtype=jnp.float32)

    def scan_offset(carry, offset_idx):
        pot = carry
        dy = offset_idx // win
        dx = offset_idx % win

        # Shifted grid and embedding at this offset (dynamic_slice for traced indices)
        shifted_grid = lax.dynamic_slice(grid_padded, (0, dy, dx), (C, N, N))
        shifted_embed = lax.dynamic_slice(embed_padded, (0, dy, dx), (embed_padded.shape[0], N, N))

        # Cosine similarity between center and shifted
        sim = jnp.sum(embedded_normed * shifted_embed, axis=0)  # (N, N)

        # Sigmoid gate
        gate = jax.nn.sigmoid(gate_beta * (sim - tau))  # (N, N)

        # Base kernel weight for this offset, per channel
        k_weight = base_weights[:, dy, dx]  # (C,) — advanced indexing works with traced scalars

        # Gated contribution: k_weight[c] * gate[n,m] * shifted_grid[c,n,m]
        contribution = k_weight[:, None, None] * gate[None, :, :] * shifted_grid

        pot = pot + contribution
        return pot, None

    potentials, _ = lax.scan(scan_offset, potentials, jnp.arange(win * win))

    return potentials


@partial(jit, static_argnums=(5,))
def run_chunk_v13(grid, resource, h_embed, base_weights, coupling, n_steps,
                  dt, channel_mus, channel_sigmas, coupling_row_sums,
                  noise_amp, resource_consume, resource_regen,
                  resource_max, resource_half_sat, decay_rate,
                  maintenance_rate, tau, gate_beta, rng):
    """Run n_steps of content-based coupling Lenia.

    Step 1: Content-based coupling (replaces FFT convolution)
    Steps 2-8: Identical to V11.4 run_chunk_hd
    """
    C = grid.shape[0]
    N = grid.shape[1]
    R = (base_weights.shape[1] - 1) // 2

    def body(carry, _):
        g, r, k = carry
        k, k_noise = random.split(k)

        # ---- STEP 1: Content-based coupling potentials ----
        potentials = _compute_content_potentials(
            g, h_embed, base_weights, tau, gate_beta, N, C, R
        )

        # ---- STEP 2: Cross-channel coupling gate ----
        # coupling: (C, C), coupling_row_sums: (C,)
        cross = jnp.einsum('cd,dnm->cnm', coupling, potentials)  # (C, N, N)
        cross = cross / coupling_row_sums[:, None, None]

        # Steep sigmoid gate: coupling only activates above threshold
        cross_gate = jax.nn.sigmoid(5.0 * (cross - 0.3))
        potentials_gated = potentials * (1.0 - 0.5 * cross_gate) + cross * 0.5 * cross_gate

        # ---- STEP 3: Growth function ----
        growth = jnp.zeros_like(g)
        for c_idx in range(C):
            growth = growth.at[c_idx].set(
                growth_fn(potentials_gated[c_idx], channel_mus[c_idx], channel_sigmas[c_idx])
            )

        # ---- STEP 4: Resource modulation ----
        resource_factor = r / (r + resource_half_sat)
        growth_modulated = growth * resource_factor[None, :, :]

        # ---- STEP 5: State update ----
        g_new = g + dt * growth_modulated

        # ---- STEP 6: Noise ----
        noise = noise_amp * random.normal(k_noise, g.shape)
        g_new = g_new + noise

        # ---- STEP 7: Clamp ----
        g_new = jnp.clip(g_new, 0.0, 1.0)

        # ---- STEP 8: Resource dynamics ----
        consumption = resource_consume * jnp.mean(g_new, axis=0)
        maintenance = maintenance_rate * (jnp.mean(g_new, axis=0) > 0.05).astype(jnp.float32)
        r_new = r + resource_regen * (1.0 - r / resource_max) - consumption - maintenance
        r_new = jnp.clip(r_new, 0.0, resource_max)

        # ---- STEP 9: Decay from resource depletion (LETHAL) ----
        # When resources are depleted, cells actively decay
        depletion_mask = (r_new < 0.05).astype(jnp.float32)
        g_new = g_new * (1.0 - decay_rate * depletion_mask[None, :, :])

        return (g_new, r_new, k), None

    (grid_final, resource_final, rng_final), _ = lax.scan(
        body, (grid, resource, rng), None, length=n_steps
    )

    return grid_final, resource_final, rng_final


# ============================================================================
# Initialization
# ============================================================================

def init_v13(config, seed=42):
    """Initialize V13 substrate: grid, resources, embedding, base weights.

    Returns:
        grid: (C, N, N) initial random grid
        resource: (N, N) initial resource field
        h_embed: (d_embed, C) embedding matrix
        base_weights: (C, 2R+1, 2R+1) spatial kernel weights
        coupling: (C, C) cross-channel coupling matrix
    """
    C = config['n_channels']
    N = config['grid_size']
    R = config['coupling_radius']
    d_embed = config['d_embed']

    rng = random.PRNGKey(seed)
    grid, resource = init_soup_hd(N, C, rng, jnp.array(config['channel_mus']))

    h_embed = init_embedding(C, d_embed, seed=seed+1000)

    base_weights = _make_base_kernel_weights(
        R, N, config['kernel_radii'], C
    )

    coupling = generate_coupling_matrix(C, bandwidth=3, seed=seed)
    coupling_row_sums = jnp.array(coupling.sum(axis=1))
    coupling = jnp.array(coupling)

    return grid, resource, h_embed, base_weights, coupling, coupling_row_sums


# ============================================================================
# Convenience runner
# ============================================================================

def run_v13_chunk(grid, resource, h_embed, base_weights, config,
                  coupling, coupling_row_sums, rng, n_steps=100,
                  drought=False):
    """Run a chunk of V13 steps with optional drought.

    Args:
        drought: if True, set resource_regen to near zero
    """
    regen = 0.001 if drought else config['resource_regen']

    return run_chunk_v13(
        grid, resource, h_embed, base_weights, coupling, n_steps,
        config['dt'],
        jnp.array(config['channel_mus']),
        jnp.array(config['channel_sigmas']),
        coupling_row_sums,
        config.get('noise_amp', 0.001),
        config['resource_consume'],
        regen,
        config['resource_max'],
        config['resource_half_sat'],
        config.get('decay_rate', 0.05),
        config['maintenance_rate'],
        config['tau'],
        config['gate_beta'],
        rng,
    )


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == '__main__':
    import time

    print("V13 Content-Based Coupling Lenia — Smoke Test")
    print("=" * 60)

    config = generate_v13_config(C=8, N=64, coupling_radius=10)
    grid, resource, h_embed, base_weights, coupling, coupling_row_sums = init_v13(config, seed=42)

    rng = random.PRNGKey(0)

    print(f"Grid: {grid.shape}, Resource: {resource.shape}")
    print(f"Embedding: {h_embed.shape}, Base weights: {base_weights.shape}")
    print(f"Coupling: {coupling.shape}")
    print(f"Initial grid mean: {float(grid.mean()):.4f}")
    print(f"Initial resource mean: {float(resource.mean()):.4f}")
    print()

    # JIT compile
    print("JIT compiling (first call)...")
    t0 = time.time()
    grid2, resource2, rng = run_v13_chunk(
        grid, resource, h_embed, base_weights, config,
        coupling, coupling_row_sums, rng, n_steps=2
    )
    jax.block_until_ready(grid2)
    t1 = time.time()
    print(f"  Compile + 2 steps: {t1-t0:.1f}s")
    print(f"  Grid mean: {float(grid2.mean()):.4f}")
    print(f"  Resource mean: {float(resource2.mean()):.4f}")
    print(f"  Alive cells: {int((grid2.mean(axis=0) > 0.05).sum())}")
    print()

    # Run a few more
    print("Running 10 steps (compiled)...")
    t0 = time.time()
    grid3, resource3, rng = run_v13_chunk(
        grid2, resource2, h_embed, base_weights, config,
        coupling, coupling_row_sums, rng, n_steps=10
    )
    jax.block_until_ready(grid3)
    t1 = time.time()
    print(f"  10 steps: {t1-t0:.2f}s ({10/(t1-t0):.1f} steps/s)")
    print(f"  Grid mean: {float(grid3.mean()):.4f}")
    print(f"  Alive cells: {int((grid3.mean(axis=0) > 0.05).sum())}")
    print()

    # Test drought
    print("Running 10 steps under drought...")
    t0 = time.time()
    grid4, resource4, rng = run_v13_chunk(
        grid3, resource3, h_embed, base_weights, config,
        coupling, coupling_row_sums, rng, n_steps=10, drought=True
    )
    jax.block_until_ready(grid4)
    t1 = time.time()
    print(f"  10 steps (drought): {t1-t0:.2f}s")
    print(f"  Grid mean: {float(grid4.mean()):.4f}")
    print(f"  Resource mean: {float(resource4.mean()):.4f}")
    print(f"  Alive cells: {int((grid4.mean(axis=0) > 0.05).sum())}")

    print()
    print("Smoke test complete.")
