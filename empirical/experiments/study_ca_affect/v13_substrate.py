"""V13: Content-Based Coupling Lenia — Experiment 0 Substrate.

The V11 series established the locality ceiling: convolutional physics with
fixed interaction topology cannot produce biological-like integration under
threat. V12 showed that attention (state-dependent coupling) is necessary
but used learned Q-K projections inherited from transformer architecture.

V13 uses a simpler, more biologically grounded mechanism: content-based
coupling MODULATION on top of standard FFT convolution.

    potential_final = potential_fft * (1 + alpha * S_local(i))

where S_local(i) is the average cosine similarity between cell i and its
neighbors within a local window. When surrounded by similar cells, the
potential is amplified — similar "chemistry" reinforces local interactions.

Key equation for the similarity field:
    S_local(i) = sigmoid(beta * (mean_j_in_window(<h(s_i), h(s_j)>) - tau))

Architecture:
1. FFT convolution for spatial potentials (standard Lenia — preserves localization)
2. Local similarity field computation (content-dependent — makes interaction state-dependent)
3. Multiplicative modulation: potential *= (1 + alpha * S_local)
4. Cross-channel coupling gate (from V11.4)
5. Growth, resources, noise, clamp, decay (from V11.4)

Key differences from V12:
1. No learned projections (no W_q, W_k) — coupling from raw state similarity
2. FFT still handles spatial structure — similarity only modulates strength
3. Simpler, cheaper, more parallelizable
4. Biologically grounded: cells with similar "chemistry" reinforce each other

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

def generate_v13_config(C=16, N=128, seed=42, similarity_radius=5,
                        d_embed=None, alpha=1.0):
    """Generate configuration for content-based coupling Lenia.

    Args:
        C: number of channels
        N: grid size
        seed: random seed
        similarity_radius: radius for local similarity computation
        d_embed: embedding dimension for h(). None = identity (d_embed = C)
        alpha: maximum modulation strength (0 = pure V11.4, 1 = up to 2x potential)
    """
    config = generate_hd_config(C=C, N=N, seed=seed)

    # Content-based coupling parameters
    config['similarity_radius'] = similarity_radius  # small window for similarity
    config['tau'] = 0.3          # similarity threshold (evolvable)
    config['gate_beta'] = 5.0    # sigmoid steepness
    config['alpha'] = alpha      # modulation strength
    config['d_embed'] = d_embed or C  # embedding dimension

    # Lethal resource dynamics
    config['maintenance_rate'] = 0.008
    config['resource_consume'] = 0.005
    config['resource_regen'] = 0.01
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
# Content-Based Similarity Field
# ============================================================================

def make_box_kernel_fft(sim_radius, N):
    """Pre-compute FFT of box averaging kernel.

    Must be called outside JIT (uses concrete shape values).
    Returns (N, N//2+1) complex array.
    """
    win = 2 * sim_radius + 1
    box = np.ones((win, win), dtype=np.float32) / (win * win)
    box_padded = np.zeros((N, N), dtype=np.float32)
    box_padded[:win, :win] = box
    # Shift so center is at (0,0) for proper circular convolution
    box_padded = np.roll(box_padded, -sim_radius, axis=0)
    box_padded = np.roll(box_padded, -sim_radius, axis=1)
    return jnp.array(np.fft.rfft2(box_padded))


def _compute_similarity_field(grid, h_embed, tau, gate_beta, box_fft):
    """Compute local content-similarity field.

    For each cell i, compute the average cosine similarity between i and
    its neighbors within a local window, then pass through sigmoid gate.

    Returns (N, N) similarity field in [0, 1].

    Uses FFT-based box blur to compute local mean embedding efficiently.
    """
    N = grid.shape[1]

    # Embed all cell states: (C, N, N) -> (d_embed, N, N)
    embedded = jnp.einsum('dc,cnm->dnm', h_embed, grid)  # (d_embed, N, N)

    # Normalize embeddings per cell
    embed_norm = jnp.sqrt(jnp.sum(embedded**2, axis=0, keepdims=True) + 1e-8)
    embedded_normed = embedded / embed_norm  # (d_embed, N, N)

    # Local mean embedding via FFT box blur
    embed_fft = jnp.fft.rfft2(embedded_normed)  # (d_embed, N, N//2+1)
    local_mean = jnp.fft.irfft2(embed_fft * box_fft[None, :, :], s=(N, N))

    # Cosine similarity between each cell and its local mean
    sim = jnp.sum(embedded_normed * local_mean, axis=0)  # (N, N)

    # Normalize by local mean magnitude
    local_mean_norm = jnp.sqrt(jnp.sum(local_mean**2, axis=0) + 1e-8)
    sim = sim / (local_mean_norm + 1e-8)

    # Sigmoid gate
    similarity_field = jax.nn.sigmoid(gate_beta * (sim - tau))

    return similarity_field


# ============================================================================
# Core Physics
# ============================================================================

@partial(jit, static_argnums=(5,))
def run_chunk_v13(grid, resource, kernel_ffts, coupling, rng, n_steps,
                  dt, channel_mus, channel_sigmas, coupling_row_sums,
                  noise_amp, resource_consume, resource_regen,
                  resource_max, resource_half_sat, decay_rate,
                  maintenance_rate,
                  h_embed, tau, gate_beta, alpha, box_fft):
    """Run n_steps of content-based coupling Lenia.

    Step 1: FFT convolution (spatial potentials — standard Lenia)
    Step 1b: Content-similarity modulation (state-dependent amplification)
    Steps 2-9: Identical to V11.4 run_chunk_hd
    """
    C = grid.shape[0]
    N = grid.shape[1]

    def body(carry, _):
        g, r, k = carry
        k, k_noise = random.split(k)

        # ---- STEP 1: Spatial potentials via FFT (standard Lenia) ----
        g_fft = jnp.fft.rfft2(g)  # (C, N, N//2+1)
        potentials = jnp.fft.irfft2(g_fft * kernel_ffts, s=(N, N))  # (C, N, N)

        # ---- STEP 1b: Content-similarity modulation ----
        # Compute local similarity field
        sim_field = _compute_similarity_field(
            g, h_embed, tau, gate_beta, box_fft)  # (N, N)

        # Modulate potentials: similar neighborhoods get amplified
        modulation = 1.0 + alpha * sim_field  # (N, N), range [1, 1+alpha]
        potentials = potentials * modulation[None, :, :]  # (C, N, N)

        # ---- STEP 2: Cross-channel coupling gate ----
        cross = jnp.einsum('cd,dnm->cnm', coupling, potentials)  # (C, N, N)
        cross = cross / coupling_row_sums[:, None, None]
        cross_gate = jax.nn.sigmoid(5.0 * (cross - 0.3))
        potentials_gated = potentials * (1.0 - 0.5 * cross_gate) + cross * 0.5 * cross_gate

        # ---- STEP 3: Growth function (vectorized) ----
        mus = channel_mus[:, None, None]
        sigs = channel_sigmas[:, None, None]
        growth = 2.0 * jnp.exp(-((potentials_gated - mus)**2) / (2 * sigs**2)) - 1.0

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
    """Initialize V13 substrate.

    Returns:
        grid: (C, N, N) initial random grid
        resource: (N, N) initial resource field
        h_embed: (d_embed, C) embedding matrix
        kernel_ffts: (C, N, N//2+1) pre-computed kernel FFTs
        coupling: (C, C) cross-channel coupling matrix
        coupling_row_sums: (C,) row normalization
        box_fft: (N, N//2+1) pre-computed box kernel FFT for similarity
    """
    C = config['n_channels']
    N = config['grid_size']
    d_embed = config['d_embed']

    rng = random.PRNGKey(seed)
    grid, resource = init_soup_hd(N, C, rng, jnp.array(config['channel_mus']))

    h_embed = init_embedding(C, d_embed, seed=seed + 1000)

    kernel_ffts = make_kernels_fft_hd(config)

    coupling = generate_coupling_matrix(C, bandwidth=3, seed=seed)
    coupling_row_sums = jnp.array(coupling.sum(axis=1))
    coupling = jnp.array(coupling)

    box_fft = make_box_kernel_fft(config['similarity_radius'], N)

    return grid, resource, h_embed, kernel_ffts, coupling, coupling_row_sums, box_fft


# ============================================================================
# Convenience runner
# ============================================================================

def run_v13_chunk(grid, resource, h_embed, kernel_ffts, config,
                  coupling, coupling_row_sums, rng, n_steps=100,
                  drought=False, box_fft=None):
    """Run a chunk of V13 steps with optional drought.

    Args:
        drought: if True, set resource_regen to near zero
        box_fft: pre-computed box kernel FFT (if None, computes it)
    """
    regen = 0.001 if drought else config['resource_regen']

    if box_fft is None:
        box_fft = make_box_kernel_fft(
            config['similarity_radius'], config['grid_size'])

    return run_chunk_v13(
        grid, resource, kernel_ffts, coupling, rng, n_steps,
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
        h_embed,
        config['tau'],
        config['gate_beta'],
        config['alpha'],
        box_fft,
    )


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == '__main__':
    import time

    print("V13 Content-Based Coupling Lenia — Smoke Test")
    print("=" * 60)

    config = generate_v13_config(C=8, N=64, similarity_radius=5)
    grid, resource, h_embed, kernel_ffts, coupling, coupling_row_sums, box_fft = init_v13(config, seed=42)

    rng = random.PRNGKey(0)

    print(f"Grid: {grid.shape}, Resource: {resource.shape}")
    print(f"Embedding: {h_embed.shape}, Kernel FFTs: {kernel_ffts.shape}")
    print(f"Coupling: {coupling.shape}")
    print(f"Alpha: {config['alpha']}, Tau: {config['tau']}, Beta: {config['gate_beta']}")
    print(f"Initial grid mean: {float(grid.mean()):.4f}")
    print(f"Initial resource mean: {float(resource.mean()):.4f}")
    print()

    # JIT compile
    print("JIT compiling (first call)...")
    t0 = time.time()
    grid2, resource2, rng = run_v13_chunk(
        grid, resource, h_embed, kernel_ffts, config,
        coupling, coupling_row_sums, rng, n_steps=2, box_fft=box_fft
    )
    jax.block_until_ready(grid2)
    t1 = time.time()
    print(f"  Compile + 2 steps: {t1-t0:.1f}s")
    print(f"  Grid mean: {float(grid2.mean()):.4f}")
    print(f"  Resource mean: {float(resource2.mean()):.4f}")
    print(f"  Alive cells: {int((grid2.mean(axis=0) > 0.05).sum())}")
    print()

    # Run a few more
    print("Running 100 steps (compiled)...")
    t0 = time.time()
    grid3, resource3, rng = run_v13_chunk(
        grid2, resource2, h_embed, kernel_ffts, config,
        coupling, coupling_row_sums, rng, n_steps=100, box_fft=box_fft
    )
    jax.block_until_ready(grid3)
    t1 = time.time()
    print(f"  100 steps: {t1-t0:.2f}s ({100/(t1-t0):.1f} steps/s)")
    print(f"  Grid mean: {float(grid3.mean()):.4f}")
    print(f"  Alive cells: {int((grid3.mean(axis=0) > 0.05).sum())}")
    print()

    # Check pattern structure
    from v11_patterns import detect_patterns_mc
    g_np = np.array(grid3)
    pats = detect_patterns_mc(g_np, threshold=0.15)
    mean_ch = g_np.mean(axis=0)
    print(f"  Patterns detected: {len(pats)}")
    print(f"  Grid coverage (>0.15): {(mean_ch > 0.15).mean():.1%}")
    for p in pats[:5]:
        print(f"    Pattern: {p.mass:.0f} cells")
    print()

    # Test drought
    print("Running 500 steps under drought...")
    t0 = time.time()
    grid4, resource4, rng = run_v13_chunk(
        grid3, resource3, h_embed, kernel_ffts, config,
        coupling, coupling_row_sums, rng, n_steps=500, drought=True, box_fft=box_fft
    )
    jax.block_until_ready(grid4)
    t1 = time.time()
    print(f"  500 steps (drought): {t1-t0:.2f}s")
    print(f"  Grid mean: {float(grid4.mean()):.4f}")
    print(f"  Resource mean: {float(resource4.mean()):.4f}")
    print(f"  Alive cells: {int((grid4.mean(axis=0) > 0.05).sum())}")

    g_np = np.array(grid4)
    pats = detect_patterns_mc(g_np, threshold=0.15)
    print(f"  Patterns after drought: {len(pats)}")

    print()
    print("Smoke test complete.")
