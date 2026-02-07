"""V12: Attention-Based Lenia — State-Dependent Interaction Topology.

V11 experiments (V11.0-V11.7) converged on Finding 4: the locality ceiling.
Convolutional physics with fixed interaction topology cannot produce
biological integration (Phi increasing under threat), regardless of
substrate complexity (C=3->64), selection pressure, or evolutionary regime.

The book's hyp:attention-bottleneck predicts that state-dependent interaction
topology is the missing ingredient. V12 tests this by replacing the fixed
FFT convolution (Step 1 of the scan body) with windowed self-attention:

  Current V11.4 (fixed topology):
    potentials[c,n,m] = sum_{i,j} grid[c, n+i, m+j] * kernel_c[i,j]

  V12 (state-dependent topology):
    Q = W_q @ grid[:, n, m]            # query from cell state
    K = W_k @ grid[:, neighbors]       # keys from neighbors
    attn = softmax(Q @ K^T / tau)      # state-dependent attention
    potentials[:, n, m] = attn @ V     # attended potential

Everything else (growth function, cross-channel coupling gate, resource
dynamics, decay, maintenance) stays IDENTICAL to V11.4.

Key design decisions:
1. Window size w_soft is evolvable (starts at R_max=13, up to w_max)
2. Attention temperature tau is evolvable (high=uniform, low=focused)
3. Projection matrices W_q, W_k are shared, evolved slowly
4. Free-lunch control: Condition A uses fixed w=R (no expansion)

Performance strategy:
- Extract ALL patches at once using roll+slice (no dynamic_slice in vmap)
- Batch key projection over all spatial positions
- Single matmul for attention scores across all positions
- jax.checkpoint for memory efficiency in scan
"""

import jax
import jax.numpy as jnp
from jax import random, jit, lax, vmap
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

def generate_attention_config(C=16, N=128, seed=42, w_max=13, d_attn=16):
    """Generate configuration for attention-based Lenia.

    Extends generate_hd_config with attention-specific parameters.

    Args:
        C: number of channels
        N: grid size
        seed: random seed
        w_max: maximum attention window radius (computation buffer)
        d_attn: attention projection dimension
    """
    config = generate_hd_config(C=C, N=N, seed=seed)

    # Attention parameters
    config['w_max'] = w_max          # max window radius (fixed computation buffer)
    config['w_soft'] = float(min(w_max, 13))  # initial soft window radius
    config['tau'] = 1.0              # attention temperature (evolvable)
    config['d_attn'] = d_attn        # attention projection dimension
    config['use_attention'] = True

    return config


def init_attention_params(C, d_attn, seed=42):
    """Initialize attention projection matrices.

    W_q: (d_attn, C) — projects cell state to query
    W_k: (d_attn, C) — projects neighbor state to key

    Initialized with small random values so initial attention is near-uniform.

    Returns dict with JAX arrays.
    """
    rng = random.PRNGKey(seed)
    k1, k2 = random.split(rng)

    scale = 1.0 / np.sqrt(d_attn)

    W_q = scale * random.normal(k1, (d_attn, C))
    W_k = scale * random.normal(k2, (d_attn, C))

    return {
        'W_q': W_q,
        'W_k': W_k,
    }


# ============================================================================
# Distance mask for soft window
# ============================================================================

def _make_distance_field(w_max):
    """Pre-compute squared distances from center for a (2*w_max+1)^2 window.

    Returns (2*w_max+1, 2*w_max+1) float32 array.
    """
    coords = np.arange(-w_max, w_max + 1, dtype=np.float32)
    dy, dx = np.meshgrid(coords, coords, indexing='ij')
    return (dy**2 + dx**2).astype(np.float32)


# ============================================================================
# Efficient Patch Extraction
# ============================================================================

def _extract_all_patches(grid_padded, w_max, N):
    """Extract (2w+1)x(2w+1) patches for every spatial position.

    Uses slicing on the padded grid — no dynamic_slice needed.

    Args:
        grid_padded: (C, N+2w, N+2w) toroidally padded grid
        w_max: window radius
        N: original grid size

    Returns:
        patches: (N, N, C, (2w+1)^2) — all patches flattened
    """
    C = grid_padded.shape[0]
    win = 2 * w_max + 1

    # Collect shifted views: for each (dy, dx) in window, extract the
    # corresponding N×N slice from padded grid
    # Result: (win, win, C, N, N)
    shifts = []
    for dy in range(win):
        row_shifts = []
        for dx in range(win):
            # grid_padded[:, dy:dy+N, dx:dx+N] gives the (C, N, N) view
            # offset by (dy, dx) relative to the top-left of the window
            row_shifts.append(grid_padded[:, dy:dy+N, dx:dx+N])
        shifts.append(jnp.stack(row_shifts, axis=0))  # (win, C, N, N)
    all_shifts = jnp.stack(shifts, axis=0)  # (win, win, C, N, N)

    # Reshape to (N, N, C, win*win) for batched attention
    # Transpose: (win, win, C, N, N) -> (N, N, C, win, win)
    all_shifts = jnp.transpose(all_shifts, (3, 4, 2, 0, 1))  # (N, N, C, win, win)
    patches = all_shifts.reshape(N, N, C, win * win)  # (N, N, C, win*win)

    return patches


# ============================================================================
# Core Attention Physics
# ============================================================================

@partial(jit, static_argnums=(5,))
def run_chunk_attention(grid, resource, W_q, W_k, coupling, n_steps,
                        dt, channel_mus, channel_sigmas, coupling_row_sums,
                        noise_amp, resource_consume, resource_regen,
                        resource_max, resource_half_sat, decay_rate,
                        maintenance_rate, tau, w_soft, dist_sq, rng):
    """Run n_steps of attention-based Lenia.

    Replaces Step 1 (FFT convolution) with windowed self-attention.
    Steps 2-8 remain identical to V11.4 run_chunk_hd.

    Performance: patches extracted via slicing (not dynamic_slice in vmap),
    attention computed as batched matmul over all spatial positions.
    """
    C = grid.shape[0]
    N = grid.shape[1]
    w_max = (dist_sq.shape[0] - 1) // 2
    win_size = (2 * w_max + 1) ** 2

    # Soft distance mask: sigmoid(beta * (w_soft^2 - dist^2))
    beta = 0.5
    soft_mask = jax.nn.sigmoid(beta * (w_soft**2 - dist_sq))  # (2w+1, 2w+1)
    log_mask = jnp.log(soft_mask.reshape(win_size) + 1e-10)  # (win_size,)

    def body(carry, _):
        g, r, k = carry
        k, k_noise = random.split(k)

        # ---- STEP 1: Windowed self-attention (REPLACES FFT convolution) ----

        # Pad grid toroidally
        g_padded = jnp.pad(g, ((0, 0), (w_max, w_max), (w_max, w_max)),
                           mode='wrap')  # (C, N+2w, N+2w)

        # Extract all patches: (N, N, C, win_size)
        patches = _extract_all_patches(g_padded, w_max, N)

        # Queries from all cells: (N, N, C) -> (N, N, d_attn)
        # g is (C, N, N), transpose to (N, N, C) for matmul
        g_nnc = jnp.transpose(g, (1, 2, 0))  # (N, N, C)
        queries = g_nnc @ W_q.T  # (N, N, d_attn) = (N,N,C) @ (C, d_attn)

        # Keys from all patches: (N, N, C, win_size) -> (N, N, d_attn, win_size)
        # For each position: keys = W_k @ patch = (d_attn, C) @ (C, win_size)
        # Batched: (N, N, d_attn, win_size)
        keys = jnp.einsum('dc,nmcw->nmdw', W_k, patches)  # (N, N, d_attn, win_size)

        # Attention scores: (N, N, win_size) via einsum
        # score[n,m,w] = sum_d queries[n,m,d] * keys[n,m,d,w] / tau
        scores = jnp.einsum('nmd,nmdw->nmw', queries, keys) / (tau + 1e-6)

        # Apply soft distance mask
        scores = scores + log_mask[None, None, :]  # broadcast (1, 1, win_size)

        # Softmax attention weights
        attn_weights = jax.nn.softmax(scores, axis=-1)  # (N, N, win_size)

        # Attended values: (N, N, C) = sum over win_size
        # potential[n,m,c] = sum_w patches[n,m,c,w] * attn[n,m,w]
        potentials_nnc = jnp.einsum('nmcw,nmw->nmc', patches, attn_weights)

        # Transpose back to (C, N, N)
        potentials = jnp.transpose(potentials_nnc, (2, 0, 1))

        # ---- STEP 2-8: Identical to V11.4 ----

        # 2. Cross-channel coupling via einsum
        cross_terms = jnp.einsum('cj,jnm->cnm', coupling, g)

        # 3. Growth from potential
        mus = channel_mus[:, None, None]
        sigs = channel_sigmas[:, None, None]
        growth = 2.0 * jnp.exp(-((potentials - mus)**2) / (2 * sigs**2)) - 1.0

        # 4. Cross-channel gate
        row_sums = coupling_row_sums[:, None, None]
        gate = jax.nn.sigmoid(5.0 * (cross_terms / row_sums - 0.3))
        growth = growth * gate

        # 5. Resource modulation
        rf = r / (r + resource_half_sat)
        growth = jnp.where(growth > 0, growth * rf[None, :, :], growth)

        # 6. Decay
        growth = growth - decay_rate

        # 7. Update grid + noise
        noise = noise_amp * random.normal(k_noise, g.shape)
        g_new = jnp.clip(g + dt * growth + noise, 0.0, 1.0)

        # 7b. Metabolic maintenance cost
        g_new = jnp.maximum(g_new - maintenance_rate * g_new * dt, 0.0)

        # 8. Resource dynamics
        total_activity = jnp.mean(g, axis=0)
        r_new = jnp.clip(
            r - resource_consume * total_activity * r * dt
            + resource_regen * (resource_max - r) * dt,
            0.0, resource_max)

        return (g_new, r_new, k), None

    (grid, resource, rng), _ = lax.scan(
        body, (grid, resource, rng), None, length=n_steps
    )
    return grid, resource, rng


def run_chunk_attention_wrapper(grid, resource, W_q, W_k, coupling, rng,
                                config, n_steps, attn_params=None):
    """Convenience wrapper that unpacks config into run_chunk_attention args."""
    channel_mus = jnp.array(config['channel_mus'])
    channel_sigmas = jnp.array(config['channel_sigmas'])
    coupling_row_sums = jnp.sum(coupling, axis=1)

    w_max = config['w_max']
    w_soft = jnp.float32(config['w_soft'] if attn_params is None
                         else attn_params.get('w_soft', config['w_soft']))
    tau = jnp.float32(config['tau'] if attn_params is None
                      else attn_params.get('tau', config['tau']))

    dist_sq = jnp.array(_make_distance_field(w_max))

    return run_chunk_attention(
        grid, resource, W_q, W_k, coupling, n_steps,
        jnp.float32(config['dt']),
        channel_mus, channel_sigmas, coupling_row_sums,
        jnp.float32(config['noise_amp']),
        jnp.float32(config['resource_consume']),
        jnp.float32(config['resource_regen']),
        jnp.float32(config['resource_max']),
        jnp.float32(config['resource_half_sat']),
        jnp.float32(config.get('decay_rate', 0.0)),
        jnp.float32(config.get('maintenance_rate', 0.0)),
        tau, w_soft, dist_sq, rng,
    )


# ============================================================================
# Attention Diagnostics
# ============================================================================

def measure_attention_entropy(grid, W_q, W_k, config, cells,
                              tau=None, w_soft=None):
    """Measure attention entropy at pattern cells.

    High entropy = diffuse attention (like convolution).
    Low entropy = focused attention (selective measurement).

    Returns mean entropy across pattern cells.
    """
    if len(cells) < 2:
        return 0.0

    C = grid.shape[0]
    N = grid.shape[1]
    w_max = config['w_max']

    if tau is None:
        tau = config['tau']
    if w_soft is None:
        w_soft = config['w_soft']

    dist_sq = _make_distance_field(w_max)
    beta = 0.5
    soft_mask = 1.0 / (1.0 + np.exp(-beta * (w_soft**2 - dist_sq)))
    mask_flat = soft_mask.reshape(-1)

    g_padded = np.array(jnp.pad(grid, ((0, 0), (w_max, w_max), (w_max, w_max)),
                                mode='wrap'))
    W_q_np = np.array(W_q)
    W_k_np = np.array(W_k)

    entropies = []
    sample_idx = np.random.choice(len(cells), min(50, len(cells)), replace=False)

    for idx in sample_idx:
        n, m = cells[idx]
        n_pad, m_pad = n + w_max, m + w_max

        cell_state = g_padded[:, n_pad, m_pad]
        query = W_q_np @ cell_state

        window = g_padded[:, n_pad - w_max:n_pad + w_max + 1,
                          m_pad - w_max:m_pad + w_max + 1]
        win_flat = window.reshape(C, -1)
        keys = W_k_np @ win_flat

        scores = query @ keys / (tau + 1e-6)
        scores = scores + np.log(mask_flat + 1e-10)

        scores = scores - scores.max()
        exp_scores = np.exp(scores)
        attn = exp_scores / (exp_scores.sum() + 1e-10)

        entropy = -np.sum(attn * np.log(attn + 1e-10))
        entropies.append(entropy)

    return float(np.mean(entropies))


def attention_spatial_map(grid, W_q, W_k, config, cell_n, cell_m,
                          tau=None, w_soft=None):
    """Get the attention weight map for a specific cell.

    Returns (2*w_max+1, 2*w_max+1) attention weight heatmap.
    """
    C = grid.shape[0]
    w_max = config['w_max']

    if tau is None:
        tau = config['tau']
    if w_soft is None:
        w_soft = config['w_soft']

    dist_sq = _make_distance_field(w_max)
    beta = 0.5
    soft_mask = 1.0 / (1.0 + np.exp(-beta * (w_soft**2 - dist_sq)))
    mask_flat = soft_mask.reshape(-1)

    g_padded = np.array(jnp.pad(grid, ((0, 0), (w_max, w_max), (w_max, w_max)),
                                mode='wrap'))
    W_q_np = np.array(W_q)
    W_k_np = np.array(W_k)

    n_pad, m_pad = cell_n + w_max, cell_m + w_max
    cell_state = g_padded[:, n_pad, m_pad]
    query = W_q_np @ cell_state

    window = g_padded[:, n_pad - w_max:n_pad + w_max + 1,
                      m_pad - w_max:m_pad + w_max + 1]
    win_flat = window.reshape(C, -1)
    keys = W_k_np @ win_flat

    scores = query @ keys / (tau + 1e-6)
    scores = scores + np.log(mask_flat + 1e-10)

    scores = scores - scores.max()
    exp_scores = np.exp(scores)
    attn = exp_scores / (exp_scores.sum() + 1e-10)

    win_size = 2 * w_max + 1
    return attn.reshape(win_size, win_size)
