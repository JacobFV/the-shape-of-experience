"""V11.4 HD Affect Measurement: Phi for high-dimensional channels.

At C=64, exact channel-partition Phi (testing all 2^C partitions) is
infeasible. Two approaches:

1. Spectral Phi (fast, routine use):
   Coupling-weighted covariance of channel values at pattern cells.
   Effective rank of eigenvalues = integration proxy.
   O(C^2 * n_cells + C^3) — fast even at C=64.

2. Sampled MIP Phi (accurate, stress-test use):
   Sample K random binary channel partitions. For each, compute
   growth with/without the other group. Take minimum across samples.
   O(K * C * N^2) with K=16 samples.

Both are vectorized for GPU.
"""

import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import numpy as np

from v11_substrate import growth_fn
from v11_affect import (
    AffectState, measure_valence, measure_arousal_mc,
    measure_effective_rank_mc, measure_self_model_salience,
    _make_partition_masks,
)


# ============================================================================
# Spectral Channel Phi (fast, routine)
# ============================================================================

def spectral_channel_phi(grid_mc, coupling, cells):
    """Fast integration proxy via coupling-weighted covariance spectrum.

    For a pattern with cells at positions P:
    1. Extract channel values: V[c, i] = grid_mc[c, P_i] for i in P
    2. Coupling-weighted covariance: C_weighted = W @ (V @ V^T / n) @ W^T
       (or simpler: just W-weighted correlation of channel activations)
    3. Effective rank of eigenvalues = integration measure

    High effective rank means many channels contribute irreducibly.
    Low effective rank means channels are redundant or decoupled.

    O(C^2 * n_cells + C^3) — fast even at C=64.
    """
    if len(cells) < 4:
        return 0.0

    C = grid_mc.shape[0]

    # Extract channel values at pattern cells: (C, n_cells)
    vals = grid_mc[:, cells[:, 0], cells[:, 1]]  # (C, n_cells)
    n = vals.shape[1]

    # Center each channel
    vals_centered = vals - vals.mean(axis=1, keepdims=True)

    # Channel covariance: (C, C)
    cov = (vals_centered @ vals_centered.T) / max(n - 1, 1)

    # Weight by coupling matrix (emphasizes coupled channels)
    # Symmetric weighting: W @ cov @ W^T
    weighted_cov = coupling @ cov @ coupling.T

    # Effective rank of weighted covariance
    eigvals = jnp.linalg.eigvalsh(weighted_cov)
    eigvals = jnp.maximum(eigvals, 0.0)  # numerical safety

    trace = jnp.sum(eigvals)
    trace_sq = jnp.sum(eigvals**2)

    # Effective rank: (tr C)^2 / tr(C^2)
    eff_rank = jnp.where(trace_sq > 1e-20, trace**2 / trace_sq, 1.0)

    # Normalize to [0, 1] range by dividing by C
    phi_spectral = float(eff_rank) / C

    # Also return the raw effective rank for diagnostics
    return float(phi_spectral), float(eff_rank)


# ============================================================================
# Sampled MIP Channel Phi (accurate, stress-test)
# ============================================================================

def sampled_channel_phi(grid_mc, kernel_ffts, coupling, cells,
                        channel_mus, channel_sigmas, N,
                        n_samples=16, rng=None):
    """Accurate integration via sampled random channel partitions.

    For each of K random binary partitions of channels into A/B:
    1. Compute full growth for all channels
    2. Compute growth with coupling zeroed from B -> A
    3. Measure growth difference at pattern cells
    4. Take minimum across samples (MIP = weakest link)

    O(K * C * N^2) with K=16 samples.
    """
    C = grid_mc.shape[0]
    if len(cells) < 4:
        return 0.0

    if rng is None:
        rng = random.PRNGKey(0)

    # Pre-compute full potentials and growth for all channels
    g_fft = jnp.fft.rfft2(grid_mc)  # (C, N, N//2+1)
    potentials = jnp.fft.irfft2(g_fft * kernel_ffts, s=(N, N))  # (C, N, N)

    mus = jnp.array(channel_mus)[:, None, None]
    sigs = jnp.array(channel_sigmas)[:, None, None]
    base_growth = 2.0 * jnp.exp(-((potentials - mus)**2) / (2 * sigs**2)) - 1.0

    # Full cross-channel coupling
    cross_full = jnp.einsum('cj,jnm->cnm', coupling, grid_mc)
    row_sums = jnp.sum(coupling, axis=1)[:, None, None]
    gate_full = jax.nn.sigmoid(5.0 * (cross_full / row_sums - 0.3))
    growth_full = base_growth * gate_full  # (C, N, N)

    # Pattern cell mask for scoring
    mask = np.zeros((N, N), dtype=np.float32)
    mask[cells[:, 0], cells[:, 1]] = 1.0
    mask_j = jnp.array(mask)
    n_cells = float(mask.sum())

    best_phi = float('inf')

    for s in range(n_samples):
        rng, k = random.split(rng)

        # Random binary partition: each channel goes to A (1) or B (0)
        # Ensure both sides have at least 1 channel
        partition = np.array(random.bernoulli(k, 0.5, shape=(C,)))
        if partition.sum() == 0:
            partition[0] = True
        if partition.sum() == C:
            partition[0] = False
        partition_j = jnp.array(partition, dtype=jnp.float32)

        # Coupling with B removed from perspective of A:
        # Zero out coupling weights from B -> A channels
        # coupling_masked[c, j] = coupling[c, j] if partition[j] == partition[c], else 0
        # Actually: for channels in A, remove coupling FROM B
        mask_A = partition_j[:, None]  # (C, 1): 1 if channel c is in A
        mask_B = 1.0 - partition_j[None, :]  # (1, C): 1 if channel j is in B

        # Zero coupling from B to A: set coupling[c,j]=0 where c in A, j in B
        coupling_masked = coupling * (1.0 - mask_A * mask_B)

        # Recompute cross-channel terms with masked coupling
        cross_masked = jnp.einsum('cj,jnm->cnm', coupling_masked, grid_mc)
        row_sums_masked = jnp.sum(coupling_masked, axis=1)[:, None, None]
        row_sums_masked = jnp.maximum(row_sums_masked, 1e-6)
        gate_masked = jax.nn.sigmoid(5.0 * (cross_masked / row_sums_masked - 0.3))
        growth_masked = base_growth * gate_masked

        # Phi = how much did removing cross-partition coupling change growth?
        # Score at pattern cells only
        diff = (growth_full - growth_masked)**2  # (C, N, N)
        phi_per_channel = jnp.sum(diff * mask_j[None, :, :], axis=(1, 2)) / (n_cells + 1e-10)
        phi_total = float(jnp.sum(phi_per_channel))

        best_phi = min(best_phi, phi_total)

    return best_phi if best_phi < float('inf') else 0.0


# ============================================================================
# Spatial Phi (vectorized for HD)
# ============================================================================

def spatial_phi_hd(grid_mc, kernel_ffts, coupling, cells,
                   channel_mus, channel_sigmas, N):
    """Spatial partition Phi, vectorized for arbitrary C.

    Same idea as V11.3 measure_integration_mc spatial part, but
    all channel operations are vectorized.
    """
    C = grid_mc.shape[0]
    if len(cells) < 4:
        return 0.0

    # Full potentials and growth
    g_fft = jnp.fft.rfft2(grid_mc)
    potentials = jnp.fft.irfft2(g_fft * kernel_ffts, s=(N, N))

    mus = jnp.array(channel_mus)[:, None, None]
    sigs = jnp.array(channel_sigmas)[:, None, None]

    cross_full = jnp.einsum('cj,jnm->cnm', coupling, grid_mc)
    row_sums = jnp.sum(coupling, axis=1)[:, None, None]
    gate_full = jax.nn.sigmoid(5.0 * (cross_full / row_sums - 0.3))
    base_growth = 2.0 * jnp.exp(-((potentials - mus)**2) / (2 * sigs**2)) - 1.0
    growth_full = base_growth * gate_full

    best_phi = float('inf')

    for split in ['vertical', 'horizontal']:
        mask_a_np, mask_b_np = _make_partition_masks(cells, N, split)
        if mask_a_np.sum() < 2 or mask_b_np.sum() < 2:
            continue

        mask_a = jnp.array(mask_a_np)
        mask_b = jnp.array(mask_b_np)

        # Remove B's spatial contribution to potential (all channels at once)
        grid_b = grid_mc * mask_b[None, :, :]  # (C, N, N)
        cross_b_fft = jnp.fft.rfft2(grid_b)
        cross_b_pot = jnp.fft.irfft2(cross_b_fft * kernel_ffts, s=(N, N))
        growth_without_b = 2.0 * jnp.exp(
            -((potentials - cross_b_pot - mus)**2) / (2 * sigs**2)) - 1.0
        growth_without_b = growth_without_b * gate_full

        # Remove A's spatial contribution
        grid_a = grid_mc * mask_a[None, :, :]
        cross_a_fft = jnp.fft.rfft2(grid_a)
        cross_a_pot = jnp.fft.irfft2(cross_a_fft * kernel_ffts, s=(N, N))
        growth_without_a = 2.0 * jnp.exp(
            -((potentials - cross_a_pot - mus)**2) / (2 * sigs**2)) - 1.0
        growth_without_a = growth_without_a * gate_full

        # Phi: sum across channels of partition impact
        phi_a = jnp.sum(mask_a[None, :, :] * (growth_full - growth_without_b)**2) / (
            jnp.sum(mask_a) + 1e-10)
        phi_b = jnp.sum(mask_b[None, :, :] * (growth_full - growth_without_a)**2) / (
            jnp.sum(mask_b) + 1e-10)
        phi_split = float(phi_a + phi_b)

        best_phi = min(best_phi, phi_split)

    return best_phi if best_phi < float('inf') else 0.0


# ============================================================================
# Combined HD measurement
# ============================================================================

def measure_all_hd(pattern, prev_mass, prev_values, history_entries,
                   grid_mc, kernel_ffts, coupling, config, grid_size,
                   step_num=-1, use_sampled_phi=False, rng=None):
    """Compute all affect dimensions for an HD multi-channel pattern.

    Uses spectral Phi by default (fast). Set use_sampled_phi=True for
    accurate MIP Phi (slower, used for fitness scoring in evolution).

    Returns (AffectState, spectral_phi, spectral_eff_rank, [sampled_phi]).
    """
    channel_mus = config['channel_mus']
    channel_sigmas = config['channel_sigmas']

    # Spectral Phi (always computed — fast)
    phi_spectral, eff_rank = spectral_channel_phi(
        grid_mc, coupling, pattern.cells)

    # Spatial Phi
    phi_spatial = spatial_phi_hd(
        grid_mc, kernel_ffts, coupling, pattern.cells,
        channel_mus, channel_sigmas, grid_size)

    # Sampled MIP Phi (optional — slower)
    phi_sampled = None
    if use_sampled_phi:
        phi_sampled = sampled_channel_phi(
            grid_mc, kernel_ffts, coupling, pattern.cells,
            channel_mus, channel_sigmas, grid_size,
            n_samples=16, rng=rng)

    # Use spectral Phi as the primary integration measure
    # Total = min(spatial, spectral_channel) — MIP across partition types
    total_phi = min(phi_spatial, phi_spectral)

    affect = AffectState(
        valence=measure_valence(pattern.mass, prev_mass),
        arousal=measure_arousal_mc(pattern.values, prev_values),
        integration=total_phi,
        effective_rank=measure_effective_rank_mc(history_entries),
        self_model_salience=measure_self_model_salience(history_entries),
        pattern_id=pattern.id,
        step=step_num,
        mass=pattern.mass,
        size=pattern.size,
    )

    return affect, phi_spectral, eff_rank, phi_spatial, phi_sampled
