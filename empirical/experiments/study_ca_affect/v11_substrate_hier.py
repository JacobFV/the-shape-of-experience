"""V11.5 Hierarchical Multi-Timescale Lenia.

Extends V11.4 HD Lenia with:
- 4 functional tiers: Sensory, Processing, Memory, Prediction
- Per-tier dt (timescale): fast sensory, slow memory, medium processing
- Asymmetric cross-tier coupling (feedforward + feedback)
- Prediction channels whose growth depends on matching future sensory state
- World-model channels that couple to resource field

Architecture (C=64 default):
  Tier 0 - Sensory:    16 ch, dt=0.1,  R=5-10   (fast, small kernels)
  Tier 1 - Processing: 24 ch, dt=0.05, R=8-18   (medium dynamics)
  Tier 2 - Memory:     16 ch, dt=0.01, R=12-25  (slow, persistent state)
  Tier 3 - Prediction:  8 ch, dt=0.05, R=7-15   (medium, special growth)

Coupling:
  Within-tier: strong (bandwidth=3)
  Sensory→Processing: strong feedforward
  Processing→Memory: strong write
  Memory→Processing: moderate feedback (read)
  Memory→Prediction: strong (prediction input)
  Prediction→Sensory: weak modulatory feedback
  Everything else: weak/zero

All vectorized for GPU — no Python loops in scan body.
"""

import jax
import jax.numpy as jnp
from jax import random, jit, lax
import numpy as np
from functools import partial

from v11_substrate import make_kernel, make_kernel_fft, growth_fn


# ============================================================================
# Tier definitions
# ============================================================================

TIER_SENSORY = 0
TIER_PROCESSING = 1
TIER_MEMORY = 2
TIER_PREDICTION = 3

TIER_NAMES = ['sensory', 'processing', 'memory', 'prediction']


def default_tier_sizes(C=64):
    """Default tier allocation for C channels."""
    # Sensory: 25%, Processing: ~38%, Memory: 25%, Prediction: ~12%
    if C < 16:
        # Small C: minimum viable allocation
        n_pred = max(1, C // 8)
        n_mem = max(1, C // 4)
        n_sens = max(1, C // 4)
        n_proc = C - n_sens - n_pred - n_mem
        n_proc = max(1, n_proc)
        # Rebalance if total doesn't match
        total = n_sens + n_proc + n_mem + n_pred
        if total > C:
            n_proc = C - n_sens - n_mem - n_pred
        if total < C:
            n_proc += C - total
    else:
        n_sens = C // 4
        n_pred = C // 8
        n_mem = C // 4
        n_proc = C - n_sens - n_pred - n_mem
    return [n_sens, n_proc, n_mem, n_pred]


# ============================================================================
# Configuration
# ============================================================================

def generate_hier_config(C=64, N=256, seed=42):
    """Generate hierarchical multi-timescale configuration.

    Returns config dict with tier assignments, per-channel parameters,
    and coupling structure.
    """
    rng = np.random.RandomState(seed)
    tier_sizes = default_tier_sizes(C)
    n_sens, n_proc, n_mem, n_pred = tier_sizes

    # Tier assignment for each channel
    tier_assignments = np.zeros(C, dtype=int)
    tier_assignments[n_sens:n_sens + n_proc] = TIER_PROCESSING
    tier_assignments[n_sens + n_proc:n_sens + n_proc + n_mem] = TIER_MEMORY
    tier_assignments[n_sens + n_proc + n_mem:] = TIER_PREDICTION

    # Per-channel dt — uniform base rate; timescale separation emerges
    # from kernel radii (memory has large kernels → slow spatial dynamics)
    # and coupling structure (memory accumulates slowly via asymmetric coupling)
    dt_per_channel = np.full(C, 0.05, dtype=np.float32)

    # Kernel radii by tier
    kernel_radii = np.zeros(C, dtype=int)
    kernel_radii[tier_assignments == TIER_SENSORY] = np.linspace(5, 10, n_sens).astype(int)
    kernel_radii[tier_assignments == TIER_PROCESSING] = np.linspace(8, 18, n_proc).astype(int)
    kernel_radii[tier_assignments == TIER_MEMORY] = np.linspace(12, 25, n_mem).astype(int)
    kernel_radii[tier_assignments == TIER_PREDICTION] = np.linspace(7, 15, n_pred).astype(int)
    kernel_radii = np.clip(kernel_radii, 5, 25)

    # Growth parameters by tier
    channel_mus = np.zeros(C, dtype=np.float32)
    channel_sigmas = np.zeros(C, dtype=np.float32)

    # Sensory: broad, responsive
    sens_idx = tier_assignments == TIER_SENSORY
    channel_mus[sens_idx] = 0.08 + rng.beta(3, 3, n_sens) * 0.12
    channel_sigmas[sens_idx] = 0.03 + rng.beta(3, 3, n_sens) * 0.03

    # Processing: medium
    proc_idx = tier_assignments == TIER_PROCESSING
    channel_mus[proc_idx] = 0.10 + rng.beta(3, 3, n_proc) * 0.15
    channel_sigmas[proc_idx] = 0.025 + rng.beta(3, 3, n_proc) * 0.025

    # Memory: narrow, stable (hard to excite, hard to lose)
    mem_idx = tier_assignments == TIER_MEMORY
    channel_mus[mem_idx] = 0.12 + rng.beta(3, 3, n_mem) * 0.10
    channel_sigmas[mem_idx] = 0.02 + rng.beta(3, 3, n_mem) * 0.015

    # Prediction: similar to processing
    pred_idx = tier_assignments == TIER_PREDICTION
    channel_mus[pred_idx] = 0.10 + rng.beta(3, 3, n_pred) * 0.12
    channel_sigmas[pred_idx] = 0.025 + rng.beta(3, 3, n_pred) * 0.025

    # Kernel shape params
    kernel_peaks = np.full(C, 0.5)
    kernel_widths = np.full(C, 0.15)

    return {
        'grid_size': N,
        'n_channels': C,
        'tier_sizes': tier_sizes,
        'tier_assignments': tier_assignments,
        'dt_per_channel': dt_per_channel,
        'kernel_radii': kernel_radii,
        'kernel_peaks': kernel_peaks,
        'kernel_widths': kernel_widths,
        'channel_mus': channel_mus,
        'channel_sigmas': channel_sigmas,
        # Resource dynamics
        'resource_max': 1.0,
        'resource_regen': 0.005,
        'resource_consume': 0.03,
        'resource_half_sat': 0.2,
        'noise_amp': 0.003,
        'decay_rate': 0.0,
        # Prediction coupling
        'prediction_lookahead': 10,    # steps ahead to predict
        'prediction_strength': 0.5,    # how much prediction error modulates growth
    }


HIER_CONFIG = generate_hier_config(C=64, N=256, seed=42)


# ============================================================================
# Hierarchical coupling matrix
# ============================================================================

def generate_hier_coupling(C, tier_assignments, seed=42,
                           bandwidth=8.0, asym_strength=0.15):
    """Hierarchical coupling: symmetric base + asymmetric directional bias.

    Base: V11.4-style banded symmetric coupling (proven to sustain patterns).
    Bias: small asymmetric component creating feedforward/feedback pathways.

    The asymmetric bias creates directional information flow:
      Sensory → Processing → Memory → Prediction → Sensory (loop)
    Without breaking the symmetric base that lets all channels sustain growth.

    Returns (C, C) float32 array.
    """
    from v11_substrate_hd import generate_coupling_matrix

    # Base: proven V11.4 symmetric coupling
    W_base = generate_coupling_matrix(C, bandwidth=bandwidth)

    # Asymmetric bias: directional flow between tiers
    W_asym = np.zeros((C, C), dtype=np.float32)

    sens_ch = np.where(tier_assignments == TIER_SENSORY)[0]
    proc_ch = np.where(tier_assignments == TIER_PROCESSING)[0]
    mem_ch = np.where(tier_assignments == TIER_MEMORY)[0]
    pred_ch = np.where(tier_assignments == TIER_PREDICTION)[0]

    def _add_directed(from_ch, to_ch, strength, spread=0.5):
        """Add directed coupling (additive, breaks symmetry)."""
        n_from, n_to = len(from_ch), len(to_ch)
        for i, ci in enumerate(to_ch):
            center = i * n_from / max(n_to, 1)
            for j, cj in enumerate(from_ch):
                d = abs(j - center) / max(n_from, 1)
                W_asym[ci, cj] += strength * np.exp(-d**2 / (2 * spread**2))

    # Feedforward: S→P→M→Pred
    _add_directed(sens_ch, proc_ch, asym_strength)
    _add_directed(proc_ch, mem_ch, asym_strength, spread=0.7)
    _add_directed(mem_ch, pred_ch, asym_strength, spread=0.7)

    # Feedback: Pred→S (closes the loop)
    _add_directed(pred_ch, sens_ch, asym_strength * 0.5, spread=0.5)

    # Combine: base symmetric + small asymmetric bias
    W = W_base + W_asym

    # Re-normalize off-diagonal to keep row sums ≤ 4.0
    np.fill_diagonal(W, 0.0)
    row_sums = W.sum(axis=1, keepdims=True)
    scale = np.where(row_sums > 0, 3.0 / np.maximum(row_sums, 1e-8), 0.0)
    W = W * np.minimum(scale, 1.0)  # only scale down, never up
    np.fill_diagonal(W, 1.0)

    return W.astype(np.float32)


# ============================================================================
# Kernel construction (same as V11.4 but with tier-specific radii)
# ============================================================================

def make_kernels_fft_hier(config):
    """Pre-compute kernel FFTs for all C channels as stacked array."""
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
# Core Physics (fully vectorized, multi-timescale)
# ============================================================================

@partial(jit, static_argnums=(5,))
def run_chunk_hier(grid, resource, kernel_ffts, coupling, rng, n_steps,
                   dt_per_channel, channel_mus, channel_sigmas,
                   coupling_row_sums,
                   noise_amp, resource_consume, resource_regen,
                   resource_max, resource_half_sat, decay_rate,
                   pred_channels_mask, sens_channels_mask,
                   prediction_strength):
    """Run n_steps of hierarchical multi-timescale Lenia.

    Key difference from V11.4: per-channel dt allows different timescales.
    Memory channels update slowly (dt=0.01), sensory channels fast (dt=0.1).

    Also: prediction channels get a bonus/penalty based on how well they
    correlate with sensory channels from the previous step (prediction signal).

    ALL channel operations are vectorized — zero Python loops in body.
    """
    C = grid.shape[0]
    N = grid.shape[1]

    def body(carry, _):
        g, r, k, g_prev_sens = carry
        k, k_noise = random.split(k)

        # 1. FFT convolution (all channels)
        g_fft = jnp.fft.rfft2(g)  # (C, N, N//2+1)
        potentials = jnp.fft.irfft2(g_fft * kernel_ffts, s=(N, N))  # (C, N, N)

        # 2. Cross-channel coupling: mix coupled activity into potential
        #    This lets sensory activity "drive" processing/memory channels
        #    by shifting their effective potential toward their growth peak
        cross_terms = jnp.einsum('cj,jnm->cnm', coupling, g)  # (C, N, N)
        row_sums = coupling_row_sums[:, None, None]
        coupling_input = cross_terms / row_sums  # normalized, (C, N, N)

        # 3. Effective potential = self-convolution + coupling injection
        #    50/50 mix: coupling provides strong cross-tier drive,
        #    allowing processing/memory channels to be sustained by sensory input
        effective_potential = 0.5 * potentials + 0.5 * coupling_input

        # 4. Growth from effective potential (per-channel)
        mus = channel_mus[:, None, None]
        sigs = channel_sigmas[:, None, None]
        growth = 2.0 * jnp.exp(-((effective_potential - mus)**2) / (2 * sigs**2)) - 1.0

        # 5. Prediction signal: prediction channels get bonus if they
        #    correlate with current sensory channels (they were "predicting"
        #    the current state from the previous step's memory)
        #    pred_signal[c] = corr(g[c], mean(g[sensory]))
        current_sensory_mean = jnp.sum(
            g * sens_channels_mask[:, None, None], axis=0
        ) / jnp.sum(sens_channels_mask)  # (N, N)

        # For prediction channels: bonus = prediction_strength * similarity
        # similarity = normalized dot product at each spatial location
        pred_match = g * current_sensory_mean[None, :, :]  # (C, N, N)
        pred_bonus = prediction_strength * pred_match * pred_channels_mask[:, None, None]
        growth = growth + pred_bonus

        # 6. Resource modulation (positive growth requires resources)
        rf = r / (r + resource_half_sat)
        growth = jnp.where(growth > 0, growth * rf[None, :, :], growth)

        # 7. Decay
        growth = growth - decay_rate

        # 8. Update grid with PER-CHANNEL dt
        dt_arr = dt_per_channel[:, None, None]  # (C, 1, 1)
        noise = noise_amp * random.normal(k_noise, g.shape)
        g_new = jnp.clip(g + dt_arr * growth + noise, 0.0, 1.0)

        # 9. Resource dynamics (consumption weighted by sensory activity)
        # Sensory channels consume resources directly; other tiers less so
        sensory_weight = 0.7 * sens_channels_mask + 0.1 * (1.0 - sens_channels_mask)
        weighted_activity = jnp.sum(
            g * sensory_weight[:, None, None], axis=0
        )  # (N, N)
        r_new = jnp.clip(
            r - resource_consume * weighted_activity * r * 0.1  # use fixed dt for resource
            + resource_regen * (resource_max - r) * 0.1,
            0.0, resource_max)

        # Store current sensory state for next step's prediction
        g_new_sens = jnp.sum(
            g_new * sens_channels_mask[:, None, None], axis=0
        ) / jnp.sum(sens_channels_mask)

        return (g_new, r_new, k, g_new_sens), None

    # Initial sensory mean
    sens_mean = jnp.sum(
        grid * sens_channels_mask[:, None, None], axis=0
    ) / jnp.sum(sens_channels_mask)

    (grid, resource, rng, _), _ = lax.scan(
        body, (grid, resource, rng, sens_mean), None, length=n_steps
    )
    return grid, resource, rng


def run_chunk_hier_wrapper(grid, resource, kernel_ffts, coupling, rng,
                           config, n_steps):
    """Convenience wrapper using V11.4 HD physics with hierarchical coupling.

    Key insight: the hierarchy lives in the COUPLING STRUCTURE (asymmetric,
    directed pathways), not in different physics per tier. All channels use
    the same proven Lenia dynamics from V11.4. The tier assignments are used
    for affect measurement, not for differential physics.
    """
    from v11_substrate_hd import run_chunk_hd

    return run_chunk_hd(
        grid, resource, kernel_ffts, coupling, rng, n_steps,
        jnp.float32(0.05),  # uniform dt
        jnp.array(config['channel_mus']),
        jnp.array(config['channel_sigmas']),
        jnp.sum(coupling, axis=1),
        jnp.float32(config['noise_amp']),
        jnp.float32(config['resource_consume']),
        jnp.float32(config['resource_regen']),
        jnp.float32(config['resource_max']),
        jnp.float32(config['resource_half_sat']),
        jnp.float32(config.get('decay_rate', 0.0)),
    )


# ============================================================================
# Initialization
# ============================================================================

def init_soup_hier(N, C, rng, config):
    """Initialize hierarchical soup using V11.4 proven init.

    All channels get equal treatment (same as V11.4 HD).
    The hierarchy lives in the coupling structure, not in initialization.
    """
    from v11_substrate_hd import init_soup_hd
    return init_soup_hd(N, C, rng, config['channel_mus'])
