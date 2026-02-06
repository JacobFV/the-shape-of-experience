"""V11 Affect Measurement: 6D affect extraction from emergent CA patterns.

Measures the affect dimensions defined in Part 1 of the thesis:
1. Valence: gradient direction on viability manifold (toward/away from persistence)
2. Arousal: magnitude of state change (processing intensity)
3. Integration (Phi): information lost under partition (causal coupling)
4. Effective Rank: dimensionality of representation space utilized
5. Self-Model Salience: how much internal config predicts behavior

Each measure is computed EXACTLY on the CA substrate (no proxies needed
for small patterns) or via principled approximation for larger ones.
This is the advantage of CA over neural nets: the math is tractable.
"""

import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

from v11_substrate import growth_fn


@dataclass
class AffectState:
    """6D affect measurement for a single pattern at a single timestep."""
    valence: float       # change in viability distance
    arousal: float       # state change rate
    integration: float   # Phi proxy (partition prediction loss)
    effective_rank: float  # dimensionality of state trajectory
    self_model_salience: float  # internal config -> behavior prediction
    # Metadata
    pattern_id: int = -1
    step: int = -1
    mass: float = 0.0
    size: int = 0


# ============================================================================
# Valence: direction on viability manifold
# ============================================================================

def measure_valence(mass, prev_mass):
    """Valence as change in distance from dissolution.

    Positive = moving toward viable interior (mass increasing)
    Negative = moving toward boundary (mass decreasing)
    Zero = stable orbit

    This is the CA-exact version of Part 1's Definition:
    V_t = d(x_{t+1}, dV) - d(x_t, dV)

    We use mass as proxy for viability distance because in Lenia,
    dissolution = mass -> 0. So mass IS distance from the boundary.
    """
    if prev_mass is None or prev_mass < 1e-6:
        return 0.0
    return (mass - prev_mass) / prev_mass


# ============================================================================
# Arousal: processing intensity
# ============================================================================

def measure_arousal(current_values, prev_values_at_cells):
    """Arousal as fraction of state that changed.

    High arousal = rapid reconfiguration (the pattern is "active")
    Low arousal = stable orbit (the pattern is "resting")

    Part 1: A_t = Hamming(x_{t+1}, x_t) / |B|
    Continuous analog: mean absolute change normalized by intensity.
    """
    if prev_values_at_cells is None or len(prev_values_at_cells) == 0:
        return 0.0
    if len(current_values) != len(prev_values_at_cells):
        # Pattern changed size; use mass-based arousal
        return abs(current_values.sum() - prev_values_at_cells.sum()) / \
               (prev_values_at_cells.sum() + 1e-10)
    diff = np.abs(current_values - prev_values_at_cells)
    return float(diff.mean() / (current_values.mean() + 1e-10))


# ============================================================================
# Integration (Phi): causal coupling across partitions
# ============================================================================

def _make_partition_masks(cells, grid_size, split='vertical'):
    """Create binary masks for left/right or top/bottom partition."""
    N = grid_size
    mask_a = np.zeros((N, N), dtype=np.float32)
    mask_b = np.zeros((N, N), dtype=np.float32)

    if split == 'vertical':
        median = np.median(cells[:, 1])
        for r, c in cells:
            if c <= median:
                mask_a[r, c] = 1.0
            else:
                mask_b[r, c] = 1.0
    else:  # horizontal
        median = np.median(cells[:, 0])
        for r, c in cells:
            if r <= median:
                mask_a[r, c] = 1.0
            else:
                mask_b[r, c] = 1.0

    return mask_a, mask_b


def measure_integration(grid_jnp, kernel_fft, cells, growth_mu, growth_sigma, grid_size):
    """Phi proxy: partition prediction loss.

    THE key measurement. Computes how much cutting the pattern in half
    changes its predicted next state. This is the CA-exact version of
    IIT's integrated information.

    Method:
    1. Compute full potential field (convolution of full grid)
    2. Compute cross-partition contribution (convolution of each half)
    3. Compare growth under full vs partitioned potential
    4. Phi = how much the other half matters

    Try both vertical and horizontal splits, take the MINIMUM
    (minimum information partition = MIP from IIT).
    """
    N = grid_size
    if len(cells) < 4:
        return 0.0

    # Full potential (already needed for dynamics)
    potential_full = jnp.fft.irfft2(
        jnp.fft.rfft2(grid_jnp) * kernel_fft, s=(N, N)
    )

    best_phi = float('inf')

    for split in ['vertical', 'horizontal']:
        mask_a_np, mask_b_np = _make_partition_masks(cells, N, split)

        if mask_a_np.sum() < 2 or mask_b_np.sum() < 2:
            continue

        mask_a = jnp.array(mask_a_np)
        mask_b = jnp.array(mask_b_np)

        # Cross-partition potential: contribution of B to the field
        cross_b = jnp.fft.irfft2(
            jnp.fft.rfft2(grid_jnp * mask_b) * kernel_fft, s=(N, N)
        )
        # By linearity: cross_a = potential_of_pattern - cross_b
        # But we need contribution of A, which includes non-pattern cells
        # Actually: for cells in A, removing B means subtracting B's contribution
        cross_a = jnp.fft.irfft2(
            jnp.fft.rfft2(grid_jnp * mask_a) * kernel_fft, s=(N, N)
        )

        # Growth under full potential vs potential-without-other-half
        g_full = growth_fn(potential_full, growth_mu, growth_sigma)
        g_a_without_b = growth_fn(potential_full - cross_b, growth_mu, growth_sigma)
        g_b_without_a = growth_fn(potential_full - cross_a, growth_mu, growth_sigma)

        # Phi for each half: how much does removing the other change growth?
        phi_a = float(jnp.sum(mask_a * (g_full - g_a_without_b)**2) /
                       (jnp.sum(mask_a) + 1e-10))
        phi_b = float(jnp.sum(mask_b * (g_full - g_b_without_a)**2) /
                       (jnp.sum(mask_b) + 1e-10))

        phi = phi_a + phi_b
        best_phi = min(best_phi, phi)

    return best_phi if best_phi < float('inf') else 0.0


# ============================================================================
# Effective Rank: dimensionality of state trajectory
# ============================================================================

def measure_effective_rank(history_entries, max_window=50, embed_size=16):
    """Effective rank of pattern's state trajectory.

    How many dimensions is the pattern actually using?
    High = exploring diverse configurations (rich dynamics)
    Low = trapped in repetitive orbit (simple dynamics)

    Part 1: r_eff = (tr C)^2 / tr(C^2)

    We embed each pattern state into a fixed-size representation
    (downsampled to embed_size x embed_size) for comparison across
    timesteps, then compute PCA effective rank.
    """
    if len(history_entries) < 5:
        return 1.0

    entries = history_entries[-max_window:]

    # Embed each state: extract bounding box, resample to fixed size
    embeddings = []
    for entry in entries:
        cells = entry['cells']
        values = entry['values']
        if len(cells) < 2:
            continue

        r_min, c_min = cells.min(axis=0)
        r_max, c_max = cells.max(axis=0)
        h = max(r_max - r_min + 1, 1)
        w = max(c_max - c_min + 1, 1)

        # Create small grid of pattern
        patch = np.zeros((h, w), dtype=np.float32)
        for (r, c), v in zip(cells, values):
            patch[r - r_min, c - c_min] = v

        # Downsample to fixed size via block averaging
        from scipy.ndimage import zoom as scipy_zoom
        if h > 1 and w > 1:
            scale_r = embed_size / h
            scale_c = embed_size / w
            embedded = scipy_zoom(patch, (scale_r, scale_c), order=1)
            # Crop/pad to exact embed_size
            embedded = embedded[:embed_size, :embed_size]
            if embedded.shape != (embed_size, embed_size):
                tmp = np.zeros((embed_size, embed_size), dtype=np.float32)
                sh, sw = embedded.shape
                tmp[:sh, :sw] = embedded
                embedded = tmp
        else:
            embedded = np.zeros((embed_size, embed_size), dtype=np.float32)

        embeddings.append(embedded.ravel().astype(np.float64))

    if len(embeddings) < 3:
        return 1.0

    try:
        X = np.array(embeddings)  # (T, embed_size^2)
        # Normalize to prevent overflow
        scale = np.abs(X).max()
        if scale > 0:
            X = X / scale
        X = X - X.mean(axis=0)

        # Use SVD for numerical stability instead of covariance
        _, S, _ = np.linalg.svd(X, full_matrices=False)
        eigenvalues = S**2 / (len(X) - 1)

        trace = eigenvalues.sum()
        trace_sq = (eigenvalues**2).sum()

        if trace_sq < 1e-20:
            return 1.0

        return float(trace**2 / trace_sq)
    except Exception:
        return 1.0


# ============================================================================
# Self-Model Salience: internal config predicts behavior
# ============================================================================

def measure_self_model_salience(history_entries, min_window=10):
    """How well does internal configuration predict behavior?

    Self-model salience = R^2 of regression from internal state
    to next-step behavior (movement direction and mass change).

    High = pattern's structure determines its future (self-effecting)
    Low = pattern's future is externally determined

    Part 1: SM = MI(self-tracking cells; effector cells) / H(effector cells)
    We approximate via linear predictability of behavior from state.
    """
    if len(history_entries) < min_window:
        return 0.0

    entries = history_entries[-50:]  # use recent history

    # Extract behavior: velocity and mass change
    centers = np.array([e['center'] for e in entries])
    masses = np.array([e['mass'] for e in entries])

    if len(centers) < 5:
        return 0.0

    # Behavior vector: dx, dy, d_mass
    velocity = np.diff(centers, axis=0)  # (T-1, 2)
    mass_change = np.diff(masses)        # (T-1,)
    behavior = np.column_stack([velocity, mass_change])  # (T-1, 3)

    # State features: mass, size, aspect ratio
    sizes = np.array([e['size'] for e in entries[:-1]])
    prev_masses = masses[:-1]
    # Compute simple shape features
    aspects = []
    for e in entries[:-1]:
        cells = e['cells']
        r_range = np.ptp(cells[:, 0]) + 1
        c_range = np.ptp(cells[:, 1]) + 1
        aspects.append(r_range / (c_range + 1e-10))
    aspects = np.array(aspects)

    state_features = np.column_stack([prev_masses, sizes, aspects])

    if state_features.shape[0] < 3 or behavior.shape[0] < 3:
        return 0.0

    # Linear regression: state -> behavior
    # R^2 = how much variance in behavior is explained by state
    try:
        X = state_features - state_features.mean(axis=0)
        Y = behavior - behavior.mean(axis=0)

        # Least squares
        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        Y_pred = X @ beta
        ss_res = ((Y - Y_pred)**2).sum()
        ss_tot = (Y**2).sum()

        if ss_tot < 1e-10:
            return 0.0

        r_squared = 1.0 - ss_res / ss_tot
        return float(np.clip(r_squared, 0.0, 1.0))
    except Exception:
        return 0.0


# ============================================================================
# Combined measurement
# ============================================================================

def measure_all(pattern, prev_mass, prev_values, history_entries,
                grid_jnp, kernel_fft, growth_mu, growth_sigma, grid_size,
                step_num=-1):
    """Compute all affect dimensions for a pattern."""
    return AffectState(
        valence=measure_valence(pattern.mass, prev_mass),
        arousal=measure_arousal(pattern.values, prev_values),
        integration=measure_integration(
            grid_jnp, kernel_fft, pattern.cells,
            growth_mu, growth_sigma, grid_size
        ),
        effective_rank=measure_effective_rank(history_entries),
        self_model_salience=measure_self_model_salience(history_entries),
        pattern_id=pattern.id,
        step=step_num,
        mass=pattern.mass,
        size=pattern.size,
    )


# ============================================================================
# V11.3: Multi-Channel Affect Measurement
# ============================================================================

def measure_integration_mc(grid_mc, kernel_ffts, coupling, cells,
                           channel_configs, N):
    """Multi-channel Phi: integration across spatial AND channel partitions.

    Two types of partition:
    1. SPATIAL: cut the pattern in half spatially (same as single-channel,
       but summed across channels). How much does the other spatial half
       matter to each channel's growth?
    2. CHANNEL: remove one channel entirely. How much does that channel
       matter to the others' growth? This is the KEY new measurement:
       cross-channel integration.

    Total Phi = min(spatial_phi, channel_phi) — MIP across both types.
    This follows IIT: the MIP is whichever partition loses LEAST info.
    """
    C = grid_mc.shape[0]

    if len(cells) < 4:
        return 0.0, 0.0, 0.0  # total, spatial, channel

    # ---- Spatial partition Phi (generalized to multi-channel) ----
    # Compute full potentials for all channels
    potentials_full = []
    for c in range(C):
        pot = jnp.fft.irfft2(
            jnp.fft.rfft2(grid_mc[c]) * kernel_ffts[c], s=(N, N))
        potentials_full.append(pot)

    best_spatial_phi = float('inf')

    for split in ['vertical', 'horizontal']:
        from v11_affect import _make_partition_masks
        mask_a_np, mask_b_np = _make_partition_masks(cells, N, split)

        if mask_a_np.sum() < 2 or mask_b_np.sum() < 2:
            continue

        mask_a = jnp.array(mask_a_np)
        mask_b = jnp.array(mask_b_np)

        phi_split = 0.0
        for c in range(C):
            cfg = channel_configs[c]
            mu_c = jnp.float32(cfg['growth_mu'])
            sigma_c = jnp.float32(cfg['growth_sigma'])

            # Cross-channel coupling at full
            cross_full = jnp.zeros((N, N))
            for j in range(C):
                cross_full = cross_full + coupling[c, j] * grid_mc[j]
            gate_full = jax.nn.sigmoid(5.0 * (cross_full - 0.3))

            g_full = growth_fn(potentials_full[c], mu_c, sigma_c) * gate_full

            # Remove B's contribution to channel c
            cross_b = jnp.fft.irfft2(
                jnp.fft.rfft2(grid_mc[c] * mask_b) * kernel_ffts[c], s=(N, N))
            g_without_b = growth_fn(
                potentials_full[c] - cross_b, mu_c, sigma_c) * gate_full

            # Remove A's contribution to channel c
            cross_a = jnp.fft.irfft2(
                jnp.fft.rfft2(grid_mc[c] * mask_a) * kernel_ffts[c], s=(N, N))
            g_without_a = growth_fn(
                potentials_full[c] - cross_a, mu_c, sigma_c) * gate_full

            phi_a = float(jnp.sum(mask_a * (g_full - g_without_b)**2) /
                          (jnp.sum(mask_a) + 1e-10))
            phi_b = float(jnp.sum(mask_b * (g_full - g_without_a)**2) /
                          (jnp.sum(mask_b) + 1e-10))
            phi_split += phi_a + phi_b

        best_spatial_phi = min(best_spatial_phi, phi_split)

    if best_spatial_phi == float('inf'):
        best_spatial_phi = 0.0

    # ---- Channel partition Phi ----
    # Remove one channel at a time, measure impact on remaining channels
    best_channel_phi = float('inf')

    for removed in range(C):
        phi_this_removal = 0.0

        for c in range(C):
            if c == removed:
                continue

            cfg = channel_configs[c]
            mu_c = jnp.float32(cfg['growth_mu'])
            sigma_c = jnp.float32(cfg['growth_sigma'])

            # Full cross-channel coupling
            cross_full = jnp.zeros((N, N))
            for j in range(C):
                cross_full = cross_full + coupling[c, j] * grid_mc[j]
            gate_full = jax.nn.sigmoid(5.0 * (cross_full - 0.3))

            g_full = growth_fn(potentials_full[c], mu_c, sigma_c) * gate_full

            # Cross-channel coupling WITHOUT the removed channel
            cross_without = jnp.zeros((N, N))
            for j in range(C):
                if j == removed:
                    continue
                cross_without = cross_without + coupling[c, j] * grid_mc[j]
            gate_without = jax.nn.sigmoid(5.0 * (cross_without - 0.3))

            g_without = growth_fn(potentials_full[c], mu_c, sigma_c) * gate_without

            # Measure impact at pattern cells
            mask = jnp.zeros((N, N))
            mask = mask.at[cells[:, 0], cells[:, 1]].set(1.0)

            phi_c = float(jnp.sum(mask * (g_full - g_without)**2) /
                          (jnp.sum(mask) + 1e-10))
            phi_this_removal += phi_c

        best_channel_phi = min(best_channel_phi, phi_this_removal)

    if best_channel_phi == float('inf'):
        best_channel_phi = 0.0

    # Total Phi = MIP across spatial and channel partitions
    total_phi = min(best_spatial_phi, best_channel_phi)

    return total_phi, best_spatial_phi, best_channel_phi


def measure_arousal_mc(current_values_mc, prev_values_mc):
    """Arousal for multi-channel patterns.

    current_values_mc, prev_values_mc: (N_cells, C) arrays.
    Arousal = mean absolute change across all channels and cells.
    """
    if prev_values_mc is None or len(prev_values_mc) == 0:
        return 0.0
    if current_values_mc.shape != prev_values_mc.shape:
        return abs(current_values_mc.sum() - prev_values_mc.sum()) / \
               (prev_values_mc.sum() + 1e-10)
    diff = np.abs(current_values_mc - prev_values_mc)
    return float(diff.mean() / (current_values_mc.mean() + 1e-10))


def measure_effective_rank_mc(history_entries, max_window=50, embed_size=16):
    """Effective rank for multi-channel patterns.

    Embeds each state as C * embed_size * embed_size, then computes PCA rank.
    More channels = richer state space = potentially higher rank.
    """
    if len(history_entries) < 5:
        return 1.0

    entries = history_entries[-max_window:]
    embeddings = []

    for entry in entries:
        cells = entry['cells']
        values = entry['values']
        if len(cells) < 2:
            continue

        r_min, c_min = cells.min(axis=0)
        r_max, c_max = cells.max(axis=0)
        h = max(r_max - r_min + 1, 1)
        w = max(c_max - c_min + 1, 1)

        # values may be (N_cells,) or (N_cells, C)
        if values.ndim == 1:
            C = 1
            values_2d = values[:, None]
        else:
            C = values.shape[1]
            values_2d = values

        from scipy.ndimage import zoom as scipy_zoom
        channel_embeds = []
        for c in range(C):
            patch = np.zeros((h, w), dtype=np.float32)
            for idx, (r, col) in enumerate(cells):
                patch[r - r_min, col - c_min] = values_2d[idx, c]

            if h > 1 and w > 1:
                scale_r = embed_size / h
                scale_c = embed_size / w
                embedded = scipy_zoom(patch, (scale_r, scale_c), order=1)
                embedded = embedded[:embed_size, :embed_size]
                if embedded.shape != (embed_size, embed_size):
                    tmp = np.zeros((embed_size, embed_size), dtype=np.float32)
                    sh, sw = embedded.shape
                    tmp[:sh, :sw] = embedded
                    embedded = tmp
            else:
                embedded = np.zeros((embed_size, embed_size), dtype=np.float32)
            channel_embeds.append(embedded.ravel())

        embeddings.append(np.concatenate(channel_embeds).astype(np.float64))

    if len(embeddings) < 3:
        return 1.0

    try:
        X = np.array(embeddings)
        scale = np.abs(X).max()
        if scale > 0:
            X = X / scale
        X = X - X.mean(axis=0)
        _, S, _ = np.linalg.svd(X, full_matrices=False)
        eigenvalues = S**2 / (len(X) - 1)
        trace = eigenvalues.sum()
        trace_sq = (eigenvalues**2).sum()
        if trace_sq < 1e-20:
            return 1.0
        return float(trace**2 / trace_sq)
    except Exception:
        return 1.0


def measure_all_mc(pattern, prev_mass, prev_values, history_entries,
                   grid_mc, kernel_ffts, coupling, channel_configs, grid_size,
                   step_num=-1):
    """Compute all affect dimensions for a multi-channel pattern.

    Returns AffectState with integration = total Phi (MIP of spatial + channel).
    """
    phi_total, phi_spatial, phi_channel = measure_integration_mc(
        grid_mc, kernel_ffts, coupling, pattern.cells,
        channel_configs, grid_size,
    )

    return AffectState(
        valence=measure_valence(pattern.mass, prev_mass),
        arousal=measure_arousal_mc(pattern.values, prev_values),
        integration=phi_total,
        effective_rank=measure_effective_rank_mc(history_entries),
        self_model_salience=measure_self_model_salience(history_entries),
        pattern_id=pattern.id,
        step=step_num,
        mass=pattern.mass,
        size=pattern.size,
    ), phi_spatial, phi_channel
