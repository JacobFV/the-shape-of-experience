"""V18: Boundary-Dependent Lenia — The Cognitive-CA Dictionary.

Three measurement experiments (5: counterfactual, 6: self-model, 9: normativity)
hit a sensory-motor coupling wall. The FFT convolution integrates over the full
128x128 grid, so patterns are always internally driven (rho_sync ~ 0 from cycle 0).
There is no reactive-to-autonomous transition because the starting point is already
autonomous.

V18 introduces an INSULATION FIELD that creates genuine boundaries:
- Boundary cells (insulation ~ 0): receive external FFT convolution signals
- Interior cells (insulation ~ 1): receive only local recurrence signals
- Small patterns: insulation ~ 0 everywhere -> pure V15 behavior (reactive)
- Large patterns: insulated cores -> internal dynamics dominate (autonomous)

This enables the reactive-to-autonomous transition that Exps 5, 6, 9 need.

Architecture (dual-signal update):
  Step 0: Compute insulation field (erosion + distance transform + sigmoid)
  Step 1a: External signal = FFT(grid) * kernels * content_sim * (1 - insulation)
  Step 1b: Internal signal = FFT(grid * insulation) * local_kernels * insulation
  Step 1c: Combine effective_potentials = external + internal
  Steps 2-9: Same as V15 (coupling, growth, resource, memory, chemotaxis, etc.)

Branches from V15 (memory + motor channels). NOT V17 (no signal fields).

Channel layout (C=16):
  - Channels 0-11: Regular growth-function channels
  - Channels 12-13: Memory channels (EMA dynamics)
  - Channels 14-15: Motor channels (chemotaxis)

Cognitive dictionary entries this enables:
  - Knows:   C_wm > 0 (boundary is information gateway)
  - Attends: selective boundary permeability (content-sim modulates entry)
  - Imagines: offline trajectory during rho_sync drop
  - Decides:  pre-motor counterfactual branching through boundary
"""

import jax
import jax.numpy as jnp
from jax import random, jit, lax
import numpy as np
from functools import partial

from v13_substrate import (
    _compute_similarity_field,
    make_box_kernel_fft, init_embedding,
)
from v14_substrate import _compute_velocity_field, _advect_grid
from v15_substrate import (
    generate_v15_config, init_v15,
    _update_memory_channels,
    generate_resource_patches, compute_patch_regen_mask,
)
from v11_substrate_hd import (
    generate_coupling_matrix, make_kernels_fft_hd,
    init_soup_hd,
)


# ============================================================================
# Constants
# ============================================================================

INSULATION_EVERY = 5   # Recompute insulation every K steps (amortizes cost)
MAX_EROSION_DIST = 8   # Max distance for Chebyshev distance transform


# ============================================================================
# Configuration
# ============================================================================

def generate_v18_config(C=16, N=128, seed=42, similarity_radius=5,
                        d_embed=None, alpha=1.0,
                        chemotaxis_strength=0.5,
                        motor_channels=2,
                        motor_sensitivity=5.0,
                        motor_threshold=0.3,
                        max_speed=1.5,
                        memory_channels=2,
                        memory_lambdas=None,
                        n_resource_patches=4,
                        patch_shift_period=500,
                        # V18-specific
                        boundary_width=1.0,
                        insulation_beta=5.0,
                        internal_kernel_sigma=2.0,
                        internal_gain=1.0,
                        activity_threshold=0.05,
                        recurrence_bandwidth=2):
    """Generate V18 config (V15 + insulation field for boundary-dependent dynamics).

    New parameters:
        boundary_width: How thick the "sensory membrane" is (cells from edge
            where transition from boundary to interior occurs). Default 1.0
            (calibrated via Phase B: gives ~10% interior at C=16, N=128).
        insulation_beta: Sigmoid steepness for boundary/interior distinction.
        internal_kernel_sigma: Gaussian sigma for short-range internal kernels.
        internal_gain: Strength of internal processing vs external sensing.
            Calibrated at 1.0 so interior cells maintain themselves.
        activity_threshold: Mean channel activity to count as "pattern" vs
            "environment". Default 0.10.
        recurrence_bandwidth: Channel-space bandwidth for internal coupling.
            Narrower than external coupling (default 2 vs 3).
    """
    config = generate_v15_config(
        C=C, N=N, seed=seed, similarity_radius=similarity_radius,
        d_embed=d_embed, alpha=alpha,
        chemotaxis_strength=chemotaxis_strength,
        motor_channels=motor_channels,
        motor_sensitivity=motor_sensitivity,
        motor_threshold=motor_threshold,
        max_speed=max_speed,
        memory_channels=memory_channels,
        memory_lambdas=memory_lambdas,
        n_resource_patches=n_resource_patches,
        patch_shift_period=patch_shift_period,
    )

    # V18 insulation parameters
    config['boundary_width'] = boundary_width
    config['insulation_beta'] = insulation_beta
    config['internal_kernel_sigma'] = internal_kernel_sigma
    config['internal_gain'] = internal_gain
    config['activity_threshold'] = activity_threshold
    config['recurrence_bandwidth'] = recurrence_bandwidth

    return config


# ============================================================================
# Insulation Field
# ============================================================================

def _erode_mask(mask):
    """Single erosion step: min over 3x3 neighborhood (Chebyshev distance).

    Pure JAX — no scipy. Uses 8 rolled copies + element-wise min.
    """
    result = mask
    for di in range(-1, 2):
        for dj in range(-1, 2):
            if di == 0 and dj == 0:
                continue
            result = jnp.minimum(
                result,
                jnp.roll(jnp.roll(mask, di, axis=0), dj, axis=1))
    return result


def _compute_insulation_field(grid, activity_threshold, insulation_beta,
                               boundary_width):
    """Compute insulation field from grid state.

    Returns (N, N) field in [0, 1]:
      ~0 at boundary (external signal passes through)
      ~1 deep inside pattern interior (internal recurrence dominates)

    Algorithm:
      1. Threshold mean activity to get pattern mask
      2. Iterated erosion to compute Chebyshev distance to edge
      3. Sigmoid to create smooth boundary/interior transition

    Small patterns (radius < boundary_width) get insulation ~ 0 everywhere,
    so they behave like pure V15 (fully reactive). Large patterns develop
    insulated cores with genuine internal dynamics.
    """
    activity = jnp.mean(grid, axis=0)  # (N, N)
    pattern_mask = (activity > activity_threshold).astype(jnp.float32)

    # Distance to edge via iterated erosion (Chebyshev approximation)
    current = pattern_mask
    dist_to_edge = jnp.zeros_like(pattern_mask)
    for _ in range(MAX_EROSION_DIST):
        current = _erode_mask(current)
        dist_to_edge = dist_to_edge + current

    # Sigmoid: smooth boundary -> interior transition
    insulation = jax.nn.sigmoid(
        insulation_beta * (dist_to_edge - boundary_width))
    return insulation


def compute_insulation_metrics(grid, config):
    """Compute insulation metrics for logging (call outside JIT)."""
    ins = np.array(_compute_insulation_field(
        jnp.array(grid),
        config['activity_threshold'],
        config['insulation_beta'],
        config['boundary_width'],
    ))
    return {
        'insulation_mean': float(np.mean(ins)),
        'insulation_max': float(np.max(ins)),
        'interior_fraction': float(np.mean(ins > 0.5)),
        'boundary_fraction': float(np.mean((ins > 0.1) & (ins < 0.9))),
    }


# ============================================================================
# Internal Kernel FFTs
# ============================================================================

def make_internal_kernel_ffts(C, N, sigma=2.0, max_radius=5):
    """Pre-compute short-range Gaussian kernel FFTs for internal recurrence.

    All channels share the same kernel shape (differentiation comes from
    the recurrence_coupling matrix). Short-range (sigma=2, r<=5) means
    internal dynamics are genuinely local — unlike the external ring kernels
    (radii 5-25) which integrate over larger neighborhoods.
    """
    y, x = np.mgrid[-max_radius:max_radius+1, -max_radius:max_radius+1]
    k = np.exp(-(x**2 + y**2) / (2 * sigma**2)).astype(np.float32)
    k = k / k.sum()

    # Pad and center for periodic convolution
    k_padded = np.zeros((N, N), dtype=np.float32)
    k_padded[:2*max_radius+1, :2*max_radius+1] = k
    k_padded = np.roll(k_padded, -max_radius, axis=0)
    k_padded = np.roll(k_padded, -max_radius, axis=1)
    kf = jnp.array(np.fft.rfft2(k_padded))

    # Same kernel for all channels
    return jnp.stack([kf] * C, axis=0)  # (C, N, N//2+1)


# ============================================================================
# Recurrence Coupling
# ============================================================================

def generate_recurrence_coupling(C, bandwidth=2, seed=42):
    """Narrow-band recurrence coupling for internal processing.

    Narrower than external coupling (bandwidth=2 vs 3), so internal channels
    mostly talk to their immediate neighbors in channel space. This creates
    local "processing circuits" within the pattern interior.
    """
    return generate_coupling_matrix(C, bandwidth=bandwidth, seed=seed + 2000)


# ============================================================================
# Core Physics
# ============================================================================

@partial(jit, static_argnums=(6, 25, 28))
def run_chunk_v18(grid, resource,                       # 0, 1
                  kernel_ffts, internal_kernel_ffts,     # 2, 3
                  coupling, rng,                         # 4, 5
                  n_steps,                               # 6 - STATIC
                  dt, channel_mus, channel_sigmas,       # 7, 8, 9
                  coupling_row_sums,                     # 10
                  noise_amp,                             # 11
                  resource_consume, resource_regen,      # 12, 13
                  resource_max, resource_half_sat,       # 14, 15
                  decay_rate, maintenance_rate,          # 16, 17
                  h_embed, tau, gate_beta, alpha,        # 18, 19, 20, 21
                  box_fft,                               # 22
                  chemotaxis_strength, motor_sensitivity, # 23, 24
                  motor_channels,                        # 25 - STATIC
                  motor_threshold, max_speed,            # 26, 27
                  memory_channels,                       # 28 - STATIC
                  memory_lambdas_arr, regen_mask,        # 29, 30
                  # V18-specific
                  insulation_beta, boundary_width,       # 31, 32
                  internal_gain, activity_threshold,     # 33, 34
                  recurrence_coupling, recurrence_crs):  # 35, 36
    """Run n_steps of V18 boundary-dependent Lenia.

    Dual-signal update rule:
      External: FFT(grid) * kernels * content_sim * (1 - insulation)
      Internal: FFT(grid * insulation) * local_kernels * recurrence * insulation
      Combined: external + internal -> growth -> resource -> memory -> chemotaxis

    The insulation field is recomputed every INSULATION_EVERY steps (amortized).
    """
    C = grid.shape[0]
    N = grid.shape[1]
    motor_start_idx = C - motor_channels
    memory_start = C - motor_channels - memory_channels

    # Initialize step counter and insulation cache
    ins_init = jnp.zeros((N, N), dtype=jnp.float32)

    def body(carry, _):
        g, r, k, step, ins = carry
        k, k_noise = random.split(k)

        # ---- STEP 0: Insulation field (recompute every K steps) ----
        ins = lax.cond(
            step % INSULATION_EVERY == 0,
            lambda op: _compute_insulation_field(
                op[0], activity_threshold, insulation_beta, boundary_width),
            lambda op: op[1],
            (g, ins),
        )

        # ---- STEP 1a: External signal (boundary-gated) ----
        g_fft = jnp.fft.rfft2(g)
        external = jnp.fft.irfft2(g_fft * kernel_ffts, s=(N, N))

        # Content-similarity modulation (attention at the boundary)
        sim_field = _compute_similarity_field(
            g, h_embed, tau, gate_beta, box_fft)
        modulation = 1.0 + alpha * sim_field
        external = external * modulation[None, :, :]

        # Gate: boundary cells receive external signal
        external_gated = external * (1.0 - ins)[None, :, :]

        # ---- STEP 1b: Internal signal (interior-gated) ----
        # Mask grid by insulation: only interior state participates
        interior_grid = g * ins[None, :, :]
        interior_fft = jnp.fft.rfft2(interior_grid)
        internal = jnp.fft.irfft2(
            interior_fft * internal_kernel_ffts, s=(N, N))

        # Recurrence coupling: mix internal channels
        internal_cross = jnp.einsum(
            'cd,dnm->cnm', recurrence_coupling, internal)
        internal_cross = internal_cross / recurrence_crs[:, None, None]

        # Gate: interior cells receive internal signal, scaled by gain
        internal_gated = internal_gain * internal_cross * ins[None, :, :]

        # ---- STEP 1c: Combine effective potentials ----
        potentials = external_gated + internal_gated

        # ---- STEP 2: Cross-channel coupling gate ----
        cross = jnp.einsum('cd,dnm->cnm', coupling, potentials)
        cross = cross / coupling_row_sums[:, None, None]
        cross_gate = jax.nn.sigmoid(5.0 * (cross - 0.3))
        potentials_gated = (potentials * (1.0 - 0.5 * cross_gate)
                            + cross * 0.5 * cross_gate)

        # ---- STEP 3: Growth function ----
        mus = channel_mus[:, None, None]
        sigs = channel_sigmas[:, None, None]
        growth = 2.0 * jnp.exp(
            -((potentials_gated - mus)**2) / (2 * sigs**2)) - 1.0

        # ---- STEP 4: Resource modulation ----
        resource_factor = r / (r + resource_half_sat)
        growth_modulated = growth * resource_factor[None, :, :]

        # ---- STEP 5: State update ----
        g_new = g + dt * growth_modulated

        # ---- STEP 5a: Memory channel EMA (V15) ----
        g_new = _update_memory_channels(
            g_new, r, memory_start, memory_channels, memory_lambdas_arr)

        # ---- STEP 5b: Chemotactic advection (V14) ----
        vx, vy = _compute_velocity_field(
            g_new, r, chemotaxis_strength, motor_start_idx,
            motor_sensitivity, motor_threshold, max_speed)
        g_new = _advect_grid(g_new, vx, vy)

        # ---- STEP 6: Noise ----
        noise = noise_amp * random.normal(k_noise, g.shape)
        g_new = g_new + noise

        # ---- STEP 7: Clamp ----
        g_new = jnp.clip(g_new, 0.0, 1.0)

        # ---- STEP 8: Resource dynamics (with patch mask) ----
        consumption = resource_consume * jnp.mean(g_new, axis=0)
        maintenance = maintenance_rate * (
            jnp.mean(g_new, axis=0) > 0.05).astype(jnp.float32)
        local_regen = resource_regen * regen_mask
        r_new = (r + local_regen * (1.0 - r / resource_max)
                 - consumption - maintenance)
        r_new = jnp.clip(r_new, 0.0, resource_max)

        # ---- STEP 9: Decay from resource depletion ----
        depletion_mask = (r_new < 0.05).astype(jnp.float32)
        g_new = g_new * (1.0 - decay_rate * depletion_mask[None, :, :])

        return (g_new, r_new, k, step + 1, ins), None

    (grid_final, resource_final, rng_final, _, _), _ = lax.scan(
        body, (grid, resource, rng, jnp.int32(0), ins_init),
        None, length=n_steps
    )

    return grid_final, resource_final, rng_final


# ============================================================================
# Initialization
# ============================================================================

def init_v18(config, seed=42):
    """Initialize V18 substrate.

    Returns:
        grid: (C, N, N) initial state
        resource: (N, N) resource field
        h_embed: (d_embed, C) embedding matrix
        kernel_ffts: (C, N, N//2+1) external kernel FFTs
        internal_kernel_ffts: (C, N, N//2+1) short-range internal kernel FFTs
        coupling: (C, C) cross-channel coupling matrix
        coupling_row_sums: (C,) row normalization for coupling
        recurrence_coupling: (C, C) internal recurrence coupling
        recurrence_crs: (C,) row normalization for recurrence
        box_fft: (N, N//2+1) box kernel FFT for similarity
    """
    # Base V15 initialization
    grid, resource, h_embed, kernel_ffts, coupling, coupling_row_sums, box_fft = \
        init_v15(config, seed=seed)

    C = config['n_channels']
    N = config['grid_size']

    # Internal kernel FFTs (short-range Gaussian)
    internal_kernel_ffts = make_internal_kernel_ffts(
        C, N, sigma=config['internal_kernel_sigma'])

    # Recurrence coupling (narrow-band)
    recurrence_coupling_np = generate_recurrence_coupling(
        C, bandwidth=config['recurrence_bandwidth'], seed=seed)
    recurrence_coupling = jnp.array(recurrence_coupling_np)
    recurrence_crs = jnp.array(recurrence_coupling_np.sum(axis=1))

    return (grid, resource, h_embed, kernel_ffts, internal_kernel_ffts,
            coupling, coupling_row_sums, recurrence_coupling, recurrence_crs,
            box_fft)


# ============================================================================
# Convenience runner
# ============================================================================

def run_v18_chunk(grid, resource, h_embed, kernel_ffts, internal_kernel_ffts,
                  config, coupling, coupling_row_sums,
                  recurrence_coupling, recurrence_crs,
                  rng, n_steps=50, drought=False,
                  box_fft=None, regen_mask=None):
    """Run a chunk of V18 steps with optional drought."""
    regen = 0.001 if drought else config['resource_regen']

    if box_fft is None:
        box_fft = make_box_kernel_fft(
            config['similarity_radius'], config['grid_size'])

    if regen_mask is None:
        N = config['grid_size']
        regen_mask = jnp.ones((N, N))

    memory_lambdas_arr = jnp.array(config['memory_lambdas'])

    return run_chunk_v18(
        grid, resource,
        kernel_ffts, internal_kernel_ffts,
        coupling, rng,
        n_steps,
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
        config['chemotaxis_strength'],
        config['motor_sensitivity'],
        int(config['motor_channels']),
        config['motor_threshold'],
        config['max_speed'],
        int(config['memory_channels']),
        memory_lambdas_arr,
        regen_mask,
        # V18
        config['insulation_beta'],
        config['boundary_width'],
        config['internal_gain'],
        config['activity_threshold'],
        recurrence_coupling,
        recurrence_crs,
    )


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == '__main__':
    import time

    print("V18 Boundary-Dependent Lenia — Smoke Test")
    print("=" * 60)

    config = generate_v18_config(
        C=8, N=64, similarity_radius=5,
        chemotaxis_strength=0.5,
        memory_channels=2,
        motor_channels=2,
        boundary_width=2.0,
        insulation_beta=5.0,
        internal_gain=1.0,
        activity_threshold=0.10,
    )

    (grid, resource, h_embed, kernel_ffts, internal_kernel_ffts,
     coupling, coupling_row_sums, recurrence_coupling, recurrence_crs,
     box_fft) = init_v18(config, seed=42)

    rng = random.PRNGKey(0)

    # Resource patches
    patches = generate_resource_patches(
        config['grid_size'], config['n_resource_patches'], seed=42)
    regen_mask = compute_patch_regen_mask(
        config['grid_size'], patches, step=0,
        shift_period=config['patch_shift_period'])

    print(f"Grid: {grid.shape}, Resource: {resource.shape}")
    print(f"External kernels: {kernel_ffts.shape}")
    print(f"Internal kernels: {internal_kernel_ffts.shape}")
    print(f"Recurrence coupling: {recurrence_coupling.shape}")
    print(f"Channel layout: regular=0-{config['n_regular']-1}, "
          f"memory={config['memory_start']}-"
          f"{config['memory_start']+config['memory_channels']-1}, "
          f"motor={config['motor_start']}-{config['n_channels']-1}")
    print(f"V18 params: boundary_width={config['boundary_width']}, "
          f"insulation_beta={config['insulation_beta']}, "
          f"internal_gain={config['internal_gain']}, "
          f"activity_threshold={config['activity_threshold']}")
    print(f"Initial grid mean: {float(grid.mean()):.4f}")
    print()

    # Check initial insulation field
    ins = _compute_insulation_field(
        grid, config['activity_threshold'],
        config['insulation_beta'], config['boundary_width'])
    print(f"Initial insulation: mean={float(ins.mean()):.4f}, "
          f"max={float(ins.max()):.4f}, "
          f"interior (>0.5): {float((ins > 0.5).mean()):.1%}")

    # JIT compile
    print("\nJIT compiling (first call)...", end=" ", flush=True)
    t0 = time.time()
    grid2, resource2, rng = run_v18_chunk(
        grid, resource, h_embed, kernel_ffts, internal_kernel_ffts,
        config, coupling, coupling_row_sums,
        recurrence_coupling, recurrence_crs,
        rng, n_steps=50, box_fft=box_fft, regen_mask=regen_mask
    )
    jax.block_until_ready(grid2)
    t1 = time.time()
    print(f"{t1-t0:.1f}s")
    print(f"  Grid mean: {float(grid2.mean()):.4f}")
    print(f"  Resource mean: {float(resource2.mean()):.4f}")

    # Check insulation after 50 steps
    ins2 = _compute_insulation_field(
        grid2, config['activity_threshold'],
        config['insulation_beta'], config['boundary_width'])
    print(f"  Insulation: mean={float(ins2.mean()):.4f}, "
          f"max={float(ins2.max()):.4f}, "
          f"interior (>0.5): {float((ins2 > 0.5).mean()):.1%}")

    # Run 200 steps (4 chunks)
    print("\nRunning 200 steps (4 chunks)...", end=" ", flush=True)
    t0 = time.time()
    for _ in range(4):
        grid2, resource2, rng = run_v18_chunk(
            grid2, resource2, h_embed, kernel_ffts, internal_kernel_ffts,
            config, coupling, coupling_row_sums,
            recurrence_coupling, recurrence_crs,
            rng, n_steps=50, box_fft=box_fft, regen_mask=regen_mask
        )
    jax.block_until_ready(grid2)
    t1 = time.time()
    throughput = 200 / (t1 - t0)
    print(f"{t1-t0:.2f}s ({throughput:.0f} steps/s)")

    from v11_patterns import detect_patterns_mc
    grid_np = np.array(grid2)
    patterns = detect_patterns_mc(grid_np, threshold=0.15)
    print(f"  Patterns: {len(patterns)}")
    print(f"  Grid mean: {float(grid2.mean()):.4f}")

    # Check memory channels
    mem_start = config['memory_start']
    mem_end = mem_start + config['memory_channels']
    print(f"  Memory ch mean: {float(grid2[mem_start:mem_end].mean()):.4f}")

    # Insulation metrics
    ins3 = _compute_insulation_field(
        grid2, config['activity_threshold'],
        config['insulation_beta'], config['boundary_width'])
    metrics = compute_insulation_metrics(grid2, config)
    print(f"  Insulation: mean={metrics['insulation_mean']:.4f}, "
          f"interior={metrics['interior_fraction']:.1%}, "
          f"boundary={metrics['boundary_fraction']:.1%}")

    # Test drought
    print("\nRunning 200 steps under drought...")
    for _ in range(4):
        grid2, resource2, rng = run_v18_chunk(
            grid2, resource2, h_embed, kernel_ffts, internal_kernel_ffts,
            config, coupling, coupling_row_sums,
            recurrence_coupling, recurrence_crs,
            rng, n_steps=50, drought=True,
            box_fft=box_fft, regen_mask=regen_mask
        )
    jax.block_until_ready(grid2)
    grid_np = np.array(grid2)
    patterns = detect_patterns_mc(grid_np, threshold=0.15)
    print(f"  Patterns after drought: {len(patterns)}")
    print(f"  Resource mean: {float(resource2.mean()):.4f}")

    print("\nV18 smoke test complete.")
