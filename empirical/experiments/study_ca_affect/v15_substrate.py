"""V15: Temporal Lenia — Memory Channels for World Model Prerequisites.

V14 validated Phase A (directed foraging) but patterns are purely reactive:
they follow current gradients with no temporal integration. World models
(Experiment 2) require patterns that know something about the future
beyond what's readable from the present.

V15 adds TWO changes to V14:
1. MEMORY CHANNELS: Dedicated channels with exponential-moving-average (EMA)
   dynamics instead of growth-function dynamics. They track slow statistics
   of the pattern's state, creating temporal memory.

2. OSCILLATING RESOURCE PATCHES: Resources regenerate in discrete zones that
   shift over time, creating temporal structure. This rewards patterns that
   remember where resources were (and anticipate where they'll be), not just
   follow current gradients.

Channel layout (C=16):
  - Channels 0-11: Regular growth-function channels (V13 physics)
  - Channels 12-13: Memory channels (EMA dynamics, evolvable time constant)
  - Channels 14-15: Motor channels (V14 chemotaxis)

Memory dynamics:
  m_new = (1 - λ) * m_old + λ * input
  input = mean(regular_channels) * local_resource_status
  λ ∈ [0.001, 0.5] — evolvable per-memory-channel time constant

The memory channels participate in:
  - Content-similarity coupling (same as regular channels)
  - Motor gating (memory influences movement decisions)
  - Cross-channel coupling (memory feeds back to regular channels)

This is the minimal substrate for world model emergence:
  - EMA provides temporal integration (prerequisite)
  - Oscillating resources provide temporal structure (forcing function)
  - Memory-motor coupling provides planning advantage (fitness pressure)
"""

import jax
import jax.numpy as jnp
from jax import random, jit, lax
import numpy as np
from functools import partial

from v13_substrate import (
    generate_v13_config, init_v13, _compute_similarity_field,
    make_box_kernel_fft, init_embedding,
)
from v14_substrate import (
    generate_v14_config, _compute_velocity_field, _advect_grid,
)
from v11_substrate_hd import (
    generate_hd_config, generate_coupling_matrix, make_kernels_fft_hd,
    init_soup_hd,
)


# ============================================================================
# Configuration
# ============================================================================

def generate_v15_config(C=16, N=128, seed=42, similarity_radius=5,
                        d_embed=None, alpha=1.0,
                        chemotaxis_strength=0.5,
                        motor_channels=2,
                        motor_sensitivity=5.0,
                        motor_threshold=0.3,
                        max_speed=1.5,
                        memory_channels=2,
                        memory_lambdas=None,
                        n_resource_patches=4,
                        patch_shift_period=500):
    """Generate V15 config (extends V14 with memory + resource patches).

    New parameters:
        memory_channels: how many channels use EMA dynamics
        memory_lambdas: per-channel EMA rate (default: [0.01, 0.1])
        n_resource_patches: number of oscillating resource zones
        patch_shift_period: steps between resource patch shifts
    """
    config = generate_v14_config(
        C=C, N=N, seed=seed, similarity_radius=similarity_radius,
        d_embed=d_embed, alpha=alpha,
        chemotaxis_strength=chemotaxis_strength,
        motor_channels=motor_channels,
        motor_sensitivity=motor_sensitivity,
        motor_threshold=motor_threshold,
        max_speed=max_speed,
    )

    # Memory channel parameters
    config['memory_channels'] = memory_channels
    if memory_lambdas is None:
        # Default: one slow (long-term), one fast (working memory)
        memory_lambdas = [0.01, 0.1]
    config['memory_lambdas'] = memory_lambdas[:memory_channels]

    # Resource patch parameters
    config['n_resource_patches'] = n_resource_patches
    config['patch_shift_period'] = patch_shift_period

    # Validate channel allocation
    n_regular = C - memory_channels - motor_channels
    assert n_regular >= 4, f"Need at least 4 regular channels, got {n_regular}"
    config['n_regular'] = n_regular
    config['memory_start'] = n_regular  # channels [n_regular, n_regular+memory_channels)
    config['motor_start'] = n_regular + memory_channels  # channels [motor_start, C)

    return config


# ============================================================================
# Memory Channel Dynamics
# ============================================================================

def _update_memory_channels(grid, resource, memory_start, memory_channels,
                            memory_lambdas_arr):
    """Update memory channels with EMA dynamics.

    Memory input = mean(regular channels) * resource_level
    This gives memory channels information about both the pattern's state
    and the resource environment.

    Args:
        grid: (C, N, N) full state
        resource: (N, N) resource field
        memory_start: int (static) — index where memory channels begin
        memory_channels: int (static) — number of memory channels
        memory_lambdas_arr: (memory_channels,) EMA rates

    Returns: updated grid with new memory channel values
    """
    N = grid.shape[1]

    # Memory input: mean of regular channels * resource level
    # Use dynamic_slice to extract regular channels (avoids traced slice)
    regular = jax.lax.dynamic_slice(
        grid, (0, 0, 0), (memory_start, N, N))
    regular_mean = jnp.mean(regular, axis=0)  # (N, N)
    resource_signal = resource / (resource + 0.1)  # normalized [0, ~1]
    memory_input = regular_mean * resource_signal  # (N, N)

    # EMA update for each memory channel
    for i in range(memory_channels):
        ch_idx = memory_start + i
        lam = memory_lambdas_arr[i]
        old = grid[ch_idx]
        new = (1.0 - lam) * old + lam * memory_input
        grid = grid.at[ch_idx].set(new)

    return grid


# ============================================================================
# Oscillating Resource Patches
# ============================================================================

def generate_resource_patches(N, n_patches, seed=42):
    """Generate resource patch centers and radii.

    Returns (n_patches, 3) array: [center_y, center_x, radius]
    """
    rng = np.random.RandomState(seed)
    centers_y = rng.randint(N // 4, 3 * N // 4, size=n_patches)
    centers_x = rng.randint(N // 4, 3 * N // 4, size=n_patches)
    radii = rng.randint(N // 8, N // 4, size=n_patches)
    return np.stack([centers_y, centers_x, radii], axis=1)


def compute_patch_regen_mask(N, patches, step, shift_period):
    """Compute spatially-varying regeneration rate from oscillating patches.

    Patches shift position over time (circular orbit), creating temporal
    structure in the resource landscape.

    Returns: (N, N) regeneration multiplier ∈ [0.3, 2.0]
    """
    mask = jnp.ones((N, N)) * 0.3  # base low regen

    yy, xx = jnp.meshgrid(jnp.arange(N), jnp.arange(N), indexing='ij')

    for i in range(patches.shape[0]):
        cy, cx, r = int(patches[i, 0]), int(patches[i, 1]), int(patches[i, 2])

        # Circular shift: patch center orbits around its initial position
        phase = 2.0 * np.pi * step / shift_period + i * 2.0 * np.pi / patches.shape[0]
        orbit_r = N // 8
        cy_shifted = (cy + int(orbit_r * np.sin(phase))) % N
        cx_shifted = (cx + int(orbit_r * np.cos(phase))) % N

        # Periodic distance
        dy = jnp.minimum(jnp.abs(yy - cy_shifted), N - jnp.abs(yy - cy_shifted))
        dx = jnp.minimum(jnp.abs(xx - cx_shifted), N - jnp.abs(xx - cx_shifted))
        dist = jnp.sqrt(dy**2 + dx**2)

        # Smooth patch: high regen within radius, falls off outside
        patch_contribution = jnp.exp(-0.5 * (dist / max(r, 1))**2)
        mask = mask + 1.7 * patch_contribution  # peak regen = 0.3 + 1.7 = 2.0

    return jnp.clip(mask, 0.3, 2.0)


# ============================================================================
# Core Physics (extends V14)
# ============================================================================

@partial(jit, static_argnums=(5, 24, 27))
def run_chunk_v15(grid, resource, kernel_ffts, coupling, rng, n_steps,
                  dt, channel_mus, channel_sigmas, coupling_row_sums,
                  noise_amp, resource_consume, resource_regen,
                  resource_max, resource_half_sat, decay_rate,
                  maintenance_rate,
                  h_embed, tau, gate_beta, alpha, box_fft,
                  # V14 chemotaxis
                  chemotaxis_strength, motor_sensitivity,
                  motor_channels,  # static (arg 24)
                  motor_threshold, max_speed,
                  # V15 memory
                  memory_channels,  # static (arg 27)
                  memory_lambdas_arr,
                  regen_mask):
    """Run n_steps of V15 Temporal Lenia.

    Extends V14 with:
    - Memory channel EMA update (after growth, before advection)
    - Spatially-varying resource regen (from regen_mask)
    """
    C = grid.shape[0]
    N = grid.shape[1]
    motor_start_idx = C - motor_channels
    # Compute memory_start from static values
    memory_start = C - motor_channels - memory_channels

    def body(carry, _):
        g, r, k = carry
        k, k_noise = random.split(k)

        # ---- STEP 1: Spatial potentials via FFT (standard Lenia) ----
        g_fft = jnp.fft.rfft2(g)
        potentials = jnp.fft.irfft2(g_fft * kernel_ffts, s=(N, N))

        # ---- STEP 1b: Content-similarity modulation ----
        sim_field = _compute_similarity_field(
            g, h_embed, tau, gate_beta, box_fft)
        modulation = 1.0 + alpha * sim_field
        potentials = potentials * modulation[None, :, :]

        # ---- STEP 2: Cross-channel coupling gate ----
        cross = jnp.einsum('cd,dnm->cnm', coupling, potentials)
        cross = cross / coupling_row_sums[:, None, None]
        cross_gate = jax.nn.sigmoid(5.0 * (cross - 0.3))
        potentials_gated = potentials * (1.0 - 0.5 * cross_gate) + cross * 0.5 * cross_gate

        # ---- STEP 3: Growth function (regular channels only) ----
        mus = channel_mus[:, None, None]
        sigs = channel_sigmas[:, None, None]
        growth = 2.0 * jnp.exp(-((potentials_gated - mus)**2) / (2 * sigs**2)) - 1.0

        # ---- STEP 4: Resource modulation ----
        resource_factor = r / (r + resource_half_sat)
        growth_modulated = growth * resource_factor[None, :, :]

        # ---- STEP 5: State update (growth for ALL channels initially) ----
        g_new = g + dt * growth_modulated

        # ---- STEP 5a: MEMORY CHANNEL UPDATE (V15 addition) ----
        # Override memory channels with EMA dynamics
        g_new = _update_memory_channels(
            g_new, r, memory_start, memory_channels, memory_lambdas_arr)

        # ---- STEP 5b: CHEMOTACTIC ADVECTION (V14) ----
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
        maintenance = maintenance_rate * (jnp.mean(g_new, axis=0) > 0.05).astype(jnp.float32)
        # Spatially-varying regen from patch mask
        local_regen = resource_regen * regen_mask
        r_new = r + local_regen * (1.0 - r / resource_max) - consumption - maintenance
        r_new = jnp.clip(r_new, 0.0, resource_max)

        # ---- STEP 9: Decay from resource depletion ----
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

def init_v15(config, seed=42):
    """Initialize V15 substrate. Same as V14 init (all channels start
    with growth-function initialization; memory channels will naturally
    diverge through EMA dynamics).
    """
    return init_v13(config, seed=seed)


# ============================================================================
# Convenience runner
# ============================================================================

def run_v15_chunk(grid, resource, h_embed, kernel_ffts, config,
                  coupling, coupling_row_sums, rng, n_steps=50,
                  drought=False, box_fft=None, regen_mask=None):
    """Run a chunk of V15 steps with optional drought."""
    regen = 0.001 if drought else config['resource_regen']

    if box_fft is None:
        box_fft = make_box_kernel_fft(
            config['similarity_radius'], config['grid_size'])

    if regen_mask is None:
        N = config['grid_size']
        regen_mask = jnp.ones((N, N))  # uniform if no patches

    memory_lambdas_arr = jnp.array(config['memory_lambdas'])

    return run_chunk_v15(
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
        # V14 chemotaxis
        config['chemotaxis_strength'],
        config['motor_sensitivity'],
        int(config['motor_channels']),  # static
        config['motor_threshold'],
        config['max_speed'],
        # V15 memory
        int(config['memory_channels']),  # static
        memory_lambdas_arr,
        regen_mask,
    )


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == '__main__':
    import time

    print("V15 Temporal Lenia — Smoke Test")
    print("=" * 60)

    config = generate_v15_config(C=8, N=64, similarity_radius=5,
                                  chemotaxis_strength=0.5,
                                  memory_channels=2,
                                  motor_channels=2)
    grid, resource, h_embed, kernel_ffts, coupling, coupling_row_sums, box_fft = \
        init_v15(config, seed=42)

    rng = random.PRNGKey(0)

    # Generate resource patches
    patches = generate_resource_patches(config['grid_size'],
                                        config['n_resource_patches'],
                                        seed=42)
    regen_mask = compute_patch_regen_mask(
        config['grid_size'], patches, step=0,
        shift_period=config['patch_shift_period'])

    print(f"Grid: {grid.shape}, Resource: {resource.shape}")
    print(f"Channel layout: regular=0-{config['n_regular']-1}, "
          f"memory={config['memory_start']}-{config['memory_start']+config['memory_channels']-1}, "
          f"motor={config['motor_start']}-{config['n_channels']-1}")
    print(f"Memory lambdas: {config['memory_lambdas']}")
    print(f"Resource patches: {config['n_resource_patches']}, "
          f"shift period: {config['patch_shift_period']}")
    print(f"Regen mask range: [{float(regen_mask.min()):.2f}, {float(regen_mask.max()):.2f}]")
    print(f"Initial grid mean: {float(grid.mean()):.4f}")
    print()

    # JIT compile
    print("JIT compiling (first call)...", end=" ", flush=True)
    t0 = time.time()
    grid2, resource2, rng = run_v15_chunk(
        grid, resource, h_embed, kernel_ffts, config,
        coupling, coupling_row_sums, rng, n_steps=50,
        box_fft=box_fft, regen_mask=regen_mask
    )
    jax.block_until_ready(grid2)
    t1 = time.time()
    print(f"{t1-t0:.1f}s")
    print(f"  Grid mean: {float(grid2.mean()):.4f}")
    print(f"  Resource mean: {float(resource2.mean()):.4f}")

    # Check memory channels
    mem_start = config['memory_start']
    mem_end = mem_start + config['memory_channels']
    print(f"  Memory channels [{mem_start}:{mem_end}] mean: "
          f"{float(grid2[mem_start:mem_end].mean()):.4f}")

    # Run 200 steps
    print("\nRunning 200 steps (4 chunks)...", end=" ", flush=True)
    t0 = time.time()
    for _ in range(4):
        grid2, resource2, rng = run_v15_chunk(
            grid2, resource2, h_embed, kernel_ffts, config,
            coupling, coupling_row_sums, rng, n_steps=50,
            box_fft=box_fft, regen_mask=regen_mask
        )
    jax.block_until_ready(grid2)
    t1 = time.time()
    print(f"{t1-t0:.2f}s")

    from v11_patterns import detect_patterns_mc
    grid_np = np.array(grid2)
    patterns = detect_patterns_mc(grid_np, threshold=0.15)
    print(f"  Patterns: {len(patterns)}")
    print(f"  Grid mean: {float(grid2.mean()):.4f}")
    print(f"  Memory ch mean: {float(grid2[mem_start:mem_end].mean()):.4f}")

    # Test oscillating patches
    print("\nTesting patch oscillation...")
    mask0 = compute_patch_regen_mask(64, patches, step=0, shift_period=500)
    mask250 = compute_patch_regen_mask(64, patches, step=250, shift_period=500)
    mask_diff = float(jnp.abs(mask250 - mask0).mean())
    print(f"  Mean mask change (0 -> 250 steps): {mask_diff:.3f}")
    print(f"  (Should be > 0 — patches moved)")

    # Run under drought
    print("\nRunning 200 steps under drought...")
    for _ in range(4):
        grid2, resource2, rng = run_v15_chunk(
            grid2, resource2, h_embed, kernel_ffts, config,
            coupling, coupling_row_sums, rng, n_steps=50,
            drought=True, box_fft=box_fft, regen_mask=regen_mask
        )
    jax.block_until_ready(grid2)
    grid_np = np.array(grid2)
    patterns = detect_patterns_mc(grid_np, threshold=0.15)
    print(f"  Patterns after drought: {len(patterns)}")
    print(f"  Resource mean: {float(resource2.mean()):.4f}")

    print("\nV15 smoke test complete.")
