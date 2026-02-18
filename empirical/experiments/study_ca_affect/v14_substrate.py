"""V14: Chemotactic Lenia — Directed Motion via Resource-Gradient Advection.

V13 established content-based coupling (state-dependent interaction topology)
and produced patterns with measurable affect geometry, but hit the sensory-motor
coupling wall: patterns are internally driven from cycle 0, with no tight
action→observation loop.

V14 adds CHEMOTACTIC ADVECTION: patterns can move toward resources by generating
a velocity field from resource gradients, gated by internal state.

    v(x) = η · ∇R(x) · gate(s[motor_channels](x))

This creates a sensory-motor loop:
1. Pattern senses resource gradient at boundary (sensory)
2. Motor channels gate chemotactic response (processing)
3. Velocity field advects pattern toward resources (action)
4. New boundary observations result from new position (feedback)

Key differences from V13:
- Advection step: velocity field displaces all channel states
- Motor channels (last 2 of C): gate chemotaxis strength
- Resource gradient sensing: finite-difference ∇R
- Velocity smoothing: box blur prevents numerical artifacts
- Speed limit: max displacement per step

Everything else is V13: FFT convolution, content coupling, cross-channel
coupling gate, growth function, resources, noise, clamp, decay.
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
from v11_substrate_hd import (
    generate_hd_config, generate_coupling_matrix, make_kernels_fft_hd,
    init_soup_hd,
)


# ============================================================================
# Configuration
# ============================================================================

def generate_v14_config(C=16, N=128, seed=42, similarity_radius=5,
                        d_embed=None, alpha=1.0,
                        chemotaxis_strength=0.5,
                        motor_channels=2,
                        motor_sensitivity=5.0,
                        motor_threshold=0.3,
                        max_speed=1.5,
                        velocity_blur=3):
    """Generate V14 config (extends V13 with chemotaxis parameters).

    New parameters:
        chemotaxis_strength: η, base speed scaling
        motor_channels: how many channels (from the end) gate chemotaxis
        motor_sensitivity: sigmoid steepness for motor gating
        motor_threshold: sigmoid center for motor gating
        max_speed: max cells/step displacement
        velocity_blur: kernel size for velocity smoothing
    """
    config = generate_v13_config(
        C=C, N=N, seed=seed, similarity_radius=similarity_radius,
        d_embed=d_embed, alpha=alpha,
    )

    # Chemotaxis parameters
    config['chemotaxis_strength'] = chemotaxis_strength
    config['motor_channels'] = motor_channels
    config['motor_sensitivity'] = motor_sensitivity
    config['motor_threshold'] = motor_threshold
    config['max_speed'] = max_speed
    config['velocity_blur'] = velocity_blur

    return config


# ============================================================================
# Chemotactic Advection
# ============================================================================

def _compute_velocity_field(grid, resource, eta, motor_start_idx,
                            sensitivity, threshold, max_speed):
    """Compute velocity field from resource gradient and motor channels.

    Args:
        grid: (C, N, N) state
        resource: (N, N) resource field
        eta: chemotaxis strength
        motor_start_idx: index where motor channels start (C - n_motor)
        sensitivity: sigmoid steepness for motor gating
        threshold: sigmoid center for motor gating
        max_speed: max cells/step displacement

    Returns (vx, vy) each of shape (N, N).
    """
    # 1. Resource gradient (central finite differences, periodic)
    grad_x = (jnp.roll(resource, -1, axis=1) - jnp.roll(resource, 1, axis=1)) / 2.0
    grad_y = (jnp.roll(resource, -1, axis=0) - jnp.roll(resource, 1, axis=0)) / 2.0

    # 2. Motor channel gating
    # Use channels from motor_start_idx onward as motor activation
    motor_activation = jnp.mean(
        jax.lax.dynamic_slice(grid, (motor_start_idx, 0, 0),
                              (grid.shape[0] - motor_start_idx, grid.shape[1], grid.shape[2])),
        axis=0)  # (N, N)
    gate = jax.nn.sigmoid(sensitivity * (motor_activation - threshold))

    # 3. Raw velocity
    vx = eta * grad_x * gate
    vy = eta * grad_y * gate

    # 4. Smooth velocity field (3x3 box blur, unrolled)
    vx_smooth = vx
    vy_smooth = vy
    for di in range(-1, 2):
        for dj in range(-1, 2):
            if di == 0 and dj == 0:
                continue
            vx_smooth = vx_smooth + jnp.roll(jnp.roll(vx, di, axis=0), dj, axis=1)
            vy_smooth = vy_smooth + jnp.roll(jnp.roll(vy, di, axis=0), dj, axis=1)
    vx = vx_smooth / 9.0
    vy = vy_smooth / 9.0

    # 5. Speed limit
    speed = jnp.sqrt(vx**2 + vy**2 + 1e-8)
    scale = jnp.minimum(1.0, max_speed / speed)
    vx = vx * scale
    vy = vy * scale

    return vx, vy


def _advect_grid(grid, vx, vy):
    """Advect grid channels by velocity field using bilinear interpolation.

    Uses backward advection: for each destination cell (i,j), look up
    where it came from at (i - vy, j - vx) and interpolate.

    Periodic boundary conditions (toroidal).
    """
    C, N, _ = grid.shape

    # Create coordinate grids
    iy = jnp.arange(N, dtype=jnp.float32)
    ix = jnp.arange(N, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(iy, ix, indexing='ij')

    # Source coordinates (backward advection)
    src_x = (xx - vx) % N
    src_y = (yy - vy) % N

    # Bilinear interpolation indices
    x0 = jnp.floor(src_x).astype(jnp.int32) % N
    x1 = (x0 + 1) % N
    y0 = jnp.floor(src_y).astype(jnp.int32) % N
    y1 = (y0 + 1) % N

    # Interpolation weights
    wx = src_x - jnp.floor(src_x)
    wy = src_y - jnp.floor(src_y)

    # Interpolate each channel
    def interp_channel(ch):
        val = (ch[y0, x0] * (1 - wx) * (1 - wy) +
               ch[y0, x1] * wx * (1 - wy) +
               ch[y1, x0] * (1 - wx) * wy +
               ch[y1, x1] * wx * wy)
        return val

    result = jax.vmap(interp_channel)(grid)  # (C, N, N)
    return result


# ============================================================================
# Core Physics
# ============================================================================

@partial(jit, static_argnums=(5, 24))
def run_chunk_v14(grid, resource, kernel_ffts, coupling, rng, n_steps,
                  dt, channel_mus, channel_sigmas, coupling_row_sums,
                  noise_amp, resource_consume, resource_regen,
                  resource_max, resource_half_sat, decay_rate,
                  maintenance_rate,
                  h_embed, tau, gate_beta, alpha, box_fft,
                  # V14-specific
                  chemotaxis_strength, motor_sensitivity,
                  motor_channels,  # static (arg 24)
                  motor_threshold, max_speed):
    """Run n_steps of V14 chemotactic Lenia.

    Same as V13 but with advection step between growth and noise.
    motor_channels is static (used for array slicing).
    """
    C = grid.shape[0]
    N = grid.shape[1]
    motor_start_idx = C - motor_channels

    def body(carry, _):
        g, r, k = carry
        k, k_noise = random.split(k)

        # ---- STEP 1: Spatial potentials via FFT (standard Lenia) ----
        g_fft = jnp.fft.rfft2(g)  # (C, N, N//2+1)
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

        # ---- STEP 3: Growth function ----
        mus = channel_mus[:, None, None]
        sigs = channel_sigmas[:, None, None]
        growth = 2.0 * jnp.exp(-((potentials_gated - mus)**2) / (2 * sigs**2)) - 1.0

        # ---- STEP 4: Resource modulation ----
        resource_factor = r / (r + resource_half_sat)
        growth_modulated = growth * resource_factor[None, :, :]

        # ---- STEP 5: State update (growth) ----
        g_new = g + dt * growth_modulated

        # ---- STEP 5b: CHEMOTACTIC ADVECTION (V14 addition) ----
        vx, vy = _compute_velocity_field(
            g_new, r, chemotaxis_strength, motor_start_idx,
            motor_sensitivity, motor_threshold, max_speed)
        g_new = _advect_grid(g_new, vx, vy)

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

def init_v14(config, seed=42):
    """Initialize V14 substrate.

    Same outputs as V13 init — all existing measurement code works unchanged.
    """
    return init_v13(config, seed=seed)


# ============================================================================
# Convenience runner
# ============================================================================

def run_v14_chunk(grid, resource, h_embed, kernel_ffts, config,
                  coupling, coupling_row_sums, rng, n_steps=100,
                  drought=False, box_fft=None):
    """Run a chunk of V14 steps with optional drought."""
    regen = 0.001 if drought else config['resource_regen']

    if box_fft is None:
        box_fft = make_box_kernel_fft(
            config['similarity_radius'], config['grid_size'])

    return run_chunk_v14(
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
        # V14-specific
        config['chemotaxis_strength'],
        config['motor_sensitivity'],
        int(config['motor_channels']),  # static arg
        config['motor_threshold'],
        config['max_speed'],
    )


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == '__main__':
    import time

    print("V14 Chemotactic Lenia — Smoke Test")
    print("=" * 60)

    config = generate_v14_config(C=8, N=64, similarity_radius=5,
                                  chemotaxis_strength=0.5)
    grid, resource, h_embed, kernel_ffts, coupling, coupling_row_sums, box_fft = \
        init_v14(config, seed=42)

    rng = random.PRNGKey(0)

    print(f"Grid: {grid.shape}, Resource: {resource.shape}")
    print(f"Chemotaxis: η={config['chemotaxis_strength']}, "
          f"motor_ch={config['motor_channels']}, "
          f"max_speed={config['max_speed']}")
    print(f"Initial grid mean: {float(grid.mean()):.4f}")
    print(f"Initial resource mean: {float(resource.mean()):.4f}")
    print()

    # JIT compile
    print("JIT compiling (first call)...")
    t0 = time.time()
    grid2, resource2, rng = run_v14_chunk(
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

    # Run more steps
    print("Running 100 steps...")
    t0 = time.time()
    grid3, resource3, rng = run_v14_chunk(
        grid2, resource2, h_embed, kernel_ffts, config,
        coupling, coupling_row_sums, rng, n_steps=100, box_fft=box_fft
    )
    jax.block_until_ready(grid3)
    t1 = time.time()
    print(f"  100 steps: {t1-t0:.2f}s ({100/(t1-t0):.1f} steps/s)")
    print(f"  Grid mean: {float(grid3.mean()):.4f}")
    print(f"  Alive cells: {int((grid3.mean(axis=0) > 0.05).sum())}")
    print()

    # Check patterns
    from v11_patterns import detect_patterns_mc
    g_np = np.array(grid3)
    pats = detect_patterns_mc(g_np, threshold=0.15)
    print(f"  Patterns detected: {len(pats)}")
    for p in pats[:5]:
        print(f"    Pattern: {p.mass:.0f} cells, center=({p.center[0]:.0f},{p.center[1]:.0f})")
    print()

    # Test directed motion: run more steps and check if patterns move toward resources
    print("Running 500 more steps to test directed motion...")
    # First record pattern positions
    positions_before = [(p.center[0], p.center[1]) for p in pats[:5]]
    resource_at_patterns = [float(resource3[int(p.center[0])%64, int(p.center[1])%64])
                           for p in pats[:5]]

    t0 = time.time()
    grid4, resource4, rng = run_v14_chunk(
        grid3, resource3, h_embed, kernel_ffts, config,
        coupling, coupling_row_sums, rng, n_steps=500, box_fft=box_fft
    )
    jax.block_until_ready(grid4)
    t1 = time.time()

    g_np = np.array(grid4)
    pats2 = detect_patterns_mc(g_np, threshold=0.15)
    print(f"  {t1-t0:.2f}s, {len(pats2)} patterns")

    positions_after = [(p.center[0], p.center[1]) for p in pats2[:5]]
    resource_at_patterns2 = [float(resource4[int(p.center[0])%64, int(p.center[1])%64])
                             for p in pats2[:5]]

    if positions_before and positions_after:
        # Compare resource levels at pattern locations
        mean_res_before = np.mean(resource_at_patterns[:len(pats)])
        mean_res_after = np.mean(resource_at_patterns2[:len(pats2)])
        print(f"  Resource at pattern locations: {mean_res_before:.3f} -> {mean_res_after:.3f}")
        if mean_res_after > mean_res_before:
            print("  *** PATTERNS MOVING TOWARD RESOURCES ***")
        else:
            print("  (No clear resource-seeking detected)")

    # Test drought
    print("\nRunning 500 steps under drought...")
    t0 = time.time()
    grid5, resource5, rng = run_v14_chunk(
        grid4, resource4, h_embed, kernel_ffts, config,
        coupling, coupling_row_sums, rng, n_steps=500, drought=True, box_fft=box_fft
    )
    jax.block_until_ready(grid5)
    t1 = time.time()

    g_np = np.array(grid5)
    pats3 = detect_patterns_mc(g_np, threshold=0.15)
    print(f"  {t1-t0:.2f}s, {len(pats3)} patterns after drought")
    print(f"  Resource mean: {float(resource5.mean()):.4f}")

    print()
    print("V14 smoke test complete.")
