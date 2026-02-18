"""V16: Plastic Lenia — Within-Lifetime Learning via Local Hebbian Rules.

V15 showed temporal memory is selectable and improves stress response.
But memory only stores — it doesn't learn. Patterns remember where resources
were but can't form new associations between internal state and outcomes.

V16 adds LOCAL PLASTICITY: coupling weights between channels can change
within a pattern's lifetime based on co-activation (Hebbian) and resource
feedback (reward-modulated).

KEY INSIGHT: V12 identified the gap — "attention is necessary but not
sufficient; missing ingredient is individual-level plasticity." V16
directly addresses this.

Changes from V15:
1. LOCAL COUPLING WEIGHTS: Each spatial location has its own C×C coupling
   matrix (initialized from global coupling, then modified by local learning).
   This replaces the single global coupling matrix with a spatially-varying
   learned coupling field.

2. HEBBIAN LEARNING RULE: Local coupling updates based on pre/post channel
   co-activation, modulated by resource level (reward signal):
     ΔW_ij(x) = η_learn · pre_i(x) · post_j(x) · reward(x) - λ_decay · W_ij(x)
   η_learn and λ_decay are evolvable.

3. COUPLING FIELD PERSISTENCE: The coupling field persists across steps
   (stored as additional state), creating genuine within-lifetime learning.
   Patterns that discover useful associations keep them.

Channel layout (C=16):
  - Channels 0-11: Regular growth-function channels
  - Channels 12-13: Memory channels (EMA dynamics, from V15)
  - Channels 14-15: Motor channels (chemotaxis, from V14)

State:
  - grid: (C, N, N) — channel activations
  - resource: (N, N) — resource field
  - coupling_field: (C, C, N, N) — local coupling weights (NEW)

Why this matters for the formal experiment program:
  - World models (Exp 2): Learning allows patterns to form predictive associations
  - Abstraction (Exp 3): Local coupling patterns = learned representations
  - Counterfactual detachment (Exp 5): Different coupling = different response
  - Imagination (Exp 6): Internal coupling dynamics can run "offline"
"""

import jax
import jax.numpy as jnp
from jax import random, jit, lax
import numpy as np
from functools import partial

from v15_substrate import (
    generate_v15_config, init_v15,
    _update_memory_channels,
    generate_resource_patches, compute_patch_regen_mask,
)
from v13_substrate import (
    _compute_similarity_field, make_box_kernel_fft, init_embedding,
)
from v14_substrate import (
    _compute_velocity_field, _advect_grid,
)


# ============================================================================
# Configuration
# ============================================================================

def generate_v16_config(C=16, N=128, seed=42, similarity_radius=5,
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
                        # V16 plasticity params
                        learning_rate=0.001,
                        decay_rate_coupling=0.0001,
                        coupling_clamp=2.0):
    """Generate V16 config (extends V15 with local plasticity).

    New parameters:
        learning_rate: Hebbian learning rate (evolvable)
        decay_rate_coupling: Weight decay preventing runaway (evolvable)
        coupling_clamp: Max absolute coupling weight
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

    config['learning_rate'] = learning_rate
    config['decay_rate_coupling'] = decay_rate_coupling
    config['coupling_clamp'] = coupling_clamp

    return config


# ============================================================================
# Local Coupling Field
# ============================================================================

def init_coupling_field(coupling, N):
    """Initialize local coupling field from global coupling matrix.

    Args:
        coupling: (C, C) global coupling matrix
        N: grid size

    Returns: (C, C, N, N) — each spatial location starts with global coupling
    """
    return jnp.broadcast_to(
        coupling[:, :, None, None], (coupling.shape[0], coupling.shape[1], N, N)
    ).copy()


def _hebbian_update(coupling_field, grid, resource,
                    learning_rate, decay_rate, coupling_clamp):
    """Update local coupling weights via reward-modulated Hebbian rule.

    ΔW_ij(x) = η · act_i(x) · act_j(x) · reward(x) - λ · (W_ij(x) - W_ij_init)

    The reward signal is the local resource level, normalized.
    Decay pulls weights back toward their initial values (homeostasis).

    Args:
        coupling_field: (C, C, N, N) local coupling weights
        grid: (C, N, N) current activations
        resource: (N, N) resource field
        learning_rate: float
        decay_rate: float
        coupling_clamp: float — max absolute weight

    Returns: updated coupling_field
    """
    C = grid.shape[0]

    # Reward signal: normalized resource level
    reward = resource / (resource + 0.1)  # (N, N), ∈ [0, ~1)

    # Channel activations (normalized)
    act = grid / (jnp.max(grid, axis=0, keepdims=True) + 0.01)  # (C, N, N)

    # Hebbian outer product: act_i(x) * act_j(x) for all i,j at each location
    # Shape: (C, 1, N, N) * (1, C, N, N) = (C, C, N, N)
    hebbian = act[:, None, :, :] * act[None, :, :, :]

    # Reward-modulated Hebbian update
    delta = learning_rate * hebbian * reward[None, None, :, :]

    # Weight decay (toward zero — keeps weights bounded)
    decay = decay_rate * coupling_field

    # Update
    coupling_field = coupling_field + delta - decay

    # Clamp
    coupling_field = jnp.clip(coupling_field, -coupling_clamp, coupling_clamp)

    # Ensure self-coupling stays positive
    diag_mask = jnp.eye(C)[:, :, None, None]
    coupling_field = jnp.where(diag_mask > 0,
                                jnp.maximum(coupling_field, 0.5),
                                coupling_field)

    return coupling_field


# ============================================================================
# Core Physics
# ============================================================================

@partial(jit, static_argnums=(5, 24, 27))
def run_chunk_v16(grid, resource, kernel_ffts, coupling_field, rng, n_steps,
                  dt, channel_mus, channel_sigmas, coupling_row_sums_unused,
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
                  regen_mask,
                  # V16 plasticity
                  learning_rate, decay_rate_coupling, coupling_clamp):
    """Run n_steps of V16 Plastic Lenia.

    Key difference from V15: coupling is now (C, C, N, N) — spatially varying,
    and updated each step via Hebbian learning.
    """
    C = grid.shape[0]
    N = grid.shape[1]
    motor_start_idx = C - motor_channels
    memory_start = C - motor_channels - memory_channels

    def body(carry, _):
        g, r, cf, k = carry
        k, k_noise = random.split(k)

        # ---- STEP 1: Spatial potentials via FFT ----
        g_fft = jnp.fft.rfft2(g)
        potentials = jnp.fft.irfft2(g_fft * kernel_ffts, s=(N, N))

        # ---- STEP 1b: Content-similarity modulation ----
        sim_field = _compute_similarity_field(
            g, h_embed, tau, gate_beta, box_fft)
        modulation = 1.0 + alpha * sim_field
        potentials = potentials * modulation[None, :, :]

        # ---- STEP 2: LOCAL cross-channel coupling (V16: spatially varying) ----
        # cf is (C, C, N, N). For each location, multiply C-dim potential by CxC matrix.
        # This is: out_c(x) = sum_d cf[c,d,x] * potentials[d,x]
        cross = jnp.einsum('cdnm,dnm->cnm', cf, potentials)
        # Normalize by row sums (computed locally)
        cf_row_sums = jnp.sum(jnp.abs(cf), axis=1)  # (C, N, N)
        cross = cross / (cf_row_sums + 0.01)
        cross_gate = jax.nn.sigmoid(5.0 * (cross - 0.3))
        potentials_gated = potentials * (1.0 - 0.5 * cross_gate) + cross * 0.5 * cross_gate

        # ---- STEP 3: Growth function ----
        mus = channel_mus[:, None, None]
        sigs = channel_sigmas[:, None, None]
        growth = 2.0 * jnp.exp(-((potentials_gated - mus)**2) / (2 * sigs**2)) - 1.0

        # ---- STEP 4: Resource modulation ----
        resource_factor = r / (r + resource_half_sat)
        growth_modulated = growth * resource_factor[None, :, :]

        # ---- STEP 5: State update ----
        g_new = g + dt * growth_modulated

        # ---- STEP 5a: Memory channels (V15) ----
        g_new = _update_memory_channels(
            g_new, r, memory_start, memory_channels, memory_lambdas_arr)

        # ---- STEP 5b: Chemotactic advection (V14) ----
        vx, vy = _compute_velocity_field(
            g_new, r, chemotaxis_strength, motor_start_idx,
            motor_sensitivity, motor_threshold, max_speed)
        g_new = _advect_grid(g_new, vx, vy)

        # ---- STEP 5c: HEBBIAN LEARNING (V16) ----
        cf_new = _hebbian_update(
            cf, g_new, r,
            learning_rate, decay_rate_coupling, coupling_clamp)

        # ---- STEP 6: Noise ----
        noise = noise_amp * random.normal(k_noise, g.shape)
        g_new = g_new + noise

        # ---- STEP 7: Clamp ----
        g_new = jnp.clip(g_new, 0.0, 1.0)

        # ---- STEP 8: Resource dynamics ----
        consumption = resource_consume * jnp.mean(g_new, axis=0)
        maintenance = maintenance_rate * (jnp.mean(g_new, axis=0) > 0.05).astype(jnp.float32)
        local_regen = resource_regen * regen_mask
        r_new = r + local_regen * (1.0 - r / resource_max) - consumption - maintenance
        r_new = jnp.clip(r_new, 0.0, resource_max)

        # ---- STEP 9: Decay from depletion ----
        depletion_mask = (r_new < 0.05).astype(jnp.float32)
        g_new = g_new * (1.0 - decay_rate * depletion_mask[None, :, :])

        return (g_new, r_new, cf_new, k), None

    (grid_final, resource_final, cf_final, rng_final), _ = lax.scan(
        body, (grid, resource, coupling_field, rng), None, length=n_steps
    )

    return grid_final, resource_final, cf_final, rng_final


# ============================================================================
# Initialization
# ============================================================================

def init_v16(config, seed=42):
    """Initialize V16 substrate. Returns everything from V15 plus coupling_field."""
    grid, resource, h_embed, kernel_ffts, coupling, coupling_row_sums, box_fft = \
        init_v15(config, seed=seed)

    N = config['grid_size']
    coupling_field = init_coupling_field(coupling, N)

    return grid, resource, h_embed, kernel_ffts, coupling, coupling_row_sums, box_fft, coupling_field


# ============================================================================
# Convenience runner
# ============================================================================

def run_v16_chunk(grid, resource, h_embed, kernel_ffts, config,
                  coupling_field, rng, n_steps=50,
                  drought=False, box_fft=None, regen_mask=None):
    """Run a chunk of V16 steps with optional drought."""
    regen = 0.001 if drought else config['resource_regen']

    if box_fft is None:
        box_fft = make_box_kernel_fft(
            config['similarity_radius'], config['grid_size'])

    N = config['grid_size']
    if regen_mask is None:
        regen_mask = jnp.ones((N, N))

    memory_lambdas_arr = jnp.array(config['memory_lambdas'])

    # Dummy coupling_row_sums (not used in V16 — computed locally)
    C = config['n_channels']
    dummy_crs = jnp.ones(C)

    return run_chunk_v16(
        grid, resource, kernel_ffts, coupling_field, rng, n_steps,
        config['dt'],
        jnp.array(config['channel_mus']),
        jnp.array(config['channel_sigmas']),
        dummy_crs,
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
        # V14
        config['chemotaxis_strength'],
        config['motor_sensitivity'],
        int(config['motor_channels']),  # static
        config['motor_threshold'],
        config['max_speed'],
        # V15
        int(config['memory_channels']),  # static
        memory_lambdas_arr,
        regen_mask,
        # V16
        config['learning_rate'],
        config['decay_rate_coupling'],
        config['coupling_clamp'],
    )


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == '__main__':
    import time

    print("V16 Plastic Lenia — Smoke Test")
    print("=" * 60)

    config = generate_v16_config(C=8, N=64, similarity_radius=5,
                                  chemotaxis_strength=0.5,
                                  memory_channels=2,
                                  motor_channels=2,
                                  learning_rate=0.001,
                                  decay_rate_coupling=0.0001)
    grid, resource, h_embed, kernel_ffts, coupling, coupling_row_sums, box_fft, coupling_field = \
        init_v16(config, seed=42)

    rng = random.PRNGKey(0)

    patches = generate_resource_patches(config['grid_size'],
                                        config['n_resource_patches'],
                                        seed=42)
    regen_mask = compute_patch_regen_mask(
        config['grid_size'], patches, step=0,
        shift_period=config['patch_shift_period'])

    print(f"Grid: {grid.shape}, Coupling field: {coupling_field.shape}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Decay rate: {config['decay_rate_coupling']}")
    print(f"Initial coupling field std: {float(coupling_field.std()):.4f}")
    print()

    # JIT compile
    print("JIT compiling...", end=" ", flush=True)
    t0 = time.time()
    grid2, resource2, cf2, rng = run_v16_chunk(
        grid, resource, h_embed, kernel_ffts, config,
        coupling_field, rng, n_steps=50,
        box_fft=box_fft, regen_mask=regen_mask
    )
    jax.block_until_ready(grid2)
    t1 = time.time()
    print(f"{t1-t0:.1f}s")
    print(f"  Grid mean: {float(grid2.mean()):.4f}")
    print(f"  Coupling field std: {float(cf2.std()):.6f}")
    cf_change = float(jnp.abs(cf2 - coupling_field).mean())
    print(f"  Mean coupling change: {cf_change:.6f}")

    # Run 200 steps
    print("\nRunning 200 steps (4 chunks)...", end=" ", flush=True)
    t0 = time.time()
    for _ in range(4):
        grid2, resource2, cf2, rng = run_v16_chunk(
            grid2, resource2, h_embed, kernel_ffts, config,
            cf2, rng, n_steps=50,
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
    cf_change_total = float(jnp.abs(cf2 - coupling_field).mean())
    print(f"  Total coupling change: {cf_change_total:.6f}")
    print(f"  Coupling field range: [{float(cf2.min()):.3f}, {float(cf2.max()):.3f}]")

    # Check that coupling actually varies spatially
    cf_spatial_var = float(cf2.var(axis=(2, 3)).mean())
    print(f"  Coupling spatial variance: {cf_spatial_var:.6f}")
    print(f"  (Should be > 0 if learning is happening)")

    print("\nV16 smoke test complete.")
