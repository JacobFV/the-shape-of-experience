"""V17: Signaling Lenia — Quorum-Sensing Communication via Diffusible Fields.

V16 showed that free-form Hebbian plasticity hurts: C×C×N×N coupling fields
add noise faster than selection filters. The lesson: constrain degrees of freedom.

V17 takes a fundamentally different approach to inter-pattern coordination.
Instead of modifying internal coupling weights, patterns can EMIT and SENSE
diffusible signal molecules that spread through the environment. This is
analogous to bacterial quorum sensing: a low-dimensional, spatially-mediated
coordination mechanism.

Architecture: V17 wraps V15's existing physics and adds a signal layer that
modulates the coupling matrix between chunks. Signal dynamics:
1. Signals diffuse and decay each inter-chunk step
2. Pattern cells emit signals based on channel activity
3. Mean signal concentration modulates coupling matrix via thresholded shifts

Changes from V15 (V17 branches from V15, NOT V16):
1. SIGNAL CHANNELS: 2 diffusible signal fields (N, N) that pattern cells emit
   into and sense from. Signals diffuse and decay between chunks.

2. EMISSION: Signal emission proportional to pattern activity in designated
   channels, gated by evolvable emission strength.

3. COUPLING MODULATION: When signal concentration exceeds sensitivity threshold,
   coupling matrix shifts by a small (evolvable) perturbation. This is a
   THRESHOLDED structural change, not continuous Hebbian drift.

Why this is different from V16:
- Signal fields are (2, N, N) — NOT (C, C, N, N). Two orders of magnitude
  fewer degrees of freedom.
- Coupling changes are GLOBAL (same shift everywhere), not per-location.
- Only 4 new evolvable scalars (emission_strength × 2, signal_sensitivity × 2),
  plus (2, C, C) coupling shift matrices.

Channel layout (C=16):
  - Channels 0-11: Regular growth-function channels
  - Channels 12-13: Memory channels (EMA dynamics, from V15)
  - Channels 14-15: Motor channels (chemotaxis, from V14)

State:
  - grid: (C, N, N) — channel activations
  - resource: (N, N) — resource field
  - signals: (2, N, N) — diffusible signal concentrations (NEW)
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

from v15_substrate import (
    generate_v15_config, init_v15, run_v15_chunk,
    generate_resource_patches, compute_patch_regen_mask,
)
from v13_substrate import make_box_kernel_fft


# ============================================================================
# Configuration
# ============================================================================

def generate_v17_config(C=16, N=128, seed=42, similarity_radius=5,
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
                        n_signals=2,
                        signal_diffusion=0.15,
                        signal_decay=0.05,
                        emission_strength=None,
                        emission_threshold=0.3,
                        signal_sensitivity=None,
                        coupling_shift_scale=0.1):
    """Generate V17 config (extends V15 with signaling channels).

    New parameters:
        n_signals: number of diffusible signal channels (default 2)
        signal_diffusion: diffusion coefficient per inter-chunk step
        signal_decay: exponential decay rate per inter-chunk step
        emission_strength: per-signal emission rate (default [0.05, 0.05])
        emission_threshold: activity threshold for emission
        signal_sensitivity: detection threshold per signal (default [0.3, 0.3])
        coupling_shift_scale: magnitude of coupling change when signal detected
    """
    v15_cfg = generate_v15_config(
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

    rng = np.random.RandomState(seed + 17)

    if emission_strength is None:
        emission_strength = [0.05] * n_signals
    if signal_sensitivity is None:
        signal_sensitivity = [0.3] * n_signals

    # Generate random coupling shifts: small perturbation per signal
    coupling_shifts = []
    for s in range(n_signals):
        shift = rng.randn(C, C).astype(np.float32) * coupling_shift_scale
        shift = (shift + shift.T) / 2  # Symmetric
        coupling_shifts.append(shift)

    v17_cfg = {
        **v15_cfg,
        'n_signals': n_signals,
        'signal_diffusion': signal_diffusion,
        'signal_decay': signal_decay,
        'emission_strength': np.array(emission_strength, dtype=np.float32),
        'emission_threshold': float(emission_threshold),
        'signal_sensitivity': np.array(signal_sensitivity, dtype=np.float32),
        'coupling_shift_scale': coupling_shift_scale,
        'coupling_shifts': np.stack(coupling_shifts),  # (n_signals, C, C)
    }

    return v17_cfg


# ============================================================================
# Initialization
# ============================================================================

def init_v17(config, seed=42):
    """Initialize V17 state.

    Returns:
        grid, resource, signals, h_embed, kernel_ffts, coupling,
        coupling_row_sums, box_fft
    """
    grid, resource, h_embed, kernel_ffts, coupling, coupling_row_sums, box_fft = \
        init_v15(config, seed=seed)
    N = config['grid_size']
    n_signals = config['n_signals']
    signals = np.zeros((n_signals, N, N), dtype=np.float32)
    return grid, resource, signals, h_embed, kernel_ffts, coupling, coupling_row_sums, box_fft


# ============================================================================
# Signal Dynamics (between V15 chunks)
# ============================================================================

def update_signals(signals, grid, config):
    """Update signal fields based on current grid state.

    Called between V15 chunks. Performs:
    1. Diffusion (Laplacian)
    2. Emission from pattern activity
    3. Decay

    signals: (n_signals, N, N) — numpy or jax array
    grid: (C, N, N) — current grid state
    config: V17 config dict

    Returns: updated signals (n_signals, N, N)
    """
    signals = jnp.array(signals)
    grid = jnp.array(grid)
    n_signals = config['n_signals']
    C = config['n_channels']

    # 1. Diffusion (5-point stencil, periodic BC)
    diffusion = config['signal_diffusion']
    up = jnp.roll(signals, -1, axis=1)
    down = jnp.roll(signals, 1, axis=1)
    left = jnp.roll(signals, -1, axis=2)
    right = jnp.roll(signals, 1, axis=2)
    laplacian = up + down + left + right - 4 * signals
    signals = signals + diffusion * laplacian

    # 2. Emission from pattern activity
    n_regular = config.get('n_regular', C - config.get('motor_channels', 2)
                           - config.get('memory_channels', 2))
    channels_per_signal = max(1, n_regular // n_signals)

    emission_strength = jnp.array(config['emission_strength'])
    threshold = config['emission_threshold']

    for s in range(n_signals):
        start_ch = s * channels_per_signal
        end_ch = min((s + 1) * channels_per_signal, n_regular)
        source = jnp.mean(grid[start_ch:end_ch], axis=0)  # (N, N)
        emission = emission_strength[s] * jax.nn.sigmoid(
            10.0 * (source - threshold)
        )
        signals = signals.at[s].add(emission)

    # 3. Decay
    signals = signals * (1.0 - config['signal_decay'])

    # Clamp
    signals = jnp.clip(signals, 0.0, 5.0)

    return signals


def compute_effective_coupling(coupling, signals, config):
    """Compute signal-modulated coupling matrix.

    When mean signal concentration exceeds sensitivity threshold,
    coupling is shifted by the corresponding coupling_shift.

    coupling: (C, C) base coupling
    signals: (n_signals, N, N) current signals
    config: V17 config

    Returns: (C, C) effective coupling
    """
    coupling = jnp.array(coupling)
    signals = jnp.array(signals)
    sensitivity = jnp.array(config['signal_sensitivity'])
    shifts = jnp.array(config['coupling_shifts'])
    n_signals = config['n_signals']

    effective = coupling.copy()
    for s in range(n_signals):
        mean_conc = jnp.mean(signals[s])
        activation = jax.nn.sigmoid(10.0 * (mean_conc - sensitivity[s]))
        effective = effective + activation * shifts[s]

    return effective


# ============================================================================
# Main runner (wraps V15)
# ============================================================================

def run_v17_chunk(grid, resource, signals, h_embed, kernel_ffts,
                  config, coupling, coupling_row_sums, rng,
                  n_steps=50, drought=False, box_fft=None,
                  regen_mask=None):
    """Run a chunk of V17 steps.

    Signal dynamics happen at the chunk level:
    1. Update signals (diffuse, emit, decay)
    2. Compute effective coupling (signal-modulated)
    3. Run V15 chunk with effective coupling
    4. Update signals again

    Returns: (grid, resource, signals, rng)
    """
    # Update signals before chunk
    signals = update_signals(signals, grid, config)

    # Compute signal-modulated coupling
    effective_coupling = compute_effective_coupling(coupling, signals, config)
    effective_crs = jnp.sum(jnp.abs(effective_coupling), axis=1)

    # Run V15 physics with modulated coupling
    grid, resource, rng = run_v15_chunk(
        grid, resource, h_embed, kernel_ffts, config,
        effective_coupling, effective_crs, rng,
        n_steps=n_steps, drought=drought,
        box_fft=box_fft, regen_mask=regen_mask,
    )

    # Update signals after chunk
    signals = update_signals(signals, grid, config)

    return grid, resource, signals, rng


# ============================================================================
# Signal metrics
# ============================================================================

def compute_signal_metrics(signals, config):
    """Compute metrics about signal state for logging."""
    signals = np.array(signals)
    metrics = {}
    for s in range(config['n_signals']):
        sig = signals[s]
        metrics[f'signal_{s}_mean'] = float(np.mean(sig))
        metrics[f'signal_{s}_max'] = float(np.max(sig))
        metrics[f'signal_{s}_std'] = float(np.std(sig))
    return metrics


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == '__main__':
    import time

    print("V17 Signaling Lenia — Smoke Test")
    print("=" * 60)

    C, N = 8, 64
    config = generate_v17_config(C=C, N=N, seed=42, motor_channels=2,
                                  memory_channels=2)
    grid, resource, signals, h_embed, kernel_ffts, coupling, crs, box_fft = \
        init_v17(config, seed=42)

    rng = random.PRNGKey(42)

    # Resource patches
    patches = generate_resource_patches(config['grid_size'],
                                        config['n_resource_patches'],
                                        seed=42)
    regen_mask = compute_patch_regen_mask(
        config['grid_size'], patches, step=0,
        shift_period=config['patch_shift_period'])

    print(f"Grid: {grid.shape}, Resource: {resource.shape}, Signals: {signals.shape}")
    print(f"Signal channels: {config['n_signals']}")
    print(f"Signal diffusion: {config['signal_diffusion']}")
    print(f"Signal decay: {config['signal_decay']}")
    print(f"Emission strength: {config['emission_strength']}")
    print(f"Coupling shifts shape: {config['coupling_shifts'].shape}")

    # Run chunks
    print("\nRunning 10 chunks of 50 steps each...")
    t0 = time.time()
    for i in range(10):
        grid, resource, signals, rng = run_v17_chunk(
            grid, resource, signals, h_embed, kernel_ffts,
            config, coupling, crs, rng,
            n_steps=50, box_fft=box_fft, regen_mask=regen_mask,
        )
        metrics = compute_signal_metrics(signals, config)
        grid_alive = float(jnp.mean(jnp.array(grid) > 0.01))
        eff_coupling = compute_effective_coupling(coupling, signals, config)
        coupling_diff = float(jnp.mean(jnp.abs(jnp.array(eff_coupling) - jnp.array(coupling))))
        print(f"  Chunk {i}: alive={grid_alive:.3f}, "
              f"sig0={metrics['signal_0_mean']:.4f}, "
              f"sig1={metrics['signal_1_mean']:.4f}, "
              f"coupling_shift={coupling_diff:.5f}")
    dt = time.time() - t0

    from v11_patterns import detect_patterns_mc
    patterns = detect_patterns_mc(np.array(grid))
    print(f"\nPatterns detected: {len(patterns)}")
    print(f"Time: {dt:.1f}s ({500/dt:.0f} steps/s)")

    # Check signal modulation effect
    print(f"\nSignal modulation analysis:")
    for s in range(config['n_signals']):
        sig = np.array(signals[s])
        thresh = config['signal_sensitivity'][s]
        activation = 1.0 / (1.0 + np.exp(-10.0 * (np.mean(sig) - thresh)))
        print(f"  Signal {s}: mean={np.mean(sig):.4f}, thresh={thresh:.2f}, "
              f"activation={activation:.4f}")

    print("\nSmoke test passed!")
