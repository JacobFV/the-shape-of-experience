"""V11.4 Visualization: labeled videos of HD Lenia dynamics.

Generates MP4 videos showing:
1. Multi-channel grid activity (PCA-projected to RGB)
2. Phase labels (BASELINE / DROUGHT / RECOVERY)
3. Live Phi and pattern count overlay
4. Resource field as background opacity

Usage:
    python v11_visualize.py hd         # C=8, N=128 quick demo
    python v11_visualize.py hd 64      # C=64, N=256 full scale
    python v11_visualize.py sweep      # C-sweep comparison video
    python v11_visualize.py perturb    # Single-channel drought/recovery
"""

import sys
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch


def channels_to_rgb(grid_np):
    """Project C channels to RGB via PCA of first 3 principal components.

    For C=3, this is just direct mapping. For C>3, we use the first 3
    PCs of the channel dimension to create a meaningful color image.
    """
    C, N, _ = grid_np.shape
    if C <= 3:
        rgb = np.zeros((N, N, 3))
        for c in range(min(C, 3)):
            rgb[:, :, c] = grid_np[c]
        return np.clip(rgb, 0, 1)

    # Reshape to (C, N*N), replace NaN/Inf, clip extremes
    flat = grid_np.reshape(C, -1).astype(np.float64)  # (C, N*N)
    flat = np.nan_to_num(flat, nan=0.0, posinf=10.0, neginf=-10.0)
    flat = np.clip(flat, -10.0, 10.0)

    # Use SVD directly on the data matrix (avoids forming CxC covariance)
    mean = flat.mean(axis=1, keepdims=True)
    centered = flat - mean
    # Thin SVD: U is (C, C), s is (C,), Vt is (C, N*N)
    try:
        U, s, Vt = np.linalg.svd(centered, full_matrices=False)
        # Top 3 PCs are first 3 rows of Vt (already sorted descending)
        projected = Vt[:3]  # (3, N*N)
    except np.linalg.LinAlgError:
        # Fallback: just use first 3 channels
        projected = flat[:3]

    # Normalize each channel to [0, 1]
    rgb = np.zeros((3, N, N))
    for i in range(3):
        ch = projected[i].reshape(N, N)
        lo, hi = np.nanmin(ch), np.nanmax(ch)
        if hi > lo:
            rgb[i] = (ch - lo) / (hi - lo)

    return np.transpose(rgb, (1, 2, 0))  # (N, N, 3)


def make_hd_video(C=8, N=128, output_path=None, fps=15, n_frames=200):
    """Generate a labeled video of HD Lenia dynamics under stress.

    Phases:
    - Warmup (not shown)
    - BASELINE: 60 frames, normal conditions
    - DROUGHT: 80 frames, resource regen -> 0
    - RECOVERY: 60 frames, normal conditions restored
    """
    from v11_substrate_hd import (
        generate_hd_config, generate_coupling_matrix,
        make_kernels_fft_hd, run_chunk_hd_wrapper, init_soup_hd
    )
    from v11_patterns import detect_patterns_mc
    from v11_affect_hd import spectral_channel_phi

    if output_path is None:
        os.makedirs('results', exist_ok=True)
        output_path = f'results/v11_4_hd_C{C}_N{N}.mp4'

    config = generate_hd_config(C=C, N=N, seed=42)
    coupling = jnp.array(generate_coupling_matrix(C, bandwidth=max(2.0, C/8), seed=42))
    kernel_ffts = make_kernels_fft_hd(config)

    rng = random.PRNGKey(42)
    rng, k = random.split(rng)
    grid, resource = init_soup_hd(N, C, k, jnp.array(config['channel_mus']))

    drought_config = {**config, 'resource_regen': 0.0001}

    # Warmup
    print(f"Generating video: C={C}, N={N}")
    print("  Warmup...", end=" ", flush=True)
    grid, resource, rng = run_chunk_hd_wrapper(
        grid, resource, kernel_ffts, coupling, rng, config, 100)
    grid.block_until_ready()
    grid, resource, rng = run_chunk_hd_wrapper(
        grid, resource, kernel_ffts, coupling, rng, config, 4900)
    grid.block_until_ready()
    print("done")

    # Collect frames
    steps_per_frame = 50
    baseline_frames = 60
    drought_frames = 80
    recovery_frames = 60
    total_frames = baseline_frames + drought_frames + recovery_frames

    frames = []
    phi_history = []
    pattern_count_history = []
    resource_history = []
    phase_labels = []

    print("  Collecting frames...")

    for frame_idx in range(total_frames):
        if frame_idx < baseline_frames:
            phase = "BASELINE"
            cfg = config
        elif frame_idx < baseline_frames + drought_frames:
            phase = "DROUGHT"
            cfg = drought_config
        else:
            phase = "RECOVERY"
            cfg = config

        grid, resource, rng = run_chunk_hd_wrapper(
            grid, resource, kernel_ffts, coupling, rng, cfg, steps_per_frame)

        if frame_idx % 5 == 0:
            grid.block_until_ready()

        grid_np = np.array(grid)
        resource_np = np.array(resource)

        # RGB projection
        rgb = channels_to_rgb(grid_np)

        # Pattern detection + Phi
        patterns = detect_patterns_mc(grid_np, threshold=0.15)
        phis = []
        for p in patterns[:30]:
            if p.size < 10:
                continue
            phi_s, _ = spectral_channel_phi(jnp.array(grid_np), coupling, p.cells)
            phis.append(phi_s)

        mean_phi = np.mean(phis) if phis else 0.0
        n_patterns = len(patterns)

        frames.append(rgb)
        phi_history.append(mean_phi)
        pattern_count_history.append(n_patterns)
        resource_history.append(float(resource_np.mean()))
        phase_labels.append(phase)

        if frame_idx % 20 == 0:
            step = frame_idx * steps_per_frame
            print(f"    Frame {frame_idx}/{total_frames} ({phase}): "
                  f"Phi={mean_phi:.4f}, n={n_patterns}, "
                  f"resource={resource_np.mean():.3f}")

    print(f"  Rendering {len(frames)} frames...")

    # Create the animation
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                              gridspec_kw={'width_ratios': [2, 1]})

    # Left: grid visualization
    ax_grid = axes[0]
    im = ax_grid.imshow(frames[0], interpolation='nearest')
    ax_grid.set_title(f'HD Lenia (C={C})', fontsize=14, fontweight='bold')
    ax_grid.axis('off')

    # Phase label
    phase_text = ax_grid.text(
        0.02, 0.98, '', transform=ax_grid.transAxes,
        fontsize=16, fontweight='bold', va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
        color='white')

    # Step counter
    step_text = ax_grid.text(
        0.98, 0.98, '', transform=ax_grid.transAxes,
        fontsize=10, va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5),
        color='white')

    # Right: time series
    ax_ts = axes[1]
    ax_ts.set_xlim(0, total_frames)
    ax_ts.set_xlabel('Frame', fontsize=11)

    # Shade phases
    ax_ts.axvspan(0, baseline_frames, alpha=0.1, color='green', label='Baseline')
    ax_ts.axvspan(baseline_frames, baseline_frames + drought_frames,
                  alpha=0.15, color='red', label='Drought')
    ax_ts.axvspan(baseline_frames + drought_frames, total_frames,
                  alpha=0.1, color='blue', label='Recovery')

    phi_line, = ax_ts.plot([], [], 'b-', linewidth=2, label='Spectral Phi')
    ax_ts.set_ylabel('Spectral Phi', color='b', fontsize=11)
    ax_ts.tick_params(axis='y', labelcolor='b')

    ax_ts2 = ax_ts.twinx()
    resource_line, = ax_ts2.plot([], [], 'r--', linewidth=1.5, label='Resource',
                                  alpha=0.7)
    ax_ts2.set_ylabel('Mean Resource', color='r', fontsize=11)
    ax_ts2.tick_params(axis='y', labelcolor='r')

    ax_ts.legend(loc='upper left', fontsize=8)
    ax_ts.set_title('Affect Dynamics', fontsize=12, fontweight='bold')

    plt.tight_layout()

    phase_colors = {'BASELINE': '#22cc22', 'DROUGHT': '#cc2222', 'RECOVERY': '#2266cc'}

    def update(frame_idx):
        im.set_data(frames[frame_idx])

        phase = phase_labels[frame_idx]
        phase_text.set_text(phase)
        phase_text.get_bbox_patch().set_facecolor(phase_colors.get(phase, 'black'))

        step = frame_idx * steps_per_frame
        step_text.set_text(f'Step {step}  |  {pattern_count_history[frame_idx]} patterns')

        # Update time series
        x = list(range(frame_idx + 1))
        phi_line.set_data(x, phi_history[:frame_idx + 1])
        resource_line.set_data(x, resource_history[:frame_idx + 1])

        ax_ts.set_ylim(0, max(phi_history[:frame_idx + 1]) * 1.3 + 0.01)
        ax_ts2.set_ylim(0, 1.0)

        return [im, phase_text, step_text, phi_line, resource_line]

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames),
        interval=1000 // fps, blit=False)

    writer = animation.FFMpegWriter(fps=fps, bitrate=3000)
    anim.save(output_path, writer=writer)
    plt.close(fig)

    print(f"  Saved: {output_path}")
    return output_path, phi_history, resource_history, phase_labels


def make_sweep_video(output_path=None, fps=12):
    """Generate a side-by-side comparison video across channel counts.

    Shows C=3, 8, 16, 32 grids simultaneously evolving through
    baseline -> drought -> recovery with Phi overlay.
    """
    from v11_substrate_hd import (
        generate_hd_config, generate_coupling_matrix,
        make_kernels_fft_hd, run_chunk_hd_wrapper, init_soup_hd
    )
    from v11_patterns import detect_patterns_mc
    from v11_affect_hd import spectral_channel_phi

    if output_path is None:
        os.makedirs('results', exist_ok=True)
        output_path = 'results/v11_4_c_sweep.mp4'

    N = 128  # smaller grid for all conditions
    C_values = [3, 8, 16, 32]
    steps_per_frame = 50
    baseline_frames = 40
    drought_frames = 60
    recovery_frames = 40
    total_frames = baseline_frames + drought_frames + recovery_frames

    # Initialize all conditions
    sims = {}
    for C in C_values:
        config = generate_hd_config(C=C, N=N, seed=42)
        coupling = jnp.array(generate_coupling_matrix(C, bandwidth=max(2.0, C/8), seed=42))
        kernel_ffts = make_kernels_fft_hd(config)

        rng = random.PRNGKey(42)
        rng, k = random.split(rng)
        grid, resource = init_soup_hd(N, C, k, jnp.array(config['channel_mus']))

        drought_config = {**config, 'resource_regen': 0.0001}

        # Warmup
        print(f"  Warming up C={C}...", end=" ", flush=True)
        grid, resource, rng = run_chunk_hd_wrapper(
            grid, resource, kernel_ffts, coupling, rng, config, 100)
        grid.block_until_ready()
        grid, resource, rng = run_chunk_hd_wrapper(
            grid, resource, kernel_ffts, coupling, rng, config, 4900)
        grid.block_until_ready()
        print("done")

        sims[C] = {
            'grid': grid, 'resource': resource, 'rng': rng,
            'config': config, 'drought_config': drought_config,
            'coupling': coupling, 'kernel_ffts': kernel_ffts,
            'frames': [], 'phi_history': [],
        }

    print(f"  Collecting {total_frames} frames for {len(C_values)} conditions...")

    for frame_idx in range(total_frames):
        if frame_idx < baseline_frames:
            phase = "BASELINE"
        elif frame_idx < baseline_frames + drought_frames:
            phase = "DROUGHT"
        else:
            phase = "RECOVERY"

        for C in C_values:
            s = sims[C]
            cfg = s['drought_config'] if phase == "DROUGHT" else s['config']
            s['grid'], s['resource'], s['rng'] = run_chunk_hd_wrapper(
                s['grid'], s['resource'], s['kernel_ffts'], s['coupling'],
                s['rng'], cfg, steps_per_frame)

            if frame_idx % 5 == 0:
                s['grid'].block_until_ready()

            grid_np = np.array(s['grid'])
            rgb = channels_to_rgb(grid_np)
            s['frames'].append(rgb)

            patterns = detect_patterns_mc(grid_np, threshold=0.15)
            phis = []
            for p in patterns[:20]:
                if p.size < 10:
                    continue
                phi_s, _ = spectral_channel_phi(
                    jnp.array(grid_np), s['coupling'], p.cells)
                phis.append(phi_s)
            s['phi_history'].append(np.mean(phis) if phis else 0.0)

        if frame_idx % 20 == 0:
            print(f"    Frame {frame_idx}/{total_frames} ({phase})")

    # Render
    print(f"  Rendering...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    phase_colors = {'BASELINE': '#22cc22', 'DROUGHT': '#cc2222', 'RECOVERY': '#2266cc'}

    ims = []
    texts = []
    phi_texts = []
    for idx, C in enumerate(C_values):
        ax = axes[idx]
        im = ax.imshow(sims[C]['frames'][0], interpolation='nearest')
        ax.set_title(f'C = {C}', fontsize=16, fontweight='bold')
        ax.axis('off')

        txt = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                      fontsize=14, fontweight='bold', va='top', ha='left',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
                      color='white')
        phi_txt = ax.text(0.98, 0.02, '', transform=ax.transAxes,
                          fontsize=12, va='bottom', ha='right',
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6),
                          color='white')

        ims.append(im)
        texts.append(txt)
        phi_texts.append(phi_txt)

    plt.suptitle('V11.4 Channel Dimensionality Sweep: Stress Response',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    def update(frame_idx):
        if frame_idx < baseline_frames:
            phase = "BASELINE"
        elif frame_idx < baseline_frames + drought_frames:
            phase = "DROUGHT"
        else:
            phase = "RECOVERY"

        for idx, C in enumerate(C_values):
            ims[idx].set_data(sims[C]['frames'][frame_idx])
            texts[idx].set_text(phase)
            texts[idx].get_bbox_patch().set_facecolor(phase_colors.get(phase, 'black'))

            phi = sims[C]['phi_history'][frame_idx]
            phi_texts[idx].set_text(f'Phi={phi:.4f}')

        return ims + texts + phi_texts

    anim = animation.FuncAnimation(
        fig, update, frames=total_frames,
        interval=1000 // fps, blit=False)

    writer = animation.FFMpegWriter(fps=fps, bitrate=4000)
    anim.save(output_path, writer=writer)
    plt.close(fig)

    print(f"  Saved: {output_path}")

    # Print Phi summary
    print("\n  Phi Summary:")
    for C in C_values:
        ph = sims[C]['phi_history']
        base_phi = np.mean(ph[:baseline_frames])
        drought_phi = np.mean(ph[baseline_frames:baseline_frames+drought_frames])
        recov_phi = np.mean(ph[baseline_frames+drought_frames:])
        delta = (drought_phi - base_phi) / (base_phi + 1e-10)
        print(f"    C={C:>2d}: base={base_phi:.4f} -> drought={drought_phi:.4f} "
              f"({delta:+.1%}) -> recovery={recov_phi:.4f}")

    return output_path


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'hd'

    if mode == 'hd':
        C = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        N = 128 if C <= 16 else 256
        make_hd_video(C=C, N=N)
    elif mode == 'sweep':
        make_sweep_video()
    else:
        print(f"Unknown mode: {mode}. Use: hd [C], sweep")
