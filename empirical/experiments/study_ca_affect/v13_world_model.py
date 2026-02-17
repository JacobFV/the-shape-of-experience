"""Experiment 2: Emergent World Model — Measurement Module.

Measures whether patterns in V13 content-based coupling Lenia develop
internal world models: predictive information about the environment
beyond current boundary observations.

Method — Prediction Gap:
    W(τ) = MSE[f_env(s_∂B → s_env(t+τ))] - MSE[f_full(s_B, s_∂B → s_env(t+τ))]

If W(τ) > 0, the pattern's internal state s_B carries information about
the future environment that is NOT available from boundary observations
alone. This is the operational definition of an emergent world model.

Derived measures:
    H_wm (horizon): max τ where W(τ) > 0 significantly
    C_wm (capacity): ∫W(τ)dτ — total predictive information across horizons

Runs on TOP of existing V13 results — no substrate changes.
"""

import warnings
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy import ndimage


# ============================================================================
# Feature Extraction
# ============================================================================

@dataclass
class PatternFeatures:
    """Features extracted from a single pattern at one timestep."""
    pattern_id: int
    step: int
    s_B: np.ndarray       # internal state vector
    s_dB: np.ndarray      # boundary observation vector
    s_env: np.ndarray     # environment target vector
    center: np.ndarray    # (2,) centroid
    size: int


def extract_internal_state(grid_mc: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Extract internal state s_B from pattern cells.

    Features (4C + 4 = 68 for C=16):
        - Per-channel mean (C)
        - Per-channel std (C)
        - Per-channel max (C)
        - Per-channel active fraction (C)
        - Spatial moments: centroid_r, centroid_c, spread_r, spread_c (4)
    """
    C = grid_mc.shape[0]
    vals = grid_mc[:, cells[:, 0], cells[:, 1]]  # (C, n_cells)

    ch_mean = vals.mean(axis=1)      # (C,)
    ch_std = vals.std(axis=1)        # (C,)
    ch_max = vals.max(axis=1)        # (C,)
    ch_active = (vals > 0.05).mean(axis=1)  # (C,)

    # Spatial moments from aggregate
    agg = vals.mean(axis=0)  # (n_cells,)
    weights = agg / (agg.sum() + 1e-10)
    centroid = np.average(cells.astype(float), axis=0, weights=weights)
    spread = np.sqrt(np.average((cells.astype(float) - centroid) ** 2,
                                axis=0, weights=weights) + 1e-10)

    return np.concatenate([ch_mean, ch_std, ch_max, ch_active,
                           centroid, spread]).astype(np.float32)


def extract_boundary_obs(grid_mc: np.ndarray, cells: np.ndarray,
                         N: int, ring_width: int = 3) -> np.ndarray:
    """Extract boundary observation s_∂B from ring around pattern.

    Features (2C + 4 = 36 for C=16):
        - Per-channel mean in ring (C)
        - Per-channel std in ring (C)
        - Ring size, ring mean intensity, resource gradient (approx), density (4)
    """
    C = grid_mc.shape[0]

    # Build pattern mask
    mask = np.zeros((N, N), dtype=bool)
    mask[cells[:, 0], cells[:, 1]] = True

    # Dilate to get ring
    struct = ndimage.generate_binary_structure(2, 2)
    dilated = ndimage.binary_dilation(mask, struct, iterations=ring_width)
    ring = dilated & ~mask

    ring_coords = np.argwhere(ring)
    if len(ring_coords) < 2:
        return np.zeros(2 * C + 4, dtype=np.float32)

    ring_vals = grid_mc[:, ring_coords[:, 0], ring_coords[:, 1]]  # (C, n_ring)

    ch_mean = ring_vals.mean(axis=1)   # (C,)
    ch_std = ring_vals.std(axis=1)     # (C,)

    ring_size = float(len(ring_coords))
    ring_intensity = float(ring_vals.mean())
    # Density = pattern cells / (pattern + ring cells)
    density = float(len(cells)) / (len(cells) + ring_size + 1e-10)
    # Gradient: mean intensity at inner vs outer ring edge (simple proxy)
    inner_dilate = ndimage.binary_dilation(mask, struct, iterations=1) & ~mask
    outer_ring = ring & ~inner_dilate
    if outer_ring.sum() > 0 and inner_dilate.sum() > 0:
        inner_coords = np.argwhere(inner_dilate)
        outer_coords = np.argwhere(outer_ring)
        inner_val = grid_mc[:, inner_coords[:, 0], inner_coords[:, 1]].mean()
        outer_val = grid_mc[:, outer_coords[:, 0], outer_coords[:, 1]].mean()
        gradient = float(inner_val - outer_val)
    else:
        gradient = 0.0

    return np.concatenate([ch_mean, ch_std,
                           [ring_size, ring_intensity, gradient, density]]
                          ).astype(np.float32)


def extract_environment(grid_mc: np.ndarray, center: np.ndarray,
                        N: int, inner_r: int = 15, outer_r: int = 25
                        ) -> np.ndarray:
    """Extract environment target s_env from annular region around pattern.

    Features (C + 2 = 18 for C=16):
        - Per-channel mean in annulus (C)
        - Annulus total intensity (1)
        - Annulus active fraction (1)
    """
    C = grid_mc.shape[0]
    cr, cc = int(round(center[0])), int(round(center[1]))

    # Build annular mask with periodic boundaries
    rr, cc_grid = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    dr = np.minimum(np.abs(rr - cr), N - np.abs(rr - cr))
    dc = np.minimum(np.abs(cc_grid - cc), N - np.abs(cc_grid - cc))
    dist = np.sqrt(dr ** 2 + dc ** 2)

    annulus = (dist >= inner_r) & (dist < outer_r)
    ann_coords = np.argwhere(annulus)

    if len(ann_coords) < 2:
        return np.zeros(C + 2, dtype=np.float32)

    ann_vals = grid_mc[:, ann_coords[:, 0], ann_coords[:, 1]]  # (C, n_ann)

    ch_mean = ann_vals.mean(axis=1)  # (C,)
    total_intensity = float(ann_vals.mean())
    active_frac = float((ann_vals > 0.05).mean())

    return np.concatenate([ch_mean, [total_intensity, active_frac]]
                          ).astype(np.float32)


# ============================================================================
# Recording Episodes
# ============================================================================

def detect_patterns_for_wm(grid_mc: np.ndarray, threshold: float = 0.15,
                           min_size: int = 16, max_size: int = 10000
                           ) -> List[dict]:
    """Lightweight pattern detection for world model recording.

    Returns list of dicts with 'cells', 'center', 'size'.
    """
    aggregate = grid_mc.mean(axis=0)
    binary = (aggregate > threshold).astype(np.int32)
    labeled, n_features = ndimage.label(binary)

    patterns = []
    for label_id in range(1, n_features + 1):
        cells = np.argwhere(labeled == label_id)
        if not (min_size <= len(cells) <= max_size):
            continue
        values_agg = aggregate[cells[:, 0], cells[:, 1]]
        weights = values_agg / (values_agg.sum() + 1e-10)
        center = np.average(cells.astype(float), axis=0, weights=weights)
        patterns.append({
            'cells': cells,
            'center': center,
            'size': len(cells),
        })
    return patterns


def run_recording_episode(grid: np.ndarray, resource: np.ndarray,
                          run_chunk_fn, chunk_kwargs: dict,
                          n_recording_steps: int = 50,
                          substrate_steps_per_record: int = 10,
                          tau_values: tuple = (1, 2, 5, 10, 20),
                          threshold: float = 0.15) -> Dict:
    """Run a recording episode from a snapshot.

    Runs substrate forward, extracting features at each recording step.
    Returns dict mapping pattern_id -> list of PatternFeatures across time.

    Args:
        grid: (C, N, N) starting grid state
        resource: (N, N) starting resource
        run_chunk_fn: callable to step the substrate
        chunk_kwargs: kwargs for run_chunk_fn (kernel_ffts, config, etc.)
        n_recording_steps: how many measurement points to record
        substrate_steps_per_record: substrate steps between measurements
        tau_values: prediction horizons (in recording steps)
        threshold: pattern detection threshold
    """
    import jax.numpy as jnp
    from jax import random

    C, N = grid.shape[0], grid.shape[1]
    max_tau = max(tau_values)

    # Total steps needed: enough for recording + max lookahead
    total_steps = n_recording_steps + max_tau

    # Run substrate and collect grid snapshots at each recording step
    grids = [np.array(grid)]  # step 0
    resources = [np.array(resource)]
    g = jnp.array(grid)
    r = jnp.array(resource)
    rng = chunk_kwargs.get('rng', random.PRNGKey(0))

    for step_idx in range(total_steps):
        g, r, rng = run_chunk_fn(
            g, r, n_steps=substrate_steps_per_record, rng=rng,
            **{k: v for k, v in chunk_kwargs.items() if k != 'rng'})
        grids.append(np.array(g))
        resources.append(np.array(r))

    # Detect patterns at each recording step and extract features
    # Track patterns by proximity matching across steps
    all_features = {}  # pattern_track_id -> list of (step, s_B, s_dB, s_env_dict)

    # Use first-frame patterns as anchors
    initial_pats = detect_patterns_for_wm(grids[0], threshold=threshold)
    if not initial_pats:
        return {}

    # Assign IDs to initial patterns
    for pid, p in enumerate(initial_pats):
        all_features[pid] = {
            'center_history': [p['center'].copy()],
            'features': [],
        }

    for t in range(n_recording_steps):
        grid_t = grids[t]
        pats_t = detect_patterns_for_wm(grid_t, threshold=threshold)
        if not pats_t:
            continue

        # Match detected patterns to tracked patterns (one-to-one)
        tracked_ids = list(all_features.keys())
        matched_tracked = set()
        matched_detected = set()

        # Build cost matrix
        costs = []
        for j, p in enumerate(pats_t):
            for pid in tracked_ids:
                last_center = all_features[pid]['center_history'][-1]
                dr = abs(p['center'][0] - last_center[0])
                dc = abs(p['center'][1] - last_center[1])
                dr = min(dr, N - dr)
                dc = min(dc, N - dc)
                dist = np.sqrt(dr ** 2 + dc ** 2)
                if dist < 40.0:
                    costs.append((dist, pid, j))

        # Greedy one-to-one matching by lowest cost
        costs.sort()
        for dist, pid, j in costs:
            if pid in matched_tracked or j in matched_detected:
                continue
            matched_tracked.add(pid)
            matched_detected.add(j)

            p = pats_t[j]
            all_features[pid]['center_history'].append(p['center'].copy())

            # Extract features at time t
            s_B = extract_internal_state(grid_t, p['cells'])
            s_dB = extract_boundary_obs(grid_t, p['cells'], N)

            # Extract environment targets at t+τ for each τ
            s_env_dict = {}
            for tau in tau_values:
                future_t = t + tau
                if future_t < len(grids):
                    s_env_dict[tau] = extract_environment(
                        grids[future_t], p['center'], N)

            all_features[pid]['features'].append({
                'step': t,
                's_B': s_B,
                's_dB': s_dB,
                's_env': s_env_dict,
            })

    return all_features


# ============================================================================
# Prediction Gap Computation
# ============================================================================

@dataclass
class WorldModelResult:
    """Results of world model measurement for a single pattern."""
    pattern_id: int
    n_samples: int
    W: Dict[int, float]        # tau -> prediction gap W(τ)
    mse_full: Dict[int, float] # tau -> MSE of full model
    mse_env: Dict[int, float]  # tau -> MSE of env-only model
    H_wm: float                # horizon: max τ where W > 0
    C_wm: float                # capacity: ∫W(τ)dτ (trapezoidal)
    lifetime: int              # how many recording steps pattern was tracked


def compute_prediction_gap(features_list: List[dict],
                           tau_values: tuple = (1, 2, 5, 10, 20),
                           alpha: float = 1.0,
                           n_folds: int = 5) -> Optional[WorldModelResult]:
    """Compute prediction gap W(τ) for a single pattern's feature trajectory.

    Args:
        features_list: list of dicts with 's_B', 's_dB', 's_env' keys
        tau_values: prediction horizons to test
        alpha: Ridge regression regularization
        n_folds: cross-validation folds

    Returns:
        WorldModelResult or None if insufficient data
    """
    if len(features_list) < max(10, n_folds + 1):
        return None

    W = {}
    mse_full = {}
    mse_env = {}

    for tau in tau_values:
        # Collect samples where we have both features and future target
        X_full_list = []
        X_env_list = []
        Y_list = []

        for feat in features_list:
            if tau not in feat['s_env']:
                continue
            s_B = feat['s_B']
            s_dB = feat['s_dB']
            s_env = feat['s_env'][tau]

            if np.any(np.isnan(s_B)) or np.any(np.isnan(s_dB)) or np.any(np.isnan(s_env)):
                continue

            X_full_list.append(np.concatenate([s_B, s_dB]))
            X_env_list.append(s_dB)
            Y_list.append(s_env)

        n = len(Y_list)
        if n < max(10, n_folds + 1):
            W[tau] = 0.0
            mse_full[tau] = float('nan')
            mse_env[tau] = float('nan')
            continue

        X_full = np.array(X_full_list, dtype=np.float64)
        X_env = np.array(X_env_list, dtype=np.float64)
        Y = np.array(Y_list, dtype=np.float64)

        # Actual folds = min(n_folds, n)
        actual_folds = min(n_folds, n)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)

            # f_full: Ridge from (s_B, s_∂B) → s_env(t+τ)
            pipe_full = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            scores_full = cross_val_score(
                pipe_full, X_full, Y, cv=actual_folds,
                scoring='neg_mean_squared_error')
            mse_f = -scores_full.mean()

            # f_env: Ridge from s_∂B → s_env(t+τ)
            pipe_env = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            scores_env = cross_val_score(
                pipe_env, X_env, Y, cv=actual_folds,
                scoring='neg_mean_squared_error')
            mse_e = -scores_env.mean()

        W[tau] = max(0.0, mse_e - mse_f)  # clamp to ≥ 0
        mse_full[tau] = float(mse_f)
        mse_env[tau] = float(mse_e)

    # Derived measures
    valid_taus = sorted([t for t in tau_values if W.get(t, 0) > 1e-8])
    H_wm = float(valid_taus[-1]) if valid_taus else 0.0

    # Trapezoidal integration for C_wm
    sorted_taus = sorted(tau_values)
    W_vals = [W.get(t, 0.0) for t in sorted_taus]
    C_wm = float(np.trapz(W_vals, sorted_taus)) if len(sorted_taus) > 1 else 0.0

    return WorldModelResult(
        pattern_id=-1,
        n_samples=len(features_list),
        W=W,
        mse_full=mse_full,
        mse_env=mse_env,
        H_wm=H_wm,
        C_wm=C_wm,
        lifetime=len(features_list),
    )


# ============================================================================
# Full Pipeline: Snapshot → World Model Measurement
# ============================================================================

def measure_world_model_from_snapshot(
    snapshot_path: str,
    seed: int,
    config_overrides: Optional[dict] = None,
    n_recording_steps: int = 50,
    substrate_steps_per_record: int = 10,
    tau_values: tuple = (1, 2, 5, 10, 20),
    threshold: float = 0.15,
    max_patterns: int = 20,
) -> Tuple[List[WorldModelResult], dict]:
    """Full pipeline: load snapshot, run recording, compute prediction gaps.

    Args:
        snapshot_path: path to .npz file with 'grid' and 'resource'
        seed: random seed for substrate init (kernel/coupling generation)
        config_overrides: override config parameters
        n_recording_steps: measurement points per episode
        substrate_steps_per_record: substrate steps between measurements
        tau_values: prediction horizons
        threshold: pattern detection threshold
        max_patterns: max patterns to analyze (by size, descending)

    Returns:
        (results, metadata) where results is a list of WorldModelResult
    """
    import jax.numpy as jnp
    from jax import random
    from v13_substrate import generate_v13_config, init_v13, run_v13_chunk

    # Load snapshot
    snap = np.load(snapshot_path)
    grid_np = snap['grid']
    resource_np = snap['resource']
    C, N = grid_np.shape[0], grid_np.shape[1]

    # Generate config (must match the one used during evolution)
    config = generate_v13_config(C=C, N=N, seed=seed)
    # Match calibrated parameters from gpu_run
    config['maintenance_rate'] = 0.003
    config['resource_consume'] = 0.003
    config['resource_regen'] = 0.01
    if config_overrides:
        config.update(config_overrides)

    # Initialize substrate components (kernels, coupling, etc.)
    _, _, h_embed, kernel_ffts, coupling, coupling_row_sums, box_fft = \
        init_v13(config, seed=seed)

    rng = random.PRNGKey(seed + 9999)

    # Wrapper for run_chunk that matches recording episode interface
    def run_chunk_wrapper(grid, resource, n_steps, rng, **kwargs):
        return run_v13_chunk(
            grid, resource, h_embed, kernel_ffts, config,
            coupling, coupling_row_sums, rng, n_steps=n_steps,
            box_fft=box_fft)

    # Run recording episode
    features = run_recording_episode(
        grid=grid_np,
        resource=resource_np,
        run_chunk_fn=run_chunk_wrapper,
        chunk_kwargs={'rng': rng},
        n_recording_steps=n_recording_steps,
        substrate_steps_per_record=substrate_steps_per_record,
        tau_values=tau_values,
        threshold=threshold,
    )

    # Compute prediction gaps for each tracked pattern
    results = []
    # Sort by number of features (longest-tracked first)
    sorted_pids = sorted(features.keys(),
                         key=lambda pid: len(features[pid]['features']),
                         reverse=True)

    for pid in sorted_pids[:max_patterns]:
        feat_list = features[pid]['features']
        wm_result = compute_prediction_gap(
            feat_list, tau_values=tau_values)
        if wm_result is not None:
            wm_result.pattern_id = pid
            results.append(wm_result)

    metadata = {
        'snapshot_path': snapshot_path,
        'seed': seed,
        'C': C,
        'N': N,
        'n_recording_steps': n_recording_steps,
        'substrate_steps_per_record': substrate_steps_per_record,
        'tau_values': list(tau_values),
        'n_patterns_detected': len(features),
        'n_patterns_analyzed': len(results),
    }

    return results, metadata


def results_to_dict(results: List[WorldModelResult], metadata: dict) -> dict:
    """Serialize results for JSON output."""
    return {
        'metadata': metadata,
        'patterns': [
            {
                'pattern_id': r.pattern_id,
                'n_samples': r.n_samples,
                'W': {str(k): v for k, v in r.W.items()},
                'mse_full': {str(k): v for k, v in r.mse_full.items()},
                'mse_env': {str(k): v for k, v in r.mse_env.items()},
                'H_wm': r.H_wm,
                'C_wm': r.C_wm,
                'lifetime': r.lifetime,
            }
            for r in results
        ],
        'summary': {
            'mean_C_wm': float(np.mean([r.C_wm for r in results])) if results else 0.0,
            'max_C_wm': float(np.max([r.C_wm for r in results])) if results else 0.0,
            'mean_H_wm': float(np.mean([r.H_wm for r in results])) if results else 0.0,
            'n_with_world_model': sum(1 for r in results if r.C_wm > 1e-6),
            'frac_with_world_model': (
                sum(1 for r in results if r.C_wm > 1e-6) / max(len(results), 1)
            ),
            'mean_W_by_tau': {
                str(tau): float(np.mean([r.W.get(tau, 0) for r in results]))
                for tau in sorted(set().union(*(r.W.keys() for r in results)))
            } if results else {},
        },
    }
