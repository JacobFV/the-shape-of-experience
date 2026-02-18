"""Experiment 3: Internal Representation Structure — Measurement Module.

Measures whether patterns develop low-dimensional, compositional internal
representations. Runs on same recording data as Experiment 2 (world model).

Key quantities:
    d_eff: Effective dimensionality of s_B across contexts
           = (tr Σ)² / tr(Σ²)  — low means compressed representation
    D:     Disentanglement score — how well individual PCA dimensions
           of s_B predict specific environmental features
    A:     Abstraction level = 1 - d_eff / d_raw  — compression ratio
    K_comp: Compositionality — whether representations compose linearly
            across environmental contexts
"""

import warnings
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy import ndimage


# ============================================================================
# Representation Analysis
# ============================================================================

@dataclass
class RepresentationResult:
    """Representation structure metrics for a single pattern."""
    pattern_id: int
    n_samples: int
    d_eff: float         # effective dimensionality
    d_raw: int           # raw dimensionality (len(s_B))
    A: float             # abstraction level = 1 - d_eff / d_raw
    D: float             # disentanglement score
    K_comp: float        # compositionality (lower = more compositional)
    eigenspectrum: list   # sorted eigenvalues for plotting
    r2_matrix: list       # (n_pca_dims, n_env_features) R² values


def effective_dimensionality(X: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute effective dimensionality of rows of X.

    d_eff = (tr Σ)² / tr(Σ²) where Σ = covariance matrix.
    Returns (d_eff, sorted_eigenvalues).
    """
    if X.shape[0] < 3:
        return float(X.shape[1]), np.ones(X.shape[1])

    X = X.astype(np.float64)
    # Standardize before covariance to avoid overflow from mixed-scale features
    stds = X.std(axis=0)
    stds[stds < 1e-12] = 1.0
    X_scaled = (X - X.mean(axis=0)) / stds
    cov = (X_scaled.T @ X_scaled) / (X.shape[0] - 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0)  # numerical safety
    eigvals = np.sort(eigvals)[::-1]  # descending

    tr = eigvals.sum()
    tr_sq = (eigvals ** 2).sum()

    if tr_sq < 1e-20:
        return float(X.shape[1]), eigvals

    d_eff = (tr ** 2) / tr_sq
    return float(d_eff), eigvals


def disentanglement_score(X_internal: np.ndarray, X_env: np.ndarray,
                          n_pca_dims: int = 10) -> Tuple[float, np.ndarray]:
    """Compute disentanglement: do individual PCA dims of s_B predict specific env features?

    D = (1/p) Σᵢ max_j r²(z_j, f_i)

    where z_j is PCA component j of internal state, f_i is environmental feature i.

    Args:
        X_internal: (n_samples, d_internal) internal state vectors
        X_env: (n_samples, d_env) environmental feature vectors
        n_pca_dims: how many PCA dimensions to use

    Returns:
        (D, r2_matrix) where r2_matrix is (n_pca_dims, d_env)
    """
    n = X_internal.shape[0]
    d_env = X_env.shape[1]

    if n < 5:
        return 0.0, np.zeros((n_pca_dims, d_env))

    X_internal = X_internal.astype(np.float64)
    X_env = X_env.astype(np.float64)

    # PCA on standardized internal state
    stds = X_internal.std(axis=0)
    stds[stds < 1e-12] = 1.0
    X_scaled = (X_internal - X_internal.mean(axis=0)) / stds
    cov = (X_scaled.T @ X_scaled) / (n - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    actual_dims = min(n_pca_dims, X_internal.shape[1], n - 1)
    Z = X_scaled @ eigvecs[:, :actual_dims]  # (n, actual_dims) — PCA projections

    # Standardize env features
    scaler = StandardScaler()
    F = scaler.fit_transform(X_env)

    # R² between each PCA dim and each env feature
    r2_matrix = np.zeros((actual_dims, d_env))

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        for j in range(actual_dims):
            z_j = Z[:, j]
            z_var = z_j.var()
            if z_var < 1e-12:
                continue
            for i in range(d_env):
                f_i = F[:, i]
                # Simple linear R²
                corr = np.corrcoef(z_j, f_i)[0, 1]
                if np.isfinite(corr):
                    r2_matrix[j, i] = corr ** 2

    # D = (1/p) Σᵢ max_j r²(z_j, f_i)
    if d_env > 0:
        D = float(r2_matrix.max(axis=0).mean())
    else:
        D = 0.0

    # Pad to requested size if needed
    if actual_dims < n_pca_dims:
        padded = np.zeros((n_pca_dims, d_env))
        padded[:actual_dims, :] = r2_matrix
        r2_matrix = padded

    return D, r2_matrix


def compositionality_score(features_list: List[dict],
                           n_context_pairs: int = 20) -> float:
    """Measure whether representations compose linearly across contexts.

    K_comp = mean ‖z_{A∩B} - (z_A + z_B - z_∅)‖ / ‖z_{A∩B}‖

    Contexts defined by environmental feature thresholds:
        A = "high resource nearby", B = "many neighbors", etc.

    Low K_comp = linear compositionality.
    """
    if len(features_list) < 20:
        return 1.0  # no data → maximally non-compositional

    # Stack internal states and environment features
    S_B = np.array([f['s_B'] for f in features_list], dtype=np.float64)
    S_env = np.array([f['s_env_raw'] for f in features_list], dtype=np.float64)

    n = S_B.shape[0]
    d_env = S_env.shape[1]

    if d_env < 2 or n < 20:
        return 1.0

    # Define contexts by median splits on env features
    medians = np.median(S_env, axis=0)
    context_masks = {}
    for i in range(min(d_env, 8)):  # up to 8 context dimensions
        context_masks[f'high_{i}'] = S_env[:, i] > medians[i]
        context_masks[f'low_{i}'] = S_env[:, i] <= medians[i]

    # Sample context pairs and test compositionality
    context_names = list(context_masks.keys())
    rng = np.random.RandomState(42)
    k_comp_values = []

    for _ in range(n_context_pairs):
        # Pick two random context features
        idx = rng.choice(len(context_names), 2, replace=False)
        name_a, name_b = context_names[idx[0]], context_names[idx[1]]
        mask_a = context_masks[name_a]
        mask_b = context_masks[name_b]

        mask_ab = mask_a & mask_b
        mask_none = ~mask_a & ~mask_b

        # Need enough samples in each partition
        if mask_a.sum() < 3 or mask_b.sum() < 3 or mask_ab.sum() < 3 or mask_none.sum() < 3:
            continue

        z_a = S_B[mask_a].mean(axis=0)
        z_b = S_B[mask_b].mean(axis=0)
        z_ab = S_B[mask_ab].mean(axis=0)
        z_none = S_B[mask_none].mean(axis=0)

        # Linear composition prediction
        z_pred = z_a + z_b - z_none
        norm_ab = np.linalg.norm(z_ab)
        if norm_ab < 1e-10:
            continue

        k = np.linalg.norm(z_ab - z_pred) / norm_ab
        k_comp_values.append(k)

    return float(np.mean(k_comp_values)) if k_comp_values else 1.0


# ============================================================================
# Pipeline: Compute representation metrics from world model recording data
# ============================================================================

def analyze_representation(features_list: List[dict],
                           n_pca_dims: int = 10) -> Optional[RepresentationResult]:
    """Compute all representation metrics for a single pattern's trajectory.

    Expects features_list items to have 's_B', 's_dB', and 's_env_raw' keys.
    's_env_raw' is the environment feature vector (used for disentanglement
    and compositionality). If 's_env_raw' is missing, falls back to the
    first available s_env target from the world model data.
    """
    if len(features_list) < 10:
        return None

    # Stack feature vectors
    S_B = np.array([f['s_B'] for f in features_list], dtype=np.float64)

    # Environment features for disentanglement
    if 's_env_raw' in features_list[0]:
        S_env = np.array([f['s_env_raw'] for f in features_list], dtype=np.float64)
    else:
        # Fall back: use s_env from smallest tau
        env_key = None
        for f in features_list:
            if 's_env' in f and isinstance(f['s_env'], dict):
                taus = sorted(f['s_env'].keys())
                if taus:
                    env_key = taus[0]
                    break
        if env_key is not None:
            S_env = np.array([f['s_env'][env_key] for f in features_list
                              if env_key in f.get('s_env', {})], dtype=np.float64)
            S_B = S_B[:len(S_env)]
        else:
            S_env = np.zeros((len(S_B), 1))

    # 1. Effective dimensionality
    d_eff, eigspectrum = effective_dimensionality(S_B)
    d_raw = S_B.shape[1]

    # 2. Abstraction level
    A = 1.0 - d_eff / d_raw

    # 3. Disentanglement
    D, r2_matrix = disentanglement_score(S_B, S_env, n_pca_dims=n_pca_dims)

    # 4. Compositionality
    K_comp = compositionality_score(features_list)

    return RepresentationResult(
        pattern_id=-1,
        n_samples=len(features_list),
        d_eff=d_eff,
        d_raw=d_raw,
        A=A,
        D=D,
        K_comp=K_comp,
        eigenspectrum=eigspectrum[:20].tolist(),
        r2_matrix=r2_matrix.tolist(),
    )


# ============================================================================
# Full pipeline: Snapshot → Representation metrics
# ============================================================================

def measure_representation_from_snapshot(
    snapshot_path: str,
    seed: int,
    config_overrides: Optional[dict] = None,
    n_recording_steps: int = 50,
    substrate_steps_per_record: int = 10,
    threshold: float = 0.15,
    max_patterns: int = 20,
) -> Tuple[List[RepresentationResult], dict]:
    """Full pipeline: load snapshot, run recording, compute representation metrics.

    Reuses the world model recording infrastructure but extracts different measures.
    """
    import jax.numpy as jnp
    from jax import random
    from v13_substrate import generate_v13_config, init_v13, run_v13_chunk
    from v13_world_model import (
        detect_patterns_for_wm, extract_internal_state,
        extract_boundary_obs, extract_environment,
    )

    # Load snapshot
    snap = np.load(snapshot_path)
    grid_np = snap['grid']
    resource_np = snap['resource']
    C, N = grid_np.shape[0], grid_np.shape[1]

    # Generate config
    config = generate_v13_config(C=C, N=N, seed=seed)
    config['maintenance_rate'] = 0.003
    config['resource_consume'] = 0.003
    config['resource_regen'] = 0.01
    if config_overrides:
        config.update(config_overrides)

    _, _, h_embed, kernel_ffts, coupling, coupling_row_sums, box_fft = \
        init_v13(config, seed=seed)
    rng = random.PRNGKey(seed + 9999)

    # Run forward to collect snapshots
    grids = [grid_np.copy()]
    g = jnp.array(grid_np)
    r = jnp.array(resource_np)

    for _ in range(n_recording_steps):
        g, r, rng = run_v13_chunk(
            g, r, h_embed, kernel_ffts, config,
            coupling, coupling_row_sums, rng,
            n_steps=substrate_steps_per_record, box_fft=box_fft)
        grids.append(np.array(g))

    # Track patterns and extract features
    initial_pats = detect_patterns_for_wm(grids[0], threshold=threshold)
    if not initial_pats:
        return [], {'n_patterns_detected': 0}

    all_features = {}
    for pid, p in enumerate(initial_pats[:max_patterns]):
        all_features[pid] = {
            'center_history': [p['center'].copy()],
            'features': [],
        }

    for t in range(n_recording_steps):
        grid_t = grids[t]
        pats_t = detect_patterns_for_wm(grid_t, threshold=threshold)
        if not pats_t:
            continue

        tracked_ids = list(all_features.keys())
        matched_tracked = set()
        matched_detected = set()
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

        costs.sort()
        for dist, pid, j in costs:
            if pid in matched_tracked or j in matched_detected:
                continue
            matched_tracked.add(pid)
            matched_detected.add(j)

            p = pats_t[j]
            all_features[pid]['center_history'].append(p['center'].copy())

            s_B = extract_internal_state(grid_t, p['cells'])
            s_dB = extract_boundary_obs(grid_t, p['cells'], N)
            # Current environment (not future — for disentanglement)
            s_env_raw = extract_environment(grid_t, p['center'], N)

            all_features[pid]['features'].append({
                's_B': s_B,
                's_dB': s_dB,
                's_env_raw': s_env_raw,
            })

    # Analyze each pattern
    results = []
    sorted_pids = sorted(all_features.keys(),
                         key=lambda pid: len(all_features[pid]['features']),
                         reverse=True)

    for pid in sorted_pids[:max_patterns]:
        feat_list = all_features[pid]['features']
        rep = analyze_representation(feat_list)
        if rep is not None:
            rep.pattern_id = pid
            results.append(rep)

    metadata = {
        'snapshot_path': snapshot_path,
        'seed': seed,
        'C': C, 'N': N,
        'n_recording_steps': n_recording_steps,
        'n_patterns_detected': len(initial_pats),
        'n_patterns_analyzed': len(results),
    }

    return results, metadata


def results_to_dict(results: List[RepresentationResult], metadata: dict) -> dict:
    """Serialize for JSON."""
    return {
        'metadata': metadata,
        'patterns': [
            {
                'pattern_id': r.pattern_id,
                'n_samples': r.n_samples,
                'd_eff': r.d_eff,
                'd_raw': r.d_raw,
                'A': r.A,
                'D': r.D,
                'K_comp': r.K_comp,
                'eigenspectrum': r.eigenspectrum,
            }
            for r in results
        ],
        'summary': {
            'mean_d_eff': float(np.mean([r.d_eff for r in results])) if results else 0,
            'mean_A': float(np.mean([r.A for r in results])) if results else 0,
            'mean_D': float(np.mean([r.D for r in results])) if results else 0,
            'mean_K_comp': float(np.mean([r.K_comp for r in results])) if results else 1,
            'n_compressed': sum(1 for r in results if r.A > 0.5),
            'n_disentangled': sum(1 for r in results if r.D > 0.3),
        },
    }
