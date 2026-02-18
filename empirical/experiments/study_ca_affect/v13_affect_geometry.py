"""Experiment 7 (partial): Affect Geometry Verification — A↔C alignment.

Tests whether structural affect (Space A, from internal dynamics) correlates
with behavioral affect (Space C, from observable behavior). If the affect
geometry is real — not a measurement artifact — internal states that are
structurally similar should drive similar behaviors.

Space A (Information-theoretic affect):
    - Valence:          resource proximity change (proxy for viability gradient)
    - Arousal:          ||Δs_B|| (magnitude of internal state change)
    - Integration:      Φ partition prediction loss
    - Effective rank:   d_eff of recent trajectory
    - CF weight:        I_img (detachment-period predictiveness)
    - Self-salience:    SM_sal (self vs env predictive advantage)

Space C (Behavioral affect):
    - Approach/avoid:   movement toward/away from nearest resource patch
    - Activity:         ||Δcenter|| (movement speed)
    - Growth:           Δsize (pattern size change)
    - Stability:        autocorrelation of trajectory direction

The test: RSA(A, C). Compute pairwise distance matrices in each space,
then correlate them (Spearman). If ρ(A,C) > 0 significantly, structural
affect drives behavior.
"""

import warnings
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy import stats, spatial, ndimage
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


@dataclass
class AffectGeometryResult:
    """Results of affect geometry alignment test."""
    n_patterns: int
    rsa_rho: float          # RSA correlation ρ(A,C)
    rsa_p: float            # p-value of RSA
    space_A: np.ndarray     # (n_patterns, dim_A) structural affect
    space_C: np.ndarray     # (n_patterns, dim_C) behavioral affect
    dim_labels_A: List[str]
    dim_labels_C: List[str]
    per_pattern: List[dict]  # per-pattern A and C values


def extract_space_A(features_list: List[dict],
                    resource_snapshots: List[np.ndarray],
                    centers: List[np.ndarray],
                    N: int) -> Optional[np.ndarray]:
    """Extract Space A (structural affect) from a pattern's trajectory.

    Returns (6,) vector: [valence, arousal, integration, d_eff, cf_weight, sm_sal]
    """
    if len(features_list) < 15:
        return None

    # --- Valence: internal vitality change (viability gradient proxy) ---
    # Use change in pattern's total activation (mass) as internal valence.
    # This is a structural property of the internal state — distinct from
    # the behavioral approach/avoid in Space C which uses spatial movement.
    valences = []
    for i in range(1, len(features_list)):
        # Pattern mass = sum of channel means (first C elements of s_B)
        C_ch = len(features_list[i]['s_B']) // 4  # s_B has 4C+4 elements
        mass_curr = float(features_list[i]['s_B'][:C_ch].sum())
        mass_prev = float(features_list[i - 1]['s_B'][:C_ch].sum())
        valences.append(mass_curr - mass_prev)

    valence = float(np.mean(valences)) if valences else 0.0

    # --- Arousal: magnitude of internal state change ---
    arousals = []
    for i in range(1, len(features_list)):
        delta = features_list[i]['s_B'] - features_list[i - 1]['s_B']
        arousals.append(np.linalg.norm(delta))
    arousal = float(np.mean(arousals)) if arousals else 0.0

    # --- Integration (Φ proxy): partition prediction loss ---
    # Use internal state cross-channel correlation as Φ proxy
    n_steps = min(len(features_list), 30)
    s_B_matrix = np.array([f['s_B'] for f in features_list[:n_steps]], dtype=np.float64)
    if s_B_matrix.shape[0] > 3:
        # Φ proxy: 1 - max variance explained by single partition
        cov = np.cov(s_B_matrix.T)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.maximum(eigvals, 0)
        total = eigvals.sum()
        if total > 1e-12:
            max_eigval = eigvals[-1]
            phi_proxy = 1.0 - (max_eigval / total)
        else:
            phi_proxy = 0.0
    else:
        phi_proxy = 0.0

    # --- Effective rank (d_eff) ---
    if s_B_matrix.shape[0] > 3 and total > 1e-12:
        p = eigvals / total
        p = p[p > 1e-10]
        d_eff = float(np.exp(-np.sum(p * np.log(p))))
    else:
        d_eff = 1.0

    # --- CF weight: detachment-period predictiveness ---
    # Use fraction of time with low synchrony as proxy
    sync_vals = []
    for i in range(1, len(features_list)):
        delta_sB = features_list[i]['s_B'] - features_list[i - 1]['s_B']
        s_dB = features_list[i]['s_dB']
        min_len = min(len(delta_sB), len(s_dB))
        if min_len < 2:
            continue
        mag_d = np.linalg.norm(delta_sB[:min_len])
        mag_b = np.linalg.norm(s_dB[:min_len])
        if mag_d > 1e-10 and mag_b > 1e-10:
            sync = abs(np.dot(delta_sB[:min_len], s_dB[:min_len]) /
                       (mag_d * mag_b))
        else:
            sync = 0.0
        sync_vals.append(sync)

    cf_weight = 1.0 - float(np.mean(sync_vals)) if sync_vals else 0.0

    # --- Self-model salience proxy ---
    # How much does s_B autocorrelate vs s_dB autocorrelate
    if len(features_list) > 5:
        sb_auto = []
        sdb_auto = []
        for i in range(1, min(len(features_list), 20)):
            sb_auto.append(np.corrcoef(features_list[0]['s_B'],
                                        features_list[i]['s_B'])[0, 1])
            min_l = min(len(features_list[0]['s_dB']),
                        len(features_list[i]['s_dB']))
            sdb_auto.append(np.corrcoef(features_list[0]['s_dB'][:min_l],
                                         features_list[i]['s_dB'][:min_l])[0, 1])
        sb_mean = np.nanmean(sb_auto)
        sdb_mean = np.nanmean(sdb_auto)
        sm_sal = float(sb_mean - sdb_mean) if np.isfinite(sb_mean) and np.isfinite(sdb_mean) else 0.0
    else:
        sm_sal = 0.0

    return np.array([valence, arousal, phi_proxy, d_eff, cf_weight, sm_sal],
                     dtype=np.float64)


def extract_space_C(features_list: List[dict],
                    centers: List[np.ndarray],
                    sizes: List[int],
                    resource_snapshots: List[np.ndarray],
                    N: int) -> Optional[np.ndarray]:
    """Extract Space C (behavioral affect) from observable behavior.

    Returns (4,) vector: [approach_avoid, activity, growth, stability]
    """
    if len(centers) < 5:
        return None

    # --- Approach/avoidance ---
    # Movement toward/away from nearest high-resource region
    approach_vals = []
    for i in range(1, min(len(centers), len(resource_snapshots))):
        resource = resource_snapshots[i]
        center = centers[i]
        center_prev = centers[i - 1]

        # Find nearest resource peak (simple: resource at current pos vs prev)
        cr, cc = int(round(center[0])) % N, int(round(center[1])) % N
        pr, pc = int(round(center_prev[0])) % N, int(round(center_prev[1])) % N
        approach_vals.append(float(resource[cr, cc] - resource[pr, pc]))

    approach = float(np.mean(approach_vals)) if approach_vals else 0.0

    # --- Activity: movement speed ---
    speeds = []
    for i in range(1, len(centers)):
        dr = abs(centers[i][0] - centers[i - 1][0])
        dc = abs(centers[i][1] - centers[i - 1][1])
        dr = min(dr, N - dr)
        dc = min(dc, N - dc)
        speeds.append(np.sqrt(dr ** 2 + dc ** 2))
    activity = float(np.mean(speeds)) if speeds else 0.0

    # --- Growth: pattern size change ---
    if len(sizes) >= 2:
        growth = float(sizes[-1] - sizes[0]) / max(sizes[0], 1)
    else:
        growth = 0.0

    # --- Stability: autocorrelation of movement direction ---
    if len(centers) >= 4:
        directions = []
        for i in range(1, len(centers)):
            dr = centers[i][0] - centers[i - 1][0]
            dc = centers[i][1] - centers[i - 1][1]
            # Handle periodic boundary
            if abs(dr) > N / 2:
                dr = dr - np.sign(dr) * N
            if abs(dc) > N / 2:
                dc = dc - np.sign(dc) * N
            angle = np.arctan2(dc, dr)
            directions.append(angle)
        if len(directions) >= 3:
            # Angular autocorrelation at lag 1
            cos_diffs = [np.cos(directions[i] - directions[i - 1])
                         for i in range(1, len(directions))]
            stability = float(np.mean(cos_diffs))
        else:
            stability = 0.0
    else:
        stability = 0.0

    return np.array([approach, activity, growth, stability], dtype=np.float64)


def compute_rsa(space_A: np.ndarray, space_C: np.ndarray) -> Tuple[float, float]:
    """Representational Similarity Analysis between two spaces.

    Computes pairwise distance matrices in each space, then correlates
    the upper triangles (Spearman).

    Returns (rho, p_value).
    """
    n = space_A.shape[0]
    if n < 4:
        return 0.0, 1.0

    # Standardize each space
    A_std = (space_A - space_A.mean(axis=0)) / (space_A.std(axis=0) + 1e-10)
    C_std = (space_C - space_C.mean(axis=0)) / (space_C.std(axis=0) + 1e-10)

    # Pairwise Euclidean distance matrices
    D_A = spatial.distance.pdist(A_std, metric='euclidean')
    D_C = spatial.distance.pdist(C_std, metric='euclidean')

    if len(D_A) < 3 or np.std(D_A) < 1e-10 or np.std(D_C) < 1e-10:
        return 0.0, 1.0

    rho, p = stats.spearmanr(D_A, D_C)
    return float(rho) if np.isfinite(rho) else 0.0, float(p) if np.isfinite(p) else 1.0


# ============================================================================
# Full pipeline
# ============================================================================

def measure_affect_geometry_from_snapshot(
    snapshot_path: str,
    seed: int,
    config_overrides: Optional[dict] = None,
    n_recording_steps: int = 50,
    substrate_steps_per_record: int = 10,
    threshold: float = 0.15,
    max_patterns: int = 20,
) -> Tuple[Optional[AffectGeometryResult], dict]:
    """Load snapshot, run recording, compute affect geometry alignment."""
    import jax.numpy as jnp
    from jax import random
    from v13_substrate import generate_v13_config, init_v13, run_v13_chunk
    from v13_world_model import (
        detect_patterns_for_wm, extract_internal_state,
        extract_boundary_obs,
    )

    snap = np.load(snapshot_path)
    grid_np = snap['grid']
    resource_np = snap['resource']
    C, N = grid_np.shape[0], grid_np.shape[1]

    config = generate_v13_config(C=C, N=N, seed=seed)
    config['maintenance_rate'] = 0.003
    config['resource_consume'] = 0.003
    config['resource_regen'] = 0.01
    if config_overrides:
        config.update(config_overrides)

    _, _, h_embed, kernel_ffts, coupling, coupling_row_sums, box_fft = \
        init_v13(config, seed=seed)
    rng = random.PRNGKey(seed + 9999)

    # Run forward collecting grids AND resources
    grids = [grid_np.copy()]
    resources = [resource_np.copy()]
    g = jnp.array(grid_np)
    r = jnp.array(resource_np)

    for _ in range(n_recording_steps):
        g, r, rng = run_v13_chunk(
            g, r, h_embed, kernel_ffts, config,
            coupling, coupling_row_sums, rng,
            n_steps=substrate_steps_per_record, box_fft=box_fft)
        grids.append(np.array(g))
        resources.append(np.array(r))

    # Track patterns
    initial_pats = detect_patterns_for_wm(grids[0], threshold=threshold)
    if not initial_pats:
        return None, {'n_patterns_detected': 0}

    all_features = {}
    for pid, p in enumerate(initial_pats[:max_patterns]):
        all_features[pid] = {
            'center_history': [p['center'].copy()],
            'size_history': [p['size']],
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
            all_features[pid]['size_history'].append(p['size'])

            s_B = extract_internal_state(grid_t, p['cells'])
            s_dB = extract_boundary_obs(grid_t, p['cells'], N)

            all_features[pid]['features'].append({
                's_B': s_B,
                's_dB': s_dB,
            })

    # Compute Space A and Space C for each pattern
    dim_A = ['valence', 'arousal', 'integration', 'd_eff', 'cf_weight', 'sm_sal']
    dim_C = ['approach_avoid', 'activity', 'growth', 'stability']

    A_list = []
    C_list = []
    per_pattern = []

    sorted_pids = sorted(all_features.keys(),
                         key=lambda pid: len(all_features[pid]['features']),
                         reverse=True)

    for pid in sorted_pids[:max_patterns]:
        data = all_features[pid]
        feat_list = data['features']
        centers = data['center_history']
        sizes = data['size_history']

        a = extract_space_A(feat_list, resources, centers, N)
        c = extract_space_C(feat_list, centers, sizes, resources, N)

        if a is not None and c is not None:
            A_list.append(a)
            C_list.append(c)
            per_pattern.append({
                'pattern_id': pid,
                'space_A': {dim_A[i]: float(a[i]) for i in range(len(dim_A))},
                'space_C': {dim_C[i]: float(c[i]) for i in range(len(dim_C))},
                'n_timesteps': len(feat_list),
            })

    if len(A_list) < 4:
        metadata = {
            'snapshot_path': snapshot_path, 'seed': seed,
            'C': C, 'N': N, 'n_patterns_detected': len(initial_pats),
            'n_patterns_analyzed': len(A_list),
        }
        return None, metadata

    space_A = np.array(A_list)
    space_C = np.array(C_list)

    rsa_rho, rsa_p = compute_rsa(space_A, space_C)

    metadata = {
        'snapshot_path': snapshot_path, 'seed': seed,
        'C': C, 'N': N,
        'n_recording_steps': n_recording_steps,
        'n_patterns_detected': len(initial_pats),
        'n_patterns_analyzed': len(A_list),
    }

    result = AffectGeometryResult(
        n_patterns=len(A_list),
        rsa_rho=rsa_rho,
        rsa_p=rsa_p,
        space_A=space_A,
        space_C=space_C,
        dim_labels_A=dim_A,
        dim_labels_C=dim_C,
        per_pattern=per_pattern,
    )

    return result, metadata


def result_to_dict(result: Optional[AffectGeometryResult], metadata: dict,
                   cycle: int = -1) -> dict:
    if result is None:
        return {
            'metadata': metadata,
            'rsa_rho': None,
            'rsa_p': None,
            'n_patterns': 0,
            'per_pattern': [],
            'cycle': cycle,
        }
    return {
        'metadata': metadata,
        'rsa_rho': result.rsa_rho,
        'rsa_p': result.rsa_p,
        'n_patterns': result.n_patterns,
        'dim_labels_A': result.dim_labels_A,
        'dim_labels_C': result.dim_labels_C,
        'per_pattern': result.per_pattern,
        'space_A_means': {result.dim_labels_A[i]: float(result.space_A[:, i].mean())
                          for i in range(len(result.dim_labels_A))},
        'space_A_stds': {result.dim_labels_A[i]: float(result.space_A[:, i].std())
                         for i in range(len(result.dim_labels_A))},
        'space_C_means': {result.dim_labels_C[i]: float(result.space_C[:, i].mean())
                          for i in range(len(result.dim_labels_C))},
        'space_C_stds': {result.dim_labels_C[i]: float(result.space_C[:, i].std())
                         for i in range(len(result.dim_labels_C))},
        'cycle': cycle,
    }
