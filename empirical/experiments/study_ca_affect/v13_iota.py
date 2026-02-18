"""Experiment 8 (adapted): Inhibition Coefficient (ι) Emergence.

Measures whether patterns develop modulable perceptual coupling —
different internal responses to other patterns vs resource patches.

Simplified for V13 (where self-models are weak/absent):

    ι(i) = 1 - MI_social(i) / (MI_social(i) + MI_resource(i))

    MI_social(i):   mean MI(s_B_i; s_B_j) across nearby patterns j
    MI_resource(i): MI(s_B_i; local_resource_distribution)
    MI_traj(i):     mean MI(s_B_i; trajectory_j) — tracking others' motion

Key questions:
    1. Do patterns model other patterns' internals (low ι, participatory)?
    2. Or only their trajectories (high ι, mechanistic)?
    3. Does ι vary across patterns / over evolution?
    4. Animism test: MI_resource vs MI_social — do patterns "model" resources
       like they model agents?
"""

import warnings
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy import stats


@dataclass
class IotaResult:
    """ι metrics for a snapshot."""
    n_patterns: int
    mean_iota: float              # mean ι across patterns
    std_iota: float               # std of ι
    mean_MI_social: float         # mean inter-pattern MI (internal states)
    mean_MI_trajectory: float     # mean MI with others' trajectories
    mean_MI_resource: float       # mean MI with local resources
    iota_range: Tuple[float, float]  # (min, max) ι
    animism_score: float          # MI_resource / MI_social (>1 = treats resources like agents)
    per_pattern: List[dict]       # per-pattern ι values


def _gaussian_mi(X: np.ndarray, Y: np.ndarray, max_dims: int = 8) -> float:
    """Estimate MI between X and Y using Gaussian correlation estimate.

    MI = -0.5 * sum(log(1 - r_d^2)) for each matched dimension d.
    """
    n = min(len(X), len(Y))
    if n < 15:
        return 0.0

    d = min(max_dims, X.shape[1], Y.shape[1])
    mi_total = 0.0
    count = 0

    for i in range(d):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            try:
                r, _ = stats.pearsonr(X[:n, i], Y[:n, i])
            except Exception:
                continue
        if np.isfinite(r):
            r_clipped = np.clip(r ** 2, 0, 0.999)
            mi_total += -0.5 * np.log(1 - r_clipped)
            count += 1

    return float(mi_total / max(count, 1))


def compute_trajectory_features(center_history: List[np.ndarray],
                                 N: int) -> np.ndarray:
    """Convert center history to trajectory features.

    For each timestep t, computes:
        [dx, dy, speed, heading_sin, heading_cos]
    where dx, dy are periodic-corrected displacements.
    """
    if len(center_history) < 3:
        return np.zeros((0, 5))

    feats = []
    for t in range(1, len(center_history)):
        prev = center_history[t - 1]
        curr = center_history[t]
        dr = curr[0] - prev[0]
        dc = curr[1] - prev[1]
        # Periodic correction
        if abs(dr) > N / 2:
            dr = dr - np.sign(dr) * N
        if abs(dc) > N / 2:
            dc = dc - np.sign(dc) * N
        speed = np.sqrt(dr ** 2 + dc ** 2)
        heading = np.arctan2(dr, dc)
        feats.append([dr, dc, speed, np.sin(heading), np.cos(heading)])

    return np.array(feats, dtype=np.float64)


def compute_resource_features(resource_history: List[np.ndarray],
                               center_history: List[np.ndarray],
                               N: int, radius: int = 15) -> np.ndarray:
    """Extract local resource distribution features around pattern center.

    For each timestep: [mean_resource, std_resource, gradient_r, gradient_c,
                        total_resource, resource_change]
    """
    if not resource_history or len(resource_history) < 2:
        return np.zeros((0, 6))

    feats = []
    for t in range(len(resource_history)):
        res = resource_history[t]
        if t >= len(center_history):
            break
        cr, cc = int(round(center_history[t][0])), int(round(center_history[t][1]))

        # Local patch (periodic)
        rr, cc_grid = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        dr = np.minimum(np.abs(rr - cr), N - np.abs(rr - cr))
        dc = np.minimum(np.abs(cc_grid - cc), N - np.abs(cc_grid - cc))
        dist = np.sqrt(dr ** 2 + dc ** 2)
        local = dist < radius

        local_vals = res[local]
        mean_r = float(local_vals.mean()) if len(local_vals) > 0 else 0.0
        std_r = float(local_vals.std()) if len(local_vals) > 0 else 0.0
        total_r = float(local_vals.sum()) if len(local_vals) > 0 else 0.0

        # Gradient (directional resource change)
        left_mask = dist < radius
        right_half = (cc_grid - cc) % N < N // 2
        top_half = (rr - cr) % N < N // 2
        grad_r = float(res[local & top_half].mean() - res[local & ~top_half].mean()) \
            if (local & top_half).sum() > 0 and (local & ~top_half).sum() > 0 else 0.0
        grad_c = float(res[local & right_half].mean() - res[local & ~right_half].mean()) \
            if (local & right_half).sum() > 0 and (local & ~right_half).sum() > 0 else 0.0

        # Resource change from previous step
        if t > 0 and t - 1 < len(resource_history):
            prev_res = resource_history[t - 1]
            prev_local = prev_res[local]
            change = mean_r - float(prev_local.mean()) if len(prev_local) > 0 else 0.0
        else:
            change = 0.0

        feats.append([mean_r, std_r, grad_r, grad_c, total_r, change])

    return np.array(feats, dtype=np.float64)


def analyze_iota(all_features: dict) -> Optional[IotaResult]:
    """Compute ι for each pattern in a snapshot.

    all_features[pid] = {
        'features': [{'s_B': ..., 's_dB': ...}, ...],
        'center_history': [...],
        'resource_history': [...],  # optional
    }
    """
    pids = sorted(all_features.keys())
    valid_pids = [pid for pid in pids
                  if len(all_features[pid]['features']) >= 15
                  and len(all_features[pid].get('center_history', [])) >= 15]

    if len(valid_pids) < 3:
        return None

    N = all_features[valid_pids[0]].get('N', 128)

    # Pre-extract internal state matrices for each pattern
    internal = {}
    for pid in valid_pids:
        feats = all_features[pid]['features']
        n = min(len(feats), 50)
        internal[pid] = np.array([f['s_B'][:8] for f in feats[:n]], dtype=np.float64)

    # Pre-extract trajectory features
    trajectories = {}
    for pid in valid_pids:
        centers = all_features[pid]['center_history']
        trajectories[pid] = compute_trajectory_features(centers, N)

    # Pre-extract resource features
    resources = {}
    has_resources = False
    for pid in valid_pids:
        res_hist = all_features[pid].get('resource_history', [])
        centers = all_features[pid]['center_history']
        if res_hist and len(res_hist) >= 10:
            resources[pid] = compute_resource_features(res_hist, centers, N)
            has_resources = True
        else:
            resources[pid] = np.zeros((0, 6))

    per_pattern = []
    for pid in valid_pids:
        X_i = internal[pid]
        n_i = len(X_i)

        # MI_social: mean MI(s_B_i; s_B_j) across other patterns
        mi_social_vals = []
        for other_pid in valid_pids:
            if other_pid == pid:
                continue
            X_j = internal[other_pid]
            n = min(n_i, len(X_j))
            if n < 15:
                continue
            mi = _gaussian_mi(X_i[:n], X_j[:n])
            mi_social_vals.append(mi)

        mi_social = float(np.mean(mi_social_vals)) if mi_social_vals else 0.0

        # MI_trajectory: mean MI(s_B_i; trajectory_j)
        mi_traj_vals = []
        for other_pid in valid_pids:
            if other_pid == pid:
                continue
            traj_j = trajectories[other_pid]
            if len(traj_j) < 15:
                continue
            n = min(n_i, len(traj_j))
            if n < 15:
                continue
            mi = _gaussian_mi(X_i[:n], traj_j[:n])
            mi_traj_vals.append(mi)

        mi_traj = float(np.mean(mi_traj_vals)) if mi_traj_vals else 0.0

        # MI_resource: MI(s_B_i; local_resource)
        mi_resource = 0.0
        if has_resources and len(resources[pid]) >= 15:
            n = min(n_i, len(resources[pid]))
            if n >= 15:
                mi_resource = _gaussian_mi(X_i[:n], resources[pid][:n])

        # ι = 1 - MI_social / (MI_social + MI_trajectory)
        denom = mi_social + mi_traj
        iota = 1.0 - mi_social / denom if denom > 1e-10 else 0.5

        # Animism score: MI_resource / MI_social
        animism = mi_resource / mi_social if mi_social > 1e-10 else 0.0

        per_pattern.append({
            'pattern_id': pid,
            'iota': float(np.clip(iota, 0, 1)),
            'MI_social': mi_social,
            'MI_trajectory': mi_traj,
            'MI_resource': mi_resource,
            'animism_score': animism,
        })

    iotas = [p['iota'] for p in per_pattern]
    mi_socials = [p['MI_social'] for p in per_pattern]
    mi_trajs = [p['MI_trajectory'] for p in per_pattern]
    mi_resources = [p['MI_resource'] for p in per_pattern]

    return IotaResult(
        n_patterns=len(valid_pids),
        mean_iota=float(np.mean(iotas)),
        std_iota=float(np.std(iotas)),
        mean_MI_social=float(np.mean(mi_socials)),
        mean_MI_trajectory=float(np.mean(mi_trajs)),
        mean_MI_resource=float(np.mean(mi_resources)),
        iota_range=(float(np.min(iotas)), float(np.max(iotas))),
        animism_score=float(np.mean([p['animism_score'] for p in per_pattern])),
        per_pattern=per_pattern,
    )


# ============================================================================
# Full pipeline
# ============================================================================

def measure_iota_from_snapshot(
    snapshot_path: str,
    seed: int,
    config_overrides: Optional[dict] = None,
    n_recording_steps: int = 50,
    substrate_steps_per_record: int = 10,
    threshold: float = 0.15,
    max_patterns: int = 20,
) -> Tuple[Optional[IotaResult], dict]:
    """Load snapshot, run recording, compute ι metrics."""
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
    rng = random.PRNGKey(seed + 8888)

    grids = [grid_np.copy()]
    resource_list = [resource_np.copy()]
    g = jnp.array(grid_np)
    r = jnp.array(resource_np)

    for _ in range(n_recording_steps):
        g, r, rng = run_v13_chunk(
            g, r, h_embed, kernel_ffts, config,
            coupling, coupling_row_sums, rng,
            n_steps=substrate_steps_per_record, box_fft=box_fft)
        grids.append(np.array(g))
        resource_list.append(np.array(r))

    initial_pats = detect_patterns_for_wm(grids[0], threshold=threshold)
    if not initial_pats:
        return None, {'n_patterns_detected': 0}

    all_features = {}
    for pid, p in enumerate(initial_pats[:max_patterns]):
        all_features[pid] = {
            'center_history': [p['center'].copy()],
            'features': [],
            'resource_history': [resource_list[0]],
            'N': N,
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
            all_features[pid]['resource_history'].append(resource_list[min(t + 1, len(resource_list) - 1)])

            s_B = extract_internal_state(grid_t, p['cells'])
            s_dB = extract_boundary_obs(grid_t, p['cells'], N)

            all_features[pid]['features'].append({
                's_B': s_B,
                's_dB': s_dB,
            })

    result = analyze_iota(all_features)

    metadata = {
        'snapshot_path': snapshot_path,
        'seed': seed,
        'C': C, 'N': N,
        'n_recording_steps': n_recording_steps,
        'n_patterns_detected': len(initial_pats),
        'n_patterns_analyzed': len([pid for pid in all_features
                                     if len(all_features[pid]['features']) >= 15]),
    }

    return result, metadata


def result_to_dict(result: Optional[IotaResult], metadata: dict,
                   cycle: int = -1) -> dict:
    if result is None:
        return {
            'metadata': metadata,
            'mean_iota': None,
            'std_iota': None,
            'mean_MI_social': None,
            'mean_MI_trajectory': None,
            'mean_MI_resource': None,
            'iota_range': None,
            'animism_score': None,
            'n_patterns': 0,
            'cycle': cycle,
        }
    return {
        'metadata': metadata,
        'n_patterns': result.n_patterns,
        'mean_iota': result.mean_iota,
        'std_iota': result.std_iota,
        'mean_MI_social': result.mean_MI_social,
        'mean_MI_trajectory': result.mean_MI_trajectory,
        'mean_MI_resource': result.mean_MI_resource,
        'iota_range': list(result.iota_range),
        'animism_score': result.animism_score,
        'per_pattern': result.per_pattern[:10],
        'cycle': cycle,
    }
