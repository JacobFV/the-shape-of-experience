"""Experiment 5: Counterfactual Detachment — Measurement Module.

Measures whether patterns decouple from external driving and run
"offline" internal simulations. Key quantities:

    ρ_sync(t): correlation between internal state change and boundary input
               High = reactive (sensory driven), Low = detached (internally driven)

    CF:        counterfactual simulation score — are detachment events
               more predictive of future than reactive episodes?

    I_img:     imagination capacity — mean CF across detachment events

    H_branch:  branch entropy during detachment — does the pattern explore
               multiple futures?
"""

import warnings
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy import stats


@dataclass
class CounterfactualResult:
    """Counterfactual detachment metrics for a single pattern."""
    pattern_id: int
    n_timesteps: int
    mean_rho_sync: float        # mean external synchrony
    std_rho_sync: float
    n_detachment_events: int    # episodes where ρ_sync < threshold
    detachment_fraction: float  # fraction of time in detached mode
    I_img: float                # imagination capacity (mean CF)
    H_branch: float             # branch entropy during detachment
    rho_sync_trajectory: list   # full ρ_sync time series


def compute_synchrony(delta_sB: np.ndarray, s_dB: np.ndarray) -> float:
    """Compute external synchrony between internal change and boundary.

    ρ_sync = Cov(Δs_B, s_∂B) / √(Var(Δs_B) · Var(s_∂B))

    Uses magnitude of vectors (scalar synchrony) to get a single number.
    """
    mag_delta = np.linalg.norm(delta_sB)
    mag_boundary = np.linalg.norm(s_dB)

    if mag_delta < 1e-10 or mag_boundary < 1e-10:
        return 0.0

    # Cosine similarity between internal change direction and boundary state
    return float(np.dot(delta_sB, s_dB[:len(delta_sB)]) /
                 (mag_delta * mag_boundary + 1e-10))


def detect_detachment_events(rho_sync: np.ndarray,
                             threshold: float = 0.3,
                             min_duration: int = 3) -> List[Tuple[int, int]]:
    """Find episodes where ρ_sync stays below threshold.

    Returns list of (start, end) index pairs.
    """
    below = rho_sync < threshold
    events = []
    start = None

    for i in range(len(below)):
        if below[i] and start is None:
            start = i
        elif not below[i] and start is not None:
            if i - start >= min_duration:
                events.append((start, i))
            start = None

    if start is not None and len(below) - start >= min_duration:
        events.append((start, len(below)))

    return events


def compute_counterfactual_score(features_list: List[dict],
                                 detachment_events: List[Tuple[int, int]],
                                 reactive_indices: np.ndarray,
                                 tau: int = 5) -> float:
    """Compute CF: is detachment-mode more predictive than reactive-mode?

    CF = mean predictive accuracy during detachment - during reactive mode.
    Uses simple autocorrelation of internal state as prediction proxy.
    """
    if not detachment_events or len(reactive_indices) < 5:
        return 0.0

    # Predictive accuracy = correlation between s_B(t) and s_env(t+tau)
    def predictive_score(indices):
        scores = []
        for i in indices:
            if i + tau >= len(features_list):
                continue
            s_B = features_list[i]['s_B']
            # Future environment from tau steps ahead
            s_env_future = features_list[i + tau].get('s_env_raw',
                                                       features_list[i + tau].get('s_dB', s_B))
            # Use correlation as prediction proxy
            min_len = min(len(s_B), len(s_env_future))
            if min_len < 2:
                continue
            corr = np.corrcoef(s_B[:min_len], s_env_future[:min_len])[0, 1]
            if np.isfinite(corr):
                scores.append(abs(corr))
        return np.mean(scores) if scores else 0.0

    # Detachment indices
    det_indices = []
    for start, end in detachment_events:
        det_indices.extend(range(start, end))
    det_indices = np.array(det_indices)

    if len(det_indices) < 3:
        return 0.0

    score_detached = predictive_score(det_indices)
    score_reactive = predictive_score(reactive_indices[:len(det_indices)])

    return float(score_detached - score_reactive)


def compute_branch_entropy(features_list: List[dict],
                           detachment_events: List[Tuple[int, int]]) -> float:
    """Entropy of internal state trajectories during detachment.

    High H_branch + positive CF = exploring multiple informative futures.
    """
    if not detachment_events:
        return 0.0

    entropies = []
    for start, end in detachment_events:
        if end - start < 3:
            continue
        # Collect internal state changes during detachment
        deltas = []
        for i in range(start + 1, min(end, len(features_list))):
            delta = features_list[i]['s_B'] - features_list[i - 1]['s_B']
            deltas.append(delta)

        if len(deltas) < 2:
            continue

        deltas = np.array(deltas, dtype=np.float64)
        # Entropy proxy: effective dimensionality of the trajectory
        cov = np.cov(deltas.T) if deltas.shape[0] > deltas.shape[1] else np.cov(deltas)
        if cov.ndim < 2:
            continue
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.maximum(eigvals, 0)
        total = eigvals.sum()
        if total < 1e-20:
            continue
        p = eigvals / total
        p = p[p > 1e-10]
        entropy = -np.sum(p * np.log(p))
        entropies.append(entropy)

    return float(np.mean(entropies)) if entropies else 0.0


def analyze_counterfactual(features_list: List[dict],
                           detach_threshold: float = 0.3,
                           min_detach_duration: int = 3,
                           prediction_tau: int = 5
                           ) -> Optional[CounterfactualResult]:
    """Full counterfactual analysis for a single pattern's trajectory."""
    if len(features_list) < 15:
        return None

    # Compute ρ_sync at each timestep
    rho_sync = []
    for i in range(1, len(features_list)):
        delta_sB = features_list[i]['s_B'] - features_list[i - 1]['s_B']
        s_dB = features_list[i]['s_dB']

        # Use the boundary-length portion of delta for correlation
        min_len = min(len(delta_sB), len(s_dB))
        rho = compute_synchrony(delta_sB[:min_len], s_dB[:min_len])
        rho_sync.append(rho)

    rho_sync = np.array(rho_sync)

    # Detect detachment events
    events = detect_detachment_events(rho_sync, detach_threshold, min_detach_duration)

    # Identify reactive indices (not in any detachment event)
    detached_set = set()
    for start, end in events:
        detached_set.update(range(start, end))
    reactive_indices = np.array([i for i in range(len(rho_sync))
                                 if i not in detached_set])

    # Counterfactual simulation score
    CF = compute_counterfactual_score(
        features_list, events, reactive_indices, tau=prediction_tau)

    # Branch entropy
    H_branch = compute_branch_entropy(features_list, events)

    det_frac = len(detached_set) / max(len(rho_sync), 1)

    return CounterfactualResult(
        pattern_id=-1,
        n_timesteps=len(features_list),
        mean_rho_sync=float(np.mean(rho_sync)),
        std_rho_sync=float(np.std(rho_sync)),
        n_detachment_events=len(events),
        detachment_fraction=det_frac,
        I_img=CF,
        H_branch=H_branch,
        rho_sync_trajectory=rho_sync.tolist(),
    )


# ============================================================================
# Full pipeline
# ============================================================================

def measure_counterfactual_from_snapshot(
    snapshot_path: str,
    seed: int,
    config_overrides: Optional[dict] = None,
    n_recording_steps: int = 50,
    substrate_steps_per_record: int = 10,
    threshold: float = 0.15,
    max_patterns: int = 20,
    detach_threshold: float = 0.3,
) -> Tuple[List[CounterfactualResult], dict]:
    """Load snapshot, run recording, compute counterfactual metrics."""
    import jax.numpy as jnp
    from jax import random
    from v13_substrate import generate_v13_config, init_v13, run_v13_chunk
    from v13_world_model import (
        detect_patterns_for_wm, extract_internal_state,
        extract_boundary_obs, extract_environment,
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

    # Run forward
    grids = [grid_np.copy()]
    g = jnp.array(grid_np)
    r = jnp.array(resource_np)

    for _ in range(n_recording_steps):
        g, r, rng = run_v13_chunk(
            g, r, h_embed, kernel_ffts, config,
            coupling, coupling_row_sums, rng,
            n_steps=substrate_steps_per_record, box_fft=box_fft)
        grids.append(np.array(g))

    # Track patterns
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
            s_env_raw = extract_environment(grid_t, p['center'], N)

            all_features[pid]['features'].append({
                's_B': s_B,
                's_dB': s_dB,
                's_env_raw': s_env_raw,
            })

    # Analyze
    results = []
    sorted_pids = sorted(all_features.keys(),
                         key=lambda pid: len(all_features[pid]['features']),
                         reverse=True)

    for pid in sorted_pids[:max_patterns]:
        feat_list = all_features[pid]['features']
        cf = analyze_counterfactual(feat_list, detach_threshold=detach_threshold)
        if cf is not None:
            cf.pattern_id = pid
            results.append(cf)

    metadata = {
        'snapshot_path': snapshot_path,
        'seed': seed,
        'C': C, 'N': N,
        'n_recording_steps': n_recording_steps,
        'detach_threshold': detach_threshold,
        'n_patterns_detected': len(initial_pats),
        'n_patterns_analyzed': len(results),
    }

    return results, metadata


def results_to_dict(results: List[CounterfactualResult], metadata: dict) -> dict:
    return {
        'metadata': metadata,
        'patterns': [
            {
                'pattern_id': r.pattern_id,
                'n_timesteps': r.n_timesteps,
                'mean_rho_sync': r.mean_rho_sync,
                'std_rho_sync': r.std_rho_sync,
                'n_detachment_events': r.n_detachment_events,
                'detachment_fraction': r.detachment_fraction,
                'I_img': r.I_img,
                'H_branch': r.H_branch,
            }
            for r in results
        ],
        'summary': {
            'mean_rho_sync': float(np.mean([r.mean_rho_sync for r in results])) if results else 0,
            'mean_detach_frac': float(np.mean([r.detachment_fraction for r in results])) if results else 0,
            'mean_I_img': float(np.mean([r.I_img for r in results])) if results else 0,
            'mean_H_branch': float(np.mean([r.H_branch for r in results])) if results else 0,
            'n_with_detachment': sum(1 for r in results if r.n_detachment_events > 0),
            'n_with_imagination': sum(1 for r in results if r.I_img > 0),
        },
    }
