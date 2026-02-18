"""Experiment 4 (adapted): Proto-Communication via Chemical Coupling.

Measures whether patterns communicate through the shared grid chemistry
in a structured, state-dependent way. Key quantities:

    MI_inter:      Mutual information between pattern i's internal state
                   and pattern j's internal state across time. Positive =
                   patterns share information.

    C_channel:     Chemical channel capacity: MI(emission_i; reception_j)
                   where emission = boundary outflow, reception = boundary
                   at the other pattern. Measures how much of i's chemical
                   output reaches j.

    ρ_topo:        Topographic similarity: correlation between signal
                   distances and context distances. Positive = communication
                   is structured (preserves relationships).

    Vocab size:    Number of distinct emission profile clusters (k-means
                   on boundary emission vectors).
"""

import warnings
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy import stats
from sklearn.cluster import KMeans


@dataclass
class CommunicationResult:
    """Communication metrics for a snapshot."""
    n_patterns: int
    n_pairs: int
    mean_MI_inter: float        # mean inter-pattern MI
    max_MI_inter: float         # max inter-pattern MI
    MI_baseline: float          # shuffled baseline MI
    MI_significant: bool        # MI > baseline + 2σ
    mean_C_channel: float       # mean chemical channel capacity
    rho_topo: float             # topographic similarity
    rho_topo_p: float           # p-value
    vocab_size: int             # emission profile clusters
    per_pair: List[dict]        # per-pair MI values


def compute_inter_pattern_MI(features_i: List[dict],
                              features_j: List[dict],
                              n_bins: int = 8) -> float:
    """Estimate MI between two patterns' internal states across time.

    Uses binned estimator on the first few PCA components.
    """
    n = min(len(features_i), len(features_j))
    if n < 15:
        return 0.0

    # Use first 8 dims of s_B for efficiency
    X_i = np.array([f['s_B'][:8] for f in features_i[:n]], dtype=np.float64)
    X_j = np.array([f['s_B'][:8] for f in features_j[:n]], dtype=np.float64)

    # Simple MI estimate: mean pairwise correlation magnitude
    # (exact MI is intractable at this dimensionality)
    mi_total = 0.0
    count = 0
    for d in range(min(8, X_i.shape[1])):
        r, _ = stats.pearsonr(X_i[:, d], X_j[:, d])
        if np.isfinite(r):
            # MI from Gaussian: MI = -0.5 * log(1 - r^2)
            r_clipped = np.clip(r ** 2, 0, 0.999)
            mi_total += -0.5 * np.log(1 - r_clipped)
            count += 1

    return float(mi_total / max(count, 1))


def compute_channel_capacity(features_i: List[dict],
                              features_j: List[dict]) -> float:
    """Estimate chemical channel capacity between patterns i and j.

    C_channel = MI(s_∂B_i; s_∂B_j) — how much of i's boundary emission
    is readable at j's boundary.
    """
    n = min(len(features_i), len(features_j))
    if n < 15:
        return 0.0

    X_i = np.array([f['s_dB'][:8] for f in features_i[:n]], dtype=np.float64)
    X_j = np.array([f['s_dB'][:8] for f in features_j[:n]], dtype=np.float64)

    mi_total = 0.0
    count = 0
    for d in range(min(8, X_i.shape[1])):
        r, _ = stats.pearsonr(X_i[:, d], X_j[:, d])
        if np.isfinite(r):
            r_clipped = np.clip(r ** 2, 0, 0.999)
            mi_total += -0.5 * np.log(1 - r_clipped)
            count += 1

    return float(mi_total / max(count, 1))


def compute_topographic_similarity(all_features: dict,
                                    n_steps: int = 30) -> Tuple[float, float]:
    """Compute topographic similarity ρ_topo.

    Correlation between signal distances (boundary emission profiles)
    and context distances (internal state). If positive, communication
    is structured — patterns with similar internal states emit similar
    chemical signatures.
    """
    pids = sorted(all_features.keys())
    if len(pids) < 4:
        return 0.0, 1.0

    # Compute mean emission profile and mean internal state for each pattern
    emissions = []
    contexts = []
    valid_pids = []

    for pid in pids:
        feat_list = all_features[pid]['features']
        if len(feat_list) < 10:
            continue
        n = min(len(feat_list), n_steps)
        mean_dB = np.mean([f['s_dB'] for f in feat_list[:n]], axis=0)
        mean_B = np.mean([f['s_B'] for f in feat_list[:n]], axis=0)
        emissions.append(mean_dB)
        contexts.append(mean_B)
        valid_pids.append(pid)

    if len(valid_pids) < 4:
        return 0.0, 1.0

    emissions = np.array(emissions, dtype=np.float64)
    contexts = np.array(contexts, dtype=np.float64)

    # Pairwise distances
    from scipy.spatial.distance import pdist
    d_signal = pdist(emissions, metric='euclidean')
    d_context = pdist(contexts, metric='euclidean')

    if np.std(d_signal) < 1e-10 or np.std(d_context) < 1e-10:
        return 0.0, 1.0

    rho, p = stats.spearmanr(d_signal, d_context)
    return (float(rho) if np.isfinite(rho) else 0.0,
            float(p) if np.isfinite(p) else 1.0)


def estimate_vocab_size(all_features: dict, n_steps: int = 30,
                        max_k: int = 10) -> int:
    """Estimate vocabulary size from emission profile clusters.

    Uses silhouette score to find optimal k for k-means.
    """
    pids = sorted(all_features.keys())

    emissions = []
    for pid in pids:
        feat_list = all_features[pid]['features']
        if len(feat_list) < 5:
            continue
        n = min(len(feat_list), n_steps)
        for f in feat_list[:n]:
            emissions.append(f['s_dB'][:16])  # first 16 dims

    if len(emissions) < max_k + 1:
        return 1

    X = np.array(emissions, dtype=np.float64)
    # Standardize
    stds = X.std(axis=0)
    stds[stds < 1e-12] = 1.0
    X = (X - X.mean(axis=0)) / stds

    from sklearn.metrics import silhouette_score

    best_k = 1
    best_score = -1

    for k in range(2, min(max_k + 1, len(X))):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                km = KMeans(n_clusters=k, n_init=3, max_iter=50, random_state=42)
                labels = km.fit_predict(X)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X, labels, sample_size=min(500, len(X)))
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue

    return best_k


def analyze_communication(all_features: dict,
                           n_shuffles: int = 100) -> Optional[CommunicationResult]:
    """Full communication analysis for a snapshot's patterns."""
    pids = sorted(all_features.keys())
    valid_pids = [pid for pid in pids if len(all_features[pid]['features']) >= 15]

    if len(valid_pids) < 3:
        return None

    # Compute pairwise MI
    mi_values = []
    cc_values = []
    per_pair = []

    for i, pid_i in enumerate(valid_pids):
        for j, pid_j in enumerate(valid_pids):
            if i >= j:
                continue
            feat_i = all_features[pid_i]['features']
            feat_j = all_features[pid_j]['features']

            mi = compute_inter_pattern_MI(feat_i, feat_j)
            cc = compute_channel_capacity(feat_i, feat_j)

            mi_values.append(mi)
            cc_values.append(cc)
            per_pair.append({
                'pattern_i': pid_i, 'pattern_j': pid_j,
                'MI_inter': mi, 'C_channel': cc,
            })

    # Shuffled baseline for MI
    shuffle_mis = []
    for _ in range(min(n_shuffles, 50)):
        # Shuffle temporal alignment between random pair
        i_idx = np.random.randint(len(valid_pids))
        j_idx = np.random.randint(len(valid_pids))
        if i_idx == j_idx:
            continue
        feat_i = all_features[valid_pids[i_idx]]['features']
        feat_j = all_features[valid_pids[j_idx]]['features']
        n = min(len(feat_i), len(feat_j))
        if n < 15:
            continue
        # Shuffle j's temporal order
        j_shuffled = [feat_j[k] for k in np.random.permutation(n)]
        mi_shuf = compute_inter_pattern_MI(feat_i, j_shuffled)
        shuffle_mis.append(mi_shuf)

    mi_baseline = float(np.mean(shuffle_mis)) if shuffle_mis else 0.0
    mi_std = float(np.std(shuffle_mis)) if shuffle_mis else 0.0
    mi_significant = float(np.mean(mi_values)) > mi_baseline + 2 * mi_std

    # Topographic similarity
    rho_topo, rho_topo_p = compute_topographic_similarity(all_features)

    # Vocabulary size
    vocab = estimate_vocab_size(all_features)

    return CommunicationResult(
        n_patterns=len(valid_pids),
        n_pairs=len(per_pair),
        mean_MI_inter=float(np.mean(mi_values)) if mi_values else 0.0,
        max_MI_inter=float(np.max(mi_values)) if mi_values else 0.0,
        MI_baseline=mi_baseline,
        MI_significant=mi_significant,
        mean_C_channel=float(np.mean(cc_values)) if cc_values else 0.0,
        rho_topo=rho_topo,
        rho_topo_p=rho_topo_p,
        vocab_size=vocab,
        per_pair=per_pair,
    )


# ============================================================================
# Full pipeline
# ============================================================================

def measure_communication_from_snapshot(
    snapshot_path: str,
    seed: int,
    config_overrides: Optional[dict] = None,
    n_recording_steps: int = 50,
    substrate_steps_per_record: int = 10,
    threshold: float = 0.15,
    max_patterns: int = 20,
) -> Tuple[Optional[CommunicationResult], dict]:
    """Load snapshot, run recording, compute communication metrics."""
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

    grids = [grid_np.copy()]
    g = jnp.array(grid_np)
    r = jnp.array(resource_np)

    for _ in range(n_recording_steps):
        g, r, rng = run_v13_chunk(
            g, r, h_embed, kernel_ffts, config,
            coupling, coupling_row_sums, rng,
            n_steps=substrate_steps_per_record, box_fft=box_fft)
        grids.append(np.array(g))

    initial_pats = detect_patterns_for_wm(grids[0], threshold=threshold)
    if not initial_pats:
        return None, {'n_patterns_detected': 0}

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

            all_features[pid]['features'].append({
                's_B': s_B,
                's_dB': s_dB,
            })

    result = analyze_communication(all_features)

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


def result_to_dict(result: Optional[CommunicationResult], metadata: dict,
                   cycle: int = -1) -> dict:
    if result is None:
        return {
            'metadata': metadata,
            'mean_MI_inter': None,
            'max_MI_inter': None,
            'MI_baseline': None,
            'MI_significant': False,
            'mean_C_channel': None,
            'rho_topo': None,
            'rho_topo_p': None,
            'vocab_size': None,
            'n_patterns': 0,
            'n_pairs': 0,
            'cycle': cycle,
        }
    return {
        'metadata': metadata,
        'n_patterns': result.n_patterns,
        'n_pairs': result.n_pairs,
        'mean_MI_inter': result.mean_MI_inter,
        'max_MI_inter': result.max_MI_inter,
        'MI_baseline': result.MI_baseline,
        'MI_significant': result.MI_significant,
        'mean_C_channel': result.mean_C_channel,
        'rho_topo': result.rho_topo,
        'rho_topo_p': result.rho_topo_p,
        'vocab_size': result.vocab_size,
        'per_pair': result.per_pair[:10],  # first 10 pairs only
        'cycle': cycle,
    }
