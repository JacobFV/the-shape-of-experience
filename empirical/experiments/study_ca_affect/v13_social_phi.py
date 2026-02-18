"""Experiment 10 (adapted): Social-Scale Integration.

Measures whether populations of interacting patterns develop collective
integration exceeding the sum of individual integrations.

Key quantity — the superorganism test:
    Φ_G > Σᵢ Φᵢ ?

If collective integration exceeds sum of parts, information exists at
the group level that doesn't exist in any individual.

Method:
    1. For each pattern, compute Φᵢ (individual integration)
    2. Compute Φ_G (group integration): prediction loss under
       population partition vs full population
    3. Superorganism ratio: Φ_G / Σ Φᵢ (>1 = synergistic)

Also measures:
    - Φ_G_stress: collective Φ under environmental perturbation
    - Group robustness: Φ_G_stress / Φ_G (>1 = group integrates under threat)
"""

import warnings
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score


@dataclass
class SocialPhiResult:
    """Social-scale integration metrics for a snapshot."""
    n_patterns: int
    n_timesteps: int

    # Individual integration
    mean_phi_individual: float    # mean Φᵢ across patterns
    sum_phi_individual: float     # Σ Φᵢ

    # Group integration
    phi_group: float              # Φ_G (collective integration)
    superorganism_ratio: float    # Φ_G / Σ Φᵢ

    # Group integration under subgroup partition
    phi_subgroup_A: float         # Φ of subgroup A
    phi_subgroup_B: float         # Φ of subgroup B
    phi_partition_sum: float      # Φ_A + Φ_B
    partition_loss: float         # Φ_G - (Φ_A + Φ_B) — info lost by partitioning

    # Group MI (social coupling strength)
    mean_pairwise_MI: float       # mean MI between all pattern pairs
    group_coherence: float        # fraction of pairs with significant MI


def compute_phi_individual(features_list: List[dict], n_parts: int = 2) -> float:
    """Compute mean Φ for a single pattern from its feature time series.

    Φ = variance(full) - mean(variance(partitions))
    """
    if len(features_list) < 10:
        return 0.0

    phis = []
    for f in features_list:
        s_B = f['s_B']
        d = len(s_B)
        if d < 4:
            continue

        full_var = float(np.var(s_B))
        if full_var < 1e-12:
            phis.append(0.0)
            continue

        block_size = d // n_parts
        part_var = 0.0
        for i in range(n_parts):
            block = s_B[i * block_size: (i + 1) * block_size]
            part_var += np.var(block)
        part_var /= n_parts

        phis.append(max(0, full_var - part_var))

    return float(np.mean(phis)) if phis else 0.0


def compute_group_phi_from_mi(pairwise_mis: List[float],
                               n_patterns: int) -> float:
    """Compute group-level Φ from pairwise MI values.

    Φ_G = sum of cross-pattern MI.
    Normalized by number of patterns for comparability.
    """
    if not pairwise_mis or n_patterns < 2:
        return 0.0
    return float(np.sum(pairwise_mis))


def compute_subgroup_phi(mi_A: List[float], mi_B: List[float],
                          n_A: int, n_B: int) -> Tuple[float, float]:
    """Compute Φ for each spatial subgroup from their pairwise MIs."""
    phi_A = compute_group_phi_from_mi(mi_A, n_A)
    phi_B = compute_group_phi_from_mi(mi_B, n_B)
    return phi_A, phi_B


def compute_pairwise_mi(s_i: np.ndarray, s_j: np.ndarray) -> float:
    """Gaussian MI between two patterns' state time series."""
    n = min(len(s_i), len(s_j))
    if n < 15:
        return 0.0

    d = min(8, s_i.shape[1], s_j.shape[1])
    mi_total = 0.0
    count = 0

    for dim in range(d):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            try:
                r, _ = stats.pearsonr(s_i[:n, dim], s_j[:n, dim])
            except Exception:
                continue
        if np.isfinite(r):
            r2 = np.clip(r ** 2, 0, 0.999)
            mi_total += -0.5 * np.log(1 - r2)
            count += 1

    return float(mi_total / max(count, 1))


def analyze_social_phi(all_features: dict) -> Optional[SocialPhiResult]:
    """Compute social-scale integration metrics."""
    pids = sorted(all_features.keys())
    valid_pids = [pid for pid in pids if len(all_features[pid]['features']) >= 15]

    if len(valid_pids) < 4:
        return None

    # Find common timestep count
    min_steps = min(len(all_features[pid]['features']) for pid in valid_pids)
    n_steps = min(min_steps, 50)

    # 1. Individual Φ
    individual_phis = []
    for pid in valid_pids:
        phi_i = compute_phi_individual(all_features[pid]['features'][:n_steps])
        individual_phis.append(phi_i)

    mean_phi_ind = float(np.mean(individual_phis))
    sum_phi_ind = float(np.sum(individual_phis))

    # 2. Build concatenated state matrix (T, D_total)
    # Use first 8 dims of s_B per pattern for tractability
    dims_per = 8
    state_matrix = np.zeros((n_steps, len(valid_pids) * dims_per))
    for i, pid in enumerate(valid_pids):
        feats = all_features[pid]['features']
        for t in range(n_steps):
            s_B = feats[t]['s_B'][:dims_per]
            state_matrix[t, i * dims_per: (i + 1) * dims_per] = s_B

    # 3. Pairwise MI (compute first, needed for group Φ)
    internal_series = {}
    for pid in valid_pids:
        feats = all_features[pid]['features']
        internal_series[pid] = np.array(
            [f['s_B'][:dims_per] for f in feats[:n_steps]], dtype=np.float64)

    # Track which pair belongs to which pattern indices
    mi_values = []
    mi_pairs = []  # (i, j) indices
    sig_count = 0
    for i, pid_i in enumerate(valid_pids):
        for j, pid_j in enumerate(valid_pids):
            if i >= j:
                continue
            mi = compute_pairwise_mi(internal_series[pid_i],
                                      internal_series[pid_j])
            mi_values.append(mi)
            mi_pairs.append((i, j))
            if mi > 0.02:  # rough significance threshold
                sig_count += 1

    n_pairs = len(mi_values)
    mean_mi = float(np.mean(mi_values)) if mi_values else 0.0
    coherence = sig_count / max(n_pairs, 1)

    # Group Φ = total cross-pattern MI
    n_pat = len(valid_pids)
    phi_group = compute_group_phi_from_mi(mi_values, n_pat)

    # 4. Subgroup partition (spatial: split by median x-coordinate)
    centers = []
    for pid in valid_pids:
        ch = all_features[pid]['center_history']
        if ch:
            centers.append(ch[0][0])
        else:
            centers.append(0)

    median_x = np.median(centers)
    group_A_set = set(i for i, x in enumerate(centers) if x <= median_x)
    group_B_set = set(i for i, x in enumerate(centers) if x > median_x)

    if len(group_A_set) < 2 or len(group_B_set) < 2:
        half = n_pat // 2
        group_A_set = set(range(half))
        group_B_set = set(range(half, n_pat))

    # MI within each subgroup
    mi_within_A = [mi for mi, (i, j) in zip(mi_values, mi_pairs)
                   if i in group_A_set and j in group_A_set]
    mi_within_B = [mi for mi, (i, j) in zip(mi_values, mi_pairs)
                   if i in group_B_set and j in group_B_set]

    phi_sub_A, phi_sub_B = compute_subgroup_phi(
        mi_within_A, mi_within_B,
        len(group_A_set), len(group_B_set))
    phi_partition_sum = phi_sub_A + phi_sub_B
    partition_loss = phi_group - phi_partition_sum

    # Superorganism ratio: Φ_G / Σ Φ_individual
    super_ratio = phi_group / sum_phi_ind if sum_phi_ind > 1e-10 else 0.0

    return SocialPhiResult(
        n_patterns=len(valid_pids),
        n_timesteps=n_steps,
        mean_phi_individual=mean_phi_ind,
        sum_phi_individual=sum_phi_ind,
        phi_group=phi_group,
        superorganism_ratio=super_ratio,
        phi_subgroup_A=phi_sub_A,
        phi_subgroup_B=phi_sub_B,
        phi_partition_sum=phi_partition_sum,
        partition_loss=partition_loss,
        mean_pairwise_MI=mean_mi,
        group_coherence=coherence,
    )


# ============================================================================
# Full pipeline
# ============================================================================

def measure_social_phi_from_snapshot(
    snapshot_path: str,
    seed: int,
    config_overrides: Optional[dict] = None,
    n_recording_steps: int = 50,
    substrate_steps_per_record: int = 10,
    threshold: float = 0.15,
    max_patterns: int = 20,
) -> Tuple[Optional[SocialPhiResult], dict]:
    """Load snapshot, run recording, compute social Φ metrics."""
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
    rng = random.PRNGKey(seed + 6666)

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

    result = analyze_social_phi(all_features)

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


def result_to_dict(result: Optional[SocialPhiResult], metadata: dict,
                   cycle: int = -1) -> dict:
    if result is None:
        return {
            'metadata': metadata,
            'n_patterns': 0,
            'phi_group': None,
            'sum_phi_individual': None,
            'superorganism_ratio': None,
            'partition_loss': None,
            'mean_pairwise_MI': None,
            'group_coherence': None,
            'cycle': cycle,
        }
    return {
        'metadata': metadata,
        'n_patterns': result.n_patterns,
        'n_timesteps': result.n_timesteps,
        'mean_phi_individual': result.mean_phi_individual,
        'sum_phi_individual': result.sum_phi_individual,
        'phi_group': result.phi_group,
        'superorganism_ratio': result.superorganism_ratio,
        'phi_subgroup_A': result.phi_subgroup_A,
        'phi_subgroup_B': result.phi_subgroup_B,
        'phi_partition_sum': result.phi_partition_sum,
        'partition_loss': result.partition_loss,
        'mean_pairwise_MI': result.mean_pairwise_MI,
        'group_coherence': result.group_coherence,
        'cycle': cycle,
    }
