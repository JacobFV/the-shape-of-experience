"""Experiment 9 (adapted): Proto-Normativity — Observational Version.

Since V13 lacks controlled intervention, we measure the OBSERVATIONAL
correlate: do patterns' internal affect states naturally differ based
on social context?

At each timestep, classify each pattern's context:
    - Isolated: no other pattern within radius R
    - Social-cooperative: neighbor(s) present AND both growing
    - Social-competitive: neighbor(s) present AND pattern growing while
      neighbor shrinking (or vice versa)

Then compare affect metrics across conditions:
    ΔΦ = Φ(cooperative) - Φ(competitive)
    ΔV = V(cooperative) - V(competitive)  [valence = mass change]
    ΔA = A(cooperative) - A(competitive)  [arousal = state change rate]

If ΔΦ > 0: integration is higher during cooperation → proto-normativity
If ΔV > 0: viability gradient favors cooperation → exploitation is
           penalized by the affect system even when locally rewarding
"""

import warnings
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy import stats


@dataclass
class NormativityResult:
    """Proto-normativity metrics for a snapshot."""
    n_patterns: int
    n_social_steps: int         # timesteps with social context
    n_isolated_steps: int       # timesteps isolated
    n_cooperative: int          # social-cooperative events
    n_competitive: int          # social-competitive events

    # Mean affect in each condition
    phi_cooperative: float      # mean Φ during cooperation
    phi_competitive: float      # mean Φ during competition
    phi_isolated: float         # mean Φ when isolated
    delta_phi: float            # Φ(coop) - Φ(comp)
    delta_phi_p: float          # p-value (Mann-Whitney)

    valence_cooperative: float  # mean valence (mass change) during coop
    valence_competitive: float
    delta_valence: float
    delta_valence_p: float

    arousal_cooperative: float  # mean arousal (state change rate)
    arousal_competitive: float
    delta_arousal: float
    delta_arousal_p: float


def compute_phi_simple(s_B: np.ndarray, n_parts: int = 2) -> float:
    """Simple Φ estimate: variance explained by full vs partitioned representation.

    Split internal state into n_parts blocks, measure information lost.
    """
    d = len(s_B)
    if d < 4:
        return 0.0

    # Full variance
    full_var = float(np.var(s_B))
    if full_var < 1e-12:
        return 0.0

    # Partitioned: split into blocks, sum block variances
    block_size = d // n_parts
    part_var = 0.0
    for i in range(n_parts):
        block = s_B[i * block_size: (i + 1) * block_size]
        part_var += np.var(block)
    part_var /= n_parts

    # Φ proxy: how much variance is lost by partitioning
    return float(max(0, full_var - part_var))


def classify_social_context(
    pid: int,
    t: int,
    all_features: dict,
    N: int,
    social_radius: float = 30.0,
) -> Tuple[str, List[int]]:
    """Classify social context: 'isolated', 'cooperative', or 'competitive'.

    Returns (context_type, neighbor_pids).
    """
    if t >= len(all_features[pid]['center_history']):
        return 'isolated', []

    center_i = all_features[pid]['center_history'][t]
    neighbors = []

    for other_pid in all_features:
        if other_pid == pid:
            continue
        if t >= len(all_features[other_pid]['center_history']):
            continue
        center_j = all_features[other_pid]['center_history'][t]

        dr = abs(center_i[0] - center_j[0])
        dc = abs(center_i[1] - center_j[1])
        dr = min(dr, N - dr)
        dc = min(dc, N - dc)
        dist = np.sqrt(dr ** 2 + dc ** 2)

        if dist < social_radius:
            neighbors.append(other_pid)

    if not neighbors:
        return 'isolated', []

    # Determine cooperative vs competitive from mass changes
    # Need features at t and t-1
    feat_idx = t  # features list index
    if feat_idx < 1 or feat_idx >= len(all_features[pid]['features']):
        return 'social', neighbors  # can't determine direction

    s_B_curr = all_features[pid]['features'][feat_idx]['s_B']
    s_B_prev = all_features[pid]['features'][feat_idx - 1]['s_B']

    # Mass proxy: sum of channel means (first C values of s_B)
    C_ch = len(s_B_curr) // 4  # 4C + 4 dimensional
    mass_change_i = float(s_B_curr[:C_ch].sum() - s_B_prev[:C_ch].sum())

    # Average neighbor mass change
    neighbor_mass_changes = []
    for npid in neighbors:
        if feat_idx < len(all_features[npid]['features']) and feat_idx >= 1:
            s_curr = all_features[npid]['features'][feat_idx]['s_B']
            s_prev = all_features[npid]['features'][feat_idx - 1]['s_B']
            mc = float(s_curr[:C_ch].sum() - s_prev[:C_ch].sum())
            neighbor_mass_changes.append(mc)

    if not neighbor_mass_changes:
        return 'social', neighbors

    mean_neighbor_change = float(np.mean(neighbor_mass_changes))

    # Cooperative: both positive or both negative (moving together)
    # Competitive: opposite directions (one gains, other loses)
    if mass_change_i * mean_neighbor_change > 0:
        return 'cooperative', neighbors
    elif mass_change_i > 0 and mean_neighbor_change < 0:
        return 'competitive', neighbors  # pattern exploiting neighbors
    elif mass_change_i < 0 and mean_neighbor_change > 0:
        return 'competitive', neighbors  # pattern being exploited
    else:
        return 'social', neighbors  # ambiguous (both zero)


def analyze_normativity(all_features: dict, N: int = 128) -> Optional[NormativityResult]:
    """Compute proto-normativity metrics for a snapshot's patterns."""
    pids = sorted(all_features.keys())
    valid_pids = [pid for pid in pids if len(all_features[pid]['features']) >= 15]

    if len(valid_pids) < 3:
        return None

    # Collect affect metrics per condition
    phi_coop, phi_comp, phi_iso = [], [], []
    val_coop, val_comp = [], []
    aro_coop, aro_comp = [], []
    n_cooperative = 0
    n_competitive = 0
    n_social = 0
    n_isolated = 0

    for pid in valid_pids:
        feats = all_features[pid]['features']
        n_steps = len(feats)

        for t in range(1, n_steps):
            context, neighbors = classify_social_context(
                pid, t, all_features, N)

            s_B = feats[t]['s_B']
            s_B_prev = feats[t - 1]['s_B']

            # Φ
            phi = compute_phi_simple(s_B)

            # Valence: mass change
            C_ch = len(s_B) // 4
            valence = float(s_B[:C_ch].sum() - s_B_prev[:C_ch].sum())

            # Arousal: state change rate
            arousal = float(np.linalg.norm(s_B - s_B_prev))

            if context == 'cooperative':
                phi_coop.append(phi)
                val_coop.append(valence)
                aro_coop.append(arousal)
                n_cooperative += 1
            elif context == 'competitive':
                phi_comp.append(phi)
                val_comp.append(valence)
                aro_comp.append(arousal)
                n_competitive += 1
            elif context == 'isolated':
                phi_iso.append(phi)
                n_isolated += 1
            else:
                n_social += 1

    if len(phi_coop) < 10 or len(phi_comp) < 10:
        return None

    # Statistical tests (Mann-Whitney)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')

        _, delta_phi_p = stats.mannwhitneyu(
            phi_coop, phi_comp, alternative='two-sided')
        _, delta_val_p = stats.mannwhitneyu(
            val_coop, val_comp, alternative='two-sided')
        _, delta_aro_p = stats.mannwhitneyu(
            aro_coop, aro_comp, alternative='two-sided')

    return NormativityResult(
        n_patterns=len(valid_pids),
        n_social_steps=n_social + n_cooperative + n_competitive,
        n_isolated_steps=n_isolated,
        n_cooperative=n_cooperative,
        n_competitive=n_competitive,
        phi_cooperative=float(np.mean(phi_coop)),
        phi_competitive=float(np.mean(phi_comp)),
        phi_isolated=float(np.mean(phi_iso)) if phi_iso else 0.0,
        delta_phi=float(np.mean(phi_coop) - np.mean(phi_comp)),
        delta_phi_p=float(delta_phi_p),
        valence_cooperative=float(np.mean(val_coop)),
        valence_competitive=float(np.mean(val_comp)),
        delta_valence=float(np.mean(val_coop) - np.mean(val_comp)),
        delta_valence_p=float(delta_val_p),
        arousal_cooperative=float(np.mean(aro_coop)),
        arousal_competitive=float(np.mean(aro_comp)),
        delta_arousal=float(np.mean(aro_coop) - np.mean(aro_comp)),
        delta_arousal_p=float(delta_aro_p),
    )


# ============================================================================
# Full pipeline
# ============================================================================

def measure_normativity_from_snapshot(
    snapshot_path: str,
    seed: int,
    config_overrides: Optional[dict] = None,
    n_recording_steps: int = 50,
    substrate_steps_per_record: int = 10,
    threshold: float = 0.15,
    max_patterns: int = 20,
) -> Tuple[Optional[NormativityResult], dict]:
    """Load snapshot, run recording, compute normativity metrics."""
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
    rng = random.PRNGKey(seed + 7777)

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

    result = analyze_normativity(all_features, N=N)

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


def result_to_dict(result: Optional[NormativityResult], metadata: dict,
                   cycle: int = -1) -> dict:
    if result is None:
        return {
            'metadata': metadata,
            'n_patterns': 0,
            'n_cooperative': None,
            'n_competitive': None,
            'delta_phi': None,
            'delta_phi_p': None,
            'delta_valence': None,
            'delta_valence_p': None,
            'delta_arousal': None,
            'delta_arousal_p': None,
            'cycle': cycle,
        }
    return {
        'metadata': metadata,
        'n_patterns': result.n_patterns,
        'n_social_steps': result.n_social_steps,
        'n_isolated_steps': result.n_isolated_steps,
        'n_cooperative': result.n_cooperative,
        'n_competitive': result.n_competitive,
        'phi_cooperative': result.phi_cooperative,
        'phi_competitive': result.phi_competitive,
        'phi_isolated': result.phi_isolated,
        'delta_phi': result.delta_phi,
        'delta_phi_p': result.delta_phi_p,
        'valence_cooperative': result.valence_cooperative,
        'valence_competitive': result.valence_competitive,
        'delta_valence': result.delta_valence,
        'delta_valence_p': result.delta_valence_p,
        'arousal_cooperative': result.arousal_cooperative,
        'arousal_competitive': result.arousal_competitive,
        'delta_arousal': result.delta_arousal,
        'delta_arousal_p': result.delta_arousal_p,
        'cycle': cycle,
    }
