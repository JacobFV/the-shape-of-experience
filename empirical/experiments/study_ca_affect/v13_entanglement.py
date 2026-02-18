"""Experiment 11: Entanglement Analysis.

Meta-analysis correlating measures across all experiments (1-10) at each
evolutionary generation. Tests whether cognitive capacities co-emerge
or are separable phase transitions.

Loads cross-seed summary JSONs from all prior experiments and computes:
1. Correlation matrix R(g) across measures at each generation
2. Co-emergence test: do world model, abstraction, and imagination correlate?
3. Language lag test: does ρ_topo lag other measures?
4. Self-model phase transition: does SM correlate with Φ jumps?
5. Overall entanglement score: mean |r| across all measure pairs
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats


# Canonical cycles across all experiments
CYCLES = [0, 5, 10, 15, 20, 25, 29]

# Measures to correlate (symbol, source file prefix, key)
MEASURES = [
    ('robustness', 'v13_progress', 'mean_robustness'),
    ('phi_increase', 'v13_progress', 'phi_increase_frac'),
    ('n_patterns', 'v13_progress', 'n_patterns'),
    ('C_wm', 'wm', 'mean_C_wm'),
    ('H_wm', 'wm', 'mean_H_wm'),
    ('d_eff', 'rep', 'mean_d_eff'),
    ('A_level', 'rep', 'mean_A'),
    ('D_disentangle', 'rep', 'mean_D'),
    ('K_comp', 'rep', 'mean_K_comp'),
    ('rho_sync', 'cf', 'mean_rho_sync'),
    ('detach_frac', 'cf', 'mean_detach_frac'),
    ('I_img', 'cf', 'mean_I_img'),
    ('rho_self', 'sm', 'mean_rho_self'),
    ('SM_capacity', 'sm', 'mean_SM_capacity'),
    ('SM_sal', 'sm', 'mean_SM_sal'),
    ('MI_inter', 'comm', 'mean_MI_inter'),
    ('rho_topo', 'comm', 'rho_topo'),
    ('rsa_rho', 'ag', 'rsa_rho'),
    ('iota', 'iota', 'mean_iota'),
    ('MI_social', 'iota', 'mean_MI_social'),
    ('animism', 'iota', 'animism_score'),
    ('phi_group', 'social_phi', 'phi_group'),
    ('super_ratio', 'social_phi', 'superorganism_ratio'),
    ('group_coherence', 'social_phi', 'group_coherence'),
]


def load_cross_seed_data(results_dir: str) -> Dict[str, Dict]:
    """Load all cross-seed summary JSONs."""
    rd = Path(results_dir)
    data = {}

    # Experiment-specific cross-seed files
    for prefix in ['wm', 'rep', 'cf', 'sm', 'comm', 'ag', 'iota',
                    'norm', 'social_phi']:
        analysis_dir = rd / f'{prefix}_analysis'
        cs_file = analysis_dir / f'{prefix}_cross_seed.json'
        if cs_file.exists():
            with open(cs_file) as f:
                data[prefix] = json.load(f)

    # V13 progress files (robustness data)
    progress_data = {'seeds': [], 'trajectories': {}}
    for seed_dir in sorted(rd.glob('v13_s*')):
        if not seed_dir.is_dir():
            continue
        # Handle nested directory structure
        for sub in seed_dir.iterdir():
            if sub.is_dir():
                prog_file = sub / 'v13_progress.json'
                if prog_file.exists():
                    # Extract seed from directory name
                    name = seed_dir.name  # e.g. v13_s7
                    seed_str = name.split('_s')[1].split('_')[0]
                    try:
                        seed = int(seed_str)
                    except ValueError:
                        continue
                    with open(prog_file) as f:
                        prog = json.load(f)
                    # Convert to trajectory format
                    traj = []
                    for cs in prog.get('cycle_stats', []):
                        cycle = cs['cycle']
                        if cycle in CYCLES:
                            traj.append({
                                'cycle': cycle,
                                'mean_robustness': cs.get('mean_robustness'),
                                'phi_increase_frac': cs.get('phi_increase_frac'),
                                'n_patterns': cs.get('n_patterns'),
                            })
                    if traj:
                        progress_data['seeds'].append(seed)
                        progress_data['trajectories'][str(seed)] = traj

    data['v13_progress'] = progress_data
    return data


def build_measure_matrix(data: Dict[str, Dict],
                          seeds: List[int] = [123, 42, 7]
                          ) -> Tuple[np.ndarray, List[str], List[int]]:
    """Build (N_observations, N_measures) matrix.

    Each observation = one (seed, cycle) pair.
    Returns matrix, measure names, and cycle labels.
    """
    measure_names = [m[0] for m in MEASURES]
    n_measures = len(measure_names)

    rows = []
    cycle_labels = []
    seed_labels = []

    for seed in seeds:
        for cycle in CYCLES:
            row = np.full(n_measures, np.nan)
            for i, (name, prefix, key) in enumerate(MEASURES):
                if prefix not in data:
                    continue
                traj = data[prefix].get('trajectories', {}).get(str(seed), [])
                for entry in traj:
                    if entry.get('cycle') == cycle:
                        val = entry.get(key)
                        if val is not None and np.isfinite(val):
                            row[i] = val
                        break
            rows.append(row)
            cycle_labels.append(cycle)
            seed_labels.append(seed)

    matrix = np.array(rows)
    return matrix, measure_names, cycle_labels, seed_labels


def compute_correlation_matrix(matrix: np.ndarray,
                                measure_names: List[str],
                                min_valid: int = 5
                                ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Compute pairwise Spearman correlation matrix.

    Returns (r_matrix, p_matrix, valid_names).
    Drops measures with fewer than min_valid non-NaN values.
    """
    n_obs, n_measures = matrix.shape

    # Filter measures with enough data
    valid_cols = []
    valid_names = []
    for i in range(n_measures):
        col = matrix[:, i]
        n_valid = np.sum(np.isfinite(col))
        # Also check variance
        finite_vals = col[np.isfinite(col)]
        if n_valid >= min_valid and np.std(finite_vals) > 1e-12:
            valid_cols.append(i)
            valid_names.append(measure_names[i])

    n_valid_measures = len(valid_cols)
    r_matrix = np.full((n_valid_measures, n_valid_measures), np.nan)
    p_matrix = np.full((n_valid_measures, n_valid_measures), np.nan)

    for i in range(n_valid_measures):
        for j in range(n_valid_measures):
            if i == j:
                r_matrix[i, j] = 1.0
                p_matrix[i, j] = 0.0
                continue

            x = matrix[:, valid_cols[i]]
            y = matrix[:, valid_cols[j]]

            # Use only paired non-NaN values
            mask = np.isfinite(x) & np.isfinite(y)
            if np.sum(mask) < min_valid:
                continue

            r, p = stats.spearmanr(x[mask], y[mask])
            if np.isfinite(r):
                r_matrix[i, j] = r
                p_matrix[i, j] = p

    return r_matrix, p_matrix, valid_names


def test_co_emergence(r_matrix: np.ndarray, p_matrix: np.ndarray,
                       measure_names: List[str]) -> Dict:
    """Test Prediction 11.1: C_wm, A_level, I_img co-emerge."""
    targets = ['C_wm', 'A_level', 'I_img']
    indices = [i for i, n in enumerate(measure_names) if n in targets]

    if len(indices) < 2:
        return {'tested': False, 'reason': 'insufficient measures'}

    pairs = []
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            ii, jj = indices[i], indices[j]
            r = r_matrix[ii, jj]
            p = p_matrix[ii, jj]
            pairs.append({
                'measure_1': measure_names[ii],
                'measure_2': measure_names[jj],
                'r': float(r) if np.isfinite(r) else None,
                'p': float(p) if np.isfinite(p) else None,
            })

    mean_r = np.nanmean([p['r'] for p in pairs if p['r'] is not None])

    return {
        'tested': True,
        'prediction': 'r > 0.7 for C_wm, A_level, I_img pairs',
        'pairs': pairs,
        'mean_r': float(mean_r) if np.isfinite(mean_r) else None,
        'confirmed': bool(mean_r > 0.7) if np.isfinite(mean_r) else False,
    }


def test_language_lag(matrix: np.ndarray, measure_names: List[str],
                       cycle_labels: List[int]) -> Dict:
    """Test Prediction 11.2: ρ_topo lags other measures.

    Check if ρ_topo becomes significant later than C_wm, MI_inter.
    """
    def first_significant_cycle(name):
        if name not in measure_names:
            return None
        idx = measure_names.index(name)
        col = matrix[:, idx]
        # Group by cycle, check if mean is above baseline
        for cycle in CYCLES:
            mask = [i for i, c in enumerate(cycle_labels) if c == cycle]
            vals = col[mask]
            finite = vals[np.isfinite(vals)]
            if len(finite) >= 2 and np.mean(finite) > 0.01:
                return cycle
        return None

    c_wm_onset = first_significant_cycle('C_wm')
    mi_onset = first_significant_cycle('MI_inter')
    topo_onset = first_significant_cycle('rho_topo')

    return {
        'tested': True,
        'prediction': 'ρ_topo onset cycle > C_wm onset cycle',
        'C_wm_onset': c_wm_onset,
        'MI_inter_onset': mi_onset,
        'rho_topo_onset': topo_onset,
        'confirmed': (topo_onset is not None and c_wm_onset is not None
                       and topo_onset > c_wm_onset),
        'note': 'ρ_topo never becomes significant in V13'
    }


def test_self_model_phi_jump(matrix: np.ndarray, measure_names: List[str],
                               cycle_labels: List[int]) -> Dict:
    """Test Prediction 11.4: SM emergence correlates with Φ jump."""
    sm_names = ['SM_sal', 'SM_capacity', 'rho_self']
    phi_names = ['robustness', 'phi_increase']

    results = []
    for sm_name in sm_names:
        if sm_name not in measure_names:
            continue
        sm_idx = measure_names.index(sm_name)
        for phi_name in phi_names:
            if phi_name not in measure_names:
                continue
            phi_idx = measure_names.index(phi_name)

            x = matrix[:, sm_idx]
            y = matrix[:, phi_idx]
            mask = np.isfinite(x) & np.isfinite(y)
            if np.sum(mask) < 5:
                continue

            r, p = stats.spearmanr(x[mask], y[mask])
            results.append({
                'sm_measure': sm_name,
                'phi_measure': phi_name,
                'r': float(r) if np.isfinite(r) else None,
                'p': float(p) if np.isfinite(p) else None,
            })

    return {
        'tested': True,
        'prediction': 'SM emergence correlates with Φ jump',
        'pairs': results,
        'confirmed': any(r['r'] is not None and r['r'] > 0.5 and
                        r['p'] is not None and r['p'] < 0.05
                        for r in results),
    }


def compute_entanglement_trajectory(matrix: np.ndarray,
                                      measure_names: List[str],
                                      cycle_labels: List[int],
                                      seed_labels: List[int],
                                      seeds: List[int] = [123, 42, 7]
                                      ) -> List[Dict]:
    """Compute entanglement score at each cycle.

    Entanglement = mean |Spearman r| across all measure pairs,
    computed across the population (seeds) at each generation.
    """
    trajectory = []
    for cycle in CYCLES:
        mask = [i for i, c in enumerate(cycle_labels) if c == cycle]
        if len(mask) < 2:
            trajectory.append({
                'cycle': cycle,
                'entanglement': None,
                'n_valid_pairs': 0,
                'mean_abs_r': None,
            })
            continue

        sub_matrix = matrix[mask]
        n_measures = sub_matrix.shape[1]

        # Cross-seed correlation at this cycle
        r_vals = []
        for i in range(n_measures):
            for j in range(i + 1, n_measures):
                x = sub_matrix[:, i]
                y = sub_matrix[:, j]
                valid = np.isfinite(x) & np.isfinite(y)
                if np.sum(valid) >= 2:
                    # With only 3 seeds, use Pearson (more stable than Spearman)
                    r, _ = stats.pearsonr(x[valid], y[valid])
                    if np.isfinite(r):
                        r_vals.append(abs(r))

        mean_abs = float(np.mean(r_vals)) if r_vals else None
        trajectory.append({
            'cycle': cycle,
            'entanglement': mean_abs,
            'n_valid_pairs': len(r_vals),
            'mean_abs_r': mean_abs,
        })

    return trajectory


def compute_evolutionary_correlations(matrix: np.ndarray,
                                        measure_names: List[str],
                                        cycle_labels: List[int],
                                        seed_labels: List[int],
                                        seeds: List[int] = [123, 42, 7]
                                        ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Compute correlations across evolutionary TIME within each seed.

    For each seed, measures how each quantity changes across cycles.
    Then averages correlations across seeds.
    """
    n_measures = len(measure_names)
    all_r = []

    for seed in seeds:
        mask = [i for i, s in enumerate(seed_labels) if s == seed]
        sub = matrix[mask]
        if len(sub) < 4:
            continue

        r_mat = np.full((n_measures, n_measures), np.nan)
        for i in range(n_measures):
            for j in range(n_measures):
                x = sub[:, i]
                y = sub[:, j]
                valid = np.isfinite(x) & np.isfinite(y)
                if np.sum(valid) >= 4:
                    r, _ = stats.spearmanr(x[valid], y[valid])
                    if np.isfinite(r):
                        r_mat[i, j] = r
        all_r.append(r_mat)

    if not all_r:
        return (np.full((n_measures, n_measures), np.nan),
                np.full((n_measures, n_measures), np.nan),
                measure_names)

    # Average across seeds
    stacked = np.stack(all_r)
    mean_r = np.nanmean(stacked, axis=0)

    # Compute p-values from Fisher-combined approach
    # For simplicity, re-compute on pooled data
    p_mat = np.full((n_measures, n_measures), np.nan)
    for i in range(n_measures):
        for j in range(n_measures):
            if i == j:
                p_mat[i, j] = 0.0
                continue
            x = matrix[:, i]
            y = matrix[:, j]
            valid = np.isfinite(x) & np.isfinite(y)
            if np.sum(valid) >= 5:
                _, p = stats.spearmanr(x[valid], y[valid])
                if np.isfinite(p):
                    p_mat[i, j] = p

    return mean_r, p_mat, measure_names


def identify_clusters(r_matrix: np.ndarray, measure_names: List[str],
                       threshold: float = 0.5) -> List[List[str]]:
    """Identify clusters of co-varying measures.

    Simple single-linkage clustering based on |r| > threshold.
    """
    n = len(measure_names)
    adj = np.abs(r_matrix) > threshold

    visited = set()
    clusters = []

    for i in range(n):
        if i in visited:
            continue
        cluster = set()
        stack = [i]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            cluster.add(node)
            for j in range(n):
                if j not in visited and adj[node, j]:
                    stack.append(j)
        if len(cluster) >= 2:
            clusters.append(sorted([measure_names[k] for k in cluster]))

    return clusters


def find_strongest_pairs(r_matrix: np.ndarray, p_matrix: np.ndarray,
                          measure_names: List[str],
                          top_k: int = 10) -> List[Dict]:
    """Find the top-k strongest correlations."""
    n = len(measure_names)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            r = r_matrix[i, j]
            p = p_matrix[i, j]
            if np.isfinite(r):
                pairs.append({
                    'measure_1': measure_names[i],
                    'measure_2': measure_names[j],
                    'r': float(r),
                    'p': float(p) if np.isfinite(p) else None,
                    'abs_r': abs(float(r)),
                })

    pairs.sort(key=lambda x: x['abs_r'], reverse=True)
    return pairs[:top_k]


def run_entanglement_analysis(results_dir: str) -> Dict:
    """Run complete entanglement analysis."""
    print("Loading cross-seed data from all experiments...")
    data = load_cross_seed_data(results_dir)
    print(f"  Loaded data from: {list(data.keys())}")

    print("\nBuilding measure matrix...")
    matrix, measure_names, cycle_labels, seed_labels = build_measure_matrix(data)
    print(f"  Matrix shape: {matrix.shape} ({matrix.shape[0]} obs × {matrix.shape[1]} measures)")

    # Count non-NaN per measure
    for i, name in enumerate(measure_names):
        n_valid = np.sum(np.isfinite(matrix[:, i]))
        if n_valid > 0:
            print(f"  {name}: {n_valid}/{matrix.shape[0]} valid")

    print("\n=== Pooled Correlation Matrix ===")
    r_matrix, p_matrix, valid_names = compute_correlation_matrix(
        matrix, measure_names)
    print(f"  Valid measures: {len(valid_names)}")

    print("\n=== Evolutionary Correlation (within-seed, across cycles) ===")
    evo_r, evo_p, evo_names = compute_evolutionary_correlations(
        matrix, measure_names, cycle_labels, seed_labels)

    # Filter evo_names to those with enough data
    valid_evo = []
    for i, name in enumerate(evo_names):
        col = matrix[:, i]
        if np.sum(np.isfinite(col)) >= 5:
            valid_evo.append(i)
    evo_names_filtered = [evo_names[i] for i in valid_evo]
    evo_r_filtered = evo_r[np.ix_(valid_evo, valid_evo)]
    evo_p_filtered = evo_p[np.ix_(valid_evo, valid_evo)]

    print(f"  Valid measures: {len(evo_names_filtered)}")

    print("\n=== Strongest Evolutionary Correlations ===")
    strongest = find_strongest_pairs(evo_r_filtered, evo_p_filtered,
                                      evo_names_filtered, top_k=15)
    for pair in strongest:
        sig = "***" if pair['p'] and pair['p'] < 0.001 else \
              "**" if pair['p'] and pair['p'] < 0.01 else \
              "*" if pair['p'] and pair['p'] < 0.05 else ""
        print(f"  {pair['measure_1']:15s} ↔ {pair['measure_2']:15s}  "
              f"r={pair['r']:+.3f}  p={pair['p']:.4f} {sig}")

    print("\n=== Co-emergence Test (Prediction 11.1) ===")
    co_emergence = test_co_emergence(evo_r_filtered, evo_p_filtered,
                                      evo_names_filtered)
    if co_emergence['tested']:
        print(f"  Mean r across C_wm, A_level, I_img: {co_emergence['mean_r']}")
        print(f"  Confirmed (r > 0.7): {co_emergence['confirmed']}")
        for pair in co_emergence.get('pairs', []):
            print(f"    {pair['measure_1']} ↔ {pair['measure_2']}: "
                  f"r={pair['r']}, p={pair['p']}")

    print("\n=== Language Lag Test (Prediction 11.2) ===")
    lang_lag = test_language_lag(matrix, measure_names, cycle_labels)
    print(f"  C_wm onset: cycle {lang_lag['C_wm_onset']}")
    print(f"  MI_inter onset: cycle {lang_lag['MI_inter_onset']}")
    print(f"  ρ_topo onset: {lang_lag['rho_topo_onset']}")
    print(f"  Confirmed: {lang_lag['confirmed']}")

    print("\n=== Self-Model × Φ Test (Prediction 11.4) ===")
    sm_phi = test_self_model_phi_jump(matrix, measure_names, cycle_labels)
    for pair in sm_phi.get('pairs', []):
        print(f"  {pair['sm_measure']} ↔ {pair['phi_measure']}: "
              f"r={pair['r']}, p={pair['p']}")
    print(f"  Confirmed: {sm_phi['confirmed']}")

    print("\n=== Measure Clusters (|r| > 0.5) ===")
    clusters = identify_clusters(evo_r_filtered, evo_names_filtered,
                                  threshold=0.5)
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i + 1}: {cluster}")

    print("\n=== Entanglement Trajectory ===")
    entanglement = compute_entanglement_trajectory(
        matrix, measure_names, cycle_labels, seed_labels)
    for e in entanglement:
        val = f"{e['entanglement']:.3f}" if e['entanglement'] else "N/A"
        print(f"  Cycle {e['cycle']:2d}: entanglement={val} "
              f"({e['n_valid_pairs']} pairs)")

    # Compile results
    results = {
        'matrix_shape': list(matrix.shape),
        'measure_names': measure_names,
        'valid_measures_pooled': valid_names,
        'valid_measures_evolutionary': evo_names_filtered,
        'strongest_evolutionary_correlations': strongest,
        'co_emergence_test': co_emergence,
        'language_lag_test': lang_lag,
        'self_model_phi_test': sm_phi,
        'clusters': clusters,
        'entanglement_trajectory': entanglement,
        'evolutionary_correlation_matrix': {
            'names': evo_names_filtered,
            'r': evo_r_filtered.tolist(),
            'p': evo_p_filtered.tolist(),
        },
    }

    return results
