"""V27 Analysis: Hidden state dimensionality and MLP head coupling.

Analyzes whether the nonlinear MLP prediction head creates richer hidden
state representations compared to V22's linear head.

Key metrics:
  - Effective rank (PCA): dimensionality of hidden state variation
  - Energy/position/resource R²: what hidden states encode (linear probes)
  - Prediction Jacobian rank: how many hidden dims the MLP actually uses
  - MLP saturation: fraction of hidden units in tanh saturation regime
"""

import numpy as np
import json
import os
import sys
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.cluster import KMeans


def analyze_snapshot(snap_path, verbose=True):
    """Analyze a single snapshot for hidden state richness."""
    data = np.load(snap_path, allow_pickle=True)
    hidden = data['hidden']
    alive = data['alive'].astype(bool)
    energy = data['energy']
    positions = data['positions']
    resources = data['resources']

    alive_idx = np.where(alive)[0]
    n_alive = len(alive_idx)

    if n_alive < 10:
        return {'n_alive': n_alive, 'skip': True, 'reason': 'too few alive'}

    h = hidden[alive_idx]
    e = energy[alive_idx]
    pos = positions[alive_idx]

    # Check for all-zero hidden states (snapshot timing bug)
    h_norms = np.sqrt(np.sum(h ** 2, axis=1))
    n_nonzero = int(np.sum(h_norms > 1e-6))
    if n_nonzero < 5:
        return {'n_alive': n_alive, 'n_nonzero': n_nonzero, 'skip': True,
                'reason': 'hidden states all zero (possible bug)'}

    # Use only nonzero hidden states
    nonzero_mask = h_norms > 1e-6
    h = h[nonzero_mask]
    e = e[alive_idx][nonzero_mask]
    pos = pos[alive_idx][nonzero_mask]
    n_used = len(h)

    # 1. PCA effective rank
    pca = PCA()
    pca.fit(h)
    explained = pca.explained_variance_ratio_
    # Effective rank = exp(entropy of explained variance)
    explained_nonzero = explained[explained > 1e-10]
    eff_rank = float(np.exp(-np.sum(explained_nonzero * np.log(explained_nonzero))))
    var_ratio_1 = float(explained[0])

    # 2. Linear probes
    from sklearn.model_selection import cross_val_score

    # Energy decoding
    if np.std(e) > 1e-6:
        ridge_e = Ridge(alpha=1.0)
        energy_r2 = float(np.mean(cross_val_score(ridge_e, h, e, cv=min(5, n_used), scoring='r2')))
    else:
        energy_r2 = 0.0

    # Position decoding (x coordinate, wrapped)
    pos_x = pos[:, 0].astype(float)
    if np.std(pos_x) > 1e-6:
        ridge_p = Ridge(alpha=1.0)
        pos_r2 = float(np.mean(cross_val_score(ridge_p, h, pos_x, cv=min(5, n_used), scoring='r2')))
    else:
        pos_r2 = 0.0

    # Resource at position
    res_at_pos = resources[pos[:, 0], pos[:, 1]]
    if np.std(res_at_pos) > 1e-6:
        ridge_r = Ridge(alpha=1.0)
        res_r2 = float(np.mean(cross_val_score(ridge_r, h, res_at_pos, cv=min(5, n_used), scoring='r2')))
    else:
        res_r2 = 0.0

    # 3. Hidden state statistics
    h_mean_abs = float(np.mean(np.abs(h)))
    h_max_abs = float(np.max(np.abs(h)))
    h_std = float(np.std(h))

    # 4. Clustering (do hidden states form distinct groups?)
    if n_used >= 10:
        km = KMeans(n_clusters=min(5, n_used // 3), n_init=3, random_state=42)
        km.fit(h)
        from sklearn.metrics import silhouette_score
        if len(set(km.labels_)) > 1:
            sil = float(silhouette_score(h, km.labels_))
        else:
            sil = 0.0
    else:
        sil = 0.0

    result = {
        'n_alive': n_alive,
        'n_used': n_used,
        'n_nonzero': n_nonzero,
        'eff_rank': eff_rank,
        'var_ratio_1': var_ratio_1,
        'energy_r2': energy_r2,
        'position_r2': pos_r2,
        'resource_r2': res_r2,
        'h_mean_abs': h_mean_abs,
        'h_max_abs': h_max_abs,
        'h_std': h_std,
        'silhouette': sil,
        'skip': False,
    }

    if verbose:
        print(f"  n_alive={n_alive}, n_used={n_used}")
        print(f"  eff_rank={eff_rank:.2f}, PC1={var_ratio_1:.2%}")
        print(f"  energy_R²={energy_r2:.3f}, pos_R²={pos_r2:.3f}, res_R²={res_r2:.3f}")
        print(f"  h_mean={h_mean_abs:.4f}, h_max={h_max_abs:.4f}, h_std={h_std:.4f}")
        print(f"  silhouette={sil:.3f}")

    return result


def analyze_seed(results_dir, seed, verbose=True):
    """Analyze all snapshots for one seed."""
    snap_files = sorted([
        f for f in os.listdir(results_dir)
        if f.startswith('snapshot_c') and f.endswith('.npz')
    ])

    if not snap_files:
        print(f"No snapshots found in {results_dir}")
        return {}

    results = {}
    for sf in snap_files:
        cycle = int(sf.split('_c')[1].split('.')[0])
        if verbose:
            print(f"\n--- Cycle {cycle} ---")
        snap_path = os.path.join(results_dir, sf)
        results[f'c{cycle:02d}'] = analyze_snapshot(snap_path, verbose=verbose)

    return results


def main():
    """Analyze V27 results across all seeds."""
    base = os.path.dirname(os.path.abspath(__file__))
    results_base = os.path.join(base, 'results')

    all_results = {}
    for seed in [42, 123, 7]:
        results_dir = os.path.join(results_base, f'v27_s{seed}')
        if not os.path.exists(results_dir):
            print(f"Skipping seed {seed} (no results)")
            continue
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        all_results[f's{seed}'] = analyze_seed(results_dir, seed)

    # Save
    out_path = os.path.join(results_base, 'v27_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary comparison vs V22
    print(f"\n{'='*60}")
    print("V27 vs V22 COMPARISON (cycle 29)")
    print(f"{'='*60}")
    print("V22 baselines: eff_rank 5.1-7.3, energy R² -3.6 to -4.6, phi ~0.097")
    print()
    for seed_key, seed_results in all_results.items():
        c29 = seed_results.get('c29', {})
        if c29.get('skip'):
            print(f"  {seed_key}: SKIPPED ({c29.get('reason', 'unknown')})")
        else:
            print(f"  {seed_key}: eff_rank={c29.get('eff_rank', 0):.2f}, "
                  f"energy_R²={c29.get('energy_r2', 0):.3f}, "
                  f"pos_R²={c29.get('position_r2', 0):.3f}, "
                  f"res_R²={c29.get('resource_r2', 0):.3f}")


if __name__ == '__main__':
    main()
