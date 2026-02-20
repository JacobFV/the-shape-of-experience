"""V26 Hidden State Analysis — Does partial observability break the 1D collapse?

Compare V26 (1×1 obs + compass, H=32) against V25 baseline where:
  - Effective rank = NaN (degenerate)
  - Energy R² = 1.0
  - Position R² ≈ 0
  - Patch R² ≈ 0

Success criteria:
  - Effective rank > 5
  - Position decode R² > 0.3
  - Resource decode R² > 0.2
  - Energy decode R² < 0.8
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json
import glob
import os


def analyze_snapshot(path, label):
    """Analyze one V26 snapshot."""
    data = np.load(path, allow_pickle=True)
    hidden = data['hidden']
    positions = data['positions']
    energy = data['energy']
    alive = data['alive']
    is_predator = data['is_predator']

    # Filter alive + finite
    alive_idx = np.where(alive)[0]
    h = hidden[alive_idx]
    pos = positions[alive_idx]
    eng = energy[alive_idx]
    is_pred = is_predator[alive_idx]

    valid = np.all(np.isfinite(h), axis=1)
    h = h[valid]
    pos = pos[valid].astype(np.float32)
    eng = eng[valid]
    is_pred = is_pred[valid]

    if len(h) < 15:
        return None

    results = {
        'label': label, 'n_alive': len(h),
        'n_prey': int((~is_pred).sum()), 'n_pred': int(is_pred.sum()),
        'hidden_dim': h.shape[1],
    }

    # 1. Effective rank (PCA)
    h_std = np.std(h, axis=0)
    n_nonzero = np.sum(h_std > 1e-6)
    results['n_nonzero_dims'] = int(n_nonzero)

    if h.shape[0] > h.shape[1] and n_nonzero > 1:
        pca = PCA(n_components=min(h.shape[1], h.shape[0]))
        pca.fit(h)
        explained = pca.explained_variance_ratio_
        eff_rank = (np.sum(explained) ** 2) / np.sum(explained ** 2)
        results['eff_rank'] = float(eff_rank)
        results['top3_var'] = float(np.sum(explained[:3]))
        results['top5_var'] = float(np.sum(explained[:5]))
        results['var_ratio_1'] = float(explained[0])
    else:
        results['eff_rank'] = float('nan')
        results['top3_var'] = float('nan')
        results['top5_var'] = float('nan')
        results['var_ratio_1'] = float('nan')

    # 2. Energy decoding
    ridge = Ridge(alpha=1.0)
    if len(h) > 20:
        scores = cross_val_score(ridge, h, eng, cv=min(5, len(h)//4), scoring='r2')
        results['energy_r2'] = float(np.mean(scores))
    else:
        ridge.fit(h, eng)
        results['energy_r2'] = float(ridge.score(h, eng))

    # 3. Position decoding
    if len(h) > 20:
        scores_row = cross_val_score(Ridge(alpha=1.0), h, pos[:, 0], cv=min(5, len(h)//4), scoring='r2')
        scores_col = cross_val_score(Ridge(alpha=1.0), h, pos[:, 1], cv=min(5, len(h)//4), scoring='r2')
        results['pos_row_r2'] = float(np.mean(scores_row))
        results['pos_col_r2'] = float(np.mean(scores_col))
        results['pos_r2'] = float((np.mean(scores_row) + np.mean(scores_col)) / 2)
    else:
        results['pos_row_r2'] = float('nan')
        results['pos_col_r2'] = float('nan')
        results['pos_r2'] = float('nan')

    # 4. On-patch decoding (is the agent on a resource patch?)
    if 'patch_mask' in data:
        patch_mask = data['patch_mask']
        on_patch = patch_mask[pos[:, 0].astype(int) % patch_mask.shape[0],
                             pos[:, 1].astype(int) % patch_mask.shape[1]]
        if np.std(on_patch) > 0.01 and len(h) > 20:
            scores = cross_val_score(Ridge(alpha=1.0), h, on_patch, cv=min(5, len(h)//4), scoring='r2')
            results['patch_r2'] = float(np.mean(scores))
        else:
            results['patch_r2'] = float('nan')
    else:
        results['patch_r2'] = float('nan')

    # 5. Agent type decoding
    if np.sum(is_pred) > 3 and np.sum(~is_pred) > 3:
        if len(h) > 20:
            lr = LogisticRegression(max_iter=500)
            scores = cross_val_score(lr, h, is_pred.astype(int), cv=min(5, len(h)//4), scoring='accuracy')
            results['type_acc'] = float(np.mean(scores))
        else:
            results['type_acc'] = float('nan')
    else:
        results['type_acc'] = float('nan')

    # 6. K-means clustering
    best_k = 2
    best_sil = -1
    for k in range(2, min(8, max(len(h) // 5, 3))):
        try:
            km = KMeans(n_clusters=k, n_init=5, random_state=42)
            labels = km.fit_predict(h)
            if len(set(labels)) < 2:
                continue
            sil = silhouette_score(h, labels)
            if sil > best_sil:
                best_sil = sil
                best_k = k
        except Exception:
            continue
    results['best_k'] = best_k
    results['silhouette'] = float(best_sil)

    # 7. Hidden state statistics
    results['h_mean_abs'] = float(np.mean(np.abs(h)))
    results['h_max_abs'] = float(np.max(np.abs(h)))
    results['h_std'] = float(np.mean(np.std(h, axis=0)))

    return results


def main():
    base = os.path.dirname(__file__)
    results_dir = os.path.join(base, 'results')

    all_results = {}

    for seed in [42, 123, 7]:
        seed_dir = os.path.join(results_dir, f'v26_s{seed}')
        snapshots = sorted(glob.glob(os.path.join(seed_dir, 'snapshot_c*.npz')))

        if not snapshots:
            print(f"No snapshots for seed {seed}")
            continue

        seed_results = []
        for snap_path in snapshots:
            cycle = int(snap_path.split('_c')[-1].split('.')[0])
            label = f'V26_s{seed}_c{cycle:02d}'
            result = analyze_snapshot(snap_path, label)
            if result:
                seed_results.append(result)
                print(f"{label}: eff_rank={result['eff_rank']:.1f} | "
                      f"energy_r2={result['energy_r2']:.3f} | "
                      f"pos_r2={result.get('pos_r2', float('nan')):.3f} | "
                      f"patch_r2={result.get('patch_r2', float('nan')):.3f} | "
                      f"type_acc={result.get('type_acc', float('nan')):.3f} | "
                      f"sil={result['silhouette']:.3f} k={result['best_k']} | "
                      f"n_nonzero={result['n_nonzero_dims']}")

        all_results[f's{seed}'] = seed_results

    # Summary comparison
    print("\n" + "=" * 80)
    print("V26 (POMDP) vs V25 (5×5 obs) Hidden State Analysis")
    print("=" * 80)
    print(f"{'Metric':<20} {'V25 baseline':<20} {'V26 (mean)':<20} {'Pass?':<10}")
    print("-" * 80)

    all_eff = []
    all_energy = []
    all_pos = []
    all_patch = []
    all_type = []
    all_sil = []
    all_nonzero = []
    all_var1 = []

    for seed_data in all_results.values():
        for r in seed_data:
            if r['n_alive'] > 50:
                if not np.isnan(r.get('eff_rank', float('nan'))):
                    all_eff.append(r['eff_rank'])
                all_energy.append(r['energy_r2'])
                if not np.isnan(r.get('pos_r2', float('nan'))):
                    all_pos.append(r['pos_r2'])
                if not np.isnan(r.get('patch_r2', float('nan'))):
                    all_patch.append(r['patch_r2'])
                if not np.isnan(r.get('type_acc', float('nan'))):
                    all_type.append(r['type_acc'])
                all_sil.append(r['silhouette'])
                all_nonzero.append(r['n_nonzero_dims'])
                if not np.isnan(r.get('var_ratio_1', float('nan'))):
                    all_var1.append(r['var_ratio_1'])

    def fmt(vals, th, direction='>', name=''):
        if not vals:
            return 'N/A', 'N/A'
        m = np.mean(vals)
        ok = (m > th) if direction == '>' else (m < th)
        return f'{m:.3f}', 'YES' if ok else 'NO'

    v, p = fmt(all_eff, 5, '>', 'eff_rank')
    print(f"{'Eff rank':<20} {'NaN (degen)':<20} {v:<20} {p:<10}")

    v, p = fmt(all_energy, 0.8, '<', 'energy_r2')
    print(f"{'Energy R²':<20} {'1.000':<20} {v:<20} {p:<10}")

    v, p = fmt(all_pos, 0.3, '>', 'pos_r2')
    print(f"{'Position R²':<20} {'~0':<20} {v:<20} {p:<10}")

    v, p = fmt(all_patch, 0.2, '>', 'patch_r2')
    print(f"{'Patch R²':<20} {'~0':<20} {v:<20} {p:<10}")

    v, p = fmt(all_type, 0.7, '>', 'type_acc')
    print(f"{'Type accuracy':<20} {'0.80 (baseline)':<20} {v:<20} {p:<10}")

    if all_nonzero:
        print(f"{'Nonzero dims':<20} {'~1-3':<20} {np.mean(all_nonzero):.1f}/{32:<13} {'YES' if np.mean(all_nonzero) > 5 else 'NO':<10}")

    if all_var1:
        print(f"{'PC1 variance %':<20} {'~95-100%':<20} {np.mean(all_var1)*100:.1f}%{'':13} {'YES' if np.mean(all_var1) < 0.8 else 'NO':<10}")

    # Save
    out_path = os.path.join(results_dir, 'v26_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: None if np.isnan(x) else x)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
