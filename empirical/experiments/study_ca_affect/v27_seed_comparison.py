"""V27 Seed Comparison: Why did seed 7 achieve Φ=0.245?

Investigates the mechanistic difference between seed 7 (Φ=0.245, highest ever)
and seeds 42/123 (Φ lower than V22 baseline).

Key hypothesis: seed 7's MLP operates in a more nonlinear regime (saturated tanh),
creating stronger gradient coupling across hidden units.

Metrics:
  1. MLP saturation: fraction of tanh units in saturated regime (|tanh(z)| > 0.9)
  2. Jacobian effective rank: how many hidden dims contribute to prediction
  3. W1 magnitude and structure: Frobenius norm, singular values
  4. Hidden state geometry: PCA structure, inter-agent distance distribution
"""

import numpy as np
import os
import json
from sklearn.decomposition import PCA


def load_snapshot(seed, cycle=29):
    """Load snapshot for a seed."""
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, 'results', f'v27_s{seed}', f'snapshot_c{cycle:02d}.npz')
    return np.load(path, allow_pickle=True)


def analyze_mlp_regime(hidden, genomes, cfg_params):
    """Analyze whether the MLP is operating in linear or nonlinear regime.

    For each agent, compute:
      z = W1 @ h + b1
      tanh(z)
    and measure saturation.
    """
    H = 16
    PH = 8

    # Extract MLP weights from genomes
    # Need to find offsets in the flat parameter vector
    # From v27_substrate._param_shapes:
    # embed_W (76,24)=1824, embed_b (24,)=24, gru_Wz (40,16)=640, gru_bz (16,)=16,
    # gru_Wr (40,16)=640, gru_br (16,)=16, gru_Wh (40,16)=640, gru_bh (16,)=16,
    # out_W (16,7)=112, out_b (7,)=7, internal_embed_W (3,24)=72, internal_embed_b (24,)=24,
    # tick_weights (8,)=8, sync_decay_raw (1,)=1
    # predict_W1 (16,8)=128, predict_b1 (8,)=8, predict_W2 (8,1)=8, predict_b2 (1,)=1, lr_raw (1,)=1

    offset_W1 = 1824 + 24 + 640 + 16 + 640 + 16 + 640 + 16 + 112 + 7 + 72 + 24 + 8 + 1
    # = 4040
    offset_b1 = offset_W1 + H * PH  # 4040 + 128 = 4168
    offset_W2 = offset_b1 + PH       # 4168 + 8 = 4176
    offset_b2 = offset_W2 + PH       # 4176 + 8 = 4184
    offset_lr = offset_b2 + 1        # 4184 + 1 = 4185

    M = genomes.shape[0]
    W1 = genomes[:, offset_W1:offset_W1 + H * PH].reshape(M, H, PH)
    b1 = genomes[:, offset_b1:offset_b1 + PH]
    W2 = genomes[:, offset_W2:offset_W2 + PH]
    b2 = genomes[:, offset_b2]

    results = {
        'saturation_fracs': [],
        'mean_abs_z': [],
        'jacobian_ranks': [],
        'W1_frobenius': [],
        'W1_sv': [],
        'W2_magnitude': [],
        'prediction_range': [],
    }

    for i in range(M):
        h = hidden[i]  # (H,)
        w1 = W1[i]     # (H, PH)
        b = b1[i]      # (PH,)
        w2 = W2[i]     # (PH,)

        # Pre-activation
        z = h @ w1 + b  # (PH,)

        # Saturation: fraction where |tanh(z)| > 0.9
        tanh_z = np.tanh(z)
        sat_frac = np.mean(np.abs(tanh_z) > 0.9)
        results['saturation_fracs'].append(float(sat_frac))
        results['mean_abs_z'].append(float(np.mean(np.abs(z))))

        # Jacobian: ∂pred/∂h = w2^T @ diag(1 - tanh²(z)) @ W1
        # Shape: (1, PH) @ (PH, PH) @ (PH, H) → (1, H)
        dtanh = 1.0 - tanh_z ** 2  # (PH,)
        J = w2[None, :] * dtanh[None, :] @ w1.T  # (1, H)

        # Effective rank of Jacobian (even though it's rank-1 in this formulation,
        # the question is how spread the sensitivity is across H dims)
        J_abs = np.abs(J.flatten())
        J_norm = J_abs / (np.sum(J_abs) + 1e-10)
        J_entropy = -np.sum(J_norm * np.log(J_norm + 1e-10))
        J_eff_rank = np.exp(J_entropy)
        results['jacobian_ranks'].append(float(J_eff_rank))

        # W1 structure
        results['W1_frobenius'].append(float(np.linalg.norm(w1, 'fro')))

        # W1 singular values
        sv = np.linalg.svd(w1, compute_uv=False)
        results['W1_sv'].append(sv.tolist())

        # W2 magnitude
        results['W2_magnitude'].append(float(np.linalg.norm(w2)))

        # Prediction
        pred = tanh_z @ w2 + b2[i]
        results['prediction_range'].append(float(pred))

    return results


def analyze_hidden_geometry(hidden, alive):
    """Analyze the geometric structure of hidden states."""
    alive_idx = np.where(alive)[0]
    h = hidden[alive_idx]
    n = len(h)

    if n < 5:
        return {'n': n, 'skip': True}

    # PCA
    pca = PCA()
    pca.fit(h)
    ev = pca.explained_variance_ratio_

    # Inter-agent distances
    from scipy.spatial.distance import pdist
    if n > 200:
        # Subsample
        idx = np.random.choice(n, 200, replace=False)
        dists = pdist(h[idx])
    else:
        dists = pdist(h)

    # Cosine similarities
    norms = np.linalg.norm(h, axis=1, keepdims=True)
    h_normed = h / (norms + 1e-10)
    if n > 200:
        cos_sims = h_normed[idx] @ h_normed[idx].T
        cos_sims = cos_sims[np.triu_indices(len(idx), k=1)]
    else:
        cos_sims = h_normed @ h_normed.T
        cos_sims = cos_sims[np.triu_indices(n, k=1)]

    return {
        'n': n,
        'skip': False,
        'pca_explained': ev[:5].tolist(),
        'pca_cumulative_3': float(np.sum(ev[:3])),
        'pca_cumulative_5': float(np.sum(ev[:5])),
        'dist_mean': float(np.mean(dists)),
        'dist_std': float(np.std(dists)),
        'dist_min': float(np.min(dists)),
        'dist_max': float(np.max(dists)),
        'cos_mean': float(np.mean(cos_sims)),
        'cos_std': float(np.std(cos_sims)),
        'h_mean_norm': float(np.mean(norms)),
        'h_std_norm': float(np.std(norms)),
    }


def main():
    print("=" * 70)
    print("V27 SEED COMPARISON: Why is seed 7 special?")
    print("=" * 70)

    all_results = {}

    for seed in [42, 123, 7]:
        print(f"\n{'='*50}")
        print(f"SEED {seed}")
        print(f"{'='*50}")

        data = load_snapshot(seed, cycle=29)
        hidden = data['hidden']
        alive = data['alive'].astype(bool)
        genomes = data['genomes']

        alive_idx = np.where(alive)[0]
        n_alive = len(alive_idx)
        print(f"  {n_alive} alive agents")

        # MLP regime analysis (alive agents only)
        h_alive = hidden[alive_idx]
        g_alive = genomes[alive_idx]

        mlp = analyze_mlp_regime(h_alive, g_alive, None)

        sat = np.array(mlp['saturation_fracs'])
        abs_z = np.array(mlp['mean_abs_z'])
        j_rank = np.array(mlp['jacobian_ranks'])
        w1_frob = np.array(mlp['W1_frobenius'])
        w2_mag = np.array(mlp['W2_magnitude'])

        print(f"\n  MLP SATURATION:")
        print(f"    Mean saturation fraction:  {np.mean(sat):.3f} (fraction of tanh units |tanh(z)|>0.9)")
        print(f"    Std saturation:            {np.std(sat):.3f}")
        print(f"    Mean |z| (pre-activation): {np.mean(abs_z):.3f}")
        print(f"    Agents with >50% saturated: {np.mean(sat > 0.5):.1%}")
        print(f"    Agents with >75% saturated: {np.mean(sat > 0.75):.1%}")

        print(f"\n  JACOBIAN SENSITIVITY:")
        print(f"    Mean Jacobian eff rank:    {np.mean(j_rank):.2f} / 16")
        print(f"    Std Jacobian eff rank:     {np.std(j_rank):.2f}")

        print(f"\n  WEIGHT MAGNITUDES:")
        print(f"    Mean W1 Frobenius norm:    {np.mean(w1_frob):.3f}")
        print(f"    Mean W2 L2 norm:           {np.mean(w2_mag):.3f}")

        # W1 singular value structure (population average)
        all_sv = np.array(mlp['W1_sv'])  # (n_alive, min(H, PH))
        mean_sv = np.mean(all_sv, axis=0)
        print(f"\n  W1 SINGULAR VALUES (pop mean):")
        for j, sv in enumerate(mean_sv):
            print(f"    SV[{j}]: {sv:.4f}")
        sv_ratio = mean_sv[0] / (np.sum(mean_sv) + 1e-10)
        print(f"    Top SV ratio: {sv_ratio:.3f} (1.0 = rank-1, 0.125 = uniform)")

        # Hidden state geometry
        print(f"\n  HIDDEN STATE GEOMETRY:")
        geom = analyze_hidden_geometry(hidden, alive)
        if not geom.get('skip'):
            print(f"    PCA top-3 cumulative:      {geom['pca_cumulative_3']:.3f}")
            print(f"    PCA top-5 cumulative:      {geom['pca_cumulative_5']:.3f}")
            print(f"    Inter-agent dist mean:     {geom['dist_mean']:.3f}")
            print(f"    Inter-agent dist std:      {geom['dist_std']:.3f}")
            print(f"    Cosine sim mean:           {geom['cos_mean']:.3f}")
            print(f"    Cosine sim std:            {geom['cos_std']:.3f}")
            print(f"    Hidden norm mean:          {geom['h_mean_norm']:.3f}")
            print(f"    Hidden norm std:           {geom['h_std_norm']:.3f}")

        all_results[f's{seed}'] = {
            'n_alive': n_alive,
            'saturation_mean': float(np.mean(sat)),
            'saturation_std': float(np.std(sat)),
            'mean_abs_z': float(np.mean(abs_z)),
            'frac_gt50_sat': float(np.mean(sat > 0.5)),
            'frac_gt75_sat': float(np.mean(sat > 0.75)),
            'jacobian_rank_mean': float(np.mean(j_rank)),
            'jacobian_rank_std': float(np.std(j_rank)),
            'W1_frobenius_mean': float(np.mean(w1_frob)),
            'W2_magnitude_mean': float(np.mean(w2_mag)),
            'W1_top_sv_ratio': float(sv_ratio),
            'W1_sv_mean': mean_sv.tolist(),
            'geometry': geom,
        }

    # Summary comparison
    print(f"\n{'='*70}")
    print("COMPARATIVE SUMMARY")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {'Seed 42':>10} {'Seed 123':>10} {'Seed 7':>10}")
    print("-" * 60)
    for metric in ['saturation_mean', 'mean_abs_z', 'frac_gt50_sat',
                   'jacobian_rank_mean', 'W1_frobenius_mean', 'W2_magnitude_mean',
                   'W1_top_sv_ratio']:
        vals = [all_results[f's{s}'][metric] for s in [42, 123, 7]]
        print(f"  {metric:<28} {vals[0]:10.3f} {vals[1]:10.3f} {vals[2]:10.3f}")

    # Save
    base = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(base, 'results', 'v27_seed_comparison.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
