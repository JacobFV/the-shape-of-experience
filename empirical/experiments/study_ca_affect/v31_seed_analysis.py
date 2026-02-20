"""V31 Seed Analysis — What determines which seeds find high integration?

With 10 seeds of V29 social prediction data, we can analyze what differs
between HIGH (>0.10) and LOW (<0.08) mean Φ seeds.

Candidate hypotheses:
1. Initial genome structure (some random genomes are closer to high-Φ basins)
2. Early population dynamics (bottleneck mortality patterns)
3. Drought response (some seeds develop drought-resilient integration)
4. Hidden state geometry at cycle 0 vs later cycles

Seeds by category:
  HIGH: s42 (0.143), s1 (0.138), s123 (0.106)
  MODERATE: s2 (0.092), s6 (0.086), s4 (0.081)
  LOW: s3 (0.074), s5 (0.073), s7 (0.062), s0 (0.054)
"""

import numpy as np
import json
import os
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


def load_seed_data(seed, results_dir='results'):
    """Load progress and snapshots for one seed."""
    seed_dir = os.path.join(results_dir, f'v31_s{seed}')

    # Load progress JSON
    progress_path = os.path.join(seed_dir, f'v29_s{seed}_progress.json')
    if not os.path.exists(progress_path):
        # V29 naming for original seeds
        progress_path = os.path.join(seed_dir, f'v31_s{seed}_progress.json')

    # Try multiple naming patterns
    for pattern in [f'v29_s{seed}_progress.json', f'v31_s{seed}_progress.json']:
        p = os.path.join(seed_dir, pattern)
        if os.path.exists(p):
            progress_path = p
            break

    with open(progress_path) as f:
        progress = json.load(f)

    # Load c00 snapshot (initial genomes after first cycle)
    snap_c00 = np.load(os.path.join(seed_dir, 'snapshot_c00.npz'))
    snap_c29 = np.load(os.path.join(seed_dir, 'snapshot_c29.npz'))

    return progress, snap_c00, snap_c29


def analyze_genome_structure(snap, label):
    """Analyze genome statistics for alive agents."""
    alive = snap['alive'].astype(bool)
    genomes = snap['genomes'][alive]

    if len(genomes) == 0:
        return {}

    # Genome statistics
    norms = np.sqrt(np.sum(genomes ** 2, axis=1))

    # Genome diversity: mean pairwise distance
    if len(genomes) > 1:
        diffs = genomes[:, None] - genomes[None, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        mean_dist = np.mean(dists[np.triu_indices(len(genomes), k=1)])

        # Cosine similarity
        cos_sim = cosine_similarity(genomes)
        mean_cos = np.mean(cos_sim[np.triu_indices(len(genomes), k=1)])
    else:
        mean_dist = 0.0
        mean_cos = 1.0

    # PCA on genomes
    pca = PCA(n_components=min(10, len(genomes), genomes.shape[1]))
    pca.fit(genomes)
    var_explained_3 = float(np.sum(pca.explained_variance_ratio_[:3]))
    eff_rank = float(np.exp(-np.sum(pca.explained_variance_ratio_ *
                                      np.log(pca.explained_variance_ratio_ + 1e-10))))

    return {
        'n_alive': len(genomes),
        'genome_mean_norm': float(np.mean(norms)),
        'genome_std_norm': float(np.std(norms)),
        'genome_mean_dist': float(mean_dist),
        'genome_mean_cos': float(mean_cos),
        'genome_pca3_var': var_explained_3,
        'genome_eff_rank': eff_rank,
    }


def analyze_hidden_geometry(snap, label):
    """Analyze hidden state geometry."""
    alive = snap['alive'].astype(bool)
    hidden = snap['hidden'][alive]

    if len(hidden) < 3:
        return {}

    norms = np.sqrt(np.sum(hidden ** 2, axis=1))

    # Filter zero-norm
    nonzero = norms > 1e-6
    if nonzero.sum() < 3:
        return {'hidden_n_nonzero': int(nonzero.sum())}

    hidden_nz = hidden[nonzero]
    norms_nz = norms[nonzero]

    cos_sim = cosine_similarity(hidden_nz)
    mean_cos = np.mean(cos_sim[np.triu_indices(len(hidden_nz), k=1)])

    pca = PCA(n_components=min(5, len(hidden_nz), hidden_nz.shape[1]))
    pca.fit(hidden_nz)
    var_3 = float(np.sum(pca.explained_variance_ratio_[:3]))

    return {
        'hidden_mean_norm': float(np.mean(norms_nz)),
        'hidden_mean_cos': float(mean_cos),
        'hidden_pca3_var': var_3,
        'hidden_n_nonzero': int(nonzero.sum()),
    }


def analyze_trajectory(progress):
    """Analyze cycle-level trajectory features."""
    cycles = progress['cycles']

    phis = [c['mean_phi'] for c in cycles]
    robs = [c['robustness'] for c in cycles]
    pops = [c['n_alive_end'] for c in cycles]
    mses = [c['mean_pred_mse'] for c in cycles]

    # Drought mortality pattern
    drought_cycles = [c for c in cycles if c['mortality'] > 0.5]
    non_drought_phis = [c['mean_phi'] for c in cycles if c['mortality'] < 0.5]
    drought_phis = [c['mean_phi'] for c in cycles if c['mortality'] > 0.5]

    # Phi trajectory features
    phi_early = np.mean(phis[:10])
    phi_late = np.mean(phis[20:])
    phi_trend = phi_late - phi_early

    # Post-drought bounce
    post_drought_bounces = []
    for i, c in enumerate(cycles):
        if c['mortality'] > 0.5 and i + 1 < len(cycles):
            post_drought_bounces.append(cycles[i+1]['mean_phi'])

    return {
        'phi_early': float(phi_early),
        'phi_late': float(phi_late),
        'phi_trend': float(phi_trend),
        'phi_max': float(np.max(phis)),
        'phi_std': float(np.std(phis)),
        'n_droughts': len(drought_cycles),
        'mean_drought_phi': float(np.mean(drought_phis)) if drought_phis else 0.0,
        'mean_non_drought_phi': float(np.mean(non_drought_phis)) if non_drought_phis else 0.0,
        'mean_post_drought_bounce': float(np.mean(post_drought_bounces)) if post_drought_bounces else 0.0,
        'mse_early': float(np.mean(mses[:5])),
        'mse_late': float(np.mean(mses[25:])),
    }


def main():
    seeds = [42, 123, 7, 0, 1, 2, 3, 4, 5, 6]
    categories = {
        42: 'HIGH', 123: 'HIGH', 1: 'HIGH',
        2: 'MOD', 6: 'MOD', 4: 'MOD',
        3: 'LOW', 5: 'LOW', 7: 'LOW', 0: 'LOW',
    }

    mean_phis = {
        42: 0.143, 123: 0.106, 7: 0.062, 0: 0.054, 1: 0.138,
        2: 0.092, 3: 0.074, 4: 0.081, 5: 0.073, 6: 0.086,
    }

    all_results = []

    for seed in seeds:
        print(f"\nSeed {seed} ({categories[seed]}, Φ={mean_phis[seed]:.3f})")
        try:
            progress, snap_c00, snap_c29 = load_seed_data(seed)
        except Exception as e:
            print(f"  Error loading: {e}")
            continue

        genome_c00 = analyze_genome_structure(snap_c00, f's{seed}_c00')
        genome_c29 = analyze_genome_structure(snap_c29, f's{seed}_c29')
        hidden_c29 = analyze_hidden_geometry(snap_c29, f's{seed}_c29')
        trajectory = analyze_trajectory(progress)

        result = {
            'seed': seed,
            'category': categories[seed],
            'mean_phi': mean_phis[seed],
            **{f'c00_{k}': v for k, v in genome_c00.items()},
            **{f'c29_{k}': v for k, v in genome_c29.items()},
            **{f'c29_h_{k}': v for k, v in hidden_c29.items()},
            **trajectory,
        }
        all_results.append(result)

        print(f"  C00 genome: norm={genome_c00.get('genome_mean_norm', '?'):.3f}, "
              f"div={genome_c00.get('genome_mean_dist', '?'):.3f}, "
              f"cos={genome_c00.get('genome_mean_cos', '?'):.3f}")
        print(f"  C29 genome: norm={genome_c29.get('genome_mean_norm', '?'):.3f}, "
              f"div={genome_c29.get('genome_mean_dist', '?'):.3f}, "
              f"cos={genome_c29.get('genome_mean_cos', '?'):.3f}")
        print(f"  C29 hidden: norm={hidden_c29.get('hidden_mean_norm', '?'):.3f}, "
              f"cos={hidden_c29.get('hidden_mean_cos', '?'):.3f}")
        print(f"  Trajectory: early={trajectory['phi_early']:.3f}, "
              f"late={trajectory['phi_late']:.3f}, "
              f"trend={trajectory['phi_trend']:+.3f}")

    # Compare HIGH vs LOW
    print("\n" + "="*60)
    print("HIGH vs LOW comparison")
    print("="*60)

    high = [r for r in all_results if r['category'] == 'HIGH']
    low = [r for r in all_results if r['category'] == 'LOW']

    for key in ['c00_genome_mean_norm', 'c00_genome_mean_dist', 'c00_genome_mean_cos',
                'c00_genome_eff_rank',
                'c29_genome_mean_norm', 'c29_genome_mean_dist', 'c29_genome_mean_cos',
                'c29_h_hidden_mean_norm', 'c29_h_hidden_mean_cos', 'c29_h_hidden_pca3_var',
                'phi_early', 'phi_late', 'phi_trend', 'phi_std',
                'mean_drought_phi', 'mean_post_drought_bounce',
                'mse_early', 'mse_late']:
        h_vals = [r[key] for r in high if key in r]
        l_vals = [r[key] for r in low if key in r]
        if h_vals and l_vals:
            h_mean = np.mean(h_vals)
            l_mean = np.mean(l_vals)
            diff = h_mean - l_mean
            print(f"  {key:40s}: HIGH={h_mean:.4f}  LOW={l_mean:.4f}  diff={diff:+.4f}")

    # Save results
    with open('results/v31_seed_analysis.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to results/v31_seed_analysis.json")


if __name__ == '__main__':
    main()
