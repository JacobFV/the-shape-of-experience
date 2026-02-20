"""V32 Analysis: What Determines the 30/70 Split?

Post-hoc analysis of 50-seed drought autopsy data.

Questions:
1. Is the 30/70 distribution stable at 50 seeds?
2. What predicts whether a seed becomes HIGH vs LOW?
3. At what cycle do HIGH and LOW seeds first diverge?
4. Is the split determined by initial genomes, first drought dynamics,
   or an ongoing trajectory-dependent process?

Usage:
    python v32_analysis.py results/  # analyze results directory
"""

import sys
import os
import json
import numpy as np
from scipy import stats


def load_all_results(results_dir):
    """Load all V32 seed results from results directory."""
    seed_results = {}
    for entry in sorted(os.listdir(results_dir)):
        if entry.startswith('v32_s') and os.path.isdir(os.path.join(results_dir, entry)):
            seed_str = entry.replace('v32_s', '')
            try:
                seed = int(seed_str)
            except ValueError:
                continue
            result_file = os.path.join(results_dir, entry, f'v32_s{seed}_results.json')
            if os.path.exists(result_file):
                with open(result_file) as f:
                    seed_results[seed] = json.load(f)
    return seed_results


def classify_seeds(seed_results):
    """Classify seeds into HIGH/MOD/LOW based on late-phase Φ."""
    categories = {'HIGH': [], 'MOD': [], 'LOW': []}
    for seed, r in seed_results.items():
        cat = r['summary']['category']
        categories[cat].append(seed)
    return categories


def extract_trajectories(seed_results):
    """Extract per-cycle Φ trajectories for all seeds."""
    trajectories = {}
    for seed, r in seed_results.items():
        phis = [c['mean_phi'] for c in r['cycles']]
        trajectories[seed] = np.array(phis)
    return trajectories


def find_divergence_point(trajectories, categories):
    """At what cycle do HIGH and LOW seeds first become distinguishable?

    Uses rolling t-test between HIGH and LOW seed trajectories.
    Returns the first cycle where p < 0.05.
    """
    high_seeds = categories['HIGH']
    low_seeds = categories['LOW']

    if len(high_seeds) < 2 or len(low_seeds) < 2:
        return None

    n_cycles = min(len(t) for t in trajectories.values())
    divergence_cycle = None

    results = []
    for c in range(n_cycles):
        high_vals = [trajectories[s][c] for s in high_seeds if s in trajectories]
        low_vals = [trajectories[s][c] for s in low_seeds if s in trajectories]

        if len(high_vals) < 2 or len(low_vals) < 2:
            results.append({'cycle': c, 'p': 1.0, 't': 0.0})
            continue

        t_stat, p_val = stats.ttest_ind(high_vals, low_vals)
        results.append({
            'cycle': c,
            't': float(t_stat),
            'p': float(p_val),
            'high_mean': float(np.mean(high_vals)),
            'low_mean': float(np.mean(low_vals)),
            'effect_size': float(np.mean(high_vals) - np.mean(low_vals)),
        })
        if divergence_cycle is None and p_val < 0.05:
            divergence_cycle = c

    return divergence_cycle, results


def analyze_drought_predictors(seed_results, categories):
    """Which drought-related quantities predict final category?

    Tests pre-drought-1 metrics, first bounce, survival rates.
    """
    predictors = {}

    # Collect per-seed features
    features = {}
    late_phis = {}

    for seed, r in seed_results.items():
        s = r['summary']
        late_phis[seed] = s['late_mean_phi']

        feat = {
            'first_bounce': s['first_bounce'],
            'mean_bounce': s['mean_bounce'],
            'final_eff_rank': s['final_eff_rank'],
            'final_weight_diversity': s['final_weight_diversity'],
            'mean_phi': s['mean_phi'],
            'max_robustness': s['max_robustness'],
            'mean_robustness': s['mean_robustness'],
        }

        # First drought details
        if r['drought_events']:
            d0 = r['drought_events'][0]
            feat['d0_pre_phi'] = d0['pre_phi']
            feat['d0_post_phi'] = d0['post_phi']
            feat['d0_survival_rate'] = d0['survivor_stats']['survival_rate']
            feat['d0_pre_eff_rank'] = d0['pre_eff_rank']
            feat['d0_post_eff_rank'] = d0['post_eff_rank']
            feat['d0_hidden_convergence'] = d0['hidden_convergence']

            ss = d0['survivor_stats']
            if 'surv_lr_mean' in ss and 'died_lr_mean' in ss:
                feat['d0_lr_diff'] = ss['surv_lr_mean'] - ss['died_lr_mean']
            if 'surv_energy_mean' in ss and 'died_energy_mean' in ss:
                feat['d0_energy_diff'] = ss['surv_energy_mean'] - ss['died_energy_mean']

        # Pre-drought metrics (from cycles before first drought)
        pre_drought_cycles = [c for c in r['cycles'] if c['cycle'] < 5]
        if pre_drought_cycles:
            feat['pre_d_mean_phi'] = float(np.mean([c['mean_phi'] for c in pre_drought_cycles]))
            feat['pre_d_mean_eff_rank'] = float(np.mean([
                c.get('eff_rank', 0) for c in pre_drought_cycles]))
            feat['pre_d_weight_div'] = float(np.mean([
                c.get('weight_diversity', 0) for c in pre_drought_cycles]))
            feat['pre_d_mean_mse'] = float(np.mean([
                c['mean_pred_mse'] for c in pre_drought_cycles]))
            feat['pre_d_mean_grad_norm'] = float(np.mean([
                c.get('mean_grad_norm', 0) for c in pre_drought_cycles]))

        # All bounces
        if len(r['drought_events']) >= 2:
            feat['bounce_variance'] = float(np.var(s['bounces']))
            feat['bounce_trend'] = float(
                np.polyfit(range(len(s['bounces'])), s['bounces'], 1)[0]
            )

        features[seed] = feat

    # Correlate each feature with late_mean_phi
    seeds_with_data = sorted(set(features.keys()) & set(late_phis.keys()))
    y = np.array([late_phis[s] for s in seeds_with_data])

    all_feature_names = set()
    for f in features.values():
        all_feature_names |= set(f.keys())

    for fname in sorted(all_feature_names):
        vals = []
        valid_seeds = []
        for s in seeds_with_data:
            if fname in features[s] and features[s][fname] is not None:
                vals.append(features[s][fname])
                valid_seeds.append(s)

        if len(vals) < 5:
            continue

        vals = np.array(vals)
        y_valid = np.array([late_phis[s] for s in valid_seeds])

        if np.std(vals) < 1e-10:
            continue

        r, p = stats.pearsonr(vals, y_valid)
        predictors[fname] = {
            'r': float(r),
            'p': float(p),
            'n': len(vals),
            'significant': p < 0.05,
        }

    # Sort by absolute correlation
    predictors = dict(sorted(predictors.items(),
                             key=lambda x: abs(x[1]['r']), reverse=True))
    return predictors, features


def analyze_survivor_genome_diff(seed_results, categories):
    """Do HIGH seeds' drought survivors differ systematically
    from LOW seeds' drought survivors?

    Compares survivor stats at each drought between categories.
    """
    high_seeds = categories['HIGH']
    low_seeds = categories['LOW']

    if not high_seeds or not low_seeds:
        return None

    comparisons = {}
    max_droughts = max(
        len(r['drought_events']) for r in seed_results.values()
    )

    for d_idx in range(max_droughts):
        high_stats = []
        low_stats = []

        for seed in high_seeds:
            if seed in seed_results and d_idx < len(seed_results[seed]['drought_events']):
                high_stats.append(seed_results[seed]['drought_events'][d_idx])
        for seed in low_seeds:
            if seed in seed_results and d_idx < len(seed_results[seed]['drought_events']):
                low_stats.append(seed_results[seed]['drought_events'][d_idx])

        if not high_stats or not low_stats:
            continue

        comp = {
            'drought_index': d_idx,
            'cycle': high_stats[0]['cycle'] if high_stats else None,
        }

        # Compare bounces
        high_bounces = [s['bounce'] for s in high_stats]
        low_bounces = [s['bounce'] for s in low_stats]
        if len(high_bounces) >= 2 and len(low_bounces) >= 2:
            t, p = stats.ttest_ind(high_bounces, low_bounces)
            comp['bounce'] = {
                'high_mean': float(np.mean(high_bounces)),
                'low_mean': float(np.mean(low_bounces)),
                't': float(t), 'p': float(p),
            }

        # Compare survival rates
        high_surv = [s['survivor_stats']['survival_rate'] for s in high_stats]
        low_surv = [s['survivor_stats']['survival_rate'] for s in low_stats]
        if len(high_surv) >= 2 and len(low_surv) >= 2:
            t, p = stats.ttest_ind(high_surv, low_surv)
            comp['survival_rate'] = {
                'high_mean': float(np.mean(high_surv)),
                'low_mean': float(np.mean(low_surv)),
                't': float(t), 'p': float(p),
            }

        # Compare post-drought eff_rank
        high_er = [s['post_eff_rank'] for s in high_stats]
        low_er = [s['post_eff_rank'] for s in low_stats]
        if len(high_er) >= 2 and len(low_er) >= 2:
            t, p = stats.ttest_ind(high_er, low_er)
            comp['post_eff_rank'] = {
                'high_mean': float(np.mean(high_er)),
                'low_mean': float(np.mean(low_er)),
                't': float(t), 'p': float(p),
            }

        comparisons[f'drought_{d_idx}'] = comp

    return comparisons


def run_analysis(results_dir):
    """Full V32 analysis pipeline."""
    print(f"Loading results from {results_dir}...")
    seed_results = load_all_results(results_dir)
    print(f"  Found {len(seed_results)} seeds")

    if len(seed_results) < 5:
        print("Need at least 5 seeds for analysis.")
        return

    # 1. Classification
    categories = classify_seeds(seed_results)
    n = len(seed_results)
    print(f"\n{'='*70}")
    print(f"1. SEED CLASSIFICATION (n={n})")
    print(f"{'='*70}")
    print(f"  HIGH (Φ>0.10): {len(categories['HIGH'])}/{n} "
          f"({len(categories['HIGH'])/n:.0%})")
    print(f"  MOD  (0.07-0.10): {len(categories['MOD'])}/{n} "
          f"({len(categories['MOD'])/n:.0%})")
    print(f"  LOW  (<0.07): {len(categories['LOW'])}/{n} "
          f"({len(categories['LOW'])/n:.0%})")

    # 2. Trajectory divergence
    print(f"\n{'='*70}")
    print(f"2. TRAJECTORY DIVERGENCE")
    print(f"{'='*70}")
    trajectories = extract_trajectories(seed_results)
    result = find_divergence_point(trajectories, categories)
    if result:
        div_cycle, div_results = result
        print(f"  First significant divergence (p<0.05): cycle {div_cycle}")
        if div_cycle is not None:
            d = div_results[div_cycle]
            print(f"    HIGH mean Φ: {d['high_mean']:.4f}")
            print(f"    LOW mean Φ:  {d['low_mean']:.4f}")
            print(f"    Effect size: {d['effect_size']:.4f}")
            print(f"    t-stat:      {d['t']:.2f}")

        # Print trajectory at key cycles
        for c in [0, 4, 5, 9, 10, 14, 15, 19, 20, 24, 25, 29]:
            if c < len(div_results):
                d = div_results[c]
                sig = "*" if d['p'] < 0.05 else ""
                drought = " [D]" if c > 0 and c % 5 == 0 else ""
                print(f"    C{c:02d}{drought}: HIGH={d.get('high_mean',0):.4f} "
                      f"LOW={d.get('low_mean',0):.4f} "
                      f"p={d['p']:.3f}{sig}")
    else:
        print("  Insufficient data for divergence analysis")

    # 3. Drought predictors
    print(f"\n{'='*70}")
    print(f"3. PREDICTORS OF LATE-PHASE Φ")
    print(f"{'='*70}")
    predictors, features = analyze_drought_predictors(seed_results, categories)
    print(f"  Correlations with late_mean_phi (sorted by |r|):")
    for name, info in list(predictors.items())[:20]:
        sig = "***" if info['p'] < 0.001 else "**" if info['p'] < 0.01 else "*" if info['p'] < 0.05 else ""
        print(f"    {name:30s}  r={info['r']:+.3f}  p={info['p']:.4f} {sig}")

    # 4. Survivor comparison across categories
    print(f"\n{'='*70}")
    print(f"4. HIGH vs LOW — DROUGHT-BY-DROUGHT")
    print(f"{'='*70}")
    comparisons = analyze_survivor_genome_diff(seed_results, categories)
    if comparisons:
        for key, comp in comparisons.items():
            print(f"\n  {key} (cycle {comp.get('cycle', '?')}):")
            for metric in ['bounce', 'survival_rate', 'post_eff_rank']:
                if metric in comp:
                    m = comp[metric]
                    sig = "*" if m['p'] < 0.05 else ""
                    print(f"    {metric:20s}: HIGH={m['high_mean']:.3f} "
                          f"LOW={m['low_mean']:.3f} p={m['p']:.3f}{sig}")

    # 5. Key finding summary
    print(f"\n{'='*70}")
    print(f"5. KEY FINDINGS")
    print(f"{'='*70}")

    # Is the 30/70 stable?
    high_frac = len(categories['HIGH']) / n
    print(f"\n  a) Distribution stability:")
    print(f"     HIGH fraction: {high_frac:.0%} (V31 was 30%)")
    print(f"     {'STABLE' if abs(high_frac - 0.30) < 0.10 else 'SHIFTED'} "
          f"at 50-seed scale")

    # Best predictor?
    if predictors:
        best_name = next(iter(predictors))
        best = predictors[best_name]
        print(f"\n  b) Best predictor of late Φ:")
        print(f"     {best_name}: r={best['r']:+.3f} (p={best['p']:.4f})")
        if best['p'] < 0.05:
            if 'pre_d_' in best_name or best_name.startswith('d0_'):
                print(f"     → EARLY DETERMINISM: measured before/during first drought")
            elif 'bounce' in best_name:
                print(f"     → TRAJECTORY-DEPENDENT: drought response is key")
            else:
                print(f"     → OUTCOME-CORRELATED: correlated but not necessarily causal")

    # Phase transition or noise?
    late_phis_arr = np.array([seed_results[s]['summary']['late_mean_phi']
                              for s in sorted(seed_results.keys())])
    # Bimodality test (Hartigan's dip test approximation via kurtosis)
    kurtosis = float(stats.kurtosis(late_phis_arr))
    skewness = float(stats.skew(late_phis_arr))
    print(f"\n  c) Distribution shape:")
    print(f"     Kurtosis: {kurtosis:.2f} (negative = bimodal, positive = peaked)")
    print(f"     Skewness: {skewness:.2f}")
    if kurtosis < -0.5:
        print(f"     → BIMODAL: consistent with phase transition")
    elif kurtosis > 0.5:
        print(f"     → PEAKED: consistent with continuous variation")
    else:
        print(f"     → AMBIGUOUS: neither clearly bimodal nor peaked")

    # Save full analysis
    analysis = {
        'n_seeds': n,
        'distribution': {
            'HIGH': len(categories['HIGH']),
            'MOD': len(categories['MOD']),
            'LOW': len(categories['LOW']),
            'HIGH_seeds': sorted(categories['HIGH']),
            'MOD_seeds': sorted(categories['MOD']),
            'LOW_seeds': sorted(categories['LOW']),
        },
        'divergence_cycle': div_cycle if result else None,
        'predictors': predictors,
        'drought_comparisons': comparisons,
        'distribution_shape': {
            'kurtosis': kurtosis,
            'skewness': skewness,
            'mean': float(np.mean(late_phis_arr)),
            'std': float(np.std(late_phis_arr)),
            'median': float(np.median(late_phis_arr)),
        },
    }

    analysis_path = os.path.join(results_dir, 'v32_analysis.json')
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\n  Analysis saved to {analysis_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python v32_analysis.py <results_directory>")
        sys.exit(1)
    run_analysis(sys.argv[1])
