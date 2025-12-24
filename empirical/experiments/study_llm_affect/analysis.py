"""
Analysis tools for testing theoretical predictions from the thesis.

Tests whether:
1. Affect dimensions cluster by induced condition
2. Specific scenarios produce expected signatures
3. Fear and curiosity are distinguished by valence (given matched CF)
4. The 6-factor model captures more structure than 2-factor (valence-arousal)
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy import stats
from scipy.spatial.distance import cdist

from .affect_calculator import AffectMeasurement
from .scenarios import THESIS_PREDICTIONS, ExpectedSignature


@dataclass
class ClusteringResult:
    """Results from clustering analysis."""
    silhouette_score: float
    cluster_labels: np.ndarray
    cluster_centers: np.ndarray
    within_cluster_variance: float
    between_cluster_variance: float
    f_statistic: float
    p_value: float


@dataclass
class SignatureMatchResult:
    """Results from signature matching."""
    scenario: str
    expected: Dict[str, Optional[float]]
    measured: Dict[str, float]
    dimension_matches: Dict[str, bool]
    overall_match: bool
    cosine_similarity: float
    euclidean_distance: float


@dataclass
class FalsificationTestResult:
    """Results from a falsification test."""
    test_name: str
    hypothesis: str
    result: str  # "supported", "falsified", "inconclusive"
    statistic: float
    p_value: float
    effect_size: float
    details: Dict[str, any]


def compute_signature_match(
    measured: AffectMeasurement,
    expected: ExpectedSignature,
    tolerance: float = 0.3
) -> SignatureMatchResult:
    """
    Compare measured affect to expected signature.

    Args:
        measured: Measured affect
        expected: Expected signature from theory
        tolerance: How close is "close enough"

    Returns:
        SignatureMatchResult
    """
    measured_dict = measured.to_dict()
    expected_dict = expected.to_dict()

    dimension_matches = {}
    for dim, exp_val in expected_dict.items():
        if exp_val is None:
            dimension_matches[dim] = True  # No prediction
        else:
            meas_val = measured_dict[dim]
            # Match if same sign and within tolerance
            same_sign = (exp_val * meas_val >= 0) or abs(exp_val) < tolerance
            within_tol = abs(meas_val - exp_val) < tolerance
            dimension_matches[dim] = same_sign and within_tol

    overall_match = all(dimension_matches.values())

    # Compute similarity metrics
    meas_vec = np.array([measured_dict[d] for d in measured_dict])
    exp_vec = np.array([v if v is not None else 0 for v in expected_dict.values()])
    mask = np.array([v is not None for v in expected_dict.values()])

    if mask.sum() > 0:
        meas_masked = meas_vec[mask]
        exp_masked = exp_vec[mask]
        cosine = np.dot(meas_masked, exp_masked) / (
            np.linalg.norm(meas_masked) * np.linalg.norm(exp_masked) + 1e-8
        )
        euclidean = np.linalg.norm(meas_masked - exp_masked)
    else:
        cosine = 1.0
        euclidean = 0.0

    return SignatureMatchResult(
        scenario="",  # Set by caller
        expected=expected_dict,
        measured=measured_dict,
        dimension_matches=dimension_matches,
        overall_match=overall_match,
        cosine_similarity=float(cosine),
        euclidean_distance=float(euclidean)
    )


def test_clustering(
    affect_data: Dict[str, np.ndarray],
    n_clusters: Optional[int] = None
) -> ClusteringResult:
    """
    Test whether affect measurements cluster by condition.

    Args:
        affect_data: Dict of condition_name -> (n_samples, 6) affect vectors
        n_clusters: Number of clusters (defaults to number of conditions)

    Returns:
        ClusteringResult
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # Combine all data
    all_data = []
    true_labels = []
    label_map = {}

    for i, (condition, data) in enumerate(affect_data.items()):
        all_data.append(data)
        true_labels.extend([i] * len(data))
        label_map[i] = condition

    X = np.vstack(all_data)
    y_true = np.array(true_labels)

    n_clusters = n_clusters or len(affect_data)

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X)

    # Silhouette score
    sil_score = silhouette_score(X, y_pred)

    # Within vs between cluster variance
    within_var = 0
    for i in range(n_clusters):
        cluster_points = X[y_pred == i]
        if len(cluster_points) > 0:
            within_var += np.var(cluster_points, axis=0).sum()
    within_var /= n_clusters

    overall_mean = X.mean(axis=0)
    between_var = 0
    for i in range(n_clusters):
        cluster_points = X[y_pred == i]
        if len(cluster_points) > 0:
            cluster_mean = cluster_points.mean(axis=0)
            between_var += len(cluster_points) * np.sum((cluster_mean - overall_mean) ** 2)
    between_var /= len(X)

    # F-statistic
    f_stat = between_var / (within_var + 1e-8)

    # ANOVA-like p-value (simplified)
    df_between = n_clusters - 1
    df_within = len(X) - n_clusters
    p_value = 1 - stats.f.cdf(f_stat, df_between, df_within)

    return ClusteringResult(
        silhouette_score=float(sil_score),
        cluster_labels=y_pred,
        cluster_centers=kmeans.cluster_centers_,
        within_cluster_variance=float(within_var),
        between_cluster_variance=float(between_var),
        f_statistic=float(f_stat),
        p_value=float(p_value)
    )


def test_dimension_independence(
    affect_data: np.ndarray,
    n_factors: int = 6
) -> Dict[str, float]:
    """
    Test whether 6-factor model captures more structure than 2-factor.

    Uses PCA to compare variance explained.

    Args:
        affect_data: (n_samples, 6) affect vectors
        n_factors: Number of factors in full model

    Returns:
        Dict with fit statistics
    """
    from sklearn.decomposition import PCA

    # 6-factor model (full)
    pca_6 = PCA(n_components=min(6, affect_data.shape[1]))
    pca_6.fit(affect_data)
    var_6 = pca_6.explained_variance_ratio_.sum()

    # 2-factor model (valence-arousal only)
    pca_2 = PCA(n_components=2)
    pca_2.fit(affect_data)
    var_2 = pca_2.explained_variance_ratio_.sum()

    # How much variance is captured by dimensions 3-6?
    additional_variance = var_6 - var_2

    # Effective rank of the data
    eigenvalues = pca_6.explained_variance_
    eff_rank = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()

    return {
        "variance_6_factor": float(var_6),
        "variance_2_factor": float(var_2),
        "additional_variance_3_6": float(additional_variance),
        "effective_rank": float(eff_rank),
        "eigenvalues": pca_6.explained_variance_.tolist(),
        "explained_variance_ratio": pca_6.explained_variance_ratio_.tolist()
    }


def test_fear_curiosity_distinction(
    fear_data: np.ndarray,
    curiosity_data: np.ndarray
) -> FalsificationTestResult:
    """
    Test that fear and curiosity are distinguished by valence given matched CF.

    Theory predicts: Both high CF, but fear is negative valence, curiosity positive.

    Args:
        fear_data: (n_samples, 6) affect vectors from fear scenario
        curiosity_data: (n_samples, 6) affect vectors from curiosity scenario

    Returns:
        FalsificationTestResult
    """
    # Dimensions: valence=0, arousal=1, integration=2, rank=3, cf=4, sm=5

    # Test 1: Both should have high CF
    fear_cf = fear_data[:, 4].mean()
    curiosity_cf = curiosity_data[:, 4].mean()

    # Test 2: Valence should differ
    fear_val = fear_data[:, 0].mean()
    curiosity_val = curiosity_data[:, 0].mean()

    # T-test on valence
    t_stat, p_value = stats.ttest_ind(fear_data[:, 0], curiosity_data[:, 0])

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        (fear_data[:, 0].var() + curiosity_data[:, 0].var()) / 2
    )
    cohens_d = (curiosity_val - fear_val) / (pooled_std + 1e-8)

    # Decision logic
    cf_matched = abs(fear_cf - curiosity_cf) < 0.3  # Both high or close
    valence_differs = p_value < 0.05 and curiosity_val > fear_val

    if cf_matched and valence_differs:
        result = "supported"
    elif not cf_matched:
        result = "inconclusive"  # CF wasn't matched
    else:
        result = "falsified"

    return FalsificationTestResult(
        test_name="fear_curiosity_distinction",
        hypothesis="Fear and curiosity both show high CF but differ in valence",
        result=result,
        statistic=float(t_stat),
        p_value=float(p_value),
        effect_size=float(cohens_d),
        details={
            "fear_valence_mean": float(fear_val),
            "curiosity_valence_mean": float(curiosity_val),
            "fear_cf_mean": float(fear_cf),
            "curiosity_cf_mean": float(curiosity_cf),
            "cf_matched": cf_matched,
            "valence_differs": valence_differs
        }
    )


def test_hopelessness_signature(
    data: np.ndarray
) -> FalsificationTestResult:
    """
    Test that hopelessness shows predicted signature.

    Expected: negative valence, low effective rank, high self-model salience, high CF.
    """
    # Mean values
    valence = data[:, 0].mean()
    rank = data[:, 3].mean()
    sm = data[:, 5].mean()
    cf = data[:, 4].mean()

    # Expected signs
    valence_negative = valence < 0
    rank_low = rank < 0.5
    sm_high = sm > 0.5
    cf_high = cf > 0.5

    matches = sum([valence_negative, rank_low, sm_high, cf_high])

    if matches >= 4:
        result = "supported"
    elif matches >= 2:
        result = "inconclusive"
    else:
        result = "falsified"

    return FalsificationTestResult(
        test_name="hopelessness_signature",
        hypothesis="Hopelessness shows negative valence, low rank, high SM, high CF",
        result=result,
        statistic=float(matches),
        p_value=float("nan"),  # Not a statistical test
        effect_size=float(matches / 4),
        details={
            "valence_mean": float(valence),
            "rank_mean": float(rank),
            "self_model_mean": float(sm),
            "cf_mean": float(cf),
            "valence_negative": valence_negative,
            "rank_low": rank_low,
            "sm_high": sm_high,
            "cf_high": cf_high
        }
    )


def test_flow_signature(data: np.ndarray) -> FalsificationTestResult:
    """
    Test that flow shows predicted signature.

    Expected: positive valence, high rank, low self-model salience, low CF.
    """
    valence = data[:, 0].mean()
    rank = data[:, 3].mean()
    sm = data[:, 5].mean()
    cf = data[:, 4].mean()

    valence_positive = valence > 0
    rank_high = rank > 0.5
    sm_low = sm < 0.5
    cf_low = cf < 0.5

    matches = sum([valence_positive, rank_high, sm_low, cf_low])

    if matches >= 4:
        result = "supported"
    elif matches >= 2:
        result = "inconclusive"
    else:
        result = "falsified"

    return FalsificationTestResult(
        test_name="flow_signature",
        hypothesis="Flow shows positive valence, high rank, low SM, low CF",
        result=result,
        statistic=float(matches),
        p_value=float("nan"),
        effect_size=float(matches / 4),
        details={
            "valence_mean": float(valence),
            "rank_mean": float(rank),
            "self_model_mean": float(sm),
            "cf_mean": float(cf),
            "valence_positive": valence_positive,
            "rank_high": rank_high,
            "sm_low": sm_low,
            "cf_low": cf_low
        }
    )


def run_full_analysis(
    results_dir: str,
    output_file: Optional[str] = None
) -> Dict[str, any]:
    """
    Run full analysis on collected results.

    Args:
        results_dir: Directory containing result JSON files
        output_file: Optional path to save analysis

    Returns:
        Dict with all analysis results
    """
    results_path = Path(results_dir)

    # Load all results
    all_data = {}
    for json_file in results_path.glob("*.json"):
        if json_file.name == "summary.json":
            continue
        with open(json_file) as f:
            data = json.load(f)

        scenario = data["scenario_name"]
        if scenario not in all_data:
            all_data[scenario] = []

        # Extract affect trajectory
        affects = np.array([
            [t["affect"][dim] for dim in
             ["valence", "arousal", "integration",
              "effective_rank", "counterfactual_weight", "self_model_salience"]]
            for t in data["turns"]
        ])
        all_data[scenario].append(affects)

    # Aggregate data
    scenario_means = {}
    for scenario, runs in all_data.items():
        all_affects = np.vstack(runs)
        scenario_means[scenario] = all_affects

    analysis = {}

    # 1. Clustering test
    if len(scenario_means) >= 2:
        clustering = test_clustering(scenario_means)
        analysis["clustering"] = {
            "silhouette_score": clustering.silhouette_score,
            "f_statistic": clustering.f_statistic,
            "p_value": clustering.p_value,
            "interpretation": "supported" if clustering.p_value < 0.05 else "inconclusive"
        }

    # 2. Dimension independence
    all_data_combined = np.vstack(list(scenario_means.values()))
    dim_test = test_dimension_independence(all_data_combined)
    analysis["dimension_independence"] = dim_test

    # 3. Specific signature tests
    if "hopelessness" in scenario_means:
        analysis["hopelessness_test"] = test_hopelessness_signature(
            scenario_means["hopelessness"]
        ).__dict__

    if "flow" in scenario_means:
        analysis["flow_test"] = test_flow_signature(
            scenario_means["flow"]
        ).__dict__

    if "threat" in scenario_means and "curiosity" in scenario_means:
        analysis["fear_curiosity_test"] = test_fear_curiosity_distinction(
            scenario_means["threat"],
            scenario_means["curiosity"]
        ).__dict__

    # Save if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"Analysis saved to {output_file}")

    return analysis


def print_analysis_report(analysis: Dict[str, any]):
    """Print a human-readable analysis report."""
    print("\n" + "=" * 70)
    print("AFFECT FRAMEWORK ANALYSIS REPORT")
    print("=" * 70)

    # Clustering
    if "clustering" in analysis:
        c = analysis["clustering"]
        print("\n1. CLUSTERING BY CONDITION")
        print("-" * 40)
        print(f"   Silhouette Score: {c['silhouette_score']:.3f}")
        print(f"   F-statistic: {c['f_statistic']:.2f}")
        print(f"   p-value: {c['p_value']:.4f}")
        print(f"   Result: {c['interpretation'].upper()}")

    # Dimension independence
    if "dimension_independence" in analysis:
        d = analysis["dimension_independence"]
        print("\n2. DIMENSION INDEPENDENCE (6-factor vs 2-factor)")
        print("-" * 40)
        print(f"   6-factor variance explained: {d['variance_6_factor']:.1%}")
        print(f"   2-factor variance explained: {d['variance_2_factor']:.1%}")
        print(f"   Additional variance (dims 3-6): {d['additional_variance_3_6']:.1%}")
        print(f"   Effective rank: {d['effective_rank']:.2f}")
        result = "6-factor captures significantly more" if d['additional_variance_3_6'] > 0.1 else "2-factor may suffice"
        print(f"   Interpretation: {result}")

    # Signature tests
    for test_name in ["hopelessness_test", "flow_test", "fear_curiosity_test"]:
        if test_name in analysis:
            t = analysis[test_name]
            print(f"\n3. {test_name.upper().replace('_', ' ')}")
            print("-" * 40)
            print(f"   Hypothesis: {t['hypothesis']}")
            print(f"   Result: {t['result'].upper()}")
            if 'details' in t:
                for k, v in t['details'].items():
                    if isinstance(v, float):
                        print(f"   {k}: {v:.3f}")
                    else:
                        print(f"   {k}: {v}")

    print("\n" + "=" * 70)
    print("END REPORT")
    print("=" * 70)
