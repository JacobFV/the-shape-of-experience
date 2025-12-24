"""
Analysis tools for testing the six-dimensional affect framework.

Includes:
- Factor analysis for dimension independence
- Correlation analysis for valence-viability
- Time series analysis for temporal dynamics
- Power analysis for study planning
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.linalg import svd


@dataclass
class FactorAnalysisResult:
    """Results from confirmatory factor analysis."""
    n_factors: int
    loadings: np.ndarray
    factor_correlations: np.ndarray
    variance_explained: np.ndarray
    fit_indices: Dict[str, float]
    dimension_names: List[str]


@dataclass
class CorrelationResult:
    """Results from correlation analysis."""
    r: float
    p: float
    ci_low: float
    ci_high: float
    n: int


def compute_effective_rank(covariance_matrix: np.ndarray) -> float:
    """
    Compute effective rank from covariance matrix.

    r_eff = (tr(C))^2 / tr(C^2) = (sum(λ))^2 / sum(λ^2)

    This is the key operationalization from the thesis:
    effective rank measures how distributed vs concentrated
    the active degrees of freedom are.
    """
    eigenvalues = np.linalg.eigvalsh(covariance_matrix)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove numerical zeros

    trace = np.sum(eigenvalues)
    trace_squared = np.sum(eigenvalues ** 2)

    if trace_squared < 1e-10:
        return 0.0

    return (trace ** 2) / trace_squared


def compute_integration_proxy(time_series: np.ndarray, partition_size: int = None) -> float:
    """
    Compute a proxy for integrated information (Φ) from time series data.

    This is a simplified approximation using mutual information between
    partitions of the system. True Φ computation is computationally intractable
    for large systems.

    Args:
        time_series: Shape (n_timesteps, n_variables)
        partition_size: Size of each partition (default: half the variables)

    Returns:
        Proxy for integration (higher = more integrated)
    """
    n_timesteps, n_vars = time_series.shape

    if partition_size is None:
        partition_size = n_vars // 2

    # Compute whole-system entropy
    cov_whole = np.cov(time_series.T)
    entropy_whole = 0.5 * np.log(np.linalg.det(cov_whole) + 1e-10)

    # Compute partition entropies
    partitions = [
        time_series[:, :partition_size],
        time_series[:, partition_size:]
    ]

    entropy_parts = 0
    for part in partitions:
        if part.shape[1] > 0:
            cov_part = np.cov(part.T)
            if cov_part.ndim == 0:
                cov_part = np.array([[cov_part]])
            entropy_parts += 0.5 * np.log(np.linalg.det(cov_part) + 1e-10)

    # Integration = whole entropy - sum of part entropies (mutual information)
    # Higher means more information is lost by partitioning
    integration = entropy_whole - entropy_parts

    # Normalize to [0, 1] approximately
    return 1 / (1 + np.exp(-integration))


def test_dimension_independence(
    data: np.ndarray,
    dimension_names: List[str] = None
) -> FactorAnalysisResult:
    """
    Test whether the six dimensions are empirically distinguishable.

    Args:
        data: Shape (n_observations, 6) - scores on each dimension
        dimension_names: Names for each dimension

    Returns:
        Factor analysis results including fit indices
    """
    if dimension_names is None:
        dimension_names = [
            "Valence", "Arousal", "Integration",
            "Effective_Rank", "Counterfactual_Weight", "Self_Model_Salience"
        ]

    n_obs, n_dims = data.shape
    assert n_dims == 6, "Expected 6 dimensions"

    # Standardize
    data_std = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-10)

    # Compute correlation matrix
    corr_matrix = np.corrcoef(data_std.T)

    # SVD for factor structure
    U, s, Vt = svd(corr_matrix)

    # Compute variance explained
    var_explained = s ** 2 / np.sum(s ** 2)

    # Simple fit indices
    # Kaiser criterion: how many eigenvalues > 1
    n_factors_kaiser = np.sum(s > 1)

    # Loadings (first 6 components)
    loadings = U * s

    # Factor correlations
    factor_corr = np.corrcoef(loadings.T)

    # Approximate fit indices
    # RMSEA approximation
    residual = corr_matrix - loadings @ loadings.T
    rmsea = np.sqrt(np.mean(residual ** 2))

    # CFI approximation (compare to null model)
    null_model_error = np.sqrt(np.mean((corr_matrix - np.eye(n_dims)) ** 2))
    cfi = 1 - (rmsea / null_model_error) if null_model_error > 0 else 1.0

    fit_indices = {
        "rmsea": rmsea,
        "cfi": cfi,
        "n_factors_kaiser": n_factors_kaiser,
        "variance_explained_6": np.sum(var_explained[:6]),
    }

    return FactorAnalysisResult(
        n_factors=6,
        loadings=loadings,
        factor_correlations=factor_corr,
        variance_explained=var_explained,
        fit_indices=fit_indices,
        dimension_names=dimension_names,
    )


def correlate_with_confidence(x: np.ndarray, y: np.ndarray) -> CorrelationResult:
    """Compute correlation with confidence interval."""
    n = len(x)
    r, p = stats.pearsonr(x, y)

    # Fisher z-transform for CI
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_low = z - 1.96 * se
    z_high = z + 1.96 * se

    ci_low = np.tanh(z_low)
    ci_high = np.tanh(z_high)

    return CorrelationResult(r=r, p=p, ci_low=ci_low, ci_high=ci_high, n=n)


def test_valence_viability_correlation(
    valence_scores: np.ndarray,
    hrv: np.ndarray,
    cortisol: np.ndarray,
    inflammatory: Optional[np.ndarray] = None
) -> Dict[str, CorrelationResult]:
    """
    Test the core prediction: valence correlates with physiological viability.

    Args:
        valence_scores: Self-reported valence
        hrv: Heart rate variability (higher = more viable)
        cortisol: Cortisol levels (lower = more viable)
        inflammatory: Inflammatory markers like IL-6 (lower = more viable)

    Returns:
        Correlation results for each physiological marker
    """
    results = {}

    # Valence should positively correlate with HRV
    results["valence_hrv"] = correlate_with_confidence(valence_scores, hrv)

    # Valence should negatively correlate with cortisol
    results["valence_cortisol"] = correlate_with_confidence(valence_scores, -cortisol)

    if inflammatory is not None:
        # Valence should negatively correlate with inflammation
        results["valence_inflammatory"] = correlate_with_confidence(valence_scores, -inflammatory)

    return results


def power_analysis_correlation(
    expected_r: float,
    alpha: float = 0.05,
    power: float = 0.80
) -> int:
    """
    Compute required sample size for detecting a correlation.

    Args:
        expected_r: Expected correlation coefficient
        alpha: Significance level
        power: Desired statistical power

    Returns:
        Required sample size
    """
    # Using Fisher z approximation
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    z_r = np.arctanh(expected_r)

    n = ((z_alpha + z_beta) / z_r) ** 2 + 3
    return int(np.ceil(n))


def temporal_precedence_test(
    valence: np.ndarray,
    physiology: np.ndarray,
    max_lag: int = 10
) -> Dict[str, float]:
    """
    Test whether valence changes precede physiological changes.

    This tests the prediction that subjective threat detection
    (negative valence) occurs before objective physiological markers.

    Args:
        valence: Time series of valence ratings
        physiology: Time series of physiological stress marker
        max_lag: Maximum lag to test (in samples)

    Returns:
        Cross-correlation at each lag, peak lag
    """
    results = {}

    # Compute cross-correlation at different lags
    n = len(valence)
    correlations = []

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            # Valence leads physiology
            v = valence[:lag]
            p = physiology[-lag:]
        elif lag > 0:
            # Physiology leads valence
            v = valence[lag:]
            p = physiology[:-lag]
        else:
            v = valence
            p = physiology

        if len(v) > 3:
            r, _ = stats.pearsonr(v, p)
            correlations.append((lag, r))

    correlations = sorted(correlations, key=lambda x: abs(x[1]), reverse=True)

    results["peak_lag"] = correlations[0][0]
    results["peak_correlation"] = correlations[0][1]
    results["all_correlations"] = dict(correlations)

    # Thesis prediction: peak should be at negative lag (valence precedes)
    results["valence_precedes"] = correlations[0][0] < 0

    return results


def simulate_affect_trajectory(
    n_timesteps: int = 100,
    initial_state: np.ndarray = None,
    noise_scale: float = 0.1,
    attractor: str = "neutral"
) -> np.ndarray:
    """
    Simulate a trajectory through affect space.

    This can be used to test analysis code and generate example data.

    Args:
        n_timesteps: Number of time points
        initial_state: Starting affect state (6-dim)
        noise_scale: Standard deviation of noise
        attractor: Type of attractor ("neutral", "depression", "anxiety", "flow")

    Returns:
        Array of shape (n_timesteps, 6)
    """
    if initial_state is None:
        initial_state = np.array([0.0, 0.5, 0.5, 0.5, 0.5, 0.5])

    # Define attractors
    attractors = {
        "neutral": np.array([0.0, 0.5, 0.6, 0.5, 0.4, 0.4]),
        "depression": np.array([-0.5, 0.2, 0.5, 0.2, 0.6, 0.8]),
        "anxiety": np.array([-0.6, 0.8, 0.6, 0.3, 0.8, 0.7]),
        "flow": np.array([0.7, 0.5, 0.8, 0.5, 0.1, 0.1]),
    }

    target = attractors.get(attractor, attractors["neutral"])

    trajectory = np.zeros((n_timesteps, 6))
    trajectory[0] = initial_state

    for t in range(1, n_timesteps):
        # Drift toward attractor
        drift = 0.1 * (target - trajectory[t-1])

        # Add noise
        noise = noise_scale * np.random.randn(6)

        # Update with bounds
        trajectory[t] = np.clip(trajectory[t-1] + drift + noise, -1, 1)

    return trajectory
