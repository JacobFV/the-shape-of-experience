"""
V10: RSA + Ablation Analysis
==============================

The core test: does the distance structure in the information-theoretic
affect space match the distance structure in the embedding-predicted
affect space?

Implements:
- RSA (Representational Similarity Analysis) with Mantel test
- CKA (Centered Kernel Alignment)
- MDS visualization of both spaces
- Ablation comparison plots
- Per-dimension analysis

This is the heart of the Geometric Alignment hypothesis.
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import os


# ============================================================================
# RSA (Representational Similarity Analysis)
# ============================================================================

def compute_rdm(X: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    """
    Compute Representational Dissimilarity Matrix.

    Args:
        X: (N, d) matrix of N representations in d dimensions
        metric: distance metric ('euclidean', 'correlation', 'cosine')

    Returns: (N, N) symmetric distance matrix
    """
    dists = pdist(X, metric=metric)
    return squareform(dists)


def rsa_correlation(
    D_a: np.ndarray,
    D_e: np.ndarray,
    method: str = 'spearman',
) -> Tuple[float, float]:
    """
    Compute RSA correlation between two RDMs.

    Args:
        D_a: (N, N) RDM from info-theoretic affect space
        D_e: (N, N) RDM from embedding-predicted affect space
        method: 'spearman' or 'pearson'

    Returns: (correlation, p_value)
    """
    # Extract upper triangle (excluding diagonal)
    n = D_a.shape[0]
    upper_tri = np.triu_indices(n, k=1)
    vec_a = D_a[upper_tri]
    vec_e = D_e[upper_tri]

    if method == 'spearman':
        rho, p = spearmanr(vec_a, vec_e)
    elif method == 'pearson':
        rho, p = pearsonr(vec_a, vec_e)
    else:
        raise ValueError(f"Unknown method: {method}")

    return float(rho), float(p)


def mantel_test(
    D_a: np.ndarray,
    D_e: np.ndarray,
    n_permutations: int = 10000,
    method: str = 'spearman',
) -> Tuple[float, float, np.ndarray]:
    """
    Mantel test: permutation-based significance for RSA.

    Permutes rows/columns of one RDM, recomputes correlation.
    P-value = fraction of permuted correlations >= observed.

    Returns: (observed_rho, p_value, null_distribution)
    """
    observed_rho, _ = rsa_correlation(D_a, D_e, method)

    n = D_a.shape[0]
    null_rhos = np.zeros(n_permutations)

    for i in range(n_permutations):
        perm = np.random.permutation(n)
        D_e_perm = D_e[perm][:, perm]
        null_rhos[i], _ = rsa_correlation(D_a, D_e_perm, method)

    p_value = np.mean(null_rhos >= observed_rho)

    return observed_rho, p_value, null_rhos


# ============================================================================
# CKA (Centered Kernel Alignment)
# ============================================================================

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute Linear CKA (Centered Kernel Alignment) between two
    representation matrices.

    Kornblith et al. (2019). Similarity of Neural Network Representations
    Revisited.

    Args:
        X: (N, p) representations in space A
        Y: (N, q) representations in space B

    Returns: CKA value in [0, 1]
    """
    # Center
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Gram matrices
    K = X @ X.T
    L = Y @ Y.T

    # HSIC
    hsic_xy = np.sum(K * L)
    hsic_xx = np.sum(K * K)
    hsic_yy = np.sum(L * L)

    if hsic_xx * hsic_yy == 0:
        return 0.0

    return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy))


def kernel_cka(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
    """
    CKA with RBF kernel.

    Args:
        X, Y: representation matrices
        sigma: RBF kernel bandwidth

    Returns: CKA value
    """
    from scipy.spatial.distance import cdist

    K = np.exp(-cdist(X, X, 'sqeuclidean') / (2 * sigma ** 2))
    L = np.exp(-cdist(Y, Y, 'sqeuclidean') / (2 * sigma ** 2))

    # Center kernels
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K = H @ K @ H
    L = H @ L @ H

    hsic_xy = np.sum(K * L) / (n - 1) ** 2
    hsic_xx = np.sum(K * K) / (n - 1) ** 2
    hsic_yy = np.sum(L * L) / (n - 1) ** 2

    if hsic_xx * hsic_yy == 0:
        return 0.0

    return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy))


# ============================================================================
# Full analysis pipeline
# ============================================================================

@dataclass
class RSAResult:
    """Results from RSA analysis."""
    rho_spearman: float
    p_value_spearman: float
    rho_pearson: float
    p_value_pearson: float
    mantel_rho: float
    mantel_p: float
    cka_linear: float
    cka_rbf: float
    n_samples: int
    null_distribution: Optional[np.ndarray] = None

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.mantel_p < alpha

    def summary(self) -> str:
        sig = "***" if self.mantel_p < 0.001 else "**" if self.mantel_p < 0.01 else "*" if self.mantel_p < 0.05 else "ns"
        return (f"RSA: ρ={self.mantel_rho:.3f} (p={self.mantel_p:.4f}) {sig} | "
                f"CKA_lin={self.cka_linear:.3f} | CKA_rbf={self.cka_rbf:.3f} | "
                f"N={self.n_samples}")

    def to_dict(self) -> Dict:
        return {
            'rho_spearman': self.rho_spearman,
            'p_value_spearman': self.p_value_spearman,
            'rho_pearson': self.rho_pearson,
            'p_value_pearson': self.p_value_pearson,
            'mantel_rho': self.mantel_rho,
            'mantel_p': self.mantel_p,
            'cka_linear': self.cka_linear,
            'cka_rbf': self.cka_rbf,
            'n_samples': self.n_samples,
        }


def run_rsa_analysis(
    affect_matrix: np.ndarray,
    embedding_affect_matrix: np.ndarray,
    n_permutations: int = 10000,
    subsample: Optional[int] = None,
) -> RSAResult:
    """
    Run full RSA analysis.

    Args:
        affect_matrix: (N, 6) info-theoretic affect vectors
        embedding_affect_matrix: (N, d) embedding-predicted affect vectors
        n_permutations: for Mantel test
        subsample: if set, randomly subsample to this many points

    Returns: RSAResult
    """
    N = len(affect_matrix)
    assert len(embedding_affect_matrix) == N

    # Subsample if needed (large N makes Mantel test slow)
    if subsample and N > subsample:
        idx = np.random.choice(N, subsample, replace=False)
        affect_matrix = affect_matrix[idx]
        embedding_affect_matrix = embedding_affect_matrix[idx]
        N = subsample

    # Compute RDMs
    D_a = compute_rdm(affect_matrix, 'euclidean')
    D_e = compute_rdm(embedding_affect_matrix, 'euclidean')

    # RSA correlations
    rho_sp, p_sp = rsa_correlation(D_a, D_e, 'spearman')
    rho_pe, p_pe = rsa_correlation(D_a, D_e, 'pearson')

    # Mantel test
    mantel_rho, mantel_p, null_dist = mantel_test(D_a, D_e, n_permutations, 'spearman')

    # CKA
    cka_lin = linear_cka(affect_matrix, embedding_affect_matrix)
    cka_r = kernel_cka(affect_matrix, embedding_affect_matrix)

    result = RSAResult(
        rho_spearman=rho_sp,
        p_value_spearman=p_sp,
        rho_pearson=rho_pe,
        p_value_pearson=p_pe,
        mantel_rho=mantel_rho,
        mantel_p=mantel_p,
        cka_linear=cka_lin,
        cka_rbf=cka_r,
        n_samples=N,
        null_distribution=null_dist,
    )

    print(result.summary())
    return result


# ============================================================================
# Per-dimension analysis
# ============================================================================

def per_dimension_correlation(
    affect_matrix: np.ndarray,
    embedding_affect_matrix: np.ndarray,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute marginal correlations for each affect dimension.

    This is weaker than RSA but useful for diagnostics.
    """
    from v10_affect import AffectVector
    dim_names = AffectVector.dim_names()

    results = {}
    for i, name in enumerate(dim_names):
        # For each affect dimension, check if nearby states in that
        # dimension are also nearby in embedding space
        dim_vals = affect_matrix[:, i]

        # Compute pairwise differences in this dimension
        D_dim = squareform(pdist(dim_vals.reshape(-1, 1)))
        D_emb = compute_rdm(embedding_affect_matrix)

        upper = np.triu_indices(len(dim_vals), k=1)
        rho, p = spearmanr(D_dim[upper], D_emb[upper])
        results[name] = (float(rho), float(p))

    return results


# ============================================================================
# Visualization
# ============================================================================

def plot_rsa_results(
    result: RSAResult,
    affect_matrix: np.ndarray,
    embedding_affect_matrix: np.ndarray,
    save_path: str = 'results/v10/rsa_analysis.png',
) -> str:
    """Create comprehensive RSA visualization."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. MDS of info-theoretic affect space
    ax = axes[0, 0]
    mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42, normalized_stress='auto')
    A_2d = mds.fit_transform(affect_matrix)
    scatter = ax.scatter(A_2d[:, 0], A_2d[:, 1], c=affect_matrix[:, 0],
                        cmap='RdYlGn', s=10, alpha=0.5)
    plt.colorbar(scatter, ax=ax, label='Valence')
    ax.set_title('Info-Theoretic Affect Space (MDS)')
    ax.set_xlabel('MDS-1')
    ax.set_ylabel('MDS-2')

    # 2. MDS of embedding-predicted affect space
    ax = axes[0, 1]
    E_2d = mds.fit_transform(embedding_affect_matrix)
    # Color by same valence for comparison
    scatter = ax.scatter(E_2d[:, 0], E_2d[:, 1], c=affect_matrix[:, 0],
                        cmap='RdYlGn', s=10, alpha=0.5)
    plt.colorbar(scatter, ax=ax, label='Valence')
    ax.set_title('Embedding-Predicted Affect Space (MDS)')
    ax.set_xlabel('MDS-1')
    ax.set_ylabel('MDS-2')

    # 3. RDM comparison
    ax = axes[0, 2]
    D_a = compute_rdm(affect_matrix[:200] if len(affect_matrix) > 200 else affect_matrix)
    D_e = compute_rdm(embedding_affect_matrix[:200] if len(embedding_affect_matrix) > 200 else embedding_affect_matrix)
    upper = np.triu_indices(D_a.shape[0], k=1)
    ax.scatter(D_a[upper], D_e[upper], s=1, alpha=0.1)
    ax.set_xlabel('Info-theoretic distance')
    ax.set_ylabel('Embedding-predicted distance')
    ax.set_title(f'RDM Correlation (ρ={result.mantel_rho:.3f})')
    # Add regression line
    z = np.polyfit(D_a[upper], D_e[upper], 1)
    p = np.poly1d(z)
    x_line = np.linspace(D_a[upper].min(), D_a[upper].max(), 100)
    ax.plot(x_line, p(x_line), 'r-', alpha=0.7)

    # 4. Mantel test null distribution
    ax = axes[1, 0]
    if result.null_distribution is not None:
        ax.hist(result.null_distribution, bins=50, alpha=0.7, label='Null')
        ax.axvline(result.mantel_rho, color='r', linestyle='--',
                   label=f'Observed (ρ={result.mantel_rho:.3f})')
        ax.set_xlabel('Spearman ρ')
        ax.set_ylabel('Count')
        ax.set_title(f'Mantel Test (p={result.mantel_p:.4f})')
        ax.legend()

    # 5. Per-dimension correlations
    ax = axes[1, 1]
    dim_corrs = per_dimension_correlation(affect_matrix, embedding_affect_matrix)
    names = list(dim_corrs.keys())
    rhos = [dim_corrs[n][0] for n in names]
    ps = [dim_corrs[n][1] for n in names]
    colors = ['green' if p < 0.05 else 'gray' for p in ps]
    bars = ax.bar(range(len(names)), rhos, color=colors, alpha=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Spearman ρ')
    ax.set_title('Per-Dimension Marginal Correlations')
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)

    # 6. Summary text
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = (
        f"RSA Results Summary\n"
        f"{'='*30}\n\n"
        f"N = {result.n_samples}\n\n"
        f"Spearman ρ = {result.rho_spearman:.4f}\n"
        f"Pearson r  = {result.rho_pearson:.4f}\n\n"
        f"Mantel test:\n"
        f"  ρ = {result.mantel_rho:.4f}\n"
        f"  p = {result.mantel_p:.4f}\n\n"
        f"CKA (linear) = {result.cka_linear:.4f}\n"
        f"CKA (RBF)    = {result.cka_rbf:.4f}\n\n"
        f"Significant: {'YES' if result.is_significant() else 'NO'}"
    )
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
            fontsize=12, fontfamily='monospace', verticalalignment='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"RSA plot saved to {save_path}")
    return save_path


def plot_ablation_comparison(
    ablation_results: Dict[str, RSAResult],
    save_path: str = 'results/v10/ablation_comparison.png',
) -> str:
    """Plot RSA correlation across ablation conditions."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    conditions = list(ablation_results.keys())
    rhos = [ablation_results[c].mantel_rho for c in conditions]
    ps = [ablation_results[c].mantel_p for c in conditions]
    ckas = [ablation_results[c].cka_linear for c in conditions]

    # RSA correlation by condition
    ax = axes[0]
    colors = ['green' if p < 0.05 else 'orange' if p < 0.1 else 'red' for p in ps]
    bars = ax.bar(range(len(conditions)), rhos, color=colors, alpha=0.7)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.set_ylabel('RSA ρ (Mantel)')
    ax.set_title('Geometric Alignment by Condition')
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    # Add significance markers
    for i, p in enumerate(ps):
        marker = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.text(i, rhos[i] + 0.01, marker, ha='center', fontsize=10)

    # CKA by condition
    ax = axes[1]
    ax.bar(range(len(conditions)), ckas, color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.set_ylabel('CKA (linear)')
    ax.set_title('CKA Alignment by Condition')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Ablation plot saved to {save_path}")
    return save_path


def plot_affect_trajectories(
    affect_matrix: np.ndarray,
    save_path: str = 'results/v10/affect_trajectories.png',
) -> str:
    """Plot affect dimension trajectories over time."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    from v10_affect import AffectVector

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    dim_names = AffectVector.dim_names()

    for i, (ax, name) in enumerate(zip(axes.flat, dim_names)):
        ax.plot(affect_matrix[:, i], alpha=0.7, linewidth=0.5)
        # Rolling average
        window = min(50, len(affect_matrix) // 10)
        if window > 1:
            rolling = np.convolve(affect_matrix[:, i], np.ones(window) / window, mode='valid')
            ax.plot(np.arange(window - 1, window - 1 + len(rolling)), rolling,
                   'r-', linewidth=2, label=f'MA({window})')
        ax.set_title(name)
        ax.set_xlabel('Step')
        ax.legend(fontsize=8)

    plt.suptitle('Affect Dimension Trajectories', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Trajectory plot saved to {save_path}")
    return save_path


# ============================================================================
# Full analysis report
# ============================================================================

def generate_report(
    affect_matrix: np.ndarray,
    embedding_affect_matrix: np.ndarray,
    ablation_results: Optional[Dict[str, RSAResult]] = None,
    output_dir: str = 'results/v10',
) -> Dict:
    """
    Run complete analysis and generate report.

    Returns dict with all results.
    """
    os.makedirs(output_dir, exist_ok=True)
    report = {}

    # 1. Main RSA analysis
    print("\n" + "=" * 60)
    print("RSA Analysis: Geometric Alignment Hypothesis")
    print("=" * 60)
    rsa_result = run_rsa_analysis(affect_matrix, embedding_affect_matrix, subsample=500)
    report['rsa'] = rsa_result.to_dict()

    # 2. Per-dimension analysis
    print("\nPer-dimension correlations:")
    dim_corrs = per_dimension_correlation(affect_matrix, embedding_affect_matrix)
    for name, (rho, p) in dim_corrs.items():
        sig = '*' if p < 0.05 else 'ns'
        print(f"  {name:20s}: ρ={rho:.3f} (p={p:.4f}) {sig}")
    report['per_dimension'] = dim_corrs

    # 3. Visualizations
    plot_rsa_results(rsa_result, affect_matrix, embedding_affect_matrix,
                    f'{output_dir}/rsa_analysis.png')
    plot_affect_trajectories(affect_matrix, f'{output_dir}/affect_trajectories.png')

    # 4. Ablation comparison (if available)
    if ablation_results:
        print("\n" + "=" * 60)
        print("Ablation Comparison")
        print("=" * 60)
        for condition, result in ablation_results.items():
            print(f"  {condition:25s}: {result.summary()}")
        report['ablation'] = {k: v.to_dict() for k, v in ablation_results.items()}
        plot_ablation_comparison(ablation_results, f'{output_dir}/ablation_comparison.png')

    # 5. Save report
    with open(f'{output_dir}/analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {output_dir}/analysis_report.json")

    # 6. Summary verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    if rsa_result.is_significant():
        print(f"Geometric Alignment SUPPORTED (ρ={rsa_result.mantel_rho:.3f}, p={rsa_result.mantel_p:.4f})")
        if rsa_result.mantel_rho > 0.3:
            print("Strong alignment: affect spaces are geometrically isomorphic")
        elif rsa_result.mantel_rho > 0.1:
            print("Moderate alignment: significant but noisy")
        else:
            print("Weak alignment: statistically significant but practically small")
    else:
        print(f"Geometric Alignment NOT SUPPORTED (ρ={rsa_result.mantel_rho:.3f}, p={rsa_result.mantel_p:.4f})")
        print("The distance structure in affect space does not match embedding space")

    return report


if __name__ == '__main__':
    # Test with synthetic data
    np.random.seed(42)
    N = 300

    # Create affect matrices with some alignment
    true_structure = np.random.randn(N, 3)  # shared latent structure
    noise_a = np.random.randn(N, 6) * 0.5
    noise_e = np.random.randn(N, 8) * 0.5

    # Info-theoretic affect: influenced by true structure
    affect_matrix = np.hstack([true_structure, np.random.randn(N, 3)]) + noise_a

    # Embedding-predicted affect: also influenced by true structure
    embedding_matrix = np.hstack([true_structure, np.random.randn(N, 5)]) + noise_e

    # Run analysis
    report = generate_report(affect_matrix, embedding_matrix, output_dir='results/v10_test')

    # Test with no alignment (shuffled)
    print("\n\n=== SHUFFLED CONTROL ===")
    shuffled_embedding = embedding_matrix[np.random.permutation(N)]
    run_rsa_analysis(affect_matrix, shuffled_embedding, n_permutations=1000)
