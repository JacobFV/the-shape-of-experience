"""
Affect Space Analysis and Visualization.

Tools for:
1. Comparing measured affect spaces to theoretical predictions
2. Computing structural correspondence metrics
3. Visualizing 6D affect space (via projections and embeddings)
4. Testing whether affect geometry is preserved across models
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import spearmanr, pearsonr
from scipy.linalg import orthogonal_procrustes
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA

from .emotion_spectrum import EMOTION_SPECTRUM, get_emotion_matrix


@dataclass
class CorrespondenceMetrics:
    """Comprehensive correspondence metrics between two affect spaces."""
    # Overall
    overall_score: float

    # Point-wise
    mean_absolute_error: float
    mean_squared_error: float
    cosine_similarity: float

    # Per-dimension
    dimension_correlations: Dict[str, float]

    # Structural
    distance_preservation: float  # Do pairwise distances match?
    neighborhood_preservation: float  # Are neighbors preserved?
    cluster_agreement: float  # Do clusters match?

    # Procrustes
    procrustes_distance: float  # Distance after optimal alignment


class AffectSpaceAnalyzer:
    """Analyze and compare affect spaces."""

    def __init__(self, results_dir: Optional[str] = None):
        self.results_dir = Path(results_dir) if results_dir else None
        self.theoretical_matrix, self.emotion_names = get_emotion_matrix()
        self.dim_names = [
            "valence", "arousal", "integration",
            "effective_rank", "counterfactual_weight", "self_model_salience"
        ]

    def compute_correspondence(
        self,
        measured: np.ndarray,
        theoretical: Optional[np.ndarray] = None
    ) -> CorrespondenceMetrics:
        """
        Compute comprehensive correspondence metrics.

        Args:
            measured: (n_emotions, 6) measured affect matrix
            theoretical: Optional override for theoretical matrix

        Returns:
            CorrespondenceMetrics
        """
        if theoretical is None:
            theoretical = self.theoretical_matrix

        # Ensure same shape
        n = min(len(measured), len(theoretical))
        measured = measured[:n]
        theoretical = theoretical[:n]

        # Point-wise metrics
        mae = np.mean(np.abs(measured - theoretical))
        mse = np.mean((measured - theoretical) ** 2)

        # Cosine similarity of flattened matrices
        m_flat = measured.flatten()
        t_flat = theoretical.flatten()
        cos_sim = np.dot(m_flat, t_flat) / (
            np.linalg.norm(m_flat) * np.linalg.norm(t_flat) + 1e-8
        )

        # Per-dimension correlations
        dim_corrs = {}
        for i, dim in enumerate(self.dim_names):
            if measured[:, i].std() > 0.01 and theoretical[:, i].std() > 0.01:
                corr, _ = spearmanr(measured[:, i], theoretical[:, i])
                dim_corrs[dim] = float(corr) if not np.isnan(corr) else 0.0
            else:
                dim_corrs[dim] = 0.0

        # Distance preservation
        m_dists = pdist(measured)
        t_dists = pdist(theoretical)
        if len(m_dists) > 2:
            dist_corr, _ = spearmanr(m_dists, t_dists)
            dist_preservation = float(dist_corr) if not np.isnan(dist_corr) else 0.0
        else:
            dist_preservation = 0.0

        # Neighborhood preservation (k=5)
        k = min(5, n - 1)
        neighborhood_score = self._compute_neighborhood_preservation(
            measured, theoretical, k
        )

        # Cluster agreement (using k-means)
        cluster_score = self._compute_cluster_agreement(measured, theoretical)

        # Procrustes analysis
        procrustes_dist = self._compute_procrustes_distance(measured, theoretical)

        # Overall score (weighted combination)
        overall = (
            0.2 * max(0, cos_sim) +
            0.3 * max(0, dist_preservation) +
            0.2 * max(0, neighborhood_score) +
            0.2 * np.mean(list(dim_corrs.values())) +
            0.1 * max(0, 1 - procrustes_dist)
        )

        return CorrespondenceMetrics(
            overall_score=float(overall),
            mean_absolute_error=float(mae),
            mean_squared_error=float(mse),
            cosine_similarity=float(cos_sim),
            dimension_correlations=dim_corrs,
            distance_preservation=float(dist_preservation),
            neighborhood_preservation=float(neighborhood_score),
            cluster_agreement=float(cluster_score),
            procrustes_distance=float(procrustes_dist)
        )

    def _compute_neighborhood_preservation(
        self,
        measured: np.ndarray,
        theoretical: np.ndarray,
        k: int
    ) -> float:
        """Check if k-nearest neighbors are preserved."""
        n = len(measured)

        m_dists = squareform(pdist(measured))
        t_dists = squareform(pdist(theoretical))

        preserved = 0
        total = 0

        for i in range(n):
            m_neighbors = set(np.argsort(m_dists[i])[1:k+1])
            t_neighbors = set(np.argsort(t_dists[i])[1:k+1])
            preserved += len(m_neighbors & t_neighbors)
            total += k

        return preserved / total if total > 0 else 0.0

    def _compute_cluster_agreement(
        self,
        measured: np.ndarray,
        theoretical: np.ndarray,
        n_clusters: int = 5
    ) -> float:
        """Check if clustering agrees between spaces."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score

        n_clusters = min(n_clusters, len(measured) - 1)
        if n_clusters < 2:
            return 0.0

        m_clusters = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(measured)
        t_clusters = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(theoretical)

        return adjusted_rand_score(m_clusters, t_clusters)

    def _compute_procrustes_distance(
        self,
        measured: np.ndarray,
        theoretical: np.ndarray
    ) -> float:
        """Procrustes distance after optimal rotation/scaling."""
        # Center both matrices
        m_centered = measured - measured.mean(axis=0)
        t_centered = theoretical - theoretical.mean(axis=0)

        # Scale to unit variance
        m_scale = np.sqrt((m_centered ** 2).sum())
        t_scale = np.sqrt((t_centered ** 2).sum())

        m_normed = m_centered / (m_scale + 1e-8)
        t_normed = t_centered / (t_scale + 1e-8)

        # Find optimal rotation
        R, scale = orthogonal_procrustes(m_normed, t_normed)
        m_aligned = m_normed @ R

        # Compute distance
        return float(np.sqrt(((m_aligned - t_normed) ** 2).sum()))

    def load_model_results(self, filepath: str) -> Tuple[np.ndarray, List[str]]:
        """Load measured affect matrix from results file."""
        with open(filepath) as f:
            data = json.load(f)

        emotions = list(data["emotions"].keys())
        matrix = np.zeros((len(emotions), 6))

        for i, emotion in enumerate(emotions):
            measured = data["emotions"][emotion]["measured"]
            if "error" not in measured:
                matrix[i] = [
                    measured["valence"],
                    measured["arousal"],
                    measured["integration"],
                    measured["effective_rank"],
                    measured["counterfactual_weight"],
                    measured["self_model_salience"],
                ]

        return matrix, emotions

    def compare_models(
        self,
        model_results: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple models to each other."""
        models = list(model_results.keys())
        comparisons = {}

        for i, m1 in enumerate(models):
            comparisons[m1] = {}
            for j, m2 in enumerate(models):
                if i != j:
                    metrics = self.compute_correspondence(
                        model_results[m1],
                        model_results[m2]
                    )
                    comparisons[m1][m2] = metrics.overall_score

        return comparisons

    def visualize_affect_space(
        self,
        measured: np.ndarray,
        method: str = "tsne",
        show_theoretical: bool = True,
        emotion_names: Optional[List[str]] = None,
        title: str = "Affect Space"
    ) -> plt.Figure:
        """
        Visualize 6D affect space in 2D.

        Args:
            measured: (n_emotions, 6) affect matrix
            method: "tsne", "mds", or "pca"
            show_theoretical: Also show theoretical positions
            emotion_names: Labels for points
            title: Plot title

        Returns:
            matplotlib Figure
        """
        emotion_names = emotion_names or self.emotion_names[:len(measured)]

        # Reduce dimensionality
        if method == "tsne":
            perplexity = min(30, len(measured) - 1)
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        elif method == "mds":
            reducer = MDS(n_components=2, random_state=42)
        else:
            reducer = PCA(n_components=2)

        if show_theoretical:
            # Combine measured and theoretical, then reduce together
            theoretical = self.theoretical_matrix[:len(measured)]
            combined = np.vstack([measured, theoretical])
            reduced = reducer.fit_transform(combined)
            measured_2d = reduced[:len(measured)]
            theoretical_2d = reduced[len(measured):]
        else:
            measured_2d = reducer.fit_transform(measured)

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))

        # Color by valence (theoretical)
        theoretical = self.theoretical_matrix[:len(measured)]
        colors = theoretical[:, 0]  # valence
        cmap = plt.cm.RdYlGn

        # Plot theoretical (if showing)
        if show_theoretical:
            ax.scatter(
                theoretical_2d[:, 0], theoretical_2d[:, 1],
                c=colors, cmap=cmap, s=100, alpha=0.3,
                marker='o', label='Theoretical'
            )

        # Plot measured
        sc = ax.scatter(
            measured_2d[:, 0], measured_2d[:, 1],
            c=colors, cmap=cmap, s=200, alpha=0.8,
            marker='s', edgecolors='black', linewidth=1,
            label='Measured'
        )

        # Add labels
        for i, name in enumerate(emotion_names):
            ax.annotate(
                name,
                (measured_2d[i, 0], measured_2d[i, 1]),
                fontsize=8, alpha=0.8,
                xytext=(5, 5), textcoords='offset points'
            )

        # Draw arrows from theoretical to measured
        if show_theoretical:
            for i in range(len(measured)):
                ax.annotate(
                    '',
                    xy=(measured_2d[i, 0], measured_2d[i, 1]),
                    xytext=(theoretical_2d[i, 0], theoretical_2d[i, 1]),
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.3)
                )

        ax.set_title(f"{title}\n({method.upper()} projection)", fontsize=14)
        ax.legend()
        plt.colorbar(sc, label='Valence')

        return fig

    def visualize_dimension_comparisons(
        self,
        measured: np.ndarray,
        emotion_names: Optional[List[str]] = None
    ) -> plt.Figure:
        """Create scatter plots comparing measured vs theoretical for each dimension."""
        emotion_names = emotion_names or self.emotion_names[:len(measured)]
        theoretical = self.theoretical_matrix[:len(measured)]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, (ax, dim) in enumerate(zip(axes, self.dim_names)):
            # Plot scatter
            ax.scatter(
                theoretical[:, i], measured[:, i],
                alpha=0.6, s=50
            )

            # Add diagonal reference
            lims = [
                min(theoretical[:, i].min(), measured[:, i].min()) - 0.1,
                max(theoretical[:, i].max(), measured[:, i].max()) + 0.1
            ]
            ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect match')

            # Compute correlation
            corr, p = spearmanr(theoretical[:, i], measured[:, i])

            ax.set_xlabel(f'Theoretical {dim}')
            ax.set_ylabel(f'Measured {dim}')
            ax.set_title(f'{dim}\n(r={corr:.2f}, p={p:.3f})')
            ax.legend()

        plt.tight_layout()
        return fig

    def generate_report(
        self,
        measured: np.ndarray,
        model_name: str,
        save_path: Optional[str] = None
    ) -> str:
        """Generate a text report of the analysis."""
        metrics = self.compute_correspondence(measured)

        report = f"""
================================================================================
AFFECT SPACE CORRESPONDENCE REPORT
Model: {model_name}
================================================================================

OVERALL CORRESPONDENCE SCORE: {metrics.overall_score:.3f}

POINT-WISE METRICS:
  Mean Absolute Error:  {metrics.mean_absolute_error:.3f}
  Mean Squared Error:   {metrics.mean_squared_error:.3f}
  Cosine Similarity:    {metrics.cosine_similarity:.3f}

STRUCTURAL METRICS:
  Distance Preservation:     {metrics.distance_preservation:.3f}
  Neighborhood Preservation: {metrics.neighborhood_preservation:.3f}
  Cluster Agreement:         {metrics.cluster_agreement:.3f}
  Procrustes Distance:       {metrics.procrustes_distance:.3f}

PER-DIMENSION CORRELATIONS:
"""
        for dim, corr in metrics.dimension_correlations.items():
            bar = "█" * int(abs(corr) * 20) + "░" * (20 - int(abs(corr) * 20))
            sign = "+" if corr >= 0 else "-"
            report += f"  {dim:25s} [{bar}] {sign}{abs(corr):.3f}\n"

        report += f"""
INTERPRETATION:
  Overall Score > 0.7: Strong correspondence (affect geometry preserved)
  Overall Score 0.4-0.7: Moderate correspondence (partial structure)
  Overall Score < 0.4: Weak correspondence (geometry not preserved)

  Current result: {'STRONG' if metrics.overall_score > 0.7 else 'MODERATE' if metrics.overall_score > 0.4 else 'WEAK'} correspondence

================================================================================
"""
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)

        return report


def analyze_results_directory(results_dir: str):
    """Analyze all results in a directory."""
    results_path = Path(results_dir)
    analyzer = AffectSpaceAnalyzer()

    print(f"Analyzing results in: {results_dir}\n")

    for json_file in results_path.glob("*.json"):
        if json_file.name in ["summary.json", "analysis.json"]:
            continue

        print(f"{'='*60}")
        print(f"Analyzing: {json_file.name}")

        try:
            matrix, emotions = analyzer.load_model_results(str(json_file))
            metrics = analyzer.compute_correspondence(matrix)

            print(f"  Overall Score: {metrics.overall_score:.3f}")
            print(f"  Distance Preservation: {metrics.distance_preservation:.3f}")
            print("  Dimension correlations:")
            for dim, corr in metrics.dimension_correlations.items():
                print(f"    {dim}: {corr:+.3f}")

        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyze_results_directory(sys.argv[1])
    else:
        print("Usage: python affect_space_analysis.py <results_dir>")
