"""
Analyze learned representations to test whether 6D affect structure emerges.

═══════════════════════════════════════════════════════════════════════════════
WARNING: THIS ANALYSIS TESTS THE WRONG THING - SEE CORRECTION.md
═══════════════════════════════════════════════════════════════════════════════

This code analyzes the dimensionality of RAW hidden representations.
The thesis does NOT claim raw representations have 6 dimensions.

The thesis claims that SIX COMPUTED QUANTITIES (valence, arousal, integration,
effective rank, counterfactual weight, self-model salience) capture affect
structure when derived from raw states.

Example of correct vs incorrect interpretation:
  WRONG: "The hidden layer should have effective rank ≈ 6"
  RIGHT: "When we compute valence = f(predicted_trajectory, viability_boundary),
          arousal = KL(belief_t+1 || belief_t), etc., these six quantities
          should predict affective behavior"

This code is preserved as an educational example. Results show that raw
representations DO track viability (PC1 correlates with viability distance),
which partially supports the theory, but the dimensionality analysis is
not the right test.

For the CORRECT approach, see: ../study_c_llm_affect/
═══════════════════════════════════════════════════════════════════════════════

Original (incorrect) predictions preserved below for reference:
- Simple RL: 2-3 dimensions (value-like, arousal-like)
- World model: 3-4 dimensions (add integration-like)
- Self model: 5-6 dimensions (add self-salience, counterfactual)
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import pearsonr
from scipy.linalg import svd
import matplotlib.pyplot as plt
import json

from viability_env import ViabilityWorld
from train_agents import SimpleAgent, WorldModelAgent, SelfModelAgent


def collect_representations(
    agent: torch.nn.Module,
    env: ViabilityWorld,
    n_episodes: int = 50,
    max_steps: int = 500
) -> Dict[str, np.ndarray]:
    """
    Collect internal representations along with ground-truth variables.

    Returns dict with:
    - representations: (N, hidden_dim) - agent's internal states
    - viability_distance: (N,) - true distance to viability boundary
    - viability_gradient: (N, 4) - true gradient direction
    - health, energy, temperature, stress: (N,) - true internal states
    - threat_active: (N,) - whether threat is present
    - reward: (N,) - received reward (proxy for valence?)
    """
    representations = []
    viability_distances = []
    healths = []
    energies = []
    temperatures = []
    stresses = []
    threat_actives = []
    rewards = []
    actions = []

    for ep in range(n_episodes):
        obs, _ = env.reset()

        for step in range(max_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

            with torch.no_grad():
                rep = agent.get_representation(obs_tensor)
                action, _, _ = agent.get_action(obs_tensor)

            representations.append(rep.squeeze().numpy())
            viability_distances.append(env.get_true_viability_distance())
            healths.append(env.health)
            energies.append(env.energy)
            temperatures.append(env.temperature)
            stresses.append(env.stress)
            threat_actives.append(float(env.threat_active))
            actions.append(action.squeeze().numpy())

            action_np = action.squeeze().numpy()
            obs, reward, terminated, truncated, _ = env.step(action_np)
            rewards.append(reward)

            if terminated or truncated:
                break

    return {
        'representations': np.array(representations),
        'viability_distance': np.array(viability_distances),
        'health': np.array(healths),
        'energy': np.array(energies),
        'temperature': np.array(temperatures),
        'stress': np.array(stresses),
        'threat_active': np.array(threat_actives),
        'reward': np.array(rewards),
        'actions': np.array(actions),
    }


def analyze_dimensionality(representations: np.ndarray) -> Dict:
    """
    Analyze effective dimensionality of representations.

    Uses multiple methods:
    1. PCA variance explained
    2. Effective rank (from thesis)
    3. Participation ratio
    """
    # Standardize
    reps_std = (representations - representations.mean(axis=0)) / (representations.std(axis=0) + 1e-8)

    # PCA
    pca = PCA()
    pca.fit(reps_std)

    # Cumulative variance explained
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    # How many components for 90%, 95%, 99%?
    n_90 = np.searchsorted(cumvar, 0.90) + 1
    n_95 = np.searchsorted(cumvar, 0.95) + 1
    n_99 = np.searchsorted(cumvar, 0.99) + 1

    # Effective rank (thesis definition)
    eigenvalues = pca.explained_variance_
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    trace = np.sum(eigenvalues)
    trace_sq = np.sum(eigenvalues ** 2)
    effective_rank = (trace ** 2) / trace_sq if trace_sq > 0 else 0

    # Participation ratio (similar but different normalization)
    participation_ratio = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)

    return {
        'n_90': n_90,
        'n_95': n_95,
        'n_99': n_99,
        'effective_rank': effective_rank,
        'participation_ratio': participation_ratio,
        'eigenvalues': eigenvalues,
        'variance_explained': pca.explained_variance_ratio_,
        'components': pca.components_,
    }


def test_theoretical_mapping(
    representations: np.ndarray,
    data: Dict[str, np.ndarray]
) -> Dict[str, Dict]:
    """
    Test whether components map to theoretical affect dimensions.

    For each principal component, compute correlation with:
    - Viability distance (should map to valence)
    - Reward (proxy for valence)
    - Health/energy/temp/stress (should relate to various dimensions)
    - Threat presence (should relate to arousal, stress)

    The thesis predicts specific mappings. If they don't emerge,
    thesis is falsified.
    """
    # Standardize and get PCA
    reps_std = (representations - representations.mean(axis=0)) / (representations.std(axis=0) + 1e-8)
    pca = PCA(n_components=min(10, representations.shape[1]))
    pcs = pca.fit_transform(reps_std)

    # Ground truth variables
    targets = {
        'viability_distance': data['viability_distance'],
        'reward': data['reward'],
        'health': data['health'],
        'energy': data['energy'],
        'temperature_deviation': np.abs(data['temperature'] - 37.0),
        'stress': data['stress'],
        'threat_active': data['threat_active'],
    }

    # Compute correlations
    results = {}
    for i in range(pcs.shape[1]):
        pc_corrs = {}
        for name, target in targets.items():
            r, p = pearsonr(pcs[:, i], target)
            pc_corrs[name] = {'r': r, 'p': p}
        results[f'PC{i+1}'] = pc_corrs

    return results


def identify_affect_dimensions(
    correlations: Dict,
    threshold: float = 0.3
) -> Dict[str, str]:
    """
    Attempt to identify which PCs correspond to which affect dimensions.

    Theoretical predictions:
    - Valence: correlates with viability_distance, reward
    - Arousal: correlates with threat_active, |action magnitude|
    - Integration: correlates with consistency of other signals
    - Effective rank: correlates with behavioral diversity
    - CF: correlates with action planning (hard to measure)
    - SM: correlates with self-state focus

    Returns mapping from PC to predicted dimension.
    """
    mappings = {}

    for pc, corrs in correlations.items():
        # Check for valence-like (viability tracking)
        viab_r = abs(corrs['viability_distance']['r'])
        reward_r = abs(corrs['reward']['r'])
        if viab_r > threshold and reward_r > threshold:
            mappings[pc] = 'valence_candidate'
            continue

        # Check for arousal-like (threat response)
        threat_r = abs(corrs['threat_active']['r'])
        if threat_r > threshold:
            mappings[pc] = 'arousal_candidate'
            continue

        # Check for internal-state tracking (self-salience proxy)
        health_r = abs(corrs['health']['r'])
        energy_r = abs(corrs['energy']['r'])
        if health_r > threshold or energy_r > threshold:
            mappings[pc] = 'self_model_candidate'
            continue

    return mappings


def visualize_representation_space(
    representations: np.ndarray,
    data: Dict[str, np.ndarray],
    save_path: Path
):
    """Create visualizations of representation structure."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Variance explained
    pca = PCA()
    pca.fit(representations)
    ax = axes[0, 0]
    ax.bar(range(1, min(11, len(pca.explained_variance_ratio_) + 1)),
           pca.explained_variance_ratio_[:10] * 100)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_title('PCA Variance Explained')
    ax.axhline(100/6, color='r', linestyle='--', label='1/6 (equal)')
    ax.legend()

    # 2. Cumulative variance
    ax = axes[0, 1]
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    ax.plot(range(1, len(cumvar) + 1), cumvar * 100, 'b-o')
    ax.axhline(90, color='r', linestyle='--', label='90%')
    ax.axhline(95, color='g', linestyle='--', label='95%')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Variance (%)')
    ax.set_title('Cumulative Variance Explained')
    ax.legend()

    # 3. 2D projection colored by viability
    pcs = PCA(n_components=2).fit_transform(representations)
    ax = axes[0, 2]
    scatter = ax.scatter(pcs[:, 0], pcs[:, 1], c=data['viability_distance'],
                         cmap='RdYlGn', alpha=0.5, s=1)
    plt.colorbar(scatter, ax=ax, label='Viability Distance')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Representation Space (colored by viability)')

    # 4. 2D projection colored by threat
    ax = axes[1, 0]
    scatter = ax.scatter(pcs[:, 0], pcs[:, 1], c=data['threat_active'],
                         cmap='coolwarm', alpha=0.5, s=1)
    plt.colorbar(scatter, ax=ax, label='Threat Active')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Representation Space (colored by threat)')

    # 5. 2D projection colored by reward
    ax = axes[1, 1]
    scatter = ax.scatter(pcs[:, 0], pcs[:, 1], c=data['reward'],
                         cmap='viridis', alpha=0.5, s=1)
    plt.colorbar(scatter, ax=ax, label='Reward')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Representation Space (colored by reward)')

    # 6. Correlation heatmap
    ax = axes[1, 2]
    pcs_6 = PCA(n_components=min(6, representations.shape[1])).fit_transform(representations)
    targets = np.column_stack([
        data['viability_distance'],
        data['reward'],
        data['threat_active'],
        data['stress'],
        data['health'],
        data['energy'],
    ])
    target_names = ['Viability', 'Reward', 'Threat', 'Stress', 'Health', 'Energy']

    corr_matrix = np.zeros((pcs_6.shape[1], len(target_names)))
    for i in range(pcs_6.shape[1]):
        for j, target in enumerate(targets.T):
            corr_matrix[i, j], _ = pearsonr(pcs_6[:, i], target)

    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(target_names)))
    ax.set_xticklabels(target_names, rotation=45, ha='right')
    ax.set_yticks(range(pcs_6.shape[1]))
    ax.set_yticklabels([f'PC{i+1}' for i in range(pcs_6.shape[1])])
    ax.set_title('PC-Variable Correlations')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_analysis(agent_type: str, model_path: Path, output_dir: Path):
    """Run full analysis for one agent type."""
    print(f"\n{'='*50}")
    print(f"Analyzing {agent_type} agent")
    print('='*50)

    # Load model
    env = ViabilityWorld(seed=123)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if agent_type == "simple":
        agent = SimpleAgent(obs_dim, action_dim)
    elif agent_type == "world_model":
        agent = WorldModelAgent(obs_dim, action_dim)
    elif agent_type == "self_model":
        agent = SelfModelAgent(obs_dim, action_dim)

    try:
        agent.load_state_dict(torch.load(model_path / f"{agent_type}_agent.pt"))
    except FileNotFoundError:
        print(f"  Model not found at {model_path}, skipping")
        return None

    agent.eval()

    # Collect representations
    print("  Collecting representations...")
    data = collect_representations(agent, env, n_episodes=50)
    print(f"  Collected {len(data['representations'])} samples")

    # Analyze dimensionality
    print("  Analyzing dimensionality...")
    dim_results = analyze_dimensionality(data['representations'])
    print(f"  Effective rank: {dim_results['effective_rank']:.2f}")
    print(f"  Components for 90% var: {dim_results['n_90']}")
    print(f"  Components for 95% var: {dim_results['n_95']}")

    # Test theoretical mapping
    print("  Testing theoretical mappings...")
    correlations = test_theoretical_mapping(data['representations'], data)
    mappings = identify_affect_dimensions(correlations)
    print(f"  Identified dimensions: {mappings}")

    # Visualize
    output_dir.mkdir(parents=True, exist_ok=True)
    visualize_representation_space(
        data['representations'],
        data,
        output_dir / f"{agent_type}_representations.png"
    )

    # Save detailed results
    results = {
        'agent_type': agent_type,
        'effective_rank': float(dim_results['effective_rank']),
        'n_90': int(dim_results['n_90']),
        'n_95': int(dim_results['n_95']),
        'n_99': int(dim_results['n_99']),
        'participation_ratio': float(dim_results['participation_ratio']),
        'variance_explained': [float(x) for x in dim_results['variance_explained']],
        'mappings': mappings,
        'correlations': {k: {k2: float(v2['r']) for k2, v2 in v.items()}
                        for k, v in correlations.items()},
    }

    with open(output_dir / f"{agent_type}_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


def compare_agents(results: List[Dict], output_dir: Path):
    """Compare dimensionality across agent types."""
    if not results or all(r is None for r in results):
        print("No results to compare")
        return

    results = [r for r in results if r is not None]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Effective rank comparison
    ax = axes[0]
    agent_types = [r['agent_type'] for r in results]
    eff_ranks = [r['effective_rank'] for r in results]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    bars = ax.bar(agent_types, eff_ranks, color=colors[:len(results)])
    ax.set_ylabel('Effective Rank')
    ax.set_title('Dimensionality by Agent Type')
    ax.axhline(6, color='r', linestyle='--', label='Thesis prediction (6D)')
    ax.axhline(2, color='gray', linestyle=':', label='Russell circumplex (2D)')
    ax.legend()

    # Add value labels
    for bar, val in zip(bars, eff_ranks):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}', ha='center', va='bottom')

    # Variance explained curves
    ax = axes[1]
    for i, r in enumerate(results):
        var_exp = np.array(r['variance_explained'])
        cumvar = np.cumsum(var_exp)
        ax.plot(range(1, len(cumvar) + 1), cumvar * 100,
                'o-', label=r['agent_type'], color=colors[i])

    ax.axhline(90, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Variance (%)')
    ax.set_title('Variance Explained by Architecture')
    ax.legend()
    ax.set_xlim(0, 15)

    plt.tight_layout()
    plt.savefig(output_dir / 'agent_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: Dimensionality by Agent Type")
    print("="*60)
    print(f"{'Agent':<15} {'Eff Rank':<12} {'N(90%)':<10} {'N(95%)':<10}")
    print("-"*60)
    for r in results:
        print(f"{r['agent_type']:<15} {r['effective_rank']:<12.2f} {r['n_90']:<10} {r['n_95']:<10}")

    print("\nThesis Prediction: Self-model agent should have ~6 dimensions")
    print("Falsification: If simple agent has same dimensionality as self-model")


def main():
    """Run complete analysis."""
    model_path = Path("trained_agents")
    output_dir = Path("analysis_results")

    results = []
    for agent_type in ["simple", "world_model", "self_model"]:
        result = run_analysis(agent_type, model_path, output_dir)
        results.append(result)

    compare_agents(results, output_dir)

    print("\n" + "="*60)
    print("Analysis complete. Results saved to:", output_dir)
    print("="*60)


if __name__ == "__main__":
    main()
