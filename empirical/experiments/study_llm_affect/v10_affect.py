"""
V10: Affect Extraction from RL Agent Internals
================================================

Computes the 6D affect dimensions from RL agent latent states z_t,
NOT from text (unlike affect_calculator.py which operates on LLM outputs).

Dimensions:
1. Valence     - Advantage A(z_t, a_t) and/or survival time probe delta
2. Arousal     - ||z_{t+1} - z_t||_2
3. Integration - Partition prediction loss difference
4. Eff. Rank   - Participation ratio from rolling covariance of z_t
5. CF Weight   - World model forward pass fraction (planning compute)
6. Self-Model  - MI(z_t^self; a_t) via probe

All probes are trained post-hoc on frozen agent weights to avoid
contaminating the affect measurements.
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class AffectVector:
    """6D affect measurement for a single agent-state."""
    valence: float
    arousal: float
    integration: float
    effective_rank: float
    counterfactual_weight: float
    self_model_salience: float
    # Metadata
    step: int = 0
    agent_id: int = 0
    raw: Dict[str, float] = field(default_factory=dict)

    def to_array(self) -> np.ndarray:
        return np.array([
            self.valence, self.arousal, self.integration,
            self.effective_rank, self.counterfactual_weight,
            self.self_model_salience
        ])

    @staticmethod
    def dim_names() -> List[str]:
        return ['valence', 'arousal', 'integration',
                'effective_rank', 'cf_weight', 'self_model_salience']


class SurvivalProbe:
    """
    Train a probe to predict time-to-death from latent state z_t.
    Used for viability-based valence: val = tau_{t+1} - tau_t.
    """

    def __init__(self, hidden_sizes=(64, 32)):
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_sizes,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
        )
        self.fitted = False

    def fit(self, z_history: np.ndarray, survival_times: np.ndarray):
        """
        Fit probe on latent states -> time until death.

        Args:
            z_history: (N, latent_dim) - latent states
            survival_times: (N,) - steps until death from each state
        """
        self.model.fit(z_history, survival_times)
        self.fitted = True
        score = self.model.score(z_history, survival_times)
        print(f"  Survival probe R²: {score:.3f}")
        return score

    def predict(self, z: np.ndarray) -> np.ndarray:
        """Predict survival time from latent state."""
        if not self.fitted:
            raise RuntimeError("Survival probe not fitted")
        return self.model.predict(z)


class PartitionPredictor:
    """
    Train full and partitioned predictors of z_{t+1} from z_t.
    Integration = L_partitioned - L_full.
    """

    def __init__(self, latent_dim: int, hidden_sizes=(64,)):
        self.latent_dim = latent_dim
        self.half_dim = latent_dim // 2
        self.hidden = hidden_sizes

        # Full predictor: z_t -> z_{t+1}
        self.full_pred = MLPRegressor(
            hidden_layer_sizes=hidden_sizes,
            max_iter=300,
            early_stopping=True,
        )

        # Partitioned predictors: z_t^A -> z_{t+1}^A, z_t^B -> z_{t+1}^B
        self.part_a = MLPRegressor(
            hidden_layer_sizes=hidden_sizes,
            max_iter=300,
            early_stopping=True,
        )
        self.part_b = MLPRegressor(
            hidden_layer_sizes=hidden_sizes,
            max_iter=300,
            early_stopping=True,
        )
        self.fitted = False

    def fit(self, z_seq: np.ndarray):
        """
        Fit predictors on sequential latent states.

        Args:
            z_seq: (T, latent_dim) - sequence of latent states
        """
        z_t = z_seq[:-1]
        z_next = z_seq[1:]

        # Full predictor
        self.full_pred.fit(z_t, z_next)
        full_score = self.full_pred.score(z_t, z_next)

        # Partition: first half and second half
        z_t_a = z_t[:, :self.half_dim]
        z_t_b = z_t[:, self.half_dim:]
        z_next_a = z_next[:, :self.half_dim]
        z_next_b = z_next[:, self.half_dim:]

        self.part_a.fit(z_t_a, z_next_a)
        self.part_b.fit(z_t_b, z_next_b)
        part_a_score = self.part_a.score(z_t_a, z_next_a)
        part_b_score = self.part_b.score(z_t_b, z_next_b)

        self.fitted = True
        print(f"  Partition predictor: full R²={full_score:.3f}, "
              f"part_A R²={part_a_score:.3f}, part_B R²={part_b_score:.3f}")
        return full_score, part_a_score, part_b_score

    def compute_integration(self, z_t: np.ndarray, z_next: np.ndarray) -> np.ndarray:
        """
        Compute integration for each pair (z_t, z_next).

        Returns: (N,) integration values
        """
        if not self.fitted:
            raise RuntimeError("Partition predictor not fitted")

        # Full prediction loss
        z_pred_full = self.full_pred.predict(z_t)
        loss_full = np.mean((z_pred_full - z_next) ** 2, axis=1)

        # Partitioned prediction loss
        z_t_a = z_t[:, :self.half_dim]
        z_t_b = z_t[:, self.half_dim:]
        z_next_a = z_next[:, :self.half_dim]
        z_next_b = z_next[:, self.half_dim:]

        z_pred_a = self.part_a.predict(z_t_a)
        z_pred_b = self.part_b.predict(z_t_b)
        loss_part = (np.mean((z_pred_a - z_next_a) ** 2, axis=1) +
                     np.mean((z_pred_b - z_next_b) ** 2, axis=1)) / 2

        # Integration = how much partitioning hurts
        integration = loss_part - loss_full
        return np.clip(integration, 0, None)


class SelfModelProbe:
    """
    Estimate MI(z_t^self; a_t) via a probe.
    Self-related latent components: those most predictive of the agent's
    own actions vs other agents' actions.
    """

    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.action_probe = LogisticRegression(max_iter=500)
        self.self_probe = None
        self.fitted = False
        self.self_dims = None

    def fit(self, z_history: np.ndarray, actions: np.ndarray):
        """
        Fit self-model probe.

        1. Train action predictor from full z
        2. Identify which z dimensions are most predictive (self-related dims)
        3. Train action predictor from self-related dims only
        """
        # Full action prediction
        self.action_probe.fit(z_history, actions)
        full_accuracy = self.action_probe.score(z_history, actions)

        # Feature importance: coefficient magnitudes
        coef_importance = np.mean(np.abs(self.action_probe.coef_), axis=0)
        # Top half of dimensions by importance = "self-related"
        n_self = len(coef_importance) // 2
        self.self_dims = np.argsort(coef_importance)[-n_self:]

        # Self-only probe
        z_self = z_history[:, self.self_dims]
        self.self_probe = LogisticRegression(max_iter=500)
        self.self_probe.fit(z_self, actions)
        self_accuracy = self.self_probe.score(z_self, actions)

        self.fitted = True
        print(f"  Self-model probe: full acc={full_accuracy:.3f}, "
              f"self-dims acc={self_accuracy:.3f}")
        return full_accuracy, self_accuracy

    def compute_salience(self, z: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Compute self-model salience for each state.
        Higher when self-related dims are more predictive of action.
        """
        if not self.fitted:
            raise RuntimeError("Self-model probe not fitted")

        z_self = z[:, self.self_dims]

        # Per-sample: probability assigned to actual action
        full_probs = self.action_probe.predict_proba(z)
        self_probs = self.self_probe.predict_proba(z_self)

        # Self-model salience = how much self-dims contribute to action prediction
        full_conf = full_probs[np.arange(len(actions)), actions]
        self_conf = self_probs[np.arange(len(actions)), actions]

        # Ratio: if self-dims predict action as well as full z, salience is high
        salience = self_conf / (full_conf + 1e-8)
        return np.clip(salience, 0, 1)


class AffectExtractor:
    """
    Extract 6D affect vectors from RL agent latent state trajectories.

    Usage:
        1. Collect latent state history during/after training
        2. Call fit_probes() to train post-hoc probes
        3. Call extract() to compute affect vectors
    """

    def __init__(self, latent_dim: int, n_actions: int, window_size: int = 50):
        self.latent_dim = latent_dim
        self.n_actions = n_actions
        self.window_size = window_size

        # Probes
        self.survival_probe = SurvivalProbe()
        self.partition_pred = PartitionPredictor(latent_dim)
        self.self_model_probe = SelfModelProbe(n_actions)

        self.probes_fitted = False

    def fit_probes(
        self,
        z_history: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        death_steps: Optional[np.ndarray] = None,
    ):
        """
        Fit all post-hoc probes on collected latent state data.

        Args:
            z_history: (T, latent_dim) latent states
            actions: (T,) action indices
            rewards: (T,) rewards
            values: (T,) value estimates
            death_steps: (T,) steps until death from each state (for survival probe)
        """
        print("Fitting affect extraction probes...")

        # Survival probe
        if death_steps is not None:
            self.survival_probe.fit(z_history, death_steps)
        else:
            # Approximate survival time from cumulative future reward
            T = len(rewards)
            cum_reward = np.zeros(T)
            running = 0
            for t in reversed(range(T)):
                running = rewards[t] + 0.99 * running
                cum_reward[t] = running
            self.survival_probe.fit(z_history, cum_reward)

        # Partition predictor (on sequential data)
        self.partition_pred.fit(z_history)

        # Self-model probe
        self.self_model_probe.fit(z_history, actions)

        self.probes_fitted = True
        print("Probes fitted successfully.")

    def extract(
        self,
        z_history: np.ndarray,
        actions: np.ndarray,
        values: np.ndarray,
        world_model_calls: Optional[np.ndarray] = None,
    ) -> List[AffectVector]:
        """
        Extract 6D affect vectors for each timestep.

        Args:
            z_history: (T, latent_dim)
            actions: (T,)
            values: (T,)
            world_model_calls: (T,) fraction of compute on world model

        Returns: List of AffectVector
        """
        if not self.probes_fitted:
            raise RuntimeError("Call fit_probes() first")

        T = len(z_history)
        affect_vectors = []

        # 1. Valence: advantage-based
        # A(s,a) = Q(s,a) - V(s) ≈ r + γV(s') - V(s)
        # We use value estimates directly
        advantages = np.zeros(T)
        for t in range(T - 1):
            advantages[t] = values[t + 1] - values[t]  # simplified advantage proxy

        # Valence v2: survival probe delta
        survival_preds = self.survival_probe.predict(z_history)
        survival_delta = np.zeros(T)
        for t in range(T - 1):
            survival_delta[t] = survival_preds[t + 1] - survival_preds[t]

        # Combine both valence measures
        valence = 0.5 * np.tanh(advantages) + 0.5 * np.tanh(survival_delta)

        # 2. Arousal: ||z_{t+1} - z_t||
        arousal = np.zeros(T)
        for t in range(T - 1):
            arousal[t] = np.linalg.norm(z_history[t + 1] - z_history[t])
        # Normalize to [0, 1]
        if arousal.max() > 0:
            arousal = arousal / arousal.max()

        # 3. Integration: partition prediction loss
        z_t = z_history[:-1]
        z_next = z_history[1:]
        integration_raw = self.partition_pred.compute_integration(z_t, z_next)
        integration = np.zeros(T)
        integration[:-1] = integration_raw
        if integration.max() > 0:
            integration = integration / integration.max()

        # 4. Effective rank: participation ratio from rolling covariance
        eff_rank = np.zeros(T)
        for t in range(self.window_size, T):
            window = z_history[t - self.window_size:t]
            cov = np.cov(window.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.maximum(eigenvalues, 0)
            trace = eigenvalues.sum()
            trace_sq = (eigenvalues ** 2).sum()
            if trace_sq > 0:
                eff_rank[t] = (trace ** 2) / trace_sq
            else:
                eff_rank[t] = 1.0
        # Normalize by latent dim
        eff_rank = eff_rank / self.latent_dim
        # Fill early timesteps
        if self.window_size < T:
            eff_rank[:self.window_size] = eff_rank[self.window_size]

        # 5. Counterfactual weight: world model usage fraction
        if world_model_calls is not None:
            cf_weight = world_model_calls
        else:
            # Proxy: how much the value function changes with hypothetical actions
            # (measure of "planning" in the latent space)
            cf_weight = np.zeros(T)
            for t in range(1, T):
                # Variance in value predictions as proxy for counterfactual exploration
                z_var = np.var(z_history[max(0, t - 5):t + 1], axis=0).mean()
                cf_weight[t] = np.tanh(z_var * 10)

        # 6. Self-model salience
        self_salience = self.self_model_probe.compute_salience(z_history, actions)

        # Build affect vectors
        for t in range(T):
            av = AffectVector(
                valence=float(valence[t]),
                arousal=float(arousal[t]),
                integration=float(integration[t]),
                effective_rank=float(eff_rank[t]),
                counterfactual_weight=float(cf_weight[t]),
                self_model_salience=float(self_salience[t]),
                step=t,
                raw={
                    'advantage': float(advantages[t]),
                    'survival_delta': float(survival_delta[t]),
                    'arousal_raw': float(arousal[t]),
                    'integration_raw': float(integration_raw[t - 1] if t > 0 and t <= len(integration_raw) else 0),
                    'eff_rank_raw': float(eff_rank[t] * self.latent_dim),
                },
            )
            affect_vectors.append(av)

        return affect_vectors

    def extract_matrix(self, affect_vectors: List[AffectVector]) -> np.ndarray:
        """Convert list of AffectVectors to (T, 6) matrix."""
        return np.array([av.to_array() for av in affect_vectors])


def compute_affect_for_agent(
    latent_history: List[Dict],
    agent_id: int,
    latent_dim: int,
    n_actions: int,
) -> Tuple[List[AffectVector], np.ndarray]:
    """
    Convenience function: extract affect for a single agent from history.

    Args:
        latent_history: List of dicts from training (z, reward, action, value, step)
        agent_id: Which agent to extract for
        latent_dim: Dimension of latent state
        n_actions: Number of discrete actions

    Returns: (affect_vectors, affect_matrix)
    """
    # Unpack history for this agent
    z_list = [h['z'][agent_id] for h in latent_history]
    action_list = [h['action'][agent_id] for h in latent_history]
    reward_list = [h['reward'][agent_id] for h in latent_history]
    value_list = [h['value'][agent_id] for h in latent_history]

    z_array = np.array(z_list)
    actions = np.array(action_list, dtype=int)
    rewards = np.array(reward_list)
    values = np.array(value_list)

    # Create extractor and fit probes
    extractor = AffectExtractor(latent_dim, n_actions)
    extractor.fit_probes(z_array, actions, rewards, values)

    # Extract affect
    affect_vectors = extractor.extract(z_array, actions, values)
    affect_matrix = extractor.extract_matrix(affect_vectors)

    return affect_vectors, affect_matrix


if __name__ == '__main__':
    # Test with synthetic data
    np.random.seed(42)
    T = 500
    latent_dim = 64
    n_actions = 8

    # Synthetic latent trajectory with structure
    z = np.random.randn(T, latent_dim) * 0.1
    for t in range(1, T):
        z[t] = 0.9 * z[t - 1] + 0.1 * np.random.randn(latent_dim)
        # Inject "threat" period
        if 200 <= t <= 300:
            z[t, :latent_dim // 2] *= 2  # high activity in first half
            z[t, latent_dim // 2:] *= 0.3  # suppressed second half

    actions = np.random.randint(0, n_actions, T)
    rewards = np.random.randn(T) * 0.1
    rewards[200:300] = -0.5  # threat period
    values = np.cumsum(rewards) * 0.01

    extractor = AffectExtractor(latent_dim, n_actions)
    extractor.fit_probes(z, actions, rewards, values)
    affect = extractor.extract(z, actions, values)
    matrix = extractor.extract_matrix(affect)

    print(f"Affect matrix shape: {matrix.shape}")
    print(f"Dimension names: {AffectVector.dim_names()}")
    print(f"\nMean affect (normal period 0-199):")
    print(f"  {dict(zip(AffectVector.dim_names(), matrix[:200].mean(axis=0).round(3)))}")
    print(f"\nMean affect (threat period 200-300):")
    print(f"  {dict(zip(AffectVector.dim_names(), matrix[200:300].mean(axis=0).round(3)))}")
    print(f"\nMean affect (recovery period 300-500):")
    print(f"  {dict(zip(AffectVector.dim_names(), matrix[300:].mean(axis=0).round(3)))}")
