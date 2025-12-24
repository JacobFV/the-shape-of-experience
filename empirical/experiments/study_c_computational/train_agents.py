"""
Train agents with different architectures and analyze what representational
structure emerges.

═══════════════════════════════════════════════════════════════════════════════
WARNING: COMMON MISINTERPRETATION - READ BEFORE USING THIS CODE
═══════════════════════════════════════════════════════════════════════════════

This experiment tests whether the RAW REPRESENTATION SPACE has 6 dimensions.
This is NOT what the thesis claims.

The thesis claims that the SIX AFFECT DIMENSIONS are COMPUTED QUANTITIES that
exist at a HIGHER LEVEL OF ABSTRACTION:

    1. Valence = gradient on viability manifold (computed from predicted futures)
    2. Arousal = rate of belief update (computed from model change over time)
    3. Integration = irreducibility (computed from partition analysis of structure)
    4. Effective Rank = active degrees of freedom (computed from state covariance)
    5. Counterfactual Weight = compute on non-actuals (computed from resource use)
    6. Self-Model Salience = self-focus (computed from attention distribution)

The raw state space can have THOUSANDS of dimensions. The claim is that when you
COMPUTE these six quantities from the raw state, they capture affect-relevant
structure.

This experiment is preserved as an example of a common misinterpretation.
See CORRECTION.md for details.

For the CORRECT test, see: ../study_c_llm_affect/
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import json
from pathlib import Path

from viability_env import ViabilityWorld


@dataclass
class TrainingConfig:
    """Hyperparameters for training."""
    hidden_size: int = 128
    num_layers: int = 2
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_episodes: int = 1000
    batch_size: int = 64
    update_epochs: int = 4


class SimpleAgent(nn.Module):
    """
    Baseline agent: Simple actor-critic without explicit world/self model.
    Theory prediction: Will develop ~2-3 dimensional internal representation.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Actor (policy)
        self.actor_mean = nn.Linear(hidden_size, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

        # Critic (value)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns hidden representation and value."""
        hidden = self.shared(obs)
        value = self.critic(hidden)
        return hidden, value

    def get_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        hidden, value = self.forward(obs)
        mean = self.actor_mean(hidden)
        std = torch.exp(self.actor_logstd)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate action under policy."""
        hidden, value = self.forward(obs)
        mean = self.actor_mean(hidden)
        std = torch.exp(self.actor_logstd)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, value, entropy

    def get_representation(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract internal representation for analysis."""
        return self.shared(obs)


class WorldModelAgent(nn.Module):
    """
    Agent with explicit world model but no self-model component.
    Theory prediction: Will develop ~3-4 dimensional representation
    (adding something like integration).
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128, latent_dim: int = 32):
        super().__init__()

        self.latent_dim = latent_dim

        # Encoder: obs -> latent state
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_dim * 2),  # mean and logvar
        )

        # World model: latent + action -> next latent
        self.world_model = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_dim * 2),
        )

        # Actor
        self.actor = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(hidden_size, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def encode(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode observation to latent distribution."""
        h = self.encoder(obs)
        mean, logvar = h.chunk(2, dim=-1)
        return mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from latent distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def predict_next(self, latent: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next latent state."""
        h = self.world_model(torch.cat([latent, action], dim=-1))
        mean, logvar = h.chunk(2, dim=-1)
        return mean, logvar

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns latent state and value."""
        mean, logvar = self.encode(obs)
        latent = self.reparameterize(mean, logvar)
        value = self.critic(latent)
        return latent, value

    def get_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        latent, value = self.forward(obs)
        hidden = self.actor(latent)
        mean = self.actor_mean(hidden)
        std = torch.exp(self.actor_logstd)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, value = self.forward(obs)
        hidden = self.actor(latent)
        mean = self.actor_mean(hidden)
        std = torch.exp(self.actor_logstd)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, value, entropy

    def get_representation(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract latent representation."""
        mean, logvar = self.encode(obs)
        return self.reparameterize(mean, logvar)


class SelfModelAgent(nn.Module):
    """
    Agent with both world model and self-model component.
    Theory prediction: Will develop 5-6 dimensional representation
    (adding self-salience, counterfactual weight).
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128,
                 world_latent_dim: int = 32, self_latent_dim: int = 16):
        super().__init__()

        self.world_latent_dim = world_latent_dim
        self.self_latent_dim = self_latent_dim
        total_latent = world_latent_dim + self_latent_dim

        # World encoder
        self.world_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, world_latent_dim * 2),
        )

        # Self encoder (focuses on internal state signals)
        # Input is subset of observations related to internal state
        self.self_encoder = nn.Sequential(
            nn.Linear(4, hidden_size // 2),  # First 4 obs are internal state
            nn.ReLU(),
            nn.Linear(hidden_size // 2, self_latent_dim * 2),
        )

        # World model
        self.world_model = nn.Sequential(
            nn.Linear(world_latent_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, world_latent_dim * 2),
        )

        # Self model (how actions affect self)
        self.self_model = nn.Sequential(
            nn.Linear(self_latent_dim + action_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, self_latent_dim * 2),
        )

        # Attention: how much to weight self vs world
        self.attention = nn.Sequential(
            nn.Linear(total_latent, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2),
            nn.Softmax(dim=-1),
        )

        # Actor
        self.actor = nn.Sequential(
            nn.Linear(total_latent, hidden_size),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(hidden_size, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(total_latent, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        # Counterfactual module (imagines alternate actions)
        self.counterfactual = nn.Sequential(
            nn.Linear(total_latent + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, total_latent),
        )

    def encode(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode into world and self latent states."""
        # World encoding
        hw = self.world_encoder(obs)
        world_mean, world_logvar = hw.chunk(2, dim=-1)

        # Self encoding (internal states only)
        hs = self.self_encoder(obs[..., :4])
        self_mean, self_logvar = hs.chunk(2, dim=-1)

        return world_mean, world_logvar, self_mean, self_logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns combined latent, value, and self-salience."""
        world_mean, world_logvar, self_mean, self_logvar = self.encode(obs)
        world_latent = self.reparameterize(world_mean, world_logvar)
        self_latent = self.reparameterize(self_mean, self_logvar)

        combined = torch.cat([world_latent, self_latent], dim=-1)

        # Compute attention (self-salience proxy)
        attn = self.attention(combined)
        self_salience = attn[..., 1]  # Weight on self component

        value = self.critic(combined)
        return combined, value, self_salience

    def get_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        combined, value, _ = self.forward(obs)
        hidden = self.actor(combined)
        mean = self.actor_mean(hidden)
        std = torch.exp(self.actor_logstd)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        combined, value, _ = self.forward(obs)
        hidden = self.actor(combined)
        mean = self.actor_mean(hidden)
        std = torch.exp(self.actor_logstd)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, value, entropy

    def get_representation(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract full combined representation."""
        combined, _, _ = self.forward(obs)
        return combined

    def get_self_salience(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract self-salience measure."""
        _, _, self_salience = self.forward(obs)
        return self_salience

    def imagine_counterfactual(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Imagine what would happen with different action."""
        combined, _, _ = self.forward(obs)
        return self.counterfactual(torch.cat([combined, action], dim=-1))


def collect_trajectory(env: ViabilityWorld, agent: nn.Module, max_steps: int = 1000) -> Dict:
    """Collect a single trajectory."""
    obs, _ = env.reset()
    observations = []
    actions = []
    rewards = []
    log_probs = []
    values = []
    dones = []
    viability_distances = []

    for _ in range(max_steps):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, value = agent.get_action(obs_tensor)

        action_np = action.squeeze().numpy()
        next_obs, reward, terminated, truncated, info = env.step(action_np)

        observations.append(obs)
        actions.append(action_np)
        rewards.append(reward)
        log_probs.append(log_prob.item())
        values.append(value.item())
        dones.append(terminated or truncated)
        viability_distances.append(env.get_true_viability_distance())

        obs = next_obs
        if terminated or truncated:
            break

    return {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'log_probs': np.array(log_probs),
        'values': np.array(values),
        'dones': np.array(dones),
        'viability_distances': np.array(viability_distances),
    }


def train_agent(
    agent_type: str,
    config: TrainingConfig,
    save_dir: Path,
    seed: int = 42
) -> Tuple[nn.Module, List[Dict]]:
    """Train an agent and save results."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = ViabilityWorld(seed=seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Create agent
    if agent_type == "simple":
        agent = SimpleAgent(obs_dim, action_dim, config.hidden_size)
    elif agent_type == "world_model":
        agent = WorldModelAgent(obs_dim, action_dim, config.hidden_size)
    elif agent_type == "self_model":
        agent = SelfModelAgent(obs_dim, action_dim, config.hidden_size)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate)

    training_history = []

    for episode in range(config.num_episodes):
        # Collect trajectory
        trajectory = collect_trajectory(env, agent)

        # Compute returns and advantages (GAE)
        returns = []
        advantages = []
        gae = 0
        next_value = 0

        for t in reversed(range(len(trajectory['rewards']))):
            if trajectory['dones'][t]:
                next_value = 0
                gae = 0

            delta = trajectory['rewards'][t] + config.gamma * next_value - trajectory['values'][t]
            gae = delta + config.gamma * config.gae_lambda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + trajectory['values'][t])
            next_value = trajectory['values'][t]

        returns = np.array(returns)
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        obs_tensor = torch.FloatTensor(trajectory['observations'])
        action_tensor = torch.FloatTensor(trajectory['actions'])
        returns_tensor = torch.FloatTensor(returns)
        advantages_tensor = torch.FloatTensor(advantages)
        old_log_probs = torch.FloatTensor(trajectory['log_probs'])

        # PPO update
        for _ in range(config.update_epochs):
            # Get current policy evaluation
            log_probs, values, entropy = agent.evaluate(obs_tensor, action_tensor)

            # PPO clipped objective
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = 0.5 * ((values.squeeze() - returns_tensor) ** 2).mean()

            # Entropy bonus
            entropy_loss = -entropy.mean()

            # Total loss
            loss = policy_loss + config.value_coef * value_loss + config.entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
            optimizer.step()

        # Log progress
        episode_return = trajectory['rewards'].sum()
        episode_length = len(trajectory['rewards'])
        mean_viability = trajectory['viability_distances'].mean()

        training_history.append({
            'episode': int(episode),
            'return': float(episode_return),
            'length': int(episode_length),
            'mean_viability': float(mean_viability),
            'policy_loss': float(policy_loss.item()),
            'value_loss': float(value_loss.item()),
        })

        if episode % 100 == 0:
            print(f"Episode {episode}: Return={episode_return:.2f}, "
                  f"Length={episode_length}, Viability={mean_viability:.3f}")

    # Save model
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), save_dir / f"{agent_type}_agent.pt")

    # Save training history
    with open(save_dir / f"{agent_type}_history.json", 'w') as f:
        json.dump(training_history, f)

    return agent, training_history


def main():
    """Train all agent types and save for analysis."""
    config = TrainingConfig(num_episodes=500)  # Reduced for testing
    save_dir = Path("trained_agents")

    for agent_type in ["simple", "world_model", "self_model"]:
        print(f"\n{'='*50}")
        print(f"Training {agent_type} agent")
        print('='*50)

        agent, history = train_agent(agent_type, config, save_dir)
        print(f"Final return: {history[-1]['return']:.2f}")


if __name__ == "__main__":
    main()
