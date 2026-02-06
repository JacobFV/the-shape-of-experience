"""
V10: Transformer Agent with Recurrent Latent State + PPO
=========================================================

Architecture:
- Observation encoder: CNN for grid + MLP for vitals/time/signals
- Recurrent latent state z_t via GRU gate
- Policy head: action logits + value estimate
- Communication head: signal token logits
- World model head (optional): predict next observation embedding
- Self-model head (optional): predict next z_{t+1}

Training: PPO with auxiliary losses for world model and self-prediction.

No pretraining. Random initialization. Everything learned from interaction.
"""

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from flax.training import train_state
import optax
from typing import Dict, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass
from functools import partial
import numpy as np

from v10_environment import EnvConfig


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class AgentConfig:
    """Agent architecture and training configuration."""
    # Latent state
    latent_dim: int = 64
    hidden_dim: int = 128

    # CNN encoder
    cnn_features: Tuple = (16, 32, 32)
    cnn_kernels: Tuple = (3, 3, 3)

    # Transformer encoder
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 128
    dropout_rate: float = 0.0

    # Communication
    vocab_size: int = 32
    n_signal_tokens: int = 2

    # PPO hyperparameters
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    n_minibatches: int = 4
    n_steps: int = 128  # steps per rollout

    # Auxiliary losses
    world_model_coef: float = 0.5
    self_pred_coef: float = 0.5
    comm_cost_coef: float = 0.001

    # Forcing function toggles (mirrors env config)
    use_world_model: bool = True
    use_self_prediction: bool = True
    use_intrinsic_motivation: bool = True


# ============================================================================
# Neural network modules
# ============================================================================

class CNNEncoder(nn.Module):
    """Encode grid observation via CNN."""
    features: Tuple = (16, 32, 32)
    kernels: Tuple = (3, 3, 3)

    @nn.compact
    def __call__(self, x, train: bool = True):
        for feat, kern in zip(self.features, self.kernels):
            x = nn.Conv(feat, (kern, kern), padding='SAME')(x)
            x = nn.relu(x)
        # Global average pooling
        x = jnp.mean(x, axis=(-3, -2))  # (batch, features)
        return x


class GRUCell(nn.Module):
    """GRU-style gated recurrent cell for latent state update."""
    hidden_dim: int

    @nn.compact
    def __call__(self, h, x):
        # Concatenate input and hidden state
        combined = jnp.concatenate([h, x], axis=-1)

        # Update gate
        z = nn.sigmoid(nn.Dense(self.hidden_dim, name='z_gate')(combined))
        # Reset gate
        r = nn.sigmoid(nn.Dense(self.hidden_dim, name='r_gate')(combined))
        # Candidate
        combined_r = jnp.concatenate([r * h, x], axis=-1)
        h_tilde = jnp.tanh(nn.Dense(self.hidden_dim, name='h_candidate')(combined_r))
        # New hidden state
        h_new = (1 - z) * h + z * h_tilde
        return h_new


class TransformerBlock(nn.Module):
    """Single transformer encoder block."""
    n_heads: int
    ff_dim: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool = True):
        d_model = x.shape[-1]
        # Self-attention
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=d_model,
        )(x, x)
        x = nn.LayerNorm()(x + attn_out)
        # Feed-forward
        ff_out = nn.Dense(self.ff_dim)(x)
        ff_out = nn.relu(ff_out)
        ff_out = nn.Dense(d_model)(ff_out)
        x = nn.LayerNorm()(x + ff_out)
        return x


class AgentNetwork(nn.Module):
    """
    Full agent network: observation → latent → action/value/signal.

    Designed to expose latent state z_t for affect extraction.
    """
    config: AgentConfig
    env_config: EnvConfig

    @nn.compact
    def __call__(
        self,
        obs: Dict[str, jnp.ndarray],
        z_prev: jnp.ndarray,
        train: bool = True,
    ) -> Dict[str, jnp.ndarray]:
        cfg = self.config
        ecfg = self.env_config

        # 1. Encode grid observation
        grid_enc = CNNEncoder(cfg.cnn_features, cfg.cnn_kernels)(obs['grid'], train)

        # 2. Encode vitals, time, signals
        vitals_enc = nn.Dense(32)(obs['vitals'])
        vitals_enc = nn.relu(vitals_enc)
        time_enc = nn.Dense(16)(obs['time'])
        time_enc = nn.relu(time_enc)
        signal_enc = nn.Dense(32)(obs['signals'])
        signal_enc = nn.relu(signal_enc)

        # 3. Combine into observation embedding
        obs_emb = jnp.concatenate([grid_enc, vitals_enc, time_enc, signal_enc], axis=-1)
        obs_emb = nn.Dense(cfg.hidden_dim)(obs_emb)
        obs_emb = nn.relu(obs_emb)

        # 4. Transformer encoder (process observation embedding as sequence of 1)
        # Expand to sequence dim for transformer
        x = obs_emb[..., None, :]  # (..., 1, hidden_dim)
        # Add z_prev as context token
        z_token = nn.Dense(cfg.hidden_dim)(z_prev)[..., None, :]  # (..., 1, hidden_dim)
        x = jnp.concatenate([z_token, x], axis=-2)  # (..., 2, hidden_dim)

        for _ in range(cfg.n_layers):
            x = TransformerBlock(cfg.n_heads, cfg.ff_dim, cfg.dropout_rate)(x, train)

        # Take the observation token output
        obs_processed = x[..., 1, :]  # (..., hidden_dim)

        # 5. Update latent state via GRU
        z_new = GRUCell(cfg.latent_dim)(z_prev, obs_processed)

        # 6. Policy head
        policy_hidden = nn.Dense(cfg.hidden_dim)(z_new)
        policy_hidden = nn.relu(policy_hidden)
        action_logits = nn.Dense(ecfg.n_actions)(policy_hidden)

        # 7. Value head
        value_hidden = nn.Dense(cfg.hidden_dim)(z_new)
        value_hidden = nn.relu(value_hidden)
        value = nn.Dense(1)(value_hidden).squeeze(-1)

        # 8. Communication head
        comm_hidden = nn.Dense(cfg.hidden_dim)(z_new)
        comm_hidden = nn.relu(comm_hidden)
        signal_logits = nn.Dense(cfg.vocab_size * cfg.n_signal_tokens)(comm_hidden)
        signal_logits = signal_logits.reshape(*z_new.shape[:-1], cfg.n_signal_tokens, cfg.vocab_size)

        outputs = {
            'z': z_new,
            'action_logits': action_logits,
            'value': value,
            'signal_logits': signal_logits,
            'obs_embedding': obs_emb,
        }

        # 9. World model head (optional)
        if cfg.use_world_model:
            wm_hidden = nn.Dense(cfg.hidden_dim, name='world_model_1')(z_new)
            wm_hidden = nn.relu(wm_hidden)
            predicted_next_obs = nn.Dense(cfg.hidden_dim, name='world_model_2')(wm_hidden)
            outputs['predicted_next_obs'] = predicted_next_obs

        # 10. Self-model head (optional)
        if cfg.use_self_prediction:
            sm_hidden = nn.Dense(cfg.hidden_dim, name='self_model_1')(z_new)
            sm_hidden = nn.relu(sm_hidden)
            predicted_next_z = nn.Dense(cfg.latent_dim, name='self_model_2')(sm_hidden)
            outputs['predicted_next_z'] = predicted_next_z

        return outputs

    def initial_latent(self, batch_shape=()):
        """Return zero-initialized latent state."""
        return jnp.zeros((*batch_shape, self.config.latent_dim))


# ============================================================================
# Rollout storage
# ============================================================================

class Transition(NamedTuple):
    """Single transition for PPO."""
    obs: Dict[str, jnp.ndarray]
    action: jnp.ndarray
    signal: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    value: jnp.ndarray
    log_prob: jnp.ndarray
    z: jnp.ndarray  # latent state (for affect extraction)
    obs_embedding: jnp.ndarray  # (for world model target)


# ============================================================================
# PPO Training
# ============================================================================

class PPOTrainer:
    """PPO trainer with auxiliary losses."""

    def __init__(self, agent_config: AgentConfig, env_config: EnvConfig):
        self.ac = agent_config
        self.ec = env_config

        # Create network
        self.network = AgentNetwork(agent_config, env_config)

        # Create optimizer
        self.tx = optax.chain(
            optax.clip_by_global_norm(agent_config.max_grad_norm),
            optax.adam(agent_config.lr),
        )

    def init_params(self, rng: jnp.ndarray) -> Dict:
        """Initialize network parameters."""
        obs_shapes = {
            'grid': jnp.zeros((1, 2 * self.ec.obs_radius + 1, 2 * self.ec.obs_radius + 1, self.ec.n_channels)),
            'vitals': jnp.zeros((1, 3)),
            'time': jnp.zeros((1, 2)),
            'signals': jnp.zeros((1, self.ec.n_agents * self.ec.n_signal_tokens)),
        }
        z_init = jnp.zeros((1, self.ac.latent_dim))
        params = self.network.init(rng, obs_shapes, z_init)
        return params

    def create_train_state(self, rng: jnp.ndarray) -> train_state.TrainState:
        """Create initial training state."""
        params = self.init_params(rng)
        return train_state.TrainState.create(
            apply_fn=self.network.apply,
            params=params,
            tx=self.tx,
        )

    def select_action(
        self,
        params: Dict,
        obs: Dict[str, jnp.ndarray],
        z: jnp.ndarray,
        rng: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict]:
        """
        Select action and signal given observation and latent state.

        Returns: (action, signal_tokens, log_prob, network_outputs)
        """
        outputs = self.network.apply(params, obs, z)

        # Sample action
        action_logits = outputs['action_logits']
        action = jax.random.categorical(rng, action_logits)
        log_prob = jax.nn.log_softmax(action_logits)[..., action]

        # Sample signal tokens
        signal_logits = outputs['signal_logits']  # (..., n_tokens, vocab_size)
        rng_signals = random.split(rng, self.ac.n_signal_tokens)
        signal_tokens = []
        for t in range(self.ac.n_signal_tokens):
            token = jax.random.categorical(rng_signals[t], signal_logits[..., t, :])
            signal_tokens.append(token)
        signal_tokens = jnp.stack(signal_tokens, axis=-1)

        return action, signal_tokens, log_prob, outputs

    def compute_gae(
        self,
        rewards: jnp.ndarray,
        values: jnp.ndarray,
        dones: jnp.ndarray,
        last_value: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute GAE advantages and returns."""
        gamma = self.ac.gamma
        lam = self.ac.gae_lambda
        n_steps = rewards.shape[0]

        advantages = jnp.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            advantages = advantages.at[t].set(last_gae)

        returns = advantages + values
        return advantages, returns

    def ppo_loss(
        self,
        params: Dict,
        batch: Dict,
        rng: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Compute PPO loss with auxiliary losses."""
        # Forward pass
        outputs = self.network.apply(params, batch['obs'], batch['z'])

        # Policy loss
        action_logits = outputs['action_logits']
        log_probs = jax.nn.log_softmax(action_logits)
        action_log_probs = jnp.take_along_axis(
            log_probs, batch['action'][..., None], axis=-1
        ).squeeze(-1)

        ratio = jnp.exp(action_log_probs - batch['old_log_prob'])
        advantages = batch['advantage']
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        surr1 = ratio * advantages
        surr2 = jnp.clip(ratio, 1 - self.ac.clip_eps, 1 + self.ac.clip_eps) * advantages
        policy_loss = -jnp.minimum(surr1, surr2).mean()

        # Value loss
        value_pred = outputs['value']
        value_loss = 0.5 * ((value_pred - batch['return']) ** 2).mean()

        # Entropy bonus
        probs = jax.nn.softmax(action_logits)
        entropy = -(probs * log_probs).sum(-1).mean()

        # Total PPO loss
        loss = policy_loss + self.ac.vf_coef * value_loss - self.ac.ent_coef * entropy

        aux_losses = {}

        # World model loss (predict next observation embedding)
        if self.ac.use_world_model and 'predicted_next_obs' in outputs:
            wm_loss = ((outputs['predicted_next_obs'] - batch['next_obs_embedding']) ** 2).mean()
            loss = loss + self.ac.world_model_coef * wm_loss
            aux_losses['world_model'] = wm_loss

        # Self-prediction loss (predict next latent state)
        if self.ac.use_self_prediction and 'predicted_next_z' in outputs:
            sp_loss = ((outputs['predicted_next_z'] - batch['next_z']) ** 2).mean()
            loss = loss + self.ac.self_pred_coef * sp_loss
            aux_losses['self_prediction'] = sp_loss

        metrics = {
            'total_loss': loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            **aux_losses,
        }

        return loss, metrics

    def compute_intrinsic_reward(
        self,
        params: Dict,
        obs: Dict[str, jnp.ndarray],
        z: jnp.ndarray,
        next_obs_embedding: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute intrinsic reward (curiosity bonus) from world model prediction error.
        """
        if not self.ac.use_intrinsic_motivation:
            return jnp.zeros(())

        outputs = self.network.apply(params, obs, z)
        if 'predicted_next_obs' not in outputs:
            return jnp.zeros(())

        pred_error = jnp.mean((outputs['predicted_next_obs'] - next_obs_embedding) ** 2)
        # Normalize to reasonable scale
        intrinsic_reward = jnp.tanh(pred_error)
        return intrinsic_reward * 0.1  # scale factor


# ============================================================================
# Rollout collection
# ============================================================================

def collect_rollout(
    trainer: PPOTrainer,
    params: Dict,
    env,
    env_state,
    z: jnp.ndarray,
    rng: jnp.ndarray,
    n_steps: int,
) -> Tuple[list, Any, jnp.ndarray]:
    """
    Collect a rollout of n_steps from the environment.

    Returns: (transitions, final_env_state, final_z)
    """
    transitions = []
    na = env.config.n_agents

    for t in range(n_steps):
        rng, step_rng = random.split(rng)
        step_keys = random.split(step_rng, na)

        # Get observations
        obs = env._get_observations(env_state)

        # Each agent selects action
        actions = []
        signals = []
        log_probs = []
        values = []
        z_list = []
        obs_embeddings = []

        for i in range(na):
            agent_obs = {k: v[i:i+1] for k, v in obs.items()}
            agent_z = z[i:i+1]

            action, signal, log_prob, outputs = trainer.select_action(
                params, agent_obs, agent_z, step_keys[i]
            )
            actions.append(action.squeeze())
            signals.append(signal.squeeze())
            log_probs.append(log_prob.squeeze())
            values.append(outputs['value'].squeeze())
            z_list.append(outputs['z'].squeeze())
            obs_embeddings.append(outputs['obs_embedding'].squeeze())

        actions = jnp.stack(actions)
        signals = jnp.stack(signals)
        log_probs = jnp.stack(log_probs)
        values_arr = jnp.stack(values)
        z_new = jnp.stack(z_list)
        obs_embs = jnp.stack(obs_embeddings)

        # Step environment
        env_state, next_obs, rewards, dones = env.step(env_state, actions, signals)

        # Compute intrinsic rewards
        if trainer.ac.use_intrinsic_motivation:
            next_obs_agent = env._get_observations(env_state)
            for i in range(na):
                agent_next_obs = {k: v[i:i+1] for k, v in next_obs_agent.items()}
                next_emb = trainer.network.apply(
                    params,
                    agent_next_obs,
                    z_new[i:i+1],
                )['obs_embedding'].squeeze()
                ir = trainer.compute_intrinsic_reward(
                    params, {k: v[i:i+1] for k, v in obs.items()},
                    z[i:i+1], next_emb[None]
                )
                rewards = rewards.at[i].add(ir)

        transitions.append(Transition(
            obs=obs,
            action=actions,
            signal=signals,
            reward=rewards,
            done=dones,
            value=values_arr,
            log_prob=log_probs,
            z=z,  # latent state BEFORE this step
            obs_embedding=obs_embs,
        ))

        z = z_new

    return transitions, env_state, z


# ============================================================================
# Training loop
# ============================================================================

def train(
    agent_config: AgentConfig,
    env_config: EnvConfig,
    total_steps: int = 1_000_000,
    seed: int = 42,
    log_interval: int = 10_000,
    save_interval: int = 100_000,
    save_dir: str = 'results/v10',
) -> Dict:
    """
    Main training loop.

    Returns dict with final params, training metrics, and latent state history.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    from v10_environment import SurvivalGridWorld

    rng = random.PRNGKey(seed)
    rng, init_rng, env_rng = random.split(rng, 3)

    # Initialize
    env = SurvivalGridWorld(env_config)
    trainer = PPOTrainer(agent_config, env_config)
    ts = trainer.create_train_state(init_rng)
    env_state, obs = env.reset(env_rng)
    z = jnp.zeros((env_config.n_agents, agent_config.latent_dim))

    # Training metrics
    metrics_history = []
    latent_history = []  # for affect extraction
    episode_rewards = []
    total_reward = jnp.zeros(env_config.n_agents)

    steps_done = 0
    n_updates = 0

    print(f"Starting V10 training: {total_steps} steps, {env_config.n_agents} agents")
    print(f"Forcing functions: WM={agent_config.use_world_model}, "
          f"SP={agent_config.use_self_prediction}, "
          f"IM={agent_config.use_intrinsic_motivation}, "
          f"PO={env_config.partial_observability}, "
          f"LH={env_config.long_horizons}, "
          f"DR={env_config.delayed_rewards}")

    while steps_done < total_steps:
        rng, rollout_rng = random.split(rng)

        # Collect rollout
        transitions, env_state, z = collect_rollout(
            trainer, ts.params, env, env_state, z, rollout_rng,
            n_steps=agent_config.n_steps,
        )

        # Compute last value for GAE
        last_obs = env._get_observations(env_state)
        last_values = []
        for i in range(env_config.n_agents):
            agent_obs = {k: v[i:i+1] for k, v in last_obs.items()}
            out = trainer.network.apply(ts.params, agent_obs, z[i:i+1])
            last_values.append(out['value'].squeeze())
        last_value = jnp.stack(last_values)

        # Stack transitions
        rewards = jnp.stack([t.reward for t in transitions])  # (n_steps, n_agents)
        values = jnp.stack([t.value for t in transitions])
        dones = jnp.stack([t.done for t in transitions])

        # Save latent states for affect extraction (subsample)
        if steps_done % (log_interval // 10) < agent_config.n_steps:
            for t in transitions:
                latent_history.append({
                    'z': np.array(t.z),
                    'reward': np.array(t.reward),
                    'action': np.array(t.action),
                    'signal': np.array(t.signal),
                    'value': np.array(t.value),
                    'step': steps_done,
                })

        # Per-agent GAE
        for i in range(env_config.n_agents):
            agent_rewards = rewards[:, i]
            agent_values = values[:, i]
            agent_dones = dones[:, i]
            advantages, returns = trainer.compute_gae(
                agent_rewards, agent_values, agent_dones, last_value[i]
            )

            # Build batch for this agent
            batch = {
                'obs': {k: jnp.stack([t.obs[k][i:i+1] for t in transitions]).squeeze(1)
                        for k in transitions[0].obs},
                'action': jnp.stack([t.action[i] for t in transitions]),
                'z': jnp.stack([t.z[i] for t in transitions]),
                'old_log_prob': jnp.stack([t.log_prob[i] for t in transitions]),
                'advantage': advantages,
                'return': returns,
            }

            # Next obs embeddings for world model
            if agent_config.use_world_model:
                next_embs = jnp.stack([t.obs_embedding[i] for t in transitions[1:]] +
                                       [transitions[-1].obs_embedding[i]])
                batch['next_obs_embedding'] = next_embs

            # Next z for self-prediction
            if agent_config.use_self_prediction:
                next_zs = jnp.stack([t.z[i] for t in transitions[1:]] +
                                     [z[i]])
                batch['next_z'] = next_zs

            # PPO update
            rng, update_rng = random.split(rng)
            for epoch in range(agent_config.n_epochs):
                loss, metrics = trainer.ppo_loss(ts.params, batch, update_rng)
                grads = jax.grad(lambda p: trainer.ppo_loss(p, batch, update_rng)[0])(ts.params)
                ts = ts.apply_gradients(grads=grads)

        steps_done += agent_config.n_steps
        n_updates += 1

        # Track rewards
        ep_reward = rewards.sum(axis=0).mean()
        episode_rewards.append(float(ep_reward))

        # Logging
        if steps_done % log_interval < agent_config.n_steps:
            avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            avg_health = float(env_state.agent_health.mean())
            avg_hunger = float(env_state.agent_hunger.mean())
            signal_entropy = _signal_entropy(env_state.agent_signals, env_config.vocab_size)
            print(f"Step {steps_done}/{total_steps} | "
                  f"Reward: {avg_reward:.3f} | "
                  f"Health: {avg_health:.1f} | "
                  f"Hunger: {avg_hunger:.1f} | "
                  f"Signal H: {signal_entropy:.2f} | "
                  f"Loss: {float(metrics.get('total_loss', 0)):.4f}")
            metrics_history.append({
                'step': steps_done,
                'reward': avg_reward,
                'health': avg_health,
                'hunger': avg_hunger,
                'signal_entropy': signal_entropy,
                **{k: float(v) for k, v in metrics.items()},
            })

        # Save checkpoint
        if steps_done % save_interval < agent_config.n_steps:
            checkpoint = {
                'params': ts.params,
                'step': steps_done,
                'metrics': metrics_history,
            }
            import pickle
            with open(f'{save_dir}/checkpoint_{steps_done}.pkl', 'wb') as f:
                pickle.dump(checkpoint, f)
            print(f"  Saved checkpoint at step {steps_done}")

    # Save final results
    results = {
        'params': ts.params,
        'metrics_history': metrics_history,
        'latent_history': latent_history,
        'episode_rewards': episode_rewards,
        'config': {
            'agent': agent_config,
            'env': env_config,
        },
    }

    import pickle
    with open(f'{save_dir}/final_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    return results


def _signal_entropy(signals: jnp.ndarray, vocab_size: int) -> float:
    """Compute entropy of signal token distribution."""
    flat = signals.flatten()
    counts = jnp.bincount(flat, length=vocab_size)
    probs = counts / counts.sum()
    probs = jnp.where(probs > 0, probs, 1e-10)
    return float(-jnp.sum(probs * jnp.log(probs)))


if __name__ == '__main__':
    # Quick architecture test
    from v10_environment import EnvConfig
    ac = AgentConfig()
    ec = EnvConfig(n_agents=4, grid_size=8)

    rng = random.PRNGKey(0)
    trainer = PPOTrainer(ac, ec)
    params = trainer.init_params(rng)

    # Test forward pass
    obs = {
        'grid': jnp.zeros((1, 7, 7, 7)),
        'vitals': jnp.zeros((1, 3)),
        'time': jnp.zeros((1, 2)),
        'signals': jnp.zeros((1, ec.n_agents * ec.n_signal_tokens)),
    }
    z = jnp.zeros((1, ac.latent_dim))

    outputs = trainer.network.apply(params, obs, z)
    print("Network outputs:")
    for k, v in outputs.items():
        if isinstance(v, jnp.ndarray):
            print(f"  {k}: {v.shape}")

    print(f"\nLatent dim: {ac.latent_dim}")
    print(f"Action space: {ec.n_actions}")
    print(f"Vocab size: {ac.vocab_size}")
