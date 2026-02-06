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
        # JIT compile forward pass after init for speed
        self._jit_apply = jax.jit(self.network.apply)
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
        apply_fn = getattr(self, '_jit_apply', self.network.apply)
        outputs = apply_fn(params, obs, z)

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
        """Compute GAE advantages and returns using reverse scan."""
        gamma = self.ac.gamma
        lam = self.ac.gae_lambda

        def _gae_step(carry, t_data):
            last_gae = carry
            reward, value, done, next_value = t_data
            next_non_terminal = 1.0 - done
            delta = reward + gamma * next_value * next_non_terminal - value
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            return last_gae, last_gae

        # Build next_values: shifted values with last_value appended
        next_values = jnp.concatenate([values[1:], last_value[None]])

        # Reverse scan for GAE
        t_data = (rewards, values, dones, next_values)
        # Reverse the inputs, scan forward, reverse outputs
        rev_data = jax.tree.map(lambda x: x[::-1], t_data)
        _, rev_advantages = jax.lax.scan(_gae_step, 0.0, rev_data)
        advantages = rev_advantages[::-1]

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

    All agents are batched into a single forward pass per step.
    Uses JIT-compiled env methods for speed.
    Returns: (transitions, final_env_state, final_z)
    """
    transitions = []
    na = env.config.n_agents
    apply_fn = getattr(trainer, '_jit_apply', trainer.network.apply)
    use_jit_env = hasattr(env, '_jit_step')

    for t in range(n_steps):
        rng, step_rng, action_rng = random.split(rng, 3)

        # Get observations — already (n_agents, ...) shaped
        obs = env.jit_get_observations(env_state) if use_jit_env else env._get_observations(env_state)

        # Batched forward pass: all agents at once
        outputs = apply_fn(params, obs, z)

        # Sample actions for all agents
        action_logits = outputs['action_logits']  # (n_agents, n_actions)
        action_keys = random.split(action_rng, na)
        actions = jax.vmap(
            lambda logits, key: jax.random.categorical(key, logits)
        )(action_logits, action_keys)

        # Log probs
        log_probs_all = jax.nn.log_softmax(action_logits)
        log_probs = jnp.take_along_axis(
            log_probs_all, actions[..., None], axis=-1
        ).squeeze(-1)

        # Sample signals — vectorized across tokens
        signal_logits = outputs['signal_logits']  # (n_agents, n_tokens, vocab)
        n_tok = trainer.ac.n_signal_tokens
        signal_keys = random.split(step_rng, na * n_tok).reshape(na, n_tok, 2)
        # Flatten to (na*n_tok, vocab), sample, reshape back
        flat_logits = signal_logits.reshape(na * n_tok, -1)
        flat_keys = signal_keys.reshape(na * n_tok, 2)
        flat_signals = jax.vmap(
            lambda logits, key: jax.random.categorical(key, logits)
        )(flat_logits, flat_keys)
        signals = flat_signals.reshape(na, n_tok).astype(jnp.int32)

        values_arr = outputs['value']  # (n_agents,)
        z_new = outputs['z']  # (n_agents, latent_dim)
        obs_embs = outputs['obs_embedding']  # (n_agents, hidden_dim)

        # Step environment (JIT-compiled)
        if use_jit_env:
            env_state, next_obs, rewards, dones = env.jit_step(env_state, actions, signals)
        else:
            env_state, next_obs, rewards, dones = env.step(env_state, actions, signals)

        # Compute intrinsic rewards (batched, but skip extra forward pass —
        # use obs_embedding difference as proxy)
        if trainer.ac.use_intrinsic_motivation and 'predicted_next_obs' in outputs:
            next_obs_batch = env.jit_get_observations(env_state) if use_jit_env else env._get_observations(env_state)
            next_outputs = apply_fn(params, next_obs_batch, z_new)
            next_embs = next_outputs['obs_embedding']
            pred_errors = jnp.mean(
                (outputs['predicted_next_obs'] - next_embs) ** 2,
                axis=-1)
            intrinsic = jnp.tanh(pred_errors) * 0.1
            rewards = rewards + intrinsic

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
    Main training loop with JIT-compiled PPO updates.

    Returns dict with final params, training metrics, and latent state history.
    """
    import os
    import time
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

    steps_done = 0
    n_updates = 0

    print(f"Starting V10 training: {total_steps} steps, {env_config.n_agents} agents", flush=True)
    print(f"Forcing functions: WM={agent_config.use_world_model}, "
          f"SP={agent_config.use_self_prediction}, "
          f"IM={agent_config.use_intrinsic_motivation}, "
          f"PO={env_config.partial_observability}, "
          f"LH={env_config.long_horizons}, "
          f"DR={env_config.delayed_rewards}", flush=True)

    # JIT-compile the PPO update step
    @jax.jit
    def _jit_update(params, batch, rng):
        """Single JIT-compiled PPO update: loss + grad + metrics."""
        (loss, metrics), grads = jax.value_and_grad(
            lambda p: trainer.ppo_loss(p, batch, rng), has_aux=True
        )(params)
        return grads, loss, metrics

    # JIT-compile GAE for a single agent
    @jax.jit
    def _jit_gae(rewards, values, dones, last_value):
        return trainer.compute_gae(rewards, values, dones, last_value)

    # Warm up JIT compilations
    print("JIT compiling env step + observations...")
    t0 = time.time()
    _ = env.jit_get_observations(env_state)
    rng, _rng = random.split(rng)
    _actions = jnp.zeros(env_config.n_agents, dtype=jnp.int32)
    _signals = jnp.zeros((env_config.n_agents, agent_config.n_signal_tokens), dtype=jnp.int32)
    _ = env.jit_step(env_state, _actions, _signals)
    print(f"  env JIT done ({time.time() - t0:.1f}s)")

    print("JIT compiling forward pass...")
    t0 = time.time()
    apply_fn = trainer._jit_apply
    _out = apply_fn(ts.params, obs, z)
    print(f"  forward JIT done ({time.time() - t0:.1f}s)")

    while steps_done < total_steps:
        rng, rollout_rng = random.split(rng)
        t_rollout = time.time()

        # Collect rollout
        transitions, env_state, z = collect_rollout(
            trainer, ts.params, env, env_state, z, rollout_rng,
            n_steps=agent_config.n_steps,
        )

        # Compute last value for GAE (batched — all agents at once)
        last_obs = env.jit_get_observations(env_state)
        last_out = apply_fn(ts.params, last_obs, z)
        last_value = last_out['value']  # (n_agents,)

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
                    'obs_embedding': np.array(t.obs_embedding),
                    'step': steps_done,
                })

        # Per-agent GAE + PPO update
        metrics = {}
        for i in range(env_config.n_agents):
            advantages, returns = _jit_gae(
                rewards[:, i], values[:, i], dones[:, i], last_value[i]
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

            # PPO update (JIT-compiled)
            rng, update_rng = random.split(rng)
            for epoch in range(agent_config.n_epochs):
                grads, loss, metrics = _jit_update(ts.params, batch, update_rng)
                ts = ts.apply_gradients(grads=grads)

        steps_done += agent_config.n_steps
        n_updates += 1

        # Track rewards
        ep_reward = rewards.sum(axis=0).mean()
        episode_rewards.append(float(ep_reward))

        # Logging
        if steps_done % log_interval < agent_config.n_steps:
            dt = time.time() - t_rollout
            avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            avg_health = float(env_state.agent_health.mean())
            avg_hunger = float(env_state.agent_hunger.mean())
            signal_entropy = _signal_entropy(env_state.agent_signals, env_config.vocab_size)
            sps = agent_config.n_steps / max(dt, 0.01)
            print(f"Step {steps_done}/{total_steps} | "
                  f"Reward: {avg_reward:.3f} | "
                  f"Health: {avg_health:.1f} | "
                  f"Hunger: {avg_hunger:.1f} | "
                  f"Signal H: {signal_entropy:.2f} | "
                  f"Loss: {float(metrics.get('total_loss', 0)):.4f} | "
                  f"{sps:.0f} steps/s", flush=True)
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
            print(f"  Saved checkpoint at step {steps_done}", flush=True)

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
