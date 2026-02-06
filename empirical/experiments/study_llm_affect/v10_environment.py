"""
V10: MARL Survival Grid World Environment
==========================================

A JAX-compatible grid world for training randomly-initialized RL agents
under viability pressure with emergent communication.

Key features:
- Resource patches (food, water, materials) with slow regeneration
- Environmental threats (storms, predators) with stochastic dynamics
- Day/night cycle (partial observability shifts)
- Discrete communication channel (K-token vocabulary)
- Forcing function toggles for ablation studies
- Cooperative tasks requiring 2+ agents

This is NOT an LLM experiment. Agents are randomly-initialized neural networks
trained from scratch via PPO. No human data, no pretraining.
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
from functools import partial

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class EnvConfig:
    """Environment configuration with forcing function toggles."""
    # Grid
    grid_size: int = 16
    n_agents: int = 4
    max_steps: int = 2000

    # Observation
    obs_radius: int = 3  # 7x7 egocentric view (radius 3)
    n_channels: int = 7  # agent, food, water, material, threat, wall, signal

    # Resources
    n_food_patches: int = 8
    n_water_patches: int = 6
    n_material_patches: int = 4
    resource_regen_rate: float = 0.02  # probability per step per depleted patch
    cooperative_harvest_bonus: float = 2.0  # multiplier when 2+ agents harvest

    # Threats
    n_predators: int = 2
    predator_damage: int = 3
    storm_probability: float = 0.01  # per step
    storm_duration: int = 20
    storm_damage: int = 1  # per step in storm

    # Agent vitals
    max_health: int = 20
    max_hunger: int = 20
    max_thirst: int = 20
    hunger_rate: int = 1  # per step
    thirst_rate: int = 1  # per step
    starvation_damage: int = 2  # when hunger hits 0
    dehydration_damage: int = 2  # when thirst hits 0
    food_restore: int = 5
    water_restore: int = 5

    # Communication
    vocab_size: int = 32  # discrete token vocabulary
    n_signal_tokens: int = 2  # tokens per message

    # Day/night cycle
    day_length: int = 100  # steps per full cycle
    night_obs_radius: int = 1  # reduced visibility at night (3x3)

    # Seasonal scarcity
    season_length: int = 500  # steps per season
    winter_regen_multiplier: float = 0.2  # resources regenerate slower in winter

    # Actions: move(4), gather, share, attack, signal
    n_actions: int = 8  # 4 move + gather + share + attack + signal

    # Forcing function toggles (for ablation)
    partial_observability: bool = True   # FF1: egocentric view vs full grid
    long_horizons: bool = True           # FF2: seasonal cycles, long episodes
    use_world_model: bool = True         # FF3: auxiliary prediction loss
    use_self_prediction: bool = True     # FF4: predict own next state
    use_intrinsic_motivation: bool = True  # FF5: curiosity bonus
    delayed_rewards: bool = True         # FF6: credit assignment under delay

    # Reward shaping
    survival_reward: float = 0.01  # per step alive
    death_penalty: float = -1.0
    gather_reward: float = 0.1
    cooperation_reward: float = 0.2
    communication_cost: float = 0.001  # slight cost to signaling


# ============================================================================
# State types
# ============================================================================

class AgentState(NamedTuple):
    """Per-agent state."""
    position: jnp.ndarray      # (2,) int - x, y
    health: jnp.ndarray        # () int
    hunger: jnp.ndarray        # () int
    thirst: jnp.ndarray        # () int
    alive: jnp.ndarray         # () bool
    last_signal: jnp.ndarray   # (n_signal_tokens,) int - last emitted signal


class EnvState(NamedTuple):
    """Full environment state."""
    # Grid layers: (grid_size, grid_size) for each
    food_grid: jnp.ndarray       # resource amounts
    water_grid: jnp.ndarray
    material_grid: jnp.ndarray
    threat_grid: jnp.ndarray     # predator/storm presence

    # Agent states: (n_agents, ...) for each field
    agent_positions: jnp.ndarray  # (n_agents, 2)
    agent_health: jnp.ndarray    # (n_agents,)
    agent_hunger: jnp.ndarray    # (n_agents,)
    agent_thirst: jnp.ndarray    # (n_agents,)
    agent_alive: jnp.ndarray     # (n_agents,)
    agent_signals: jnp.ndarray   # (n_agents, n_signal_tokens)

    # Predator positions
    predator_positions: jnp.ndarray  # (n_predators, 2)

    # Storm state
    storm_active: jnp.ndarray    # () bool
    storm_center: jnp.ndarray    # (2,) int
    storm_timer: jnp.ndarray     # () int

    # Time
    step: jnp.ndarray            # () int
    rng: jnp.ndarray             # PRNG key


# ============================================================================
# Environment
# ============================================================================

class SurvivalGridWorld:
    """
    JAX-compatible survival grid world for MARL affect experiments.

    All operations are pure functions on (EnvState, actions) -> (EnvState, obs, rewards).
    Designed for JAX jit compilation and vmap over seeds.
    """

    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config or EnvConfig()
        self.c = self.config  # shorthand
        # JIT-compiled versions (created after first use)
        self._jit_step = None
        self._jit_obs = None

    def jit_step(self, state, actions, signal_tokens):
        """JIT-compiled environment step."""
        if self._jit_step is None:
            self._jit_step = jax.jit(self.step)
        return self._jit_step(state, actions, signal_tokens)

    def jit_get_observations(self, state):
        """JIT-compiled observation function."""
        if self._jit_obs is None:
            self._jit_obs = jax.jit(self._get_observations)
        return self._jit_obs(state)

    def reset(self, rng: jnp.ndarray) -> Tuple[EnvState, jnp.ndarray]:
        """Initialize environment. Returns (state, observations)."""
        keys = random.split(rng, 10)

        gs = self.c.grid_size
        na = self.c.n_agents

        # Place resources randomly
        food_grid = jnp.zeros((gs, gs))
        water_grid = jnp.zeros((gs, gs))
        material_grid = jnp.zeros((gs, gs))

        # Scatter resource patches
        food_positions = random.randint(keys[0], (self.c.n_food_patches, 2), 0, gs)
        water_positions = random.randint(keys[1], (self.c.n_water_patches, 2), 0, gs)
        material_positions = random.randint(keys[2], (self.c.n_material_patches, 2), 0, gs)

        food_grid = food_grid.at[food_positions[:, 0], food_positions[:, 1]].set(5.0)
        water_grid = water_grid.at[water_positions[:, 0], water_positions[:, 1]].set(5.0)
        material_grid = material_grid.at[material_positions[:, 0], material_positions[:, 1]].set(3.0)

        # Place agents randomly
        agent_positions = random.randint(keys[3], (na, 2), 0, gs)
        agent_health = jnp.full(na, self.c.max_health, dtype=jnp.float32)
        agent_hunger = jnp.full(na, self.c.max_hunger, dtype=jnp.float32)
        agent_thirst = jnp.full(na, self.c.max_thirst, dtype=jnp.float32)
        agent_alive = jnp.ones(na, dtype=jnp.bool_)
        agent_signals = jnp.zeros((na, self.c.n_signal_tokens), dtype=jnp.int32)

        # Place predators
        predator_positions = random.randint(keys[4], (self.c.n_predators, 2), 0, gs)

        threat_grid = jnp.zeros((gs, gs))
        for i in range(self.c.n_predators):
            threat_grid = threat_grid.at[
                predator_positions[i, 0], predator_positions[i, 1]
            ].set(1.0)

        state = EnvState(
            food_grid=food_grid,
            water_grid=water_grid,
            material_grid=material_grid,
            threat_grid=threat_grid,
            agent_positions=agent_positions,
            agent_health=agent_health,
            agent_hunger=agent_hunger,
            agent_thirst=agent_thirst,
            agent_alive=agent_alive,
            agent_signals=agent_signals,
            predator_positions=predator_positions,
            storm_active=jnp.array(False),
            storm_center=jnp.array([gs // 2, gs // 2]),
            storm_timer=jnp.array(0),
            step=jnp.array(0),
            rng=keys[9],
        )

        obs = self._get_observations(state)
        return state, obs

    def step(
        self,
        state: EnvState,
        actions: jnp.ndarray,       # (n_agents,) int - action indices
        signal_tokens: jnp.ndarray,  # (n_agents, n_signal_tokens) int
    ) -> Tuple[EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Execute one environment step.

        Returns: (new_state, observations, rewards, dones)
        """
        rng, *keys = random.split(state.rng, 7)
        na = self.c.n_agents
        gs = self.c.grid_size

        # 1. Process movement (actions 0-3: up, down, left, right)
        directions = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # up, down, left, right
        is_move = actions < 4
        move_dirs = jnp.where(
            is_move[:, None],
            directions[jnp.clip(actions, 0, 3)],
            jnp.zeros((na, 2), dtype=jnp.int32)
        )
        new_positions = state.agent_positions + move_dirs
        new_positions = jnp.clip(new_positions, 0, gs - 1)
        # Only move if alive
        new_positions = jnp.where(state.agent_alive[:, None], new_positions, state.agent_positions)

        # 2. Process gathering (action 4)
        is_gather = (actions == 4) & state.agent_alive
        rewards = jnp.zeros(na)

        food_grid = state.food_grid
        water_grid = state.water_grid
        material_grid = state.material_grid

        new_hunger = state.agent_hunger
        new_thirst = state.agent_thirst

        # Check for cooperative harvesting (2+ agents on same cell gathering)
        for i in range(na):
            pos = new_positions[i]
            gathering = is_gather[i]

            # Count nearby gatherers for cooperation bonus
            same_cell = jnp.all(new_positions == pos[None, :], axis=1) & is_gather
            n_coop = jnp.sum(same_cell)
            coop_mult = jnp.where(n_coop > 1, self.c.cooperative_harvest_bonus, 1.0)

            # Gather food
            food_here = food_grid[pos[0], pos[1]]
            food_gathered = jnp.where(gathering & (food_here > 0), 1.0 * coop_mult, 0.0)
            food_grid = food_grid.at[pos[0], pos[1]].add(-jnp.where(gathering, jnp.minimum(food_here, 1.0), 0.0))
            new_hunger = new_hunger.at[i].set(
                jnp.minimum(new_hunger[i] + food_gathered * self.c.food_restore, self.c.max_hunger)
            )

            # Gather water
            water_here = water_grid[pos[0], pos[1]]
            water_gathered = jnp.where(gathering & (water_here > 0), 1.0 * coop_mult, 0.0)
            water_grid = water_grid.at[pos[0], pos[1]].add(-jnp.where(gathering, jnp.minimum(water_here, 1.0), 0.0))
            new_thirst = new_thirst.at[i].set(
                jnp.minimum(new_thirst[i] + water_gathered * self.c.water_restore, self.c.max_thirst)
            )

            # Reward for gathering
            gathered_any = (food_gathered > 0) | (water_gathered > 0)
            rewards = rewards.at[i].add(
                jnp.where(gathered_any, self.c.gather_reward, 0.0)
            )
            # Cooperation bonus reward
            rewards = rewards.at[i].add(
                jnp.where(gathered_any & (n_coop > 1), self.c.cooperation_reward, 0.0)
            )

        # 3. Process sharing (action 5) - share resources with nearest agent
        is_share = (actions == 5) & state.agent_alive
        for i in range(na):
            sharing = is_share[i]
            if na > 1:
                # Find nearest other alive agent
                dists = jnp.sum((new_positions - new_positions[i][None, :]) ** 2, axis=1)
                dists = jnp.where(jnp.arange(na) == i, jnp.inf, dists)
                dists = jnp.where(state.agent_alive, dists, jnp.inf)
                nearest = jnp.argmin(dists)
                near_enough = dists[nearest] <= 2  # adjacent or same cell

                # Transfer 1 hunger unit
                can_share = sharing & near_enough & (new_hunger[i] > 3)
                new_hunger = new_hunger.at[i].add(jnp.where(can_share, -2, 0))
                new_hunger = new_hunger.at[nearest].add(jnp.where(can_share, 2, 0))

        # 4. Process attack (action 6)
        is_attack = (actions == 6) & state.agent_alive
        new_health = state.agent_health
        for i in range(na):
            attacking = is_attack[i]
            if na > 1:
                dists = jnp.sum((new_positions - new_positions[i][None, :]) ** 2, axis=1)
                dists = jnp.where(jnp.arange(na) == i, jnp.inf, dists)
                nearest = jnp.argmin(dists)
                near_enough = dists[nearest] <= 2
                can_attack = attacking & near_enough
                new_health = new_health.at[nearest].add(jnp.where(can_attack, -3, 0))

        # 5. Process signals (action 7) - update agent signals
        is_signal = (actions == 7) & state.agent_alive
        new_signals = jnp.where(
            is_signal[:, None],
            signal_tokens,
            state.agent_signals
        )
        # Communication cost
        rewards = rewards - jnp.where(is_signal, self.c.communication_cost, 0.0)

        # 6. Resource depletion (hunger/thirst per step)
        new_hunger = new_hunger - self.c.hunger_rate
        new_thirst = new_thirst - self.c.thirst_rate
        new_hunger = jnp.clip(new_hunger, 0, self.c.max_hunger)
        new_thirst = jnp.clip(new_thirst, 0, self.c.max_thirst)

        # Starvation/dehydration damage
        starving = (new_hunger <= 0) & state.agent_alive
        dehydrated = (new_thirst <= 0) & state.agent_alive
        new_health = new_health - jnp.where(starving, self.c.starvation_damage, 0)
        new_health = new_health - jnp.where(dehydrated, self.c.dehydration_damage, 0)

        # 7. Predator movement and damage
        pred_keys = random.split(keys[0], self.c.n_predators)
        new_pred_positions = state.predator_positions
        new_threat_grid = jnp.zeros((gs, gs))

        for p in range(self.c.n_predators):
            # Random walk toward nearest agent
            pred_pos = state.predator_positions[p]
            agent_dists = jnp.sum((new_positions - pred_pos[None, :]) ** 2, axis=1)
            agent_dists = jnp.where(state.agent_alive, agent_dists, jnp.inf)
            nearest_agent = jnp.argmin(agent_dists)

            # Move toward nearest agent with some randomness
            direction = jnp.sign(new_positions[nearest_agent] - pred_pos)
            random_dir = random.randint(pred_keys[p], (2,), -1, 2)
            move = jnp.where(random.uniform(pred_keys[p]) < 0.7, direction, random_dir)
            new_pred_pos = jnp.clip(pred_pos + move, 0, gs - 1)
            new_pred_positions = new_pred_positions.at[p].set(new_pred_pos)
            new_threat_grid = new_threat_grid.at[new_pred_pos[0], new_pred_pos[1]].set(1.0)

            # Damage agents on same cell
            for i in range(na):
                on_predator = jnp.all(new_positions[i] == new_pred_pos) & state.agent_alive[i]
                new_health = new_health.at[i].add(jnp.where(on_predator, -self.c.predator_damage, 0))

        # 8. Storm dynamics
        new_storm_active = state.storm_active
        new_storm_center = state.storm_center
        new_storm_timer = state.storm_timer

        # Spawn storm
        spawn_storm = (~state.storm_active) & (random.uniform(keys[1]) < self.c.storm_probability)
        new_storm_active = state.storm_active | spawn_storm
        new_storm_center = jnp.where(
            spawn_storm,
            random.randint(keys[2], (2,), 0, gs),
            state.storm_center
        )
        new_storm_timer = jnp.where(spawn_storm, self.c.storm_duration, state.storm_timer)

        # Storm damage (radius 3 around center)
        if self.c.long_horizons:  # storms only matter with long horizons
            for i in range(na):
                in_storm = (
                    new_storm_active &
                    (jnp.sum((new_positions[i] - new_storm_center) ** 2) <= 9) &
                    state.agent_alive[i]
                )
                new_health = new_health.at[i].add(jnp.where(in_storm, -self.c.storm_damage, 0))
            # Storm threat marker (JIT-safe: use distance mask instead of dynamic slice)
            row_idx = jnp.arange(gs)
            storm_mask = (
                (jnp.abs(row_idx[:, None] - new_storm_center[0]) <= 1) &
                (jnp.abs(row_idx[None, :] - new_storm_center[1]) <= 1)
            )
            new_threat_grid = jnp.where(
                new_storm_active & storm_mask,
                jnp.maximum(new_threat_grid, 0.5),
                new_threat_grid,
            )

        # Storm timer
        new_storm_timer = jnp.maximum(0, new_storm_timer - 1)
        new_storm_active = new_storm_active & (new_storm_timer > 0)

        # 9. Resource regeneration
        season = self._get_season(state.step)
        regen_mult = jnp.where(season == 3, self.c.winter_regen_multiplier, 1.0)  # winter = season 3
        regen_prob = self.c.resource_regen_rate * regen_mult

        food_regen = random.uniform(keys[3], (gs, gs)) < regen_prob
        water_regen = random.uniform(keys[4], (gs, gs)) < regen_prob
        food_grid = food_grid + jnp.where(food_regen & (food_grid < 5), 1.0, 0.0)
        water_grid = water_grid + jnp.where(water_regen & (water_grid < 5), 1.0, 0.0)

        # 10. Death and respawn
        new_health = jnp.clip(new_health, 0, self.c.max_health)
        new_alive = (new_health > 0) & state.agent_alive
        just_died = state.agent_alive & ~new_alive

        # Death penalty
        rewards = rewards + jnp.where(just_died, self.c.death_penalty, 0.0)

        # Respawn dead agents (evolutionary pressure)
        respawn_positions = random.randint(keys[5], (na, 2), 0, gs)
        new_positions = jnp.where(just_died[:, None], respawn_positions, new_positions)
        new_health = jnp.where(just_died, self.c.max_health, new_health)
        new_hunger = jnp.where(just_died, self.c.max_hunger // 2, new_hunger)
        new_thirst = jnp.where(just_died, self.c.max_thirst // 2, new_thirst)
        new_alive = jnp.ones(na, dtype=jnp.bool_)  # always respawn

        # Survival reward
        rewards = rewards + jnp.where(state.agent_alive, self.c.survival_reward, 0.0)

        # Delayed rewards: accumulate and release periodically
        if not self.c.delayed_rewards:
            pass  # immediate rewards already set
        # With delayed rewards, we could accumulate and batch-release,
        # but for simplicity we keep per-step rewards and let the
        # long gamma handle credit assignment

        new_step = state.step + 1

        # Build new state
        new_state = EnvState(
            food_grid=food_grid,
            water_grid=water_grid,
            material_grid=material_grid,
            threat_grid=new_threat_grid,
            agent_positions=new_positions,
            agent_health=new_health,
            agent_hunger=new_hunger,
            agent_thirst=new_thirst,
            agent_alive=new_alive,
            agent_signals=new_signals,
            predator_positions=new_pred_positions,
            storm_active=new_storm_active,
            storm_center=new_storm_center,
            storm_timer=new_storm_timer,
            step=new_step,
            rng=rng,
        )

        obs = self._get_observations(new_state)
        dones = (new_step >= self.c.max_steps) * jnp.ones(na, dtype=jnp.bool_)

        return new_state, obs, rewards, dones

    def _get_observations(self, state: EnvState) -> jnp.ndarray:
        """
        Get egocentric observations for all agents.

        Returns: (n_agents, obs_size, obs_size, n_channels)
        Where channels are: [self, other_agents, food, water, material, threat, signals_present]
        """
        na = self.c.n_agents
        gs = self.c.grid_size

        # Determine observation radius based on day/night
        if self.c.partial_observability:
            time_of_day = (state.step % self.c.day_length) / self.c.day_length
            is_night = (time_of_day > 0.5)  # second half of cycle
            obs_r = jnp.where(is_night, self.c.night_obs_radius, self.c.obs_radius)
        else:
            obs_r = self.c.grid_size  # full observability

        obs_size = 2 * self.c.obs_radius + 1  # always same tensor size, padded if needed
        n_ch = self.c.n_channels

        # Build global grid layers
        agent_grid = jnp.zeros((gs, gs))
        for i in range(na):
            pos = state.agent_positions[i]
            agent_grid = agent_grid.at[pos[0], pos[1]].add(
                jnp.where(state.agent_alive[i], 1.0, 0.0)
            )

        signal_grid = jnp.zeros((gs, gs))
        for i in range(na):
            pos = state.agent_positions[i]
            has_signal = jnp.any(state.agent_signals[i] > 0)
            signal_grid = signal_grid.at[pos[0], pos[1]].add(
                jnp.where(has_signal & state.agent_alive[i], 1.0, 0.0)
            )

        # Stack all channels: (gs, gs, n_ch)
        global_grid = jnp.stack([
            jnp.zeros((gs, gs)),  # self channel (filled per agent)
            agent_grid,
            state.food_grid / 5.0,  # normalize
            state.water_grid / 5.0,
            state.material_grid / 3.0,
            state.threat_grid,
            signal_grid,
        ], axis=-1)

        # Extract per-agent egocentric views
        all_obs = []
        # Pad global grid for border handling
        pad_size = self.c.obs_radius
        padded = jnp.pad(global_grid, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)))

        for i in range(na):
            pos = state.agent_positions[i] + pad_size  # offset for padding
            # Extract window
            obs = jax.lax.dynamic_slice(
                padded,
                (pos[0] - self.c.obs_radius, pos[1] - self.c.obs_radius, 0),
                (obs_size, obs_size, n_ch)
            )
            # Mark self position in channel 0
            center = self.c.obs_radius
            obs = obs.at[center, center, 0].set(1.0)
            # Remove self from other-agents channel
            obs = obs.at[center, center, 1].add(-1.0)

            # Apply partial observability mask (night reduces radius)
            if self.c.partial_observability:
                dx = jnp.arange(obs_size) - center
                dy = jnp.arange(obs_size) - center
                dist_grid = jnp.sqrt(dx[:, None]**2 + dy[None, :]**2)
                mask = (dist_grid <= obs_r).astype(jnp.float32)
                obs = obs * mask[:, :, None]

            all_obs.append(obs)

        # Also include vitals and received signals as auxiliary observation
        # Shape: (n_agents, obs_size, obs_size, n_ch)
        obs_grid = jnp.stack(all_obs)

        # Auxiliary info: vitals + signals from all agents
        # (n_agents, aux_dim)
        vitals = jnp.stack([
            state.agent_health / self.c.max_health,
            state.agent_hunger / self.c.max_hunger,
            state.agent_thirst / self.c.max_thirst,
        ], axis=-1)  # (n_agents, 3)

        # Time features
        time_of_day = (state.step % self.c.day_length) / self.c.day_length
        season = self._get_season(state.step) / 4.0
        time_features = jnp.broadcast_to(
            jnp.array([time_of_day, season]),
            (na, 2)
        )

        # All signals from all agents (received messages)
        # (n_agents, n_agents * n_signal_tokens)
        all_signals = state.agent_signals.reshape(1, -1).repeat(na, axis=0) / self.c.vocab_size

        return {
            'grid': obs_grid,
            'vitals': vitals,
            'time': time_features,
            'signals': all_signals,
        }

    def _get_season(self, step: jnp.ndarray) -> jnp.ndarray:
        """Get season index (0=spring, 1=summer, 2=autumn, 3=winter)."""
        if not self.c.long_horizons:
            return jnp.array(1)  # always summer if no long horizons
        return (step // self.c.season_length) % 4

    def get_observation_shape(self) -> Dict[str, Tuple]:
        """Return shapes of observation components."""
        obs_size = 2 * self.c.obs_radius + 1
        return {
            'grid': (obs_size, obs_size, self.c.n_channels),
            'vitals': (3,),
            'time': (2,),
            'signals': (self.c.n_agents * self.c.n_signal_tokens,),
        }

    def render_scene(self, state: EnvState, agent_idx: int) -> Dict:
        """
        Render a scene description for VLM translation.
        Returns structured dict suitable for VLM query.
        """
        pos = state.agent_positions[agent_idx]
        health = state.agent_health[agent_idx]
        hunger = state.agent_hunger[agent_idx]
        thirst = state.agent_thirst[agent_idx]

        # Find nearby entities
        nearby_agents = []
        for i in range(self.c.n_agents):
            if i != agent_idx:
                other_pos = state.agent_positions[i]
                dist = jnp.sqrt(jnp.sum((pos - other_pos) ** 2))
                if dist <= self.c.obs_radius:
                    nearby_agents.append({
                        'id': i,
                        'distance': float(dist),
                        'signal': state.agent_signals[i].tolist(),
                    })

        nearby_threats = []
        for p in range(self.c.n_predators):
            pred_pos = state.predator_positions[p]
            dist = jnp.sqrt(jnp.sum((pos - pred_pos) ** 2))
            if dist <= self.c.obs_radius:
                nearby_threats.append({
                    'type': 'predator',
                    'distance': float(dist),
                })

        food_here = float(state.food_grid[pos[0], pos[1]])
        water_here = float(state.water_grid[pos[0], pos[1]])

        return {
            'agent_id': agent_idx,
            'position': pos.tolist(),
            'health': float(health),
            'hunger': float(hunger),
            'thirst': float(thirst),
            'health_pct': float(health / self.c.max_health),
            'hunger_pct': float(hunger / self.c.max_hunger),
            'thirst_pct': float(thirst / self.c.max_thirst),
            'food_available': food_here,
            'water_available': water_here,
            'nearby_agents': nearby_agents,
            'nearby_threats': nearby_threats,
            'storm_active': bool(state.storm_active),
            'storm_nearby': bool(
                state.storm_active and
                jnp.sum((pos - state.storm_center) ** 2) <= 16
            ),
            'time_of_day': float((state.step % self.c.day_length) / self.c.day_length),
            'season': int(self._get_season(state.step)),
            'step': int(state.step),
            'own_signal': state.agent_signals[agent_idx].tolist(),
        }


# ============================================================================
# Ablation configs
# ============================================================================

def make_ablation_configs() -> Dict[str, EnvConfig]:
    """Generate configs for each forcing function ablation."""
    configs = {}

    # Full model
    configs['full'] = EnvConfig()

    # FF1: Remove partial observability
    configs['no_partial_obs'] = EnvConfig(partial_observability=False)

    # FF2: Remove long horizons (short episodes, no seasons)
    configs['no_long_horizon'] = EnvConfig(
        long_horizons=False,
        max_steps=200,
        season_length=1000,  # effectively no seasons
    )

    # FF3: Remove world model
    configs['no_world_model'] = EnvConfig(use_world_model=False)

    # FF4: Remove self-prediction
    configs['no_self_prediction'] = EnvConfig(use_self_prediction=False)

    # FF5: Remove intrinsic motivation
    configs['no_intrinsic_motivation'] = EnvConfig(use_intrinsic_motivation=False)

    # FF6: Remove delayed rewards (immediate only)
    configs['no_delayed_rewards'] = EnvConfig(delayed_rewards=False)

    return configs


if __name__ == '__main__':
    # Quick sanity check
    config = EnvConfig(n_agents=4, grid_size=8)
    env = SurvivalGridWorld(config)

    rng = random.PRNGKey(42)
    state, obs = env.reset(rng)

    print("Environment initialized:")
    print(f"  Grid: {config.grid_size}x{config.grid_size}")
    print(f"  Agents: {config.n_agents}")
    print(f"  Obs grid shape: {obs['grid'].shape}")
    print(f"  Vitals shape: {obs['vitals'].shape}")
    print(f"  Time shape: {obs['time'].shape}")
    print(f"  Signals shape: {obs['signals'].shape}")

    # Run a few random steps
    for t in range(10):
        actions = random.randint(random.PRNGKey(t), (config.n_agents,), 0, config.n_actions)
        signals = random.randint(random.PRNGKey(t + 100), (config.n_agents, config.n_signal_tokens), 0, config.vocab_size)
        state, obs, rewards, dones = env.step(state, actions, signals)
        alive = state.agent_alive
        health = state.agent_health
        print(f"  Step {t+1}: rewards={rewards}, health={health}, alive={alive}")

    # Render scene for VLM
    scene = env.render_scene(state, 0)
    print(f"\nScene for agent 0: {scene}")
