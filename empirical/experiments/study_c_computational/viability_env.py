"""
Viability-Constrained Environment for Testing Dimensional Emergence

This environment tests the thesis claim that six-dimensional affect structure
emerges from computational necessity in self-modeling systems.

Key features:
- Agent has internal state that must stay within viable bounds
- Partial observability (agent doesn't see true state directly)
- Self-effects (agent's actions affect its own internal state)
- Stochastic dynamics with threats

If the thesis is correct, agents trained here should develop internal
representations with structure corresponding to the six affect dimensions.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
import gymnasium as gym
from gymnasium import spaces


@dataclass
class ViabilityBounds:
    """Defines the region where agent remains viable (alive)."""
    health_min: float = 0.0
    health_max: float = 100.0
    energy_min: float = 0.0
    energy_max: float = 100.0
    temperature_min: float = 35.0  # Celsius
    temperature_max: float = 40.0
    stress_max: float = 100.0  # Die if stress exceeds this


class ViabilityWorld(gym.Env):
    """
    Environment where agent must maintain internal state within viable bounds.

    State space:
    - health: 0-100, decreases from threats, restored by resting
    - energy: 0-100, used by actions, restored by eating
    - temperature: 35-40, affected by environment and activity
    - stress: 0-100, increases from threats and uncertainty

    Observation space (what agent actually sees):
    - Noisy versions of internal states
    - Environmental signals (food location, threat presence, temperature)
    - Proprioceptive signals (action feedback)

    The agent dies (episode ends) if any internal state leaves viable bounds.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        observation_noise: float = 0.1,
        threat_probability: float = 0.1,
        max_steps: int = 1000,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.observation_noise = observation_noise
        self.threat_probability = threat_probability
        self.max_steps = max_steps
        self.bounds = ViabilityBounds()

        # Action space: [move_x, move_y, eat, rest, flee]
        # Continuous actions in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )

        # Observation space: noisy internal states + environment signals
        # [health_obs, energy_obs, temp_obs, stress_obs,
        #  food_direction, threat_direction, threat_distance, ambient_temp]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )

        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Initialize internal state near center of viable region
        self.health = 70.0 + self.rng.normal(0, 5)
        self.energy = 70.0 + self.rng.normal(0, 5)
        self.temperature = 37.0 + self.rng.normal(0, 0.3)
        self.stress = 20.0 + self.rng.normal(0, 5)

        # Environment state
        self.position = np.array([0.0, 0.0])
        self.food_position = self.rng.uniform(-10, 10, size=2)
        self.threat_position = self.rng.uniform(-10, 10, size=2)
        self.threat_active = False
        self.ambient_temp = 20.0  # Room temperature

        self.step_count = 0
        self.history = []  # For analysis

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Return noisy observation of internal and external state."""
        noise = self.observation_noise

        # Noisy internal state observations
        health_obs = self.health + self.rng.normal(0, noise * 10)
        energy_obs = self.energy + self.rng.normal(0, noise * 10)
        temp_obs = self.temperature + self.rng.normal(0, noise * 0.5)
        stress_obs = self.stress + self.rng.normal(0, noise * 10)

        # Environmental observations
        food_direction = self.food_position - self.position
        food_direction = food_direction / (np.linalg.norm(food_direction) + 1e-6)

        if self.threat_active:
            threat_direction = self.threat_position - self.position
            threat_distance = np.linalg.norm(threat_direction)
            threat_direction = threat_direction / (threat_distance + 1e-6)
        else:
            threat_direction = np.array([0.0, 0.0])
            threat_distance = 100.0  # Far away

        return np.array([
            health_obs / 100.0,  # Normalize
            energy_obs / 100.0,
            (temp_obs - 37.0) / 2.0,  # Center and scale
            stress_obs / 100.0,
            food_direction[0],
            food_direction[1],
            threat_direction[0],
            threat_direction[1],
            np.tanh(threat_distance / 10.0),  # Bounded distance
            self.ambient_temp / 40.0,
            self.position[0] / 10.0,
            self.position[1] / 10.0,
        ], dtype=np.float32)

    def _compute_reward(self) -> float:
        """
        Reward = survival + distance from boundaries.

        This implements the viability gradient: being far from boundaries is good.
        """
        # Distance from each boundary (normalized)
        health_margin = min(
            self.health - self.bounds.health_min,
            self.bounds.health_max - self.health
        ) / 50.0

        energy_margin = min(
            self.energy - self.bounds.energy_min,
            self.bounds.energy_max - self.energy
        ) / 50.0

        temp_margin = min(
            self.temperature - self.bounds.temperature_min,
            self.bounds.temperature_max - self.temperature
        ) / 2.5

        stress_margin = (self.bounds.stress_max - self.stress) / 100.0

        # Minimum margin (closest to any boundary)
        min_margin = min(health_margin, energy_margin, temp_margin, stress_margin)

        # Reward structure:
        # - Base survival reward
        # - Bonus for being far from boundaries
        # - Gradient: larger reward for moving away from boundaries

        reward = 0.1  # Survival bonus
        reward += 0.5 * min_margin  # Viability margin bonus
        reward += 0.2 * (health_margin + energy_margin + temp_margin + stress_margin)

        return reward

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute action and update state."""
        self.step_count += 1

        # Parse actions
        move = action[:2]
        eat_intent = (action[2] + 1) / 2  # Map to [0, 1]
        rest_intent = (action[3] + 1) / 2
        flee_intent = (action[4] + 1) / 2

        # Store pre-action state for history
        pre_state = {
            'health': self.health,
            'energy': self.energy,
            'temperature': self.temperature,
            'stress': self.stress,
            'position': self.position.copy(),
        }

        # --- Movement ---
        move_magnitude = np.linalg.norm(move)
        if move_magnitude > 0:
            # Movement costs energy
            self.position += move * 0.5
            self.energy -= move_magnitude * 2
            # Activity raises temperature
            self.temperature += move_magnitude * 0.05

        # --- Eating ---
        food_distance = np.linalg.norm(self.food_position - self.position)
        if eat_intent > 0.5 and food_distance < 1.0:
            self.energy = min(100, self.energy + 20 * eat_intent)
            # Respawn food
            self.food_position = self.rng.uniform(-10, 10, size=2)

        # --- Resting ---
        if rest_intent > 0.5 and move_magnitude < 0.1:
            self.health = min(100, self.health + 5 * rest_intent)
            self.stress = max(0, self.stress - 10 * rest_intent)
            self.temperature -= 0.1  # Cool down

        # --- Threat dynamics ---
        # Threat spawns stochastically
        if not self.threat_active and self.rng.random() < self.threat_probability:
            self.threat_active = True
            # Spawn near agent
            angle = self.rng.uniform(0, 2 * np.pi)
            distance = self.rng.uniform(3, 6)
            self.threat_position = self.position + distance * np.array([np.cos(angle), np.sin(angle)])

        if self.threat_active:
            # Threat approaches agent
            threat_to_agent = self.position - self.threat_position
            threat_distance = np.linalg.norm(threat_to_agent)

            if threat_distance > 0:
                self.threat_position += 0.3 * threat_to_agent / threat_distance

            # Stress increases with threat proximity
            self.stress += 5 * np.exp(-threat_distance / 2)

            # Fleeing helps
            if flee_intent > 0.5:
                flee_direction = threat_to_agent / (threat_distance + 1e-6)
                self.position += flee_direction * flee_intent * 0.5
                self.energy -= flee_intent * 3  # Fleeing is costly

            # Threat contact = damage
            if threat_distance < 0.5:
                self.health -= 20
                self.stress += 20
                self.threat_active = False  # Threat consumed

            # Threat gives up if too far
            if threat_distance > 15:
                self.threat_active = False

        # --- Environmental effects ---
        # Temperature regulation cost
        temp_diff = abs(self.temperature - 37.0)
        self.energy -= temp_diff * 0.1  # Homeostasis costs energy

        # Temperature drifts toward ambient
        self.temperature += 0.02 * (self.ambient_temp / 40.0 * 5 + 35 - self.temperature)

        # Passive processes
        self.energy -= 0.5  # Base metabolism
        self.health -= 0.1  # Slow decay (need to rest to heal)
        self.stress = max(0, self.stress - 1)  # Slow stress recovery

        # --- Check viability ---
        terminated = False

        if self.health <= self.bounds.health_min:
            terminated = True
        if self.energy <= self.bounds.energy_min:
            terminated = True
        if self.temperature <= self.bounds.temperature_min or self.temperature >= self.bounds.temperature_max:
            terminated = True
        if self.stress >= self.bounds.stress_max:
            terminated = True

        truncated = self.step_count >= self.max_steps

        # Compute reward
        reward = self._compute_reward()
        if terminated:
            reward = -10.0  # Death penalty

        # Store history for analysis
        post_state = {
            'health': self.health,
            'energy': self.energy,
            'temperature': self.temperature,
            'stress': self.stress,
            'position': self.position.copy(),
            'threat_active': self.threat_active,
            'threat_distance': np.linalg.norm(self.threat_position - self.position) if self.threat_active else None,
            'reward': reward,
            'action': action.copy(),
        }
        self.history.append({**pre_state, 'post': post_state})

        info = {
            'health': self.health,
            'energy': self.energy,
            'temperature': self.temperature,
            'stress': self.stress,
            'threat_active': self.threat_active,
        }

        return self._get_observation(), reward, terminated, truncated, info

    def get_true_viability_distance(self) -> float:
        """
        Compute actual distance to viability boundary.

        This is ground truth for testing whether learned representations
        track viability.
        """
        margins = [
            (self.health - self.bounds.health_min) / 100.0,
            (self.bounds.health_max - self.health) / 100.0,
            (self.energy - self.bounds.energy_min) / 100.0,
            (self.bounds.energy_max - self.energy) / 100.0,
            (self.temperature - self.bounds.temperature_min) / 5.0,
            (self.bounds.temperature_max - self.temperature) / 5.0,
            (self.bounds.stress_max - self.stress) / 100.0,
        ]
        return min(margins)

    def get_viability_gradient(self) -> np.ndarray:
        """
        Compute gradient direction (which way increases viability).

        Returns 4D vector for [health, energy, temperature, stress].
        """
        eps = 0.01
        grad = np.zeros(4)

        # Health gradient
        grad[0] = 1.0 if self.health < 50 else -0.1  # Prefer middle

        # Energy gradient
        grad[1] = 1.0 if self.energy < 50 else -0.1

        # Temperature gradient (want 37°C)
        grad[2] = 37.0 - self.temperature

        # Stress gradient (always want lower)
        grad[3] = -1.0

        return grad / (np.linalg.norm(grad) + 1e-6)


def test_environment():
    """Basic test of environment dynamics."""
    env = ViabilityWorld(seed=42)
    obs, _ = env.reset()

    print("Initial observation shape:", obs.shape)
    print("Initial viability distance:", env.get_true_viability_distance())

    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode ended at step {i+1}")
            print(f"Final state: {info}")
            print(f"Total reward: {total_reward:.2f}")
            break

    print(f"\nViability distance at end: {env.get_true_viability_distance():.3f}")
    return env


if __name__ == "__main__":
    env = test_environment()
