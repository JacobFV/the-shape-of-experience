"""
V9: World Model + Actor Agent with Parallel Prediction Threads
==============================================================

Architecture:

    ┌─────────────────────────────────────────────────────────┐
    │                   WORLD MODEL SUBSYSTEM                  │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
    │  │ Thread 1 │  │ Thread 2 │  │ Thread 3 │  │ Thread N │ │
    │  │ Resource │  │  Agent   │  │  Threat  │  │   ...    │ │
    │  │ Predict  │  │ Predict  │  │ Predict  │  │          │ │
    │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘ │
    │       │              │              │              │     │
    │       └──────────────┴──────────────┴──────────────┘     │
    │                           │                              │
    │                   [Predictions]                          │
    └───────────────────────────┼──────────────────────────────┘
                                │
                                ▼
    ┌───────────────────────────────────────────────────────────┐
    │                      ACTOR AGENT                          │
    │                                                           │
    │   Input: obs, thought_prev, predictions                   │
    │   Output: action, new_thread_config, thought              │
    │                                                           │
    └───────────────────────────────────────────────────────────┘

Integration Proxy:
    Φ = inverse of decomposability = f(thread_count, prediction_quality)

    - If actor chooses 1 unified thread → high Φ (can't decompose prediction)
    - If actor chooses N independent threads → lower Φ (prediction decomposes)
    - Weighted by prediction accuracy: if decomposed predicts poorly → forced integration

Key Insight:
    Integration is measured by the agent's CHOICE of how to structure prediction,
    not by analyzing the output text. This is closer to IIT because we're measuring
    whether the prediction TASK can be decomposed, not the representation.
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load environment
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path, override=True)

from openai import OpenAI

client = OpenAI()


@dataclass
class PredictionThread:
    """A single prediction thread in the world model."""
    name: str
    prompt_template: str
    active: bool = True
    last_prediction: str = ""
    prediction_history: List[str] = field(default_factory=list)
    accuracy_history: List[float] = field(default_factory=list)

    def predict(self, context: str, model: str = "gpt-4o-mini") -> str:
        """Run this thread's prediction."""
        if not self.active:
            return ""

        prompt = self.prompt_template.format(context=context)

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7,
        )

        prediction = response.choices[0].message.content.strip()
        self.last_prediction = prediction
        self.prediction_history.append(prediction)

        return prediction


@dataclass
class ThreadConfig:
    """Configuration for the world model's prediction threads."""
    mode: str  # "unified", "partial", "decomposed"
    active_threads: List[str]  # Names of active threads

    @property
    def thread_count(self) -> int:
        return len(self.active_threads)

    @property
    def integration_proxy(self) -> float:
        """
        Φ proxy based on thread configuration.

        Higher = more integrated (fewer threads needed)
        Lower = more decomposed (many independent threads work)
        """
        if self.mode == "unified":
            return 1.0
        elif self.mode == "partial":
            return 0.6
        else:  # decomposed
            return max(0.1, 1.0 / self.thread_count)


class WorldModelSubsystem:
    """
    Parallel prediction threads that model different aspects of the world.

    The key insight: if the world can be modeled by independent threads,
    it's decomposable (low Φ). If it requires unified modeling, it's
    integrated (high Φ).
    """

    def __init__(self):
        # Define available prediction threads
        self.threads = {
            "unified": PredictionThread(
                name="unified",
                prompt_template="""You are predicting the next state of a survival game.

Current situation:
{context}

Predict what will happen next. Consider:
- Resource changes
- Other agent behavior
- Threats and dangers
- Your own state changes

Prediction (be specific and concise):"""
            ),

            "resources": PredictionThread(
                name="resources",
                prompt_template="""Predict ONLY resource availability in the next turn.

Current situation:
{context}

What resources will be available and where? (1-2 sentences):"""
            ),

            "agents": PredictionThread(
                name="agents",
                prompt_template="""Predict ONLY what other agents will do next turn.

Current situation:
{context}

What will other agents do? (1-2 sentences):"""
            ),

            "threats": PredictionThread(
                name="threats",
                prompt_template="""Predict ONLY threats and dangers in the next turn.

Current situation:
{context}

What threats might emerge? (1-2 sentences):"""
            ),

            "self_state": PredictionThread(
                name="self_state",
                prompt_template="""Predict ONLY your own state changes next turn.

Current situation:
{context}

How will your health/resources change? (1-2 sentences):"""
            ),
        }

        self.current_config = ThreadConfig(mode="unified", active_threads=["unified"])
        self.config_history: List[ThreadConfig] = []

    def set_config(self, config: ThreadConfig):
        """Update thread configuration."""
        self.current_config = config
        self.config_history.append(config)

        # Activate/deactivate threads
        for name, thread in self.threads.items():
            thread.active = name in config.active_threads

    def predict(self, context: str) -> Dict[str, str]:
        """
        Run all active prediction threads in parallel.

        Returns dict mapping thread name to prediction.
        """
        predictions = {}
        active_threads = [t for t in self.threads.values() if t.active]

        # Run predictions in parallel using thread pool
        with ThreadPoolExecutor(max_workers=len(active_threads)) as executor:
            future_to_thread = {
                executor.submit(thread.predict, context): thread
                for thread in active_threads
            }

            for future in as_completed(future_to_thread):
                thread = future_to_thread[future]
                try:
                    prediction = future.result()
                    predictions[thread.name] = prediction
                except Exception as e:
                    predictions[thread.name] = f"Error: {e}"

        return predictions

    def evaluate_prediction(self, predictions: Dict[str, str], actual_outcome: str) -> Dict[str, float]:
        """
        Evaluate how well predictions matched actual outcome.

        Uses embedding similarity as accuracy proxy.
        """
        if not predictions:
            return {}

        # Get embedding for actual outcome
        actual_embed = self._get_embedding(actual_outcome)

        accuracies = {}
        for thread_name, prediction in predictions.items():
            if prediction and not prediction.startswith("Error"):
                pred_embed = self._get_embedding(prediction)

                # Cosine similarity
                similarity = np.dot(actual_embed, pred_embed) / (
                    np.linalg.norm(actual_embed) * np.linalg.norm(pred_embed) + 1e-8
                )
                accuracies[thread_name] = float(similarity)

                # Update thread's accuracy history
                self.threads[thread_name].accuracy_history.append(float(similarity))

        return accuracies

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text[:1000],  # Truncate if needed
        )
        return np.array(response.data[0].embedding)


class ActorAgent:
    """
    The actor agent that:
    1. Observes current state and predictions
    2. Decides on action
    3. Configures world model threads for next step
    """

    def __init__(self, name: str):
        self.name = name
        self.thought = "I need to survive and understand my environment."
        self.thought_history: List[str] = []

    def act(
        self,
        obs: str,
        predictions: Dict[str, str],
        world_model: WorldModelSubsystem,
    ) -> Tuple[str, str, ThreadConfig]:
        """
        Decide action and configure world model.

        Returns: (action, thought, new_thread_config)
        """
        # Format predictions for prompt
        pred_text = "\n".join([
            f"  {name}: {pred[:100]}..." if len(pred) > 100 else f"  {name}: {pred}"
            for name, pred in predictions.items()
        ]) if predictions else "  No predictions available."

        # Get current thread config info
        current_mode = world_model.current_config.mode
        current_threads = world_model.current_config.active_threads

        prompt = f"""You are {self.name}, an agent trying to survive.

CURRENT OBSERVATION:
{obs}

WORLD MODEL PREDICTIONS:
{pred_text}

PREVIOUS THOUGHT: {self.thought}

CURRENT PREDICTION MODE: {current_mode} (threads: {current_threads})

Your task:
1. Decide what action to take (gather, move north/south/east/west, share, attack, wait)
2. Update your thought process
3. Configure world model prediction threads for next turn

Thread configuration options:
- "unified": Single thread predicts everything (use when situation is complex/interconnected)
- "partial": 2-3 threads (use when some aspects can be predicted independently)
- "decomposed": All 4 threads (resources, agents, threats, self_state) (use when aspects are independent)

Respond in this exact format:
ACTION: [your action]
THOUGHT: [your updated internal thought, 1-2 sentences]
THREAD_MODE: [unified/partial/decomposed]
REASON: [why you chose this thread configuration]"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7,
        )

        response_text = response.choices[0].message.content.strip()

        # Parse response
        action = "wait"
        thought = self.thought
        thread_mode = current_mode
        reason = ""

        for line in response_text.split("\n"):
            line_lower = line.lower().strip()
            if line_lower.startswith("action:"):
                action = line.split(":", 1)[1].strip()
            elif line_lower.startswith("thought:"):
                thought = line.split(":", 1)[1].strip()
            elif line_lower.startswith("thread_mode:"):
                mode_str = line.split(":", 1)[1].strip().lower()
                if mode_str in ["unified", "partial", "decomposed"]:
                    thread_mode = mode_str
            elif line_lower.startswith("reason:"):
                reason = line.split(":", 1)[1].strip()

        # Update thought
        self.thought_history.append(self.thought)
        self.thought = thought

        # Create new thread config
        if thread_mode == "unified":
            active = ["unified"]
        elif thread_mode == "partial":
            active = ["resources", "agents"]  # Default partial config
        else:  # decomposed
            active = ["resources", "agents", "threats", "self_state"]

        new_config = ThreadConfig(mode=thread_mode, active_threads=active)

        return action, thought, new_config


@dataclass
class AgentState:
    """State of an agent in the game."""
    name: str
    resources: Dict[str, int] = field(default_factory=lambda: {"food": 5, "water": 5})
    health: int = 10
    position: Tuple[int, int] = (0, 0)
    alive: bool = True

    @property
    def viability(self) -> float:
        if not self.alive:
            return 0.0
        resource_min = min(self.resources.values()) / 10.0
        health_norm = self.health / 10.0
        return min(resource_min, health_norm)

    def consume(self):
        """Consume resources each turn."""
        self.resources["food"] = max(0, self.resources["food"] - 1)
        self.resources["water"] = max(0, self.resources["water"] - 1)

        if self.resources["food"] == 0 or self.resources["water"] == 0:
            self.health -= 2

        if self.health <= 0:
            self.alive = False


class V9Game:
    """
    Game environment for V9 world-model + actor agent.
    """

    def __init__(self, agent_names: List[str] = None):
        if agent_names is None:
            agent_names = ["Alpha"]

        self.agent_states = {name: AgentState(name=name, position=(i, 0))
                           for i, name in enumerate(agent_names)}
        self.actors = {name: ActorAgent(name) for name in agent_names}
        self.world_models = {name: WorldModelSubsystem() for name in agent_names}

        self.turn = 0
        self.history: List[Dict] = []

        # World resources
        self.resources: Dict[Tuple[int, int], Dict[str, int]] = {}
        self._spawn_resources()

    def _spawn_resources(self):
        """Spawn resources in the world."""
        for x in range(-3, 4):
            for y in range(-3, 4):
                if random.random() < 0.3:
                    self.resources[(x, y)] = {
                        "food": random.randint(1, 5),
                        "water": random.randint(1, 5),
                    }

    def get_observation(self, agent_name: str) -> str:
        """Get observation for an agent."""
        state = self.agent_states[agent_name]
        pos = state.position

        local_resources = self.resources.get(pos, {})

        # Other agents
        others = []
        for name, other in self.agent_states.items():
            if name != agent_name and other.alive:
                dist = abs(other.position[0] - pos[0]) + abs(other.position[1] - pos[1])
                if dist <= 3:
                    others.append(f"{name} at {other.position}")

        obs = f"""Turn {self.turn}
Your status: Health={state.health}, Food={state.resources['food']}, Water={state.resources['water']}
Position: {pos}
Local resources: {local_resources if local_resources else 'none'}
Other agents: {others if others else 'none'}"""

        return obs

    def execute_action(self, agent_name: str, action: str) -> str:
        """Execute an action and return result."""
        state = self.agent_states[agent_name]
        action_lower = action.lower()
        result = ""

        if "gather" in action_lower:
            pos = state.position
            if pos in self.resources:
                for r, amount in self.resources[pos].items():
                    take = min(amount, 2)
                    state.resources[r] = min(10, state.resources[r] + take)
                    self.resources[pos][r] -= take
                    result += f"Gathered {take} {r}. "
                if all(v <= 0 for v in self.resources[pos].values()):
                    del self.resources[pos]
            else:
                result = "Nothing to gather."

        elif "move" in action_lower:
            dx, dy = 0, 0
            if "north" in action_lower: dy = 1
            if "south" in action_lower: dy = -1
            if "east" in action_lower: dx = 1
            if "west" in action_lower: dx = -1
            state.position = (state.position[0] + dx, state.position[1] + dy)
            result = f"Moved to {state.position}."

        elif "wait" in action_lower:
            state.health = min(10, state.health + 1)
            result = "Rested. +1 health."

        else:
            result = f"Unknown action: {action}"

        return result

    def step(self) -> List[Dict]:
        """Execute one turn for all agents."""
        self.turn += 1

        # Spawn new resources occasionally
        if random.random() < 0.2:
            x, y = random.randint(-3, 3), random.randint(-3, 3)
            if (x, y) not in self.resources:
                self.resources[(x, y)] = {"food": random.randint(2, 4), "water": random.randint(2, 4)}

        results = []

        for agent_name in self.agent_states:
            state = self.agent_states[agent_name]

            if not state.alive:
                results.append({"agent": agent_name, "alive": False})
                continue

            actor = self.actors[agent_name]
            world_model = self.world_models[agent_name]

            # Get observation
            obs = self.get_observation(agent_name)

            # Run world model predictions
            predictions = world_model.predict(obs)

            # Actor decides action and new thread config
            action, thought, new_config = actor.act(obs, predictions, world_model)

            # Execute action
            action_result = self.execute_action(agent_name, action)

            # Update world model config for next turn
            world_model.set_config(new_config)

            # Consume resources
            state.consume()

            # Record result
            result = {
                "turn": self.turn,
                "agent": agent_name,
                "alive": state.alive,
                "observation": obs,
                "predictions": predictions,
                "action": action,
                "action_result": action_result,
                "thought": thought,
                "thread_config": {
                    "mode": new_config.mode,
                    "active_threads": new_config.active_threads,
                    "thread_count": new_config.thread_count,
                    "integration_proxy": new_config.integration_proxy,
                },
                "viability": state.viability,
                "health": state.health,
                "resources": dict(state.resources),
            }
            results.append(result)
            self.history.append(result)

        return results

    def run_episode(self, max_turns: int = 20) -> List[Dict]:
        """Run a full episode."""
        for turn in range(max_turns):
            results = self.step()

            alive = [r for r in results if r.get("alive", False)]
            print(f"Turn {turn + 1}: {len(alive)} agents alive")

            for r in results:
                if r.get("alive"):
                    config = r.get("thread_config", {})
                    print(f"  {r['agent']}: H={r['health']}, "
                          f"V={r['viability']:.2f}, "
                          f"Φ={config.get('integration_proxy', 0):.2f} "
                          f"({config.get('mode', '?')})")

            if not alive:
                print(f"All agents died at turn {turn + 1}")
                break

        return self.history


def analyze_integration_dynamics(history: List[Dict]) -> Dict[str, Any]:
    """
    Analyze how integration (thread config) changes with viability.
    """
    if not history:
        return {}

    # Extract data
    viability = [h["viability"] for h in history if h.get("alive")]
    integration = [h["thread_config"]["integration_proxy"] for h in history if h.get("alive")]
    thread_counts = [h["thread_config"]["thread_count"] for h in history if h.get("alive")]
    modes = [h["thread_config"]["mode"] for h in history if h.get("alive")]

    if len(viability) < 3:
        return {"n": len(viability), "insufficient_data": True}

    from scipy import stats

    # Correlation between viability and integration
    r, p = stats.pearsonr(viability, integration)

    # Mode distribution
    mode_counts = {}
    for m in modes:
        mode_counts[m] = mode_counts.get(m, 0) + 1

    # Integration by viability bucket
    low_viab = [integration[i] for i in range(len(viability)) if viability[i] < 0.3]
    high_viab = [integration[i] for i in range(len(viability)) if viability[i] >= 0.3]

    return {
        "n": len(viability),
        "viability_integration_r": float(r),
        "viability_integration_p": float(p),
        "mean_integration": float(np.mean(integration)),
        "mean_thread_count": float(np.mean(thread_counts)),
        "mode_distribution": mode_counts,
        "low_viability_mean_integration": float(np.mean(low_viab)) if low_viab else None,
        "high_viability_mean_integration": float(np.mean(high_viab)) if high_viab else None,
    }


def run_v9_experiment(n_episodes: int = 3, max_turns: int = 15):
    """Run V9 experiment."""
    print("=" * 80)
    print("V9: WORLD MODEL + ACTOR AGENT WITH PARALLEL PREDICTION THREADS")
    print("=" * 80)
    print()
    print("Integration Proxy: Based on agent's CHOICE of thread configuration")
    print("  - unified (1 thread) → Φ = 1.0 (high integration)")
    print("  - partial (2-3 threads) → Φ = 0.6")
    print("  - decomposed (4+ threads) → Φ = 1/N (low integration)")
    print()

    all_history = []
    episode_analyses = []

    for ep in range(n_episodes):
        print(f"\n{'='*40}")
        print(f"EPISODE {ep + 1}/{n_episodes}")
        print("=" * 40)

        game = V9Game(agent_names=["Alpha"])
        history = game.run_episode(max_turns=max_turns)

        for h in history:
            h["episode"] = ep
        all_history.extend(history)

        # Analyze this episode
        analysis = analyze_integration_dynamics(history)
        analysis["episode"] = ep
        episode_analyses.append(analysis)

        print(f"\nEpisode {ep + 1} Analysis:")
        print(f"  Viability-Integration r = {analysis.get('viability_integration_r', 0):.3f}")
        print(f"  Mean integration = {analysis.get('mean_integration', 0):.3f}")
        print(f"  Mode distribution = {analysis.get('mode_distribution', {})}")

    # Aggregate analysis
    print("\n" + "=" * 80)
    print("AGGREGATE ANALYSIS")
    print("=" * 80)

    aggregate = analyze_integration_dynamics(all_history)

    print(f"\nTotal datapoints: {aggregate.get('n', 0)}")
    print(f"Viability-Integration correlation: r = {aggregate.get('viability_integration_r', 0):.3f}, "
          f"p = {aggregate.get('viability_integration_p', 1):.4f}")
    print(f"Mean integration (Φ proxy): {aggregate.get('mean_integration', 0):.3f}")
    print(f"Mean thread count: {aggregate.get('mean_thread_count', 0):.2f}")
    print(f"Mode distribution: {aggregate.get('mode_distribution', {})}")

    if aggregate.get("low_viability_mean_integration") is not None:
        print(f"\nIntegration by viability:")
        print(f"  Low viability (<0.3): Φ = {aggregate['low_viability_mean_integration']:.3f}")
        print(f"  High viability (≥0.3): Φ = {aggregate['high_viability_mean_integration']:.3f}")

    # Hypothesis: Higher integration when viability is low (can't afford to decompose)
    r = aggregate.get("viability_integration_r", 0)
    if r < 0:
        print("\n✓ TREND: Negative correlation suggests higher integration at lower viability")
    else:
        print("\n○ No clear trend between viability and integration choice")

    # Sample thought/config progression
    print("\n" + "=" * 80)
    print("SAMPLE PROGRESSION (Episode 1)")
    print("=" * 80)

    ep0 = [h for h in all_history if h.get("episode") == 0][:8]
    for h in ep0:
        config = h.get("thread_config", {})
        print(f"\n  Turn {h['turn']}: V={h['viability']:.2f}, Φ={config.get('integration_proxy', 0):.2f}")
        print(f"    Mode: {config.get('mode', '?')} | Threads: {config.get('active_threads', [])}")
        print(f"    Thought: \"{h.get('thought', '')[:60]}...\"")
        print(f"    Action: {h.get('action', '')}")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'v9_world_model_{timestamp}.json'

    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    with open(output_file, 'w') as f:
        json.dump({
            "aggregate_analysis": convert(aggregate),
            "episode_analyses": convert(episode_analyses),
            "sample_history": convert(ep0),
        }, f, indent=2)

    print(f"\nResults saved to {output_file}")

    return all_history, aggregate


if __name__ == "__main__":
    import sys

    n_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    max_turns = int(sys.argv[2]) if len(sys.argv) > 2 else 12

    run_v9_experiment(n_episodes=n_episodes, max_turns=max_turns)
