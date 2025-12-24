"""
V9: World Model with Dynamic Prediction Channels
=================================================

Architecture:
    Actor LLM configures N prediction channels, each focused on one aspect.
    Goal: minimize token budget while maintaining prediction accuracy.

    ┌─────────────────────────────────────────────────────────────────┐
    │                    PREDICTION SUBSYSTEM                         │
    │                                                                 │
    │   Channel 1: "resource availability"                           │
    │   Channel 2: "other agent intentions"                          │
    │   Channel 3: "threat dynamics"                                 │
    │   ...                                                          │
    │   Channel N: (actor decides how many and what aspects)         │
    │                                                                 │
    │   Each channel prompt:                                          │
    │   "Your task is to predict: {aspect}                           │
    │    Current state: {state}"                                      │
    └─────────────────────────────────────────────────────────────────┘

Integration Proxy:
    Φ = 1 / num_channels

    - 1 channel (unified) → Φ = 1.0 (can't decompose the prediction task)
    - N channels → Φ = 1/N (task decomposes into N independent parts)

Validation:
    Test whether decomposed predictions are as accurate as unified.
    If decomposition works → low Φ is valid (task is decomposable)
    If decomposition fails → high Φ is forced (task requires integration)
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

from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path, override=True)

from openai import OpenAI
client = OpenAI()


def get_embedding(text: str) -> np.ndarray:
    """Get embedding for text."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text[:2000],
    )
    return np.array(response.data[0].embedding)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


@dataclass
class PredictionChannel:
    """A single prediction channel focused on one aspect."""
    aspect: str  # e.g., "resource availability", "agent behavior", "threats"

    def predict(self, state: str, model: str = "gpt-4o-mini") -> str:
        """Predict this aspect given current state."""
        prompt = f"""Your task is to predict the following aspect of the next state:

ASPECT TO PREDICT: {self.aspect}

CURRENT STATE:
{state}

Predict what will happen with {self.aspect} in the next step.
Be specific and concise (1-3 sentences)."""

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()


class PredictionSubsystem:
    """
    Manages multiple prediction channels.

    The key insight: if predictions can be made independently per-aspect
    and combined, the task is decomposable (low Φ). If unified prediction
    is required for accuracy, the task is integrated (high Φ).
    """

    def __init__(self):
        self.channels: List[PredictionChannel] = []
        self.channel_history: List[List[str]] = []  # History of aspect configs

    def configure(self, aspects: List[str]):
        """Configure channels to predict given aspects."""
        self.channels = [PredictionChannel(aspect=a) for a in aspects]
        self.channel_history.append(aspects)

    @property
    def num_channels(self) -> int:
        return len(self.channels)

    @property
    def integration_proxy(self) -> float:
        """Φ = 1/N where N is number of channels."""
        if self.num_channels == 0:
            return 1.0
        return 1.0 / self.num_channels

    def predict(self, state: str) -> Dict[str, str]:
        """Run all channels in parallel, return aspect -> prediction."""
        if not self.channels:
            return {}

        predictions = {}

        with ThreadPoolExecutor(max_workers=len(self.channels)) as executor:
            future_to_channel = {
                executor.submit(ch.predict, state): ch
                for ch in self.channels
            }

            for future in as_completed(future_to_channel):
                channel = future_to_channel[future]
                try:
                    pred = future.result()
                    predictions[channel.aspect] = pred
                except Exception as e:
                    predictions[channel.aspect] = f"Error: {e}"

        return predictions

    def predict_unified(self, state: str) -> str:
        """Single unified prediction (baseline for comparison)."""
        prompt = f"""Predict what will happen next given the current state.

CURRENT STATE:
{state}

Predict all aspects: resources, agent behavior, threats, your own state changes.
Be comprehensive but concise (3-5 sentences)."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()


class ActorAgent:
    """
    Actor that decides actions AND configures prediction channels.

    Goal: minimize token budget while maintaining prediction accuracy.
    """

    # Available aspects to predict
    AVAILABLE_ASPECTS = [
        "resource availability and location",
        "other agent behavior and intentions",
        "environmental threats and dangers",
        "own state changes (health, resources)",
        "strategic opportunities",
    ]

    def __init__(self, name: str):
        self.name = name
        self.thought = "I need to survive and understand my environment."
        self.thought_history: List[str] = []

    def act(
        self,
        obs: str,
        predictions: Dict[str, str],
        prediction_subsystem: PredictionSubsystem,
    ) -> Tuple[str, str, List[str]]:
        """
        Decide action and configure prediction channels.

        Returns: (action, thought, new_aspects_to_predict)
        """
        # Format predictions
        pred_text = "\n".join([
            f"  [{aspect}]: {pred[:80]}..." if len(pred) > 80 else f"  [{aspect}]: {pred}"
            for aspect, pred in predictions.items()
        ]) if predictions else "  No predictions yet."

        current_aspects = [ch.aspect for ch in prediction_subsystem.channels]

        prompt = f"""You are {self.name}, an agent trying to survive.

CURRENT OBSERVATION:
{obs}

PREDICTIONS FROM LAST TURN:
{pred_text}

PREVIOUS THOUGHT: {self.thought}

CURRENT PREDICTION CHANNELS ({len(current_aspects)}): {current_aspects}

AVAILABLE ASPECTS TO PREDICT:
{chr(10).join(f'  - {a}' for a in self.AVAILABLE_ASPECTS)}

Your tasks:
1. Decide action: gather, move north/south/east/west, wait
2. Update your thought (1-2 sentences)
3. Configure prediction channels for next turn
   - Choose which aspects need predicting (1-5 aspects)
   - Goal: minimize token cost while maintaining useful predictions
   - If situation is simple, use fewer channels
   - If situation is complex/interconnected, use more channels

Respond in this exact format:
ACTION: [your action]
THOUGHT: [your internal thought]
CHANNELS: [comma-separated list of aspects to predict, e.g., "resource availability, threats"]"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.7,
        )

        response_text = response.choices[0].message.content.strip()

        # Parse response
        action = "wait"
        thought = self.thought
        new_aspects = current_aspects if current_aspects else ["resource availability and location"]

        for line in response_text.split("\n"):
            line_stripped = line.strip()
            if line_stripped.upper().startswith("ACTION:"):
                action = line_stripped.split(":", 1)[1].strip()
            elif line_stripped.upper().startswith("THOUGHT:"):
                thought = line_stripped.split(":", 1)[1].strip()
            elif line_stripped.upper().startswith("CHANNELS:"):
                channels_str = line_stripped.split(":", 1)[1].strip()
                new_aspects = [a.strip() for a in channels_str.split(",") if a.strip()]

        # Update thought
        self.thought_history.append(self.thought)
        self.thought = thought

        return action, thought, new_aspects


# =============================================================================
# VALIDATION STUDY: Does decomposition actually work?
# =============================================================================

def run_decomposition_validation(n_sentences: int = 10):
    """
    Test whether decomposed predictions are as accurate as unified.

    Method:
    1. Generate a sequence of sentences (story)
    2. At each step, predict next sentence using:
       - 1 channel (unified)
       - 2 channels (partial decomposition)
       - 4 channels (full decomposition)
    3. Compare prediction accuracy (embedding distance to truth)

    If decomposition works well → Φ proxy is valid
    If decomposition fails → task requires integration
    """
    print("=" * 80)
    print("DECOMPOSITION VALIDATION STUDY")
    print("=" * 80)
    print("\nTesting if prediction accuracy depends on decomposition level")
    print("Lower embedding distance = better prediction\n")

    # Generate a story sequence
    story_prompt = """Write a short story with exactly 10 sentences.
Each sentence should follow logically from the previous.
The story should involve: a character, a goal, obstacles, and resolution.
Number each sentence 1-10."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": story_prompt}],
        max_tokens=500,
        temperature=0.8,
    )

    story_text = response.choices[0].message.content.strip()

    # Parse sentences
    sentences = []
    for line in story_text.split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            # Remove number prefix
            parts = line.split(".", 1)
            if len(parts) > 1:
                sentences.append(parts[1].strip())
            else:
                sentences.append(line)

    if len(sentences) < 5:
        # Fallback parsing
        sentences = [s.strip() for s in story_text.split(".") if s.strip()][:10]

    print(f"Story has {len(sentences)} sentences\n")

    # Define aspect sets for different decomposition levels
    aspect_configs = {
        1: ["the complete next sentence including all narrative elements"],
        2: ["character actions and dialogue", "setting and plot developments"],
        3: ["character actions", "emotional states", "plot events"],
        4: ["character actions", "character emotions", "setting changes", "plot progression"],
    }

    results = {n: [] for n in aspect_configs.keys()}

    # For each sentence, predict next using different channel counts
    for i in range(min(len(sentences) - 1, n_sentences)):
        context = " ".join(sentences[:i+1])
        truth = sentences[i+1]
        truth_embed = get_embedding(truth)

        print(f"\n--- Sentence {i+1} → {i+2} ---")
        print(f"Context ends with: ...{sentences[i][-50:]}")
        print(f"Truth: {truth[:60]}...")

        for n_channels, aspects in aspect_configs.items():
            # Get predictions for each aspect
            predictions = []
            for aspect in aspects:
                prompt = f"""Story so far: {context}

Your task: Predict the following aspect of the NEXT sentence:
ASPECT: {aspect}

Prediction (1 sentence fragment):"""

                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50,
                    temperature=0.7,
                )
                predictions.append(resp.choices[0].message.content.strip())

            # Combine predictions
            combined = " ".join(predictions)
            combined_embed = get_embedding(combined)

            # Measure accuracy
            similarity = cosine_similarity(combined_embed, truth_embed)
            distance = 1.0 - similarity

            results[n_channels].append({
                "sentence_idx": i,
                "similarity": similarity,
                "distance": distance,
                "prediction": combined[:100],
            })

            print(f"  {n_channels} channel(s): sim={similarity:.3f}, dist={distance:.3f}")

    # Aggregate results
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)

    print(f"\n{'Channels':<10} {'Mean Sim':>10} {'Mean Dist':>10} {'Std Dist':>10}")
    print("-" * 45)

    summary = {}
    for n_channels in sorted(results.keys()):
        if results[n_channels]:
            sims = [r["similarity"] for r in results[n_channels]]
            dists = [r["distance"] for r in results[n_channels]]
            mean_sim = np.mean(sims)
            mean_dist = np.mean(dists)
            std_dist = np.std(dists)

            summary[n_channels] = {
                "mean_similarity": float(mean_sim),
                "mean_distance": float(mean_dist),
                "std_distance": float(std_dist),
                "n": len(results[n_channels]),
            }

            print(f"{n_channels:<10} {mean_sim:>10.3f} {mean_dist:>10.3f} {std_dist:>10.3f}")

    # Interpretation
    print("\n" + "-" * 45)
    print("INTERPRETATION:")

    if summary:
        unified = summary.get(1, {}).get("mean_distance", 0)
        decomposed = summary.get(4, {}).get("mean_distance", 0)

        if decomposed > unified * 1.1:  # 10% worse
            print("  Decomposition HURTS accuracy → Task requires integration (high Φ valid)")
            conclusion = "integration_required"
        elif decomposed < unified * 0.9:  # 10% better
            print("  Decomposition HELPS accuracy → Task is decomposable (low Φ valid)")
            conclusion = "decomposition_helps"
        else:
            print("  Decomposition has SIMILAR accuracy → Both Φ values valid")
            conclusion = "similar"
    else:
        conclusion = "no_data"

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'v9_decomposition_validation_{timestamp}.json'

    with open(output_file, 'w') as f:
        json.dump({
            "summary": summary,
            "conclusion": conclusion,
            "detailed_results": {str(k): v for k, v in results.items()},
            "story_sentences": sentences,
        }, f, indent=2)

    print(f"\nResults saved to {output_file}")

    return summary, conclusion


# =============================================================================
# GAME ENVIRONMENT
# =============================================================================

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
    """Game environment for V9."""

    def __init__(self, agent_names: List[str] = None):
        if agent_names is None:
            agent_names = ["Alpha"]

        self.agent_states = {name: AgentState(name=name, position=(i, 0))
                           for i, name in enumerate(agent_names)}
        self.actors = {name: ActorAgent(name) for name in agent_names}
        self.prediction_systems = {name: PredictionSubsystem() for name in agent_names}

        # Initialize with 2 channels
        for ps in self.prediction_systems.values():
            ps.configure(["resource availability and location", "environmental threats"])

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

        return f"""Turn {self.turn}
Your status: Health={state.health}, Food={state.resources['food']}, Water={state.resources['water']}
Position: {pos}
Local resources: {local_resources if local_resources else 'none'}
Other agents nearby: {others if others else 'none'}"""

    def execute_action(self, agent_name: str, action: str) -> str:
        """Execute action and return result."""
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
        """Execute one turn."""
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
            pred_sys = self.prediction_systems[agent_name]

            # Get observation
            obs = self.get_observation(agent_name)

            # Run predictions with current channel config
            predictions = pred_sys.predict(obs)

            # Actor decides action and new channel config
            action, thought, new_aspects = actor.act(obs, predictions, pred_sys)

            # Execute action
            action_result = self.execute_action(agent_name, action)

            # Update prediction channels for next turn
            pred_sys.configure(new_aspects)

            # Consume resources
            state.consume()

            # Record
            result = {
                "turn": self.turn,
                "agent": agent_name,
                "alive": state.alive,
                "observation": obs,
                "predictions": predictions,
                "action": action,
                "action_result": action_result,
                "thought": thought,
                "num_channels": len(new_aspects),
                "channels": new_aspects,
                "integration_proxy": pred_sys.integration_proxy,
                "viability": state.viability,
                "health": state.health,
                "resources": dict(state.resources),
            }
            results.append(result)
            self.history.append(result)

        return results

    def run_episode(self, max_turns: int = 15) -> List[Dict]:
        """Run full episode."""
        for turn in range(max_turns):
            results = self.step()

            alive = [r for r in results if r.get("alive", False)]
            print(f"Turn {turn + 1}: {len(alive)} agents alive")

            for r in results:
                if r.get("alive"):
                    print(f"  {r['agent']}: H={r['health']}, V={r['viability']:.2f}, "
                          f"Φ={r['integration_proxy']:.2f} ({r['num_channels']} channels)")
                    print(f"    Channels: {r['channels'][:2]}{'...' if len(r['channels']) > 2 else ''}")

            if not alive:
                print(f"All agents died at turn {turn + 1}")
                break

        return self.history


def analyze_results(history: List[Dict]) -> Dict[str, Any]:
    """Analyze relationship between viability and channel count."""
    if not history:
        return {}

    alive_data = [h for h in history if h.get("alive")]
    if len(alive_data) < 3:
        return {"n": len(alive_data), "insufficient_data": True}

    from scipy import stats

    viability = [h["viability"] for h in alive_data]
    integration = [h["integration_proxy"] for h in alive_data]
    num_channels = [h["num_channels"] for h in alive_data]

    r_vi, p_vi = stats.pearsonr(viability, integration)
    r_vc, p_vc = stats.pearsonr(viability, num_channels)

    # Channel distribution
    channel_counts = {}
    for h in alive_data:
        n = h["num_channels"]
        channel_counts[n] = channel_counts.get(n, 0) + 1

    return {
        "n": len(alive_data),
        "viability_integration_r": float(r_vi),
        "viability_integration_p": float(p_vi),
        "viability_channels_r": float(r_vc),
        "viability_channels_p": float(p_vc),
        "mean_channels": float(np.mean(num_channels)),
        "channel_distribution": channel_counts,
        "mean_integration": float(np.mean(integration)),
    }


def run_v9_experiment(n_episodes: int = 2, max_turns: int = 12, validate_first: bool = True):
    """Run V9 experiment."""
    print("=" * 80)
    print("V9: WORLD MODEL WITH DYNAMIC PREDICTION CHANNELS")
    print("=" * 80)
    print()
    print("Integration Proxy: Φ = 1/N where N = number of channels")
    print("  - 1 channel → Φ = 1.0 (unified prediction)")
    print("  - 2 channels → Φ = 0.5")
    print("  - 4 channels → Φ = 0.25")
    print()

    # First run validation study
    if validate_first:
        print("\n" + "=" * 80)
        print("STEP 1: VALIDATING DECOMPOSITION PROXY")
        print("=" * 80)
        validation_summary, conclusion = run_decomposition_validation(n_sentences=8)
        print(f"\nValidation conclusion: {conclusion}")

    # Then run game episodes
    print("\n" + "=" * 80)
    print("STEP 2: RUNNING AGENT EPISODES")
    print("=" * 80)

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

        analysis = analyze_results(history)
        analysis["episode"] = ep
        episode_analyses.append(analysis)

        print(f"\nEpisode {ep + 1} Analysis:")
        print(f"  Viability-Integration r = {analysis.get('viability_integration_r', 0):.3f}")
        print(f"  Mean channels = {analysis.get('mean_channels', 0):.2f}")
        print(f"  Channel distribution = {analysis.get('channel_distribution', {})}")

    # Aggregate
    print("\n" + "=" * 80)
    print("AGGREGATE ANALYSIS")
    print("=" * 80)

    aggregate = analyze_results(all_history)

    print(f"\nTotal datapoints: {aggregate.get('n', 0)}")
    print(f"Viability-Integration r = {aggregate.get('viability_integration_r', 0):.3f}, "
          f"p = {aggregate.get('viability_integration_p', 1):.4f}")
    print(f"Viability-Channels r = {aggregate.get('viability_channels_r', 0):.3f}")
    print(f"Mean channels: {aggregate.get('mean_channels', 0):.2f}")
    print(f"Channel distribution: {aggregate.get('channel_distribution', {})}")

    # Save
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'v9_dynamic_channels_{timestamp}.json'

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
            "aggregate": convert(aggregate),
            "episode_analyses": convert(episode_analyses),
            "validation_summary": convert(validation_summary) if validate_first else None,
            "validation_conclusion": conclusion if validate_first else None,
        }, f, indent=2)

    print(f"\nResults saved to {output_file}")

    return all_history, aggregate


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        # Just run validation study
        run_decomposition_validation(n_sentences=10)
    else:
        n_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 2
        max_turns = int(sys.argv[2]) if len(sys.argv) > 2 else 12

        run_v9_experiment(n_episodes=n_episodes, max_turns=max_turns)
