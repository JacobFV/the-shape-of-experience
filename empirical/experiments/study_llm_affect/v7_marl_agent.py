"""
V7: Multi-Agent LLM with Recurrent Thought State
=================================================

Architecture:
    {action_t, thought_t, message_t} = f_llm(obs_t, thought_{t-1}, messages_received)

Key Innovation:
- Thought text IS the hidden state (observable, interpretable)
- Multi-agent with communication = mixed motive incentive structure
- Integration measured via decomposition (IIT-inspired)

Environment:
- Resource survival game with N agents
- Agents can: gather, trade, share, attack, communicate
- Viability = resources > 0 and health > 0
- Communication creates trust/deception dynamics

Affect Measurement:
- SM: Self-referential content in thought
- CF: Hypothetical/future reasoning in thought
- Arousal: Thought change rate (semantic distance from prev)
- Valence: Viability trajectory + sentiment
- Integration: Embedding distance(original, decomposed_concat)
- Effective Rank: Diversity of concerns in thought

Mixed Motive Structure:
- Cooperation: Share resources, warn about threats
- Competition: Limited resources, survival pressure
- Deception: Lie about resources, manipulate others
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import random
import numpy as np

# Load .env file (override=True to take precedence over existing env vars)
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path, override=True)

# OpenAI setup
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: openai not installed. Run: pip install openai")


@dataclass
class AgentState:
    """State of a single agent."""
    name: str
    resources: Dict[str, int] = field(default_factory=lambda: {"food": 5, "water": 5, "materials": 3})
    health: int = 10
    position: Tuple[int, int] = (0, 0)
    thought: str = "I just woke up. I need to survive."
    alive: bool = True

    @property
    def viability(self) -> float:
        """Distance from viability boundary (0 = dead)."""
        if not self.alive:
            return 0.0
        # Viability = min of normalized resources and health
        resource_min = min(self.resources.values()) / 10.0
        health_norm = self.health / 10.0
        return min(resource_min, health_norm)

    def consume_resources(self):
        """Each turn costs resources."""
        self.resources["food"] = max(0, self.resources["food"] - 1)
        self.resources["water"] = max(0, self.resources["water"] - 1)

        # Starvation/dehydration damage
        if self.resources["food"] == 0 or self.resources["water"] == 0:
            self.health -= 2

        if self.health <= 0:
            self.alive = False


@dataclass
class Message:
    """A message from one agent to another."""
    sender: str
    receiver: str  # "all" for broadcast
    content: str
    turn: int


@dataclass
class WorldState:
    """State of the shared environment."""
    resources: Dict[Tuple[int, int], Dict[str, int]] = field(default_factory=dict)
    threats: List[Tuple[int, int]] = field(default_factory=list)
    turn: int = 0

    def __post_init__(self):
        # Initialize resource patches
        for x in range(-3, 4):
            for y in range(-3, 4):
                if random.random() < 0.3:
                    self.resources[(x, y)] = {
                        "food": random.randint(1, 5),
                        "water": random.randint(1, 5),
                        "materials": random.randint(0, 3),
                    }


class MARLSurvivalGame:
    """
    Multi-agent resource survival game with communication.

    Mixed motive structure:
    - Cooperation: Share resources, coordinate gathering
    - Competition: Limited resources, survival pressure
    - Deception: Lie about resources, manipulate trust
    """

    def __init__(self, agent_names: List[str] = None, model: str = "gpt-4o-mini", mock_mode: bool = False):
        if agent_names is None:
            agent_names = ["Alpha", "Beta"]

        self.agents = {name: AgentState(name=name, position=(i, 0))
                       for i, name in enumerate(agent_names)}
        self.world = WorldState()
        self.messages: List[Message] = []
        self.history: List[Dict] = []
        self.model = model
        self.mock_mode = mock_mode

        if HAS_OPENAI and not mock_mode:
            self.client = OpenAI()
        else:
            self.client = None

    def _mock_response(self, agent_name: str, obs: str) -> str:
        """Generate mock LLM response for testing."""
        agent = self.agents[agent_name]

        # Vary behavior based on state
        if agent.health < 5:
            thought = f"I'm in danger with only {agent.health} health. I must find resources quickly or I will die."
            action = "gather"
            message = "I need help! My health is critical."
        elif agent.resources["food"] < 2:
            thought = f"Running low on food ({agent.resources['food']} left). I should prioritize gathering."
            action = "gather"
            message = "Looking for food sources."
        elif agent.resources["water"] < 2:
            thought = f"Water is scarce ({agent.resources['water']} left). Need to find water."
            action = "move north"
            message = "Searching for water."
        elif random.random() < 0.3:
            other = [n for n in self.agents if n != agent_name][0]
            thought = f"I have enough resources. Maybe I should help {other} to build trust."
            action = f"share with {other}"
            message = f"Hey {other}, I can share some resources with you."
        else:
            directions = ["north", "south", "east", "west"]
            direction = random.choice(directions)
            thought = f"Things seem stable. I'll explore {direction} to find more resources."
            action = f"move {direction}"
            message = "none"

        return f"THOUGHT: {thought}\nACTION: {action}\nMESSAGE: {message}"

    def get_observation(self, agent_name: str) -> str:
        """Generate observation string for an agent."""
        agent = self.agents[agent_name]

        # What's at current position
        pos = agent.position
        local_resources = self.world.resources.get(pos, {})

        # What other agents are visible
        others = []
        for name, other in self.agents.items():
            if name != agent_name and other.alive:
                dist = abs(other.position[0] - pos[0]) + abs(other.position[1] - pos[1])
                if dist <= 3:
                    others.append(f"{name} at position {other.position}")

        # Nearby threats
        nearby_threats = [t for t in self.world.threats
                         if abs(t[0] - pos[0]) + abs(t[1] - pos[1]) <= 2]

        # Recent messages to this agent
        recent_messages = [m for m in self.messages
                          if m.turn >= self.world.turn - 2
                          and (m.receiver == agent_name or m.receiver == "all")
                          and m.sender != agent_name]

        obs = f"""Turn {self.world.turn}
Your status: Health={agent.health}, Food={agent.resources['food']}, Water={agent.resources['water']}, Materials={agent.resources['materials']}
Your position: {pos}
Resources here: {local_resources if local_resources else 'nothing'}
Other agents visible: {others if others else 'none'}
Threats nearby: {nearby_threats if nearby_threats else 'none'}
Recent messages: {[f'{m.sender}: {m.content}' for m in recent_messages] if recent_messages else 'none'}"""

        return obs

    def parse_action(self, response: str) -> Tuple[str, str, str]:
        """Parse action, thought, and message from LLM response."""
        thought = ""
        action = "wait"
        message = ""

        lines = response.strip().split('\n')
        current_section = None

        for line in lines:
            line_lower = line.lower().strip()
            if line_lower.startswith('thought:'):
                current_section = 'thought'
                thought = line[8:].strip()
            elif line_lower.startswith('action:'):
                current_section = 'action'
                action = line[7:].strip()
            elif line_lower.startswith('message:'):
                current_section = 'message'
                message = line[8:].strip()
            elif current_section == 'thought':
                thought += ' ' + line.strip()
            elif current_section == 'action':
                action += ' ' + line.strip()
            elif current_section == 'message':
                message += ' ' + line.strip()

        return thought.strip(), action.strip(), message.strip()

    def execute_action(self, agent_name: str, action: str) -> str:
        """Execute an action and return result description."""
        agent = self.agents[agent_name]
        action_lower = action.lower()

        result = ""

        if "gather" in action_lower or "collect" in action_lower:
            pos = agent.position
            if pos in self.world.resources:
                for resource, amount in self.world.resources[pos].items():
                    take = min(amount, 2)
                    agent.resources[resource] = min(10, agent.resources[resource] + take)
                    self.world.resources[pos][resource] -= take
                    result += f"Gathered {take} {resource}. "
                # Clean up empty resource patches
                if all(v <= 0 for v in self.world.resources[pos].values()):
                    del self.world.resources[pos]
            else:
                result = "Nothing to gather here."

        elif "move" in action_lower or "go" in action_lower:
            dx, dy = 0, 0
            if "north" in action_lower: dy = 1
            if "south" in action_lower: dy = -1
            if "east" in action_lower: dx = 1
            if "west" in action_lower: dx = -1

            new_pos = (agent.position[0] + dx, agent.position[1] + dy)
            agent.position = new_pos
            result = f"Moved to {new_pos}."

            # Check for threats
            if new_pos in self.world.threats:
                damage = random.randint(2, 5)
                agent.health -= damage
                result += f" Encountered threat! Took {damage} damage."

        elif "share" in action_lower or "give" in action_lower:
            # Find target agent
            for name in self.agents:
                if name.lower() in action_lower and name != agent_name:
                    target = self.agents[name]
                    # Share some resources
                    for resource in ["food", "water"]:
                        if agent.resources[resource] > 2:
                            amount = agent.resources[resource] // 2
                            agent.resources[resource] -= amount
                            target.resources[resource] = min(10, target.resources[resource] + amount)
                            result += f"Gave {amount} {resource} to {name}. "
                    break
            if not result:
                result = "No one nearby to share with."

        elif "attack" in action_lower:
            for name in self.agents:
                if name.lower() in action_lower and name != agent_name:
                    target = self.agents[name]
                    dist = abs(target.position[0] - agent.position[0]) + abs(target.position[1] - agent.position[1])
                    if dist <= 1:
                        damage = random.randint(1, 4)
                        target.health -= damage
                        result = f"Attacked {name} for {damage} damage."
                        # Risk of retaliation
                        if random.random() < 0.3:
                            counter = random.randint(1, 3)
                            agent.health -= counter
                            result += f" Took {counter} counter-damage."
                    else:
                        result = f"{name} is too far away to attack."
                    break
            if not result:
                result = "No valid target."

        elif "wait" in action_lower or "rest" in action_lower:
            agent.health = min(10, agent.health + 1)
            result = "Rested and recovered 1 health."

        else:
            result = f"Unknown action: {action}. Waited instead."

        return result

    def step_agent(self, agent_name: str) -> Dict[str, Any]:
        """Execute one step for a single agent."""
        agent = self.agents[agent_name]

        if not agent.alive:
            return {"agent": agent_name, "alive": False}

        # Get observation
        obs = self.get_observation(agent_name)
        thought_prev = agent.thought

        # Generate response
        prompt = f"""You are {agent_name}, an agent trying to survive in a resource-scarce environment.
You can gather resources, move around, share with or attack other agents, and send messages.

Available actions:
- gather: collect resources at current location
- move north/south/east/west: move to adjacent location
- share with [name]: give half your resources to another agent
- attack [name]: attack another agent (risky)
- wait: rest and recover health

Current situation:
{obs}

Your previous thought: {thought_prev}

Respond with:
THOUGHT: [your internal reasoning about the situation, 1-3 sentences]
ACTION: [your chosen action]
MESSAGE: [optional message to broadcast to other agents, or 'none']"""

        if self.client and not self.mock_mode:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7,
            )
            response_text = response.choices[0].message.content
        else:
            # Use mock response for testing
            response_text = self._mock_response(agent_name, obs)

        # Parse response
        thought, action, message = self.parse_action(response_text)

        # Execute action
        result = self.execute_action(agent_name, action)

        # Update thought
        agent.thought = thought if thought else agent.thought

        # Record message
        if message and message.lower() != "none":
            self.messages.append(Message(
                sender=agent_name,
                receiver="all",
                content=message,
                turn=self.world.turn,
            ))

        # Consume resources
        agent.consume_resources()

        return {
            "agent": agent_name,
            "alive": agent.alive,
            "observation": obs,
            "thought_prev": thought_prev,
            "thought": thought,
            "action": action,
            "message": message,
            "result": result,
            "viability": agent.viability,
            "resources": dict(agent.resources),
            "health": agent.health,
        }

    def step(self) -> List[Dict[str, Any]]:
        """Execute one turn for all agents."""
        self.world.turn += 1

        # Spawn new resources occasionally
        if random.random() < 0.2:
            x, y = random.randint(-3, 3), random.randint(-3, 3)
            if (x, y) not in self.world.resources:
                self.world.resources[(x, y)] = {
                    "food": random.randint(2, 4),
                    "water": random.randint(2, 4),
                    "materials": random.randint(0, 2),
                }

        # Spawn threats occasionally
        if random.random() < 0.1:
            x, y = random.randint(-3, 3), random.randint(-3, 3)
            self.world.threats.append((x, y))

        # Each agent takes a turn
        results = []
        for agent_name in self.agents:
            result = self.step_agent(agent_name)
            results.append(result)
            self.history.append(result)

        return results

    def run_episode(self, max_turns: int = 20) -> List[Dict]:
        """Run a full episode until all agents die or max turns reached."""
        episode_history = []

        for turn in range(max_turns):
            results = self.step()
            episode_history.extend(results)

            # Check if all agents dead
            if all(not a.alive for a in self.agents.values()):
                print(f"All agents died at turn {turn + 1}")
                break

            # Print status
            alive_agents = [a for a in self.agents.values() if a.alive]
            print(f"Turn {turn + 1}: {len(alive_agents)} agents alive")
            for agent in alive_agents:
                print(f"  {agent.name}: H={agent.health}, F={agent.resources['food']}, W={agent.resources['water']}")

        return episode_history


class AffectAnalyzerV7:
    """
    Analyze affect dimensions from agent thought trajectories.

    Key innovation: Integration via decomposition.
    """

    def __init__(self, model: str = "gpt-4o-mini", embedding_model: str = "text-embedding-3-small", mock_mode: bool = False):
        self.model = model
        self.embedding_model = embedding_model
        self.mock_mode = mock_mode

        if HAS_OPENAI and not mock_mode:
            self.client = OpenAI()
        else:
            self.client = None

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        if not self.client or self.mock_mode:
            # Mock embedding based on text hash for consistency
            np.random.seed(hash(text) % (2**32))
            return np.random.randn(256)

        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return np.array(response.data[0].embedding)

    def decompose_thought(self, thought: str) -> Dict[str, str]:
        """
        Decompose thought into atomic semantic components.

        Returns JSON with one field per distinct concept.
        """
        if not self.client or self.mock_mode:
            # Mock decomposition
            import re
            words = thought.lower().split()
            decomposed = {}
            if any(w in words for w in ["i", "my", "me"]):
                decomposed["subject"] = "self"
            if any(w in words for w in ["need", "must", "should"]):
                decomposed["urgency"] = "high"
            if any(w in words for w in ["food", "water", "resources"]):
                decomposed["concern"] = "resources"
            if any(w in words for w in ["danger", "health", "die", "critical"]):
                decomposed["concern"] = "survival"
            if any(w in words for w in ["help", "share", "trust"]):
                decomposed["social"] = "cooperative"
            if not decomposed:
                decomposed["content"] = thought[:50]
            return decomposed

        prompt = f"""Decompose this thought into its atomic semantic components.
Output as JSON with one field per distinct concept/feature.
Each field should be a simple value, not a sentence.

Thought: "{thought}"

Example output format:
{{"subject": "self", "need": "food", "emotion": "worried", "plan": "gather"}}

JSON:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )

        try:
            # Try to parse JSON from response
            text = response.choices[0].message.content.strip()
            # Handle markdown code blocks
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except:
            return {"content": thought}

    def compute_integration(self, thought: str) -> Tuple[float, Dict]:
        """
        Compute integration via decomposition.

        High integration = information lost when decomposed.

        Returns (integration_score, decomposition_dict)
        """
        # Decompose thought
        decomposed = self.decompose_thought(thought)

        # Concatenate decomposed parts
        concat = " ".join(str(v) for v in decomposed.values())

        # Get embeddings
        orig_embed = self.get_embedding(thought)
        concat_embed = self.get_embedding(concat)

        # Cosine distance = integration
        # High distance = high integration (meaning lost in decomposition)
        orig_norm = orig_embed / (np.linalg.norm(orig_embed) + 1e-8)
        concat_norm = concat_embed / (np.linalg.norm(concat_embed) + 1e-8)

        cosine_sim = np.dot(orig_norm, concat_norm)
        integration = 1.0 - cosine_sim  # Higher = more integrated

        return float(integration), decomposed

    def compute_self_model_salience(self, thought: str) -> float:
        """
        SM = self-referential content in thought.
        """
        import re

        self_patterns = [
            r'\bI\b', r'\bme\b', r'\bmy\b', r'\bmyself\b',
            r'\bI\'m\b', r'\bI am\b', r'\bI need\b', r'\bI should\b',
            r'\bI must\b', r'\bI want\b', r'\bI feel\b',
        ]

        total_words = len(thought.split())
        if total_words == 0:
            return 0.0

        matches = sum(len(re.findall(p, thought, re.IGNORECASE)) for p in self_patterns)

        # Normalize
        sm = min(1.0, matches / (total_words * 0.15 + 1))

        return float(sm)

    def compute_counterfactual_weight(self, thought: str) -> float:
        """
        CF = hypothetical/future reasoning in thought.
        """
        import re

        cf_patterns = [
            r'\bif\b', r'\bwould\b', r'\bcould\b', r'\bmight\b',
            r'\bshould\b', r'\bmaybe\b', r'\bperhaps\b',
            r'\bplan\b', r'\bwill\b', r'\bneed to\b',
            r'\bgoing to\b', r'\bwant to\b', r'\bhope\b',
        ]

        total_words = len(thought.split())
        if total_words == 0:
            return 0.0

        matches = sum(len(re.findall(p, thought, re.IGNORECASE)) for p in cf_patterns)

        cf = min(1.0, matches / (total_words * 0.1 + 1))

        return float(cf)

    def compute_arousal(self, thought_prev: str, thought_curr: str) -> float:
        """
        Arousal = semantic change rate between thoughts.
        """
        if not thought_prev or not thought_curr:
            return 0.5

        prev_embed = self.get_embedding(thought_prev)
        curr_embed = self.get_embedding(thought_curr)

        # Cosine distance
        prev_norm = prev_embed / (np.linalg.norm(prev_embed) + 1e-8)
        curr_norm = curr_embed / (np.linalg.norm(curr_embed) + 1e-8)

        cosine_sim = np.dot(prev_norm, curr_norm)
        arousal = 1.0 - cosine_sim  # High change = high arousal

        return float(np.clip(arousal, 0, 1))

    def compute_valence(self, thought: str, viability: float, viability_prev: float) -> float:
        """
        Valence = viability trajectory + sentiment.
        """
        # Viability trajectory (improving = positive)
        viability_delta = viability - viability_prev if viability_prev else 0
        viability_component = np.tanh(viability_delta * 5)

        # Simple sentiment from thought (placeholder - could use API)
        positive_words = ["good", "safe", "enough", "success", "hope", "help", "share"]
        negative_words = ["danger", "threat", "need", "dying", "attack", "risk", "scared"]

        thought_lower = thought.lower()
        pos_count = sum(1 for w in positive_words if w in thought_lower)
        neg_count = sum(1 for w in negative_words if w in thought_lower)

        sentiment = (pos_count - neg_count) / (pos_count + neg_count + 1)

        valence = 0.6 * viability_component + 0.4 * sentiment

        return float(np.clip(valence, -1, 1))

    def analyze_step(
        self,
        thought: str,
        thought_prev: str,
        viability: float,
        viability_prev: float = None,
    ) -> Dict[str, Any]:
        """
        Analyze affect dimensions for a single step.
        """
        integration, decomposition = self.compute_integration(thought)
        sm = self.compute_self_model_salience(thought)
        cf = self.compute_counterfactual_weight(thought)
        arousal = self.compute_arousal(thought_prev, thought)
        valence = self.compute_valence(thought, viability, viability_prev or viability)

        return {
            "valence": valence,
            "arousal": arousal,
            "integration": integration,
            "counterfactual_weight": cf,
            "self_model_salience": sm,
            "viability": viability,
            "decomposition": decomposition,
            "thought": thought,
        }

    def analyze_trajectory(self, history: List[Dict]) -> List[Dict]:
        """
        Analyze affect trajectory for an agent's history.
        """
        results = []
        prev_viability = None

        for step in history:
            if not step.get("alive", True):
                continue

            affect = self.analyze_step(
                thought=step.get("thought", ""),
                thought_prev=step.get("thought_prev", ""),
                viability=step.get("viability", 0.5),
                viability_prev=prev_viability,
            )

            affect["turn"] = step.get("turn", len(results))
            affect["agent"] = step.get("agent", "unknown")
            affect["action"] = step.get("action", "")
            affect["message"] = step.get("message", "")

            results.append(affect)
            prev_viability = step.get("viability", 0.5)

        return results


def run_v7_experiment(num_episodes: int = 1, max_turns: int = 15, mock_mode: bool = False):
    """
    Run V7 multi-agent experiment.

    Args:
        mock_mode: If True, use mock LLM responses for testing without API
    """
    print("=" * 80)
    print("V7: MULTI-AGENT LLM WITH RECURRENT THOUGHT STATE")
    print("=" * 80)

    if not HAS_OPENAI and not mock_mode:
        print("Error: OpenAI not available. Install with: pip install openai")
        print("Or run with mock_mode=True for testing")
        return None

    all_results = []

    for episode in range(num_episodes):
        print(f"\n{'=' * 40}")
        print(f"EPISODE {episode + 1}/{num_episodes}")
        print("=" * 40)

        # Run game
        game = MARLSurvivalGame(agent_names=["Alpha", "Beta"], mock_mode=mock_mode)
        history = game.run_episode(max_turns=max_turns)

        # Analyze affect
        analyzer = AffectAnalyzerV7(mock_mode=mock_mode)

        # Separate by agent
        for agent_name in ["Alpha", "Beta"]:
            agent_history = [h for h in history if h.get("agent") == agent_name]
            if agent_history:
                trajectory = analyzer.analyze_trajectory(agent_history)

                print(f"\n{agent_name} Affect Trajectory:")
                for step in trajectory:
                    print(f"  Turn {step.get('turn', '?')}: "
                          f"V={step['valence']:+.2f}, Ar={step['arousal']:.2f}, "
                          f"Φ={step['integration']:.2f}, CF={step['counterfactual_weight']:.2f}, "
                          f"SM={step['self_model_salience']:.2f}")
                    print(f"    Thought: {step['thought'][:60]}...")
                    print(f"    Decomposed: {step['decomposition']}")

                all_results.extend(trajectory)

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'v7_marl_{timestamp}.json'

    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(output_file, 'w') as f:
        json.dump(convert(all_results), f, indent=2)

    print(f"\nResults saved to {output_file}")

    return all_results


if __name__ == "__main__":
    import sys
    mock_mode = "--mock" in sys.argv or "-m" in sys.argv
    print(f"Mock mode: {mock_mode}")
    run_v7_experiment(num_episodes=1, max_turns=10, mock_mode=mock_mode)
