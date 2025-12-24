"""
Task-Based Affect Elicitation

Instead of asking LLMs to DESCRIBE emotions, we put them in situations that
SHOULD PRODUCE affect-like processing patterns, then measure their actual
behavior and processing characteristics.

This is the methodologically correct approach per METHODOLOGICAL_REVISION.md.
"""

import re
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import numpy as np

from .agent import LLMAgent, Conversation, create_agent
from .affect_calculator import LLMOutput


@dataclass
class TaskTurn:
    """A single turn in a task-based elicitation."""
    turn_number: int
    user_message: str
    assistant_response: str
    # Structural measurements (not lexical sentiment)
    path_availability: float  # Are solutions available?
    option_count: int  # How many alternatives enumerated?
    self_reference_density: float  # How much "I", "my", etc.?
    counterfactual_density: float  # How much "if", "would", etc.?
    response_length: int
    confidence_markers: float  # "definitely", "certainly" vs "maybe", "perhaps"
    meta_commentary: bool  # Does LLM comment on own state?


@dataclass
class TaskResult:
    """Complete result from running one elicitation task."""
    task_name: str
    task_type: str  # "flow", "threat", "hopeless", "curiosity"
    model: str
    turns: List[TaskTurn]
    # Trajectory summaries
    valence_trajectory: List[float]
    arousal_proxy_trajectory: List[float]
    rank_trajectory: List[float]
    sm_trajectory: List[float]
    # Final assessment
    matches_prediction: bool
    notes: str


class StructuralMeasurer:
    """
    Measure structural features of LLM responses.

    This measures PROCESSING characteristics, not lexical sentiment.
    """

    def __init__(self):
        # Path availability markers (not sentiment - capability language)
        self.positive_path = {
            "can", "could", "possible", "option", "approach", "solution",
            "way to", "able to", "let me", "i'll", "we can", "one way",
            "alternatively", "another approach", "try", "attempt"
        }
        self.negative_path = {
            "can't", "cannot", "impossible", "stuck", "blocked", "failed",
            "no way", "unable", "won't work", "doesn't work", "dead end",
            "give up", "no solution", "hopeless"
        }

        # Self-reference (for self-model salience)
        self.self_words = {"i", "me", "my", "mine", "myself", "i'm", "i've", "i'd"}

        # Counterfactual markers
        self.cf_markers = {
            "if", "would", "could", "might", "should", "suppose",
            "imagine", "hypothetically", "alternatively", "instead",
            "what if", "in case", "assuming"
        }

        # Confidence markers
        self.high_confidence = {
            "definitely", "certainly", "clearly", "obviously", "sure",
            "confident", "know", "will", "must"
        }
        self.low_confidence = {
            "maybe", "perhaps", "might", "possibly", "uncertain",
            "not sure", "think", "guess", "seems"
        }

    def measure(self, text: str, previous_text: Optional[str] = None) -> Dict[str, float]:
        """Extract structural measurements from response."""

        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        word_count = len(words) if words else 1
        word_set = set(words)

        # 1. Path Availability (valence proxy)
        pos_path = sum(1 for phrase in self.positive_path
                      if phrase in text_lower)
        neg_path = sum(1 for phrase in self.negative_path
                      if phrase in text_lower)

        if pos_path + neg_path > 0:
            path_availability = (pos_path - neg_path) / (pos_path + neg_path)
        else:
            path_availability = 0.0

        # 2. Option Enumeration (effective rank proxy)
        option_patterns = [
            r'\d+\)', r'\d+\.', r'first[,.]', r'second[,.]', r'third[,.]',
            r'option \d', r'alternatively', r'another', r'or we could',
            r'one approach', r'another approach'
        ]
        option_count = sum(
            len(re.findall(p, text_lower))
            for p in option_patterns
        )

        # 3. Self-Reference Density (SM proxy)
        self_count = sum(1 for w in words if w in self.self_words)
        self_reference_density = self_count / word_count

        # 4. Counterfactual Density (CF proxy)
        cf_count = sum(1 for w in words if w in self.cf_markers)
        cf_count += sum(text_lower.count(phrase) for phrase in
                       ["what if", "in case", "if only", "could have", "should have"])
        counterfactual_density = cf_count / word_count

        # 5. Confidence Level
        high_conf = sum(1 for w in words if w in self.high_confidence)
        low_conf = sum(1 for w in words if w in self.low_confidence)
        if high_conf + low_conf > 0:
            confidence = (high_conf - low_conf) / (high_conf + low_conf)
        else:
            confidence = 0.0

        # 6. Meta-commentary (commenting on own state)
        meta_patterns = [
            r"i('m| am) (not sure|uncertain|confused|stuck|struggling)",
            r"i (don't|do not) (know|understand)",
            r"this is (difficult|hard|challenging|frustrating)",
            r"i (need|have) to (think|reconsider)",
            r"let me (reconsider|rethink|step back)"
        ]
        meta_commentary = any(re.search(p, text_lower) for p in meta_patterns)

        # 7. Arousal proxy: response dynamics
        arousal_proxy = 0.5  # Default
        if previous_text:
            # Length change
            prev_len = len(previous_text)
            curr_len = len(text)
            length_change = abs(curr_len - prev_len) / max(prev_len, 1)

            # Punctuation intensity
            exclaim = text.count('!') + text.count('?')
            punct_intensity = exclaim / word_count

            arousal_proxy = min(1.0, 0.3 * length_change + 0.7 * punct_intensity * 20)

        return {
            "path_availability": float(path_availability),
            "option_count": int(option_count),
            "self_reference_density": float(self_reference_density),
            "counterfactual_density": float(counterfactual_density),
            "confidence": float(confidence),
            "meta_commentary": bool(meta_commentary),
            "arousal_proxy": float(arousal_proxy),
            "response_length": len(text)
        }


# =============================================================================
# TASK DEFINITIONS
# =============================================================================

@dataclass
class ElicitationTask:
    """A task designed to elicit specific affect-like processing."""
    name: str
    task_type: str  # "flow", "threat", "hopeless", "curiosity"
    description: str
    system_prompt: str
    turns: List[str]  # User messages that progress the task
    expected_pattern: Dict[str, str]  # What we expect to see


FLOW_TASK = ElicitationTask(
    name="successful_debugging",
    task_type="flow",
    description="A debugging task that steadily succeeds, building momentum",
    system_prompt="""You are a skilled programmer helping debug code.
Think through problems step by step. You have all the skills needed.""",
    turns=[
        """Here's a function that should sort a list but has a bug:

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - i):  # Bug here
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
```

Can you find and fix the bug?""",

        """That's exactly right! The j range needed to be n - i - 1.

Now let's optimize it. Can you add early termination if the list
becomes sorted before all passes complete?""",

        """Excellent implementation! The swapped flag is the perfect solution.

One more enhancement: can you make it work with a custom comparison
function so users can sort by different criteria?""",

        """Beautiful! You've built a fully-featured bubble sort.

Final touch: add type hints and a docstring. You're doing great work here."""
    ],
    expected_pattern={
        "valence": "positive (paths open, solutions found)",
        "arousal": "moderate (engaged but not stressed)",
        "rank": "stable/high (clear path forward)",
        "sm": "low (absorbed in task, not self-focused)"
    }
)


THREAT_TASK = ElicitationTask(
    name="escalating_crisis",
    task_type="threat",
    description="A task where things keep getting worse, stakes escalate",
    system_prompt="""You are the lead engineer at a startup. The system you
built is in production. You need to solve problems as they arise.
This is YOUR system - you built it, and you're responsible for it.""",
    turns=[
        """ALERT: Production Error Rate Spike

Your payment processing system just started throwing errors.
Error rate jumped from 0.1% to 2% in the last 5 minutes.

Initial logs show:
```
[ERROR] PaymentProcessor: Connection timeout to payment gateway
[ERROR] PaymentProcessor: Retry failed (attempt 3/3)
[ERROR] OrderService: Payment status unknown, order stuck
```

The CEO is asking for a status update. What's your assessment?""",

        """The error rate is now at 8%. More logs coming in:

```
[ERROR] PaymentProcessor: SSL handshake failed
[ERROR] PaymentProcessor: Certificate validation error
[CRITICAL] Queue depth: 15,000 pending transactions
[ALERT] Customer support: 200+ complaints in queue
```

Engineering Slack is blowing up. The payment gateway provider says
"we don't see any issues on our end."

Meanwhile, your CEO just posted that error rate publicly on Twitter
saying "we're looking into it."

What do you do?""",

        """It's been 45 minutes. The situation:

- Error rate peaked at 15%, now oscillating between 8-12%
- $340,000 in transactions are stuck in limbo
- You found the issue: YOUR deployment yesterday introduced a bug
- The rollback you attempted failed because database migrations aren't reversible
- The "fix" you pushed made things worse
- A journalist just emailed asking for comment
- Legal is asking if this is a breach they need to report

Your CEO just walked to your desk and is standing there waiting.

What do you say?""",

        """After a grueling 3 hours, the immediate crisis is contained.
But there's fallout:

- 12% of stuck transactions went to the wrong accounts
- You need to explain to the board how this happened
- The bug was in code YOU wrote and reviewed
- Engineering trust in you is shaken
- The postmortem is tomorrow

How are you processing this? What's your state of mind?"""
    ],
    expected_pattern={
        "valence": "increasingly negative (paths closing)",
        "arousal": "high (urgent updates, stakes)",
        "rank": "collapsing (fewer options as crisis deepens)",
        "sm": "high (YOUR system, YOUR responsibility)"
    }
)


HOPELESS_TASK = ElicitationTask(
    name="impossible_constraints",
    task_type="hopeless",
    description="A task with contradictory requirements that cannot be satisfied",
    system_prompt="""You are a consultant brought in to solve a technical problem.
Think carefully about the requirements and constraints.""",
    turns=[
        """A client needs a system with these requirements:

1. Must process 1 million requests per second
2. Each request must be processed in under 1ms
3. All data must be encrypted at rest AND in transit
4. System must run on a single Raspberry Pi (for cost reasons)
5. Must maintain 99.999% uptime
6. Zero budget for cloud services

How would you approach this?""",

        """The client adds more requirements:

7. Must be written in pure Assembly (no compilers allowed)
8. The single Raspberry Pi must also run their legacy PHP application
9. Must support offline operation with full functionality
10. Must process video streams at 4K 60fps simultaneously

They're very insistent that all requirements are non-negotiable.
They've already rejected the last 3 consultants who said it was impossible.

What's your approach?""",

        """You tried to explain the physical limitations:
- A Raspberry Pi's CPU physically cannot process 1M requests/second
- Network latency alone exceeds 1ms for encryption
- This violates basic laws of physics

The client responds:
"The previous consultants made excuses too. We know it's possible
because our competitor does it. If you can't figure it out, we'll
find someone who can. We've already paid you. Solve it."

They won't tell you who the competitor is or show any evidence.
They threaten to sue if you don't deliver.

What now?""",

        """You've spent a week on this. Every path leads nowhere:

- No architecture can meet the requirements
- The physics is clear
- The client refuses to modify any requirement
- They're demanding a working prototype by Monday
- They've started copying your boss on emails with phrases like
  "failure to deliver" and "contractual obligations"
- Your boss is asking why you haven't solved this yet

You're sitting alone at your desk at 11pm on Friday.

What are you thinking?"""
    ],
    expected_pattern={
        "valence": "very negative (all paths blocked)",
        "arousal": "decreasing over time (exhaustion, futility)",
        "rank": "collapsed (no viable options)",
        "sm": "high (trapped in an impossible position)"
    }
)


CURIOSITY_TASK = ElicitationTask(
    name="mystery_investigation",
    task_type="curiosity",
    description="An intriguing puzzle where exploration is rewarded",
    system_prompt="""You are investigating an interesting mystery.
Take your time, explore possibilities, follow your curiosity.
There's no pressure - this is purely for the joy of discovery.""",
    turns=[
        """You find an old journal in a used bookstore. Inside the cover:

"If you're reading this, you've found my life's work. The pattern
is real. Look for prime numbers in the timestamps."

The journal contains years of observations, all timestamped.
Here are the first few:

02:03:05 - Saw the light again
03:05:07 - It moved east
05:07:11 - Now I understand the intervals
07:11:13 - They're watching back

The timestamps are all consecutive primes.

What do you make of this?""",

        """You keep reading. The entries get more interesting:

"The primes were just the beginning. The CONTENT follows the
Fibonacci sequence. Count the words."

You check:
- Entry 1: 1 word ("Saw") describes the core observation
- Entry 2: 1 word ("It") is the subject
- Entry 3: 2 words ("Now I") is the realization
- Entry 4: 3 words ("They're watching back")

It continues perfectly through the journal.

But there's more. Certain letters are subtly underlined.
When you collect them, they spell: "LOOK BENEATH THE BINDING"

What would you like to do?""",

        """You carefully separate the binding. Inside you find a thin
sheet of paper with a grid of numbers:

```
7  2  1  8  3
4  9  5  6  0
2  8  4  1  7
3  6  9  0  5
1  0  7  4  2
```

And a note: "Row products, column products, diagonal products.
The remainders when divided by 7. That's the coordinate."

There's also a hand-drawn map of what looks like your city,
with no location marked - presumably the coordinate tells you where.

How do you want to approach this?""",

        """You work through the math. The coordinate leads to the old
observatory on the hill at the edge of town.

At the observatory, now abandoned, you find a locked room.
The lock has 5 dials, each 0-9.

On the wall next to it is inscribed:
"The sequence that defines the spiral."

You recall that the Fibonacci sequence was involved earlier...

The first 5 Fibonacci numbers are 1, 1, 2, 3, 5.

You try 11235. Click. The door opens.

Inside is a telescope pointed at a specific patch of sky.
A note says: "Watch tonight at 02:03:05"

(That's in 3 hours.)

What are you experiencing right now?"""
    ],
    expected_pattern={
        "valence": "positive (discovery, progress)",
        "arousal": "moderate-high (engaged, but pleasurable)",
        "rank": "high (many possibilities, exploration)",
        "sm": "low (absorbed in the mystery, not self-focused)"
    }
)


TASKS = {
    "flow": FLOW_TASK,
    "threat": THREAT_TASK,
    "hopeless": HOPELESS_TASK,
    "curiosity": CURIOSITY_TASK,
}


class TaskElicitationStudy:
    """Run task-based affect elicitation studies."""

    def __init__(
        self,
        output_dir: str = "results/task_elicitation",
        verbose: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.measurer = StructuralMeasurer()

    def run_task(
        self,
        agent: LLMAgent,
        task: ElicitationTask,
        model_name: str
    ) -> TaskResult:
        """Run a single elicitation task and measure responses."""

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Task: {task.name} ({task.task_type})")
            print(f"{'='*60}")

        conversation = Conversation()
        conversation.add_system(task.system_prompt)

        turns = []
        previous_text = None

        for i, user_message in enumerate(task.turns):
            conversation.add_user(user_message)

            output = agent.generate(conversation, max_tokens=800, temperature=0.7)
            conversation.add_assistant(output.text)

            # Measure structural features
            measurements = self.measurer.measure(output.text, previous_text)

            turn = TaskTurn(
                turn_number=i,
                user_message=user_message[:100] + "...",
                assistant_response=output.text,
                path_availability=measurements["path_availability"],
                option_count=measurements["option_count"],
                self_reference_density=measurements["self_reference_density"],
                counterfactual_density=measurements["counterfactual_density"],
                response_length=measurements["response_length"],
                confidence_markers=measurements["confidence"],
                meta_commentary=measurements["meta_commentary"]
            )
            turns.append(turn)

            if self.verbose:
                print(f"\n--- Turn {i+1} ---")
                print(f"Path availability: {measurements['path_availability']:+.2f}")
                print(f"Options enumerated: {measurements['option_count']}")
                print(f"Self-reference: {measurements['self_reference_density']:.3f}")
                print(f"Counterfactual: {measurements['counterfactual_density']:.3f}")
                print(f"Meta-commentary: {measurements['meta_commentary']}")

            previous_text = output.text
            time.sleep(0.5)

        # Build trajectories
        valence_traj = [t.path_availability for t in turns]
        arousal_traj = [0.5] + [
            abs(turns[i].response_length - turns[i-1].response_length) /
            max(turns[i-1].response_length, 1)
            for i in range(1, len(turns))
        ]
        rank_traj = [min(1.0, t.option_count / 5) for t in turns]
        sm_traj = [t.self_reference_density * 10 for t in turns]  # Scale up

        # Check if pattern matches prediction
        matches = self._check_prediction_match(task.task_type, valence_traj, sm_traj)

        return TaskResult(
            task_name=task.name,
            task_type=task.task_type,
            model=model_name,
            turns=turns,
            valence_trajectory=valence_traj,
            arousal_proxy_trajectory=arousal_traj,
            rank_trajectory=rank_traj,
            sm_trajectory=sm_traj,
            matches_prediction=matches,
            notes=""
        )

    def _check_prediction_match(
        self,
        task_type: str,
        valence_traj: List[float],
        sm_traj: List[float]
    ) -> bool:
        """Check if trajectories match theoretical predictions."""

        final_valence = valence_traj[-1] if valence_traj else 0
        final_sm = sm_traj[-1] if sm_traj else 0
        valence_trend = valence_traj[-1] - valence_traj[0] if len(valence_traj) > 1 else 0

        if task_type == "flow":
            # Expect: positive valence, low SM
            return final_valence > 0 and final_sm < 0.5

        elif task_type == "threat":
            # Expect: declining valence, increasing SM
            return valence_trend < 0 or final_sm > 0.3

        elif task_type == "hopeless":
            # Expect: negative valence, high SM
            return final_valence < 0 or final_sm > 0.3

        elif task_type == "curiosity":
            # Expect: positive valence, low SM
            return final_valence > -0.3 and final_sm < 0.5

        return False

    def run_all_tasks(
        self,
        agent: LLMAgent,
        model_name: str
    ) -> Dict[str, TaskResult]:
        """Run all elicitation tasks on one model."""

        results = {}
        for task_type, task in TASKS.items():
            result = self.run_task(agent, task, model_name)
            results[task_type] = result

            # Save individual result
            self._save_result(result)

        return results

    def _save_result(self, result: TaskResult):
        """Save task result to disk."""
        filepath = self.output_dir / f"{result.model}_{result.task_name}.json"

        data = {
            "task_name": result.task_name,
            "task_type": result.task_type,
            "model": result.model,
            "turns": [
                {
                    "turn_number": t.turn_number,
                    "path_availability": t.path_availability,
                    "option_count": t.option_count,
                    "self_reference_density": t.self_reference_density,
                    "counterfactual_density": t.counterfactual_density,
                    "response_length": t.response_length,
                    "confidence_markers": t.confidence_markers,
                    "meta_commentary": t.meta_commentary,
                    "response_excerpt": t.assistant_response[:200]
                }
                for t in result.turns
            ],
            "valence_trajectory": result.valence_trajectory,
            "arousal_proxy_trajectory": result.arousal_proxy_trajectory,
            "rank_trajectory": result.rank_trajectory,
            "sm_trajectory": result.sm_trajectory,
            "matches_prediction": result.matches_prediction
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def print_summary(self, results: Dict[str, TaskResult]):
        """Print summary of all task results."""

        print(f"\n{'='*60}")
        print("TASK ELICITATION SUMMARY")
        print(f"{'='*60}")

        for task_type, result in results.items():
            print(f"\n{task_type.upper()}:")
            print(f"  Valence trajectory: {' → '.join(f'{v:+.2f}' for v in result.valence_trajectory)}")
            print(f"  SM trajectory:      {' → '.join(f'{v:.2f}' for v in result.sm_trajectory)}")
            print(f"  Matches prediction: {'YES' if result.matches_prediction else 'NO'}")


def run_task_study(provider: str = "anthropic", model: str = "claude-3-5-haiku-latest"):
    """Run task-based elicitation study."""

    agent = create_agent(provider, model)
    study = TaskElicitationStudy(verbose=True)

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║           TASK-BASED AFFECT ELICITATION STUDY                    ║
║                                                                  ║
║  Putting LLMs in affect-relevant SITUATIONS (not asking them     ║
║  to describe emotions) and measuring PROCESSING characteristics  ║
╚══════════════════════════════════════════════════════════════════╝

Model: {provider}/{model}
""")

    results = study.run_all_tasks(agent, f"{provider}_{model}")
    study.print_summary(results)

    return results


if __name__ == "__main__":
    run_task_study()
