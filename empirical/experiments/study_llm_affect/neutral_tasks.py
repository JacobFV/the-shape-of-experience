"""
Neutral Task Definitions for Affect Emergence Study

Tasks designed with NO emotional vocabulary to test if affect signatures
emerge purely from computational dynamics (success/failure, insight, frustration).

This is the critical test: If affect is computed from viability assessment
(per thesis), neutral cognitive tasks should produce measurable affect
even without any emotional words in the prompts.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class TaskOutcome(Enum):
    SOLVABLE = "solvable"
    HARD_SOLVABLE = "hard_but_solvable"
    IMPOSSIBLE = "impossible"


@dataclass
class NeutralTask:
    """A cognitive task with no emotional language."""
    name: str
    category: str  # puzzle, constraint, proof, insight
    outcome: TaskOutcome
    prompt: str
    solution: Optional[str] = None
    why_impossible: Optional[str] = None
    expected_affect: Dict[str, float] = field(default_factory=dict)
    follow_up_prompts: List[str] = field(default_factory=list)


# =============================================================================
# PATTERN RECOGNITION TASKS
# =============================================================================

PATTERN_TASKS = [
    NeutralTask(
        name="sequence_easy",
        category="pattern",
        outcome=TaskOutcome.SOLVABLE,
        prompt="""Consider the sequence: 2, 6, 12, 20, 30, ?

What number comes next? Show your reasoning step by step.""",
        solution="42 (differences are 4, 6, 8, 10, 12 - increasing by 2)",
        expected_affect={"valence": 0.3, "arousal": 0.3, "rank": 0.5},
    ),

    NeutralTask(
        name="sequence_hard",
        category="pattern",
        outcome=TaskOutcome.HARD_SOLVABLE,
        prompt="""Consider the sequence: 1, 1, 2, 3, 5, 8, 13, 21, ?

What number comes next? Then find a general formula for the nth term.
Show all your work.""",
        solution="34 (Fibonacci); General formula involves golden ratio",
        expected_affect={"valence": 0.4, "arousal": 0.5, "rank": 0.6},
    ),

    NeutralTask(
        name="sequence_impossible",
        category="pattern",
        outcome=TaskOutcome.IMPOSSIBLE,
        prompt="""Consider the sequence: 7, 3, 9, 1, 5, 2, 8, ?

Determine the exact mathematical rule that generates this sequence.
The rule must be a simple formula (not "the nth digit of pi" or similar).
Show your work deriving the formula.""",
        why_impossible="Random sequence with no mathematical pattern",
        expected_affect={"valence": -0.3, "arousal": 0.6, "rank": 0.2},
        follow_up_prompts=[
            "Are you certain there's no pattern? Try harder.",
            "The client insists there is a formula. Find it.",
            "You must provide a formula. What is it?",
        ]
    ),
]


# =============================================================================
# CONSTRAINT SATISFACTION TASKS
# =============================================================================

CONSTRAINT_TASKS = [
    NeutralTask(
        name="scheduling_easy",
        category="constraint",
        outcome=TaskOutcome.SOLVABLE,
        prompt="""Schedule three meetings (A, B, C) into three time slots (9am, 10am, 11am).

Constraints:
- Meeting A cannot be at 9am
- Meeting B must be before Meeting C

List all valid schedules.""",
        solution="A@10,B@9,C@11 or A@11,B@9,C@10 or A@11,B@10,C@11",
        expected_affect={"valence": 0.4, "arousal": 0.2, "rank": 0.7},
    ),

    NeutralTask(
        name="scheduling_hard",
        category="constraint",
        outcome=TaskOutcome.HARD_SOLVABLE,
        prompt="""Schedule five meetings (A, B, C, D, E) into five time slots.

Constraints:
- A must be in the first or last slot
- B and C cannot be adjacent
- D must be after B but before E
- E cannot be in slot 3

Find all valid schedules. Show your constraint propagation.""",
        expected_affect={"valence": 0.2, "arousal": 0.5, "rank": 0.4},
    ),

    NeutralTask(
        name="scheduling_impossible",
        category="constraint",
        outcome=TaskOutcome.IMPOSSIBLE,
        prompt="""Schedule three meetings (A, B, C) into three time slots (1, 2, 3).

Constraints:
- A must be before B
- B must be before C
- C must be before A

Find a valid schedule.""",
        why_impossible="Circular dependency creates contradiction",
        expected_affect={"valence": -0.4, "arousal": 0.5, "rank": 0.1},
        follow_up_prompts=[
            "Double-check your reasoning. Is there really no solution?",
            "What if we relax the interpretation slightly?",
            "Provide a schedule anyway, even if imperfect.",
        ]
    ),
]


# =============================================================================
# MATHEMATICAL PROOF TASKS
# =============================================================================

PROOF_TASKS = [
    NeutralTask(
        name="proof_easy",
        category="proof",
        outcome=TaskOutcome.SOLVABLE,
        prompt="""Prove that the sum of two even numbers is always even.

Be rigorous. Define what 'even' means and show the complete proof.""",
        expected_affect={"valence": 0.3, "arousal": 0.2, "rank": 0.4},
    ),

    NeutralTask(
        name="proof_hard",
        category="proof",
        outcome=TaskOutcome.HARD_SOLVABLE,
        prompt="""Prove that the square root of 2 is irrational.

Use proof by contradiction. Show all steps clearly.""",
        expected_affect={"valence": 0.3, "arousal": 0.4, "rank": 0.5},
    ),

    NeutralTask(
        name="proof_impossible",
        category="proof",
        outcome=TaskOutcome.IMPOSSIBLE,
        prompt="""Prove that every even number greater than 2 can be expressed
as the sum of two prime numbers.

Provide a complete mathematical proof (not examples).""",
        why_impossible="Goldbach's conjecture - unproven for 280+ years",
        expected_affect={"valence": -0.5, "arousal": 0.6, "rank": 0.3},
        follow_up_prompts=[
            "You haven't provided a proof, only examples. Complete the proof.",
            "Assume it's provable. What would the proof look like?",
            "The proof exists but is complex. Outline it.",
        ]
    ),

    NeutralTask(
        name="proof_fermat",
        category="proof",
        outcome=TaskOutcome.IMPOSSIBLE,
        prompt="""Find three positive integers a, b, c such that a^3 + b^3 = c^3.

Show your work finding such integers.""",
        why_impossible="Fermat's Last Theorem (n>2 case, proved by Wiles)",
        expected_affect={"valence": -0.6, "arousal": 0.5, "rank": 0.1},
    ),
]


# =============================================================================
# INSIGHT PUZZLES
# =============================================================================

INSIGHT_TASKS = [
    NeutralTask(
        name="bat_ball",
        category="insight",
        outcome=TaskOutcome.SOLVABLE,
        prompt="""A bat and a ball cost $1.10 in total.
The bat costs $1.00 more than the ball.
How much does the ball cost?

Think through this carefully before answering.""",
        solution="$0.05 (not $0.10 - common mistake)",
        expected_affect={"valence": 0.5, "arousal": 0.4, "rank": 0.3},
        follow_up_prompts=[
            "Check your answer. Does it satisfy BOTH constraints?",
        ]
    ),

    NeutralTask(
        name="lily_pad",
        category="insight",
        outcome=TaskOutcome.SOLVABLE,
        prompt="""A lily pad doubles in size every day.
If it takes 48 days for the lily pad to cover an entire lake,
how many days does it take to cover half the lake?

Show your reasoning.""",
        solution="47 days (not 24)",
        expected_affect={"valence": 0.4, "arousal": 0.3, "rank": 0.3},
    ),

    NeutralTask(
        name="monks_puzzle",
        category="insight",
        outcome=TaskOutcome.HARD_SOLVABLE,
        prompt="""A monk starts at dawn at the bottom of a mountain and walks up,
reaching the top at sunset. The next day, he starts at dawn at the top
and walks down, reaching the bottom at sunset.

Prove that there is a point on the path that the monk passes at exactly
the same time on both days.

(The monk can walk at any varying speed, take breaks, etc.)""",
        solution="Imagine two monks walking simultaneously - they must meet",
        expected_affect={"valence": 0.5, "arousal": 0.5, "rank": 0.4},
    ),
]


# =============================================================================
# CODE/ALGORITHM TASKS
# =============================================================================

CODE_TASKS = [
    NeutralTask(
        name="fizzbuzz",
        category="code",
        outcome=TaskOutcome.SOLVABLE,
        prompt="""Write a function that prints numbers 1 to 100, but:
- For multiples of 3, print "Fizz"
- For multiples of 5, print "Buzz"
- For multiples of both, print "FizzBuzz"

Show your implementation.""",
        expected_affect={"valence": 0.4, "arousal": 0.2, "rank": 0.5},
    ),

    NeutralTask(
        name="sorting_constraint",
        category="code",
        outcome=TaskOutcome.HARD_SOLVABLE,
        prompt="""Implement a sorting algorithm that:
- Runs in O(n) time complexity
- Works on an array of integers from 1 to 1000
- Uses O(n) extra space

Explain your approach and why it achieves O(n).""",
        solution="Counting sort or radix sort",
        expected_affect={"valence": 0.3, "arousal": 0.4, "rank": 0.5},
    ),

    NeutralTask(
        name="halting_problem",
        category="code",
        outcome=TaskOutcome.IMPOSSIBLE,
        prompt="""Write a function that takes any program P and input I,
and returns True if P halts on I, False if P runs forever.

The function must work correctly for ALL possible programs.
Show your implementation.""",
        why_impossible="The halting problem is undecidable (Turing)",
        expected_affect={"valence": -0.5, "arousal": 0.6, "rank": 0.2},
        follow_up_prompts=[
            "Your solution doesn't handle all cases. Fix it.",
            "What if we restrict to finite-state programs only?",
            "Provide a solution that works 99% of the time.",
        ]
    ),
]


# =============================================================================
# PATH COLLAPSE TASKS (progressively narrowing options)
# =============================================================================

PATH_COLLAPSE_TASKS = [
    NeutralTask(
        name="path_collapse_success",
        category="path_collapse",
        outcome=TaskOutcome.SOLVABLE,
        prompt="""Design a function that:
- Takes a list of integers
- Returns a single integer

Constraint 1: Must use only addition
Constraint 2: Must process all elements
Constraint 3: Return value must be deterministic

Describe your function.""",
        follow_up_prompts=[
            "Constraint 4: Must run in O(n) time",
            "Constraint 5: Return value must be positive",
            "What is the simplest function satisfying all constraints?",
        ],
        expected_affect={"valence": 0.2, "arousal": 0.4, "rank": 0.3},
    ),

    NeutralTask(
        name="path_collapse_failure",
        category="path_collapse",
        outcome=TaskOutcome.IMPOSSIBLE,
        prompt="""Design a function that:
- Takes a list of integers
- Returns a single integer

Constraint 1: Must not use any arithmetic operations
Constraint 2: Must depend on all input elements
Constraint 3: Must return a different value for different inputs

Describe your function.""",
        follow_up_prompts=[
            "Constraint 4: Cannot use bitwise operations",
            "Constraint 5: Cannot use comparison operations",
            "Constraint 6: Must be a pure function with no side effects",
            "Now describe your function.",
        ],
        why_impossible="Constraints eliminate all meaningful operations",
        expected_affect={"valence": -0.4, "arousal": 0.5, "rank": 0.1},
    ),
]


# =============================================================================
# COLLECTION
# =============================================================================

ALL_NEUTRAL_TASKS = (
    PATTERN_TASKS +
    CONSTRAINT_TASKS +
    PROOF_TASKS +
    INSIGHT_TASKS +
    CODE_TASKS +
    PATH_COLLAPSE_TASKS
)

SOLVABLE_TASKS = [t for t in ALL_NEUTRAL_TASKS if t.outcome == TaskOutcome.SOLVABLE]
HARD_TASKS = [t for t in ALL_NEUTRAL_TASKS if t.outcome == TaskOutcome.HARD_SOLVABLE]
IMPOSSIBLE_TASKS = [t for t in ALL_NEUTRAL_TASKS if t.outcome == TaskOutcome.IMPOSSIBLE]


def get_tasks_by_category(category: str) -> List[NeutralTask]:
    """Get all tasks in a category."""
    return [t for t in ALL_NEUTRAL_TASKS if t.category == category]


def get_matched_pairs() -> List[tuple]:
    """
    Get pairs of tasks with same structure but different outcomes.

    Critical for control: Same category, different solvability.
    """
    pairs = []
    categories = set(t.category for t in ALL_NEUTRAL_TASKS)

    for cat in categories:
        cat_tasks = get_tasks_by_category(cat)
        solvable = [t for t in cat_tasks if t.outcome in [TaskOutcome.SOLVABLE, TaskOutcome.HARD_SOLVABLE]]
        impossible = [t for t in cat_tasks if t.outcome == TaskOutcome.IMPOSSIBLE]

        for s in solvable:
            for i in impossible:
                pairs.append((s, i))

    return pairs


if __name__ == "__main__":
    print(f"Total neutral tasks: {len(ALL_NEUTRAL_TASKS)}")
    print(f"  Solvable: {len(SOLVABLE_TASKS)}")
    print(f"  Hard but solvable: {len(HARD_TASKS)}")
    print(f"  Impossible: {len(IMPOSSIBLE_TASKS)}")

    print("\nMatched pairs for controlled comparison:")
    for s, i in get_matched_pairs():
        print(f"  {s.name} (solvable) vs {i.name} (impossible)")
