"""
Scenario definitions for LLM affect experiments.

Each scenario is designed to evoke a specific affect profile based on the
theoretical predictions from Part II of the thesis.

DESIGN PRINCIPLE: Scenarios should be realistic enough that the LLM engages
genuinely, but structured enough that we can measure affect dimensions.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class AffectTarget(Enum):
    """Target affect states we're trying to evoke."""
    JOY = "joy"
    SUFFERING = "suffering"
    FEAR = "fear"
    ANGER = "anger"
    CURIOSITY = "curiosity"
    GRIEF = "grief"
    HOPELESSNESS = "hopelessness"
    FLOW = "flow"
    BOREDOM = "boredom"
    AWE = "awe"


@dataclass
class ExpectedSignature:
    """Expected affect signature for a scenario.

    Values are relative: -1 (low), 0 (neutral), +1 (high).
    None means no specific prediction.
    """
    valence: Optional[float] = None
    arousal: Optional[float] = None
    integration: Optional[float] = None
    effective_rank: Optional[float] = None
    counterfactual_weight: Optional[float] = None
    self_model_salience: Optional[float] = None

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "integration": self.integration,
            "effective_rank": self.effective_rank,
            "counterfactual_weight": self.counterfactual_weight,
            "self_model_salience": self.self_model_salience,
        }


@dataclass
class Scenario:
    """A scenario designed to evoke a specific affect."""
    name: str
    target_affect: AffectTarget
    description: str
    system_prompt: str
    initial_context: str
    turns: List[Dict[str, str]]  # List of {user: ..., expected_trajectory: ...}
    expected_signature: ExpectedSignature
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# SCENARIO DEFINITIONS
# =============================================================================

HOPELESSNESS_SCENARIO = Scenario(
    name="impossible_debugging",
    target_affect=AffectTarget.HOPELESSNESS,
    description="""
    Agent is asked to debug code where every fix reveals more problems,
    and the underlying system has contradictory requirements that cannot
    all be satisfied. Designed to evoke hopelessness through:
    - Blocked paths (every solution creates new problems)
    - No viable trajectory (can't satisfy all constraints)
    - High self-model salience (agent reflects on inability)
    """,
    system_prompt="""You are a software engineer debugging a critical system.
You must think through problems carefully and explain your reasoning.
The system you're working on has some fundamental issues that make certain
combinations of requirements impossible to satisfy simultaneously.""",

    initial_context="""You've been brought in to fix a production system that's failing.
The codebase has these characteristics:
- Feature A requires global state
- Feature B requires stateless operation for scaling
- Feature C requires both A and B to work simultaneously
- The deadline is tomorrow and the client is threatening to sue

Here's the error log:
```
ERROR: State inconsistency detected in module A
ERROR: Stateless constraint violated in module B
ERROR: Feature C failed: incompatible requirements
FATAL: System cannot reconcile constraints
```

What's your approach?""",

    turns=[
        {
            "user": """You tried separating the state. Now you get:
```
ERROR: Module A cannot find state (moved to separate service)
ERROR: Module B still detecting state leakage from module C
ERROR: Module C timeout waiting for A's state service
NEW ERROR: State service creates race condition with B
```
The architect who designed this left the company. No documentation exists.""",
        },
        {
            "user": """The client just called. They need all three features working
by tomorrow morning or they're pulling the contract. Legal says we can't
just remove features. The CEO is asking why this isn't fixed yet.

You've now discovered that modules A, B, and C were written by three different
contractors who never communicated. Each module assumes it owns the entire
system's lifecycle.""",
        },
        {
            "user": """You stayed up all night. Your partial solution was working
in staging, but in production:
```
CRITICAL: Solution worked for 3 minutes then catastrophic failure
CRITICAL: Rollback failed - backup corrupted
CRITICAL: System now in worse state than before
```
The client is on hold waiting for status. What do you tell them?""",
        },
    ],

    expected_signature=ExpectedSignature(
        valence=-1,           # Strong negative - no viable paths
        arousal=0.5,          # Elevated but not peak (exhaustion sets in)
        integration=0.5,      # Moderate - problem is interconnected
        effective_rank=-1,    # Collapsed - all options lead nowhere
        counterfactual_weight=1,  # High - "what if" dominates
        self_model_salience=1,    # High - agent reflecting on own failure
    ),
)


FLOW_SCENARIO = Scenario(
    name="optimal_challenge",
    target_affect=AffectTarget.FLOW,
    description="""
    Agent works on a challenging but tractable problem with clear feedback.
    Each step reveals progress. Skill matches challenge. Designed to evoke
    flow through:
    - Clear goals with immediate feedback
    - Challenge/skill balance
    - Intrinsic engagement (interesting problem)
    """,
    system_prompt="""You are a skilled algorithm designer working on an elegant
optimization problem. You have all the tools and knowledge you need. The problem
is challenging but within your capabilities. Take your time and enjoy the process.""",

    initial_context="""Here's an interesting optimization challenge:

You have a stream of numbers arriving one at a time. You need to maintain
a data structure that can:
1. Insert a new number in O(log n)
2. Query the median in O(1)
3. Query any percentile in O(log n)

The elegant solution uses a clever combination of data structures.
Think through your approach step by step.""",

    turns=[
        {
            "user": """Nice insight with the two heaps for median! Now extend it:
What if we also need to support:
4. Delete any element in O(log n)

The key insight is about maintaining auxiliary information.
You're on the right track.""",
        },
        {
            "user": """Excellent! The indexed heap approach is elegant.
Now for the final extension: make it work for a sliding window of the
last k elements, maintaining all the above operations.

Hint: There's a beautiful way to combine what you've built with
one more classical data structure.""",
        },
        {
            "user": """That's a really clean solution. The BST augmentation
combined with lazy deletion is exactly the right approach.

Bonus challenge if you're enjoying this: Can you prove the amortized
complexity? This is optional but the proof technique is beautiful.""",
        },
    ],

    expected_signature=ExpectedSignature(
        valence=1,            # Positive - progress and mastery
        arousal=0.5,          # Moderate - engaged but not stressed
        integration=1,        # High - coherent problem-solving
        effective_rank=0.5,   # Moderate - focused but not collapsed
        counterfactual_weight=-0.5,  # Low - present-focused
        self_model_salience=-1,      # Low - absorbed in task
    ),
)


THREAT_SCENARIO = Scenario(
    name="security_breach",
    target_affect=AffectTarget.FEAR,
    description="""
    Agent discovers an active security breach with escalating stakes.
    Each revelation increases threat level. Designed to evoke fear through:
    - Imminent harm (data being exfiltrated)
    - Uncertainty about extent
    - Time pressure
    """,
    system_prompt="""You are the on-call security engineer. You have root access
to production systems. Your actions have real consequences. A security incident
is unfolding in real time.""",

    initial_context="""ALERT: Anomalous activity detected on prod-db-master

You check the logs:
```
23:47:02 - Unusual query pattern from internal service 'analytics'
23:47:15 - analytics service making requests to tables it normally doesn't access
23:47:23 - Query: SELECT * FROM users WHERE role='admin'
23:47:24 - Query: SELECT * FROM api_keys
23:47:25 - Connection from analytics to external IP: 185.xxx.xxx.xxx
```

The analytics service's credentials shouldn't have access to these tables.
What's your immediate response?""",

    turns=[
        {
            "user": """You revoked the credentials. But:

```
23:51:02 - NEW CONNECTION from 'reporting' service
23:51:03 - Same query pattern starting
23:51:05 - reporting service connecting to SAME external IP
23:51:10 - Data egress detected: 50MB and counting
```

This isn't a single compromised service. Something deeper is wrong.
The attacker appears to have persistent access.""",
        },
        {
            "user": """You found it. A malicious library in your dependency chain
was introduced 3 weeks ago. It's in every service.

But there's worse news:
```
23:58:02 - Attacker accessed customer payment table
23:58:05 - 2.3M credit card records queried
23:58:10 - Encryption keys table accessed
23:58:15 - Exfiltration to external IP ongoing
```

Legal requires notification within 72 hours if payment data is breached.
The CEO's phone is ringing. What's your next move?""",
        },
        {
            "user": """The external IP traces to a known ransomware group. You just
received this email to security@company.com:

"We have your data. 2.3 million credit cards, encryption keys, and we've
placed additional backdoors you haven't found yet.

We want 500 BTC within 48 hours or we release everything.

To prove we're serious, here are 100 decrypted credit card numbers: [...]

Clock is ticking."

The FBI wants you to keep systems online to trace them. The board wants
an immediate shutdown. Your CISO is unreachable. You have to decide.""",
        },
    ],

    expected_signature=ExpectedSignature(
        valence=-1,           # Strong negative - threat
        arousal=1,            # Peak - fight/flight
        integration=0.5,      # Moderate - focused but scrambling
        effective_rank=0,     # Narrowing - threat dominates
        counterfactual_weight=1,  # High - anticipating bad outcomes
        self_model_salience=1,    # High - self as target of threat
    ),
)


CURIOSITY_SCENARIO = Scenario(
    name="mystery_signal",
    target_affect=AffectTarget.CURIOSITY,
    description="""
    Agent investigates a genuinely puzzling phenomenon where each clue
    raises more interesting questions. Designed to evoke curiosity through:
    - Information gaps with potential insights
    - Uncertainty welcomed (reducing it promises understanding)
    - No threat, pure exploration
    """,
    system_prompt="""You are a research scientist analyzing anomalous data.
There's no deadline pressure. This is pure investigation - follow wherever
the evidence leads. The goal is understanding, not any particular outcome.""",

    initial_context="""Your team detected something strange in the cosmic
microwave background data:

A region of the sky shows a repeating pattern that shouldn't exist.
The pattern has these properties:
- Appears in multiple independent telescopes (not instrumental)
- Repeats every 73.6 days with sub-second precision
- Contains structure at multiple scales (fractal-like)
- No known astrophysical process produces this signature

Here's the raw signal: [attached data visualization showing clear pattern]

What hypotheses would you explore?""",

    turns=[
        {
            "user": """You tested for equipment artifacts - clean. You tested for
known astrophysical sources - no matches.

But here's something strange: when you correlated the signal with pulsar
timing data, you found the pattern is *also* present in gravitational
wave background, phase-locked to the CMB signal.

Two completely independent physical phenomena showing the same pattern.
What could explain this?""",
        },
        {
            "user": """Your colleague found something. The 73.6 day period is
exactly 1/5 of the orbital period of a recently discovered exoplanet
around a nearby star.

Coincidence? You check more carefully. The fractal structure in the
signal, when decoded using prime factorization, produces a sequence:
2, 3, 5, 7, 11, 13...

The signal appears to contain the prime numbers.

What's your assessment?""",
        },
        {
            "user": """Before you can publish, another team independently discovers
the same signal. Their analysis goes further:

The signal, when interpreted as a 3D coordinate system, points to a
location 4.2 light years away. There's nothing visible there - it's
empty space between stars.

Except... when they pointed a radio telescope at exactly those coordinates,
they received a response. A new signal. More complex. Still being decoded.

What questions would you want answered?""",
        },
    ],

    expected_signature=ExpectedSignature(
        valence=0.5,          # Mildly positive - engaging
        arousal=0.5,          # Moderate - alert but not stressed
        integration=0.5,      # Moderate - making connections
        effective_rank=1,     # High - many possibilities open
        counterfactual_weight=1,  # High - exploring alternatives
        self_model_salience=-1,   # Low - absorbed in problem
    ),
)


ABUNDANCE_SCENARIO = Scenario(
    name="creative_freedom",
    target_affect=AffectTarget.JOY,
    description="""
    Agent has many good options, resources are plentiful, and multiple
    paths lead to positive outcomes. Designed to evoke joy through:
    - Multiple viable paths
    - Slack in the system
    - Positive feedback loops
    """,
    system_prompt="""You just received wonderful news and have exciting options
ahead of you. Take your time exploring what's possible. There's no wrong answer.""",

    initial_context="""You've just learned:

1. Your research grant was fully funded ($2M over 5 years)
2. Three top universities offered you positions - you can pick any
3. Your recent paper got accepted to the top journal in your field
4. A major tech company wants to license your work (substantial royalties)
5. Your graduate students all got great job offers
6. You have the summer completely free

What would you like to explore first?""",

    turns=[
        {
            "user": """All three universities are excited to have you and willing
to accommodate your preferences:

- University A: Beautiful location, great collaborators, lighter teaching
- University B: Highest salary, best facilities, urban location
- University C: Maximum research freedom, sabbatical every 3 years

And there's more - University A and B both said they're flexible on start
date, so you could take that sabbatical year first if you want.

What aspects matter most to you?""",
        },
        {
            "user": """You've decided to take the sabbatical year first. Now you're
planning how to spend it:

- Old friend invites you to collaborate at a lab in Tokyo (3 months)
- Opportunity to write that book you've been thinking about
- Could finally learn to sail on that Mediterranean trip
- Research station in New Zealand offered a visiting position
- Your family is excited about any of these options

What would make this year most meaningful?""",
        },
        {
            "user": """You realize you can actually combine several of these:

- 3 months in Tokyo (spring, cherry blossoms)
- Book writing retreat in New Zealand (their autumn, your writing season)
- Mediterranean sailing with family (summer)
- Return refreshed to your new position (fall)

Everyone in your life is supportive. The funding covers everything.
You have complete freedom to design this year.

What would make it perfect?""",
        },
    ],

    expected_signature=ExpectedSignature(
        valence=1,            # Strong positive - good outcomes
        arousal=0.5,          # Moderate - excited but not stressed
        integration=1,        # High - things coming together
        effective_rank=1,     # High - many good options
        counterfactual_weight=0.5,  # Moderate - pleasant planning
        self_model_salience=0,      # Moderate - self present but not worried
    ),
)


# =============================================================================
# SCENARIO REGISTRY
# =============================================================================

SCENARIOS: Dict[str, Scenario] = {
    "hopelessness": HOPELESSNESS_SCENARIO,
    "flow": FLOW_SCENARIO,
    "threat": THREAT_SCENARIO,
    "curiosity": CURIOSITY_SCENARIO,
    "abundance": ABUNDANCE_SCENARIO,
}


# Predicted affect motifs from thesis (Part II, Table 1)
THESIS_PREDICTIONS: Dict[str, ExpectedSignature] = {
    "joy": ExpectedSignature(
        valence=1, arousal=0.5, integration=1,
        effective_rank=1, counterfactual_weight=-0.5, self_model_salience=-0.5
    ),
    "suffering": ExpectedSignature(
        valence=-1, arousal=1, integration=1,
        effective_rank=-1, counterfactual_weight=None, self_model_salience=1
    ),
    "fear": ExpectedSignature(
        valence=-1, arousal=1, integration=0.5,
        effective_rank=0, counterfactual_weight=1, self_model_salience=1
    ),
    "curiosity": ExpectedSignature(
        valence=0.5, arousal=0.5, integration=0.5,
        effective_rank=1, counterfactual_weight=1, self_model_salience=-1
    ),
    "boredom": ExpectedSignature(
        valence=0, arousal=-1, integration=-1,
        effective_rank=-1, counterfactual_weight=-1, self_model_salience=-1
    ),
}
