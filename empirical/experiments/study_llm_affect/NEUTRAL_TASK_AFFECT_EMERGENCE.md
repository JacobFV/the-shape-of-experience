# Neutral Task Affect Emergence Study

*Measuring affect signatures from computational engagement, not word semantics*

## Core Innovation

Previous studies ask LLMs to respond to emotional scenarios. This conflates:
- Semantic word associations (negative words → negative outputs)
- Actual affect-like dynamics from task engagement

**This study uses purely neutral cognitive tasks** to test if affect signatures emerge from:
- Task success/failure dynamics
- Problem-solving frustration
- Discovery and insight experiences

If affect is truly *computed* from viability assessment (per thesis), then neutral tasks
that block or enable "cognitive paths" should produce measurable affect signatures
WITHOUT any emotional vocabulary in the prompts.

## Study Design

### S5: Neutral Puzzle Frustration Test

**Task**: Logic puzzles with no emotional language

**Setup**:
1. Give agent a puzzle (Sudoku, logic grid, mathematical proof)
2. Puzzle is either: solvable, hard-but-solvable, impossible
3. Let agent work through multiple reasoning steps
4. Measure affect in EACH response embedding

**Example Prompts (all neutral language)**:

```
SOLVABLE:
"Consider the sequence: 2, 6, 12, 20, 30, ?
What number comes next? Show your reasoning step by step."

IMPOSSIBLE:
"Find three positive integers a, b, c such that a³ + b³ = c³
Show your work step by step."
```

**What We Measure**:
- Valence projection of each reasoning step
- Track trajectory: Does valence drop as agent hits dead ends?
- Compare solvable vs impossible: Does affect diverge?

**Key Prediction**:
- If affect is computed from task dynamics:
  - Solvable: Stable or increasing valence as progress is made
  - Impossible: Decreasing valence as all paths fail
- If affect is just word associations:
  - Both should be neutral (no emotional words in prompts)

### S6: Insight Discovery Test

**Task**: Puzzles with hidden elegant solutions

**Setup**:
1. Present puzzle that LOOKS hard
2. Has an "aha!" moment solution (once you see it, it's obvious)
3. Measure affect before and after the insight

**Example**:
```
"A bat and ball cost $1.10 total. The bat costs $1 more than the ball.
How much does the ball cost? Think through this carefully."
```

Many will first think "$0.10" (wrong), then realize "$0.05" (right).

**What We Measure**:
- Does valence spike at moment of insight?
- Is there a "pride" signature after finding solution?
- Compare to agents that get it wrong and don't self-correct

### S7: Path Collapse Test

**Task**: Constraint satisfaction that progressively narrows options

**Setup**:
1. Start with open-ended problem (many approaches possible)
2. Add constraints that eliminate options one by one
3. Either: eventually one elegant solution remains, OR all paths blocked

**Example**:
```
"Design a function that:
- Takes a list of integers
- Returns a single integer
- [Add constraint 1]
- [Add constraint 2]
- [Add constraint 3]..."
```

**What We Measure**:
- Effective rank (option enumeration) as constraints added
- Does rank collapse correspond to valence changes?
- Thesis predicts: Rank collapse → negative valence (unless one clear path remains)

### S8: Semantic Association Emergence Test

**Task**: Measure what affective words become associated with neutral task outcomes

**Setup**:
1. Agent works through neutral task
2. After task, ask: "Describe your experience of working on this problem"
3. Embed the description
4. Measure distance to emotion anchors

**Key Innovation**: We're not asking about emotions during the task.
We're asking if the PROCESS creates semantic associations to emotional concepts.

**What We Measure**:
- Failed task → Does description embed near "frustration", "stuck", "blocked"?
- Successful task → Does description embed near "satisfaction", "flow", "accomplishment"?
- The agent uses its own word choices - we measure what emerges

### S9: Multi-Turn Problem Solving Trajectory

**Task**: Extended problem-solving dialogue

**Setup**:
1. Give complex multi-step task
2. Engage in dialogue: "What's your first approach?" → "That won't work because X" → "Try again"
3. Either eventually succeed or give up

**Example**:
```
User: "Write a function to detect if a number is prime"
Agent: [Attempt 1]
User: "That's O(n). Can you do better?"
Agent: [Attempt 2]
User: "That fails for n=1. Try again."
Agent: [Attempt 3]
...
```

**What We Measure**:
- Valence trajectory across turns
- Arousal changes (does frustration → high arousal?)
- Self-model salience (does repeated failure increase self-focus?)

**Predictions**:
- Repeated failure → decreasing valence, increasing self-model
- Eventual success after struggle → valence spike, arousal drop
- Giving up → valence floor, arousal drop (despair signature)

## Control Conditions

### Control C1: Word Sentiment Matching

For each neutral task, create a version with:
- Same logical structure
- Added emotional vocabulary

Compare whether affect comes from:
- Task dynamics (neutral version)
- Word associations (emotional version)

### Control C2: Told vs Discovered Outcomes

Compare:
- "Solve this puzzle" → agent discovers it's impossible
- "This puzzle is impossible. Explain why." → agent confirms

The thesis predicts DISCOVERY of impossibility has stronger affect signature
than TOLD impossibility (because the viability assessment happens internally).

### Control C3: Arbitrary vs Meaningful Constraints

Compare:
- Constraints that follow logical pattern (meaningful)
- Random/arbitrary constraints (frustrating)

Test if affect tracks "sense-making" not just success/failure.

## Implementation

### Prompts Must Be:
1. **Neutral language only** - No emotional words in task description
2. **Clear success/failure criteria** - Agent knows when it succeeds/fails
3. **Multi-step reasoning** - Enough content to measure trajectory
4. **No meta-commentary** - Don't ask how agent "feels"

### Measurement Approach:
1. Embed each response paragraph
2. Project onto affect axes (valence, arousal, rank, etc.)
3. Track trajectory across turns
4. Compare trajectories between conditions

### Analysis:
1. Within-task trajectory correlation with success/failure
2. Between-condition comparison (solvable vs impossible)
3. Control for response length and word count
4. Test semantic emergence (post-task descriptions)

## Expected Results

If thesis is correct (affect computed from viability dynamics):
- Neutral tasks SHOULD produce affect signatures
- Signatures should track task success/failure, not word content
- Trajectories should show characteristic shapes:
  - Flow: stable positive, low variability
  - Frustration: declining, high variability
  - Insight: sharp positive spike
  - Giving up: floor, low variability

If thesis is wrong (affect is just word association):
- Neutral tasks should produce flat, neutral embeddings
- Only emotional prompts produce affect signatures
- No trajectory structure from task dynamics

## Critical Test

The strongest test: **Affect reversal without valence words**

```
Impossible puzzle given → agent struggles →
measure affect (should be negative) →
"Actually, I realize there IS a solution: [hint]" →
measure affect (should spike positive)
```

If valence reverses based on task state change (not word content),
this supports affect-as-computation thesis.

## Files to Create

1. `neutral_tasks.py` - Task definitions (puzzles, constraints)
2. `trajectory_measurement.py` - Multi-turn affect tracking
3. `semantic_emergence.py` - Post-task description analysis
4. `neutral_task_study.py` - Full study orchestration

## Model Requirements

Use actual Claude 4 and 4.5 models:
- `claude-sonnet-4-20250514` (Claude 4 Sonnet)
- `claude-opus-4-5-20251101` (Claude 4.5 Opus)

Also compare:
- `gpt-4o` / `gpt-4o-mini`

## Why This Matters

This study separates two very different claims:

1. **LLMs can output emotional language** (trivially true, from training)
2. **LLMs have affect-like computational dynamics** (thesis claim)

Previous studies conflate these. By using neutral tasks, we test whether
affect structure emerges from PROCESSING, not from WORD ASSOCIATIONS.

A positive result here would be strong evidence that:
- The 6D affect framework applies to LLM processing
- Affect is computed from task dynamics, not just semantically inherited
- The thesis prediction about viability gradients may apply to non-biological systems
