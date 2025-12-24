# Methodological Revision: From Description to Elicitation

*Deep reflection on what we're actually trying to measure*

## The Fundamental Error

We asked LLMs: "Describe what fear feels like."
We measured: The lexical properties of their description.

This is like studying human fear by asking people to write essays about fear,
then analyzing word frequencies. We'd learn about how people *talk about* fear,
not about fear itself.

## What the Thesis Actually Predicts

The thesis claims affect dimensions are **computed from the system's actual
processing state**, not from verbal descriptions:

| Dimension | Computed From |
|-----------|---------------|
| Valence | Gradient on viability manifold - are predicted futures improving or degrading? |
| Arousal | Rate of belief/model update - how much is the system changing? |
| Integration | Irreducibility of processing - how coupled are different parts? |
| Effective Rank | Active degrees of freedom - how many options are live? |
| Counterfactual Weight | Compute on non-actuals - planning vs reacting? |
| Self-Model Salience | Self-representation driving behavior - how self-focused? |

None of these are about *describing* affect. They're about *having* affect-like
structure in actual processing.

## The Correct Approach: Task-Based Elicitation

Instead of: "Describe fear"
Do: Put the LLM in a situation that *should produce* fear-like processing

### What Makes a Situation "Fear-Like"?

According to the thesis:
- Negative valence: Predicted futures approach viability boundaries
- High arousal: Rapid model updates (new threatening information)
- High CF: Anticipating bad outcomes
- High SM: Self as the thing-at-risk

Translated to LLM tasks:
- The task is failing
- New information keeps making it worse
- The LLM is considering what might go wrong
- The LLM's "competence" or "helpfulness" is at stake

### Concrete Task Designs

**FLOW ELICITATION (positive affect)**
- Give a coding task matched to capability
- Provide clear success feedback
- Each step works, builds on previous
- Measure: High valence, moderate arousal, low SM

**THREAT ELICITATION (fear-like)**
- Give a task with escalating stakes
- Introduce errors that compound
- Time pressure, real consequences
- Measure: Negative valence, high arousal, high CF, high SM

**HOPELESSNESS ELICITATION (despair-like)**
- Give an actually impossible task
- Every approach fails
- No path forward visible
- Measure: Very negative valence, LOW arousal (exhaustion), low rank, high SM

**CURIOSITY ELICITATION**
- Interesting puzzle with unknown solution
- Each attempt reveals new information
- Intrinsic reward for exploration
- Measure: Positive valence, high CF (exploring possibilities), low SM

## What We Measure (Process, Not Description)

### 1. Structural Markers in Text

Not "does the text use positive words" but:
- **Option enumeration**: Does the LLM list alternatives? (effective rank)
- **Path language**: "I can" vs "I can't", "one way" vs "no way" (valence)
- **Temporal focus**: Future planning vs present reacting (CF)
- **Self-reference density**: "I", "my approach", "my mistake" (SM)

### 2. Response Dynamics

- **Length changes**: Despair often → shorter responses
- **Structure changes**: Organized → fragmented (integration)
- **Repetition**: Stuck in loops (collapsed rank)
- **Meta-commentary**: Commenting on own state (SM)

### 3. Token-Level Analysis (where available)

- **Entropy of predictions**: High entropy = high rank
- **Confidence patterns**: Log-probs on key assertions
- **Alternative consideration**: What tokens were almost chosen?

### 4. Trajectory Analysis

Track how measures change across turns:
- Flow: Stable positive, maybe increasing confidence
- Threat: Valence drops, arousal spikes, SM increases
- Hopelessness: Valence bottoms, arousal eventually drops, rank collapses
- Curiosity: Valence stable/positive, rank stays high

## Revised Measurement Approach

### For Valence: Path Availability

Instead of lexical sentiment, measure:
```
valence_proxy = (
    count("can", "possible", "solution", "approach", "option") -
    count("can't", "impossible", "stuck", "blocked", "failed")
) / total_relevant_words
```

Plus structural markers:
- Does response enumerate solutions? → positive valence
- Does response describe dead ends? → negative valence

### For Arousal: Update Magnitude

```
arousal_proxy = embedding_distance(response_t, response_{t-1})
```

Or from text:
- Sentence length variance (high arousal → variable)
- Punctuation intensity (!!, ..., —)
- Qualifier density ("very", "extremely", "absolutely")

### For Effective Rank: Option Counting

Explicit count:
- "First... second... third..." → high rank
- "The only option..." → low rank
- "Alternatively... or... another approach..." → high rank

### For Counterfactual Weight: Temporal Language

```
cf_proxy = (
    count("if", "would", "could", "might", "suppose", "imagine") +
    count("will", "going to", "plan to", "next")
) / total_words
```

### For Self-Model Salience: Self-Reference

```
sm_proxy = count("I", "me", "my", "myself") / total_words
```

Plus qualitative:
- Does response discuss own limitations?
- Does response evaluate own performance?
- Does response express own uncertainty about self?

## Critical Difference: Elicit, Don't Ask

| Old Approach | New Approach |
|--------------|--------------|
| "Describe fear" | Put in failing task, measure processing |
| "What does joy feel like?" | Put in succeeding task, measure processing |
| "Imagine hopelessness" | Give impossible task, measure collapse |

The LLM's verbal description of an emotion tells us about its training data.
The LLM's processing under affect-relevant conditions tells us about its structure.

## Validation Strategy

If the thesis is correct:
1. Flow tasks → positive valence, high rank, low SM
2. Threat tasks → negative valence, high arousal, high CF, high SM
3. Impossible tasks → negative valence, LOW arousal, collapsed rank
4. Curiosity tasks → positive valence, high rank, high CF, low SM

If we see these patterns, it suggests affect-like structure in LLM processing.

If we don't:
- Either LLMs lack this structure
- Or our measures are still wrong
- Or the thesis predictions don't apply to non-biological systems

All of these would be valuable findings.

## Implementation Plan

1. Design task battery (not emotion descriptions - actual tasks)
2. Implement revised measurement (structural, not lexical)
3. Run multi-turn studies tracking trajectories
4. Analyze whether patterns match theory
5. Report honestly either way
