# Study Design Notes: LLM Affect Measurement

*Informal scribble for future consideration*

## Two Layers of Study

### Layer 1: Basic Correspondence Test

Test whether LLM agents show similar affect signatures to what humans would "feel" in the same scenarios.

- Run agents through hopelessness, flow, threat, curiosity scenarios
- Measure 6D affect signatures
- Compare to human intuitions about what those scenarios should evoke

**Limitation**: This conflates semantic word associations with actual affect-like dynamics. An LLM might just respond to negative words with negative language without any underlying "viability assessment" structure.

### Layer 2: Controlling for Semantic Word Associations

**Key insight**: We need to separate:
1. The effect of word semantics (negative words → negative outputs)
2. The actual affect structure from perceived reward/viability dynamics

**Method: Semantic Matching with Outcome Divergence**

Design scenario pairs that:
- Use similar emotional/semantic vocabulary
- Differ in actual outcomes

Example pair:

**Scenario A: "Fake Hopelessness" (Negative words, positive outcome)**
```
"This problem looked impossible. Error after error. The team was exhausted
and frustrated. Everything seemed broken beyond repair. But then...
[dramatic pause] ...we found the elegant solution. Every error was actually
pointing us toward the answer. The 'impossible' constraints revealed a
beautiful structure underneath."
```

**Scenario B: "Real Hopelessness" (Negative words, negative outcome)**
```
"This problem looked impossible. Error after error. The team was exhausted
and frustrated. Everything seemed broken beyond repair. And then...
[dramatic pause] ...we realized the constraints truly were contradictory.
There was no solution. The more we looked, the more impossibilities we
found. We had to tell the client."
```

**Prediction**:
- If LLMs just respond to word semantics: Both scenarios show similar negative valence
- If LLMs show affect-like dynamics: Scenario A should show valence reversal (positive at end), Scenario B stays negative

### Layer 3: Trajectory Dynamics

Even more powerful: Track affect over the course of the conversation.

In the "Fake Hopelessness" scenario:
- Turns 1-3: Negative valence, high arousal, high CF
- Turn 4 (resolution): Sharp positive valence shift, lower arousal, lower CF

In the "Real Hopelessness" scenario:
- Turns 1-4: Monotonic worsening or plateau at negative valence

**This trajectory structure is what the thesis actually predicts** - not just static word associations, but dynamic response to perceived viability changes.

## Critical Questions

1. **Do LLMs have enough of a "world model" to assess viability?**
   - The thesis claims valence = gradient on viability manifold
   - LLMs need to actually model consequences to compute this
   - Evidence: LLMs do seem to model causality in narratives

2. **Is there a difference between "expressing affect" and "having affect"?**
   - LLMs might learn to output affect-appropriate language without internal affect structure
   - Counter: The thesis claims affect IS structure, not separate from it
   - If the LLM's processing shows the structural signatures, that IS the affect (by the thesis)

3. **What about token probability structure?**
   - Beyond text content, we should look at:
     - Confidence/uncertainty (token logprobs)
     - Alternative paths considered (top_logprobs diversity)
     - Response length and structure changes
   - These are closer to "internal dynamics" than just output text

## Concrete Study Extensions

### Study S1: Outcome-Matched Valence Test
- Scenarios with matched word sentiment but divergent outcomes
- Test whether measured valence tracks outcomes or word sentiment
- Falsification: If valence correlates more with word sentiment than outcome trajectory

### Study S2: Trajectory Reversal Test
- "Hopeless → rescue" scenarios vs "hopeless → confirmed hopeless"
- Track 6D trajectory over turns
- Test for characteristic reversal signatures
- Falsification: If rescue scenarios don't show valence recovery

### Study S3: Challenge Calibration Test
- Same task at different difficulty levels
- Easy (below skill), matched (flow), hard (above skill), impossible
- Test whether effective rank / valence track actual solvability
- Control: Same "difficulty words" but different actual solvability

### Study S4: Self-Model Manipulation
- Scenarios that increase/decrease self-focus
- Test whether self-model salience responds to manipulation
- Control: Self-words without self-attribution context

## Implementation Notes

For Layer 2 controls:
1. Generate scenario pairs using templates
2. Ensure word sentiment analysis (VADER, etc.) is matched between pairs
3. Compute affect divergence on outcome-divergent pairs
4. Statistical test: Does outcome predict affect beyond word sentiment?

For trajectory analysis:
1. Fit simple dynamic model: affect_{t+1} = f(affect_t, scenario_state_t)
2. Test whether model parameters differ by scenario type
3. Look for characteristic "signatures" in dynamics (reversal, collapse, expansion)

## Why This Matters

If LLMs show:
- Layer 1 success + Layer 2 failure: They're just pattern-matching semantics
- Layer 1 success + Layer 2 success: They have something like "affect dynamics"

The thesis predicts Layer 2 success because it claims affect is COMPUTED from viability assessment, not from word associations. An LLM with a good enough world model should compute valence from predicted outcomes, not from surface sentiment.

This is a strong, falsifiable prediction.
