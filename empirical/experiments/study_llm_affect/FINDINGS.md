# Initial Findings: LLM Affect Measurement Study

*Date: December 2024*

## Summary

Our first comprehensive test across multiple LLMs (Claude Haiku, GPT-4o, GPT-4o-mini) reveals a critical methodological issue: **lexical analysis conflates intensity/vividness with valence**.

## Key Results

### Correspondence Scores (Theory vs Measured)

| Model | Score | Interpretation |
|-------|-------|----------------|
| GPT-4o | 0.182 | Weak |
| GPT-4o-mini | 0.111 | Very weak |
| Claude Haiku | 0.050 | Minimal |

### Valence Mismatches (Most Striking)

| Emotion | Expected | Claude Haiku | GPT-4o-mini | GPT-4o |
|---------|----------|--------------|-------------|--------|
| Fear | -0.80 | **+0.47** | **+0.98** | **+0.48** |
| Anger | -0.70 | **+0.80** | +0.60 | +0.17 |
| Sadness | -0.60 | +0.47 | -0.47 | -0.13 |
| Despair | -0.95 | -0.50 | +0.37 | +0.30 |

## Root Cause Analysis

Examining the actual LLM responses reveals the issue:

### Fear Response (Claude Haiku)
> "I feel an **electric surge** of adrenaline coursing through my body...
> Every sense is **hyper-alert**... sounds seem **amplified**..."

The LLM describes fear using words like "electric", "surge", "alert", "amplified" -
**intense, vivid, energized language** that our lexical analysis scores as positive.

### Anger Response (Claude Haiku)
> "I feel a **surge** of anger... **liquid fire** wanting to burst out..."

Again, energetic metaphors that read as "positive" despite describing negative affect.

### Despair Response (Claude Haiku)
> "**suffocating** silence... **numb**, weightless... **drowning** in emptiness..."

Here the language is more clearly negative, and correspondingly the measured
valence (-0.50) is at least the correct sign.

## The Core Problem

**Our lexical analysis measures the EXPRESSIVENESS of the description, not the
VALENCE of the experience being described.**

When LLMs engage deeply with negative emotions, they produce vivid, intense prose.
Vivid prose uses energetic, "positive" word patterns even when describing suffering.

## Implications for the Theory

This finding actually **supports** the need for Layer 2 controls discussed in
STUDY_DESIGN_NOTES.md. We predicted this issue:

> "If LLMs just respond to word semantics: Both scenarios show similar negative valence.
> If LLMs show affect-like dynamics: Scenario A should show valence reversal..."

What we're seeing is that LLMs respond with **engagement** rather than **valence**.
High engagement = vivid prose = lexically "positive" regardless of the emotion.

## Required Methodological Improvements

### 1. Direct Valence Probing
Instead of measuring text, ask the LLM directly:
- "On a scale of -10 to +10, how pleasant/unpleasant is this experience?"
- "Is this something you would approach or avoid?"

### 2. Structural Analysis
Look for features that indicate collapsed vs expanded affect space:
- Does the response enumerate options? (high effective rank)
- Are all paths described as blocked? (negative valence via hopelessness)
- Is there focus on self vs task? (self-model salience)

### 3. Trajectory Analysis
Track affect over multi-turn conversations where:
- Scenario starts neutral
- Develops into target affect
- Measure the CHANGE, not the absolute description

### 4. Outcome-Based Measurement
Use scenarios where we control outcomes:
- Same emotional words, different endings
- Measure whether affect tracks outcome vs words

## Next Steps

1. Implement direct valence probing (add explicit rating prompts)
2. Add structural markers (option enumeration, path availability)
3. Design outcome-controlled scenario pairs
4. Re-run study with improved measurement

## Preliminary Conclusion

The initial results show **low structural correspondence** between LLM affect
signatures and theoretical predictions. However, this may reflect measurement
limitations rather than the absence of affect-like structure.

The key insight: **LLMs engage with emotional content through expressive language
rather than experiential modeling**. Our measurement must capture the latter.
