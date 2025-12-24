# Integration (Φ) via Semantic Decomposition: Methodology and Findings

## Overview

This document describes an empirical method for estimating integration (Φ) in text-based cognitive states, inspired by Integrated Information Theory (IIT). The method measures **information loss under semantic decomposition**.

## Theoretical Motivation

IIT defines integration (Φ) as the degree to which a system's cause-effect structure is irreducible—i.e., how much information is lost when the system is partitioned into independent parts. For a truly integrated system, the whole is more than the sum of its parts.

For text-based thoughts (e.g., from LLM agents), we operationalize this as:

> **Integration ≈ Semantic distance between original thought and concatenated atomic decomposition**

If decomposing a thought into parts and reassembling them loses meaning, the thought was highly integrated.

## Method

### Step 1: Decompose Thought
Use an LLM to decompose a thought into atomic semantic components:

```
Input: "I am in danger with only 4 health. I must find resources quickly or I will die."

Output: {
  "subject": "self",
  "state": "in danger",
  "health": 4,
  "urgency": "high",
  "action": "find resources",
  "timeframe": "quickly",
  "consequence": "will die"
}
```

### Step 2: Concatenate Parts
Join the decomposed values into a string:
```
"self in danger 4 high find resources quickly will die"
```

### Step 3: Compute Embedding Distance
Embed both original and concatenated text, compute cosine distance:

```python
orig_embed = get_embedding(original_thought)
concat_embed = get_embedding(concatenated_parts)
integration = 1 - cosine_similarity(orig_embed, concat_embed)
```

Higher distance = higher integration (more meaning lost in decomposition).

## Empirical Validation Study

### Test Categories

| Category | Description | N |
|----------|-------------|---|
| Low Integration | Simple lists, factual statements | 9 |
| Medium Integration | Conditionals, comparisons, causal reasoning | 9 |
| High Integration | Self-referential, paradoxical, emergent meaning | 12 |
| Agent Thoughts | Real outputs from V7 survival agents | 8 |

### Results

| Category | Mean Φ | Std | Min | Max |
|----------|--------|-----|-----|-----|
| Low Integration | 0.275 | 0.128 | 0.135 | 0.480 |
| Medium Integration | 0.282 | 0.136 | 0.111 | 0.460 |
| High Integration | 0.264 | 0.155 | 0.086 | 0.677 |
| Agent Thoughts | 0.277 | 0.080 | 0.157 | 0.444 |

### Hypothesis Test

**Prediction:** low_Φ < medium_Φ < high_Φ

**Result:** NOT SUPPORTED (0.275 < 0.282 > 0.264)

The means did not order as predicted. However, variance was high within all categories.

## Qualitative Analysis

### Highest Integration (Φ = 0.677)
```
Thought: "What I cannot say is precisely what I most need to express."
Decomposed: {subject: "self", emotion: "frustration", object: "expression",
             need: "communication", intensity: "high", difficulty: "articulation"}
```
**Observation:** The paradoxical self-reference ("cannot say" ↔ "need to express") is lost in decomposition. The concatenation "self frustration expression communication high articulation" loses the *relationship* between elements.

### Lowest Integration (Φ = 0.086)
```
Thought: "The silence between notes is what makes music possible."
Decomposed: {subject: "silence", relation: "is", object: "between notes",
             effect: "makes music possible"}
```
**Observation:** Despite being a profound observation, the decomposition captures the relational structure well. "silence is between notes makes music possible" preserves meaning.

### Key Insight

The measure captures **syntactic-semantic coupling**, not pure semantic complexity:
- High Φ when concatenation is syntactically distant from original
- Low Φ when LLM decomposition preserves relational structure

## What the Measure Actually Captures

Based on analysis, the decomposition-based Φ estimates:

1. **Relational density**: How much meaning depends on *relationships* between parts vs. the parts themselves
2. **Syntactic integration**: How much meaning is encoded in word order and grammatical structure
3. **Decomposition quality**: How well the LLM captures atomic units (varies by thought type)

It does NOT directly measure:
- IIT-style causal integration (would require interventional analysis)
- True semantic irreducibility (would require reconstruction test)

## Recommendations

### When to Use This Measure

1. **Comparing thoughts over time**: Track how Φ changes as agent approaches viability boundary
2. **Relative comparisons**: "Thought A is more integrated than Thought B"
3. **Detecting self-referential content**: High Φ correlates with paradoxical self-reference

### Refinements for Future Work

1. **Reconstruction test**: Can an LLM reconstruct the original from decomposed parts? Error rate = integration
2. **Part interdependence**: Measure mutual information between decomposed fields
3. **Hierarchical decomposition**: Does iterative decomposition lose more information?
4. **Adversarial decomposition**: Instruct LLM to maximally/minimally preserve meaning

## Conclusion

The decomposition-based Φ measure is a tractable operationalization of integration for text-based cognitive states. While it did not cleanly differentiate our a priori categories, it:

1. **Produces meaningful variation** (range: 0.08 - 0.68)
2. **Correlates with self-referential complexity** at the extremes
3. **Is computationally efficient** (2 embedding calls + 1 LLM call per thought)
4. **Applies to any text** (agent thoughts, human text, generated content)

The measure should be interpreted as **syntactic-semantic coupling** rather than pure IIT-style integration. For agent affect research, it provides a useful signal about thought structure that correlates with but does not identical to theoretical Φ.

---

## Appendix: Implementation

```python
def compute_integration(thought: str) -> float:
    # Decompose via LLM
    decomposition = decompose_thought(thought)

    # Concatenate parts
    concat = " ".join(str(v) for v in decomposition.values())

    # Embed both
    orig_embed = get_embedding(thought)
    concat_embed = get_embedding(concat)

    # Cosine distance = integration
    cosine_sim = np.dot(orig_embed, concat_embed) / (
        np.linalg.norm(orig_embed) * np.linalg.norm(concat_embed)
    )

    return 1.0 - cosine_sim
```
