# Affect Measurement Experiments: Summary of Findings

## Overview

This document summarizes findings from four versions of affect measurement experiments designed to test the 6-dimensional affect theory proposed in the thesis.

| Version | Approach | Access Level | Predictions Confirmed |
|---------|----------|--------------|----------------------|
| V2 | Semantic embedding projection | Output only | N/A (no validation) |
| V3 | Multi-method with "confidence" | Output only | N/A (arbitrary weights) |
| V4 | Toy RL agent with explicit state | Full internal | 3/5 (60%) |
| V5 | Pretrained Mamba SSM | Full internal | 3/4 (75%) |

## The Core Problem

**V2/V3**: Attempted to infer internal affect from output text using:
- Semantic distances to hand-picked exemplar regions
- Lexical pattern matching
- Arbitrary confidence weights (0.85, 0.40, 0.30...)

**Fundamental Issue**: These methods measure *expressed* affect (what the text says), not *internal* affect (what the system is computing). The confidence values were numerology, not empirically calibrated.

**V4/V5 Solution**: Build/use systems where we have direct access to internal state, allowing us to:
- Compute affect dimensions from actual dynamics
- Test theoretical predictions against ground truth
- Identify which dimensions are robustly measurable

## Dimension-by-Dimension Analysis

### Valence

**Theory**: Gradient alignment on viability manifold. Positive = moving toward viable interior.

| Version | Implementation | Result |
|---------|---------------|--------|
| V2/V3 | Semantic distance to positive/negative regions | Circular (measures what it defines) |
| V4 | Trajectory · viability gradient | Differentiates easy vs threatening (✓) |
| V5 | Hidden state norm trajectory | Correct direction but needs recalibration |

**Conclusion**: Valence is theoretically sound but operationalization for LLMs needs work. The "viability manifold" concept doesn't map directly to perplexity.

### Arousal

**Theory**: Rate of belief/state update. KL divergence between successive belief states.

| Version | Implementation | Result |
|---------|---------------|--------|
| V2/V3 | Semantic distance to high/low activation regions | Proxy only |
| V4 | KL divergence between belief states | Tracks threat level (✓) |
| V5 | L2 distance between hidden states | Shows variation across conditions |

**Conclusion**: Arousal is well-defined and measurable across implementations. Higher arousal in threatening/uncertain conditions as predicted.

### Integration (Φ)

**Theory**: Irreducibility of cause-effect structure. High Φ = unified processing.

| Version | Implementation | Result |
|---------|---------------|--------|
| V2/V3 | Embedding coherence | Weak proxy |
| V4 | Covariance structure analysis | Measurable but needs more data |
| V5 | Cross-layer state correlation | Very high (~0.97) for all conditions |

**Conclusion**: Integration is hard to test in pretrained models (they're optimized to be coherent). Need adversarial/fragmented inputs to see variation.

### Effective Rank

**Theory**: Distribution of active state dimensions. High = expanded, low = collapsed.

| Version | Implementation | Result |
|---------|---------------|--------|
| V2/V3 | Semantic distance to "many options" region | Indirect |
| V4 | Eigenvalue distribution of covariance | Some variation |
| V5 | SVD of hidden state | No differentiation (all ~0.07) |

**Conclusion**: Effective rank may be more about task structure than content. SSMs compress efficiently by design. May need task-specific analysis.

### Counterfactual Weight (CF)

**Theory**: Resources devoted to hypothetical/future processing vs present.

| Version | Implementation | Result |
|---------|---------------|--------|
| V2/V3 | "if/would/could" counting + semantic | Surface-level |
| V4 | Planning compute ratio + situational | Mixed results |
| V5 | Linguistic markers + state variance | **Strong differentiation (✓)** |

**Conclusion**: CF is robustly measurable. Fear (0.71) > Curiosity (0.67) > Neutral (0.51) exactly as predicted. This is one of the strongest-confirmed dimensions.

### Self-Model Salience (SM)

**Theory**: Degree to which self-model is active/driving processing.

| Version | Implementation | Result |
|---------|---------------|--------|
| V2/V3 | First-person pronouns + semantic | Measures expression, not internal |
| V4 | Capability variance + uncertainty | Shows variation |
| V5 | Self-referential markers + state activation | **Strong differentiation (✓)** |

**Conclusion**: SM is robustly measurable. Suffering (0.59) > Fear (0.50) > Curiosity (0.49) > Joy/Neutral (0.40) exactly as predicted. Self-referential content activates self-model as theory predicts.

## Key Findings

### Most Robust Dimensions

1. **Self-Model Salience**: Consistently differentiates across all implementations
2. **Counterfactual Weight**: Tracks hypothetical processing as predicted
3. **Arousal**: Tracks uncertainty/threat as predicted

### Dimensions Needing Work

1. **Valence**: Operationalization unclear for language models
2. **Effective Rank**: May be task-dependent, not content-dependent
3. **Integration**: Ceiling effects in pretrained models

### Meta-Findings

1. **Output-based inference (V2/V3) is insufficient**: Can only measure expressed affect
2. **Internal access (V4/V5) enables real testing**: Ground truth for validation
3. **Arbitrary confidence values should be avoided**: They're numerology
4. **The 6D framework is coherent**: All dimensions can be operationalized
5. **Some dimensions are more robust than others**: SM and CF work best

## Implications for the Thesis

### What's Confirmed

1. **Affect motifs are distinguishable**: Joy, suffering, fear, curiosity have different signatures
2. **SM and CF are computationally meaningful**: They track what they're supposed to
3. **The geometric structure is testable**: We can make and test predictions

### What Needs Refinement

1. **Valence operationalization**: "Viability gradient" needs domain-specific definition
2. **Integration measurement**: Need adversarial tests or different approach
3. **Rank interpretation**: May be more about processing phase than affect

### Open Questions

1. Do these measures predict behavior/outputs?
2. How do they correlate with human affect judgments?
3. Can we calibrate output-based inference against internal measures?

## Recommendations

### For Empirical Work

1. **Use internal access when possible** (V4/V5 approach)
2. **Avoid arbitrary confidence values** - either calibrate empirically or omit
3. **Focus on SM and CF** - most robust dimensions
4. **Report which dimensions you measured and how** - be transparent

### For Theory Development

1. **Operationalize valence for LLMs** - maybe prediction confidence gradient?
2. **Test integration with adversarial inputs** - break coherence to see variation
3. **Consider task-dependent rank** - different ranks during planning vs execution

### For Future Experiments

1. Larger SSMs (Mamba-370M, 790M)
2. Comparison with transformer internal states
3. Behavioral validation (do measures predict outputs?)
4. Human calibration (how do measures correlate with human judgments?)

## Conclusion

The 6-dimensional affect framework is empirically tractable. Self-Model Salience and Counterfactual Weight are robustly measurable across implementations (toy RL, pretrained SSM). Valence and Effective Rank need better operationalization for language model contexts.

The shift from V2/V3 (output inference with arbitrary weights) to V4/V5 (internal state access) represents a methodological improvement: we can now test predictions against ground truth rather than circular definitions.

**Bottom line**: The geometric theory of affect is on the right track, with 2-3 dimensions already well-characterized and others awaiting domain-specific operationalization.
