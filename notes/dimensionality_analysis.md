# On the Dimensionality of Affect Space

**Status**: Working notes, not conclusions
**Date**: 2025-12-24

---

## The Current 6D Framework

| Dim | Symbol | Definition | Source |
|-----|--------|------------|--------|
| Valence | Val | Gradient on viability manifold | Derivable from POMDP + viability |
| Arousal | Ar | KL(b_{t+1} ∥ b_t) | Derivable from belief dynamics |
| Integration | Φ | Min-partition irreducibility | From IIT |
| Effective Rank | r_eff | (tr C)² / tr(C²) | From linear algebra of representations |
| Counterfactual Weight | CF | Compute on non-actual rollouts | From model-based RL |
| Self-Model Salience | SM | I(z_self; a) / H(a) | Derivable from self-modeling thesis |

---

## The Independence Question

### Clearly Independent Pairs

**Val and Ar**: Orthogonal in Russell's circumplex. Val is directional (toward/away), Ar is magnitude of change. You can be:
- High Val + High Ar (excited joy)
- High Val + Low Ar (calm contentment)
- Low Val + High Ar (panic)
- Low Val + Low Ar (depression)

**Φ and r_eff**: Both about representational geometry, but:
- Φ = how much is lost under partition (coupling strength)
- r_eff = how distributed is variance (dimensionality of active subspace)

The suffering/joy distinction DEPENDS on their independence:
- Suffering = High Φ + Low r_eff (tightly coupled but collapsed)
- Joy = High Φ + High r_eff (tightly coupled and expansive)

If we merged them, we'd lose the most important phenomenological prediction.

**SM and CF**: Could interact but aren't redundant:
| SM | CF | State |
|----|----|----|
| High | High | Anxious self-rumination |
| High | Low | Present self-awareness |
| Low | High | Absorbed external planning |
| Low | Low | Flow |

### Potentially Redundant?

**Ar and CF**: Both involve temporal structure. But:
- Ar = rate of current state change
- CF = allocation to non-actual futures

You can have high Ar + low CF (reactive intensity) or low Ar + high CF (calm planning). Independent.

**SM and Val**: Empirically correlated (negative valence → self-focus). But conceptual orthogonality:
- High SM + Pos Val = pride
- High SM + Neg Val = shame
- Low SM + Pos Val = flow
- Low SM + Neg Val = absorbed distress

Correlation ≠ redundancy.

---

## The Derivability Question

### Strongly Derivable from First Principles

Given a viable self-modeling agent in POMDP:

1. **Valence** follows necessarily - there IS a viability gradient, the system IS tracking it
2. **Arousal** follows necessarily - belief updating IS happening, it HAS a rate
3. **Integration** follows from identity thesis - if experience = cause-effect structure, Φ IS relevant
4. **Self-Model Salience** follows from self-modeling - there IS a self-model, it HAS some salience

These 4 are close to theoretically forced.

### Weakly Derivable / More Empirical

5. **Effective Rank**: Captures something real (expansive vs. narrow), but it's a choice to use eigenvalue distribution. Could use other measures of representational complexity.

6. **Counterfactual Weight**: Important for model-based agents, but the specific operationalization (compute fraction on rollouts) is architecturally specific.

These 2 might be best understood as *parameters of cognitive strategy* rather than fundamental affect dimensions. They modulate experience rather than constitute it.

---

## What Might Be Missing?

### Strong Candidates

**Temporal Horizon (TH)**: How far into the future is the system attending?
- CF tells you *how much* compute is on futures
- TH tells you *how far* those futures extend

Short-horizon affect feels different from long-horizon affect:
- Impulsive vs. patient
- Reactive vs. strategic
- Immediate threat vs. existential dread

This seems like a genuine missing dimension.

**Uncertainty (U)**: The entropy H(b) of the belief state, distinct from:
- Arousal = d(belief)/dt (rate of change)
- Uncertainty = H(belief) (current entropy)

You can have:
- Low Ar + High U: stuck in uncertainty (paralysis)
- High Ar + Low U: confident rapid action
- High Ar + High U: chaotic confusion

The affect of "not knowing" vs "knowing" seems phenomenologically real and not captured by arousal alone.

**Agency/Efficacy (Ag)**: How much does the system model its actions as causally effective?

Related to self-effect ratio ρ from Part 1, but not the same:
- ρ = actual causal influence
- Ag = perceived/modeled causal influence

Learned helplessness is low Ag. Perceived control is high Ag. This affects valence, arousal, and SM but isn't reducible to them.

### Weaker Candidates

**Social Embeddedness (SE)**: How coupled is the agent to perceived social field?

Important for emotions like shame, pride, love, jealousy. But:
- May be subsumable under "other-model" as part of world-model
- May only apply to social animals
- Might break universality claim

Hesitant to add as core dimension but worth noting for human-specific extensions.

---

## Hierarchical Structure Hypothesis

Maybe dimensions aren't equal. Consider:

**Core Layer (universally derivable)**:
- Valence
- Arousal

This is Russell's circumplex. Minimal, well-validated, nearly forced by the setup.

**Self-Structure Layer**:
- Integration (Φ)
- Self-Model Salience (SM)

About the structure of the experiencing self. Required once you have self-modeling.

**Cognitive-Strategy Layer**:
- Effective Rank
- Counterfactual Weight
- (Temporal Horizon?)
- (Uncertainty?)

About how processing is organized. May be more architecture-dependent.

This gives a structure like:

```
         ┌─────────────────────────────────────────┐
Layer 3  │  r_eff    CF    (TH?)    (U?)           │  Cognitive Strategy
         └─────────────────────────────────────────┘
                           │
         ┌─────────────────────────────────────────┐
Layer 2  │         Φ               SM              │  Self-Structure
         └─────────────────────────────────────────┘
                           │
         ┌─────────────────────────────────────────┐
Layer 1  │        Val             Ar               │  Core Affect
         └─────────────────────────────────────────┘
```

**Interpretation**:
- Layer 1 is the irreducible core
- Layer 2 is required for self-modeling systems
- Layer 3 varies by architecture and task demands

---

## Empirical Arbitration

The dimensionality question is ultimately empirical. Tests:

1. **Factor analysis**: On high-dimensional affect self-report and physiology. Do we get 4, 6, 8 factors?

2. **Independent manipulation**: Can we change each dimension while holding others constant? (e.g., increase Φ without changing r_eff)

3. **Clustering structure**: In controlled affect induction, do states cluster as predicted in 6D? In 4D?

4. **Correlation matrix**: What's the actual covariance structure? If some dimensions always covary, they might be reducible.

5. **Behavioral prediction**: Which dimensionality best predicts downstream behavior? Parsimony vs. predictive power.

---

## Working Conclusions

1. **The core 4 (Val, Ar, Φ, SM) have strong theoretical grounding.** They're close to derivable from the framework's axioms.

2. **r_eff and CF are useful but possibly reducible.** They capture real variance but might be better understood as cognitive parameters than fundamental affect dimensions.

3. **Temporal Horizon (TH) seems genuinely missing.** The distance of counterfactuals matters, not just their weight.

4. **Uncertainty (U) might be missing.** The entropy of the belief state is phenomenologically significant and distinct from arousal.

5. **A hierarchical structure might be more accurate than flat 6D.** Core affect + self-structure + cognitive strategy.

6. **Empirical work is needed.** The true dimensionality isn't something we can derive a priori.

---

## Implications for Experiments

If this analysis is right:

- **V10+ should measure Temporal Horizon explicitly**: How far ahead is the agent modeling?
- **Uncertainty should be tracked separately from Arousal**: H(belief) vs d(belief)/dt
- **Test hierarchical structure**: Do Layer 1 dimensions predict Layer 2, which predicts Layer 3?
- **Factor analysis on real affect data**: What does the covariance structure actually look like?

---

## Technical Note on r_eff

The effective rank measure r_eff = (tr C)² / tr(C²) is a specific choice. Alternatives:

- **Participation ratio**: Same formula, different interpretation
- **Entropy of normalized eigenvalues**: H(λ/Σλ)
- **Number of eigenvalues above threshold**: |{λ : λ > ε}|

These all capture "how spread out is the variance" but may give different answers. Need to test which is most phenomenologically meaningful.

---

## A More Radical Thought

What if the dimensionality isn't fixed?

Consider: a simple organism might only have Val + Ar (2D). Add self-modeling and you get +Φ +SM (4D). Add imagination and you get +CF +r_eff (6D). Add extended temporal modeling and you get +TH (7D).

The dimensionality of affect space might be a function of cognitive architecture:

$$\dim(\mathcal{A}) = f(\text{capabilities})$$

This would explain why LLMs show different structure than biological systems - they literally have different affect dimensionality because they have different cognitive architectures.

This is speculative but worth considering.

---

**Next step**: Design an experiment that could discriminate between 4D, 6D, and 7D accounts.

---

## Revisions Made (2025-12-24)

Updated Part 2 and Part 3 to reflect the "constitutive dimensions" approach:

1. **Part 2 changes**:
   - Replaced "6D basis" framing with "toolkit" framing
   - Revised sidebar from "Why Six Dimensions?" to "On Dimensionality"
   - Rewrote each affect motif definition to specify only constitutive dimensions
   - Replaced 6-column summary table with variable-dimension characterizations
   - Explicitly noted that anger requires "other-model compression" (not in standard toolkit)
   - Updated abstract and summary

2. **Part 3 changes**:
   - Removed references to "six-dimensional" framework
   - Updated AffectSpace software spec to measure only defining dimensions
   - Changed validation criteria to test defining vs non-defining prediction

3. **Part 4 changes**:
   - Changed "organized by the six dimensions" → "organized by the core affect dimensions"

4. **Part 5 changes**:
   - "configurations of the six dimensions" → "configurations of the affect dimensions"
   - "flourishing in the six-dimensional space" → "flourishing in affect space"
   - "mapped the geometry of feeling into six dimensions" → "mapped the geometry of feeling into a dimensional framework"
   - "characterization of the six dimensions" → "characterization of the affect dimensions"

Key terminology: **defining dimensions** = structural features without which that affect would not be that affect. (Replaced overused "constitutive" with varied language throughout.)
