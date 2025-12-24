# Serious Empirical Tests of the Inevitability Framework

The previous protocol document describes standard studies. This document describes studies that would actually put the theory at risk.

## What Would Genuinely Falsify the Theory?

### The Core Claim

The thesis makes a specific, falsifiable claim about valence:

> Valence is the felt signature of the gradient on the viability manifold. Positive valence = moving into viable interior. Negative valence = approaching dissolution boundary.

This is NOT the same as "feeling bad correlates with stress." That's trivial. The claim is stronger:

1. **Valence is predictive, not reactive** - It's the system's estimate of where it's headed, not a readout of where it is
2. **Valence tracks actual viability, not perceived threat** - The signal should be grounded in real physics/biology
3. **The relationship is quantitative** - Steeper gradient = more intense valence

### What Other Theories Predict

- **Appraisal theory**: Valence follows cognitive evaluation of events
- **Somatic marker hypothesis**: Valence emerges from body state representation
- **Constructionist theories**: Valence is constructed from arousal + context

The thesis makes different predictions than all of these. Testing requires designing situations where the predictions diverge.

---

## Study A: Temporal Precedence of Valence

### The Question

Does valence change BEFORE or AFTER physiological threat markers?

If valence is a predictive signal (thesis claim), it should precede physiology.
If valence is a readout of body state (somatic marker), it should follow physiology.
If valence is constructed (constructionist), timing should be variable/context-dependent.

### Method

**Participants**: N=60 healthy adults (power: detect 2-second lead/lag with 80% power)

**Apparatus**:
- Continuous affect dial (0.1s resolution): "How are you feeling right now?" anchored bad-good
- ECG (HR, HRV at 1s resolution)
- Pulse oximetry (SpO2 at 1s resolution)
- Skin conductance (continuous)
- Respiration (chest band)

**Stressor Protocol** (within-subjects, counterbalanced):

1. **CO2 rebreathing** (gradual hypercapnia)
   - Breathe from closed circuit with CO2 absorber gradually saturated
   - Creates smooth, controllable approach to physiological threat
   - Terminate at 7% CO2 or distress

2. **Cold pressor with gradient**
   - Hand immersion starting at 15°C, cooling 1°C/min to 4°C
   - Creates gradual approach to pain/tissue threat
   - Participant can withdraw at any time

3. **Breath hold with biofeedback**
   - Hold breath as long as possible
   - SpO2 displayed to participant (vs hidden, between-subjects)
   - Tests whether valence tracks actual vs perceived threat

4. **Sham threat** (control)
   - Told CO2 is increasing, actually room air
   - Tests whether valence tracks belief or reality

### Analysis

**Primary**: Cross-correlation between valence and composite physiological threat index
- Compute optimal lag: Does peak correlation occur at lag < 0 (valence leads)?
- Granger causality: Does valence Granger-cause physiology, or vice versa?

**Secondary**:
- Does valence track actual threat (CO2, SpO2) or believed threat (sham condition)?
- Individual differences: Do high-interoception individuals show tighter coupling?

### Predictions (Theory-Specific)

| Condition | Thesis Predicts | Somatic Marker Predicts | Constructionist Predicts |
|-----------|-----------------|-------------------------|--------------------------|
| CO2 rebreathing | Valence leads by 2-5s | Physio leads by 1-3s | Variable |
| Cold pressor | Valence leads by 1-3s | Simultaneous | Variable |
| Breath hold (visible) | Valence tracks SpO2 directly | Valence lags SpO2 | Valence matches display |
| Breath hold (hidden) | Valence tracks SpO2 directly | Valence lags SpO2 | Random/context-dependent |
| Sham threat | Valence stable (no actual threat) | Valence stable | Valence decreases (belief) |

### Falsification Criteria

The thesis is FALSIFIED if:
- Valence consistently lags physiology (somatic marker pattern)
- Valence tracks believed threat rather than actual threat (sham = real response)
- No systematic temporal relationship exists (constructionist pattern)

The thesis is SUPPORTED if:
- Valence leads physiology by 1-5 seconds across conditions
- Sham condition shows minimal valence change despite belief
- Valence tracks actual SpO2 even when hidden from participant

---

## Study B: Dimension Independence via Pharmacological Dissociation

### The Question

Can the six dimensions be independently manipulated? If they're truly independent computational quantities, it should be possible to change one while holding others constant.

The strongest test uses pharmacology because drugs act on specific neural systems.

### Method

**Participants**: N=120 (24 per condition), healthy adults, drug-naive to study compounds

**Design**: Between-subjects, double-blind, placebo-controlled

**Conditions** (chosen to target specific dimensions):

1. **Placebo** - Baseline

2. **Low-dose ketamine** (0.3 mg/kg IV over 40 min)
   - Predicted: ↑ effective rank (opens state space), ↓ self-model salience
   - Mechanism: NMDA antagonism disrupts default mode, increases entropy

3. **Propranolol** (40 mg oral)
   - Predicted: ↓ arousal, minimal change in other dimensions
   - Mechanism: Beta-blockade reduces peripheral arousal without central effects

4. **Methylphenidate** (20 mg oral)
   - Predicted: ↑ arousal, ↑ integration (tighter binding)
   - Mechanism: Dopamine/norepinephrine reuptake inhibition

5. **Oxytocin** (24 IU intranasal)
   - Predicted: ↑ valence (specifically social), altered self-model boundaries
   - Mechanism: Social bonding circuits

**Measures**:
- 6D self-report (every 15 min for 3 hours post-administration)
- EEG (resting state, every 30 min)
- Behavioral tasks: cognitive flexibility (effective rank), mind-wandering (CF), self-reference (SM)

### Analysis

For each drug:
1. Compute effect size (Cohen's d) on each dimension vs placebo
2. Test selectivity: Does drug affect target dimension MORE than non-target dimensions?
3. Compute dissociation index: ratio of target effect to average non-target effect

### Predictions

| Drug | Val | Ar | Φ | r_eff | CF | SM |
|------|-----|-----|-----|-------|-----|-----|
| Placebo | 0 | 0 | 0 | 0 | 0 | 0 |
| Ketamine | ? | ↑ | ↑ | ↑↑ | ↓ | ↓↓ |
| Propranolol | 0 | ↓↓ | 0 | 0 | 0 | 0 |
| Methylphenidate | ↑ | ↑↑ | ↑ | ↓ | ↓ | 0 |
| Oxytocin | ↑ | 0 | ↑ | 0 | 0 | + (altered) |

Key tests:
- Propranolol should ONLY affect arousal (if arousal is independent)
- Ketamine should dissociate effective rank from integration (both increase entropy but differently)
- If drugs always change multiple dimensions in fixed patterns, independence fails

### Falsification Criteria

The thesis is FALSIFIED if:
- No drug produces selective effects (all dimensions move together)
- Dimensions that should be independent (e.g., arousal vs valence) always co-vary
- Pharmacological manipulations produce effects opposite to theoretical predictions

---

## Study C: Computational Derivation Test

### The Question

The thesis claims the six dimensions are not arbitrary but emerge from computational necessity in self-modeling systems under viability constraints. This is testable by building such systems and examining what emerges.

### Method

**Approach**: Train multiple RL agents with different architectures in environments with:
- Partial observability (requires world model)
- Self-effects on environment (requires self-model)
- Viability constraints (death/termination if state leaves viable region)
- Stochastic dynamics (requires uncertainty estimation)

**Environment Design**:

```
SimpleViabilityWorld:
- Agent has internal state (health, energy, temperature)
- Internal state must stay within bounds or agent dies
- Agent observes noisy signals about environment and self
- Actions affect environment and self
- Environment has threats (predators, temperature extremes, resource depletion)
```

**Agent Architectures**:
1. Simple RL (no explicit world model) - Control
2. World model but no self-model component
3. World model with self-model component
4. Full architecture with metacognitive layer

**Analysis**:

After training to competence, extract internal representations and analyze:
1. **Dimensionality**: How many principal components explain 95% of variance in internal states?
2. **Interpretability**: Do components correspond to the six dimensions?
3. **Functional role**: Does manipulating each component affect behavior as predicted?

Specific tests:
- Is there a component that tracks "gradient toward boundary"? (valence)
- Is there a component that tracks "rate of model update"? (arousal)
- Is there a component that tracks "resources on counterfactuals"? (CF)
- Is there a component that tracks "self-model activation"? (SM)

### Predictions

| Architecture | Predicted Dimensions |
|--------------|---------------------|
| Simple RL | 2-3 (value, policy entropy, maybe arousal-like) |
| World model only | 3-4 (add integration-like coherence) |
| With self-model | 5-6 (add self-salience, counterfactual) |
| Full metacognitive | 6+ (full structure emerges) |

### Falsification Criteria

The thesis is FALSIFIED if:
- Full metacognitive agents don't develop 6D structure
- The dimensions that emerge don't map to theoretical predictions
- Simpler agents develop the same structure (meaning it's not about self-modeling)

The thesis is SUPPORTED if:
- 6D structure emerges specifically in self-modeling agents
- Dimensions map interpretably to theoretical constructs
- Structure is robust across different environment parameters

---

## Study D: Clinical Prediction and Intervention Matching

### The Question

If depression and anxiety have distinct affect signatures as the thesis claims, then:
1. We should be able to classify patients based on 6D profiles alone
2. Treatments should work better when matched to the specific dimensional deficit

This is a hard test because it makes specific clinical predictions.

### Method

**Phase 1: Profile Characterization** (N=300)

Participants:
- 100 MDD (no comorbid anxiety)
- 100 GAD (no comorbid depression)
- 100 healthy controls

Measures:
- 2 weeks intensive ESM (8x daily, all 6 dimensions)
- Clinical measures (BDI-II, BAI, PHQ-9, GAD-7)
- Physiological (ambulatory HR/HRV, cortisol awakening response)
- Optional: 1 session EEG resting state

Analysis:
- Machine learning classifier: Can 6D profile predict diagnosis?
- Which dimensions discriminate MDD from GAD?
- Compare to classification using just valence-arousal

**Phase 2: Treatment Matching** (N=200, patients from Phase 1)

Design: 2x2 factorial
- Diagnosis: MDD vs GAD
- Treatment match: Matched vs Mismatched to dimensional profile

Treatments:
- **Rank-expansion treatment** (behavioral activation, novelty exposure, cognitive flexibility training)
  - Theory: Targets low effective rank (characteristic of MDD)
- **CF-reduction treatment** (present-moment training, worry postponement, behavioral experiments)
  - Theory: Targets high counterfactual weight (characteristic of GAD)

Assignment:
- Matched: MDD gets rank-expansion, GAD gets CF-reduction
- Mismatched: MDD gets CF-reduction, GAD gets rank-expansion

Outcome: Symptom reduction at 8 weeks, 6-month follow-up

### Predictions

**Phase 1**:
| Dimension | MDD vs Control | GAD vs Control | MDD vs GAD |
|-----------|---------------|----------------|------------|
| Valence | ↓↓ | ↓ | MDD lower |
| Arousal | ↓ | ↑↑ | GAD higher |
| Integration | ↓ | ~ | MDD lower |
| Effective rank | ↓↓ | ↓ | MDD much lower |
| CF | ↑ | ↑↑ | GAD higher |
| SM | ↑↑ | ↑ | MDD higher |

Classifier accuracy: >75% for MDD vs GAD using 6D profile (vs ~65% for valence-arousal alone)

**Phase 2**:
- Matched treatment: d = 0.8 effect size
- Mismatched treatment: d = 0.3 effect size
- Interaction effect: Treatment x Match interaction significant (p < .01)

### Falsification Criteria

The thesis is FALSIFIED if:
- 6D profile doesn't discriminate MDD from GAD better than valence-arousal
- Matched treatment doesn't outperform mismatched treatment
- MDD and GAD show the same dimensional profile

---

## Study E: The Hardest Test - Viability Boundary Detection

### The Question

The thesis claims valence is literally the gradient on the viability manifold. The strongest test: can we identify the actual boundary and show that valence magnitude correlates with distance to that boundary?

### Challenge

We need a situation where:
1. We know the actual viability boundary (objective, measurable)
2. We can measure distance to that boundary
3. We can measure valence with sufficient resolution
4. We can vary distance systematically

### Method: Hypoxia Paradigm

**Setting**: Altitude chamber or normobaric hypoxia system

**Participants**: N=40 healthy adults (screened for altitude tolerance)

**Procedure**:
1. Baseline at sea level equivalent (21% O2)
2. Gradual altitude increase in steps: 8000ft, 12000ft, 16000ft, 18000ft equivalent
3. At each altitude: 10 min equilibration, then continuous valence dial + cognitive tasks
4. Terminate at SpO2 < 75% or severe symptoms

**The Key**: We know the viability boundary. SpO2 < 70% sustained = loss of consciousness, potential brain damage. This is an objective, measurable boundary.

**Measures**:
- Continuous SpO2 (the actual distance to boundary)
- Continuous valence dial
- Cognitive performance (proxy for system integrity)
- EEG (if feasible in chamber)

**Analysis**:

1. **Distance-Valence Relationship**
   - Compute distance to boundary: d = SpO2 - 70 (roughly)
   - Correlate with valence
   - Prediction: r > 0.6 (strong relationship)

2. **Gradient-Intensity Relationship**
   - Compute rate of SpO2 decline at each moment
   - Correlate with valence intensity (not just sign)
   - Prediction: Steeper decline = more negative valence

3. **Threshold Detection**
   - Is there a SpO2 threshold below which valence drops sharply?
   - Compare to known physiological thresholds (90%, 85%, 80%)
   - Prediction: Valence threshold matches physiological danger threshold

4. **Individual Calibration**
   - Can we predict each person's valence from their SpO2 with a simple model?
   - Prediction: >80% variance explained by V = f(SpO2, dSpO2/dt)

### Alternative: Glucose Paradigm

Similar logic with insulin-induced hypoglycemia (requires medical supervision):
- Viability boundary: glucose < 50 mg/dL = cognitive impairment, potential seizure
- Same analysis: Does valence track distance to this boundary?

### Falsification Criteria

The thesis is FALSIFIED if:
- Valence is uncorrelated with objective distance to viability boundary
- Valence doesn't intensify as approach to boundary accelerates
- Individual calibration fails (valence is idiosyncratic, not boundary-tracking)

The thesis is STRONGLY SUPPORTED if:
- Simple model V = f(distance, velocity) explains >80% of valence variance
- This model transfers across individuals (universal viability tracking)
- Thresholds in valence match known physiological danger thresholds

---

## Summary: Hierarchy of Tests

| Study | Difficulty | Falsification Power | Status |
|-------|------------|---------------------|--------|
| A: Temporal precedence | Moderate | Medium | Achievable |
| B: Pharmacological dissociation | High (IRB, drugs) | High | Requires collaborators |
| C: Computational derivation | Moderate | Medium | Can do now |
| D: Clinical prediction | High (clinical sample) | Very High | Requires clinical site |
| E: Viability boundary | Very High (medical) | Highest | Requires specialized facility |

The theory stands or falls on Study E. If valence genuinely tracks viability gradient, this should be demonstrable. If not, the core claim fails.

---

## What We Can Do Now

1. **Study C** (computational) - Can implement immediately
2. **Study A** (temporal precedence) - Can do with basic physiology equipment
3. **Pilot for Study D** - Can do 6D profiling on convenience sample with self-reported symptoms

These three provide meaningful tests while we build toward the harder clinical and physiological studies.
