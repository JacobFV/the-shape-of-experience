# Empirical Studies for Testing the Inevitability Thesis

This document specifies concrete, pre-registered studies to test the falsifiable claims of the six-dimensional affect framework.

## Study 1: Dimension Independence (Factor Analysis)

### Hypothesis
The six affect dimensions are empirically distinguishable and show the predicted independence structure.

### Method
- **N**: 500 participants (power analysis: detect r = 0.2 with 95% power)
- **Design**: Cross-sectional experience sampling over 2 weeks
- **Measures**:
  - 6x daily self-report on all six dimensions
  - Total: ~84 observations per participant
- **Analysis**:
  - Confirmatory factor analysis (6-factor model)
  - Compare fit to 2-factor (valence-arousal), 3-factor (core affect + cognition), and 1-factor models

### Predictions
1. 6-factor model will show superior fit (RMSEA < 0.06, CFI > 0.95)
2. Dimensions will be distinguishable (factor loadings > 0.5 on intended factors)
3. Correlations between dimensions will match theoretical predictions:
   - Valence-Arousal: weak (r ~ 0.1-0.2)
   - Integration-EffectiveRank: moderate positive (r ~ 0.3-0.4)
   - CF-SM: moderate positive (r ~ 0.3-0.4)

### Falsification Criteria
- If a 2-factor model fits equally well or better
- If dimensions load on unexpected factors
- If predicted independence patterns fail

---

## Study 2: Valence-Viability Correlation

### Hypothesis
Self-reported valence correlates with objective measures of physiological viability/threat.

### Method
- **N**: 100 participants with continuous physiological monitoring
- **Design**: 7-day ambulatory assessment
- **Measures**:
  - Continuous: HR, HRV, accelerometry, skin temperature
  - Daily: Morning cortisol, evening cortisol
  - Event-contingent: Valence rating (10x daily random prompts)
  - Weekly: Blood markers (inflammatory cytokines, immune function)

### Predictions
1. Momentary valence correlates with concurrent HRV (r > 0.2)
2. Chronic valence predicts cortisol slope (flatter slope = worse)
3. Valence predicts next-day inflammatory markers (negative valence → higher IL-6)
4. Acute threats (detected via HR spike) precede negative valence by seconds

### Falsification Criteria
- No correlation between valence and physiological threat markers
- Valence disconnected from objective safety/danger
- People report positive valence during measurable physiological crisis

---

## Study 3: Integration-Unity Correspondence

### Hypothesis
Neural integration measures correlate with self-reported experiential unity.

### Method
- **N**: 60 participants
- **Design**: Within-subjects manipulation of integration
- **Conditions**:
  1. Baseline (resting state)
  2. Fragmentation induction (rapid task-switching, divided attention)
  3. Integration enhancement (single-task deep focus, meditation)
  4. Pharmacological (optional: low-dose anesthetic vs caffeine)
- **Measures**:
  - EEG: Lempel-Ziv complexity, PCI (if TMS available), connectivity
  - Self-report: Unity/fragmentation scale (immediately after each condition)
  - Behavioral: Task coherence, response consistency

### Predictions
1. EEG complexity correlates with reported unity (r > 0.3)
2. Task-switching reduces both neural integration and reported unity
3. Meditation increases both
4. Within-person changes in neural integration predict changes in reported unity

### Falsification Criteria
- Neural integration and experiential unity dissociate
- High integration with reported fragmentation
- Anesthesia doesn't reduce integration measures

---

## Study 4: Cultural Form Affect Signatures

### Hypothesis
Different cultural forms (art, music, practices) produce distinct, predictable affect signatures.

### Method
- **N**: 200 participants (40 per condition)
- **Design**: Between-subjects exposure to different stimuli
- **Conditions**:
  1. Tragedy (sad film scene)
  2. Comedy (humorous content)
  3. Awe-inspiring (nature documentary, space imagery)
  4. Horror (threat-based content)
  5. Meditative music
- **Measures**: Full 6-dimension self-report before, during, after exposure

### Predictions

| Condition | Val | Ar | Φ | r_eff | CF | SM |
|-----------|-----|-----|---|-------|-----|-----|
| Tragedy | - | + | + | - | + | + |
| Comedy | + | + | + | + | - | - |
| Awe | + | + | + | + | + | - |
| Horror | - | ++ | - | - | + | + |
| Meditative | + | - | + | + | - | - |

### Falsification Criteria
- Cultural forms do not produce distinguishable signatures
- Comedy increases self-model salience
- Horror increases effective rank

---

## Study 5: Flow State Validation

### Hypothesis
Flow states have the predicted signature: low SM, high Φ, moderate Ar, positive Val.

### Method
- **N**: 80 participants
- **Design**: Flow induction via skill-challenge matching
- **Task**: Video game with adaptive difficulty
- **Conditions**:
  1. Too easy (skill >> challenge)
  2. Matched (flow zone)
  3. Too hard (challenge >> skill)
- **Measures**:
  - Continuous gameplay metrics
  - Interruption-contingent affect sampling
  - Post-session flow questionnaire

### Predictions
1. Flow zone shows: SM < 0.3, Φ > 0.7, Ar ~ 0.5, Val > 0.5
2. Too easy: Low Ar, moderate SM, negative Val (boredom)
3. Too hard: High Ar, high SM, negative Val (anxiety)
4. Time perception distorted in flow (underestimation)

### Falsification Criteria
- Flow characterized by high self-model salience
- No difference in integration across conditions
- Flow does not show positive valence

---

## Study 6: Meditation Affect Signatures

### Hypothesis
Different meditation types produce distinct, predicted affect signatures.

### Method
- **N**: 90 experienced meditators (30 per style)
- **Design**: Within-subjects comparison of meditation styles
- **Conditions**:
  1. Focused attention (breath focus)
  2. Open monitoring (choiceless awareness)
  3. Loving-kindness (metta)
- **Measures**:
  - EEG during meditation
  - Periodic self-report (every 5 min)
  - Post-session detailed phenomenology

### Predictions

| Style | Val | Ar | Φ | r_eff | CF | SM |
|-------|-----|-----|---|-------|-----|-----|
| Focused | + | - | + | - | - | ~ |
| Open | + | - | ++ | ++ | - | - |
| Metta | ++ | ~ | + | + | + | + |

### Falsification Criteria
- No difference between meditation styles
- Open monitoring doesn't increase effective rank
- Focused attention increases mind-wandering (CF)

---

## Study 7: Clinical Signatures

### Hypothesis
Depression and anxiety have distinct, predictable affect signatures.

### Method
- **N**: 150 total (50 MDD, 50 GAD, 50 controls)
- **Design**: Case-control with experience sampling
- **Measures**:
  - 2 weeks of 6x daily self-report
  - Standardized clinical measures (BDI, BAI)
  - Optional: EEG resting state

### Predictions

| Group | Val | Ar | Φ | r_eff | CF | SM |
|-------|-----|-----|---|-------|-----|-----|
| MDD | -- | - | ~ | -- | + | ++ |
| GAD | - | ++ | ~ | - | ++ | + |
| Control | + | ~ | + | + | ~ | ~ |

Key distinctions:
- MDD: Collapsed effective rank (tunnel vision), low arousal
- GAD: High arousal, very high counterfactual weight (worry)
- Both: Elevated self-model salience

### Falsification Criteria
- Depression shows high effective rank
- Anxiety shows low counterfactual weight
- No difference in self-model salience between clinical and control

---

## Study 8: Real-Time Viability Threat

### Hypothesis
Approach to viability boundary is felt as negative valence before objective markers.

### Method
- **N**: 40 participants
- **Design**: Controlled physiological stressor
- **Protocol**:
  1. Baseline
  2. Cold pressor task (hand in ice water)
  3. Breath holding task
  4. Recovery
- **Measures**:
  - Continuous valence dial (moment-to-moment report)
  - Continuous physiology (HR, BP, SpO2, HRV)
  - Temperature sensors

### Predictions
1. Valence decreases during stressor
2. Valence change precedes physiological markers by ~2 seconds
3. Valence tracks proximity to tolerance threshold
4. Recovery valence predicts physiological recovery speed

### Falsification Criteria
- Positive valence during physiological threat
- No temporal ordering (valence doesn't precede or correlate with stress)
- Valence doesn't track threshold proximity

---

## Data Management and Analysis

### Pre-registration
All studies will be pre-registered on OSF before data collection.

### Code Repository
Analysis code will be version-controlled and made public upon publication.

### Data Sharing
De-identified data will be shared on OSF upon publication.

### Replication
Key findings will be subject to pre-registered replication attempts.

---

## Falsification Summary

The thesis is falsified if:

1. **Dimension Independence Fails**: Six factors are not empirically distinguishable
2. **Valence-Viability Disconnect**: Valence doesn't correlate with objective threat/safety
3. **Integration-Unity Dissociation**: Neural integration and experiential unity are unrelated
4. **Cultural Signatures Fail**: Art/practice forms don't produce predicted signatures
5. **Flow Signature Wrong**: Flow shows high self-salience or low integration
6. **Meditation Predictions Fail**: Different styles don't produce different signatures
7. **Clinical Patterns Fail**: Depression/anxiety don't show predicted profiles
8. **No Temporal Priority**: Valence doesn't precede physiological threat markers

Any single major failure requires theory revision. Multiple failures falsify the framework.
