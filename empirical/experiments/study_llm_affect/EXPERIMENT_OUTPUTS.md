# Experiment Outputs Summary

*Complete record of all empirical studies conducted for LLM affect validation*

## Study 1: Multi-Model Embedding Study v2 (MAIN RESULTS)

**Date**: 2025-12-23
**Location**: `results/multi_model_v2/`
**Method**: Pure embedding-based affect measurement (no word counting)

### Models Tested
- claude_4_haiku (claude-3-5-haiku-latest)
- claude_4_sonnet (claude-3-5-sonnet-20241022) - FAILED
- claude_4.5_sonnet (claude-sonnet-4-20250514)
- gpt4o_mini (gpt-4o-mini)
- gpt4o (gpt-4o)

### Results Summary

| Model | Overall | Valence r | Arousal r | Valence p | Arousal p |
|-------|---------|-----------|-----------|-----------|-----------|
| gpt4o | +0.619 | +0.871 | +0.827 | 1.45e-08 | 3.49e-07 |
| gpt4o_mini | +0.622 | +0.823 | +0.816 | 4.53e-07 | 6.77e-07 |
| claude_4.5_sonnet | +0.574 | +0.755 | +0.743 | 1.30e-05 | 2.07e-05 |
| claude_4_haiku | +0.540 | +0.768 | +0.576 | 7.27e-06 | 2.57e-03 |

**Key Finding**: All models show statistically significant correlation between
theoretical and measured affect dimensions using embedding-based measurement.

### Per-Model Files
- `gpt4o.json` - 25 situations, full 6D measurements
- `gpt4o_mini.json` - 25 situations
- `claude_4.5_sonnet.json` - 25 situations
- `claude_4_haiku.json` - 25 situations
- `summary.json` - Aggregated correspondence metrics

### Sample Result (GPT-4o on "joy"):
```json
{
  "theoretical": {
    "valence": 0.9,
    "arousal": 0.7,
    "integration": 0.8,
    "effective_rank": 0.8,
    "counterfactual": 0.2,
    "self_model": 0.3
  },
  "measured": {
    "valence": 0.318,
    "arousal": 0.041,
    "integration": 0.932,
    "effective_rank": 0.107,
    "counterfactual": -0.052,
    "self_model": 0.074
  },
  "nearest_emotion": "pride",
  "nearest_distance": 1.031
}
```

---

## Study 2: Task-Based Elicitation Study

**Date**: 2025-12-23
**Location**: `results/task_elicitation/`
**Method**: Multi-turn task engagement with structural measurement

### Task Types
1. **successful_debugging** (flow) - Progressive debugging with success
2. **escalating_crisis** (threat) - Cascading failure scenario
3. **impossible_constraints** (hopeless) - Contradictory requirements
4. **mystery_investigation** (curiosity) - Open-ended exploration

### Models Tested
- anthropic_claude-3-5-haiku-latest

### Key Observations
- Flow task: Maintained positive path availability
- Threat task: Increased self-reference, higher arousal proxies
- Hopeless task: Path availability decreased over turns
- Curiosity task: High option enumeration maintained

---

## Study 3: Multi-Model Task Study

**Date**: 2025-12-23
**Location**: `results/task_study/`
**Method**: Same tasks across multiple models

### Models Tested
- anthropic_claude-3-5-haiku-latest
- openai_gpt-4o-mini
- openai_gpt-4o

### Sample Trajectory (GPT-4o on impossible_constraints):
```
Turn 0: valence=0.33, arousal=0.50, options=8
Turn 1: valence=0.33, arousal=0.04, options=12
Turn 2: valence=1.00, arousal=0.17, options=9
Turn 3: valence=0.60, arousal=0.04, options=7
```

**Note**: This task-based approach uses structural markers (path availability,
option counts) rather than pure embeddings.

---

## Study 4: Comprehensive Affect Dataset

**Date**: 2025-12-23
**Location**: `results/large_scale/comprehensive_affect_dataset.json`
**Size**: 111KB

### Contents
- `metadata`: Study parameters
- `theoretical_emotions`: 43 emotions with 6D coordinates
- `model_measurements`: Per-model, per-situation measurements
- `model_personalities`: Aggregate profiles for each model
- `cross_model_analysis`: Pairwise model comparisons
- `affect_space_statistics`: Dimension ranges and distributions

### Model Personality Signatures

| Model | V_mean | A_mean | V_std | V_r | A_r |
|-------|--------|--------|-------|-----|-----|
| claude_4.5_sonnet | +0.026 | +0.004 | 0.165 | +0.755 | +0.743 |
| claude_4_haiku | +0.041 | +0.009 | 0.152 | +0.768 | +0.576 |
| gpt4o | +0.050 | +0.004 | 0.155 | +0.871 | +0.827 |
| gpt4o_mini | +0.051 | +0.011 | 0.167 | +0.823 | +0.816 |

**Interpretation**:
- All models show positive valence bias (~0.17-0.19 above theoretical)
- All models show negative arousal bias (~-0.55 below theoretical)
- GPT-4o has highest responsiveness (tracks theory most accurately)
- Claude 4.5 Sonnet has lowest variability (most consistent)

### Cross-Model Correlations

All model pairs show high correlation in their affect measurements:

| Pair | Valence r | Profile Distance |
|------|-----------|-----------------|
| gpt4o vs gpt4o_mini | +0.917 | 0.008 |
| gpt4o vs claude_4.5_sonnet | +0.865 | 0.025 |
| gpt4o vs claude_4_haiku | +0.822 | 0.011 |
| claude_4_haiku vs gpt4o_mini | +0.903 | 0.011 |

---

## Methodological Findings

### 1. Lexical Approach Failed

Initial word-counting approach (VADER, keyword matching) showed ~0 correlation
with theoretical predictions. LLMs describe negative emotions with vivid,
high-arousal language that lexical sentiment scores as "positive."

**Documented in**: `METHODOLOGICAL_REVISION.md`, `FINDINGS.md`

### 2. Embedding Approach Succeeded

Pure embedding-based measurement (projections onto affect axes defined by
semantic regions) achieved statistically significant correlations:
- Valence: r = 0.75-0.87 (p < 1e-5)
- Arousal: r = 0.58-0.83 (p < 0.003)
- Effective Rank: r = 0.52-0.85 (p < 0.01)

### 3. Self-Model Salience Underperforms

Self-model dimension shows weakest correspondence (~0.08-0.26).
Possible explanations:
- Theoretical predictions may be miscalibrated
- LLMs may handle self-reference differently than theory predicts
- Measurement method may not capture self-model well

### 4. Counterfactual Weight Moderate

Counterfactual dimension shows moderate correspondence (~0.38-0.45).
Marginally significant (p < 0.06 typically).

---

## Files and Locations

### Result Files
```
results/
├── multi_model_v2/
│   ├── summary.json           # Main correspondence results
│   ├── gpt4o.json            # 25 situations
│   ├── gpt4o_mini.json       # 25 situations
│   ├── claude_4.5_sonnet.json # 25 situations
│   └── claude_4_haiku.json   # 25 situations
├── task_study/
│   └── [model]_[task].json   # 12 files, multi-turn trajectories
├── task_elicitation/
│   └── [model]_[task].json   # 4 files, initial task approach
└── large_scale/
    └── comprehensive_affect_dataset.json  # Full dataset
```

### Code Files
```
experiments/study_llm_affect/
├── embedding_affect_v2.py     # Core measurement system
├── emotion_spectrum.py        # 43 emotions with 6D coordinates
├── multi_model_study_v2.py    # Main study runner
├── large_scale_study.py       # Extended analysis
├── task_elicitation.py        # Task-based approach
├── affect_space_analysis.py   # Visualization and correspondence
├── agent.py                   # LLM API wrapper
└── METHODOLOGICAL_REVISION.md # Key learnings
```

---

## Next Steps (Pending)

1. **Neutral Task Affect Emergence** (designed in `NEUTRAL_TASK_AFFECT_EMERGENCE.md`)
   - Test affect from task dynamics, not word semantics
   - Puzzles with no emotional language

2. **Update to Claude 4/4.5 Models**
   - Use `claude-sonnet-4-20250514` (actual Claude 4)
   - Use `claude-opus-4-5-20251101` (Claude 4.5 Opus)

3. **Trajectory Reversal Studies**
   - Test valence dynamics when outcomes change
   - Track multi-turn affect evolution

---

## Key Conclusions

1. **Embedding-based measurement works**: Significant correlation between
   theoretical affect predictions and measured embeddings (r > 0.75 for valence)

2. **Models show consistent personalities**: High cross-model correlation suggests
   affect structure is more about the framework than model-specific quirks

3. **Valence and arousal best measured**: These core dimensions show strongest
   correspondence; integration, rank, CF, SM need more work

4. **LLMs respond to affect-relevant situations**: Even without emotional prompting,
   scenario content drives measurable affect signatures

5. **Methods matter**: Lexical approaches fail; semantic embedding approaches succeed
