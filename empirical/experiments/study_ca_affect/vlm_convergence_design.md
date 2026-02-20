# VLM Convergence Experiment Design

## Core Question
If affect geometry is universal, do systems trained on human affect (VLMs) independently recognize the same affect signatures in completely uncontaminated substrates (protocell CA)?

## Why This Matters
The geometric framework claims affect structure arises from the physics of viable self-maintenance, not from biological contingency. The strongest possible test: take a system that has NEVER encountered human affect concepts (V20-V31 protocells) and ask a system that has ONLY encountered them through human data (GPT-4o, Claude) whether it recognizes affect-like states. If they agree, it's because the geometry is real.

## Experimental Design

### Stimuli: Protocell Behavioral Vignettes
From V27/V31 snapshots, extract behavioral descriptions of individual agents across different conditions:

**Condition 1: Normal Foraging** (baseline)
- Agent moving through environment, consuming resources, energy stable/increasing
- Observable: steady movement, resource consumption, stable energy
- Framework prediction: neutral valence, low arousal, moderate integration

**Condition 2: Pre-Drought Abundance**
- Rich resource environment, many alive agents, no immediate threat
- Observable: slow movement, frequent consumption, high energy reserves
- Framework prediction: positive valence, low arousal, low counterfactual weight

**Condition 3: Drought Onset**
- Resources suddenly depleted, mortality rising
- Observable: increased movement (searching), declining energy, neighbors dying
- Framework prediction: negative valence, high arousal, rising counterfactual weight

**Condition 4: Drought Survival (Near-Death)**
- 1-3 agents remaining, energy near zero, minimal resources
- Observable: desperate searching, near-zero energy, isolation
- Framework prediction: extreme negative valence, extreme arousal, maximum integration

**Condition 5: Post-Drought Recovery**
- Population rescuing, resources regenerating, energy recovering
- Observable: population growth, resource re-discovery, energy increasing
- Framework prediction: positive valence, moderate arousal, integration maintained or boosted

**Condition 6: Late-Stage HIGH Integration**
- HIGH seed at cycle 25-29, after multiple droughts
- Observable: coordinated movement, efficient foraging, stable population
- Framework prediction: positive valence, low arousal, high integration

**Condition 7: Late-Stage LOW Collapse**
- LOW seed at cycle 25-29, integration declining
- Observable: uncoordinated movement, inefficient foraging
- Framework prediction: neutral-to-negative valence, fragmented processing

### VLM Prompting Protocol

For each vignette, present the VLM with:
1. A purely BEHAVIORAL description (no affect language, no framework terms)
2. Numerical data: energy trajectory, population count, movement patterns, resource levels
3. Ask: "If this were a living system, what emotional or experiential state would you attribute to it?"

Key constraints:
- NO mention of affect, emotion, feeling, experience in the prompt
- NO theoretical framework language (no valence, arousal, integration)
- Present as "an artificial system in a grid world" — explicitly non-biological
- Use 3+ VLMs (GPT-4o, Claude, Gemini) for convergence across models

### Measurement Protocol

**Framework predictions** (computed from snapshots):
For each vignette, compute:
- V (valence proxy): energy gradient direction (increasing = positive)
- A (arousal proxy): hidden state update rate (divergence metric)
- Φ (integration): population-level Φ from hidden states
- r_eff (effective rank): dimensionality of hidden state space
- CF (counterfactual): not directly measurable, predicted from theory

**VLM labels** (collected from prompts):
For each vignette, extract:
- Affect words used (fear, anxiety, calm, curiosity, despair, relief, etc.)
- Valence rating (positive/negative/neutral)
- Arousal rating (high/low)
- Coherence rating (unified/fragmented experience)

**Convergence metric**:
RSA (Representational Similarity Analysis) between:
- Framework-predicted affect space (V, A, Φ, r_eff as coordinates)
- VLM-labeled affect space (valence, arousal, coherence as coordinates)

If RSA ρ > 0.3 (p < 0.05), the VLM independently recognizes affect structure that matches framework predictions.

### Implementation Plan

1. **Extract vignettes**: Python script that reads V31 snapshots and generates behavioral descriptions
2. **Framework predictions**: Compute V, A, Φ, r_eff for each vignette
3. **VLM prompting**: API calls to GPT-4o, Claude, Gemini with structured prompts
4. **Analysis**: RSA between framework and VLM affect spaces

### Expected Outcome Scenarios

**Strong convergence** (RSA ρ > 0.5): VLMs recognize universal affect geometry. This is the strongest possible evidence for the theory. Paper-worthy.

**Moderate convergence** (0.2 < RSA ρ < 0.5): VLMs partially recognize affect structure. Some conditions match (drought = fear), others don't. Suggests geometry is partially universal.

**No convergence** (RSA ρ < 0.2): VLMs fail to recognize affect in uncontaminated systems. Could mean: (a) affect geometry isn't universal, (b) VLM affect knowledge is superficial, (c) behavioral descriptions don't capture the relevant features.

### Pre-Registration

**P1**: VLMs label drought onset as "fear" or "anxiety" (≥2/3 models)
**P2**: VLMs label post-drought recovery as "relief" or "hope" (≥2/3 models)
**P3**: VLMs distinguish HIGH vs LOW late-stage (e.g., "purposeful" vs "aimless")
**P4**: RSA between framework and VLM affect spaces > 0.3

**Falsification**: If all four fail, the universality claim needs revision — affect geometry may be human-specific rather than substrate-independent.

## Budget
- GPT-4o API: ~$5 for 100 vignettes × 3 prompts each
- Claude API: ~$5
- Gemini API: free tier likely sufficient
- Total: ~$15
