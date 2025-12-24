# Study C Results: Computational Derivation Test

**Author**: Jacob Valdez
**Date**: December 2024
**Status**: Initial experiment complete; results require extended replication

---

## Summary

This experiment tested whether six-dimensional affect structure emerges from computational necessity in self-modeling systems under viability constraints.

**Main Finding**: Self-modeling agents develop higher-dimensional representations than simple RL agents, but the effect is smaller than predicted (effective rank 2.2 vs 1.4, not 6 vs 2-3).

**Verdict**: Partial support for the thesis. The direction is correct but the magnitude is wrong.

---

## Methods

### Environment
- ViabilityWorld: Agent must maintain internal states (health, energy, temperature, stress) within viable bounds
- Death occurs if any state crosses boundary
- Partial observability, stochastic threats
- Reward = f(viability margin)

### Agents
1. **SimpleAgent**: Basic actor-critic (128-dim hidden layer)
2. **WorldModelAgent**: Adds latent world model (32-dim)
3. **SelfModelAgent**: Adds self-model + self-attention (32+16 dim)

### Training
- 500 episodes per agent
- PPO algorithm
- Same random seed for reproducibility

### Analysis
- Collected 1200+ state representations per agent
- PCA to extract principal components
- Computed effective rank: r_eff = (Σλ)² / Σλ²
- Correlated PCs with ground-truth viability variables

---

## Results

### Dimensionality

| Agent | Effective Rank | N(90% var) | N(95% var) |
|-------|---------------|------------|------------|
| Simple | **1.40** | 4 | 7 |
| World Model | **1.42** | 5 | 13 |
| Self Model | **2.21** | 17 | 27 |

**Observation**: Self-model agent has ~58% higher effective rank than simple agent. However, all agents have effective rank < 3, far below the predicted 6 dimensions.

### Variance Concentration

All agents show highly concentrated variance:
- SimpleAgent: PC1 explains **84.3%** of variance
- WorldModelAgent: PC1 explains **78.0%** of variance
- SelfModelAgent: PC1 explains **67.0%** of variance

The self-model agent distributes variance more evenly, but still has one dominant component.

### PC1 Correlations (All Agents)

| Variable | Simple | World Model | Self Model |
|----------|--------|-------------|------------|
| Viability distance | 0.67 | 0.73 | 0.72 |
| Energy | 0.97 | 0.97 | 0.97 |
| Temperature deviation | -0.92 | -0.87 | -0.86 |
| Threat active | -0.38 | -0.37 | -0.41 |
| Reward | 0.39 | 0.39 | 0.37 |

**Critical Finding**: PC1 consistently tracks viability across all agents. This supports the thesis claim that learned representations encode viability information.

### Dimensional Mapping

Theoretical predictions vs observed:

| Predicted Dimension | Expected Correlate | Observed? |
|--------------------|-------------------|-----------|
| Valence | Viability distance | **YES** (PC1 all agents) |
| Arousal | Threat presence | **Partial** (PC2-4 in some agents) |
| Integration | Response coherence | Not clearly identifiable |
| Effective rank | Behavioral diversity | Not clearly identifiable |
| Counterfactual weight | Planning activity | Not measured |
| Self-model salience | Self-state attention | Not clearly identifiable |

---

## Interpretation

### What the Results Support

1. **Viability tracking emerges reliably**. All agents, regardless of architecture, develop representations that correlate with distance to viability boundary. This is consistent with the thesis claim that valence = viability gradient.

2. **Self-modeling increases dimensionality**. The self-model agent has higher effective rank, more distributed variance, and requires more PCs to explain the same variance. This directionally supports the prediction.

### What the Results Do Not Support

1. **Six-dimensional structure**. Effective rank is ~2, not ~6. Even with generous interpretation (using N(90%) instead of effective rank), we get 4-17 dimensions—not a clear 6.

2. **World model adding dimensions**. World model agent has nearly identical dimensionality to simple agent (1.42 vs 1.40). The thesis predicts world models should add 1-2 dimensions.

3. **Identifiable affect dimensions**. Beyond valence (PC1) and possibly arousal (threat-correlated PCs), the other four dimensions are not clearly identifiable in the learned representations.

---

## Limitations

### Experimental

1. **Insufficient training**: 500 episodes may not be enough for complex representations to emerge. Published RL work typically uses 10^6+ timesteps.

2. **Simple environment**: ViabilityWorld may not require the computational sophistication that would necessitate 6D structure. A more complex environment with:
   - Planning horizons requiring counterfactual reasoning
   - Social interactions requiring self-modeling
   - Temporal credit assignment requiring integration

   ...might produce richer representations.

3. **Architecture constraints**: Explicit separation into world/self components may not be how these dimensions naturally emerge. End-to-end learning might produce different structure.

### Analytical

1. **Linear analysis**: PCA assumes linear structure. The six dimensions might be nonlinearly entangled.

2. **Effective rank is one metric**: Other measures (participation ratio, intrinsic dimensionality) might tell different stories.

3. **Ground truth proxies**: We don't have direct measures of "integration" or "counterfactual weight" to correlate against.

---

## Conclusions

### For the Thesis

The computational derivation argument is **partially supported but not validated**.

The thesis should be revised to make weaker claims:
- ✓ "Self-modeling systems develop representations that track viability"
- ? "Self-modeling adds computational dimensions" (true but effect size unclear)
- ✗ "Six specific dimensions emerge from computational necessity" (not demonstrated)

### For Future Work

1. **Scale up**: Train for 10^6+ steps with larger networks
2. **Richer environments**: Add planning requirements, social dynamics
3. **Ablation studies**: Systematically vary environment complexity
4. **Nonlinear analysis**: Use autoencoders, manifold learning
5. **Direct probes**: Design tasks that specifically require each dimension

### Scientific Status

This experiment demonstrates that the thesis makes testable predictions that can be partially falsified. The 6D claim is not supported by this evidence. The viability-tracking claim is supported.

This is how science should work: make predictions, test them, revise theory based on results.

---

## Data Availability

All code, trained models, and analysis results are available in this repository.

To reproduce:
```bash
cd empirical/experiments/study_c_computational
uv sync
uv run python train_agents.py
uv run python analyze_representations.py
```
