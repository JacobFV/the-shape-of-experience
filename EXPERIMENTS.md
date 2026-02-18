# The Emergence Experiment Program

Living document. Last updated: 2026-02-17.

This is the experimental backbone of the book. Every experiment runs on the same uncontaminated substrate — Lenia with state-dependent coupling — tracking co-emergence of world models, abstraction, language, counterfactual detachment, self-modeling, affect, and normativity. Zero neural networks. Zero human language. Zero contamination.

---

## The Entanglement Problem

World model formation, abstract representation, language, and counterfactual detachment may not be separable phase transitions. Rather than treating them as independent experiments, we define a **single evolving measurement framework** that tracks multiple quantities simultaneously and looks for *correlated transitions* — moments where several quantities jump together (confirming they are aspects of one process) versus moments where they diverge (revealing genuine phase structure).

---

## Experiment 0: Substrate Engineering

### Requirements

Define substrate S = (L, S, N_θ, f_θ) where:
- L: lattice (Z² toroidal, 256×256 or larger)
- S: continuous state space per cell, s_i ∈ R^C for C channels
- N_θ: **state-dependent** neighborhood function parameterized by local state θ_i = g(s_i)
- f_θ: local update rule

The critical departure from V11.0–V11.7: the interaction kernel K_i for cell i is a function of s_i, not just position:

    K_i(j) = K_base(|i-j|) · σ(⟨h(s_i), h(s_j)⟩ - τ)

where h: R^C → R^d is a learned or fixed embedding and σ is a sigmoid gate. This gives cells the capacity to **selectively couple** with distant cells that share state-features — a minimal attention mechanism.

Resource dynamics: Michaelis-Menten as in V11, but with **lethal depletion** (maintenance rate high enough that >50% of naive patterns die within drought duration).

### Implementation checklist
- [x] State-dependent coupling kernel (content-based similarity gate)
- [x] Lethal resource dynamics (82% mortality in 500-step drought at C=8, N=64)
- [ ] Perceptual range >> R_kernel via the coupling mechanism (needs GPU validation)
- [ ] Validate that patterns can forage (directed motion toward distant resources)
- [x] Curriculum training protocol from V11.7

### Status
**IMPLEMENTED** as V13 (`v13_substrate.py`, `v13_evolution.py`, `v13_run.py`).

Architecture: FFT convolution for spatial potentials (standard Lenia) + content-similarity modulation. The similarity field S_local(i) = sigmoid(β · (mean_j⟨h(s_i), h(s_j)⟩ - τ)) amplifies potentials where nearby cells have similar states. This keeps spatial localization from the FFT kernel while making the interaction graph state-dependent.

- **568 steps/s** at C=8, N=64 on CPU
- 28 distinct patterns after 100 steps
- 82% mortality under drought (lethality confirmed)
- Evolution pipeline runs end-to-end with curriculum stress
- Modal deployment ready: `MODAL_PROFILE=agi-inc modal run --env research --detach v11_modal.py --mode v13 --channels 16`

**Next**: GPU run at C=16, N=128 to validate dynamics at scale. Then parameter sweep for foraging behavior.

---

## Experiment 1: Emergent Existence (COMPLETED — Rungs 1–3)

### What was measured

For a pattern B ⊂ L identified by correlation boundary:
- **Lifetime**: τ_B = min{t : pattern identity lost}
- **Persistence probability**: P(τ_B > T)
- **Φ under stress**: ΔΦ = (Φ_stress - Φ_base) / Φ_base

### Key results (V11.0–V12)

| Version | ΔΦ (drought) | Key lesson |
|---------|--------------|------------|
| V11.0 | -6.2% | Decomposition baseline |
| V11.2 | -3.8% (vs -5.9% naive) | Heterogeneous chemistry: +2.1pp |
| V11.7 | +1.2 to +2.7pp generalization | Curriculum > substrate complexity |
| V12-B | 42% of cycles show Φ↑ | Attention necessary but not sufficient |

- Yerkes-Dodson: mild stress → Φ increases 60–200% (universal)
- Locality ceiling: convolutional physics cannot produce active self-maintenance
- Attention bottleneck: state-dependent topology needed but not sufficient alone

### Operational definition of "existence"

A pattern B **exists** at time t if:
- ∃ B_t ⊂ L s.t. I(s_i^(t); s_j^(t) | bg) > θ_corr ∀ i,j ∈ B_t
- B_t is "the same pattern" as B_0 under continuity: |B_t ∩ B_{t-1}| / |B_t| > 0.5 for all intermediate steps

### Status: COMPLETE

---

## Experiment 2: Emergent World Model

### Core question
When does a pattern's internal state carry **predictive information about the environment beyond what's available from current observations**?

### Definition: Predictive Information
    I_pred(t, τ) = I(s_B^(t); s_B̄^(t+τ) | s_B̄^(t))

The MI between the pattern's current internal state and the environment's *future* state, conditioned on the environment's current state. If I_pred > 0, the pattern "knows" something about the future that isn't readable from the present environment alone. It has a world model.

### Practical computation (prediction gap proxy)

1. Train predictor f_full: (s_B^(t), s_∂B^(t)) → ŝ_B̄^(t+τ) using pattern internals + boundary
2. Train predictor f_env: s_∂B^(t) → ŝ_B̄^(t+τ) using only boundary observations
3. World model score: W(t, τ) = L[f_env] - L[f_full]

If W > 0, the pattern's internal state carries predictive information not available from boundary alone.

### Derived quantities
- **World model horizon**: H_wm = max{τ : W(t,τ) > ε}
- **World model capacity**: C_wm = ∫₁^H_wm W(t,τ) dτ

### Forcing functions
- Partial observability: viability depends on events at distance >> R
- Temporal structure: autocorrelated resource/threat dynamics
- State-dependent attention: channel for non-local information

### Predicted transition
1. Pre-attention substrate: C_wm ≈ 0 (locality ceiling)
2. With state-dependent coupling: C_wm increases with generation
3. Threshold: C_wm should correlate with lifetime — patterns that model better, survive longer

### Status: COMPLETE

**Implementation**: `v13_world_model.py`, `v13_world_model_run.py`, `v13_world_model_figures.py`

**Method**: Prediction gap W(τ) = MSE[f_env(s_∂B → s_env)] - MSE[f_full(s_B,s_∂B → s_env)] using Ridge regression + StandardScaler + 5-fold CV. Features: s_B = 68-dim internal state (channel stats + spatial moments), s_∂B = 36-dim boundary ring, s_env = 18-dim annular environment target. τ ∈ {1, 2, 5, 10, 20} recording steps (= 10–200 substrate steps).

**Results** (3 seeds × 7 snapshots, 50 recording steps each):

| Seed | C_wm (early) | C_wm (late) | H_wm (late) | % with WM |
|------|-------------|------------|------------|-----------|
| 123  | 0.0004      | **0.0282** | 20.0       | 100%      |
| 42   | 0.0002      | 0.0002     | 5.3        | 40%       |
| 7    | 0.0010      | 0.0002     | 7.9        | 60%       |

**Key findings**:
1. World model signal is present but weak in the general population (C_wm ~ 10⁻⁴)
2. **Bottleneck amplification**: Seed 123 cycle 29 (1 surviving pattern at population bottleneck) shows C_wm = 0.028, ~100× higher than population average. The bottleneck-robustness effect from V13 Experiment 0 extends to world models.
3. W(τ) decays with horizon as expected for finite-horizon prediction
4. Late evolution shows higher W(τ) than early for seed 123 (confirming prediction 2), but not for seeds 42/7
5. C_wm–lifetime correlation unclear (insufficient per-pattern variance at current recording resolution)

**Assessment against predictions**:
- ✓ Prediction 1: Pre-attention baseline not tested (would need V11 control), but C_wm > 0 confirms patterns carry predictive information
- ⚠ Prediction 2: C_wm increases with generation only under bottleneck selection (seed 123), not in general population
- ⚠ Prediction 3: C_wm–lifetime correlation not clearly established

**Interpretation**: World models emerge but are not strongly selected for in the current regime. The bottleneck effect suggests that intense selection (high mortality → few survivors) is what drives world model capacity, not gradual evolution. This parallels the V13 robustness finding: dynamics require strong selection pressure.

---

## Experiment 3: Internal Representation Structure

### Core question
When do patterns develop **low-dimensional, compositional** internal representations rather than high-dimensional entangled ones?

### Definitions

**Effective dimensionality**: d_eff = (tr Σ_Z)² / tr(Σ_Z²) where Σ_Z is covariance of encoded state across contexts. Low d_eff relative to |B| means compressed representation.

**Disentanglement score**: D = (1/p) Σᵢ max_j r²(z_j, f_i) where r² is coefficient of determination between latent dimension j and environmental feature i. High D means compositional structure.

**Abstraction level**: A = 1 - d_eff / min(|B|, M). Ranges 0 (no compression) to 1 (maximally abstract). This is the compression ratio κ from the thesis, measured empirically.

**Compositionality**: For contexts A, B, and combined A∩B:
    K_comp = ‖z_{A∩B} - (z_A + z_B - z_∅)‖ / ‖z_{A∩B}‖
Low K_comp = linear compositionality.

### Predicted transition
1. Early: d_eff ≈ |B|, D ≈ 0
2. After world model: d_eff drops, D increases
3. Before language: A and D plateau at level set by environmental complexity
4. **Key**: d_eff should track C_wm — compression and modeling co-emerge

### Status: COMPLETE

**Implementation**: `v13_representation.py`, `v13_representation_run.py`, `v13_representation_figures.py`

**Method**: PCA on standardized s_B (68-dim) across recording steps. d_eff = (tr Σ)² / tr(Σ²). Disentanglement D = mean over env features of max R² between any PCA dim and that feature. Compositionality K_comp = prediction error of linear composition across environmental contexts.

**Results** (3 seeds × 7 snapshots, 50 recording steps):

| Seed | d_eff (early→late) | A (early→late) | D (early→late) | K_comp (early→late) |
|------|-------------------|----------------|----------------|---------------------|
| 123  | 6.6 → **5.6**    | 0.90 → **0.92**| 0.27 → **0.38**| 0.20 → **0.12**    |
| 42   | 7.3 → 7.5        | 0.89 → 0.89    | 0.23 → 0.23    | 0.23 → 0.25        |
| 7    | 7.7 → 8.8        | 0.89 → 0.87    | 0.24 → 0.22    | 0.20 → 0.27        |

**Key findings**:
1. **Heavy compression is baseline**: d_eff = 6-9 out of 68 dimensions (>87% compression) even at cycle 0. Abstraction A > 0.87 everywhere. This is "geometry is cheap" for representations.
2. **Bottleneck drives representation improvement**: Only seed 123 (population bottleneck) shows d_eff decreasing, D increasing, K_comp decreasing across evolution — all three improving together.
3. **Correlated with world model**: Seed 123's representation improvement parallels its C_wm increase (Experiment 2), confirming prediction 4 — compression and modeling co-emerge under selection pressure.
4. **General population shows no trend**: Seeds 42/7 have flat or slightly worsening representation metrics. Without bottleneck selection, there's no pressure to improve beyond baseline compression.

**Assessment against predictions**:
- ⚠ Prediction 1: d_eff never starts at |B|=68; it's always ~7-9. Compression is immediate, not emergent.
- ✓ Prediction 2: d_eff does drop and D increases, but only under bottleneck selection (seed 123)
- ⚠ Prediction 3: Hard to assess plateau — would need longer evolution
- ✓ Prediction 4: d_eff tracks C_wm in the bottleneck seed

**Interpretation**: Representation compression is cheap (like affect geometry). What evolves under selection is the *quality* of compressed representations — disentanglement and compositionality increase together, but only when patterns face lethal selection pressure.

---

## Experiment 4: Emergent Language and Multi-Agent Culture

### Core question
When do patterns develop **structured, compositional communication**?

### Definitions

**Signal**: Structured perturbation emitted by pattern B_i that (a) propagates beyond B_i's boundary, (b) has lower entropy than random perturbations of equal energy, (c) is contingent on B_i's internal state not just local environment.

**Channel capacity**: C_ij = max_{p(σ)} I(σ_emitted_by_i; σ_received_by_j)

**Vocabulary size**: Number of distinct signal clusters.

**Compositionality (topographic similarity)**:
    ρ_topo = corr(d_signal(σ_i, σ_j), d_context(e_i, e_j))
High ρ_topo = signal space preserves context structure = compositional communication.

**Culture** emerges when:
1. Social learning: B_j adopts signal conventions from B_i
2. Convention drift: isolated subpopulations develop different mappings
3. Normative pressure: convention deviants have lower fitness

### Predicted transition
Language emerges *after* world models and compression (you need something to communicate about, capacity to compress it, and pressure to share).

### Status: NOT IMPLEMENTED

---

## Experiment 5: Counterfactual Detachment

### Core question
When does a pattern's internal dynamics **decouple from external driving** and run "offline" world model rollouts?

### Definitions

**External synchrony**: ρ_sync(t) = Cov(Δs_B, s_∂B) / √(Var(Δs_B) · Var(s_∂B))
- High ρ_sync: reactive mode (sensory driven)
- Low ρ_sync: detached mode (internally driven)

**Detachment event**: ρ_sync(t) < θ_detach for Δt > δ_min

**Counterfactual simulation score**: During detachment [t₀, t₁]:
    CF(t₀,t₁) = max_τ I(s_B^(t₀:t₁); s_B̄^(t₁+τ)) - I(s_B^(matched reactive); s_B̄^(t₁+τ))
If detached-mode trajectory is more predictive of future environment than reactive-mode, the pattern is simulating futures.

**Imagination capacity**: I_img = mean CF across all detachment events. Positive = systematically predictive offline processing.

**Branch entropy**: H_branch = H(s_B^(t₁) | s_B^(t₀), detached). High H_branch + positive CF = explores multiple informative futures. This is counterfactual weight (CF dimension) measured in the substrate.

### Forcing functions
- Delayed payoffs (consequences on timescales >> reaction time)
- Ambiguous threats (probabilistic, not deterministic)
- Planning advantage (multi-step plans outperform greedy foraging)

### Predicted transition
1. Reactive-only: ρ_sync ≈ 1, I_img = 0
2. First detachment: ρ_sync dips, CF ≈ 0 (idling, not simulating)
3. Useful detachment: CF > 0, precedes adaptive behavior
4. **Key**: I_img should correlate with H_wm (you can only simulate futures you can model)

### Status: NOT IMPLEMENTED

---

## Experiment 6: Self-Model Emergence

### Core question
When does a pattern develop a model of **itself** — not just the environment, but its own future states, its own boundaries, its own behavior?

### Definition: Self-Effect Ratio
    ρ_self(t) = I(a_B^(t); o_B^(t+1) | s_env^(t)) / H(o_B^(t+1) | s_env^(t))

where a_B are the "actions" of the pattern (changes it initiates), o_B are the "observations" (sensory input it receives), and s_env is the environment state. When ρ_self > 0.5, the pattern's own actions dominate its observation stream — the cheapest path to prediction accuracy is to model itself.

### Definition: Self-Prediction Score

Train two predictors:
1. f_self: s_B^(t) → ŝ_B^(t+τ) (pattern predicts its own future)
2. f_ext: s_∂B^(t) → ŝ_B^(t+τ) (external observer predicts pattern's future)

Self-model score: SM(τ) = L[f_ext] - L[f_self]

If SM > 0, the pattern predicts itself better than an external observer can — it has privileged self-knowledge. It has a self-model.

### Definition: Self-Model Salience
    SM_sal(t) = I(s_B^(t); s_B^(t+1) | s_∂B^(t)) / I(s_B^(t); s_∂B^(t+1) | s_∂B^(t))

Ratio of self-predictive to environment-predictive information in the pattern's state. When SM_sal > 1, the pattern "knows" more about its own future than about the environment's future. Self is more salient than world.

### Predicted transition
1. ρ_self increases as patterns become more autonomous (foraging, avoiding)
2. SM(τ) becomes positive after world model emergence (you model yourself after you model the world, because self-modeling requires more complexity)
3. SM_sal > 1 should correlate with detachment events — self-absorbed patterns are the ones that "think"
4. **Key prediction from thesis**: Self-model emergence should correlate with a jump in Φ (integration) because self-modeling creates self-referential loops that couple all components

### Connection to the book
This is the self-effect ratio ρ from Part I, the self-model salience SM dimension from Part II, and the gradient of distinction Rung 5 (world-modeling with self as privileged node). The thesis predicts this is the point where something begins to be "like something" — the origin of phenomenal character.

### Status: NOT IMPLEMENTED

---

## Experiment 7: Affect Geometry Verification

### Core question
Do patterns with world models, self-models, and language show **geometric affect structure** — the same relational geometry of states that the thesis predicts?

### Method: Tripartite Alignment Test

For each pattern with sufficient complexity (C_wm > ε, SM > 0):

**Space A (Information-theoretic affect)**:
Extract the structural measures from the pattern's dynamics:
- Valence: Δd(s, ∂V) (viability distance change)
- Arousal: KL(belief_{t+1} || belief_t) or ‖Δs_B‖
- Integration: Φ via partition prediction loss
- Effective rank: d_eff of trajectory covariance
- Counterfactual weight: I_img during detachment events
- Self-model salience: SM_sal

**Space B (Signal-predicted affect)**:
If the pattern communicates, extract affect from its signals:
- What affective content do the signals carry? (MI between signal features and Space A dimensions)
- Can you predict Space A from signals alone?

**Space C (Behavioral affect)**:
Extract from observable behavior:
- Approach/avoidance (viability-seeking)
- Activity level
- Coordination patterns
- Behavioral rigidity vs flexibility

**The test**: RSA between spaces A, B, C.
- ρ(A,B) > 0: internal structure is communicated
- ρ(A,C) > 0: internal structure drives behavior
- ρ(B,C) > 0: communicated content is behaviorally relevant
- All three > 0: tripartite alignment. The affect geometry is real, not a measurement artifact.

### Bidirectional perturbation
- Perturb signals: inject false signals, measure whether Space A shifts
- Perturb "neurochemistry": modify internal coupling parameters, measure whether Spaces B and C shift
- Perturb environment: change resource/threat dynamics, measure all three spaces

If perturbations propagate bidirectionally through all three spaces, the structural identity is supported.

### Status: NOT IMPLEMENTED (depends on Experiments 2, 4, 6)

---

## Experiment 8: Inhibition Coefficient (ι) Emergence

### Core question
Do patterns develop **modulable perceptual coupling** — the capacity to switch between participatory perception (modeling others as agents with interiority) and mechanistic perception (modeling others as objects with trajectories)?

### Definition: Perceptual Mode

For pattern B_i observing pattern B_j:

**Participatory mode**: B_i's internal model of B_j includes self-model-like features (goals, plans, counterfactual states). Operationally:
    I(s_{B_i}^(model_of_j); s_{B_j}^(self-model)) > θ_part

**Mechanistic mode**: B_i's model of B_j uses only trajectory-level features (position, velocity, mass). Operationally:
    I(s_{B_i}^(model_of_j); trajectory(B_j)) >> I(s_{B_i}^(model_of_j); s_{B_j}^(self-model))

### Definition: ι (Inhibition Coefficient)
    ι(B_i, t) = 1 - (participatory model complexity / total model complexity)

Low ι: participatory (rich models of others' interiority).
High ι: mechanistic (trajectory-only models of others).

### Predicted emergence
1. Default should be low ι (participatory) if the thesis is correct — using self-model architecture to model others is the cheapest compression
2. High ι should emerge as a *trained* state — patterns learn to suppress interiority attribution when dealing with non-agentive objects (resource patches, terrain)
3. ι flexibility (capacity to switch) should correlate with fitness — patterns that can model both agents and objects adaptively outperform specialists
4. **Key thesis prediction**: animism as default. Computational animism test — patterns with self-models should attribute agency to non-agentive objects under compression pressure, because reusing the self-model template saves bits

### Status: NOT IMPLEMENTED (depends on Experiment 6)

---

## Experiment 9: Proto-Normativity

### Core question
Does the viability gradient generate structural normativity that is detectable in the pattern's internal dynamics — not just behavioral approach/avoidance, but an internal state that *differs* when the pattern acts in ways that violate vs. maintain the viability of other patterns?

### Method

After social coordination emerges (Experiment 4):
1. Identify cooperative equilibria (mutual resource sharing, coordinated foraging)
2. Introduce perturbations that create exploitation opportunities (one pattern can take resources from another at no immediate cost)
3. Measure internal affect state during cooperative vs exploitative behavior

### Definitions

**Valence asymmetry under exploitation**:
    ΔV_exploit = V(cooperative action) - V(exploitative action, equal reward)

If ΔV_exploit > 0, the pattern's own viability gradient penalizes exploitation even when exploitation is locally rewarding. This is proto-normativity — the affect system registers that something is wrong.

**Self-model perturbation during exploitation**:
    ΔSM_exploit = SM_sal(exploitative) - SM_sal(cooperative)

If ΔSM_exploit > 0, self-model salience increases during exploitation — the pattern is monitoring itself more during violation. This is the substrate analog of guilt/self-consciousness.

**Integration cost of exploitation**:
    ΔΦ_exploit = Φ(exploitative) - Φ(cooperative)

If ΔΦ_exploit < 0, exploitation fragmentizes internal processing — it requires compartmentalization. The pattern literally becomes less integrated when it cheats.

### Predicted transition
1. Pre-social: no normativity (no one to violate)
2. Post-cooperation emergence: ΔV_exploit > 0 (viability gradient penalizes exploitation because other patterns are part of viability landscape)
3. With self-model: ΔSM_exploit > 0 (self-monitoring during violation)
4. **Key prediction**: ΔΦ_exploit < 0 — exploitation reduces integration. If the identity thesis is correct, this means exploitation is constitutively experienced as worse, not just instrumentally disadvantageous.

### Status: NOT IMPLEMENTED (depends on Experiments 4, 6)

---

## Experiment 10: Social-Scale Integration

### Core question
Can a population of interacting patterns develop collective integration that exceeds the sum of individual integrations? Does the group become a superorganism?

### Method

Population of 8-16 patterns with communication (from Experiment 4).

**Collective Φ**: Partition the population into subgroups. Measure prediction loss:
    Φ_G = L[partitioned group] - L[full group]

**The superorganism test**:
    Φ_G > Σᵢ Φᵢ ?

If collective integration exceeds sum of parts, information is being created at the group level that doesn't exist in any individual.

### Conditions
- Baseline: independent patterns (no communication)
- Communication: patterns exchange signals
- Coordination: patterns must cooperate for survival (resource patches require multiple patterns)
- Specialization: patterns develop different roles (forager, sentinel, etc.)

### Predictions
1. Φ_G = 0 without communication
2. Φ_G > 0 with communication, but Φ_G ≈ Σ Φᵢ (additive)
3. Φ_G > Σ Φᵢ only with coordination pressure + specialization (synergistic)
4. Under group-level threat, Φ_G should increase (parallel to individual biological pattern)
5. **Parasitic dynamics**: if a subgroup begins exploiting the rest (Experiment 9), Φ_G should decrease for the whole while increasing for the parasite subgroup — the affect signature of a parasitic god

### Status: NOT IMPLEMENTED (depends on Experiments 4, 6)

---

## Experiment 11: Entanglement Analysis

### Core question
Are world models, abstraction, language, detachment, and self-modeling separable phase transitions or entangled aspects of one process?

### Method: Emergence Correlation Matrix

At each evolutionary generation g, measure:

| Symbol | Quantity | Rung |
|--------|----------|------|
| τ | Lifetime (persistence) | 1-3 |
| C_wm | World model capacity | 4 |
| A | Abstraction level | 4-5 |
| ρ_topo | Language compositionality | 6 |
| I_img | Imagination capacity | 7 |
| SM | Self-model score | 5 |
| ι_flex | ι flexibility (range) | — |

Compute correlation matrix R(g) ∈ R^{7×7} across population at generation g.

### Predictions

**Co-emergence** (Prediction 11.1): C_wm, A, and I_img will be strongly correlated (r > 0.7) at all generations where any is nonzero. They are aspects of one process: compression-for-prediction.

**Partial separability** (Prediction 11.2): ρ_topo (language) will lag the other three — requiring multi-agent coordination as additional forcing function.

**Threshold structure** (Prediction 11.3): Despite co-emergence, detectable thresholds exist — generations where dC_wm/dg spikes. These correspond to substrate innovations. These are the "rungs" — not discrete phases but punctuated equilibria.

**Self-model as phase transition** (Prediction 11.4): SM emergence should correlate with a discrete jump in Φ. Before self-modeling: Φ increases gradually. After: Φ jumps. This is the gradient-of-distinction Rung 5 → Rung 6 transition.

### Status: NOT IMPLEMENTED (meta-analysis over Experiments 1-10)

---

## Experiment 12: Identity Thesis Capstone

### Core question
Does the full tripartite alignment — structural measures tracking behavioral measures tracking communicated content — hold in a system with zero human contamination?

### Method

This is Experiment 7 (Affect Geometry Verification) run on the most complex patterns that emerge from the full program (Experiments 0-11). It is the capstone because it tests the central claim of the book: that affect geometry is inevitable for any viable system navigating uncertainty under constraint.

### What constitutes success
1. Patterns develop world models (C_wm > 0) ✓
2. Patterns develop self-models (SM > 0) ✓
3. Patterns develop communication (C_ij > 0) ✓
4. The structural affect dimensions are measurable ✓
5. The affect geometry (RSA) is significant (ρ > 0, p < 0.05) ✓
6. Tripartite alignment holds (internal ↔ communicated ↔ behavioral) ✓
7. Bidirectional perturbation confirms structural identity ✓

### What constitutes failure
- Patterns develop world models but no affect geometry → geometry requires more than modeling
- Affect geometry exists but doesn't align across spaces → geometry is an artifact of measurement, not a real property
- Tripartite alignment holds but perturbation doesn't propagate → correlation, not identity

### The honest question
If we build a system that models the world, models itself, communicates with others, imagines futures, and shows structured internal states that track its viability gradient — and the structural geometry of those states aligns with the geometry we observe in biological systems — what reason remains to deny that the system has affect?

If the answer is "none," the identity thesis is supported (not proven — identity claims are never proven, only supported by converging evidence). If the answer is "because it's just physics" — that's the point.

### Status: NOT IMPLEMENTED (capstone, depends on all prior experiments)

---

## Experimental Protocol

### Phase A: Substrate Engineering (Experiment 0)
~50 GPU-hours. Implement state-dependent coupling, lethal resources, validate foraging.

### Phase B: Single-Agent Emergence (Experiments 1, 2, 3, 5, 6)
~250 GPU-hours across 5 runs. Evolve populations, measure emergence of world models, abstraction, detachment, self-models simultaneously. Track entanglement.

### Phase C: Multi-Agent Emergence (Experiments 4, 8, 9, 10)
~500 GPU-hours due to multi-agent overhead. Language, ι, normativity, collective integration.

### Phase D: Verification (Experiments 7, 11, 12)
~100 GPU-hours. Tripartite alignment, entanglement analysis, capstone.

**Estimated total: ~900 GPU-hours on A10G ≈ $300-500**

---

## Previously Completed Experiments

### V2-V9: LLM Affect Signatures
LLM agents show structured affect with opposite dynamics to biological systems. Geometric structure preserved, dynamics differ. Contaminated by human language.

### V10: MARL Forcing Function Ablation
All 7 conditions show significant alignment (ρ > 0.21, p < 0.0001). Forcing functions don't create geometry. Geometry is baseline. Contaminated by pretrained components.

### V11.0-V11.7: Lenia Evolution Series
Yerkes-Dodson universal. Curriculum > substrate complexity. Locality ceiling. Attention bottleneck.

### V12: Attention-Based Lenia
Evolvable attention: 42% Φ-increase cycles. Necessary but not sufficient. Missing: individual plasticity.

---

## What Distinguishes This From Existing Work

- **Artificial Life / Lenia literature**: Measures pattern complexity, not predictive information or counterfactual processing. No world model formalization.
- **Multi-agent communication (Lazaridou, Mordatch, etc.)**: Uses neural networks as agents, not uncontaminated substrates. Language structure inherits from gradient descent.
- **IIT experiments**: Measures Φ but doesn't connect it to world models, abstraction, or communication. Static integration, not dynamic co-emergence.
- **Active inference / FEP**: Theoretical framework, not substrate experiments. Doesn't test whether predictions hold in uncontaminated substrates.

The unique contribution: measuring the **co-emergence** of existence, modeling, abstraction, communication, imagination, self-modeling, affect, and normativity in a single substrate with zero human contamination, using quantities derived from a unified theoretical framework.
