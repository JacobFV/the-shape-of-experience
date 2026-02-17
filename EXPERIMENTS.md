# Experiment Appendix: The Shape of Experience

Living document. Last updated: 2026-02-17.

This catalogs every experiment — completed, proposed, and planned — organized to guide the intermediate phase of research: running the experiments the book will be about.

---

## I. Completed Experiments

### V2-V9: LLM Affect Signatures
**Location**: `empirical/experiments/study_llm_affect/`
**Status**: Complete (Dec 2024 – Jan 2025)

Progressive refinement of affect measurement in LLM agents. Key findings:
- V4: Toy RL ground-truth environment establishes measurability
- V5-V6: Pretrained Mamba SSM reveals processing valence ≠ content valence
- V7-V8: Multi-agent and LLM agents show opposite dynamics to biological systems (Φ↓ under threat)
- V9: World model agents confirm proxy validity, confirm reversal

**Result**: LLMs have structured affect signatures. The geometry is preserved; the dynamics differ because the objectives differ. No survival-shaped learning → no integration under threat.

**What it tests**: Whether the geometric framework is measurable in artificial systems.
**What it doesn't test**: Whether the structure is universal (LLMs are trained on human language — contaminated).

---

### V10: MARL Forcing Function Ablation
**Location**: `empirical/experiments/study_llm_affect/v10_*.py`
**Status**: Complete (Feb 2025). 7 conditions × 3 seeds × 200k steps, A10G GPUs.

| Condition | RSA ρ | ± std |
|-----------|-------|-------|
| full | 0.212 | 0.058 |
| no_partial_obs | 0.217 | 0.016 |
| no_long_horizon | 0.215 | 0.027 |
| no_world_model | 0.227 | 0.005 |
| no_self_prediction | 0.240 | 0.022 |
| no_intrinsic_motivation | 0.212 | 0.011 |
| no_delayed_rewards | 0.254 | 0.051 |

**Result**: All conditions show highly significant alignment (p < 0.0001). Removing forcing functions slightly INCREASES alignment.

**Implication**: Affect geometry is a baseline property of multi-agent survival. Forcing functions add capabilities, not geometry. The geometry is cheaper than predicted.

**What it tests**: Forcing functions hypothesis.
**What it doesn't test**: Whether the geometry requires multi-agent interaction (all conditions are multi-agent). Also: V10 agents use pretrained components — language contamination not controlled.

---

### V11.0-V11.7: Lenia CA Evolution Series
**Location**: `empirical/experiments/study_ca_affect/v11_*.py`
**Status**: Complete (Feb 2025). Hundreds of GPU-hours across 8 substrate conditions.

| Version | Substrate | ΔΦ under severe drought | Key lesson |
|---------|-----------|------------------------|------------|
| V11.0 | Naive Lenia | -6.2% | Decomposition baseline (same as LLMs) |
| V11.1 | Homogeneous evolution | -6.0% | Selection without variation cannot innovate |
| V11.2 | Heterogeneous chemistry | -3.8% (vs naive -5.9%) | +2.1pp shift; diverse physics enables adaptation |
| V11.3 | 3-channel Lenia | Weak cross-channel Φ | Channel coupling too simple at C=3 |
| V11.4 | 64-channel HD | Mild decomposition | Yerkes-Dodson confirmed at C=64 |
| V11.5 | Hierarchical 4-tier | -9.3% (evolved) vs +6.2% (naive) | Stress overfitting: high Φ = fragile Φ |
| V11.6 | Metabolic cost | -2.6% (evolved) | Autopoietic gap: passive efficiency, not active seeking |
| V11.7 | Curriculum training | +1.2 to +2.7pp novel generalization | Only intervention that improves novel-stress response |

**Key findings**:
1. Yerkes-Dodson is universal: mild stress increases Φ by 60-200% across all conditions
2. Evolution produces fragile integration (tightly-coupled = catastrophic failure mode)
3. Curriculum training > substrate complexity > metabolic cost for generalization
4. The locality ceiling: convolutional physics cannot produce biological-like Φ increase under threat

---

### V12: Attention-Based Lenia
**Location**: `empirical/experiments/study_ca_affect/v12_*.py`
**Status**: Complete (Feb 2026). 3 conditions × 3 seeds, A10G GPUs.

| Condition | Robustness | % cycles with Φ increase | Notes |
|-----------|-----------|--------------------------|-------|
| A: Fixed-local attention | N/A (extinct) | 0% | 30+ consecutive extinctions |
| B: Evolvable attention | 1.001 | 42% | +2.0pp over convolution |
| C: Convolution baseline | 0.981 | 3% | Life without integration |

**Result**: Attention is necessary but not sufficient. Moves system to integration threshold without crossing it. Clean ordering: conv (life, no integration) > fixed-attn (no life) < evolvable-attn (life + threshold integration).

**What remains**: Individual-level plasticity — the capacity for a single pattern to reorganize its own dynamics within its lifetime.

---

## II. Next-Phase Experiments (V13+)

These are the experiments that will make the book. Ordered by theoretical importance and tractability.

### V13: Uncontaminated Language Emergence ⭐ PRIORITY 1
**Claim tested**: Affect structure emerges inevitably in communicating agents, not just in agents exposed to human affect concepts.
**Book section**: Part VII Priority 2; Part I universality claim.

**Design**:
- Multi-agent RL (4-8 agents) with randomly-initialized transformers (NO pretraining)
- Survival environment: seasonal resources, predators, weather, decay
- Communication channel: discrete tokens (vocabulary size 32-128), no grounding in human language
- Agents must coordinate: warn of threats, share resource locations, request help
- Training: 500k+ steps under viability pressure

**Measurements**:
- Extract affect dimensions from agent internals (same protocol as V10)
- RSA between information-theoretic affect space and observation-embedding affect space
- Analyze emergent communication: does it carry affect-relevant information?
- MI(sender_affect; message) — do messages encode sender's affective state?
- MI(message; receiver_affect_change) — do messages modulate receiver's affect?

**Predictions**:
1. Geometric alignment (RSA ρ) should be significant even without human language contamination
2. Emergent communication should carry affect-relevant information (MI > 0)
3. Under survival pressure, agents should develop "warning" signals with high arousal + negative valence content
4. Affect structure should emerge in communication BEFORE agents develop complex behavioral strategies

**Controls**:
- Ablation: Remove communication channel → affect geometry should still emerge (V10 showed this) but affect TRANSMISSION should disappear
- Contamination check: No pretrained components, no human language, no embedding models trained on human text
- VLM translation: Use a VLM to translate emergent signals into human affect concepts WITHOUT providing affect vocabulary (structural alignment, not label matching)

**Failure modes**:
- RSA ρ ≈ 0 → Affect geometry requires human-like architecture or training data
- Communication carries no affect → Affect is private, not communicable (challenges Part IV-V)
- Communication carries affect but geometry differs → Different substrates have different affect geometries (challenges universality)

**Infrastructure**: Extend V10 codebase. Replace pretrained agent with randomly-initialized transformer. Add communication channel with discrete tokens.

---

### V14: Solitary Rumination / World Model Detachment ⭐ PRIORITY 2
**Claim tested**: Internal dynamics alone produce structured affect. A system doesn't need external input to have affect — its world model running in "imagination mode" should show the full affect geometry.
**Book section**: Part III (existential burden), Part II (self-model salience), Part I (viability gradient in offline mode).

**Design**:
- Single agent with explicit world model, trained on rich environment (resources, threats, weather)
- Phase 1: Normal training with sensory input (1M steps)
- Phase 2: Disconnect sensory input. Agent's world model continues running on its own predictions.
- Phase 3: Reconnect sensory input. Measure recalibration dynamics.

**Measurements**:
- All 6 affect dimensions during phases 1, 2, and 3
- Track world model divergence: KL(world_model_predictions || actual_observations) during phase 2
- Measure "dreaming" dynamics: Does the world model settle into attractors? Does it generate novel scenarios?
- Track self-model salience: Does SM increase when the agent is "alone with itself"?

**Predictions**:
1. Counterfactual weight (CF) should increase during phase 2 (more resources to non-actual trajectories)
2. Integration (Φ) should initially increase (world model running without sensory interruption) then slowly decrease (internal model collapses to attractors)
3. Effective rank should decrease over phase 2 (internal model has fewer degrees of freedom than reality)
4. Valence should drift negative (viability estimates degrade without sensory calibration)
5. On reconnection (phase 3): Arousal spike (large belief updates), then gradual normalization
6. SM should increase during phase 2 (self is the most salient signal in the absence of external input)

**Why this matters**: This is the computational equivalent of:
- Meditation (voluntary detachment from sensory input)
- Dreaming (involuntary detachment during sleep)
- Dissociation (pathological detachment)
- Solitary confinement (forced detachment)
- The "existential burden" (self-model running even when you'd rather it didn't)

If the predictions hold, the framework explains why prolonged isolation is psychologically devastating (valence drift, rank collapse), why meditation requires training (resisting attractor collapse), and why reconnection with reality is disorienting (arousal spike).

**Infrastructure**: New single-agent architecture with explicit world model (variational autoencoder or transformer-based). Can be built on top of V10 environment with single agent.

---

### V15: World Model Derailment / Distribution Shift ⭐ PRIORITY 3
**Claim tested**: The affect system detects world model failure. Systematic prediction error has a specific affect signature that corresponds to what humans call "confusion," "disorientation," and (if persistent) "psychosis."
**Book section**: Part II (arousal as belief-update rate), Part III (existential burden responses).

**Design**:
- Train agent on environment A (stable, predictable)
- Transfer to environment B (similar structure but different dynamics — physics changed, rewards inverted, social norms shifted)
- Measure affect dimensions during the transition and adaptation period

**Conditions**:
- Gradual shift (environment slowly morphs from A to B)
- Sudden shift (instantaneous change)
- Partial shift (some aspects change, others stable)
- Adversarial shift (environment actively contradicts agent's world model)

**Predictions**:
1. Arousal spikes proportional to prediction error magnitude
2. Valence goes negative (viability estimates become unreliable)
3. Self-model salience increases (agent needs to recalibrate "what am I?")
4. Effective rank initially expands (agent considers more hypotheses) then contracts (new model crystallizes)
5. Integration may temporarily decrease (coordinated response overwhelmed by conflicting signals)
6. Gradual shift produces less arousal but LONGER adaptation period (boiling frog effect)
7. Adversarial shift produces persistent high arousal + negative valence (closest to anxiety/paranoia)

**Why this matters**: Tests the framework's prediction about culture shock, cognitive dissonance, paradigm shifts, and psychotic breaks. If the predictions hold, the affect dimensions capture the phenomenology of being systematically wrong about the world.

**Infrastructure**: Modify V10 environment to support mid-training environment changes. Measure affect continuously across the transition.

---

### V16: Affect Contagion in Multi-Agent Communication
**Claim tested**: Affect propagates through communication. When agent A is under threat and communicates with safe agent B, B's affect state shifts — even though B's local environment hasn't changed.
**Book section**: Part IV (shared observation), Part V (superorganism integration).

**Design**:
- Extension of V13 (uncontaminated MARL with emergent communication)
- Controlled affect induction: One agent faces local threat while others are safe
- Measure: Does communication from threatened agent shift recipients' affect states?

**Measurements**:
- MI(sender_affect; receiver_affect_change | receiver_local_obs) — affect transmission controlling for local conditions
- Temporal dynamics: How fast does affect propagate? Does it decay with distance?
- Specificity: Does negative valence (threat) propagate more readily than positive (discovery)?
- Group-level: Does collective affect state emerge that is distinct from any individual?

**Predictions**:
1. Affect contagion is real and measurable (MI > 0)
2. Negative valence propagates faster than positive (threat-detection bias)
3. Arousal is most contagious (simplest signal — just "something happened")
4. Integration contagion requires shared context (you can only "feel with" someone if your world models overlap)
5. Group-level affect state emerges with higher integration than any individual (collective Φ > Σ individual Φ)

**Why this matters**: Foundation for Part IV (social bonds) and Part V (gods). If affect doesn't propagate through communication, the social-scale claims collapse. If it does, this is evidence that superorganism-level dynamics are real.

---

### V17: Proto-Normativity Detection
**Claim tested**: The grounded normativity claim — that valence is constitutively normative, not just descriptively correlated with behavior.
**Book section**: Part I (is-ought dissolution), Part IV (manifold contamination).

**Design**:
- Multi-agent environment with cooperative and exploitative strategies
- Agents can: cooperate (mutual benefit), defect (self-benefit at other's cost), communicate
- Key manipulation: After cooperation emerges, introduce opportunity for one agent to exploit trust
- Measure: Does the exploiting agent's affect state differ from the cooperating agent's?

**Measurements**:
- Valence during cooperative vs exploitative actions
- Self-model salience during exploitation (does the agent "know" what it's doing?)
- Integration during moral choice (does the system coordinate around the decision?)
- Behavioral hesitation: Response time differences between cooperative and exploitative choices

**Predictions**:
1. Exploitation produces lower valence than cooperation, even when exploitation is more rewarding
2. Self-model salience increases during exploitation (self-monitoring)
3. Integration may decrease during exploitation (internal conflict = partial decomposition)
4. Response time is longer for exploitation (processing cost of coordinating a deceptive action)
5. After repeated exploitation, valence penalty diminishes (desensitization — high ι)

**Why this matters**: If computational agents show proto-normative signals (valence penalties for exploitation), this supports the claim that normativity is structural, not cultural. If they don't, the normativity claim is more fragile than presented.

---

### V18: Superorganism Integration Measurement
**Claim tested**: Multi-agent groups can develop collective integration that exceeds the sum of individual integrations.
**Book section**: Part V (gods and superorganisms), Part VII Priority 5.

**Design**:
- 8-16 agents with communication, specialization, and resource sharing
- Group-level threat (drought, predator swarm) that requires coordination
- Measure collective Φ: partition the group into subgroups, measure prediction loss

**Measurements**:
- Collective Φ_G = prediction_loss(partitioned_group) - prediction_loss(full_group)
- Individual Φ_i for each agent
- Key test: Φ_G > Σ Φ_i ? (collective integration exceeds sum of parts)
- Track across time: Does collective Φ grow with training?
- Under group-level threat: Does collective Φ increase (biological pattern) or decrease?

**Predictions**:
1. Collective Φ_G > 0 (some group-level integration)
2. Φ_G > Σ Φ_i only when agents share information and coordinate (not in independent populations)
3. Group-level threat increases Φ_G (parallel to individual biological pattern)
4. Communication ablation eliminates Φ_G (integration requires information sharing)
5. Emergent specialization correlates with higher Φ_G (division of labor = functional coupling)

**Failure mode**: If Φ_G ≈ Σ Φ_i, superorganism integration is additive, not synergistic — the "gods" framework collapses to a metaphor rather than a literal claim.

---

### V19: ι Operationalization Battery
**Claim tested**: ι is a real, measurable parameter — not just a theoretical construct.
**Book section**: Part II (perceptual configuration), Part III (ι modulation).

**Design** (computational version):
- Train RL agents with self-model module
- Vary conditions that should modulate ι:
  - Low ι: social environment (other agents), survival pressure, novel stimuli
  - High ι: mechanical environment (physics puzzles), repetitive tasks, familiar stimuli
- Measure: Do the 4 proposed ι proxies load on a single factor?

**Proxies**:
1. Agency attribution rate: How often does the agent's world model attribute goals/intentions to objects?
2. Affect-perception coupling: MI(perceptual features; affect state)
3. Integration: Φ of the agent's representations
4. Self-model activation: How active is the self-model during different conditions?

**Predictions**:
1. All 4 proxies correlate positively (high ι = low agency attribution, low coupling, lower Φ, lower SM)
2. Social environments produce lower ι than mechanical environments
3. Survival pressure reduces ι (participatory perception is more efficient under threat)
4. ι varies within agent across conditions (not a fixed trait)

**Human version** (when IRB available):
- Heider-Simmel animations (agency attribution)
- Affect-perception coupling (self-report + physiology)
- Kelemen paradigm (teleological reasoning)
- Mismatch negativity amplitude (EEG)

---

## III. Experiments Proposed in the Book (Not Yet Implemented)

These are formally proposed in Experiment boxes within Parts I-IV. Each should be implemented as part of the V13+ program or as standalone studies.

### From Part I
1. **Lenia Affect Emergence** — Status: V11-V12 series (partially implemented)
2. **Attention-as-Measurement Test** — Status: Proposed. Test whether attention (α) literally selects trajectories.
3. **Computational Animism Test** — Status: Proposed. Self-modeling agents attribute agency to non-agentive objects under compression pressure.

### From Part II
4. **ι Operationalization Battery** — Status: Proposed (see V19 above)
5. **Shame vs. Guilt Dissociation** — Status: Proposed. Requires human subjects.
6. **Affect Similarity Topology** — Status: Proposed. Is affect similarity symmetric? Requires human judgments.

### From Part III
7. **ι Oscillation in Science** — Status: Proposed. ι range predicts scientific novelty.
8. **ι Rigidity as Transdiagnostic Factor** — Status: Proposed. Requires clinical population.
9. **Unified ι Modulation Test** — Status: Proposed. Flow/awe/psychedelics/contemplation with same battery.

### From Part IV
10. **Contamination Detection** — Status: Proposed. Physiological response to manifold mismatch.
11. **Ordering Principle** — Status: Proposed. Broader-first relationship orderings more stable.
12. **Contamination Asymmetry** — Status: Proposed. Contamination faster than decontamination.
13. **Digital Manifold Confusion** — Status: Proposed. Digital mediation produces manifold ambiguity.

---

## IV. Human Studies (Require IRB + Funding)

Eight pre-registered protocols exist in `empirical/protocols/study_protocols.md`:

| # | N | Method | Key Prediction |
|---|---|--------|----------------|
| 1 | 500 | CFA factor analysis | Multi-factor model beats 2-factor |
| 2 | 100 | Ambulatory + physiology | Valence ↔ HRV, cortisol, threat proximity |
| 3 | 60 | EEG + meditation | Integration ↔ experiential unity |
| 4 | 200 | Cultural exposure | Different art forms produce distinct signatures |
| 5 | 80 | Flow induction | Flow = low SM, high Φ, positive V |
| 6 | 90 | Meditation training | Meditation increases Φ, reduces CF |
| 7 | 150 | Clinical (MDD vs GAD) | Depression = low r_eff; Anxiety = high CF |
| 8 | 40 | Real-time threat | Threat increases Φ, SM, A |

---

## V. Priority Ordering for Next Phase

### Tier 1: Run Now (extend existing infrastructure)
1. **V13: Uncontaminated Language Emergence** — Extends V10. Tests universality without contamination.
2. **V14: Solitary Rumination** — New single-agent architecture. Tests internal dynamics.
3. **V15: World Model Derailment** — Modifies V10. Tests confusion/disorientation signature.

### Tier 2: Run After Tier 1 Results
4. **V16: Affect Contagion** — Extends V13. Tests social transmission.
5. **V17: Proto-Normativity** — Extends V13. Tests grounded normativity.
6. **V18: Superorganism Detection** — Extends V13/V16. Tests collective integration.
7. **V19: ι Battery (computational)** — New architecture. Tests ι measurability.

### Tier 3: Human Studies (Require External Resources)
8. **Unified ι Modulation Test** — Flow/awe/psychedelics/contemplation
9. **Affect Similarity Topology** — Is affect similarity symmetric?
10. **Contamination Detection** — Manifold mismatch physiology
11. **Factor Validation (Study 1)** — CFA of affect dimensions in humans

---

## VI. Infrastructure Needed

### For V13 (Uncontaminated MARL)
- [ ] Randomly-initialized transformer agent (no pretrained weights)
- [ ] Discrete communication channel (learnable tokens)
- [ ] Communication-affect analysis pipeline (MI computation)
- [ ] VLM-free translation protocol (structural alignment only)

### For V14 (Solitary Rumination)
- [ ] Single-agent architecture with explicit world model (VAE or transformer)
- [ ] "Offline mode" — world model runs on own predictions
- [ ] Affect extraction during offline phase
- [ ] Phase transition detection (attractor collapse timing)

### For V15 (World Model Derailment)
- [ ] Mid-training environment modification (gradual, sudden, partial, adversarial)
- [ ] Continuous affect tracking across transition
- [ ] Prediction error measurement (KL divergence of world model)

### For V18 (Superorganism)
- [ ] Group-level Φ computation (partition multi-agent system, measure prediction loss)
- [ ] Collective vs individual affect comparison
- [ ] Communication topology analysis

---

## VII. What Would Falsify the Theory

| Experiment | If we find... | Then... |
|------------|--------------|---------|
| V13 | No geometric alignment without human language | Affect geometry may require human-like architecture or training data. Universality claim fails. |
| V14 | No structured affect during offline mode | Affect requires external input. Internal dynamics insufficient. Existential burden claim weakens. |
| V15 | No consistent affect signature of prediction error | Arousal is not belief-update rate. Foundational definition wrong. |
| V16 | No affect contagion through communication | Social-scale affect claims (Parts IV-V) are metaphorical, not literal. |
| V17 | No valence difference between cooperation and exploitation | Normativity is not structural. Is-ought dissolution fails. |
| V18 | Collective Φ = sum of individual Φ | Superorganism integration is additive. "Gods" framework collapses to metaphor. |
| V19 | ι proxies don't correlate | ι is not a unitary construct. May need vector treatment. |

---

## VIII. Completed Experiment Code Reference

```
empirical/experiments/
├── study_llm_affect/          # V2-V10
│   ├── v10_environment.py     # JAX grid world (8×8, 4 agents)
│   ├── v10_agent.py           # Transformer + GRU + PPO
│   ├── v10_affect.py          # 6D affect extraction
│   ├── v10_translation.py     # VLM scene annotation
│   ├── v10_analysis.py        # RSA, CKA, MDS
│   ├── v10_run.py             # Training pipeline
│   └── v10_modal.py           # GPU deployment
│
├── study_ca_affect/           # V11-V12
│   ├── v11_substrate.py       # Lenia physics + resources
│   ├── v11_substrate_hd.py    # 64-channel vectorized
│   ├── v11_substrate_hier.py  # 4-tier hierarchical
│   ├── v11_evolution.py       # Selection + evolution (3842 lines)
│   ├── v11_affect.py          # 6D measurement from CA
│   ├── v11_affect_hd.py       # Spectral Φ for high-D
│   ├── v12_substrate_attention.py  # Self-attention physics
│   ├── v12_evolution.py       # Attention-specific selection
│   ├── v11_run.py             # CLI runner
│   ├── v12_run.py             # V12 runner
│   └── v11_modal.py           # GPU deployment
│
└── study_c_computational/     # Early viability study
    ├── viability_env.py       # Custom gym environment
    ├── train_agents.py        # Simple/WorldModel/SelfModel agents
    └── RESULTS.md             # Partial support (viability yes, 6D no)
```

---

*This document is the roadmap. Update it as experiments are designed, run, and completed. Every strong claim in the book should trace back to an entry here.*
