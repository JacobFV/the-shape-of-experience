# Formalizing the Emergence Experiment Program

## Preamble: The Entanglement Problem

You noted that world model formation, abstract representation, language, and counterfactual detachment may not be separable phase transitions. This formalization takes that seriously. Rather than treating them as four independent experiments, we define a **single evolving measurement framework** that tracks multiple quantities simultaneously and looks for *correlated transitions* — moments where several quantities jump together, confirming they are aspects of one underlying process, versus moments where they diverge, revealing genuine phase structure.

The substrate for all experiments is Lenia or a successor CA with the key modification your V11 series identified as necessary: **state-dependent interaction topology** (attention). Without this, the locality ceiling prevents anything above Rung 4 on the ladder.

---

## 0. Substrate Requirements

Define a substrate $\mathcal{S} = (L, S, N_\theta, f_\theta)$ where:

- $L$: lattice (e.g., $\mathbb{Z}^2$ toroidal, $256 \times 256$ or larger)  
- $S$: continuous state space per cell, $s_i \in \mathbb{R}^C$ for $C$ channels  
- $N_\theta$: **state-dependent** neighborhood function parameterized by local state $\theta_i = g(s_i)$  
- $f_\theta$: local update rule  

The critical departure from V11.0–V11.7: the interaction kernel $K_i$ for cell $i$ is a function of $s_i$, not just position. Concretely:

$$K_i(j) = K_{\text{base}}(|i - j|) \cdot \sigma\!\bigl(\langle h(s_i),\, h(s_j) \rangle - \tau\bigr)$$

where $h: \mathbb{R}^C \to \mathbb{R}^d$ is a learned or fixed embedding and $\sigma$ is a sigmoid gate. This gives cells the capacity to **selectively couple** with distant cells that share state-features — a minimal attention mechanism.

Resource dynamics: Michaelis-Menten as in V11, but with **lethal depletion** (maintenance rate high enough that $>50\%$ of naive patterns die within drought duration).

---

## 1. Experiment 1: Emergent Existence (Completed — Rungs 1–3)

### What was measured

For a pattern $B \subset L$ identified by correlation boundary (Def. 5.9 in thesis):

**Lifetime:** $\tau_B = \min\{t : \text{pattern identity lost}\}$

**Persistence probability:** $P(\tau_B > T)$ for various $T$

**Φ under stress:** $\Delta\Phi = (\Phi_{\text{stress}} - \Phi_{\text{base}}) / \Phi_{\text{base}}$

### Key results (V11.0–V11.7)

- Yerkes-Dodson: mild stress → Φ increases 60–200%
- Curriculum evolution (V11.7) produces generalization to novel stressors (+1.2 to +2.7pp)
- Locality ceiling: convolutional physics cannot produce active self-maintenance

### What "existence" means operationally

A pattern $B$ **exists** at time $t$ if:

$$\exists\, B_t \subset L \text{ s.t. } I(s_i^{(t)};\, s_j^{(t)} \mid \text{bg}) > \theta_{\text{corr}} \;\;\forall\, i,j \in B_t$$

and $B_t$ is "the same pattern" as $B_0$ under a continuity criterion (e.g., overlap $|B_t \cap B_{t-1}| / |B_t| > 0.5$ for all intermediate steps).

---

## 2. Experiment 2: Emergent World Model

### The core question

When does a pattern's internal state carry **predictive information about the environment beyond what's available from current observations**?

### Definition 2.1 (Predictive Information)

$$\mathcal{I}_{\text{pred}}(t, \tau) = I\!\bigl(s_B^{(t)};\; s_{\bar{B}}^{(t+\tau)} \;\big|\; s_{\bar{B}}^{(t)}\bigr)$$

where $s_B^{(t)}$ is the state of cells in the pattern, $s_{\bar{B}}^{(t)}$ is the state of cells outside the pattern (environment), and $\tau$ is the prediction horizon.

**Interpretation:** This is the mutual information between the pattern's current internal state and the environment's *future* state, **conditioned on** the environment's current state. If $\mathcal{I}_{\text{pred}} > 0$, the pattern "knows" something about the future that isn't readable from the present environment alone. It has a world model.

### Practical computation

For continuous-state CAs, direct MI computation is intractable. Use the **prediction gap** proxy:

**Step 1.** Train a predictor $\hat{f}_{\text{full}}: (s_B^{(t)}, s_{\partial B}^{(t)}) \to \hat{s}_{\bar{B}}^{(t+\tau)}$ using the pattern's internal state plus boundary observations.

**Step 2.** Train a predictor $\hat{f}_{\text{env}}: s_{\partial B}^{(t)} \to \hat{s}_{\bar{B}}^{(t+\tau)}$ using only boundary observations (no access to pattern internals).

**Step 3.** The world model score is:

$$\mathcal{W}(t, \tau) = \mathcal{L}[\hat{f}_{\text{env}}] - \mathcal{L}[\hat{f}_{\text{full}}]$$

where $\mathcal{L}$ is prediction loss (MSE or log-likelihood). If $\mathcal{W} > 0$, the pattern's internal state carries predictive information not available from boundary observations alone.

### Definition 2.2 (World Model Horizon)

$$H_{\text{wm}} = \max\{\tau : \mathcal{W}(t, \tau) > \epsilon\}$$

The maximum prediction horizon at which the pattern's internals provide useful information. A longer horizon implies a richer world model.

### Definition 2.3 (World Model Capacity)

$$\mathcal{C}_{\text{wm}} = \int_1^{H_{\text{wm}}} \mathcal{W}(t, \tau)\, d\tau$$

Total predictive information across all horizons. This is the "depth" of the world model.

### Forcing functions

World models should emerge under:
- **Partial observability:** Pattern sees only $R$-radius neighborhood, but viability depends on events at distance $\gg R$
- **Temporal structure:** Resources, threats, conspecifics have autocorrelated dynamics (knowing what happened predicts what will happen)
- **State-dependent attention:** The substrate modification from §0 lets patterns selectively attend to distant signals, creating the channel for non-local information

### Predicted transition

Plot $\mathcal{C}_{\text{wm}}$ vs. evolutionary generation. Expect:

1. Pre-attention substrate: $\mathcal{C}_{\text{wm}} \approx 0$ (locality ceiling)
2. With state-dependent coupling: $\mathcal{C}_{\text{wm}}$ increases with generation
3. Threshold: $\mathcal{C}_{\text{wm}}$ should correlate with lifetime improvement — patterns that model better, survive longer

---

## 3. Experiment 3: Internal Representation Structure

### The core question

When do patterns develop **low-dimensional, compositional** internal representations rather than high-dimensional entangled ones?

### Definition 3.1 (Representational Geometry)

For a pattern $B$ observed across $T$ timesteps and $M$ distinct environmental contexts $\{e_1, \ldots, e_M\}$, collect the internal state vectors $\{s_B^{(t)} : t \in \text{context } e_m\}$. Define:

$$\mathbf{Z} = \text{encoder}(s_B) \in \mathbb{R}^k$$

where the encoder is trained to predict viability-relevant outcomes from $s_B$. Then:

**Effective dimensionality:**
$$d_{\text{eff}} = \frac{(\text{tr}\, \Sigma_Z)^2}{\text{tr}(\Sigma_Z^2)}$$

where $\Sigma_Z$ is the covariance of $\mathbf{Z}$ across contexts. Low $d_{\text{eff}}$ relative to $|B|$ means compressed representation.

**Disentanglement score:** For environmental features $\{f_1, \ldots, f_p\}$ (resource level, threat proximity, conspecific count, etc.):

$$\mathcal{D} = \frac{1}{p} \sum_{i=1}^p \max_j \; r^2(z_j, f_i)$$

where $r^2(z_j, f_i)$ is the coefficient of determination between latent dimension $j$ and environmental feature $i$. High $\mathcal{D}$ means each latent dimension tracks a single environmental feature — compositional structure.

### Definition 3.2 (Abstraction Level)

$$\mathcal{A} = 1 - \frac{d_{\text{eff}}}{\min(|B|, M)}$$

Ranges from 0 (no compression) to 1 (maximally abstract). This is the compression ratio $\kappa$ from Theorem 4.6, measured empirically.

### Definition 3.3 (Compositionality)

If patterns develop signal emissions (§4 below), test whether internal representations compose:

Given contexts $A$, $B$, and the combined context $A \cap B$:

$$\mathcal{K}_{\text{comp}} = \frac{\|z_{A \cap B} - (z_A + z_B - z_\emptyset)\|}{\|z_{A \cap B}\|}$$

Low $\mathcal{K}_{\text{comp}}$ means the representation of combined contexts is approximately the sum of individual context representations — linear compositionality.

### Predicted transition

1. Early evolution: $d_{\text{eff}} \approx |B|$, $\mathcal{D} \approx 0$ — no structure
2. After world model emergence: $d_{\text{eff}}$ drops, $\mathcal{D}$ increases — compression begins
3. Before language: $\mathcal{A}$ and $\mathcal{D}$ should plateau at a level set by environmental complexity
4. **Key prediction:** $d_{\text{eff}}$ should track $\mathcal{C}_{\text{wm}}$ — compression and modeling co-emerge

---

## 4. Experiment 4: Emergent Language and Multi-Agent Culture

### The core question

When do patterns develop **structured, compositional communication** that goes beyond reflexive signaling?

### Setup

Multiple patterns coexist in the same substrate. Viability depends on coordination: resource patches too large for one pattern, threats too fast for individual escape, etc.

### Definition 4.1 (Signal)

A signal is a structured perturbation emitted by pattern $B_i$ that:

1. Propagates through the medium (travels beyond $B_i$'s immediate boundary)
2. Is **distinct from noise** — has lower entropy than random perturbations of equal energy
3. Is **contingent** — its form depends on $B_i$'s internal state, not just local environment

Operationally: identify candidate signals by detecting outgoing wavefronts from $B_i$ with entropy below threshold, then verify contingency via $I(\sigma; s_{B_i}) > I(\sigma; s_{\partial B_i})$.

### Definition 4.2 (Communication Channel Capacity)

$$C_{ij} = \max_{p(\sigma)} I(\sigma_{\text{emitted by } i};\; \sigma_{\text{received by } j})$$

If $C_{ij} > 0$ and both patterns' behaviors condition on received signals, a communication channel exists.

### Definition 4.3 (Language Structure)

Given a corpus of signal-context pairs $\{(\sigma_k, e_k)\}_{k=1}^N$:

**Vocabulary size:** Number of distinct signal clusters (cluster by waveform similarity).

**Context-dependence:** $I(\sigma; e)$ — how much the signal tells you about the environmental context.

**Compositionality (topographic similarity):**

$$\rho_{\text{topo}} = \text{corr}\!\bigl(d_{\text{signal}}(\sigma_i, \sigma_j),\; d_{\text{context}}(e_i, e_j)\bigr)$$

High $\rho_{\text{topo}}$ means similar contexts produce similar signals, and the signal space preserves the structure of context space. This is the hallmark of compositional (as opposed to holistic) communication.

**Productivity:** Can patterns produce and interpret novel signal combinations? Test by presenting novel composite contexts and checking whether emitted signals compose from known components.

### Definition 4.4 (Culture)

Culture emerges when:

1. **Social learning:** Pattern $B_j$ adopts signal conventions from $B_i$ (signal repertoire similarity increases with interaction time)
2. **Convention drift:** Isolated subpopulations develop different signal-context mappings for the same environmental features
3. **Normative pressure:** Patterns that deviate from local conventions have lower fitness (coordination failure)

Measure: compare signal-context mutual information $I(\sigma; e)$ within vs. between subpopulations.

### Predicted transition

Language should emerge *after* world models and compression, because communication requires: (a) something to communicate about (world model content), (b) the capacity to compress that content into a transmissible signal (abstraction), and (c) selective pressure to share information (multi-agent coordination payoff).

---

## 5. Experiment 5: Counterfactual Detachment

### The core question

When does a pattern's internal dynamics **decouple from external driving** and run "offline" world model rollouts — imagination, planning, counterfactual reasoning?

### Definition 5.1 (External Synchrony)

At each timestep, measure the correlation between the pattern's internal dynamics and its boundary inputs:

$$\rho_{\text{sync}}(t) = \frac{\text{Cov}(s_B^{(t+1)} - s_B^{(t)},\; s_{\partial B}^{(t)})}{\sqrt{\text{Var}(\Delta s_B) \cdot \text{Var}(s_{\partial B})}}$$

High $\rho_{\text{sync}}$: pattern's updates are driven by sensory input (reactive mode).  
Low $\rho_{\text{sync}}$: pattern's updates are driven by internal dynamics (detached mode).

### Definition 5.2 (Detachment Event)

A detachment event occurs at time $t$ if:

$$\rho_{\text{sync}}(t) < \theta_{\text{detach}} \quad \text{for} \quad \Delta t > \delta_{\text{min}}$$

where $\theta_{\text{detach}}$ is calibrated from the distribution of $\rho_{\text{sync}}$ (e.g., below the 10th percentile) and $\delta_{\text{min}}$ is a minimum duration (ruling out transient desynchronization from noise).

### Definition 5.3 (Counterfactual Simulation Score)

During a detachment event $[t_0, t_1]$, the pattern's internal dynamics run some trajectory $s_B^{(t_0:t_1)}$ that is decorrelated from actual environmental input. Define:

$$\text{CF}(t_0, t_1) = \max_{\tau > 0} I\!\bigl(s_B^{(t_0:t_1)};\; s_{\bar{B}}^{(t_1 + \tau)}\bigr) - I\!\bigl(s_B^{(t_0^{\text{match}}:t_1^{\text{match}})};\; s_{\bar{B}}^{(t_1 + \tau)}\bigr)$$

where $[t_0^{\text{match}}, t_1^{\text{match}}]$ is a matched-duration period of high-synchrony (reactive) operation.

**Interpretation:** If the detached-mode internal trajectory is *more predictive of future environment* than a reactive-mode trajectory of equal duration, the pattern is doing something useful with its offline processing. It's simulating futures, not just idling.

### Definition 5.4 (Imagination Capacity)

$$\mathcal{I}_{\text{img}} = \frac{1}{|\mathcal{E}|} \sum_{e \in \mathcal{E}} \text{CF}(e)$$

Average counterfactual simulation score across all detachment events $\mathcal{E}$. Positive $\mathcal{I}_{\text{img}}$ means offline processing is systematically predictive — the pattern imagines usefully.

### Definition 5.5 (Branch Entropy During Detachment)

During a detachment event, estimate the diversity of internal trajectories the pattern "considers":

$$H_{\text{branch}} = H\!\bigl(s_B^{(t_1)} \;\big|\; s_B^{(t_0)},\; \text{detached}\bigr)$$

High $H_{\text{branch}}$ combined with positive $\text{CF}$ means the pattern explores multiple possible futures and the exploration is informative. This is the counterfactual weight dimension ($w_c$) from Part II, measured in the substrate.

### Forcing functions for detachment

Detachment should emerge when:
- **Delayed payoffs:** Actions have consequences on timescales $\gg$ reaction time
- **Ambiguous threats:** Environment contains cues that are probabilistically but not deterministically threatening (pattern must "think" before acting)
- **Planning advantage:** Resource patches are distributed such that multi-step plans outperform greedy foraging

### Predicted transition

1. Reactive-only patterns: $\rho_{\text{sync}} \approx 1$ always, $\mathcal{I}_{\text{img}} = 0$
2. First detachment events: $\rho_{\text{sync}}$ dips intermittently, $\text{CF} \approx 0$ (idling, not simulating)
3. Useful detachment: $\text{CF} > 0$, detachment events precede adaptive behavior changes
4. **Key prediction:** $\mathcal{I}_{\text{img}}$ should correlate with $H_{\text{wm}}$ (world model horizon) — you can only simulate futures you can model

---

## 6. The Entanglement Prediction

### Definition 6.1 (Emergence Correlation Matrix)

At each evolutionary generation $g$, measure all five quantities:

| Symbol | Quantity | Rung |
|--------|----------|------|
| $\tau$ | Lifetime (persistence) | 1–3 |
| $\mathcal{C}_{\text{wm}}$ | World model capacity | 4 |
| $\mathcal{A}$ | Abstraction level | 4–5 |
| $\rho_{\text{topo}}$ | Language compositionality | 6 |
| $\mathcal{I}_{\text{img}}$ | Imagination capacity | 7 |

Compute the correlation matrix $\mathbf{R}(g) \in \mathbb{R}^{5 \times 5}$ across the population at generation $g$.

### Prediction 6.1 (Co-Emergence)

$\mathcal{C}_{\text{wm}}$, $\mathcal{A}$, and $\mathcal{I}_{\text{img}}$ will be strongly correlated ($r > 0.7$) at all generations where any of them is nonzero. They are aspects of one process: compression-for-prediction.

### Prediction 6.2 (Partial Separability)

$\rho_{\text{topo}}$ (language) will lag the other three — requiring the additional forcing function of multi-agent coordination. It will correlate with the others only in populations where coordination pressure is present.

### Prediction 6.3 (Threshold Structure)

Despite co-emergence, there should be detectable thresholds — generations where $d\mathcal{C}_{\text{wm}}/dg$ spikes. These correspond to substrate innovations (e.g., the first pattern to develop state-dependent long-range coupling). These are the "rungs" — not discrete phases but punctuated equilibria in a continuous process.

---

## 7. Experimental Protocol Summary

### Phase A: Substrate Engineering (prerequisite)

1. Implement state-dependent coupling (§0)
2. Implement lethal resource dynamics with perceptual range $\gg R_{\text{kernel}}$ via the coupling mechanism
3. Validate that patterns can now forage (directed motion toward distant resources)
4. This is the foundation — nothing above Rung 3 works without it

### Phase B: Single-Agent Emergence (Experiments 1–2, 5)

1. Evolve populations with curriculum training (V11.7 protocol)
2. Measure $\tau$, $\mathcal{C}_{\text{wm}}$, $\mathcal{A}$, $\mathcal{I}_{\text{img}}$ per generation
3. Track emergence correlation matrix
4. Identify threshold generations
5. ~50–100 GPU-hours per evolutionary run, ~5 runs for statistics

### Phase C: Multi-Agent Emergence (Experiments 3–4)

1. Introduce multi-pattern environments with coordination pressure
2. Measure signal emergence, $\rho_{\text{topo}}$, convention formation
3. Track whether language emergence correlates with jumps in $\mathcal{C}_{\text{wm}}$ and $\mathcal{I}_{\text{img}}$
4. ~200 GPU-hours per run due to multi-agent overhead

### Phase D: Uncontaminated Affect Verification

1. On patterns from Phase C with language: apply the tripartite alignment test from the thesis
2. Affect signature ↔ translated signal ↔ observable behavior
3. Bidirectional perturbation (signal, "neurochemistry", environment)
4. This is the capstone — tests the identity thesis on systems with genuine internal complexity

---

## 8. What Distinguishes This From Existing Work

- **Artificial Life / Lenia literature:** Measures pattern complexity, not predictive information or counterfactual processing. No world model formalization.
- **Multi-agent communication (Lazaridou, Mordatch, etc.):** Uses neural networks as agents, not uncontaminated substrates. Language structure inherits from gradient descent, not from thermodynamic emergence.
- **IIT experiments:** Measures Φ but doesn't connect it to world models, abstraction, or communication. Static integration, not dynamic co-emergence.
- **Active inference / FEP:** Theoretical framework, not substrate experiments. Doesn't test whether the framework's predictions hold in uncontaminated substrates.

The unique contribution: measuring the **co-emergence** of existence, modeling, abstraction, communication, and imagination in a single substrate with zero human contamination, using quantities derived from a unified theoretical framework (the thesis).
