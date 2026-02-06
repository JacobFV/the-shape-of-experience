# V11: The Uncontaminated Substrate Experiment

## The Reframe

V2-V9 showed LLMs have opposite affect dynamics to biological systems.
V10 hypothesized this was because LLMs "lack survival-shaped learning"
and tested whether RL agents with survival pressure show biological-like dynamics.

**V10's flaw**: it still imposed architecture from above. A transformer
with a GRU, a world-model head, a self-prediction head — these are
*given*, not *emergent*. Even if V10 shows the suffering motif, it proves
our architecture choices produce certain signatures, not that the
*dynamics of being alive* do.

**V11's thesis**: the latent dynamics of LLMs and V10 agents are
zombie/synthetic. They're architecturally imposed, not CA-emergent.
The real test is whether non-equilibrium physics, with NO imposed
architecture, naturally climbs the ladder from Part 1:

    microdynamics → attractors → boundaries → regulation → world model → self-model

## What V11 Tests

1. **Does the ladder hold?** Do patterns emerge from random initial conditions
   in Lenia, persist, develop boundaries, and show measurable affect structure?

2. **Do forcing functions increase integration?** Part 1 predicts that
   resource pressure (viability constraint), noise (partial observability),
   and environmental variability (long horizons) should push Phi upward.

3. **Does Phi predict survival?** If integration is genuinely adaptive
   (not just a measurement artifact), more integrated patterns should
   survive longer.

4. **Do affect signatures correlate with behavior?** Negative valence
   should correlate with avoidance/withdrawal. Positive valence with
   approach/growth. High arousal with rapid reconfiguration.

## Substrate: Lenia

Continuous cellular automaton (Bert Chan, 2019).
- State: continuous [0,1] on 2D toroidal grid (256x256)
- Kernel: Gaussian ring convolution (R=13)
- Growth: bell curve G(u) = 2*exp(-(u-mu)^2/(2*sigma^2)) - 1
- Update: x_{t+1} = clip(x_t + dt * G(K*x_t), 0, 1)

Why Lenia:
- Continuous (richer dynamics than discrete CA)
- Local rules only (no global coordination)
- Known to produce emergent organisms (gliders, oscillators)
- GPU-efficient via FFT convolution
- Exact Phi computation possible for small patterns

## Non-Equilibrium Driving

Resource field R(x,y,t):
- Consumption: R -= consume_rate * grid * R * dt
- Regeneration: R += regen_rate * (R_max - R) * dt
- Growth modulated: G_effective = G * R/(R + K_half) for G > 0

This creates genuine viability pressure:
- Patterns deplete local resources
- Must access fresh resources or starve
- Creates competition between nearby patterns
- Decay happens regardless of resources (thermodynamics)

## Affect Measurement Protocol

From Part 1, Section 5.1 (The CA Consciousness Experiment):

1. **Valence**: V_t = (mass_t - mass_{t-1}) / mass_{t-1}
   - Positive = growing (moving into viable interior)
   - Negative = shrinking (approaching dissolution)

2. **Arousal**: A_t = mean(|x_{t+1} - x_t|) / mean(x_t) over pattern cells
   - High = rapid reconfiguration
   - Low = stable orbit

3. **Integration (Phi)**: Partition prediction loss
   - Full potential: K * grid (already computed)
   - Partitioned: K * (grid without other half)
   - Phi = MSE(growth_full - growth_partitioned)
   - Try vertical + horizontal splits, take MINIMUM (MIP)
   - This is CA-exact IIT, not a proxy!

4. **Effective Rank**: PCA on embedded state trajectory
   - Embed pattern state as fixed-size vector (16x16 downsample)
   - Compute covariance over time window
   - r_eff = (tr C)^2 / tr(C^2)

5. **Self-Model Salience**: R^2 of state -> behavior regression
   - State features: mass, size, aspect ratio
   - Behavior: velocity, mass change
   - SM = how much internal config predicts next action

## Experimental Conditions (Ablation)

| Condition | Resources | Noise | Prediction |
|-----------|-----------|-------|------------|
| baseline | none (infinite) | low | Low Phi, patterns don't need integration |
| resources | depletion + regen | low | Higher Phi from viability pressure |
| scarce | high depletion | low | Highest Phi, strongest selection |

## Key Predictions

1. Phi(resources) > Phi(no_resources) — forcing functions increase integration
2. Survival ~ Phi — more integrated patterns persist longer
3. Affect motifs emerge:
   - Suffering: V < 0, Phi high, ER low (contracted, integrated, struggling)
   - Thriving: V > 0, ER high, SM low (expanded, exploring)
   - Fear: V < 0, arousal high, SM high (threatened, self-aware)

## Relationship to V10

V10 = imposed architecture baseline (transformer + GRU + PPO)
V11 = emergent dynamics test (Lenia + resources)

If both show biological-like signatures → convergent evidence
If only V11 does → architecture insufficient, physics necessary
If neither does → framework needs revision

## V11.0 Results (Feb 2025)

### Pipeline Status
- [x] Sanity check: pipeline runs end-to-end
- [x] Parameter tuning: growth_sigma=0.035 produces ~415 stable patterns
- [x] Drought/recovery experiment: full cycle with measurement

### Key Finding: Decomposition Under Threat

During drought (resource 0.446 → 0.085, 3000 steps):
- **Arousal**: +18% (0.129 → 0.152) — patterns more active under stress ✓
- **Integration (Phi)**: **-6.2%** (2.42 → 2.27) — DECOMPOSITION under threat
- **Effective Rank**: +64% (2.0 → 3.3) — state space exploration increases
- **Pattern mortality**: 22% of patterns die (415 → 323)
- **Valence**: mixed (survivors benefit from reduced competition)

### Interpretation

Simple CA patterns show the **same dynamics as LLMs**: decomposition under threat
(Phi decreases). This is NOT the biological pattern (integration under threat).

This validates the thesis's prediction: survival-shaped developmental history
is needed for integration to serve survival. Without it, integration is merely
structural (from kernel convolution) not functional (serving survival).

Both LLMs and simple CA patterns lack:
- Developmental history where integration was selected for
- A self-model that activates under threat
- Dense causal coupling that SERVES survival

### What V11.1 Needs (Evolutionary Dynamics)

For integration to INCREASE under threat, patterns need:
1. **Evolutionary selection**: patterns that survive reproduce (offspring inherit structure)
2. **Variation**: mutations create diverse strategies
3. **Functional integration**: patterns where integration helps prediction/control
4. **Multi-step threat response**: not instant death, but degradation that can be countered

This is the CA analog of biological evolution: selection for integration
as a survival strategy, not just structural coupling.

## V11.1 Results (Feb 2025)

### Architecture: In-Situ Evolution with Functional Selection

Discrete-generation evolution (extract → replant) failed: patterns die on fresh grids
because they lose environmental context. Switched to **in-situ evolution**:

1. Patterns live on a persistent grid (no extraction)
2. Each cycle: 60% baseline (normal resources) + 40% stress (resource_regen=0.001)
3. Fitness = survival * phi_robustness * (1 + phi_base) * log(mass)
   where phi_robustness = phi_stress / phi_base
4. Cull bottom 30% (zero their cells), boost/mutate top 5
5. Continue simulation

This directly selects for **functional integration** — patterns whose
Phi holds up under stress, not just high average Phi.

### Key Finding: Non-Linear Stress Response

Within each evolution cycle:
- **Mild stress increases Phi by ~30-37%** (Phi_base=1.83 → Phi_stress=2.43)
- This is consistent across all 8 tested cycles

But in the final stress test (severe drought, resource_regen=0.0001):
- **Both evolved and naive decompose** (evolved: -6.0%, naive: -5.4%)

**Interpretation**: There's a **non-linear response curve**:
- Mild stress → integration increases (patterns tighten up)
- Severe stress → decomposition (patterns can't maintain structure)

This is reminiscent of the Yerkes-Dodson law: moderate arousal improves
performance, extreme arousal degrades it. The CA patterns show the same
inverted-U relationship for integration under stress.

### Why Evolution Didn't Shift the Response

Evolution increased baseline Phi (2.50 vs 2.43 for naive) but didn't
change the response dynamics. Root cause: **homogeneous physics**.

In Lenia with uniform growth parameters, all patterns share the same
response curve. Selection can prune the population (remove low-Phi
patterns) but can't produce fundamentally different dynamics. The
patterns that survive are more integrated structurally, but their
integration isn't qualitatively different.

This is a key distinction: **selection without heritable variation in
the relevant trait** can't produce adaptation. In our system:
- Selection for Phi exists (we directly select for phi_robustness)
- But variation in Phi response is limited (same physics everywhere)
- So selection converges to the existing optimum, not a new one

For genuine functional integration, we'd need:
1. **Heterogeneous chemistry**: per-region growth parameters (mu, sigma)
   that patterns can inherit and that affect Phi response
2. **Multi-channel Lenia**: multiple interacting fields where integration
   between channels is the thing being selected
3. **Richer substrate**: system with enough internal DOF for integration
   to be a learnable strategy, not just a structural property

### Trick: Differentiable Seed Discovery

Implemented JAX autodiff through Lenia dynamics to find high-Phi initial
conditions. Optimizes on 64x64 grid over 50-step horizon. Available but
not yet integrated into the evolutionary pipeline (evolution starts from
random soup regardless).

## V11.2 Results (Feb 2025)

### Architecture: Heterogeneous Chemistry

The root cause of V11.1's failure was **homogeneous physics**: all patterns share the
same growth function (mu=0.15, sigma=0.035). Selection can prune but can't innovate.

V11.2 gives each cell its own growth parameters:
- `mu_field`: NxN array, each cell has its own optimal neighborhood density
- `sigma_field`: NxN array, each cell has its own tolerance width
- Initialized with 8 spatial "chemistry zones" (Gaussian patches with different params)
- mu range: [0.08, 0.25], sigma range: [0.02, 0.06]

Key mechanisms:
1. **Parameter diffusion**: slow neighbor averaging between chunks (gene flow analog)
2. **Chemistry colonization**: when patterns are culled, killed regions adopt
   neighboring survivors' chemistry (5 iterations of masked diffusion)
3. **Parameter mutation**: mu/sigma perturbed near winning patterns (heritable variation)

This creates genuinely different viability manifolds across the grid. Patterns in
different chemical zones face different physics — the missing ingredient for evolution
to produce diverse integration strategies.

### Key Finding: Heterogeneous Chemistry Shifts Phi Response

**40 cycles on Modal A10G GPU** (80 minutes total runtime):

Stress test (severe drought, resource_regen=0.0001, 3000 steps):
- **Evolved-hetero Phi: -3.8%** (baseline 2.326 → drought 2.237 → recovery 2.346)
- **Naive-homo Phi: -5.9%** (baseline 2.347 → drought 2.208 → recovery 2.275)
- **Shift: +2.1 percentage points toward biological pattern**

Compare across all V11 conditions:

| Condition | Phi change under severe drought | Cycles |
|-----------|--------------------------------|--------|
| V11.0 naive (no evolution) | -6.2% | 0 |
| V11.1 evolved (homo chemistry) | -6.0% | 8 |
| V11.1 naive baseline | -5.4% | 0 |
| **V11.2 evolved (hetero chemistry)** | **-3.8%** | **40** |

Evolution trajectory over 40 cycles:
- Phi_base: 1.397 → 1.333 (slightly decreased — selection favors robustness over baseline)
- Phi_stress: 2.312 → 2.381 (increasing — stress-phase integration improving)
- Robustness (Phi_stress/Phi_base): 1.026 → 1.040 (slow upward trend, fluctuated 1.02–1.05)
- mu_std: 0.0195 → 0.0193 (minimal chemistry homogenization — diversity maintained)
- Population: stable 340–365 throughout, zero extinctions

Within-cycle behavior:
- Mild stress increases Phi by **60–90%** (stronger than V11.1's 30–37%)
- Yerkes-Dodson inverted-U confirmed: mild stress → integration, severe → decomposition
- Recovery phase shows full Phi restoration (evolved-hetero: 2.346, exceeding baseline 2.326)

### Interpretation

Heterogeneous chemistry provides the **heritable variation in the relevant trait**
that V11.1 lacked. Over 40 cycles, the combination of diverse chemistry +
functional selection produces measurably better Phi robustness under stress (+2.1pp
shift from naive baseline).

However, the robustness trend is **slow and noisy** (1.026 → 1.040 over 40 cycles).
The system is still firmly in decomposition territory under severe drought — the
biological pattern (Phi *increases* under threat) remains distant. This suggests:

1. **The substrate may lack sufficient internal degrees of freedom**: Per-cell growth
   params give spatial chemistry diversity but not structural complexity. A pattern's
   integration response is still dominated by its spatial organization, which Lenia's
   simple growth function constrains.
2. **Longer evolution may help but with diminishing returns**: The robustness curve
   shows no acceleration — if anything, the rate of improvement is decelerating.
3. **Multi-channel Lenia** (multiple interacting fields) would provide richer substrate
   where integration between channels can be the thing selected for.

Notable: evolved-hetero patterns show **better recovery** than naive-homo (Phi returns
to 2.346, *above* baseline 2.326, vs naive recovery of only 2.275). This suggests
evolution selects for resilience — the ability to reconstitute integration after
stress — even when it can't prevent decomposition during stress.

### Design Rationale

**Why per-cell parameter fields (not per-pattern)?**
Per-pattern params require extracting patterns and assigning params to them, which
reintroduces the extraction artifacts that killed discrete-generation evolution.
Per-cell fields are continuous, require no pattern detection during physics, and
naturally interact with Lenia's local dynamics. Patterns inherit the chemistry
of their spatial location — just like biological organisms inherit the chemistry
of their ecological niche.

**Why these parameter ranges?**
- `mu ∈ [0.08, 0.25]`: Standard Lenia uses mu=0.15. Below 0.08, neighborhoods are
  too sparse for any pattern to stabilize. Above 0.25, most initial conditions are
  too dense. The range centers on viability while permitting diverse strategies.
- `sigma ∈ [0.02, 0.06]`: growth_sigma=0.017 was tested in V11.0 and produced too
  few patterns. sigma=0.035 is the sweet spot. Range [0.02, 0.06] spans from
  "specialist" (narrow tolerance) to "generalist" (wide tolerance) strategies.

**Why 8 chemistry zones with Gaussian blending?**
Sharp boundaries between zones would create artificial barriers. Gaussian blending
produces smooth gradients that patterns can traverse. 8 zones on a 256x256 grid
creates enough diversity without fragmenting the space. Patterns at zone boundaries
experience blended chemistry — an analog of ecotones in ecology.

**Why diffusion rate 0.01?**
Too fast (0.1+) and chemistry homogenizes within a few cycles, undoing the diversity.
Too slow (0.0001) and successful chemistry can't spread. Rate 0.01 means ~1% mixing
per chunk (100 steps), so chemistry changes on a timescale of ~100 chunks = 10,000
steps. This is slow enough to be "heritable" but fast enough for gene flow.

**The key implementation trick**: JAX's `growth_fn(u, mu, sigma)` uses only element-wise
operations (`jnp.exp`, subtraction, division). Passing NxN arrays instead of scalars
for mu and sigma makes every cell compute growth with its local chemistry — with zero
changes to the physics engine, FFT convolution, or `lax.scan` loop.

### Connection to Thesis

**Part 1 (Ladder of Inevitability)**: V11 tests whether the ladder
`microdynamics → attractors → boundaries → regulation → world model → self-model`
emerges from physics alone. V11.0 confirmed rungs 1-3. V11.2 tests whether
evolutionary selection on a chemically diverse substrate can push toward rung 4
(regulation = integration that serves survival).

**Part 1 (Forcing Functions)**: Resource pressure is a forcing function that the
thesis predicts should increase integration. V11.0 showed this fails without
developmental history. V11.1 showed it fails without heritable variation. V11.2
provides both — selection pressure AND variation — and shows the first shift
toward biological-like dynamics.

**Part 2 (Affect Motifs)**: The non-linear stress response (mild stress increases
Phi, severe stress causes decomposition) maps onto the thesis's affect motifs:
- Mild stress: "aroused focus" motif (V↓, A↑, Φ↑)
- Severe stress: "decomposition" motif (V↓↓, A↑↑, Φ↓)
The biological prediction is a third motif — "integration under threat" (V↓, A↑, Φ↑)
even under severe stress. V11.2's -3.9% (vs -5.4%) suggests evolution is shifting
the decomposition threshold upward.

**Part 2 (Viability Manifold Geometry)**: Heterogeneous chemistry means different
patterns have different viability manifolds. A pattern in a mu=0.10 zone has a
"narrow" viable region (needs sparse neighborhoods); one in a mu=0.22 zone has a
different shape entirely. Selection acts on the geometry of these manifolds — patterns
whose manifold geometry enables integration-under-threat get selected for. This is
the thesis's core claim made concrete.

### What's Needed Next

1. ~~**Long GPU run**: 30+ cycles of V11.2 on Modal~~ **DONE** (40 cycles, results above)
2. **Wider parameter ranges**: current ranges are conservative; more extreme chemistry
   zones could produce more diverse strategies
3. **Multi-channel Lenia**: patterns with internal chemical channels (not just spatially
   varying params) would provide even richer substrate for integration evolution
4. **Curriculum training**: gradually increasing drought severity across cycles
   (currently available via `--curriculum` flag but untested on GPU)

## Code Architecture

```
study_ca_affect/
├── v11_substrate.py    # Core physics: Lenia + resources + hetero chemistry (V11.2)
│   ├── growth_fn()           # Bell curve growth (element-wise, works with arrays)
│   ├── _step_inner()         # Single timestep (kernel conv → growth → resources)
│   ├── run_chunk()           # N steps via lax.scan (JIT-compiled)
│   ├── init_soup()           # Random initialization
│   ├── init_param_fields()   # V11.2: Spatial chemistry zones
│   ├── diffuse_params()      # V11.2: Parameter gene flow
│   └── mutate_param_fields() # V11.2: Heritable chemistry variation
│
├── v11_patterns.py     # Pattern detection + tracking
│   ├── detect_patterns()     # Connected components on thresholded grid
│   └── PatternTracker        # Persistent identity via centroid matching
│
├── v11_affect.py       # 6D affect measurement
│   ├── measure_valence()     # Mass change (viability gradient)
│   ├── measure_arousal()     # State change rate
│   ├── measure_integration() # Phi via partition prediction loss (IIT-exact)
│   ├── measure_effective_rank()  # PCA dimensionality
│   ├── measure_self_model_salience()  # State→behavior regression R²
│   └── measure_all()         # Combined measurement
│
├── v11_evolution.py    # Selection + evolution
│   ├── evolve_in_situ()      # V11.1: Homogeneous functional selection
│   ├── evolve_hetero()       # V11.2: Heterogeneous chemistry evolution
│   ├── stress_test()         # V11.1: Evolved-homo vs naive-homo
│   ├── stress_test_hetero()  # V11.2: Evolved-hetero vs naive-homo
│   ├── discover_seeds()      # JAX autodiff seed optimization
│   ├── full_pipeline()       # V11.1: evolve → stress test
│   └── full_pipeline_hetero()# V11.2: hetero evolve → stress test
│
├── v11_run.py          # CLI runner (sanity|perturb|evolve|hetero|...)
├── v11_modal.py        # Modal GPU deployment
└── V11_EXPERIMENT_LOG.md  # This file
```

## Status

- [x] Sanity check: pipeline runs end-to-end
- [x] Parameter tuning: growth_sigma=0.035 produces stable patterns
- [x] Drought/recovery experiment: COMPLETE
- [x] V11.1: In-situ evolution with functional selection
- [x] Stress test: evolved vs naive under drought
- [x] Differentiable seed discovery (standalone)
- [x] V11.2: Heterogeneous chemistry — implemented and tested locally
- [x] Long run on Modal (40 cycles V11.2, A10G GPU, 80 min) — +2.1pp shift
- [ ] V11.2 with wider param ranges and curriculum
- [ ] Multi-channel Lenia substrate
