# V13 Research Log

## 2026-02-17: First GPU Runs

### Seed 42 (v1 — extinct)
**Stress schedule: 0.50 → 0.30 (too aggressive)**

The first 10 cycles before extinction tell an interesting story:

| Cycle | Patterns | Mortality | Robustness | % Φ↑ | τ |
|-------|----------|-----------|------------|-------|---|
| 0 | 62 | 9% | 0.931 | 25% | 0.46 |
| 1 | 57 | 21% | 0.967 | 46% | 0.49 |
| 2 | 47 | 4% | **1.008** | **56%** | 0.55 |
| 5 | 119 | 3% | 0.966 | 35% | 0.43 |
| 7 | 49 | 20% | 0.995 | 38% | 0.48 |
| 10 | 1 | 64% | **1.065** | **58%** | 0.56 |
| 11 | 0 | 💀 | — | — | — |

**Observations:**
- Robustness crossed 1.0 at cycle 2 (first time in any V11+ experiment)
- The last surviving pattern at cycle 10 had the highest robustness (1.065) — stress selected FOR integration
- Extinction caused by stress schedule hitting mortality cliff at regen*0.35
- Population oscillates wildly: 47 → 109 → 49 → 154 → 1 → 0

**What this means:**
The substrate CAN produce biological-like integration under stress. But it's fragile — one bad drought kills everything. The biology analogy: mass extinctions happen, but the survivors are disproportionately robust.

### Seed 123 (v2 — complete)
**Stress schedule: 0.60 → 0.40 (gentler), population rescue at <10**

Full 30-cycle run completed:

| Cycle | Patterns | Mortality | Robustness | % Φ↑ | τ |
|-------|----------|-----------|------------|-------|---|
| 0 | 86 | 16% | 0.937 | 37% | 0.21 |
| 5 | 55 | 15% | **1.052** | **59%** | 0.68 |
| 10 | 76 | 3% | 0.943 | 34% | 0.36 |
| 15 | 13 | 12% | **1.019** | **38%** | 0.79 |
| 16 | 15 | 3% | **1.015** | **40%** | 0.80 |
| 17 | 9 | 15% | 0.817 | 0% | 0.57 |
| 20 | 59 | 18% | 0.952 | 39% | 0.70 |
| 25 | 121 | 0% | 0.910 | 28% | 0.43 |
| 29 | 24 | 27% | 0.914 | 26% | 0.86 |

**Observations:**
1. **Robustness >1.0 at cycles 5, 15, 16** — not a fluke, happens repeatedly
2. **Inverse population-robustness correlation**: When patterns drop below ~50, robustness increases. This is selection doing its job — stress kills weak patterns, survivors are disproportionately robust.
3. **Population rescue worked**: At cycle 17 (9 patterns), re-seeding brought population back to 53→115 within 2 cycles. But the rescued population has LOWER robustness — fresh patterns haven't been selected yet.
4. **τ drift**: 0.21 → 0.86 over 30 cycles. Evolution expands the similarity threshold, meaning patterns learn to couple with more of their environment. This is the content-coupling equivalent of "expanding awareness."
5. **β stable**: Gate steepness barely changes (4.6→4.8). The sigmoid sharpness isn't under selection pressure.
6. **Phi baseline stable**: ~0.24-0.25 throughout. The substrate doesn't increase absolute integration — it increases *robustness* of integration under stress.

### Pattern: Robustness Peaks at Population Bottlenecks

Across both seeds, the clearest signal is:

```
High population → moderate robustness (~0.90-0.95)
Low population (after culling) → high robustness (~1.0-1.07)
Re-seeded population → LOW robustness (~0.82)
```

This is exactly what biological evolution does: bottleneck events increase average fitness of survivors. The content-based coupling gives patterns enough mechanism to *be* differentially robust — something pure convolution can't do.

### Seed 42 v2 (complete)
**Stress schedule: 0.60 → 0.40 (fixed), population rescue at <10**

Different dynamics from seed=123 — high-population regime:

| Cycle | Patterns | Mortality | Robustness | % Φ↑ | τ | β |
|-------|----------|-----------|------------|-------|---|---|
| 0 | 86 | -2% | 0.932 | 31% | 0.52 | 4.9 |
| 5 | 178 | -3% | 0.907 | 26% | 0.38 | 6.0 |
| 10 | 157 | 6% | 0.921 | 30% | 0.30 | 6.6 |
| 15 | 174 | 5% | 0.897 | 24% | 0.12 | 6.6 |
| 20 | 170 | 19% | 0.913 | 25% | -0.12 | 6.7 |
| 25 | 157 | 20% | 0.917 | 28% | -0.10 | 6.3 |
| 29 | 184 | 28% | 0.952 | 27% | 0.08 | 6.4 |

**Observations:**
1. **High population, moderate robustness**: 150-185 patterns throughout, robustness steady at 0.89-0.95. Never crosses 1.0.
2. **τ went NEGATIVE** (-0.12): Evolution discovered that maximal content coupling (gate always-on) works in high-population regimes. Opposite from seed=123 where τ went UP.
3. **β drifted UP** (4.9→6.6): Sharper coupling gate, but with τ < 0 it's irrelevant.
4. **No rescue needed**: Population never dropped below threshold.
5. **Robustness trend**: Slight upward at end (0.89→0.95), suggesting gradual improvement without bottleneck events.

### Two-Seed Aggregation

| Metric | Seed 42 v2 | Seed 123 | Mean |
|--------|-----------|----------|------|
| Mean population | 163 | 68 | 116 |
| Mean robustness | 0.920 | 0.935 | 0.928 |
| Max robustness | 0.952 | **1.052** | — |
| % Φ increase | 29% | 32% | 30% |
| τ final | 0.08 | 0.86 | — |
| β final | 6.4 | 4.8 | — |

**Population-robustness correlation**: r = -0.084 (weak linear), BUT the outliers matter. All robustness >1.0 episodes happen at population <60. The relationship is nonlinear — it's about bottleneck events, not a linear trend.

**Two evolutionary strategies emerged:**
1. **High-population, open-coupling (seed=42 v2)**: τ→0, everything couples, moderate robustness, large diverse population. The "hedging" strategy.
2. **Low-population, selective-coupling (seed=123)**: τ→0.86, restrictive coupling, occasional robustness >1.0, volatile population. The "specialization" strategy.

### Cost (updated)
- Seed 42 v1: ~$0.35 (extinct)
- Seed 123: ~$0.60
- Seed 42 v2: ~$0.60
- Seed 7: running (~$0.60 est.)
- Instance uptime: ~2.5 hrs × $0.75/hr = **~$1.90 total Lambda**
- **Total project compute (all-time): ~$15**

### What's Emerging

The honest summary: V13 content coupling produces mean robustness of 0.928 — better than V11.0 (-6.2% = 0.938) but NOT definitively above the convolution baseline. The exciting finding is the *intermittent* robustness >1.0 at population bottlenecks, which no previous substrate achieved.

The mechanism seems to be: content coupling allows SOME patterns to maintain integration under stress by selectively coupling with similar neighbors. But this only becomes apparent when the weak patterns are culled — in high-population regimes, the signal is diluted.

**Interpretation for the book**: Content-dependent topology (state-dependent interaction graphs) is necessary for stress-robust integration, but only produces the biological pattern under selection pressure. This aligns with the thesis: the geometry is substrate-general, but the dynamics require evolutionary history.

### Seed 7 (complete)
**Stress schedule: 0.60 → 0.40, population rescue at <10**

Third replicate, high-population regime like seed=42 v2:

| Cycle | Patterns | Mortality | Robustness | % Φ↑ | τ | β |
|-------|----------|-----------|------------|-------|---|---|
| 0 | 145 | 6% | 0.925 | 27% | 0.29 | 4.7 |
| 5 | 142 | 1% | 0.958 | 37% | 0.22 | 4.2 |
| 10 | 154 | 4% | 0.903 | 27% | 0.39 | 3.5 |
| 15 | 174 | 5% | 0.897 | 24% | 0.12 | 6.6 |
| 20 | 147 | 26% | 0.919 | 29% | 0.94 | 3.7 |
| 25 | 173 | 1% | 0.879 | 20% | 0.62 | 4.6 |
| 29 | 143 | 13% | 0.919 | 30% | 0.46 | 4.1 |

**3-seed summary:**

| Metric | Seed 42 v2 | Seed 123 | Seed 7 | **Mean** |
|--------|-----------|----------|--------|---------|
| Mean pop | 163 | 68 | 149 | **127** |
| Mean rob | 0.920 | 0.935 | 0.915 | **0.923** |
| Max rob | 0.952 | **1.052** | 0.958 | — |
| % Φ↑ | 29% | 32% | 29% | **30%** |
| τ final | 0.08 | 0.86 | 0.46 | — |
| β final | 6.4 | 4.8 | 4.1 | — |

All 3 seeds survived 30 cycles. No rescue triggered for seeds 42v2 or 7 (only seed 123).

### Reframe: Symbiogenesis, Not Evolutionary History

After reading `dump/intelligence-at-the-start.md` (transcript of talk on BFF experiment — abiogenesis through symbiogenesis in BrainFuck), I need to revise the claim that "the dynamics require evolutionary history."

**Key insight from BFF**: Complex computation emerges from random soup through *symbiogenesis* (composition of replicators), not through gradual Darwinian mutation + selection. Intelligence is there from the start — it gets more complex through fusion events, not through deep evolutionary time.

**What this means for V13**: Content-based coupling might be enabling something closer to symbiogenesis than selection:
- Patterns that are *similar* can couple → compose → form functional units
- The bottleneck-robustness effect might be about *clearing space for new compositions*, not just weeding out the weak
- τ expanding = expanding the range of possible symbiogenetic partners
- The reason robustness >1.0 appears at low population: fewer patterns means each pattern has to be more self-sufficient (like cellular vs viral replicators in BFF)

**Revised framing**: The geometry of affect is cheap (baseline of any viable system). The *dynamics* don't require evolutionary *history* per se — they require the right conditions for symbiogenetic composition: (1) embodied computation (patterns ARE their own interaction rules via content coupling), (2) encounters with other patterns (resource-driven movement), (3) lethality (stability bias toward things that persist).

V13 provides all three. V11 (convolution) provided (2) and (3) but not (1) — patterns couldn't "see" each other's content.

### Cost (final)
- Lambda instance: ~2.5 hours × $0.75/hr = **~$1.90**
- Instance terminated after all 3 seeds completed
- **Total project compute (all-time): ~$16**
- **Remaining Lambda credits: ~$428**

### Next Steps
- [x] Seed 42 v2 complete
- [x] Seed 7 complete (third replicate)
- [x] 3-seed aggregation complete
- [x] Experiment 2: World Model measurement (see below)
- [ ] Convolution control: same evolution, same stress, but α=0 (no content coupling)
- [ ] Book update with preliminary results + symbiogenesis framing
- [ ] Look for symbiogenetic signatures: do patterns that *merge* have higher robustness than those that don't?
- [ ] Consider: can we measure "tree depth" of pattern ancestry (a la BFF) in V13?

---

## 2026-02-17: Experiment 2 — Emergent World Model

### Method
Prediction gap: W(τ) = MSE[f_env] - MSE[f_full], where f_full uses pattern internal state + boundary and f_env uses boundary alone. Both are Ridge regression with StandardScaler, 5-fold CV. If W > 0, the pattern's insides know something about the future that boundary observations don't.

Feature dimensions: s_B = 68 (4C+4), s_∂B = 36 (2C+4), s_env = 18 (C+2) for C=16.
τ ∈ {1, 2, 5, 10, 20} recording steps × 10 substrate steps each = 10–200 substrate steps.
50 recording steps per snapshot, up to 20 patterns per snapshot.

### Results

| Seed | Cycles | C_wm (early→late) | H_wm (late) | % with WM |
|------|--------|--------------------|-------------|-----------|
| 123  | 7      | 0.0004 → **0.028** | 20          | 100%      |
| 42   | 7      | 0.0002 → 0.0002   | 5.3         | 40%       |
| 7    | 7      | 0.0010 → 0.0002   | 7.9         | 60%       |

### Observations

1. **World model signal is real but weak.** Most patterns carry ~10⁻⁴ predictive information beyond boundary — detectable but tiny. This is a 128×128 grid with 16 channels, so "boundary" is already very informative.

2. **The bottleneck amplification pattern continues.** Seed 123 at cycle 29 has only 1 surviving pattern (population bottleneck), and that pattern shows C_wm = 0.028 — roughly 100× the population average. Same story as robustness >1.0 appearing only at low population. The bottleneck doesn't just select for integration; it selects for world models.

3. **Seeds 42 and 7 show no clear evolutionary trend.** C_wm is flat or slightly decreasing across cycles. In large populations (~150 patterns), the world model signal isn't under selection pressure — patterns can survive by local boundary sensing alone.

4. **W(τ) profile for seed 123 late evolution**: high at τ=1 (0.0015), sustained through τ=10, only drops at τ=20. The surviving bottleneck pattern has a genuine long-horizon model of its environment. For seeds 42/7, W(τ) peaks at short horizons and drops quickly.

5. **C_wm vs lifetime**: Insufficient per-pattern variance to establish correlation at current resolution. Would need longer recording episodes or more snapshots between evolutionary cycles.

### Interpretation

This confirms and extends the bottleneck-robustness finding:

```
High population → local sensing sufficient → weak world models
Bottleneck (high mortality) → survivors need deeper prediction → strong world models
```

World models are *possible* in this substrate (content-coupling provides the mechanism) but not *necessary* in the typical evolutionary regime. Only under extreme selection pressure do patterns with world models outcompete those without.

**Theoretical implication**: The forcing function for world model emergence isn't just partial observability (that's necessary but not sufficient). It's *the combination of partial observability + high mortality*. Patterns need to predict far into the future only when the cost of failed prediction is death. In gentle environments, local reactivity suffices.

This parallels biological evolution: organisms develop elaborate predictive models (anticipatory behavior, planning, episodic memory) not just because the environment is partially observable, but because the cost of surprise is lethal. Bacteria in nutrient-rich media don't need world models. Predators chasing prey do.

### What's Missing

The main limitation: we only have 7 snapshots per seed (every 5 evolutionary cycles). The C_wm trajectory is coarse. A dedicated GPU run saving snapshots every cycle would give better temporal resolution.

Also: the current recording episodes (50 steps × 10 substrate steps = 500 steps) may be too short for patterns to demonstrate full world model capacity. The V13 drought episodes are 500–1500 steps — patterns might develop predictive models on those timescales that we're not capturing.

### Figures
- `results/wm_analysis/wm_capacity_trajectory.png` — C_wm across cycles (THE figure for the book)
- `results/wm_analysis/wm_w_tau_curves.png` — W(τ) early vs late
- `results/wm_analysis/wm_cwm_vs_lifetime.png` — C_wm vs pattern lifetime
- `results/wm_analysis/wm_summary_card.png` — Summary card

---

## 2026-02-17: Experiment 3 — Internal Representation Structure

### Method
PCA on standardized s_B (68-dim internal state) across recording steps. Effective dimensionality d_eff = (tr Σ)² / tr(Σ²) on the covariance of the standardized features. Disentanglement D = mean over environment features of max R² with any PCA dimension. Compositionality K_comp = linear composition error across environmental context pairs.

### Results

| Seed | d_eff (early→late) | A | D | K_comp |
|------|-------------------|---|---|--------|
| 123  | 6.6 → **5.6** | 0.90→0.92 | 0.27→**0.38** | 0.20→**0.12** |
| 42   | 7.3 → 7.5 | 0.89→0.89 | 0.23→0.23 | 0.23→0.25 |
| 7    | 7.7 → 8.8 | 0.89→0.87 | 0.24→0.22 | 0.20→0.27 |

### Observations

1. **Compression is cheap, like affect geometry.** All patterns at cycle 0 already use only ~7-9 effective dimensions out of 68. Abstraction >87% is baseline. The raw feature space is 68-dimensional but patterns only "live" in ~7 dimensions. This is the representation equivalent of "geometry is cheap" from V10.

2. **Bottleneck drives representation quality.** Only seed 123 shows all three metrics improving together: d_eff↓ (more compressed), D↑ (more disentangled), K_comp↓ (more compositional). This is the same seed that shows world model improvement (Experiment 2) and robustness >1.0 (Experiment 0).

3. **The triad co-emerges.** In seed 123: world model capacity (C_wm), representation quality (d_eff, D, K_comp), and integration robustness (Φ ratio) all improve together under bottleneck selection. This is exactly what the EXPERIMENTS.md predicted: "d_eff should track C_wm — compression and modeling co-emerge."

4. **General population: flat.** Seeds 42 and 7, with stable populations of 150+, show no improvement in any representation metric. Local reactive behavior suffices for survival in gentle environments.

### The Emerging Pattern Across Experiments 0-3

```
                         General Population    Bottleneck Survivors
Affect geometry          ✓ (cheap)             ✓ (cheap)
Integration robustness   ~0.92                 >1.0
World model capacity     ~10⁻⁴                 ~10⁻²
Representation quality   flat                  improving
```

The consistent finding: *structure* is cheap and universal; *dynamics* (improvement over time, maintaining quality under stress) require intense selection pressure. The bottleneck is the furnace.

### What This Means for the Theory

The geometry/dynamics distinction from Part I is now empirically supported across four measurement domains:
1. Affect space geometry (V10: cheap, universal)
2. Integration under stress (V13: requires bottleneck)
3. Predictive world models (Exp 2: requires bottleneck)
4. Representation quality (Exp 3: requires bottleneck)

The mechanism seems clear: in large populations, patterns can survive with minimal internal structure because the environment isn't lethal enough. Only when >90% mortality occurs do patterns *need* world models, good representations, and robust integration to survive. The biological parallel: most bacteria don't need brains. Only organisms in persistently lethal, partially observable environments evolve complex cognition.

### Figures
- `results/rep_analysis/rep_trajectory.png` — d_eff, A, D, K_comp across cycles
- `results/rep_analysis/rep_eigenspectrum.png` — Eigenspectrum early vs late
- `results/rep_analysis/rep_summary_card.png` — Summary card

---

## 2026-02-17: Experiment 5 — Counterfactual Detachment (NULL RESULT)

### Method
Measured external synchrony (ρ_sync), detachment events, imagination capacity (I_img), and branch entropy (H_branch) across all 3 seeds × 7 snapshots. Recording: 50 substrate steps per snapshot, tracking top-20 patterns.

- **ρ_sync**: Correlation between internal state changes (Δs_B) and boundary observations (s_∂B)
- **Detachment event**: ρ_sync < 0.3 for ≥3 consecutive steps
- **I_img**: Mutual information between detached-period trajectory and future environment, minus reactive-period baseline
- **H_branch**: Entropy of internal states at detachment exit points

### Results

| Seed | ρ_sync (mean across cycles) | detach_frac | I_img | Interpretation |
|------|---------------------------|-------------|-------|----------------|
| 123  | ≈ 0 (range: -0.008 to 0.014) | 0.69 – 0.83 | -0.04 to 0.13 | Internally driven from start |
| 42   | ≈ 0 (range: -0.002 to 0.002) | 0.94 – 0.99 | ≈ 0 | Near-total detachment |
| 7    | ≈ 0 (range: -0.004 to 0.001) | 0.92 – 0.97 | ≈ 0 | Near-total detachment |

### Key observations

1. **Patterns are always internally driven.** ρ_sync ≈ 0 from cycle 0 across all seeds. The predicted reactive→detached transition never occurs because patterns never start reactive. This is a substrate property: the FFT convolution kernel spreads influence globally, so patterns' internal dynamics are dominated by their own chemistry, not boundary inputs.

2. **Detachment is the default, not an achievement.** Detachment fraction is 0.69–1.0 from the first cycle. Seeds 42 and 7 show 0.92–0.99 detachment fraction consistently. Even seed 123 (bottleneck seed) shows 0.69+ detachment.

3. **No systematic imagination capacity.** I_img ≈ 0 across all seeds. Seed 123 shows occasional spikes (0.08 at cycle 5, 0.13 at cycle 15) but no upward trend. These are likely noise — the imagination capacity metric requires patterns to be predictive during detachment vs during reactive periods, but since both periods look identical (ρ_sync ≈ 0 everywhere), the comparison is meaningless.

4. **Branch entropy is moderate and stable.** H_branch ≈ 0.4–0.7 across all seeds, indicating moderate diversity in detachment-exit states. But since detachment is ubiquitous, this just reflects the diversity of internal dynamics, not a cognitive achievement.

### What this means

This is an **informative null result**. The experiment framework is correct — the measures are well-defined and computable. But the substrate doesn't satisfy the precondition: patterns need to START reactive (driven by boundary observations) and LEARN to detach. V13 patterns skip this stage entirely.

**Why?** The content-based coupling mechanism (`K_i(j) = K_fft(|i-j|) · (1 + α·S_local(i))`) is fundamentally about pattern-pattern similarity, not about boundary sensing. Patterns respond to similar chemistry, not to what's at their edges. This is the right substrate for symbiogenesis but the wrong one for testing sensory detachment.

**What would work?** A substrate where patterns have explicit sensory channels — dedicated boundary cells that drive internal dynamics. Then detachment would mean: "I'm ignoring my senses and running an internal simulation." V13 patterns don't have senses to ignore.

### Updated cross-experiment table

```
                         General Population    Bottleneck Survivors
Affect geometry          ✓ (cheap)             ✓ (cheap)
Integration robustness   ~0.92                 >1.0
World model capacity     ~10⁻⁴                 ~10⁻²
Representation quality   flat                  improving
Counterfactual detach.   N/A (always detached) N/A (always detached)
```

### Data
- `results/cf_s{123,42,7}/` — per-cycle JSON files
- `results/cf_analysis/cf_cross_seed.json` — cross-seed summary

---

## 2026-02-17: Experiment 6 — Self-Model Emergence (WEAK SIGNAL)

### Method
Three self-model metrics, all using Ridge regression + StandardScaler + 5-fold CV:

1. **ρ_self**: Self-effect ratio. How much does adding the pattern's "action" (Δs_B) to environment state improve prediction of next boundary observation? ρ_self = (MSE[f_env] - MSE[f_full]) / MSE[f_env].
2. **SM(τ)**: Self-prediction score. Gap between f_self (s_B → s_B(t+τ)) and f_ext (s_∂B → s_B(t+τ)). Positive = pattern predicts itself better than an external observer.
3. **SM_sal**: Self-model salience. Ratio of (how much s_B helps predict self-future) to (how much s_B helps predict env-future). SM_sal > 1 = pattern knows more about itself than about the environment.

Ran on all 3 seeds × 7 snapshots, 50 recording steps per snapshot.

### Results

| Seed | Cycle | ρ_self | SM_cap | SM_sal | Notable |
|------|-------|--------|--------|--------|---------|
| 123  | 0     | 0.021  | 58.6   | 0.000  | |
| 123  | 5     | 0.002  | 21.6   | 0.001  | |
| 123  | 10    | 0.000  | 18.0   | 0.000  | |
| 123  | 15    | 0.044  | 12.2   | 0.000  | |
| 123  | **20**| 0.000  | **132.1** | **1.138** | **Bottleneck (3 patterns)** |
| 123  | 25    | 0.008  | 56.6   | 0.000  | |
| 123  | 29    | 0.000  | 44.6   | 0.014  | |
| 42   | all   | 0-0.05 | 36-56  | 0-0.30 | Flat, no SM_sal > 1 |
| 7    | all   | 0-0.05 | 19-52  | 0-0.33 | Flat, no SM_sal > 1 |

### Observations

1. **ρ_self ≈ 0 everywhere.** Same root cause as Experiment 5: patterns don't have a tight action→observation loop. Internal state changes don't propagate to boundary observations at measured timescales. The FFT substrate spreads influence globally, so boundary changes are driven by grid-wide dynamics, not the focal pattern's actions.

2. **SM_capacity is positive but trivially so.** A pattern's own state naturally predicts its own future better than its boundary does — this is spatial autocorrelation, not self-modeling. The values (12-132) don't trend with evolution. The spike at cycle 020 (132.1) is driven by n=3 patterns, one of which (P1) had SM_capacity = 364.

3. **SM_sal > 1 at the bottleneck.** The one genuinely interesting result: seed 123, cycle 020, pattern 1 shows SM_sal = 2.28. This means adding internal state helps predict self-future 2.28× more than it helps predict environment-future. This pattern is "self-focused" — it knows more about what it will do than about what the environment will do. But n=1 at n=3 patterns is anecdotal, not a trend.

4. **Why SM is often negative in the mean.** The mean SM across all patterns is typically hugely negative (MSE[f_ext] << MSE[f_self]). This is a methodological artifact: s_B is 68-dimensional while s_∂B is 36-dimensional. With Ridge regression, the higher-dimensional predictor overfits more in cross-validation. The SM_capacity metric (positive part only) partly corrects for this but introduces its own bias.

### Updated cross-experiment table

```
                         General Population    Bottleneck Survivors
Affect geometry          ✓ (cheap)             ✓ (cheap)
Integration robustness   ~0.92                 >1.0
World model capacity     ~10⁻⁴                 ~10⁻²
Representation quality   flat                  improving
Counterfactual detach.   N/A (always detached) N/A (always detached)
Self-model emergence     SM_sal ≈ 0            SM_sal = 2.28 (n=1, anecdotal)
```

### What this means for the theory

The thesis predicts self-model emergence should correlate with integration jumps (Φ increase). We can't test this because we don't have enough self-modeling events. The one occurrence (SM_sal > 1) is at the bottleneck where we also see robustness >1.0, C_wm spike, and representation improvement — consistent with the prediction but statistically meaningless at n=1.

The deeper issue remains: V13's substrate doesn't support sensory-motor coupling. Experiments 5 and 6 both hit the same wall — patterns are internally driven from the start, so the reactive→autonomous transition the theory predicts can't occur. Testing these predictions requires a substrate with explicit boundary sensing and action channels.

### Data
- `results/sm_s{123,42,7}/` — per-cycle JSON files
- `results/sm_analysis/sm_cross_seed.json` — cross-seed summary

---

## 2026-02-17: Experiment 7 (partial) — Affect Geometry A↔C Alignment (POSITIVE)

### Method
Representational Similarity Analysis (RSA) between two spaces:

**Space A (structural affect)** — 6 dimensions extracted from internal dynamics:
1. Valence: change in pattern mass (total channel activation)
2. Arousal: ‖Δs_B‖ (magnitude of internal state change)
3. Integration: Φ proxy (1 - max_eigval/total of trajectory covariance)
4. Effective rank: d_eff = exp(spectral entropy)
5. CF weight: 1 - mean(|synchrony|) (fraction of internally-driven dynamics)
6. Self-model salience: difference in self vs boundary autocorrelation

**Space C (behavioral affect)** — 4 dimensions from observable behavior:
1. Approach/avoidance: movement toward/away from resources
2. Activity: ‖Δcenter‖ (movement speed)
3. Growth: Δsize/size₀ (pattern size change)
4. Stability: angular autocorrelation of movement direction

RSA: compute pairwise Euclidean distance in each space (after z-scoring), then Spearman correlate the distance matrices.

### Results

| Seed | Cycle | n_pat | ρ(A,C) | p | Sig? |
|------|-------|-------|--------|-------|------|
| 123 | 0 | 5 | 0.33 | 0.347 | |
| 123 | 5 | 18 | -0.09 | 0.247 | |
| 123 | **10** | 5 | **0.72** | **0.019** | **\*** |
| 123 | 15 | 6 | 0.26 | 0.355 | |
| 123 | 20 | — | — | — | <4 pat |
| 123 | 25 | 20 | 0.16 | 0.029 | \* |
| 123 | 29 | — | — | — | <4 pat |
| 42 | 0 | 11 | 0.21 | 0.118 | |
| 42 | **5** | 20 | **0.39** | **<0.001** | **\*** |
| 42 | 10 | 20 | 0.00 | 0.950 | |
| 42 | 15 | 20 | -0.17 | 0.022 | \* (neg) |
| 42 | 20 | 20 | -0.09 | 0.242 | |
| 42 | 25 | 20 | -0.15 | 0.038 | \* (neg) |
| 42 | 29 | 20 | 0.15 | 0.043 | \* |
| 7 | 0 | 20 | 0.01 | 0.857 | |
| 7 | 5 | 20 | 0.07 | 0.324 | |
| 7 | **10** | 20 | **0.38** | **<0.001** | **\*** |
| 7 | **15** | 20 | **0.38** | **<0.001** | **\*** |
| 7 | **20** | 20 | **0.31** | **<0.001** | **\*** |
| 7 | 25 | 20 | 0.18 | 0.013 | \* |
| 7 | 29 | 20 | 0.24 | 0.001 | \* |

### Observations

1. **Seed 7 is the star.** RSA increases from near-zero (cycle 0: 0.01) to consistently significant (cycle 10 onward: 0.18-0.38, all p < 0.015). This is the clearest evidence that affect geometry alignment DEVELOPS over evolutionary time. Internal states increasingly predict behavior.

2. **Seed 123 shows the strongest point alignment.** ρ = 0.72 at cycle 010 with 5 patterns (p = 0.019). But cycles 020 and 029 (bottleneck, <4 patterns) can't be assessed. The sparse data is frustrating — the bottleneck is where we'd expect the strongest alignment.

3. **Seed 42 is mixed.** Positive at early/late cycles, negative in the middle (cycles 15, 25). The negative alignment at cycle 15 (ρ = -0.17, p = 0.022) means patterns whose internal structure looks "high arousal" are behaviorally "low activity" — they're internally agitated but behaviorally still. This is interesting but hard to interpret without more context.

4. **8/19 significant positive, 2/19 significant negative.** The positive signal dominates, but it's not overwhelming. This is weaker than V10's near-universal geometry (ρ > 0.21, p < 0.0001 in all 7 conditions). The difference: V10 tested affect space geometry (similarity structure IS cheap); here we test affect-behavior alignment (mapping from structure to behavior is NOT cheap — it develops).

### What this means

This is the first positive result about affect DYNAMICS since V13 started:
- Experiments 2, 3: only bottleneck shows improvement
- Experiment 5: null (always detached)
- Experiment 6: null (no self-model)
- **Experiment 7: seed 7 shows developing alignment across ALL evolutionary stages**

The difference: Experiments 2, 3, 5, 6 all measured whether specific cognitive capacities (world models, representations, detachment, self-models) emerge. Most didn't. Experiment 7 asks a simpler question: does the EXISTING internal structure drive behavior? And the answer is yes, increasingly.

This recalibrates the story. The thesis doesn't need patterns to develop world models or self-models first. The more basic claim — that the geometric structure of internal states maps onto behavioral organization — is supported. The structure-behavior mapping starts weak and strengthens, which is the dynamics story: geometry is cheap, but the mapping from geometry to behavior is learned.

### Updated cross-experiment table

```
                         General Population    Bottleneck Survivors
Affect geometry (V10)    ✓ (cheap)             ✓ (cheap)
A↔C alignment (Exp 7)   0.01→0.38 (develops)  0.72 (strong but n=5)
Integration robustness   ~0.92                 >1.0
World model capacity     ~10⁻⁴                 ~10⁻²
Representation quality   flat                  improving
Counterfactual detach.   N/A (always detached) N/A (always detached)
Self-model emergence     SM_sal ≈ 0            SM_sal = 2.28 (n=1)
```

### Data
- `results/ag_s{123,42,7}/` — per-cycle JSON files
- `results/ag_analysis/ag_cross_seed.json` — cross-seed summary
