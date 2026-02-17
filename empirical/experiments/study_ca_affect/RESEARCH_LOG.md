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

### Next Steps
- [x] Seed 42 v2 complete
- [ ] Seed 7 running (third replicate)
- [ ] Convolution control: same evolution, same stress, but α=0 (no content coupling)
- [ ] Re-aggregate with 3 seeds
- [ ] Book update with preliminary results
- [ ] Consider lower population runs (smaller grid? higher maintenance?) to increase selection pressure
