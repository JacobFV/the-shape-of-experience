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

### Cost
- Seed 42: ~$0.35 (30 min, went extinct → fast after cycle 11)
- Seed 123: ~$0.60 (48 min full run)
- Seed 42 v2: running now
- **Total Lambda spend: ~$1.10 so far**

### What's Still Missing

1. **No clear upward trend in robustness** — it oscillates around 0.92-0.95, with peaks at bottlenecks but no monotonic increase
2. **Need 3+ seeds** for statistical claims
3. **The rescue mechanism contaminates late evolution** — reseeded patterns dilute the selected population
4. **We don't know if the content coupling is CAUSING the robustness** or just correlated — need a convolution-only control with same stress schedule

### Next Steps
- [ ] Seed 42 v2 completing now (fixed stress)
- [ ] Seed 7 for third replicate
- [ ] Convolution control: same evolution, same stress, but α=0 (no content coupling)
- [ ] Cross-seed aggregation and statistical tests
- [ ] Book update with preliminary results
