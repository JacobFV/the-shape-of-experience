# V11.4: High-Dimensional Multi-Channel Lenia

## Design

V11.4 extends the multi-channel Lenia substrate from 3 channels (V11.3) to 64 continuous channels with fully vectorized physics.

### Architecture

| Component | V11.3 (3-channel) | V11.4 (64-channel) |
|---|---|---|
| Physics loop | Python for-loop over C | `jnp.einsum` + broadcast |
| Kernel FFTs | Python list of 3 arrays | Stacked `(C, N, N//2+1)` array |
| Coupling | 3x3 matrix, Python inner loop | CxC banded toroidal matrix, einsum |
| Growth | Per-channel with `channel_configs[c]` | Broadcast with `(C,1,1)` arrays |
| Gate | `sigmoid(5*(cross_term - 0.3))` per channel | Normalized: `sigmoid(5*(cross/rowsum - 0.3))` |
| XLA graph | Scales with C (unrolled loops) | Independent of C |

### Key Design Decisions

**Coupling matrix**: Banded toroidal structure `W[i,j] = exp(-d(i,j)^2 / (2*bw^2))` where `d` is toroidal channel distance. Row sums normalized to 3.0. This creates local coupling in "channel space" — nearby channels strongly coupled, distant channels weakly. Bandwidth is evolvable.

**Growth parameters**: Per-channel `mu` and `sigma` sampled from Beta distributions, kernel radii log-spaced from 5 to 25. This gives channels different spatial scales (some local, some global) and different viability manifold geometries.

**Gate normalization**: `cross_terms / row_sums` before sigmoid, so the gate operates on [0,1] regardless of C. Without this, increasing C would saturate the gate.

### Phi Measurement

Two approaches for channel integration at C=64:

1. **Spectral Phi** (fast, O(C^2 n + C^3)): Coupling-weighted covariance of channel values at pattern cells, effective rank of eigenvalues / C. Used for routine per-cycle measurement.

2. **Sampled MIP Phi** (accurate, O(K*C*N^2)): K=16 random binary channel partitions, measure growth disruption when cross-partition coupling is zeroed. MIP = minimum across samples. Used for fitness scoring.

**Validation**: At C=3, spectral Phi correlates with exact channel-partition Phi (Spearman rho=0.75, p=0.013). Correctly identifies the one genuinely integrated pattern vs zero-integration patterns.

## Results

### Compilation and Performance

| C | N | Compile | Steps/s (CPU) | Patterns | Grid Memory |
|---|---|---|---|---|---|
| 8 | 128 | 0.2s | 800 | 31 | 0.5 MB |
| 16 | 128 | 0.3s | ~400 | ~50 | 1.0 MB |
| 32 | 256 | 1.9s | ~100 | ~45 | 8.4 MB |
| 64 | 256 | 1.4s | 40 | 480 | 16.8 MB |

### C-Sweep: Naive Stress Response by Dimensionality

Protocol: 5000-step warmup, 1500-step baseline, 1500-step drought (regen -> 0), measure spectral Phi change. No evolution — pure substrate physics. 3 random seeds per condition.

| C | Mean Delta Phi | Std | Pattern |
|---|---|---|---|
| 3 | -0.0% | 0.0% | Flat (stuck at floor) |
| 8 | +0.5% | 1.3% | Weak integration |
| 16 | +1.2% | 6.7% | Integration (high variance) |
| 32 | -2.5% | 1.4% | Decomposition |
| 64 | -1.0% | 1.9% | Weak decomposition |

**Interpretation**: At C=3, spectral Phi is at its floor (1/C = 0.333) — no cross-channel coupling to disrupt. At C=8-16, the mid-range dimensionality produces weak but positive Phi change under stress (the biological pattern) even without evolution. At C=32-64, the higher dimensionality returns to decomposition — the coupling is too diffuse for spontaneous integration.

This is consistent with the hypothesis that evolution is needed to *discover* integration strategies at high C. The substrate at C=8-16 accidentally produces integration-like responses because the coupling bandwidth is well-matched to the channel count. At C=64, the space is too large for random coupling to produce coherent integration — this is exactly where evolution should make the biggest difference.

### Spectral Phi Validation (C=3)

| Metric | Value |
|---|---|
| Pearson correlation (exact vs spectral) | r = 1.000 |
| Spearman correlation (exact vs spectral) | rho = 0.745, p = 0.013 |
| Exact Phi range | [0, 0.037] |
| Spectral Phi range | [0.333, 0.852] |

Note: Pearson is degenerate (9 of 10 patterns have zero exact Phi, mapping to spectral 0.333). Spearman is the meaningful test — correctly orders patterns by integration level.

## Files

| File | Purpose |
|---|---|
| `v11_substrate_hd.py` | Vectorized physics: config gen, coupling gen, `run_chunk_hd`, `init_soup_hd` |
| `v11_affect_hd.py` | Spectral Phi, sampled MIP Phi, spatial Phi, `measure_all_hd` |
| `v11_evolution.py` | `evolve_hd`, `stress_test_hd`, `full_pipeline_hd` (appended) |
| `v11_run.py` | `hd` and `hd-pipeline` modes, `--channels` flag |
| `v11_modal.py` | `run_hd` function, `--channels` parameter |
| `v11_visualize.py` | Labeled video generation (HD demo + C-sweep) |

## Running

```bash
# Quick local test (C=8)
python v11_run.py hd 2 --channels 8

# Full local run (C=64, ~slow on CPU)
python v11_run.py hd 5 --channels 64

# GPU run (recommended for real results)
modal run v11_modal.py --mode hd --hours 4

# GPU with different C
modal run v11_modal.py --mode hd --hours 4 --channels 32

# Generate videos
python v11_visualize.py hd 8       # C=8 demo
python v11_visualize.py sweep       # C-sweep comparison
```

## Next Steps

1. **GPU evolutionary run**: `modal run v11_modal.py --mode hd --hours 4` — the main experiment. Does evolution at C=64 produce better stress robustness than C=3?

2. **C-sweep with evolution**: Run 10 cycles of evolution at each C, then stress test. Compare evolved vs naive delta at each dimensionality. Tests whether the dimensionality threshold shifts under selection.

3. **Bandwidth optimization**: The coupling bandwidth is currently fixed or randomly perturbed. Systematic sweep of bandwidth vs C to find the optimal coupling structure.

4. **Book update with GPU results**: Once we have evolved trajectories, update Part 1 with quantitative results for V11.4.

## Connection to Thesis

This experiment directly tests the "internal dimensionality" hypothesis from Part 1: that biological-like integration requires a substrate with enough degrees of freedom for evolution to sculpt. V11.0-V11.2 showed that even with evolution, a single-channel substrate stalls at +2.1pp. V11.3's three channels added channel-partition Phi but the integration signal was near zero. V11.4 provides the high-dimensional substrate where channel integration becomes the primary phenomenon.

The C-sweep finding — that mid-range C spontaneously shows integration while high C decomposes without evolution — is especially interesting. It suggests that the "ladder" from the book has a dimensionality component: each rung may require not just the right selection pressure, but also a minimum (and perhaps optimal) internal dimensionality.
