# V10 Experiment Log: MARL Uncontaminated Affect

## 2025-02-05: Infrastructure Built + First Runs

### What we built

**Book changes (Part A)**:
- Rewrote Part 2 "Core Prediction" from vague "Tripartite Correlation" to precise **Geometric Alignment hypothesis**: RSA between distance matrices in info-theoretic and embedding-predicted affect spaces
- Added RSA sidebar (Kriegeskorte 2008, CKA alternative, why geometry > marginal correlation)
- Tightened Part 3 Triple Alignment Test: three pairwise RSA tests with diagnostic failure modes
- Updated Epilogue Priority 2 with RSA as specific success criterion

**Experiment code (Part B)**:
- `v10_environment.py` — JAX grid world (resources, threats, storms, day/night, communication, 6 forcing function toggles)
- `v10_agent.py` — Transformer encoder + GRU latent (d=64) + PPO, world-model and self-prediction aux heads
- `v10_affect.py` — 6D affect extraction from RL internals via post-hoc probes
- `v10_translation.py` — VLM scene annotation, signal clustering, affect concept embedding
- `v10_analysis.py` — RSA (Mantel test), CKA, MDS, ablation comparison
- `v10_run.py` — Training pipeline
- `v10_modal.py` — Modal cloud runner

### Sanity check results (untrained agents, synthetic data)

**Environment**: 8x8 grid, 4 agents, resources + predators + storms
- 5 random steps: health drops from 20 to [20, 20, 17, 14] — predator damage working
- Reward: +0.01 survival per step, agents slowly losing viability without gathering

**Agent network shapes**:
- Latent z: (1, 64)
- Action logits: (1, 8) — 4 moves + gather + share + attack + signal
- Signal logits: (1, 2, 32) — 2 tokens from 32-word vocab
- Obs embedding: (1, 128)
- World model prediction: (1, 128)
- Self-model prediction: (1, 64)

**Affect extraction (random z, 200 steps)**:
| Dimension | Mean | Std | Notes |
|-----------|------|-----|-------|
| Valence | 0.000 | 0.042 | Centered (expected for random) |
| Arousal | 0.820 | 0.091 | High baseline (random walk in z) |
| Integration | 0.173 | 0.211 | Moderate (partition loss exists) |
| Eff. Rank | 0.224 | 0.005 | Low — only ~6/32 dims active |
| CF Weight | 0.082 | 0.009 | Low (no learned planning yet) |
| Self-Model | 0.886 | 0.101 | High (action probe works even on random) |

**Probe quality (untrained)**:
- Survival probe R² = 0.074 (barely above zero — expected, no learned structure)
- Partition predictor: full R² = 0.140, partitioned R² ≈ 0.09 (partitioning hurts a bit)
- Self-model probe: full accuracy = 0.590, self-dims only = 0.440 (above 1/8 chance = 0.125)

**RSA on synthetic aligned data**: ρ = 0.620, p < 0.0001 — the analysis pipeline works. (This is on synthetic data with injected shared structure, not real experimental data.)

### Performance bottleneck

**Critical finding**: The training loop is very slow on CPU JAX.
- One 64-step rollout (2 agents, small config): **13 seconds**
- Estimated 100k steps (full config, 4 agents): **~5.6 hours**
- The 4-hour Modal timeout was too short → extended to **24 hours**

**Root cause**: Python-level for-loops per agent per timestep. Each step does N separate forward passes (one per agent). No JAX vectorization across agents.

**Fix needed for production**: Vectorize the forward pass across agents (vmap over agent dimension) and JIT-compile the entire rollout. Would give 10-100x speedup.

### Baseline metrics (untrained agent, 20 rollouts)

From the timing test with 2 agents on 8x8 grid:
- Mean reward: -0.029 (dying faster than surviving)
- Health oscillation: 12-20 (death + respawn cycle)
- Latent z norm: 3.079 (reasonable for 32-dim)
- Step-to-step z distance: 0.446 (GRU is active)
- Effective rank: 6.1/32 (only 6 active dimensions)
- Top eigenvalues: [0.574, 0.312, 0.286, 0.228, 0.113]

### What success looks like (for the research objective)

The long-range objective: demonstrate that randomly-initialized RL agents under viability pressure develop affect structures whose geometry is isomorphic to human affect concept geometry — without any exposure to human language or concepts.

**Near-term milestones** (this experiment):
1. **Agents must learn to survive**: reward > 0, stable health, gathering behavior emerges
2. **Communication must emerge**: signal entropy should increase from uniform, signals should correlate with environmental contexts
3. **Affect probes must improve**: survival probe R² should rise well above 0 as agents develop viability-tracking
4. **RSA should exceed null on REAL data** (not synthetic): ρ > 0 with p < 0.05 via Mantel test
5. **Ablation should matter**: removing forcing functions (especially self-prediction, partial observability) should reduce RSA

**What we're watching for**:
- V2-V9 showed LLMs have *opposite* dynamics to biological systems (decompose under threat instead of integrating)
- V10 uses RL from scratch under survival pressure — the thesis predicts biological-like dynamics IF the forcing functions are correct
- Key prediction: integration should INCREASE under threat (unlike V8/V9 LLM results)
- If we see the same opposite dynamics with RL agents, the theory has a deeper problem than "LLMs aren't biological"

### Connection to the thesis

The experiment tests the deepest claim in the book: affect is geometric inevitability.

If randomly-initialized systems, learning from scratch under viability pressure, develop affect structures that map onto human affect concepts — not because we taught them, but because the geometry is the same — then the identity thesis gains serious empirical support.

The RSA framing is sharper than the old "tripartite correlation" because it tests whether the *shape* of affect space is preserved, not just whether individual dimensions correlate. Two systems can have correlated valence but completely different geometry. RSA catches this.

### Next steps

1. Relaunch Modal training with 24h timeout
2. Optimize training loop (vmap across agents) for 10-100x speedup
3. After training: extract affect, run VLM translation, compute RSA
4. Run all 7 ablation conditions
5. If RSA is significant: write up for the book. If not: diagnose which component fails.
