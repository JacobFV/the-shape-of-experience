# The 1D Collapse: RETRACTED — Snapshot Timing Bug

## Date: 2026-02-19 (original), 2026-02-20 (retraction)
## Status: RETRACTED. The original finding was an artifact of a bug.

---

## The Bug

V20, V25, and V26 evolution loops reset hidden states to zero BEFORE saving
snapshots. The code order was:

```python
# Reset hidden states and energy for next cycle
state['hidden'] = jnp.zeros_like(state['hidden'])
state['energy'] = jnp.where(state['alive'], cfg['initial_energy'], 0.0)

# ... print, logging ...

# Save snapshot  <-- captures ZERO hidden states!
snap = extract_snapshot(state, cycle, cfg)
```

This meant ALL hidden state analysis (effective rank, energy R², position R²,
silhouette scores) was performed on **all-zero vectors**. The energy R²=1.0
result was because both hidden=0 and energy=initial_energy are constants
after reset — trivially perfect regression on constants.

## Corrected Results (V22-V24)

V22-V24 evolution loops reset at cycle START (line 339-349), so their
snapshots captured the actual post-cycle hidden states. Re-analysis shows:

| Metric | Original (buggy V20b) | Corrected (V22-V24) |
|--------|-----------------------|---------------------|
| Effective rank | 1-3 | **5.1-7.3** |
| Energy R² | 1.0 | **-3.6 to -4.6** (negative!) |
| Position R² | ~0 | -0.06 to -0.11 |
| PC1 variance | ~95-100% | **25-38%** |
| N nonzero dims | 0-3 | ~32/32 (all active) |

The hidden states are **moderately high-dimensional** and do NOT encode
energy linearly. The "1D energy counter" story was completely wrong.

## What This Means

1. **The agents DO use their hidden state richly** — effective rank 5-7 means
   ~5-7 independent dimensions of variation across the population
2. **Energy is NOT the dominant feature** — negative R² means a linear model
   predicting energy from hidden state does WORSE than predicting the mean
3. **The "environment doesn't demand representation" story was premature** —
   it was built on artifactual data

## What We Don't Yet Know

The corrected V22-V24 data shows rich hidden states, but what ARE they encoding?
Energy R² is negative, position R² is negative. The hidden state varies across
agents (eff rank 5-7) but we can't decode any environmental feature from it
using linear regression. Candidates:
- Temporal patterns (movement history, resource encounter sequence)
- Action policy state (which decision mode the agent is in)
- Random walk accumulation (GRU integrating noise without useful structure)
- Nonlinear encodings of environmental features that linear regression can't decode

## Files Affected

- `v25_evolution.py` — FIXED (snapshot before reset)
- `v26_evolution.py` — FIXED (snapshot before reset)
- `v20_evolution.py` — BUG STILL PRESENT (not fixed, historical)
- `v22_evolution.py`, `v23_evolution.py`, `v24_evolution.py` — NOT affected
  (reset at cycle start, snapshot at end)

## Lesson

Always verify that analysis data captures the intended state. When hidden states
appear degenerate, first check that you're not analyzing zeroed/reset buffers.
