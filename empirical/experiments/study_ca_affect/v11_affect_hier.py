"""V11.5 Hierarchical Affect Measurement.

Extends V11.4 affect measures with tier-aware measurements that map
directly to Part 2's affective dimensions:

Core 6D:
1. Valence       — Mass change (viability gradient proxy)
2. Arousal       — Rate of state change (KL divergence proxy)
3. Integration   — Cross-tier + within-tier Phi
4. Effective Rank — Eigenvalue distribution of tier activations
5. CF Weight     — Memory channel divergence from current state
6. Self-Model    — Prediction channel accuracy

Extended measures:
7. World-Model Accuracy  — How well memory predicts resource field
8. Tier Integration      — Information flow between tiers (MIP across tiers)
9. Self-Model Scope      — What fraction of the pattern is self-modeled
10. Metacognitive Depth  — Does prediction accuracy modulate behavior?

These measures characterize the affect motifs from Part 2:
- Suffering: high integration + low effective rank (trapped)
- Fear: high CF weight, high self-model salience, negative valence
- Curiosity: positive valence toward uncertainty, high CF weight
- Boredom: low arousal, low integration, low rank
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from v11_affect import AffectState, measure_valence, measure_self_model_salience
from v11_affect_hd import spectral_channel_phi
from v11_substrate_hier import (
    TIER_SENSORY, TIER_PROCESSING, TIER_MEMORY, TIER_PREDICTION
)


# ============================================================================
# Extended Affect State
# ============================================================================

@dataclass
class HierAffectState:
    """Extended affect measurement for hierarchical patterns.

    Core 6D (Part 2 framework):
    """
    # Core 6D
    valence: float = 0.0
    arousal: float = 0.0
    integration: float = 0.0
    effective_rank: float = 0.0
    counterfactual_weight: float = 0.0
    self_model_salience: float = 0.0

    # Extended hierarchical measures
    world_model_accuracy: float = 0.0
    tier_integration: float = 0.0      # cross-tier Phi
    self_model_scope: float = 0.0      # fraction of pattern self-modeled
    metacognitive_depth: float = 0.0   # prediction accuracy modulates growth

    # Per-tier integration
    sensory_phi: float = 0.0
    processing_phi: float = 0.0
    memory_phi: float = 0.0
    prediction_phi: float = 0.0

    # Metadata
    pattern_id: int = -1
    step: int = 0
    mass: float = 0.0
    size: int = 0

    def to_array_6d(self) -> np.ndarray:
        """Core 6D affect vector (Part 2 compatible)."""
        return np.array([
            self.valence, self.arousal, self.integration,
            self.effective_rank, self.counterfactual_weight,
            self.self_model_salience
        ])

    def to_array_full(self) -> np.ndarray:
        """Full 10D affect vector including extended measures."""
        return np.array([
            self.valence, self.arousal, self.integration,
            self.effective_rank, self.counterfactual_weight,
            self.self_model_salience,
            self.world_model_accuracy, self.tier_integration,
            self.self_model_scope, self.metacognitive_depth,
        ])

    @staticmethod
    def dim_names_6d():
        return ['valence', 'arousal', 'integration',
                'effective_rank', 'cf_weight', 'self_model']

    @staticmethod
    def dim_names_full():
        return ['valence', 'arousal', 'integration',
                'effective_rank', 'cf_weight', 'self_model',
                'world_model', 'tier_integration',
                'self_scope', 'metacog_depth']


# ============================================================================
# Dimension 2: Arousal (KL divergence proxy)
# ============================================================================

def measure_arousal_hier(pattern_values, prev_values, tier_assignments):
    """Arousal = rate of state change, weighted by tier.

    Sensory arousal weighted 2x (fast changes matter more).
    Memory arousal weighted 0.5x (slow changes less salient).

    Args:
        pattern_values: (N_cells, C) current channel values
        prev_values: (N_cells, C) previous channel values
        tier_assignments: (C,) int array of tier assignments
    """
    if prev_values is None or pattern_values is None:
        return 0.0

    v_curr = np.asarray(pattern_values)
    v_prev = np.asarray(prev_values)

    if v_curr.shape != v_prev.shape:
        return 0.0

    # Per-channel L2 change, averaged over cells
    delta = np.mean((v_curr - v_prev)**2, axis=0)  # (C,)

    # Tier weighting
    weights = np.ones_like(delta)
    weights[tier_assignments == TIER_SENSORY] = 2.0
    weights[tier_assignments == TIER_PROCESSING] = 1.0
    weights[tier_assignments == TIER_MEMORY] = 0.5
    weights[tier_assignments == TIER_PREDICTION] = 1.5

    return float(np.sum(delta * weights) / np.sum(weights))


# ============================================================================
# Dimension 3: Integration (cross-tier Phi)
# ============================================================================

def measure_tier_integration(grid_mc, coupling, cells, tier_assignments):
    """Cross-tier integration: how much information flows between tiers.

    Measures: what happens if we partition by tier (remove all cross-tier
    coupling)? The growth change quantifies tier integration.

    This is the hierarchical analogue of channel-partition Phi.
    """
    C = grid_mc.shape[0]
    if len(cells) < 4:
        return 0.0

    # Extract values at pattern cells: (C, n_cells)
    vals = np.asarray(grid_mc[:, cells[:, 0], cells[:, 1]], dtype=np.float64)
    vals = np.clip(vals, 0.0, 1.0)
    n = vals.shape[1]
    if n < 4:
        return 0.0

    # Only include channels with nonzero variance
    var_per_ch = np.var(vals, axis=1)
    active_ch = var_per_ch > 1e-12
    if np.sum(active_ch) < 2:
        return 0.0
    vals = vals[active_ch]
    # Remap tier assignments for active channels only
    active_tier_assignments = np.asarray(tier_assignments)[active_ch]

    # Full covariance (float64 for numerical stability at C=64)
    vals_c = vals - vals.mean(axis=1, keepdims=True)
    cov_full = (vals_c @ vals_c.T) / max(n - 1, 1)

    # Block-diagonal covariance (zero out cross-tier blocks)
    n_active = int(np.sum(active_ch))
    mask = np.zeros((n_active, n_active), dtype=np.float64)
    for tier in range(4):
        idx = np.where(active_tier_assignments == tier)[0]
        if len(idx) > 0:
            mask[np.ix_(idx, idx)] = 1.0

    cov_within = cov_full * mask

    # Effective rank difference: full vs block-diagonal (all numpy float64)
    eigvals_full = np.linalg.eigvalsh(cov_full)
    eigvals_full = np.maximum(eigvals_full, 0.0)
    sum_sq_full = np.sum(eigvals_full**2)
    eff_rank_full = np.sum(eigvals_full)**2 / sum_sq_full if sum_sq_full > 1e-20 else 1.0

    eigvals_within = np.linalg.eigvalsh(cov_within)
    eigvals_within = np.maximum(eigvals_within, 0.0)
    sum_sq_within = np.sum(eigvals_within**2)
    eff_rank_within = np.sum(eigvals_within)**2 / sum_sq_within if sum_sq_within > 1e-20 else 1.0

    # Tier integration = fraction of effective rank from cross-tier correlations
    tier_phi = float((eff_rank_full - eff_rank_within) / max(eff_rank_full, 1.0))

    return max(0.0, tier_phi)


def measure_per_tier_phi(grid_mc, coupling, cells, tier_assignments):
    """Per-tier spectral Phi: integration within each tier.

    Returns dict with phi for each tier.
    """
    C = grid_mc.shape[0]
    result = {}

    for tier, name in enumerate(['sensory', 'processing', 'memory', 'prediction']):
        tier_ch = np.where(np.asarray(tier_assignments) == tier)[0]
        if len(tier_ch) < 2 or len(cells) < 4:
            result[name] = 0.0
            continue

        # Extract just this tier's channels
        tier_vals = grid_mc[tier_ch][:, cells[:, 0], cells[:, 1]]  # (n_tier, n_cells)
        n = tier_vals.shape[1]

        vals_c = tier_vals - tier_vals.mean(axis=1, keepdims=True)
        cov = (vals_c @ vals_c.T) / max(n - 1, 1)

        eigvals = jnp.linalg.eigvalsh(cov)
        eigvals = jnp.maximum(eigvals, 0.0)

        tr = jnp.sum(eigvals)
        tr_sq = jnp.sum(eigvals**2)
        eff_rank = jnp.where(tr_sq > 1e-20, tr**2 / tr_sq, 1.0)

        result[name] = float(eff_rank) / len(tier_ch)

    return result


# ============================================================================
# Dimension 4: Effective Rank (tier-aware)
# ============================================================================

def measure_effective_rank_hier(pattern_values, tier_assignments):
    """Effective rank of pattern state, computed per-tier and globally.

    Maps to Part 2's r_eff: dimensionality of representation being utilized.
    Low rank = trapped in narrow manifold (suffering signature).
    High rank = expansive (joy/curiosity signature).
    """
    if pattern_values is None:
        return 0.0, {}

    vals = np.asarray(pattern_values, dtype=np.float64)  # (N_cells, C)
    vals = np.clip(vals, 0.0, 1.0)  # ensure valid range
    if vals.shape[0] < 4:
        return 0.0, {}

    # Only include channels with nonzero variance
    var_per_ch = np.var(vals, axis=0)
    active_ch = var_per_ch > 1e-12
    if np.sum(active_ch) < 2:
        return 0.0, {}
    vals_active = vals[:, active_ch]

    # Global effective rank (float64 for numerical stability)
    vals_c = vals_active - vals_active.mean(axis=0, keepdims=True)
    cov = (vals_c.T @ vals_c) / max(vals_active.shape[0] - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0.0)

    tr = np.sum(eigvals)
    tr_sq = np.sum(eigvals**2)
    global_rank = (tr**2 / tr_sq) / vals_active.shape[1] if tr_sq > 1e-20 else 0.0

    # Per-tier rank
    tier_ranks = {}
    for tier, name in enumerate(['sensory', 'processing', 'memory', 'prediction']):
        tier_ch = np.where(np.asarray(tier_assignments) == tier)[0]
        if len(tier_ch) < 2:
            tier_ranks[name] = 0.0
            continue
        tier_vals = vals[:, tier_ch]
        tv_c = tier_vals - tier_vals.mean(axis=0, keepdims=True)
        tcov = (tv_c.T @ tv_c) / max(vals.shape[0] - 1, 1)
        teig = np.linalg.eigvalsh(tcov)
        teig = np.maximum(teig, 0.0)
        ttr = np.sum(teig)
        ttr_sq = np.sum(teig**2)
        tier_ranks[name] = float((ttr**2 / ttr_sq) / len(tier_ch)) if ttr_sq > 1e-20 else 0.0

    return float(global_rank), tier_ranks


# ============================================================================
# Dimension 5: Counterfactual Weight
# ============================================================================

def measure_counterfactual_weight(pattern_values, tier_assignments):
    """CF weight = how much memory channels diverge from current sensory state.

    In a system with memory, counterfactual processing manifests as
    memory channels maintaining representations of non-current states.
    High CF weight = memory is divergent from present (planning/worrying).
    Low CF weight = memory tracks present (reactive/flow).

    Maps to Part 2's CF dimension: resources allocated to non-actual trajectories.
    """
    if pattern_values is None:
        return 0.0

    vals = np.asarray(pattern_values)  # (N_cells, C)

    sens_ch = np.where(np.asarray(tier_assignments) == TIER_SENSORY)[0]
    mem_ch = np.where(np.asarray(tier_assignments) == TIER_MEMORY)[0]

    if len(sens_ch) < 1 or len(mem_ch) < 1:
        return 0.0

    # Current sensory state: average activation per cell
    sens_mean = vals[:, sens_ch].mean(axis=1)  # (N_cells,)

    # Memory state: average activation per cell
    mem_mean = vals[:, mem_ch].mean(axis=1)  # (N_cells,)

    # CF weight = normalized divergence between memory and sensory
    sens_norm = np.linalg.norm(sens_mean)
    mem_norm = np.linalg.norm(mem_mean)

    if sens_norm < 1e-8 or mem_norm < 1e-8:
        return 0.0

    # Cosine distance (1 - cosine similarity)
    cos_sim = np.dot(sens_mean, mem_mean) / (sens_norm * mem_norm)
    cf_weight = 1.0 - cos_sim  # 0 = identical, 2 = opposite

    return float(np.clip(cf_weight, 0.0, 1.0))


# ============================================================================
# Dimension 6: Self-Model Salience
# ============================================================================

def measure_self_model_hier(pattern_values, tier_assignments):
    """Self-model salience from prediction channel activity.

    High prediction channel activity relative to total = high self-model salience.
    Maps to Part 2's SM: how prominently self appears in processing.
    """
    if pattern_values is None:
        return 0.0

    vals = np.asarray(pattern_values)  # (N_cells, C)

    pred_ch = np.where(np.asarray(tier_assignments) == TIER_PREDICTION)[0]
    if len(pred_ch) < 1:
        return 0.0

    # Prediction channel energy vs total energy
    pred_energy = np.mean(vals[:, pred_ch]**2)
    total_energy = np.mean(vals**2)

    if total_energy < 1e-10:
        return 0.0

    # Normalized: fraction of total energy in prediction channels
    # Adjusted for tier size (prediction is smaller, so raw fraction is biased)
    C = vals.shape[1]
    expected_fraction = len(pred_ch) / C
    actual_fraction = pred_energy / total_energy

    # Salience = how much prediction is overrepresented, normalized to [0, 1]
    # 1.0 = prediction has ALL the energy; 0.0 = prediction has expected share or less
    ratio = actual_fraction / max(expected_fraction, 1e-8)
    # Sigmoid mapping: ratio=1 → 0, ratio=3 → ~0.73, ratio=5+ → ~0.95
    salience = 1.0 - np.exp(-max(0.0, ratio - 1.0) / 2.0)

    return float(np.clip(salience, 0.0, 1.0))


# ============================================================================
# Extended: World-Model Accuracy
# ============================================================================

def measure_world_model_accuracy(grid_mc, resource, cells, tier_assignments):
    """How well do memory channels predict the local resource field?

    For each pattern cell, compare memory channel values to local resource.
    High correlation = pattern has implicit world model.

    Maps to Part 2's learned world model forcing function.
    """
    if len(cells) < 4:
        return 0.0

    mem_ch = np.where(np.asarray(tier_assignments) == TIER_MEMORY)[0]
    if len(mem_ch) < 1:
        return 0.0

    # Resource values at pattern cells
    resource_np = np.asarray(resource)
    r_vals = resource_np[cells[:, 0], cells[:, 1]]  # (N_cells,)

    # Memory channel values at pattern cells
    grid_np = np.asarray(grid_mc)
    mem_vals = grid_np[mem_ch][:, cells[:, 0], cells[:, 1]]  # (n_mem, N_cells)

    # For each memory channel, correlation with resource
    best_corr = 0.0
    r_centered = r_vals - r_vals.mean()
    r_std = r_vals.std()
    if r_std < 1e-8:
        return 0.0

    for mc in range(len(mem_ch)):
        m_centered = mem_vals[mc] - mem_vals[mc].mean()
        m_std = mem_vals[mc].std()
        if m_std < 1e-8:
            continue
        corr = abs(np.dot(r_centered, m_centered) / (len(cells) * r_std * m_std))
        best_corr = max(best_corr, corr)

    return float(best_corr)


# ============================================================================
# Extended: Self-Model Scope
# ============================================================================

def measure_self_model_scope(pattern_values, tier_assignments):
    """What fraction of the pattern's state is captured by prediction channels?

    Measures explained variance: how much of processing+sensory variance
    is linearly predictable from prediction channels.

    Wide scope = prediction channels capture most of the pattern's dynamics.
    Narrow scope = prediction only covers part of the state.

    Maps to Part 2's self-model scope concept.
    """
    if pattern_values is None:
        return 0.0

    vals = np.asarray(pattern_values)
    if vals.shape[0] < 10:
        return 0.0

    pred_ch = np.where(np.asarray(tier_assignments) == TIER_PREDICTION)[0]
    other_ch = np.where(np.asarray(tier_assignments) != TIER_PREDICTION)[0]

    if len(pred_ch) < 1 or len(other_ch) < 1:
        return 0.0

    # Simple measure: canonical correlation between prediction and other channels
    pred_vals = vals[:, pred_ch]   # (N_cells, n_pred)
    other_vals = vals[:, other_ch]  # (N_cells, n_other)

    # Center
    pred_c = pred_vals - pred_vals.mean(axis=0, keepdims=True)
    other_c = other_vals - other_vals.mean(axis=0, keepdims=True)

    # Cross-covariance
    cross = (pred_c.T @ other_c) / max(vals.shape[0] - 1, 1)

    # Singular values of cross-covariance = canonical correlations
    try:
        sv = np.linalg.svd(cross, compute_uv=False)
        # Scope = fraction of variance captured by top singular values
        total_sv = np.sum(sv)
        if total_sv < 1e-8:
            return 0.0
        # Normalized effective rank of singular values
        sv_norm = sv / total_sv
        scope = float(-np.sum(sv_norm * np.log(sv_norm + 1e-10)) / np.log(len(sv) + 1))
    except np.linalg.LinAlgError:
        return 0.0

    return float(np.clip(scope, 0.0, 1.0))


# ============================================================================
# Extended: Metacognitive Depth
# ============================================================================

def measure_metacognitive_depth(pattern_values, prev_values, tier_assignments):
    """Does the prediction tier's state change predict future processing changes?

    Metacognition = the system modeling its own cognitive process.
    Operationalized as: do changes in prediction channels at time t
    correlate with changes in processing channels at time t?

    If prediction changes lead processing changes, the pattern has
    metacognitive depth — its self-model is driving behavior.
    """
    if pattern_values is None or prev_values is None:
        return 0.0

    vals = np.asarray(pattern_values)
    prev = np.asarray(prev_values)

    if vals.shape != prev.shape or vals.shape[0] < 4:
        return 0.0

    pred_ch = np.where(np.asarray(tier_assignments) == TIER_PREDICTION)[0]
    proc_ch = np.where(np.asarray(tier_assignments) == TIER_PROCESSING)[0]

    if len(pred_ch) < 1 or len(proc_ch) < 1:
        return 0.0

    # Change in prediction channels
    delta_pred = np.mean(np.abs(vals[:, pred_ch] - prev[:, pred_ch]), axis=0)  # (n_pred,)
    # Change in processing channels
    delta_proc = np.mean(np.abs(vals[:, proc_ch] - prev[:, proc_ch]), axis=0)  # (n_proc,)

    # Cross-correlation between prediction change and processing change
    dp_c = delta_pred - delta_pred.mean()
    dr_c = delta_proc - delta_proc.mean()

    if dp_c.std() < 1e-8 or dr_c.std() < 1e-8:
        return 0.0

    # Use the magnitude of the cross-correlation matrix
    cross = np.outer(dp_c, dr_c) / (dp_c.std() * dr_c.std() * len(dp_c))
    metacog = float(np.mean(np.abs(cross)))

    return float(np.clip(metacog, 0.0, 1.0))


# ============================================================================
# Combined measurement
# ============================================================================

def measure_all_hier(pattern, prev_mass, prev_values, history_entries,
                     grid_mc, resource, coupling, config, grid_size,
                     step_num=-1, fast=True):
    """Compute all affect dimensions for a hierarchical pattern.

    Returns HierAffectState with 6D core + extended measures.
    """
    tier_assignments = config['tier_assignments']

    # Core: Valence
    valence = measure_valence(pattern.mass, prev_mass)

    # Core: Arousal (tier-weighted)
    arousal = measure_arousal_hier(pattern.values, prev_values, tier_assignments)

    # Core: Integration (spectral Phi)
    phi_spec, eff_rank_raw = spectral_channel_phi(grid_mc, coupling, pattern.cells)

    # Core: Effective Rank
    global_rank, tier_ranks = measure_effective_rank_hier(
        pattern.values, tier_assignments)

    # Core: Counterfactual Weight
    cf_weight = measure_counterfactual_weight(pattern.values, tier_assignments)

    # Core: Self-Model Salience
    sm_salience = measure_self_model_hier(pattern.values, tier_assignments)

    # Extended: World-Model Accuracy
    wm_accuracy = measure_world_model_accuracy(
        grid_mc, resource, pattern.cells, tier_assignments)

    # Extended: Tier Integration (cross-tier Phi)
    tier_int = measure_tier_integration(
        grid_mc, coupling, pattern.cells, tier_assignments)

    # Extended: Self-Model Scope
    sm_scope = measure_self_model_scope(pattern.values, tier_assignments)

    # Extended: Metacognitive Depth
    metacog = measure_metacognitive_depth(
        pattern.values, prev_values, tier_assignments)

    # Per-tier Phi
    if not fast:
        tier_phis = measure_per_tier_phi(
            grid_mc, coupling, pattern.cells, tier_assignments)
    else:
        tier_phis = {'sensory': 0.0, 'processing': 0.0,
                     'memory': 0.0, 'prediction': 0.0}

    return HierAffectState(
        valence=valence,
        arousal=arousal,
        integration=phi_spec,
        effective_rank=global_rank,
        counterfactual_weight=cf_weight,
        self_model_salience=sm_salience,
        world_model_accuracy=wm_accuracy,
        tier_integration=tier_int,
        self_model_scope=sm_scope,
        metacognitive_depth=metacog,
        sensory_phi=tier_phis['sensory'],
        processing_phi=tier_phis['processing'],
        memory_phi=tier_phis['memory'],
        prediction_phi=tier_phis['prediction'],
        pattern_id=pattern.id,
        step=step_num,
        mass=pattern.mass,
        size=pattern.size,
    )


# ============================================================================
# Affect motif detection (Part 2)
# ============================================================================

def classify_affect_motif(affect: HierAffectState) -> str:
    """Classify pattern's affect state into Part 2 motif categories.

    Returns the dominant motif name, or 'neutral' if no strong signature.
    Based on Part 2's structural definitions.
    """
    v = affect.valence
    a = affect.arousal
    phi = affect.integration
    r_eff = affect.effective_rank
    cf = affect.counterfactual_weight
    sm = affect.self_model_salience

    # Suffering: ν(-), Φ↑, r_eff↓ (hyper-integrated, trapped)
    if v < -0.1 and phi > 0.3 and r_eff < 0.2:
        return 'suffering'

    # Fear: ν(-), CF↑, SM↑ (anticipated negative, self-foregrounded)
    if v < -0.05 and cf > 0.5 and sm > 1.5:
        return 'fear'

    # Joy: ν(+), Φ↑, r_eff↑, SM↓ (positive, unified, expansive, self-light)
    if v > 0.05 and phi > 0.2 and r_eff > 0.4 and sm < 1.0:
        return 'joy'

    # Curiosity: ν(+), CF↑ (positive toward uncertainty)
    if v > 0.0 and cf > 0.4 and a > 0.1:
        return 'curiosity'

    # Boredom: α↓, Φ↓, r_eff↓ (understimulated, fragmented)
    if a < 0.02 and phi < 0.1 and r_eff < 0.15:
        return 'boredom'

    # Awe: Φ↑↑, r_eff↑↑, SM↓ (self-dissolution through scale)
    if phi > 0.5 and r_eff > 0.6 and sm < 0.8:
        return 'awe'

    return 'neutral'
