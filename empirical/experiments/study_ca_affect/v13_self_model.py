"""Experiment 6: Self-Model Emergence — Measurement Module.

Measures whether patterns develop models of *themselves* — privileged
self-knowledge beyond what an external observer can infer.

Key quantities:

    ρ_self(t):  Self-effect ratio. How much does the pattern's own action
                improve prediction of its next observation, given the
                environment? When ρ_self > 0.5, the pattern's own actions
                dominate its observation stream.

    SM(τ):      Self-prediction score. The gap between self-prediction
                (s_B → ŝ_B(t+τ)) and external prediction (s_∂B → ŝ_B(t+τ)).
                Positive = pattern predicts itself better than external
                observer can. This is privileged self-knowledge.

    SM_sal(t):  Self-model salience. Ratio of self-predictive to
                environment-predictive information in pattern state.
                SM_sal > 1 means pattern knows more about its own future
                than about the environment's.
"""

import warnings
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


@dataclass
class SelfModelResult:
    """Self-model metrics for a single pattern."""
    pattern_id: int
    n_timesteps: int
    rho_self: float           # self-effect ratio (mean)
    SM: Dict[int, float]      # tau -> self-prediction score
    SM_capacity: float        # ∫SM(τ)dτ — total self-knowledge
    SM_sal: float             # self-model salience
    f_self_mse: Dict[int, float]  # tau -> MSE of self-prediction
    f_ext_mse: Dict[int, float]   # tau -> MSE of external prediction


def compute_self_effect_ratio(features_list: List[dict],
                               n_folds: int = 5,
                               alpha: float = 1.0) -> float:
    """Compute ρ_self: how much does internal state help predict next boundary?

    ρ_self = (MSE[f_env] - MSE[f_full]) / MSE[f_env]

    where:
        f_env:  s_env(t) → o_B(t+1)  (predict next boundary from environment alone)
        f_full: (a_B(t), s_env(t)) → o_B(t+1)  (add pattern's action)
        a_B(t) = s_B(t) - s_B(t-1)   (change initiated by pattern)

    Returns ρ_self ∈ [0, 1] (clamped). Higher = pattern's actions matter more.
    """
    if len(features_list) < max(12, n_folds + 2):
        return 0.0

    X_env_list = []
    X_full_list = []
    Y_list = []

    for i in range(1, len(features_list) - 1):
        s_B_prev = features_list[i - 1]['s_B']
        s_B_curr = features_list[i]['s_B']
        s_dB_curr = features_list[i]['s_dB']
        s_dB_next = features_list[i + 1]['s_dB']

        action = s_B_curr - s_B_prev  # pattern's "action"

        # Environment context: current boundary
        s_env = s_dB_curr

        if (np.any(np.isnan(action)) or np.any(np.isnan(s_env))
                or np.any(np.isnan(s_dB_next))):
            continue

        X_env_list.append(s_env)
        X_full_list.append(np.concatenate([action, s_env]))
        Y_list.append(s_dB_next)

    n = len(Y_list)
    if n < max(10, n_folds + 1):
        return 0.0

    X_env = np.array(X_env_list, dtype=np.float64)
    X_full = np.array(X_full_list, dtype=np.float64)
    Y = np.array(Y_list, dtype=np.float64)

    actual_folds = min(n_folds, n)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        pipe_env = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        scores_env = cross_val_score(
            pipe_env, X_env, Y, cv=actual_folds,
            scoring='neg_mean_squared_error')
        mse_env = -scores_env.mean()

        pipe_full = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        scores_full = cross_val_score(
            pipe_full, X_full, Y, cv=actual_folds,
            scoring='neg_mean_squared_error')
        mse_full = -scores_full.mean()

    if mse_env < 1e-12:
        return 0.0

    rho = (mse_env - mse_full) / mse_env
    return float(np.clip(rho, 0.0, 1.0))


def compute_self_prediction(features_list: List[dict],
                             tau_values: tuple = (1, 2, 5, 10),
                             n_folds: int = 5,
                             alpha: float = 1.0
                             ) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    """Compute SM(τ): self-prediction advantage at multiple horizons.

    SM(τ) = MSE[f_ext] - MSE[f_self]

    f_self: s_B(t) → s_B(t+τ)   (pattern predicts itself)
    f_ext:  s_∂B(t) → s_B(t+τ)  (external observer predicts pattern)

    Returns (SM, f_self_mse, f_ext_mse) dicts.
    """
    SM = {}
    f_self_mse = {}
    f_ext_mse = {}

    for tau in tau_values:
        X_self_list = []
        X_ext_list = []
        Y_list = []

        for i in range(len(features_list) - tau):
            s_B = features_list[i]['s_B']
            s_dB = features_list[i]['s_dB']
            s_B_future = features_list[i + tau]['s_B']

            if (np.any(np.isnan(s_B)) or np.any(np.isnan(s_dB))
                    or np.any(np.isnan(s_B_future))):
                continue

            X_self_list.append(s_B)
            X_ext_list.append(s_dB)
            Y_list.append(s_B_future)

        n = len(Y_list)
        if n < max(10, n_folds + 1):
            SM[tau] = 0.0
            f_self_mse[tau] = float('nan')
            f_ext_mse[tau] = float('nan')
            continue

        X_self = np.array(X_self_list, dtype=np.float64)
        X_ext = np.array(X_ext_list, dtype=np.float64)
        Y = np.array(Y_list, dtype=np.float64)

        actual_folds = min(n_folds, n)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)

            pipe_self = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            scores_self = cross_val_score(
                pipe_self, X_self, Y, cv=actual_folds,
                scoring='neg_mean_squared_error')
            mse_self = -scores_self.mean()

            pipe_ext = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            scores_ext = cross_val_score(
                pipe_ext, X_ext, Y, cv=actual_folds,
                scoring='neg_mean_squared_error')
            mse_ext = -scores_ext.mean()

        SM[tau] = float(mse_ext - mse_self)  # positive = self-knowledge advantage
        f_self_mse[tau] = float(mse_self)
        f_ext_mse[tau] = float(mse_ext)

    return SM, f_self_mse, f_ext_mse


def compute_self_model_salience(features_list: List[dict],
                                 n_folds: int = 5,
                                 alpha: float = 1.0) -> float:
    """Compute SM_sal: ratio of self-predictive to environment-predictive info.

    SM_sal = I(s_B; s_B(t+1) | s_∂B) / I(s_B; s_∂B(t+1) | s_∂B)

    Operationally:
        numerator:   (MSE[s_∂B → s_B(t+1)] - MSE[(s_B, s_∂B) → s_B(t+1)])
        denominator: (MSE[s_∂B → s_∂B(t+1)] - MSE[(s_B, s_∂B) → s_∂B(t+1)])

    SM_sal > 1 means pattern knows more about its own future than environment's.
    """
    if len(features_list) < max(12, n_folds + 2):
        return 0.0

    X_boundary_list = []
    X_full_list = []
    Y_self_list = []     # s_B(t+1)
    Y_env_list = []      # s_∂B(t+1)

    for i in range(len(features_list) - 1):
        s_B = features_list[i]['s_B']
        s_dB = features_list[i]['s_dB']
        s_B_next = features_list[i + 1]['s_B']
        s_dB_next = features_list[i + 1]['s_dB']

        if (np.any(np.isnan(s_B)) or np.any(np.isnan(s_dB))
                or np.any(np.isnan(s_B_next)) or np.any(np.isnan(s_dB_next))):
            continue

        X_boundary_list.append(s_dB)
        X_full_list.append(np.concatenate([s_B, s_dB]))
        Y_self_list.append(s_B_next)
        Y_env_list.append(s_dB_next)

    n = len(Y_self_list)
    if n < max(10, n_folds + 1):
        return 0.0

    X_bnd = np.array(X_boundary_list, dtype=np.float64)
    X_full = np.array(X_full_list, dtype=np.float64)
    Y_self = np.array(Y_self_list, dtype=np.float64)
    Y_env = np.array(Y_env_list, dtype=np.float64)

    actual_folds = min(n_folds, n)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        # Numerator: I(s_B; s_B(t+1) | s_∂B)
        pipe_bnd_self = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        s1 = cross_val_score(pipe_bnd_self, X_bnd, Y_self, cv=actual_folds,
                             scoring='neg_mean_squared_error')
        mse_bnd_self = -s1.mean()

        pipe_full_self = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        s2 = cross_val_score(pipe_full_self, X_full, Y_self, cv=actual_folds,
                             scoring='neg_mean_squared_error')
        mse_full_self = -s2.mean()

        # Denominator: I(s_B; s_∂B(t+1) | s_∂B)
        pipe_bnd_env = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        s3 = cross_val_score(pipe_bnd_env, X_bnd, Y_env, cv=actual_folds,
                             scoring='neg_mean_squared_error')
        mse_bnd_env = -s3.mean()

        pipe_full_env = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        s4 = cross_val_score(pipe_full_env, X_full, Y_env, cv=actual_folds,
                             scoring='neg_mean_squared_error')
        mse_full_env = -s4.mean()

    numerator = max(0.0, mse_bnd_self - mse_full_self)
    denominator = max(0.0, mse_bnd_env - mse_full_env)

    if denominator < 1e-12:
        # If adding s_B doesn't help predict environment either,
        # SM_sal is undefined. Return 0 to indicate "no self-model".
        return 0.0 if numerator < 1e-12 else float('inf')

    return float(numerator / denominator)


def analyze_self_model(features_list: List[dict],
                        tau_values: tuple = (1, 2, 5, 10),
                        ) -> Optional[SelfModelResult]:
    """Full self-model analysis for a single pattern's trajectory."""
    if len(features_list) < 15:
        return None

    rho_self = compute_self_effect_ratio(features_list)
    SM, f_self_mse, f_ext_mse = compute_self_prediction(
        features_list, tau_values=tau_values)
    SM_sal = compute_self_model_salience(features_list)

    # Capacity: trapezoidal integral of SM(τ) where SM > 0
    sorted_taus = sorted(tau_values)
    sm_vals = [max(0.0, SM.get(t, 0.0)) for t in sorted_taus]
    SM_capacity = float(np.trapz(sm_vals, sorted_taus)) if len(sorted_taus) > 1 else 0.0

    return SelfModelResult(
        pattern_id=-1,
        n_timesteps=len(features_list),
        rho_self=rho_self,
        SM=SM,
        SM_capacity=SM_capacity,
        SM_sal=SM_sal,
        f_self_mse=f_self_mse,
        f_ext_mse=f_ext_mse,
    )


# ============================================================================
# Full pipeline
# ============================================================================

def measure_self_model_from_snapshot(
    snapshot_path: str,
    seed: int,
    config_overrides: Optional[dict] = None,
    n_recording_steps: int = 50,
    substrate_steps_per_record: int = 10,
    threshold: float = 0.15,
    max_patterns: int = 20,
    tau_values: tuple = (1, 2, 5, 10),
) -> Tuple[List[SelfModelResult], dict]:
    """Load snapshot, run recording, compute self-model metrics."""
    import jax.numpy as jnp
    from jax import random
    from v13_substrate import generate_v13_config, init_v13, run_v13_chunk
    from v13_world_model import (
        detect_patterns_for_wm, extract_internal_state,
        extract_boundary_obs,
    )

    snap = np.load(snapshot_path)
    grid_np = snap['grid']
    resource_np = snap['resource']
    C, N = grid_np.shape[0], grid_np.shape[1]

    config = generate_v13_config(C=C, N=N, seed=seed)
    config['maintenance_rate'] = 0.003
    config['resource_consume'] = 0.003
    config['resource_regen'] = 0.01
    if config_overrides:
        config.update(config_overrides)

    _, _, h_embed, kernel_ffts, coupling, coupling_row_sums, box_fft = \
        init_v13(config, seed=seed)
    rng = random.PRNGKey(seed + 9999)

    # Run forward and collect grids
    grids = [grid_np.copy()]
    g = jnp.array(grid_np)
    r = jnp.array(resource_np)

    for _ in range(n_recording_steps):
        g, r, rng = run_v13_chunk(
            g, r, h_embed, kernel_ffts, config,
            coupling, coupling_row_sums, rng,
            n_steps=substrate_steps_per_record, box_fft=box_fft)
        grids.append(np.array(g))

    # Track patterns
    initial_pats = detect_patterns_for_wm(grids[0], threshold=threshold)
    if not initial_pats:
        return [], {'n_patterns_detected': 0}

    all_features = {}
    for pid, p in enumerate(initial_pats[:max_patterns]):
        all_features[pid] = {
            'center_history': [p['center'].copy()],
            'features': [],
        }

    for t in range(n_recording_steps):
        grid_t = grids[t]
        pats_t = detect_patterns_for_wm(grid_t, threshold=threshold)
        if not pats_t:
            continue

        tracked_ids = list(all_features.keys())
        matched_tracked = set()
        matched_detected = set()
        costs = []
        for j, p in enumerate(pats_t):
            for pid in tracked_ids:
                last_center = all_features[pid]['center_history'][-1]
                dr = abs(p['center'][0] - last_center[0])
                dc = abs(p['center'][1] - last_center[1])
                dr = min(dr, N - dr)
                dc = min(dc, N - dc)
                dist = np.sqrt(dr ** 2 + dc ** 2)
                if dist < 40.0:
                    costs.append((dist, pid, j))

        costs.sort()
        for dist, pid, j in costs:
            if pid in matched_tracked or j in matched_detected:
                continue
            matched_tracked.add(pid)
            matched_detected.add(j)

            p = pats_t[j]
            all_features[pid]['center_history'].append(p['center'].copy())

            s_B = extract_internal_state(grid_t, p['cells'])
            s_dB = extract_boundary_obs(grid_t, p['cells'], N)

            all_features[pid]['features'].append({
                's_B': s_B,
                's_dB': s_dB,
            })

    # Analyze
    results = []
    sorted_pids = sorted(all_features.keys(),
                         key=lambda pid: len(all_features[pid]['features']),
                         reverse=True)

    for pid in sorted_pids[:max_patterns]:
        feat_list = all_features[pid]['features']
        sm = analyze_self_model(feat_list, tau_values=tau_values)
        if sm is not None:
            sm.pattern_id = pid
            results.append(sm)

    metadata = {
        'snapshot_path': snapshot_path,
        'seed': seed,
        'C': C, 'N': N,
        'n_recording_steps': n_recording_steps,
        'tau_values': list(tau_values),
        'n_patterns_detected': len(initial_pats),
        'n_patterns_analyzed': len(results),
    }

    return results, metadata


def results_to_dict(results: List[SelfModelResult], metadata: dict,
                    cycle: int = -1) -> dict:
    return {
        'metadata': metadata,
        'patterns': [
            {
                'pattern_id': r.pattern_id,
                'n_timesteps': r.n_timesteps,
                'rho_self': r.rho_self,
                'SM': {str(k): v for k, v in r.SM.items()},
                'SM_capacity': r.SM_capacity,
                'SM_sal': r.SM_sal,
                'f_self_mse': {str(k): v for k, v in r.f_self_mse.items()},
                'f_ext_mse': {str(k): v for k, v in r.f_ext_mse.items()},
            }
            for r in results
        ],
        'summary': {
            'mean_rho_self': float(np.mean([r.rho_self for r in results])) if results else 0,
            'mean_SM_capacity': float(np.mean([r.SM_capacity for r in results])) if results else 0,
            'mean_SM_sal': float(np.mean([r.SM_sal for r in results
                                           if np.isfinite(r.SM_sal)])) if results else 0,
            'n_with_self_model': sum(1 for r in results if r.SM_capacity > 0),
            'n_with_high_rho': sum(1 for r in results if r.rho_self > 0.1),
            'n_with_salience': sum(1 for r in results
                                   if np.isfinite(r.SM_sal) and r.SM_sal > 1.0),
            'mean_SM_by_tau': {
                str(tau): float(np.mean([r.SM.get(tau, 0) for r in results]))
                for tau in (results[0].SM.keys() if results else [])
            },
        },
        'cycle': cycle,
    }
