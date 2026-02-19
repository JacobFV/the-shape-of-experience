"""V20: Language Precursor Measurements

Tests the "transmitted compressed imagination" account of language emergence.

Theory: language drops out of compressed imagination + multiagent communication
+ derailment mixing diverse sharp counterfactuals sharp enough to feel symbolic.

In a GRU, the update gate z is the exact architectural analog of detachment:
  z ≈ 1  → hidden state preserved from previous step (memory-dominant / "imagination")
  z ≈ 0  → observation drives the update (reactive)

If the theory is correct:
  (1) Emissions during high-z windows carry MORE information about h than low-z emissions
  (2) High-z windows are more predictive of future states (C_wm higher in memory-mode)
  (3) Mean z increases over evolution (more time in memory-dominant mode)
  (4) Cross-agent: received signal during emitter's high-z window updates receiver h more

If emissions are equally informative regardless of z, language is NOT emerging from
transmitted imagination — it's pure reflex. That would falsify the theory.

Usage:
    python v20_language_precursors.py --snapshot results/v20_s42/snapshots/snapshot_c29.npz
    python v20_language_precursors.py --dir results/v20_s42 --all_snapshots
    python v20_language_precursors.py --all_seeds
"""

import sys
import os
import json
import argparse
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v20_substrate import generate_v20_config, unpack_params, MOVE_DELTAS


# ---------------------------------------------------------------------------
# Sigmoid (numpy)
# ---------------------------------------------------------------------------

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _linear_r2(X, Y):
    """R² of linear regression X → Y (proxy for MI). Returns scalar."""
    X = np.array(X, dtype=np.float64)
    Y = np.array(Y, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    if len(X) < 5:
        return 0.0
    std = X.std(0)
    active = std > 1e-6
    if active.sum() == 0:
        return 0.0
    X = X[:, active]
    X = (X - X.mean(0)) / std[active]
    n_comp = min(8, X.shape[1], X.shape[0] - 1)
    try:
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            X_red = X @ Vt[:n_comp].T
    except np.linalg.LinAlgError:
        return 0.0
    if not np.isfinite(X_red).all():
        return 0.0
    r2s = []
    for col in range(Y.shape[1]):
        y = Y[:, col]
        y = y - y.mean()
        if y.std() < 1e-8:
            continue
        try:
            coef, _, _, _ = np.linalg.lstsq(X_red, y, rcond=None)
            y_pred = X_red @ coef
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum(y ** 2)
            r2s.append(1.0 - ss_res / (ss_tot + 1e-12))
        except Exception:
            continue
    return float(np.mean(r2s)) if r2s else 0.0


# ---------------------------------------------------------------------------
# Per-step GRU with z-gate exposure (numpy, no JIT — for measurement only)
# ---------------------------------------------------------------------------

def agent_step_np(obs, h, params_flat, cfg):
    """Single agent GRU step. Returns new_h, z_mean, raw_actions."""
    p = unpack_params(params_flat, cfg)
    # Unpack to numpy
    embed_W = np.array(p['embed_W'])
    embed_b = np.array(p['embed_b'])
    Wz = np.array(p['gru_Wz'])
    bz = np.array(p['gru_bz'])
    Wr = np.array(p['gru_Wr'])
    br = np.array(p['gru_br'])
    Wh = np.array(p['gru_Wh'])
    bh = np.array(p['gru_bh'])
    out_W = np.array(p['out_W'])
    out_b = np.array(p['out_b'])

    x = np.tanh(obs @ embed_W + embed_b)
    xh = np.concatenate([x, h])
    z = _sigmoid(xh @ Wz + bz)
    r = _sigmoid(xh @ Wr + br)
    h_tilde = np.tanh(np.concatenate([x, r * h]) @ Wh + bh)
    new_h = z * h + (1.0 - z) * h_tilde
    actions = new_h @ out_W + out_b
    return new_h, float(np.mean(z)), actions


def batch_step_np(obs_batch, h_batch, params_batch, alive, cfg):
    """Vectorized numpy step over all agents. Returns new_h, z_vec, actions."""
    M = h_batch.shape[0]
    new_h = h_batch.copy()
    z_vec = np.zeros(M)
    actions = np.zeros((M, cfg['n_actions']))
    for i in range(M):
        if alive[i]:
            new_h[i], z_vec[i], actions[i] = agent_step_np(
                obs_batch[i], h_batch[i], params_batch[i], cfg
            )
    return new_h, z_vec, actions


# ---------------------------------------------------------------------------
# Build observations (numpy, matches v20_substrate.py logic)
# ---------------------------------------------------------------------------

def gather_patch_np(field, pos, radius, N):
    """Extract (2r+1)×(2r+1) patch with circular wrap."""
    offsets = np.arange(-radius, radius + 1)
    rows = (pos[0] + offsets) % N
    cols = (pos[1] + offsets) % N
    return field[np.ix_(rows, cols)]


def build_obs_batch_np(positions, resources, signals, agent_count, energy, cfg):
    """Build observation batch (M, obs_flat)."""
    M = positions.shape[0]
    r = cfg['obs_radius']
    N = cfg['N']
    obs_flat = cfg['obs_flat']
    obs = np.zeros((M, obs_flat))
    for i in range(M):
        R = gather_patch_np(resources, positions[i], r, N).flatten()
        S = gather_patch_np(signals, positions[i], r, N).flatten()
        A = gather_patch_np(agent_count, positions[i], r, N).flatten()
        obs[i] = np.concatenate([R, S, A, [energy[i]]])
    return obs


def build_count_grid_np(positions, alive, N):
    grid = np.zeros((N, N))
    for i in range(len(positions)):
        if alive[i]:
            grid[positions[i, 0], positions[i, 1]] += 1
    return grid


MOVE_DELTAS_NP = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])


# ---------------------------------------------------------------------------
# Measurement rollout
# ---------------------------------------------------------------------------

def measurement_rollout(snapshot, cfg, n_steps=600, seed=0):
    """Run agents step-by-step recording per-step data.

    Returns list of records, one per step.
    Each record: {t, h_before, obs, z, emit_logit, emit_prob, alive, positions, signals}
    """
    rng = np.random.RandomState(seed)
    N = cfg['N']
    if 'resources' in snapshot:
        resources = np.array(snapshot['resources'], dtype=np.float32)
    else:
        resources = np.full((N, N), 0.5, dtype=np.float32)
    if 'signals' in snapshot:
        signals = np.array(snapshot['signals'], dtype=np.float32)
    else:
        signals = np.zeros((N, N), dtype=np.float32)
    positions = np.array(snapshot['positions'], dtype=np.int32)
    hidden = np.array(snapshot['hidden'], dtype=np.float32)
    energy = np.array(snapshot['energy'], dtype=np.float32)
    alive = np.array(snapshot['alive'], dtype=bool)
    params = np.array(snapshot['params'], dtype=np.float32)

    # Re-initialize hidden and energy for clean measurement
    hidden = np.zeros_like(hidden)
    energy = np.where(alive, cfg['initial_energy'], 0.0).astype(np.float32)

    records = []

    for t in range(n_steps):
        count_grid = build_count_grid_np(positions, alive, N)
        obs = build_obs_batch_np(positions, resources, signals, count_grid, energy, cfg)

        h_before = hidden.copy()
        new_h, z_vec, actions_batch = batch_step_np(obs, hidden, params, alive, cfg)

        # Decode actions
        move_idx = np.argmax(actions_batch[:, :5], axis=-1)
        consume_prob = _sigmoid(actions_batch[:, 5])
        emit_logit = actions_batch[:, 6]
        emit_prob = _sigmoid(emit_logit)

        records.append({
            't': t,
            'h': h_before,
            'obs': obs.copy(),
            'z': z_vec.copy(),
            'emit_logit': emit_logit.copy(),
            'emit_prob': emit_prob.copy(),
            'alive': alive.copy(),
            'positions': positions.copy(),
            'signals': signals.copy(),
        })

        # Apply movements
        new_positions = positions.copy()
        for i in range(cfg['M_max']):
            if alive[i]:
                delta = MOVE_DELTAS_NP[move_idx[i]]
                new_positions[i] = (positions[i] + delta) % N

        # Apply consumption
        local_r = resources[positions[:, 0], positions[:, 1]]
        actual_consume = consume_prob * local_r * alive
        for i in range(cfg['M_max']):
            if alive[i]:
                resources[positions[i, 0], positions[i, 1]] -= actual_consume[i]
        resources = np.clip(resources, 0.0, 1.0)

        # Apply emissions
        for i in range(cfg['M_max']):
            if alive[i]:
                signals[positions[i, 0], positions[i, 1]] += emit_prob[i] * 0.1
        signals = np.clip(signals, 0.0, 1.0)

        # Regen + diffuse (simplified)
        resources = resources + cfg['resource_regen'] * (1.0 - resources)
        signals = signals * (1.0 - cfg['signal_decay'])

        # Energy update
        energy += actual_consume * cfg['resource_value']
        energy -= cfg['metabolic_cost']
        alive = (energy > 0.0) & alive

        positions = new_positions
        hidden = new_h

    return records


# ---------------------------------------------------------------------------
# Measurement 1: z distribution and evolution trend
# ---------------------------------------------------------------------------

def measure_z_distribution(records, cfg):
    """Mean z per agent, fraction of high-z steps."""
    M = cfg['M_max']
    z_sum = np.zeros(M)
    z_count = np.zeros(M)

    for rec in records:
        mask = rec['alive']
        for i in range(M):
            if mask[i]:
                z_sum[i] += rec['z'][i]
                z_count[i] += 1

    mean_z = np.where(z_count > 0, z_sum / np.maximum(z_count, 1), np.nan)
    alive_mean_z = mean_z[~np.isnan(mean_z)]

    # High-z fraction
    all_z = np.array([rec['z'][i]
                      for rec in records
                      for i in range(M)
                      if rec['alive'][i]])

    return {
        'mean_z': float(np.nanmean(mean_z)),
        'std_z': float(np.nanstd(mean_z)),
        'frac_high_z': float(np.mean(all_z > 0.7)) if len(all_z) else 0.0,
        'frac_low_z': float(np.mean(all_z < 0.3)) if len(all_z) else 0.0,
        'n_alive_agents': int(z_count.sum()),
    }


# ---------------------------------------------------------------------------
# Measurement 2: Emission informativeness by z-window
# ---------------------------------------------------------------------------

def measure_emission_informativeness(records, cfg, z_high=0.7, z_low=0.3):
    """MI(h; emit_logit) split by z-window.

    Key test: if high-z > low-z, emissions carry more info about hidden state
    during memory-dominant windows → "transmitted imagination" signature.
    """
    M = cfg['M_max']
    high_h, high_e = [], []
    low_h, low_e = [], []

    for rec in records:
        for i in range(M):
            if not rec['alive'][i]:
                continue
            z = rec['z'][i]
            h = rec['h'][i]
            e = rec['emit_logit'][i]
            if z > z_high:
                high_h.append(h)
                high_e.append(e)
            elif z < z_low:
                low_h.append(h)
                low_e.append(e)

    r2_high = _linear_r2(np.array(high_h), np.array(high_e)) if len(high_h) >= 10 else 0.0
    r2_low = _linear_r2(np.array(low_h), np.array(low_e)) if len(low_h) >= 10 else 0.0

    ratio = r2_high / max(r2_low, 1e-6)

    return {
        'r2_emit_high_z': r2_high,
        'r2_emit_low_z': r2_low,
        'ratio_high_vs_low': ratio,
        'n_high_z_steps': len(high_h),
        'n_low_z_steps': len(low_h),
        'signature_present': bool(r2_high > r2_low and ratio > 1.5),
    }


# ---------------------------------------------------------------------------
# Measurement 3: Future-state prediction quality by z-window
# ---------------------------------------------------------------------------

def measure_future_prediction(records, cfg, k=5, z_high=0.7, z_low=0.3):
    """R²(h_t → obs_{t+k}) split by z at time t.

    If high-z → better future prediction, memory-dominant mode is more
    predictive of what comes next (consistent with "running useful rollouts").
    """
    M = cfg['M_max']
    n = len(records)
    high_h, high_obs_future = [], []
    low_h, low_obs_future = [], []

    for t in range(n - k):
        rec_t = records[t]
        rec_future = records[t + k]
        for i in range(M):
            if not rec_t['alive'][i] or not rec_future['alive'][i]:
                continue
            z = rec_t['z'][i]
            h = rec_t['h'][i]
            obs_future = rec_future['obs'][i]
            if z > z_high:
                high_h.append(h)
                high_obs_future.append(obs_future)
            elif z < z_low:
                low_h.append(h)
                low_obs_future.append(obs_future)

    r2_high = _linear_r2(np.array(high_h), np.array(high_obs_future)) if len(high_h) >= 10 else 0.0
    r2_low = _linear_r2(np.array(low_h), np.array(low_obs_future)) if len(low_h) >= 10 else 0.0

    return {
        'r2_future_high_z': r2_high,
        'r2_future_low_z': r2_low,
        'ratio_high_vs_low': r2_high / max(r2_low, 1e-6),
        'n_high_z': len(high_h),
        'n_low_z': len(low_h),
        'signature_present': bool(r2_high > r2_low),
    }


# ---------------------------------------------------------------------------
# Measurement 4: Cross-agent signal utility
# ---------------------------------------------------------------------------

def measure_cross_agent_utility(records, cfg, z_high=0.7, radius=3):
    """When agent i emits during high-z, does agent j update h more?

    Strategy: compare h_update magnitude for agent j when nearby agent i
    emits during high-z vs. no nearby emission, controlling for j's own z.
    """
    M = cfg['M_max']
    N = cfg['N']
    n = len(records)

    # Build emission events: (t, pos, emit_prob, emitter_z, emitter_h)
    high_z_emit_steps = []  # steps where any agent emits during high-z
    baseline_steps = []     # steps where no nearby high-z emission

    for t in range(n - 1):
        rec = records[t]
        rec_next = records[t + 1]

        for j in range(M):
            if not rec['alive'][j] or not rec_next['alive'][j]:
                continue

            # h update magnitude for agent j
            h_update = float(np.linalg.norm(rec_next['h'][j] - rec['h'][j]))

            # Check if any nearby agent (within radius) emitted during high-z
            pos_j = rec['positions'][j]
            nearby_high_z_emit = False
            for i in range(M):
                if i == j or not rec['alive'][i]:
                    continue
                pos_i = rec['positions'][i]
                dist = np.max(np.abs(pos_j - pos_i))  # Chebyshev
                if dist <= radius and rec['z'][i] > z_high and rec['emit_prob'][i] > 0.5:
                    nearby_high_z_emit = True
                    break

            if nearby_high_z_emit:
                high_z_emit_steps.append((h_update, rec['z'][j]))
            else:
                baseline_steps.append((h_update, rec['z'][j]))

    if len(high_z_emit_steps) < 5 or len(baseline_steps) < 5:
        return {'insufficient_data': True, 'n_high_z_emit': len(high_z_emit_steps)}

    h_updates_high = np.array([x[0] for x in high_z_emit_steps])
    h_updates_base = np.array([x[0] for x in baseline_steps])

    t_stat, p_val = stats.ttest_ind(h_updates_high, h_updates_base)

    return {
        'mean_h_update_near_high_z_emit': float(np.mean(h_updates_high)),
        'mean_h_update_baseline': float(np.mean(h_updates_base)),
        'effect': float(np.mean(h_updates_high) - np.mean(h_updates_base)),
        't_stat': float(t_stat),
        'p_value': float(p_val),
        'n_high_z_emit': len(high_z_emit_steps),
        'n_baseline': len(baseline_steps),
        'signature_present': bool(p_val < 0.05 and np.mean(h_updates_high) > np.mean(h_updates_base)),
    }


# ---------------------------------------------------------------------------
# Full snapshot analysis
# ---------------------------------------------------------------------------

def analyze_snapshot(snap_path, cfg, verbose=True):
    """Run all language precursor measurements on one snapshot."""
    snap = dict(np.load(snap_path, allow_pickle=True))
    cycle = int(snap.get('cycle', -1))

    if verbose:
        print(f"  Running measurement rollout (600 steps)...")
    records = measurement_rollout(snap, cfg, n_steps=600)

    n_alive = int(np.sum(records[0]['alive']))
    if verbose:
        print(f"  n_alive={n_alive}, running measurements...")

    z_dist = measure_z_distribution(records, cfg)
    emit_info = measure_emission_informativeness(records, cfg)
    future_pred = measure_future_prediction(records, cfg)
    cross_agent = measure_cross_agent_utility(records, cfg)

    result = {
        'cycle': cycle,
        'snap_path': snap_path,
        'n_alive': n_alive,
        'z_distribution': z_dist,
        'emission_informativeness': emit_info,
        'future_prediction': future_pred,
        'cross_agent_utility': cross_agent,
    }

    if verbose:
        print(f"  C{cycle:02d} | mean_z={z_dist['mean_z']:.3f} "
              f"| frac_high_z={z_dist['frac_high_z']:.2f} "
              f"| emit_r2_high={emit_info['r2_emit_high_z']:.4f} "
              f"| emit_r2_low={emit_info['r2_emit_low_z']:.4f} "
              f"| ratio={emit_info['ratio_high_vs_low']:.2f}x "
              f"| sig={'YES' if emit_info['signature_present'] else 'no'}")

    return result


# ---------------------------------------------------------------------------
# Cross-cycle evolution analysis
# ---------------------------------------------------------------------------

def analyze_evolution_trend(results):
    """Test whether key metrics trend over cycles."""
    cycles = [r['cycle'] for r in results]
    mean_z = [r['z_distribution']['mean_z'] for r in results]
    emit_ratio = [r['emission_informativeness']['ratio_high_vs_low'] for r in results]
    future_ratio = [r['future_prediction']['ratio_high_vs_low'] for r in results]

    def trend(xs, ys):
        if len(ys) < 3:
            return None, None
        slope, _, r, p, _ = stats.linregress(xs, ys)
        return float(slope), float(p)

    z_slope, z_p = trend(cycles, mean_z)
    emit_slope, emit_p = trend(cycles, emit_ratio)
    future_slope, future_p = trend(cycles, future_ratio)

    return {
        'mean_z_trend': {'slope': z_slope, 'p': z_p,
                         'increasing': bool(z_slope > 0 and z_p < 0.1) if z_slope else False},
        'emit_ratio_trend': {'slope': emit_slope, 'p': emit_p,
                             'increasing': bool(emit_slope > 0 and emit_p < 0.1) if emit_slope else False},
        'future_ratio_trend': {'slope': future_slope, 'p': future_p,
                               'increasing': bool(future_slope > 0 and future_p < 0.1) if future_slope else False},
        'n_snapshots': len(results),
        'cycle_range': [min(cycles), max(cycles)] if cycles else [],
    }


# ---------------------------------------------------------------------------
# Summarize across seeds
# ---------------------------------------------------------------------------

def summarize_signatures(all_seed_results):
    """Aggregate signature detection across seeds and snapshots."""
    n_total = 0
    n_emit_sig = 0
    n_future_sig = 0
    n_cross_sig = 0
    mean_z_vals = []
    emit_ratio_vals = []

    for seed_results in all_seed_results.values():
        for r in seed_results['snapshots']:
            n_total += 1
            if r['emission_informativeness']['signature_present']:
                n_emit_sig += 1
            if r['future_prediction']['signature_present']:
                n_future_sig += 1
            cross = r['cross_agent_utility']
            if not cross.get('insufficient_data') and cross.get('signature_present'):
                n_cross_sig += 1
            mean_z_vals.append(r['z_distribution']['mean_z'])
            emit_ratio_vals.append(r['emission_informativeness']['ratio_high_vs_low'])

    return {
        'n_snapshots_total': n_total,
        'emit_sig_fraction': n_emit_sig / max(n_total, 1),
        'future_sig_fraction': n_future_sig / max(n_total, 1),
        'cross_sig_fraction': n_cross_sig / max(n_total, 1),
        'mean_z_overall': float(np.mean(mean_z_vals)) if mean_z_vals else 0.0,
        'mean_emit_ratio': float(np.mean(emit_ratio_vals)) if emit_ratio_vals else 0.0,
        'theory_supported': bool(
            n_emit_sig / max(n_total, 1) > 0.3
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all_seeds(base_dir, seeds=(42, 123, 7), variant='v20'):
    """Run language precursor analysis on all snapshots for all seeds."""
    cfg = generate_v20_config(N=128, M_max=256, steps_per_cycle=5000, chunk_size=50)
    all_results = {}

    for seed in seeds:
        seed_dir = os.path.join(base_dir, f'{variant}_s{seed}')
        # V20b saves snapshots directly in seed_dir; V18 uses a snapshots/ subdir
        snap_dir = seed_dir
        subdir = os.path.join(seed_dir, 'snapshots')
        if os.path.isdir(subdir):
            snap_dir = subdir

        if not os.path.isdir(snap_dir):
            print(f"[seed {seed}] No dir at {snap_dir}, skipping")
            continue

        snaps = sorted([
            os.path.join(snap_dir, f)
            for f in os.listdir(snap_dir)
            if f.endswith('.npz')
        ])

        if not snaps:
            print(f"[seed {seed}] No snapshots found")
            continue

        print(f"\n{'='*60}")
        print(f"Seed {seed}: {len(snaps)} snapshots")
        print(f"{'='*60}")

        snap_results = []
        for sp in snaps:
            result = analyze_snapshot(sp, cfg)
            snap_results.append(result)

        trend = analyze_evolution_trend(snap_results)

        seed_summary = {
            'seed': seed,
            'snapshots': snap_results,
            'evolution_trend': trend,
        }
        all_results[seed] = seed_summary

        print(f"\n  Seed {seed} evolution trends:")
        print(f"    mean_z trend: slope={trend['mean_z_trend']['slope']:.4f}, "
              f"p={trend['mean_z_trend']['p']:.3f}, "
              f"increasing={trend['mean_z_trend']['increasing']}")
        print(f"    emit_ratio trend: slope={trend['emit_ratio_trend']['slope']:.4f}, "
              f"p={trend['emit_ratio_trend']['p']:.3f}")

        out_file = os.path.join(seed_dir, f'language_precursors_{variant}_s{seed}.json')
        with open(out_file, 'w') as f:
            json.dump(seed_summary, f, indent=2, default=str)
        print(f"  Saved to {out_file}")

    # Aggregate summary
    summary = summarize_signatures(all_results)
    print(f"\n{'='*60}")
    print(f"LANGUAGE PRECURSOR SUMMARY ({variant})")
    print(f"{'='*60}")
    print(f"  Snapshots analyzed: {summary['n_snapshots_total']}")
    print(f"  Emission signature (high-z more informative): "
          f"{summary['emit_sig_fraction']:.0%} of snapshots")
    print(f"  Future prediction signature: "
          f"{summary['future_sig_fraction']:.0%} of snapshots")
    print(f"  Cross-agent utility signature: "
          f"{summary['cross_sig_fraction']:.0%} of snapshots")
    print(f"  Mean z (memory-dominant tendency): {summary['mean_z_overall']:.3f}")
    print(f"  Mean emit ratio (high/low z): {summary['mean_emit_ratio']:.2f}x")
    print(f"  Theory supported: {'YES' if summary['theory_supported'] else 'NO'}")

    summary_path = os.path.join(base_dir, f'language_precursors_summary_{variant}.json')
    with open(summary_path, 'w') as f:
        json.dump({**summary, 'all_results': all_results}, f, indent=2, default=str)
    print(f"\n  Summary saved to {summary_path}")

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='results', help='Base results directory')
    parser.add_argument('--variant', default='v20', help='v20 or v20b')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 7])
    parser.add_argument('--snapshot', type=str, help='Analyze single snapshot')
    args = parser.parse_args()

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.dir)

    if args.snapshot:
        cfg = generate_v20_config(N=128, M_max=256, steps_per_cycle=5000, chunk_size=50)
        result = analyze_snapshot(args.snapshot, cfg)
        print(json.dumps(result, indent=2, default=str))
    else:
        run_all_seeds(base, seeds=tuple(args.seeds), variant=args.variant)
