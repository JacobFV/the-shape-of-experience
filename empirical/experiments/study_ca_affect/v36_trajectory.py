"""V36: Egocentric Affect Trajectory Recorder

Records detailed per-timestep data from a focal agent's perspective
during key ecological transitions (normal → pre-drought → drought →
recovery → late-stage). Produces data for:

1. 3D egocentric rendering (agent's-eye view of the world)
2. 3D affect trajectory (hidden state PCA'd to 3D)
3. VLM narration pipeline (behavioral descriptions per segment)
4. Framework affect predictions per timestep

Architecture: V27 (GRU + MLP prediction head, the highest-Φ configuration).
Uses a pre-evolved population from V32 results (seed 23, max Φ=0.473).
Re-runs the final cycles with detailed recording.

The output is a JSON file containing:
- Per-timestep focal agent data (position, observations, hidden state,
  energy, actions, predictions, affect coordinates)
- Per-timestep environment state (resource grid downsampled, all agent
  positions/energies)
- Segment boundaries with ecological condition labels
- Framework-predicted affect coordinates per segment
"""

import jax
import jax.numpy as jnp
import numpy as np
import json
import os
import time
from functools import partial

from v27_substrate import (
    generate_v27_config, init_v27, make_v27_chunk_runner,
    agent_multi_tick_v27_batch,
    build_observation_batch,
    build_agent_count_grid, apply_consumption, apply_emissions,
    diffuse_signals, regen_resources, decode_actions, MOVE_DELTAS,
    compute_phi_hidden,
)


# ---------------------------------------------------------------------------
# Trajectory recording step (NOT jitted — we need to extract numpy data)
# ---------------------------------------------------------------------------

def recording_step(state, cfg):
    """Single env step that returns full agent-level detail.

    Not JIT-compiled so we can extract per-agent data as numpy arrays.
    """
    N = cfg['N']

    # Build observation
    agent_count = build_agent_count_grid(state['positions'], state['alive'], N)
    obs = build_observation_batch(
        state['positions'], state['resources'], state['signals'],
        agent_count, state['energy'], cfg
    )

    # Forward pass
    new_hidden, new_sync, raw_actions, predictions = agent_multi_tick_v27_batch(
        obs, state['hidden'], state['sync_matrices'], state['phenotypes'], cfg
    )
    new_hidden = new_hidden * state['alive'][:, None]
    new_sync = new_sync * state['alive'][:, None, None]

    # Decode actions
    move_idx, consume, emit = decode_actions(raw_actions, cfg)

    # Movement
    deltas = MOVE_DELTAS[move_idx]
    new_positions = (state['positions'] + deltas) % N
    new_positions = jnp.where(state['alive'][:, None], new_positions, state['positions'])

    # Consumption
    resources, actual_consumed = apply_consumption(
        state['resources'], new_positions, consume * state['alive'], state['alive'], N
    )

    # Emission
    signals = apply_emissions(state['signals'], new_positions, emit, state['alive'], N)

    # Environment dynamics
    signals = diffuse_signals(signals, cfg)
    resources = regen_resources(resources, state['regen_rate'])

    # Energy update
    pre_energy = state['energy']
    energy = state['energy'] - cfg['metabolic_cost'] + actual_consumed * cfg['resource_value']
    energy = jnp.clip(energy, 0.0, 2.0)

    # Kill starved
    new_alive = state['alive'] & (energy > 0.0)

    new_state = {
        'resources': resources,
        'signals': signals,
        'positions': new_positions,
        'hidden': new_hidden,
        'energy': energy,
        'alive': new_alive,
        'phenotypes': state['phenotypes'],
        'genomes': state['genomes'],
        'sync_matrices': new_sync,
        'regen_rate': state['regen_rate'],
        'step_count': state['step_count'] + 1,
    }

    # Return step data for recording
    step_data = {
        'obs': obs,
        'hidden': new_hidden,
        'actions_raw': raw_actions,
        'move_idx': move_idx,
        'consume': consume,
        'emit': emit,
        'predictions': predictions,
        'energy_delta': energy - pre_energy,
        'positions': new_positions,
        'energy': energy,
        'alive': new_alive,
        'resources': resources,
        'signals': signals,
        'agent_count': agent_count,
    }

    return new_state, step_data


def select_focal_agent(state):
    """Pick the focal agent: alive agent with highest energy."""
    alive = np.array(state['alive'])
    energy = np.array(state['energy'])
    alive_energy = energy * alive
    return int(np.argmax(alive_energy))


def extract_focal_data(step_data, focal_idx, cfg):
    """Extract per-timestep data for the focal agent."""
    obs_side = cfg['obs_side']

    # Focal agent's observation (reshape to 5x5x3 + energy)
    focal_obs = np.array(step_data['obs'][focal_idx])
    obs_spatial = focal_obs[:obs_side*obs_side*3].reshape(obs_side, obs_side, 3)
    focal_energy_obs = float(focal_obs[-1])

    # Hidden state
    focal_hidden = np.array(step_data['hidden'][focal_idx]).tolist()

    # Position
    pos = np.array(step_data['positions'][focal_idx]).tolist()

    # Actions
    move = int(step_data['move_idx'][focal_idx])
    consume = float(step_data['consume'][focal_idx])
    emit = float(step_data['emit'][focal_idx])

    # Energy
    energy = float(step_data['energy'][focal_idx])
    energy_delta = float(step_data['energy_delta'][focal_idx])
    prediction = float(step_data['predictions'][focal_idx])

    # Local environment (resource density in 5x5 window)
    local_resources = obs_spatial[:, :, 0].tolist()  # channel 0 = resources
    local_signals = obs_spatial[:, :, 1].tolist()     # channel 1 = signals
    local_agents = obs_spatial[:, :, 2].tolist()      # channel 2 = agent counts

    return {
        'position': pos,
        'energy': energy,
        'energy_delta': energy_delta,
        'prediction': prediction,
        'prediction_error': abs(prediction - energy_delta),
        'move_direction': move,  # 0=stay, 1=up, 2=down, 3=left, 4=right
        'consume': consume,
        'emit': emit,
        'hidden_state': focal_hidden,
        'local_resources': local_resources,
        'local_signals': local_signals,
        'local_agents': local_agents,
    }


def extract_environment_snapshot(step_data, cfg, downsample=4):
    """Extract downsampled environment state for global context."""
    N = cfg['N']
    ds = downsample

    resources = np.array(step_data['resources'])
    signals = np.array(step_data['signals'])

    # Downsample by averaging
    rs_ds = resources[:N//ds*ds, :N//ds*ds].reshape(N//ds, ds, N//ds, ds).mean(axis=(1,3))
    sg_ds = signals[:N//ds*ds, :N//ds*ds].reshape(N//ds, ds, N//ds, ds).mean(axis=(1,3))

    # All alive agent positions and energies
    alive = np.array(step_data['alive'])
    positions = np.array(step_data['positions'])
    energies = np.array(step_data['energy'])

    alive_idx = np.where(alive)[0]
    agents = [
        {'pos': positions[i].tolist(), 'energy': float(energies[i])}
        for i in alive_idx
    ]

    return {
        'resource_grid': rs_ds.tolist(),
        'signal_grid': sg_ds.tolist(),
        'agents': agents,
        'n_alive': int(alive.sum()),
        'mean_energy': float(energies[alive].mean()) if alive.any() else 0.0,
    }


def compute_segment_affect(hidden_states, energies, energy_deltas, n_alive_history):
    """Compute framework-predicted affect coordinates for a segment."""
    hidden_arr = np.array(hidden_states)  # (T, H)

    # Valence: mean energy delta (proxy for viability gradient)
    valence = float(np.mean(energy_deltas))
    valence_norm = max(-1.0, min(1.0, valence * 100))  # scale to [-1, 1]

    # Arousal: mean hidden state update rate
    if len(hidden_arr) > 1:
        diffs = np.diff(hidden_arr, axis=0)
        arousal = float(np.mean(np.linalg.norm(diffs, axis=1)))
    else:
        arousal = 0.0

    # Effective rank of hidden state covariance
    if len(hidden_arr) > 5:
        cov = np.cov(hidden_arr.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.maximum(eigenvalues, 0)
        total = eigenvalues.sum()
        if total > 1e-10:
            p = eigenvalues / total
            p = p[p > 1e-10]
            eff_rank = float(np.exp(-np.sum(p * np.log(p))))
        else:
            eff_rank = 1.0
    else:
        eff_rank = 1.0

    # Population trend (proxy for ecological condition)
    if len(n_alive_history) > 1:
        pop_slope = float(np.polyfit(range(len(n_alive_history)), n_alive_history, 1)[0])
    else:
        pop_slope = 0.0

    return {
        'valence': valence_norm,
        'arousal': arousal,
        'effective_rank': eff_rank,
        'pop_slope': pop_slope,
        'mean_energy': float(np.mean(energies)),
        'energy_std': float(np.std(energies)),
    }


# ---------------------------------------------------------------------------
# Main trajectory recording
# ---------------------------------------------------------------------------

def record_trajectory(seed=23, output_dir='results/v36_trajectory',
                      record_cycles=None, steps_per_record=50,
                      downsample_env=4):
    """Record detailed trajectory from a V27 evolution.

    Uses the same evolution machinery as V27/V32 (JIT chunk runner + proper
    tournament selection) for non-recorded cycles (FAST), and a per-step
    recording loop for recorded cycles (SLOW but detailed).

    Args:
        seed: Random seed (23 = max Phi in V32)
        output_dir: Where to save results
        record_cycles: Which cycles to record in detail (default: all)
        steps_per_record: How often to save environment snapshots (focal agent
                         is recorded every step)
        downsample_env: Factor for downsampling environment grids
    """
    from v20_evolution import tournament_selection
    from v27_evolution import (
        run_cycle_with_metrics, rescue_population_v27,
    )

    os.makedirs(output_dir, exist_ok=True)

    cfg = generate_v27_config()
    n_cycles = cfg['n_cycles']
    steps_per_cycle = cfg['steps_per_cycle']

    if record_cycles is None:
        record_cycles = list(range(n_cycles))

    print(f"=== V36 Trajectory Recorder ===")
    print(f"Seed: {seed}")
    print(f"Recording cycles: {record_cycles}")
    print(f"Steps per cycle: {steps_per_cycle}")

    # Initialize
    state, key = init_v27(seed, cfg)

    # JIT-compiled chunk runner (used for non-recorded cycles)
    chunk_runner = make_v27_chunk_runner(cfg)

    # Warmup JIT
    print("Warming up JIT...")
    t0 = time.time()
    state, _ = chunk_runner(state)
    print(f"  JIT warmup: {time.time()-t0:.1f}s")

    # Re-initialize after warmup (warmup consumed one chunk)
    state, key = init_v27(seed, cfg)

    all_cycle_data = []

    for cycle in range(n_cycles):
        t0 = time.time()

        # === Cycle setup (same as run_v27) ===
        state['phenotypes'] = state['genomes'].copy()
        state['hidden'] = jnp.zeros_like(state['hidden'])
        state['sync_matrices'] = jnp.zeros_like(state['sync_matrices'])
        state['energy'] = jnp.where(
            state['alive'], cfg['initial_energy'], 0.0
        )
        state['step_count'] = jnp.array(0)

        drought_every = cfg.get('drought_every', 0)
        is_drought = (drought_every > 0) and (cycle > 0) and (cycle % drought_every == 0)
        if is_drought:
            state['resources'] = state['resources'] * cfg.get('drought_depletion', 0.05)
            state['regen_rate'] = jnp.array(cfg.get('drought_regen', 0.00002))
        else:
            state['regen_rate'] = jnp.array(cfg['resource_regen'])

        n_alive_start = int(state['alive'].sum())
        recording = cycle in record_cycles

        if recording:
            # === SLOW PATH: per-step recording ===
            focal_idx = select_focal_agent(state)
            focal_timeline = []
            env_snapshots = []
            hidden_states = []
            energies = []
            energy_deltas = []
            n_alive_history = []

            print(f"  Cycle {cycle}: RECORDING (focal={focal_idx}, "
                  f"{'DROUGHT' if is_drought else 'normal'}, pop={n_alive_start})")

            for step in range(steps_per_cycle):
                state, step_data = recording_step(state, cfg)

                # Check focal agent is still alive
                if not bool(step_data['alive'][focal_idx]):
                    focal_idx = select_focal_agent(
                        {k: step_data.get(k, state.get(k)) for k in ['alive', 'energy']}
                    )
                    if not bool(step_data['alive'][focal_idx]):
                        break  # everyone dead

                focal_data = extract_focal_data(step_data, focal_idx, cfg)
                focal_data['step'] = step
                focal_data['focal_idx'] = int(focal_idx)
                focal_timeline.append(focal_data)

                hidden_states.append(focal_data['hidden_state'])
                energies.append(focal_data['energy'])
                energy_deltas.append(focal_data['energy_delta'])
                n_alive_history.append(int(step_data['alive'].sum()))

                if step % steps_per_record == 0:
                    env_snap = extract_environment_snapshot(
                        step_data, cfg, downsample_env
                    )
                    env_snap['step'] = step
                    env_snapshots.append(env_snap)

                if step % 1000 == 0 and step > 0:
                    print(f"    step {step}/{steps_per_cycle} "
                          f"({time.time()-t0:.1f}s)")

            n_alive_end = int(state['alive'].sum())

            segment_affect = compute_segment_affect(
                hidden_states, energies, energy_deltas, n_alive_history
            )
            phi = float(compute_phi_hidden(state['hidden'], state['alive']))

            cycle_record = {
                'cycle': cycle,
                'is_drought': is_drought,
                'condition': 'drought' if is_drought else 'normal',
                'n_alive_start': n_alive_start,
                'n_alive_end': n_alive_end,
                'mortality': 1.0 - n_alive_end / max(n_alive_start, 1),
                'phi': phi,
                'segment_affect': segment_affect,
                'focal_timeline_length': len(focal_timeline),
                'n_env_snapshots': len(env_snapshots),
            }
            all_cycle_data.append(cycle_record)

            cycle_file = os.path.join(output_dir, f'cycle_{cycle:03d}.json')
            with open(cycle_file, 'w') as f:
                json.dump({
                    'meta': cycle_record,
                    'focal_timeline': focal_timeline,
                    'env_snapshots': env_snapshots,
                }, f)

            print(f"    → {len(focal_timeline)} steps, "
                  f"{len(env_snapshots)} snapshots, "
                  f"Φ={phi:.3f}, pop {n_alive_start}→{n_alive_end} "
                  f"({time.time()-t0:.1f}s)")

        else:
            # === FAST PATH: JIT chunk runner (same as run_v27) ===
            print(f"  Cycle {cycle}: evolving "
                  f"({'DROUGHT' if is_drought else 'normal'}, "
                  f"pop={n_alive_start})", end='')

            drought_regen = cfg.get('drought_regen', 0.00002) if is_drought else None
            state, metrics, fitness_np, survival_steps = run_cycle_with_metrics(
                state, chunk_runner, cfg, regen_override=drought_regen
            )

            n_alive_end = int(state['alive'].sum())
            print(f" → pop {n_alive_end} "
                  f"(Φ={metrics['mean_phi']:.3f}, {time.time()-t0:.1f}s)")

        # === Evolution (same as run_v27) ===
        if cycle < n_cycles - 1:
            key, k_sel, k_pos = jax.random.split(key, 3)

            # For recorded cycles, compute fitness from energy (proxy)
            if recording:
                alive_f = np.array(state['alive']).astype(np.float32)
                fitness_np = np.array(state['energy']) * alive_f

            state['genomes'] = tournament_selection(
                state['genomes'], jnp.array(fitness_np), state['alive'], k_sel, cfg
            )

            # Activate offspring
            if cfg.get('activate_offspring', False):
                M = cfg['M_max']
                n_alive = int(jnp.sum(state['alive']))
                n_keep = max(1, min(n_alive, int(M * cfg['elite_fraction'])))
                fitness_masked = np.where(
                    np.array(state['alive']), fitness_np, -np.inf
                )
                ranked = np.argsort(fitness_masked)[::-1]
                elite_set = set(ranked[:n_keep].tolist())

                alive_np = np.array(state['alive'])
                positions_np = np.array(state['positions'])
                energy_np = np.array(state['energy'])
                new_positions = np.array(
                    jax.random.randint(k_pos, (M, 2), 0, cfg['N'])
                )

                for i in range(M):
                    if i not in elite_set:
                        alive_np[i] = True
                        positions_np[i] = new_positions[i]
                        energy_np[i] = cfg['initial_energy']

                state['alive'] = jnp.array(alive_np)
                state['positions'] = jnp.array(positions_np)
                state['energy'] = jnp.array(energy_np)

            # Rescue if needed
            state, key = rescue_population_v27(state, key, cfg)

    # Save summary
    summary = {
        'seed': seed,
        'config': {k: v for k, v in cfg.items()
                   if isinstance(v, (int, float, str, bool))},
        'cycles': all_cycle_data,
    }

    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Done. {len(all_cycle_data)} cycles recorded. ===")
    print(f"Output: {output_dir}")

    return summary


# ---------------------------------------------------------------------------
# VLM narration pipeline
# ---------------------------------------------------------------------------

def generate_trajectory_vignettes(output_dir='results/v36_trajectory',
                                   segment_length=500):
    """Generate behavioral vignettes from recorded trajectory data.

    Each vignette describes a segment of the focal agent's experience:
    - What the agent sees (local environment)
    - What the agent does (actions taken)
    - How the agent's internal state changes
    - Population context (who's around, who's dying)

    These vignettes are fed to VLMs for narration.
    """
    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file) as f:
        summary = json.load(f)

    vignettes = []

    for cycle_info in summary['cycles']:
        cycle = cycle_info['cycle']
        cycle_file = os.path.join(output_dir, f'cycle_{cycle:03d}.json')

        if not os.path.exists(cycle_file):
            continue

        with open(cycle_file) as f:
            cycle_data = json.load(f)

        timeline = cycle_data['focal_timeline']
        meta = cycle_data['meta']

        # Split timeline into segments
        for seg_start in range(0, len(timeline), segment_length):
            seg_end = min(seg_start + segment_length, len(timeline))
            segment = timeline[seg_start:seg_end]

            if len(segment) < 10:
                continue

            # Compute segment statistics
            energies = [s['energy'] for s in segment]
            deltas = [s['energy_delta'] for s in segment]
            moves = [s['move_direction'] for s in segment]
            consumes = [s['consume'] for s in segment]
            pred_errors = [s['prediction_error'] for s in segment]

            # Resource availability in observation window
            local_res = [np.mean(s['local_resources']) for s in segment]
            local_agents_count = [np.sum(np.array(s['local_agents']) > 0)
                                  for s in segment]

            # Movement pattern
            unique_moves = len(set(moves))
            stay_fraction = moves.count(0) / len(moves)

            # Build behavioral description (NO affect vocabulary)
            desc = []
            desc.append(f"Observation window: Cycle {cycle}, "
                       f"steps {seg_start}-{seg_end}.")
            desc.append(f"Ecological condition: "
                       f"{'DROUGHT — resources depleted to 1%, zero regeneration' if meta['is_drought'] else 'normal resource regeneration'}.")
            desc.append(f"Population: {meta['n_alive_start']} agents at cycle start"
                       f"{', declining to ' + str(meta['n_alive_end']) + ' by cycle end' if meta['n_alive_end'] < meta['n_alive_start'] else ''}.")
            desc.append(f"Mortality this cycle: {meta['mortality']*100:.0f}%.")
            desc.append("")
            desc.append("Focal agent behavioral observations:")
            desc.append(f"  Energy trajectory: {energies[0]:.3f} → {energies[-1]:.3f} "
                       f"(mean delta: {np.mean(deltas):.5f}/step)")
            desc.append(f"  Resource density in observation window: "
                       f"mean {np.mean(local_res):.4f}, "
                       f"{'declining' if local_res[-1] < local_res[0] else 'stable or increasing'}.")
            desc.append(f"  Neighbors visible: mean {np.mean(local_agents_count):.1f} "
                       f"agents in 5×5 window.")
            desc.append(f"  Movement: {stay_fraction*100:.0f}% stationary, "
                       f"{unique_moves} distinct movement directions used.")
            desc.append(f"  Consumption attempts: {np.mean(consumes)*100:.0f}% of steps.")
            desc.append(f"  Prediction accuracy: mean |error| = {np.mean(pred_errors):.5f}.")
            desc.append("")
            desc.append(f"Internal state dynamics:")
            desc.append(f"  State update rate: {meta['segment_affect']['arousal']:.4f}")
            desc.append(f"  Effective representation dimensionality: "
                       f"{meta['segment_affect']['effective_rank']:.1f}")
            desc.append(f"  Information integration (Φ): {meta['phi']:.3f}")
            desc.append("")
            desc.append("System context: These are evolved GRU neural networks "
                       "(~4200 parameters each) on a 128×128 toroidal grid. "
                       "No biological components. No training on human data. "
                       "Genome evolved through tournament selection with "
                       "Gaussian mutation. Within-lifetime gradient learning "
                       "on energy prediction error through a 2-layer MLP.")

            vignette = {
                'cycle': cycle,
                'segment': [seg_start, seg_end],
                'condition': meta['condition'],
                'description': '\n'.join(desc),
                'numerical_summary': {
                    'cycle': cycle,
                    'is_drought': meta['is_drought'],
                    'population_start': meta['n_alive_start'],
                    'population_end': meta['n_alive_end'],
                    'mortality': meta['mortality'],
                    'energy_start': energies[0],
                    'energy_end': energies[-1],
                    'mean_energy_delta': float(np.mean(deltas)),
                    'mean_local_resource': float(np.mean(local_res)),
                    'mean_neighbors': float(np.mean(local_agents_count)),
                    'stay_fraction': stay_fraction,
                    'mean_pred_error': float(np.mean(pred_errors)),
                    'state_update_rate': meta['segment_affect']['arousal'],
                    'effective_rank': meta['segment_affect']['effective_rank'],
                    'phi': meta['phi'],
                    'robustness': meta['robustness'],
                },
                'framework_prediction': meta['segment_affect'],
            }
            vignettes.append(vignette)

    # Save vignettes
    vignette_file = os.path.join(output_dir, 'vignettes.json')
    with open(vignette_file, 'w') as f:
        json.dump(vignettes, f, indent=2)

    print(f"Generated {len(vignettes)} vignettes from "
          f"{len(summary['cycles'])} recorded cycles.")
    return vignettes


# ---------------------------------------------------------------------------
# PCA for 3D affect trajectory
# ---------------------------------------------------------------------------

def compute_affect_trajectory_3d(output_dir='results/v36_trajectory'):
    """Compute PCA of hidden states across all recorded cycles.

    Returns 3D coordinates for the affect trajectory visualization.
    """
    from sklearn.decomposition import PCA

    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file) as f:
        summary = json.load(f)

    # Collect all hidden states
    all_hidden = []
    all_meta = []  # cycle, step, energy, condition

    for cycle_info in summary['cycles']:
        cycle = cycle_info['cycle']
        cycle_file = os.path.join(output_dir, f'cycle_{cycle:03d}.json')

        if not os.path.exists(cycle_file):
            continue

        with open(cycle_file) as f:
            cycle_data = json.load(f)

        # Subsample to manageable size (every 10th step)
        timeline = cycle_data['focal_timeline']
        for i in range(0, len(timeline), 10):
            s = timeline[i]
            all_hidden.append(s['hidden_state'])
            all_meta.append({
                'cycle': cycle,
                'step': s['step'],
                'energy': s['energy'],
                'condition': cycle_data['meta']['condition'],
                'n_alive': cycle_data['meta']['n_alive_end'],
            })

    if len(all_hidden) < 10:
        print("Not enough data for PCA")
        return None

    hidden_arr = np.array(all_hidden)

    # PCA to 3D
    pca = PCA(n_components=3)
    coords_3d = pca.fit_transform(hidden_arr)

    # Build trajectory data
    trajectory = {
        'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
        'points': [
            {
                'x': float(coords_3d[i, 0]),
                'y': float(coords_3d[i, 1]),
                'z': float(coords_3d[i, 2]),
                **all_meta[i]
            }
            for i in range(len(coords_3d))
        ]
    }

    traj_file = os.path.join(output_dir, 'affect_trajectory_3d.json')
    with open(traj_file, 'w') as f:
        json.dump(trajectory, f)

    print(f"3D trajectory: {len(trajectory['points'])} points, "
          f"variance explained: {pca.explained_variance_ratio_}")

    return trajectory


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='V36 Trajectory Recorder')
    parser.add_argument('--seed', type=int, default=23,
                       help='Random seed (default: 23, highest Phi in V32)')
    parser.add_argument('--output', type=str, default='results/v36_trajectory',
                       help='Output directory')
    parser.add_argument('--cycles', type=str, default=None,
                       help='Cycles to record (comma-separated, default: all)')
    parser.add_argument('--steps-per-record', type=int, default=50,
                       help='Environment snapshot frequency')
    parser.add_argument('--vignettes-only', action='store_true',
                       help='Only generate vignettes from existing data')
    parser.add_argument('--pca-only', action='store_true',
                       help='Only compute 3D PCA from existing data')

    args = parser.parse_args()

    if args.vignettes_only:
        generate_trajectory_vignettes(args.output)
    elif args.pca_only:
        compute_affect_trajectory_3d(args.output)
    else:
        record_cycles = None
        if args.cycles:
            record_cycles = [int(c) for c in args.cycles.split(',')]

        record_trajectory(
            seed=args.seed,
            output_dir=args.output,
            record_cycles=record_cycles,
            steps_per_record=args.steps_per_record,
        )

        # Generate vignettes and PCA
        generate_trajectory_vignettes(args.output)
        compute_affect_trajectory_3d(args.output)
