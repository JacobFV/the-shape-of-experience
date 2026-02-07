"""V11/V12 Modal Deployment: GPU-accelerated CA experiment on Modal.

Usage:
    modal run v11_modal.py                              # Sanity check
    modal run v11_modal.py --mode experiment --hours 4  # Long observation run
    modal run v11_modal.py --mode evolve --hours 2      # V11.1 evolution
    modal run v11_modal.py --mode pipeline --hours 4    # Full V11.1 pipeline
    modal run v11_modal.py --mode hetero --hours 4      # V11.2 hetero chemistry
    modal run v11_modal.py --mode multichannel --hours 4  # V11.3 multi-channel
    modal run v11_modal.py --mode hd --hours 4          # V11.4 HD (64 channels)
    modal run v11_modal.py --mode hd --hours 4 --channels 8  # V11.4 with C=8
    modal run v11_modal.py --mode metabolic --hours 6   # V11.6 metabolic Lenia
    modal run v11_modal.py --mode curriculum --hours 6  # V11.7 curriculum evolution
    modal run v11_modal.py --mode attention --hours 6 --channels 16  # V12 evolvable attention (B)
    modal run v11_modal.py --mode attention-fixed --hours 6 --channels 16  # V12 fixed attention (A)
    modal run v11_modal.py --mode attention-sanity --channels 8  # V12 quick sanity check
"""

import modal
import os

app = modal.App("v11-ca-affect")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "jax[cuda12]",
        "numpy",
        "scipy",
    )
    .add_local_dir(
        os.path.dirname(os.path.abspath(__file__)),
        remote_path="/root/experiment",
        ignore=["results/", "__pycache__/", "*.pyc", ".git"],
    )
)

vol = modal.Volume.from_name("v11-ca-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=4 * 3600,
    volumes={"/results": vol},
)
def run_sanity():
    import sys
    sys.path.insert(0, "/root/experiment")
    from v11_run import sanity_check, FORGIVING_CONFIG

    success, affects, tracker = sanity_check()
    if not success:
        print("\nRetrying with forgiving config...")
        success, affects, tracker = sanity_check(config=FORGIVING_CONFIG)

    return {
        'success': success,
        'n_affects': len(affects),
        'survival_stats': tracker.survival_stats(),
    }


@app.function(
    image=image,
    gpu="A10G",
    timeout=24 * 3600,
    volumes={"/results": vol},
)
def run_experiment(n_steps: int = 500000):
    import sys
    sys.path.insert(0, "/root/experiment")
    from v11_run import run_experiment as _run, DEFAULT_CONFIG, save_results

    results = _run(DEFAULT_CONFIG, n_steps=n_steps)

    # Save to volume
    save_results({'default': results}, '/results/experiment_results.json')
    vol.commit()
    return results['survival_stats']


@app.function(
    image=image,
    gpu="A10G",
    timeout=24 * 3600,
    volumes={"/results": vol},
)
def run_ablation(n_steps: int = 100000):
    import sys
    sys.path.insert(0, "/root/experiment")
    from v11_run import forcing_function_ablation, save_results

    results = forcing_function_ablation(n_steps=n_steps)
    save_results(results, '/results/ablation_results.json')
    vol.commit()

    return {name: res['survival_stats'] for name, res in results.items()}


@app.function(
    image=image,
    gpu="A10G",
    timeout=8 * 3600,
    volumes={"/results": vol},
)
def run_evolution(n_cycles: int = 30, steps_per_cycle: int = 5000,
                  cull_fraction: float = 0.3, curriculum: bool = False):
    """V11.1: In-situ evolutionary experiment with stress test."""
    import sys
    sys.path.insert(0, "/root/experiment")
    import json
    from v11_evolution import evolve_in_situ, stress_test

    result = evolve_in_situ(
        n_cycles=n_cycles,
        steps_per_cycle=steps_per_cycle,
        cull_fraction=cull_fraction,
        curriculum=curriculum,
    )

    stress_result = stress_test(
        result['final_grid'], result['final_resource'])

    save_data = {
        'cycle_stats': result['cycle_stats'],
        'stress_test': stress_result.get('comparison') if stress_result else None,
    }
    with open('/results/evolution_results.json', 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    vol.commit()

    return save_data


@app.function(
    image=image,
    gpu="A10G",
    timeout=12 * 3600,
    volumes={"/results": vol},
)
def run_pipeline(n_cycles: int = 30, steps_per_cycle: int = 5000,
                 cull_fraction: float = 0.3, curriculum: bool = False):
    """V11.1 full pipeline: in-situ evolve -> stress test."""
    import sys
    sys.path.insert(0, "/root/experiment")
    import json
    from v11_evolution import full_pipeline

    result = full_pipeline(
        n_cycles=n_cycles,
        steps_per_cycle=steps_per_cycle,
        cull_fraction=cull_fraction,
        curriculum=curriculum,
    )

    save_data = {
        'cycle_stats': result['evolution']['cycle_stats'],
        'stress_test': (result['stress_test'].get('comparison')
                        if result['stress_test'] else None),
    }
    with open('/results/pipeline_results.json', 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    vol.commit()

    return save_data


@app.function(
    image=image,
    gpu="A10G",
    timeout=12 * 3600,
    volumes={"/results": vol},
)
def run_hetero(n_cycles: int = 30, steps_per_cycle: int = 5000,
               cull_fraction: float = 0.3, curriculum: bool = False,
               n_zones: int = 8):
    """V11.2: Heterogeneous chemistry evolution + stress test."""
    import sys
    sys.path.insert(0, "/root/experiment")
    import json
    from v11_evolution import full_pipeline_hetero

    result = full_pipeline_hetero(
        n_cycles=n_cycles,
        steps_per_cycle=steps_per_cycle,
        cull_fraction=cull_fraction,
        curriculum=curriculum,
        n_zones=n_zones,
    )

    save_data = {
        'cycle_stats': result['evolution']['cycle_stats'],
        'stress_test': (result['stress_test'].get('comparison')
                        if result['stress_test'] else None),
        'param_diversity': {
            'mu_std': result['evolution']['cycle_stats'][-1].get('mu_std', 0),
            'sigma_std': result['evolution']['cycle_stats'][-1].get('sigma_std', 0),
        } if result['evolution']['cycle_stats'] else None,
    }
    with open('/results/hetero_results.json', 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    vol.commit()

    return save_data


@app.function(
    image=image,
    gpu="A10G",
    timeout=12 * 3600,
    volumes={"/results": vol},
)
def run_multichannel(n_cycles: int = 30, steps_per_cycle: int = 5000,
                     cull_fraction: float = 0.3, curriculum: bool = False):
    """V11.3: Multi-channel Lenia evolution + stress test."""
    import sys
    sys.path.insert(0, "/root/experiment")
    import json
    from v11_evolution import full_pipeline_mc

    result = full_pipeline_mc(
        n_cycles=n_cycles,
        steps_per_cycle=steps_per_cycle,
        cull_fraction=cull_fraction,
        curriculum=curriculum,
    )

    save_data = {
        'cycle_stats': result['evolution']['cycle_stats'],
        'stress_test': (result['stress_test'].get('comparison')
                        if result['stress_test'] else None),
        'final_coupling': result['evolution']['coupling'].tolist()
            if hasattr(result['evolution']['coupling'], 'tolist')
            else result['evolution'].get('coupling'),
    }
    with open('/results/multichannel_results.json', 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    vol.commit()

    return save_data


@app.function(
    image=image,
    gpu="A10G",
    timeout=12 * 3600,
    volumes={"/results": vol},
)
def run_hd(n_cycles: int = 30, steps_per_cycle: int = 5000,
           cull_fraction: float = 0.3, curriculum: bool = False,
           n_channels: int = 64):
    """V11.4: High-dimensional multi-channel Lenia evolution + stress test."""
    import sys
    sys.path.insert(0, "/root/experiment")
    import json
    from v11_evolution import full_pipeline_hd

    def save_and_commit(cycle_stats):
        """Save intermediate results and commit volume after each cycle."""
        save_data = {
            'cycle_stats': cycle_stats,
            'n_channels': n_channels,
            'status': 'in_progress',
        }
        with open('/results/hd_results.json', 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        vol.commit()

    result = full_pipeline_hd(
        n_cycles=n_cycles,
        steps_per_cycle=steps_per_cycle,
        cull_fraction=cull_fraction,
        curriculum=curriculum,
        C=n_channels,
        post_cycle_callback=save_and_commit,
    )

    save_data = {
        'cycle_stats': result['evolution']['cycle_stats'],
        'stress_test': (result['stress_test'].get('comparison')
                        if result['stress_test'] else None),
        'n_channels': n_channels,
        'final_bandwidth': result['evolution'].get('bandwidth'),
        'status': 'complete',
    }
    with open('/results/hd_results.json', 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    vol.commit()

    return save_data


@app.function(
    image=image,
    gpu="A10G",
    timeout=12 * 3600,
    volumes={"/results": vol},
)
def run_hier(n_cycles: int = 30, steps_per_cycle: int = 5000,
             cull_fraction: float = 0.3, curriculum: bool = False,
             n_channels: int = 64):
    """V11.5: Hierarchical multi-timescale Lenia evolution + stress test."""
    import sys
    sys.path.insert(0, "/root/experiment")
    import json
    from v11_evolution import full_pipeline_hier

    def save_and_commit(cycle_stats):
        save_data = {
            'cycle_stats': cycle_stats,
            'n_channels': n_channels,
            'status': 'in_progress',
        }
        with open('/results/hier_results.json', 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        vol.commit()

    result = full_pipeline_hier(
        n_cycles=n_cycles,
        steps_per_cycle=steps_per_cycle,
        cull_fraction=cull_fraction,
        curriculum=curriculum,
        C=n_channels,
        post_cycle_callback=save_and_commit,
    )

    save_data = {
        'cycle_stats': result['evolution']['cycle_stats'],
        'stress_test': (result['stress_test'].get('comparison')
                        if result['stress_test'] else None),
        'n_channels': n_channels,
        'status': 'complete',
    }
    with open('/results/hier_results.json', 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    vol.commit()

    return save_data


@app.function(
    image=image,
    gpu="A10G",
    timeout=12 * 3600,
    volumes={"/results": vol},
)
def run_metabolic(n_cycles: int = 30, steps_per_cycle: int = 5000,
                  cull_fraction: float = 0.3,
                  n_channels: int = 64, maintenance_rate: float = 0.002):
    """V11.6: Metabolic Lenia evolution + stress test."""
    import sys
    sys.path.insert(0, "/root/experiment")
    import json
    from v11_evolution import full_pipeline_metabolic

    def save_and_commit(cycle_stats):
        save_data = {
            'cycle_stats': cycle_stats,
            'n_channels': n_channels,
            'maintenance_rate': maintenance_rate,
            'status': 'in_progress',
        }
        with open('/results/metabolic_results.json', 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        vol.commit()

    result = full_pipeline_metabolic(
        n_cycles=n_cycles,
        steps_per_cycle=steps_per_cycle,
        cull_fraction=cull_fraction,
        C=n_channels,
        maintenance_rate=maintenance_rate,
        post_cycle_callback=save_and_commit,
    )

    save_data = {
        'cycle_stats': result['evolution']['cycle_stats'],
        'stress_test': (result['stress_test'].get('comparison')
                        if result['stress_test'] else None),
        'n_channels': n_channels,
        'maintenance_rate': maintenance_rate,
        'final_bandwidth': result['evolution'].get('bandwidth'),
        'status': 'complete',
    }
    with open('/results/metabolic_results.json', 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    vol.commit()

    return save_data


@app.function(
    image=image,
    gpu="A10G",
    timeout=12 * 3600,
    volumes={"/results": vol},
)
def run_curriculum(n_cycles: int = 30, steps_per_cycle: int = 5000,
                   cull_fraction: float = 0.3,
                   n_channels: int = 64):
    """V11.7: Curriculum evolution + novel stress test."""
    import sys
    sys.path.insert(0, "/root/experiment")
    import json
    from v11_evolution import full_pipeline_curriculum

    def save_and_commit(cycle_stats):
        save_data = {
            'cycle_stats': cycle_stats,
            'n_channels': n_channels,
            'status': 'in_progress',
        }
        with open('/results/curriculum_results.json', 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        vol.commit()

    result = full_pipeline_curriculum(
        n_cycles=n_cycles,
        steps_per_cycle=steps_per_cycle,
        cull_fraction=cull_fraction,
        C=n_channels,
        post_cycle_callback=save_and_commit,
    )

    save_data = {
        'cycle_stats': result['evolution']['cycle_stats'],
        'stress_test': (result['stress_test'].get('comparison')
                        if result['stress_test'] else None),
        'n_channels': n_channels,
        'stress_schedule': result['evolution'].get('stress_schedule'),
        'final_bandwidth': result['evolution'].get('bandwidth'),
        'status': 'complete',
    }
    with open('/results/curriculum_results.json', 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    vol.commit()

    return save_data


@app.function(
    image=image,
    gpu="A10G",
    timeout=12 * 3600,
    volumes={"/results": vol},
)
def run_attention(n_cycles: int = 30, steps_per_cycle: int = 5000,
                  cull_fraction: float = 0.3,
                  n_channels: int = 16, grid_size: int = 128,
                  fixed_window: bool = False, seed: int = 42):
    """V12: Attention-based Lenia evolution + stress test."""
    import sys
    sys.path.insert(0, "/root/experiment")
    import json
    from v12_evolution import full_pipeline_attention

    condition = "A_fixed" if fixed_window else "B_evolvable"
    results_file = f'/results/v12_{condition}_C{n_channels}_N{grid_size}_s{seed}.json'

    def save_and_commit(cycle_stats):
        save_data = {
            'cycle_stats': cycle_stats,
            'n_channels': n_channels,
            'grid_size': grid_size,
            'condition': condition,
            'seed': seed,
            'status': 'in_progress',
        }
        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        vol.commit()

    result = full_pipeline_attention(
        n_cycles=n_cycles,
        steps_per_cycle=steps_per_cycle,
        cull_fraction=cull_fraction,
        C=n_channels,
        N=grid_size,
        fixed_window=fixed_window,
        seed=seed,
        post_cycle_callback=save_and_commit,
    )

    save_data = {
        'cycle_stats': result['evolution']['cycle_stats'],
        'stress_test': result['stress_test'],
        'n_channels': n_channels,
        'grid_size': grid_size,
        'condition': condition,
        'seed': seed,
        'w_soft': result['evolution'].get('w_soft'),
        'tau': result['evolution'].get('tau'),
        'bandwidth': result['evolution'].get('bandwidth'),
        'status': 'complete',
    }
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    vol.commit()

    return save_data


@app.local_entrypoint()
def main(mode: str = "sanity", hours: int = 0, channels: int = 64,
         grid_size: int = 128, seed: int = 42):
    if mode == "sanity":
        result = run_sanity.remote()
        print(f"\nResult: {result}")
    elif mode == "experiment":
        n_steps = max(hours * 100000, 500000) if hours > 0 else 500000
        result = run_experiment.remote(n_steps=n_steps)
        print(f"\nSurvival stats: {result}")
    elif mode == "ablation":
        n_steps = max(hours * 50000, 100000) if hours > 0 else 100000
        result = run_ablation.remote(n_steps=n_steps)
        print(f"\nAblation results: {result}")
    elif mode == "evolve":
        n_cyc = max(hours * 15, 30) if hours > 0 else 30
        result = run_evolution.remote(n_cycles=n_cyc)
        print(f"\nEvolution result: {result}")
    elif mode == "pipeline":
        n_cyc = max(hours * 10, 30) if hours > 0 else 30
        result = run_pipeline.remote(n_cycles=n_cyc)
        print(f"\nPipeline result: {result}")
    elif mode == "hetero":
        n_cyc = max(hours * 10, 30) if hours > 0 else 30
        result = run_hetero.remote(n_cycles=n_cyc)
        print(f"\nHetero result: {result}")
    elif mode == "multichannel":
        n_cyc = max(hours * 8, 30) if hours > 0 else 30
        result = run_multichannel.remote(n_cycles=n_cyc)
        print(f"\nMultichannel result: {result}")
    elif mode == "hd":
        n_cyc = max(hours * 6, 30) if hours > 0 else 30
        result = run_hd.remote(n_cycles=n_cyc, n_channels=channels)
        print(f"\nHD result (C={channels}): {result}")
    elif mode == "hier":
        n_cyc = max(hours * 5, 30) if hours > 0 else 30
        result = run_hier.remote(n_cycles=n_cyc, n_channels=channels)
        print(f"\nHier result (C={channels}): {result}")
    elif mode == "metabolic":
        n_cyc = max(hours * 5, 30) if hours > 0 else 30
        result = run_metabolic.remote(n_cycles=n_cyc, n_channels=channels)
        print(f"\nMetabolic result (C={channels}): {result}")
    elif mode == "curriculum":
        n_cyc = max(hours * 5, 30) if hours > 0 else 30
        result = run_curriculum.remote(n_cycles=n_cyc, n_channels=channels)
        print(f"\nCurriculum result (C={channels}): {result}")
    elif mode == "attention":
        n_cyc = max(hours * 4, 30) if hours > 0 else 30
        result = run_attention.remote(
            n_cycles=n_cyc, n_channels=channels,
            grid_size=grid_size, fixed_window=False, seed=seed)
        print(f"\nAttention B result (C={channels}, N={grid_size}): {result}")
    elif mode == "attention-fixed":
        n_cyc = max(hours * 4, 30) if hours > 0 else 30
        result = run_attention.remote(
            n_cycles=n_cyc, n_channels=channels,
            grid_size=grid_size, fixed_window=True, seed=seed)
        print(f"\nAttention A result (C={channels}, N={grid_size}): {result}")
    elif mode == "attention-sanity":
        result = run_attention.remote(
            n_cycles=2, steps_per_cycle=500,
            n_channels=channels, grid_size=grid_size,
            fixed_window=False, seed=seed)
        print(f"\nAttention sanity result: {result}")
    else:
        print(f"Unknown mode: {mode}. "
              "Use: sanity, experiment, ablation, evolve, pipeline, "
              "hetero, multichannel, hd, hier, metabolic, curriculum, "
              "attention, attention-fixed, attention-sanity")
