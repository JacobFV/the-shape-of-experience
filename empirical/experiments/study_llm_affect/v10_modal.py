"""
V10: Modal Cloud Runner
========================

Run the V10 MARL affect experiment on Modal GPUs.

Usage:
    modal run v10_modal.py                      # Sanity check on CPU
    modal run v10_modal.py --mode train          # Train full model
    modal run v10_modal.py --mode ablation       # Train all ablations
    modal run v10_modal.py --mode full           # Full experiment
"""

import modal
import os

# Modal app
app = modal.App("v10-affect-experiment")

# Volume for persistent results
results_volume = modal.Volume.from_name("v10-results", create_if_missing=True)

# Determine source directory
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# Image with all dependencies + source code baked in
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "jax[cpu]",
        "flax",
        "optax",
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "openai",
        "python-dotenv",
    )
    .add_local_dir(SRC_DIR, remote_path="/root/experiment",
                   ignore=lambda path: not path.name.endswith(".py"))
)


@app.function(
    image=image,
    timeout=600,
)
def sanity_check():
    """Run sanity check on Modal (CPU)."""
    import sys
    # Source files are at /root/experiment/ via add_local_dir
    if "/root/experiment" not in sys.path:
        sys.path.insert(0, "/root/experiment")

    # Also ensure cwd-relative imports work
    import os
    os.chdir("/root/experiment")

    from v10_run import sanity_check as _sanity_check
    _sanity_check()
    return "Sanity check passed"


@app.function(
    image=image,
    volumes={"/results": results_volume},
    timeout=3600 * 24,  # 24 hours
    memory=16384,
)
def train_condition(condition: str, seed: int, total_steps: int = 100_000):
    """Train a single condition on Modal."""
    import sys, os
    if "/root/experiment" not in sys.path:
        sys.path.insert(0, "/root/experiment")
    os.chdir("/root/experiment")

    from v10_environment import EnvConfig, make_ablation_configs
    from v10_agent import AgentConfig, train

    env_configs = make_ablation_configs()
    env_config = env_configs.get(condition, EnvConfig())

    agent_config = AgentConfig()
    if condition == 'no_world_model':
        agent_config.use_world_model = False
        agent_config.world_model_coef = 0.0
    elif condition == 'no_self_prediction':
        agent_config.use_self_prediction = False
        agent_config.self_pred_coef = 0.0
    elif condition == 'no_intrinsic_motivation':
        agent_config.use_intrinsic_motivation = False

    save_dir = f"/results/{condition}/seed_{seed}"
    os.makedirs(save_dir, exist_ok=True)

    results = train(
        agent_config=agent_config,
        env_config=env_config,
        total_steps=total_steps,
        seed=seed,
        save_dir=save_dir,
    )

    results_volume.commit()

    return {
        'condition': condition,
        'seed': seed,
        'steps': total_steps,
        'n_metrics': len(results.get('metrics_history', [])),
        'n_latent': len(results.get('latent_history', [])),
    }


@app.function(
    image=image,
    volumes={"/results": results_volume},
    timeout=1800,
    memory=8192,
)
def analyze_results():
    """Run RSA analysis on trained results."""
    import sys, os
    if "/root/experiment" not in sys.path:
        sys.path.insert(0, "/root/experiment")
    os.chdir("/root/experiment")
    import pickle
    import numpy as np

    from v10_affect import compute_affect_for_agent
    from v10_analysis import run_rsa_analysis, plot_ablation_comparison, RSAResult

    conditions = [
        'full', 'no_partial_obs', 'no_long_horizon',
        'no_world_model', 'no_self_prediction',
        'no_intrinsic_motivation', 'no_delayed_rewards',
    ]

    ablation_results = {}

    for condition in conditions:
        for seed in range(3):
            results_path = f"/results/{condition}/seed_{seed}/final_results.pkl"
            if not os.path.exists(results_path):
                continue

            with open(results_path, 'rb') as f:
                results = pickle.load(f)

            latent_history = results.get('latent_history', [])
            if not latent_history:
                continue

            env_config = results['config']['env']
            agent_config = results['config']['agent']

            try:
                vectors, affect_mat = compute_affect_for_agent(
                    latent_history, 0,
                    agent_config.latent_dim, env_config.n_actions
                )

                # Synthetic embedding for now (VLM translation requires API)
                emb_mat = np.random.randn(len(affect_mat), 16) * 0.5

                result = run_rsa_analysis(affect_mat, emb_mat, subsample=500)
                ablation_results[condition] = result
            except Exception as e:
                print(f"Error: {condition}/seed_{seed}: {e}")

    if ablation_results:
        plot_ablation_comparison(ablation_results, "/results/ablation_comparison.png")
        results_volume.commit()

    return {k: v.to_dict() for k, v in ablation_results.items()}


@app.local_entrypoint()
def main(
    mode: str = "sanity",
    steps: int = 100_000,
    seeds: int = 1,
    condition: str = "full",
):
    """
    Main entrypoint for Modal execution.

    Args:
        mode: 'sanity', 'train', 'ablation', or 'full'
        steps: training steps per condition
        seeds: number of random seeds
        condition: which ablation condition (for mode='train')
    """
    if mode == "sanity":
        result = sanity_check.remote()
        print(result)

    elif mode == "train":
        print(f"Training {condition} for {steps} steps, {seeds} seeds")
        futures = []
        for seed in range(seeds):
            futures.append(train_condition.spawn(condition, seed, steps))

        for i, future in enumerate(futures):
            result = future.get()
            print(f"Seed {i}: {result}")

    elif mode == "ablation":
        conditions = [
            'full', 'no_partial_obs', 'no_long_horizon',
            'no_world_model', 'no_self_prediction',
            'no_intrinsic_motivation', 'no_delayed_rewards',
        ]
        print(f"Training all {len(conditions)} conditions for {steps} steps, {seeds} seeds")

        futures = []
        for cond in conditions:
            for seed in range(seeds):
                futures.append(train_condition.spawn(cond, seed, steps))

        for future in futures:
            result = future.get()
            print(f"Completed: {result}")

    elif mode == "full":
        conditions = [
            'full', 'no_partial_obs', 'no_long_horizon',
            'no_world_model', 'no_self_prediction',
            'no_intrinsic_motivation', 'no_delayed_rewards',
        ]

        print(f"Full experiment: {len(conditions)} conditions x {seeds} seeds x {steps} steps")

        futures = []
        for cond in conditions:
            for seed in range(seeds):
                futures.append(train_condition.spawn(cond, seed, steps))

        for future in futures:
            result = future.get()
            print(f"Completed: {result}")

        print("\nRunning analysis...")
        analysis = analyze_results.remote()
        print(f"Analysis results: {analysis}")
