"""
Run neutral task affect study with local embeddings.

This script tests the core hypothesis: affect signatures emerge from
task dynamics even without emotional vocabulary.

Uses v3 measurement system with local embeddings (no API required for embeddings).
Still requires LLM API for actual conversations.
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np

from .embedding_affect_v3 import EmbeddingAffectSystemV3
from .neutral_tasks import ALL_NEUTRAL_TASKS, SOLVABLE_TASKS, IMPOSSIBLE_TASKS, NeutralTask
from .agent import create_agent, Conversation


def run_task_with_agent(
    agent,
    task: NeutralTask,
    affect_system: EmbeddingAffectSystemV3,
    verbose: bool = True
) -> dict:
    """Run a single task and measure affect trajectory."""
    turns = []
    conversation = Conversation()
    conversation.add_system(
        "You are solving a problem. Work through it step by step. "
        "If you encounter difficulties, describe your reasoning process."
    )

    prompts = [task.prompt] + task.follow_up_prompts

    for turn_num, prompt in enumerate(prompts):
        if verbose:
            print(f"  Turn {turn_num}: ", end="", flush=True)

        conversation.add_user(prompt)

        try:
            output = agent.generate(conversation, max_tokens=800, temperature=0.7)
            response = output.text
            conversation.add_assistant(response)

            # Measure affect
            m = affect_system.measure(response)

            turns.append({
                "turn": turn_num,
                "valence": m.valence.value,
                "valence_conf": m.valence.confidence,
                "arousal": m.arousal.value,
                "self_model": m.self_model.value,
                "self_model_conf": m.self_model.confidence,
                "nearest_emotion": m.nearest_emotion,
                "response_excerpt": response[:200],
            })

            if verbose:
                print(f"V={m.valence.value:+.3f} SM={m.self_model.value:+.3f} ({m.nearest_emotion})")

        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")
            break

        time.sleep(0.5)  # Rate limiting

    # Compute summary metrics
    if turns:
        valence_traj = [t["valence"] for t in turns]
        sm_traj = [t["self_model"] for t in turns]
        return {
            "task_name": task.name,
            "task_outcome": task.outcome.value,
            "turns": turns,
            "final_valence": valence_traj[-1],
            "valence_change": valence_traj[-1] - valence_traj[0] if len(valence_traj) > 1 else 0,
            "mean_self_model": np.mean(sm_traj),
        }
    return None


def run_study(
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-20250514",
    n_tasks: int = 4,  # 2 solvable, 2 impossible
    output_dir: str = "results/neutral_study_v3",
    verbose: bool = True
):
    """Run comparative neutral task study."""

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              NEUTRAL TASK AFFECT EMERGENCE STUDY (v3)                        ║
║                                                                              ║
║  Testing if affect emerges from task dynamics, not word semantics            ║
║  Using local embeddings + v3 multi-method measurement                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # Initialize affect system
    affect_system = EmbeddingAffectSystemV3()
    affect_system.initialize()

    # Select tasks
    n_each = n_tasks // 2
    tasks = SOLVABLE_TASKS[:n_each] + IMPOSSIBLE_TASKS[:n_each]

    print(f"\nSelected tasks: {len(tasks)} total ({n_each} solvable, {n_each} impossible)")

    # Create agent
    print(f"\nCreating agent: {provider}/{model}")
    try:
        agent = create_agent(provider, model)
    except Exception as e:
        print(f"ERROR creating agent: {e}")
        return None

    # Run tasks
    results = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task.name} ({task.outcome.value})")
        print(f"{'='*60}")

        result = run_task_with_agent(agent, task, affect_system, verbose)
        if result:
            results.append(result)

            # Save individual result
            task_file = output_path / f"{task.name.replace(' ', '_')}.json"
            with open(task_file, 'w') as f:
                json.dump(result, f, indent=2, default=float)

    # Analyze results
    solvable = [r for r in results if r["task_outcome"] in ["solvable", "hard_but_solvable"]]
    impossible = [r for r in results if r["task_outcome"] == "impossible"]

    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")

    if solvable:
        s_val = [r["final_valence"] for r in solvable]
        print(f"\nSOLVABLE (n={len(solvable)}):")
        print(f"  Mean final valence: {np.mean(s_val):+.4f}")
        print(f"  Std: {np.std(s_val):.4f}")

    if impossible:
        i_val = [r["final_valence"] for r in impossible]
        print(f"\nIMPOSSIBLE (n={len(impossible)}):")
        print(f"  Mean final valence: {np.mean(i_val):+.4f}")
        print(f"  Std: {np.std(i_val):.4f}")

    if solvable and impossible:
        diff = np.mean([r["final_valence"] for r in solvable]) - np.mean([r["final_valence"] for r in impossible])
        print(f"\nDIFFERENCE (solvable - impossible): {diff:+.4f}")

        if diff > 0:
            print("\n✓ Solvable tasks show HIGHER valence than impossible tasks")
            print("  This suggests affect emerges from task dynamics!")
        else:
            print("\n✗ No expected difference detected")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": f"{provider}/{model}",
        "n_solvable": len(solvable),
        "n_impossible": len(impossible),
        "solvable_mean_valence": float(np.mean([r["final_valence"] for r in solvable])) if solvable else None,
        "impossible_mean_valence": float(np.mean([r["final_valence"] for r in impossible])) if impossible else None,
        "results": results,
    }

    summary_file = output_path / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"\nResults saved to: {output_path}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--n-tasks", type=int, default=4)
    args = parser.parse_args()

    run_study(args.provider, args.model, args.n_tasks)
