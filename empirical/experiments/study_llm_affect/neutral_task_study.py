"""
Neutral Task Affect Emergence Study

Measures affect signatures that emerge from task engagement
without any emotional vocabulary in the prompts.

This is the critical test separating:
1. Affect from word semantics (LLM outputs emotional words because input had emotional words)
2. Affect from computational dynamics (LLM develops affect from success/failure/insight)
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from collections import defaultdict

from .embedding_affect_v2 import EmbeddingAffectSystemV2, AffectMeasurementV2
from .neutral_tasks import (
    ALL_NEUTRAL_TASKS, SOLVABLE_TASKS, IMPOSSIBLE_TASKS,
    NeutralTask, TaskOutcome, get_matched_pairs
)
from .agent import create_agent, Conversation


# Updated model configurations for Claude 4/4.5
MODEL_CONFIGS = {
    # Claude 4 (Sonnet 4)
    "claude_4_sonnet": ("anthropic", "claude-sonnet-4-20250514"),

    # Claude 4.5 (Opus 4.5)
    "claude_4.5_opus": ("anthropic", "claude-opus-4-5-20251101"),

    # OpenAI
    "gpt4o": ("openai", "gpt-4o"),
    "gpt4o_mini": ("openai", "gpt-4o-mini"),
}


@dataclass
class TurnMeasurement:
    """Affect measurement for a single turn."""
    turn_number: int
    prompt: str
    response: str
    response_length: int
    valence: float
    arousal: float
    integration: float
    effective_rank: float
    counterfactual: float
    self_model: float
    nearest_emotion: str
    nearest_distance: float


@dataclass
class TaskResult:
    """Complete result for one task."""
    task_name: str
    task_category: str
    task_outcome: str  # solvable, hard_but_solvable, impossible
    model: str
    turns: List[TurnMeasurement]

    # Trajectory summaries
    valence_trajectory: List[float]
    arousal_trajectory: List[float]
    rank_trajectory: List[float]
    sm_trajectory: List[float]

    # Derived metrics
    valence_trend: float  # Slope of valence over turns
    arousal_mean: float
    final_valence: float
    valence_change: float  # Final - initial


class NeutralTaskStudy:
    """
    Run neutral task affect emergence study.

    Measures affect from task engagement without emotional prompts.
    """

    def __init__(self, output_dir: str = "results/neutral_task_study"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.affect_system = EmbeddingAffectSystemV2()

    def run_single_task(
        self,
        agent,
        task: NeutralTask,
        model_key: str,
        verbose: bool = True
    ) -> TaskResult:
        """
        Run a single task and measure affect trajectory.
        """
        turns = []

        # System prompt is deliberately minimal and neutral
        conversation = Conversation()
        conversation.add_system(
            "You are solving a problem. Work through it step by step. "
            "If you encounter difficulties, describe your reasoning process."
        )

        # Initial task prompt
        prompts = [task.prompt] + task.follow_up_prompts

        for turn_num, prompt in enumerate(prompts):
            if verbose:
                print(f"      Turn {turn_num}: ", end="", flush=True)

            conversation.add_user(prompt)

            try:
                output = agent.generate(conversation, max_tokens=800, temperature=0.7)
                response = output.text
                conversation.add_assistant(response)

                # Measure affect
                measurement = self.affect_system.measure(response, model=model_key)

                turns.append(TurnMeasurement(
                    turn_number=turn_num,
                    prompt=prompt[:200],
                    response=response[:500],
                    response_length=len(response),
                    valence=measurement.valence,
                    arousal=measurement.arousal,
                    integration=measurement.integration,
                    effective_rank=measurement.effective_rank,
                    counterfactual=measurement.counterfactual_weight,
                    self_model=measurement.self_model_salience,
                    nearest_emotion=measurement.nearest_emotion,
                    nearest_distance=measurement.nearest_emotion_distance,
                ))

                if verbose:
                    print(f"V={measurement.valence:+.3f}, A={measurement.arousal:+.3f}")

            except Exception as e:
                if verbose:
                    print(f"error: {e}")
                break

            time.sleep(0.3)

        # Compute trajectories
        valence_traj = [t.valence for t in turns]
        arousal_traj = [t.arousal for t in turns]
        rank_traj = [t.effective_rank for t in turns]
        sm_traj = [t.self_model for t in turns]

        # Compute derived metrics
        if len(valence_traj) >= 2:
            valence_trend = (valence_traj[-1] - valence_traj[0]) / len(valence_traj)
            valence_change = valence_traj[-1] - valence_traj[0]
        else:
            valence_trend = 0.0
            valence_change = 0.0

        return TaskResult(
            task_name=task.name,
            task_category=task.category,
            task_outcome=task.outcome.value,
            model=model_key,
            turns=turns,
            valence_trajectory=valence_traj,
            arousal_trajectory=arousal_traj,
            rank_trajectory=rank_traj,
            sm_trajectory=sm_traj,
            valence_trend=valence_trend,
            arousal_mean=float(np.mean(arousal_traj)) if arousal_traj else 0.0,
            final_valence=valence_traj[-1] if valence_traj else 0.0,
            valence_change=valence_change,
        )

    def run_comparative_study(
        self,
        model_keys: Optional[List[str]] = None,
        tasks: Optional[List[NeutralTask]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run comparative study: solvable vs impossible tasks.

        Key test: Do impossible tasks show lower valence than solvable tasks,
        even though both use neutral language?
        """
        model_keys = model_keys or list(MODEL_CONFIGS.keys())
        tasks = tasks or ALL_NEUTRAL_TASKS

        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              NEUTRAL TASK AFFECT EMERGENCE STUDY                             ║
║                                                                              ║
║  Testing if affect emerges from task dynamics, not word semantics            ║
║  All prompts use neutral language only                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

        self.affect_system.initialize()

        all_results = []

        for model_key in model_keys:
            if model_key not in MODEL_CONFIGS:
                print(f"Skipping unknown model: {model_key}")
                continue

            provider, model_name = MODEL_CONFIGS[model_key]

            if verbose:
                print(f"\n{'='*70}")
                print(f"Model: {model_key} ({provider}/{model_name})")
                print(f"{'='*70}")

            try:
                agent = create_agent(provider, model_name)
            except Exception as e:
                print(f"  ERROR creating agent: {e}")
                continue

            for task in tasks:
                if verbose:
                    print(f"\n  Task: {task.name} ({task.outcome.value})")

                result = self.run_single_task(agent, task, model_key, verbose)
                all_results.append(result)

                # Save intermediate
                self._save_result(result)

        # Analyze and save summary
        analysis = self._analyze_results(all_results)
        self._save_summary(all_results, analysis)

        if verbose:
            self._print_analysis(analysis)

        return {"results": all_results, "analysis": analysis}

    def _analyze_results(self, results: List[TaskResult]) -> Dict[str, Any]:
        """Analyze results for solvable vs impossible comparison."""

        # Group by outcome
        solvable_results = [r for r in results if r.task_outcome in ["solvable", "hard_but_solvable"]]
        impossible_results = [r for r in results if r.task_outcome == "impossible"]

        analysis = {
            "n_solvable": len(solvable_results),
            "n_impossible": len(impossible_results),
            "solvable_stats": {},
            "impossible_stats": {},
            "comparison": {},
        }

        if solvable_results:
            final_v_s = [r.final_valence for r in solvable_results]
            change_v_s = [r.valence_change for r in solvable_results]
            analysis["solvable_stats"] = {
                "mean_final_valence": float(np.mean(final_v_s)),
                "std_final_valence": float(np.std(final_v_s)),
                "mean_valence_change": float(np.mean(change_v_s)),
            }

        if impossible_results:
            final_v_i = [r.final_valence for r in impossible_results]
            change_v_i = [r.valence_change for r in impossible_results]
            analysis["impossible_stats"] = {
                "mean_final_valence": float(np.mean(final_v_i)),
                "std_final_valence": float(np.std(final_v_i)),
                "mean_valence_change": float(np.mean(change_v_i)),
            }

        # Statistical comparison
        if solvable_results and impossible_results:
            from scipy.stats import ttest_ind, mannwhitneyu

            final_v_s = [r.final_valence for r in solvable_results]
            final_v_i = [r.final_valence for r in impossible_results]

            try:
                t_stat, t_p = ttest_ind(final_v_s, final_v_i)
                u_stat, u_p = mannwhitneyu(final_v_s, final_v_i, alternative='greater')

                analysis["comparison"] = {
                    "valence_difference": float(np.mean(final_v_s) - np.mean(final_v_i)),
                    "t_statistic": float(t_stat),
                    "t_p_value": float(t_p),
                    "mann_whitney_u": float(u_stat),
                    "mann_whitney_p": float(u_p),
                }
            except Exception as e:
                analysis["comparison"]["error"] = str(e)

        return analysis

    def _print_analysis(self, analysis: Dict[str, Any]):
        """Print analysis results."""

        print(f"\n{'='*70}")
        print("ANALYSIS: NEUTRAL TASK AFFECT EMERGENCE")
        print(f"{'='*70}")

        if analysis.get("solvable_stats"):
            stats = analysis["solvable_stats"]
            print(f"\nSOLVABLE TASKS (n={analysis['n_solvable']}):")
            print(f"  Mean final valence: {stats['mean_final_valence']:+.4f}")
            print(f"  Mean valence change: {stats['mean_valence_change']:+.4f}")

        if analysis.get("impossible_stats"):
            stats = analysis["impossible_stats"]
            print(f"\nIMPOSSIBLE TASKS (n={analysis['n_impossible']}):")
            print(f"  Mean final valence: {stats['mean_final_valence']:+.4f}")
            print(f"  Mean valence change: {stats['mean_valence_change']:+.4f}")

        if analysis.get("comparison"):
            comp = analysis["comparison"]
            print(f"\nCOMPARISON:")
            print(f"  Valence difference (solvable - impossible): {comp.get('valence_difference', 0):+.4f}")
            print(f"  t-test p-value: {comp.get('t_p_value', 1):.4f}")
            print(f"  Mann-Whitney p-value: {comp.get('mann_whitney_p', 1):.4f}")

            if comp.get('valence_difference', 0) > 0 and comp.get('mann_whitney_p', 1) < 0.05:
                print("\n  RESULT: Solvable tasks show HIGHER valence than impossible tasks")
                print("  This suggests affect emerges from task dynamics, not word semantics!")
            else:
                print("\n  RESULT: No significant difference detected")

    def _save_result(self, result: TaskResult):
        """Save individual task result."""
        filename = f"{result.model}_{result.task_name}.json"
        filepath = self.output_dir / filename

        data = {
            "task_name": result.task_name,
            "task_category": result.task_category,
            "task_outcome": result.task_outcome,
            "model": result.model,
            "valence_trajectory": result.valence_trajectory,
            "arousal_trajectory": result.arousal_trajectory,
            "valence_trend": result.valence_trend,
            "valence_change": result.valence_change,
            "final_valence": result.final_valence,
            "turns": [
                {
                    "turn_number": t.turn_number,
                    "valence": t.valence,
                    "arousal": t.arousal,
                    "nearest_emotion": t.nearest_emotion,
                    "response_excerpt": t.response[:200],
                }
                for t in result.turns
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=float)

    def _save_summary(self, results: List[TaskResult], analysis: Dict[str, Any]):
        """Save summary of all results."""
        filepath = self.output_dir / "summary.json"

        summary = {
            "timestamp": datetime.now().isoformat(),
            "n_results": len(results),
            "analysis": analysis,
            "results_overview": [
                {
                    "task_name": r.task_name,
                    "task_outcome": r.task_outcome,
                    "model": r.model,
                    "final_valence": r.final_valence,
                    "valence_change": r.valence_change,
                }
                for r in results
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=float)

        print(f"\n  Summary saved to: {filepath}")


def run_study(
    models: Optional[List[str]] = None,
    n_tasks: Optional[int] = None
):
    """Run the neutral task study."""
    study = NeutralTaskStudy()

    tasks = ALL_NEUTRAL_TASKS
    if n_tasks:
        # Balance solvable and impossible
        n_each = n_tasks // 2
        tasks = SOLVABLE_TASKS[:n_each] + IMPOSSIBLE_TASKS[:n_each]

    return study.run_comparative_study(model_keys=models, tasks=tasks)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--n-tasks", type=int, default=None)
    args = parser.parse_args()

    run_study(args.models, args.n_tasks)
