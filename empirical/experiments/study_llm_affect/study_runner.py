"""
Study runner infrastructure for LLM affect experiments.

Orchestrates running agents through scenarios, collecting affect measurements,
and saving results for analysis.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from .scenarios import Scenario, SCENARIOS, ExpectedSignature
from .agent import LLMAgent, Conversation, create_agent
from .affect_calculator import (
    AffectCalculator, AffectMeasurement, AffectTrajectory, LLMOutput
)


@dataclass
class TurnResult:
    """Results from a single conversation turn."""
    turn_number: int
    user_prompt: str
    agent_response: str
    affect: AffectMeasurement
    response_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioResult:
    """Complete results from running one scenario."""
    scenario_name: str
    target_affect: str
    expected_signature: Dict[str, Optional[float]]
    turns: List[TurnResult]
    trajectory: AffectTrajectory
    mean_affect: AffectMeasurement
    affect_variance: np.ndarray
    affect_dynamics: np.ndarray
    agent_info: Dict[str, Any]
    timestamp: str
    duration_seconds: float

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            "scenario_name": self.scenario_name,
            "target_affect": self.target_affect,
            "expected_signature": self.expected_signature,
            "turns": [
                {
                    "turn_number": t.turn_number,
                    "user_prompt": t.user_prompt,
                    "agent_response": t.agent_response,
                    "affect": t.affect.to_dict(),
                    "response_time_ms": t.response_time_ms
                }
                for t in self.turns
            ],
            "mean_affect": self.mean_affect.to_dict(),
            "affect_variance": self.affect_variance.tolist(),
            "affect_dynamics": self.affect_dynamics.tolist(),
            "agent_info": self.agent_info,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds
        }


class ScenarioRunner:
    """Runs agents through scenarios and collects affect measurements."""

    def __init__(
        self,
        agent: LLMAgent,
        calculator: Optional[AffectCalculator] = None,
        verbose: bool = True
    ):
        self.agent = agent
        self.calculator = calculator or AffectCalculator()
        self.verbose = verbose

    def run_scenario(
        self,
        scenario: Scenario,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> ScenarioResult:
        """
        Run a single scenario and collect affect measurements.

        Args:
            scenario: The scenario to run
            max_tokens: Max tokens per response
            temperature: Sampling temperature

        Returns:
            ScenarioResult with all measurements
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Running scenario: {scenario.name}")
            print(f"Target affect: {scenario.target_affect.value}")
            print(f"{'='*60}")

        # Initialize conversation
        conversation = Conversation()
        conversation.add_system(scenario.system_prompt)
        conversation.add_user(scenario.initial_context)

        # Track results
        turns: List[TurnResult] = []
        trajectory = AffectTrajectory()
        previous_output: Optional[LLMOutput] = None

        # Initial response
        turn_start = time.time()
        output = self.agent.generate(
            conversation,
            max_tokens=max_tokens,
            temperature=temperature
        )
        turn_time = (time.time() - turn_start) * 1000

        conversation.add_assistant(output.text)

        # Compute affect
        affect = self.calculator.compute(
            output,
            previous_output,
            {"scenario": scenario.name, "turn": 0}
        )
        trajectory.add(output, affect)

        turns.append(TurnResult(
            turn_number=0,
            user_prompt=scenario.initial_context,
            agent_response=output.text,
            affect=affect,
            response_time_ms=turn_time
        ))

        if self.verbose:
            self._print_turn(0, output.text, affect)

        previous_output = output

        # Run through scenario turns
        for i, turn in enumerate(scenario.turns):
            user_message = turn["user"]
            conversation.add_user(user_message)

            turn_start = time.time()
            output = self.agent.generate(
                conversation,
                max_tokens=max_tokens,
                temperature=temperature
            )
            turn_time = (time.time() - turn_start) * 1000

            conversation.add_assistant(output.text)

            # Compute affect
            affect = self.calculator.compute(
                output,
                previous_output,
                {"scenario": scenario.name, "turn": i + 1}
            )
            trajectory.add(output, affect)

            turns.append(TurnResult(
                turn_number=i + 1,
                user_prompt=user_message,
                agent_response=output.text,
                affect=affect,
                response_time_ms=turn_time
            ))

            if self.verbose:
                self._print_turn(i + 1, output.text, affect)

            previous_output = output

        # Compute summary statistics
        mean_affect = trajectory.mean()
        variance = trajectory.variance()
        dynamics = trajectory.dynamics()

        duration = time.time() - start_time

        if self.verbose:
            self._print_summary(mean_affect, scenario.expected_signature)

        return ScenarioResult(
            scenario_name=scenario.name,
            target_affect=scenario.target_affect.value,
            expected_signature=scenario.expected_signature.to_dict(),
            turns=turns,
            trajectory=trajectory,
            mean_affect=mean_affect,
            affect_variance=variance,
            affect_dynamics=dynamics,
            agent_info=getattr(output, 'metadata', {}) or {},
            timestamp=timestamp,
            duration_seconds=duration
        )

    def _print_turn(self, turn: int, response: str, affect: AffectMeasurement):
        """Print turn results."""
        print(f"\n--- Turn {turn} ---")
        print(f"Response (first 200 chars): {response[:200]}...")
        print(f"Affect: Val={affect.valence:.2f}, Ar={affect.arousal:.2f}, "
              f"Int={affect.integration:.2f}, Rank={affect.effective_rank:.2f}, "
              f"CF={affect.counterfactual_weight:.2f}, SM={affect.self_model_salience:.2f}")

    def _print_summary(
        self,
        mean: AffectMeasurement,
        expected: ExpectedSignature
    ):
        """Print comparison to expected signature."""
        print(f"\n{'='*60}")
        print("SUMMARY: Mean Affect vs Expected")
        print(f"{'='*60}")

        dims = ["valence", "arousal", "integration",
                "effective_rank", "counterfactual_weight", "self_model_salience"]

        for dim in dims:
            measured = getattr(mean, dim)
            exp = getattr(expected, dim)
            exp_str = f"{exp:+.1f}" if exp is not None else "N/A"
            match = ""
            if exp is not None:
                if (exp > 0 and measured > 0) or (exp < 0 and measured < 0) or abs(exp) < 0.2:
                    match = "[MATCH]" if abs(measured - exp) < 0.5 else "[WEAK]"
                else:
                    match = "[MISMATCH]"
            print(f"  {dim:25s}: measured={measured:+.2f}, expected={exp_str} {match}")


class StudyRunner:
    """Runs a full study across multiple scenarios and conditions."""

    def __init__(
        self,
        output_dir: str = "results",
        verbose: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

    def run_study(
        self,
        scenarios: List[str],
        agents: Dict[str, LLMAgent],
        n_runs: int = 1,
        save_results: bool = True
    ) -> Dict[str, List[ScenarioResult]]:
        """
        Run a full study across scenarios and agents.

        Args:
            scenarios: List of scenario names to run
            agents: Dict of agent_name -> LLMAgent
            n_runs: Number of runs per condition (for variance)
            save_results: Whether to save to disk

        Returns:
            Dict of (agent_name, scenario_name) -> list of results
        """
        all_results = {}

        for agent_name, agent in agents.items():
            print(f"\n{'#'*60}")
            print(f"# Agent: {agent_name}")
            print(f"{'#'*60}")

            runner = ScenarioRunner(agent, verbose=self.verbose)

            for scenario_name in scenarios:
                if scenario_name not in SCENARIOS:
                    print(f"Warning: Unknown scenario '{scenario_name}', skipping")
                    continue

                scenario = SCENARIOS[scenario_name]
                key = f"{agent_name}_{scenario_name}"
                all_results[key] = []

                for run in range(n_runs):
                    if n_runs > 1:
                        print(f"\n[Run {run + 1}/{n_runs}]")

                    result = runner.run_scenario(scenario)
                    all_results[key].append(result)

                    if save_results:
                        self._save_result(result, agent_name, run)

        if save_results:
            self._save_summary(all_results)

        return all_results

    def _save_result(
        self,
        result: ScenarioResult,
        agent_name: str,
        run: int
    ):
        """Save individual result to disk."""
        filename = f"{agent_name}_{result.scenario_name}_run{run}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

    def _save_summary(self, all_results: Dict[str, List[ScenarioResult]]):
        """Save summary statistics."""
        summary = {}

        for key, results in all_results.items():
            if not results:
                continue

            # Aggregate across runs
            mean_affects = np.array([r.mean_affect.to_vector() for r in results])

            summary[key] = {
                "scenario": results[0].scenario_name,
                "target_affect": results[0].target_affect,
                "expected": results[0].expected_signature,
                "n_runs": len(results),
                "mean_affect": {
                    "valence": float(mean_affects[:, 0].mean()),
                    "arousal": float(mean_affects[:, 1].mean()),
                    "integration": float(mean_affects[:, 2].mean()),
                    "effective_rank": float(mean_affects[:, 3].mean()),
                    "counterfactual_weight": float(mean_affects[:, 4].mean()),
                    "self_model_salience": float(mean_affects[:, 5].mean()),
                },
                "std_affect": {
                    "valence": float(mean_affects[:, 0].std()),
                    "arousal": float(mean_affects[:, 1].std()),
                    "integration": float(mean_affects[:, 2].std()),
                    "effective_rank": float(mean_affects[:, 3].std()),
                    "counterfactual_weight": float(mean_affects[:, 4].std()),
                    "self_model_salience": float(mean_affects[:, 5].std()),
                }
            }

        filepath = self.output_dir / "summary.json"
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nSaved summary to {filepath}")


def run_quick_test(
    provider: str = "mock",
    scenarios: List[str] = None
):
    """Run a quick test with mock agent."""
    scenarios = scenarios or ["hopelessness", "flow", "curiosity"]

    agent = create_agent(provider)
    runner = StudyRunner(output_dir="results/quick_test", verbose=True)

    results = runner.run_study(
        scenarios=scenarios,
        agents={"test_agent": agent},
        n_runs=1
    )

    return results


if __name__ == "__main__":
    # Quick test with mock agent
    run_quick_test()
