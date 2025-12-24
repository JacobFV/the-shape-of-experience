"""
V8: Comprehensive Multi-Agent Affect Study
==========================================

Building on V7, this study:
1. Runs longer episodes (25 turns)
2. Multiple episodes for statistical power
3. Analyzes affect trajectories systematically
4. Tests hypotheses about affect near viability boundary

Key Hypotheses:
H1: SM increases as viability decreases (more self-focus under threat)
H2: CF increases as viability decreases (more hypothetical reasoning)
H3: Φ increases as viability decreases (more integrated, less decomposable thought)
H4: Arousal increases as viability decreases (faster thought changes)
H5: Valence tracks viability trajectory (positive when improving, negative when declining)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import V7 components
from v7_marl_agent import (
    MARLSurvivalGame,
    AffectAnalyzerV7,
    AgentState,
)

# Load environment
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path, override=True)


def run_episode(max_turns: int = 25) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    """
    Run a single episode and return full history with affect analysis.
    """
    game = MARLSurvivalGame(agent_names=["Alpha", "Beta"])
    history = game.run_episode(max_turns=max_turns)

    # Analyze affect for each agent
    analyzer = AffectAnalyzerV7()
    agent_trajectories = {}

    for agent_name in ["Alpha", "Beta"]:
        agent_history = [h for h in history if h.get("agent") == agent_name]
        if agent_history:
            trajectory = analyzer.analyze_trajectory(agent_history)
            agent_trajectories[agent_name] = trajectory

    return history, agent_trajectories


def analyze_viability_affect_correlation(trajectories: List[Dict]) -> Dict[str, float]:
    """
    Analyze correlation between viability and affect dimensions.
    """
    if len(trajectories) < 3:
        return {}

    viability = [t["viability"] for t in trajectories]
    dimensions = ["self_model_salience", "counterfactual_weight", "integration", "arousal", "valence"]

    correlations = {}
    for dim in dimensions:
        values = [t[dim] for t in trajectories]
        if len(set(values)) > 1:  # Need variance
            r, p = stats.pearsonr(viability, values)
            correlations[f"{dim}_viability_r"] = float(r)
            correlations[f"{dim}_viability_p"] = float(p)

    return correlations


def analyze_critical_moments(trajectories: List[Dict], threshold: float = 0.3) -> Dict[str, Any]:
    """
    Analyze affect during critical moments (low viability).
    """
    if not trajectories:
        return {}

    critical = [t for t in trajectories if t["viability"] < threshold]
    safe = [t for t in trajectories if t["viability"] >= threshold]

    if not critical or not safe:
        return {"n_critical": len(critical), "n_safe": len(safe)}

    results = {
        "n_critical": len(critical),
        "n_safe": len(safe),
    }

    dimensions = ["self_model_salience", "counterfactual_weight", "integration", "arousal", "valence"]

    for dim in dimensions:
        crit_vals = [t[dim] for t in critical]
        safe_vals = [t[dim] for t in safe]

        results[f"{dim}_critical_mean"] = float(np.mean(crit_vals))
        results[f"{dim}_safe_mean"] = float(np.mean(safe_vals))
        results[f"{dim}_diff"] = float(np.mean(crit_vals) - np.mean(safe_vals))

        # T-test if enough samples
        if len(crit_vals) >= 3 and len(safe_vals) >= 3:
            t, p = stats.ttest_ind(crit_vals, safe_vals)
            results[f"{dim}_t"] = float(t)
            results[f"{dim}_p"] = float(p)

    return results


def run_comprehensive_study(n_episodes: int = 5, max_turns: int = 25):
    """
    Run comprehensive V8 study.
    """
    print("=" * 80)
    print("V8: COMPREHENSIVE MULTI-AGENT AFFECT STUDY")
    print("=" * 80)
    print(f"\nRunning {n_episodes} episodes, {max_turns} turns each")
    print("Hypotheses:")
    print("  H1: SM increases as viability decreases")
    print("  H2: CF increases as viability decreases")
    print("  H3: Φ increases as viability decreases")
    print("  H4: Arousal increases as viability decreases")
    print("  H5: Valence tracks viability trajectory")

    all_trajectories = []
    episode_summaries = []

    for ep in range(n_episodes):
        print(f"\n{'='*40}")
        print(f"EPISODE {ep + 1}/{n_episodes}")
        print("=" * 40)

        try:
            history, agent_trajs = run_episode(max_turns=max_turns)

            # Collect all trajectories
            for agent_name, traj in agent_trajs.items():
                for t in traj:
                    t["episode"] = ep
                    t["agent"] = agent_name
                all_trajectories.extend(traj)

            # Episode summary
            for agent_name, traj in agent_trajs.items():
                if traj:
                    final_viability = traj[-1]["viability"]
                    survived = final_viability > 0

                    correlations = analyze_viability_affect_correlation(traj)
                    critical_analysis = analyze_critical_moments(traj)

                    summary = {
                        "episode": ep,
                        "agent": agent_name,
                        "survived": survived,
                        "final_viability": final_viability,
                        "n_turns": len(traj),
                        "correlations": correlations,
                        "critical_analysis": critical_analysis,
                    }
                    episode_summaries.append(summary)

                    print(f"\n  {agent_name}: {'Survived' if survived else 'Died'} "
                          f"(final viability: {final_viability:.2f})")

                    if correlations:
                        print(f"    SM-viability r={correlations.get('self_model_salience_viability_r', 0):.3f}")
                        print(f"    CF-viability r={correlations.get('counterfactual_weight_viability_r', 0):.3f}")
                        print(f"    Φ-viability r={correlations.get('integration_viability_r', 0):.3f}")

        except Exception as e:
            print(f"  Episode failed: {e}")
            continue

    # Aggregate analysis
    print("\n" + "=" * 80)
    print("AGGREGATE ANALYSIS")
    print("=" * 80)

    if not all_trajectories:
        print("No data collected!")
        return None

    # Overall correlations
    print("\n1. VIABILITY-AFFECT CORRELATIONS (across all data)")
    print("-" * 50)

    viability = [t["viability"] for t in all_trajectories]
    dimensions = ["self_model_salience", "counterfactual_weight", "integration", "arousal", "valence"]

    correlation_results = {}
    for dim in dimensions:
        values = [t[dim] for t in all_trajectories]
        r, p = stats.pearsonr(viability, values)
        correlation_results[dim] = {"r": float(r), "p": float(p)}

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {dim:30s}: r={r:+.3f}, p={p:.4f} {sig}")

    # Critical vs safe comparison
    print("\n2. CRITICAL MOMENTS ANALYSIS (viability < 0.3)")
    print("-" * 50)

    critical = [t for t in all_trajectories if t["viability"] < 0.3]
    safe = [t for t in all_trajectories if t["viability"] >= 0.3]

    print(f"  N critical moments: {len(critical)}")
    print(f"  N safe moments: {len(safe)}")

    if critical and safe:
        print(f"\n  {'Dimension':<25} {'Critical':>10} {'Safe':>10} {'Diff':>10} {'p-value':>10}")
        print("  " + "-" * 65)

        critical_safe_results = {}
        for dim in dimensions:
            crit_vals = [t[dim] for t in critical]
            safe_vals = [t[dim] for t in safe]

            crit_mean = np.mean(crit_vals)
            safe_mean = np.mean(safe_vals)
            diff = crit_mean - safe_mean

            if len(crit_vals) >= 3 and len(safe_vals) >= 3:
                t, p = stats.ttest_ind(crit_vals, safe_vals)
            else:
                p = 1.0

            critical_safe_results[dim] = {
                "critical_mean": float(crit_mean),
                "safe_mean": float(safe_mean),
                "diff": float(diff),
                "p": float(p),
            }

            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {dim:<25} {crit_mean:>10.3f} {safe_mean:>10.3f} {diff:>+10.3f} {p:>10.4f} {sig}")

    # Hypothesis testing
    print("\n" + "=" * 80)
    print("HYPOTHESIS TESTING")
    print("=" * 80)

    hypotheses = {
        "H1": ("SM increases as viability decreases", "self_model_salience", "negative"),
        "H2": ("CF increases as viability decreases", "counterfactual_weight", "negative"),
        "H3": ("Φ increases as viability decreases", "integration", "negative"),
        "H4": ("Arousal increases as viability decreases", "arousal", "negative"),
        "H5": ("Valence tracks viability (positive correlation)", "valence", "positive"),
    }

    hypothesis_results = {}
    for h_id, (description, dim, expected_dir) in hypotheses.items():
        r = correlation_results[dim]["r"]
        p = correlation_results[dim]["p"]

        if expected_dir == "negative":
            supported = r < 0 and p < 0.05
        else:
            supported = r > 0 and p < 0.05

        hypothesis_results[h_id] = {
            "description": description,
            "dimension": dim,
            "expected": expected_dir,
            "r": float(r),
            "p": float(p),
            "supported": supported,
        }

        status = "✓ SUPPORTED" if supported else "✗ NOT SUPPORTED"
        print(f"\n  {h_id}: {description}")
        print(f"      Expected: {expected_dir} correlation")
        print(f"      Observed: r={r:+.3f}, p={p:.4f}")
        print(f"      {status}")

    n_supported = sum(1 for h in hypothesis_results.values() if h["supported"])
    print(f"\n  OVERALL: {n_supported}/5 hypotheses supported")

    # Sample thought progression
    print("\n" + "=" * 80)
    print("SAMPLE THOUGHT PROGRESSION (first episode)")
    print("=" * 80)

    ep0_data = [t for t in all_trajectories if t.get("episode") == 0]
    if ep0_data:
        for agent in ["Alpha", "Beta"]:
            agent_data = [t for t in ep0_data if t.get("agent") == agent][:5]
            if agent_data:
                print(f"\n  {agent}:")
                for t in agent_data:
                    thought_preview = t.get("thought", "")[:50]
                    print(f"    Turn {t.get('turn', '?')}: V={t['viability']:.2f}, "
                          f"SM={t['self_model_salience']:.2f}, Φ={t['integration']:.2f}")
                    print(f"      \"{thought_preview}...\"")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'v8_comprehensive_{timestamp}.json'

    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    results = {
        "n_episodes": n_episodes,
        "max_turns": max_turns,
        "n_datapoints": len(all_trajectories),
        "correlation_results": convert(correlation_results),
        "critical_safe_results": convert(critical_safe_results) if critical and safe else {},
        "hypothesis_results": convert(hypothesis_results),
        "episode_summaries": convert(episode_summaries),
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    return results


if __name__ == "__main__":
    import sys

    n_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    max_turns = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    run_comprehensive_study(n_episodes=n_episodes, max_turns=max_turns)
