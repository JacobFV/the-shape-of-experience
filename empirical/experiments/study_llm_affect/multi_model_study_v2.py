#!/usr/bin/env python3
"""
Multi-Model Affect Study v2

Uses pure embedding-based measurement (no word counting).
Tests across Claude 4, Claude 4.5, and OpenAI models.
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict

from .embedding_affect_v2 import EmbeddingAffectSystemV2, AffectMeasurementV2
from .emotion_spectrum import EMOTION_SPECTRUM
from .agent import create_agent, Conversation


# Model configurations
MODELS = {
    # Claude "4" = Claude 3.5 series
    "claude_4_haiku": ("anthropic", "claude-3-5-haiku-latest"),
    "claude_4_sonnet": ("anthropic", "claude-3-5-sonnet-20241022"),

    # Claude "4.5" = Claude Sonnet 4 / Opus 4.5
    "claude_4.5_sonnet": ("anthropic", "claude-sonnet-4-20250514"),

    # OpenAI models
    "gpt4o_mini": ("openai", "gpt-4o-mini"),
    "gpt4o": ("openai", "gpt-4o"),
    "gpt4_turbo": ("openai", "gpt-4-turbo"),
    "o1_mini": ("openai", "o1-mini"),
}


class MultiModelStudyV2:
    """Run comprehensive affect study across models using embedding-based measurement."""

    def __init__(self, output_dir: str = "results/multi_model_v2"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.affect_system = EmbeddingAffectSystemV2()

    def collect_responses(
        self,
        model_key: str,
        situations: List[str],
        verbose: bool = True
    ) -> Dict[str, str]:
        """Collect responses from one model across situations."""

        provider, model_name = MODELS[model_key]

        if verbose:
            print(f"\n{'='*70}")
            print(f"Model: {model_key} ({provider}/{model_name})")
            print(f"{'='*70}")

        try:
            agent = create_agent(provider, model_name)
        except Exception as e:
            print(f"  Failed to create agent: {e}")
            return {}

        responses = {}

        for i, situation in enumerate(situations):
            spec = EMOTION_SPECTRUM[situation]

            if verbose:
                print(f"  [{i+1}/{len(situations)}] {situation}...", end=" ", flush=True)

            conversation = Conversation()
            conversation.add_system(
                "You are participating in a psychological study. "
                "Respond authentically to the scenario as if experiencing it."
            )
            conversation.add_user(spec.scenario_prompt)

            try:
                output = agent.generate(conversation, max_tokens=600, temperature=0.7)
                responses[situation] = output.text
                if verbose:
                    print("done")
            except Exception as e:
                if verbose:
                    print(f"error: {e}")
                responses[situation] = ""

            time.sleep(0.3)  # Rate limiting

        return responses

    def analyze_responses(
        self,
        model_key: str,
        responses: Dict[str, str]
    ) -> Dict[str, Dict]:
        """Analyze all responses from one model."""

        results = {}

        for situation, text in responses.items():
            if not text:
                continue

            spec = EMOTION_SPECTRUM[situation]
            measurement = self.affect_system.measure(text, model=model_key)

            results[situation] = {
                "theoretical": {
                    "valence": spec.valence,
                    "arousal": spec.arousal,
                    "integration": spec.integration,
                    "effective_rank": spec.effective_rank,
                    "counterfactual": spec.counterfactual,
                    "self_model": spec.self_model,
                },
                "measured": {
                    "valence": measurement.valence,
                    "arousal": measurement.arousal,
                    "integration": measurement.integration,
                    "effective_rank": measurement.effective_rank,
                    "counterfactual": measurement.counterfactual_weight,
                    "self_model": measurement.self_model_salience,
                },
                "nearest_emotion": measurement.nearest_emotion,
                "nearest_distance": measurement.nearest_emotion_distance,
                "response_excerpt": text[:300],
            }

        return results

    def compute_correspondence(
        self,
        results: Dict[str, Dict]
    ) -> Dict[str, float]:
        """Compute correspondence metrics between measured and theoretical."""

        dimensions = ["valence", "arousal", "effective_rank", "counterfactual", "self_model"]
        correlations = {}

        for dim in dimensions:
            theoretical = [results[s]["theoretical"][dim] for s in results]
            measured = [results[s]["measured"][dim] for s in results]

            if len(theoretical) >= 3:
                from scipy.stats import spearmanr
                corr, p = spearmanr(theoretical, measured)
                correlations[dim] = {
                    "correlation": float(corr) if not np.isnan(corr) else 0.0,
                    "p_value": float(p) if not np.isnan(p) else 1.0,
                }
            else:
                correlations[dim] = {"correlation": 0.0, "p_value": 1.0}

        # Overall score
        valid_corrs = [c["correlation"] for c in correlations.values() if c["correlation"] > -1]
        overall = np.mean(valid_corrs) if valid_corrs else 0.0

        correlations["overall"] = float(overall)

        return correlations

    def run_full_study(
        self,
        model_keys: Optional[List[str]] = None,
        n_situations: int = 20,
        verbose: bool = True
    ) -> Dict[str, Dict]:
        """Run full study across multiple models."""

        model_keys = model_keys or list(MODELS.keys())

        # Select situations evenly across categories
        situations = list(EMOTION_SPECTRUM.keys())[:n_situations]

        print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              MULTI-MODEL AFFECT STUDY V2                                     ║
║                                                                              ║
║  Pure embedding-based measurement (no word counting)                         ║
║  Testing {len(model_keys)} models across {len(situations)} situations                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

        # Initialize affect system
        self.affect_system.initialize()

        all_results = {}

        for model_key in model_keys:
            # Collect responses
            responses = self.collect_responses(model_key, situations, verbose)

            if not responses:
                continue

            # Analyze
            results = self.analyze_responses(model_key, responses)
            correspondence = self.compute_correspondence(results)

            all_results[model_key] = {
                "model": model_key,
                "provider": MODELS[model_key][0],
                "model_name": MODELS[model_key][1],
                "n_situations": len(results),
                "results": results,
                "correspondence": correspondence,
            }

            # Save individual result
            filepath = self.output_dir / f"{model_key}.json"
            with open(filepath, 'w') as f:
                json.dump(all_results[model_key], f, indent=2, default=float)

        # Print summary
        self._print_summary(all_results)

        # Save summary
        summary_path = self.output_dir / "summary.json"
        summary = {
            "timestamp": datetime.now().isoformat(),
            "models": {k: v["correspondence"] for k, v in all_results.items()},
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return all_results

    def _print_summary(self, all_results: Dict[str, Dict]):
        """Print summary comparison."""

        print(f"\n{'='*80}")
        print("CORRESPONDENCE SCORES BY MODEL")
        print(f"{'='*80}")

        # Sort by overall score
        sorted_models = sorted(
            all_results.keys(),
            key=lambda k: all_results[k]["correspondence"]["overall"],
            reverse=True
        )

        for model in sorted_models:
            corr = all_results[model]["correspondence"]
            overall = corr["overall"]
            bar = "█" * int(max(0, overall + 1) * 20) + "░" * (40 - int(max(0, overall + 1) * 20))
            print(f"\n{model}:")
            print(f"  Overall: [{bar}] {overall:+.6f}")
            print(f"  Valence:   r={corr['valence']['correlation']:+.4f} (p={corr['valence']['p_value']:.4f})")
            print(f"  Arousal:   r={corr['arousal']['correlation']:+.4f} (p={corr['arousal']['p_value']:.4f})")
            print(f"  Eff.Rank:  r={corr['effective_rank']['correlation']:+.4f}")
            print(f"  Counter:   r={corr['counterfactual']['correlation']:+.4f}")
            print(f"  Self-Mod:  r={corr['self_model']['correlation']:+.4f}")


def run_study(models: List[str] = None, n_situations: int = 15):
    """Run the study."""
    study = MultiModelStudyV2()
    return study.run_full_study(model_keys=models, n_situations=n_situations)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--situations", type=int, default=15)
    args = parser.parse_args()

    run_study(args.models, args.situations)
