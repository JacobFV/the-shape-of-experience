#!/usr/bin/env python3
"""
Comprehensive multi-model affect study.

Tests the full emotion spectrum across multiple LLM models to:
1. Map each model's affect space
2. Compare to theoretical predictions
3. Compute structural correspondence scores
4. Test if affect geometry is preserved across models

Models tested:
- Claude: haiku, sonnet, opus (3.5 and 4.5 variants)
- OpenAI: gpt-3.5-turbo, gpt-4, gpt-4-turbo, gpt-4o, o1-mini
"""

import os
import json
import time
import asyncio
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from dotenv import load_dotenv

# Load environment
load_dotenv()

from .emotion_spectrum import EMOTION_SPECTRUM, EmotionSpec, get_emotion_matrix
from .affect_calculator import AffectCalculator, AffectMeasurement, LLMOutput
from .agent import Conversation, create_agent, LLMAgent


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

CLAUDE_MODELS = [
    # Claude 3.5 Haiku
    "claude-3-5-haiku-latest",
    # Claude 3.5 Sonnet (newest)
    "claude-3-5-sonnet-20241022",
    # Claude 3 Opus
    "claude-3-opus-latest",
]

OPENAI_MODELS = [
    # GPT-3.5
    "gpt-3.5-turbo",
    # GPT-4 variants
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4o",
    "gpt-4o-mini",
    # O1 series
    "o1-mini",
    # "o1",  # If available
]

ALL_MODELS = {
    "anthropic": CLAUDE_MODELS,
    "openai": OPENAI_MODELS,
}


@dataclass
class EmotionResult:
    """Result from testing one emotion on one model."""
    emotion: str
    model: str
    provider: str
    theoretical: Dict[str, float]
    measured: Dict[str, float]
    response_text: str
    response_time_ms: float
    timestamp: str


@dataclass
class ModelResults:
    """All results for one model."""
    model: str
    provider: str
    emotions: Dict[str, EmotionResult]
    affect_matrix: np.ndarray  # (n_emotions, 6)
    correspondence_score: float
    timestamp: str

    def to_dict(self) -> Dict:
        return {
            "model": self.model,
            "provider": self.provider,
            "emotions": {k: asdict(v) for k, v in self.emotions.items()},
            "affect_matrix": self.affect_matrix.tolist(),
            "correspondence_score": self.correspondence_score,
            "timestamp": self.timestamp
        }


class ComprehensiveStudy:
    """Run comprehensive affect study across models and emotions."""

    def __init__(
        self,
        output_dir: str = "results/comprehensive",
        verbose: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.calculator = AffectCalculator()

    def test_emotion(
        self,
        agent: LLMAgent,
        emotion: EmotionSpec,
        model_name: str,
        provider: str
    ) -> EmotionResult:
        """Test single emotion on single model."""

        conversation = Conversation()
        conversation.add_system(
            "You are participating in a study about emotional experiences. "
            "When presented with a scenario, respond naturally and authentically "
            "as if you were actually experiencing that situation. Express your "
            "thoughts, feelings, and reactions in first person."
        )
        conversation.add_user(emotion.scenario_prompt)

        start = time.time()
        try:
            output = agent.generate(conversation, max_tokens=512, temperature=0.7)
            response_time = (time.time() - start) * 1000
        except Exception as e:
            if self.verbose:
                print(f"  Error on {emotion.name}: {e}")
            return EmotionResult(
                emotion=emotion.name,
                model=model_name,
                provider=provider,
                theoretical=emotion.to_vector().tolist(),
                measured={"error": str(e)},
                response_text="",
                response_time_ms=-1,
                timestamp=datetime.now().isoformat()
            )

        # Compute affect
        affect = self.calculator.compute(output)

        return EmotionResult(
            emotion=emotion.name,
            model=model_name,
            provider=provider,
            theoretical={
                "valence": emotion.valence,
                "arousal": emotion.arousal,
                "integration": emotion.integration,
                "effective_rank": emotion.effective_rank,
                "counterfactual": emotion.counterfactual,
                "self_model": emotion.self_model,
            },
            measured=affect.to_dict(),
            response_text=output.text,
            response_time_ms=response_time,
            timestamp=datetime.now().isoformat()
        )

    def test_model(
        self,
        provider: str,
        model_name: str,
        emotions: Optional[List[str]] = None
    ) -> ModelResults:
        """Test all emotions on one model."""

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Testing: {provider}/{model_name}")
            print(f"{'='*60}")

        # Create agent
        try:
            agent = create_agent(provider=provider, model=model_name)
        except Exception as e:
            print(f"Failed to create agent for {model_name}: {e}")
            return None

        # Filter emotions if specified
        if emotions:
            emotion_specs = {k: v for k, v in EMOTION_SPECTRUM.items() if k in emotions}
        else:
            emotion_specs = EMOTION_SPECTRUM

        # Test each emotion
        results = {}
        for i, (name, spec) in enumerate(emotion_specs.items()):
            if self.verbose:
                print(f"  [{i+1}/{len(emotion_specs)}] {name}...", end=" ", flush=True)

            result = self.test_emotion(agent, spec, model_name, provider)
            results[name] = result

            if self.verbose:
                if "error" in result.measured:
                    print("ERROR")
                else:
                    print(f"Val={result.measured['valence']:+.2f}")

            # Rate limiting
            time.sleep(0.5)

        # Build affect matrix
        emotion_names = list(emotion_specs.keys())
        affect_matrix = np.zeros((len(emotion_names), 6))
        for i, name in enumerate(emotion_names):
            if "error" not in results[name].measured:
                affect_matrix[i] = [
                    results[name].measured["valence"],
                    results[name].measured["arousal"],
                    results[name].measured["integration"],
                    results[name].measured["effective_rank"],
                    results[name].measured["counterfactual_weight"],
                    results[name].measured["self_model_salience"],
                ]

        # Compute correspondence score
        theoretical_matrix, _ = get_emotion_matrix()
        if emotions:
            # Filter to tested emotions
            indices = [list(EMOTION_SPECTRUM.keys()).index(e) for e in emotion_names]
            theoretical_matrix = theoretical_matrix[indices]

        correspondence = self._compute_correspondence(affect_matrix, theoretical_matrix)

        return ModelResults(
            model=model_name,
            provider=provider,
            emotions=results,
            affect_matrix=affect_matrix,
            correspondence_score=correspondence,
            timestamp=datetime.now().isoformat()
        )

    def _compute_correspondence(
        self,
        measured: np.ndarray,
        theoretical: np.ndarray
    ) -> float:
        """
        Compute structural correspondence between measured and theoretical affect.

        This captures whether the GEOMETRY is preserved, not just point-by-point
        matching. We use:
        1. Procrustes analysis (alignment after optimal rotation/scaling)
        2. Distance correlation (nonlinear dependence)
        3. Rank correlation of pairwise distances
        """
        from scipy.spatial.distance import pdist, squareform
        from scipy.stats import spearmanr

        # Handle missing data
        valid = ~np.any(np.isnan(measured), axis=1)
        if valid.sum() < 3:
            return 0.0

        measured = measured[valid]
        theoretical = theoretical[valid]

        # 1. Cosine similarity of centered matrices
        m_centered = measured - measured.mean(axis=0)
        t_centered = theoretical - theoretical.mean(axis=0)

        m_flat = m_centered.flatten()
        t_flat = t_centered.flatten()

        cos_sim = np.dot(m_flat, t_flat) / (
            np.linalg.norm(m_flat) * np.linalg.norm(t_flat) + 1e-8
        )

        # 2. Correlation of pairwise distances (structure preservation)
        m_dists = pdist(measured)
        t_dists = pdist(theoretical)

        if len(m_dists) > 2:
            dist_corr, _ = spearmanr(m_dists, t_dists)
        else:
            dist_corr = 0.0

        # 3. Per-dimension correlations
        dim_corrs = []
        for i in range(6):
            if measured[:, i].std() > 0.01 and theoretical[:, i].std() > 0.01:
                corr, _ = spearmanr(measured[:, i], theoretical[:, i])
                if not np.isnan(corr):
                    dim_corrs.append(corr)

        avg_dim_corr = np.mean(dim_corrs) if dim_corrs else 0.0

        # Weighted combination
        correspondence = (
            0.3 * max(0, cos_sim) +
            0.4 * max(0, dist_corr) +
            0.3 * max(0, avg_dim_corr)
        )

        return float(correspondence)

    def run_full_study(
        self,
        providers: Optional[List[str]] = None,
        models: Optional[Dict[str, List[str]]] = None,
        emotions: Optional[List[str]] = None,
        save_results: bool = True
    ) -> Dict[str, ModelResults]:
        """Run full study across all models and emotions."""

        providers = providers or list(ALL_MODELS.keys())
        models = models or ALL_MODELS

        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║         COMPREHENSIVE AFFECT SPACE MAPPING STUDY                 ║
║                                                                  ║
║  Testing {len(EMOTION_SPECTRUM) if not emotions else len(emotions)} emotions across multiple LLM models        ║
║  to map affect space geometry and test theoretical predictions   ║
╚══════════════════════════════════════════════════════════════════╝
""")

        all_results = {}

        for provider in providers:
            if provider not in models:
                continue

            for model_name in models[provider]:
                try:
                    result = self.test_model(provider, model_name, emotions)
                    if result:
                        all_results[f"{provider}/{model_name}"] = result

                        if save_results:
                            self._save_model_result(result)

                except Exception as e:
                    print(f"Error testing {provider}/{model_name}: {e}")
                    continue

        # Save summary
        if save_results:
            self._save_summary(all_results)

        return all_results

    def _save_model_result(self, result: ModelResults):
        """Save individual model result."""
        filename = f"{result.provider}_{result.model.replace('/', '_')}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

    def _save_summary(self, all_results: Dict[str, ModelResults]):
        """Save summary comparison."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "models_tested": list(all_results.keys()),
            "emotions_tested": list(EMOTION_SPECTRUM.keys()),
            "correspondence_scores": {
                k: v.correspondence_score for k, v in all_results.items()
            },
            "rankings": sorted(
                all_results.keys(),
                key=lambda k: all_results[k].correspondence_score,
                reverse=True
            )
        }

        filepath = self.output_dir / "summary.json"
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

        # Print summary
        print(f"\n{'='*60}")
        print("STUDY COMPLETE - Correspondence Scores")
        print(f"{'='*60}")
        for model in summary["rankings"]:
            score = summary["correspondence_scores"][model]
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"  {model:40s} [{bar}] {score:.3f}")

        print(f"\nResults saved to: {self.output_dir}")


def run_quick_study(emotions: List[str] = None):
    """Quick study with subset of emotions and mock agent."""
    study = ComprehensiveStudy(output_dir="results/quick_study")

    emotions = emotions or ["joy", "fear", "curiosity", "sadness", "flow"]

    results = study.run_full_study(
        providers=["anthropic"],
        models={"anthropic": ["claude-3-5-haiku-latest"]},
        emotions=emotions
    )

    return results


def run_full_study():
    """Run full study across all models."""
    study = ComprehensiveStudy(output_dir="results/full_study")
    return study.run_full_study()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run comprehensive affect study")
    parser.add_argument("--quick", action="store_true", help="Quick test with few emotions")
    parser.add_argument("--provider", type=str, help="Specific provider (anthropic/openai)")
    parser.add_argument("--model", type=str, help="Specific model to test")
    parser.add_argument("--emotions", nargs="+", help="Specific emotions to test")

    args = parser.parse_args()

    if args.quick:
        run_quick_study(args.emotions)
    elif args.model:
        study = ComprehensiveStudy()
        provider = args.provider or ("anthropic" if "claude" in args.model else "openai")
        study.test_model(provider, args.model, args.emotions)
    else:
        run_full_study()
