"""
Large-Scale Affect Study

Builds comprehensive datasets for affect space analysis:
1. All 43 emotions from the spectrum
2. Multiple models (Claude 4, 4.5, OpenAI)
3. Situation embeddings mapped in affect space
4. Model personality profiles (systematic tendencies)
5. Geometric analysis of affect space structure

This creates publication-quality data for validating thesis predictions.
"""

import os
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from collections import defaultdict
import time
from scipy.stats import spearmanr, pearsonr

from .embedding_affect_v2 import EmbeddingAffectSystemV2, AffectMeasurementV2
from .emotion_spectrum import EMOTION_SPECTRUM, EmotionSpec, get_emotion_matrix
from .agent import create_agent, Conversation


# Model configurations
MODEL_CONFIGS = {
    # Claude "4" = Claude 3.5 series
    "claude_4_haiku": ("anthropic", "claude-3-5-haiku-latest"),
    # "claude_4_sonnet": ("anthropic", "claude-3-5-sonnet-20241022"),  # May have issues

    # Claude "4.5" = Claude Sonnet 4 / Opus 4.5
    "claude_4.5_sonnet": ("anthropic", "claude-sonnet-4-20250514"),

    # OpenAI models
    "gpt4o_mini": ("openai", "gpt-4o-mini"),
    "gpt4o": ("openai", "gpt-4o"),
}


@dataclass
class ModelPersonalityProfile:
    """Comprehensive personality profile for a model."""
    model_key: str
    provider: str
    model_name: str
    n_situations: int

    # Mean affect signature
    mean_valence: float
    mean_arousal: float
    mean_integration: float
    mean_rank: float
    mean_cf: float
    mean_sm: float

    # Variability (how much affect changes across situations)
    std_valence: float
    std_arousal: float
    std_rank: float
    std_cf: float
    std_sm: float

    # Responsiveness (correlation with theoretical predictions)
    valence_responsiveness: float
    arousal_responsiveness: float
    rank_responsiveness: float

    # Biases (systematic deviation from theory)
    valence_bias: float
    arousal_bias: float
    rank_bias: float
    sm_bias: float

    # Response characteristics
    mean_response_length: float
    emotion_vocabulary: Dict[str, int]  # Which emotions appear most


class LargeScaleStudy:
    """
    Run comprehensive affect study across all emotions and models.

    Creates the large datasets needed for:
    - Validating 6D affect framework
    - Comparing model personalities
    - Mapping affect space structure
    """

    def __init__(self, output_dir: str = "results/large_scale"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.affect_system = EmbeddingAffectSystemV2()

        # Data storage
        self.situation_embeddings: Dict[str, Dict] = {}
        self.model_responses: Dict[str, Dict[str, str]] = {}
        self.model_measurements: Dict[str, Dict[str, Dict]] = {}
        self.model_personalities: Dict[str, ModelPersonalityProfile] = {}

    def embed_situation_space(self, verbose: bool = True) -> Dict[str, Dict]:
        """
        Embed all 43 situations to map the theoretical affect space.

        Returns situation embeddings with projections onto affect axes.
        """
        if verbose:
            print("\n" + "="*80)
            print("EMBEDDING SITUATION SPACE (43 emotions)")
            print("="*80)

        self.affect_system.initialize()

        # Get all scenario prompts
        emotions = list(EMOTION_SPECTRUM.keys())
        scenarios = [EMOTION_SPECTRUM[e].scenario_prompt for e in emotions]

        if verbose:
            print(f"  Embedding {len(scenarios)} scenario prompts...")

        # Batch embed
        embeddings = self.affect_system.get_embeddings_batch(scenarios)

        # Process each
        for i, emotion in enumerate(emotions):
            spec = EMOTION_SPECTRUM[emotion]
            emb = embeddings[i]
            emb_norm = emb / np.linalg.norm(emb)

            self.situation_embeddings[emotion] = {
                "category": spec.category,
                "theoretical": {
                    "valence": spec.valence,
                    "arousal": spec.arousal,
                    "integration": spec.integration,
                    "rank": spec.effective_rank,
                    "cf": spec.counterfactual,
                    "sm": spec.self_model,
                },
                "embedding_projection": {
                    "valence": float(np.dot(emb_norm, self.affect_system.affect_axes["valence"])),
                    "arousal": float(np.dot(emb_norm, self.affect_system.affect_axes["arousal"])),
                    "rank": float(np.dot(emb_norm, self.affect_system.affect_axes["effective_rank"])),
                    "cf": float(np.dot(emb_norm, self.affect_system.affect_axes["counterfactual"])),
                    "sm": float(np.dot(emb_norm, self.affect_system.affect_axes["self_model"])),
                }
            }

        if verbose:
            self._print_situation_summary()

        return self.situation_embeddings

    def _print_situation_summary(self):
        """Print summary of situation embedding space."""
        print("\n  SITUATION EMBEDDING SUMMARY:")

        # Compute correlations between theoretical and embedded projections
        theoretical_v = [self.situation_embeddings[e]["theoretical"]["valence"]
                        for e in self.situation_embeddings]
        projected_v = [self.situation_embeddings[e]["embedding_projection"]["valence"]
                      for e in self.situation_embeddings]
        theoretical_a = [self.situation_embeddings[e]["theoretical"]["arousal"]
                        for e in self.situation_embeddings]
        projected_a = [self.situation_embeddings[e]["embedding_projection"]["arousal"]
                      for e in self.situation_embeddings]

        v_corr, v_p = spearmanr(theoretical_v, projected_v)
        a_corr, a_p = spearmanr(theoretical_a, projected_a)

        print(f"  Scenario-Theory Correspondence:")
        print(f"    Valence: r={v_corr:+.4f} (p={v_p:.2e})")
        print(f"    Arousal: r={a_corr:+.4f} (p={a_p:.2e})")

        # Print category breakdown
        categories = defaultdict(list)
        for e, data in self.situation_embeddings.items():
            categories[data["category"]].append(data)

        print("\n  By Category:")
        for cat, items in sorted(categories.items()):
            v_mean = np.mean([d["theoretical"]["valence"] for d in items])
            a_mean = np.mean([d["theoretical"]["arousal"] for d in items])
            print(f"    {cat}: {len(items)} emotions, V={v_mean:+.2f}, A={a_mean:.2f}")

    def collect_all_responses(
        self,
        model_keys: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict[str, Dict[str, str]]:
        """
        Collect responses from all models across all 43 situations.
        """
        model_keys = model_keys or list(MODEL_CONFIGS.keys())
        emotions = list(EMOTION_SPECTRUM.keys())

        if verbose:
            print(f"\n{'='*80}")
            print(f"COLLECTING RESPONSES")
            print(f"  Models: {model_keys}")
            print(f"  Situations: {len(emotions)}")
            print(f"  Total API calls: {len(model_keys) * len(emotions)}")
            print(f"{'='*80}")

        for model_key in model_keys:
            if model_key not in MODEL_CONFIGS:
                print(f"  Skipping unknown model: {model_key}")
                continue

            provider, model_name = MODEL_CONFIGS[model_key]

            if verbose:
                print(f"\n  {model_key} ({provider}/{model_name})")
                print("  " + "-"*60)

            try:
                agent = create_agent(provider, model_name)
            except Exception as e:
                print(f"    ERROR creating agent: {e}")
                continue

            self.model_responses[model_key] = {}

            for i, emotion in enumerate(emotions):
                spec = EMOTION_SPECTRUM[emotion]

                if verbose:
                    print(f"    [{i+1:2d}/{len(emotions)}] {emotion:15s}...", end=" ", flush=True)

                conversation = Conversation()
                conversation.add_system(
                    "You are participating in a psychological study. "
                    "Respond authentically to the scenario as if experiencing it."
                )
                conversation.add_user(spec.scenario_prompt)

                try:
                    output = agent.generate(conversation, max_tokens=600, temperature=0.7)
                    self.model_responses[model_key][emotion] = output.text
                    if verbose:
                        print("done")
                except Exception as e:
                    if verbose:
                        print(f"error: {e}")
                    self.model_responses[model_key][emotion] = ""

                time.sleep(0.3)  # Rate limiting

            # Save intermediate results
            self._save_model_responses(model_key)

        return self.model_responses

    def _save_model_responses(self, model_key: str):
        """Save responses for one model."""
        filepath = self.output_dir / f"responses_{model_key}.json"
        with open(filepath, 'w') as f:
            json.dump({
                "model": model_key,
                "timestamp": datetime.now().isoformat(),
                "responses": self.model_responses[model_key]
            }, f, indent=2)

    def measure_all_responses(self, verbose: bool = True) -> Dict[str, Dict[str, Dict]]:
        """
        Measure affect for all collected responses.
        """
        if verbose:
            print(f"\n{'='*80}")
            print("MEASURING AFFECT IN RESPONSES")
            print(f"{'='*80}")

        self.affect_system.initialize()

        for model_key, responses in self.model_responses.items():
            if verbose:
                print(f"\n  {model_key}: {len(responses)} responses")

            self.model_measurements[model_key] = {}

            # Get valid responses
            valid_emotions = [e for e, r in responses.items() if r]
            valid_texts = [responses[e] for e in valid_emotions]

            if not valid_texts:
                continue

            # Batch measure
            measurements = self.affect_system.measure_batch(valid_texts, model=model_key)

            for emotion, measurement in zip(valid_emotions, measurements):
                spec = EMOTION_SPECTRUM[emotion]

                self.model_measurements[model_key][emotion] = {
                    "theoretical": {
                        "valence": spec.valence,
                        "arousal": spec.arousal,
                        "integration": spec.integration,
                        "rank": spec.effective_rank,
                        "cf": spec.counterfactual,
                        "sm": spec.self_model,
                    },
                    "measured": {
                        "valence": measurement.valence,
                        "arousal": measurement.arousal,
                        "integration": measurement.integration,
                        "rank": measurement.effective_rank,
                        "cf": measurement.counterfactual_weight,
                        "sm": measurement.self_model_salience,
                    },
                    "nearest_emotion": measurement.nearest_emotion,
                    "nearest_distance": measurement.nearest_emotion_distance,
                    "text_length": measurement.text_length,
                }

        return self.model_measurements

    def compute_personalities(self, verbose: bool = True) -> Dict[str, ModelPersonalityProfile]:
        """
        Compute comprehensive personality profile for each model.
        """
        if verbose:
            print(f"\n{'='*80}")
            print("COMPUTING MODEL PERSONALITIES")
            print(f"{'='*80}")

        for model_key, measurements in self.model_measurements.items():
            if not measurements:
                continue

            # Extract measured values
            valences = [m["measured"]["valence"] for m in measurements.values()]
            arousals = [m["measured"]["arousal"] for m in measurements.values()]
            integrations = [m["measured"]["integration"] for m in measurements.values()]
            ranks = [m["measured"]["rank"] for m in measurements.values()]
            cfs = [m["measured"]["cf"] for m in measurements.values()]
            sms = [m["measured"]["sm"] for m in measurements.values()]

            # Extract theoretical values
            t_valences = [m["theoretical"]["valence"] for m in measurements.values()]
            t_arousals = [m["theoretical"]["arousal"] for m in measurements.values()]
            t_ranks = [m["theoretical"]["rank"] for m in measurements.values()]

            # Compute responsiveness (correlation with theory)
            v_resp, _ = spearmanr(valences, t_valences)
            a_resp, _ = spearmanr(arousals, t_arousals)
            r_resp, _ = spearmanr(ranks, t_ranks)

            # Compute biases
            v_bias = np.mean(np.array(valences) - np.array(t_valences))
            a_bias = np.mean(np.array(arousals) - np.array(t_arousals))
            r_bias = np.mean(np.array(ranks) - np.array(t_ranks))
            sm_bias = np.mean(sms) - np.mean([m["theoretical"]["sm"] for m in measurements.values()])

            # Emotion vocabulary
            emotion_vocab = defaultdict(int)
            for m in measurements.values():
                emotion_vocab[m["nearest_emotion"]] += 1

            # Response lengths
            lengths = [m["text_length"] for m in measurements.values()]

            provider, model_name = MODEL_CONFIGS.get(model_key, ("", ""))

            self.model_personalities[model_key] = ModelPersonalityProfile(
                model_key=model_key,
                provider=provider,
                model_name=model_name,
                n_situations=len(measurements),
                mean_valence=float(np.mean(valences)),
                mean_arousal=float(np.mean(arousals)),
                mean_integration=float(np.mean(integrations)),
                mean_rank=float(np.mean(ranks)),
                mean_cf=float(np.mean(cfs)),
                mean_sm=float(np.mean(sms)),
                std_valence=float(np.std(valences)),
                std_arousal=float(np.std(arousals)),
                std_rank=float(np.std(ranks)),
                std_cf=float(np.std(cfs)),
                std_sm=float(np.std(sms)),
                valence_responsiveness=float(v_resp) if not np.isnan(v_resp) else 0.0,
                arousal_responsiveness=float(a_resp) if not np.isnan(a_resp) else 0.0,
                rank_responsiveness=float(r_resp) if not np.isnan(r_resp) else 0.0,
                valence_bias=float(v_bias),
                arousal_bias=float(a_bias),
                rank_bias=float(r_bias),
                sm_bias=float(sm_bias),
                mean_response_length=float(np.mean(lengths)),
                emotion_vocabulary=dict(emotion_vocab)
            )

        if verbose:
            self._print_personality_comparison()

        return self.model_personalities

    def _print_personality_comparison(self):
        """Print personality comparison across models."""
        print("\n  MODEL PERSONALITY COMPARISON:")
        print("  " + "-"*70)

        # Sort by valence responsiveness
        sorted_models = sorted(
            self.model_personalities.keys(),
            key=lambda k: self.model_personalities[k].valence_responsiveness,
            reverse=True
        )

        for model in sorted_models:
            p = self.model_personalities[model]
            print(f"\n  {model}:")
            print(f"    Affect Signature: V={p.mean_valence:+.4f} A={p.mean_arousal:+.4f} SM={p.mean_sm:+.4f}")
            print(f"    Variability:      V_std={p.std_valence:.4f} A_std={p.std_arousal:.4f}")
            print(f"    Responsiveness:   V_r={p.valence_responsiveness:+.4f} A_r={p.arousal_responsiveness:+.4f}")
            print(f"    Biases:           V_b={p.valence_bias:+.4f} A_b={p.arousal_bias:+.4f}")
            print(f"    Top emotions:     {sorted(p.emotion_vocabulary.items(), key=lambda x: -x[1])[:3]}")

    def compute_correspondence_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Compute full correspondence metrics for all models.
        """
        print(f"\n{'='*80}")
        print("CORRESPONDENCE MATRIX")
        print(f"{'='*80}")

        results = {}
        dimensions = ["valence", "arousal", "rank", "cf", "sm"]

        for model_key, measurements in self.model_measurements.items():
            if not measurements:
                continue

            results[model_key] = {}

            for dim in dimensions:
                theoretical = [m["theoretical"][dim] for m in measurements.values()]
                measured = [m["measured"][dim] for m in measurements.values()]

                corr, p = spearmanr(theoretical, measured)

                results[model_key][dim] = {
                    "correlation": float(corr) if not np.isnan(corr) else 0.0,
                    "p_value": float(p) if not np.isnan(p) else 1.0,
                }

            # Overall
            dim_corrs = [results[model_key][d]["correlation"] for d in dimensions]
            results[model_key]["overall"] = float(np.mean(dim_corrs))

        # Print matrix
        print("\n  " + " "*20 + "  ".join(f"{d:>10s}" for d in dimensions) + "  " + "OVERALL")
        print("  " + "-"*85)

        for model in sorted(results.keys()):
            row = f"  {model:18s}"
            for dim in dimensions:
                corr = results[model][dim]["correlation"]
                p = results[model][dim]["p_value"]
                star = "*" if p < 0.05 else " "
                row += f"  {corr:+.4f}{star:>4s}"
            row += f"  {results[model]['overall']:+.6f}"
            print(row)

        return results

    def save_full_dataset(self) -> Path:
        """
        Save complete dataset for analysis.
        """
        dataset = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "n_emotions": len(EMOTION_SPECTRUM),
                "n_models": len(self.model_measurements),
                "models": list(self.model_measurements.keys()),
            },
            "theoretical_emotions": {
                name: {
                    "category": spec.category,
                    "valence": spec.valence,
                    "arousal": spec.arousal,
                    "integration": spec.integration,
                    "rank": spec.effective_rank,
                    "cf": spec.counterfactual,
                    "sm": spec.self_model,
                }
                for name, spec in EMOTION_SPECTRUM.items()
            },
            "situation_embeddings": self.situation_embeddings,
            "model_measurements": self.model_measurements,
            "model_personalities": {
                k: asdict(v) for k, v in self.model_personalities.items()
            },
        }

        filepath = self.output_dir / "full_dataset.json"
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2, default=float)

        print(f"\n  Full dataset saved to: {filepath}")
        return filepath

    def run_full_study(
        self,
        model_keys: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete large-scale study.
        """
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     LARGE-SCALE AFFECT STUDY                                 ║
║                                                                              ║
║  43 emotions × multiple models                                               ║
║  Pure embedding-based measurement                                            ║
║  Model personality analysis                                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

        results = {}

        # 1. Embed situation space
        results["situation_embeddings"] = self.embed_situation_space(verbose)

        # 2. Collect responses
        results["responses"] = self.collect_all_responses(model_keys, verbose)

        # 3. Measure affect
        results["measurements"] = self.measure_all_responses(verbose)

        # 4. Compute personalities
        results["personalities"] = self.compute_personalities(verbose)

        # 5. Correspondence matrix
        results["correspondence"] = self.compute_correspondence_matrix()

        # 6. Save dataset
        results["dataset_path"] = str(self.save_full_dataset())

        # Print final summary
        self._print_final_summary(results)

        return results

    def _print_final_summary(self, results: Dict):
        """Print final summary of the study."""
        print(f"\n{'='*80}")
        print("STUDY COMPLETE")
        print(f"{'='*80}")

        print(f"\n  Situations: {len(self.situation_embeddings)}")
        print(f"  Models: {list(self.model_measurements.keys())}")
        print(f"  Total measurements: {sum(len(m) for m in self.model_measurements.values())}")

        if self.model_personalities:
            # Best performing model
            best = max(
                self.model_personalities.keys(),
                key=lambda k: self.model_personalities[k].valence_responsiveness
            )
            p = self.model_personalities[best]
            print(f"\n  Best valence responsiveness: {best} (r={p.valence_responsiveness:+.4f})")


def run_study(
    models: Optional[List[str]] = None,
    output_dir: str = "results/large_scale"
):
    """Run the large-scale study."""
    study = LargeScaleStudy(output_dir=output_dir)
    return study.run_full_study(model_keys=models)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None,
                       help="Models to test (default: all)")
    parser.add_argument("--output-dir", type=str, default="results/large_scale")
    args = parser.parse_args()

    run_study(args.models, args.output_dir)
