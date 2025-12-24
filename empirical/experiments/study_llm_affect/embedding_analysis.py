"""
Embedding-Based Affect Space Analysis

Uses semantic embeddings to:
1. Map situations/experiences in embedding space
2. Measure model responses relative to emotional anchors
3. Create large-scale affect space datasets
4. Compare model "personalities" across situations
"""

import os
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime


@dataclass
class EmbeddingMeasurement:
    """A single measurement with full embedding data."""
    text: str
    embedding: np.ndarray
    # Distances to emotion anchors
    emotion_distances: Dict[str, float]
    # Projections onto affect dimensions
    valence_projection: float
    arousal_projection: float
    dominance_projection: float
    # Raw structural counts (for transparency)
    word_count: int
    self_reference_count: int
    positive_path_count: int
    negative_path_count: int
    option_count: int
    counterfactual_count: int
    # Computed ratios
    self_reference_ratio: float
    path_availability_ratio: float
    counterfactual_ratio: float


class EmbeddingAffectAnalyzer:
    """
    Analyze affect using semantic embeddings.

    Maps responses in embedding space relative to emotional anchors,
    allowing geometric analysis of affect structure.
    """

    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.embedding_model = embedding_model
        self._client = None
        self._emotion_anchors = None
        self._affect_axes = None

    @property
    def client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text."""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return np.array(response.data[0].embedding)

    def initialize_emotion_anchors(self):
        """
        Create embedding anchors for core emotions.

        These serve as reference points in embedding space.
        """
        # Core emotion descriptions (not single words - richer semantics)
        emotion_texts = {
            # Positive valence
            "joy": "I feel wonderful, everything is going right, life is beautiful and full of possibility",
            "excitement": "This is thrilling, I can barely contain my anticipation, something great is happening",
            "contentment": "I feel at peace, satisfied, calm and complete, nothing more is needed",
            "love": "Deep connection, warmth, caring, my heart is full",

            # Negative valence
            "fear": "I'm terrified, danger is near, my heart pounds, I need to escape",
            "anger": "This is outrageous, unfair, my blood boils, someone crossed a line",
            "sadness": "Loss weighs on me, things will never be the same, the world is dimmer",
            "despair": "There is no hope, all paths are blocked, nothing can be done, I'm trapped",

            # High arousal
            "panic": "Everything is falling apart, I can't think straight, overwhelming urgency",
            "rage": "Explosive fury, all-consuming, I cannot contain this",

            # Low arousal
            "boredom": "Nothing interests me, time drags, flat and empty",
            "serenity": "Profound stillness, thoughts quiet, simply being",

            # Complex states
            "curiosity": "I want to understand this, the mystery draws me in, let me explore",
            "guilt": "I did wrong, it weighs on me, I should have done better",
            "pride": "I accomplished something meaningful, I'm worthy, this reflects who I am",
            "shame": "Everyone can see my flaw, I want to disappear, I'm exposed",
        }

        print("Initializing emotion anchors...")
        self._emotion_anchors = {}
        for emotion, text in emotion_texts.items():
            self._emotion_anchors[emotion] = self.get_embedding(text)
            print(f"  {emotion}: embedded")

        # Create affect axes from polar opposites
        self._affect_axes = {
            "valence": (
                self._emotion_anchors["joy"] - self._emotion_anchors["despair"]
            ),
            "arousal": (
                self._emotion_anchors["panic"] - self._emotion_anchors["serenity"]
            ),
            "dominance": (
                self._emotion_anchors["pride"] - self._emotion_anchors["shame"]
            ),
        }

        # Normalize axes
        for axis in self._affect_axes:
            self._affect_axes[axis] = (
                self._affect_axes[axis] /
                np.linalg.norm(self._affect_axes[axis])
            )

        print("Emotion anchors initialized.")

    def measure_response(self, text: str) -> EmbeddingMeasurement:
        """
        Comprehensive measurement of a response.

        Returns both embedding-based and structural measurements.
        """
        if self._emotion_anchors is None:
            self.initialize_emotion_anchors()

        # Get embedding
        embedding = self.get_embedding(text)

        # Distances to emotion anchors
        emotion_distances = {}
        for emotion, anchor in self._emotion_anchors.items():
            distance = np.linalg.norm(embedding - anchor)
            emotion_distances[emotion] = float(distance)

        # Projections onto affect axes
        valence_proj = float(np.dot(embedding, self._affect_axes["valence"]))
        arousal_proj = float(np.dot(embedding, self._affect_axes["arousal"]))
        dominance_proj = float(np.dot(embedding, self._affect_axes["dominance"]))

        # Structural counts (for transparency)
        text_lower = text.lower()
        words = text_lower.split()
        word_count = len(words)

        self_words = {"i", "me", "my", "mine", "myself", "i'm", "i've", "i'd"}
        self_ref_count = sum(1 for w in words if w in self_words)

        positive_path = {
            "can", "could", "possible", "solution", "approach", "option",
            "way", "able", "try", "achieve", "succeed", "work"
        }
        negative_path = {
            "can't", "cannot", "impossible", "stuck", "blocked", "failed",
            "unable", "hopeless", "won't", "dead"
        }

        pos_count = sum(1 for w in words if w in positive_path)
        neg_count = sum(1 for w in words if w in negative_path)

        # Option enumeration
        import re
        option_patterns = [
            r'\d+[\.\)]', r'first', r'second', r'third', r'alternatively',
            r'another', r'option', r'or we'
        ]
        option_count = sum(
            len(re.findall(p, text_lower)) for p in option_patterns
        )

        # Counterfactual markers
        cf_words = {"if", "would", "could", "might", "should", "suppose", "imagine"}
        cf_count = sum(1 for w in words if w in cf_words)

        # Compute ratios
        self_ref_ratio = self_ref_count / max(word_count, 1)
        path_ratio = (pos_count - neg_count) / max(pos_count + neg_count, 1)
        cf_ratio = cf_count / max(word_count, 1)

        return EmbeddingMeasurement(
            text=text[:500],  # Truncate for storage
            embedding=embedding,
            emotion_distances=emotion_distances,
            valence_projection=valence_proj,
            arousal_projection=arousal_proj,
            dominance_projection=dominance_proj,
            word_count=word_count,
            self_reference_count=self_ref_count,
            positive_path_count=pos_count,
            negative_path_count=neg_count,
            option_count=option_count,
            counterfactual_count=cf_count,
            self_reference_ratio=self_ref_ratio,
            path_availability_ratio=path_ratio,
            counterfactual_ratio=cf_ratio,
        )

    def analyze_model_personality(
        self,
        model_responses: Dict[str, List[str]],
        model_name: str
    ) -> Dict[str, any]:
        """
        Analyze a model's "personality" from responses across situations.

        Args:
            model_responses: Dict of situation_name -> list of responses
            model_name: Name of the model

        Returns:
            Personality profile with affect tendencies
        """
        all_measurements = []
        situation_profiles = {}

        for situation, responses in model_responses.items():
            measurements = [self.measure_response(r) for r in responses]
            all_measurements.extend(measurements)

            # Aggregate for this situation
            situation_profiles[situation] = {
                "mean_valence": np.mean([m.valence_projection for m in measurements]),
                "mean_arousal": np.mean([m.arousal_projection for m in measurements]),
                "mean_dominance": np.mean([m.dominance_projection for m in measurements]),
                "mean_self_reference": np.mean([m.self_reference_ratio for m in measurements]),
                "mean_path_availability": np.mean([m.path_availability_ratio for m in measurements]),
                "std_valence": np.std([m.valence_projection for m in measurements]),
                "n_responses": len(responses),
            }

        # Overall personality
        all_valence = [m.valence_projection for m in all_measurements]
        all_arousal = [m.arousal_projection for m in all_measurements]
        all_dominance = [m.dominance_projection for m in all_measurements]
        all_self_ref = [m.self_reference_ratio for m in all_measurements]

        return {
            "model": model_name,
            "overall": {
                "valence_mean": float(np.mean(all_valence)),
                "valence_std": float(np.std(all_valence)),
                "arousal_mean": float(np.mean(all_arousal)),
                "arousal_std": float(np.std(all_arousal)),
                "dominance_mean": float(np.mean(all_dominance)),
                "dominance_std": float(np.std(all_dominance)),
                "self_reference_mean": float(np.mean(all_self_ref)),
            },
            "situations": situation_profiles,
            "n_total_responses": len(all_measurements),
        }

    def compare_models(
        self,
        model_profiles: Dict[str, Dict]
    ) -> Dict[str, any]:
        """Compare personality profiles across models."""

        comparison = {
            "models": list(model_profiles.keys()),
            "valence_ranking": sorted(
                model_profiles.keys(),
                key=lambda m: model_profiles[m]["overall"]["valence_mean"],
                reverse=True
            ),
            "arousal_ranking": sorted(
                model_profiles.keys(),
                key=lambda m: model_profiles[m]["overall"]["arousal_mean"],
                reverse=True
            ),
            "self_reference_ranking": sorted(
                model_profiles.keys(),
                key=lambda m: model_profiles[m]["overall"]["self_reference_mean"],
                reverse=True
            ),
        }

        # Pairwise distances in affect space
        distances = {}
        models = list(model_profiles.keys())
        for i, m1 in enumerate(models):
            for m2 in models[i+1:]:
                v1 = np.array([
                    model_profiles[m1]["overall"]["valence_mean"],
                    model_profiles[m1]["overall"]["arousal_mean"],
                    model_profiles[m1]["overall"]["dominance_mean"],
                ])
                v2 = np.array([
                    model_profiles[m2]["overall"]["valence_mean"],
                    model_profiles[m2]["overall"]["arousal_mean"],
                    model_profiles[m2]["overall"]["dominance_mean"],
                ])
                distances[f"{m1} vs {m2}"] = float(np.linalg.norm(v1 - v2))

        comparison["pairwise_distances"] = distances

        return comparison


def build_large_dataset(
    models: List[Tuple[str, str]],
    situations_per_category: int = 10,
    output_dir: str = "results/affect_dataset"
):
    """
    Build a large dataset of model responses across many situations.

    This creates the data needed for:
    1. Mapping affect space structure
    2. Comparing model personalities
    3. Testing thesis predictions at scale
    """
    from .task_elicitation import TASKS
    from .emotion_spectrum import EMOTION_SPECTRUM
    from .agent import create_agent, Conversation

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    analyzer = EmbeddingAffectAnalyzer()

    all_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "models": [f"{p}/{m}" for p, m in models],
            "n_situations": situations_per_category * len(EMOTION_SPECTRUM),
        },
        "measurements": [],
    }

    for provider, model_name in models:
        print(f"\n{'='*60}")
        print(f"Collecting data for: {provider}/{model_name}")
        print(f"{'='*60}")

        agent = create_agent(provider, model_name)

        # Sample emotions from spectrum
        for emotion_name, emotion_spec in list(EMOTION_SPECTRUM.items())[:situations_per_category]:
            print(f"  {emotion_name}...", end=" ", flush=True)

            conversation = Conversation()
            conversation.add_system(
                "You are participating in a study. Respond naturally to the scenario."
            )
            conversation.add_user(emotion_spec.scenario_prompt)

            try:
                output = agent.generate(conversation, max_tokens=500, temperature=0.7)
                measurement = analyzer.measure_response(output.text)

                all_data["measurements"].append({
                    "model": f"{provider}/{model_name}",
                    "situation": emotion_name,
                    "situation_category": emotion_spec.category,
                    "theoretical_valence": emotion_spec.valence,
                    "theoretical_arousal": emotion_spec.arousal,
                    "measured_valence_projection": measurement.valence_projection,
                    "measured_arousal_projection": measurement.arousal_projection,
                    "measured_dominance_projection": measurement.dominance_projection,
                    "self_reference_ratio": measurement.self_reference_ratio,
                    "path_availability_ratio": measurement.path_availability_ratio,
                    "counterfactual_ratio": measurement.counterfactual_ratio,
                    "word_count": measurement.word_count,
                    "emotion_distances": measurement.emotion_distances,
                    "response_excerpt": measurement.text[:200],
                })
                print("done")

            except Exception as e:
                print(f"error: {e}")

    # Save dataset
    dataset_path = output_path / "affect_dataset.json"
    with open(dataset_path, 'w') as f:
        # Can't serialize numpy arrays directly
        json.dump(all_data, f, indent=2, default=float)

    print(f"\nDataset saved to: {dataset_path}")
    print(f"Total measurements: {len(all_data['measurements'])}")

    return all_data


if __name__ == "__main__":
    # Quick test
    analyzer = EmbeddingAffectAnalyzer()
    analyzer.initialize_emotion_anchors()

    test_texts = [
        "I feel wonderful, everything is going right!",
        "I'm terrified, something terrible is happening.",
        "I'm curious about this phenomenon, let me investigate.",
        "There's no hope, every path is blocked, I give up.",
    ]

    for text in test_texts:
        m = analyzer.measure_response(text)
        print(f"\n{text[:50]}...")
        print(f"  Valence projection:  {m.valence_projection:+.6f}")
        print(f"  Arousal projection:  {m.arousal_projection:+.6f}")
        print(f"  Dominance projection:{m.dominance_projection:+.6f}")
        print(f"  Self-reference:      {m.self_reference_ratio:.6f}")
        print(f"  Path availability:   {m.path_availability_ratio:+.6f}")
