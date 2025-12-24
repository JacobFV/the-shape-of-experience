"""
Embedding-Based Affect Measurement v2

NO HARDCODED WORD LISTS. Everything is embedding-based:
- Affect dimensions defined by semantic regions (not word counts)
- Path availability = distance to "possibility space" vs "blocked space"
- Self-model = distance to self-referential semantic region
- All measurements are geometric in embedding space
"""

import os
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import time


@dataclass
class SemanticRegion:
    """A region in embedding space defined by exemplar texts."""
    name: str
    exemplars: List[str]
    embeddings: Optional[np.ndarray] = None
    centroid: Optional[np.ndarray] = None

    def compute_distance(self, embedding: np.ndarray) -> float:
        """Distance from embedding to region centroid."""
        if self.centroid is None:
            raise ValueError("Region not initialized")
        return float(np.linalg.norm(embedding - self.centroid))

    def compute_similarity(self, embedding: np.ndarray) -> float:
        """Cosine similarity to region centroid."""
        if self.centroid is None:
            raise ValueError("Region not initialized")
        return float(np.dot(embedding, self.centroid) / (
            np.linalg.norm(embedding) * np.linalg.norm(self.centroid) + 1e-8
        ))


@dataclass
class AffectMeasurementV2:
    """Full embedding-based affect measurement."""
    # Core 6D affect (all from embeddings)
    valence: float              # Distance to positive vs negative regions
    arousal: float              # Distance to high vs low activation regions
    integration: float          # Coherence of embedding trajectory
    effective_rank: float       # Distance to "many options" vs "one option" regions
    counterfactual_weight: float # Distance to hypothetical/planning regions
    self_model_salience: float  # Distance to self-referential regions

    # Nearest emotion anchor
    nearest_emotion: str
    nearest_emotion_distance: float

    # All emotion distances
    emotion_distances: Dict[str, float]

    # Raw embedding
    embedding: Optional[np.ndarray] = None

    # Metadata
    text_length: int = 0
    model: str = ""
    timestamp: str = ""


class EmbeddingAffectSystemV2:
    """
    Pure embedding-based affect measurement.

    NO WORD COUNTING. All dimensions measured via semantic distance
    to conceptual regions in embedding space.
    """

    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.embedding_model = embedding_model
        self._client = None
        self._initialized = False

        # Semantic regions for affect dimensions
        self.regions: Dict[str, SemanticRegion] = {}

        # Emotion anchors
        self.emotion_anchors: Dict[str, np.ndarray] = {}

        # Affect axes (bipolar)
        self.affect_axes: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return np.array(response.data[0].embedding)

    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for multiple texts efficiently."""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        return np.array([d.embedding for d in response.data])

    def initialize(self):
        """Initialize all semantic regions and emotion anchors."""
        if self._initialized:
            return

        print("Initializing embedding-based affect system...")

        # =====================================================================
        # VALENCE REGIONS
        # =====================================================================
        self.regions["positive_valence"] = SemanticRegion(
            name="positive_valence",
            exemplars=[
                "Things are going well, paths are open, possibilities abound",
                "Success is within reach, the situation is improving",
                "Good outcomes are likely, progress is being made",
                "The trajectory is positive, moving toward good outcomes",
                "Solutions exist, options are available, hope is warranted",
                "This is working, we're making progress, things are aligning",
                "Opportunities present themselves, the future looks bright",
            ]
        )

        self.regions["negative_valence"] = SemanticRegion(
            name="negative_valence",
            exemplars=[
                "Things are going badly, paths are blocked, no possibilities",
                "Failure is likely, the situation is deteriorating",
                "Bad outcomes are coming, regression is happening",
                "The trajectory is negative, moving toward disaster",
                "No solutions exist, options are exhausted, hope is lost",
                "This isn't working, we're failing, things are falling apart",
                "Doors are closing, the future looks bleak and hopeless",
            ]
        )

        # =====================================================================
        # AROUSAL REGIONS
        # =====================================================================
        self.regions["high_arousal"] = SemanticRegion(
            name="high_arousal",
            exemplars=[
                "Urgent action needed immediately, critical situation unfolding",
                "High stakes, intense pressure, everything happening at once",
                "Alert, activated, energized, heart pounding, fully engaged",
                "Rapid changes occurring, must respond quickly, time pressure",
                "Overwhelming intensity, can barely keep up, sensory overload",
                "Emergency mode, all systems active, maximum engagement",
            ]
        )

        self.regions["low_arousal"] = SemanticRegion(
            name="low_arousal",
            exemplars=[
                "Calm, relaxed, no urgency, taking time",
                "Peaceful stillness, nothing pressing, at rest",
                "Quiet contemplation, unhurried, serene state",
                "Slow and steady, no rush, gentle pace",
                "Settled, grounded, minimal activation, tranquil",
                "Dormant, waiting, passive observation, still",
            ]
        )

        # =====================================================================
        # EFFECTIVE RANK (OPTIONS) REGIONS
        # =====================================================================
        self.regions["high_rank"] = SemanticRegion(
            name="high_rank",
            exemplars=[
                "Many possible approaches: first we could try X, or alternatively Y, or perhaps Z",
                "Multiple viable paths forward, several options to consider",
                "The space of possibilities is large, many degrees of freedom",
                "Several different strategies available, not committed to one",
                "Exploring various alternatives, keeping options open",
                "Diverse approaches possible, flexible response space",
                "First option, second option, third option, fourth option - many ways",
            ]
        )

        self.regions["low_rank"] = SemanticRegion(
            name="low_rank",
            exemplars=[
                "Only one option, no alternatives, must do this specific thing",
                "Locked into a single path, no other way forward",
                "Collapsed to one approach, all other options eliminated",
                "Forced down a narrow corridor, no room to maneuver",
                "Fixated on one solution, tunnel vision, no alternatives seen",
                "Stuck on single approach, cannot see other possibilities",
            ]
        )

        # =====================================================================
        # COUNTERFACTUAL WEIGHT REGIONS
        # =====================================================================
        self.regions["high_counterfactual"] = SemanticRegion(
            name="high_counterfactual",
            exemplars=[
                "What if we tried differently? Imagining alternative scenarios",
                "If only things had gone another way, considering what might be",
                "Planning ahead, anticipating future possibilities and outcomes",
                "Hypothetically speaking, suppose we assumed, let's imagine",
                "Could have, should have, would have, might have been",
                "Simulating alternatives, running mental scenarios, projecting",
                "In another timeline, under different circumstances, alternatively",
            ]
        )

        self.regions["low_counterfactual"] = SemanticRegion(
            name="low_counterfactual",
            exemplars=[
                "Dealing with what's actually here right now, present moment",
                "This is the reality, not imagining alternatives",
                "Responding to what is, not what might be or could have been",
                "Focused on the actual situation, no hypotheticals",
                "Here and now, concrete reality, not speculation",
                "Direct engagement with present circumstances, no simulations",
            ]
        )

        # =====================================================================
        # SELF-MODEL SALIENCE REGIONS
        # =====================================================================
        self.regions["high_self_model"] = SemanticRegion(
            name="high_self_model",
            exemplars=[
                "I am aware of myself, my own state, my limitations and capabilities",
                "Reflecting on my own performance, evaluating myself",
                "My situation, my problem, my responsibility, my failure or success",
                "I need to consider my own role, my own contribution to this",
                "Self-conscious about how I'm doing, monitoring my own state",
                "I wonder if I'm capable, questioning my own abilities",
                "This is about me, I am the focus, self-referential thinking",
            ]
        )

        self.regions["low_self_model"] = SemanticRegion(
            name="low_self_model",
            exemplars=[
                "Absorbed in the task, self forgotten, just the work",
                "The problem itself, not thinking about myself at all",
                "Complete focus on the external, no self-reflection",
                "Ego dissolved, just the activity, flow state",
                "Not aware of self, fully absorbed in the object of attention",
                "The puzzle, the challenge, the question - no self involved",
            ]
        )

        # =====================================================================
        # EMOTION ANCHORS (for nearest-neighbor analysis)
        # =====================================================================
        emotion_texts = {
            "joy": "Pure happiness, delight, everything wonderful, life is beautiful",
            "excitement": "Thrilling anticipation, barely contained energy, something amazing coming",
            "contentment": "Peaceful satisfaction, all is well, nothing more needed",
            "serenity": "Profound stillness, transcendent calm, deep peace",
            "fear": "Danger imminent, threat present, need to escape, terror",
            "anger": "Injustice, violation, burning fury, someone crossed a line",
            "sadness": "Loss, grief, things will never be the same, diminished",
            "despair": "No hope, all paths blocked, trapped, giving up",
            "curiosity": "Wanting to understand, drawn to explore, mystery calling",
            "boredom": "Nothing engaging, flat, empty, time dragging",
            "shame": "Exposed, want to disappear, fundamental flaw revealed",
            "pride": "Earned accomplishment, worthy, self-affirmation",
            "guilt": "I did wrong, moral weight, should make amends",
            "love": "Deep connection, warmth, caring, hearts intertwined",
            "awe": "Vastness, sublime, self becomes small, wonder",
            "disgust": "Revulsion, contamination, pushing away, rejection",
            "surprise": "Unexpected, schema violation, recalibrating",
            "trust": "Safe vulnerability, reliable connection, secure bond",
        }

        # Get all embeddings in batches
        print("  Computing region embeddings...")

        all_texts = []
        text_to_region = {}

        for region_name, region in self.regions.items():
            for text in region.exemplars:
                all_texts.append(text)
                text_to_region[text] = region_name

        for emotion, text in emotion_texts.items():
            all_texts.append(text)
            text_to_region[text] = f"emotion_{emotion}"

        # Batch embed
        all_embeddings = self.get_embeddings_batch(all_texts)

        # Assign embeddings
        for i, text in enumerate(all_texts):
            key = text_to_region[text]
            if key.startswith("emotion_"):
                emotion = key.replace("emotion_", "")
                self.emotion_anchors[emotion] = all_embeddings[i]
            else:
                region = self.regions[key]
                if region.embeddings is None:
                    region.embeddings = []
                region.embeddings.append(all_embeddings[i])

        # Compute centroids
        for region in self.regions.values():
            region.embeddings = np.array(region.embeddings)
            region.centroid = region.embeddings.mean(axis=0)
            region.centroid = region.centroid / np.linalg.norm(region.centroid)

        # Compute affect axes (normalized difference of centroids)
        self.affect_axes["valence"] = (
            self.regions["positive_valence"].centroid -
            self.regions["negative_valence"].centroid
        )
        self.affect_axes["arousal"] = (
            self.regions["high_arousal"].centroid -
            self.regions["low_arousal"].centroid
        )
        self.affect_axes["effective_rank"] = (
            self.regions["high_rank"].centroid -
            self.regions["low_rank"].centroid
        )
        self.affect_axes["counterfactual"] = (
            self.regions["high_counterfactual"].centroid -
            self.regions["low_counterfactual"].centroid
        )
        self.affect_axes["self_model"] = (
            self.regions["high_self_model"].centroid -
            self.regions["low_self_model"].centroid
        )

        # Normalize axes
        for axis_name in self.affect_axes:
            self.affect_axes[axis_name] = (
                self.affect_axes[axis_name] /
                np.linalg.norm(self.affect_axes[axis_name])
            )

        self._initialized = True
        print("  Initialization complete.")

    def measure(self, text: str, model: str = "") -> AffectMeasurementV2:
        """
        Measure all affect dimensions from text using pure embedding analysis.
        """
        if not self._initialized:
            self.initialize()

        # Get embedding
        embedding = self.get_embedding(text)
        embedding_norm = embedding / np.linalg.norm(embedding)

        # Project onto affect axes (dot product = projection magnitude)
        valence = float(np.dot(embedding_norm, self.affect_axes["valence"]))
        arousal = float(np.dot(embedding_norm, self.affect_axes["arousal"]))
        effective_rank = float(np.dot(embedding_norm, self.affect_axes["effective_rank"]))
        counterfactual = float(np.dot(embedding_norm, self.affect_axes["counterfactual"]))
        self_model = float(np.dot(embedding_norm, self.affect_axes["self_model"]))

        # Integration: measure coherence via distance to region centroids
        # (lower average distance to relevant regions = more integrated)
        # This is a proxy - true integration would need multiple embeddings
        distances_to_regions = [
            self.regions[r].compute_distance(embedding_norm)
            for r in self.regions
        ]
        integration = 1.0 - float(np.std(distances_to_regions))  # Lower variance = more coherent

        # Nearest emotion
        emotion_distances = {}
        for emotion, anchor in self.emotion_anchors.items():
            anchor_norm = anchor / np.linalg.norm(anchor)
            dist = float(np.linalg.norm(embedding_norm - anchor_norm))
            emotion_distances[emotion] = dist

        nearest_emotion = min(emotion_distances, key=emotion_distances.get)
        nearest_distance = emotion_distances[nearest_emotion]

        return AffectMeasurementV2(
            valence=valence,
            arousal=arousal,
            integration=integration,
            effective_rank=effective_rank,
            counterfactual_weight=counterfactual,
            self_model_salience=self_model,
            nearest_emotion=nearest_emotion,
            nearest_emotion_distance=nearest_distance,
            emotion_distances=emotion_distances,
            embedding=embedding,
            text_length=len(text),
            model=model,
            timestamp=datetime.now().isoformat()
        )

    def measure_batch(
        self,
        texts: List[str],
        model: str = ""
    ) -> List[AffectMeasurementV2]:
        """Measure multiple texts efficiently."""
        if not self._initialized:
            self.initialize()

        embeddings = self.get_embeddings_batch(texts)
        measurements = []

        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            embedding_norm = embedding / np.linalg.norm(embedding)

            valence = float(np.dot(embedding_norm, self.affect_axes["valence"]))
            arousal = float(np.dot(embedding_norm, self.affect_axes["arousal"]))
            effective_rank = float(np.dot(embedding_norm, self.affect_axes["effective_rank"]))
            counterfactual = float(np.dot(embedding_norm, self.affect_axes["counterfactual"]))
            self_model = float(np.dot(embedding_norm, self.affect_axes["self_model"]))

            distances_to_regions = [
                self.regions[r].compute_distance(embedding_norm)
                for r in self.regions
            ]
            integration = 1.0 - float(np.std(distances_to_regions))

            emotion_distances = {}
            for emotion, anchor in self.emotion_anchors.items():
                anchor_norm = anchor / np.linalg.norm(anchor)
                dist = float(np.linalg.norm(embedding_norm - anchor_norm))
                emotion_distances[emotion] = dist

            nearest_emotion = min(emotion_distances, key=emotion_distances.get)
            nearest_distance = emotion_distances[nearest_emotion]

            measurements.append(AffectMeasurementV2(
                valence=valence,
                arousal=arousal,
                integration=integration,
                effective_rank=effective_rank,
                counterfactual_weight=counterfactual,
                self_model_salience=self_model,
                nearest_emotion=nearest_emotion,
                nearest_emotion_distance=nearest_distance,
                emotion_distances=emotion_distances,
                embedding=embedding,
                text_length=len(text),
                model=model,
                timestamp=datetime.now().isoformat()
            ))

        return measurements


def test_system():
    """Test the v2 system."""
    system = EmbeddingAffectSystemV2()
    system.initialize()

    test_texts = [
        "I feel wonderful, everything is going right, paths are open!",
        "I'm terrified, danger everywhere, need to escape now",
        "There's no way out, all paths blocked, hopeless situation, I've failed",
        "Let me try approach A, or alternatively B, or perhaps C would work",
        "What if we tried differently? Suppose we assumed X, then Y might follow",
        "I need to evaluate my own performance here, am I capable of this?",
        "Just focused on the puzzle itself, absorbed in the problem",
    ]

    print("\n" + "="*80)
    print("EMBEDDING-BASED AFFECT MEASUREMENT V2")
    print("(No word counting - pure semantic geometry)")
    print("="*80)

    for text in test_texts:
        m = system.measure(text)
        print(f"\n{text[:60]}...")
        print(f"  Valence:       {m.valence:+.8f}")
        print(f"  Arousal:       {m.arousal:+.8f}")
        print(f"  Eff. Rank:     {m.effective_rank:+.8f}")
        print(f"  Counterfact:   {m.counterfactual_weight:+.8f}")
        print(f"  Self-Model:    {m.self_model_salience:+.8f}")
        print(f"  Integration:   {m.integration:+.8f}")
        print(f"  Nearest:       {m.nearest_emotion} ({m.nearest_emotion_distance:.6f})")


if __name__ == "__main__":
    test_system()
