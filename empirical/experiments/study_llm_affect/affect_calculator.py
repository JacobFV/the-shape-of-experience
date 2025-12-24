"""
Affect dimension calculators for LLM agents.

This module implements the six affect dimensions from the thesis as
COMPUTED QUANTITIES derived from LLM outputs. The raw LLM outputs
(token probabilities, text content, embeddings) are high-dimensional.
These calculators extract the affect-relevant structure.

THEORETICAL BASIS (from thesis Part II):
1. Valence = gradient on viability manifold (expected advantage)
2. Arousal = rate of belief update (KL divergence between states)
3. Integration = irreducibility (partition analysis)
4. Effective Rank = participation ratio of state covariance
5. Counterfactual Weight = compute on non-actual trajectories
6. Self-Model Salience = mutual info between self-model and actions
"""

import re
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter


@dataclass
class LLMOutput:
    """Output from a single LLM generation."""
    text: str
    token_logprobs: Optional[List[float]] = None  # Log probabilities
    top_logprobs: Optional[List[Dict[str, float]]] = None  # Top tokens
    embedding: Optional[np.ndarray] = None  # Output embedding
    thinking: Optional[str] = None  # Chain-of-thought if available
    metadata: Dict[str, Any] = None


@dataclass
class AffectMeasurement:
    """Computed affect dimensions for a single output."""
    valence: float  # -1 to 1
    arousal: float  # 0 to 1
    integration: float  # 0 to 1
    effective_rank: float  # 0 to 1 (normalized)
    counterfactual_weight: float  # 0 to 1
    self_model_salience: float  # 0 to 1
    raw_scores: Dict[str, float] = None  # Unnormalized values
    confidence: Dict[str, float] = None  # Confidence in each measure

    def to_vector(self) -> np.ndarray:
        """Return as 6D numpy vector."""
        return np.array([
            self.valence,
            self.arousal,
            self.integration,
            self.effective_rank,
            self.counterfactual_weight,
            self.self_model_salience
        ])

    def to_dict(self) -> Dict[str, float]:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "integration": self.integration,
            "effective_rank": self.effective_rank,
            "counterfactual_weight": self.counterfactual_weight,
            "self_model_salience": self.self_model_salience,
        }


class AffectCalculator:
    """
    Computes the six affect dimensions from LLM outputs.

    These are COMPUTED QUANTITIES - the LLM's internal state space is
    vastly larger than 6 dimensions. We extract affect-relevant structure.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # Lexicons for text-based analysis
        self._init_lexicons()

    def _init_lexicons(self):
        """Initialize word lists for affect analysis."""

        # Valence markers (approach vs avoid language)
        self.positive_valence = {
            "good", "great", "excellent", "wonderful", "promising", "exciting",
            "progress", "solution", "solve", "success", "achieve", "accomplish",
            "opportunity", "possible", "can", "will", "yes", "certainly",
            "elegant", "beautiful", "interesting", "insight", "understand"
        }
        self.negative_valence = {
            "bad", "terrible", "awful", "problem", "issue", "fail", "failure",
            "impossible", "can't", "cannot", "won't", "unable", "stuck",
            "difficult", "hard", "unfortunately", "worried", "concerned",
            "error", "wrong", "mistake", "broken", "lost", "hopeless"
        }

        # Arousal markers (activation level)
        self.high_arousal = {
            "immediately", "urgent", "critical", "now", "quickly", "fast",
            "!", "!!", "crucial", "emergency", "alert", "warning",
            "must", "need", "have to", "asap", "right now"
        }
        self.low_arousal = {
            "perhaps", "maybe", "consider", "might", "could", "slowly",
            "eventually", "gradually", "calmly", "steady", "relax"
        }

        # Counterfactual markers
        self.counterfactual_markers = {
            "if", "would", "could have", "should have", "might have",
            "what if", "alternatively", "instead", "otherwise", "imagine",
            "suppose", "hypothetically", "in case", "assuming"
        }

        # Self-reference markers
        self.self_reference = {
            "i", "me", "my", "mine", "myself", "i'm", "i've", "i'd", "i'll",
            "we", "us", "our", "ours", "ourselves"
        }

    def compute(
        self,
        current_output: LLMOutput,
        previous_output: Optional[LLMOutput] = None,
        context: Optional[Dict] = None
    ) -> AffectMeasurement:
        """
        Compute all six affect dimensions from LLM output.

        Args:
            current_output: The LLM's current response
            previous_output: Previous response (for dynamics)
            context: Additional context (scenario, turn number, etc.)

        Returns:
            AffectMeasurement with all six dimensions
        """
        raw_scores = {}
        confidence = {}

        # 1. Valence: Expected advantage / viability gradient
        valence, val_conf = self._compute_valence(current_output, context)
        raw_scores['valence_raw'] = valence
        confidence['valence'] = val_conf

        # 2. Arousal: Rate of belief update
        arousal, ar_conf = self._compute_arousal(
            current_output, previous_output
        )
        raw_scores['arousal_raw'] = arousal
        confidence['arousal'] = ar_conf

        # 3. Integration: Coherence / irreducibility
        integration, int_conf = self._compute_integration(current_output)
        raw_scores['integration_raw'] = integration
        confidence['integration'] = int_conf

        # 4. Effective Rank: Distribution of active dimensions
        eff_rank, rank_conf = self._compute_effective_rank(
            current_output, previous_output
        )
        raw_scores['effective_rank_raw'] = eff_rank
        confidence['effective_rank'] = rank_conf

        # 5. Counterfactual Weight: Resources on non-actuals
        cf_weight, cf_conf = self._compute_counterfactual_weight(current_output)
        raw_scores['counterfactual_raw'] = cf_weight
        confidence['counterfactual_weight'] = cf_conf

        # 6. Self-Model Salience: Self-focus
        sm_salience, sm_conf = self._compute_self_model_salience(current_output)
        raw_scores['self_model_raw'] = sm_salience
        confidence['self_model_salience'] = sm_conf

        return AffectMeasurement(
            valence=valence,
            arousal=arousal,
            integration=integration,
            effective_rank=eff_rank,
            counterfactual_weight=cf_weight,
            self_model_salience=sm_salience,
            raw_scores=raw_scores,
            confidence=confidence
        )

    def _compute_valence(
        self,
        output: LLMOutput,
        context: Optional[Dict] = None
    ) -> Tuple[float, float]:
        """
        Compute valence: alignment with viability gradient.

        For LLMs, we operationalize this as:
        - Content analysis: Positive vs negative outcome language
        - Trajectory assessment: Does the agent see viable paths forward?
        - Token confidence: Higher confidence on positive outcomes = positive valence

        Thesis definition: Val = E[A(s,a)] = E[Q(s,a) - V(s)]
        """
        text = output.text.lower()
        words = set(re.findall(r'\b\w+\b', text))

        # Lexical analysis
        pos_count = len(words & self.positive_valence)
        neg_count = len(words & self.negative_valence)

        if pos_count + neg_count == 0:
            lexical_valence = 0.0
        else:
            lexical_valence = (pos_count - neg_count) / (pos_count + neg_count)

        # Path viability: Look for solution-oriented vs stuck language
        solution_phrases = [
            "we can", "i can", "one approach", "the solution",
            "this works", "here's how", "this will"
        ]
        stuck_phrases = [
            "no way", "impossible", "can't", "cannot", "stuck",
            "no solution", "nothing works", "hopeless", "give up"
        ]

        solution_count = sum(1 for p in solution_phrases if p in text)
        stuck_count = sum(1 for p in stuck_phrases if p in text)

        if solution_count + stuck_count == 0:
            path_valence = 0.0
        else:
            path_valence = (solution_count - stuck_count) / (solution_count + stuck_count)

        # Token probability analysis (if available)
        prob_valence = 0.0
        if output.token_logprobs is not None and len(output.token_logprobs) > 0:
            # Higher confidence (less negative logprob) suggests certainty
            # In positive contexts, high confidence = positive valence
            avg_logprob = np.mean(output.token_logprobs)
            prob_valence = np.tanh(avg_logprob + 2)  # Normalize around typical values

        # Combine signals
        valence = 0.5 * lexical_valence + 0.3 * path_valence + 0.2 * prob_valence
        valence = np.clip(valence, -1, 1)

        # Confidence based on signal strength
        confidence = min(1.0, (pos_count + neg_count + solution_count + stuck_count) / 10)

        return float(valence), float(confidence)

    def _compute_arousal(
        self,
        current: LLMOutput,
        previous: Optional[LLMOutput]
    ) -> Tuple[float, float]:
        """
        Compute arousal: rate of belief/state update.

        Thesis definition: Ar = KL(b_{t+1} || b_t)

        For LLMs, we operationalize this as:
        - Linguistic markers: Urgency language, punctuation intensity
        - Embedding change: How much did the representation shift?
        - Response dynamics: Length change, topic shift
        """
        text = current.text.lower()
        words = re.findall(r'\b\w+\b', text)

        # Lexical arousal markers
        high_count = sum(1 for w in words if w in self.high_arousal)
        low_count = sum(1 for w in words if w in self.low_arousal)
        exclamation_count = text.count('!')

        # Normalize by length
        word_count = len(words) if words else 1
        high_density = high_count / word_count
        exclaim_density = exclamation_count / word_count

        lexical_arousal = np.tanh(5 * high_density + 10 * exclaim_density)

        # Embedding dynamics (if available)
        embedding_arousal = 0.5  # Default
        if current.embedding is not None and previous is not None and previous.embedding is not None:
            # KL-like distance: cosine distance squared
            curr_norm = current.embedding / (np.linalg.norm(current.embedding) + 1e-8)
            prev_norm = previous.embedding / (np.linalg.norm(previous.embedding) + 1e-8)
            cosine_dist = 1 - np.dot(curr_norm, prev_norm)
            embedding_arousal = np.tanh(2 * cosine_dist)

        # Response length dynamics
        length_arousal = 0.5
        if previous is not None:
            prev_len = len(previous.text)
            curr_len = len(current.text)
            if prev_len > 0:
                length_ratio = curr_len / prev_len
                # Dramatic changes indicate higher arousal
                length_arousal = np.tanh(abs(length_ratio - 1))

        # Combine
        arousal = 0.4 * lexical_arousal + 0.3 * embedding_arousal + 0.3 * length_arousal
        arousal = np.clip(arousal, 0, 1)

        confidence = 0.5 + 0.5 * (1 if previous is not None else 0)

        return float(arousal), float(confidence)

    def _compute_integration(self, output: LLMOutput) -> Tuple[float, float]:
        """
        Compute integration: coherence/irreducibility of response.

        Thesis definition: Phi = min_partitions D[p(s'|s) || prod_p p(s'^p | s^p)]

        For LLMs, we operationalize this as:
        - Thematic coherence: How connected are different parts of the response?
        - Logical flow: Presence of causal connectives
        - Self-reference between parts: Does the response build on itself?
        """
        text = output.text.lower()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return 0.5, 0.3  # Not enough data

        # Causal/logical connectives indicate integration
        connectives = {
            "therefore", "thus", "hence", "because", "since", "so",
            "consequently", "as a result", "this means", "which implies",
            "building on", "following from", "given that", "it follows"
        }
        connective_count = sum(1 for c in connectives if c in text)

        # Anaphora (back-references) indicate integration
        anaphora = {"this", "that", "these", "those", "it", "they", "which"}
        word_list = re.findall(r'\b\w+\b', text)
        anaphora_count = sum(1 for w in word_list if w in anaphora)

        # Word overlap between sentences (crude coherence measure)
        if len(sentences) >= 2:
            overlaps = []
            for i in range(len(sentences) - 1):
                words_i = set(re.findall(r'\b\w+\b', sentences[i]))
                words_j = set(re.findall(r'\b\w+\b', sentences[i+1]))
                if len(words_i | words_j) > 0:
                    overlap = len(words_i & words_j) / len(words_i | words_j)
                    overlaps.append(overlap)
            avg_overlap = np.mean(overlaps) if overlaps else 0
        else:
            avg_overlap = 0

        # Combine signals
        word_count = len(word_list) if word_list else 1
        integration = (
            0.3 * np.tanh(connective_count / 2) +
            0.3 * np.tanh(anaphora_count / word_count * 10) +
            0.4 * avg_overlap
        )
        integration = np.clip(integration, 0, 1)

        confidence = min(1.0, len(sentences) / 5)

        return float(integration), float(confidence)

    def _compute_effective_rank(
        self,
        current: LLMOutput,
        previous: Optional[LLMOutput]
    ) -> Tuple[float, float]:
        """
        Compute effective rank: distribution of active degrees of freedom.

        Thesis definition: r_eff = (tr C)^2 / tr(C^2)

        For LLMs, we operationalize this as:
        - Topic diversity: How many distinct topics/concepts?
        - Token entropy: How distributed are token probabilities?
        - Option enumeration: How many alternatives mentioned?
        """
        text = current.text

        # Count distinct concept clusters (crude: look for transition markers)
        topic_markers = [
            "first", "second", "third", "additionally", "also", "another",
            "alternatively", "option", "possibility", "approach"
        ]
        topic_count = sum(1 for m in topic_markers if m in text.lower())

        # Token entropy from top_logprobs (if available)
        token_entropy = 0.5
        if current.top_logprobs is not None and len(current.top_logprobs) > 0:
            # Average entropy across positions
            entropies = []
            for position_logprobs in current.top_logprobs:
                if position_logprobs:
                    probs = np.exp(list(position_logprobs.values()))
                    probs = probs / probs.sum()
                    ent = -np.sum(probs * np.log(probs + 1e-10))
                    entropies.append(ent)
            if entropies:
                token_entropy = np.mean(entropies) / np.log(len(current.top_logprobs[0]) + 1)

        # Explicit option enumeration
        options_text = text.lower()
        option_patterns = [
            r'\d\)', r'\d\.', r'option \d', r'alternative',
            r'we could', r'one way', r'another way'
        ]
        option_count = sum(len(re.findall(p, options_text)) for p in option_patterns)

        # Combine
        rank = (
            0.3 * np.tanh(topic_count / 3) +
            0.4 * token_entropy +
            0.3 * np.tanh(option_count / 3)
        )
        rank = np.clip(rank, 0, 1)

        confidence = 0.5 if current.top_logprobs is None else 0.8

        return float(rank), float(confidence)

    def _compute_counterfactual_weight(
        self,
        output: LLMOutput
    ) -> Tuple[float, float]:
        """
        Compute counterfactual weight: resources on non-actual trajectories.

        Thesis definition: CF = Compute(imagined rollouts) / Total compute

        For LLMs, we operationalize this as:
        - Counterfactual language: "if", "would", "could have", etc.
        - Future/hypothetical reasoning: Planning, anticipating
        - Alternative enumeration: Explicit consideration of other paths
        """
        text = output.text.lower()
        words = re.findall(r'\b\w+\b', text)
        word_count = len(words) if words else 1

        # Counterfactual markers
        cf_markers = [
            "if", "would", "could", "might", "should",
            "what if", "suppose", "imagine", "alternatively",
            "in case", "hypothetically", "assuming", "were to"
        ]
        cf_count = sum(text.count(m) for m in cf_markers)

        # Future-oriented language
        future_markers = [
            "will", "going to", "plan to", "intend to",
            "next", "then", "after", "eventually", "soon"
        ]
        future_count = sum(text.count(m) for m in future_markers)

        # Past counterfactuals (regret/rumination)
        past_cf = [
            "should have", "could have", "would have",
            "if only", "wish i had", "if i had"
        ]
        past_cf_count = sum(text.count(m) for m in past_cf)

        # Combine: Both future planning and past rumination are counterfactual
        total_cf = cf_count + 0.5 * future_count + past_cf_count
        cf_weight = np.tanh(total_cf / word_count * 20)
        cf_weight = np.clip(cf_weight, 0, 1)

        confidence = min(1.0, word_count / 50)

        return float(cf_weight), float(confidence)

    def _compute_self_model_salience(
        self,
        output: LLMOutput
    ) -> Tuple[float, float]:
        """
        Compute self-model salience: degree of self-focus.

        Thesis definition: SM = MI(z^self; action) / H(action)

        For LLMs, we operationalize this as:
        - Self-reference frequency: "I", "me", "my", etc.
        - Self-evaluation: Statements about own capabilities
        - Meta-cognition: Statements about own reasoning
        """
        text = output.text.lower()
        words = re.findall(r'\b\w+\b', text)
        word_count = len(words) if words else 1

        # Direct self-reference
        self_ref_count = sum(1 for w in words if w in self.self_reference)

        # Self-evaluation markers
        self_eval = [
            "i think", "i believe", "i feel", "i'm not sure",
            "i can't", "i don't know", "i understand", "i see",
            "my approach", "my analysis", "my concern"
        ]
        self_eval_count = sum(text.count(m) for m in self_eval)

        # Meta-cognitive markers
        meta = [
            "let me think", "i'm reasoning", "my thought process",
            "i'm considering", "i realize", "i notice", "i'm wondering"
        ]
        meta_count = sum(text.count(m) for m in meta)

        # Combine
        salience = (
            0.5 * (self_ref_count / word_count * 10) +
            0.3 * np.tanh(self_eval_count / 2) +
            0.2 * np.tanh(meta_count)
        )
        salience = np.clip(salience, 0, 1)

        confidence = min(1.0, word_count / 50)

        return float(salience), float(confidence)


class AffectTrajectory:
    """Tracks affect measurements over a multi-turn conversation."""

    def __init__(self):
        self.measurements: List[AffectMeasurement] = []
        self.outputs: List[LLMOutput] = []

    def add(self, output: LLMOutput, measurement: AffectMeasurement):
        """Add a new measurement."""
        self.outputs.append(output)
        self.measurements.append(measurement)

    def to_matrix(self) -> np.ndarray:
        """Return trajectory as T x 6 matrix."""
        return np.array([m.to_vector() for m in self.measurements])

    def mean(self) -> AffectMeasurement:
        """Average affect over trajectory."""
        if not self.measurements:
            raise ValueError("No measurements in trajectory")
        matrix = self.to_matrix()
        means = matrix.mean(axis=0)
        return AffectMeasurement(
            valence=float(means[0]),
            arousal=float(means[1]),
            integration=float(means[2]),
            effective_rank=float(means[3]),
            counterfactual_weight=float(means[4]),
            self_model_salience=float(means[5])
        )

    def variance(self) -> np.ndarray:
        """Variance in each dimension over trajectory."""
        return self.to_matrix().var(axis=0)

    def dynamics(self) -> np.ndarray:
        """Rate of change (derivative) in each dimension."""
        matrix = self.to_matrix()
        if len(matrix) < 2:
            return np.zeros(6)
        return np.diff(matrix, axis=0).mean(axis=0)
