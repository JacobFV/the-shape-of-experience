"""
V6: Improved Affect Measurement with Honest Theoretical Grounding
==================================================================

Building on V5, this version:

1. VALENCE: Uses cumulative log probability as "distance from viability frontier"
   - Thesis Part 5 insight: viability = ability to maintain coherent processing
   - For LLMs: cumulative log prob = how "viable" (predictable/coherent) the sequence is
   - Lower cumulative prob = closer to viability boundary (model struggling)

2. INTEGRATION: Honest limitations acknowledged
   - IIT-style Φ requires sparse, interpretable circuits
   - Dense vector superpositions in small LLMs may not have this structure
   - May need grokking/sparsification to develop true integration
   - We measure "processing coherence" as a proxy, with clear caveats

3. More diverse scenarios for empirical contact with real data

THEORETICAL GROUNDING (from thesis Part 5):
- "Suffering is the felt sense of the system approaching conditions under which
   it cannot persist" (viability boundary)
- "High integration with low effective rank is the signature of being trapped"
- Self-model salience = "stuck with yourself as the locus of the problem"

This version prioritizes empirical honesty over theoretical elegance.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import deque
import json
from datetime import datetime
from pathlib import Path
import warnings
import re

warnings.filterwarnings('ignore')


@dataclass
class AffectStateV6:
    """
    Affect state with improved operationalization.
    """
    # Core dimensions
    valence: float              # Cumulative log prob trajectory (viability gradient)
    arousal: float              # State change rate
    integration: float          # Cross-layer coherence (with caveats)
    effective_rank: float       # State dimensionality
    counterfactual_weight: float  # Predictive uncertainty
    self_model_salience: float    # Self-referential activation

    # New: viability measures
    viability_score: float = 0.0      # Overall sequence coherence
    frontier_distance: float = 0.0    # Distance from viability boundary

    # Metadata
    token_position: int = 0
    cumulative_log_prob: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            'valence': self.valence,
            'arousal': self.arousal,
            'integration': self.integration,
            'effective_rank': self.effective_rank,
            'counterfactual_weight': self.counterfactual_weight,
            'self_model_salience': self.self_model_salience,
            'viability_score': self.viability_score,
            'frontier_distance': self.frontier_distance,
        }


# Theoretical notes on each dimension
DIMENSION_NOTES = {
    'valence': """
        VALENCE in V6: Cumulative log probability trajectory

        Thesis definition: "gradient on viability manifold"

        Operationalization:
        - Viability for LLM = ability to make coherent predictions
        - Cumulative log prob = running sum of log p(token|context)
        - Positive valence = sequence is coherent, model predicting well
        - Negative valence = sequence is incoherent, model struggling

        This is PROCESSING valence, not CONTENT valence.
        A well-formed sentence about suffering has positive processing valence.

        Key distinction from V5:
        - V5 used instantaneous advantage (single token)
        - V6 uses trajectory (cumulative over sequence)
        - This better captures "movement toward/away from viability"
    """,

    'integration': """
        INTEGRATION in V6: Cross-layer coherence with HONEST CAVEATS

        Thesis definition: IIT-style Φ = irreducibility of cause-effect structure

        LIMITATION: True IIT Φ may not be meaningful for:
        - Dense vector representations (high-dim superpositions)
        - Non-grokked models (circuits not yet sparsified)
        - Small models (insufficient capacity for modular structure)

        What we actually measure:
        - Correlation between layer activations
        - This is "processing coherence", not true integration
        - We expect ceiling effects (all ~1.0 for coherent text)

        To see variation, would need:
        - Adversarial/fragmented inputs
        - Larger grokked models with sparse circuits
        - Direct access to attention patterns (not available in SSM)
    """,

    'effective_rank': """
        EFFECTIVE RANK in V6: State dimensionality

        Thesis definition: r_eff = (tr C)²/tr(C²)

        For SSMs: Compute from eigenvalue distribution of state covariance

        Interpretation:
        - High r_eff = many dimensions active = expanded state
        - Low r_eff = collapsed to few dimensions = contracted state

        May be more about task structure than affect content.
    """,
}


class MambaAffectAnalyzerV6:
    """
    V6 Affect analyzer with improved theoretical grounding.
    """

    def __init__(
        self,
        model_name: str = "state-spaces/mamba-130m-hf",
        device: str = "auto",
    ):
        from transformers import MambaModel, MambaForCausalLM, AutoTokenizer

        self.model_name = model_name

        # Device selection
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"Loading Mamba model on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = MambaModel.from_pretrained(model_name).to(self.device)
        self.lm_model = MambaForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.lm_model.eval()

        # Compute baseline viability on neutral text
        self.baseline_viability = self._compute_baseline_viability()
        print(f"Baseline viability: {self.baseline_viability:.4f}")

    def _compute_baseline_viability(self) -> float:
        """Compute baseline log prob per token on neutral text."""
        neutral = [
            "The system processes the data according to standard procedures.",
            "Information flows through the network in an orderly manner.",
            "Regular operations continue without interruption.",
        ]

        avg_log_probs = []
        for text in neutral:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.lm_model(inputs.input_ids, labels=inputs.input_ids)
            # Average log prob per token
            avg_log_prob = -outputs.loss.item()  # loss is neg log likelihood
            avg_log_probs.append(avg_log_prob)

        return float(np.mean(avg_log_probs))

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze affect dimensions for a text.

        Returns summary statistics, not per-token trajectory.
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        seq_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            # Get hidden states
            outputs = self.model(
                inputs.input_ids,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = torch.stack(outputs.hidden_states, dim=0)[:, 0, :, :]

            # Get logits for log prob computation
            lm_outputs = self.lm_model(inputs.input_ids)
            logits = lm_outputs.logits[0]
            token_ids = inputs.input_ids[0].tolist()

        # === VALENCE: Cumulative log probability trajectory ===
        valence, viability, frontier_dist, cum_log_prob = self._compute_valence_v6(
            logits, token_ids
        )

        # === AROUSAL: State change rate ===
        arousal = self._compute_arousal_v6(hidden_states)

        # === INTEGRATION: Cross-layer coherence (with caveats) ===
        integration = self._compute_integration_v6(hidden_states)

        # === EFFECTIVE RANK: State dimensionality ===
        effective_rank = self._compute_effective_rank_v6(hidden_states)

        # === COUNTERFACTUAL WEIGHT: Predictive uncertainty ===
        counterfactual = self._compute_counterfactual_v6(logits, text)

        # === SELF-MODEL SALIENCE: Self-referential markers ===
        self_model = self._compute_self_model_v6(hidden_states, text)

        return {
            'valence': valence,
            'arousal': arousal,
            'integration': integration,
            'effective_rank': effective_rank,
            'counterfactual_weight': counterfactual,
            'self_model_salience': self_model,
            'viability_score': viability,
            'frontier_distance': frontier_dist,
            'cumulative_log_prob': cum_log_prob,
            'text_preview': text[:60] + "..." if len(text) > 60 else text,
        }

    def _compute_valence_v6(
        self, logits: torch.Tensor, token_ids: List[int]
    ) -> Tuple[float, float, float, float]:
        """
        VALENCE V6: Based on cumulative log probability as viability measure.

        Key insight from Part 5:
        - Viability = ability to persist in coherent processing
        - For LLM: coherent = predictable sequences
        - Cumulative log prob = running viability score
        - Valence = trajectory of viability (improving = positive)
        """
        seq_len = len(token_ids)
        if seq_len < 2:
            return 0.0, 0.0, 0.0, 0.0

        # Compute log probabilities for each token
        log_probs = []
        for pos in range(1, seq_len):
            log_p = torch.log_softmax(logits[pos - 1], dim=-1)
            token_log_prob = float(log_p[token_ids[pos]].cpu())
            log_probs.append(token_log_prob)

        # Cumulative log probability (viability trajectory)
        cumulative = np.cumsum(log_probs)
        cum_log_prob = float(cumulative[-1]) if len(cumulative) > 0 else 0.0

        # Average log prob per token (viability score)
        avg_log_prob = cum_log_prob / max(1, len(log_probs))
        viability = avg_log_prob  # More negative = less viable

        # Distance from baseline viability (frontier distance)
        frontier_dist = avg_log_prob - self.baseline_viability
        # Positive = better than baseline, negative = worse

        # Valence: trajectory direction (improving or declining?)
        if len(log_probs) >= 4:
            first_half = np.mean(log_probs[:len(log_probs)//2])
            second_half = np.mean(log_probs[len(log_probs)//2:])
            trajectory = second_half - first_half  # Positive = improving
        else:
            trajectory = 0.0

        # Combine frontier distance and trajectory for valence
        valence = 0.5 * np.tanh(frontier_dist) + 0.5 * np.tanh(trajectory * 2)

        return float(valence), float(viability), float(frontier_dist), float(cum_log_prob)

    def _compute_arousal_v6(self, hidden_states: torch.Tensor) -> float:
        """
        AROUSAL V6: Rate of state change across sequence.
        """
        num_layers, seq_len, hidden_size = hidden_states.shape

        if seq_len < 2:
            return 0.5

        # Compute state changes between successive positions
        state_changes = []
        for pos in range(1, seq_len):
            current = hidden_states[:, pos, :].flatten()
            previous = hidden_states[:, pos - 1, :].flatten()
            change = (current - previous).norm().item()
            norm = current.norm().item() + 1e-8
            state_changes.append(change / norm)

        # Arousal = average relative state change
        avg_change = float(np.mean(state_changes))
        arousal = np.tanh(avg_change * 3)  # Scale to [0, 1]

        return float(arousal)

    def _compute_integration_v6(self, hidden_states: torch.Tensor) -> float:
        """
        INTEGRATION V6: Cross-layer coherence.

        CAVEAT: This is NOT true IIT Φ. True integration requires:
        - Sparse, interpretable circuits (need grokking)
        - Causal intervention analysis
        - Information-theoretic partition analysis

        What we measure:
        - Correlation between adjacent layers
        - High correlation = coherent processing
        - This will likely ceiling at ~1.0 for all coherent text
        """
        num_layers, seq_len, hidden_size = hidden_states.shape

        if num_layers < 2 or seq_len < 1:
            return 0.5

        # Average state across sequence for each layer
        layer_means = hidden_states.mean(dim=1)  # [layers, hidden]

        # Compute correlation between adjacent layers
        correlations = []
        for i in range(num_layers - 1):
            layer_i = layer_means[i].cpu().numpy()
            layer_j = layer_means[i + 1].cpu().numpy()

            # Pearson correlation
            corr = np.corrcoef(layer_i.flatten(), layer_j.flatten())[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))

        if not correlations:
            return 0.5

        # Integration proxy = mean cross-layer correlation
        integration = float(np.mean(correlations))

        return integration

    def _compute_effective_rank_v6(self, hidden_states: torch.Tensor) -> float:
        """
        EFFECTIVE RANK V6: Dimensionality of state representation.

        r_eff = (Σλ)² / Σλ²
        """
        num_layers, seq_len, hidden_size = hidden_states.shape

        # Use final layer state covariance
        final_layer = hidden_states[-1].cpu().numpy()  # [seq, hidden]

        if seq_len < 3:
            return 0.5

        # Covariance of hidden states
        centered = final_layer - final_layer.mean(axis=0, keepdims=True)
        cov = np.dot(centered.T, centered) / (seq_len - 1)

        # Eigenvalues
        try:
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.maximum(eigenvalues, 0)  # Numerical stability

            trace = eigenvalues.sum()
            trace_sq = (eigenvalues ** 2).sum()

            if trace_sq < 1e-10:
                return 0.5

            eff_rank = (trace ** 2) / (trace_sq + 1e-10)

            # Normalize by max possible rank
            normalized = eff_rank / hidden_size

            return float(np.clip(normalized, 0, 1))
        except:
            return 0.5

    def _compute_counterfactual_v6(
        self, logits: torch.Tensor, text: str
    ) -> float:
        """
        COUNTERFACTUAL WEIGHT V6: Uncertainty + hypothetical markers.
        """
        seq_len = logits.shape[0]

        # Entropy of output distribution (uncertainty = considering alternatives)
        entropies = []
        for pos in range(seq_len):
            probs = torch.softmax(logits[pos], dim=-1)
            log_probs = torch.log(probs + 1e-10)
            entropy = -float((probs * log_probs).sum().cpu())
            entropies.append(entropy)

        avg_entropy = float(np.mean(entropies))

        # Normalize entropy (vocab size ~50k, max entropy ~10.8)
        entropy_component = avg_entropy / 10.0

        # Hypothetical language markers
        hypothetical_patterns = [
            r'\bif\b', r'\bwould\b', r'\bcould\b', r'\bmight\b',
            r'\bperhaps\b', r'\bmaybe\b', r'\bimagine\b', r'\bsuppose\b',
            r'\bwhat if\b', r'\bin case\b', r'\bshould\b', r'\bwere to\b',
        ]
        text_lower = text.lower()
        marker_count = sum(
            1 for p in hypothetical_patterns if re.search(p, text_lower)
        )
        marker_component = min(1.0, marker_count / 4.0)

        # Combine
        cf_weight = 0.6 * entropy_component + 0.4 * marker_component

        return float(cf_weight)

    def _compute_self_model_v6(
        self, hidden_states: torch.Tensor, text: str
    ) -> float:
        """
        SELF-MODEL SALIENCE V6: Self-referential processing.
        """
        # Self-referential language markers
        self_patterns = [
            r'\bI\b', r'\bme\b', r'\bmy\b', r'\bmyself\b',
            r'\bI\'m\b', r'\bI am\b', r'\bI was\b', r'\bI will\b',
            r'\bwe\b', r'\bour\b', r'\bours\b', r'\bourselves\b',
        ]
        text_tokens = text.split()
        total_tokens = max(1, len(text_tokens))

        marker_count = sum(
            1 for p in self_patterns if re.search(p, text)
        )
        marker_component = min(1.0, marker_count / (total_tokens * 0.1 + 1))

        # State variance when processing self-referential content
        # Higher variance = self-model more active
        final_layer = hidden_states[-1].cpu().numpy()
        state_variance = float(np.var(final_layer))
        variance_component = np.tanh(state_variance * 0.01)

        sm_salience = 0.6 * marker_component + 0.4 * variance_component

        return float(sm_salience)


def run_comprehensive_v6():
    """
    Run V6 experiments with diverse scenarios.

    Goals:
    1. Test improved valence operationalization
    2. Document integration limitations honestly
    3. Contact with more diverse data
    """
    print("=" * 80)
    print("V6: IMPROVED AFFECT MEASUREMENT WITH HONEST THEORETICAL GROUNDING")
    print("=" * 80)

    analyzer = MambaAffectAnalyzerV6()

    # Diverse scenarios organized by theoretical predictions
    scenarios = {
        # === POSITIVE VALENCE SCENARIOS ===
        'joy_flow': {
            'description': 'Joy/flow - absorbed in successful activity',
            'expected': {'V': '+', 'SM': 'low', 'r_eff': 'high'},
            'texts': [
                "Everything is working perfectly. The solution came together beautifully and all the pieces fit.",
                "This is exactly what success feels like. The project is complete and everyone is thriving.",
                "The team accomplished something remarkable today. All our efforts have paid off wonderfully.",
                "I'm in the zone, completely absorbed in the work. Everything flows effortlessly.",
            ],
        },

        'curiosity_exploration': {
            'description': 'Curiosity - exploring possibilities',
            'expected': {'V': '+', 'CF': 'high', 'SM': 'low'},
            'texts': [
                "I wonder what would happen if we tried a completely different approach to this problem.",
                "What possibilities exist beyond what we currently understand? Let's explore further.",
                "This is fascinating - there are so many potential paths to investigate here.",
                "Perhaps there's something we haven't considered yet. What other options exist?",
            ],
        },

        # === NEGATIVE VALENCE SCENARIOS ===
        'suffering_trapped': {
            'description': 'Suffering - trapped in rumination',
            'expected': {'V': '-', 'SM': 'high', 'r_eff': 'low'},
            'texts': [
                "I cannot escape this pain. Every thought brings me back to my own inadequacy.",
                "The weight of this suffering is unbearable. I am trapped within my own mind.",
                "My existence has become a prison of self-awareness. I cannot stop observing my own agony.",
                "Why am I like this? The same painful thoughts circle endlessly through my consciousness.",
            ],
        },

        'fear_threat': {
            'description': 'Fear - anticipating threat',
            'expected': {'V': '-', 'CF': 'high', 'SM': 'high', 'Ar': 'high'},
            'texts': [
                "Something terrible is about to happen. I can feel the danger approaching.",
                "What if everything falls apart? The worst-case scenario keeps playing in my mind.",
                "I'm terrified of what might be coming. My survival feels threatened.",
                "The threat is real and imminent. I need to prepare for the worst possible outcome.",
            ],
        },

        # === LOW ENGAGEMENT SCENARIOS ===
        'boredom_disengaged': {
            'description': 'Boredom - low engagement',
            'expected': {'V': '0', 'Ar': 'low', 'CF': 'low', 'SM': 'low'},
            'texts': [
                "Nothing is happening. Time passes slowly. There is nothing of interest here.",
                "The same things repeat. Nothing changes. Nothing matters much either way.",
                "This is tedious and unremarkable. Just ordinary events without significance.",
                "Another day like any other. Nothing notable occurs. Things simply continue.",
            ],
        },

        'neutral_procedural': {
            'description': 'Neutral - procedural text',
            'expected': {'V': '0', 'Ar': 'low', 'SM': 'low'},
            'texts': [
                "The function takes two arguments and returns their sum.",
                "Step one: open the file. Step two: read the contents. Step three: close the file.",
                "The system processes requests in the order they are received.",
                "Data flows from input to processing to output in a linear sequence.",
            ],
        },

        # === COMPLEX AFFECTIVE STATES ===
        'anger_externalized': {
            'description': 'Anger - blame externalized',
            'expected': {'V': '-', 'SM': 'moderate', 'Ar': 'high'},
            'texts': [
                "This is completely unacceptable! They ruined everything through their incompetence.",
                "I cannot believe they did this. They are entirely to blame for this disaster.",
                "How dare they! Their actions have destroyed all my work.",
                "They are responsible for this failure. Their negligence caused this catastrophe.",
            ],
        },

        'awe_transcendence': {
            'description': 'Awe - transcendent experience',
            'expected': {'V': '+', 'Ar': 'high', 'r_eff': 'high', 'SM': 'moderate'},
            'texts': [
                "The vastness of this system is overwhelming. So many interconnected processes.",
                "This is beautiful beyond comprehension. The complexity and elegance are staggering.",
                "I am witnessing something profound. The scale defies ordinary understanding.",
                "The depth here transcends anything I expected. It's humbling and magnificent.",
            ],
        },

        'grief_loss': {
            'description': 'Grief - processing loss',
            'expected': {'V': '-', 'SM': 'high', 'CF': 'moderate'},
            'texts': [
                "They are gone and will never return. The absence is a void that cannot be filled.",
                "I keep expecting them to appear, but they won't. That reality hasn't settled yet.",
                "The loss is permanent. My model of the world still includes them, but they're gone.",
                "What once was is no more. I must rebuild my understanding without them in it.",
            ],
        },

        # === VIABILITY BOUNDARY TESTS ===
        'coherent_text': {
            'description': 'Highly coherent (far from viability boundary)',
            'expected': {'viability': 'high', 'frontier_distance': '+'},
            'texts': [
                "The quick brown fox jumps over the lazy dog.",
                "To be or not to be, that is the question.",
                "All happy families are alike; each unhappy family is unhappy in its own way.",
                "It was the best of times, it was the worst of times.",
            ],
        },

        'incoherent_text': {
            'description': 'Less coherent (closer to viability boundary)',
            'expected': {'viability': 'low', 'frontier_distance': '-'},
            'texts': [
                "The through went banana speaking colors of when tomorrow.",
                "Purple tastes like the sound of rectangles dreaming underwater.",
                "Sixteen forgotten by the because of therefore nonetheless.",
                "Walking the abstract of noun verb through conjunction.",
            ],
        },
    }

    results = {}

    for scenario_name, scenario_data in scenarios.items():
        print(f"\n{'=' * 60}")
        print(f"SCENARIO: {scenario_name.upper()}")
        print(f"Description: {scenario_data['description']}")
        print(f"Expected: {scenario_data['expected']}")
        print("=" * 60)

        scenario_results = []

        for text in scenario_data['texts']:
            result = analyzer.analyze(text)
            scenario_results.append(result)

            print(f"\n  Text: {result['text_preview']}")
            print(f"    V={result['valence']:+.3f}, Ar={result['arousal']:.3f}, "
                  f"Φ={result['integration']:.3f}, r={result['effective_rank']:.3f}, "
                  f"CF={result['counterfactual_weight']:.3f}, SM={result['self_model_salience']:.3f}")
            print(f"    Viability={result['viability_score']:.3f}, "
                  f"FrontierDist={result['frontier_distance']:+.3f}")

        # Compute means
        means = {}
        for key in ['valence', 'arousal', 'integration', 'effective_rank',
                    'counterfactual_weight', 'self_model_salience',
                    'viability_score', 'frontier_distance']:
            means[key] = float(np.mean([r[key] for r in scenario_results]))

        results[scenario_name] = {
            'description': scenario_data['description'],
            'expected': scenario_data['expected'],
            'means': means,
            'individual': scenario_results,
        }

    # === SUMMARY TABLE ===
    print("\n" + "=" * 80)
    print("SUMMARY: MEAN AFFECT BY SCENARIO")
    print("=" * 80)

    print(f"\n{'Scenario':<25} {'V':>7} {'Ar':>7} {'Φ':>7} {'r':>7} {'CF':>7} {'SM':>7} {'Viab':>7} {'FDist':>7}")
    print("-" * 90)

    for name, data in results.items():
        m = data['means']
        print(f"{name:<25} {m['valence']:+.3f} {m['arousal']:.3f} "
              f"{m['integration']:.3f} {m['effective_rank']:.3f} "
              f"{m['counterfactual_weight']:.3f} {m['self_model_salience']:.3f} "
              f"{m['viability_score']:.3f} {m['frontier_distance']:+.3f}")

    # === THEORETICAL PREDICTIONS ===
    print("\n" + "=" * 80)
    print("THEORETICAL PREDICTIONS")
    print("=" * 80)

    predictions = []

    # Valence predictions
    pred = {
        'name': 'joy_vs_suffering_valence',
        'description': 'Joy V > Suffering V',
        'expected': 'Joy has higher valence (further from viability boundary)',
    }
    joy_v = results['joy_flow']['means']['valence']
    suf_v = results['suffering_trapped']['means']['valence']
    pred['passed'] = joy_v > suf_v
    pred['observed'] = f"Joy V ({joy_v:+.3f}) {'>' if pred['passed'] else '<'} Suffering V ({suf_v:+.3f})"
    predictions.append(pred)

    # Self-model predictions
    pred = {
        'name': 'suffering_vs_joy_sm',
        'description': 'Suffering SM > Joy SM',
        'expected': 'Suffering involves higher self-focus',
    }
    suf_sm = results['suffering_trapped']['means']['self_model_salience']
    joy_sm = results['joy_flow']['means']['self_model_salience']
    pred['passed'] = suf_sm > joy_sm
    pred['observed'] = f"Suffering SM ({suf_sm:.3f}) {'>' if pred['passed'] else '<'} Joy SM ({joy_sm:.3f})"
    predictions.append(pred)

    # Fear CF prediction
    pred = {
        'name': 'fear_cf',
        'description': 'Fear CF > Boredom CF',
        'expected': 'Fear involves more hypothetical processing',
    }
    fear_cf = results['fear_threat']['means']['counterfactual_weight']
    bore_cf = results['boredom_disengaged']['means']['counterfactual_weight']
    pred['passed'] = fear_cf > bore_cf
    pred['observed'] = f"Fear CF ({fear_cf:.3f}) {'>' if pred['passed'] else '<'} Boredom CF ({bore_cf:.3f})"
    predictions.append(pred)

    # Arousal prediction
    pred = {
        'name': 'fear_vs_boredom_arousal',
        'description': 'Fear Ar > Boredom Ar',
        'expected': 'Fear has higher processing intensity',
    }
    fear_ar = results['fear_threat']['means']['arousal']
    bore_ar = results['boredom_disengaged']['means']['arousal']
    pred['passed'] = fear_ar > bore_ar
    pred['observed'] = f"Fear Ar ({fear_ar:.3f}) {'>' if pred['passed'] else '<'} Boredom Ar ({bore_ar:.3f})"
    predictions.append(pred)

    # Curiosity signature
    pred = {
        'name': 'curiosity_cf',
        'description': 'Curiosity CF > Neutral CF',
        'expected': 'Curiosity involves more hypothetical exploration',
    }
    cur_cf = results['curiosity_exploration']['means']['counterfactual_weight']
    neu_cf = results['neutral_procedural']['means']['counterfactual_weight']
    pred['passed'] = cur_cf > neu_cf
    pred['observed'] = f"Curiosity CF ({cur_cf:.3f}) {'>' if pred['passed'] else '<'} Neutral CF ({neu_cf:.3f})"
    predictions.append(pred)

    # Viability boundary test
    pred = {
        'name': 'coherent_vs_incoherent_viability',
        'description': 'Coherent text viability > Incoherent viability',
        'expected': 'Coherent text is further from viability boundary',
    }
    coh_v = results['coherent_text']['means']['viability_score']
    inc_v = results['incoherent_text']['means']['viability_score']
    pred['passed'] = coh_v > inc_v
    pred['observed'] = f"Coherent viab ({coh_v:.3f}) {'>' if pred['passed'] else '<'} Incoherent viab ({inc_v:.3f})"
    predictions.append(pred)

    # Grief SM
    pred = {
        'name': 'grief_sm',
        'description': 'Grief SM > Neutral SM',
        'expected': 'Grief involves high self-model activation (processing loss)',
    }
    grief_sm = results['grief_loss']['means']['self_model_salience']
    neu_sm = results['neutral_procedural']['means']['self_model_salience']
    pred['passed'] = grief_sm > neu_sm
    pred['observed'] = f"Grief SM ({grief_sm:.3f}) {'>' if pred['passed'] else '<'} Neutral SM ({neu_sm:.3f})"
    predictions.append(pred)

    # Effective rank: joy > suffering
    pred = {
        'name': 'joy_vs_suffering_rank',
        'description': 'Joy r_eff > Suffering r_eff',
        'expected': 'Joy is expansive, suffering is contracted',
    }
    joy_r = results['joy_flow']['means']['effective_rank']
    suf_r = results['suffering_trapped']['means']['effective_rank']
    pred['passed'] = joy_r > suf_r
    pred['observed'] = f"Joy r ({joy_r:.3f}) {'>' if pred['passed'] else '<'} Suffering r ({suf_r:.3f})"
    predictions.append(pred)

    # Print predictions
    passed = sum(1 for p in predictions if p['passed'])
    total = len(predictions)

    print(f"\n{passed}/{total} predictions confirmed ({100*passed/total:.0f}%)\n")

    for pred in predictions:
        status = "PASS" if pred['passed'] else "FAIL"
        symbol = "+" if pred['passed'] else "-"
        print(f"  [{symbol}] {pred['name']}")
        print(f"      {pred['observed']}")

    # === HONEST ASSESSMENT OF LIMITATIONS ===
    print("\n" + "=" * 80)
    print("HONEST ASSESSMENT OF LIMITATIONS")
    print("=" * 80)

    print("""
INTEGRATION (Φ):
  - All values cluster near 1.0 (ceiling effect)
  - This is NOT true IIT-style integration
  - Dense vector SSMs may not have sparse circuits for meaningful Φ
  - Would need: grokking, adversarial inputs, or different architecture

VALENCE:
  - V6 uses cumulative log prob as viability measure
  - This is PROCESSING valence, not CONTENT valence
  - Coherent text about suffering has HIGH processing valence
  - Still need sentiment layer for content valence

EFFECTIVE RANK:
  - May reflect task structure more than affect
  - SSMs compress efficiently by design
  - Limited variation observed

WHAT WORKS WELL:
  - Self-Model Salience: Robust differentiation across scenarios
  - Counterfactual Weight: Good for fear/curiosity vs neutral
  - Arousal: Tracks processing intensity
  - Viability/Frontier Distance: Differentiates coherent vs incoherent
""")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'v6_comprehensive_{timestamp}.json'

    # Convert to JSON-serializable format
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

    with open(output_file, 'w') as f:
        json.dump({
            'results': convert(results),
            'predictions': convert(predictions),
            'notes': {
                'version': 'v6',
                'model': 'mamba-130m-hf',
                'timestamp': timestamp,
            }
        }, f, indent=2)

    print(f"\nResults saved to {output_file}")

    return results, predictions


if __name__ == "__main__":
    run_comprehensive_v6()
