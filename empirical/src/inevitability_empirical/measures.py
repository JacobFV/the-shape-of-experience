"""
Operational measures for the six affect dimensions.

Each dimension has multiple measurement modalities:
- Self-report: Experience sampling items
- Behavioral: Task-based and observational measures
- Physiological: Autonomic and hormonal markers
- Neural: EEG, fMRI, etc.

The thesis is falsified if:
1. The six dimensions are not empirically distinguishable (factor analysis fails)
2. Predicted correlations between dimensions and external variables fail
3. Cross-modal convergence fails (self-report doesn't match physiology)
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Callable
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class MeasurementResult:
    """Result from a single measurement."""
    dimension: str
    value: float
    confidence: float
    modality: str  # "self_report", "behavioral", "physiological", "neural"
    raw_data: Optional[Any] = None
    timestamp: Optional[float] = None


class DimensionMeasure(ABC):
    """Abstract base class for measuring an affect dimension."""

    @property
    @abstractmethod
    def dimension_name(self) -> str:
        pass

    @abstractmethod
    def measure_self_report(self, responses: Dict[str, Any]) -> MeasurementResult:
        """Compute dimension score from self-report items."""
        pass

    @abstractmethod
    def get_self_report_items(self) -> List[Dict[str, Any]]:
        """Return the self-report items for this dimension."""
        pass


class ValenceMeasure(DimensionMeasure):
    """
    Measure valence: the good/bad, approach/avoid dimension.

    Key theoretical claim: Valence = gradient alignment on viability manifold.
    Falsification: If valence doesn't correlate with objective safety/threat.
    """

    @property
    def dimension_name(self) -> str:
        return "valence"

    def get_self_report_items(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "val_pleasantness",
                "text": "How pleasant or unpleasant do you feel right now?",
                "type": "slider",
                "min": -3, "max": 3,
                "anchors": ["Very unpleasant", "Neutral", "Very pleasant"],
                "weight": 1.0,
            },
            {
                "id": "val_approach",
                "text": "Do you want this current state to continue or end?",
                "type": "slider",
                "min": -3, "max": 3,
                "anchors": ["Want it to end", "Neutral", "Want it to continue"],
                "weight": 0.8,
            },
            {
                "id": "val_wellbeing",
                "text": "How would you rate your overall wellbeing right now?",
                "type": "slider",
                "min": 0, "max": 10,
                "anchors": ["Worst possible", "Best possible"],
                "weight": 0.6,
            },
        ]

    def measure_self_report(self, responses: Dict[str, Any]) -> MeasurementResult:
        items = self.get_self_report_items()
        weighted_sum = 0.0
        weight_total = 0.0

        for item in items:
            if item["id"] in responses:
                raw = responses[item["id"]]
                # Normalize to [-1, 1]
                normalized = (raw - item["min"]) / (item["max"] - item["min"]) * 2 - 1
                weighted_sum += normalized * item["weight"]
                weight_total += item["weight"]

        value = weighted_sum / weight_total if weight_total > 0 else 0.0
        confidence = weight_total / sum(i["weight"] for i in items)

        return MeasurementResult(
            dimension="valence",
            value=value,
            confidence=confidence,
            modality="self_report",
            raw_data=responses,
        )

    @staticmethod
    def from_physiology(hrv: float, cortisol: float, scl: float) -> MeasurementResult:
        """
        Estimate valence from physiological signals.

        Higher HRV, lower cortisol, lower tonic SCL → more positive valence.
        This is a testable prediction of the thesis.
        """
        # Normalize each signal (these would be z-scored in practice)
        hrv_norm = np.tanh(hrv / 50)  # Higher HRV = positive
        cortisol_norm = -np.tanh(cortisol / 20)  # Lower cortisol = positive
        scl_norm = -np.tanh(scl / 10)  # Lower SCL = positive

        value = (hrv_norm + cortisol_norm + scl_norm) / 3
        return MeasurementResult(
            dimension="valence",
            value=float(np.clip(value, -1, 1)),
            confidence=0.6,  # Lower confidence for physiology
            modality="physiological",
            raw_data={"hrv": hrv, "cortisol": cortisol, "scl": scl},
        )


class ArousalMeasure(DimensionMeasure):
    """
    Measure arousal: the activation/energy dimension.

    Key theoretical claim: Arousal = rate of belief update (KL divergence over time).
    Falsification: If arousal doesn't correlate with prediction error magnitude.
    """

    @property
    def dimension_name(self) -> str:
        return "arousal"

    def get_self_report_items(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "ar_activation",
                "text": "How activated or calm do you feel right now?",
                "type": "slider",
                "min": 0, "max": 6,
                "anchors": ["Very calm", "Very activated"],
                "weight": 1.0,
            },
            {
                "id": "ar_energy",
                "text": "How much energy do you have right now?",
                "type": "slider",
                "min": 0, "max": 6,
                "anchors": ["No energy", "Full of energy"],
                "weight": 0.8,
            },
            {
                "id": "ar_alertness",
                "text": "How alert or drowsy are you?",
                "type": "slider",
                "min": 0, "max": 6,
                "anchors": ["Very drowsy", "Very alert"],
                "weight": 0.7,
            },
        ]

    def measure_self_report(self, responses: Dict[str, Any]) -> MeasurementResult:
        items = self.get_self_report_items()
        weighted_sum = 0.0
        weight_total = 0.0

        for item in items:
            if item["id"] in responses:
                raw = responses[item["id"]]
                normalized = (raw - item["min"]) / (item["max"] - item["min"])
                weighted_sum += normalized * item["weight"]
                weight_total += item["weight"]

        value = weighted_sum / weight_total if weight_total > 0 else 0.5
        confidence = weight_total / sum(i["weight"] for i in items)

        return MeasurementResult(
            dimension="arousal",
            value=value,
            confidence=confidence,
            modality="self_report",
            raw_data=responses,
        )

    @staticmethod
    def from_physiology(heart_rate: float, scr: float, pupil: float) -> MeasurementResult:
        """
        Estimate arousal from physiological signals.

        Higher HR, higher phasic SCR, larger pupils → higher arousal.
        """
        hr_norm = np.tanh((heart_rate - 70) / 30)
        scr_norm = np.tanh(scr / 5)
        pupil_norm = np.tanh((pupil - 4) / 2)

        value = (hr_norm + scr_norm + pupil_norm) / 3
        value = (value + 1) / 2  # Map to [0, 1]

        return MeasurementResult(
            dimension="arousal",
            value=float(np.clip(value, 0, 1)),
            confidence=0.7,
            modality="physiological",
            raw_data={"heart_rate": heart_rate, "scr": scr, "pupil": pupil},
        )


class IntegrationMeasure(DimensionMeasure):
    """
    Measure integration: the unity/fragmentation dimension.

    Key theoretical claim: Integration (Φ) = irreducibility of cause-effect structure.
    Falsification: If neural integration doesn't correlate with experiential unity.
    """

    @property
    def dimension_name(self) -> str:
        return "integration"

    def get_self_report_items(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "int_unity",
                "text": "How unified or fragmented does your experience feel?",
                "type": "slider",
                "min": 0, "max": 6,
                "anchors": ["Completely fragmented", "Completely unified"],
                "weight": 1.0,
            },
            {
                "id": "int_coherence",
                "text": "How connected or disconnected do your thoughts and feelings seem?",
                "type": "slider",
                "min": 0, "max": 6,
                "anchors": ["Disconnected/scattered", "Coherent/connected"],
                "weight": 0.9,
            },
            {
                "id": "int_presence",
                "text": "How present and whole do you feel?",
                "type": "slider",
                "min": 0, "max": 6,
                "anchors": ["Absent/partial", "Fully present/whole"],
                "weight": 0.8,
            },
        ]

    def measure_self_report(self, responses: Dict[str, Any]) -> MeasurementResult:
        items = self.get_self_report_items()
        weighted_sum = 0.0
        weight_total = 0.0

        for item in items:
            if item["id"] in responses:
                raw = responses[item["id"]]
                normalized = (raw - item["min"]) / (item["max"] - item["min"])
                weighted_sum += normalized * item["weight"]
                weight_total += item["weight"]

        value = weighted_sum / weight_total if weight_total > 0 else 0.5
        confidence = weight_total / sum(i["weight"] for i in items)

        return MeasurementResult(
            dimension="integration",
            value=value,
            confidence=confidence,
            modality="self_report",
            raw_data=responses,
        )

    @staticmethod
    def from_eeg(lempel_ziv: float, pci: float, connectivity: float) -> MeasurementResult:
        """
        Estimate integration from EEG-derived measures.

        Lempel-Ziv complexity, Perturbational Complexity Index, functional connectivity.
        """
        # Normalize (these would be calibrated to population norms)
        lz_norm = np.clip(lempel_ziv, 0, 1)
        pci_norm = np.clip(pci / 0.5, 0, 1)  # PCI ~0.5 in healthy wakeful state
        conn_norm = np.clip(connectivity, 0, 1)

        value = (lz_norm + pci_norm + conn_norm) / 3

        return MeasurementResult(
            dimension="integration",
            value=float(value),
            confidence=0.8,
            modality="neural",
            raw_data={"lempel_ziv": lempel_ziv, "pci": pci, "connectivity": connectivity},
        )


class EffectiveRankMeasure(DimensionMeasure):
    """
    Measure effective rank: the openness/narrowness dimension.

    Key theoretical claim: r_eff = distribution of active degrees of freedom.
    Falsification: If neural dimensionality doesn't correlate with cognitive flexibility.
    """

    @property
    def dimension_name(self) -> str:
        return "effective_rank"

    def get_self_report_items(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "er_options",
                "text": "How many options or possibilities feel available to you right now?",
                "type": "slider",
                "min": 0, "max": 6,
                "anchors": ["No options/stuck", "Many possibilities"],
                "weight": 1.0,
            },
            {
                "id": "er_flexibility",
                "text": "How flexible or rigid does your thinking feel?",
                "type": "slider",
                "min": 0, "max": 6,
                "anchors": ["Very rigid", "Very flexible"],
                "weight": 0.9,
            },
            {
                "id": "er_openness",
                "text": "How open or closed does your attention feel?",
                "type": "slider",
                "min": 0, "max": 6,
                "anchors": ["Narrow/closed", "Broad/open"],
                "weight": 0.8,
            },
        ]

    def measure_self_report(self, responses: Dict[str, Any]) -> MeasurementResult:
        items = self.get_self_report_items()
        weighted_sum = 0.0
        weight_total = 0.0

        for item in items:
            if item["id"] in responses:
                raw = responses[item["id"]]
                normalized = (raw - item["min"]) / (item["max"] - item["min"])
                weighted_sum += normalized * item["weight"]
                weight_total += item["weight"]

        value = weighted_sum / weight_total if weight_total > 0 else 0.5
        confidence = weight_total / sum(i["weight"] for i in items)

        return MeasurementResult(
            dimension="effective_rank",
            value=value,
            confidence=confidence,
            modality="self_report",
            raw_data=responses,
        )

    @staticmethod
    def from_behavior(response_diversity: float, flexibility_score: float) -> MeasurementResult:
        """Estimate from behavioral measures of cognitive flexibility."""
        value = (response_diversity + flexibility_score) / 2
        return MeasurementResult(
            dimension="effective_rank",
            value=float(np.clip(value, 0, 1)),
            confidence=0.7,
            modality="behavioral",
            raw_data={"response_diversity": response_diversity, "flexibility": flexibility_score},
        )


class CounterfactualWeightMeasure(DimensionMeasure):
    """
    Measure counterfactual weight: the present/elsewhere dimension.

    Key theoretical claim: CF = resources on non-actual trajectories.
    Falsification: If DMN activation doesn't correlate with mind-wandering.
    """

    @property
    def dimension_name(self) -> str:
        return "counterfactual_weight"

    def get_self_report_items(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "cf_temporal",
                "text": "How much were you thinking about the past/future vs the present?",
                "type": "slider",
                "min": 0, "max": 6,
                "anchors": ["Entirely present", "Entirely past/future"],
                "weight": 1.0,
            },
            {
                "id": "cf_location",
                "text": "Where was your mind just now?",
                "type": "slider",
                "min": 0, "max": 6,
                "anchors": ["Right here", "Somewhere else"],
                "weight": 0.9,
            },
            {
                "id": "cf_absorption",
                "text": "How absorbed were you in what you were doing?",
                "type": "slider",
                "min": 0, "max": 6,
                "anchors": ["Completely absorbed", "Mind wandering"],
                "weight": 0.8,
            },
        ]

    def measure_self_report(self, responses: Dict[str, Any]) -> MeasurementResult:
        items = self.get_self_report_items()
        weighted_sum = 0.0
        weight_total = 0.0

        for item in items:
            if item["id"] in responses:
                raw = responses[item["id"]]
                normalized = (raw - item["min"]) / (item["max"] - item["min"])
                weighted_sum += normalized * item["weight"]
                weight_total += item["weight"]

        value = weighted_sum / weight_total if weight_total > 0 else 0.5
        confidence = weight_total / sum(i["weight"] for i in items)

        return MeasurementResult(
            dimension="counterfactual_weight",
            value=value,
            confidence=confidence,
            modality="self_report",
            raw_data=responses,
        )

    @staticmethod
    def from_neural(dmn_activation: float, frontopolar: float) -> MeasurementResult:
        """Estimate from neural markers of prospection/retrospection."""
        value = (dmn_activation + frontopolar) / 2
        return MeasurementResult(
            dimension="counterfactual_weight",
            value=float(np.clip(value, 0, 1)),
            confidence=0.7,
            modality="neural",
            raw_data={"dmn": dmn_activation, "frontopolar": frontopolar},
        )


class SelfModelSalienceMeasure(DimensionMeasure):
    """
    Measure self-model salience: the self-focused/absorbed dimension.

    Key theoretical claim: SM = degree to which self-model dominates attention.
    Falsification: If meditation doesn't reduce midline activation.
    """

    @property
    def dimension_name(self) -> str:
        return "self_model_salience"

    def get_self_report_items(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "sm_awareness",
                "text": "How aware of yourself were you?",
                "type": "slider",
                "min": 0, "max": 6,
                "anchors": ["Not at all self-aware", "Extremely self-aware"],
                "weight": 1.0,
            },
            {
                "id": "sm_consciousness",
                "text": "How self-conscious did you feel?",
                "type": "slider",
                "min": 0, "max": 6,
                "anchors": ["Not at all", "Extremely"],
                "weight": 0.9,
            },
            {
                "id": "sm_object",
                "text": "To what extent were you the object of your own attention?",
                "type": "slider",
                "min": 0, "max": 6,
                "anchors": ["Not at all", "Completely"],
                "weight": 0.8,
            },
        ]

    def measure_self_report(self, responses: Dict[str, Any]) -> MeasurementResult:
        items = self.get_self_report_items()
        weighted_sum = 0.0
        weight_total = 0.0

        for item in items:
            if item["id"] in responses:
                raw = responses[item["id"]]
                normalized = (raw - item["min"]) / (item["max"] - item["min"])
                weighted_sum += normalized * item["weight"]
                weight_total += item["weight"]

        value = weighted_sum / weight_total if weight_total > 0 else 0.5
        confidence = weight_total / sum(i["weight"] for i in items)

        return MeasurementResult(
            dimension="self_model_salience",
            value=value,
            confidence=confidence,
            modality="self_report",
            raw_data=responses,
        )

    @staticmethod
    def from_neural(mpfc: float, pcc: float, name_response: float) -> MeasurementResult:
        """Estimate from neural markers of self-referential processing."""
        value = (mpfc + pcc + name_response) / 3
        return MeasurementResult(
            dimension="self_model_salience",
            value=float(np.clip(value, 0, 1)),
            confidence=0.7,
            modality="neural",
            raw_data={"mpfc": mpfc, "pcc": pcc, "name_response": name_response},
        )

    @staticmethod
    def from_behavior(pronoun_ratio: float, mirror_latency: float) -> MeasurementResult:
        """Estimate from behavioral markers."""
        pronoun_norm = np.clip(pronoun_ratio * 5, 0, 1)  # More self-reference = higher
        mirror_norm = 1 - np.clip(mirror_latency / 5, 0, 1)  # Faster = higher
        value = (pronoun_norm + mirror_norm) / 2
        return MeasurementResult(
            dimension="self_model_salience",
            value=float(value),
            confidence=0.6,
            modality="behavioral",
            raw_data={"pronoun_ratio": pronoun_ratio, "mirror_latency": mirror_latency},
        )
