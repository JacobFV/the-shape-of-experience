"""
Core affect state representation and the six-dimensional framework.

The thesis claims experience has a specific geometric structure with six dimensions:
1. Valence (Val): Good/bad, approach/avoid - gradient on viability manifold
2. Arousal (Ar): Rate of belief/state update
3. Integration (Φ): Irreducibility of cause-effect structure
4. Effective Rank (r_eff): Distribution of active degrees of freedom
5. Counterfactual Weight (CF): Resources on non-actual trajectories
6. Self-Model Salience (SM): Degree of self-focus

Each dimension has:
- A theoretical definition (from the thesis)
- Operational proxies (self-report, behavioral, physiological, neural)
- Predicted relationships with other constructs
- Falsification criteria
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Tuple
import numpy as np


class AffectDimension(Enum):
    """The six affect dimensions from the thesis."""

    VALENCE = "valence"
    AROUSAL = "arousal"
    INTEGRATION = "integration"
    EFFECTIVE_RANK = "effective_rank"
    COUNTERFACTUAL_WEIGHT = "counterfactual_weight"
    SELF_MODEL_SALIENCE = "self_model_salience"


@dataclass
class DimensionSpec:
    """Specification for an affect dimension including theoretical and operational definitions."""

    name: str
    symbol: str
    theoretical_definition: str
    phenomenological_poles: Tuple[str, str]  # (high, low)

    # Operational proxies at different levels
    self_report_items: List[str] = field(default_factory=list)
    behavioral_proxies: List[str] = field(default_factory=list)
    physiological_proxies: List[str] = field(default_factory=list)
    neural_proxies: List[str] = field(default_factory=list)

    # Theoretical predictions
    predicted_correlates: Dict[str, str] = field(default_factory=dict)
    falsification_criteria: List[str] = field(default_factory=list)


# Define each dimension with full specification
DIMENSION_SPECS = {
    AffectDimension.VALENCE: DimensionSpec(
        name="Valence",
        symbol="Val",
        theoretical_definition=(
            "The felt quality of approach versus avoidance. Formally, the structural "
            "signature of gradient direction on the viability landscape: "
            "Val = f(∇d(s, ∂V) · ṡ). Positive valence indicates movement into viable "
            "interior; negative indicates approach toward dissolution."
        ),
        phenomenological_poles=("good, pleasant, approach", "bad, unpleasant, avoid"),
        self_report_items=[
            "How pleasant/unpleasant do you feel right now? (1-7)",
            "Do you want this moment to continue or end? (continue/end)",
            "How would you rate your current wellbeing? (1-10)",
        ],
        behavioral_proxies=[
            "Approach vs avoidance behaviors",
            "Facial expression (smile vs frown)",
            "Persistence on current activity",
        ],
        physiological_proxies=[
            "Heart rate variability (HRV) - higher = more positive",
            "Cortisol levels - lower = more positive",
            "Skin conductance level (tonic)",
            "Zygomatic/corrugator EMG",
        ],
        neural_proxies=[
            "Left vs right prefrontal asymmetry (EEG)",
            "Nucleus accumbens activation (fMRI)",
            "Amygdala activation pattern",
        ],
        predicted_correlates={
            "objective_safety": "Positive correlation with distance from viability boundary",
            "immune_function": "Positive correlation with valence over time",
            "allostatic_load": "Negative correlation with chronic valence",
        },
        falsification_criteria=[
            "If valence does not correlate with objective viability measures",
            "If chronic negative valence does not predict health outcomes",
            "If valence can be high during actual dissolution approach",
        ],
    ),

    AffectDimension.AROUSAL: DimensionSpec(
        name="Arousal",
        symbol="Ar",
        theoretical_definition=(
            "The rate of belief/state update. Formally: Ar = KL(b_{t+1} || b_t). "
            "How rapidly the system's internal model is changing."
        ),
        phenomenological_poles=("activated, alert, intense", "calm, settled, quiet"),
        self_report_items=[
            "How activated/calm do you feel right now? (1-7)",
            "How much energy do you have? (1-7)",
            "How alert vs drowsy are you? (1-7)",
        ],
        behavioral_proxies=[
            "Movement speed and frequency",
            "Speech rate",
            "Response latency",
            "Motor restlessness",
        ],
        physiological_proxies=[
            "Heart rate",
            "Skin conductance response (phasic)",
            "Pupil dilation",
            "Respiratory rate",
        ],
        neural_proxies=[
            "Beta power (EEG)",
            "Locus coeruleus activation",
            "Thalamic activity",
        ],
        predicted_correlates={
            "prediction_error": "Positive correlation with surprise/novelty",
            "attention": "Inverted U relationship (Yerkes-Dodson)",
            "memory_encoding": "Enhanced at moderate levels",
        },
        falsification_criteria=[
            "If arousal does not correlate with prediction error magnitude",
            "If arousal is independent of environmental novelty/change",
        ],
    ),

    AffectDimension.INTEGRATION: DimensionSpec(
        name="Integration",
        symbol="Φ",
        theoretical_definition=(
            "Measures the irreducibility of the system's cause-effect structure under "
            "partition. Φ(s) = min over partitions of divergence between whole and "
            "product of parts."
        ),
        phenomenological_poles=("unified, coherent, connected", "fragmented, dissociated, scattered"),
        self_report_items=[
            "How unified/fragmented does your experience feel? (1-7)",
            "Do your thoughts and feelings seem connected or scattered? (1-7)",
            "How present and whole do you feel? (1-7)",
        ],
        behavioral_proxies=[
            "Task coherence",
            "Narrative continuity",
            "Response consistency",
        ],
        physiological_proxies=[
            "Inter-system coupling (cardiac-respiratory)",
            "Hormonal coherence",
        ],
        neural_proxies=[
            "EEG complexity measures (Lempel-Ziv, entropy)",
            "Functional connectivity (fMRI)",
            "Global workspace activation",
            "Perturbational complexity index (TMS-EEG)",
        ],
        predicted_correlates={
            "consciousness_level": "Monotonic relationship",
            "anesthesia_depth": "Negative correlation",
            "meditation_expertise": "Positive correlation with practice",
        },
        falsification_criteria=[
            "If integration measures do not decrease under anesthesia",
            "If fragmented neural activity accompanies reported unity",
            "If high integration occurs without any phenomenal report",
        ],
    ),

    AffectDimension.EFFECTIVE_RANK: DimensionSpec(
        name="Effective Rank",
        symbol="r_eff",
        theoretical_definition=(
            "Measures how distributed versus concentrated the active degrees of freedom "
            "are. r_eff = (tr C)² / tr(C²) where C is the state covariance matrix."
        ),
        phenomenological_poles=("open, many possibilities, flexible", "narrow, tunnel vision, rigid"),
        self_report_items=[
            "How many options feel available to you right now? (1-7)",
            "How flexible/rigid does your thinking feel? (1-7)",
            "How open vs closed does your attention feel? (1-7)",
        ],
        behavioral_proxies=[
            "Response diversity",
            "Cognitive flexibility tasks",
            "Breadth of attention",
            "Number of considered alternatives",
        ],
        physiological_proxies=[
            "Gaze dispersion",
            "Postural variability",
        ],
        neural_proxies=[
            "Dimensionality of neural activity",
            "Entropy of brain state trajectories",
            "Breadth of cortical activation",
        ],
        predicted_correlates={
            "creativity": "Positive correlation",
            "depression": "Negative correlation (collapsed rank)",
            "psychedelics": "Increased rank under influence",
        },
        falsification_criteria=[
            "If rumination does not show reduced effective rank",
            "If open awareness meditation does not increase rank",
            "If neural dimensionality is uncorrelated with reported openness",
        ],
    ),

    AffectDimension.COUNTERFACTUAL_WEIGHT: DimensionSpec(
        name="Counterfactual Weight",
        symbol="CF",
        theoretical_definition=(
            "The fraction of computational resources devoted to modeling non-actual "
            "possibilities: CF = Compute(imagined) / Compute(total)."
        ),
        phenomenological_poles=("elsewhere, planning, remembering", "here, present, absorbed"),
        self_report_items=[
            "How much are you thinking about past/future vs present? (1-7)",
            "Where is your mind right now? (here/elsewhere)",
            "How absorbed are you in the present moment? (1-7)",
        ],
        behavioral_proxies=[
            "Mind-wandering probes",
            "Response to immediate stimuli",
            "Planning behavior",
            "Rumination measures",
        ],
        physiological_proxies=[
            "Pupil diameter stability",
            "Microsaccade rate",
        ],
        neural_proxies=[
            "Default mode network activation",
            "Frontopolar cortex activity",
            "Hippocampal-prefrontal coupling",
        ],
        predicted_correlates={
            "anxiety": "High CF (worry about futures)",
            "depression": "High CF (rumination about pasts)",
            "flow": "Low CF (present absorption)",
        },
        falsification_criteria=[
            "If DMN activity does not correlate with self-reported mind-wandering",
            "If anxiety is not characterized by future-oriented thought",
            "If present-moment awareness does not reduce CF markers",
        ],
    ),

    AffectDimension.SELF_MODEL_SALIENCE: DimensionSpec(
        name="Self-Model Salience",
        symbol="SM",
        theoretical_definition=(
            "The degree to which the self-model dominates attention and processing. "
            "SM = I(z^self; a) / H(a) - mutual information of self-model and action "
            "normalized by action entropy."
        ),
        phenomenological_poles=("self-conscious, self-aware, self-focused", "absorbed, self-forgotten, ego-dissolved"),
        self_report_items=[
            "How aware are you of yourself right now? (1-7)",
            "How self-conscious do you feel? (1-7)",
            "To what extent are you the object of your own attention? (1-7)",
        ],
        behavioral_proxies=[
            "Self-reference in speech",
            "Mirror self-recognition latency",
            "Pronoun use (I, me, my)",
            "Self-focused attention tasks",
        ],
        physiological_proxies=[
            "Interoceptive accuracy",
            "Response to self-relevant stimuli",
        ],
        neural_proxies=[
            "Cortical midline structures activation (mPFC, PCC)",
            "Default mode network self-referential component",
            "Response to own name/face",
        ],
        predicted_correlates={
            "social_anxiety": "Positive correlation (excessive SM)",
            "flow_state": "Negative correlation",
            "meditation": "Reduced SM with expertise",
            "ego_dissolution": "Very low SM (psychedelics, mystical states)",
        },
        falsification_criteria=[
            "If meditation does not reduce self-referential processing",
            "If flow states show high self-focus",
            "If ego dissolution reports occur with high midline activation",
        ],
    ),
}


@dataclass
class AffectState:
    """
    A point in the six-dimensional affect space.

    Each dimension is represented as a continuous value, typically normalized
    to [0, 1] or [-1, 1] depending on whether the dimension is unipolar or bipolar.
    """

    valence: float  # [-1, 1] bad to good
    arousal: float  # [0, 1] calm to activated
    integration: float  # [0, 1] fragmented to unified
    effective_rank: float  # [0, 1] narrow to open
    counterfactual_weight: float  # [0, 1] present to elsewhere
    self_model_salience: float  # [0, 1] absorbed to self-focused

    timestamp: Optional[float] = None
    context: Optional[str] = None

    def to_vector(self) -> np.ndarray:
        """Return as numpy array for analysis."""
        return np.array([
            self.valence,
            self.arousal,
            self.integration,
            self.effective_rank,
            self.counterfactual_weight,
            self.self_model_salience,
        ])

    @classmethod
    def from_vector(cls, vec: np.ndarray, timestamp: Optional[float] = None) -> "AffectState":
        """Construct from numpy array."""
        return cls(
            valence=vec[0],
            arousal=vec[1],
            integration=vec[2],
            effective_rank=vec[3],
            counterfactual_weight=vec[4],
            self_model_salience=vec[5],
            timestamp=timestamp,
        )

    def distance_to(self, other: "AffectState") -> float:
        """Euclidean distance in affect space."""
        return np.linalg.norm(self.to_vector() - other.to_vector())


# Predicted affect signatures for various states/activities
PREDICTED_SIGNATURES = {
    "flow": AffectState(
        valence=0.7, arousal=0.5, integration=0.8,
        effective_rank=0.5, counterfactual_weight=0.1, self_model_salience=0.1,
    ),
    "anxiety": AffectState(
        valence=-0.6, arousal=0.8, integration=0.6,
        effective_rank=0.3, counterfactual_weight=0.8, self_model_salience=0.7,
    ),
    "depression": AffectState(
        valence=-0.5, arousal=0.2, integration=0.5,
        effective_rank=0.2, counterfactual_weight=0.6, self_model_salience=0.8,
    ),
    "meditation_focused": AffectState(
        valence=0.3, arousal=0.3, integration=0.8,
        effective_rank=0.3, counterfactual_weight=0.1, self_model_salience=0.3,
    ),
    "meditation_open": AffectState(
        valence=0.4, arousal=0.3, integration=0.9,
        effective_rank=0.8, counterfactual_weight=0.1, self_model_salience=0.2,
    ),
    "psychedelic_peak": AffectState(
        valence=0.5, arousal=0.7, integration=0.9,
        effective_rank=0.9, counterfactual_weight=0.2, self_model_salience=0.1,
    ),
    "boredom": AffectState(
        valence=-0.2, arousal=0.2, integration=0.4,
        effective_rank=0.3, counterfactual_weight=0.5, self_model_salience=0.4,
    ),
    "awe": AffectState(
        valence=0.8, arousal=0.6, integration=0.7,
        effective_rank=0.8, counterfactual_weight=0.3, self_model_salience=0.2,
    ),
}
