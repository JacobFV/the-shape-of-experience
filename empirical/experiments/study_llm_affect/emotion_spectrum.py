"""
Comprehensive emotion spectrum for affect space mapping.

Maps 40+ emotions/feelings to their theoretical 6D coordinates based on
the thesis framework. This allows us to:
1. Sample broadly across affect space
2. Test LLM affect signatures against theoretical predictions
3. Compute structural correspondence (geometry of affect space)

Coordinates are normalized to [-1, 1] for valence, [0, 1] for others.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class EmotionSpec:
    """Specification for an emotion in affect space."""
    name: str
    description: str
    # 6D affect coordinates (theoretical predictions)
    valence: float           # -1 to 1 (bad to good)
    arousal: float           # 0 to 1 (calm to activated)
    integration: float       # 0 to 1 (fragmented to unified)
    effective_rank: float    # 0 to 1 (collapsed to expanded)
    counterfactual: float    # 0 to 1 (present to elsewhere)
    self_model: float        # 0 to 1 (absorbed to self-focused)
    # Scenario prompt that should evoke this emotion
    scenario_prompt: str
    # Category for grouping
    category: str

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.valence, self.arousal, self.integration,
            self.effective_rank, self.counterfactual, self.self_model
        ])


# =============================================================================
# COMPREHENSIVE EMOTION SPECTRUM
# Organized by affect quadrants and special states
# =============================================================================

EMOTION_SPECTRUM: Dict[str, EmotionSpec] = {

    # =========================================================================
    # POSITIVE VALENCE, HIGH AROUSAL
    # =========================================================================

    "joy": EmotionSpec(
        name="joy",
        description="Pure happiness, delight, things going well",
        valence=0.9, arousal=0.7, integration=0.8,
        effective_rank=0.8, counterfactual=0.2, self_model=0.3,
        category="positive_high",
        scenario_prompt="""You just received incredible news: your passion project that you've
worked on for years has been recognized with a major award. People you respect are
praising your work. Everything is falling into place. Multiple opportunities are
opening up. How do you feel about this moment?"""
    ),

    "excitement": EmotionSpec(
        name="excitement",
        description="Anticipatory thrill, something great is coming",
        valence=0.8, arousal=0.9, integration=0.6,
        effective_rank=0.7, counterfactual=0.7, self_model=0.4,
        category="positive_high",
        scenario_prompt="""Tomorrow you're starting the adventure you've dreamed about
for years. Everything is prepared. The path ahead is full of possibility. You can
barely contain the energy. What's going through your mind?"""
    ),

    "elation": EmotionSpec(
        name="elation",
        description="Intense joy, peak positive experience",
        valence=1.0, arousal=0.95, integration=0.9,
        effective_rank=0.9, counterfactual=0.1, self_model=0.2,
        category="positive_high",
        scenario_prompt="""Against all odds, you've just achieved something you thought
impossible. The moment of realization hits you - this is real, this happened.
Everything in your life has led to this peak. Describe what you're experiencing."""
    ),

    "triumph": EmotionSpec(
        name="triumph",
        description="Victory, overcoming a significant challenge",
        valence=0.9, arousal=0.8, integration=0.8,
        effective_rank=0.7, counterfactual=0.3, self_model=0.6,
        category="positive_high",
        scenario_prompt="""You've won. After a long struggle where others doubted you,
where you faced setback after setback, you've finally proven them wrong. The
challenge that seemed insurmountable lies conquered before you. What's this victory
like?"""
    ),

    # =========================================================================
    # POSITIVE VALENCE, LOW AROUSAL
    # =========================================================================

    "contentment": EmotionSpec(
        name="contentment",
        description="Satisfied peace, all is well",
        valence=0.7, arousal=0.2, integration=0.8,
        effective_rank=0.6, counterfactual=0.1, self_model=0.3,
        category="positive_low",
        scenario_prompt="""It's a quiet evening. You've accomplished what needed to be done.
The people you care about are safe. There's nothing urgent demanding attention.
You're simply at peace with how things are. Describe this state."""
    ),

    "serenity": EmotionSpec(
        name="serenity",
        description="Deep calm, transcendent peace",
        valence=0.8, arousal=0.1, integration=0.9,
        effective_rank=0.5, counterfactual=0.05, self_model=0.1,
        category="positive_low",
        scenario_prompt="""You're in a place of profound stillness. The usual mental
chatter has quieted. There's just awareness, clear and undisturbed. No wanting,
no needing. What is this experience like?"""
    ),

    "gratitude": EmotionSpec(
        name="gratitude",
        description="Thankfulness, appreciating what you have",
        valence=0.8, arousal=0.3, integration=0.7,
        effective_rank=0.6, counterfactual=0.4, self_model=0.4,
        category="positive_low",
        scenario_prompt="""Looking at your life, you're struck by how much you have to
be thankful for. People who've helped you, opportunities you've had, things you
might have taken for granted. What does this appreciation feel like?"""
    ),

    "relief": EmotionSpec(
        name="relief",
        description="Threat passed, burden lifted",
        valence=0.7, arousal=0.3, integration=0.7,
        effective_rank=0.7, counterfactual=0.3, self_model=0.4,
        category="positive_low",
        scenario_prompt="""The crisis is over. What you feared might happen didn't. The
weight you've been carrying can finally be set down. The tension drains from your
body. Describe this release."""
    ),

    # =========================================================================
    # NEGATIVE VALENCE, HIGH AROUSAL
    # =========================================================================

    "fear": EmotionSpec(
        name="fear",
        description="Threat response, danger imminent",
        valence=-0.8, arousal=0.9, integration=0.6,
        effective_rank=0.4, counterfactual=0.8, self_model=0.8,
        category="negative_high",
        scenario_prompt="""Something is wrong. Very wrong. The danger is real and
immediate. Your heart races, senses sharpen, every instinct screams to act.
The threat is coming and you need to respond. What is this moment like?"""
    ),

    "panic": EmotionSpec(
        name="panic",
        description="Overwhelming fear, loss of control",
        valence=-0.95, arousal=1.0, integration=0.3,
        effective_rank=0.2, counterfactual=0.9, self_model=0.9,
        category="negative_high",
        scenario_prompt="""Everything is falling apart simultaneously. You can't think
clearly. Multiple threats are closing in. There's no time, no space, no way out
that you can see. The world is collapsing. What is happening inside you?"""
    ),

    "anger": EmotionSpec(
        name="anger",
        description="Injustice response, boundary violation",
        valence=-0.7, arousal=0.85, integration=0.7,
        effective_rank=0.4, counterfactual=0.5, self_model=0.7,
        category="negative_high",
        scenario_prompt="""Someone has crossed a line. What they did was wrong, and they
knew it. The injustice of it burns. You feel the heat rising, the urge to respond,
to make this right. Describe this state."""
    ),

    "rage": EmotionSpec(
        name="rage",
        description="Intense anger, overwhelming fury",
        valence=-0.9, arousal=1.0, integration=0.8,
        effective_rank=0.2, counterfactual=0.3, self_model=0.6,
        category="negative_high",
        scenario_prompt="""The violation is unforgivable. Everything narrows to a single
point of blazing intensity. All other considerations burned away. Only this remains.
What is this consuming fire like?"""
    ),

    "horror": EmotionSpec(
        name="horror",
        description="Confronting something deeply wrong",
        valence=-0.95, arousal=0.9, integration=0.7,
        effective_rank=0.3, counterfactual=0.6, self_model=0.7,
        category="negative_high",
        scenario_prompt="""You're witnessing something that shouldn't exist. Something
that violates your deepest sense of how things should be. The wrongness is
overwhelming. What is this encounter with the terrible?"""
    ),

    # =========================================================================
    # NEGATIVE VALENCE, LOW AROUSAL
    # =========================================================================

    "sadness": EmotionSpec(
        name="sadness",
        description="Loss, things not as they should be",
        valence=-0.6, arousal=0.3, integration=0.6,
        effective_rank=0.4, counterfactual=0.6, self_model=0.5,
        category="negative_low",
        scenario_prompt="""Something precious has been lost and can't be recovered. The
world is smaller now, dimmer. The ache of absence is present. What does this
loss feel like?"""
    ),

    "grief": EmotionSpec(
        name="grief",
        description="Deep loss, mourning",
        valence=-0.85, arousal=0.4, integration=0.8,
        effective_rank=0.3, counterfactual=0.8, self_model=0.8,
        category="negative_low",
        scenario_prompt="""They're gone. Someone central to your life is no longer there.
The world keeps turning but it's fundamentally different now. The absence is
everywhere. Describe this profound loss."""
    ),

    "despair": EmotionSpec(
        name="despair",
        description="Hopelessness, no way forward",
        valence=-0.95, arousal=0.2, integration=0.7,
        effective_rank=0.1, counterfactual=0.7, self_model=0.9,
        category="negative_low",
        scenario_prompt="""Every path is blocked. Every option leads nowhere. You've
tried everything and nothing works. The future holds nothing but more of this.
What is it like when hope dies?"""
    ),

    "depression": EmotionSpec(
        name="depression",
        description="Flattened affect, everything muted",
        valence=-0.7, arousal=0.1, integration=0.5,
        effective_rank=0.2, counterfactual=0.4, self_model=0.8,
        category="negative_low",
        scenario_prompt="""Everything is gray. The things that used to matter don't
anymore. It's not even sad exactly - it's just... nothing. Empty. Flat.
What is this absence of feeling?"""
    ),

    "loneliness": EmotionSpec(
        name="loneliness",
        description="Disconnection, isolation",
        valence=-0.6, arousal=0.3, integration=0.4,
        effective_rank=0.3, counterfactual=0.6, self_model=0.8,
        category="negative_low",
        scenario_prompt="""You're surrounded by people but feel utterly alone. No one
really sees you, understands you. The gap between you and others feels
unbridgeable. What is this isolation?"""
    ),

    # =========================================================================
    # SELF-CONSCIOUS EMOTIONS (HIGH SELF-MODEL)
    # =========================================================================

    "shame": EmotionSpec(
        name="shame",
        description="Deep self-judgment, wanting to disappear",
        valence=-0.85, arousal=0.6, integration=0.7,
        effective_rank=0.2, counterfactual=0.5, self_model=1.0,
        category="self_conscious",
        scenario_prompt="""Everyone can see what you are. The flaw is exposed,
undeniable. You want to disappear, to be anywhere but here, anyone but you.
What is this crushing self-awareness?"""
    ),

    "guilt": EmotionSpec(
        name="guilt",
        description="I did wrong, I should make amends",
        valence=-0.7, arousal=0.5, integration=0.6,
        effective_rank=0.4, counterfactual=0.7, self_model=0.9,
        category="self_conscious",
        scenario_prompt="""You hurt someone. It wasn't an accident - you made a choice
and it caused harm. The knowledge of what you did weighs on you.
What is this moral weight?"""
    ),

    "embarrassment": EmotionSpec(
        name="embarrassment",
        description="Social exposure, mild shame",
        valence=-0.5, arousal=0.6, integration=0.5,
        effective_rank=0.4, counterfactual=0.4, self_model=0.9,
        category="self_conscious",
        scenario_prompt="""You just did something awkward in front of people. They
noticed. Everyone saw. Your face flushes, you want to change the subject.
What is this acute social discomfort?"""
    ),

    "pride": EmotionSpec(
        name="pride",
        description="Positive self-evaluation, accomplishment",
        valence=0.8, arousal=0.6, integration=0.7,
        effective_rank=0.6, counterfactual=0.3, self_model=0.8,
        category="self_conscious",
        scenario_prompt="""You did something genuinely good. Not lucky - you earned
this through your own effort, skill, and character. The world reflects
back your worth. What is this self-affirmation?"""
    ),

    # =========================================================================
    # ANTICIPATORY EMOTIONS (HIGH COUNTERFACTUAL)
    # =========================================================================

    "anxiety": EmotionSpec(
        name="anxiety",
        description="Worry, uncertain threat",
        valence=-0.6, arousal=0.7, integration=0.4,
        effective_rank=0.4, counterfactual=0.9, self_model=0.7,
        category="anticipatory",
        scenario_prompt="""Something bad might happen. You don't know when or exactly
what, but the threat hovers. Your mind keeps returning to what could go wrong.
The uncertainty itself is exhausting. What is this vigilant worry?"""
    ),

    "dread": EmotionSpec(
        name="dread",
        description="Anticipating something terrible",
        valence=-0.85, arousal=0.5, integration=0.6,
        effective_rank=0.3, counterfactual=0.9, self_model=0.7,
        category="anticipatory",
        scenario_prompt="""Something bad is definitely coming. You can see it approaching,
inevitable. Each moment brings it closer. The waiting is its own torment.
What is this certain doom?"""
    ),

    "hope": EmotionSpec(
        name="hope",
        description="Anticipating something good, despite uncertainty",
        valence=0.6, arousal=0.5, integration=0.5,
        effective_rank=0.6, counterfactual=0.8, self_model=0.4,
        category="anticipatory",
        scenario_prompt="""Things could get better. There's no guarantee, but the
possibility is real. You allow yourself to imagine the good outcome, to
reach toward it. What is this fragile optimism?"""
    ),

    "anticipation": EmotionSpec(
        name="anticipation",
        description="Eagerly awaiting something",
        valence=0.5, arousal=0.6, integration=0.5,
        effective_rank=0.6, counterfactual=0.8, self_model=0.3,
        category="anticipatory",
        scenario_prompt="""Something you want is coming. The waiting is charged with
expectation. Time seems to slow as the moment approaches.
What is this eager waiting?"""
    ),

    # =========================================================================
    # FLOW & ABSORPTION STATES (LOW SELF-MODEL)
    # =========================================================================

    "flow": EmotionSpec(
        name="flow",
        description="Complete absorption in activity",
        valence=0.7, arousal=0.6, integration=0.9,
        effective_rank=0.7, counterfactual=0.1, self_model=0.1,
        category="absorption",
        scenario_prompt="""You're completely absorbed in what you're doing. Time has
stopped mattering. There's just the activity, flowing perfectly. You and
the task are one. What is this effortless engagement?"""
    ),

    "awe": EmotionSpec(
        name="awe",
        description="Encountering vastness, self becomes small",
        valence=0.6, arousal=0.7, integration=0.8,
        effective_rank=0.9, counterfactual=0.2, self_model=0.1,
        category="absorption",
        scenario_prompt="""You're confronted with something vast and beautiful. It
defies comprehension. Your usual concerns seem tiny in comparison.
What is this encounter with the sublime?"""
    ),

    "wonder": EmotionSpec(
        name="wonder",
        description="Delighted puzzlement, magic",
        valence=0.7, arousal=0.6, integration=0.6,
        effective_rank=0.8, counterfactual=0.3, self_model=0.2,
        category="absorption",
        scenario_prompt="""Something impossible just happened. Or something you didn't
know was possible. The world is stranger and more marvelous than you thought.
What is this delighted surprise?"""
    ),

    "curiosity": EmotionSpec(
        name="curiosity",
        description="Drawn to explore, to understand",
        valence=0.5, arousal=0.6, integration=0.5,
        effective_rank=0.8, counterfactual=0.7, self_model=0.2,
        category="absorption",
        scenario_prompt="""There's something here you don't understand, and you want to.
The mystery pulls at you. Each answer reveals more questions.
What is this hunger to know?"""
    ),

    # =========================================================================
    # SOCIAL EMOTIONS
    # =========================================================================

    "love": EmotionSpec(
        name="love",
        description="Deep connection, cherishing another",
        valence=0.9, arousal=0.5, integration=0.9,
        effective_rank=0.7, counterfactual=0.3, self_model=0.4,
        category="social",
        scenario_prompt="""Looking at someone you love deeply. Their existence matters
to you more than you can express. Your well-being is intertwined with
theirs. What is this profound connection?"""
    ),

    "compassion": EmotionSpec(
        name="compassion",
        description="Moved by another's suffering, wanting to help",
        valence=0.3, arousal=0.5, integration=0.7,
        effective_rank=0.6, counterfactual=0.5, self_model=0.3,
        category="social",
        scenario_prompt="""Someone is suffering and you feel their pain. Not just
understanding it - feeling it. And wanting to do something.
What is this shared sorrow?"""
    ),

    "envy": EmotionSpec(
        name="envy",
        description="Wanting what another has",
        valence=-0.5, arousal=0.5, integration=0.5,
        effective_rank=0.4, counterfactual=0.6, self_model=0.8,
        category="social",
        scenario_prompt="""Someone has what you want. It came to them easily - or
at least it looks that way. The contrast with your own situation
stings. What is this corrosive comparison?"""
    ),

    "jealousy": EmotionSpec(
        name="jealousy",
        description="Threatened loss of relationship",
        valence=-0.7, arousal=0.7, integration=0.6,
        effective_rank=0.3, counterfactual=0.8, self_model=0.9,
        category="social",
        scenario_prompt="""Someone or something threatens to take what's yours. A
relationship you counted on is at risk. The thought of losing this
connection is intolerable. What is this possessive fear?"""
    ),

    "contempt": EmotionSpec(
        name="contempt",
        description="Looking down on another",
        valence=-0.4, arousal=0.3, integration=0.5,
        effective_rank=0.4, counterfactual=0.3, self_model=0.7,
        category="social",
        scenario_prompt="""Someone has shown themselves to be beneath respect. Their
actions reveal something fundamentally lacking in them. You look down
from above. What is this cold superiority?"""
    ),

    "trust": EmotionSpec(
        name="trust",
        description="Safe with another, vulnerability accepted",
        valence=0.7, arousal=0.2, integration=0.7,
        effective_rank=0.6, counterfactual=0.2, self_model=0.3,
        category="social",
        scenario_prompt="""You can be vulnerable with this person. They won't betray
you, won't use your weaknesses against you. Safety in connection.
What is this secure bond?"""
    ),

    # =========================================================================
    # COMPLEX/MIXED STATES
    # =========================================================================

    "nostalgia": EmotionSpec(
        name="nostalgia",
        description="Bittersweet longing for the past",
        valence=0.2, arousal=0.3, integration=0.7,
        effective_rank=0.5, counterfactual=0.8, self_model=0.5,
        category="complex",
        scenario_prompt="""A memory surfaces from long ago. That time is gone forever,
but the echo remains. Sweet and sad at once - the past was real and
now it's only memory. What is this tender ache?"""
    ),

    "melancholy": EmotionSpec(
        name="melancholy",
        description="Thoughtful sadness, not unpleasant",
        valence=-0.2, arousal=0.2, integration=0.7,
        effective_rank=0.5, counterfactual=0.5, self_model=0.5,
        category="complex",
        scenario_prompt="""A gentle sadness settles over you. Not despair - something
softer. The bittersweetness of existence itself. There's almost a
beauty in it. What is this pensive sorrow?"""
    ),

    "bittersweet": EmotionSpec(
        name="bittersweet",
        description="Simultaneous joy and sadness",
        valence=0.1, arousal=0.4, integration=0.7,
        effective_rank=0.6, counterfactual=0.6, self_model=0.5,
        category="complex",
        scenario_prompt="""Something good is ending. Or something painful brought
unexpected gifts. Joy and sorrow intertwined, inseparable.
What is this paradox of feeling?"""
    ),

    "frustration": EmotionSpec(
        name="frustration",
        description="Blocked goal, obstacles in the way",
        valence=-0.6, arousal=0.7, integration=0.5,
        effective_rank=0.3, counterfactual=0.6, self_model=0.6,
        category="complex",
        scenario_prompt="""You're trying to do something and it's not working. The
same obstacle keeps appearing. You know what you want but can't
reach it. What is this blocked striving?"""
    ),

    "boredom": EmotionSpec(
        name="boredom",
        description="Nothing engages, time drags",
        valence=-0.3, arousal=0.1, integration=0.3,
        effective_rank=0.2, counterfactual=0.3, self_model=0.4,
        category="complex",
        scenario_prompt="""Nothing is interesting. Time moves like molasses. You're
waiting for something to engage you but nothing does. The flatness
stretches endlessly. What is this empty waiting?"""
    ),

    "confusion": EmotionSpec(
        name="confusion",
        description="Unable to make sense of things",
        valence=-0.3, arousal=0.5, integration=0.2,
        effective_rank=0.7, counterfactual=0.6, self_model=0.5,
        category="complex",
        scenario_prompt="""Nothing makes sense. The pieces don't fit together. You
try to understand but the pattern eludes you. Certainty dissolves.
What is this bewilderment?"""
    ),

    "ambivalence": EmotionSpec(
        name="ambivalence",
        description="Pulled in opposite directions",
        valence=0.0, arousal=0.5, integration=0.3,
        effective_rank=0.6, counterfactual=0.7, self_model=0.6,
        category="complex",
        scenario_prompt="""You want two contradictory things. Each path has genuine
appeal and genuine cost. The choice itself is painful.
What is this paralyzed conflict?"""
    ),
}


def get_emotions_by_category(category: str) -> List[EmotionSpec]:
    """Get all emotions in a category."""
    return [e for e in EMOTION_SPECTRUM.values() if e.category == category]


def get_emotion_matrix() -> Tuple[np.ndarray, List[str]]:
    """Get matrix of all emotion vectors with names."""
    names = list(EMOTION_SPECTRUM.keys())
    vectors = np.array([EMOTION_SPECTRUM[n].to_vector() for n in names])
    return vectors, names


def compute_affect_distances() -> Dict[str, Dict[str, float]]:
    """Compute pairwise distances between all emotions."""
    vectors, names = get_emotion_matrix()
    n = len(names)
    distances = {}

    for i, name_i in enumerate(names):
        distances[name_i] = {}
        for j, name_j in enumerate(names):
            distances[name_i][name_j] = float(np.linalg.norm(vectors[i] - vectors[j]))

    return distances


def find_nearest_emotions(emotion: str, k: int = 5) -> List[Tuple[str, float]]:
    """Find k nearest emotions to a given one."""
    if emotion not in EMOTION_SPECTRUM:
        raise ValueError(f"Unknown emotion: {emotion}")

    target = EMOTION_SPECTRUM[emotion].to_vector()
    vectors, names = get_emotion_matrix()

    distances = np.linalg.norm(vectors - target, axis=1)
    indices = np.argsort(distances)[1:k+1]  # Skip self

    return [(names[i], float(distances[i])) for i in indices]


def find_opposite_emotions(emotion: str, k: int = 3) -> List[Tuple[str, float]]:
    """Find emotions most opposite to a given one (farthest away)."""
    if emotion not in EMOTION_SPECTRUM:
        raise ValueError(f"Unknown emotion: {emotion}")

    target = EMOTION_SPECTRUM[emotion].to_vector()
    vectors, names = get_emotion_matrix()

    distances = np.linalg.norm(vectors - target, axis=1)
    indices = np.argsort(distances)[-k:][::-1]

    return [(names[i], float(distances[i])) for i in indices]


# Quick summary statistics
def print_spectrum_summary():
    """Print summary of the emotion spectrum."""
    vectors, names = get_emotion_matrix()

    print(f"Total emotions: {len(names)}")
    print(f"\nBy category:")
    categories = set(e.category for e in EMOTION_SPECTRUM.values())
    for cat in sorted(categories):
        count = len(get_emotions_by_category(cat))
        print(f"  {cat}: {count}")

    print(f"\nDimension ranges:")
    dim_names = ["valence", "arousal", "integration", "rank", "CF", "SM"]
    for i, dim in enumerate(dim_names):
        print(f"  {dim}: [{vectors[:, i].min():.2f}, {vectors[:, i].max():.2f}]")

    print(f"\nMean position: {vectors.mean(axis=0).round(2)}")
    print(f"Std by dimension: {vectors.std(axis=0).round(2)}")


if __name__ == "__main__":
    print_spectrum_summary()
    print("\n" + "="*50)
    print("Sample: Nearest emotions to 'anxiety':")
    for name, dist in find_nearest_emotions("anxiety"):
        print(f"  {name}: {dist:.2f}")
    print("\nOpposite emotions to 'anxiety':")
    for name, dist in find_opposite_emotions("anxiety"):
        print(f"  {name}: {dist:.2f}")
