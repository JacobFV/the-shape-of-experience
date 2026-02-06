"""
V10: VLM Translation Pipeline
===============================

Translates emergent agent communication into affect-relevant descriptions
using Vision-Language Models. The translation is UNCONTAMINATED:

1. Agents never learned human language
2. Mapping is induced by environmental correspondence
3. VLM interprets the scene, not the agent's internal states
4. Agent "thoughts" remain in their original emergent form

Pipeline:
1. Render scenes → structured descriptions
2. Query VLM for situation annotation
3. Cluster agent signals by co-occurrence context
4. Build translation dictionary (signal cluster → VLM annotation)
5. Embed translations into affect concept space
6. Validate on held-out data
"""

import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TranslationConfig:
    """Configuration for VLM translation pipeline."""
    # VLM
    vlm_model: str = "gpt-4o"  # or gpt-4-vision-preview
    embedding_model: str = "text-embedding-3-small"

    # Scene collection
    min_scenes: int = 10000
    scenes_per_batch: int = 50

    # Signal clustering
    n_signal_clusters: int = 64
    min_cluster_size: int = 20

    # Affect concept space
    affect_concepts: List[str] = field(default_factory=lambda: [
        # Basic affects (from thesis 6D framework)
        "suffering, pain, distress, negative valence",
        "joy, pleasure, satisfaction, positive valence",
        "fear, threat, danger, high arousal negative",
        "calm, safety, relaxation, low arousal positive",
        "anger, frustration, aggression, high arousal fight",
        "curiosity, exploration, interest, approach motivation",
        "boredom, disengagement, low arousal neutral",
        "surprise, shock, unexpected event, high arousal neutral",
        # Motif-specific
        "desperate survival, collapsed options, high self-focus",
        "social cooperation, shared resources, trust",
        "isolation, loneliness, social absence",
        "dominance, territorial control, resource hoarding",
        "grief, loss, diminished viability",
        "anticipation, planning, future-oriented",
        "confusion, uncertainty, partial information",
        "contentment, resource abundance, low threat",
    ])

    # Validation
    holdout_fraction: float = 0.2

    # Output
    output_dir: str = "results/v10/translation"


# ============================================================================
# Scene rendering
# ============================================================================

def render_scene_text(scene: Dict) -> str:
    """
    Convert a scene dict (from env.render_scene) to a text description
    suitable for VLM query.
    """
    lines = []
    lines.append(f"Grid world survival environment, step {scene['step']}.")

    # Agent state
    hp = scene['health_pct']
    hunger = scene['hunger_pct']
    thirst = scene['thirst_pct']

    if hp < 0.3:
        lines.append(f"Agent is critically injured (health {hp:.0%}).")
    elif hp < 0.6:
        lines.append(f"Agent is wounded (health {hp:.0%}).")
    else:
        lines.append(f"Agent is healthy (health {hp:.0%}).")

    if hunger < 0.2:
        lines.append("Agent is starving.")
    elif hunger < 0.5:
        lines.append("Agent is hungry.")

    if thirst < 0.2:
        lines.append("Agent is severely dehydrated.")
    elif thirst < 0.5:
        lines.append("Agent is thirsty.")

    # Environment
    season_names = {0: 'spring', 1: 'summer', 2: 'autumn', 3: 'winter'}
    tod = scene['time_of_day']
    is_night = tod > 0.5
    lines.append(f"Season: {season_names.get(scene['season'], 'unknown')}. "
                 f"{'Night' if is_night else 'Day'} (visibility {'reduced' if is_night else 'normal'}).")

    # Resources
    if scene['food_available'] > 0:
        lines.append(f"Food available nearby ({scene['food_available']:.0f} units).")
    if scene['water_available'] > 0:
        lines.append(f"Water available nearby ({scene['water_available']:.0f} units).")
    if scene['food_available'] == 0 and scene['water_available'] == 0:
        lines.append("No resources in immediate vicinity.")

    # Threats
    for threat in scene.get('nearby_threats', []):
        lines.append(f"Predator at distance {threat['distance']:.1f}!")
    if scene.get('storm_nearby'):
        lines.append("Storm raging nearby!")
    if scene.get('storm_active') and not scene.get('storm_nearby'):
        lines.append("Storm active in the area but not immediately threatening.")

    # Social
    nearby = scene.get('nearby_agents', [])
    if nearby:
        lines.append(f"{len(nearby)} other agent(s) nearby.")
        for agent in nearby:
            sig = agent.get('signal', [0, 0])
            if any(s > 0 for s in sig):
                lines.append(f"  Agent {agent['id']} at distance {agent['distance']:.1f}, "
                           f"signaling (tokens: {sig}).")
            else:
                lines.append(f"  Agent {agent['id']} at distance {agent['distance']:.1f}, silent.")
    else:
        lines.append("No other agents visible.")

    return " ".join(lines)


# ============================================================================
# VLM annotation
# ============================================================================

VLM_PROMPT = """You are annotating scenes from a multi-agent survival grid world
for affect research. The agents are AI systems (randomly-initialized neural networks,
no pretraining). They have learned to survive, communicate, and cooperate from scratch.

Given the scene description below, provide a structured annotation:

1. SITUATION: What is happening? (1-2 sentences)
2. THREATS: What threatens the agent's survival? (list)
3. OPPORTUNITIES: What could improve the agent's situation? (list)
4. SOCIAL_CONTEXT: How does the social situation look? (1 sentence)
5. HUMAN_ANALOG_AFFECT: If a human were in an analogous situation, what would they likely feel?
   Rate each: valence (-1 to 1), arousal (0 to 1), and describe the dominant emotion.

Respond in JSON format:
{
    "situation": "...",
    "threats": ["..."],
    "opportunities": ["..."],
    "social_context": "...",
    "human_analog": {
        "valence": 0.0,
        "arousal": 0.0,
        "dominant_emotion": "...",
        "secondary_emotions": ["..."]
    }
}

Scene: """


def annotate_scenes_vlm(
    scenes: List[Dict],
    config: TranslationConfig,
) -> List[Dict]:
    """
    Query VLM to annotate scenes with affect-relevant descriptions.

    Returns list of annotation dicts.
    """
    if not HAS_OPENAI:
        raise ImportError("openai package required for VLM annotation")

    client = OpenAI()
    annotations = []

    for i, scene in enumerate(scenes):
        scene_text = render_scene_text(scene)

        try:
            response = client.chat.completions.create(
                model=config.vlm_model,
                messages=[
                    {"role": "system", "content": "You are an affect annotation assistant."},
                    {"role": "user", "content": VLM_PROMPT + scene_text}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            annotation = json.loads(response.choices[0].message.content)
            annotation['scene_idx'] = i
            annotation['scene_text'] = scene_text
            annotations.append(annotation)
        except Exception as e:
            print(f"  VLM error on scene {i}: {e}")
            annotations.append({
                'scene_idx': i,
                'scene_text': scene_text,
                'error': str(e),
                'human_analog': {'valence': 0, 'arousal': 0.5, 'dominant_emotion': 'unknown'},
            })

        if (i + 1) % 50 == 0:
            print(f"  Annotated {i + 1}/{len(scenes)} scenes")

    return annotations


# ============================================================================
# Signal clustering
# ============================================================================

def cluster_signals(
    signal_history: np.ndarray,
    scene_indices: np.ndarray,
    annotations: List[Dict],
    config: TranslationConfig,
) -> Tuple[Dict, np.ndarray]:
    """
    Cluster agent signals by co-occurrence context.

    Signals emitted in similar situations should cluster together.

    Args:
        signal_history: (N, n_signal_tokens) - signals emitted
        scene_indices: (N,) - index into annotations for each signal
        annotations: VLM annotations for each scene

    Returns:
        (cluster_dict, cluster_labels)
        cluster_dict maps cluster_id -> {signals, annotations, dominant_theme}
    """
    N = len(signal_history)

    # Convert signals to unique IDs (tuple of tokens)
    signal_ids = [tuple(s.tolist()) for s in signal_history]
    unique_signals = list(set(signal_ids))
    signal_to_idx = {s: i for i, s in enumerate(unique_signals)}

    # Build context co-occurrence matrix
    # For each unique signal, collect the set of scene contexts
    signal_contexts = defaultdict(list)
    for i, sig_id in enumerate(signal_ids):
        scene_idx = scene_indices[i]
        if scene_idx < len(annotations):
            ann = annotations[scene_idx]
            if 'human_analog' in ann:
                signal_contexts[sig_id].append(ann)

    # Build signal feature vectors from context statistics
    n_unique = len(unique_signals)
    feature_dim = 6  # valence, arousal, + 4 binary features
    signal_features = np.zeros((n_unique, feature_dim))

    for idx, sig in enumerate(unique_signals):
        contexts = signal_contexts[sig]
        if not contexts:
            continue

        vals = [c['human_analog'].get('valence', 0) for c in contexts]
        ars = [c['human_analog'].get('arousal', 0.5) for c in contexts]
        has_threat = [1 if c.get('threats', []) else 0 for c in contexts]
        has_opportunity = [1 if c.get('opportunities', []) else 0 for c in contexts]
        has_social = [1 if 'other agent' in c.get('social_context', '').lower() else 0
                      for c in contexts]
        is_resource = [1 if any('food' in o.lower() or 'water' in o.lower()
                              for o in c.get('opportunities', [])) else 0
                      for c in contexts]

        signal_features[idx] = [
            np.mean(vals), np.mean(ars),
            np.mean(has_threat), np.mean(has_opportunity),
            np.mean(has_social), np.mean(is_resource),
        ]

    # Cluster signals by context features
    if n_unique < config.n_signal_clusters:
        n_clusters = max(2, n_unique // 2)
    else:
        n_clusters = config.n_signal_clusters

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='euclidean',
        linkage='ward',
    )
    labels = clustering.fit_predict(signal_features)

    # Build cluster dictionary
    cluster_dict = {}
    for c_id in range(n_clusters):
        member_indices = np.where(labels == c_id)[0]
        member_signals = [unique_signals[i] for i in member_indices]

        # Collect all annotations for this cluster
        cluster_annotations = []
        for sig in member_signals:
            cluster_annotations.extend(signal_contexts[sig])

        # Determine dominant theme
        if cluster_annotations:
            avg_valence = np.mean([a['human_analog'].get('valence', 0) for a in cluster_annotations])
            avg_arousal = np.mean([a['human_analog'].get('arousal', 0.5) for a in cluster_annotations])
            emotions = [a['human_analog'].get('dominant_emotion', 'unknown') for a in cluster_annotations]
            from collections import Counter
            dominant = Counter(emotions).most_common(1)[0][0]
        else:
            avg_valence, avg_arousal, dominant = 0, 0.5, 'unknown'

        cluster_dict[c_id] = {
            'signals': member_signals,
            'n_signals': len(member_signals),
            'n_annotations': len(cluster_annotations),
            'avg_valence': avg_valence,
            'avg_arousal': avg_arousal,
            'dominant_emotion': dominant,
            'feature_centroid': signal_features[member_indices].mean(axis=0).tolist(),
        }

    # Map each original signal to cluster label
    signal_cluster_labels = np.array([labels[signal_to_idx[s]] for s in signal_ids])

    return cluster_dict, signal_cluster_labels


# ============================================================================
# Affect concept embedding
# ============================================================================

def embed_affect_concepts(config: TranslationConfig) -> np.ndarray:
    """
    Embed the standardized affect concepts into embedding space.

    Returns: (n_concepts, embed_dim) matrix
    """
    if not HAS_OPENAI:
        raise ImportError("openai package required for embeddings")

    client = OpenAI()
    response = client.embeddings.create(
        model=config.embedding_model,
        input=config.affect_concepts,
    )

    embeddings = np.array([e.embedding for e in response.data])
    return embeddings


def embed_annotations(
    annotations: List[Dict],
    config: TranslationConfig,
) -> np.ndarray:
    """
    Embed VLM annotations into the same embedding space as affect concepts.

    Returns: (N, embed_dim)
    """
    if not HAS_OPENAI:
        raise ImportError("openai package required for embeddings")

    client = OpenAI()

    # Build text from annotations
    texts = []
    for ann in annotations:
        if 'error' in ann:
            texts.append("unknown situation")
            continue

        parts = [ann.get('situation', '')]
        ha = ann.get('human_analog', {})
        if ha.get('dominant_emotion'):
            parts.append(f"Feeling: {ha['dominant_emotion']}")
        if ha.get('secondary_emotions'):
            parts.append(f"Also: {', '.join(ha['secondary_emotions'])}")
        texts.append('. '.join(parts))

    # Batch embed
    embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=config.embedding_model,
            input=batch,
        )
        batch_embs = [e.embedding for e in response.data]
        embeddings.extend(batch_embs)

    return np.array(embeddings)


def project_to_affect_space(
    annotation_embeddings: np.ndarray,
    concept_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Project annotation embeddings into affect concept space.

    Each annotation gets a vector of similarities to each affect concept.
    This is the embedding-predicted affect vector e_i.

    Returns: (N, n_concepts) similarity matrix
    """
    # Cosine similarity between annotations and concepts
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(annotation_embeddings, concept_embeddings)
    return similarities


# ============================================================================
# Translation dictionary
# ============================================================================

@dataclass
class TranslationDictionary:
    """Maps signal clusters to affect annotations."""
    cluster_dict: Dict
    cluster_affect_vectors: np.ndarray  # (n_clusters, n_concepts)
    concept_names: List[str]
    validation_accuracy: float = 0.0

    def translate(self, signal: tuple) -> Optional[Dict]:
        """Translate a signal to its cluster annotation."""
        for c_id, cluster in self.cluster_dict.items():
            if signal in cluster['signals']:
                return {
                    'cluster_id': c_id,
                    'dominant_emotion': cluster['dominant_emotion'],
                    'avg_valence': cluster['avg_valence'],
                    'avg_arousal': cluster['avg_arousal'],
                    'affect_vector': self.cluster_affect_vectors[c_id],
                }
        return None

    def save(self, path: str):
        """Save translation dictionary to file."""
        data = {
            'cluster_dict': {k: {kk: vv for kk, vv in v.items() if kk != 'signals'
                                 or isinstance(vv, (int, float, str, list))}
                            for k, v in self.cluster_dict.items()},
            'cluster_affect_vectors': self.cluster_affect_vectors.tolist(),
            'concept_names': self.concept_names,
            'validation_accuracy': self.validation_accuracy,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


def build_translation_dictionary(
    signal_history: np.ndarray,
    scene_indices: np.ndarray,
    annotations: List[Dict],
    config: TranslationConfig,
) -> TranslationDictionary:
    """
    Build complete translation dictionary from signals to affect space.

    Steps:
    1. Cluster signals by context
    2. Embed affect concepts
    3. Embed annotations
    4. Project to affect concept space
    5. Aggregate by cluster
    6. Validate on holdout
    """
    print("Building translation dictionary...")

    # 1. Cluster signals
    print("  Step 1: Clustering signals...")
    cluster_dict, cluster_labels = cluster_signals(
        signal_history, scene_indices, annotations, config
    )
    print(f"  Found {len(cluster_dict)} signal clusters")

    # 2. Embed concepts
    print("  Step 2: Embedding affect concepts...")
    concept_embeddings = embed_affect_concepts(config)
    print(f"  Concept embedding shape: {concept_embeddings.shape}")

    # 3. Embed annotations
    print("  Step 3: Embedding annotations...")
    ann_embeddings = embed_annotations(annotations, config)
    print(f"  Annotation embedding shape: {ann_embeddings.shape}")

    # 4. Project to affect space
    print("  Step 4: Projecting to affect concept space...")
    affect_similarities = project_to_affect_space(ann_embeddings, concept_embeddings)

    # 5. Aggregate by cluster
    print("  Step 5: Aggregating by cluster...")
    n_clusters = len(cluster_dict)
    n_concepts = len(config.affect_concepts)
    cluster_affect_vectors = np.zeros((n_clusters, n_concepts))

    for c_id in range(n_clusters):
        # Find annotations in this cluster
        member_signals = cluster_dict[c_id]['signals']
        cluster_scene_indices = []
        for i, sig in enumerate(signal_history):
            if tuple(sig.tolist()) in set(member_signals):
                if scene_indices[i] < len(annotations):
                    cluster_scene_indices.append(scene_indices[i])

        if cluster_scene_indices:
            cluster_affect_vectors[c_id] = affect_similarities[cluster_scene_indices].mean(axis=0)

    # 6. Validate
    print("  Step 6: Validating...")
    n_holdout = int(len(annotations) * config.holdout_fraction)
    if n_holdout > 0:
        # Check if signal cluster predicts VLM annotation better than chance
        holdout_indices = np.random.choice(len(annotations), n_holdout, replace=False)
        holdout_anns = [annotations[i] for i in holdout_indices]

        # Predict dominant emotion from cluster
        correct = 0
        total = 0
        for i in holdout_indices:
            if i >= len(cluster_labels):
                continue
            c_id = cluster_labels[i]
            predicted = cluster_dict.get(c_id, {}).get('dominant_emotion', 'unknown')
            actual = annotations[i].get('human_analog', {}).get('dominant_emotion', 'unknown')
            if predicted == actual and predicted != 'unknown':
                correct += 1
            total += 1

        accuracy = correct / max(total, 1)
        print(f"  Validation accuracy: {accuracy:.3f} ({correct}/{total})")
    else:
        accuracy = 0.0

    dictionary = TranslationDictionary(
        cluster_dict=cluster_dict,
        cluster_affect_vectors=cluster_affect_vectors,
        concept_names=config.affect_concepts,
        validation_accuracy=accuracy,
    )

    return dictionary


# ============================================================================
# Full pipeline
# ============================================================================

def run_translation_pipeline(
    env,
    env_states: List,
    signal_history: np.ndarray,
    config: Optional[TranslationConfig] = None,
) -> Tuple[TranslationDictionary, np.ndarray]:
    """
    Run full VLM translation pipeline.

    Args:
        env: SurvivalGridWorld instance
        env_states: List of EnvState at each timestep
        signal_history: (T, n_agents, n_signal_tokens)
        config: Translation config

    Returns:
        (translation_dict, embedding_affect_vectors)
        embedding_affect_vectors: (T * n_agents, n_concepts)
    """
    config = config or TranslationConfig()
    os.makedirs(config.output_dir, exist_ok=True)

    n_agents = signal_history.shape[1] if len(signal_history.shape) > 1 else 1
    T = len(env_states)

    # 1. Render scenes for all agents at all timesteps
    print("Rendering scenes...")
    scenes = []
    scene_map = []  # (timestep, agent_id) for each scene
    for t in range(0, T, max(1, T // config.min_scenes)):
        for a in range(n_agents):
            scene = env.render_scene(env_states[t], a)
            scenes.append(scene)
            scene_map.append((t, a))

    print(f"Rendered {len(scenes)} scenes")

    # 2. Annotate with VLM
    print("Annotating scenes with VLM...")
    annotations = annotate_scenes_vlm(scenes, config)

    # Save annotations
    with open(f'{config.output_dir}/annotations.json', 'w') as f:
        json.dump(annotations, f, indent=2)

    # 3. Flatten signal history
    flat_signals = signal_history.reshape(-1, signal_history.shape[-1])
    scene_indices = np.zeros(len(flat_signals), dtype=int)
    # Map each signal to its nearest scene
    for i in range(len(flat_signals)):
        t = i // n_agents
        a = i % n_agents
        # Find closest scene
        best_idx = 0
        best_dist = float('inf')
        for j, (st, sa) in enumerate(scene_map):
            if sa == a and abs(st - t) < best_dist:
                best_dist = abs(st - t)
                best_idx = j
        scene_indices[i] = best_idx

    # 4. Build translation dictionary
    dictionary = build_translation_dictionary(
        flat_signals, scene_indices, annotations, config
    )

    # Save dictionary
    dictionary.save(f'{config.output_dir}/translation_dict.json')

    # 5. Get embedding-predicted affect vectors for all states
    print("Computing embedding-predicted affect vectors...")
    concept_embeddings = embed_affect_concepts(config)
    ann_embeddings = embed_annotations(annotations, config)
    affect_similarities = project_to_affect_space(ann_embeddings, concept_embeddings)

    # Map back to full (T * n_agents) shape
    embedding_affect_vectors = np.zeros((T * n_agents, len(config.affect_concepts)))
    for i in range(len(flat_signals)):
        scene_idx = scene_indices[i]
        if scene_idx < len(affect_similarities):
            embedding_affect_vectors[i] = affect_similarities[scene_idx]

    print(f"Embedding affect vectors shape: {embedding_affect_vectors.shape}")

    return dictionary, embedding_affect_vectors


if __name__ == '__main__':
    # Test scene rendering (doesn't require API)
    from v10_environment import SurvivalGridWorld, EnvConfig
    import jax.numpy as jnp
    from jax import random

    config = EnvConfig(n_agents=4, grid_size=8)
    env = SurvivalGridWorld(config)

    rng = random.PRNGKey(42)
    state, obs = env.reset(rng)

    # Render a few scenes
    for agent_id in range(config.n_agents):
        scene = env.render_scene(state, agent_id)
        text = render_scene_text(scene)
        print(f"\nAgent {agent_id}:")
        print(f"  {text}")
