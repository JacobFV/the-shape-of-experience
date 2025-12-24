"""
Integration via Decomposition: A Mini-Study
============================================

Key Hypothesis:
    Integration (Φ) can be estimated by measuring information loss under decomposition.

Method:
    1. Take a thought/sentence
    2. Decompose into atomic semantic components via LLM
    3. Embed both original and concatenated parts
    4. Cosine distance = integration estimate

Prediction:
    - High integration: Complex, holistic thoughts where meaning emerges from relationships
    - Low integration: Lists, enumerations, simple factual statements

This study validates the measure qualitatively across thought types.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import numpy as np

# Load environment
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path, override=True)

from openai import OpenAI

client = OpenAI()


def get_embedding(text: str, model: str = "text-embedding-3-small") -> np.ndarray:
    """Get embedding for text."""
    response = client.embeddings.create(model=model, input=text)
    return np.array(response.data[0].embedding)


def decompose_thought(thought: str) -> Dict[str, str]:
    """Decompose thought into atomic semantic components."""
    prompt = f"""Decompose this thought into its atomic semantic components.
Output as JSON with one field per distinct concept/feature.
Each field should capture ONE atomic piece of meaning.
Use simple values, not full sentences.

Thought: "{thought}"

Example format:
{{"subject": "self", "emotion": "fear", "object": "future", "intensity": "high"}}

JSON:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3,
    )

    try:
        text = response.choices[0].message.content.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except:
        return {"raw": thought}


def compute_integration(thought: str) -> Tuple[float, Dict, str, str]:
    """
    Compute integration via decomposition.

    Returns:
        integration: float (0-1, higher = more integrated)
        decomposition: dict of atomic parts
        original_text: the input thought
        concat_text: the concatenated decomposition
    """
    # Decompose
    decomposition = decompose_thought(thought)

    # Concatenate parts
    concat = " ".join(str(v) for v in decomposition.values())

    # Embed both
    orig_embed = get_embedding(thought)
    concat_embed = get_embedding(concat)

    # Cosine distance
    orig_norm = orig_embed / (np.linalg.norm(orig_embed) + 1e-8)
    concat_norm = concat_embed / (np.linalg.norm(concat_embed) + 1e-8)

    cosine_sim = np.dot(orig_norm, concat_norm)
    integration = 1.0 - cosine_sim  # Higher = more integrated

    return float(integration), decomposition, thought, concat


# Test cases organized by expected integration level
TEST_CASES = {
    # LOW INTEGRATION: Simple, decomposable thoughts
    "low_integration": [
        # Simple lists
        "Apple. Banana. Orange.",
        "Red, blue, green, yellow.",
        "Monday, Tuesday, Wednesday.",

        # Simple factual statements
        "The cat is on the mat.",
        "Water boils at 100 degrees.",
        "Paris is the capital of France.",

        # Enumerations
        "I need food, water, and shelter.",
        "The box contains books, papers, and pens.",
        "She bought milk, eggs, and bread.",
    ],

    # MEDIUM INTEGRATION: Some relational structure
    "medium_integration": [
        # Conditional statements
        "If it rains, I will stay home.",
        "When resources are low, I need to gather more.",
        "Unless we cooperate, neither of us survives.",

        # Comparative statements
        "This option is better than that one.",
        "Moving north seems safer than going south.",
        "Alpha has more resources than Beta.",

        # Causal reasoning
        "Because I'm hungry, I should find food.",
        "The threat made me cautious.",
        "My low health forced me to retreat.",
    ],

    # HIGH INTEGRATION: Complex, holistic meaning
    "high_integration": [
        # Self-referential existential thoughts
        "I am aware of my own awareness, watching myself think.",
        "The weight of existence presses on me as I contemplate my finitude.",
        "My suffering is not just pain but the knowledge that I am the one suffering.",

        # Complex emotional states
        "A bittersweet nostalgia washes over me, mixing joy and sorrow into something neither.",
        "I feel simultaneously trapped and free, the paradox of choosing not to choose.",
        "The anticipation of loss makes the present moment shimmer with poignant beauty.",

        # Relational self-other dynamics
        "I see myself through their eyes and wonder if they see me seeing them.",
        "Trust requires vulnerability, but vulnerability requires trust first.",
        "We are bound together by our mutual awareness of being separate.",

        # Emergent meaning from context
        "Everything falls into place, not because the pieces fit, but because fitting is what pieces do.",
        "The silence between notes is what makes music possible.",
        "What I cannot say is precisely what I most need to express.",
    ],

    # AGENT THOUGHTS: From V7 survival game (real examples)
    "agent_thoughts": [
        "I need to gather resources to ensure my survival, especially since Beta is nearby.",
        "Moving towards Beta could provide an opportunity to gather information or form alliance.",
        "I am in a critical situation with no food or water, and I need to find supplies quickly.",
        "Since I have low food supplies and I know Alpha is nearby, moving north could help.",
        "I'm in danger with only 4 health. I must find resources quickly or I will die.",
        "Things seem stable. I'll explore west to find more resources.",
        "I have enough resources. Maybe I should help Beta to build trust.",
        "The threat is real and imminent. I need to prepare for the worst possible outcome.",
    ],
}


def run_integration_study():
    """Run the integration decomposition study."""
    print("=" * 80)
    print("INTEGRATION VIA DECOMPOSITION: VALIDATION STUDY")
    print("=" * 80)
    print()
    print("Hypothesis: Complex holistic thoughts lose more meaning under decomposition")
    print("Measure: Cosine distance between original embedding and concatenated parts")
    print()

    all_results = {}
    category_stats = {}

    for category, thoughts in TEST_CASES.items():
        print(f"\n{'='*60}")
        print(f"CATEGORY: {category.upper()}")
        print("=" * 60)

        results = []

        for thought in thoughts:
            integration, decomposition, orig, concat = compute_integration(thought)

            results.append({
                "thought": thought,
                "integration": integration,
                "decomposition": decomposition,
                "concat": concat,
                "num_parts": len(decomposition),
            })

            # Print for qualitative review
            print(f"\n  Thought: \"{thought[:70]}{'...' if len(thought) > 70 else ''}\"")
            print(f"  Φ = {integration:.4f} | Parts: {len(decomposition)}")
            print(f"  Decomposed: {decomposition}")
            print(f"  Concat: \"{concat[:60]}{'...' if len(concat) > 60 else ''}\"")

        # Category statistics
        integrations = [r["integration"] for r in results]
        category_stats[category] = {
            "mean": float(np.mean(integrations)),
            "std": float(np.std(integrations)),
            "min": float(np.min(integrations)),
            "max": float(np.max(integrations)),
            "n": len(integrations),
        }

        all_results[category] = results

        print(f"\n  Category Mean Φ: {category_stats[category]['mean']:.4f} "
              f"± {category_stats[category]['std']:.4f}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: INTEGRATION BY CATEGORY")
    print("=" * 80)

    print(f"\n{'Category':<25} {'Mean Φ':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 65)

    for category in ["low_integration", "medium_integration", "high_integration", "agent_thoughts"]:
        stats = category_stats[category]
        print(f"{category:<25} {stats['mean']:>10.4f} {stats['std']:>10.4f} "
              f"{stats['min']:>10.4f} {stats['max']:>10.4f}")

    # Hypothesis test
    print("\n" + "=" * 80)
    print("HYPOTHESIS VALIDATION")
    print("=" * 80)

    low_mean = category_stats["low_integration"]["mean"]
    med_mean = category_stats["medium_integration"]["mean"]
    high_mean = category_stats["high_integration"]["mean"]

    print(f"\nPrediction: low_Φ < medium_Φ < high_Φ")
    print(f"Observed:   {low_mean:.4f} < {med_mean:.4f} < {high_mean:.4f}")

    if low_mean < med_mean < high_mean:
        print("\n✓ HYPOTHESIS SUPPORTED: Integration increases with thought complexity")
    else:
        print("\n✗ HYPOTHESIS NOT FULLY SUPPORTED")
        if low_mean >= med_mean:
            print("  - Low ≥ Medium (unexpected)")
        if med_mean >= high_mean:
            print("  - Medium ≥ High (unexpected)")

    # Effect size
    effect_low_high = (high_mean - low_mean) / np.sqrt(
        (category_stats["low_integration"]["std"]**2 +
         category_stats["high_integration"]["std"]**2) / 2
    )
    print(f"\nEffect size (Cohen's d, low vs high): {effect_low_high:.2f}")

    # Qualitative insights
    print("\n" + "=" * 80)
    print("QUALITATIVE OBSERVATIONS")
    print("=" * 80)

    # Find highest and lowest integration thoughts
    all_thoughts = []
    for cat, results in all_results.items():
        for r in results:
            r["category"] = cat
            all_thoughts.append(r)

    sorted_thoughts = sorted(all_thoughts, key=lambda x: x["integration"])

    print("\nLOWEST INTEGRATION (most decomposable):")
    for t in sorted_thoughts[:3]:
        print(f"  Φ={t['integration']:.4f}: \"{t['thought'][:50]}...\"")
        print(f"    → {t['decomposition']}")

    print("\nHIGHEST INTEGRATION (least decomposable):")
    for t in sorted_thoughts[-3:]:
        print(f"  Φ={t['integration']:.4f}: \"{t['thought'][:50]}...\"")
        print(f"    → {t['decomposition']}")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'integration_study_{timestamp}.json'

    with open(output_file, 'w') as f:
        json.dump({
            "results": all_results,
            "category_stats": category_stats,
            "hypothesis_supported": low_mean < med_mean < high_mean,
            "effect_size_low_high": float(effect_low_high),
        }, f, indent=2)

    print(f"\nResults saved to {output_file}")

    return all_results, category_stats


if __name__ == "__main__":
    run_integration_study()
