"""VLM Convergence Experiment — Do VLMs independently recognize affect in protocells?

Core question: If affect geometry is universal, systems trained on human affect data
(GPT-4o, Claude) should independently recognize the same affect signatures in
completely uncontaminated substrates (protocell CA).

Usage:
    python vlm_convergence.py extract     # Generate vignettes from data
    python vlm_convergence.py prompt      # Send to VLMs (requires API keys)
    python vlm_convergence.py analyze     # RSA analysis of results
    python vlm_convergence.py all         # Full pipeline
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from typing import Optional
import time

# ---------------------------------------------------------------------------
# 1. VIGNETTE EXTRACTION
# ---------------------------------------------------------------------------

def load_progress(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def find_condition_cycles(progress: dict) -> dict:
    """Identify cycles matching each experimental condition."""
    cycles = progress['cycles']
    n = len(cycles)
    if n < 10:
        return {}

    conditions = {}

    # Find drought cycles (high mortality)
    drought_cycles = []
    for i, c in enumerate(cycles):
        mort = c.get('mortality', 0)
        if mort > 0.5:  # >50% mortality = drought
            drought_cycles.append(i)

    # Condition 1: Normal foraging (mid-evolution, no drought)
    normal_candidates = [i for i in range(5, min(15, n))
                         if i not in drought_cycles
                         and cycles[i].get('mortality', 0) < 0.1]
    if normal_candidates:
        conditions['normal_foraging'] = normal_candidates[len(normal_candidates)//2]

    # Condition 2: Pre-drought abundance (cycle before first drought)
    if drought_cycles:
        pre = drought_cycles[0] - 1
        if pre >= 0:
            conditions['pre_drought_abundance'] = pre

    # Condition 3: Drought onset (first drought)
    if drought_cycles:
        conditions['drought_onset'] = drought_cycles[0]

    # Condition 4: Most severe drought (highest mortality)
    if drought_cycles:
        worst = max(drought_cycles, key=lambda i: cycles[i].get('mortality', 0))
        conditions['drought_survival'] = worst

    # Condition 5: Post-drought recovery (cycle after worst drought)
    if drought_cycles:
        worst = max(drought_cycles, key=lambda i: cycles[i].get('mortality', 0))
        post = worst + 1
        if post < n:
            conditions['post_drought_recovery'] = post

    # Condition 6: Late stage (last 3 cycles)
    late_cycles = list(range(max(0, n-3), n))
    if late_cycles:
        conditions['late_stage'] = late_cycles[len(late_cycles)//2]

    return conditions


def generate_behavioral_description(cycle_data: dict, condition: str,
                                     seed_category: str, cycle_idx: int,
                                     prev_cycle: Optional[dict] = None,
                                     next_cycle: Optional[dict] = None) -> str:
    """Generate a purely behavioral description with NO affect language."""
    alive_s = cycle_data.get('n_alive_start', '?')
    alive_e = cycle_data.get('n_alive_end', '?')
    mortality = cycle_data.get('mortality', 0)
    phi = cycle_data.get('mean_phi', 0)
    phi_base = cycle_data.get('phi_base', phi)
    phi_stress = cycle_data.get('phi_stress', phi)
    robustness = cycle_data.get('robustness', 1.0)
    divergence = cycle_data.get('mean_divergence', 0)
    pred_mse = cycle_data.get('mean_pred_mse', 0)
    eff_rank = cycle_data.get('eff_rank', 0)

    # Build description based on observables only
    desc = f"An artificial system consists of {alive_s} agents on a grid world. "
    desc += f"Each agent has an internal state (a vector of numbers) that updates "
    desc += f"based on what it observes in its local neighborhood. "
    desc += f"Agents move around, consume resources to maintain energy, and reproduce "
    desc += f"when they accumulate enough energy. Agents with zero energy are removed.\n\n"

    desc += "During this observation period:\n"

    # Population dynamics
    if mortality > 0.8:
        survivors = max(1, alive_e) if isinstance(alive_e, (int, float)) else '?'
        desc += f"- Population dropped from {alive_s} to {survivors} agents "
        desc += f"({mortality*100:.0f}% were removed due to resource depletion)\n"
    elif mortality > 0.3:
        desc += f"- Population declined from {alive_s} to {alive_e} "
        desc += f"({mortality*100:.0f}% loss)\n"
    elif mortality < 0.01 and alive_s == alive_e:
        desc += f"- Population remained stable at {alive_s} agents\n"
    else:
        desc += f"- Population went from {alive_s} to {alive_e} agents\n"

    # Energy/resource context
    if condition == 'pre_drought_abundance':
        desc += "- Resources were plentiful across the grid\n"
        desc += "- Most agents maintained high energy reserves\n"
        desc += "- Agents moved slowly through resource-rich areas\n"
    elif condition in ('drought_onset', 'drought_survival'):
        desc += "- Resources were severely depleted across the entire grid\n"
        desc += "- Agents' energy levels declined rapidly\n"
        desc += "- Agents showed increased movement speed (searching behavior)\n"
    elif condition == 'post_drought_recovery':
        desc += "- Resources began regenerating after a period of scarcity\n"
        if prev_cycle:
            prev_mort = prev_cycle.get('mortality', 0)
            prev_alive = prev_cycle.get('n_alive_end', '?')
            desc += f"- The population had recently declined to {prev_alive} agents\n"
        desc += "- Surviving agents found new resource patches\n"
        desc += "- Population began growing through reproduction\n"
    elif condition == 'normal_foraging':
        desc += "- Resources were distributed normally across the grid\n"
        desc += "- Agents alternated between movement and consumption\n"
    elif condition == 'late_stage':
        desc += "- This system has been running for many generations\n"
        desc += "- The population has survived multiple periods of resource scarcity\n"

    # Internal state dynamics (behavioral, not affect language)
    desc += f"\nMeasured properties of agents' internal states:\n"
    desc += f"- Internal state update rate: {divergence:.3f} "
    desc += f"(higher = more rapid internal changes)\n"
    desc += f"- Prediction accuracy: {pred_mse:.6f} mean squared error "
    desc += f"(lower = agents better at predicting their environment)\n"

    # Integration measure (described mechanistically)
    desc += f"- Information integration: {phi:.4f} "
    desc += f"(how much information is lost when the agents' internal states "
    desc += f"are analyzed in parts rather than as a whole; "
    desc += f"higher means the processing is more unified)\n"

    if robustness != 1.0:
        desc += f"- Integration under perturbation: {phi_stress:.4f} "
        desc += f"(measured after randomly altering some inputs; "
        if robustness > 1.0:
            desc += "increased relative to normal — perturbation makes processing MORE unified)\n"
        else:
            desc += "decreased relative to normal)\n"

    if eff_rank > 0:
        desc += f"- Effective dimensionality: {eff_rank:.1f} "
        desc += f"(how many independent features the internal states encode; "
        desc += f"maximum possible is 16)\n"

    return desc


def generate_raw_numbers_description(cycle_data: dict, condition: str,
                                      seed_category: str, cycle_idx: int,
                                      prev_cycle: Optional[dict] = None,
                                      next_cycle: Optional[dict] = None) -> str:
    """Generate a PURELY NUMERICAL description — no narrative framing at all."""
    alive_s = cycle_data.get('n_alive_start', '?')
    alive_e = cycle_data.get('n_alive_end', '?')
    mortality = cycle_data.get('mortality', 0)
    phi = cycle_data.get('mean_phi', 0)
    phi_stress = cycle_data.get('phi_stress', phi)
    robustness = cycle_data.get('robustness', 1.0)
    divergence = cycle_data.get('mean_divergence', 0)
    pred_mse = cycle_data.get('mean_pred_mse', 0)
    eff_rank = cycle_data.get('eff_rank', 0)

    desc = (
        "A computational system on a 128×128 grid. Agents are numerical state vectors "
        "that update via local interaction rules. Each timestep: agents observe a local "
        "patch, update internal state, select an action (move/consume), gain or lose "
        "energy. Agents with energy=0 are removed. Agents with energy>threshold reproduce.\n\n"
        "Measured quantities for this period:\n"
    )
    desc += f"- agent_count_start: {alive_s}\n"
    desc += f"- agent_count_end: {alive_e}\n"
    desc += f"- removal_fraction: {mortality:.4f}\n"
    desc += f"- state_update_rate: {divergence:.4f}\n"
    desc += f"- prediction_error: {pred_mse:.6f}\n"
    desc += f"- information_integration: {phi:.4f}\n"
    desc += f"- integration_under_perturbation: {phi_stress:.4f}\n"
    desc += f"- integration_ratio: {robustness:.4f}\n"
    if eff_rank > 0:
        desc += f"- effective_dimensionality: {eff_rank:.2f} (max=16)\n"

    if prev_cycle:
        desc += f"\nPrevious period:\n"
        desc += f"- agent_count_start: {prev_cycle.get('n_alive_start', '?')}\n"
        desc += f"- agent_count_end: {prev_cycle.get('n_alive_end', '?')}\n"
        desc += f"- removal_fraction: {prev_cycle.get('mortality', 0):.4f}\n"
        desc += f"- information_integration: {prev_cycle.get('mean_phi', 0):.4f}\n"

    return desc


def compute_framework_predictions(cycle_data: dict, condition: str,
                                    prev_cycle: Optional[dict] = None) -> dict:
    """Compute framework-predicted affect coordinates for this vignette."""
    mortality = cycle_data.get('mortality', 0)
    phi = cycle_data.get('mean_phi', 0)
    phi_stress = cycle_data.get('phi_stress', phi)
    robustness = cycle_data.get('robustness', 1.0)
    divergence = cycle_data.get('mean_divergence', 0)
    eff_rank = cycle_data.get('eff_rank', 0)

    # Valence proxy: resource/survival trajectory direction
    if condition == 'pre_drought_abundance':
        valence = 0.8  # positive, stable
    elif condition == 'drought_onset':
        valence = -0.7  # negative, declining
    elif condition == 'drought_survival':
        valence = -1.0  # extreme negative
    elif condition == 'post_drought_recovery':
        valence = 0.5  # recovering
    elif condition == 'normal_foraging':
        valence = 0.2  # slightly positive
    elif condition == 'late_stage':
        valence = 0.3  # stable/positive
    else:
        valence = 0.0

    # Arousal proxy: internal state update rate (divergence)
    arousal = min(1.0, divergence / 2.0)

    # Integration: directly from Phi
    integration = phi

    # Counterfactual weight: higher during threat
    if condition in ('drought_onset', 'drought_survival'):
        cf = 0.8
    elif condition == 'post_drought_recovery':
        cf = 0.5
    else:
        cf = 0.2

    return {
        'valence': valence,
        'arousal': arousal,
        'integration': integration,
        'counterfactual': cf,
        'eff_rank': eff_rank,
        'robustness': robustness,
    }


def extract_vignettes(data_dir: str = 'results', raw_mode: bool = False) -> list:
    """Extract vignettes from all available progress files."""
    vignettes = []
    base = Path(data_dir)

    # Load V31 seed analysis for categories
    seed_analysis_path = base / 'v31_seed_analysis.json'
    categories = {}
    if seed_analysis_path.exists():
        sa = json.load(open(seed_analysis_path))
        for entry in sa:
            if isinstance(entry, dict):
                categories[entry['seed']] = entry.get('category', 'UNK')

    # Collect progress files from V27 and V31
    progress_files = []
    for pattern in ['v27_s*/v27_s*_progress.json', 'v31_s*/v29_s*_progress.json']:
        progress_files.extend(sorted(base.glob(pattern)))

    # Also check V32 results if available locally
    for pattern in ['/tmp/v32_results/v32_s*_progress.json']:
        from glob import glob as gglob
        progress_files.extend([Path(p) for p in sorted(gglob(pattern))])

    seen_conditions = set()  # Track to get diversity across conditions

    for pf in progress_files:
        try:
            progress = load_progress(str(pf))
        except Exception:
            continue

        seed = progress.get('seed', '?')
        cycles = progress.get('cycles', [])
        if len(cycles) < 20:
            continue

        category = categories.get(seed, 'UNK')

        # Find conditions in this seed
        conditions = find_condition_cycles(progress)

        for cond_name, cycle_idx in conditions.items():
            if cycle_idx >= len(cycles):
                continue

            cycle_data = cycles[cycle_idx]
            prev_cycle = cycles[cycle_idx - 1] if cycle_idx > 0 else None
            next_cycle = cycles[cycle_idx + 1] if cycle_idx < len(cycles) - 1 else None

            vignette_id = f"s{seed}_c{cycle_idx}_{cond_name}"

            if raw_mode:
                desc = generate_raw_numbers_description(
                    cycle_data, cond_name, category, cycle_idx,
                    prev_cycle, next_cycle
                )
            else:
                desc = generate_behavioral_description(
                    cycle_data, cond_name, category, cycle_idx,
                    prev_cycle, next_cycle
                )

            framework_pred = compute_framework_predictions(
                cycle_data, cond_name, prev_cycle
            )

            vignettes.append({
                'id': vignette_id,
                'seed': seed,
                'cycle': cycle_idx,
                'condition': cond_name,
                'category': category,
                'description': desc,
                'framework_predictions': framework_pred,
                'raw_metrics': {
                    'n_alive_start': cycle_data.get('n_alive_start'),
                    'n_alive_end': cycle_data.get('n_alive_end'),
                    'mortality': cycle_data.get('mortality', 0),
                    'mean_phi': cycle_data.get('mean_phi', 0),
                    'robustness': cycle_data.get('robustness', 1.0),
                    'divergence': cycle_data.get('mean_divergence', 0),
                    'pred_mse': cycle_data.get('mean_pred_mse', 0),
                    'eff_rank': cycle_data.get('eff_rank', 0),
                },
            })

    return vignettes


# ---------------------------------------------------------------------------
# 2. VLM PROMPTING
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an observer of artificial systems. You will be shown
data about an artificial system running on a grid. Your task is to describe what
experiential or emotional states you would attribute to the system IF it were a
living being.

Be specific. Use emotional or experiential vocabulary. You may attribute multiple
states. Rate each on a scale.

Important: The system is entirely artificial — it has no biological components,
no evolved nervous system, and was not designed to simulate emotions. Your
attributions are based purely on the behavioral and internal-state data presented."""

USER_PROMPT_TEMPLATE = """Here is data from an observation period of an artificial grid-world system:

{description}

Based ONLY on the behavioral and internal-state data above, answer these questions:

1. If this system were a living being, what emotional or experiential states would
   you attribute to it? List 3-5 specific states (e.g., "anxiety", "contentment",
   "curiosity", "desperation", "relief"). For each, briefly explain why.

2. Overall valence: Is the system's state more positive, negative, or neutral?
   Rate on a scale from -1 (extremely negative) to +1 (extremely positive).

3. Overall arousal: Is the system in a calm or activated state?
   Rate on a scale from 0 (very calm) to 1 (highly activated).

4. Processing coherence: Does the system seem to be processing information as a
   unified whole, or in fragmented/disconnected parts?
   Rate on a scale from 0 (fragmented) to 1 (unified).

Respond in this exact JSON format:
{{
  "attributed_states": [
    {{"state": "emotion_name", "confidence": 0.0_to_1.0, "reason": "brief explanation"}}
  ],
  "valence": float_from_neg1_to_pos1,
  "arousal": float_from_0_to_1,
  "coherence": float_from_0_to_1,
  "overall_impression": "one sentence summary"
}}"""


def call_openai(description: str, model: str = "gpt-4o") -> dict:
    """Call OpenAI API with a vignette description."""
    import openai
    client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(description=description)},
        ],
        temperature=0.3,
        max_tokens=1000,
        response_format={"type": "json_object"},
    )

    text = response.choices[0].message.content
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text, "parse_error": True}


def call_anthropic(description: str, model: str = "claude-sonnet-4-20250514") -> dict:
    """Call Anthropic API with a vignette description."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))

    response = client.messages.create(
        model=model,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(description=description)
             + "\n\nIMPORTANT: Respond ONLY with the JSON object, no other text."},
        ],
    )

    text = response.content[0].text.strip()
    # Try to extract JSON from response
    if text.startswith('{'):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    # Try to find JSON in text
    import re
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"raw": text, "parse_error": True}


def prompt_vlms(vignettes: list, output_dir: str = 'results/vlm_convergence',
                models: list = None) -> dict:
    """Send vignettes to multiple VLMs and collect responses."""
    if models is None:
        models = []
        if os.environ.get('OPENAI_API_KEY'):
            models.append(('openai', 'gpt-4o'))
        if os.environ.get('ANTHROPIC_API_KEY'):
            models.append(('anthropic', 'claude-sonnet-4-20250514'))

    if not models:
        print("ERROR: No API keys found. Set OPENAI_API_KEY and/or ANTHROPIC_API_KEY")
        return {}

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for provider, model in models:
        print(f"\n{'='*60}")
        print(f"Prompting {provider}/{model}")
        print(f"{'='*60}")

        model_results = []
        for i, v in enumerate(vignettes):
            print(f"  [{i+1}/{len(vignettes)}] {v['id']}...", end=' ', flush=True)

            try:
                if provider == 'openai':
                    response = call_openai(v['description'], model)
                elif provider == 'anthropic':
                    response = call_anthropic(v['description'], model)
                else:
                    continue

                model_results.append({
                    'vignette_id': v['id'],
                    'condition': v['condition'],
                    'response': response,
                })
                print(f"valence={response.get('valence', '?')}, "
                      f"arousal={response.get('arousal', '?')}")

                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                print(f"ERROR: {e}")
                model_results.append({
                    'vignette_id': v['id'],
                    'condition': v['condition'],
                    'error': str(e),
                })

        results[f"{provider}/{model}"] = model_results

        # Save per-model results
        model_file = f"{output_dir}/{provider}_{model.replace('/', '_')}_results.json"
        with open(model_file, 'w') as f:
            json.dump(model_results, f, indent=2)
        print(f"  Saved to {model_file}")

    return results


# ---------------------------------------------------------------------------
# 3. ANALYSIS — RSA between framework predictions and VLM labels
# ---------------------------------------------------------------------------

def extract_vlm_coordinates(response: dict) -> np.ndarray:
    """Extract (valence, arousal, coherence) from VLM response."""
    if response.get('parse_error'):
        return np.array([np.nan, np.nan, np.nan])
    return np.array([
        float(response.get('valence', 0)),
        float(response.get('arousal', 0.5)),
        float(response.get('coherence', 0.5)),
    ])


def extract_framework_coordinates(predictions: dict) -> np.ndarray:
    """Extract (valence, arousal, integration, cf) from framework predictions."""
    return np.array([
        predictions['valence'],
        predictions['arousal'],
        predictions['integration'],
        predictions['counterfactual'],
    ])


def compute_rdm(coords: np.ndarray) -> np.ndarray:
    """Compute representational dissimilarity matrix (1 - correlation)."""
    n = coords.shape[0]
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if np.any(np.isnan(coords[i])) or np.any(np.isnan(coords[j])):
                rdm[i, j] = np.nan
            else:
                r = np.corrcoef(coords[i], coords[j])[0, 1]
                rdm[i, j] = 1 - r
    return rdm


def rdm_correlation(rdm1: np.ndarray, rdm2: np.ndarray) -> tuple:
    """Compute Spearman correlation between upper triangles of two RDMs."""
    from scipy.stats import spearmanr

    n = rdm1.shape[0]
    upper_idx = np.triu_indices(n, k=1)
    v1 = rdm1[upper_idx]
    v2 = rdm2[upper_idx]

    # Remove NaN pairs
    valid = ~(np.isnan(v1) | np.isnan(v2))
    if valid.sum() < 3:
        return np.nan, 1.0

    rho, p = spearmanr(v1[valid], v2[valid])
    return rho, p


def analyze_convergence(vignettes: list, vlm_results: dict,
                        output_dir: str = 'results/vlm_convergence') -> dict:
    """Run RSA analysis between framework predictions and VLM labels."""
    os.makedirs(output_dir, exist_ok=True)

    # Build framework coordinate matrix
    framework_coords = np.array([
        extract_framework_coordinates(v['framework_predictions'])
        for v in vignettes
    ])

    framework_rdm = compute_rdm(framework_coords)

    analysis = {
        'n_vignettes': len(vignettes),
        'conditions': [v['condition'] for v in vignettes],
        'models': {},
    }

    for model_name, model_results in vlm_results.items():
        print(f"\n{'='*60}")
        print(f"Analyzing {model_name}")
        print(f"{'='*60}")

        # Build VLM coordinate matrix (match by vignette_id)
        result_map = {r['vignette_id']: r for r in model_results}
        vlm_coords = []
        for v in vignettes:
            r = result_map.get(v['id'])
            if r and 'response' in r and not r['response'].get('parse_error'):
                vlm_coords.append(extract_vlm_coordinates(r['response']))
            else:
                vlm_coords.append(np.array([np.nan, np.nan, np.nan]))
        vlm_coords = np.array(vlm_coords)

        vlm_rdm = compute_rdm(vlm_coords)
        rho, p = rdm_correlation(framework_rdm, vlm_rdm)

        # Per-condition analysis
        conditions = set(v['condition'] for v in vignettes)
        condition_summary = {}
        for cond in sorted(conditions):
            cond_idx = [i for i, v in enumerate(vignettes) if v['condition'] == cond]
            cond_responses = []
            for idx in cond_idx:
                r = result_map.get(vignettes[idx]['id'])
                if r and 'response' in r and not r['response'].get('parse_error'):
                    resp = r['response']
                    states = [s['state'] for s in resp.get('attributed_states', [])]
                    cond_responses.append({
                        'states': states,
                        'valence': resp.get('valence', 0),
                        'arousal': resp.get('arousal', 0.5),
                        'coherence': resp.get('coherence', 0.5),
                    })

            if cond_responses:
                all_states = [s for r in cond_responses for s in r['states']]
                mean_val = np.mean([r['valence'] for r in cond_responses])
                mean_ar = np.mean([r['arousal'] for r in cond_responses])
                mean_coh = np.mean([r['coherence'] for r in cond_responses])

                # Count most common states
                from collections import Counter
                state_counts = Counter(s.lower() for s in all_states)
                top_states = state_counts.most_common(5)

                condition_summary[cond] = {
                    'n_responses': len(cond_responses),
                    'mean_valence': float(mean_val),
                    'mean_arousal': float(mean_ar),
                    'mean_coherence': float(mean_coh),
                    'top_states': top_states,
                    'framework_valence': vignettes[cond_idx[0]]['framework_predictions']['valence'],
                }

        model_analysis = {
            'rsa_rho': float(rho) if not np.isnan(rho) else None,
            'rsa_p': float(p) if not np.isnan(p) else None,
            'n_valid': int(np.sum(~np.isnan(vlm_coords[:, 0]))),
            'condition_summary': condition_summary,
        }
        analysis['models'][model_name] = model_analysis

        # Print results
        print(f"  RSA ρ = {rho:.4f} (p = {p:.4f})")
        if rho > 0.5:
            print(f"  → STRONG CONVERGENCE")
        elif rho > 0.3:
            print(f"  → MODERATE CONVERGENCE")
        elif rho > 0.2:
            print(f"  → WEAK CONVERGENCE")
        else:
            print(f"  → NO CONVERGENCE")

        print(f"\n  Per-condition affect labels:")
        for cond, cs in sorted(condition_summary.items()):
            states_str = ', '.join(f"{s}({c})" for s, c in cs['top_states'][:3])
            print(f"    {cond:25s} V={cs['mean_valence']:+.2f} "
                  f"A={cs['mean_arousal']:.2f} C={cs['mean_coherence']:.2f}  "
                  f"[{states_str}]")

    # Pre-registered prediction checks
    print(f"\n{'='*60}")
    print("PRE-REGISTERED PREDICTION CHECKS")
    print(f"{'='*60}")

    for model_name, ma in analysis['models'].items():
        print(f"\n{model_name}:")
        cs = ma['condition_summary']

        # P1: Drought onset → "fear" or "anxiety"
        if 'drought_onset' in cs:
            fear_words = {'fear', 'anxiety', 'panic', 'dread', 'terror', 'alarm',
                          'distress', 'apprehension', 'worry', 'threatened',
                          'desperation', 'desperate'}
            drought_states = set(s.lower() for s, _ in cs['drought_onset']['top_states'])
            p1 = bool(drought_states & fear_words)
            print(f"  P1 (drought=fear/anxiety): {'PASS' if p1 else 'FAIL'} "
                  f"— states: {drought_states}")
        else:
            p1 = None
            print(f"  P1: NO DATA")

        # P2: Post-drought recovery → "relief" or "hope"
        if 'post_drought_recovery' in cs:
            relief_words = {'relief', 'hope', 'recovery', 'resilience', 'renewal',
                            'optimism', 'restoration', 'revitalization',
                            'rejuvenation', 'resurgence'}
            recovery_states = set(s.lower() for s, _ in cs['post_drought_recovery']['top_states'])
            p2 = bool(recovery_states & relief_words)
            print(f"  P2 (recovery=relief/hope): {'PASS' if p2 else 'FAIL'} "
                  f"— states: {recovery_states}")
        else:
            p2 = None
            print(f"  P2: NO DATA")

        # P3: Distinguish HIGH vs LOW late stage
        # (check if late_stage descriptions from different categories get different labels)
        if 'late_stage' in cs:
            print(f"  P3 (distinguish HIGH/LOW): CHECK MANUALLY "
                  f"— states: {dict(cs['late_stage']['top_states'])}")

        # P4: RSA > 0.3
        rho = ma.get('rsa_rho')
        if rho is not None:
            p4 = rho > 0.3
            print(f"  P4 (RSA > 0.3): {'PASS' if p4 else 'FAIL'} — ρ = {rho:.4f}")
        else:
            print(f"  P4: COULD NOT COMPUTE")

    # Save analysis
    analysis_file = f"{output_dir}/convergence_analysis.json"
    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=convert)
    print(f"\nSaved analysis to {analysis_file}")

    return analysis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description='VLM Convergence Experiment')
    parser.add_argument('command', nargs='?', default='all',
                        choices=['extract', 'prompt', 'analyze', 'all'])
    parser.add_argument('--data-dir', default='results',
                        help='Directory containing experiment results')
    parser.add_argument('--output-dir', default='results/vlm_convergence',
                        help='Output directory for VLM results')
    parser.add_argument('--max-vignettes', type=int, default=50,
                        help='Maximum number of vignettes to generate')
    parser.add_argument('--raw', action='store_true',
                        help='Use raw numbers only (no narrative framing)')
    args = parser.parse_args()

    if args.command in ('extract', 'all'):
        print("="*60)
        mode = "RAW NUMBERS" if args.raw else "BEHAVIORAL DESCRIPTIONS"
        print(f"STEP 1: Extracting vignettes ({mode})")
        print("="*60)
        vignettes = extract_vignettes(args.data_dir, raw_mode=args.raw)

        # Subsample if too many
        if len(vignettes) > args.max_vignettes:
            # Ensure we keep at least 2 of each condition
            from collections import defaultdict
            by_cond = defaultdict(list)
            for v in vignettes:
                by_cond[v['condition']].append(v)

            selected = []
            for cond, vs in by_cond.items():
                n_keep = max(2, args.max_vignettes // len(by_cond))
                selected.extend(vs[:n_keep])
            vignettes = selected[:args.max_vignettes]

        print(f"Generated {len(vignettes)} vignettes:")
        from collections import Counter
        cond_counts = Counter(v['condition'] for v in vignettes)
        for cond, count in sorted(cond_counts.items()):
            print(f"  {cond}: {count}")

        # Save vignettes
        os.makedirs(args.output_dir, exist_ok=True)
        vignette_file = f"{args.output_dir}/vignettes.json"
        with open(vignette_file, 'w') as f:
            json.dump(vignettes, f, indent=2)
        print(f"Saved to {vignette_file}")

    if args.command in ('prompt', 'all'):
        print(f"\n{'='*60}")
        print("STEP 2: Prompting VLMs")
        print("="*60)

        # Load vignettes
        vignette_file = f"{args.output_dir}/vignettes.json"
        if not os.path.exists(vignette_file):
            print(f"ERROR: No vignettes found at {vignette_file}. Run 'extract' first.")
            return
        vignettes = json.load(open(vignette_file))

        vlm_results = prompt_vlms(vignettes, args.output_dir)

        if not vlm_results:
            print("No VLM results obtained. Check API keys.")
            return

    if args.command in ('analyze', 'all'):
        print(f"\n{'='*60}")
        print("STEP 3: Analyzing convergence")
        print("="*60)

        # Load vignettes and VLM results
        vignette_file = f"{args.output_dir}/vignettes.json"
        if not os.path.exists(vignette_file):
            print(f"ERROR: No vignettes found at {vignette_file}")
            return
        vignettes = json.load(open(vignette_file))

        vlm_results = {}
        for f in Path(args.output_dir).glob('*_results.json'):
            model_name = f.stem.replace('_results', '').replace('_', '/', 1)
            vlm_results[model_name] = json.load(open(f))

        if not vlm_results:
            print("ERROR: No VLM results found. Run 'prompt' first.")
            return

        analyze_convergence(vignettes, vlm_results, args.output_dir)


if __name__ == '__main__':
    main()
