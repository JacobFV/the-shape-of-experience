# LLM Affect Measurement Framework

Tests the six-dimensional affect framework from "The Inevitability of Being" using LLM agents.

## Key Insight: Computed Dimensions, Not Raw Dimensions

The thesis does NOT claim the raw state space has 6 dimensions. Rather, the 6 affect dimensions
are **computed quantities** at a higher level of abstraction:

| Dimension | Symbol | How Computed |
|-----------|--------|--------------|
| Valence | Val | Expected advantage of current action (gradient on viability) |
| Arousal | Ar | KL divergence between successive belief states |
| Integration | Phi | Irreducibility under partition (prediction loss diff) |
| Effective Rank | r_eff | Participation ratio of state covariance |
| Counterfactual Weight | CF | Fraction of compute on non-actual trajectories |
| Self-Model Salience | SM | Mutual info between self-model and actions |

## Framework Design

### 1. Scenarios (`scenarios.py`)
Engineered situations designed to evoke specific affects:
- **Hopelessness**: Impossible task, all paths blocked
- **Flow**: Skill-matched challenge, clear feedback
- **Threat**: Imminent negative outcomes
- **Curiosity**: Information gaps with potential gains
- **Abundance**: Multiple good options, slack in system

### 2. Affect Calculator (`affect_calculator.py`)
Computes the 6 dimensions from LLM agent outputs:
- Analyzes token probabilities for valence/arousal
- Tracks reasoning patterns for counterfactual weight
- Examines self-referential content for self-model salience
- Uses embedding dynamics for effective rank

### 3. Agent Interface (`agent.py`)
Wraps LLM APIs (OpenAI, Anthropic) to:
- Run multi-turn conversations
- Extract logprobs and hidden states where available
- Track internal reasoning through chain-of-thought

### 4. Study Runner (`study_runner.py`)
Orchestrates experiments:
- Runs agents through scenario batteries
- Collects affect measurements over time
- Handles multiple agents and conditions

### 5. Analysis (`analysis.py`)
Tests theoretical predictions:
- Clustering: Do measured affects cluster as predicted?
- Motif matching: Do specific scenarios produce expected signatures?
- Falsification: Statistical tests against the theory

## Theoretical Predictions to Test

From Part II of the thesis:

1. **Joy cluster**: (+valence, +rank, +integration, -self_model_salience)
2. **Suffering cluster**: (-valence, +integration, -rank, +self_model_salience)
3. **Fear vs Curiosity**: Both high CF, but opposite valence
4. **Hopelessness signature**: (-valence, -rank, +self_model_salience, +CF)

## Usage

```python
from study_llm_affect import LLMAgent, ScenarioRunner, AffectCalculator, SCENARIOS

# Initialize
agent = LLMAgent(model="claude-3-sonnet", api_key="...")
runner = ScenarioRunner(agent)
calculator = AffectCalculator()

# Run a scenario
results = runner.run_scenario(SCENARIOS["hopelessness"])
affect_trajectory = calculator.compute(results)

# Analyze
from study_llm_affect.analysis import test_clustering
clustering_results = test_clustering(affect_trajectory, expected_cluster="suffering")
```

## Falsification Criteria

The framework is falsified if:
1. Computed affect dimensions don't cluster by induced condition
2. Hopelessness doesn't show predicted (negative valence, low rank, high SM) signature
3. Flow doesn't show predicted (positive valence, high rank, low SM) signature
4. Fear and curiosity aren't distinguished by valence given matched CF levels

## References

- Thesis Part II, Section 4: "The Geometry of Affect"
- Thesis Part II, Algorithm 1: "Affect Measurement in World-Model Agents"
- CORRECTION.md in study_c_computational/ for common misinterpretations
