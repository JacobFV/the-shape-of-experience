# Inevitability Empirical

Empirical validation tools for the **six-dimensional affect framework** from "The Inevitability of Being" thesis.

## The Framework

The thesis proposes that experience has a specific geometric structure with six dimensions:

| Dimension | Symbol | Phenomenology |
|-----------|--------|---------------|
| **Valence** | Val | Good/bad, approach/avoid |
| **Arousal** | Ar | Activated/calm, energized/settled |
| **Integration** | Φ | Unified/fragmented, coherent/scattered |
| **Effective Rank** | r_eff | Open/narrow, flexible/rigid |
| **Counterfactual Weight** | CF | Present/elsewhere, here/wandering |
| **Self-Model Salience** | SM | Self-aware/absorbed, self-conscious/ego-dissolved |

## Key Predictions

1. **Dimension Independence**: The six dimensions are empirically distinguishable via factor analysis
2. **Valence = Viability Gradient**: Positive valence correlates with objective safety/low threat markers
3. **Integration = Experiential Unity**: Neural integration (Φ) correlates with reported experiential coherence
4. **Cultural Form Signatures**: Art, meditation, etc. produce distinct, predictable affect profiles
5. **Clinical Signatures**: Depression (low r_eff, high SM) vs anxiety (high CF, high Ar) are distinguishable

## Falsification Criteria

The framework is falsified if:
- A 2-factor model (valence-arousal) fits as well as 6-factor
- Valence is uncorrelated with physiological threat markers
- Neural integration and reported unity dissociate
- Different cultural forms produce indistinguishable signatures

## Installation

Using [uv](https://docs.astral.sh/uv/) (recommended):
```bash
cd empirical
uv sync
```

Or with pip:
```bash
pip install -e .
```

## Quick Start

```python
from inevitability_empirical.affect import AffectState, PREDICTED_SIGNATURES
from inevitability_empirical.measures import ValenceMeasure, ArousalMeasure
from inevitability_empirical.analysis import test_dimension_independence

# Define an affect state
state = AffectState(
    valence=0.7,
    arousal=0.5,
    integration=0.8,
    effective_rank=0.5,
    counterfactual_weight=0.1,
    self_model_salience=0.1
)

# Compare to predicted flow signature
flow_signature = PREDICTED_SIGNATURES["flow"]
distance = state.distance_to(flow_signature)
print(f"Distance from flow: {distance:.2f}")

# Run factor analysis on data
import numpy as np
data = np.random.randn(100, 6)  # Replace with real data
result = test_dimension_independence(data)
print(f"RMSEA: {result.fit_indices['rmsea']:.3f}")
```

## Study Protocols

See `protocols/study_protocols.md` for detailed study designs including:

1. **Study 1**: Dimension independence (factor analysis, N=500)
2. **Study 2**: Valence-viability correlation (ambulatory assessment, N=100)
3. **Study 3**: Integration-unity correspondence (EEG + phenomenology, N=60)
4. **Study 4**: Cultural form signatures (between-subjects exposure, N=200)
5. **Study 5**: Flow state validation (skill-challenge matching, N=80)
6. **Study 6**: Meditation signatures (within-subjects comparison, N=90)
7. **Study 7**: Clinical signatures (MDD vs GAD vs control, N=150)
8. **Study 8**: Real-time viability threat (cold pressor + breath hold, N=40)

## Directory Structure

```
empirical/
├── src/
│   └── inevitability_empirical/
│       ├── __init__.py
│       ├── affect.py         # Core affect state definitions
│       ├── measures.py       # Operational measures for each dimension
│       └── analysis.py       # Statistical analysis tools
├── protocols/
│   └── study_protocols.md    # Detailed study designs
├── notebooks/
│   └── 01_dimension_validation.ipynb
├── data/                     # Collected data (not tracked in git)
├── pyproject.toml
└── README.md
```

## Contributing

All studies will be pre-registered on OSF. Analysis code and de-identified data will be shared upon publication.

## License

MIT License

## Citation

If you use this framework, please cite:

```bibtex
@misc{valdez2024inevitability,
  title={The Inevitability of Being: A Six-Dimensional Framework for Affect},
  author={Valdez, Jacob},
  year={2024},
  url={https://github.com/JacobFV/the-inevitability-thesis}
}
```
