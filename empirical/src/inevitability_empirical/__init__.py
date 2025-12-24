"""
Inevitability Empirical: Tools for testing the six-dimensional affect framework.

This package provides:
- Operationalizations of the six affect dimensions
- Experience sampling protocols
- Analysis tools for validating theoretical predictions
- Simulation environments for edge case testing
"""

__version__ = "0.1.0"

from .affect import AffectState, AffectDimension
from .measures import (
    ValenceMeasure,
    ArousalMeasure,
    IntegrationMeasure,
    EffectiveRankMeasure,
    CounterfactualWeightMeasure,
    SelfModelSalienceMeasure,
)
