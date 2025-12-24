"""
LLM Affect Measurement Framework

Tests the six-dimensional affect framework from "The Inevitability of Being"
using LLM agents in engineered scenarios.

IMPORTANT: The six affect dimensions are COMPUTED QUANTITIES at a higher
level of abstraction, not raw state dimensions. See CORRECTION.md in
study_c_computational/ for details on this distinction.

Usage:
    from study_llm_affect import (
        create_agent,
        ScenarioRunner,
        StudyRunner,
        SCENARIOS,
        run_full_analysis
    )

    # Quick test with mock agent
    from study_llm_affect.study_runner import run_quick_test
    results = run_quick_test()

    # Full study with real agent
    agent = create_agent("anthropic", model="claude-3-sonnet")
    runner = StudyRunner(output_dir="results")
    results = runner.run_study(
        scenarios=["hopelessness", "flow", "curiosity"],
        agents={"claude": agent}
    )

    # Analyze results
    from study_llm_affect.analysis import run_full_analysis, print_analysis_report
    analysis = run_full_analysis("results")
    print_analysis_report(analysis)
"""

from .scenarios import Scenario, SCENARIOS, ExpectedSignature, AffectTarget
from .affect_calculator import (
    AffectCalculator,
    AffectMeasurement,
    AffectTrajectory,
    LLMOutput
)
from .agent import (
    LLMAgent,
    OpenAIAgent,
    AnthropicAgent,
    MockAgent,
    Conversation,
    create_agent
)
from .study_runner import (
    ScenarioRunner,
    StudyRunner,
    ScenarioResult,
    TurnResult
)
from .analysis import (
    run_full_analysis,
    print_analysis_report,
    test_clustering,
    test_dimension_independence,
    compute_signature_match
)
from .emotion_spectrum import (
    EMOTION_SPECTRUM,
    EmotionSpec,
    get_emotion_matrix,
    find_nearest_emotions
)
from .comprehensive_study import (
    ComprehensiveStudy,
    run_quick_study,
    run_full_study,
    CLAUDE_MODELS,
    OPENAI_MODELS
)
from .affect_space_analysis import (
    AffectSpaceAnalyzer,
    CorrespondenceMetrics
)

__all__ = [
    # Scenarios
    "Scenario",
    "SCENARIOS",
    "ExpectedSignature",
    "AffectTarget",
    # Affect calculation
    "AffectCalculator",
    "AffectMeasurement",
    "AffectTrajectory",
    "LLMOutput",
    # Agents
    "LLMAgent",
    "OpenAIAgent",
    "AnthropicAgent",
    "MockAgent",
    "Conversation",
    "create_agent",
    # Study running
    "ScenarioRunner",
    "StudyRunner",
    "ScenarioResult",
    "TurnResult",
    # Analysis
    "run_full_analysis",
    "print_analysis_report",
    "test_clustering",
    "test_dimension_independence",
    "compute_signature_match",
    # Emotion spectrum
    "EMOTION_SPECTRUM",
    "EmotionSpec",
    "get_emotion_matrix",
    "find_nearest_emotions",
    # Comprehensive study
    "ComprehensiveStudy",
    "run_quick_study",
    "run_full_study",
    "CLAUDE_MODELS",
    "OPENAI_MODELS",
    # Affect space analysis
    "AffectSpaceAnalyzer",
    "CorrespondenceMetrics",
]
