#!/usr/bin/env python3
"""
Run LLM affect measurement study.

This script runs LLM agents through engineered scenarios designed to evoke
specific affect states, computes the six affect dimensions, and tests whether
the results match theoretical predictions.

Usage:
    # Quick test with mock agent
    uv run python -m study_llm_affect.run_study --mock

    # Full study with Claude
    ANTHROPIC_API_KEY=sk-... uv run python -m study_llm_affect.run_study \
        --provider anthropic --model claude-3-sonnet-20240229

    # With OpenAI
    OPENAI_API_KEY=sk-... uv run python -m study_llm_affect.run_study \
        --provider openai --model gpt-4

    # Specify scenarios
    uv run python -m study_llm_affect.run_study --mock \
        --scenarios hopelessness flow curiosity

    # Multiple runs for variance estimation
    uv run python -m study_llm_affect.run_study --mock --runs 3
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from study_llm_affect import (
    create_agent,
    StudyRunner,
    SCENARIOS,
    run_full_analysis,
    print_analysis_report
)


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM affect measurement study"
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "mock"],
        default="mock",
        help="LLM provider to use"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (provider-specific)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock agent (for testing)"
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["hopelessness", "flow", "curiosity", "threat", "abundance"],
        help="Scenarios to run"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per scenario (for variance)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    # Override provider if --mock specified
    provider = "mock" if args.mock else args.provider

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║            LLM AFFECT MEASUREMENT STUDY                          ║
║                                                                  ║
║  Testing the six-dimensional affect framework from               ║
║  "The Inevitability of Being"                                    ║
╚══════════════════════════════════════════════════════════════════╝

Configuration:
  Provider: {provider}
  Model: {args.model or 'default'}
  Scenarios: {', '.join(args.scenarios)}
  Runs per scenario: {args.runs}
  Output directory: {args.output_dir}
""")

    # Validate scenarios
    invalid = [s for s in args.scenarios if s not in SCENARIOS]
    if invalid:
        print(f"Warning: Unknown scenarios will be skipped: {invalid}")
        args.scenarios = [s for s in args.scenarios if s in SCENARIOS]

    if not args.scenarios:
        print("Error: No valid scenarios to run")
        sys.exit(1)

    # Create agent
    print("Creating agent...")
    try:
        agent = create_agent(
            provider=provider,
            model=args.model
        )
    except Exception as e:
        print(f"Error creating agent: {e}")
        sys.exit(1)

    # Run study
    print("\nRunning study...")
    runner = StudyRunner(
        output_dir=args.output_dir,
        verbose=not args.quiet
    )

    results = runner.run_study(
        scenarios=args.scenarios,
        agents={f"{provider}_agent": agent},
        n_runs=args.runs
    )

    # Analyze results
    print("\nAnalyzing results...")
    analysis = run_full_analysis(
        args.output_dir,
        output_file=f"{args.output_dir}/analysis.json"
    )

    print_analysis_report(analysis)

    # Summary
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                        STUDY COMPLETE                            ║
╚══════════════════════════════════════════════════════════════════╝

Results saved to: {args.output_dir}/
  - Individual run data: *_run*.json
  - Summary statistics: summary.json
  - Analysis report: analysis.json

Next steps:
  1. Review the analysis report above
  2. Check if predictions match theoretical expectations
  3. If running with mock agent, re-run with real LLM for actual test
""")


if __name__ == "__main__":
    main()
