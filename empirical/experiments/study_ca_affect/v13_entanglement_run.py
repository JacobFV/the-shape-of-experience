#!/usr/bin/env python3
"""Runner for Experiment 11: Entanglement Analysis.

Usage:
    python v13_entanglement_run.py full
"""

import json
import sys
from pathlib import Path

RESULTS_DIR = str(Path(__file__).parent / 'results')


def run_full():
    """Run full entanglement analysis."""
    from v13_entanglement import run_entanglement_analysis

    results = run_entanglement_analysis(RESULTS_DIR)

    # Save results
    out_dir = Path(RESULTS_DIR) / 'entanglement_analysis'
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / 'entanglement_results.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT 11 SUMMARY: Entanglement Analysis")
    print("=" * 60)

    # Top findings
    strongest = results['strongest_evolutionary_correlations']
    sig_pairs = [p for p in strongest if p['p'] is not None and p['p'] < 0.05]
    print(f"\nSignificant evolutionary correlations: {len(sig_pairs)}/{len(strongest)}")
    for p in sig_pairs[:5]:
        print(f"  {p['measure_1']:15s} ↔ {p['measure_2']:15s}  r={p['r']:+.3f}  p={p['p']:.4f}")

    # Co-emergence
    co = results['co_emergence_test']
    print(f"\nCo-emergence (C_wm, A, I_img): mean r = {co.get('mean_r', 'N/A')}")
    print(f"  Prediction 11.1 confirmed: {co.get('confirmed', False)}")

    # Language lag
    lang = results['language_lag_test']
    print(f"\nLanguage lag: ρ_topo onset = {lang.get('rho_topo_onset', 'never')}")
    print(f"  Prediction 11.2 confirmed: {lang.get('confirmed', False)}")

    # SM-Φ
    sm = results['self_model_phi_test']
    print(f"\nSM × Φ jump: {len(sm.get('pairs', []))} pairs tested")
    print(f"  Prediction 11.4 confirmed: {sm.get('confirmed', False)}")

    # Clusters
    clusters = results['clusters']
    print(f"\nMeasure clusters (|r| > 0.5): {len(clusters)}")
    for i, c in enumerate(clusters):
        print(f"  {i + 1}: {c}")

    # Entanglement trajectory
    print("\nEntanglement trajectory:")
    for e in results['entanglement_trajectory']:
        val = f"{e['entanglement']:.3f}" if e['entanglement'] else "N/A"
        print(f"  Cycle {e['cycle']:2d}: {val}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python v13_entanglement_run.py full")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == 'full':
        run_full()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
