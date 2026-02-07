"""V12 Runner: Attention-based Lenia experiments.

Usage:
    python v12_run.py attention [N] [--channels C] [--grid G]
        V12 evolution with evolvable attention (Condition B)
        N = number of cycles (default 30)
        C = channels (default 16), G = grid size (default 128)

    python v12_run.py attention-fixed [N] [--channels C] [--grid G]
        V12 evolution with fixed-local attention (Condition A)

    python v12_run.py attention-pipeline [--channels C] [--grid G]
        Full V12 pipeline: evolve (Condition B) -> stress test

    python v12_run.py attention-fixed-pipeline [--channels C] [--grid G]
        Full V12 pipeline: evolve (Condition A) -> stress test

    python v12_run.py convolution-baseline [N] [--channels C] [--grid G]
        V11.4 FFT convolution baseline (Condition C) for comparison

    python v12_run.py compare [--channels C] [--grid G]
        Run all three conditions and compare results
"""

import sys
import os
import json
import time
import numpy as np


def parse_args():
    """Parse CLI arguments."""
    args = {
        'mode': sys.argv[1] if len(sys.argv) > 1 else 'attention',
        'n_cycles': 30,
        'C': 16,
        'N': 128,
    }

    # Parse positional n_cycles
    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        args['n_cycles'] = int(sys.argv[2])

    # Parse flags
    for i, arg in enumerate(sys.argv):
        if arg == '--channels' and i + 1 < len(sys.argv):
            args['C'] = int(sys.argv[i + 1])
        elif arg == '--grid' and i + 1 < len(sys.argv):
            args['N'] = int(sys.argv[i + 1])

    return args


def save_results(data, path):
    """Save results to JSON."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

    def serialize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return str(obj)

    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=serialize)
    print(f"Results saved to {path}")


if __name__ == '__main__':
    args = parse_args()
    mode = args['mode']
    C = args['C']
    N = args['N']
    n_cycles = args['n_cycles']

    if mode == 'attention':
        from v12_evolution import evolve_attention
        result = evolve_attention(
            n_cycles=n_cycles, C=C, N=N, fixed_window=False)
        save_results({
            'cycle_stats': result['cycle_stats'],
            'w_soft': result['w_soft'],
            'tau': result['tau'],
            'condition': result['condition'],
        }, f'results/v12_attention_C{C}_N{N}.json')

    elif mode == 'attention-fixed':
        from v12_evolution import evolve_attention
        result = evolve_attention(
            n_cycles=n_cycles, C=C, N=N, fixed_window=True)
        save_results({
            'cycle_stats': result['cycle_stats'],
            'w_soft': result['w_soft'],
            'tau': result['tau'],
            'condition': result['condition'],
        }, f'results/v12_attention_fixed_C{C}_N{N}.json')

    elif mode == 'attention-pipeline':
        from v12_evolution import full_pipeline_attention
        result = full_pipeline_attention(
            n_cycles=n_cycles, C=C, N=N, fixed_window=False)
        save_results({
            'cycle_stats': result['evolution']['cycle_stats'],
            'stress_test': result['stress_test'],
            'condition': 'B_evolvable',
        }, f'results/v12_pipeline_B_C{C}_N{N}.json')

    elif mode == 'attention-fixed-pipeline':
        from v12_evolution import full_pipeline_attention
        result = full_pipeline_attention(
            n_cycles=n_cycles, C=C, N=N, fixed_window=True)
        save_results({
            'cycle_stats': result['evolution']['cycle_stats'],
            'stress_test': result['stress_test'],
            'condition': 'A_fixed',
        }, f'results/v12_pipeline_A_C{C}_N{N}.json')

    elif mode == 'convolution-baseline':
        from v12_evolution import full_pipeline_convolution_baseline
        result = full_pipeline_convolution_baseline(
            n_cycles=n_cycles, C=C, N=N)
        save_results({
            'cycle_stats': result['evolution']['cycle_stats'],
            'stress_test': (result['stress_test'].get('comparison')
                            if result.get('stress_test') else None),
            'condition': 'C_convolution',
        }, f'results/v12_baseline_C_C{C}_N{N}.json')

    elif mode == 'compare':
        from v12_evolution import (
            full_pipeline_attention,
            full_pipeline_convolution_baseline,
        )

        print("\n" + "#" * 60)
        print("# V12: THREE-CONDITION COMPARISON")
        print("#" * 60)
        print(f"# C={C}, N={N}, {n_cycles} cycles each")
        print("#" * 60)

        all_results = {}

        # Condition A: Fixed-local attention
        print("\n>>> CONDITION A: Fixed-local attention <<<")
        result_a = full_pipeline_attention(
            n_cycles=n_cycles, C=C, N=N, fixed_window=True, seed=42)
        all_results['A_fixed'] = result_a

        # Condition B: Evolvable attention
        print("\n>>> CONDITION B: Evolvable attention <<<")
        result_b = full_pipeline_attention(
            n_cycles=n_cycles, C=C, N=N, fixed_window=False, seed=42)
        all_results['B_evolvable'] = result_b

        # Condition C: FFT convolution baseline
        print("\n>>> CONDITION C: FFT convolution baseline <<<")
        result_c = full_pipeline_convolution_baseline(
            n_cycles=n_cycles, C=C, N=N, seed=42)
        all_results['C_convolution'] = result_c

        # Summary
        print("\n" + "=" * 60)
        print("V12 THREE-CONDITION SUMMARY")
        print("=" * 60)

        for cond_name, result in all_results.items():
            evo = result.get('evolution', result)
            stress = result.get('stress_test', {})
            comparison = stress.get('comparison', {})

            last_stats = evo.get('cycle_stats', [{}])[-1] if evo.get('cycle_stats') else {}

            print(f"\n  {cond_name}:")
            print(f"    Final patterns: {last_stats.get('n_patterns', '?')}")
            print(f"    Phi (base):     {last_stats.get('mean_phi_base', 0):.4f}")
            print(f"    Phi (stress):   {last_stats.get('mean_phi_stress', 0):.4f}")
            print(f"    Robustness:     {last_stats.get('mean_robustness', 0):.3f}")
            if 'w_soft' in last_stats:
                print(f"    Window:         {last_stats.get('w_soft', 0):.1f}")
                print(f"    Temperature:    {last_stats.get('tau', 0):.2f}")
            phi_delta = comparison.get('evolved_phi_delta', None)
            if phi_delta is not None:
                print(f"    Stress Phi Δ:   {phi_delta:+.1%}")
                if comparison.get('biological_shift', False):
                    print(f"    *** BIOLOGICAL PATTERN ***")

        # Save combined results
        save_data = {}
        for cond_name, result in all_results.items():
            evo = result.get('evolution', result)
            stress = result.get('stress_test', {})
            save_data[cond_name] = {
                'cycle_stats': evo.get('cycle_stats', []),
                'stress_comparison': stress.get('comparison', {}),
            }

        save_results(save_data, f'results/v12_comparison_C{C}_N{N}.json')

    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python v12_run.py "
              "[attention|attention-fixed|attention-pipeline|"
              "attention-fixed-pipeline|convolution-baseline|compare]")
        print("  Options: --channels C  --grid G")
