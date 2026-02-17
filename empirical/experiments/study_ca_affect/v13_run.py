"""V13 Runner: Content-based coupling Lenia experiments.

Usage:
    python v13_run.py evolve [N] [--channels C] [--grid G] [--radius R]
        V13 evolution with content-based coupling
        N = number of cycles (default 30)

    python v13_run.py pipeline [--channels C] [--grid G] [--radius R]
        Full V13 pipeline: evolve -> stress test

    python v13_run.py smoke
        Quick smoke test (2 cycles, small grid)
"""

import sys
import os
import json
import time
import numpy as np


def parse_args():
    """Parse CLI arguments."""
    args = {
        'mode': sys.argv[1] if len(sys.argv) > 1 else 'smoke',
        'n_cycles': 30,
        'C': 16,
        'N': 128,
        'R': 20,
        'seed': 42,
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
        elif arg == '--radius' and i + 1 < len(sys.argv):
            args['R'] = int(sys.argv[i + 1])
        elif arg == '--seed' and i + 1 < len(sys.argv):
            args['seed'] = int(sys.argv[i + 1])

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
        if hasattr(obj, 'item'):
            return obj.item()
        return str(obj)

    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=serialize)
    print(f"Results saved to {path}")


if __name__ == '__main__':
    args = parse_args()
    mode = args['mode']
    C = args['C']
    N = args['N']
    R = args['R']
    seed = args['seed']
    n_cycles = args['n_cycles']

    if mode == 'smoke':
        print("V13 Smoke Test (2 cycles, C=8, N=64, R=10)")
        print("=" * 50)
        from v13_evolution import evolve_v13
        result = evolve_v13(n_cycles=2, steps_per_cycle=1000,
                            C=8, N=64, similarity_radius=10, seed=seed)
        save_results({
            'cycle_stats': result['cycle_stats'],
            'tau': result['tau'],
            'gate_beta': result['gate_beta'],
        }, 'results/v13_smoke.json')

    elif mode == 'evolve':
        from v13_evolution import evolve_v13
        result = evolve_v13(n_cycles=n_cycles, C=C, N=N,
                            similarity_radius=R, seed=seed)
        save_results({
            'cycle_stats': result['cycle_stats'],
            'tau': result['tau'],
            'gate_beta': result['gate_beta'],
        }, f'results/v13_evolve_C{C}_N{N}_R{R}.json')

    elif mode == 'pipeline':
        from v13_evolution import full_pipeline_v13
        result = full_pipeline_v13(n_cycles=n_cycles, C=C, N=N,
                                   similarity_radius=R, seed=seed)
        save_results({
            'cycle_stats': result['evolution']['cycle_stats'],
            'stress_test': result['stress_test'],
            'tau': result['evolution']['tau'],
            'gate_beta': result['evolution']['gate_beta'],
        }, f'results/v13_pipeline_C{C}_N{N}_R{R}.json')

    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python v13_run.py [smoke|evolve|pipeline]")
        print("  Options: --channels C  --grid G  --radius R  --seed S")
