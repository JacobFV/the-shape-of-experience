#!/usr/bin/env python3
"""Runner for Experiment 12: Identity Thesis Capstone.

Usage:
    python v13_capstone_run.py full
"""

import json
import sys
from pathlib import Path

RESULTS_DIR = str(Path(__file__).parent / 'results')


def run_full():
    """Run capstone analysis."""
    from v13_capstone import run_capstone

    results = run_capstone(RESULTS_DIR)

    # Save
    out_dir = Path(RESULTS_DIR) / 'capstone_analysis'
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / 'capstone_results.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python v13_capstone_run.py full")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == 'full':
        run_full()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
