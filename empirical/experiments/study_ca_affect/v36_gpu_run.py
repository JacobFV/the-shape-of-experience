"""V36 GPU Runner: Record egocentric trajectory on Lambda Labs.

Records key ecological transitions only (not all 30 cycles):
- Cycle 0: Initial state (random agents, no evolution)
- Cycle 3-4: Pre-drought normal foraging
- Cycle 5: First drought
- Cycle 6: First recovery
- Cycle 9-10: Second drought cycle + recovery
- Cycle 14-16: Mid-evolution + third drought
- Cycle 19-21: Late-mid evolution + fourth drought
- Cycle 24-26: Late evolution + fifth drought
- Cycle 28-29: Final state

This captures the full arc: naive→foraging→first_crisis→recovery→
repeated_crises→late_stage with enough temporal coverage for
VLM narration and 3D trajectory.

Total: ~15 cycles recorded × 5000 steps × focal agent data ≈ 75k datapoints.
Estimated runtime: ~20 minutes on A10/A100.
"""

import sys
import os

# Add experiment directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v36_trajectory import record_trajectory, generate_trajectory_vignettes, compute_affect_trajectory_3d


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--output', type=str, default='results/v36_trajectory')
    parser.add_argument('--all-cycles', action='store_true',
                       help='Record all 30 cycles (large output)')
    args = parser.parse_args()

    if args.all_cycles:
        record_cycles = list(range(30))
    else:
        # Key ecological transitions
        record_cycles = [
            0,          # Initial (random agents)
            3, 4,       # Pre-drought foraging
            5,          # First drought
            6, 7,       # First recovery
            10,         # Second drought
            11,         # Second recovery
            15,         # Third drought
            16,         # Third recovery
            20,         # Fourth drought
            21,         # Fourth recovery
            25,         # Fifth drought
            26,         # Fifth recovery
            28, 29,     # Late evolution
        ]

    print(f"Recording {len(record_cycles)} cycles for seed {args.seed}")
    print(f"Key transitions: drought cycles at 5, 10, 15, 20, 25")

    record_trajectory(
        seed=args.seed,
        output_dir=args.output,
        record_cycles=record_cycles,
        steps_per_record=25,  # env snapshot every 25 steps (200 per cycle)
    )

    # Post-processing
    print("\n--- Generating vignettes ---")
    generate_trajectory_vignettes(args.output, segment_length=500)

    print("\n--- Computing 3D trajectory ---")
    try:
        compute_affect_trajectory_3d(args.output)
    except ImportError:
        print("sklearn not available — skipping PCA. Run locally.")


if __name__ == '__main__':
    main()
