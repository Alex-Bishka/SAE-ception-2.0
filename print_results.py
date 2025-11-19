import torch
import argparse
from pathlib import Path

# Find your results file
def main(checkpoint_num: int | None = None, cycle: int | None = None):
    main_path = "/home1/abishka@hmc.edu/scratch/saeception/run_19436/multirun/2025-11-17/23-27-43"
    
    if checkpoint_num is None:
        checkpoint_num = 0
    if cycle is None:
        cycle = 0

    path = f"{main_path}/{checkpoint_num}/checkpoints/results_cycle_{cycle}.pt"
    print(path)
    results_path = Path(path)
    results = torch.load(results_path)

    # Check what metrics exist
    print("Available metrics:")
    for key in sorted(results.keys()):
        print(f"  {key}: {results[key]}")

    # Check for new metrics (will be missing)
    new_metrics = [
        'sae_dead_features_pct',
        'sae_feature_usage_entropy', 
        'k_sparse_probe_k1',
        'k_sparse_probe_k25',
    ]

    print("\nNew metrics present?")
    for metric in new_metrics:
        print(f"  {metric}: {'✓' if metric in results else '✗ MISSING'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-num",
        type=int,
        default=None,
        help="Optional checkpoint number (default: script's internal default)",
    )
    parser.add_argument(
        "--cycle",
        type=int,
        default=None,
        help="Optional cycle index (default: script's internal default)",
    )

    args = parser.parse_args()

    main(
        checkpoint_num=args.checkpoint_num,
        cycle=args.cycle,
    )