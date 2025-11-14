import torch
from pathlib import Path

# Find your results file
path = "/home1/abishka@hmc.edu/scratch/saeception/run_19340/multirun/2025-11-13/23-33-39/1/checkpoints/results_cycle_0.pt"
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