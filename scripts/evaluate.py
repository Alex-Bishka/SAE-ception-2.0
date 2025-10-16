#!/usr/bin/env python3
"""Evaluate model and SAE."""

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Evaluate model."""
    print("TODO: Implement evaluation")
    print("Metrics:")
    print("- Task accuracy")
    print("- Monosemanticity (CSI, U)")
    print("- Clustering metrics (ARI, Silhouette, etc.)")
    print("- SAE reconstruction quality")


if __name__ == "__main__":
    main()
