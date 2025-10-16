#!/usr/bin/env python3
"""Train baseline model (Cycle 0)."""

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Train baseline model."""
    print("TODO: Implement baseline training")
    print(f"Dataset: {cfg.dataset.name}")
    print(f"Model: {cfg.model.name}")


if __name__ == "__main__":
    main()
