#!/usr/bin/env python3
"""Train SAE on frozen model activations."""

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Train SAE."""
    print("TODO: Implement SAE training")
    print(f"Target layer: {cfg.model.target_layer}")
    print(f"Expansion factor: {cfg.sae.expansion_factor}")


if __name__ == "__main__":
    main()
