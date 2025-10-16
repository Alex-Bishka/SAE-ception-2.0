#!/usr/bin/env python3
"""Main script for running SAE-ception training cycles."""

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Run SAE-ception training cycle."""
    print("=" * 60)
    print("SAE-ception Training")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    
    cycle = cfg.cycle.current
    print(f"\nCurrent cycle: {cycle}")
    print(f"Sharpening strategy: {cfg.sharpening.type}")
    
    print("\nTODO: Implement full training cycle")
    print("Steps:")
    print("1. Load/train base model")
    print("2. Train SAE on activations")
    print("3. Generate sharpened targets")
    print("4. Train next model with auxiliary loss")
    print("5. Evaluate")


if __name__ == "__main__":
    main()
