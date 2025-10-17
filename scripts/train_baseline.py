#!/usr/bin/env python3
"""Train baseline model (Cycle 0)."""

import hydra
from omegaconf import DictConfig
import torch
from pathlib import Path
import logging

from sae_ception.training.baseline import train_baseline_model
from sae_ception.utils.logger import setup_logging

# Setup logging before Hydra runs
setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Train baseline classification model."""
    
    # Set random seeds
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    
    # Train baseline model
    model = train_baseline_model(cfg)
    
    # Save final model
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    final_path = checkpoint_dir / "model_cycle_0.pt"
    torch.save(model.state_dict(), final_path)
    logger.info(f"Saved final model to {final_path}")
    
    logger.info("✓ Baseline training complete!")
    logger.info(f"Model saved to: {cfg.checkpoint_dir}")


if __name__ == "__main__":
    main()
