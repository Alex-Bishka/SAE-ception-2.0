#!/usr/bin/env python3
"""Train model with auxiliary loss from sharpened SAE features."""

import hydra
from omegaconf import DictConfig
import torch
from pathlib import Path
import logging

from sae_ception.training.auxiliary import train_with_auxiliary_loss
from sae_ception.utils.logger import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Train with auxiliary loss."""
    
    # Set random seeds
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    
    # Get checkpoint paths if specified
    baseline_checkpoint = cfg.get('baseline_checkpoint', None)
    sae_checkpoint = cfg.get('sae_checkpoint', None)
    
    # Determine cycle
    prev_cycle = cfg.cycle.current
    next_cycle = prev_cycle + 1
    
    # Train with auxiliary loss
    model = train_with_auxiliary_loss(
        cfg,
        prev_cycle=prev_cycle,
        baseline_checkpoint=baseline_checkpoint,
        sae_checkpoint=sae_checkpoint,
    )
    
    # Save final model
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    final_path = checkpoint_dir / f"model_cycle_{next_cycle}.pt"
    torch.save(model.state_dict(), final_path)
    logger.info(f"Saved final model to {final_path}")
    
    logger.info(f"✓ Auxiliary training complete!")
    logger.info(f"Model for cycle {next_cycle} saved to: {cfg.checkpoint_dir}")


if __name__ == "__main__":
    main()