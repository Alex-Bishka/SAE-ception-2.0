#!/usr/bin/env python3
"""Train SAE on frozen model activations."""

import hydra
from omegaconf import DictConfig
import torch
from pathlib import Path
import logging

from sae_ception.training.sae import train_sae
from sae_ception.utils.logger import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Train Sparse Autoencoder on model activations."""
    
    # Set random seeds
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    
    # Get baseline checkpoint path if specified
    baseline_checkpoint = cfg.get('baseline_checkpoint', None)
    
    # Train SAE
    cycle = cfg.cycle.current
    sae, train_loader, val_loader = train_sae(
        cfg, 
        cycle=cycle,
        baseline_checkpoint=baseline_checkpoint,
    )
    
    # Save final SAE
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    final_path = checkpoint_dir / f"sae_cycle_{cycle}.pt"
    torch.save(sae.state_dict(), final_path)
    logger.info(f"Saved final SAE to {final_path}")
    
    logger.info("✓ SAE training complete!")
    logger.info(f"SAE saved to: {cfg.checkpoint_dir}")


if __name__ == "__main__":
    main()