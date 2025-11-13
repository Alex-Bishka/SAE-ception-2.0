#!/usr/bin/env python3
"""
Complete SAE-ception training cycle.

This script orchestrates the full SAE-ception process:
1. Train baseline (if cycle 0) OR load previous model
2. Train SAE on frozen model activations
3. Generate sharpened targets
4. Train next model with auxiliary loss
5. Evaluate and save results

Usage:
    # Run cycle 0 (baseline + first SAE + first aux training)
    python scripts/train_cycle.py cycle.current=0

    # Run cycle 1 (uses cycle 0 model, trains new SAE, creates cycle 1 model)
    python scripts/train_cycle.py cycle.current=1
    
    # Run multiple cycles in sequence
    python scripts/train_cycle.py cycle.current=0 cycle.max_cycles=3
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from pathlib import Path
import logging

from sae_ception.training.baseline import train_baseline_model, load_baseline_model
from sae_ception.training.sae import train_sae, load_sae
from sae_ception.training.auxiliary import train_with_auxiliary_loss
from sae_ception.evaluation import (
    evaluate_classification_accuracy,
    evaluate_model_and_sae,
    train_linear_probe,
)
from sae_ception.utils.logger import setup_logging
from sae_ception.utils.data import (
    create_dataloaders,
    extract_activations_from_model,
    create_activation_dataloader,
)
from transformers import AutoTokenizer

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def evaluate_cycle(
    cfg: DictConfig,
    model,
    sae,
    cycle: int,
    device: str = 'cuda',
) -> dict:
    """
    Comprehensive evaluation of a cycle.
    
    Args:
        cfg: Configuration
        model: Trained model
        sae: Trained SAE
        cycle: Cycle number
        device: Device
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("=" * 60)
    logger.info(f"Evaluating Cycle {cycle}")
    logger.info("=" * 60)
    
    # Load tokenizer and data
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.hf_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataloaders = create_dataloaders(cfg, tokenizer)
    val_loader = dataloaders['val']
    train_loader = dataloaders['train']
    
    # Extract activations
    logger.info("Extracting activations for evaluation...")
    train_acts, train_labels = extract_activations_from_model(
        model, train_loader, str(cfg.model.target_layer), device, 'last'
    )
    val_acts, val_labels = extract_activations_from_model(
        model, val_loader, str(cfg.model.target_layer), device, 'last'
    )
    
    # Get SAE sparse codes
    logger.info("Computing SAE sparse codes...")
    sae.eval()
    with torch.no_grad():
        sparse_train = sae.encode(train_acts.to(device)).cpu()
        sparse_val = sae.encode(val_acts.to(device)).cpu()
    
    # Full evaluation
    results = evaluate_model_and_sae(
        model=model,
        sae=sae,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        train_activations=train_acts,
        train_labels=train_labels,
        val_activations=val_acts,
        val_labels=val_labels,
        sparse_codes_train=sparse_train,
        sparse_codes_val=sparse_val,
        device=device,
    )
    
    return results


def run_single_cycle(cfg: DictConfig, cycle: int) -> tuple:
    """
    Run a single SAE-ception cycle.
    
    Args:
        cfg: Configuration
        cycle: Cycle number
        
    Returns:
        Tuple of (model, sae, results)
    """
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    
    # Step 1: Get base model for this cycle
    if cycle == 0:
        logger.info("=" * 60)
        logger.info("STEP 1: Training Baseline Model (Cycle 0)")
        logger.info("=" * 60)
        model = train_baseline_model(cfg)
        
        # Save baseline
        checkpoint_dir = Path(cfg.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_dir / "model_cycle_0.pt")
        
    else:
        logger.info("=" * 60)
        logger.info(f"STEP 1: Loading Model from Cycle {cycle - 1}")
        logger.info("=" * 60)
        
        # Find previous model
        outputs_dir = Path('outputs')
        checkpoint_pattern = f"**/checkpoints/model_cycle_{cycle - 1}_best.pt"
        checkpoints = sorted(outputs_dir.glob(checkpoint_pattern), 
                           key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not checkpoints:
            raise FileNotFoundError(
                f"No model found for cycle {cycle - 1}. "
                f"Run cycle {cycle - 1} first."
            )
        
        model = load_baseline_model(cfg, checkpoint_path=str(checkpoints[0]))
    
    # Step 2: Train SAE
    logger.info("=" * 60)
    logger.info(f"STEP 2: Training SAE (Cycle {cycle})")
    logger.info("=" * 60)
    
    sae, _, _ = train_sae(cfg, model=model, cycle=cycle)
    
    # Save SAE
    checkpoint_dir = Path(cfg.checkpoint_dir)
    torch.save(sae.state_dict(), checkpoint_dir / f"sae_cycle_{cycle}.pt")
    
    # Step 3: Evaluate current cycle
    results = evaluate_cycle(cfg, model, sae, cycle, device)
    
    # Save results
    results_path = checkpoint_dir / f"results_cycle_{cycle}.pt"
    torch.save(results, results_path)
    logger.info(f"Saved results to {results_path}")
    
    # Log key metrics
    logger.info("=" * 60)
    logger.info(f"Cycle {cycle} Summary")
    logger.info("=" * 60)
    logger.info(f"Task Accuracy: {results['task_accuracy']:.4f}")
    logger.info(f"Probe Accuracy: {results['probe_test_accuracy']:.4f}")
    logger.info(f"Base Model CSI: {results['base_csi']:.4f}")
    logger.info(f"SAE Features U: {results['sae_features_u']:.4f}")
    logger.info(f"Clustering ARI: {results['clustering_ari']:.4f}")
    logger.info(f"Sparsity L0: {results['sparsity_l0_mean']:.2f}")
    
    # Step 4: Train next model if not at max cycles
    next_model = None
    if cycle < cfg.cycle.max_cycles:
        logger.info("=" * 60)
        logger.info(f"STEP 3: Training with Auxiliary Loss (Cycle {cycle} -> {cycle + 1})")
        logger.info("=" * 60)
        
        next_model = train_with_auxiliary_loss(cfg, prev_cycle=cycle)
        
        # Save next model
        torch.save(next_model.state_dict(), 
                  checkpoint_dir / f"model_cycle_{cycle + 1}.pt")
    else:
        logger.info(f"Reached max_cycles ({cfg.cycle.max_cycles}). Stopping.")
    
    return model, sae, results, next_model


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Run SAE-ception training cycle(s)."""
    
    logger.info("=" * 80)
    logger.info("SAE-ception Training")
    logger.info("=" * 80)
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Set random seeds
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    
    current_cycle = cfg.cycle.current
    max_cycles = cfg.cycle.max_cycles
    
    logger.info(f"Starting from cycle {current_cycle}")
    logger.info(f"Will run up to cycle {max_cycles}")
    logger.info(f"Sharpening strategy: {cfg.sharpening.type}")
    logger.info(f"Auxiliary loss weight: {cfg.cycle.aux_loss_weight}")
    
    # Track all results
    all_results = {}
    
    # Run cycles
    for cycle in range(current_cycle, max_cycles + 1):
        logger.info("\n" + "=" * 80)
        logger.info(f"RUNNING CYCLE {cycle}")
        logger.info("=" * 80 + "\n")
        
        model, sae, results, next_model = run_single_cycle(cfg, cycle)
        all_results[cycle] = results
        
        # If we created a next model and haven't hit max, continue
        if next_model is not None and cycle < max_cycles:
            logger.info(f"✓ Cycle {cycle} complete, continuing to cycle {cycle + 1}...")
        else:
            logger.info(f"✓ Cycle {cycle} complete")
            break
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("SAE-ception Training Complete!")
    logger.info("=" * 80)
    
    logger.info("\nResults Summary:")
    logger.info("-" * 80)
    logger.info(f"{'Cycle':<8} {'Accuracy':<12} {'Probe Acc':<12} {'ARI':<10} {'U':<10} {'L0':<10}")
    logger.info("-" * 80)
    
    for cycle, results in all_results.items():
        logger.info(
            f"{cycle:<8} "
            f"{results['task_accuracy']:<12.4f} "
            f"{results['probe_test_accuracy']:<12.4f} "
            f"{results['clustering_ari']:<10.4f} "
            f"{results['sae_features_u']:<10.4f} "
            f"{results['sparsity_l0_mean']:<10.2f}"
        )
    
    logger.info("-" * 80)
    logger.info(f"\nAll checkpoints saved to: {cfg.checkpoint_dir}")
    logger.info(f"Results saved to: {cfg.checkpoint_dir}/results_cycle_*.pt")


if __name__ == "__main__":
    main()
