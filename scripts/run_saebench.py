#!/usr/bin/env python3
"""
Standalone SAEBench evaluation script.

Runs SAEBench metrics (first-letter absorption) on saved SAE checkpoints.
This is separate from train_cycle.py to allow flexible evaluation timing.

Usage:
    # Full evaluation (all latents, paper-accurate)
    python scripts/run_saebench.py checkpoint_dir=outputs/run_123/checkpoints cycle_to_eval=0

    # Quick smoke test (top 50 latents only)
    python scripts/run_saebench.py checkpoint_dir=outputs/run_123/checkpoints cycle_to_eval=0 n_candidates=50
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from pathlib import Path
import logging

from sae_ception.evaluation.saebench import evaluate_sae_saebench
from sae_ception.utils.data import extract_activations_all_tokens, create_dataloaders
from sae_ception.training.sae import load_sae  # Use existing load function
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_path: str, cfg: DictConfig, device: str):
    """Load model from checkpoint."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.hf_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.hf_path,
        num_labels=cfg.dataset.num_classes,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    
    return model, tokenizer


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Run SAEBench evaluation on saved SAE."""
    
    # Get evaluation-specific parameters
    cycle = cfg.get('cycle_to_eval', cfg.cycle.current)
    n_candidates = cfg.get('n_candidates', None)  # None = all latents
    checkpoint_dir = Path(cfg.checkpoint_dir)
    
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    
    logger.info("=" * 60)
    logger.info("SAEBench Evaluation")
    logger.info("=" * 60)
    logger.info(f"Checkpoint dir: {checkpoint_dir}")
    logger.info(f"Cycle: {cycle}")
    logger.info(f"N candidates: {n_candidates or 'ALL (paper-accurate)'}")
    logger.info(f"Model: {cfg.model.hf_path}")
    logger.info(f"Dataset: {cfg.dataset.name}")
    logger.info(f"Device: {device}")
    
    # Check for SAE checkpoint
    sae_path = checkpoint_dir / f"sae_cycle_{cycle}_best.pt"
    if not sae_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {sae_path}")
    
    # Load SAE using existing function
    logger.info(f"Loading SAE from {sae_path}")
    sae = load_sae(cfg, cycle=cycle, checkpoint_path=str(sae_path))
    sae.to(device)
    sae.eval()
    logger.info(f"Loaded SAE: type={cfg.sae.sae_type}")
    
    # Check for cached token-level activations
    token_metadata_path = checkpoint_dir / f"token_metadata_cycle_{cycle}.pt"
    activations_path = checkpoint_dir / f"val_activations_cycle_{cycle}.pt"
    
    if token_metadata_path.exists() and activations_path.exists():
        logger.info("Loading cached token-level activations...")
        token_metadata = torch.load(token_metadata_path)
        activations_data = torch.load(activations_path)
        
        val_acts = activations_data['val_acts']
        val_token_ids = token_metadata['val_token_ids']
        
        # Load tokenizer for decoding
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.hf_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
    else:
        # Need to extract activations from model
        logger.info("No cached activations found. Extracting from model...")
        
        model_path = checkpoint_dir / f"model_cycle_{cycle}_best.pt"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_path}\n"
                "Either provide cached activations or model checkpoint."
            )
        
        # Load model and tokenizer
        model, tokenizer = load_model_from_checkpoint(str(model_path), cfg, device)
        
        # Create dataloader
        dataloaders = create_dataloaders(cfg, tokenizer)
        val_loader = dataloaders['val']
        
        # Extract activations
        logger.info(f"Extracting activations from layer: {cfg.model.target_layer}")
        val_acts, val_token_ids, val_labels = extract_activations_all_tokens(
            model=model,
            dataloader=val_loader,
            layer_name=str(cfg.model.target_layer),
            device=device,
        )
        
        logger.info(f"Extracted {val_acts.shape[0]} token activations")
    
    logger.info(f"Activations shape: {val_acts.shape}")
    logger.info(f"Token IDs shape: {val_token_ids.shape}")
    
    # Run SAEBench evaluation
    results = evaluate_sae_saebench(
        sae=sae,
        activations=val_acts,
        token_ids=val_token_ids,
        tokenizer=tokenizer,
        device=device,
        n_latent_candidates=n_candidates,
    )
    
    # Log final summary
    logger.info("\n" + "=" * 60)
    logger.info("SAEBench Evaluation Complete")
    logger.info("=" * 60)
    logger.info(f"  Absorption:            {results.get('absorption', 0):.4f}")
    logger.info(f"  Absorption Complement: {results.get('absorption_complement', 0):.4f}")
    logger.info(f"  Letters Evaluated:     {results.get('n_letters_evaluated', 0)}")
    logger.info(f"  Test Samples:          {results.get('n_test_samples', 0)}")
    
    # Save results
    output_path = checkpoint_dir / f"saebench_cycle_{cycle}.pt"
    torch.save(results, output_path)
    logger.info(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()