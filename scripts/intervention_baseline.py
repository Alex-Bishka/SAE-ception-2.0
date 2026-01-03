#!/usr/bin/env python3
"""
Intervention Baseline: Test perplexity degradation from activation replacement.

This script measures the "information bottleneck" cost of replacing model
activations with SAE-sharpened reconstructions at inference time.

Usage:
    # Quick test (100 samples)
    python scripts/intervention_baseline.py --quick
    
    # Full sweep of k values
    python scripts/intervention_baseline.py --k_values 5,10,25,50,100
    
    # Custom SAE checkpoint
    python scripts/intervention_baseline.py --sae_checkpoint path/to/sae.pt
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from typing import List, Optional
import json

from sae_ception.models.sae import create_sae
from sae_ception.utils.hooks import create_sae_intervention, ActivationIntervention
from sae_ception.utils.data import create_causal_lm_dataloader
from sae_ception.evaluation.lm_metrics import (
    evaluate_perplexity,
    evaluate_perplexity_with_intervention,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_or_train_sae(
    model: torch.nn.Module,
    tokenizer,
    layer_idx: int,
    sae_checkpoint: Optional[str],
    hidden_size: int,
    expansion_factor: int = 8,
    k: int = 50,
    device: str = 'cuda',
) -> torch.nn.Module:
    """Load SAE from checkpoint or train a fresh one."""
    
    sae_hidden = hidden_size * expansion_factor
    
    if sae_checkpoint and Path(sae_checkpoint).exists():
        logger.info(f"Loading SAE from {sae_checkpoint}")
        sae = create_sae(
            input_dim=hidden_size,
            hidden_dim=sae_hidden,
            sae_type='topk',
            k=k,
        )
        checkpoint = torch.load(sae_checkpoint, map_location=device)
        sae.load_state_dict(checkpoint['model_state_dict'])
        sae.to(device)
        return sae
    
    # Train a fresh SAE
    logger.info("No SAE checkpoint provided. Training fresh SAE...")
    logger.info("(For proper experiments, you should train SAE separately)")
    
    from sae_ception.utils.data import extract_activations_all_tokens
    
    # Create small dataloader for SAE training
    train_loader = create_causal_lm_dataloader(
        tokenizer=tokenizer,
        dataset_name='wikitext',
        split='train',
        batch_size=8,
        max_length=512,
        max_samples=1000,  # Small for quick training
    )
    
    # Extract activations
    logger.info(f"Extracting activations from layer {layer_idx}...")
    acts, token_ids, _ = extract_activations_all_tokens(
        model=model,
        dataloader=train_loader,
        layer_name=str(layer_idx),
        device=device,
    )
    
    logger.info(f"Extracted {acts.shape[0]} token activations")
    
    # Create and train SAE
    sae = create_sae(
        input_dim=hidden_size,
        hidden_dim=sae_hidden,
        sae_type='topk',
        k=k,
    )
    sae.to(device)
    
    # Quick training loop
    from sae_ception.utils.data import create_activation_dataloader
    from torch.optim import Adam
    
    # For SAE training, we don't need labels - just activations
    # Use dummy labels (zeros) since create_activation_dataloader requires them
    dummy_labels = torch.zeros(len(acts), dtype=torch.long)
    
    act_loader = create_activation_dataloader(
        activations=acts,
        labels=dummy_labels,  # Dummy labels for compatibility
        batch_size=256,
        shuffle=True,
    )
    
    optimizer = Adam(sae.parameters(), lr=1e-4)
    
    logger.info("Training SAE (5 epochs)...")
    sae.train()
    for epoch in range(5):
        total_loss = 0
        for batch in act_loader:
            x = batch['activations'].to(device)
            recon, sparse, info = sae(x)
            loss_dict = sae.loss(x, recon, sparse, info)
            loss = loss_dict['total_loss']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if hasattr(sae, 'normalize_decoder_'):
                sae.normalize_decoder_()
            
            total_loss += loss.item()
        
        logger.info(f"  Epoch {epoch+1}: loss={total_loss/len(act_loader):.4f}")
    
    sae.eval()
    return sae


def run_intervention_baseline(
    model_name: str = 'EleutherAI/pythia-70m',
    layer_idx: int = 3,
    k_values: List[int] = [5, 10, 25, 50, 100],
    sae_checkpoint: Optional[str] = None,
    sae_k_train: int = 50,
    max_eval_samples: Optional[int] = None,
    device: str = 'cuda',
    output_path: Optional[str] = None,
):
    """
    Run the intervention baseline experiment.
    
    Tests perplexity degradation at different k_sharp values.
    """
    logger.info("=" * 60)
    logger.info("Intervention Baseline Experiment")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Layer: {layer_idx}")
    logger.info(f"K values to test: {k_values}")
    
    # Load model and tokenizer
    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    hidden_size = model.config.hidden_size
    logger.info(f"Hidden size: {hidden_size}")
    
    # Load or train SAE
    sae = load_or_train_sae(
        model=model,
        tokenizer=tokenizer,
        layer_idx=layer_idx,
        sae_checkpoint=sae_checkpoint,
        hidden_size=hidden_size,
        k=sae_k_train,
        device=device,
    )
    
    # Create evaluation dataloader
    logger.info("Loading evaluation data...")
    eval_loader = create_causal_lm_dataloader(
        tokenizer=tokenizer,
        dataset_name='wikitext',
        split='test',
        batch_size=4,
        max_length=1024,
        max_samples=max_eval_samples,
    )
    
    # Baseline: No intervention
    logger.info("\n" + "-" * 40)
    logger.info("Baseline (no intervention)...")
    baseline_results = evaluate_perplexity(model, eval_loader, device)
    logger.info(f"Baseline Perplexity: {baseline_results['perplexity']:.2f}")
    
    # Full SAE reconstruction (no sharpening)
    logger.info("\n" + "-" * 40)
    logger.info(f"Full SAE reconstruction (k={sae_k_train})...")
    
    full_recon_fn = create_sae_intervention(sae, k_sharp=sae_k_train, device=device)
    full_recon_results = evaluate_perplexity_with_intervention(
        model, eval_loader, full_recon_fn, str(layer_idx), device
    )
    logger.info(f"Full Reconstruction Perplexity: {full_recon_results['perplexity']:.2f}")
    logger.info(f"Degradation: {full_recon_results['perplexity'] - baseline_results['perplexity']:.2f}")
    
    # Sweep k_sharp values
    results = {
        'model': model_name,
        'layer': layer_idx,
        'sae_k_train': sae_k_train,
        'baseline_perplexity': baseline_results['perplexity'],
        'full_recon_perplexity': full_recon_results['perplexity'],
        'k_sharp_results': {},
    }
    
    logger.info("\n" + "-" * 40)
    logger.info("Sweeping k_sharp values...")
    
    for k_sharp in k_values:
        logger.info(f"\nk_sharp = {k_sharp}")
        
        intervention_fn = create_sae_intervention(sae, k_sharp=k_sharp, device=device)
        k_results = evaluate_perplexity_with_intervention(
            model, eval_loader, intervention_fn, str(layer_idx), device
        )
        
        ppl = k_results['perplexity']
        degradation = ppl - baseline_results['perplexity']
        relative_degradation = (degradation / baseline_results['perplexity']) * 100
        
        logger.info(f"  Perplexity: {ppl:.2f}")
        logger.info(f"  Absolute degradation: +{degradation:.2f}")
        logger.info(f"  Relative degradation: +{relative_degradation:.1f}%")
        
        results['k_sharp_results'][k_sharp] = {
            'perplexity': ppl,
            'degradation': degradation,
            'relative_degradation_pct': relative_degradation,
        }
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'k_sharp':<10} {'Perplexity':<15} {'Degradation':<15} {'Relative':<10}")
    logger.info("-" * 50)
    logger.info(f"{'baseline':<10} {baseline_results['perplexity']:<15.2f} {'--':<15} {'--':<10}")
    logger.info(f"{'full':<10} {full_recon_results['perplexity']:<15.2f} "
                f"{full_recon_results['perplexity'] - baseline_results['perplexity']:<15.2f} "
                f"{(full_recon_results['perplexity'] - baseline_results['perplexity']) / baseline_results['perplexity'] * 100:<10.1f}%")
    
    for k_sharp in k_values:
        r = results['k_sharp_results'][k_sharp]
        logger.info(f"{k_sharp:<10} {r['perplexity']:<15.2f} "
                    f"{r['degradation']:<15.2f} {r['relative_degradation_pct']:<10.1f}%")
    
    # Find sweet spot
    logger.info("\n" + "-" * 40)
    sweet_spot = None
    for k_sharp in sorted(k_values, reverse=True):
        r = results['k_sharp_results'][k_sharp]
        if r['relative_degradation_pct'] < 10:  # Less than 10% degradation
            sweet_spot = k_sharp
    
    if sweet_spot:
        logger.info(f"Recommended k_sharp: {sweet_spot}")
        logger.info(f"  (Lowest k with <10% perplexity degradation)")
    else:
        logger.info("Warning: All k values caused >10% degradation")
        logger.info("Consider using higher k_sharp or improving SAE quality")
    
    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run intervention baseline experiment")
    
    parser.add_argument('--model', type=str, default='EleutherAI/pythia-70m',
                        help='Model to test')
    parser.add_argument('--layer', type=int, default=3,
                        help='Layer to intervene on')
    parser.add_argument('--k_values', type=str, default='5,10,25,50,100',
                        help='Comma-separated k_sharp values to test')
    parser.add_argument('--sae_checkpoint', type=str, default=None,
                        help='Path to pre-trained SAE checkpoint')
    parser.add_argument('--sae_k', type=int, default=50,
                        help='k value used to train SAE')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max evaluation samples (for quick tests)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with 100 samples')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for results JSON')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    k_values = [int(k) for k in args.k_values.split(',')]
    max_samples = 100 if args.quick else args.max_samples
    
    run_intervention_baseline(
        model_name=args.model,
        layer_idx=args.layer,
        k_values=k_values,
        sae_checkpoint=args.sae_checkpoint,
        sae_k_train=args.sae_k,
        max_eval_samples=max_samples,
        device=args.device,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()