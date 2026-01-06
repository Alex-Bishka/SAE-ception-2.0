#!/usr/bin/env python3
"""
Continued Pre-Training (CPT) with L1 penalty on activations (Run B baseline).

This script implements the L1 regularization baseline:
    loss = next_token_loss + λ * ||activations||₁

This tests whether simple L1 sparsity pressure achieves similar effects to
SAE-ception's structured auxiliary loss.

Usage:
    # Quick test
    python scripts/train_cpt_l1.py --quick
    
    # Full training
    python scripts/train_cpt_l1.py \
        --model EleutherAI/pythia-70m \
        --layer 3 \
        --l1_weight 0.001 \
        --train_samples 100000 \
        --epochs 1 \
        --output checkpoints/pythia_cpt_l1.pt
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from typing import Optional, Dict
from tqdm import tqdm
import json
import math

from sae_ception.utils.hooks import ActivationExtractor
from sae_ception.utils.data import create_causal_lm_dataloader
from sae_ception.evaluation.lm_metrics import evaluate_perplexity

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def train_epoch_cpt_l1(
    model: nn.Module,
    train_loader,
    optimizer,
    scheduler,
    layer_idx: int,
    l1_weight: float,
    device: str,
    max_grad_norm: float = 1.0,
    log_interval: int = 100,
) -> Dict[str, float]:
    """
    Train one epoch with L1 penalty on activations.
    
    Args:
        model: Language model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        layer_idx: Which layer to apply L1 penalty
        l1_weight: Weight λ for L1 penalty
        device: Device
        max_grad_norm: Gradient clipping
        log_interval: Steps between logging
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    
    total_loss = 0.0
    total_lm_loss = 0.0
    total_l1_loss = 0.0
    total_tokens = 0
    n_batches = 0
    
    # Track activation statistics
    total_l0 = 0.0  # Average number of "active" features
    total_l1_norm = 0.0  # Raw L1 norm
    
    # Get the base model for hook attachment
    if hasattr(model, 'gpt_neox'):
        base_model = model.gpt_neox
    elif hasattr(model, 'transformer'):
        base_model = model.transformer
    else:
        base_model = model
    
    progress_bar = tqdm(train_loader, desc="Training CPT + L1")
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Create labels (shifted input_ids for causal LM)
        labels = input_ids.clone()
        if attention_mask is not None:
            labels[attention_mask == 0] = -100  # Ignore padding
        
        # Forward pass with activation capture
        with ActivationExtractor(base_model, str(layer_idx)) as extractor:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            # Get activations at target layer
            activations = extractor.get_activations()  # [batch, seq, hidden]
        
        # Language modeling loss (next-token prediction)
        lm_loss = outputs.loss
        
        # L1 penalty on activations
        # Only compute on non-padding positions
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # [batch, seq, 1]
            # Masked L1 norm
            l1_norm = (activations.abs() * mask).sum() / mask.sum()
        else:
            l1_norm = activations.abs().mean()
        
        l1_loss = l1_weight * l1_norm
        
        # Combined loss
        loss = lm_loss + l1_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        
        # Track metrics
        if attention_mask is not None:
            n_tokens = attention_mask.sum().item()
        else:
            n_tokens = input_ids.numel()
        
        total_loss += loss.item() * n_tokens
        total_lm_loss += lm_loss.item() * n_tokens
        total_l1_loss += l1_loss.item() * n_tokens
        total_tokens += n_tokens
        n_batches += 1
        
        # Track activation statistics
        with torch.no_grad():
            # L0: count of "active" features (> threshold)
            threshold = 0.1
            if attention_mask is not None:
                active_count = ((activations.abs() > threshold) * mask).sum()
                total_elements = mask.sum() * activations.shape[-1]
                l0 = active_count / total_elements
            else:
                l0 = (activations.abs() > threshold).float().mean()
            total_l0 += l0.item()
            total_l1_norm += l1_norm.item()
        
        # Update progress bar
        if batch_idx % log_interval == 0:
            avg_loss = total_loss / total_tokens
            avg_lm = total_lm_loss / total_tokens
            avg_l1 = total_l1_loss / total_tokens
            ppl = math.exp(avg_lm) if avg_lm < 20 else float('inf')
            
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lm': f'{avg_lm:.4f}',
                'l1': f'{avg_l1:.4f}',
                'ppl': f'{ppl:.2f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
            })
    
    avg_loss = total_loss / total_tokens
    avg_lm_loss = total_lm_loss / total_tokens
    avg_l1_loss = total_l1_loss / total_tokens
    
    return {
        'loss': avg_loss,
        'lm_loss': avg_lm_loss,
        'l1_loss': avg_l1_loss,
        'perplexity': math.exp(avg_lm_loss) if avg_lm_loss < 20 else float('inf'),
        'tokens': total_tokens,
        'avg_l0': total_l0 / n_batches,  # Sparsity metric
        'avg_l1_norm': total_l1_norm / n_batches,
    }


def train_cpt_l1(
    model_name: str = 'EleutherAI/pythia-70m',
    layer_idx: int = 3,
    l1_weight: float = 0.001,
    train_samples: int = 100000,
    eval_samples: int = 500,
    epochs: int = 1,
    batch_size: int = 4,
    lr: float = 1e-5,
    warmup_steps: int = 100,
    device: str = 'cuda',
    output_path: Optional[str] = None,
    quick: bool = False,
):
    """
    Run CPT with L1 penalty on activations.
    
    Args:
        model_name: HuggingFace model name
        layer_idx: Layer to apply L1 penalty
        l1_weight: Weight λ for L1 penalty
        train_samples: Number of training samples
        eval_samples: Number of eval samples
        epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        warmup_steps: Warmup steps
        device: Device
        output_path: Path to save trained model
        quick: Quick test mode
    """
    if quick:
        train_samples = 1000
        eval_samples = 100
        epochs = 1
        logger.info("Quick mode: 1k samples, 1 epoch")
    
    logger.info("=" * 60)
    logger.info("CPT with L1 Penalty (Run B Baseline)")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Layer: {layer_idx}")
    logger.info(f"L1 weight (λ): {l1_weight}")
    logger.info(f"Train samples: {train_samples}")
    logger.info(f"Epochs: {epochs}")
    
    # Load tokenizer and model
    logger.info("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    
    hidden_size = model.config.hidden_size
    logger.info(f"Hidden size: {hidden_size}")
    
    # Create data loaders
    logger.info("\nLoading training data...")
    train_loader = create_causal_lm_dataloader(
        tokenizer=tokenizer,
        dataset_name='wikitext',
        split='train',
        batch_size=batch_size,
        max_length=1024,
        max_samples=train_samples,
    )
    
    logger.info("Loading evaluation data...")
    eval_loader = create_causal_lm_dataloader(
        tokenizer=tokenizer,
        dataset_name='wikitext',
        split='test',
        batch_size=batch_size,
        max_length=1024,
        max_samples=eval_samples,
    )
    
    # Baseline evaluation
    logger.info("\nBaseline evaluation...")
    baseline_results = evaluate_perplexity(model, eval_loader, device, show_progress=False)
    logger.info(f"Baseline perplexity: {baseline_results['perplexity']:.2f}")
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Calculate total steps
    total_steps = len(train_loader) * epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.1)
    
    # Training loop
    logger.info(f"\nTraining for {epochs} epochs...")
    
    results = {
        'model_name': model_name,
        'layer_idx': layer_idx,
        'l1_weight': l1_weight,
        'baseline_perplexity': baseline_results['perplexity'],
        'epochs': [],
    }
    
    best_ppl = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        logger.info('='*60)
        
        # Train
        train_metrics = train_epoch_cpt_l1(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            layer_idx=layer_idx,
            l1_weight=l1_weight,
            device=device,
        )
        
        logger.info(f"\nTrain metrics:")
        logger.info(f"  Loss: {train_metrics['loss']:.4f}")
        logger.info(f"  LM Loss: {train_metrics['lm_loss']:.4f}")
        logger.info(f"  L1 Loss: {train_metrics['l1_loss']:.4f}")
        logger.info(f"  Perplexity: {train_metrics['perplexity']:.2f}")
        logger.info(f"  Avg L0 (sparsity): {train_metrics['avg_l0']:.4f}")
        logger.info(f"  Avg L1 Norm: {train_metrics['avg_l1_norm']:.4f}")
        
        # Evaluate
        logger.info("\nEvaluating...")
        eval_results = evaluate_perplexity(model, eval_loader, device, show_progress=False)
        
        logger.info(f"  Eval Perplexity: {eval_results['perplexity']:.2f}")
        logger.info(f"  Change from baseline: {eval_results['perplexity'] - baseline_results['perplexity']:+.2f}")
        
        # Track best
        if eval_results['perplexity'] < best_ppl:
            best_ppl = eval_results['perplexity']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"  New best perplexity!")
        
        results['epochs'].append({
            'epoch': epoch + 1,
            'train': train_metrics,
            'eval_perplexity': eval_results['perplexity'],
        })
    
    # Load best model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        logger.info(f"\nLoaded best model (PPL: {best_ppl:.2f})")
    
    # Final evaluation
    logger.info("\nFinal evaluation...")
    final_results = evaluate_perplexity(model, eval_loader, device)
    
    results['final_perplexity'] = final_results['perplexity']
    results['perplexity_change'] = final_results['perplexity'] - baseline_results['perplexity']
    results['perplexity_change_pct'] = (final_results['perplexity'] - baseline_results['perplexity']) / baseline_results['perplexity'] * 100
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Baseline Perplexity: {baseline_results['perplexity']:.2f}")
    logger.info(f"Final Perplexity:    {final_results['perplexity']:.2f}")
    logger.info(f"Change:              {results['perplexity_change']:+.2f} ({results['perplexity_change_pct']:+.1f}%)")
    
    # Save model
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': {
                'model_name': model_name,
                'layer_idx': layer_idx,
                'l1_weight': l1_weight,
            },
            'results': results,
        }
        torch.save(checkpoint, output_path)
        logger.info(f"\nSaved model to {output_path}")
        
        # Save results JSON
        results_path = output_path.with_suffix('.results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {results_path}")
    
    return model, results


def main():
    parser = argparse.ArgumentParser(description="CPT with L1 penalty (Run B baseline)")
    
    parser.add_argument('--model', type=str, default='EleutherAI/pythia-70m')
    parser.add_argument('--layer', type=int, default=3,
                        help='Layer to apply L1 penalty')
    parser.add_argument('--l1_weight', type=float, default=0.001,
                        help='Weight λ for L1 penalty')
    parser.add_argument('--train_samples', type=int, default=100000)
    parser.add_argument('--eval_samples', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    train_cpt_l1(
        model_name=args.model,
        layer_idx=args.layer,
        l1_weight=args.l1_weight,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        device=args.device,
        output_path=args.output,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()