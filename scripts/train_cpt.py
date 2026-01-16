#!/usr/bin/env python3
"""
Continued Pre-Training (CPT) with SAE-ception auxiliary loss.

This script implements the core SAE-ception training loop for causal language models:
    loss = next_token_loss + λ * MSE(activations, sharpened_reconstruction)

The model is trained on The Pile while being guided toward activations that,
when sharpened through the SAE, still preserve task-relevant information.

Usage:
    # Quick test
    python scripts/train_cpt.py --quick
    
    # Full training
    python scripts/train_cpt.py \
        --model EleutherAI/pythia-70m \
        --sae_checkpoint checkpoints/sae_k768.pt \
        --sae_k 768 \
        --k_sharp 512 \
        --aux_weight 0.01 \
        --train_samples 100000 \
        --epochs 1 \
        --output checkpoints/pythia_cpt_cycle1.pt
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

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from sae_ception.models.sae import create_sae
from sae_ception.utils.hooks import ActivationExtractor, create_sae_intervention
from sae_ception.utils.data import create_causal_lm_dataloader
from sae_ception.evaluation.lm_metrics import evaluate_perplexity

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_sae_from_checkpoint(
    checkpoint_path: str,
    device: str = 'cuda',
) -> nn.Module:
    """Load SAE from checkpoint with config."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get('config', {})
    
    sae = create_sae(
        input_dim=config.get('input_dim', 512),
        hidden_dim=config.get('hidden_dim', 4096),
        sae_type='topk',
        k=config.get('k', 768),
    )
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.to(device)
    sae.eval()
    
    logger.info(f"Loaded SAE from {checkpoint_path}")
    logger.info(f"  Input dim: {config.get('input_dim')}, Hidden dim: {config.get('hidden_dim')}")
    logger.info(f"  k: {config.get('k')}")
    
    return sae, config


def compute_sharpened_target(
    activations: torch.Tensor,
    sae: nn.Module,
    k_sharp: int,
) -> torch.Tensor:
    """
    Compute sharpened reconstruction target.
    
    Args:
        activations: [batch, seq_len, hidden_dim] model activations
        sae: Trained SAE
        k_sharp: Number of top features to keep
        
    Returns:
        sharpened_recon: [batch, seq_len, hidden_dim] sharpened target
    """
    original_shape = activations.shape
    batch_size, seq_len, hidden_dim = original_shape
    
    # Flatten to [batch * seq, hidden]
    flat_acts = activations.view(-1, hidden_dim)
    
    with torch.no_grad():
        # Encode
        sparse_codes = sae.encode(flat_acts)
        
        # Sharpen: keep only top-k
        topk_vals, topk_idx = torch.topk(sparse_codes, k=k_sharp, dim=-1)
        sharpened_codes = torch.zeros_like(sparse_codes)
        sharpened_codes.scatter_(-1, topk_idx, topk_vals)
        
        # Decode
        sharpened_recon = sae.decode(sharpened_codes)
    
    return sharpened_recon.view(original_shape)


def train_epoch_cpt(
    model: nn.Module,
    sae: nn.Module,
    train_loader,
    optimizer,
    scheduler,
    layer_idx: int,
    k_sharp: int,
    aux_weight: float,
    device: str,
    max_grad_norm: float = 1.0,
    log_interval: int = 100,
) -> Dict[str, float]:
    """
    Train one epoch with CPT + auxiliary loss.
    
    Args:
        model: Language model to train
        sae: Frozen SAE for generating targets
        train_loader: DataLoader for training data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        layer_idx: Which layer to apply auxiliary loss
        k_sharp: Sharpening k for auxiliary targets
        aux_weight: Weight λ for auxiliary loss
        device: Device
        max_grad_norm: Gradient clipping
        log_interval: Steps between logging
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    sae.eval()
    
    total_loss = 0.0
    total_lm_loss = 0.0
    total_aux_loss = 0.0
    total_tokens = 0
    n_batches = 0
    
    # Get the base model for hook attachment
    if hasattr(model, 'gpt_neox'):
        base_model = model.gpt_neox
    elif hasattr(model, 'transformer'):
        base_model = model.transformer
    else:
        base_model = model
    
    progress_bar = tqdm(train_loader, desc="Training CPT")
    
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
        # NOTE: detach=False is CRITICAL - it allows aux_loss gradients to flow back
        # through the model, which is the whole point of CPT with auxiliary loss!
        with ActivationExtractor(base_model, str(layer_idx), detach=False) as extractor:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            # Get activations at target layer (with gradients preserved)
            activations = extractor.get_activations()  # [batch, seq, hidden]
        
        # Language modeling loss (next-token prediction)
        lm_loss = outputs.loss
        
        # Compute sharpened target
        sharpened_target = compute_sharpened_target(activations, sae, k_sharp)
        
        # Auxiliary loss: MSE between activations and sharpened reconstruction
        # Only compute on non-padding positions
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # [batch, seq, 1]
            aux_loss = ((activations - sharpened_target) ** 2 * mask).sum()
            aux_loss = aux_loss / mask.sum() / activations.shape[-1]
        else:
            aux_loss = F.mse_loss(activations, sharpened_target)
        
        # Combined loss
        loss = lm_loss + aux_weight * aux_loss
        
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
        total_aux_loss += aux_loss.item() * n_tokens
        total_tokens += n_tokens
        n_batches += 1
        
        # Update progress bar
        if batch_idx % log_interval == 0:
            avg_loss = total_loss / total_tokens
            avg_lm = total_lm_loss / total_tokens
            avg_aux = total_aux_loss / total_tokens
            ppl = math.exp(avg_lm) if avg_lm < 20 else float('inf')
            
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lm': f'{avg_lm:.4f}',
                'aux': f'{avg_aux:.4f}',
                'ppl': f'{ppl:.2f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
            })
    
    avg_loss = total_loss / total_tokens
    avg_lm_loss = total_lm_loss / total_tokens
    avg_aux_loss = total_aux_loss / total_tokens
    
    return {
        'loss': avg_loss,
        'lm_loss': avg_lm_loss,
        'aux_loss': avg_aux_loss,
        'perplexity': math.exp(avg_lm_loss) if avg_lm_loss < 20 else float('inf'),
        'tokens': total_tokens,
    }


def train_cpt(
    model_name: str = 'EleutherAI/pythia-70m',
    sae_checkpoint: str = None,
    sae_k: int = 768,
    k_sharp: int = 512,
    layer_idx: int = 3,
    aux_weight: float = 0.01,
    train_samples: int = 100000,
    eval_samples: int = 500,
    epochs: int = 1,
    batch_size: int = 4,
    lr: float = 1e-5,
    warmup_steps: int = 100,
    device: str = 'cuda',
    output_path: Optional[str] = None,
    quick: bool = False,
    wandb_run=None,
    gradient_checkpointing: bool = False,
):
    """
    Run CPT with SAE-ception auxiliary loss.

    Args:
        model_name: HuggingFace model name
        sae_checkpoint: Path to pre-trained SAE
        sae_k: k value the SAE was trained with
        k_sharp: Sharpening k for auxiliary targets
        layer_idx: Layer to apply auxiliary loss
        aux_weight: Weight λ for auxiliary loss
        train_samples: Number of training samples
        eval_samples: Number of eval samples
        epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        warmup_steps: Warmup steps
        device: Device
        output_path: Path to save trained model
        quick: Quick test mode
        wandb_run: Weights & Biases run object (optional)
        gradient_checkpointing: Enable gradient checkpointing to reduce memory usage
                                (trades compute for memory, ~30% slower but much less VRAM)
    """
    if quick:
        train_samples = 1000
        eval_samples = 100
        epochs = 1
        logger.info("Quick mode: 1k samples, 1 epoch")
    
    logger.info("=" * 60)
    logger.info("CPT with SAE-ception Auxiliary Loss")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Layer: {layer_idx}")
    logger.info(f"SAE k: {sae_k}, Sharpen k: {k_sharp}")
    logger.info(f"Aux weight (λ): {aux_weight}")
    logger.info(f"Train samples: {train_samples}")
    logger.info(f"Epochs: {epochs}")
    
    # Load tokenizer and model
    logger.info("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    # Enable gradient checkpointing if requested (reduces memory, increases compute)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing ENABLED (memory-efficient mode)")

    hidden_size = model.config.hidden_size
    logger.info(f"Hidden size: {hidden_size}")
    
    # Load or create SAE
    logger.info(f"\nLooking for SAE checkpoint...")
    logger.info(f"  sae_checkpoint arg: {sae_checkpoint}")
    if sae_checkpoint:
        sae_path = Path(sae_checkpoint)
        logger.info(f"  Resolved path: {sae_path.absolute()}")
        logger.info(f"  Path exists: {sae_path.exists()}")
    
    if sae_checkpoint and Path(sae_checkpoint).exists():
        sae, sae_config = load_sae_from_checkpoint(sae_checkpoint, device)
    else:
        logger.error("SAE checkpoint not found!")
        if sae_checkpoint:
            logger.error(f"  Checked path: {Path(sae_checkpoint).absolute()}")
            # List what's in the parent directory
            parent = Path(sae_checkpoint).parent
            if parent.exists():
                logger.error(f"  Files in {parent}:")
                for f in parent.iterdir():
                    logger.error(f"    - {f.name}")
            else:
                logger.error(f"  Parent directory doesn't exist: {parent}")
        raise ValueError("SAE checkpoint required for CPT. Train one first with train_sae_pretrained.py")
    
    # Create data loaders
    logger.info("\nLoading training data (The Pile)...")
    
    # For now, use WikiText for training too (Pile loading to be added)
    # This is a placeholder - replace with create_pile_dataloader when ready
    train_loader = create_causal_lm_dataloader(
        tokenizer=tokenizer,
        dataset_name='wikitext',
        split='train',
        batch_size=batch_size,
        max_length=1024,
        max_samples=train_samples,
    )
    
    logger.info("Loading evaluation data (WikiText)...")
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
        'sae_checkpoint': sae_checkpoint,
        'sae_k': sae_k,
        'k_sharp': k_sharp,
        'layer_idx': layer_idx,
        'aux_weight': aux_weight,
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
        train_metrics = train_epoch_cpt(
            model=model,
            sae=sae,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            layer_idx=layer_idx,
            k_sharp=k_sharp,
            aux_weight=aux_weight,
            device=device,
        )
        
        logger.info(f"\nTrain metrics:")
        logger.info(f"  Loss: {train_metrics['loss']:.4f}")
        logger.info(f"  LM Loss: {train_metrics['lm_loss']:.4f}")
        logger.info(f"  Aux Loss: {train_metrics['aux_loss']:.4f}")
        logger.info(f"  Perplexity: {train_metrics['perplexity']:.2f}")
        
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

        # W&B logging
        if wandb_run is not None:
            wandb_run.log({
                'epoch': epoch + 1,
                'train/loss': train_metrics['loss'],
                'train/lm_loss': train_metrics['lm_loss'],
                'train/aux_loss': train_metrics['aux_loss'],
                'train/perplexity': train_metrics['perplexity'],
                'eval/perplexity': eval_results['perplexity'],
                'eval/ppl_change': eval_results['perplexity'] - baseline_results['perplexity'],
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
                'sae_k': sae_k,
                'k_sharp': k_sharp,
                'aux_weight': aux_weight,
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

    # Log final results to W&B
    if wandb_run is not None:
        wandb_run.log({
            'final/baseline_perplexity': baseline_results['perplexity'],
            'final/perplexity': final_results['perplexity'],
            'final/ppl_change': results['perplexity_change'],
            'final/ppl_change_pct': results['perplexity_change_pct'],
        })
        wandb_run.finish()
        logger.info("W&B run finished")

    return model, results


def main():
    parser = argparse.ArgumentParser(description="CPT with SAE-ception auxiliary loss")
    
    parser.add_argument('--model', type=str, default='EleutherAI/pythia-70m')
    parser.add_argument('--sae_checkpoint', type=str, required=True,
                        help='Path to pre-trained SAE checkpoint')
    parser.add_argument('--sae_k', type=int, default=768,
                        help='k value the SAE was trained with')
    parser.add_argument('--k_sharp', type=int, default=512,
                        help='Sharpening k for auxiliary targets')
    parser.add_argument('--layer', type=int, default=3,
                        help='Layer to apply auxiliary loss')
    parser.add_argument('--aux_weight', type=float, default=0.01,
                        help='Weight λ for auxiliary loss')
    parser.add_argument('--train_samples', type=int, default=100000)
    parser.add_argument('--eval_samples', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Enable gradient checkpointing (reduces memory ~50%%, increases compute ~30%%)')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='sae-ception', help='W&B project name')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='W&B run name')

    args = parser.parse_args()

    # Initialize W&B if enabled
    wandb_run = None
    if args.wandb:
        if not WANDB_AVAILABLE:
            logger.warning("W&B requested but not installed. Skipping.")
        else:
            run_name = args.wandb_run_name or f"cpt_{args.model.split('/')[-1]}_k{args.k_sharp}_aux{args.aux_weight}"
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    'model': args.model,
                    'sae_checkpoint': args.sae_checkpoint,
                    'sae_k': args.sae_k,
                    'k_sharp': args.k_sharp,
                    'layer': args.layer,
                    'aux_weight': args.aux_weight,
                    'train_samples': args.train_samples,
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'lr': args.lr,
                },
            )
            logger.info(f"W&B initialized: {wandb_run.url}")

    train_cpt(
        model_name=args.model,
        sae_checkpoint=args.sae_checkpoint,
        sae_k=args.sae_k,
        k_sharp=args.k_sharp,
        layer_idx=args.layer,
        aux_weight=args.aux_weight,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        device=args.device,
        output_path=args.output,
        quick=args.quick,
        wandb_run=wandb_run,
        gradient_checkpointing=args.gradient_checkpointing,
    )


if __name__ == "__main__":
    main()