#!/usr/bin/env python3
"""
Train a high-quality SAE on Pythia-70M activations.

This script trains an SAE with proper hyperparameters and includes
comprehensive diagnostics to verify quality before intervention experiments.

Usage:
    # Quick test (10k tokens, 10 epochs)
    python train_sae_pythia.py --quick
    
    # Full training (500k tokens, 50 epochs)
    python train_sae_pythia.py --output checkpoints/sae_pythia70m_layer3.pt
    
    # With diagnostics only (load existing SAE)
    python train_sae_pythia.py --checkpoint path/to/sae.pt --diagnose-only
"""

import argparse
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from typing import Dict, Optional
from tqdm import tqdm
import json
import math

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from sae_ception.models.sae import create_sae
from sae_ception.utils.hooks import ActivationExtractor
from sae_ception.utils.data import (
    create_causal_lm_dataloader,
    extract_activations_all_tokens,
    create_activation_dataloader,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def diagnose_sae(
    sae: torch.nn.Module,
    activations: torch.Tensor,
    device: str = 'cuda',
    n_samples: int = 10000,
) -> Dict[str, float]:
    """
    Compute comprehensive diagnostics for SAE quality.
    
    Returns metrics that indicate whether the SAE is good enough for intervention.
    """
    sae.eval()
    sae.to(device)
    
    # Sample activations if we have too many
    if len(activations) > n_samples:
        indices = torch.randperm(len(activations))[:n_samples]
        sample_acts = activations[indices].to(device)
    else:
        sample_acts = activations.to(device)
    
    with torch.no_grad():
        # Forward pass
        recon, sparse, info = sae(sample_acts)
        
        # 1. Reconstruction metrics
        mse = F.mse_loss(recon, sample_acts).item()
        var = sample_acts.var().item()
        relative_error = mse / var if var > 0 else float('inf')
        
        # Cosine similarity
        cos_sim = F.cosine_similarity(recon, sample_acts, dim=-1).mean().item()
        
        # 2. Sparsity metrics
        # L0: number of active features per sample
        active_mask = sparse > 0
        l0 = active_mask.float().sum(dim=1).mean().item()
        
        # 3. Dead features: features that never activate
        feature_usage = active_mask.float().mean(dim=0)  # [hidden_dim]
        dead_features = (feature_usage == 0).sum().item()
        total_features = sparse.shape[1]
        dead_pct = 100 * dead_features / total_features
        
        # 4. Feature activation statistics
        if active_mask.any():
            active_values = sparse[active_mask]
            mean_activation = active_values.mean().item()
            max_activation = active_values.max().item()
            std_activation = active_values.std().item()
        else:
            mean_activation = max_activation = std_activation = 0.0
        
        # 5. Reconstruction by magnitude
        # Check if we reconstruct high-magnitude activations better
        act_norms = sample_acts.norm(dim=-1)
        top_quartile_mask = act_norms > act_norms.quantile(0.75)
        if top_quartile_mask.any():
            top_mse = F.mse_loss(recon[top_quartile_mask], sample_acts[top_quartile_mask]).item()
            top_relative_error = top_mse / sample_acts[top_quartile_mask].var().item()
        else:
            top_relative_error = relative_error
    
    metrics = {
        'mse': mse,
        'relative_error': relative_error,
        'cosine_similarity': cos_sim,
        'l0': l0,
        'dead_features': dead_features,
        'dead_pct': dead_pct,
        'total_features': total_features,
        'mean_activation': mean_activation,
        'max_activation': max_activation,
        'std_activation': std_activation,
        'top_quartile_relative_error': top_relative_error,
    }
    
    return metrics


def print_diagnostics(metrics: Dict[str, float], expected_k: int):
    """Print SAE diagnostics with quality assessment."""
    
    print("\n" + "=" * 60)
    print("SAE DIAGNOSTICS")
    print("=" * 60)
    
    print("\n📊 Reconstruction Quality:")
    print(f"  MSE:              {metrics['mse']:.6f}")
    print(f"  Relative Error:   {metrics['relative_error']:.2%}")
    print(f"  Cosine Similarity:{metrics['cosine_similarity']:.4f}")
    print(f"  Top-25% Rel Error:{metrics['top_quartile_relative_error']:.2%}")
    
    print("\n📊 Sparsity:")
    print(f"  Mean L0:          {metrics['l0']:.1f} (expected ~{expected_k})")
    print(f"  Dead Features:    {metrics['dead_features']}/{metrics['total_features']} ({metrics['dead_pct']:.1f}%)")
    
    print("\n📊 Activation Statistics:")
    print(f"  Mean (when active):{metrics['mean_activation']:.4f}")
    print(f"  Std:              {metrics['std_activation']:.4f}")
    print(f"  Max:              {metrics['max_activation']:.4f}")
    
    # Quality assessment
    print("\n" + "-" * 60)
    print("QUALITY ASSESSMENT:")
    
    issues = []
    
    if metrics['relative_error'] > 0.50:
        issues.append("⚠️  HIGH RECONSTRUCTION ERROR (>50%) - SAE needs more training")
    elif metrics['relative_error'] > 0.20:
        issues.append("⚠️  MODERATE RECONSTRUCTION ERROR (>20%) - Consider more training")
    else:
        print("✓ Reconstruction error is acceptable (<20%)")
    
    if metrics['dead_pct'] > 50:
        issues.append("⚠️  MANY DEAD FEATURES (>50%) - Reduce L1/sparsity penalty")
    elif metrics['dead_pct'] > 30:
        issues.append("⚠️  SOME DEAD FEATURES (>30%) - Consider reducing L1")
    else:
        print(f"✓ Dead features acceptable ({metrics['dead_pct']:.1f}%)")
    
    if metrics['cosine_similarity'] < 0.8:
        issues.append("⚠️  LOW COSINE SIMILARITY (<0.8) - Reconstructions poorly aligned")
    else:
        print(f"✓ Cosine similarity good ({metrics['cosine_similarity']:.3f})")
    
    l0_ratio = metrics['l0'] / expected_k
    if l0_ratio < 0.5 or l0_ratio > 2.0:
        issues.append(f"⚠️  L0 ({metrics['l0']:.1f}) far from expected k={expected_k}")
    else:
        print(f"✓ L0 close to expected ({metrics['l0']:.1f} vs {expected_k})")
    
    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("\n✅ SAE looks good for intervention experiments!")
    
    # Estimate perplexity impact
    print("\n" + "-" * 60)
    print("ESTIMATED PERPLEXITY IMPACT:")
    
    # Very rough heuristic based on relative error
    if metrics['relative_error'] < 0.05:
        print("  Expected PPL degradation: <5% (excellent)")
    elif metrics['relative_error'] < 0.10:
        print("  Expected PPL degradation: 5-15% (good)")
    elif metrics['relative_error'] < 0.20:
        print("  Expected PPL degradation: 15-50% (moderate)")
    elif metrics['relative_error'] < 0.50:
        print("  Expected PPL degradation: 50-200% (high)")
    else:
        print("  Expected PPL degradation: >200% (very high - SAE not ready)")
    
    print("=" * 60 + "\n")
    
    return len(issues) == 0


def train_sae(
    model: torch.nn.Module,
    tokenizer,
    layer_idx: int,
    hidden_size: int,
    expansion_factor: int = 8,
    k: int = 32,
    n_samples: int = 50000,
    n_epochs: int = 50,
    batch_size: int = 256,
    lr: float = 3e-4,
    device: str = 'cuda',
    output_path: Optional[str] = None,
    dataset_name: str = 'wikitext',
    wandb_run=None,
) -> torch.nn.Module:
    """Train an SAE with proper hyperparameters."""

    sae_hidden = hidden_size * expansion_factor

    # Extract activations
    logger.info(f"Extracting activations from layer {layer_idx}...")
    logger.info(f"Target: {n_samples} samples from {dataset_name}")

    # Estimate how many sequences we need
    seq_len = 512
    avg_tokens_per_sample = seq_len * 0.7  # Account for padding
    n_sequences = int(n_samples / avg_tokens_per_sample) + 100  # Extra buffer

    train_loader = create_causal_lm_dataloader(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        split='train',
        batch_size=8,
        max_length=seq_len,
        max_samples=n_sequences,
    )
    
    acts, token_ids, _ = extract_activations_all_tokens(
        model=model,
        dataloader=train_loader,
        layer_name=str(layer_idx),
        device=device,
        include_padding=False,  # Don't include padding tokens
    )
    
    logger.info(f"Extracted {len(acts)} token activations")
    
    # Limit to n_samples if we got more
    if len(acts) > n_samples:
        indices = torch.randperm(len(acts))[:n_samples]
        acts = acts[indices]
        logger.info(f"Subsampled to {len(acts)} activations")
    
    # Create SAE
    sae = create_sae(
        input_dim=hidden_size,
        hidden_dim=sae_hidden,
        sae_type='topk',
        k=k,
    )
    sae.to(device)
    
    # Create dataloader for SAE training
    dummy_labels = torch.zeros(len(acts), dtype=torch.long)
    act_loader = create_activation_dataloader(
        activations=acts,
        labels=dummy_labels,
        batch_size=batch_size,
        shuffle=True,
    )
    
    # Optimizer and scheduler
    optimizer = Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.1)
    
    # Training loop
    logger.info(f"\nTraining SAE for {n_epochs} epochs...")
    logger.info(f"  Hidden dim: {sae_hidden} ({expansion_factor}x)")
    logger.info(f"  k (TopK): {k}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {lr}")
    
    best_loss = float('inf')
    best_state = None
    
    sae.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_aux_loss = 0
        n_batches = 0
        
        for batch in act_loader:
            x = batch['activations'].to(device)
            
            # Forward pass
            recon, sparse, info = sae(x)
            loss_dict = sae.loss(x, recon, sparse, info)
            loss = loss_dict['total_loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            
            optimizer.step()
            
            # Normalize decoder columns
            if hasattr(sae, 'normalize_decoder_'):
                sae.normalize_decoder_()
            
            epoch_loss += loss.item()
            epoch_recon_loss += loss_dict.get('recon_loss', loss_dict['total_loss']).item()
            if 'aux_loss' in loss_dict:
                epoch_aux_loss += loss_dict['aux_loss'].item()
            n_batches += 1
        
        scheduler.step()
        
        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon_loss / n_batches
        avg_aux = epoch_aux_loss / n_batches if epoch_aux_loss > 0 else 0
        
        # Track best
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in sae.state_dict().items()}
        
        # Logging
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch+1:3d}/{n_epochs}: "
                f"loss={avg_loss:.4f} (recon={avg_recon:.4f}, aux={avg_aux:.4f}), "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        # W&B logging
        if wandb_run is not None:
            wandb_run.log({
                'epoch': epoch + 1,
                'train/loss': avg_loss,
                'train/recon_loss': avg_recon,
                'train/aux_loss': avg_aux,
                'train/lr': scheduler.get_last_lr()[0],
            })
    
    # Load best state
    if best_state is not None:
        sae.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        logger.info(f"\nLoaded best checkpoint (loss={best_loss:.4f})")
    
    sae.eval()
    
    # Save checkpoint
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': sae.state_dict(),
            'config': {
                'input_dim': hidden_size,
                'hidden_dim': sae_hidden,
                'expansion_factor': expansion_factor,
                'k': k,
                'layer_idx': layer_idx,
            },
            'training': {
                'n_samples': len(acts),
                'n_epochs': n_epochs,
                'best_loss': best_loss,
            }
        }
        torch.save(checkpoint, output_path)
        logger.info(f"Saved checkpoint to {output_path}")
    
    return sae, acts


def main():
    parser = argparse.ArgumentParser(description="Train high-quality SAE on Pythia")
    
    parser.add_argument('--model', type=str, default='EleutherAI/pythia-70m')
    parser.add_argument('--layer', type=int, default=3)
    parser.add_argument('--expansion', type=int, default=8)
    parser.add_argument('--k', type=int, default=32, help='TopK sparsity')
    parser.add_argument('--samples', type=int, default=500000, help='Number of token activations')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--output', type=str, default=None, help='Output checkpoint path')
    parser.add_argument('--checkpoint', type=str, default=None, help='Load existing checkpoint')
    parser.add_argument('--diagnose-only', action='store_true', help='Only run diagnostics')
    parser.add_argument('--quick', action='store_true', help='Quick test (10k samples, 10 epochs)')
    parser.add_argument('--dataset', type=str, default='wikitext', help='Dataset name (wikitext, pile, etc.)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='sae-ception', help='W&B project name')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='W&B run name (auto-generated if not set)')
    
    args = parser.parse_args()
    
    # Quick mode overrides
    if args.quick:
        args.samples = 10000
        args.epochs = 10
        logger.info("Quick mode: 10k samples, 10 epochs")

    # Initialize W&B if enabled
    wandb_run = None
    if args.wandb:
        if not WANDB_AVAILABLE:
            logger.warning("W&B requested but not installed. Skipping.")
        else:
            run_name = args.wandb_run_name or f"sae_{args.model.split('/')[-1]}_k{args.k}_layer{args.layer}"
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    'model': args.model,
                    'layer': args.layer,
                    'expansion_factor': args.expansion,
                    'k': args.k,
                    'samples': args.samples,
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'lr': args.lr,
                    'dataset': args.dataset,
                },
            )
            logger.info(f"W&B initialized: {wandb_run.url}")

    # Determine base model name for tokenizer
    if args.model.endswith('.pt'):
        # Load checkpoint to get base model name
        checkpoint = torch.load(args.model, map_location='cpu')
        base_model_name = checkpoint['config'].get('model_name', 'EleutherAI/pythia-70m')
        logger.info(f"Loading tokenizer from base model: {base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    else:
        logger.info(f"Loading tokenizer from: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def load_model(model_path: str, device: str = 'cuda'):
        """Load model from HuggingFace name or local checkpoint."""
        
        if model_path.endswith('.pt'):
            # Local checkpoint from train_cpt.py
            logger.info(f"Loading model from checkpoint: {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            
            # Get the base model name from checkpoint config
            base_model_name = checkpoint['config'].get('model_name', 'EleutherAI/pythia-70m')
            
            # Load architecture
            model = AutoModelForCausalLM.from_pretrained(base_model_name)
            
            # Load trained weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            
            logger.info(f"  Loaded from base: {base_model_name}")
        else:
            # HuggingFace model name
            logger.info(f"Loading model from HuggingFace: {model_path}")
            model = AutoModelForCausalLM.from_pretrained(model_path)
            model.to(device)
        
        return model
    model = load_model(args.model, args.device)
    model.eval()
    
    hidden_size = model.config.hidden_size
    logger.info(f"Hidden size: {hidden_size}")
    
    # Either load or train SAE
    if args.checkpoint and Path(args.checkpoint).exists():
        logger.info(f"Loading SAE from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        
        config = checkpoint.get('config', {})
        sae = create_sae(
            input_dim=config.get('input_dim', hidden_size),
            hidden_dim=config.get('hidden_dim', hidden_size * args.expansion),
            sae_type='topk',
            k=config.get('k', args.k),
        )
        sae.load_state_dict(checkpoint['model_state_dict'])
        sae.to(args.device)
        
        # Get activations for diagnostics
        logger.info("Extracting activations for diagnostics...")
        train_loader = create_causal_lm_dataloader(
            tokenizer=tokenizer,
            dataset_name='wikitext',
            split='test',  # Use test set for diagnostics
            batch_size=8,
            max_length=512,
            max_samples=1000,
        )
        acts, _, _ = extract_activations_all_tokens(
            model=model,
            dataloader=train_loader,
            layer_name=str(args.layer),
            device=args.device,
        )
    else:
        if args.diagnose_only:
            logger.error("--diagnose-only requires --checkpoint")
            return
        
        sae, acts = train_sae(
            model=model,
            tokenizer=tokenizer,
            layer_idx=args.layer,
            hidden_size=hidden_size,
            expansion_factor=args.expansion,
            k=args.k,
            n_samples=args.samples,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            output_path=args.output,
            dataset_name=args.dataset,
            wandb_run=wandb_run,
        )
    
    # Run diagnostics
    logger.info("\nRunning diagnostics...")
    metrics = diagnose_sae(sae, acts, device=args.device)
    is_good = print_diagnostics(metrics, expected_k=args.k)
    
    # Save diagnostics
    if args.output:
        diag_path = Path(args.output).with_suffix('.diagnostics.json')
        with open(diag_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Diagnostics saved to {diag_path}")

    # Log final diagnostics to W&B
    if wandb_run is not None:
        wandb_run.log({
            'diagnostics/l0_sparsity': metrics.get('l0_sparsity', 0),
            'diagnostics/dead_features_pct': metrics.get('dead_features_pct', 0),
            'diagnostics/mean_cosine_sim': metrics.get('mean_cosine_sim', 0),
            'diagnostics/reconstruction_mse': metrics.get('reconstruction_mse', 0),
            'diagnostics/relative_reconstruction_error': metrics.get('relative_reconstruction_error', 0),
        })
        wandb_run.finish()
        logger.info("W&B run finished")

    return is_good


if __name__ == "__main__":
    main()