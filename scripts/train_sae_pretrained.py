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
import math
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from typing import Dict, Optional
import json

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from sae_ception.models.sae import create_sae
from sae_ception.utils.data import (
    create_causal_lm_dataloader,
    stream_activations,
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
    n_samples: int = 10000000,
    n_epochs: int = 4,
    batch_size: int = 4096,
    lr: float = 3e-4,
    device: str = 'cuda',
    output_path: Optional[str] = None,
    dataset_name: str = 'wikitext',
    wandb_run=None,
    # Resampling options
    resampling: bool = False,
    resampling_interval: int = 500,
    resampling_stop: float = 0.5,
    # Checkpoint options
    checkpoint_interval: int = 10,
) -> torch.nn.Module:
    """Train an SAE with streaming activation extraction (memory-efficient).

    Instead of extracting all activations upfront (which causes OOM for large
    datasets), this streams activations batch-by-batch during training.

    Saves periodic checkpoints every checkpoint_interval% of training for
    Pareto frontier evaluation.
    """
    sae_hidden = hidden_size * expansion_factor

    # Create SAE first
    sae = create_sae(
        input_dim=hidden_size,
        hidden_dim=sae_hidden,
        sae_type='topk',
        k=k,
    )
    sae.to(device)

    # Calculate total steps first (needed for scheduler)
    total_steps = (n_samples * n_epochs) // batch_size
    warmup_steps = min(1000, total_steps // 10)  # 1000 steps or 10% of training

    # Optimizer with warmup + cosine decay (SAELens best practice)
    optimizer = Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))

    def lr_lambda(step):
        """Warmup + cosine decay schedule."""
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # Setup checkpoint directory
    checkpoint_dir = None
    if output_path:
        checkpoint_dir = Path(output_path)
        if checkpoint_dir.suffix:  # If it's a file path, use parent dir
            checkpoint_dir = checkpoint_dir.parent / checkpoint_dir.stem
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_every = max(1, total_steps * checkpoint_interval // 100)
    resampling_stop_step = int(total_steps * resampling_stop)

    logger.info(f"\nTraining SAE (streaming mode)...")
    logger.info(f"  Target samples: {n_samples:,}")
    logger.info(f"  Epochs: {n_epochs} (total tokens: {n_samples * n_epochs:,})")
    logger.info(f"  Hidden dim: {sae_hidden} ({expansion_factor}x)")
    logger.info(f"  k (TopK): {k}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {lr} (warmup={warmup_steps} steps, cosine decay)")
    logger.info(f"  Dataset: {dataset_name}")
    logger.info(f"  Total steps: ~{total_steps:,}")
    logger.info(f"  Checkpoint every: {checkpoint_every} steps ({checkpoint_interval}%)")
    if resampling:
        logger.info(f"  Resampling: enabled (interval={resampling_interval}, stop at step {resampling_stop_step})")

    # Setup data loader - estimate sequences needed
    seq_len = 512
    avg_tokens_per_seq = seq_len * 0.7  # Account for padding
    n_sequences = int(n_samples / avg_tokens_per_seq) + 100

    best_loss = float('inf')
    best_state = None
    total_samples_seen = 0
    global_step = 0
    next_checkpoint_step = checkpoint_every
    saved_checkpoints = []

    def save_checkpoint(step: int, is_final: bool = False):
        """Save checkpoint with current metrics."""
        if checkpoint_dir is None:
            return
        name = "final.pt" if is_final else f"step_{step:06d}.pt"
        ckpt_path = checkpoint_dir / name
        ckpt = {
            'model_state_dict': sae.state_dict(),
            'config': {
                'input_dim': hidden_size,
                'hidden_dim': sae_hidden,
                'expansion_factor': expansion_factor,
                'k': k,
                'layer_idx': layer_idx,
            },
            'training': {
                'step': step,
                'total_samples': total_samples_seen,
                'n_epochs': n_epochs,
                'loss': best_loss,
            }
        }
        torch.save(ckpt, ckpt_path)
        saved_checkpoints.append(str(ckpt_path))
        logger.info(f"  Saved checkpoint: {ckpt_path}")

    for epoch in range(n_epochs):
        # Create fresh dataloader each epoch for streaming datasets
        train_loader = create_causal_lm_dataloader(
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            split='train',
            batch_size=64,  # Larger batch OK with streaming (no OOM risk)
            max_length=seq_len,
            max_samples=n_sequences,
        )

        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_aux_loss = 0
        n_batches = 0
        epoch_samples = 0

        sae.train()
        activation_buffer = []

        for act_batch in stream_activations(
            model=model,
            dataloader=train_loader,
            layer_idx=layer_idx,
            device=device,
            max_tokens=n_samples,
            show_progress=(epoch == 0),  # Only show progress on first epoch
        ):
            # Accumulate until we have enough for a training batch
            activation_buffer.append(act_batch.cpu())
            buffer_size = sum(a.shape[0] for a in activation_buffer)

            if buffer_size >= batch_size:
                # Concatenate and take a batch
                all_acts = torch.cat(activation_buffer, dim=0)
                x = all_acts[:batch_size].to(device)

                # Keep remainder in buffer
                if all_acts.shape[0] > batch_size:
                    activation_buffer = [all_acts[batch_size:]]
                else:
                    activation_buffer = []

                # Forward pass
                recon, sparse, info = sae(x)
                loss_dict = sae.loss(x, recon, sparse, info)
                loss = loss_dict['total_loss']

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # Normalize decoder columns
                if hasattr(sae, 'normalize_decoder_'):
                    sae.normalize_decoder_()

                epoch_loss += loss.item()
                epoch_recon_loss += loss_dict.get('reconstruction_loss', loss_dict['total_loss']).item()
                if 'sparsity_loss' in loss_dict:
                    epoch_aux_loss += loss_dict['sparsity_loss'].item()
                n_batches += 1
                epoch_samples += batch_size
                global_step += 1

                # Periodic checkpoint saving
                if global_step >= next_checkpoint_step:
                    save_checkpoint(global_step)
                    next_checkpoint_step += checkpoint_every

                # Resampling (if enabled and before stop point)
                if resampling and global_step % resampling_interval == 0:
                    if global_step < resampling_stop_step:
                        if hasattr(sae, 'resample_dead_neurons'):
                            n_resampled = sae.resample_dead_neurons(
                                activations=x,
                                optimizer=optimizer,
                            )
                            if n_resampled > 0:
                                logger.info(f"  Step {global_step}: resampled {n_resampled} dead neurons")

            # Early stop if we've hit target
            if epoch_samples >= n_samples:
                break

        # Handle remaining buffer
        if activation_buffer and len(activation_buffer) > 0:
            remaining = torch.cat(activation_buffer, dim=0)
            if remaining.shape[0] >= 32:  # Only if enough for a mini-batch
                x = remaining.to(device)
                recon, sparse, info = sae(x)
                loss_dict = sae.loss(x, recon, sparse, info)
                loss = loss_dict['total_loss']

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                if hasattr(sae, 'normalize_decoder_'):
                    sae.normalize_decoder_()

                epoch_loss += loss.item()
                epoch_recon_loss += loss_dict.get('reconstruction_loss', loss_dict['total_loss']).item()
                if 'sparsity_loss' in loss_dict:
                    epoch_aux_loss += loss_dict['sparsity_loss'].item()
                n_batches += 1


        if n_batches == 0:
            logger.warning(f"Epoch {epoch+1}: No batches processed!")
            continue

        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon_loss / n_batches
        avg_aux = epoch_aux_loss / n_batches if epoch_aux_loss > 0 else 0

        total_samples_seen += epoch_samples

        # Track best
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in sae.state_dict().items()}

        # Logging
        logger.info(
            f"  Epoch {epoch+1:3d}/{n_epochs}: "
            f"loss={avg_loss:.4f} (recon={avg_recon:.4f}, aux={avg_aux:.4f}), "
            f"samples={epoch_samples:,}, lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # W&B logging
        if wandb_run is not None:
            wandb_run.log({
                'epoch': epoch + 1,
                'train/loss': avg_loss,
                'train/recon_loss': avg_recon,
                'train/aux_loss': avg_aux,
                'train/lr': scheduler.get_last_lr()[0],
                'train/samples_this_epoch': epoch_samples,
                'train/total_samples': total_samples_seen,
            })

    # Load best state
    if best_state is not None:
        sae.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        logger.info(f"\nLoaded best checkpoint (loss={best_loss:.4f})")

    sae.eval()

    # Save final checkpoint
    save_checkpoint(global_step, is_final=True)

    logger.info(f"\nTraining complete!")
    logger.info(f"  Total steps: {global_step}")
    logger.info(f"  Total samples: {total_samples_seen:,}")
    logger.info(f"  Best loss: {best_loss:.4f}")
    if checkpoint_dir:
        logger.info(f"  Checkpoints saved to: {checkpoint_dir}/")
        logger.info(f"  Use evaluate_sae_checkpoints.py for Pareto frontier evaluation")

    return sae, checkpoint_dir


def main():
    parser = argparse.ArgumentParser(description="Train high-quality SAE on Pythia")
    
    parser.add_argument('--model', type=str, default='EleutherAI/pythia-70m')
    parser.add_argument('--layer', type=int, default=3)
    parser.add_argument('--expansion', type=int, default=8)
    parser.add_argument('--k', type=int, default=32, help='TopK sparsity')
    parser.add_argument('--samples', type=int, default=10000000, help='Number of token activations')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--output', type=str, default=None, help='Output checkpoint dir (will save periodic checkpoints)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Load existing checkpoint')
    parser.add_argument('--diagnose-only', action='store_true', help='Only run diagnostics')
    parser.add_argument('--quick', action='store_true', help='Quick test (10k samples, 10 epochs)')
    parser.add_argument('--dataset', type=str, default='wikitext', help='Dataset name (wikitext, pile, etc.)')
    parser.add_argument('--device', type=str, default='cuda')
    # Resampling options
    parser.add_argument('--resampling', action='store_true', help='Enable dead neuron resampling (aux-k)')
    parser.add_argument('--resampling-interval', type=int, default=500, help='Steps between resampling attempts')
    parser.add_argument('--resampling-stop', type=float, default=0.5, help='Stop resampling after this fraction of training')
    # Checkpoint options
    parser.add_argument('--checkpoint-interval', type=int, default=10, help='Save checkpoint every N%% of training')
    # W&B options
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
    if args.diagnose_only:
        if not args.checkpoint:
            logger.error("--diagnose-only requires --checkpoint")
            return
        logger.info("For diagnostics, use evaluate_sae_checkpoints.py instead:")
        logger.info(f"  python scripts/evaluate_sae_checkpoints.py \\")
        logger.info(f"    --checkpoint-dir {Path(args.checkpoint).parent} \\")
        logger.info(f"    --model {args.model} --layer {args.layer}")
        return

    if args.checkpoint and Path(args.checkpoint).exists():
        logger.info(f"Resuming from checkpoint: {args.checkpoint}")
        # TODO: implement checkpoint resumption if needed
        logger.warning("Checkpoint resumption not yet implemented, starting fresh")

    # Train new SAE
    sae, checkpoint_dir = train_sae(
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
        resampling=args.resampling,
        resampling_interval=args.resampling_interval,
        resampling_stop=args.resampling_stop,
        checkpoint_interval=args.checkpoint_interval,
    )

    # Finish W&B run
    if wandb_run is not None:
        wandb_run.finish()
        logger.info("W&B run finished")

    # Print next steps
    if checkpoint_dir:
        logger.info(f"\nNext: Run Pareto frontier evaluation:")
        logger.info(f"  python scripts/evaluate_sae_checkpoints.py \\")
        logger.info(f"    --checkpoint-dir {checkpoint_dir} \\")
        logger.info(f"    --model {args.model} --layer {args.layer}")


if __name__ == "__main__":
    main()