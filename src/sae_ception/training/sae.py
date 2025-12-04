"""SAE training on frozen model activations."""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from pathlib import Path
from omegaconf import DictConfig
from tqdm import tqdm
import wandb

from ..models.sae import create_sae, SparseAutoencoder, TopKSparseAutoencoder
from ..utils.data import (
    create_dataloaders, 
    extract_activations_from_model,
    extract_activations_all_tokens,
    extract_activations_final_token,
    create_activation_dataloader,
)
from ..utils.logger import get_logger
from ..utils.checkpointing import save_checkpoint
from ..training.baseline import load_baseline_model
from transformers import AutoTokenizer

logger = get_logger(__name__)


def load_sae(
    cfg: DictConfig,
    cycle: int,
    checkpoint_path: str = None,
) -> nn.Module:
    """
    Load a trained SAE from checkpoint.
    
    Args:
        cfg: Configuration
        cycle: Cycle number
        checkpoint_path: Path to checkpoint (default: sae_cycle_X_best.pt)
        
    Returns:
        Loaded SAE
    """
    if checkpoint_path is None:
        checkpoint_path = Path(cfg.checkpoint_dir) / f"sae_cycle_{cycle}_best.pt"
    
    # Determine dimensions
    if hasattr(cfg.model, 'hidden_size'):
        activation_dim = cfg.model.hidden_size
    else:
        raise ValueError("Cannot determine activation dimension from config")
    
    hidden_dim = activation_dim * cfg.sae.expansion_factor
    
    # Determine SAE type
    sae_type = getattr(cfg.sae, 'sae_type', 'l1')
    
    # Create SAE using factory function
    sae = create_sae(
        input_dim=activation_dim,
        hidden_dim=hidden_dim,
        sae_type=sae_type,
        l1_penalty=cfg.sae.l1_penalty,
        k=getattr(cfg.sae, 'k', 50),
        aux_k_coef=getattr(cfg.sae, 'aux_k_coef', 1/32),
        dead_steps_threshold=getattr(cfg.sae, 'dead_steps_threshold', 100),
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
    sae.load_state_dict(checkpoint['model_state_dict'])
    
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    sae.to(device)
    sae.eval()
    
    logger.info(f"Loaded {sae_type.upper()} SAE from {checkpoint_path}")
    if 'val_l0_norm' in checkpoint.get('metrics', {}):
        logger.info(f"Checkpoint val L0 norm: {checkpoint['metrics']['val_l0_norm']:.2f}")
    if 'val_dead_pct' in checkpoint.get('metrics', {}):
        logger.info(f"Checkpoint val dead features: {checkpoint['metrics']['val_dead_pct']:.1f}%")
    
    return sae

def compute_geometric_median(x: torch.Tensor, num_iters: int = 10) -> torch.Tensor:
    """
    Compute approximate geometric median using Weiszfeld's algorithm.
    
    This is used to initialize the pre_bias, which helps center the data
    in a way that's robust to outliers.
    """
    # Start with the mean as initial estimate
    median = x.mean(dim=0)
    
    for _ in range(num_iters):
        # Compute distances from current estimate
        dists = torch.norm(x - median, dim=1, keepdim=True)
        # Avoid division by zero
        dists = torch.clamp(dists, min=1e-8)
        # Weighted average (inverse distance weighting)
        weights = 1.0 / dists
        median = (x * weights).sum(dim=0) / weights.sum()
    
    return median


def init_sae_from_data_(
    sae: nn.Module, 
    activations: torch.Tensor,
    device: str,
) -> None:
    """
    Initialize SAE parameters from data statistics (OpenAI's approach).
    
    This helps prevent dead features by:
    1. Setting pre_bias to geometric median (centers data robustly)
    2. Scaling encoder so initial reconstructions have reasonable norm
    """
    logger.info("Initializing SAE from data statistics...")
    
    # Use subset for efficiency
    sample = activations[:min(32768, len(activations))].to(device)
    
    # 1. Initialize pre_bias to geometric median
    with torch.no_grad():
        geometric_med = compute_geometric_median(sample.float())
        sae.pre_bias.data = geometric_med.to(sae.pre_bias.dtype)
    
    logger.info(f"  Pre-bias initialized (norm: {sae.pre_bias.norm().item():.4f})")
    
    # 2. Scale encoder based on initial reconstruction norm
    # This ensures activations start in a reasonable range
    if hasattr(sae, 'encoder'):
        with torch.no_grad():
            # adaptation from OpenAI's implementation that uses random activations
            batch_size = min(512, len(sample))
            x = sample[:batch_size]
            x = x / x.norm(dim=-1, keepdim=True)
            x = x + sae.pre_bias.data
            
            # Get initial reconstruction
            recons, _, _ = sae(x)
            recons_centered = recons - sae.pre_bias.data
            recons_norm = recons_centered.norm(dim=-1).mean()
            
            # Scale encoder to normalize reconstruction magnitude
            if recons_norm > 1e-6:
                sae.encoder.weight.data /= recons_norm.item()
                logger.info(f"  Encoder scaled by 1/{recons_norm.item():.4f}")


def train_sae_epoch(
    sae: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    clip_grad: float = None,
) -> dict:
    """Train SAE for one epoch."""
    sae.train()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_sparsity_loss = 0.0
    total_l0 = 0.0
    total_dead_pct = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training SAE")
    
    for batch in progress_bar:
        activations = batch['activations'].to(device)
        
        # Forward pass
        reconstruction, sparse_code, info = sae(activations)
        
        # Compute loss
        loss_dict = sae.loss(activations, reconstruction, sparse_code, info=info)
        loss = loss_dict['total_loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Optional gradient clipping (OpenAI uses this)
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(sae.parameters(), clip_grad)
        
        optimizer.step()
        
        # Normalize decoder after each step (important for TopK!)
        if hasattr(sae, 'normalize_decoder_'):
            sae.normalize_decoder_()
        
        # Track metrics
        total_loss += loss_dict['total_loss'].item()
        total_recon_loss += loss_dict['reconstruction_loss'].item()
        sparsity = loss_dict.get('sparsity_loss', loss_dict.get('aux_loss', 0.0))
        if isinstance(sparsity, torch.Tensor):
            sparsity = sparsity.item()
        total_sparsity_loss += sparsity
        total_l0 += loss_dict['l0_norm'].item()
        
        # Track dead features for TopK
        if 'dead_pct' in loss_dict:
            dead_pct = loss_dict['dead_pct']
            if isinstance(dead_pct, torch.Tensor):
                dead_pct = dead_pct.item()
            total_dead_pct += dead_pct
        
        num_batches += 1
        
        # Update progress bar
        postfix = {
            'loss': loss.item(),
            'l0': loss_dict['l0_norm'].item(),
        }
        if 'dead_pct' in loss_dict:
            dead_val = loss_dict['dead_pct']
            if isinstance(dead_val, torch.Tensor):
                dead_val = dead_val.item()
            postfix['dead%'] = f"{dead_val:.1f}"
        progress_bar.set_postfix(postfix)
    
    metrics = {
        'loss': total_loss / num_batches,
        'reconstruction_loss': total_recon_loss / num_batches,
        'sparsity_loss': total_sparsity_loss / num_batches,
        'l0_norm': total_l0 / num_batches,
    }
    
    if total_dead_pct > 0:
        metrics['dead_pct'] = total_dead_pct / num_batches
    
    return metrics


def evaluate_sae(
    sae: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> dict:
    """Evaluate SAE on validation set."""
    sae.eval()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_sparsity_loss = 0.0
    total_l0 = 0.0
    total_dead_pct = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating SAE"):
            activations = batch['activations'].to(device)
            
            # Forward pass
            reconstruction, sparse_code, info = sae(activations)
            
            # Compute loss
            loss_dict = sae.loss(activations, reconstruction, sparse_code, info=info)
            
            # Track metrics
            total_loss += loss_dict['total_loss'].item()
            total_recon_loss += loss_dict['reconstruction_loss'].item()
            
            sparsity = loss_dict.get('sparsity_loss', loss_dict.get('aux_loss', 0.0))
            if isinstance(sparsity, torch.Tensor):
                sparsity = sparsity.item()
            total_sparsity_loss += sparsity
            
            total_l0 += loss_dict['l0_norm'].item()
            
            if 'dead_pct' in loss_dict:
                dead_pct = loss_dict['dead_pct']
                if isinstance(dead_pct, torch.Tensor):
                    dead_pct = dead_pct.item()
                total_dead_pct += dead_pct
            
            num_batches += 1
    
    metrics = {
        'loss': total_loss / num_batches,
        'reconstruction_loss': total_recon_loss / num_batches,
        'sparsity_loss': total_sparsity_loss / num_batches,
        'l0_norm': total_l0 / num_batches,
    }
    
    if total_dead_pct > 0:
        metrics['dead_pct'] = total_dead_pct / num_batches
    
    return metrics


def train_sae(
    cfg: DictConfig,
    model: nn.Module = None,
    cycle: int = 0,
    baseline_checkpoint: str = None,
) -> tuple[nn.Module, DataLoader, DataLoader]:
    """
    Train Sparse Autoencoder on model activations.
    """
    # Determine SAE type
    sae_type = getattr(cfg.sae, 'sae_type', 'l1')
    
    logger.info("=" * 60)
    logger.info(f"Training SAE (Cycle {cycle})")
    logger.info(f"SAE Type: {sae_type.upper()}")
    logger.info("=" * 60)
    
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load model if not provided
    if model is None:
        logger.info(f"Loading baseline model from cycle {cycle}")
        if baseline_checkpoint:
            logger.info(f"Using checkpoint: {baseline_checkpoint}")
            model = load_baseline_model(cfg, checkpoint_path=baseline_checkpoint)
        else:
            checkpoint_path = None
            
            if hasattr(cfg, 'checkpoint_dir'):
                potential_path = Path(cfg.checkpoint_dir) / f"model_cycle_{cycle}_best.pt"
                if potential_path.exists():
                    checkpoint_path = potential_path
            
            if checkpoint_path is None:
                outputs_dir = Path.cwd().parent.parent if 'outputs' in str(Path.cwd()) else Path('outputs')
                logger.info(f"Searching for baseline checkpoint in: {outputs_dir}")
                
                checkpoint_pattern = f"**/checkpoints/model_cycle_{cycle}_best.pt"
                checkpoints = sorted(outputs_dir.glob(checkpoint_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
                
                if checkpoints:
                    checkpoint_path = checkpoints[0]
                    logger.info(f"Found checkpoint: {checkpoint_path}")
                else:
                    raise FileNotFoundError(
                        f"No baseline checkpoint found for cycle {cycle}. "
                        f"Please train baseline first with: python scripts/train_baseline.py\n"
                        f"Or specify checkpoint path with: baseline_checkpoint=/path/to/checkpoint.pt"
                    )
            
            model = load_baseline_model(cfg, checkpoint_path=str(checkpoint_path))
    
    model.eval()
    model.to(device)
    
    # Load tokenizer for data loading
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.hf_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataloaders for text data
    logger.info("Loading dataset...")
    dataloaders = create_dataloaders(cfg, tokenizer)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    # Extract activations based on training mode
    logger.info(f"Extracting activations from layer: {cfg.model.target_layer}")
    
    # Check token_level setting (NEW - default to True for SAEBench compatibility)
    token_level = getattr(cfg.sae, 'token_level', True)
    
    if token_level:
        # NEW: Token-level extraction for SAEBench compatibility
        logger.info("Extracting ALL token activations (SAEBench compatible)...")
        
        train_acts, train_token_ids, train_labels = extract_activations_all_tokens(
            model=model,
            dataloader=train_loader,
            layer_name=str(cfg.model.target_layer),
            device=device,
        )
        
        val_acts, val_token_ids, val_labels = extract_activations_all_tokens(
            model=model,
            dataloader=val_loader,
            layer_name=str(cfg.model.target_layer),
            device=device,
        )
        
        logger.info(f"Train tokens: {train_acts.shape[0]:,} (from {len(train_loader.dataset)} sequences)")
        logger.info(f"Val tokens: {val_acts.shape[0]:,} (from {len(val_loader.dataset)} sequences)")
        
        # Store token_ids for SAEBench evaluation later
        checkpoint_dir = Path(cfg.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'train_token_ids': train_token_ids,
            'val_token_ids': val_token_ids,
            'train_labels': train_labels,
            'val_labels': val_labels,
        }, checkpoint_dir / f"token_metadata_cycle_{cycle}.pt")
        logger.info(f"Saved token metadata for SAEBench evaluation")
        
    else:
        # Legacy: Sequence-level extraction (final token only)
        logger.info("Extracting FINAL token activations only...")
        
        train_acts, train_labels = extract_activations_final_token(
            model=model,
            dataloader=train_loader,
            layer_name=str(cfg.model.target_layer),
            device=device,
        )
        
        val_acts, val_labels = extract_activations_final_token(
            model=model,
            dataloader=val_loader,
            layer_name=str(cfg.model.target_layer),
            device=device,
        )
        
        train_token_ids = None
        val_token_ids = None
        
        logger.info(f"Train activations: {train_acts.shape}")
        logger.info(f"Val activations: {val_acts.shape}")
    
    activation_dim = train_acts.shape[1]
    logger.info(f"Activation dimension: {activation_dim}")
    
    # Create SAE using factory function
    hidden_dim = activation_dim * cfg.sae.expansion_factor
    
    if sae_type == "topk":
        k = getattr(cfg.sae, 'k', 50)
        aux_k_coef = getattr(cfg.sae, 'aux_k_coef', 1/32)
        dead_steps_threshold = getattr(cfg.sae, 'dead_steps_threshold', 100)
        logger.info(f"Creating TopK SAE: {activation_dim} -> {hidden_dim} -> {activation_dim}")
        logger.info(f"  k={k} (target L0 sparsity)")
        logger.info(f"  aux_k_coef={aux_k_coef}")
        logger.info(f"  dead_steps_threshold={dead_steps_threshold}")
    else:
        logger.info(f"Creating L1 SAE: {activation_dim} -> {hidden_dim} -> {activation_dim}")
        logger.info(f"  L1 penalty: {cfg.sae.l1_penalty}")
    
    sae = create_sae(
        input_dim=activation_dim,
        hidden_dim=hidden_dim,
        sae_type=sae_type,
        l1_penalty=cfg.sae.l1_penalty,
        k=getattr(cfg.sae, 'k', 50),
        aux_k_coef=getattr(cfg.sae, 'aux_k_coef', 1/32),
        dead_steps_threshold=getattr(cfg.sae, 'dead_steps_threshold', 100),
    )
    sae.to(device)
    
    # Initialize from data (important for TopK!)
    if sae_type == "topk":
        init_sae_from_data_(sae, train_acts, device)
    
    # Count parameters
    total_params = sum(p.numel() for p in sae.parameters())
    logger.info(f"SAE parameters: {total_params:,}")
    
    # Create activation dataloaders
    train_act_loader = create_activation_dataloader(
        train_acts, train_labels,
        token_ids=train_token_ids,  # NEW
        batch_size=cfg.sae.batch_size,
        shuffle=True,
    )
    
    val_act_loader = create_activation_dataloader(
        val_acts, val_labels,
        token_ids=val_token_ids,  # NEW
        batch_size=cfg.sae.batch_size,
        shuffle=False,
    )
    
    # Setup optimizer
    optimizer = Adam(
        sae.parameters(),
        lr=cfg.sae.learning_rate,
        weight_decay=cfg.sae.weight_decay,
        eps=getattr(cfg.sae, 'adam_eps', 1e-8),  # OpenAI uses 6.25e-10
    )
    
    # Get gradient clipping value
    clip_grad = getattr(cfg.sae, 'clip_grad', None)
    
    # Initialize wandb
    if cfg.wandb.mode != 'disabled':
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"sae_{sae_type}_cycle_{cycle}_{cfg.dataset.name}",
            config=dict(cfg),
            tags=['sae', sae_type, f'cycle_{cycle}'],
        )
    
    # Training loop
    logger.info(f"Training SAE for {cfg.sae.epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(cfg.sae.epochs):
        logger.info(f"Epoch {epoch + 1}/{cfg.sae.epochs}")
        
        # Train
        train_metrics = train_sae_epoch(
            sae=sae,
            dataloader=train_act_loader,
            optimizer=optimizer,
            device=device,
            clip_grad=clip_grad,
        )
        
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
        logger.info(f"Train Reconstruction Loss: {train_metrics['reconstruction_loss']:.4f}")
        if sae_type == "l1":
            logger.info(f"Train Sparsity Loss: {train_metrics['sparsity_loss']:.4f}")
        elif sae_type == "topk" and 'dead_pct' in train_metrics:
            logger.info(f"Train Aux Loss: {train_metrics['sparsity_loss']:.4f}")
            logger.info(f"Train Dead Features: {train_metrics['dead_pct']:.1f}%")
        logger.info(f"Train L0 Norm: {train_metrics['l0_norm']:.2f}")
        
        # Evaluate
        logger.info("Evaluating SAE...")
        val_metrics = evaluate_sae(sae, val_act_loader, device)
        
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"Val Reconstruction Loss: {val_metrics['reconstruction_loss']:.4f}")
        logger.info(f"Val L0 Norm: {val_metrics['l0_norm']:.2f}")
        if 'dead_pct' in val_metrics:
            logger.info(f"Val Dead Features: {val_metrics['dead_pct']:.1f}%")
        
        # Log to wandb
        if cfg.wandb.mode != 'disabled':
            log_dict = {
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'train/reconstruction_loss': train_metrics['reconstruction_loss'],
                'train/sparsity_loss': train_metrics['sparsity_loss'],
                'train/l0_norm': train_metrics['l0_norm'],
                'val/loss': val_metrics['loss'],
                'val/reconstruction_loss': val_metrics['reconstruction_loss'],
                'val/l0_norm': val_metrics['l0_norm'],
            }
            
            # Add dead feature tracking for TopK
            if 'dead_pct' in train_metrics:
                log_dict['train/dead_pct'] = train_metrics['dead_pct']
            if 'dead_pct' in val_metrics:
                log_dict['val/dead_pct'] = val_metrics['dead_pct']
            
            wandb.log(log_dict)
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            logger.info(f"New best validation loss: {best_val_loss:.4f}")
            
            checkpoint_dir = Path(cfg.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = checkpoint_dir / f"sae_cycle_{cycle}_best.pt"
            
            metrics_to_save = {
                'val_loss': val_metrics['loss'],
                'val_l0_norm': val_metrics['l0_norm'],
            }
            if 'dead_pct' in val_metrics:
                metrics_to_save['val_dead_pct'] = val_metrics['dead_pct']
            
            save_checkpoint(
                path=checkpoint_path,
                model=sae,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics_to_save,
            )
    
    # Load best model
    logger.info(f"Loading best SAE (val_loss: {best_val_loss:.4f})")
    checkpoint_path = Path(cfg.checkpoint_dir) / f"sae_cycle_{cycle}_best.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        sae.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    logger.info("Final evaluation on validation set:")
    final_metrics = evaluate_sae(sae, val_act_loader, device)
    
    logger.info(f"Final Val Loss: {final_metrics['loss']:.4f}")
    logger.info(f"Final Val Reconstruction Loss: {final_metrics['reconstruction_loss']:.4f}")
    logger.info(f"Final Val L0 Norm: {final_metrics['l0_norm']:.2f}")
    if 'dead_pct' in final_metrics:
        logger.info(f"Final Val Dead Features: {final_metrics['dead_pct']:.1f}%")
    
    if cfg.wandb.mode != 'disabled':
        final_log = {
            'final/val_loss': final_metrics['loss'],
            'final/val_reconstruction_loss': final_metrics['reconstruction_loss'],
            'final/val_l0_norm': final_metrics['l0_norm'],
        }
        if 'dead_pct' in final_metrics:
            final_log['final/val_dead_pct'] = final_metrics['dead_pct']
        wandb.log(final_log)
        wandb.finish()
    
    logger.info("SAE training complete!")
    
    return sae, train_act_loader, val_act_loader