"""SAE training on frozen model activations."""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from pathlib import Path
from omegaconf import DictConfig
from tqdm import tqdm
import wandb

from ..models.sae import SparseAutoencoder
from ..utils.data import create_dataloaders, extract_activations_from_model, create_activation_dataloader
from ..utils.logger import get_logger
from ..utils.checkpointing import save_checkpoint
from ..training.baseline import load_baseline_model
from transformers import AutoTokenizer

logger = get_logger(__name__)


def train_sae_epoch(
    sae: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> dict:
    """Train SAE for one epoch."""
    sae.train()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_sparsity_loss = 0.0
    total_l0 = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training SAE")
    
    for batch in progress_bar:
        activations = batch['activations'].to(device)
        
        # Forward pass
        reconstruction, sparse_code = sae(activations)
        
        # Compute loss
        loss_dict = sae.loss(activations, reconstruction, sparse_code)
        loss = loss_dict['total_loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss_dict['total_loss'].item()
        total_recon_loss += loss_dict['reconstruction_loss'].item()
        total_sparsity_loss += loss_dict['sparsity_loss'].item()
        total_l0 += loss_dict['l0_norm'].item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(),
            'l0': loss_dict['l0_norm'].item(),
        })
    
    return {
        'loss': total_loss / num_batches,
        'reconstruction_loss': total_recon_loss / num_batches,
        'sparsity_loss': total_sparsity_loss / num_batches,
        'l0_norm': total_l0 / num_batches,
    }


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
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating SAE"):
            activations = batch['activations'].to(device)
            
            # Forward pass
            reconstruction, sparse_code = sae(activations)
            
            # Compute loss
            loss_dict = sae.loss(activations, reconstruction, sparse_code)
            
            # Track metrics
            total_loss += loss_dict['total_loss'].item()
            total_recon_loss += loss_dict['reconstruction_loss'].item()
            total_sparsity_loss += loss_dict['sparsity_loss'].item()
            total_l0 += loss_dict['l0_norm'].item()
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'reconstruction_loss': total_recon_loss / num_batches,
        'sparsity_loss': total_sparsity_loss / num_batches,
        'l0_norm': total_l0 / num_batches,
    }


def train_sae(
    cfg: DictConfig,
    model: nn.Module = None,
    cycle: int = 0,
    baseline_checkpoint: str = None,
) -> tuple[SparseAutoencoder, DataLoader, DataLoader]:
    """
    Train Sparse Autoencoder on model activations.
    
    Args:
        cfg: Hydra configuration
        model: Pre-trained model (if None, loads from checkpoint)
        cycle: Current SAE-ception cycle
        baseline_checkpoint: Path to baseline model checkpoint (optional)
        
    Returns:
        Tuple of (trained_sae, train_activation_loader, val_activation_loader)
    """
    logger.info("=" * 60)
    logger.info(f"Training SAE (Cycle {cycle})")
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
            # Try to find checkpoint in current or parent directories
            checkpoint_path = None
            
            # First, try the configured checkpoint_dir
            if hasattr(cfg, 'checkpoint_dir'):
                potential_path = Path(cfg.checkpoint_dir) / f"model_cycle_{cycle}_best.pt"
                if potential_path.exists():
                    checkpoint_path = potential_path
            
            # If not found, look in parent outputs directory
            if checkpoint_path is None:
                outputs_dir = Path.cwd().parent.parent if 'outputs' in str(Path.cwd()) else Path('outputs')
                logger.info(f"Searching for baseline checkpoint in: {outputs_dir}")
                
                # Find most recent baseline checkpoint
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
    
    # Extract activations from target layer
    logger.info(f"Extracting activations from layer: {cfg.model.target_layer}")
    logger.info("Extracting training activations...")
    train_acts, train_labels = extract_activations_from_model(
        model=model,
        dataloader=train_loader,
        layer_name=str(cfg.model.target_layer),
        device=device,
        token_position='last',  # Use last token for classification
    )
    
    logger.info("Extracting validation activations...")
    val_acts, val_labels = extract_activations_from_model(
        model=model,
        dataloader=val_loader,
        layer_name=str(cfg.model.target_layer),
        device=device,
        token_position='last',
    )
    
    activation_dim = train_acts.shape[1]
    logger.info(f"Activation dimension: {activation_dim}")
    logger.info(f"Train activations: {train_acts.shape}")
    logger.info(f"Val activations: {val_acts.shape}")
    
    # Create SAE
    hidden_dim = activation_dim * cfg.sae.expansion_factor
    logger.info(f"Creating SAE: {activation_dim} -> {hidden_dim} -> {activation_dim}")
    
    sae = SparseAutoencoder(
        input_dim=activation_dim,
        hidden_dim=hidden_dim,
        l1_penalty=cfg.sae.l1_penalty,
    )
    sae.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in sae.parameters())
    logger.info(f"SAE parameters: {total_params:,}")
    
    # Create activation dataloaders
    train_act_loader = create_activation_dataloader(
        train_acts, train_labels,
        batch_size=cfg.sae.batch_size,
        shuffle=True,
    )
    
    val_act_loader = create_activation_dataloader(
        val_acts, val_labels,
        batch_size=cfg.sae.batch_size,
        shuffle=False,
    )
    
    # Setup optimizer
    optimizer = Adam(
        sae.parameters(),
        lr=cfg.sae.learning_rate,
        weight_decay=cfg.sae.weight_decay,
    )
    
    # Initialize wandb
    if cfg.wandb.mode != 'disabled':
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"sae_cycle_{cycle}_{cfg.dataset.name}",
            config=dict(cfg),
            tags=['sae', f'cycle_{cycle}'],
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
        )
        
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
        logger.info(f"Train Reconstruction Loss: {train_metrics['reconstruction_loss']:.4f}")
        logger.info(f"Train Sparsity Loss: {train_metrics['sparsity_loss']:.4f}")
        logger.info(f"Train L0 Norm: {train_metrics['l0_norm']:.2f}")
        
        # Evaluate
        logger.info("Evaluating SAE...")
        val_metrics = evaluate_sae(sae, val_act_loader, device)
        
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"Val Reconstruction Loss: {val_metrics['reconstruction_loss']:.4f}")
        logger.info(f"Val L0 Norm: {val_metrics['l0_norm']:.2f}")
        
        # Log to wandb
        if cfg.wandb.mode != 'disabled':
            wandb.log({
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'train/reconstruction_loss': train_metrics['reconstruction_loss'],
                'train/sparsity_loss': train_metrics['sparsity_loss'],
                'train/l0_norm': train_metrics['l0_norm'],
                'val/loss': val_metrics['loss'],
                'val/reconstruction_loss': val_metrics['reconstruction_loss'],
                'val/l0_norm': val_metrics['l0_norm'],
            })
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            logger.info(f"New best validation loss: {best_val_loss:.4f}")
            
            # Save checkpoint
            checkpoint_dir = Path(cfg.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = checkpoint_dir / f"sae_cycle_{cycle}_best.pt"
            save_checkpoint(
                path=checkpoint_path,
                model=sae,
                optimizer=optimizer,
                epoch=epoch,
                metrics={
                    'val_loss': val_metrics['loss'],
                    'val_l0_norm': val_metrics['l0_norm'],
                },
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
    
    if cfg.wandb.mode != 'disabled':
        wandb.log({
            'final/val_loss': final_metrics['loss'],
            'final/val_reconstruction_loss': final_metrics['reconstruction_loss'],
            'final/val_l0_norm': final_metrics['l0_norm'],
        })
        wandb.finish()
    
    logger.info("SAE training complete!")
    
    return sae, train_act_loader, val_act_loader


def load_sae(
    cfg: DictConfig,
    cycle: int,
    checkpoint_path: str = None,
) -> SparseAutoencoder:
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
    # We need to know the activation dimension - get from model config
    if hasattr(cfg.model, 'hidden_size'):
        activation_dim = cfg.model.hidden_size
    else:
        raise ValueError("Cannot determine activation dimension from config")
    
    hidden_dim = activation_dim * cfg.sae.expansion_factor
    
    # Create SAE
    sae = SparseAutoencoder(
        input_dim=activation_dim,
        hidden_dim=hidden_dim,
        l1_penalty=cfg.sae.l1_penalty,
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
    sae.load_state_dict(checkpoint['model_state_dict'])
    
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    sae.to(device)
    sae.eval()
    
    logger.info(f"Loaded SAE from {checkpoint_path}")
    if 'val_l0_norm' in checkpoint:
        logger.info(f"Checkpoint val L0 norm: {checkpoint['val_l0_norm']:.2f}")
    
    return sae