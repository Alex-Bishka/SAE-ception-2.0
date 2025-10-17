"""Utilities for saving and loading checkpoints."""

import torch
from pathlib import Path
from typing import Dict, Optional


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict] = None,
    **kwargs,
):
    """
    Save a checkpoint.
    
    Args:
        path: Path to save checkpoint
        model: Model to save
        optimizer: Optional optimizer state
        scheduler: Optional scheduler state
        epoch: Current epoch
        metrics: Dictionary of metrics
        **kwargs: Additional items to save
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if metrics is not None:
        checkpoint.update(metrics)
    
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = 'cuda',
) -> Dict:
    """
    Load a checkpoint.
    
    Args:
        path: Path to checkpoint
        model: Model to load into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load checkpoint on
        
    Returns:
        Dictionary with checkpoint contents
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Loaded checkpoint from {path}")
    
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    
    # Print any metrics in checkpoint
    metric_keys = [k for k in checkpoint.keys() 
                   if k not in ['model_state_dict', 'optimizer_state_dict', 
                               'scheduler_state_dict', 'epoch']]
    if metric_keys:
        print("  Metrics:")
        for key in metric_keys:
            print(f"    {key}: {checkpoint[key]}")
    
    return checkpoint