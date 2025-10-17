"""Baseline model training (Cycle 0) - Fine-tune pre-trained model on classification task."""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig
import wandb

from ..utils.data import create_dataloaders
from ..utils.logger import get_logger
from ..evaluation import evaluate_classification_accuracy

logger = get_logger(__name__)


class ClassificationModel(nn.Module):
    """Wrapper for pre-trained model with classification head."""
    
    def __init__(self, base_model, num_classes: int):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    scheduler,
    device: str,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    optimizer.zero_grad()
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get logits
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        # Compute loss
        loss = nn.functional.cross_entropy(logits, labels)
        
        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights
        if (step + 1) % gradient_accumulation_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Track metrics
        total_loss += loss.item() * gradient_accumulation_steps
        
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item() * gradient_accumulation_steps,
            'acc': total_correct / total_samples,
        })
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': total_correct / total_samples,
    }


def train_baseline_model(cfg: DictConfig) -> nn.Module:
    """
    Train baseline classification model (Cycle 0).
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        Trained model
    """
    logger.info("=" * 60)
    logger.info("Training Baseline Model (Cycle 0)")
    logger.info("=" * 60)
    
    # Set device
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {cfg.model.hf_path}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.hf_path)
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataloaders
    logger.info("Loading dataset...")
    dataloaders = create_dataloaders(cfg, tokenizer)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    
    # Load pre-trained model with classification head
    logger.info(f"Loading model: {cfg.model.hf_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.hf_path,
        num_labels=cfg.dataset.num_classes,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
    )
    
    # Calculate total training steps
    total_steps = len(train_loader) * cfg.model.epochs_per_cycle // cfg.model.gradient_accumulation_steps
    
    # Setup learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.model.warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Initialize wandb
    if cfg.wandb.mode != 'disabled':
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"baseline_{cfg.dataset.name}",
            config=dict(cfg),
            tags=['baseline', 'cycle_0'],
        )
    
    # Training loop
    logger.info(f"Training for {cfg.model.epochs_per_cycle} epochs...")
    best_val_acc = 0.0
    
    for epoch in range(cfg.model.epochs_per_cycle):
        logger.info(f"Epoch {epoch + 1}/{cfg.model.epochs_per_cycle}")
        
        # Train
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            gradient_accumulation_steps=cfg.model.gradient_accumulation_steps,
            max_grad_norm=cfg.model.max_grad_norm,
        )
        
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
        logger.info(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
        
        # Evaluate
        logger.info("Evaluating...")
        val_metrics = evaluate_classification_accuracy(model, val_loader, device)
        
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"Val F1 (macro): {val_metrics['f1_macro']:.4f}")
        
        # Log to wandb
        if cfg.wandb.mode != 'disabled':
            wandb.log({
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'train/accuracy': train_metrics['accuracy'],
                'val/loss': val_metrics['loss'],
                'val/accuracy': val_metrics['accuracy'],
                'val/f1_macro': val_metrics['f1_macro'],
                'learning_rate': scheduler.get_last_lr()[0],
            })
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            logger.info(f"New best validation accuracy: {best_val_acc:.4f}")
            
            # Save checkpoint
            checkpoint_dir = Path(cfg.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = checkpoint_dir / "model_cycle_0_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_accuracy': val_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
            }, checkpoint_path)
            
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Load best model
    logger.info(f"Loading best model (val_acc: {best_val_acc:.4f})")
    checkpoint_path = Path(cfg.checkpoint_dir) / "model_cycle_0_best.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    logger.info("Final evaluation on validation set:")
    final_metrics = evaluate_classification_accuracy(model, val_loader, device)
    
    logger.info(f"Final Val Accuracy: {final_metrics['accuracy']:.4f}")
    logger.info(f"Final Val F1 (macro): {final_metrics['f1_macro']:.4f}")
    
    if cfg.wandb.mode != 'disabled':
        wandb.log({
            'final/val_accuracy': final_metrics['accuracy'],
            'final/val_f1_macro': final_metrics['f1_macro'],
        })
        wandb.finish()
    
    return model


def load_baseline_model(
    cfg: DictConfig,
    checkpoint_path: str = None,
) -> nn.Module:
    """
    Load a trained baseline model from checkpoint.
    
    Args:
        cfg: Configuration
        checkpoint_path: Path to checkpoint (default: cycle_0_best.pt)
        
    Returns:
        Loaded model
    """
    if checkpoint_path is None:
        checkpoint_path = Path(cfg.checkpoint_dir) / "model_cycle_0_best.pt"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.hf_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model architecture
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.hf_path,
        num_labels=cfg.dataset.num_classes,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded baseline model from {checkpoint_path}")
    if 'val_accuracy' in checkpoint:
        logger.info(f"Checkpoint val accuracy: {checkpoint['val_accuracy']:.4f}")
    
    return model
