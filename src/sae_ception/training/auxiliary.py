"""Training with auxiliary loss from sharpened SAE features."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig
import wandb

from ..utils.data import create_dataloaders, extract_activations_from_model
from ..utils.logger import get_logger
from ..utils.checkpointing import save_checkpoint
from ..evaluation import evaluate_classification_accuracy
from ..training.baseline import load_baseline_model
from ..training.sae import load_sae
from ..sharpening import get_sharpener

logger = get_logger(__name__)


def freeze_layers_before_target(
    model: nn.Module,
    target_layer: int,
) -> int:
    """
    Freeze all layers before the target layer.
    
    Only the target layer and classifier head remain trainable.
    This is the approach used in the original SAE-ception paper.
    
    Args:
        model: The model to freeze layers in
        target_layer: Target layer index (e.g., -1 for last layer)
        
    Returns:
        Number of layers frozen
    """
    # Find the transformer layers
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # GPT-2 style: model.transformer.h
        layers = model.transformer.h
        embeddings = model.transformer.wte, model.transformer.wpe
        ln_f = model.transformer.ln_f
    elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        # GPT-NeoX style
        layers = model.gpt_neox.layers
        embeddings = [model.gpt_neox.embed_in]
        ln_f = model.gpt_neox.final_layer_norm
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # LLaMA style
        layers = model.model.layers
        embeddings = [model.model.embed_tokens]
        ln_f = model.model.norm
    else:
        raise ValueError(
            f"Cannot identify transformer layers for model type {type(model).__name__}. "
            f"Supported: GPT-2, GPT-NeoX, LLaMA style architectures."
        )
    
    n_layers = len(layers)
    
    # Convert negative index to positive
    if target_layer < 0:
        target_layer_idx = n_layers + target_layer
    else:
        target_layer_idx = target_layer
    
    # Freeze embeddings
    for emb in embeddings:
        if emb is not None:
            for param in emb.parameters():
                param.requires_grad = False
    
    # Freeze all layers BEFORE the target layer
    n_frozen = 0
    for idx, layer in enumerate(layers):
        if idx < target_layer_idx:
            for param in layer.parameters():
                param.requires_grad = False
            n_frozen += 1
    
    # Log what's trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"Frozen {n_frozen}/{n_layers} transformer layers (layers 0-{target_layer_idx - 1})")
    logger.info(f"Trainable: layer {target_layer_idx}, final LN, classifier head")
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    return n_frozen


def compute_auxiliary_loss(
    activations: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Compute auxiliary loss (cosine distance).
    
    Args:
        activations: Current model activations [batch, hidden_dim]
        targets: Sharpened target activations [batch, hidden_dim]
        
    Returns:
        Cosine distance loss
    """
    # Cosine distance = 1 - cosine_similarity
    cos_sim = F.cosine_similarity(activations, targets, dim=-1)
    loss = 1.0 - cos_sim.mean()
    return loss


def train_epoch_with_aux_loss(
    model: nn.Module,
    sae: nn.Module,
    dataloader,
    targets_dict: dict,
    optimizer,
    scheduler,
    device: str,
    aux_loss_weight: float,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    target_layer: str = '-1',
) -> dict:
    """Train for one epoch with auxiliary loss."""
    model.train()
    sae.eval()
    
    total_loss = 0.0
    total_task_loss = 0.0
    total_aux_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    optimizer.zero_grad()
    
    # We need to track which samples we're processing to get the right targets
    sample_idx = 0
    
    # Get the base model (transformer) from the classification model
    # For GPT2ForSequenceClassification: model.transformer
    # For other models: model.base_model or similar
    if hasattr(model, 'transformer'):
        base_model = model.transformer
    elif hasattr(model, 'base_model'):
        base_model = model.base_model
    elif hasattr(model, 'model'):
        base_model = model.model
    else:
        # Fallback: use model directly
        base_model = model
    
    progress_bar = tqdm(dataloader, desc="Training with Aux Loss")
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        batch_size = input_ids.size(0)
        
        # Get targets for this batch
        batch_targets = targets_dict['targets'][sample_idx:sample_idx + batch_size].to(device)
        sample_idx += batch_size
        
        # Forward pass with activation capture
        from ..utils.hooks import ActivationExtractor, extract_final_token_activation
        
        with ActivationExtractor(base_model, target_layer) as extractor:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get activations
            activations = extractor.get_activations()
            activations = extract_final_token_activation(activations, token_idx=-1)
        
        # Get logits
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        # Compute task loss
        task_loss = F.cross_entropy(logits, labels)
        
        # Compute auxiliary loss
        aux_loss = compute_auxiliary_loss(activations, batch_targets)
        
        # Combined loss
        loss = task_loss + aux_loss_weight * aux_loss
        
        # Scale for gradient accumulation
        loss = loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Track metrics
        total_loss += loss.item() * gradient_accumulation_steps
        total_task_loss += task_loss.item()
        total_aux_loss += aux_loss.item()
        
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item() * gradient_accumulation_steps,
            'task': task_loss.item(),
            'aux': aux_loss.item(),
            'acc': total_correct / total_samples,
        })
    
    return {
        'loss': total_loss / len(dataloader),
        'task_loss': total_task_loss / len(dataloader),
        'aux_loss': total_aux_loss / len(dataloader),
        'accuracy': total_correct / total_samples,
    }


def train_with_auxiliary_loss(
    cfg: DictConfig,
    prev_cycle: int = 0,
    baseline_checkpoint: str = None,
    sae_checkpoint: str = None,
) -> nn.Module:
    """
    Train model with auxiliary loss from sharpened SAE features.
    
    IMPORTANT: Only the target layer and classifier head are trained.
    All layers before the target layer are frozen. This is the approach
    used in the original SAE-ception paper.
    
    Args:
        cfg: Hydra configuration
        prev_cycle: Previous cycle number (loads model/SAE from this cycle)
        baseline_checkpoint: Path to baseline checkpoint (optional)
        sae_checkpoint: Path to SAE checkpoint (optional)
        
    Returns:
        Trained model for next cycle
    """
    next_cycle = prev_cycle + 1
    
    logger.info("=" * 60)
    logger.info(f"Training with Auxiliary Loss (Cycle {prev_cycle} -> {next_cycle})")
    logger.info("=" * 60)
    
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.hf_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load previous cycle's model
    logger.info(f"Loading model from cycle {prev_cycle}")
    if baseline_checkpoint:
        model = load_baseline_model(cfg, checkpoint_path=baseline_checkpoint)
    else:
        # First try current run's checkpoint directory (for multirun)
        checkpoint_dir = Path(cfg.checkpoint_dir)
        checkpoint_path = checkpoint_dir / f"model_cycle_{prev_cycle}_best.pt"
        
        if checkpoint_path.exists():
            logger.info(f"Found checkpoint in current run dir: {checkpoint_path}")
            model = load_baseline_model(cfg, checkpoint_path=str(checkpoint_path))
        else:
            # Fallback: search in outputs/ (for single runs)
            logger.info(f"Checkpoint not in {checkpoint_dir}, searching outputs/...")
            outputs_dir = Path('outputs')
            checkpoint_pattern = f"**/checkpoints/model_cycle_{prev_cycle}_best.pt"
            checkpoints = sorted(outputs_dir.glob(checkpoint_pattern), 
                               key=lambda p: p.stat().st_mtime, reverse=True)
            if checkpoints:
                logger.info(f"Found checkpoint: {checkpoints[0]}")
                model = load_baseline_model(cfg, checkpoint_path=str(checkpoints[0]))
            else:
                raise FileNotFoundError(
                    f"No model checkpoint found for cycle {prev_cycle}. "
                    f"Searched: {checkpoint_dir} and {outputs_dir}"
                )
    model.to(device)
    
    # ==========================================================================
    # FREEZE LAYERS BEFORE TARGET (SAE-ception paper approach)
    # ==========================================================================
    logger.info("Freezing layers before target layer...")
    target_layer = int(cfg.model.target_layer)
    n_frozen = freeze_layers_before_target(model, target_layer)
    
    # Load SAE
    logger.info(f"Loading SAE from cycle {prev_cycle}")
    if sae_checkpoint:
        sae = load_sae(cfg, cycle=prev_cycle, checkpoint_path=sae_checkpoint)
    else:
        # First try current run's checkpoint directory (for multirun)
        checkpoint_dir = Path(cfg.checkpoint_dir)
        checkpoint_path = checkpoint_dir / f"sae_cycle_{prev_cycle}_best.pt"
        
        if checkpoint_path.exists():
            logger.info(f"Found SAE in current run dir: {checkpoint_path}")
            sae = load_sae(cfg, cycle=prev_cycle, checkpoint_path=str(checkpoint_path))
        else:
            # Fallback: search in outputs/ (for single runs)
            logger.info(f"SAE not in {checkpoint_dir}, searching outputs/...")
            outputs_dir = Path('outputs')
            checkpoint_pattern = f"**/checkpoints/sae_cycle_{prev_cycle}_best.pt"
            checkpoints = sorted(outputs_dir.glob(checkpoint_pattern), 
                               key=lambda p: p.stat().st_mtime, reverse=True)
            if checkpoints:
                logger.info(f"Found SAE: {checkpoints[0]}")
                sae = load_sae(cfg, cycle=prev_cycle, checkpoint_path=str(checkpoints[0]))
            else:
                raise FileNotFoundError(
                    f"No SAE checkpoint found for cycle {prev_cycle}. "
                    f"Searched: {checkpoint_dir} and {outputs_dir}"
                )
    sae.to(device)
    sae.eval()
    
    # Create dataloaders
    logger.info("Loading dataset...")
    dataloaders = create_dataloaders(cfg, tokenizer)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    # Extract activations and generate sharpened targets
    logger.info("Extracting activations for target generation...")
    logger.info("Processing training set...")
    train_acts, train_labels = extract_activations_from_model(
        model=model,
        dataloader=train_loader,
        layer_name=str(cfg.model.target_layer),
        device=device,
        token_position='last',
    )
    
    # Initialize sharpener
    logger.info(f"Using sharpening strategy: {cfg.sharpening.type}")
    sharpener = get_sharpener(cfg.sharpening)
    
    # Create temporary dataloader for activation processing
    from ..utils.data import create_activation_dataloader
    temp_loader = create_activation_dataloader(
        train_acts, train_labels,
        batch_size=cfg.sae.batch_size,
        shuffle=False,
    )
    
    # Generate sharpened targets
    logger.info("Generating sharpened targets...")
    targets = sharpener.generate_targets(sae, temp_loader, device)
    
    logger.info(f"Generated targets shape: {targets.shape}")
    logger.info(f"Target mean: {targets.mean():.4f}, std: {targets.std():.4f}")
    
    # Store targets in dictionary for easy batching
    targets_dict = {
        'targets': targets,
        'labels': train_labels,
    }
    
    # Setup optimizer - ONLY for trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
    )
    
    # Calculate total training steps
    total_steps = len(train_loader) * cfg.model.epochs_per_cycle // cfg.model.gradient_accumulation_steps
    
    # Setup scheduler
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
            name=f"aux_cycle_{next_cycle}_{cfg.dataset.name}",
            config=dict(cfg),
            tags=['auxiliary', f'cycle_{next_cycle}'],
        )
    
    # Training loop
    logger.info(f"Training for {cfg.model.epochs_per_cycle} epochs...")
    logger.info(f"Auxiliary loss weight (λ): {cfg.cycle.aux_loss_weight}")
    
    best_val_acc = 0.0
    
    for epoch in range(cfg.model.epochs_per_cycle):
        logger.info(f"Epoch {epoch + 1}/{cfg.model.epochs_per_cycle}")
        
        # Train
        train_metrics = train_epoch_with_aux_loss(
            model=model,
            sae=sae,
            dataloader=train_loader,
            targets_dict=targets_dict,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            aux_loss_weight=cfg.cycle.aux_loss_weight,
            gradient_accumulation_steps=cfg.model.gradient_accumulation_steps,
            max_grad_norm=cfg.model.max_grad_norm,
            target_layer=str(cfg.model.target_layer),
        )
        
        logger.info(f"Train Total Loss: {train_metrics['loss']:.4f}")
        logger.info(f"Train Task Loss: {train_metrics['task_loss']:.4f}")
        logger.info(f"Train Aux Loss: {train_metrics['aux_loss']:.4f}")
        logger.info(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
        
        # Evaluate (task performance only)
        logger.info("Evaluating...")
        val_metrics = evaluate_classification_accuracy(model, val_loader, device)
        
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"Val F1 (macro): {val_metrics['f1_macro']:.4f}")
        
        # Log to wandb
        if cfg.wandb.mode != 'disabled':
            wandb.log({
                'epoch': epoch,
                'train/total_loss': train_metrics['loss'],
                'train/task_loss': train_metrics['task_loss'],
                'train/aux_loss': train_metrics['aux_loss'],
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
            
            checkpoint_path = checkpoint_dir / f"model_cycle_{next_cycle}_best.pt"
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={
                    'val_accuracy': val_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                },
            )
    
    # Load best model
    logger.info(f"Loading best model (val_acc: {best_val_acc:.4f})")
    checkpoint_path = Path(cfg.checkpoint_dir) / f"model_cycle_{next_cycle}_best.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
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
    
    logger.info("Auxiliary training complete!")
    
    return model