#!/usr/bin/env python3
"""
Complete SAE-ception training cycle.

This script orchestrates the full SAE-ception process:
- Cycle 0: Create baseline model (pretrained or finetuned) + train SAE + evaluate
- Cycle 1+: Apply aux loss training using previous cycle's SAE, train new SAE, evaluate

Usage:
    # Run just cycle 0 (baseline only, no SAE-ception)
    python scripts/train_cycle.py cycle.current=0 cycle.max_cycles=0

    # Run cycle 0 and cycle 1 (baseline + first SAE-ception iteration)
    python scripts/train_cycle.py cycle.current=0 cycle.max_cycles=1
    
    # Run multiple cycles
    python scripts/train_cycle.py cycle.current=0 cycle.max_cycles=3
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from pathlib import Path
import logging

from sae_ception.training.baseline import train_baseline_model, load_baseline_model
from sae_ception.training.sae import train_sae, load_sae
from sae_ception.training.auxiliary import train_with_auxiliary_loss
from sae_ception.evaluation import (
    evaluate_classification_accuracy,
    evaluate_model_and_sae,
    train_linear_probe,
)
from sae_ception.utils.logger import setup_logging
from sae_ception.utils.data import (
    create_dataloaders,
    extract_activations_from_model,
    create_activation_dataloader,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def evaluate_cycle(
    cfg: DictConfig,
    model,
    sae,
    cycle: int,
    device: str = 'cuda',
) -> dict:
    """
    Comprehensive evaluation of a cycle.
    
    Uses two extraction strategies:
    - Token-level: For SAE quality metrics (dead features %)
    - Sequence-level (final token): For classification metrics (probe, clustering)
    
    Args:
        cfg: Configuration
        model: Trained model
        sae: Trained SAE
        cycle: Cycle number
        device: Device
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("=" * 60)
    if cycle == 0:
        logger.info(f"Evaluating Cycle 0 (baseline)")
    else:
        logger.info(f"Evaluating Cycle {cycle}")
    logger.info("=" * 60)
    
    # Load tokenizer and data
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.hf_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataloaders = create_dataloaders(cfg, tokenizer)
    val_loader = dataloaders['val']
    train_loader = dataloaders['train']
    
    token_level = getattr(cfg.sae, 'token_level', True)
    
    # =========================================================================
    # SEQUENCE-LEVEL EXTRACTION (for classification metrics)
    # =========================================================================
    logger.info("Extracting FINAL token activations for classification metrics...")
    train_acts_seq, train_labels_seq = extract_activations_from_model(
        model, train_loader, str(cfg.model.target_layer), device, 'last'
    )
    val_acts_seq, val_labels_seq = extract_activations_from_model(
        model, val_loader, str(cfg.model.target_layer), device, 'last'
    )
    logger.info(f"Sequence-level: train={train_acts_seq.shape}, val={val_acts_seq.shape}")
    
    # Compute sparse codes for sequence-level (small, fits in memory)
    sae.eval()
    with torch.no_grad():
        sparse_train_seq = sae.encode(train_acts_seq.to(device)).cpu()
        sparse_val_seq = sae.encode(val_acts_seq.to(device)).cpu()
    
    # =========================================================================
    # TOKEN-LEVEL EXTRACTION (for SAE quality / dead features)
    # =========================================================================
    if token_level:
        logger.info("Extracting ALL token activations for SAE quality metrics...")
        from sae_ception.utils.data import extract_activations_all_tokens
        
        # Only need val set for dead feature calculation
        val_acts_tok, _, _ = extract_activations_all_tokens(
            model, val_loader, str(cfg.model.target_layer), device
        )
        logger.info(f"Token-level val: {val_acts_tok.shape}")
        
        # Compute dead features % using token-level activations (batched)
        logger.info("Computing SAE quality metrics on token-level activations...")
        sae_quality = compute_sae_quality_batched(sae, val_acts_tok, device)
    else:
        # Use sequence-level for SAE quality too
        from sae_ception.evaluation import evaluate_sae_quality
        sae_quality = evaluate_sae_quality(sae, val_acts_seq, device)
    
    # =========================================================================
    # COMPUTE ALL OTHER METRICS (using sequence-level data)
    # =========================================================================
    from sae_ception.evaluation import (
        evaluate_classification_accuracy,
        train_linear_probe,
        compute_monosemanticity_metrics,
        compute_all_clustering_metrics,
        compute_sparsity_metrics,
        k_sparse_probe_accuracy,
    )
    
    results = {}
    
    # 1. Task performance
    print("Evaluating task performance...")
    task_metrics = evaluate_classification_accuracy(model, val_loader, device)
    results.update({f'task_{k}': v for k, v in task_metrics.items()})
    
    # 2. Linear probe on SAE features
    print("Training linear probe on SAE features...")
    probe_metrics = train_linear_probe(
        sparse_train_seq, train_labels_seq,
        sparse_val_seq, val_labels_seq,
    )
    results.update({f'probe_{k}': v for k, v in probe_metrics.items()})
    
    # 3. K-sparse probing
    print("Evaluating k-sparse probe accuracy...")
    k_sparse_metrics = k_sparse_probe_accuracy(
        sparse_val_seq, val_labels_seq,
        k_values=[1, 5, 10, 25, 50],
    )
    results.update(k_sparse_metrics)
    
    # 4. SAE quality (from token-level if enabled)
    results.update({f'sae_{k}': v for k, v in sae_quality.items()})
    
    # 5. Monosemanticity of base activations
    print("Computing monosemanticity metrics (base model)...")
    mono_base = compute_monosemanticity_metrics(val_acts_seq, val_labels_seq)
    results.update({f'base_{k}': v for k, v in mono_base.items()})
    
    # 6. Monosemanticity of SAE features
    print("Computing monosemanticity metrics (SAE features)...")
    mono_sae = compute_monosemanticity_metrics(sparse_val_seq, val_labels_seq)
    results.update({f'sae_features_{k}': v for k, v in mono_sae.items()})
    
    # 7. Clustering metrics
    print("Computing clustering metrics...")
    clustering = compute_all_clustering_metrics(sparse_val_seq, val_labels_seq)
    results.update({f'clustering_{k}': v for k, v in clustering.items()})
    
    # 8. Sparsity metrics (from sequence-level sparse codes)
    print("Computing sparsity metrics...")
    sparsity = compute_sparsity_metrics(sparse_val_seq)
    results.update({f'sparsity_{k}': v for k, v in sparsity.items()})
    
    return results


def compute_sae_quality_batched(
    sae: torch.nn.Module,
    activations: torch.Tensor,
    device: str = 'cuda',
    batch_size: int = 4096,
) -> dict:
    """
    Compute SAE quality metrics in batches to avoid OOM.
    
    This is the token-level version that correctly computes dead features %.
    """
    import numpy as np
    
    sae.eval()
    sae.to(device)
    
    all_recon_loss = []
    all_explained_var = []
    all_l0 = []
    
    # Track which features ever fire (for dead feature %)
    n_features = sae.hidden_dim
    feature_ever_fired = torch.zeros(n_features, dtype=torch.bool)
    
    n_samples = len(activations)
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = activations[i:i+batch_size].to(device)
            
            # Forward pass
            reconstruction, sparse_code, _ = sae(batch)
            
            # Reconstruction loss
            recon_loss = torch.nn.functional.mse_loss(reconstruction, batch)
            all_recon_loss.append(recon_loss.item())
            
            # Explained variance
            var_original = batch.var(dim=0).mean()
            var_residual = (batch - reconstruction).var(dim=0).mean()
            explained_var = 1 - (var_residual / (var_original + 1e-8))
            all_explained_var.append(explained_var.item())
            
            # Sparsity (L0)
            l0 = (sparse_code > 0).float().sum(dim=-1).mean()
            all_l0.append(l0.item())
            
            # Track which features fired in this batch
            batch_fired = (sparse_code > 0).any(dim=0).cpu()
            feature_ever_fired |= batch_fired
            
            # Clear GPU cache periodically
            if i % (batch_size * 10) == 0 and i > 0:
                torch.cuda.empty_cache()
    
    # Compute dead features %
    dead_features_pct = 100.0 * (~feature_ever_fired).float().mean().item()
    
    # Compute feature usage entropy
    # (This is approximate - we'd need to track full counts for exact entropy)
    n_active = feature_ever_fired.sum().item()
    if n_active > 0:
        # Approximate entropy assuming uniform distribution among active features
        feature_usage_entropy = np.log(n_active)
    else:
        feature_usage_entropy = 0.0
    
    return {
        'reconstruction_loss': float(np.mean(all_recon_loss)),
        'explained_variance': float(np.mean(all_explained_var)),
        'mean_l0': float(np.mean(all_l0)),
        'dead_features_pct': float(dead_features_pct),
        'feature_usage_entropy': float(feature_usage_entropy),
    }


def load_pretrained_model_for_eval(cfg):
    """Load pretrained model with classification head (no training)."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.hf_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.hf_path,
        num_labels=cfg.dataset.num_classes,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    return model


def get_or_create_cycle_0(cfg, device):
    """
    Get cached cycle 0 (baseline) or create it.
    
    Cycle 0 is the baseline:
    - If start_from='baseline': Load pretrained model (no fine-tuning)
    - If start_from='finetuned': Fine-tune model on task (no aux loss)
    
    Then train SAE and evaluate.
    
    Returns:
        Tuple of (model, sae, results)
    """
    # Cache location based on model + dataset + SAE config
    sae_type = getattr(cfg.sae, 'sae_type', 'l1')
    if sae_type == 'topk':
        cache_key = f"{cfg.model.name}_{cfg.dataset.name}_topk-k{cfg.sae.k}_exp-{cfg.sae.expansion_factor}"
    else:
        cache_key = f"{cfg.model.name}_{cfg.dataset.name}_l1-{cfg.sae.l1_penalty}_exp-{cfg.sae.expansion_factor}"
    cache_dir = Path(cfg.cache_dir) / cache_key

    # Clear cache if requested
    if getattr(cfg, 'clear_cache', False) and cache_dir.exists():
        logger.info(f"Clearing cache: {cache_dir}")
        import shutil
        shutil.rmtree(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = cfg.start_from  # 'baseline' or 'finetuned'
    model_path = cache_dir / f"model_cycle_0_{prefix}.pt"
    sae_path = cache_dir / f"sae_cycle_0_{prefix}.pt"
    results_path = cache_dir / f"results_cycle_0_{prefix}.pt"
    
    # Check if cached
    if all(p.exists() for p in [model_path, sae_path, results_path]):
        logger.info(f"✓ Loading cached cycle 0 ({prefix}) from {cache_dir}")
        model = load_baseline_model(cfg, checkpoint_path=str(model_path))
        sae = load_sae(cfg, cycle=0, checkpoint_path=str(sae_path))
        results = torch.load(results_path)
        return model, sae, results
    
    # Create new
    logger.info(f"Creating cycle 0 ({prefix})...")
    
    if cfg.start_from == 'baseline':
        # Load pretrained, no training
        logger.info("Loading pretrained model (no fine-tuning)...")
        model = load_pretrained_model_for_eval(cfg)
        model.to(device)
    else:  # 'finetuned'
        # Fine-tune from pretrained (no aux loss - this is standard fine-tuning)
        logger.info("Fine-tuning model on task (no aux loss)...")
        model = train_baseline_model(cfg)
    
    # Train SAE on this model
    logger.info("Training SAE for cycle 0...")
    sae, _, _ = train_sae(cfg, model=model, cycle=0)
    
    # Evaluate
    logger.info("Evaluating cycle 0...")
    results = evaluate_cycle(cfg, model, sae, cycle=0, device=device)
    
    # Cache everything
    torch.save({
        'model_state_dict': model.state_dict(),
        'val_accuracy': results.get('task_accuracy', 0.0),
    }, model_path)
    
    torch.save({
        'model_state_dict': sae.state_dict(),
        'val_l0_norm': results.get('sparsity_l0_mean', 0.0),
    }, sae_path)
    
    torch.save(results, results_path)
    logger.info(f"✓ Cached cycle 0 ({prefix}) to {cache_dir}")
    
    return model, sae, results


def log_evaluation_results(results: dict, label: str = "Results"):
    """
    Log full evaluation results in a consistent format.
    
    Args:
        results: Dictionary of evaluation metrics
        label: Label for this set of results
    """
    logger.info("=" * 60)
    logger.info(f"{label}")
    logger.info("=" * 60)
    logger.info(f"  Task Accuracy:     {results.get('task_accuracy', 0):.4f}")
    logger.info(f"  Probe Accuracy:    {results.get('probe_test_accuracy', 0):.4f}")
    logger.info(f"  Base Model CSI:    {results.get('base_csi', 0):.4f}")
    logger.info(f"  SAE Features U:    {results.get('sae_features_u', 0):.4f}")
    logger.info(f"  Clustering ARI:    {results.get('clustering_ari', 0):.4f}")
    logger.info(f"  Silhouette (sup):  {results.get('clustering_silhouette_supervised', 0):.4f}")
    logger.info(f"  Davies-Bouldin:    {results.get('clustering_davies_bouldin', 0):.4f}")
    logger.info(f"  Sparsity L0:       {results.get('sparsity_l0_mean', 0):.2f}")
    logger.info(f"  Dead Features %:   {results.get('sae_dead_features_pct', 0):.1f}%")


def run_single_cycle(cfg: DictConfig, cycle: int, checkpoint_dir: Path) -> tuple:
    """
    Run a single SAE-ception cycle.
    
    Args:
        cfg: Configuration
        cycle: Cycle number
        checkpoint_dir: Directory to save checkpoints
        
    Returns:
        Tuple of (model, sae, results)
    """
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    
    if cycle == 0:
        # =====================================================================
        # CYCLE 0: Baseline (no aux loss)
        # =====================================================================
        logger.info("=" * 60)
        logger.info("CYCLE 0: Creating Baseline")
        logger.info(f"  Mode: {cfg.start_from}")
        logger.info("=" * 60)
        
        # Get or create the baseline model + SAE + evaluation
        model, sae, results = get_or_create_cycle_0(cfg, device)
        
        # Log results
        log_evaluation_results(results, f"Cycle 0 (baseline - {cfg.start_from}) Summary")
        
        # Save to this run's checkpoint directory as well
        torch.save(results, checkpoint_dir / "results_cycle_0.pt")
        
        # Save model and SAE checkpoints for subsequent cycles
        torch.save({
            'model_state_dict': model.state_dict(),
            'val_accuracy': results.get('task_accuracy', 0.0),
        }, checkpoint_dir / "model_cycle_0_best.pt")
        
        torch.save({
            'model_state_dict': sae.state_dict(),
            'val_l0_norm': results.get('sparsity_l0_mean', 0.0),
        }, checkpoint_dir / "sae_cycle_0_best.pt")
        
        return model, sae, results
    
    else:
        # =====================================================================
        # CYCLE >= 1: SAE-ception (with aux loss)
        # =====================================================================
        prev_cycle = cycle - 1
        
        logger.info("=" * 60)
        logger.info(f"CYCLE {cycle}: SAE-ception")
        logger.info(f"  Loading from cycle {prev_cycle}")
        logger.info("=" * 60)
        
        # Step 1: Load previous cycle's model
        logger.info(f"Step 1: Loading model from cycle {prev_cycle}...")
        prev_model_path = checkpoint_dir / f"model_cycle_{prev_cycle}_best.pt"
        
        if not prev_model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {prev_model_path}\n"
                f"Run cycle {prev_cycle} first."
            )
        
        # We need to load the model architecture first, then load weights
        # For aux loss training, we pass the checkpoint path directly
        
        # Step 2: Load previous cycle's SAE
        logger.info(f"Step 2: Loading SAE from cycle {prev_cycle}...")
        prev_sae_path = checkpoint_dir / f"sae_cycle_{prev_cycle}_best.pt"
        
        if not prev_sae_path.exists():
            raise FileNotFoundError(
                f"SAE checkpoint not found: {prev_sae_path}\n"
                f"Run cycle {prev_cycle} first."
            )
        
        # Step 3: Apply auxiliary loss training
        logger.info(f"Step 3: Training with auxiliary loss (cycle {prev_cycle} -> {cycle})...")
        model = train_with_auxiliary_loss(
            cfg,
            prev_cycle=prev_cycle,
            baseline_checkpoint=str(prev_model_path),
            sae_checkpoint=str(prev_sae_path),
        )
        
        # Step 4: Train new SAE on the updated model
        logger.info(f"Step 4: Training SAE for cycle {cycle}...")
        sae, _, _ = train_sae(cfg, model=model, cycle=cycle)
        
        # Step 5: Evaluate
        logger.info(f"Step 5: Evaluating cycle {cycle}...")
        results = evaluate_cycle(cfg, model, sae, cycle, device)
        
        # Save results
        torch.save(results, checkpoint_dir / f"results_cycle_{cycle}.pt")
        
        # Log results
        log_evaluation_results(results, f"Cycle {cycle} Summary")
        
        # Save model and SAE checkpoints for subsequent cycles
        torch.save({
            'model_state_dict': model.state_dict(),
            'val_accuracy': results.get('task_accuracy', 0.0),
        }, checkpoint_dir / f"model_cycle_{cycle}_best.pt")
        
        torch.save({
            'model_state_dict': sae.state_dict(),
            'val_l0_norm': results.get('sparsity_l0_mean', 0.0),
        }, checkpoint_dir / f"sae_cycle_{cycle}_best.pt")
        
        return model, sae, results


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Run SAE-ception training cycle(s)."""
    
    logger.info("=" * 80)
    logger.info("SAE-ception Training")
    logger.info("=" * 80)
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Set random seeds
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    
    current_cycle = cfg.cycle.current
    max_cycles = cfg.cycle.max_cycles
    
    logger.info(f"Starting from cycle {current_cycle}")
    logger.info(f"Will run up to cycle {max_cycles}")
    logger.info(f"Start from: {cfg.start_from}")
    logger.info(f"Sharpening strategy: {cfg.sharpening.type}")
    logger.info(f"Auxiliary loss weight: {cfg.cycle.aux_loss_weight}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Track all results
    all_results = {}
    
    # Run cycles
    for cycle in range(current_cycle, max_cycles + 1):
        logger.info("\n" + "=" * 80)
        logger.info(f"RUNNING CYCLE {cycle}")
        logger.info("=" * 80 + "\n")
        
        model, sae, results = run_single_cycle(cfg, cycle, checkpoint_dir)
        all_results[cycle] = results
        
        logger.info(f"✓ Cycle {cycle} complete")
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("SAE-ception Training Complete!")
    logger.info("=" * 80)
    
    logger.info("\nResults Summary:")
    logger.info("-" * 110)
    logger.info(
        f"{'Cycle':<22} {'Accuracy':<12} {'Probe Acc':<12} {'ARI':<10} {'U':<10} {'L0':<10} {'Dead %':<10}"
    )
    logger.info("-" * 110)
    
    for cycle in sorted(all_results.keys()):
        results = all_results[cycle]
        
        # Label cycle 0 as baseline
        if cycle == 0:
            label = f"Cycle 0 (baseline)"
        else:
            label = f"Cycle {cycle}"
        
        logger.info(
            f"{label:<22} "
            f"{results.get('task_accuracy', 0):<12.4f} "
            f"{results.get('probe_test_accuracy', 0):<12.4f} "
            f"{results.get('clustering_ari', 0):<10.4f} "
            f"{results.get('sae_features_u', 0):<10.4f} "
            f"{results.get('sparsity_l0_mean', 0):<10.2f} "
            f"{results.get('sae_dead_features_pct', 0):<10.1f}"
        )
    
    logger.info("-" * 110)
    
    logger.info(f"\nAll checkpoints saved to: {cfg.checkpoint_dir}")
    logger.info(f"Results saved to: {cfg.checkpoint_dir}/results_cycle_*.pt")


if __name__ == "__main__":
    main()