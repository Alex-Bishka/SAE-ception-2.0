#!/usr/bin/env python3
"""
Structure Check Analysis: Compare SAEs trained on control vs SAE-ception models.

This script implements the "Structure Check" from the SAE-ception 2.0 research plan:
- Compare L0 sparsity, dead features %, convergence speed
- Compare decoder weight geometry (clustering, similarity)
- Compare feature co-activation patterns
- Compare feature absorption (SAEBench first-letter metric)
- Hypothesis: SAE on SAE-ception model should be "lazier" (lower L0, faster convergence,
  tighter clusters, lower absorption)

Uses existing evaluation infrastructure from sae_ception.evaluation:
- clustering.py: ARI, Silhouette, Davies-Bouldin, etc.
- interpretability.py: dead_features_pct, feature_usage_entropy
- saebench.py: feature absorption (first-letter absorption metric)

Usage:
    # Basic comparison (from diagnostics files only)
    python scripts/compare_sae_structure.py \
        --control_sae checkpoints/sae_on_control.pt \
        --saeception_sae checkpoints/sae_on_saeception.pt \
        --output results/structure_check.json
        
    # Full analysis with models (enables co-activation & absorption)
    python scripts/compare_sae_structure.py \
        --control_sae checkpoints/sae_on_control.pt \
        --saeception_sae checkpoints/sae_on_saeception.pt \
        --control_model checkpoints/model_cpt_no_aux.pt \
        --saeception_model checkpoints/model_cpt_with_aux.pt \
        --base_model EleutherAI/pythia-70m \
        --output results/structure_check.json
        
    # Quick absorption test (fewer latent candidates)
    python scripts/compare_sae_structure.py \
        --control_sae checkpoints/sae_on_control.pt \
        --saeception_sae checkpoints/sae_on_saeception.pt \
        --control_model checkpoints/model_cpt_no_aux.pt \
        --saeception_model checkpoints/model_cpt_with_aux.pt \
        --base_model EleutherAI/pythia-70m \
        --n_absorption_candidates 50 \
        --output results/structure_check.json
"""

import argparse
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import logging
from tabulate import tabulate
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import existing evaluation infrastructure
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sae_ception.models.sae import create_sae
from sae_ception.evaluation.interpretability import evaluate_sae_quality, compute_sparsity_metrics
from sae_ception.evaluation.clustering import compute_all_clustering_metrics
from sae_ception.evaluation.saebench import compute_first_letter_absorption
from sae_ception.utils.data import create_causal_lm_dataloader
from sae_ception.utils.hooks import ActivationExtractor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# TOKEN-LEVEL ACTIVATION EXTRACTION FOR ABSORPTION
# =============================================================================

def extract_token_activations_for_absorption(
    model: torch.nn.Module,
    tokenizer,
    layer_idx: int,
    device: str = 'cuda',
    max_samples: int = 2000,
    max_length: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract token-level activations for absorption analysis.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        layer_idx: Layer to extract from
        device: Device to run on
        max_samples: Max sequences to process
        max_length: Max sequence length
        
    Returns:
        Tuple of (activations [N_tokens, hidden], token_ids [N_tokens])
    """
    model.eval()
    model.to(device)
    
    # Create dataloader
    dataloader = create_causal_lm_dataloader(
        tokenizer=tokenizer,
        dataset_name='wikitext',
        split='test',
        batch_size=4,
        max_length=max_length,
        max_samples=max_samples,
    )
    
    all_activations = []
    all_token_ids = []
    
    # Get the base model for hook attachment
    if hasattr(model, 'gpt_neox'):
        base_model = model.gpt_neox
    elif hasattr(model, 'transformer'):
        base_model = model.transformer
    else:
        base_model = model
    
    logger.info(f"Extracting activations from layer {layer_idx}...")
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            else:
                attention_mask = torch.ones_like(input_ids)
            
            batch_size, seq_len = input_ids.shape
            
            # Extract activations
            with ActivationExtractor(base_model, str(layer_idx)) as extractor:
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
                activations = extractor.get_activations()  # [batch, seq, hidden]
            
            # Flatten and filter padding
            for b in range(batch_size):
                for t in range(seq_len):
                    if attention_mask[b, t] == 1:
                        all_activations.append(activations[b, t].cpu())
                        all_token_ids.append(input_ids[b, t].cpu())
    
    activations_tensor = torch.stack(all_activations)
    token_ids_tensor = torch.stack(all_token_ids)
    
    logger.info(f"Extracted {len(activations_tensor)} token activations")
    
    return activations_tensor, token_ids_tensor


def load_model_from_checkpoint(
    checkpoint_path: str,
    base_model_name: str,
    device: str = 'cuda',
) -> torch.nn.Module:
    """
    Load a model from a CPT checkpoint.
    
    Args:
        checkpoint_path: Path to .pt checkpoint
        base_model_name: HuggingFace model name for architecture
        device: Device to load on
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get base model name from checkpoint if available
    config = checkpoint.get('config', {})
    model_name = config.get('model_name', base_model_name)
    
    # Load architecture
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


# =============================================================================
# DECODER WEIGHT GEOMETRY ANALYSIS
# =============================================================================

def compute_decoder_geometry(sae: torch.nn.Module, n_clusters: int = 10) -> Dict:
    """
    Analyze decoder weight geometry.
    
    Computes:
    - Decoder column cosine similarity distribution
    - Natural clustering of decoder columns
    - Cluster tightness (silhouette score)
    
    Args:
        sae: SAE model
        n_clusters: Number of clusters for K-means
        
    Returns:
        Dictionary with geometry metrics
    """
    # Get decoder weights [input_dim, hidden_dim]
    if hasattr(sae, 'decoder') and hasattr(sae.decoder, 'weight'):
        decoder = sae.decoder.weight.data.cpu()  # [input_dim, hidden_dim]
        # Transpose so each column is a feature direction
        decoder = decoder.T  # [hidden_dim, input_dim]
    else:
        return {'error': 'No decoder weights found'}
    
    # Normalize decoder columns
    decoder_norm = F.normalize(decoder, dim=1)
    
    # 1. Pairwise cosine similarity statistics
    # Sample to avoid O(n^2) for large hidden dims
    n_features = decoder.shape[0]
    if n_features > 1000:
        # Sample 1000 features for statistics
        sample_idx = torch.randperm(n_features)[:1000]
        sample = decoder_norm[sample_idx]
    else:
        sample = decoder_norm
    
    cos_sim_matrix = sample @ sample.T
    # Get upper triangle (excluding diagonal)
    triu_idx = torch.triu_indices(cos_sim_matrix.shape[0], cos_sim_matrix.shape[1], offset=1)
    pairwise_sims = cos_sim_matrix[triu_idx[0], triu_idx[1]]
    
    # 2. K-means clustering on decoder columns
    try:
        kmeans = KMeans(n_clusters=min(n_clusters, n_features // 10 + 1), random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(decoder_norm.numpy())
        
        # Silhouette score (cluster quality)
        if len(np.unique(cluster_labels)) > 1:
            cluster_silhouette = silhouette_score(decoder_norm.numpy(), cluster_labels)
        else:
            cluster_silhouette = 0.0
    except Exception as e:
        logger.warning(f"Clustering failed: {e}")
        cluster_silhouette = 0.0
        cluster_labels = np.zeros(n_features)
    
    # 3. Compute cluster sizes (feature grouping)
    unique, counts = np.unique(cluster_labels, return_counts=True)
    cluster_size_std = float(np.std(counts))
    cluster_size_gini = _gini_coefficient(counts)
    
    return {
        'cos_sim_mean': float(pairwise_sims.mean()),
        'cos_sim_std': float(pairwise_sims.std()),
        'cos_sim_max': float(pairwise_sims.max()),
        'cos_sim_min': float(pairwise_sims.min()),
        'cluster_silhouette': float(cluster_silhouette),
        'cluster_size_std': cluster_size_std,
        'cluster_size_gini': cluster_size_gini,
        'n_clusters': len(unique),
    }


def compute_coactivation_patterns(
    sae: torch.nn.Module,
    activations: torch.Tensor,
    device: str = 'cuda',
    batch_size: int = 1024,
    top_k_pairs: int = 100,
) -> Dict:
    """
    Analyze which features tend to fire together.
    
    Computes:
    - Pairwise feature correlation
    - Co-firing frequency
    - Feature redundancy score
    
    Args:
        sae: SAE model
        activations: Input activations [N, hidden_dim]
        device: Device to run on
        batch_size: Batch size for processing
        top_k_pairs: Number of top correlated pairs to track
        
    Returns:
        Dictionary with co-activation metrics
    """
    sae.eval()
    sae.to(device)
    
    # Collect sparse codes
    all_sparse = []
    n_samples = len(activations)
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = activations[i:i+batch_size].to(device)
            sparse = sae.encode(batch)
            all_sparse.append((sparse > 0).float().cpu())  # Binary firing pattern
    
    firing_patterns = torch.cat(all_sparse, dim=0)  # [N, sae_hidden]
    
    # 1. Feature firing frequency
    firing_freq = firing_patterns.mean(dim=0)  # [sae_hidden]
    
    # 2. Co-firing matrix (sampled for efficiency)
    n_features = firing_patterns.shape[1]
    if n_features > 500:
        # Sample features that actually fire
        active_features = torch.where(firing_freq > 0.001)[0]
        if len(active_features) > 500:
            sample_idx = active_features[torch.randperm(len(active_features))[:500]]
        else:
            sample_idx = active_features
        sample = firing_patterns[:, sample_idx]
    else:
        sample = firing_patterns
        sample_idx = torch.arange(n_features)
    
    # Compute correlation matrix
    sample_centered = sample - sample.mean(dim=0, keepdim=True)
    sample_std = sample.std(dim=0, keepdim=True) + 1e-8
    sample_norm = sample_centered / sample_std
    
    corr_matrix = (sample_norm.T @ sample_norm) / sample.shape[0]
    
    # Get upper triangle
    triu_idx = torch.triu_indices(corr_matrix.shape[0], corr_matrix.shape[1], offset=1)
    pairwise_corr = corr_matrix[triu_idx[0], triu_idx[1]]
    
    # 3. Find highly correlated pairs (potential redundancy)
    top_corr_vals, top_corr_idx = torch.topk(pairwise_corr.abs(), min(top_k_pairs, len(pairwise_corr)))
    
    # 4. Gini coefficient of firing frequency (concentration)
    firing_gini = _gini_coefficient(firing_freq.numpy())
    
    # 5. Effective feature count (how many features carry 90% of activations)
    sorted_freq, _ = torch.sort(firing_freq, descending=True)
    cumsum = torch.cumsum(sorted_freq, dim=0) / (sorted_freq.sum() + 1e-8)
    effective_features = int((cumsum < 0.9).sum()) + 1
    
    return {
        'firing_freq_mean': float(firing_freq.mean()),
        'firing_freq_std': float(firing_freq.std()),
        'correlation_mean': float(pairwise_corr.mean()),
        'correlation_std': float(pairwise_corr.std()),
        'correlation_max': float(pairwise_corr.max()),
        'high_corr_pairs': int((pairwise_corr.abs() > 0.5).sum()),
        'firing_gini': firing_gini,
        'effective_features': effective_features,
        'effective_features_pct': effective_features / n_features * 100,
    }


def _gini_coefficient(values: np.ndarray) -> float:
    """Compute Gini coefficient (0 = equal, 1 = maximally concentrated)."""
    values = np.sort(values.flatten())
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    cumsum = np.cumsum(values)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n


def load_diagnostics(checkpoint_path: str) -> Dict:
    """Load diagnostics from .diagnostics.json file."""
    diag_path = Path(checkpoint_path).with_suffix('.diagnostics.json')
    
    if diag_path.exists():
        with open(diag_path) as f:
            return json.load(f)
    else:
        logger.warning(f"No diagnostics file found at {diag_path}")
        return {}


def load_training_history(history_path: str) -> Optional[List[Dict]]:
    """Load per-epoch training history."""
    if history_path and Path(history_path).exists():
        with open(history_path) as f:
            return json.load(f)
    return None


def load_checkpoint_info(checkpoint_path: str) -> Dict:
    """Load checkpoint and extract config/training info."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    info = {
        'config': checkpoint.get('config', {}),
        'training': checkpoint.get('training', {}),
    }
    
    return info


def compute_convergence_metrics(history: List[Dict]) -> Dict:
    """
    Compute convergence speed metrics from training history.
    
    Metrics:
    - epochs_to_90pct: Epochs to reach 90% of final performance
    - initial_loss: Loss at epoch 1
    - final_loss: Loss at last epoch
    - loss_improvement_rate: Average loss decrease per epoch
    """
    if not history:
        return {}
    
    losses = [h.get('loss', h.get('reconstruction_loss', float('inf'))) for h in history]
    
    if not losses or all(l == float('inf') for l in losses):
        return {}
    
    initial_loss = losses[0]
    final_loss = losses[-1]
    n_epochs = len(losses)
    
    # Epochs to reach 90% of final improvement
    total_improvement = initial_loss - final_loss
    target_loss = initial_loss - 0.9 * total_improvement
    
    epochs_to_90pct = n_epochs  # Default to all epochs
    for i, loss in enumerate(losses):
        if loss <= target_loss:
            epochs_to_90pct = i + 1
            break
    
    # Loss improvement rate
    if n_epochs > 1:
        improvement_rate = total_improvement / (n_epochs - 1)
    else:
        improvement_rate = 0.0
    
    return {
        'epochs_to_90pct': epochs_to_90pct,
        'initial_loss': float(initial_loss),
        'final_loss': float(final_loss),
        'total_improvement': float(total_improvement),
        'improvement_rate': float(improvement_rate),
        'n_epochs': n_epochs,
    }


def load_sae_from_checkpoint(checkpoint_path: str, device: str = 'cuda') -> torch.nn.Module:
    """Load SAE model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    sae = create_sae(
        input_dim=config.get('input_dim', 512),
        hidden_dim=config.get('hidden_dim', 4096),
        sae_type='topk',
        k=config.get('k', 32),
    )
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.to(device)
    sae.eval()
    
    return sae


def compare_saes(
    control_sae_path: str,
    saeception_sae_path: str,
    control_model_path: Optional[str] = None,
    saeception_model_path: Optional[str] = None,
    base_model_name: str = 'EleutherAI/pythia-70m',
    layer_idx: int = 3,
    control_history_path: Optional[str] = None,
    saeception_history_path: Optional[str] = None,
    n_absorption_candidates: Optional[int] = None,
    max_absorption_samples: int = 2000,
    device: str = 'cuda',
) -> Dict:
    """
    Compare two SAEs trained on control vs SAE-ception models.
    
    Args:
        control_sae_path: Path to control SAE checkpoint
        saeception_sae_path: Path to SAE-ception SAE checkpoint
        control_model_path: Path to control model checkpoint (for full analysis)
        saeception_model_path: Path to SAE-ception model checkpoint (for full analysis)
        base_model_name: Base model name for loading architecture
        layer_idx: Layer to extract activations from
        control_history_path: Optional training history JSON
        saeception_history_path: Optional training history JSON
        n_absorption_candidates: Number of latent candidates for absorption (None = all)
        max_absorption_samples: Max sequences for absorption analysis
        device: Device for computation
        
    Returns:
        Dictionary with comparison metrics
    """
    results = {
        'control': {},
        'saeception': {},
        'comparison': {},
    }
    
    # Load diagnostics (basic metrics from training)
    logger.info("Loading control SAE diagnostics...")
    control_diag = load_diagnostics(control_sae_path)
    control_info = load_checkpoint_info(control_sae_path)
    
    logger.info("Loading SAE-ception SAE diagnostics...")
    saeception_diag = load_diagnostics(saeception_sae_path)
    saeception_info = load_checkpoint_info(saeception_sae_path)
    
    # Load SAE models
    logger.info("Loading SAE models...")
    control_sae = load_sae_from_checkpoint(control_sae_path, device)
    saeception_sae = load_sae_from_checkpoint(saeception_sae_path, device)
    
    # Core metrics from diagnostics
    for name, diag, info in [('control', control_diag, control_info), 
                              ('saeception', saeception_diag, saeception_info)]:
        results[name]['l0'] = diag.get('l0', info['training'].get('best_l0'))
        results[name]['dead_pct'] = diag.get('dead_pct')
        results[name]['relative_error'] = diag.get('relative_error')
        results[name]['cosine_similarity'] = diag.get('cosine_similarity')
        results[name]['mse'] = diag.get('mse')
    
    # Decoder geometry analysis
    logger.info("Analyzing decoder weight geometry...")
    results['control']['geometry'] = compute_decoder_geometry(control_sae)
    results['saeception']['geometry'] = compute_decoder_geometry(saeception_sae)
    
    # ==========================================================================
    # FULL ANALYSIS (requires model checkpoints)
    # ==========================================================================
    
    if control_model_path and saeception_model_path:
        logger.info("=" * 60)
        logger.info("Running full analysis with model activations...")
        logger.info("=" * 60)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load models and extract activations
        logger.info("\nExtracting control model activations...")
        control_model = load_model_from_checkpoint(control_model_path, base_model_name, device)
        control_acts, control_token_ids = extract_token_activations_for_absorption(
            control_model, tokenizer, layer_idx, device, max_absorption_samples
        )
        del control_model  # Free memory
        torch.cuda.empty_cache()
        
        logger.info("\nExtracting SAE-ception model activations...")
        saeception_model = load_model_from_checkpoint(saeception_model_path, base_model_name, device)
        saeception_acts, saeception_token_ids = extract_token_activations_for_absorption(
            saeception_model, tokenizer, layer_idx, device, max_absorption_samples
        )
        del saeception_model  # Free memory
        torch.cuda.empty_cache()
        
        # Co-activation analysis
        logger.info("\nAnalyzing control SAE co-activation patterns...")
        results['control']['coactivation'] = compute_coactivation_patterns(
            control_sae, control_acts, device
        )
        
        logger.info("Analyzing SAE-ception SAE co-activation patterns...")
        results['saeception']['coactivation'] = compute_coactivation_patterns(
            saeception_sae, saeception_acts, device
        )
        
        # SAE quality metrics
        logger.info("\nComputing full SAE quality metrics (control)...")
        results['control']['quality'] = evaluate_sae_quality(
            control_sae, control_acts, device
        )
        
        logger.info("Computing full SAE quality metrics (SAE-ception)...")
        results['saeception']['quality'] = evaluate_sae_quality(
            saeception_sae, saeception_acts, device
        )
        
        # ======================================================================
        # FEATURE ABSORPTION ANALYSIS (SAEBench first-letter)
        # ======================================================================
        logger.info("\n" + "=" * 60)
        logger.info("FEATURE ABSORPTION ANALYSIS")
        logger.info("=" * 60)
        
        logger.info("\nComputing absorption for control SAE...")
        try:
            control_absorption = compute_first_letter_absorption(
                sae=control_sae,
                activations=control_acts,
                token_ids=control_token_ids,
                tokenizer=tokenizer,
                device=device,
                n_latent_candidates=n_absorption_candidates,
                show_progress=True,
            )
            results['control']['absorption'] = control_absorption
            logger.info(f"Control absorption: {control_absorption.get('absorption', 'N/A'):.4f}")
        except Exception as e:
            logger.error(f"Control absorption failed: {e}")
            results['control']['absorption'] = {'error': str(e)}
        
        logger.info("\nComputing absorption for SAE-ception SAE...")
        try:
            saeception_absorption = compute_first_letter_absorption(
                sae=saeception_sae,
                activations=saeception_acts,
                token_ids=saeception_token_ids,
                tokenizer=tokenizer,
                device=device,
                n_latent_candidates=n_absorption_candidates,
                show_progress=True,
            )
            results['saeception']['absorption'] = saeception_absorption
            logger.info(f"SAE-ception absorption: {saeception_absorption.get('absorption', 'N/A'):.4f}")
        except Exception as e:
            logger.error(f"SAE-ception absorption failed: {e}")
            results['saeception']['absorption'] = {'error': str(e)}
    
    # Convergence analysis from training histories
    control_history = load_training_history(control_history_path)
    saeception_history = load_training_history(saeception_history_path)
    
    if control_history:
        results['control']['convergence'] = compute_convergence_metrics(control_history)
    
    if saeception_history:
        results['saeception']['convergence'] = compute_convergence_metrics(saeception_history)
    
    # ==========================================================================
    # COMPUTE COMPARISONS
    # ==========================================================================
    
    # L0 comparison
    if results['control']['l0'] is not None and results['saeception']['l0'] is not None:
        results['comparison']['l0_diff'] = results['saeception']['l0'] - results['control']['l0']
        results['comparison']['l0_reduction_pct'] = (
            (results['control']['l0'] - results['saeception']['l0']) / results['control']['l0'] * 100
        )
    
    # Dead features comparison
    if results['control']['dead_pct'] is not None and results['saeception']['dead_pct'] is not None:
        results['comparison']['dead_pct_diff'] = results['saeception']['dead_pct'] - results['control']['dead_pct']
    
    # Geometry comparison
    ctrl_geo = results['control'].get('geometry', {})
    sae_geo = results['saeception'].get('geometry', {})
    
    if ctrl_geo and sae_geo:
        results['comparison']['geometry'] = {
            'cos_sim_mean_diff': sae_geo.get('cos_sim_mean', 0) - ctrl_geo.get('cos_sim_mean', 0),
            'cluster_silhouette_diff': sae_geo.get('cluster_silhouette', 0) - ctrl_geo.get('cluster_silhouette', 0),
        }
    
    # Co-activation comparison
    ctrl_coact = results['control'].get('coactivation', {})
    sae_coact = results['saeception'].get('coactivation', {})
    
    if ctrl_coact and sae_coact:
        results['comparison']['coactivation'] = {
            'correlation_mean_diff': sae_coact.get('correlation_mean', 0) - ctrl_coact.get('correlation_mean', 0),
            'effective_features_diff': sae_coact.get('effective_features', 0) - ctrl_coact.get('effective_features', 0),
            'firing_gini_diff': sae_coact.get('firing_gini', 0) - ctrl_coact.get('firing_gini', 0),
        }
    
    # Absorption comparison
    ctrl_abs = results['control'].get('absorption', {})
    sae_abs = results['saeception'].get('absorption', {})
    
    if ctrl_abs and sae_abs and 'absorption' in ctrl_abs and 'absorption' in sae_abs:
        results['comparison']['absorption'] = {
            'absorption_diff': sae_abs['absorption'] - ctrl_abs['absorption'],
            'absorption_reduction_pct': (ctrl_abs['absorption'] - sae_abs['absorption']) / (ctrl_abs['absorption'] + 1e-8) * 100,
        }
    
    # Convergence comparison
    if 'convergence' in results['control'] and 'convergence' in results['saeception']:
        ctrl_conv = results['control']['convergence']
        sae_conv = results['saeception']['convergence']
        
        results['comparison']['convergence_speedup'] = (
            ctrl_conv.get('epochs_to_90pct', 0) - sae_conv.get('epochs_to_90pct', 0)
        )
        results['comparison']['improvement_rate_ratio'] = (
            sae_conv.get('improvement_rate', 0) / (ctrl_conv.get('improvement_rate', 1e-10))
        )
    
    return results


def print_comparison_table(results: Dict):
    """Print a formatted comparison table."""
    
    print("\n" + "=" * 80)
    print("STRUCTURE CHECK: SAE COMPARISON")
    print("=" * 80)
    
    headers = ["Metric", "Control", "SAE-ception", "Δ", "Better?"]
    
    # ==========================================================================
    # CORE METRICS
    # ==========================================================================
    print("\n📊 CORE METRICS")
    print("-" * 80)
    
    table_data = []
    metrics = [
        ('L0 Sparsity', 'l0', '.1f', 'lower'),
        ('Dead Features %', 'dead_pct', '.1f', 'lower'),
        ('Relative Error', 'relative_error', '.2%', 'lower'),
        ('Cosine Similarity', 'cosine_similarity', '.4f', 'higher'),
        ('MSE', 'mse', '.6f', 'lower'),
    ]
    
    for name, key, fmt, better in metrics:
        ctrl_val = results['control'].get(key)
        sae_val = results['saeception'].get(key)
        
        if ctrl_val is not None and sae_val is not None:
            diff = sae_val - ctrl_val
            is_better = diff < 0 if better == 'lower' else diff > 0
            indicator = "✓" if is_better else "✗"
            table_data.append([name, f"{ctrl_val:{fmt}}", f"{sae_val:{fmt}}", f"{diff:+{fmt}}", indicator])
        else:
            table_data.append([name, "N/A", "N/A", "N/A", "-"])
    
    print(tabulate(table_data, headers=headers, tablefmt="simple"))
    
    # ==========================================================================
    # DECODER GEOMETRY
    # ==========================================================================
    ctrl_geo = results['control'].get('geometry', {})
    sae_geo = results['saeception'].get('geometry', {})
    
    if ctrl_geo and sae_geo:
        print("\n🔷 DECODER WEIGHT GEOMETRY")
        print("-" * 80)
        
        geo_data = []
        geo_metrics = [
            ('Mean Cosine Sim', 'cos_sim_mean', '.4f', 'higher'),  # Tighter clusters = higher similarity
            ('Cluster Silhouette', 'cluster_silhouette', '.4f', 'higher'),
            ('Cluster Size Gini', 'cluster_size_gini', '.4f', 'higher'),  # More concentrated = higher
        ]
        
        for name, key, fmt, better in geo_metrics:
            ctrl_val = ctrl_geo.get(key)
            sae_val = sae_geo.get(key)
            
            if ctrl_val is not None and sae_val is not None:
                diff = sae_val - ctrl_val
                is_better = diff < 0 if better == 'lower' else diff > 0
                indicator = "✓" if is_better else "✗"
                geo_data.append([name, f"{ctrl_val:{fmt}}", f"{sae_val:{fmt}}", f"{diff:+{fmt}}", indicator])
        
        print(tabulate(geo_data, headers=headers, tablefmt="simple"))
    
    # ==========================================================================
    # CO-ACTIVATION PATTERNS
    # ==========================================================================
    ctrl_coact = results['control'].get('coactivation', {})
    sae_coact = results['saeception'].get('coactivation', {})
    
    if ctrl_coact and sae_coact:
        print("\n🔗 FEATURE CO-ACTIVATION")
        print("-" * 80)
        
        coact_data = []
        coact_metrics = [
            ('Firing Gini (concentration)', 'firing_gini', '.4f', 'higher'),  # More absorption = higher
            ('Effective Features', 'effective_features', 'd', 'lower'),  # Fewer = more concentrated
            ('Effective Features %', 'effective_features_pct', '.1f', 'lower'),
            ('Mean Correlation', 'correlation_mean', '.4f', 'context'),  # Context-dependent
            ('High Corr Pairs (>0.5)', 'high_corr_pairs', 'd', 'context'),
        ]
        
        for name, key, fmt, better in coact_metrics:
            ctrl_val = ctrl_coact.get(key)
            sae_val = sae_coact.get(key)
            
            if ctrl_val is not None and sae_val is not None:
                diff = sae_val - ctrl_val
                if better == 'context':
                    indicator = "~"
                else:
                    is_better = diff < 0 if better == 'lower' else diff > 0
                    indicator = "✓" if is_better else "✗"
                coact_data.append([name, f"{ctrl_val:{fmt}}", f"{sae_val:{fmt}}", f"{diff:+{fmt}}", indicator])
        
        print(tabulate(coact_data, headers=headers, tablefmt="simple"))
    
    # ==========================================================================
    # FEATURE ABSORPTION (NEW!)
    # ==========================================================================
    ctrl_abs = results['control'].get('absorption', {})
    sae_abs = results['saeception'].get('absorption', {})
    
    if ctrl_abs and sae_abs and 'error' not in ctrl_abs and 'error' not in sae_abs:
        print("\n🎯 FEATURE ABSORPTION (SAEBench First-Letter)")
        print("-" * 80)
        
        abs_data = []
        abs_metrics = [
            ('Absorption Score', 'absorption', '.4f', 'lower'),  # Lower = features stay clean
            ('Absorption Complement', 'absorption_complement', '.4f', 'higher'),  # Higher = better
            ('Letters Evaluated', 'n_letters_evaluated', 'd', 'context'),
            ('Test Samples', 'n_test_samples', 'd', 'context'),
        ]
        
        for name, key, fmt, better in abs_metrics:
            ctrl_val = ctrl_abs.get(key)
            sae_val = sae_abs.get(key)
            
            if ctrl_val is not None and sae_val is not None:
                diff = sae_val - ctrl_val
                if better == 'context':
                    indicator = "~"
                else:
                    is_better = diff < 0 if better == 'lower' else diff > 0
                    indicator = "✓" if is_better else "✗"
                abs_data.append([name, f"{ctrl_val:{fmt}}", f"{sae_val:{fmt}}", f"{diff:+{fmt}}", indicator])
        
        print(tabulate(abs_data, headers=headers, tablefmt="simple"))
        
        # Show absorption reduction percentage
        if 'absorption' in results.get('comparison', {}).get('absorption', {}):
            reduction = results['comparison']['absorption']['absorption_reduction_pct']
            print(f"\n  Absorption reduction: {reduction:+.1f}%")
    
    elif ctrl_abs.get('error') or sae_abs.get('error'):
        print("\n🎯 FEATURE ABSORPTION")
        print("-" * 80)
        print(f"  Control error: {ctrl_abs.get('error', 'None')}")
        print(f"  SAE-ception error: {sae_abs.get('error', 'None')}")
    
    # ==========================================================================
    # CONVERGENCE ANALYSIS
    # ==========================================================================
    if 'convergence' in results['control'] and 'convergence' in results['saeception']:
        print("\n⏱️  CONVERGENCE SPEED")
        print("-" * 80)
        
        conv_data = []
        ctrl_conv = results['control']['convergence']
        sae_conv = results['saeception']['convergence']
        
        conv_metrics = [
            ('Epochs to 90% improvement', 'epochs_to_90pct', 'd', 'lower'),
            ('Initial Loss', 'initial_loss', '.4f', 'lower'),
            ('Final Loss', 'final_loss', '.4f', 'lower'),
            ('Improvement Rate', 'improvement_rate', '.6f', 'higher'),
        ]
        
        for name, key, fmt, better in conv_metrics:
            ctrl_val = ctrl_conv.get(key)
            sae_val = sae_conv.get(key)
            
            if ctrl_val is not None and sae_val is not None:
                diff = sae_val - ctrl_val
                is_better = diff < 0 if better == 'lower' else diff > 0
                indicator = "✓" if is_better else "✗"
                conv_data.append([name, f"{ctrl_val:{fmt}}", f"{sae_val:{fmt}}", f"{diff:+{fmt}}", indicator])
        
        print(tabulate(conv_data, headers=headers, tablefmt="simple"))
    
    # ==========================================================================
    # HYPOTHESIS SUMMARY
    # ==========================================================================
    print("\n" + "=" * 80)
    print("📋 HYPOTHESIS CHECK: Is SAE on SAE-ception model 'lazier'?")
    print("=" * 80)
    
    hypothesis_checks = []
    
    # Check 1: Lower L0
    l0_reduction = results['comparison'].get('l0_reduction_pct', 0)
    hypothesis_checks.append(('Lower L0 (fewer features needed)', l0_reduction > 0, f"{l0_reduction:+.1f}%"))
    
    # Check 2: Dead features
    dead_diff = results['comparison'].get('dead_pct_diff', 0)
    hypothesis_checks.append(('Fewer dead features', dead_diff < 0, f"{dead_diff:+.1f}%"))
    
    # Check 3: Tighter clusters (if available)
    if ctrl_geo and sae_geo:
        silhouette_diff = sae_geo.get('cluster_silhouette', 0) - ctrl_geo.get('cluster_silhouette', 0)
        hypothesis_checks.append(('Tighter decoder clusters', silhouette_diff > 0, f"{silhouette_diff:+.4f}"))
    
    # Check 4: More concentrated features (if available)
    if ctrl_coact and sae_coact:
        gini_diff = sae_coact.get('firing_gini', 0) - ctrl_coact.get('firing_gini', 0)
        hypothesis_checks.append(('Higher feature concentration', gini_diff > 0, f"{gini_diff:+.4f}"))
        
        eff_diff = sae_coact.get('effective_features', 0) - ctrl_coact.get('effective_features', 0)
        hypothesis_checks.append(('Fewer effective features', eff_diff < 0, f"{eff_diff:+d}"))
    
    # Check 5: Lower absorption (NEW!)
    if ctrl_abs and sae_abs and 'absorption' in ctrl_abs and 'absorption' in sae_abs:
        abs_diff = sae_abs['absorption'] - ctrl_abs['absorption']
        hypothesis_checks.append(('Lower feature absorption', abs_diff < 0, f"{abs_diff:+.4f}"))
    
    # Check 6: Faster convergence
    if 'convergence_speedup' in results['comparison']:
        speedup = results['comparison']['convergence_speedup']
        hypothesis_checks.append(('Faster convergence', speedup > 0, f"{speedup:+d} epochs"))
    
    print()
    n_passed = 0
    for check_name, passed, value in hypothesis_checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}: {value}")
        if passed:
            n_passed += 1
    
    print("\n" + "-" * 80)
    pct_passed = n_passed / len(hypothesis_checks) * 100 if hypothesis_checks else 0
    
    if pct_passed >= 60:
        print(f"  ✅ HYPOTHESIS SUPPORTED ({n_passed}/{len(hypothesis_checks)} checks passed)")
        print("     SAE on SAE-ception model shows signs of learning over structured features")
    elif pct_passed >= 40:
        print(f"  ⚠️  MIXED RESULTS ({n_passed}/{len(hypothesis_checks)} checks passed)")
        print("     Some evidence for hypothesis, may need tuning")
    else:
        print(f"  ❌ HYPOTHESIS NOT SUPPORTED ({n_passed}/{len(hypothesis_checks)} checks passed)")
        print("     Check training parameters or model quality")
    
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare SAEs trained on control vs SAE-ception models"
    )
    
    parser.add_argument(
        '--control_sae',
        type=str,
        required=True,
        help='Path to SAE trained on control model'
    )
    parser.add_argument(
        '--saeception_sae',
        type=str,
        required=True,
        help='Path to SAE trained on SAE-ception model'
    )
    parser.add_argument(
        '--control_model',
        type=str,
        default=None,
        help='Path to control model checkpoint (enables full analysis including absorption)'
    )
    parser.add_argument(
        '--saeception_model',
        type=str,
        default=None,
        help='Path to SAE-ception model checkpoint (enables full analysis including absorption)'
    )
    parser.add_argument(
        '--base_model',
        type=str,
        default='EleutherAI/pythia-70m',
        help='Base model name for architecture (default: EleutherAI/pythia-70m)'
    )
    parser.add_argument(
        '--layer',
        type=int,
        default=3,
        help='Layer to extract activations from (default: 3)'
    )
    parser.add_argument(
        '--control_history',
        type=str,
        default=None,
        help='Path to control SAE training history JSON'
    )
    parser.add_argument(
        '--saeception_history',
        type=str,
        default=None,
        help='Path to SAE-ception SAE training history JSON'
    )
    parser.add_argument(
        '--n_absorption_candidates',
        type=int,
        default=None,
        help='Number of latent candidates for absorption (None = all, 50-100 for quick test)'
    )
    parser.add_argument(
        '--max_absorption_samples',
        type=int,
        default=2000,
        help='Max sequences for absorption analysis (default: 2000)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for results JSON'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (default: cuda)'
    )
    
    args = parser.parse_args()
    
    # Run comparison
    results = compare_saes(
        control_sae_path=args.control_sae,
        saeception_sae_path=args.saeception_sae,
        control_model_path=args.control_model,
        saeception_model_path=args.saeception_model,
        base_model_name=args.base_model,
        layer_idx=args.layer,
        control_history_path=args.control_history,
        saeception_history_path=args.saeception_history,
        n_absorption_candidates=args.n_absorption_candidates,
        max_absorption_samples=args.max_absorption_samples,
        device=args.device,
    )
    
    # Print table
    print_comparison_table(results)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    return results


if __name__ == "__main__":
    main()