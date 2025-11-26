"""Interpretability metrics for evaluating monosemanticity and feature quality."""

import torch
import numpy as np
from typing import Dict, Optional
from sklearn.metrics import mutual_info_score


def class_selectivity_index(
    activations: torch.Tensor,
    labels: torch.Tensor,
    num_classes: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute Class Selectivity Index (CSI) for features.
    
    CSI measures how selective each feature is to a single class.
    Higher values (closer to 1) indicate more monosemantic features.
    
    Formula for feature i:
        CSI_i = (μ_max - μ_other) / (μ_max + μ_other)
    
    where:
        μ_max = mean activation for the maximally activating class
        μ_other = mean activation across all other classes
    
    Args:
        activations: Shape [N, hidden_dim] - feature activations
        labels: Shape [N] - class labels
        num_classes: Number of classes (inferred if None)
        
    Returns:
        Dictionary with:
            - mean_csi: Average CSI across all features
            - std_csi: Standard deviation of CSI
            - per_feature_csi: CSI for each feature [hidden_dim]
    """
    if num_classes is None:
        num_classes = labels.max().item() + 1
    
    N, hidden_dim = activations.shape
    csi_scores = []
    
    # Convert to numpy for easier indexing
    acts = activations.numpy() if isinstance(activations, torch.Tensor) else activations
    labs = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    
    for feat_idx in range(hidden_dim):
        feature_acts = acts[:, feat_idx]
        
        # Compute mean activation per class
        class_means = np.zeros(num_classes)
        for c in range(num_classes):
            class_mask = labs == c
            if class_mask.sum() > 0:
                class_means[c] = feature_acts[class_mask].mean()
        
        # Find max activating class
        max_class = class_means.argmax()
        mu_max = class_means[max_class]
        
        # Mean activation for all other classes
        other_mask = labs != max_class
        if other_mask.sum() > 0:
            mu_other = feature_acts[other_mask].mean()
        else:
            mu_other = 0.0
        
        # Compute CSI (handle division by zero)
        if mu_max + mu_other > 1e-8:
            csi = (mu_max - mu_other) / (mu_max + mu_other)
        else:
            csi = 0.0
        
        csi_scores.append(csi)
    
    csi_scores = np.array(csi_scores)
    
    return {
        'mean_csi': float(csi_scores.mean()),
        'std_csi': float(csi_scores.std()),
        'per_feature_csi': csi_scores,
    }


def uncertainty_coefficient(
    activations: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.0,
    num_classes: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute Uncertainty Coefficient (U) for sparse features.
    
    U measures the normalized mutual information between feature firing
    events and class labels. Better for sparse features than CSI.
    
    U ranges from 0 (no information) to 1 (perfect prediction).
    
    Args:
        activations: Shape [N, hidden_dim] - sparse feature activations
        labels: Shape [N] - class labels
        threshold: Threshold for considering a feature "active" (default: 0)
        num_classes: Number of classes (inferred if None)
        
    Returns:
        Dictionary with:
            - mean_u: Average U across all features
            - std_u: Standard deviation of U
            - per_feature_u: U for each feature [hidden_dim]
    """
    if num_classes is None:
        num_classes = labels.max().item() + 1
    
    N, hidden_dim = activations.shape
    u_scores = []
    
    # Convert to numpy
    acts = activations.numpy() if isinstance(activations, torch.Tensor) else activations
    labs = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    
    for feat_idx in range(hidden_dim):
        feature_acts = acts[:, feat_idx]
        
        # Binarize: is feature active?
        feature_active = (feature_acts > threshold).astype(int)
        
        # Skip features that never fire or always fire
        if feature_active.sum() == 0 or feature_active.sum() == N:
            u_scores.append(0.0)
            continue
        
        # Compute mutual information
        mi = mutual_info_score(feature_active, labs)
        
        # Normalize by entropy of labels
        # U(Y|X) = I(X;Y) / H(Y)
        label_counts = np.bincount(labs, minlength=num_classes)
        label_probs = label_counts / label_counts.sum()
        label_probs = label_probs[label_probs > 0]  # Remove zeros
        h_labels = -np.sum(label_probs * np.log2(label_probs))
        
        if h_labels > 0:
            u = mi / h_labels
        else:
            u = 0.0
        
        u_scores.append(u)
    
    u_scores = np.array(u_scores)
    
    return {
        'mean_u': float(u_scores.mean()),
        'std_u': float(u_scores.std()),
        'per_feature_u': u_scores,
    }


def compute_monosemanticity_metrics(
    activations: torch.Tensor,
    labels: torch.Tensor,
    num_classes: Optional[int] = None,
    threshold: float = 0.0,
) -> Dict[str, float]:
    """
    Compute all monosemanticity metrics.
    
    Args:
        activations: Shape [N, hidden_dim]
        labels: Shape [N]
        num_classes: Number of classes
        threshold: Threshold for U computation
        
    Returns:
        Dictionary with CSI and U metrics
    """
    csi_results = class_selectivity_index(activations, labels, num_classes)
    u_results = uncertainty_coefficient(activations, labels, threshold, num_classes)
    
    return {
        'csi': csi_results['mean_csi'],
        'csi_std': csi_results['std_csi'],
        'u': u_results['mean_u'],
        'u_std': u_results['std_u'],
    }


def compute_sparsity_metrics(
    activations: torch.Tensor,
    threshold: float = 0.0,
) -> Dict[str, float]:
    """
    Compute sparsity metrics for activations.
    
    Args:
        activations: Shape [N, hidden_dim]
        threshold: Threshold for considering a feature active
        
    Returns:
        Dictionary with:
            - l0_mean: Average number of active features per example
            - l0_std: Standard deviation of L0
            - l1_mean: Average L1 norm
    """
    # L0 norm (number of active features)
    active = (activations.abs() > threshold).float()
    l0_per_example = active.sum(dim=-1)
    
    # L1 norm
    l1_per_example = activations.abs().sum(dim=-1)
    
    return {
        'l0_mean': float(l0_per_example.mean()),
        'l0_std': float(l0_per_example.std()),
        'l1_mean': float(l1_per_example.mean()),
    }


def evaluate_sae_quality(
    sae: torch.nn.Module,
    activations: torch.Tensor,
    device: str = 'cuda',
) -> Dict[str, float]:
    """
    Evaluate SAE reconstruction quality and feature statistics.
    
    Args:
        sae: Trained SAE
        activations: Original activations [N, hidden_dim]
        device: Device to run on
        
    Returns:
        Dictionary with:
            - reconstruction_loss: MSE between input and reconstruction
            - explained_variance: Proportion of variance explained
            - mean_l0: Average sparsity of codes
            - dead_features_pct: Percentage of features that never activate
            - feature_usage_entropy: Entropy of feature usage distribution (in nats)
    """
    sae.eval()
    sae.to(device)
    
    all_recon_loss = []
    all_explained_var = []
    all_l0 = []
    all_sparse_codes = []
    
    # Process in batches to avoid memory issues
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(activations), batch_size):
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
            
            # Collect sparse codes for feature statistics
            all_sparse_codes.append(sparse_code.cpu())
    
    # Concatenate all sparse codes
    all_sparse_codes = torch.cat(all_sparse_codes, dim=0)  # [N, hidden_dim]
    
    # Compute dead features percentage
    # A feature is "dead" if it never fires (always zero) across the entire dataset
    feature_ever_active = (all_sparse_codes > 0).any(dim=0)  # [hidden_dim]
    dead_features_pct = 100.0 * (~feature_ever_active).float().mean().item()
    
    # Compute feature usage entropy
    # How evenly are features being used?
    feature_fire_counts = (all_sparse_codes > 0).float().sum(dim=0)  # [hidden_dim]
    total_fires = feature_fire_counts.sum()
    
    if total_fires > 0:
        # Normalize to get probability distribution
        feature_usage_probs = feature_fire_counts / total_fires
        # Remove zeros to avoid log(0)
        feature_usage_probs = feature_usage_probs[feature_usage_probs > 0]
        # Compute entropy (in nats)
        feature_usage_entropy = -(feature_usage_probs * torch.log(feature_usage_probs)).sum().item()
    else:
        feature_usage_entropy = 0.0
    
    return {
        'reconstruction_loss': float(np.mean(all_recon_loss)),
        'explained_variance': float(np.mean(all_explained_var)),
        'mean_l0': float(np.mean(all_l0)),
        'dead_features_pct': float(dead_features_pct),
        'feature_usage_entropy': float(feature_usage_entropy),
    }