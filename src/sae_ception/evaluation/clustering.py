"""Clustering metrics for evaluating feature separation and organization."""

import torch
import numpy as np
from typing import Dict
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.cluster import KMeans


def adjusted_rand_index(
    activations: torch.Tensor,
    labels: torch.Tensor,
    num_clusters: int = None,
) -> float:
    """
    Compute Adjusted Rand Index (ARI).
    
    ARI measures agreement between clustering and ground-truth labels.
    Ranges from -1 to 1, where:
        - 1 = perfect agreement
        - 0 = random clustering
        - negative = worse than random
    
    Args:
        activations: Shape [N, hidden_dim]
        labels: Ground truth labels [N]
        num_clusters: Number of clusters (default: num unique labels)
        
    Returns:
        ARI score
    """
    if num_clusters is None:
        num_clusters = len(torch.unique(labels))
    
    # Convert to numpy
    X = activations.numpy() if isinstance(activations, torch.Tensor) else activations
    y_true = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    
    # Perform clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X)
    
    # Compute ARI
    return float(adjusted_rand_score(y_true, y_pred))


def silhouette_unsupervised(
    activations: torch.Tensor,
    num_clusters: int = None,
    labels: torch.Tensor = None,
) -> float:
    """
    Compute Silhouette score using unsupervised clustering.
    
    Measures how well-separated clusters are.
    Ranges from -1 to 1, where:
        - 1 = clusters are well-separated
        - 0 = clusters overlap
        - negative = points may be in wrong cluster
    
    Args:
        activations: Shape [N, hidden_dim]
        num_clusters: Number of clusters (inferred from labels if not provided)
        labels: Optional ground truth labels to determine num_clusters
        
    Returns:
        Silhouette score
    """
    if num_clusters is None:
        if labels is None:
            raise ValueError("Must provide either num_clusters or labels")
        num_clusters = len(torch.unique(labels))
    
    # Convert to numpy
    X = activations.numpy() if isinstance(activations, torch.Tensor) else activations
    
    # Perform clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Compute silhouette score
    if len(np.unique(cluster_labels)) > 1:
        return float(silhouette_score(X, cluster_labels))
    else:
        return 0.0


def silhouette_supervised(
    activations: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Compute Silhouette score using ground truth labels.
    
    Measures how compact and well-separated classes are in feature space.
    
    Args:
        activations: Shape [N, hidden_dim]
        labels: Ground truth labels [N]
        
    Returns:
        Silhouette score
    """
    # Convert to numpy
    X = activations.numpy() if isinstance(activations, torch.Tensor) else activations
    y = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    
    # Compute silhouette score using true labels
    if len(np.unique(y)) > 1:
        return float(silhouette_score(X, y))
    else:
        return 0.0


def davies_bouldin_index(
    activations: torch.Tensor,
    labels: torch.Tensor = None,
    num_clusters: int = None,
) -> float:
    """
    Compute Davies-Bouldin Index (DBI).
    
    Measures average similarity between each cluster and its most similar cluster.
    Lower values indicate better clustering (more compact, well-separated).
    
    Args:
        activations: Shape [N, hidden_dim]
        labels: Optional ground truth labels
        num_clusters: Number of clusters (required if labels not provided)
        
    Returns:
        DBI score (lower is better)
    """
    # Convert to numpy
    X = activations.numpy() if isinstance(activations, torch.Tensor) else activations
    
    if labels is not None:
        # Use ground truth labels
        y = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    else:
        # Perform clustering
        if num_clusters is None:
            raise ValueError("Must provide either labels or num_clusters")
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        y = kmeans.fit_predict(X)
    
    # Compute DBI
    if len(np.unique(y)) > 1:
        return float(davies_bouldin_score(X, y))
    else:
        return float('inf')


def calinski_harabasz_index(
    activations: torch.Tensor,
    labels: torch.Tensor = None,
    num_clusters: int = None,
) -> float:
    """
    Compute Calinski-Harabasz Index (CHI).
    
    Ratio of between-cluster dispersion to within-cluster dispersion.
    Higher values indicate better-defined, denser clusters.
    
    Args:
        activations: Shape [N, hidden_dim]
        labels: Optional ground truth labels
        num_clusters: Number of clusters (required if labels not provided)
        
    Returns:
        CHI score (higher is better)
    """
    # Convert to numpy
    X = activations.numpy() if isinstance(activations, torch.Tensor) else activations
    
    if labels is not None:
        # Use ground truth labels
        y = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    else:
        # Perform clustering
        if num_clusters is None:
            raise ValueError("Must provide either labels or num_clusters")
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        y = kmeans.fit_predict(X)
    
    # Compute CHI
    if len(np.unique(y)) > 1:
        return float(calinski_harabasz_score(X, y))
    else:
        return 0.0


def compute_all_clustering_metrics(
    activations: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute all clustering metrics.
    
    Args:
        activations: Shape [N, hidden_dim]
        labels: Ground truth labels [N]
        
    Returns:
        Dictionary with all clustering metrics
    """
    num_clusters = len(torch.unique(labels))
    
    results = {
        'ari': adjusted_rand_index(activations, labels, num_clusters),
        'silhouette_unsupervised': silhouette_unsupervised(activations, num_clusters, labels),
        'silhouette_supervised': silhouette_supervised(activations, labels),
        'davies_bouldin': davies_bouldin_index(activations, labels),
        'calinski_harabasz': calinski_harabasz_index(activations, labels),
    }
    
    return results


def compute_within_class_variance(
    activations: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute within-class variance metrics.
    
    Args:
        activations: Shape [N, hidden_dim]
        labels: Ground truth labels [N]
        
    Returns:
        Dictionary with variance metrics
    """
    num_classes = len(torch.unique(labels))
    
    # Convert to numpy
    X = activations.numpy() if isinstance(activations, torch.Tensor) else activations
    y = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    
    # Compute per-class statistics
    within_class_vars = []
    class_sizes = []
    
    for c in range(num_classes):
        mask = y == c
        if mask.sum() > 1:
            class_acts = X[mask]
            var = np.var(class_acts, axis=0).mean()
            within_class_vars.append(var)
            class_sizes.append(mask.sum())
    
    # Overall within-class variance (weighted by class size)
    total_samples = sum(class_sizes)
    weighted_var = sum(v * s for v, s in zip(within_class_vars, class_sizes)) / total_samples
    
    # Between-class variance
    class_means = []
    for c in range(num_classes):
        mask = y == c
        if mask.sum() > 0:
            class_means.append(X[mask].mean(axis=0))
    
    class_means = np.array(class_means)
    global_mean = X.mean(axis=0)
    between_class_var = np.var(class_means, axis=0).mean()
    
    return {
        'within_class_variance': float(weighted_var),
        'between_class_variance': float(between_class_var),
        'variance_ratio': float(between_class_var / (weighted_var + 1e-8)),
    }