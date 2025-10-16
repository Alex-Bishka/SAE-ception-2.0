"""Evaluation module for SAE-ception experiments."""

from .interpretability import (
    class_selectivity_index,
    uncertainty_coefficient,
    compute_monosemanticity_metrics,
    compute_sparsity_metrics,
    evaluate_sae_quality,
)

from .clustering import (
    adjusted_rand_index,
    silhouette_unsupervised,
    silhouette_supervised,
    davies_bouldin_index,
    calinski_harabasz_index,
    compute_all_clustering_metrics,
    compute_within_class_variance,
)

from .performance import (
    evaluate_classification_accuracy,
    train_linear_probe,
    evaluate_perplexity,
    compare_models,
    evaluate_model_and_sae,
)


__all__ = [
    # Interpretability
    'class_selectivity_index',
    'uncertainty_coefficient',
    'compute_monosemanticity_metrics',
    'compute_sparsity_metrics',
    'evaluate_sae_quality',
    # Clustering
    'adjusted_rand_index',
    'silhouette_unsupervised',
    'silhouette_supervised',
    'davies_bouldin_index',
    'calinski_harabasz_index',
    'compute_all_clustering_metrics',
    'compute_within_class_variance',
    # Performance
    'evaluate_classification_accuracy',
    'train_linear_probe',
    'evaluate_perplexity',
    'compare_models',
    'evaluate_model_and_sae',
]