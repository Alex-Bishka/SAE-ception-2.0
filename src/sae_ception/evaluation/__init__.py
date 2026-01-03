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
    k_sparse_probe_accuracy,
)

from .saebench import (
    compute_first_letter_absorption,
    evaluate_sae_saebench,
)

from .lm_metrics import (
    evaluate_perplexity,
    evaluate_perplexity_with_intervention,
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
    'k_sparse_probe_accuracy',
    # SAEBench
    'compute_first_letter_absorption',
    'evaluate_sae_saebench',
    # LM Metrics
    'evaluate_perplexity',
    'evaluate_perplexity_with_intervention',
]