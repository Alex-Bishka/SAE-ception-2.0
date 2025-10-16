#!/usr/bin/env python3
"""
Test script to verify all evaluation metrics work correctly.
Run this to ensure the evaluation pipeline is functioning.
"""

import torch
import numpy as np
from sae_ception.evaluation import (
    compute_monosemanticity_metrics,
    compute_all_clustering_metrics,
    compute_sparsity_metrics,
    train_linear_probe,
)


def generate_synthetic_data(
    n_samples: int = 1000,
    hidden_dim: int = 128,
    num_classes: int = 4,
    seed: int = 42,
):
    """Generate synthetic data for testing."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate random labels
    labels = torch.randint(0, num_classes, (n_samples,))
    
    # Generate activations with some class structure
    activations = torch.randn(n_samples, hidden_dim)
    
    # Add class-specific signal
    for c in range(num_classes):
        mask = labels == c
        # Define feature range for this class
        feature_start = c * 10
        feature_end = min((c + 1) * 10, hidden_dim)
        # Boolean mask + slice indexing works correctly in PyTorch
        activations[mask, feature_start:feature_end] += 2.0
    
    # Add some sparsity
    activations = torch.relu(activations)
    
    return activations, labels


def test_monosemanticity_metrics():
    """Test monosemanticity metrics."""
    print("=" * 60)
    print("Testing Monosemanticity Metrics")
    print("=" * 60)
    
    activations, labels = generate_synthetic_data()
    
    results = compute_monosemanticity_metrics(activations, labels, num_classes=4)
    
    print("\nResults:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    
    assert 0 <= results['csi'] <= 1, "CSI should be between 0 and 1"
    assert 0 <= results['u'] <= 1, "U should be between 0 and 1"
    
    print("\n✓ Monosemanticity metrics passed!")


def test_clustering_metrics():
    """Test clustering metrics."""
    print("\n" + "=" * 60)
    print("Testing Clustering Metrics")
    print("=" * 60)
    
    activations, labels = generate_synthetic_data()
    
    results = compute_all_clustering_metrics(activations, labels)
    
    print("\nResults:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    
    assert -1 <= results['ari'] <= 1, "ARI should be between -1 and 1"
    assert -1 <= results['silhouette_supervised'] <= 1, "Silhouette should be between -1 and 1"
    assert results['davies_bouldin'] >= 0, "DBI should be non-negative"
    assert results['calinski_harabasz'] >= 0, "CHI should be non-negative"
    
    print("\n✓ Clustering metrics passed!")


def test_sparsity_metrics():
    """Test sparsity metrics."""
    print("\n" + "=" * 60)
    print("Testing Sparsity Metrics")
    print("=" * 60)
    
    activations, _ = generate_synthetic_data()
    
    results = compute_sparsity_metrics(activations)
    
    print("\nResults:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    
    assert results['l0_mean'] > 0, "L0 should be positive"
    assert results['l1_mean'] > 0, "L1 should be positive"
    
    print("\n✓ Sparsity metrics passed!")


def test_linear_probe():
    """Test linear probe training."""
    print("\n" + "=" * 60)
    print("Testing Linear Probe")
    print("=" * 60)
    
    # Generate train and test data
    train_acts, train_labels = generate_synthetic_data(n_samples=800, seed=42)
    test_acts, test_labels = generate_synthetic_data(n_samples=200, seed=43)
    
    results = train_linear_probe(
        train_acts, train_labels,
        test_acts, test_labels,
    )
    
    print("\nResults:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    
    assert 0 <= results['train_accuracy'] <= 1, "Accuracy should be between 0 and 1"
    assert 0 <= results['test_accuracy'] <= 1, "Accuracy should be between 0 and 1"
    
    print("\n✓ Linear probe passed!")


def test_comparison_across_conditions():
    """Test that metrics can distinguish between good and bad representations."""
    print("\n" + "=" * 60)
    print("Testing Metric Sensitivity")
    print("=" * 60)
    
    # Generate well-separated data
    torch.manual_seed(42)
    n_samples = 400
    hidden_dim = 64
    num_classes = 4
    
    labels = torch.randint(0, num_classes, (n_samples,))
    
    # Good representation: clear class structure
    good_acts = torch.randn(n_samples, hidden_dim)
    for c in range(num_classes):
        mask = labels == c
        feature_start = c * 10
        feature_end = min((c + 1) * 10, hidden_dim)
        good_acts[mask, feature_start:feature_end] += 5.0  # Strong signal
    good_acts = torch.relu(good_acts)
    
    # Bad representation: no class structure
    bad_acts = torch.randn(n_samples, hidden_dim)
    bad_acts = torch.relu(bad_acts)
    
    # Compare monosemanticity
    good_mono = compute_monosemanticity_metrics(good_acts, labels, num_classes)
    bad_mono = compute_monosemanticity_metrics(bad_acts, labels, num_classes)
    
    print("\nMonosemanticity Comparison:")
    print(f"  Good CSI: {good_mono['csi']:.4f} | Bad CSI: {bad_mono['csi']:.4f}")
    print(f"  Good U:   {good_mono['u']:.4f} | Bad U:   {bad_mono['u']:.4f}")
    
    # Compare clustering
    good_clust = compute_all_clustering_metrics(good_acts, labels)
    bad_clust = compute_all_clustering_metrics(bad_acts, labels)
    
    print("\nClustering Comparison:")
    print(f"  Good ARI: {good_clust['ari']:.4f} | Bad ARI: {bad_clust['ari']:.4f}")
    print(f"  Good Silhouette: {good_clust['silhouette_supervised']:.4f} | Bad Silhouette: {bad_clust['silhouette_supervised']:.4f}")
    
    # Good representation should have better metrics
    assert good_mono['csi'] > bad_mono['csi'], "Good representation should have higher CSI"
    assert good_clust['ari'] > bad_clust['ari'], "Good representation should have higher ARI"
    assert good_clust['silhouette_supervised'] > bad_clust['silhouette_supervised'], "Good representation should have higher silhouette"
    
    print("\n✓ Metrics successfully distinguish good from bad representations!")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("SAE-ception Evaluation Metrics Test Suite")
    print("=" * 60)
    
    try:
        test_monosemanticity_metrics()
        test_clustering_metrics()
        test_sparsity_metrics()
        test_linear_probe()
        test_comparison_across_conditions()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nEvaluation metrics are working correctly.")
        print("You can now use them to evaluate your SAE-ception experiments.")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TEST FAILED!")
        print("=" * 60)
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()