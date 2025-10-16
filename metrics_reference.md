# SAE-ception Evaluation Metrics Reference

This document describes all evaluation metrics used in SAE-ception experiments.

## Quick Start

Test all metrics:
```bash
python scripts/test_metrics.py
```

## Metric Categories

### 1. Monosemanticity Metrics

These measure how "pure" or "single-concept" features are.

#### Class Selectivity Index (CSI)
- **Range**: -1 to 1 (typically 0 to 1)
- **Higher is better**
- **Interpretation**: Measures how selective each feature is to a single class
- **Formula**: `CSI = (μ_max - μ_other) / (μ_max + μ_other)`
  - `μ_max`: mean activation for maximally activating class
  - `μ_other`: mean activation across all other classes

**When to use**: Measuring monosemanticity of dense activations (model layers)

**Example values**:
- CSI = 0.8: Highly monosemantic (good!)
- CSI = 0.3: Polysemantic (mixed concepts)
- CSI = 0.0: No class selectivity

#### Uncertainty Coefficient (U)
- **Range**: 0 to 1
- **Higher is better**
- **Interpretation**: Normalized mutual information between feature firing and class labels
- **Formula**: `U = I(X;Y) / H(Y)`
  - Better for sparse features than CSI
  - Measures predictive power of firing events

**When to use**: Measuring monosemanticity of sparse SAE features

**Example values**:
- U = 0.7: Highly informative features
- U = 0.2: Some class information
- U = 0.0: No information about classes

### 2. Clustering Metrics

These measure how well-organized and separated features are in representation space.

#### Adjusted Rand Index (ARI)
- **Range**: -1 to 1
- **Higher is better**
- **Interpretation**: Agreement between K-means clustering and ground truth labels
- **Adjusted for chance**: 0 = random clustering, 1 = perfect agreement

**Use case**: Does the learned representation naturally cluster by class?

#### Silhouette Score (Supervised)
- **Range**: -1 to 1
- **Higher is better**
- **Interpretation**: How compact and well-separated classes are
  - Compares intra-class distance to inter-class distance
  - Uses ground truth labels

**Example values**:
- Silhouette = 0.7: Very well-separated classes
- Silhouette = 0.3: Some overlap
- Silhouette < 0: Poor separation

#### Silhouette Score (Unsupervised)
- **Range**: -1 to 1
- **Higher is better**
- **Interpretation**: Same as supervised, but uses K-means clusters
- **Use case**: Measures intrinsic cluster quality without labels

#### Davies-Bouldin Index (DBI)
- **Range**: 0 to ∞
- **Lower is better** ⚠️
- **Interpretation**: Average similarity between each cluster and its most similar cluster
- **Formula**: Ratio of within-cluster scatter to between-cluster separation

**Example values**:
- DBI = 0.5: Excellent separation
- DBI = 1.5: Moderate separation
- DBI = 3.0: Poor separation

#### Calinski-Harabasz Index (CHI)
- **Range**: 0 to ∞
- **Higher is better**
- **Interpretation**: Ratio of between-cluster to within-cluster variance
- **Alternative name**: Variance Ratio Criterion

**Example values**:
- CHI = 3000: Very good clustering
- CHI = 1000: Moderate clustering
- CHI = 100: Poor clustering

### 3. Sparsity Metrics

These measure how sparse SAE activations are.

#### L0 Norm
- **Range**: 0 to hidden_dim
- **Interpretation**: Average number of active (non-zero) features per example
- **Target**: Should be much smaller than hidden_dim for good sparsity

**Example values**:
- L0 = 45 out of 1024 dimensions: Very sparse (good for interpretability)
- L0 = 512 out of 1024 dimensions: Not very sparse

#### L1 Norm
- **Range**: 0 to ∞
- **Interpretation**: Average sum of absolute activations
- **Use case**: Sparsity penalty term in SAE loss

### 4. Performance Metrics

These measure task performance and information preservation.

#### Task Accuracy
- **Range**: 0 to 1 (0% to 100%)
- **Higher is better**
- **Interpretation**: Classification accuracy on the main task
- **Critical**: Should remain stable across SAE-ception cycles

**Red flags**:
- Drop > 2% after cycle 1: Auxiliary loss may be too strong
- Drop > 5% after cycle 3: Method may be interfering with task learning

#### Linear Probe Accuracy
- **Range**: 0 to 1
- **Higher is better**
- **Interpretation**: How much task information is preserved in SAE features
- **Method**: Train logistic regression on frozen SAE sparse codes

**Example values**:
- Probe accuracy ≈ Task accuracy: Features preserve all information (good!)
- Probe accuracy << Task accuracy: Information loss in SAE

#### SAE Reconstruction Loss
- **Range**: 0 to ∞
- **Lower is better**
- **Interpretation**: MSE between original activations and SAE reconstruction
- **Use case**: Measuring SAE quality

#### Explained Variance
- **Range**: 0 to 1
- **Higher is better**
- **Interpretation**: Proportion of activation variance explained by SAE
- **Formula**: `1 - var(residual) / var(original)`

## Comprehensive Evaluation Pipeline

Use the full evaluation function:

```python
from sae_ception.evaluation import evaluate_model_and_sae

results = evaluate_model_and_sae(
    model=model,
    sae=sae,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    train_activations=train_acts,
    train_labels=train_labels,
    val_activations=val_acts,
    val_labels=val_labels,
    sparse_codes_train=sparse_train,
    sparse_codes_val=sparse_val,
    device='cuda',
)
```

Returns dictionary with:
- `task_*`: Task performance metrics
- `probe_*`: Linear probe metrics
- `sae_*`: SAE quality metrics
- `base_*`: Base model monosemanticity
- `sae_features_*`: SAE features monosemanticity
- `clustering_*`: Clustering metrics
- `sparsity_*`: Sparsity metrics

## Interpreting Results Across Cycles

### Success Indicators

**Cycle 0 → Cycle 1 (Good signs)**:
- ✓ ARI increases (e.g., 0.81 → 0.90)
- ✓ Silhouette increases
- ✓ DBI decreases
- ✓ U increases (for SAE features)
- ✓ Task accuracy stable (±0.5%)

**Cycle 1 → Cycle 2 (Watch for)**:
- Peak performance on clustering metrics
- May start to plateau or decline after 2-3 cycles
- Task accuracy should remain stable

### Warning Signs

- ⚠️ Task accuracy drops > 2%
- ⚠️ Linear probe accuracy << task accuracy (information loss)
- ⚠️ Clustering metrics decreasing
- ⚠️ L0 norm very high (not sparse)

## Comparing to Paper Results

### ViT-H on CIFAR-10 (from paper)

| Metric | Baseline | Cycle 1 | Target |
|--------|----------|---------|--------|
| ARI | 0.81 | 0.90 | ↑ |
| Silhouette | 0.49 | 0.56 | ↑ |
| DBI | 1.19 | 1.00 | ↓ |
| U (SAE) | 0.05 | 0.10 | ↑ |
| Accuracy | 99.56% | 99.47% | ≈ |

### MLP on MNIST (from paper)

| Metric | Baseline | Cycle 3 | Target |
|--------|----------|---------|--------|
| ARI | 0.26 | 0.28 | ↑ |
| CSI | 0.28 | 0.31 | ↑ |
| U | 0.09 | 0.11 | ↑ |
| Accuracy | 91.11% | 93.66% | ↑ |

## Usage Examples

### Individual Metrics

```python
from sae_ception.evaluation import (
    compute_monosemanticity_metrics,
    compute_all_clustering_metrics,
    train_linear_probe,
)

# Monosemanticity
mono = compute_monosemanticity_metrics(activations, labels)
print(f"CSI: {mono['csi']:.3f}, U: {mono['u']:.3f}")

# Clustering
clustering = compute_all_clustering_metrics(sparse_codes, labels)
print(f"ARI: {clustering['ari']:.3f}")

# Linear probe
probe = train_linear_probe(train_acts, train_labels, test_acts, test_labels)
print(f"Probe accuracy: {probe['test_accuracy']:.1%}")
```

### Tracking Across Cycles

```python
results_by_cycle = {}

for cycle in range(4):
    # ... train model and SAE ...
    
    results = evaluate_model_and_sae(...)
    results_by_cycle[cycle] = results
    
    # Log key metrics
    print(f"Cycle {cycle}:")
    print(f"  Task Acc: {results['task_accuracy']:.2%}")
    print(f"  ARI: {results['clustering_ari']:.3f}")
    print(f"  U: {results['sae_features_u']:.3f}")
```

## Troubleshooting

**Q: My CSI is very low (< 0.2)**
- This is normal for polysemantic representations
- SAE-ception aims to improve this
- Check if it increases across cycles

**Q: My clustering metrics are decreasing after cycle 1**
- This is the "optimal cycle" phenomenon
- The paper observed this too
- Consider using cycle 1 or 2 model

**Q: Task accuracy dropped significantly**
- Reduce auxiliary loss weight (λ)
- Check if you're overfitting to sharpened targets
- Try random baseline to debug

**Q: Linear probe accuracy << task accuracy**
- SAE may be losing task-relevant information
- Increase L1 penalty (more sparsity)
- Try different target layer

## References

- Class Selectivity: Morcos et al. (2018)
- Uncertainty Coefficient: Theil (1972)
- Clustering metrics: scikit-learn documentation
- SAE-ception paper: Tables 2, 3, 4