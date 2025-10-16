"""Performance evaluation metrics for models and SAE features."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from tqdm import tqdm


def evaluate_classification_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda',
) -> Dict[str, float]:
    """
    Evaluate classification accuracy of a model.
    
    Args:
        model: Classification model
        dataloader: DataLoader with 'input_ids', 'attention_mask', 'labels'
        device: Device to run on
        
    Returns:
        Dictionary with accuracy, loss, and other metrics
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    all_losses = []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Handle different output formats
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Compute loss
            loss = criterion(logits, labels)
            all_losses.append(loss.item())
            
            # Get predictions
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate all batches
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    return {
        'accuracy': float(accuracy),
        'loss': float(np.mean(all_losses)),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
    }


def train_linear_probe(
    activations: torch.Tensor,
    labels: torch.Tensor,
    test_activations: Optional[torch.Tensor] = None,
    test_labels: Optional[torch.Tensor] = None,
    max_iter: int = 1000,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Train a linear probe on SAE features.
    
    This measures how much task-relevant information is preserved
    in the sparse features.
    
    Args:
        activations: Training features [N, hidden_dim]
        labels: Training labels [N]
        test_activations: Test features (optional, uses train if not provided)
        test_labels: Test labels (optional)
        max_iter: Maximum iterations for LogisticRegression
        random_state: Random seed
        
    Returns:
        Dictionary with probe accuracy and other metrics
    """
    # Convert to numpy
    X_train = activations.numpy() if isinstance(activations, torch.Tensor) else activations
    y_train = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    
    if test_activations is not None:
        X_test = test_activations.numpy() if isinstance(test_activations, torch.Tensor) else test_activations
        y_test = test_labels.numpy() if isinstance(test_labels, torch.Tensor) else test_labels
    else:
        X_test = X_train
        y_test = y_train
    
    # Train logistic regression
    clf = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        solver='lbfgs',
        multi_class='multinomial',
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    train_preds = clf.predict(X_train)
    test_preds = clf.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    
    test_f1_macro = f1_score(y_test, test_preds, average='macro')
    
    return {
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'test_f1_macro': float(test_f1_macro),
    }


def evaluate_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda',
) -> float:
    """
    Evaluate perplexity for language models.
    
    Args:
        model: Language model
        dataloader: DataLoader providing input_ids and labels
        device: Device to run on
        
    Returns:
        Perplexity score
    """
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    total_tokens = 0
    
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing perplexity"):
            input_ids = batch['input_ids'].to(device)
            
            # For language modeling, labels are typically shifted input_ids
            if 'labels' in batch:
                labels = batch['labels'].to(device)
            else:
                labels = input_ids.clone()
            
            # Forward pass
            outputs = model(input_ids=input_ids)
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Compute loss
            # Reshape for cross entropy: [batch * seq, vocab_size]
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            
            # Ignore padding tokens (typically -100)
            valid_mask = labels_flat != -100
            if valid_mask.sum() > 0:
                loss = criterion(logits_flat[valid_mask], labels_flat[valid_mask])
                total_loss += loss.item()
                total_tokens += valid_mask.sum().item()
    
    # Compute perplexity
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return float(perplexity)


def compare_models(
    baseline_results: Dict[str, float],
    current_results: Dict[str, float],
) -> Dict[str, float]:
    """
    Compare current model performance to baseline.
    
    Args:
        baseline_results: Metrics from baseline model
        current_results: Metrics from current model
        
    Returns:
        Dictionary with differences and relative changes
    """
    comparison = {}
    
    for key in baseline_results:
        if key in current_results:
            baseline_val = baseline_results[key]
            current_val = current_results[key]
            
            diff = current_val - baseline_val
            rel_change = (diff / baseline_val * 100) if baseline_val != 0 else 0
            
            comparison[f'{key}_diff'] = float(diff)
            comparison[f'{key}_rel_change'] = float(rel_change)
    
    return comparison


def evaluate_model_and_sae(
    model: nn.Module,
    sae: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    train_activations: torch.Tensor,
    train_labels: torch.Tensor,
    val_activations: torch.Tensor,
    val_labels: torch.Tensor,
    sparse_codes_train: torch.Tensor,
    sparse_codes_val: torch.Tensor,
    device: str = 'cuda',
) -> Dict[str, float]:
    """
    Comprehensive evaluation of model and SAE.
    
    Args:
        model: Classification model
        sae: Trained SAE
        train_dataloader: Training data
        val_dataloader: Validation data
        train_activations: Pre-extracted training activations
        train_labels: Training labels
        val_activations: Pre-extracted validation activations
        val_labels: Validation labels
        sparse_codes_train: SAE sparse codes for training data
        sparse_codes_val: SAE sparse codes for validation data
        device: Device to run on
        
    Returns:
        Dictionary with all metrics
    """
    from .interpretability import (
        compute_monosemanticity_metrics,
        compute_sparsity_metrics,
        evaluate_sae_quality,
    )
    from .clustering import compute_all_clustering_metrics
    
    results = {}
    
    # 1. Task performance
    print("Evaluating task performance...")
    task_metrics = evaluate_classification_accuracy(model, val_dataloader, device)
    results.update({f'task_{k}': v for k, v in task_metrics.items()})
    
    # 2. Linear probe on SAE features
    print("Training linear probe on SAE features...")
    probe_metrics = train_linear_probe(
        sparse_codes_train, train_labels,
        sparse_codes_val, val_labels,
    )
    results.update({f'probe_{k}': v for k, v in probe_metrics.items()})
    
    # 3. SAE quality
    print("Evaluating SAE quality...")
    sae_metrics = evaluate_sae_quality(sae, val_activations, device)
    results.update({f'sae_{k}': v for k, v in sae_metrics.items()})
    
    # 4. Monosemanticity of base activations
    print("Computing monosemanticity metrics (base model)...")
    mono_base = compute_monosemanticity_metrics(val_activations, val_labels)
    results.update({f'base_{k}': v for k, v in mono_base.items()})
    
    # 5. Monosemanticity of SAE features
    print("Computing monosemanticity metrics (SAE features)...")
    mono_sae = compute_monosemanticity_metrics(sparse_codes_val, val_labels)
    results.update({f'sae_features_{k}': v for k, v in mono_sae.items()})
    
    # 6. Clustering metrics
    print("Computing clustering metrics...")
    clustering = compute_all_clustering_metrics(sparse_codes_val, val_labels)
    results.update({f'clustering_{k}': v for k, v in clustering.items()})
    
    # 7. Sparsity metrics
    print("Computing sparsity metrics...")
    sparsity = compute_sparsity_metrics(sparse_codes_val)
    results.update({f'sparsity_{k}': v for k, v in sparsity.items()})
    
    return results