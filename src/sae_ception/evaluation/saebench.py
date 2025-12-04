"""
SAEBench-compatible evaluations for SAE quality.

This module implements standard SAE evaluation metrics from SAEBench,
particularly the first-letter absorption metric.

Reference: "SAEBench: A Comprehensive Benchmark for Sparse Autoencoders 
in Language Model Interpretability"

Note: These evaluations require token-level SAE training (not sequence-level).

Usage:
    # From standalone script
    python scripts/run_saebench.py --checkpoint_dir outputs/run_123 --cycle 0
    
    # Quick smoke test (only top 50 latents)
    python scripts/run_saebench.py --checkpoint_dir outputs/run_123 --cycle 0 --n_candidates 50
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from tqdm import tqdm
import warnings


def compute_first_letter_absorption(
    sae: nn.Module,
    activations: torch.Tensor,
    token_ids: torch.Tensor,
    tokenizer,
    device: str = 'cuda',
    k_sparse_threshold: float = 0.03,
    absorption_threshold: float = 0.1,
    max_absorbing_latents: int = 5,
    min_samples_per_letter: int = 10,
    n_latent_candidates: Optional[int] = None,  # None = all latents (paper-accurate)
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    SAEBench first-letter absorption metric.
    
    Measures whether main SAE latents for a first-letter feature get "absorbed"
    into other latents, reducing interpretability.
    
    Args:
        sae: Trained SAE (must be token-level trained)
        activations: Token-level activations [N_tokens, hidden_dim]
        token_ids: Token IDs [N_tokens]
        tokenizer: Tokenizer for decoding tokens
        device: Device to run on
        k_sparse_threshold: F1 improvement threshold for feature splitting (ω_fs)
        absorption_threshold: Minimum absorption ratio to count (τ_pa)
        max_absorbing_latents: Max latents to consider as absorbing (A_max)
        min_samples_per_letter: Minimum samples needed per letter
        n_latent_candidates: Number of top latents to consider for k-sparse probing.
                            None = all latents (paper-accurate, slower)
                            50-100 = fast approximation for smoke tests
        show_progress: Whether to show progress bars
    
    Returns:
        Dict with:
            - absorption: Mean absorption score (lower is better)
            - absorption_complement: 1 - absorption (higher is better)
            - n_letters_evaluated: Number of letters with enough data
            - n_test_samples: Number of test samples evaluated
            - per_letter_absorption: Absorption score per letter
    """
    sae.eval()
    sae.to(device)
    
    # Step 1: Filter to alphabetic tokens and get first letters
    valid_indices = []
    first_letters = []
    
    print("Filtering tokens to alphabetic...")
    for i, tid in enumerate(token_ids):
        token_str = tokenizer.decode([tid.item()])
        clean = token_str.replace('Ġ', '').replace('▁', '').replace(' ', '').strip()
        
        if clean and len(clean) > 0 and clean[0].isalpha():
            valid_indices.append(i)
            first_letters.append(clean[0].upper())
    
    if len(valid_indices) < 100:
        warnings.warn(f"Only {len(valid_indices)} valid alphabetic tokens found.")
        return {
            'absorption': 0.0,
            'absorption_complement': 1.0,
            'n_letters_evaluated': 0,
            'n_test_samples': 0,
            'error': 'insufficient_tokens',
        }
    
    print(f"Found {len(valid_indices)} valid alphabetic tokens")
    
    valid_indices = torch.tensor(valid_indices)
    valid_acts = activations[valid_indices]
    
    # Train/test split (80/20)
    n = len(valid_indices)
    perm = torch.randperm(n)
    split = int(0.8 * n)
    
    train_idx, test_idx = perm[:split], perm[split:]
    train_acts = valid_acts[train_idx]
    test_acts = valid_acts[test_idx]
    train_letters = [first_letters[i] for i in train_idx.tolist()]
    test_letters = [first_letters[i] for i in test_idx.tolist()]
    
    # Step 2: Train ground-truth probes for each letter
    print("Training ground-truth probes for each letter...")
    unique_letters = sorted(set(train_letters))
    probes = {}
    
    for letter in tqdm(unique_letters, disable=not show_progress, desc="Training probes"):
        y_train = np.array([1 if l == letter else 0 for l in train_letters])
        
        if y_train.sum() < min_samples_per_letter or (1 - y_train).sum() < min_samples_per_letter:
            continue
        
        try:
            clf = LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs')
            clf.fit(train_acts.numpy(), y_train)
            
            probe_dir = torch.tensor(clf.coef_[0], dtype=torch.float32)
            probe_dir = probe_dir / probe_dir.norm()
            
            probes[letter] = (probe_dir, clf)
        except Exception as e:
            continue
    
    if len(probes) == 0:
        return {
            'absorption': 0.0,
            'absorption_complement': 1.0,
            'n_letters_evaluated': 0,
            'n_test_samples': 0,
            'error': 'no_valid_probes',
        }
    
    print(f"Trained probes for {len(probes)} letters")
    
    # Step 3: Get SAE sparse codes
    print("Computing SAE sparse codes...")
    with torch.no_grad():
        train_sparse = sae.encode(train_acts.to(device)).cpu()
        test_sparse = sae.encode(test_acts.to(device)).cpu()
    
    # Get decoder weights
    if hasattr(sae, 'decoder') and hasattr(sae.decoder, 'weight'):
        decoder_weights = sae.decoder.weight.data.cpu()
        if decoder_weights.shape[0] == train_sparse.shape[1]:
            decoder_weights = decoder_weights.T
    else:
        raise ValueError("SAE must have decoder.weight attribute")
    
    hidden_dim, n_latents = decoder_weights.shape
    train_sparse_np = train_sparse.numpy()
    
    # Determine which latents to consider
    if n_latent_candidates is None:
        print(f"Using ALL {n_latents} latents (paper-accurate, may be slow)")
    else:
        print(f"Using top {n_latent_candidates} latent candidates (fast approximation)")
    
    # Step 4: Find main latents for each letter via k-sparse probing
    print("Finding main latents for each letter...")
    main_latents = {}
    
    for letter in tqdm(probes.keys(), disable=not show_progress, desc="K-sparse probing"):
        y_train = np.array([1 if l == letter else 0 for l in train_letters])
        
        # Determine candidate latents for this letter
        if n_latent_candidates is not None:
            # Fast path: use correlation-based filtering
            candidates = _get_top_candidates_by_correlation(
                train_sparse_np, y_train, n_latent_candidates
            )
        else:
            # Full path: all non-dead latents
            candidates = [i for i in range(n_latents) if train_sparse_np[:, i].max() > 0]
        
        if len(candidates) == 0:
            continue
        
        # K-sparse probing on candidates
        latents = _find_main_latents(
            sparse_codes=train_sparse_np,
            y_train=y_train,
            candidates=candidates,
            max_k=5,
            f1_threshold=k_sparse_threshold,
        )
        
        if latents:
            main_latents[letter] = latents
    
    print(f"Found main latents for {len(main_latents)} letters")
    
    # Step 5: Measure absorption on test set
    print("Measuring absorption on test set...")
    absorption_scores = []
    per_letter_scores = {letter: [] for letter in main_latents}
    
    for i, (act, letter) in enumerate(tqdm(
        zip(test_acts, test_letters),
        total=len(test_acts),
        disable=not show_progress,
        desc="Computing absorption"
    )):
        if letter not in probes or letter not in main_latents:
            continue
        
        probe_dir, probe_clf = probes[letter]
        
        try:
            probe_pred = probe_clf.predict(act.unsqueeze(0).numpy())[0]
        except:
            continue
            
        if probe_pred != 1:
            continue
        
        model_proj = (act @ probe_dir).item()
        if model_proj <= 0:
            continue
        
        sparse = test_sparse[i]
        main_proj = 0.0
        
        for lat_idx in main_latents[letter]:
            lat_act = sparse[lat_idx].item()
            if lat_act > 0:
                lat_dir = decoder_weights[:, lat_idx]
                main_proj += lat_act * (lat_dir @ probe_dir).item()
        
        if main_proj >= model_proj:
            absorption_scores.append(0.0)
            per_letter_scores[letter].append(0.0)
            continue
        
        # Find absorbing latents
        absorbing_contributions = []
        
        for lat_idx in range(n_latents):
            if lat_idx in main_latents[letter]:
                continue
            
            lat_act = sparse[lat_idx].item()
            if lat_act <= 0:
                continue
            
            lat_dir = decoder_weights[:, lat_idx]
            lat_probe_proj = (lat_dir @ probe_dir).item()
            
            if lat_probe_proj <= 0:
                continue
            
            contribution = lat_act * lat_probe_proj
            absorbing_contributions.append(contribution)
        
        absorbing_contributions = sorted(absorbing_contributions, reverse=True)[:max_absorbing_latents]
        total_absorbing = sum(absorbing_contributions)
        
        if total_absorbing < absorption_threshold * model_proj:
            absorption_scores.append(0.0)
            per_letter_scores[letter].append(0.0)
            continue
        
        score = total_absorbing / (total_absorbing + max(main_proj, 0) + 1e-8)
        absorption_scores.append(score)
        per_letter_scores[letter].append(score)
    
    mean_absorption = float(np.mean(absorption_scores)) if absorption_scores else 0.0
    
    per_letter_mean = {
        letter: float(np.mean(scores)) if scores else 0.0
        for letter, scores in per_letter_scores.items()
    }
    
    return {
        'absorption': mean_absorption,
        'absorption_complement': 1.0 - mean_absorption,
        'n_letters_evaluated': len(main_latents),
        'n_test_samples': len(absorption_scores),
        'per_letter_absorption': per_letter_mean,
        'main_latents_per_letter': {k: len(v) for k, v in main_latents.items()},
    }


def _get_top_candidates_by_correlation(
    sparse_codes: np.ndarray,
    y_train: np.ndarray,
    n_candidates: int,
) -> List[int]:
    """
    Get top N latent candidates by correlation with label.
    
    Uses point-biserial correlation as a fast filter before
    expensive k-sparse probing.
    """
    n_latents = sparse_codes.shape[1]
    
    pos_mask = y_train == 1
    neg_mask = ~pos_mask
    
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return []
    
    correlations = np.zeros(n_latents)
    
    for i in range(n_latents):
        feat = sparse_codes[:, i]
        if feat.max() == 0:
            continue
        
        mean_pos = feat[pos_mask].mean()
        mean_neg = feat[neg_mask].mean()
        std = feat.std()
        
        if std > 0:
            correlations[i] = abs(mean_pos - mean_neg) / std
    
    top_indices = np.argsort(correlations)[-n_candidates:]
    return [i for i in top_indices if correlations[i] > 0]


def _find_main_latents(
    sparse_codes: np.ndarray,
    y_train: np.ndarray,
    candidates: List[int],
    max_k: int = 5,
    f1_threshold: float = 0.03,
) -> List[int]:
    """
    Find main latents via k-sparse probing on given candidates.
    
    Args:
        sparse_codes: [n_samples, n_latents]
        y_train: Binary labels [n_samples]
        candidates: List of latent indices to consider
        max_k: Maximum number of main latents
        f1_threshold: F1 improvement threshold to add another latent
    
    Returns:
        List of main latent indices
    """
    if len(candidates) == 0:
        return []
    
    # k=1: find best single latent
    best_latent = None
    best_f1 = 0.0
    
    for lat_idx in candidates:
        feat = sparse_codes[:, lat_idx:lat_idx+1]
        
        if feat.max() == 0:
            continue
        
        try:
            clf = LogisticRegression(max_iter=200, solver='lbfgs')
            clf.fit(feat, y_train)
            preds = clf.predict(feat)
            f1 = f1_score(y_train, preds, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_latent = lat_idx
        except:
            continue
    
    if best_latent is None:
        return []
    
    main_latents = [best_latent]
    current_f1 = best_f1
    
    # Greedy forward selection for k > 1
    for _ in range(max_k - 1):
        best_additional = None
        best_new_f1 = current_f1
        
        for lat_idx in candidates:
            if lat_idx in main_latents:
                continue
            
            feat = sparse_codes[:, main_latents + [lat_idx]]
            
            try:
                clf = LogisticRegression(max_iter=200, solver='lbfgs')
                clf.fit(feat, y_train)
                preds = clf.predict(feat)
                f1 = f1_score(y_train, preds, zero_division=0)
                
                if f1 > best_new_f1 + f1_threshold:
                    best_new_f1 = f1
                    best_additional = lat_idx
            except:
                continue
        
        if best_additional is not None:
            main_latents.append(best_additional)
            current_f1 = best_new_f1
        else:
            break
    
    return main_latents


def evaluate_sae_saebench(
    sae: nn.Module,
    activations: torch.Tensor,
    token_ids: torch.Tensor,
    tokenizer,
    device: str = 'cuda',
    n_latent_candidates: Optional[int] = None,
) -> Dict[str, float]:
    """
    Run all SAEBench-style evaluations.
    
    Args:
        sae: Trained SAE
        activations: Token-level activations
        token_ids: Token IDs
        tokenizer: Tokenizer
        device: Device
        n_latent_candidates: Number of latents to consider (None = all)
    
    Returns:
        Dict with all SAEBench metrics
    """
    results = {}
    
    print("\n" + "=" * 60)
    print("Running SAEBench First-Letter Absorption Evaluation")
    print("=" * 60)
    
    absorption_results = compute_first_letter_absorption(
        sae=sae,
        activations=activations,
        token_ids=token_ids,
        tokenizer=tokenizer,
        device=device,
        n_latent_candidates=n_latent_candidates,
    )
    
    for key, value in absorption_results.items():
        if isinstance(value, (int, float)):
            results[key] = value
    
    return results