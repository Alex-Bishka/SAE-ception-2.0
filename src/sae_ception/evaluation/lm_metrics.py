"""Language model evaluation metrics."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm
import math


def evaluate_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda',
    max_batches: Optional[int] = None,
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    Evaluate perplexity of a causal language model.
    
    Args:
        model: Causal LM (e.g., Pythia, GPT-2)
        dataloader: DataLoader providing input_ids and attention_mask
        device: Device to run on
        max_batches: Limit number of batches (for quick tests)
        show_progress: Whether to show progress bar
        
    Returns:
        Dictionary with:
            - perplexity: exp(average cross-entropy loss)
            - loss: average cross-entropy loss
            - total_tokens: number of tokens evaluated
    """
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    total_tokens = 0
    
    iterator = tqdm(dataloader, desc="Evaluating perplexity") if show_progress else dataloader
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(iterator):
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            labels = input_ids.clone()
            if attention_mask is not None:
                # Set padding positions to -100 (ignored in loss)
                labels[attention_mask == 0] = -100

            # Forward pass with labels = input_ids (standard causal LM)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            # outputs.loss is average over all tokens
            # We need to weight by number of tokens for accurate aggregation
            if attention_mask is not None:
                n_tokens = attention_mask.sum().item()
            else:
                n_tokens = input_ids.numel()
            
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens
            
            if show_progress:
                current_ppl = math.exp(total_loss / total_tokens)
                iterator.set_postfix({'ppl': f'{current_ppl:.2f}'})
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return {
        'perplexity': perplexity,
        'loss': avg_loss,
        'total_tokens': total_tokens,
    }


def evaluate_perplexity_with_intervention(
    model: nn.Module,
    dataloader: DataLoader,
    intervention_fn,
    layer_name: str,
    device: str = 'cuda',
    max_batches: Optional[int] = None,
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    Evaluate perplexity with activation intervention.
    
    This replaces activations at a specific layer during the forward pass.
    
    Args:
        model: Causal LM
        dataloader: DataLoader
        intervention_fn: Function that transforms activations
        layer_name: Layer to intervene on
        device: Device
        max_batches: Limit batches
        show_progress: Show progress
        
    Returns:
        Same as evaluate_perplexity
    """
    from ..utils.hooks import ActivationIntervention
    
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    total_tokens = 0
    
    iterator = tqdm(dataloader, desc="Evaluating with intervention") if show_progress else dataloader
    
    with ActivationIntervention(model, layer_name, intervention_fn):
        with torch.no_grad():
            for batch_idx, batch in enumerate(iterator):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                
                # masking padding in labels
                labels = input_ids.clone()
                if attention_mask is not None:
                    labels[attention_mask == 0] = -100

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                
                if attention_mask is not None:
                    n_tokens = attention_mask.sum().item()
                else:
                    n_tokens = input_ids.numel()
                
                total_loss += outputs.loss.item() * n_tokens
                total_tokens += n_tokens
                
                if show_progress:
                    current_ppl = math.exp(total_loss / total_tokens)
                    iterator.set_postfix({'ppl': f'{current_ppl:.2f}'})
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return {
        'perplexity': perplexity,
        'loss': avg_loss,
        'total_tokens': total_tokens,
    }