"""Utilities for extracting activations from model layers."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable


class ActivationCache:
    """Cache for storing activations during forward pass."""
    
    def __init__(self):
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
    
    def clear(self):
        """Clear cached activations."""
        self.activations.clear()
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()


def get_activation_hook(cache: ActivationCache, name: str) -> Callable:
    """
    Create a hook function that caches activations.
    
    Args:
        cache: ActivationCache to store activations in
        name: Name to store activations under
        
    Returns:
        Hook function compatible with PyTorch's register_forward_hook
    """
    def hook(module, input, output):
        # Handle different output types
        if isinstance(output, tuple):
            # For models that return (hidden_states, ...), take first element
            output = output[0]
        
        # If output is 3D (batch, seq, hidden), we might want different handling
        # For now, just cache as-is
        cache.activations[name] = output.detach()
    
    return hook


def register_activation_hook(
    model: nn.Module,
    layer_name: str,
    cache: ActivationCache,
) -> None:
    """
    Register a forward hook on a specific layer.
    
    Args:
        model: The model to hook
        layer_name: Name of the layer (e.g., 'transformer.h.11' or '-1' for last layer)
        cache: ActivationCache to store activations
    """
    # Handle negative indices for layer names
    if layer_name.lstrip('-').isdigit():
        layer_idx = int(layer_name)
        # For transformers, typically model.transformer.h is the layer list
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h
            target_layer = layers[layer_idx]
            hook_name = f"layer_{layer_idx}"
        else:
            raise ValueError(f"Cannot handle layer index {layer_idx} for this model type")
    else:
        # Handle named layers
        target_layer = dict(model.named_modules())[layer_name]
        hook_name = layer_name
    
    hook_fn = get_activation_hook(cache, hook_name)
    handle = target_layer.register_forward_hook(hook_fn)
    cache.hooks.append(handle)


def extract_layer_output(
    model: nn.Module,
    layer_name: str,
) -> ActivationCache:
    """
    Setup activation extraction for a specific layer.
    
    Args:
        model: The model to extract from
        layer_name: Layer to extract (e.g., '-1' for last layer)
        
    Returns:
        ActivationCache that will be populated during forward passes
        
    Example:
        >>> cache = extract_layer_output(model, layer_name='-1')
        >>> outputs = model(inputs)
        >>> activations = cache.activations['layer_-1']
        >>> cache.remove_hooks()  # Clean up when done
    """
    cache = ActivationCache()
    register_activation_hook(model, layer_name, cache)
    return cache


class ActivationExtractor:
    """
    Context manager for extracting activations from a model.
    
    Example:
        >>> with ActivationExtractor(model, 'transformer.h.11') as extractor:
        ...     outputs = model(inputs)
        ...     activations = extractor.get_activations()
    """
    
    def __init__(self, model: nn.Module, layer_name: str):
        self.model = model
        self.layer_name = layer_name
        self.cache = None
    
    def __enter__(self):
        self.cache = extract_layer_output(self.model, self.layer_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cache is not None:
            self.cache.remove_hooks()
    
    def get_activations(self, key: Optional[str] = None) -> torch.Tensor:
        """Get cached activations."""
        if self.cache is None:
            raise RuntimeError("Extractor not active. Use within 'with' block.")
        
        if key is None:
            # Return the first (and usually only) cached activation
            return next(iter(self.cache.activations.values()))
        return self.cache.activations[key]


def extract_final_token_activation(
    activation: torch.Tensor,
    token_idx: int = -1,
) -> torch.Tensor:
    """
    Extract activation for a specific token position.
    
    Args:
        activation: Shape [batch, seq_len, hidden_dim]
        token_idx: Token position to extract (default: -1 for last token)
        
    Returns:
        Extracted activation of shape [batch, hidden_dim]
    """
    if activation.dim() == 2:
        # Already [batch, hidden_dim], no token dimension
        return activation
    elif activation.dim() == 3:
        # [batch, seq_len, hidden_dim]
        return activation[:, token_idx, :]
    else:
        raise ValueError(f"Unexpected activation shape: {activation.shape}")


def extract_cls_token_activation(activation: torch.Tensor) -> torch.Tensor:
    """
    Extract [CLS] token activation (first token).
    
    Args:
        activation: Shape [batch, seq_len, hidden_dim]
        
    Returns:
        [CLS] activation of shape [batch, hidden_dim]
    """
    return extract_final_token_activation(activation, token_idx=0)