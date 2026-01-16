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


def get_activation_hook(cache: ActivationCache, name: str, detach: bool = True) -> Callable:
    """
    Create a hook function that caches activations.

    Args:
        cache: ActivationCache to store activations in
        name: Name to store activations under
        detach: If True, detach activations from computation graph (saves memory).
                If False, preserve gradients for backpropagation through activations.

    Returns:
        Hook function compatible with PyTorch's register_forward_hook
    """
    def hook(module, input, output):
        # Handle different output types
        if isinstance(output, tuple):
            # For models that return (hidden_states, ...), take first element
            output = output[0]

        # If output is 3D (batch, seq, hidden), we might want different handling
        # Optionally detach to save memory (but breaks gradient flow for aux loss)
        if detach:
            cache.activations[name] = output.detach()
        else:
            cache.activations[name] = output

    return hook


def register_activation_hook(
    model: nn.Module,
    layer_name: str,
    cache: ActivationCache,
    detach: bool = True,
) -> None:
    """
    Register a forward hook on a specific layer.

    Args:
        model: The model to hook
        layer_name: Name of the layer (e.g., 'transformer.h.11' or '-1' for last layer)
        cache: ActivationCache to store activations
        detach: If True, detach activations (saves memory). If False, preserve gradients.
    """
    # Handle negative indices for layer names
    if layer_name.lstrip('-').isdigit():
        layer_idx = int(layer_name)
        
        # Try different common transformer architectures
        layers = None
        
        # Option 1: model.transformer.h (GPT2ForSequenceClassification)
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h
        # Option 2: model.h (GPT2Model, already unwrapped transformer)
        elif hasattr(model, 'h'):
            layers = model.h
        # Option 3: model.layers (some other architectures)
        elif hasattr(model, 'layers'):
            layers = model.layers
        # Option 4: model.encoder.layer (BERT-style)
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            layers = model.encoder.layer
        # Option 5: model.gpt_neox.layers (GPT-NeoX)
        elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
            layers = model.gpt_neox.layers
        
        if layers is None:
            raise ValueError(
                f"Cannot find layer list for model type {type(model).__name__}. "
                f"Model attributes: {list(model.__dict__.keys())}"
            )
        
        target_layer = layers[layer_idx]
        hook_name = f"layer_{layer_idx}"
    else:
        # Handle named layers
        target_layer = dict(model.named_modules())[layer_name]
        hook_name = layer_name
    
    hook_fn = get_activation_hook(cache, hook_name, detach=detach)
    handle = target_layer.register_forward_hook(hook_fn)
    cache.hooks.append(handle)


def extract_layer_output(
    model: nn.Module,
    layer_name: str,
    detach: bool = True,
) -> ActivationCache:
    """
    Setup activation extraction for a specific layer.

    Args:
        model: The model to extract from
        layer_name: Layer to extract (e.g., '-1' for last layer)
        detach: If True, detach activations (saves memory). If False, preserve gradients.

    Returns:
        ActivationCache that will be populated during forward passes

    Example:
        >>> cache = extract_layer_output(model, layer_name='-1')
        >>> outputs = model(inputs)
        >>> activations = cache.activations['layer_-1']
        >>> cache.remove_hooks()  # Clean up when done
    """
    cache = ActivationCache()
    register_activation_hook(model, layer_name, cache, detach=detach)
    return cache


class ActivationExtractor:
    """
    Context manager for extracting activations from a model.

    Example:
        >>> with ActivationExtractor(model, 'transformer.h.11') as extractor:
        ...     outputs = model(inputs)
        ...     activations = extractor.get_activations()

    For training with auxiliary loss (gradients through activations):
        >>> with ActivationExtractor(model, 'transformer.h.11', detach=False) as extractor:
        ...     outputs = model(inputs)
        ...     activations = extractor.get_activations()
        ...     aux_loss = F.mse_loss(activations, target)  # Gradients flow back!
    """

    def __init__(self, model: nn.Module, layer_name: str, detach: bool = True):
        """
        Args:
            model: Model to extract activations from
            layer_name: Layer to target (e.g., 'transformer.h.11' or '11')
            detach: If True, detach activations (saves memory, no gradients).
                    If False, preserve gradients for backprop through activations.
        """
        self.model = model
        self.layer_name = layer_name
        self.detach = detach
        self.cache = None

    def __enter__(self):
        self.cache = extract_layer_output(self.model, self.layer_name, detach=self.detach)
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

# =============================================================================
# INTERVENTION HOOKS (NEW - for activation replacement)
# =============================================================================

class ActivationIntervention:
    """
    Context manager for intervening on model activations during forward pass.
    
    Unlike ActivationExtractor (read-only), this MODIFIES activations in-place.
    
    Example:
        >>> def sharpen_fn(acts):
        ...     # acts: [batch, seq, hidden]
        ...     return sae.decode(sharpen(sae.encode(acts)))
        >>> 
        >>> with ActivationIntervention(model, 'gpt_neox.layers.3', sharpen_fn):
        ...     outputs = model(input_ids)  # Uses modified activations
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        layer_name: str, 
        intervention_fn: Callable[[torch.Tensor], torch.Tensor],
        enabled: bool = True,
    ):
        """
        Args:
            model: Model to intervene on
            layer_name: Layer to target (e.g., 'gpt_neox.layers.3' or '3')
            intervention_fn: Function that takes activations and returns modified activations
            enabled: Whether intervention is active (useful for toggling)
        """
        self.model = model
        self.layer_name = layer_name
        self.intervention_fn = intervention_fn
        self.enabled = enabled
        self.handle = None
        self._original_output = None  # For debugging/comparison
    
    def _get_target_layer(self) -> nn.Module:
        """Resolve layer name to module."""
        # Handle integer indices
        if self.layer_name.lstrip('-').isdigit():
            layer_idx = int(self.layer_name)
            
            # Try different architectures
            if hasattr(self.model, 'gpt_neox') and hasattr(self.model.gpt_neox, 'layers'):
                return self.model.gpt_neox.layers[layer_idx]
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                return self.model.transformer.h[layer_idx]
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                return self.model.model.layers[layer_idx]
            else:
                raise ValueError(f"Cannot find layers for model type {type(self.model)}")
        else:
            # Handle dotted path names
            return dict(self.model.named_modules())[self.layer_name]
    
    def _intervention_hook(self, module, input, output):
        """Hook that modifies layer output."""
        # Handle tuple outputs (common in transformers)
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None
        
        # Store original for debugging
        self._original_output = hidden_states.detach().clone()
        
        # Apply intervention if enabled
        if self.enabled:
            modified = self.intervention_fn(hidden_states)
        else:
            modified = hidden_states
        
        # Reconstruct output format
        if rest is not None:
            return (modified,) + rest
        return modified
    
    def __enter__(self):
        target_layer = self._get_target_layer()
        self.handle = target_layer.register_forward_hook(self._intervention_hook)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
    
    def get_original_output(self) -> Optional[torch.Tensor]:
        """Get the unmodified output from last forward pass (for debugging)."""
        return self._original_output


def create_sae_intervention(
    sae: nn.Module,
    k_sharp: int,
    device: str = 'cuda',
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Create an intervention function that applies SAE sharpening.
    
    Args:
        sae: Trained SAE
        k_sharp: Number of top features to keep per token
        device: Device to run on
        
    Returns:
        Function: activations -> sharpened_reconstructions
    """
    sae.eval()
    sae.to(device)
    
    def intervention_fn(activations: torch.Tensor) -> torch.Tensor:
        """
        Apply SAE encode -> sharpen -> decode.
        
        Args:
            activations: [batch, seq_len, hidden_dim]
        Returns:
            sharpened: [batch, seq_len, hidden_dim]
        """
        original_shape = activations.shape
        batch_size, seq_len, hidden_dim = original_shape
        
        # Flatten to [batch * seq, hidden]
        flat_acts = activations.view(-1, hidden_dim)
        
        with torch.no_grad():
            # Encode
            sparse_codes = sae.encode(flat_acts)  # [batch * seq, sae_hidden]
            
            # Sharpen: keep only top-k per token
            topk_vals, topk_idx = torch.topk(sparse_codes, k=k_sharp, dim=-1)
            sharpened_codes = torch.zeros_like(sparse_codes)
            sharpened_codes.scatter_(-1, topk_idx, topk_vals)
            
            # Decode
            reconstructed = sae.decode(sharpened_codes)  # [batch * seq, hidden]
        
        # Reshape back
        return reconstructed.view(original_shape)
    
    return intervention_fn