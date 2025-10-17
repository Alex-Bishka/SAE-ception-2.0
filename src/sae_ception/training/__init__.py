"""Training module for SAE-ception."""

from .baseline import train_baseline_model, load_baseline_model

__all__ = [
    'train_baseline_model',
    'load_baseline_model',
]