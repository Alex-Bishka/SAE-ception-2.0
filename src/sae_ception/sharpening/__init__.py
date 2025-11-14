from .base import FeatureSharpener
from .per_class import PerClassSharpener
from .per_example import PerExampleSharpener
from .random import RandomSharpener


def get_sharpener(cfg) -> FeatureSharpener:
    """Factory function to create sharpener from config."""
    sharpener_type = cfg.type
    
    if sharpener_type == "per_class":
        return PerClassSharpener(top_k_pct=cfg.get("top_k_pct", 0.5))
    elif sharpener_type == "per_example":
        return PerExampleSharpener(top_k_pct=cfg.get("top_k_pct", 0.5))
    elif sharpener_type == "random":
        return RandomSharpener(seed=cfg.get("seed", 42))
    else:
        raise ValueError(f"Unknown sharpener type: {sharpener_type}")


__all__ = [
    "FeatureSharpener",
    "PerClassSharpener", 
    "PerExampleSharpener",
    "RandomSharpener",
    "get_sharpener",
]
