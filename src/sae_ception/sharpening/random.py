import torch
from .base import FeatureSharpener


class RandomSharpener(FeatureSharpener):
    """Use random unit vectors as targets (baseline)."""
    
    def __init__(self, seed: int = 42):
        super().__init__()
        self.seed = seed
    
    def identify_salient_features(self, *args, **kwargs) -> dict:
        """No features to identify for random targets."""
        return {}
    
    def sharpen_features(
        self,
        sparse_code: torch.Tensor,
        labels: torch.Tensor,
        salient_features: dict,
    ) -> torch.Tensor:
        """Not used - we override generate_targets entirely."""
        raise NotImplementedError("Use generate_targets() directly")
    
    def generate_targets(
        self,
        sae: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Generate random unit vectors as targets."""
        # Get total dataset size and activation dim
        total_size = len(dataloader.dataset)
        activation_dim = sae.input_dim
        
        # Generate random vectors
        torch.manual_seed(self.seed)
        random_targets = torch.randn(total_size, activation_dim)
        
        # Normalize to unit vectors
        random_targets = torch.nn.functional.normalize(random_targets, dim=-1)
        
        return random_targets
