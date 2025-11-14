from abc import ABC, abstractmethod
import torch


class FeatureSharpener(ABC):
    """Abstract base class for feature sharpening strategies."""
    
    def __init__(self, top_k_pct: float = 0.5):
        """
        Args:
            top_k_pct: Percentage of features to keep (0-100)
                      e.g., 0.5 = keep top 0.5% of features
                      1.0 = keep top 1% of features
        """
        self.top_k_pct = top_k_pct
    
    def _compute_k(self, hidden_dim: int) -> int:
        """Convert percentage to absolute k value."""
        k = int(hidden_dim * self.top_k_pct / 100.0)
        # Ensure at least 1 feature is kept
        return max(1, k)
    
    @abstractmethod
    def identify_salient_features(
        self,
        sae: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cuda",
    ) -> dict:
        """
        Identify which features are salient.
        
        Returns a dict that can be used by sharpen_features().
        """
        pass
    
    @abstractmethod
    def sharpen_features(
        self,
        sparse_code: torch.Tensor,
        labels: torch.Tensor,
        salient_features: dict,
    ) -> torch.Tensor:
        """
        Given a sparse code, zero out non-salient features.
        
        Args:
            sparse_code: [batch_size, hidden_dim] SAE sparse code
            labels: [batch_size] class labels
            salient_features: Output from identify_salient_features()
            
        Returns:
            sharpened_code: [batch_size, hidden_dim] with non-salient features zeroed
        """
        pass
    
    def generate_targets(
        self,
        sae: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Generate sharpened reconstruction targets for entire dataset.
        
        Returns:
            targets: [dataset_size, input_dim] sharpened reconstructions
        """
        sae.eval()
        
        # First, identify salient features
        salient_features = self.identify_salient_features(sae, dataloader, device)
        
        # Then generate sharpened targets
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                activations = batch["activations"].to(device)
                labels = batch["labels"].to(device)
                
                # Encode to sparse code
                sparse_code = sae.encode(activations)
                
                # Sharpen the sparse code
                sharpened_code = self.sharpen_features(
                    sparse_code, labels, salient_features
                )
                
                # Decode back to activation space
                targets = sae.decode(sharpened_code)
                all_targets.append(targets.cpu())
        
        return torch.cat(all_targets, dim=0)