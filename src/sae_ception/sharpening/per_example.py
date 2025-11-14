import torch
from .base import FeatureSharpener


class PerExampleSharpener(FeatureSharpener):
    """Sharpen features based on per-example top-k% activations."""
    
    def identify_salient_features(
        self,
        sae: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cuda",
    ) -> dict:
        """No precomputation needed for per-example sharpening."""
        # Just compute k from percentage
        # Get hidden_dim from first batch
        for batch in dataloader:
            activations = batch["activations"].to(device)
            with torch.no_grad():
                sparse_code = sae.encode(activations)
            hidden_dim = sparse_code.shape[1]
            break
        
        k = self._compute_k(hidden_dim)
        
        return {
            "k": k,
            "top_k_pct": self.top_k_pct,
        }
    
    def sharpen_features(
        self,
        sparse_code: torch.Tensor,
        labels: torch.Tensor,
        salient_features: dict,
    ) -> torch.Tensor:
        """Keep only top-k% activated features for each example."""
        k = salient_features["k"]
        batch_size, hidden_dim = sparse_code.shape
        
        # Get top-k indices for each example
        top_k_values, top_k_indices = torch.topk(sparse_code, k=k, dim=-1)
        
        # Create sharpened code with only top-k features
        sharpened = torch.zeros_like(sparse_code)
        
        # Scatter top-k values back
        sharpened.scatter_(dim=-1, index=top_k_indices, src=top_k_values)
        
        return sharpened