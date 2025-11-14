import torch
from .base import FeatureSharpener


class PerClassSharpener(FeatureSharpener):
    """Sharpen features based on per-class mean activations."""
    
    def identify_salient_features(
        self,
        sae: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cuda",
    ) -> dict:
        """Identify top-k% features for each class."""
        sae.eval()
        
        # Accumulate activations per class
        class_activations = {}
        class_counts = {}
        
        with torch.no_grad():
            for batch in dataloader:
                activations = batch["activations"].to(device)
                labels = batch["labels"].to(device)
                
                # Get sparse codes
                sparse_codes = sae.encode(activations)
                
                # Accumulate per class
                for label in labels.unique():
                    mask = labels == label
                    label_item = label.item()
                    
                    if label_item not in class_activations:
                        class_activations[label_item] = torch.zeros_like(
                            sparse_codes[0]
                        )
                        class_counts[label_item] = 0
                    
                    class_activations[label_item] += sparse_codes[mask].sum(dim=0)
                    class_counts[label_item] += mask.sum().item()
        
        # Compute mean activations and get top-k% indices per class
        hidden_dim = next(iter(class_activations.values())).shape[0]
        k = self._compute_k(hidden_dim)
        
        top_k_per_class = {}
        
        for label, activations in class_activations.items():
            mean_activation = activations / class_counts[label]
            top_k_indices = torch.topk(mean_activation, k=k).indices
            top_k_per_class[label] = top_k_indices.cpu()
        
        return {
            "top_k_per_class": top_k_per_class,
            "k": k,
            "top_k_pct": self.top_k_pct,
        }
    
    def sharpen_features(
        self,
        sparse_code: torch.Tensor,
        labels: torch.Tensor,
        salient_features: dict,
    ) -> torch.Tensor:
        """Zero out all but top-k% features for each sample's class."""
        top_k_per_class = salient_features["top_k_per_class"]
        
        sharpened = torch.zeros_like(sparse_code)
        
        for label in labels.unique():
            mask = labels == label
            top_k_indices = top_k_per_class[label.item()].to(sparse_code.device)
            
            # Get row indices where this class appears
            row_indices = torch.where(mask)[0]
            
            # Use advanced indexing with broadcasting
            sharpened[row_indices[:, None], top_k_indices[None, :]] = \
                sparse_code[row_indices[:, None], top_k_indices[None, :]]
        
        return sharpened