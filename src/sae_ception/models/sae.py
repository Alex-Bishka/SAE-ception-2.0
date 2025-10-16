import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder for extracting interpretable features."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        l1_penalty: float = 1e-4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.l1_penalty = l1_penalty
        
        # Encoder: input_dim -> hidden_dim
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        
        # Decoder: hidden_dim -> input_dim
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse representation."""
        return F.relu(self.encoder(x))
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode sparse representation back to input space."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through SAE.
        
        Returns:
            reconstruction: Reconstructed input
            sparse_code: Sparse hidden representation
        """
        sparse_code = self.encode(x)
        reconstruction = self.decode(sparse_code)
        return reconstruction, sparse_code
    
    def loss(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        sparse_code: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute SAE loss.
        
        Returns dictionary with:
            - total_loss
            - reconstruction_loss
            - sparsity_loss
            - l0_norm (for logging)
        """
        # Reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(reconstruction, x)
        
        # Sparsity loss (L1 on activations)
        sparsity_loss = self.l1_penalty * sparse_code.abs().sum(dim=-1).mean()
        
        # Total loss
        total_loss = reconstruction_loss + sparsity_loss
        
        # L0 norm for logging (average number of active features)
        with torch.no_grad():
            l0_norm = (sparse_code > 0).float().sum(dim=-1).mean()
        
        return {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "sparsity_loss": sparsity_loss,
            "l0_norm": l0_norm,
        }
