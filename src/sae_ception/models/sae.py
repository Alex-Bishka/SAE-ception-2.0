import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """
    Standard L1-penalized Sparse Autoencoder.
    
    Uses L1 penalty on activations to encourage sparsity.
    Warning: Can suffer from dead features at high sparsity levels.
    """
    
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
        return reconstruction, sparse_code, {}
    
    def loss(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        sparse_code: torch.Tensor,
        info_dict: dict | None = None,
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


class TopKSparseAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        k: int = 50,
        auxk: int | None = None,
        aux_k_coef: float = 1/32,
        dead_steps_threshold: int = 75,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.auxk = auxk if auxk is not None else k * 2
        self.aux_k_coef = aux_k_coef
        self.dead_steps_threshold = dead_steps_threshold
        
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        self.latent_bias = nn.Parameter(torch.zeros(hidden_dim))
        
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        
        self.register_buffer('stats_last_nonzero', torch.zeros(hidden_dim, dtype=torch.long))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity='relu')
        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data.T.clone()
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    @torch.no_grad()
    def normalize_decoder_(self):
        """Call after optimizer.step()"""
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
    
    def encode_pre_act(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pre-activations (before ReLU and TopK)."""
        x = x - self.pre_bias
        return self.encoder(x) + self.latent_bias

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse representation (for inference/eval)."""
        pre_acts = self.encode_pre_act(x)
        pre_acts_relu = F.relu(pre_acts)
        topk_values, topk_indices = torch.topk(pre_acts_relu, k=self.k, dim=-1)

        # Use mask multiplication for robust gradient flow
        mask = torch.zeros_like(pre_acts_relu)
        mask.scatter_(-1, topk_indices, 1.0)
        sparse_code = pre_acts_relu * mask
        return sparse_code

    def decode(self, sparse_code: torch.Tensor) -> torch.Tensor:
        """Decode sparse representation back to input space."""
        return self.decoder(sparse_code) + self.pre_bias

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Training forward pass - tracks info needed for aux loss."""
        # 1. Compute pre-activations
        pre_acts = self.encode_pre_act(x)

        # 2. Top-K Selection
        pre_acts_relu = F.relu(pre_acts)
        topk_values, topk_indices = torch.topk(pre_acts_relu, k=self.k, dim=-1)

        # 3. Create Sparse Code using mask multiplication (robust gradient flow)
        # Create binary mask: 1 at top-k positions, 0 elsewhere
        mask = torch.zeros_like(pre_acts_relu)
        mask.scatter_(-1, topk_indices, 1.0)
        # Multiply: gradient flows through pre_acts_relu, mask just selects positions
        sparse_code = pre_acts_relu * mask

        # 4. Decode
        reconstruction = self.decode(sparse_code)
        
        # 5. Dead Feature Tracking (Training Only)
        if self.training:
            with torch.no_grad():
                # Only count as "fired" if activation value is meaningful
                fired_counts = torch.zeros(self.hidden_dim, device=x.device)
                fired_counts.scatter_add_(
                    0,
                    topk_indices.flatten(),
                    (topk_values > 1e-3).float().flatten()
                )
                fired_mask = fired_counts > 0
                
                # Reset counter for features that fired, increment for those that didn't
                self.stats_last_nonzero *= (~fired_mask).long()
                self.stats_last_nonzero += 1

        info = {
            'pre_acts': pre_acts,
            'topk_indices': topk_indices,
            'topk_values': topk_values,
        }
        
        return reconstruction, sparse_code, info

    def loss(
        self, 
        x: torch.Tensor, 
        reconstruction: torch.Tensor, 
        sparse_code: torch.Tensor, 
        info: dict
    ) -> dict:
        """Compute loss with aux-k for dead feature revival."""
        
        # 1. Main Reconstruction Loss
        reconstruction_loss = F.mse_loss(reconstruction, x)
        
        # 2. Auxiliary Loss for dead features
        aux_loss = torch.tensor(0.0, device=x.device)
        
        if self.training and self.aux_k_coef > 0:
            dead_mask = self.stats_last_nonzero > self.dead_steps_threshold
            
            if dead_mask.any():
                # Compute residual: what the main reconstruction missed
                # OpenAI computes: x - recons.detach() + pre_bias.detach()
                # This is equivalent to: (x - pre_bias) - (recons - pre_bias)
                # i.e., the residual in the centered space
                residual = x - reconstruction.detach()
                
                # Get pre-activations, mask out living features
                pre_acts = info['pre_acts']
                dead_pre_acts = pre_acts.clone()
                dead_pre_acts[:, ~dead_mask] = -float('inf')
                
                # Select top aux_k dead features
                n_dead = int(dead_mask.sum())
                k_aux = min(self.auxk, n_dead)
                
                if k_aux > 0:
                    aux_vals, aux_inds = torch.topk(dead_pre_acts, k=k_aux, dim=-1)

                    # Create sparse code using mask multiplication (robust gradient flow)
                    aux_mask = torch.zeros_like(pre_acts)
                    aux_mask.scatter_(-1, aux_inds, 1.0)
                    aux_sparse = F.relu(pre_acts) * aux_mask

                    # Decode aux features
                    aux_recon = self.decode(aux_sparse)
                    
                    # Aux loss: how well can dead features reconstruct the residual?
                    # OpenAI uses normalized MSE here
                    aux_loss = self._normalized_mse(aux_recon, residual)

        total_loss = reconstruction_loss + self.aux_k_coef * aux_loss
        
        # Logging metrics
        with torch.no_grad():
            l0 = (sparse_code > 0).float().sum(dim=-1).mean()
            dead_pct = (self.stats_last_nonzero > self.dead_steps_threshold).float().mean() * 100

        return {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "sparsity_loss": aux_loss,
            "l0_norm": l0,
            "dead_pct": dead_pct,
        }
    
    def _normalized_mse(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        MSE normalized by target variance.
        This prevents the aux loss scale from depending on activation magnitude.
        """
        mse = (recon - target).pow(2).mean(dim=-1)
        target_variance = (target - target.mean(dim=-1, keepdim=True)).pow(2).mean(dim=-1)
        
        # Avoid division by zero; if target has no variance, use raw MSE
        normalized = mse / (target_variance + 1e-6)
        return normalized.mean()

    @torch.no_grad()
    def resample_dead_neurons(
        self,
        activations: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        bias_resample_scale: float = 0.0,
        encoder_resample_scale: float = 0.2,
    ) -> int:
        """
        Resample dead neurons using high-loss examples (OpenAI approach).
        
        Args:
            activations: Batch of input activations [batch, input_dim]
            optimizer: Optimizer (to reset momentum for resampled neurons)
            bias_resample_scale: Scale for resampled encoder bias (0 = zero)
            encoder_resample_scale: Scale for encoder weights relative to avg alive norm
            
        Returns:
            Number of neurons resampled
        """
        # 1. Find dead neurons
        dead_mask = self.stats_last_nonzero > self.dead_steps_threshold
        n_dead = dead_mask.sum().item()
        
        if n_dead == 0:
            return 0
        
        dead_indices = torch.where(dead_mask)[0]
        
        # 2. Compute per-example reconstruction loss
        recon, _, _ = self(activations)
        per_example_loss = (recon - activations).pow(2).mean(dim=-1)  # [batch]
        
        # 3. Sample examples proportional to squared loss
        # (high-loss examples are underrepresented by current features)
        sampling_probs = per_example_loss.pow(2)
        sampling_probs = sampling_probs / sampling_probs.sum()
        
        # Sample with replacement if we need more dead neurons than examples
        n_samples = min(n_dead, len(activations))
        sampled_indices = torch.multinomial(
            sampling_probs, 
            num_samples=n_samples, 
            replacement=(n_dead > len(activations))
        )
        
        # If we have more dead neurons than samples, cycle through
        if n_dead > n_samples:
            sampled_indices = sampled_indices.repeat(
                (n_dead // n_samples) + 1
            )[:n_dead]
        
        sampled_activations = activations[sampled_indices]  # [n_dead, input_dim]
        
        # 4. Center the sampled activations (subtract pre_bias)
        centered = sampled_activations - self.pre_bias
        
        # 5. Compute average encoder norm of alive neurons for scaling
        alive_mask = ~dead_mask
        if alive_mask.any():
            alive_encoder_norms = self.encoder.weight[alive_mask].norm(dim=1)
            avg_alive_norm = alive_encoder_norms.mean().item()
        else:
            avg_alive_norm = 1.0
        
        # 6. Reinitialize dead neurons
        for i, dead_idx in enumerate(dead_indices):
            direction = centered[i]
            
            # Normalize and scale encoder weight
            direction_norm = direction.norm()
            if direction_norm > 1e-6:
                normalized_direction = direction / direction_norm
            else:
                # Fallback to random if example has zero norm
                normalized_direction = torch.randn_like(direction)
                normalized_direction = normalized_direction / normalized_direction.norm()
            
            # Set encoder row (scaled to average alive norm)
            self.encoder.weight.data[dead_idx] = (
                normalized_direction * avg_alive_norm * encoder_resample_scale
            )
            
            # Set decoder column to same direction (will be normalized after)
            self.decoder.weight.data[:, dead_idx] = normalized_direction
            
            # Reset encoder bias for this neuron
            self.latent_bias.data[dead_idx] = bias_resample_scale
            
            # Reset dead tracking counter
            self.stats_last_nonzero[dead_idx] = 0
        
        # 7. Normalize decoder (maintains unit norm constraint)
        self.normalize_decoder_()
        
        # 8. Reset optimizer state for resampled neurons
        self._reset_optimizer_state(optimizer, dead_indices)
        
        return n_dead

    def _reset_optimizer_state(
        self, 
        optimizer: torch.optim.Optimizer, 
        neuron_indices: torch.Tensor
    ):
        """Reset optimizer momentum/state for specific neurons."""
        neuron_indices = neuron_indices.tolist()
        
        for group in optimizer.param_groups:
            for param in group['params']:
                if param not in optimizer.state:
                    continue
                    
                state = optimizer.state[param]
                
                # Handle Adam optimizer state
                if 'exp_avg' in state:
                    # Check if this is encoder weight [hidden, input]
                    if param.shape == self.encoder.weight.shape:
                        state['exp_avg'][neuron_indices] = 0
                        state['exp_avg_sq'][neuron_indices] = 0
                    # Check if this is decoder weight [input, hidden]
                    elif param.shape == self.decoder.weight.shape:
                        state['exp_avg'][:, neuron_indices] = 0
                        state['exp_avg_sq'][:, neuron_indices] = 0
                    # Check if this is latent_bias [hidden]
                    elif param.shape == self.latent_bias.shape:
                        state['exp_avg'][neuron_indices] = 0
                        state['exp_avg_sq'][neuron_indices] = 0


class TopKSparseAutoencoderSTE(TopKSparseAutoencoder):
    """
    TopK SAE with Straight-Through Estimator for better gradient flow.
    
    The standard top-k operation has zero gradients for non-selected features.
    This variant uses a straight-through estimator to allow gradients to flow
    to all features during backprop, while still enforcing top-k during forward.
    
    This can help with training dynamics and feature utilization.
    """
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode with straight-through estimator for top-k.
        """
        # Get pre-activations
        pre_acts = self.encoder(x)
        acts = F.relu(pre_acts)
        
        actual_k = min(self.k, self.hidden_dim)
        
        # Get top-k mask
        topk_values, topk_indices = torch.topk(acts, k=actual_k, dim=-1)
        
        # Create mask
        mask = torch.zeros_like(acts)
        mask.scatter_(dim=-1, index=topk_indices, value=1.0)
        
        # Straight-through: forward uses masked, backward gets full gradients
        # Detach the mask so gradients flow through acts for all positions
        sparse_code = acts * mask.detach()
        
        return sparse_code


def create_sae(
    sae_type: str,
    input_dim: int,
    hidden_dim: int,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create the appropriate SAE type.
    
    Args:
        sae_type: One of 'l1', 'topk', 'topk_ste'
        input_dim: Dimension of input activations
        hidden_dim: Dimension of sparse hidden layer
        **kwargs: Additional arguments passed to SAE constructor
            - For 'l1': l1_penalty (float)
            - For 'topk'/'topk_ste': k (int), aux_k_coef (float)
    
    Returns:
        SAE module
    """
    sae_type = sae_type.lower()
    
    if sae_type == 'l1':
        l1_penalty = kwargs.get('l1_penalty', 1e-4)
        return SparseAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            l1_penalty=l1_penalty,
        )
    
    elif sae_type == 'topk':
        return TopKSparseAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            k=kwargs.get('k', 50),
            auxk=kwargs.get('auxk', None),
            aux_k_coef=kwargs.get('aux_k_coef', 1/32),
            dead_steps_threshold=kwargs.get('dead_steps_threshold', 75),
        )
    
    elif sae_type == 'topk_ste':
        k = kwargs.get('k', 50)
        aux_k_coef = kwargs.get('aux_k_coef', 1/32)
        return TopKSparseAutoencoderSTE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            k=k,
            aux_k_coef=aux_k_coef,
        )
    
    else:
        raise ValueError(
            f"Unknown SAE type: {sae_type}. "
            f"Choose from: 'l1', 'topk', 'topk_ste'"
        )