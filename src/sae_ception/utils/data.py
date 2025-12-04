"""Data loading and preprocessing utilities."""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer
from typing import Dict, Optional, List, Tuple
from omegaconf import DictConfig
from tqdm import tqdm


class TextClassificationDataset(Dataset):
    """Dataset for text classification tasks."""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 128,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
        }


class ActivationDataset(Dataset):
    """Dataset storing pre-extracted activations and labels."""
    
    def __init__(
        self,
        activations: torch.Tensor,
        labels: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            activations: Shape [N, hidden_dim]
            labels: Shape [N]
            targets: Optional sharpened targets, shape [N, hidden_dim]
            token_ids: Optional token IDs, shape [N] (for SAEBench)
        """
        self.activations = activations
        self.labels = labels
        self.targets = targets
        self.token_ids = token_ids
    
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        item = {
            'activations': self.activations[idx],
            'labels': self.labels[idx],
        }
        
        if self.targets is not None:
            item['targets'] = self.targets[idx]
        
        if self.token_ids is not None:
            item['token_ids'] = self.token_ids[idx]
        
        return item


def load_classification_dataset(cfg: DictConfig, tokenizer, split: str = 'train'):
    """
    Load a classification dataset from HuggingFace.
    
    Args:
        cfg: Dataset configuration
        tokenizer: Tokenizer to use
        split: Dataset split ('train', 'validation', 'test')
        
    Returns:
        TextClassificationDataset
    """
    # Load dataset
    if cfg.hf_name:
        dataset = load_dataset(cfg.hf_path, cfg.hf_name, split=split)
    else:
        dataset = load_dataset(cfg.hf_path, split=split)
    
    # Extract texts and labels
    texts = dataset[cfg.text_column]
    labels = dataset[cfg.label_column]
    
    # Create dataset
    return TextClassificationDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
    )


def create_dataloaders(
    cfg: DictConfig,
    tokenizer,
) -> Dict[str, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        cfg: Configuration
        tokenizer: Tokenizer to use
        
    Returns:
        Dictionary with 'train' and 'val' dataloaders
    """
    # Load datasets
    train_dataset = load_classification_dataset(cfg.dataset, tokenizer, split='train')
    
    # Try to load validation set
    try:
        val_dataset = load_classification_dataset(cfg.dataset, tokenizer, split='validation')
    except:
        # If no validation split, use test or create from train
        try:
            val_dataset = load_classification_dataset(cfg.dataset, tokenizer, split='test')
        except:
            # Split from train
            val_size = int(len(train_dataset) * cfg.dataset.validation_split)
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(cfg.seed)
            )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True,
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
    }


# =============================================================================
# TOKEN-LEVEL ACTIVATION EXTRACTION (NEW - SAEBench Compatible)
# =============================================================================

def extract_activations_all_tokens(
    model: torch.nn.Module,
    dataloader: DataLoader,
    layer_name: str,
    device: str = 'cuda',
    include_padding: bool = False,
    show_progress: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract activations for ALL tokens in the dataset.
    
    This is the SAEBench-compatible extraction method. Each token gets its own
    activation vector, rather than extracting only the final token per sequence.
    
    Args:
        model: Model to extract from
        dataloader: DataLoader providing inputs
        layer_name: Layer to extract activations from (e.g., '-1' for last)
        device: Device to run on
        include_padding: Whether to include padding token activations
        show_progress: Whether to show progress bar
        
    Returns:
        activations: Shape [total_tokens, hidden_dim]
        token_ids: Shape [total_tokens] - the token ID at each position
        sequence_labels: Shape [total_tokens] - inherited class label from sequence
    """
    from .hooks import ActivationExtractor
    
    model.eval()
    model.to(device)
    
    all_activations = []
    all_token_ids = []
    all_labels = []
    
    iterator = tqdm(dataloader, desc="Extracting all-token activations") if show_progress else dataloader
    
    with torch.no_grad():
        for batch in iterator:
            input_ids = batch['input_ids'].to(device)      # [batch, seq_len]
            attention_mask = batch['attention_mask'].to(device)  # [batch, seq_len]
            labels = batch['labels']  # [batch]
            
            batch_size, seq_len = input_ids.shape
            
            # Get activations for all positions
            with ActivationExtractor(model, layer_name) as extractor:
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
                activations = extractor.get_activations()  # [batch, seq_len, hidden]
            
            # Handle case where activations might be 2D (already pooled)
            if activations.dim() == 2:
                # Model already returns pooled output, can't do token-level
                raise ValueError(
                    f"Model returns 2D activations (shape {activations.shape}). "
                    "Token-level extraction requires 3D activations [batch, seq, hidden]. "
                    "Check your target_layer setting."
                )
            
            # Flatten and filter
            for b in range(batch_size):
                for t in range(seq_len):
                    # Skip padding tokens unless requested
                    if not include_padding and attention_mask[b, t] == 0:
                        continue
                    
                    all_activations.append(activations[b, t].cpu())
                    all_token_ids.append(input_ids[b, t].cpu())
                    all_labels.append(labels[b])  # Inherit sequence label
    
    return (
        torch.stack(all_activations),
        torch.stack(all_token_ids),
        torch.stack(all_labels),
    )


def extract_activations_final_token(
    model: torch.nn.Module,
    dataloader: DataLoader,
    layer_name: str,
    device: str = 'cuda',
    show_progress: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract activations for FINAL token only (for classification).
    
    This is the original extraction method - used for:
    - Sharpening (class labels only meaningful at sequence level)
    - Auxiliary loss computation
    - Sequence-level evaluation metrics
    
    Args:
        model: Model to extract from
        dataloader: DataLoader providing inputs
        layer_name: Layer to extract activations from
        device: Device to run on
        show_progress: Whether to show progress bar
        
    Returns:
        activations: Shape [N_examples, hidden_dim]
        labels: Shape [N_examples]
    """
    from .hooks import ActivationExtractor, extract_final_token_activation
    
    model.eval()
    model.to(device)
    
    all_activations = []
    all_labels = []
    
    iterator = tqdm(dataloader, desc="Extracting final-token activations") if show_progress else dataloader
    
    with torch.no_grad():
        for batch in iterator:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            # Forward pass (activations are captured by hook)
            with ActivationExtractor(model, layer_name) as extractor:
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
                activations = extractor.get_activations()
            
            # Extract final token
            activations = extract_final_token_activation(activations, token_idx=-1)
            
            all_activations.append(activations.cpu())
            all_labels.append(labels)
    
    activations = torch.cat(all_activations, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    return activations, labels


# Legacy function - redirects to final_token version
def extract_activations_from_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    layer_name: str,
    device: str = 'cuda',
    token_position: str = 'last',  # 'last', 'first' ([CLS]), 'mean', or 'all'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract activations from a model for entire dataset.
    
    DEPRECATED: Use extract_activations_final_token() or extract_activations_all_tokens()
    
    Args:
        model: Model to extract from
        dataloader: DataLoader providing inputs
        layer_name: Layer to extract activations from
        device: Device to run on
        token_position: Which token to extract ('last', 'first', 'mean')
        
    Returns:
        activations: Shape [N, hidden_dim]
        labels: Shape [N]
    """
    from .hooks import ActivationExtractor, extract_final_token_activation, extract_cls_token_activation
    
    if token_position == 'all':
        # Return token-level activations
        acts, token_ids, labels = extract_activations_all_tokens(
            model, dataloader, layer_name, device
        )
        return acts, labels  # Note: token_ids not returned for backward compat
    
    model.eval()
    model.to(device)
    
    all_activations = []
    all_labels = []
    
    with torch.no_grad():
        with ActivationExtractor(model, layer_name) as extractor:
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels']
                
                # Forward pass (activations are captured by hook)
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Get cached activations
                activations = extractor.get_activations()
                
                # Extract token position
                if token_position == 'last':
                    activations = extract_final_token_activation(activations, token_idx=-1)
                elif token_position == 'first':
                    activations = extract_cls_token_activation(activations)
                elif token_position == 'mean':
                    # Mean pooling over sequence
                    if activations.dim() == 3:
                        # Weight by attention mask
                        mask = attention_mask.unsqueeze(-1).float()
                        activations = (activations * mask).sum(dim=1) / mask.sum(dim=1)
                else:
                    raise ValueError(f"Unknown token_position: {token_position}")
                
                all_activations.append(activations.cpu())
                all_labels.append(labels)
                
                # Clear cache for next batch
                extractor.cache.clear()
    
    activations = torch.cat(all_activations, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    return activations, labels


def create_activation_dataloader(
    activations: torch.Tensor,
    labels: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    token_ids: Optional[torch.Tensor] = None,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a dataloader from pre-extracted activations.
    
    Args:
        activations: Pre-extracted activations [N, hidden_dim]
        labels: Labels [N]
        targets: Optional sharpened targets [N, hidden_dim]
        token_ids: Optional token IDs [N] (for SAEBench evaluations)
        batch_size: Batch size
        shuffle: Whether to shuffle
        
    Returns:
        DataLoader
    """
    dataset = ActivationDataset(activations, labels, targets, token_ids)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
    )