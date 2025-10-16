"""Data loading and preprocessing utilities."""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer
from typing import Dict, Optional, List
from omegaconf import DictConfig


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
    ):
        """
        Args:
            activations: Shape [N, hidden_dim]
            labels: Shape [N]
            targets: Optional sharpened targets, shape [N, hidden_dim]
        """
        self.activations = activations
        self.labels = labels
        self.targets = targets
    
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        item = {
            'activations': self.activations[idx],
            'labels': self.labels[idx],
        }
        
        if self.targets is not None:
            item['targets'] = self.targets[idx]
        
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


def extract_activations_from_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    layer_name: str,
    device: str = 'cuda',
    token_position: str = 'last',  # 'last', 'first' ([CLS]), or 'mean'
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract activations from a model for entire dataset.
    
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
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a dataloader from pre-extracted activations.
    
    Args:
        activations: Pre-extracted activations [N, hidden_dim]
        labels: Labels [N]
        targets: Optional sharpened targets [N, hidden_dim]
        batch_size: Batch size
        shuffle: Whether to shuffle
        
    Returns:
        DataLoader
    """
    dataset = ActivationDataset(activations, labels, targets)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
    )