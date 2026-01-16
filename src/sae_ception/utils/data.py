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
    has_labels = None  # Will be determined from first batch
    
    iterator = tqdm(dataloader, desc="Extracting all-token activations") if show_progress else dataloader
    
    with torch.no_grad():
        for batch in iterator:
            input_ids = batch['input_ids'].to(device)      # [batch, seq_len]
            attention_mask = batch['attention_mask'].to(device)  # [batch, seq_len]
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)  # [batch, seq_len]
            else:
                # If no attention mask, assume all tokens are valid
                attention_mask = torch.ones_like(input_ids)

            # Check if labels exist (first batch only)
            labels = batch.get('labels', None)
            if has_labels is None:
                has_labels = 'labels' in batch
            
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
                    
                    if has_labels:
                        all_labels.append(labels[b])

    result_activations = torch.stack(all_activations)
    result_token_ids = torch.stack(all_token_ids)
    
    if has_labels and all_labels:
        result_labels = torch.stack(all_labels)
    else:
        result_labels = None
    
    return result_activations, result_token_ids, result_labels


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


# =============================================================================
# STREAMING ACTIVATION EXTRACTION (Memory-efficient for large datasets)
# =============================================================================

def stream_activations(
    model: torch.nn.Module,
    dataloader: DataLoader,
    layer_idx: int,
    device: str = 'cuda',
    max_tokens: Optional[int] = None,
    show_progress: bool = True,
    exclude_bos: bool = False,
):
    """
    Generator that yields batches of activations on-the-fly.

    This is memory-efficient as it doesn't store all activations at once.
    Each yield returns a batch of token activations extracted from the model.

    Args:
        model: The language model
        dataloader: DataLoader yielding tokenized text batches
        layer_idx: Which layer to extract activations from
        device: Device to run on
        max_tokens: Maximum number of tokens to yield (None = unlimited)
        show_progress: Whether to show progress bar
        exclude_bos: If True, exclude position 0 (BOS) tokens from each sequence

    Yields:
        torch.Tensor: Batch of activations [batch_tokens, hidden_dim]
    """
    from sae_ception.utils.hooks import extract_layer_output

    model.eval()
    layer_name = str(layer_idx)
    cache = extract_layer_output(model, layer_name)

    total_tokens = 0
    pbar = tqdm(dataloader, desc="Streaming activations", disable=not show_progress)

    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Forward pass to capture activations (no grad for LM only)
        with torch.no_grad():
            _ = model(input_ids, attention_mask=attention_mask)

            # Get activations from cache (hook stores as "layer_{idx}")
            cache_key = f"layer_{layer_idx}"
            acts = cache.activations[cache_key]  # [batch, seq_len, hidden]

            batch_size, seq_len, hidden_dim = acts.shape

            # Build mask for valid tokens
            if attention_mask is not None:
                mask = attention_mask.bool()  # [batch, seq_len]
            else:
                mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

            # Exclude BOS (position 0) if requested
            if exclude_bos:
                mask[:, 0] = False

            # Flatten and apply mask
            mask_flat = mask.view(-1)
            acts_flat = acts.view(-1, hidden_dim)[mask_flat]

            cache.clear()

        # Check if we've hit the limit
        if max_tokens is not None:
            remaining = max_tokens - total_tokens
            if remaining <= 0:
                break
            if len(acts_flat) > remaining:
                acts_flat = acts_flat[:remaining]

        total_tokens += len(acts_flat)
        if show_progress:
            pbar.set_postfix({'tokens': total_tokens})

        # Yield OUTSIDE of no_grad so caller can use gradients
        # Clone to release the source tensor's storage and prevent memory accumulation
        yield acts_flat.clone()

        if max_tokens is not None and total_tokens >= max_tokens:
            break

    cache.remove_hooks()


class StreamingActivationDataset(torch.utils.data.IterableDataset):
    """
    Iterable dataset that streams activations from a model.

    This wraps stream_activations for use with DataLoader.
    Note: Due to streaming nature, len() is not available.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        text_dataloader: DataLoader,
        layer_idx: int,
        device: str = 'cuda',
        max_tokens: Optional[int] = None,
    ):
        self.model = model
        self.text_dataloader = text_dataloader
        self.layer_idx = layer_idx
        self.device = device
        self.max_tokens = max_tokens

    def __iter__(self):
        for batch in stream_activations(
            model=self.model,
            dataloader=self.text_dataloader,
            layer_idx=self.layer_idx,
            device=self.device,
            max_tokens=self.max_tokens,
            show_progress=False,
        ):
            # Yield individual samples for DataLoader to batch
            for i in range(len(batch)):
                yield batch[i]


# =============================================================================
# CAUSAL LM DATA LOADING (NEW - for CPT experiments)
# =============================================================================

class CausalLMDataset(Dataset):
    """Dataset for causal language modeling."""
    
    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            input_ids: [N, seq_len] tokenized sequences
            attention_mask: [N, seq_len] optional attention mask
        """
        self.input_ids = input_ids
        self.attention_mask = attention_mask
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        item = {'input_ids': self.input_ids[idx]}
        if self.attention_mask is not None:
            item['attention_mask'] = self.attention_mask[idx]
        return item


def load_wikitext_for_eval(
    tokenizer,
    split: str = 'test',
    max_length: int = 1024,
    max_samples: Optional[int] = None,
) -> CausalLMDataset:
    """
    Load WikiText-103 for perplexity evaluation.
    
    Args:
        tokenizer: Tokenizer to use
        split: 'train', 'validation', or 'test'
        max_length: Maximum sequence length
        max_samples: Limit number of samples (for quick tests)
        
    Returns:
        CausalLMDataset
    """
    from datasets import load_dataset
    
    dataset = load_dataset('wikitext', 'wikitext-103-v1', split=split)
    
    # Filter empty lines and tokenize
    texts = [t for t in dataset['text'] if len(t.strip()) > 0]
    
    if max_samples is not None:
        texts = texts[:max_samples]
    
    # Tokenize
    encodings = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
    )
    
    return CausalLMDataset(
        input_ids=encodings['input_ids'],
        attention_mask=encodings['attention_mask'],
    )


def create_causal_lm_dataloader(
    tokenizer,
    dataset_name: str = 'wikitext',
    split: str = 'test',
    batch_size: int = 8,
    max_length: int = 1024,
    max_samples: Optional[int] = None,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create a dataloader for causal LM evaluation.

    Args:
        tokenizer: Tokenizer
        dataset_name: 'wikitext' or 'pile'
        split: Dataset split
        batch_size: Batch size
        max_length: Max sequence length
        max_samples: Limit samples
        num_workers: DataLoader workers

    Returns:
        DataLoader
    """
    if dataset_name == 'wikitext':
        dataset = load_wikitext_for_eval(
            tokenizer, split, max_length, max_samples
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    elif dataset_name == 'pile':
        # Use the batched pile loader which pre-downloads samples
        # train: 0-10M samples, test: 20M+ (held-out for Pareto evaluation)
        skip = 0
        if split == 'test':
            skip = 20_000_000  # Held-out test set, well past training data
            print(f"Test (held-out): skipping first {skip:,} Pile samples")

        return create_pile_dataloader_batched(
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            max_samples=max_samples or 100000,
            num_workers=num_workers,
            skip_samples=skip,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: wikitext, pile")

# =============================================================================
# ADD THESE FUNCTIONS TO src/sae_ception/utils/data.py
# Place after the create_causal_lm_dataloader function
# =============================================================================

def load_pile_streaming(
    tokenizer,
    max_length: int = 1024,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> "IterableDataset":
    """
    Load The Pile dataset in streaming mode.
    
    Uses the uncopyrighted subset which is more accessible.
    
    Args:
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        max_samples: Limit number of samples (None = unlimited)
        seed: Random seed for shuffling
        
    Returns:
        IterableDataset that yields tokenized examples
    """
    from datasets import load_dataset
    
    # Load streaming dataset
    dataset = load_dataset(
        "monology/pile-uncopyrighted",
        split="train",
        streaming=True,
    )
    
    # Shuffle with buffer
    dataset = dataset.shuffle(seed=seed, buffer_size=10000)
    
    # Limit samples if specified
    if max_samples is not None:
        dataset = dataset.take(max_samples)
    
    def tokenize_fn(examples):
        """Tokenize and prepare for causal LM."""
        # Tokenize
        tokenized = tokenizer(
            examples["text"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
        }
    
    # Apply tokenization
    dataset = dataset.map(tokenize_fn, remove_columns=["text", "meta"])
    
    return dataset


def create_pile_dataloader(
    tokenizer,
    batch_size: int = 4,
    max_length: int = 1024,
    max_samples: Optional[int] = None,
    num_workers: int = 0,  # Streaming doesn't work well with multiprocessing
    seed: int = 42,
) -> DataLoader:
    """
    Create a DataLoader for The Pile (streaming).
    
    Args:
        tokenizer: Tokenizer
        batch_size: Batch size
        max_length: Max sequence length
        max_samples: Limit samples (None = unlimited)
        num_workers: DataLoader workers (0 recommended for streaming)
        seed: Random seed
        
    Returns:
        DataLoader that yields batches from The Pile
    """
    dataset = load_pile_streaming(
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=max_samples,
        seed=seed,
    )
    
    # For streaming datasets, we need to use IterableDataset
    # Can't use standard DataLoader shuffle
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )


def create_pile_dataloader_batched(
    tokenizer,
    batch_size: int = 4,
    max_length: int = 1024,
    max_samples: int = 100000,
    num_workers: int = 4,
    seed: int = 42,
    skip_samples: int = 0,
) -> DataLoader:
    """
    Create a DataLoader for The Pile by pre-loading samples.

    This loads a fixed number of samples into memory, which allows
    proper shuffling and multi-worker loading. Better for training
    but requires more memory.

    Args:
        tokenizer: Tokenizer
        batch_size: Batch size
        max_length: Max sequence length
        max_samples: Number of samples to load
        num_workers: DataLoader workers
        seed: Random seed
        skip_samples: Number of samples to skip (for validation offset)

    Returns:
        DataLoader with pre-loaded Pile samples
    """
    from datasets import load_dataset
    import torch

    skip_msg = f" (skipping first {skip_samples:,})" if skip_samples > 0 else ""
    print(f"Loading {max_samples:,} samples from The Pile{skip_msg}...")

    # Load streaming dataset
    dataset = load_dataset(
        "monology/pile-uncopyrighted",
        split="train",
        streaming=True,
    )

    # Skip samples efficiently using HuggingFace's .skip() method
    # This doesn't load skipped samples, just advances the iterator
    if skip_samples > 0:
        print(f"  Skipping {skip_samples:,} samples (this may take a moment)...")
        dataset = dataset.skip(skip_samples)

    # Shuffle after skip (so validation gets different ordering than if we hadn't skipped)
    dataset = dataset.shuffle(seed=seed, buffer_size=10000)

    # Collect samples
    texts = []
    for i, example in enumerate(dataset):
        if i >= max_samples:
            break
        if len(example["text"].strip()) > 0:
            texts.append(example["text"])
        if (i + 1) % 10000 == 0:
            print(f"  Loaded {len(texts):,} samples...")
    
    print(f"Loaded {len(texts)} non-empty samples")
    
    # Tokenize all at once
    print("Tokenizing...")
    encodings = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    
    # Create dataset
    pile_dataset = CausalLMDataset(
        input_ids=encodings["input_ids"],
        attention_mask=encodings["attention_mask"],
    )
    
    return DataLoader(
        pile_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )