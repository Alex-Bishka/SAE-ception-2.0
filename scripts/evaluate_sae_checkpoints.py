#!/usr/bin/env python3
"""
Evaluate SAE checkpoints for Pareto frontier selection.

This script evaluates multiple SAE checkpoints on a held-out test set and
computes metrics for Pareto frontier analysis:
- L0: Average number of active features per token
- CE Loss Recovered: Reconstruction quality metric

Usage:
    python evaluate_sae_checkpoints.py \
        --checkpoint-dir checkpoints/sae_k32/ \
        --model EleutherAI/pythia-410m \
        --layer 12 \
        --test-samples 500000 \
        --output results/pareto_k32.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from sae_ception.models.sae import create_sae
from sae_ception.utils.data import create_causal_lm_dataloader, stream_activations

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def evaluate_checkpoint(
    sae: torch.nn.Module,
    model: torch.nn.Module,
    tokenizer,
    layer_idx: int,
    n_samples: int = 500000,
    device: str = 'cuda',
    dataset_name: str = 'pile',
    exclude_bos: bool = True,
) -> Dict[str, float]:
    """
    Evaluate an SAE checkpoint on held-out test data.

    Returns:
        Dictionary with L0, reconstruction metrics, etc.
    """
    sae.eval()
    model.eval()

    # Create test dataloader (uses 20M offset for pile)
    seq_len = 512
    avg_tokens_per_seq = seq_len * 0.7
    n_sequences = int(n_samples / avg_tokens_per_seq) + 100

    test_loader = create_causal_lm_dataloader(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        split='test',  # Uses 20M offset for held-out data
        batch_size=64,
        max_length=seq_len,
        max_samples=n_sequences,
    )

    # Collect metrics
    total_l0 = 0.0
    total_recon_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for act_batch in tqdm(
            stream_activations(
                model=model,
                dataloader=test_loader,
                layer_idx=layer_idx,
                device=device,
                max_tokens=n_samples,
                show_progress=False,
                exclude_bos=exclude_bos,
            ),
            desc="Evaluating",
            total=n_samples // 256,  # Rough estimate
        ):
            x = act_batch.to(device)

            # Forward through SAE
            recon, sparse, info = sae(x)

            # L0: number of non-zero features per token
            if sparse.dim() == 2:  # [batch, hidden]
                l0 = (sparse != 0).float().sum(dim=-1).mean().item()
            else:
                l0 = (sparse != 0).float().mean().item() * sparse.shape[-1]

            # Reconstruction MSE
            mse = torch.nn.functional.mse_loss(recon, x).item()

            batch_size = x.shape[0]
            total_l0 += l0 * batch_size
            total_recon_loss += mse * batch_size
            total_samples += batch_size

            if total_samples >= n_samples:
                break

    if total_samples == 0:
        return {'error': 'No samples evaluated'}

    avg_l0 = total_l0 / total_samples
    avg_recon_loss = total_recon_loss / total_samples

    return {
        'l0': avg_l0,
        'reconstruction_mse': avg_recon_loss,
        'samples_evaluated': total_samples,
    }


def load_checkpoint(checkpoint_path: Path, device: str = 'cuda') -> torch.nn.Module:
    """Load SAE from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})

    sae = create_sae(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        sae_type='topk',
        k=config['k'],
    )
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.to(device)
    sae.eval()

    return sae, config, checkpoint.get('training', {})


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAE checkpoints for Pareto frontier")

    parser.add_argument('--checkpoint-dir', type=str, required=True,
                        help='Directory containing SAE checkpoints')
    parser.add_argument('--model', type=str, default='EleutherAI/pythia-410m',
                        help='Base model for activation extraction')
    parser.add_argument('--layer', type=int, required=True,
                        help='Layer to extract activations from')
    parser.add_argument('--test-samples', type=int, default=500000,
                        help='Number of test samples for evaluation')
    parser.add_argument('--dataset', type=str, default='pile',
                        help='Dataset for evaluation (pile, wikitext)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path for results')
    parser.add_argument('--plot', type=str, default=None,
                        help='Output path for Pareto frontier plot')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--exclude-bos', action='store_true', default=True,
                        help='Exclude BOS tokens from evaluation')

    args = parser.parse_args()

    # Find all checkpoints
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
        return

    checkpoints = sorted(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        logger.error(f"No .pt files found in {checkpoint_dir}")
        return

    logger.info(f"Found {len(checkpoints)} checkpoints")

    # Load model and tokenizer
    logger.info(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(args.device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Evaluate each checkpoint
    results: List[Dict] = []

    for ckpt_path in checkpoints:
        logger.info(f"\nEvaluating: {ckpt_path.name}")

        try:
            sae, config, training_info = load_checkpoint(ckpt_path, args.device)

            metrics = evaluate_checkpoint(
                sae=sae,
                model=model,
                tokenizer=tokenizer,
                layer_idx=args.layer,
                n_samples=args.test_samples,
                device=args.device,
                dataset_name=args.dataset,
                exclude_bos=args.exclude_bos,
            )

            result = {
                'checkpoint': ckpt_path.name,
                'step': training_info.get('step', 0),
                'k': config.get('k', 0),
                **metrics,
            }
            results.append(result)

            logger.info(f"  L0: {metrics['l0']:.2f}, Recon MSE: {metrics['reconstruction_mse']:.6f}")

        except Exception as e:
            logger.error(f"  Error evaluating {ckpt_path.name}: {e}")
            continue

    # Sort by step
    results.sort(key=lambda x: x.get('step', 0))

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("PARETO FRONTIER SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Checkpoint':<25} {'Step':>8} {'L0':>8} {'Recon MSE':>12}")
    logger.info("-" * 60)

    for r in results:
        logger.info(
            f"{r['checkpoint']:<25} {r.get('step', 0):>8} "
            f"{r['l0']:>8.2f} {r['reconstruction_mse']:>12.6f}"
        )

    # Find best checkpoint (L0 closest to k with lowest reconstruction error)
    if results:
        target_k = results[0].get('k', 32)
        best = min(results, key=lambda x: (abs(x['l0'] - target_k), x['reconstruction_mse']))
        logger.info("-" * 60)
        logger.info(f"Recommended (L0 ≈ {target_k}): {best['checkpoint']}")
        logger.info(f"  L0: {best['l0']:.2f}, Recon MSE: {best['reconstruction_mse']:.6f}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                'model': args.model,
                'layer': args.layer,
                'test_samples': args.test_samples,
                'checkpoints': results,
            }, f, indent=2)
        logger.info(f"\nResults saved to: {output_path}")

    # Generate plot if requested
    if args.plot:
        try:
            import matplotlib.pyplot as plt

            l0_values = [r['l0'] for r in results]
            mse_values = [r['reconstruction_mse'] for r in results]
            steps = [r.get('step', i) for i, r in enumerate(results)]

            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(l0_values, mse_values, c=steps, cmap='viridis', s=100)

            # Add labels
            for r in results:
                ax.annotate(
                    r['checkpoint'].replace('.pt', ''),
                    (r['l0'], r['reconstruction_mse']),
                    fontsize=8,
                    alpha=0.7,
                )

            ax.set_xlabel('L0 (Average Active Features)')
            ax.set_ylabel('Reconstruction MSE')
            ax.set_title(f'Pareto Frontier - {args.model} Layer {args.layer}')

            # Add target k line
            if results:
                target_k = results[0].get('k', 32)
                ax.axvline(x=target_k, color='r', linestyle='--', alpha=0.5, label=f'Target k={target_k}')
                ax.legend()

            plt.colorbar(scatter, label='Training Step')
            plt.tight_layout()

            plot_path = Path(args.plot)
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path, dpi=150)
            logger.info(f"Plot saved to: {plot_path}")

        except ImportError:
            logger.warning("matplotlib not installed, skipping plot generation")


if __name__ == "__main__":
    main()
