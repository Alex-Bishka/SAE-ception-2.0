#!/usr/bin/env python3
"""
Unified perplexity comparison across checkpoints.

Loads multiple model checkpoints and evaluates them on identical held-out data
to ensure fair comparison.

Usage:
    python scripts/compare_perplexity.py \
        --checkpoints \
            checkpoints/model_control.pt \
            checkpoints/model_l1_0.001.pt \
            checkpoints/model_saeception.pt \
        --baseline EleutherAI/pythia-70m \
        --eval_samples 1000 \
        --output results/comparison.json
        
    # Or compare all checkpoints in a directory
    python scripts/compare_perplexity.py \
        --checkpoint_dir checkpoints/ \
        --baseline EleutherAI/pythia-70m \
        --output results/comparison.json
"""

import argparse
import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Tuple
import logging
from tabulate import tabulate

from sae_ception.utils.data import create_causal_lm_dataloader
from sae_ception.evaluation.lm_metrics import evaluate_perplexity

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: str = 'cuda',
) -> Tuple[torch.nn.Module, Dict]:
    """
    Load a model from a training checkpoint.
    
    Returns:
        Tuple of (model, config_dict)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get the base model name from checkpoint config
    config = checkpoint.get('config', {})
    model_name = config.get('model_name', 'EleutherAI/pythia-70m')
    
    # Load base model architecture
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def load_model(
    model_path: str,
    device: str = 'cuda',
) -> Tuple[torch.nn.Module, str, Dict]:
    """
    Load a model from either HuggingFace name or local checkpoint.
    
    Returns:
        Tuple of (model, display_name, config_dict)
    """
    path = Path(model_path)
    
    if path.exists() and path.suffix == '.pt':
        # Local checkpoint
        model, config = load_model_from_checkpoint(path, device)
        display_name = path.stem
        return model, display_name, config
    else:
        # HuggingFace model
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.to(device)
        model.eval()
        display_name = model_path.split('/')[-1]
        return model, display_name, {'model_name': model_path}


def find_checkpoints(checkpoint_dir: Path) -> List[Path]:
    """Find all .pt checkpoint files in a directory."""
    checkpoints = list(checkpoint_dir.glob('*.pt'))
    # Filter out SAE checkpoints (we only want model checkpoints)
    checkpoints = [c for c in checkpoints if not c.stem.startswith('sae')]
    return sorted(checkpoints)


def compare_perplexity(
    checkpoints: List[str],
    baseline: Optional[str] = None,
    eval_samples: int = 1000,
    batch_size: int = 4,
    max_length: int = 1024,
    device: str = 'cuda',
    output_path: Optional[str] = None,
) -> Dict:
    """
    Compare perplexity across multiple model checkpoints.
    
    Args:
        checkpoints: List of checkpoint paths or HuggingFace model names
        baseline: Optional baseline model for comparison (e.g., pretrained model)
        eval_samples: Number of evaluation samples
        batch_size: Batch size for evaluation
        max_length: Max sequence length
        device: Device to use
        output_path: Path to save results JSON
        
    Returns:
        Dictionary with comparison results
    """
    logger.info("=" * 70)
    logger.info("PERPLEXITY COMPARISON")
    logger.info("=" * 70)
    
    # Load tokenizer (use baseline or first checkpoint to determine)
    tokenizer_source = baseline if baseline else checkpoints[0]
    if Path(tokenizer_source).exists():
        # Load from checkpoint config
        ckpt = torch.load(tokenizer_source, map_location='cpu')
        tokenizer_source = ckpt.get('config', {}).get('model_name', 'EleutherAI/pythia-70m')
    
    logger.info(f"Loading tokenizer from: {tokenizer_source}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create evaluation dataloader (same for all models)
    logger.info(f"Loading evaluation data ({eval_samples} samples)...")
    eval_loader = create_causal_lm_dataloader(
        tokenizer=tokenizer,
        dataset_name='wikitext',
        split='test',
        batch_size=batch_size,
        max_length=max_length,
        max_samples=eval_samples,
    )
    
    results = {
        'eval_samples': eval_samples,
        'models': {},
    }
    
    # Evaluate baseline first if provided
    baseline_ppl = None
    if baseline:
        logger.info(f"\nEvaluating baseline: {baseline}")
        model, name, config = load_model(baseline, device)
        
        with torch.no_grad():
            eval_results = evaluate_perplexity(model, eval_loader, device, show_progress=True)
        
        baseline_ppl = eval_results['perplexity']
        results['baseline'] = {
            'name': name,
            'path': baseline,
            'perplexity': baseline_ppl,
        }
        logger.info(f"  Perplexity: {baseline_ppl:.2f}")
        
        # Free memory
        del model
        torch.cuda.empty_cache()
    
    # Evaluate each checkpoint
    for checkpoint_path in checkpoints:
        logger.info(f"\nEvaluating: {checkpoint_path}")
        
        try:
            model, name, config = load_model(checkpoint_path, device)
            
            with torch.no_grad():
                eval_results = evaluate_perplexity(model, eval_loader, device, show_progress=True)
            
            ppl = eval_results['perplexity']
            
            model_result = {
                'name': name,
                'path': str(checkpoint_path),
                'config': config,
                'perplexity': ppl,
            }
            
            # Compute change from baseline
            if baseline_ppl is not None:
                model_result['perplexity_change'] = ppl - baseline_ppl
                model_result['perplexity_change_pct'] = (ppl - baseline_ppl) / baseline_ppl * 100
            
            results['models'][name] = model_result
            
            logger.info(f"  Perplexity: {ppl:.2f}")
            if baseline_ppl:
                logger.info(f"  Change from baseline: {ppl - baseline_ppl:+.2f} ({(ppl - baseline_ppl) / baseline_ppl * 100:+.1f}%)")
            
            # Free memory
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"  Failed to evaluate: {e}")
            results['models'][str(checkpoint_path)] = {'error': str(e)}
    
    # Print summary table
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    table_data = []
    
    if baseline_ppl:
        table_data.append([
            results['baseline']['name'] + " (baseline)",
            f"{baseline_ppl:.2f}",
            "-",
            "-",
        ])
    
    for name, data in results['models'].items():
        if 'error' in data:
            table_data.append([name, "ERROR", "-", "-"])
        else:
            ppl = data['perplexity']
            if 'perplexity_change' in data:
                change = f"{data['perplexity_change']:+.2f}"
                change_pct = f"{data['perplexity_change_pct']:+.1f}%"
            else:
                change = "-"
                change_pct = "-"
            table_data.append([name, f"{ppl:.2f}", change, change_pct])
    
    # Sort by perplexity
    table_data_sorted = sorted(table_data, key=lambda x: float(x[1]) if x[1] != "ERROR" else float('inf'))
    
    headers = ["Model", "Perplexity", "Δ PPL", "Δ %"]
    print("\n" + tabulate(table_data_sorted, headers=headers, tablefmt="simple"))
    
    # Find best model
    valid_models = {k: v for k, v in results['models'].items() if 'perplexity' in v}
    if valid_models:
        best_name = min(valid_models, key=lambda k: valid_models[k]['perplexity'])
        best_ppl = valid_models[best_name]['perplexity']
        logger.info(f"\n✓ Best model: {best_name} (PPL: {best_ppl:.2f})")
        results['best_model'] = best_name
        results['best_perplexity'] = best_ppl
    
    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare perplexity across model checkpoints"
    )
    
    parser.add_argument(
        '--checkpoints', 
        nargs='+', 
        help='Paths to model checkpoints'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        help='Directory containing checkpoints (alternative to --checkpoints)'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        default='EleutherAI/pythia-70m',
        help='Baseline model for comparison (HuggingFace name or checkpoint path)'
    )
    parser.add_argument(
        '--eval_samples',
        type=int,
        default=1000,
        help='Number of evaluation samples'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/perplexity_comparison.json',
        help='Output path for results JSON'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use'
    )
    
    args = parser.parse_args()
    
    # Gather checkpoints
    checkpoints = []
    
    if args.checkpoints:
        checkpoints.extend(args.checkpoints)
    
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        if checkpoint_dir.exists():
            found = find_checkpoints(checkpoint_dir)
            checkpoints.extend([str(c) for c in found])
            logger.info(f"Found {len(found)} checkpoints in {checkpoint_dir}")
    
    if not checkpoints:
        parser.error("Must provide --checkpoints or --checkpoint_dir")
    
    logger.info(f"Comparing {len(checkpoints)} checkpoints")
    
    compare_perplexity(
        checkpoints=checkpoints,
        baseline=args.baseline,
        eval_samples=args.eval_samples,
        batch_size=args.batch_size,
        device=args.device,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()