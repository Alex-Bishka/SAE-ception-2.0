#!/usr/bin/env python3
"""
Quick analysis of L1 penalty test results.

Usage: python analyze_l1_test.py <multirun_dir>
Example: python analyze_l1_test.py multirun/2025-11-14/14-30-00
"""

import sys
import torch
from pathlib import Path
from omegaconf import OmegaConf


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_l1_test.py <multirun_dir>")
        sys.exit(1)
    
    base = Path(sys.argv[1])
    
    if not base.exists():
        print(f"Error: Directory not found: {base}")
        sys.exit(1)
    
    print("=" * 80)
    print("L1 PENALTY TEST RESULTS")
    print("=" * 80)
    
    results = []
    
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        
        # Load config
        config_path = run_dir / '.hydra/config.yaml'
        if not config_path.exists():
            continue
        config = OmegaConf.load(config_path)
        
        # Load results
        results_path = run_dir / 'checkpoints/results_cycle_0.pt'
        if not results_path.exists():
            print(f"⚠ Run {run_dir.name}: No results file found")
            continue
        
        r = torch.load(results_path, map_location='cpu')
        
        l1 = config.sae.l1_penalty
        dead = r['sae_dead_features_pct']
        l0 = r['sparsity_l0_mean']
        u = r['sae_features_u']
        entropy = r['sae_feature_usage_entropy']
        ari = r['clustering_ari']
        
        results.append({
            'run': run_dir.name,
            'l1': l1,
            'dead': dead,
            'l0': l0,
            'u': u,
            'entropy': entropy,
            'ari': ari,
        })
        
        # Status emoji
        if dead < 30:
            status = "✅ GOOD"
        elif dead < 50:
            status = "⚠️  OK"
        else:
            status = "❌ BAD"
        
        print(f"\nRun {run_dir.name}: l1_penalty = {l1}")
        print(f"  Dead features: {dead:>6.1f}%  {status}")
        print(f"  L0 sparsity:   {l0:>6.1f}")
        print(f"  U (interp):    {u:>6.4f}")
        print(f"  Entropy:       {entropy:>6.2f} nats")
        print(f"  ARI:           {ari:>6.4f}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if not results:
        print("No results found!")
        return
    
    # Find best by dead %
    best = min(results, key=lambda x: x['dead'])
    print(f"\n✨ Best by dead %: l1={best['l1']}")
    print(f"   Dead: {best['dead']:.1f}%, L0: {best['l0']:.0f}, U: {best['u']:.4f}")
    
    # Check if any are good
    good_runs = [r for r in results if r['dead'] < 30]
    if good_runs:
        print(f"\n🎉 {len(good_runs)} run(s) achieved <30% dead features!")
        print("   Ready to add SAE-ception cycles.")
    else:
        min_dead = min(r['dead'] for r in results)
        print(f"\n⚠️  All runs have >30% dead features (best: {min_dead:.1f}%)")
        print("   Recommendations:")
        print("   1. Try even lower l1: 5e-5, 1e-4")
        print("   2. Increase epochs: 10-15")
        print("   3. Check initialization/learning rate")
        print("   4. Consider pre-training SAE without sharpening first")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()