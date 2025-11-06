# SAE-ception Workflow

## The Complete Loop

```
┌─────────────────────────────────────────────────────────────┐
│                        CYCLE 0                               │
└─────────────────────────────────────────────────────────────┘

[Pre-trained Model]
        │
        ▼
┌──────────────────┐
│ train_baseline.py│  ← Fine-tune on dataset
└────────┬─────────┘
         │ Saves: model_cycle_0.pt
         ▼
┌──────────────────┐
│   Freeze Model   │
└────────┬─────────┘
         │ Extract activations
         ▼
┌──────────────────┐
│  train_sae.py    │  ← Train SAE on activations
└────────┬─────────┘
         │ Saves: sae_cycle_0.pt
         ▼
┌──────────────────┐
│   Evaluation     │  ← Compute all metrics
└────────┬─────────┘
         │ Saves: results_cycle_0.pt
         ▼
┌──────────────────┐
│ Feature          │
│ Sharpening       │  ← Generate sharpened targets
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│train_auxiliary.py│  ← Retrain with aux loss
└────────┬─────────┘
         │ Saves: model_cycle_1.pt
         │
         │
┌────────┴─────────────────────────────────────────────────────┐
│                        CYCLE 1                               │
└──────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│   Freeze Model   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  train_sae.py    │  ← Train SAE on new activations
└────────┬─────────┘
         │ Saves: sae_cycle_1.pt
         ▼
┌──────────────────┐
│   Evaluation     │
└────────┬─────────┘
         │ Saves: results_cycle_1.pt
         ▼
┌──────────────────┐
│train_auxiliary.py│  ← Create cycle 2 model
└────────┬─────────┘
         │ Saves: model_cycle_2.pt
         │
         ▼
       [Repeat...]
```

## Script Responsibilities

### Individual Scripts (Manual Control)

```
train_baseline.py    →  Creates: model_cycle_0.pt
                        Status: ✅ Tested
                        
train_sae.py         →  Creates: sae_cycle_X.pt
                        Status: ✅ Tested
                        
train_auxiliary.py   →  Creates: model_cycle_X+1.pt
                        Status: 🔨 Needs testing
```

### Automated Pipeline

```
train_cycle.py       →  Runs entire loop automatically
                        Creates: All checkpoints + results
                        Status: 🔨 Needs testing
```

## File Flow

```
Training Run
     │
     ├─→ outputs/YYYY-MM-DD/HH-MM-SS/
     │        │
     │        ├─→ checkpoints/
     │        │     ├─→ model_cycle_0_best.pt    ✅ Have
     │        │     ├─→ sae_cycle_0_best.pt      ✅ Have
     │        │     ├─→ model_cycle_1_best.pt    🔨 Next
     │        │     ├─→ sae_cycle_1_best.pt      🔨 After
     │        │     └─→ results_cycle_*.pt       🔨 Auto-generated
     │        │
     │        └─→ train_cycle.log                ✅ Working
     │
     └─→ .hydra/config.yaml                      ✅ Working
```

## Quick Command Reference

### Test Each Component
```bash
# 1. Baseline (TESTED ✅)
python scripts/train_baseline.py model=gpt2_small dataset=sst2

# 2. SAE (TESTED ✅)
python scripts/train_sae.py cycle.current=0

# 3. Auxiliary (TEST NEXT 🔨)
python scripts/train_auxiliary.py cycle.current=0

# 4. Full Pipeline (TEST AFTER 🔨)
python scripts/train_cycle.py cycle.max_cycles=3
```

### What Each Script Needs

```
train_baseline.py:   Nothing (starts fresh)
                     
train_sae.py:        Needs model_cycle_X.pt from baseline
                     
train_auxiliary.py:  Needs model_cycle_X.pt + sae_cycle_X.pt
                     
train_cycle.py:      Nothing (orchestrates everything)
```

## Your Progress

```
✅ [████████████████████░░░] 90%

Completed:
  ✅ Project structure
  ✅ All configurations
  ✅ SAE architecture
  ✅ Feature sharpening (3 strategies)
  ✅ All evaluation metrics
  ✅ Baseline training (tested)
  ✅ SAE training (tested)
  ✅ Logging system

Remaining:
  🔨 Test auxiliary training
  🔨 Test full cycle
  📊 Optional: Visualizations
```

## Next Action

**Run this command to test auxiliary training:**
```bash
python scripts/train_auxiliary.py \
    model=gpt2_small \
    dataset=sst2 \
    cycle.current=0 \
    model.epochs_per_cycle=1 \
    cycle.aux_loss_weight=0.01 \
    wandb.mode=disabled
```

**Expected time**: ~30 minutes

**What success looks like**:
- ✅ Script completes without errors
- ✅ Creates `model_cycle_1_best.pt`
- ✅ Task accuracy stays within ±2% of baseline
- ✅ Aux loss decreases during training

Then you're ready for the full pipeline! 🚀
