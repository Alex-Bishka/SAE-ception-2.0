#!/bin/bash
#SBATCH --job-name=saeception
#SBATCH --partition=gpu
#SBATCH --gpus=1                 # if your cluster uses gres: use --gres=gpu:1 instead
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out

set -euo pipefail

module purge

# If uv isn't available as a module, ensure it's on PATH (one-time installer okay on login node).
if ! command -v uv >/dev/null 2>&1; then
  export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi

# Guard against leaked envs from login node
if [ -n "${VIRTUAL_ENV:-}" ]; then deactivate 2>/dev/null || true; unset VIRTUAL_ENV; fi
if command -v conda >/dev/null 2>&1; then conda deactivate 2>/dev/null || true; fi

# Scratch + caches
export SCRATCH_DIR="${SCRATCH:-$HOME/scratch}/saeception"
mkdir -p "$SCRATCH_DIR" "$SCRATCH_DIR/caches"
export UV_CACHE_DIR="$SCRATCH_DIR/caches/uv"
export PIP_CACHE_DIR="$SCRATCH_DIR/caches/pip"
export HF_HOME="$SCRATCH_DIR/caches/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"

# Per-job workdir on fast storage
WORKDIR="$SCRATCH_DIR/run_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"
cp -r "$SLURM_SUBMIT_DIR"/* "$WORKDIR"/
cd "$WORKDIR"

# Install/resolve env into .venv here (respects uv.lock / pyproject)
uv sync --frozen

# Quick GPU sanity (won't fail job if absent)
uv run nvidia-smi || true
uv run python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

# =============================================================================
# TEST 0: Verify Pythia loads and runs
# =============================================================================
echo "=========================================="
echo "TEST 0: Verify Pythia model loads"
echo "=========================================="

uv run python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading Pythia-70M...")
model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-70m')
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f"Model loaded on {device}")
print(f"Hidden size: {model.config.hidden_size}")
print(f"Num layers: {model.config.num_hidden_layers}")

# Quick generation test
input_ids = tokenizer("The capital of France is", return_tensors='pt')['input_ids'].to(device)
with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=10, do_sample=False)
print(f"Generation test: {tokenizer.decode(output[0])}")

print("✓ TEST 0 PASSED: Pythia loads and generates")
PY

# =============================================================================
# TEST 1: Verify intervention hooks work
# =============================================================================
echo ""
echo "=========================================="
echo "TEST 1: Verify intervention hooks"
echo "=========================================="

uv run python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_ception.utils.hooks import ActivationIntervention

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-70m')
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()

# Test 1a: Identity intervention (should not change output)
print("\nTest 1a: Identity intervention...")
hook_called = [False]
captured_shape = [None]

def identity_fn(x):
    hook_called[0] = True
    captured_shape[0] = x.shape
    return x

input_ids = tokenizer("Hello world", return_tensors='pt')['input_ids'].to(device)

# Get baseline output
with torch.no_grad():
    baseline_logits = model(input_ids).logits

# Get output with identity intervention
with ActivationIntervention(model, '3', identity_fn):
    with torch.no_grad():
        intervened_logits = model(input_ids).logits

assert hook_called[0], "Hook was not called!"
print(f"  Hook called: ✓")
print(f"  Activation shape: {captured_shape[0]}")
assert captured_shape[0][0] == 1, "Batch size should be 1"
assert captured_shape[0][2] == 512, "Hidden dim should be 512 for Pythia-70M"
print(f"  Shape correct: ✓")

# Check outputs are identical (identity should not change anything)
diff = (baseline_logits - intervened_logits).abs().max().item()
print(f"  Max logit difference: {diff:.6f}")
assert diff < 1e-5, f"Identity intervention changed output! Max diff: {diff}"
print(f"  Output unchanged: ✓")

# Test 1b: Zeroing intervention (should change output)
print("\nTest 1b: Zeroing intervention...")

def zero_fn(x):
    return torch.zeros_like(x)

with ActivationIntervention(model, '3', zero_fn):
    with torch.no_grad():
        zeroed_logits = model(input_ids).logits

diff = (baseline_logits - zeroed_logits).abs().max().item()
print(f"  Max logit difference: {diff:.2f}")
assert diff > 1.0, "Zeroing should significantly change output!"
print(f"  Output changed: ✓")

print("\n✓ TEST 1 PASSED: Intervention hooks work correctly")
PY

# =============================================================================
# TEST 2: Verify perplexity evaluation works
# =============================================================================
echo ""
echo "=========================================="
echo "TEST 2: Verify perplexity evaluation"
echo "=========================================="

uv run python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_ception.utils.data import create_causal_lm_dataloader
from sae_ception.evaluation.lm_metrics import evaluate_perplexity

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-70m')
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()

print("\nLoading WikiText test data (50 samples)...")
loader = create_causal_lm_dataloader(
    tokenizer=tokenizer,
    dataset_name='wikitext',
    split='test',
    batch_size=4,
    max_length=512,
    max_samples=50,
)
print(f"  Loaded {len(loader.dataset)} samples")

print("\nEvaluating perplexity...")
results = evaluate_perplexity(model, loader, device=device, max_batches=10)

print(f"\nResults:")
print(f"  Perplexity: {results['perplexity']:.2f}")
print(f"  Loss: {results['loss']:.4f}")
print(f"  Tokens evaluated: {results['total_tokens']}")

# Sanity check: Pythia-70M should have reasonable perplexity
assert 10 < results['perplexity'] < 100, f"Perplexity {results['perplexity']} seems wrong for Pythia-70M"
print(f"\n✓ TEST 2 PASSED: Perplexity evaluation works (PPL={results['perplexity']:.2f})")
PY

# =============================================================================
# TEST 3: Verify SAE + intervention integration
# =============================================================================
echo ""
echo "=========================================="
echo "TEST 3: SAE + intervention integration"
echo "=========================================="

uv run python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_ception.models.sae import create_sae
from sae_ception.utils.hooks import create_sae_intervention, ActivationIntervention
from sae_ception.utils.data import create_causal_lm_dataloader
from sae_ception.evaluation.lm_metrics import evaluate_perplexity, evaluate_perplexity_with_intervention

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-70m')
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()

hidden_size = model.config.hidden_size
print(f"Hidden size: {hidden_size}")

# Create a randomly initialized SAE (not trained, just for testing integration)
print("\nCreating SAE...")
sae = create_sae(
    input_dim=hidden_size,
    hidden_dim=hidden_size * 8,
    sae_type='topk',
    k=50,
)
sae.to(device)
sae.eval()
print(f"  SAE: {hidden_size} -> {hidden_size * 8} -> {hidden_size}")

# Test create_sae_intervention
print("\nTesting create_sae_intervention...")
intervention_fn = create_sae_intervention(sae, k_sharp=10, device=device)

# Test on a single batch
input_ids = tokenizer("The quick brown fox", return_tensors='pt')['input_ids'].to(device)

with ActivationIntervention(model, '3', intervention_fn) as intervener:
    with torch.no_grad():
        output = model(input_ids)
    
    # Check we captured the original
    original = intervener.get_original_output()
    print(f"  Original activation shape: {original.shape}")
    print(f"  Original activation mean: {original.mean():.4f}")

print("  Intervention ran without error: ✓")

# Test evaluate_perplexity_with_intervention
print("\nTesting perplexity with intervention...")
loader = create_causal_lm_dataloader(
    tokenizer=tokenizer,
    dataset_name='wikitext',
    split='test',
    batch_size=2,
    max_length=256,
    max_samples=20,
)

# Baseline
baseline = evaluate_perplexity(model, loader, device=device, show_progress=False)
print(f"  Baseline perplexity: {baseline['perplexity']:.2f}")

# With intervention (using untrained SAE, expect high degradation)
with_intervention = evaluate_perplexity_with_intervention(
    model, loader, intervention_fn, '3', device=device, show_progress=False
)
print(f"  Intervened perplexity: {with_intervention['perplexity']:.2f}")
print(f"  Degradation: +{with_intervention['perplexity'] - baseline['perplexity']:.2f}")

# Untrained SAE should cause significant degradation
assert with_intervention['perplexity'] > baseline['perplexity'], \
    "Untrained SAE intervention should increase perplexity"

print("\n✓ TEST 3 PASSED: SAE + intervention integration works")
PY

# =============================================================================
# TEST 4: Full intervention baseline script (quick mode)
# =============================================================================
echo ""
echo "=========================================="
echo "TEST 4: Full intervention baseline (quick)"
echo "=========================================="

uv run python scripts/intervention_baseline.py \
    --model EleutherAI/pythia-70m \
    --layer 3 \
    --k_values 10,25,50 \
    --quick \
    --output "$WORKDIR/intervention_baseline_results.json"

echo ""
echo "=========================================="
echo "ALL TESTS PASSED"
echo "=========================================="
echo "Results saved to: $WORKDIR/intervention_baseline_results.json"