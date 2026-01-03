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
# Comprehensive SAE Sweep (with smart k filtering)
# =============================================================================

TRAIN_K_VALUES=(64 128 256 384 512 768 1024)
ALL_SHARP_K=(32 64 128 192 256 384 512 768 1024)

# Create results summary file
SUMMARY_FILE="$WORKDIR/sweep_summary.csv"
echo "train_k,sharp_k,perplexity,degradation_pct" > "$SUMMARY_FILE"

for train_k in "${TRAIN_K_VALUES[@]}"; do
    echo ""
    echo "========================================"
    echo "Training SAE with k=$train_k"
    echo "========================================"
    
    SAE_CHECKPOINT="$WORKDIR/checkpoints/sae_k${train_k}.pt"
    
    uv run python scripts/train_sae_pretrained.py \
        --model EleutherAI/pythia-70m \
        --layer 3 \
        --k "$train_k" \
        --epochs 30 \
        --samples 100000 \
        --output "$SAE_CHECKPOINT"
    
    if [ ! -f "$SAE_CHECKPOINT" ]; then
        echo "ERROR: SAE checkpoint not created for k=$train_k"
        continue
    fi
    
    # Build list of valid sharp_k values (<= train_k)
    VALID_SHARP_K=""
    for sharp_k in "${ALL_SHARP_K[@]}"; do
        if [ "$sharp_k" -le "$train_k" ]; then
            if [ -z "$VALID_SHARP_K" ]; then
                VALID_SHARP_K="$sharp_k"
            else
                VALID_SHARP_K="$VALID_SHARP_K,$sharp_k"
            fi
        fi
    done
    
    echo ""
    echo "Testing interventions for k=$train_k SAE..."
    echo "Valid sharp_k values: $VALID_SHARP_K"
    
    uv run python scripts/intervention_baseline.py \
        --model EleutherAI/pythia-70m \
        --layer 3 \
        --k_values "$VALID_SHARP_K" \
        --sae_checkpoint "$SAE_CHECKPOINT" \
        --sae_k "$train_k" \
        --max_samples 500 \
        --output "$WORKDIR/results_k${train_k}.json"
    
    echo "Completed k=$train_k"
done

# =============================================================================
# Combine results into summary
# =============================================================================
echo ""
echo "========================================"
echo "Generating summary..."
echo "========================================"

uv run python -c "
import json
import glob

results = []
for f in sorted(glob.glob('$WORKDIR/results_k*.json')):
    with open(f) as fp:
        data = json.load(fp)
    train_k = data.get('sae_k_train', f.split('_k')[-1].replace('.json',''))
    baseline = data['baseline_perplexity']
    
    # Full reconstruction
    full_ppl = data['full_recon_perplexity']
    full_deg = (full_ppl - baseline) / baseline * 100
    results.append((train_k, train_k, full_ppl, full_deg))
    
    # Each sharpening value
    for k_str, info in data.get('k_sharp_results', {}).items():
        results.append((train_k, k_str, info['perplexity'], info['relative_degradation_pct']))

# Print table
print()
print('=' * 70)
print('FULL RESULTS GRID')
print('=' * 70)
print(f'{\"Train k\":<10} {\"Sharp k\":<10} {\"PPL\":<12} {\"Degradation\":<12}')
print('-' * 50)

for train_k, sharp_k, ppl, deg in sorted(results, key=lambda x: (int(x[0]), int(x[1]))):
    print(f'{train_k:<10} {sharp_k:<10} {ppl:<12.2f} {deg:<12.1f}%')

# Save CSV
with open('$SUMMARY_FILE', 'a') as f:
    for train_k, sharp_k, ppl, deg in results:
        f.write(f'{train_k},{sharp_k},{ppl:.2f},{deg:.1f}\n')

print()
print(f'Summary saved to: $SUMMARY_FILE')
"

# =============================================================================
# Copy results back
# =============================================================================
echo ""
echo "========================================"
echo "COMPLETE"
echo "========================================"

cp "$WORKDIR"/results_k*.json "$SLURM_SUBMIT_DIR/" 2>/dev/null || true
cp "$SUMMARY_FILE" "$SLURM_SUBMIT_DIR/" 2>/dev/null || true
cp "$WORKDIR/checkpoints/"*.pt "$SLURM_SUBMIT_DIR/checkpoints/" 2>/dev/null || true

echo "Results copied to: $SLURM_SUBMIT_DIR"
echo "Summary CSV: $SLURM_SUBMIT_DIR/sweep_summary.csv"