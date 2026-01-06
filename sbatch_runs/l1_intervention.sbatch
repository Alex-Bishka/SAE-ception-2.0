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

# ============================================================
# L1 Comparison
# ============================================================
echo "========================================"
echo "L1 PENALTY BASELINE (Run B)"
echo "========================================"

# Parameters
MODEL="EleutherAI/pythia-70m"
LAYER=3
TRAIN_SAMPLES=10000
EVAL_SAMPLES=500
EPOCHS=1

# L1 weights to sweep
L1_WEIGHTS="0.0001 0.0005 0.001 0.005 0.01"

mkdir -p checkpoints results

for L1_WEIGHT in $L1_WEIGHTS; do
    echo ""
    echo "----------------------------------------"
    echo "Training with L1 weight = $L1_WEIGHT"
    echo "----------------------------------------"
    
    uv run python scripts/train_cpt_l1.py \
        --model "$MODEL" \
        --layer $LAYER \
        --l1_weight $L1_WEIGHT \
        --train_samples $TRAIN_SAMPLES \
        --eval_samples $EVAL_SAMPLES \
        --epochs $EPOCHS \
        --output "checkpoints/model_l1_${L1_WEIGHT}.pt"
done

# Copy results back
mkdir -p "$SLURM_SUBMIT_DIR/results" "$SLURM_SUBMIT_DIR/checkpoints"
cp "$WORKDIR"/checkpoints/*.pt "$SLURM_SUBMIT_DIR/checkpoints/" 2>/dev/null || true
cp "$WORKDIR"/checkpoints/*.json "$SLURM_SUBMIT_DIR/checkpoints/" 2>/dev/null || true

echo ""
echo "L1 baseline complete. Results in $SLURM_SUBMIT_DIR/checkpoints/"