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
# SAE-ception e2e smoke test
# ============================================================
echo "========================================"
echo "SAE-CEPTION END-TO-END SMOKE TEST"
echo "========================================"

# Parameters for smoke test
SAE_SAMPLES=5000
SAE_EPOCHS=15
CPT_SAMPLES=1000
CPT_EPOCHS=1
SAE_K=768
SHARP_K=512

mkdir -p checkpoints

# ============================================================
# Step 1: Train SAE on pretrained model
# ============================================================
echo ""
echo "Step 1: Training SAE on pretrained Pythia-70M..."
uv run python scripts/train_sae_pretrained.py \
    --model EleutherAI/pythia-70m \
    --layer 3 \
    --k $SAE_K \
    --epochs $SAE_EPOCHS \
    --samples $SAE_SAMPLES \
    --output checkpoints/sae_k${SAE_K}.pt

# ============================================================
# Step 2: CPT WITHOUT aux loss (control)
# ============================================================
echo ""
echo "Step 2: CPT without aux loss (control)..."
uv run python scripts/train_cpt.py \
    --model EleutherAI/pythia-70m \
    --layer 3 \
    --sae_checkpoint checkpoints/sae_k${SAE_K}.pt \
    --sae_k $SAE_K \
    --k_sharp $SHARP_K \
    --aux_weight 0.0 \
    --train_samples $CPT_SAMPLES \
    --epochs $CPT_EPOCHS \
    --output checkpoints/model_cpt_no_aux.pt

# ============================================================
# Step 3: CPT WITH aux loss (SAE-ception)
# ============================================================
echo ""
echo "Step 3: CPT with aux loss (SAE-ception)..."
uv run python scripts/train_cpt.py \
    --model EleutherAI/pythia-70m \
    --layer 3 \
    --sae_checkpoint checkpoints/sae_k${SAE_K}.pt \
    --sae_k $SAE_K \
    --k_sharp $SHARP_K \
    --aux_weight 0.01 \
    --train_samples $CPT_SAMPLES \
    --epochs $CPT_EPOCHS \
    --output checkpoints/model_cpt_with_aux.pt

# ============================================================
# Step 4: Structure Check - Train SAE on control model
# ============================================================
echo ""
echo "Step 4: Training SAE on control model..."
uv run python scripts/train_sae_pretrained.py \
    --model checkpoints/model_cpt_no_aux.pt \
    --layer 3 \
    --k $SAE_K \
    --epochs $SAE_EPOCHS \
    --samples $SAE_SAMPLES \
    --output checkpoints/sae_on_control.pt

# ============================================================
# Step 5: Structure Check - Train SAE on SAE-ception model
# ============================================================
echo ""
echo "Step 5: Training SAE on SAE-ception model..."
uv run python scripts/train_sae_pretrained.py \
    --model checkpoints/model_cpt_with_aux.pt \
    --layer 3 \
    --k $SAE_K \
    --epochs $SAE_EPOCHS \
    --samples $SAE_SAMPLES \
    --output checkpoints/sae_on_saeception.pt

# ============================================================
# Summary
# ============================================================
echo ""
echo "========================================"
echo "SMOKE TEST COMPLETE"
echo "========================================"
echo ""
echo "Checkpoints created:"
echo "  - checkpoints/sae_k${SAE_K}.pt (SAE on pretrained)"
echo "  - checkpoints/model_cpt_no_aux.pt (CPT control)"
echo "  - checkpoints/model_cpt_with_aux.pt (CPT + SAE-ception)"
echo "  - checkpoints/sae_on_control.pt (SAE on control)"
echo "  - checkpoints/sae_on_saeception.pt (SAE on SAE-ception)"
echo ""
echo "Check the .diagnostics.json files to compare:"
echo "  - L0 sparsity"
echo "  - Dead features %"
echo "  - Reconstruction error"