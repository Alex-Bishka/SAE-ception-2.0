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

# ---- gp2_small real run ----
# uv run python scripts/train_cycle.py -m \
#     model=gpt2_small \
#     dataset=sst2 \
#     cycle.current=0 \
#     cycle.max_cycles=3 \
#     model.epochs_per_cycle=2 \
#     sharpening.top_k_pct=0.50,1.00,1.50,2.00 \
#     sae=topk \
#     sae.epochs=3 \
#     sae.k=100,150,200,250 \
#     sae.token_level=true \
#     clear_cache=true \
    # cycle.aux_loss_weight=0.01,0.05,0.10,0.15,0.20,0.25,0.30 \
#     start_from=finetuned \
#     wandb.mode=enabled
# ----------------------------

# find best k for gpt2 small
# uv run python scripts/train_cycle.py -m \
#     model=gpt2_small \
#     dataset=sst2 \
#     cycle.current=0 \
#     cycle.max_cycles=1 \
#     model.epochs_per_cycle=2 \
#     sharpening.top_k_pct=1.00 \
#     sae=topk \
#     sae.expansion_factor=8 \
#     sae.epochs=5 \
#     sae.k=256,384,512,768,1000 \
#     sae.resample_interval=100 \
#     sae.token_level=true \
#     start_from=finetuned \
#     wandb.mode=online

# smoke test on gpt2_small
uv run python scripts/train_cycle.py -m \
    model=gpt2_small \
    dataset=sst2 \
    cycle.current=0 \
    cycle.max_cycles=1 \
    model.epochs_per_cycle=2 \
    sharpening.top_k_pct=1.00 \
    sae=topk \
    sae.epochs=3 \
    sae.k=400 \
    clear_cache=true \
    sae.token_level=true \
    start_from=baseline \
    wandb.mode=disabled

# pythyia 70m dims: TopK SAE: 512 -> 4096 -> 512
# 0.5% of 4096 is about 20 active features
# 200 is about 4.88% of sparse codes
# ---- SMOKE TEST SCRIPT ----
# uv run python scripts/train_cycle.py -m \
#     model=gpt2_small \
#     model.hf_path=EleutherAI/pythia-70m \
#     model.hidden_size=512 \
#     dataset=sst2 \
#     cycle.current=0 \
#     cycle.max_cycles=1 \
#     model.epochs_per_cycle=1 \
#     sharpening.top_k_pct=0.50 \
#     sae=topk \
#     sae.epochs=2 \
#     sae.k=100 \
#     sae.resample_interval=100 \
#     sae.token_level=true \
#     clear_cache=true \
#     start_from=baseline \
#     wandb.mode=disabled
# ----------------------------

# SAE bench testing:
# uv run python scripts/run_saebench.py \
#     model=gpt2_small \
#     model.hf_path=EleutherAI/pythia-70m \
#     model.hidden_size=512 \
#     dataset=sst2 \
#     sae=topk \
#     checkpoint_dir=/home1/abishka@hmc.edu/scratch/saeception/run_20033/multirun/2025-12-01/18-35-39/0/checkpoints \
#     cycle_to_eval=0 \
#     n_candidates=100