# SAE-ception

Iterative training method that uses Sparse Autoencoders (SAEs) to improve model interpretability. Follow up to the original [SAE-ception](https://github.com/Alex-Bishka/SAE-ception).

## Setup
```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

## Quick Start

### Train Baseline Model (Cycle 0)
```bash
python scripts/train_cycle.py
```

### Run First SAE-ception Cycle
```bash
python scripts/train_cycle.py cycle.current=1
```

### Try Different Sharpening Strategies
```bash
# Per-class (default)
python scripts/train_cycle.py sharpening=per_class

# Per-example
python scripts/train_cycle.py sharpening=per_example

# Random baseline
python scripts/train_cycle.py sharpening=random
```

### Use Different Datasets
```bash
python scripts/train_cycle.py dataset=ag_news
```

## Project Structure
```
sae-ception/
├── configs/          # Hydra configuration files
├── src/              # Source code
│   └── sae_ception/
│       ├── models/       # SAE architecture
│       ├── training/     # Training loops
│       ├── sharpening/   # Feature sharpening strategies
│       ├── evaluation/   # Metrics and evaluation
│       └── utils/        # Utilities
├── scripts/          # Executable scripts
└── outputs/          # Experiment outputs (created by Hydra)
```

## Configuration

All experiments are configured via Hydra YAML files in `configs/`. Key configuration groups:

- `dataset/`: Dataset configuration (sst2, ag_news)
- `model/`: Model configuration (gpt2_small, gpt_neox_20b)
- `sae/`: SAE hyperparameters
- `sharpening/`: Feature sharpening strategy
- `experiment/`: Pre-configured experiment setups

## Development
```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Format code
black src/ scripts/

# Lint
ruff check src/ scripts/
```

## Creating a Job

To make a lean run dir on the cluster:
```
mkdir -p /home1/$USER/jobs/saeception-run

rsync -av \
  --exclude '.venv' --exclude '.git' --exclude '.cache' \
  --exclude 'data' --exclude 'outputs' --exclude 'wandb' \
  /home1/$USER/SAE-ception/ /home1/$USER/jobs/saeception-run/
```


Submit job from login node:
```
cd /home1/$USER/jobs/saeception-run/
sbatch run.sbatch
squeue -u $USER
tail -f slurm-<jobid>.out
```

## Citation

Based on "SAE-ception: Iteratively Using Sparse Autoencoders as a Training Signal"
