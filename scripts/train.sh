#!/bin/bash
# scripts/train.sh — SLURM array job to train all 5 folds of one LogFiT
# dataset configuration.
#
# Usage:
#   sbatch scripts/train.sh <config.yaml> <paragraphs.pkl> <splits.json> [backbone_decision.json]
#
# Examples:
#   # HDFS (RoBERTa-base)
#   sbatch scripts/train.sh \
#       configs/hdfs.yaml \
#       data/hdfs_subset/paragraphs.pkl \
#       data/hdfs_subset/splits.json \
#       data/hdfs_subset/backbone_decision.json
#
#   # BGL 30s (Longformer)
#   sbatch scripts/train.sh \
#       configs/bgl_30s.yaml \
#       data/bgl_subset/paragraphs.pkl \
#       data/bgl_subset/splits.json \
#       data/bgl_subset/backbone_decision.json
#
# Submits 5 parallel array tasks, one per fold. $SLURM_ARRAY_TASK_ID becomes
# the --fold-idx passed to `python -m src.train`.
#
# Account (repo is public — account is NOT hard-coded here):
#   Set once in ~/.bashrc on Narval:
#       export SBATCH_ACCOUNT=<your-rrg-or-def-account>
#   sbatch reads SBATCH_ACCOUNT automatically. One-off override:
#       sbatch --account=<your-account> scripts/train.sh ...
#
# Prerequisites:
#   - ~/sdd_env activated venv with pyproject.toml deps installed
#   - data/ symlinked to $SCRATCH/log-fit/data (paragraphs.pkl + splits.json present)
#   - results/ symlinked to $SCRATCH/log-fit/results (will hold logs + models)
#   - submitted from the project root (./ contains pyproject.toml and src/)

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --job-name=logfit-train
#SBATCH --output=results/logs/train-%A_%a.out
#SBATCH --error=results/logs/train-%A_%a.err

set -euo pipefail

# ----- positional args -----
CONFIG_PATH="${1:?Usage: sbatch scripts/train.sh <config.yaml> <paragraphs.pkl> <splits.json> [backbone_decision.json]}"
PARAGRAPHS_PATH="${2:?missing paragraphs.pkl path}"
SPLITS_PATH="${3:?missing splits.json path}"
BACKBONE_DECISION_PATH="${4:-}"   # optional

# ----- Narval modules -----
module load StdEnv/2023
module load arrow/24.0.0

# ----- project venv (shared with SDD) -----
source ~/sdd_env/bin/activate

# ----- determinism env (spec Component 14) -----
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR"

# Sanity check: we're in the project root
if [ ! -f pyproject.toml ] || [ ! -d src ]; then
    echo "[train.sh] ERROR: not in the project root (expected pyproject.toml + src/)."
    echo "[train.sh] SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
    exit 1
fi

# Ensure output directories exist (data/ + results/ are symlinks to $SCRATCH)
mkdir -p results/logs results/models

FOLD_IDX="${SLURM_ARRAY_TASK_ID:-0}"

EXTRA_ARGS=()
if [ -n "$BACKBONE_DECISION_PATH" ]; then
    EXTRA_ARGS+=(--backbone-decision "$BACKBONE_DECISION_PATH")
fi

echo "================================================================"
echo "LogFiT training — fold $FOLD_IDX"
echo "  config:      $CONFIG_PATH"
echo "  paragraphs:  $PARAGRAPHS_PATH"
echo "  splits:      $SPLITS_PATH"
echo "  decision:    ${BACKBONE_DECISION_PATH:-<none — YAML default>}"
echo "  runs-root:   results/models"
echo "  job_id:      ${SLURM_JOB_ID:-N/A}"
echo "  array_job:   ${SLURM_ARRAY_JOB_ID:-N/A}"
echo "  task_id:     ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "  account:     ${SLURM_JOB_ACCOUNT:-<unset — check SBATCH_ACCOUNT>}"
echo "  node:        $(hostname)"
echo "  gpu:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  start:       $(date -Iseconds)"
echo "================================================================"

python -m src.train \
    --config "$CONFIG_PATH" \
    --paragraphs "$PARAGRAPHS_PATH" \
    --splits "$SPLITS_PATH" \
    --fold-idx "$FOLD_IDX" \
    --runs-root results/models \
    "${EXTRA_ARGS[@]}"

echo "================================================================"
echo "Fold $FOLD_IDX done at $(date -Iseconds)"
echo "================================================================"
