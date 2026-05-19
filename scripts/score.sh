#!/bin/bash
# scripts/score.sh — SLURM array job to score all 5 folds of one LogFiT
# dataset configuration. Scores BOTH tune and test splits per fold under
# `--split=both`, producing scores_tune.json + scores_test.json.
#
# Usage:
#   sbatch scripts/score.sh \
#       <config.yaml> \
#       <paragraphs.pkl> \
#       <splits.json> \
#       <model_template> \
#       <output_root> \
#       [backbone_decision.json]
#
#   <model_template> contains the literal substring '{FOLD_IDX}' which is
#   replaced with the SLURM array task ID (0..4) inside the script.
#
# Examples:
#   # HDFS (RoBERTa-base)
#   sbatch scripts/score.sh \
#       configs/hdfs.yaml \
#       data/processed/hdfs_subset/paragraphs.pkl \
#       data/processed/hdfs_subset/splits.json \
#       'results/models/hdfs/fold_{FOLD_IDX}' \
#       results/scores/hdfs \
#       data/processed/hdfs_subset/backbone_decision.json
#
#   # BGL 30s (Longformer)
#   sbatch scripts/score.sh \
#       configs/bgl_30s.yaml \
#       data/processed/bgl_30s_subset/paragraphs.pkl \
#       data/processed/bgl_30s_subset/splits.json \
#       'results/models/bgl_30s/fold_{FOLD_IDX}' \
#       results/scores/bgl_30s
#
# Submits 5 parallel array tasks, one per fold. $SLURM_ARRAY_TASK_ID becomes
# the --fold-idx passed to `python -m src.score`. Output goes to
#   ${OUTPUT_ROOT}/fold_${FOLD_IDX}/scores_tune.json
#   ${OUTPUT_ROOT}/fold_${FOLD_IDX}/scores_test.json
#
# Account: set SBATCH_ACCOUNT in ~/.bashrc (same convention as train.sh).
#
# Prerequisites:
#   - ~/sdd_activate.sh exists (project's canonical activation script)
#   - data/ + results/ symlinks present
#   - models from train.sh exist under <model_template> for each fold
#   - submitted from the project root

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --job-name=logfit-score
#SBATCH --output=results/logs/score-%A_%a.out
#SBATCH --error=results/logs/score-%A_%a.err

set -euo pipefail

# ----- positional args -----
CONFIG_PATH="${1:?Usage: sbatch scripts/score.sh <config.yaml> <paragraphs.pkl> <splits.json> <model_template> <output_root> [backbone_decision.json]}"
PARAGRAPHS_PATH="${2:?missing paragraphs.pkl path}"
SPLITS_PATH="${3:?missing splits.json path}"
MODEL_TEMPLATE="${4:?missing model template (must contain literal {FOLD_IDX})}"
OUTPUT_ROOT="${5:?missing output_root}"
BACKBONE_DECISION_PATH="${6:-}"   # optional

# ----- environment: use the project's canonical activation script -----
# Mirrors the interactive setup; guarantees pyarrow visibility (via the
# arrow/24.0.0 module's PYTHONPATH) and venv state regardless of which
# shell submitted this job.
source ~/sdd_activate.sh

# ----- determinism env (same as train.sh) -----
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false

# ----- HF offline mode (compute nodes have no outbound internet) -----
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"

# Sanity check: in project root
if [ ! -f pyproject.toml ] || [ ! -d src ]; then
    echo "[score.sh] ERROR: not in the project root (expected pyproject.toml + src/)."
    echo "[score.sh] SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
    exit 1
fi

FOLD_IDX="${SLURM_ARRAY_TASK_ID:-0}"

# Substitute {FOLD_IDX} -> 0/1/2/3/4 in the model template
MODEL_PATH="${MODEL_TEMPLATE//\{FOLD_IDX\}/$FOLD_IDX}"

if [ ! -e "$MODEL_PATH" ]; then
    echo "[score.sh] ERROR: model path does not exist: $MODEL_PATH"
    echo "[score.sh] (Template was: $MODEL_TEMPLATE)"
    exit 1
fi

# Per-fold output directory
PER_FOLD_OUTPUT="$OUTPUT_ROOT/fold_$FOLD_IDX"
mkdir -p "$PER_FOLD_OUTPUT" results/logs

EXTRA_ARGS=()
if [ -n "$BACKBONE_DECISION_PATH" ]; then
    EXTRA_ARGS+=(--backbone-decision "$BACKBONE_DECISION_PATH")
fi

echo "================================================================"
echo "LogFiT scoring — fold $FOLD_IDX"
echo "  config:        $CONFIG_PATH"
echo "  paragraphs:    $PARAGRAPHS_PATH"
echo "  splits:        $SPLITS_PATH"
echo "  model:         $MODEL_PATH"
echo "  output:        $PER_FOLD_OUTPUT"
echo "  decision:      ${BACKBONE_DECISION_PATH:-<none — YAML default>}"
echo "  splits-scored: tune + test (--split both)"
echo "  job_id:        ${SLURM_JOB_ID:-N/A}"
echo "  array_job:     ${SLURM_ARRAY_JOB_ID:-N/A}"
echo "  task_id:       ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "  account:       ${SLURM_JOB_ACCOUNT:-<unset — check SBATCH_ACCOUNT>}"
echo "  node:          $(hostname)"
echo "  gpu:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  start:         $(date -Iseconds)"
echo "================================================================"

python -m src.score \
    --config "$CONFIG_PATH" \
    --paragraphs "$PARAGRAPHS_PATH" \
    --splits "$SPLITS_PATH" \
    --fold-idx "$FOLD_IDX" \
    --model "$MODEL_PATH" \
    --output "$PER_FOLD_OUTPUT" \
    --split both \
    "${EXTRA_ARGS[@]}"

echo "================================================================"
echo "Fold $FOLD_IDX scoring done at $(date -Iseconds)"
echo "================================================================"
