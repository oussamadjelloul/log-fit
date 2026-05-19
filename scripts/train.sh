#!/bin/bash
#SBATCH --job-name=logfit-train
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/train_%j.out
#SBATCH --error=results/logs/train_%j.err

# Usage:
#   sbatch --account="${SLURM_ACCOUNT}" scripts/train.sh hdfs 0
#   sbatch --account="${SLURM_ACCOUNT}" scripts/train.sh bgl_30s 2

set -euo pipefail

DATASET=${1:-hdfs}   # hdfs | bgl_10s | bgl_30s | bgl_60s | tbird_10s | tbird_30s | tbird_60s
FOLD_IDX=${2:-0}

echo "=== LogFiT Train ==="
echo "Dataset: ${DATASET}"
echo "Fold:    ${FOLD_IDX}"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo ""

module load StdEnv/2023 gcc arrow/24.0.0 python/3.11
source ~/sdd_env/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

cd ~/log-fit

# Map dataset -> config + data folder
case "${DATASET}" in
  hdfs)
    CFG="configs/hdfs.yaml"
    DATA_DIR="data/hdfs_subset"
    ;;
  bgl_10s|bgl_30s|bgl_60s)
    CFG="configs/${DATASET}.yaml"
    DATA_DIR="data/bgl_subset"
    ;;
  tbird_10s|tbird_30s|tbird_60s)
    CFG="configs/${DATASET}.yaml"
    DATA_DIR="data/tbird_subset"
    ;;
  *)
    echo "ERROR: unknown dataset ${DATASET}"
    exit 1
    ;;
esac

mkdir -p results/models results/logs

python - <<PY
from src.train import train_fold_from_paths

train_fold_from_paths(
    config_path="${CFG}",
    paragraphs_pkl_path="${DATA_DIR}/paragraphs.pkl",
    splits_path="${DATA_DIR}/splits.json",
    fold_idx=int("${FOLD_IDX}"),
    backbone_decision_path="${DATA_DIR}/backbone_decision.json",
    runs_root="results/models",
    fp16_verification=False,
)
PY

echo ""
echo "End: $(date)"
echo "=== Done ==="
