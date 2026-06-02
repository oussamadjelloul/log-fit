#!/bin/bash
# scripts/prep_hdfs_canon.sh — build the CANONICALIZED HDFS paragraphs.pkl from
# the full HDFS_v1 parquet, replicating the logfit-project canonicalization.
# CPU-only job (no GPU). Companion to the raw src/prepare_hdfs.py, which is left
# UNTOUCHED. Produces a drop-in paragraphs.pkl for the usual pipeline:
#   token_length_gate -> splits -> train -> score -> eval
#
# ONE-TIME download on a LOGIN node (compute nodes have no internet):
#   source ~/sdd_activate.sh
#   hf download logfit-project/HDFS_v1 --repo-type dataset \
#       --local-dir $SCRATCH/log-fit/data/raw/HDFS_v1_hf
#   # anomaly_label.csv is your EXISTING loghub label file (reuse the raw arm's).
#
# Usage:
#   sbatch scripts/prep_hdfs_canon.sh <hdfs_v1_parquet_dir> <anomaly_label.csv> <output_dir>
#
# Example:
#   sbatch scripts/prep_hdfs_canon.sh \
#       data/raw/HDFS_v1_hf \
#       data/raw/hdfs/anomaly_label.csv \
#       data/processed/hdfs_canon_full
#
# (CPU + a few minutes; can also be run directly on a login node for testing:
#   python -m src.prepare_hdfs_canon --hdfs-v1-parquet ... --label-csv ... --output-dir ...)
#
# Account: set SBATCH_ACCOUNT in ~/.bashrc (same convention as train.sh/score.sh).

#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=logfit-prep-canon
#SBATCH --output=results/logs/prep-canon-%j.out
#SBATCH --error=results/logs/prep-canon-%j.err

set -euo pipefail

PARQUET_DIR="${1:?Usage: sbatch scripts/prep_hdfs_canon.sh <hdfs_v1_parquet_dir> <anomaly_label.csv> <output_dir>}"
LABEL_CSV="${2:?missing anomaly_label.csv path}"
OUTPUT_DIR="${3:?missing output_dir}"

source ~/sdd_activate.sh
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"
if [ ! -f pyproject.toml ] || [ ! -d src ]; then
    echo "[prep_hdfs_canon.sh] ERROR: not in the project root (expected pyproject.toml + src/)."
    echo "[prep_hdfs_canon.sh] SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
    exit 1
fi
mkdir -p results/logs "$OUTPUT_DIR"

echo "================================================================"
echo "LogFiT HDFS canon prep"
echo "  parquet:   $PARQUET_DIR"
echo "  label-csv: $LABEL_CSV"
echo "  output:    $OUTPUT_DIR"
echo "  node:      $(hostname)"
echo "  start:     $(date -Iseconds)"
echo "================================================================"

python -m src.prepare_hdfs_canon \
    --hdfs-v1-parquet "$PARQUET_DIR" \
    --label-csv "$LABEL_CSV" \
    --output-dir "$OUTPUT_DIR"

echo "================================================================"
echo "Canon prep done at $(date -Iseconds)"
echo "================================================================"
