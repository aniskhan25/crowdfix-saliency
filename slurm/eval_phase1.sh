#!/bin/bash
# Phase 1 evaluation pipeline:
#   Step 1: compute mean GT saliency map (centre-bias prior)
#   Step 2: evaluate model with debiasing (debiased NSS)
#   Step 3: evaluate centre-bias upper-bound baseline
#   Step 4: save per-clip metrics for SP group-size analysis
#
# Usage:
#   sbatch slurm/eval_phase1.sh [checkpoint] [model]
#   sbatch slurm/eval_phase1.sh checkpoints/best.pth density_swin

#SBATCH --job-name=crowdfix-eval-p1
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_phase1_%j.log

set -euo pipefail

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

cd "${SLURM_SUBMIT_DIR}"
source env.sh

: "${CONTAINER:?}"
: "${DATA_DIR:?}"
: "${SQSH_PATH:?}"
[[ -f "$SQSH_PATH" ]] || { echo "ERROR: missing sqsh $SQSH_PATH" >&2; exit 1; }

export TORCH_HOME="${SCRATCH}/.torch_hub"
export MIOPEN_USER_DB_PATH="/tmp/${USER}_${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="/tmp/${USER}_${SLURM_JOB_ID}"
mkdir -p "${MIOPEN_USER_DB_PATH}" logs results

CKPT="${1:-checkpoints/best.pth}"
MODEL="${2:-density_swin}"
MEAN_GT="results/mean_gt_saliency.npy"

PY="singularity exec -B ${SQSH_PATH}:/user-software:image-src=/ ${CONTAINER} /user-software/bin/python"

# ── Step 1: compute mean GT saliency (CPU-only, runs fast) ────────────────────
echo "=== Step 1: compute mean GT saliency ==="
$PY scripts/compute_mean_gt.py \
    --data-dir  "$DATA_DIR" \
    --splits    splits.json \
    --out       "$MEAN_GT" \
    --frame-size 224 384

# ── Step 2: standard oracle eval (baseline numbers, save per-clip JSON) ───────
echo "=== Step 2: oracle eval (standard) ==="
$PY evaluate.py \
    --checkpoint   "$CKPT" \
    --model        "$MODEL" \
    --data-dir     "$DATA_DIR" \
    --splits       splits.json \
    --batch-size   4 \
    --clip-len     8 \
    --density-mode oracle \
    --save-clips   results/eval_clips_oracle.json

# ── Step 3: debiased eval ─────────────────────────────────────────────────────
echo "=== Step 3: debiased NSS eval ==="
$PY evaluate.py \
    --checkpoint   "$CKPT" \
    --model        "$MODEL" \
    --data-dir     "$DATA_DIR" \
    --splits       splits.json \
    --batch-size   4 \
    --clip-len     8 \
    --density-mode oracle \
    --debias-map   "$MEAN_GT"

# ── Step 4: centre-bias upper-bound baseline ──────────────────────────────────
echo "=== Step 4: centre-bias baseline ==="
$PY evaluate.py \
    --checkpoint   "$CKPT" \
    --model        "$MODEL" \
    --data-dir     "$DATA_DIR" \
    --splits       splits.json \
    --batch-size   4 \
    --clip-len     8 \
    --baseline     centre-bias \
    --debias-map   "$MEAN_GT"

echo "=== Phase 1 eval complete ==="
echo "Per-clip results at: results/eval_clips_oracle.json"
echo "Run analyze_sp_groupsize.py locally after syncing results/"
