#!/bin/bash
# Phase 2 evaluation: oracle + debiased NSS for DensitySwinSoft and DensitySwinMultiscale.
# Assumes results/mean_gt_saliency.npy already exists (from eval_phase1.sh Step 1).
# If it doesn't exist, Step 0 recomputes it.
#
# Usage: cd <repo-root> && sbatch slurm/eval_phase2.sh

#SBATCH --job-name=crowdfix-eval-p2
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_phase2_%j.log

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

MEAN_GT="results/mean_gt_saliency.npy"
PY="singularity exec -B ${SQSH_PATH}:/user-software:image-src=/ ${CONTAINER} /user-software/bin/python"

# ── Step 0: recompute mean GT if missing ─────────────────────────────────────
if [[ ! -f "$MEAN_GT" ]]; then
    echo "=== Step 0: compute mean GT saliency ==="
    $PY scripts/compute_mean_gt.py \
        --data-dir  "$DATA_DIR" \
        --splits    splits.json \
        --out       "$MEAN_GT" \
        --frame-size 224 384
fi

# ── DensitySwinSoft ──────────────────────────────────────────────────────────
echo "=== Soft — oracle eval ==="
$PY evaluate.py \
    --checkpoint   checkpoints/soft/best.pth \
    --model        density_swin_soft \
    --data-dir     "$DATA_DIR" \
    --splits       splits.json \
    --batch-size   4 \
    --clip-len     8 \
    --density-mode oracle \
    --save-clips   results/eval_clips_soft.json

echo "=== Soft — debiased eval ==="
$PY evaluate.py \
    --checkpoint   checkpoints/soft/best.pth \
    --model        density_swin_soft \
    --data-dir     "$DATA_DIR" \
    --splits       splits.json \
    --batch-size   4 \
    --clip-len     8 \
    --density-mode oracle \
    --debias-map   "$MEAN_GT"

# ── DensitySwinMultiscale ─────────────────────────────────────────────────────
echo "=== Multiscale — oracle eval ==="
$PY evaluate.py \
    --checkpoint   checkpoints/multiscale/best.pth \
    --model        density_swin_multiscale \
    --data-dir     "$DATA_DIR" \
    --splits       splits.json \
    --batch-size   4 \
    --clip-len     8 \
    --density-mode oracle \
    --save-clips   results/eval_clips_multiscale.json

echo "=== Multiscale — debiased eval ==="
$PY evaluate.py \
    --checkpoint   checkpoints/multiscale/best.pth \
    --model        density_swin_multiscale \
    --data-dir     "$DATA_DIR" \
    --splits       splits.json \
    --batch-size   4 \
    --clip-len     8 \
    --density-mode oracle \
    --debias-map   "$MEAN_GT"

echo "=== Phase 2 eval complete ==="
