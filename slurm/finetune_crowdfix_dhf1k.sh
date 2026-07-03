#!/bin/bash
# Phase 3b: Fine-tune DensitySwinSaliency on CrowdFix, initialised from the
# DHF1K-pretrained VideoSwinSaliency checkpoint (encoder + decoder transfer;
# FiLM layers initialise fresh via strict=False loading).
#
# Run AFTER pretrain_dhf1k.sh completes.
#
# Usage: cd <repo-root> && sbatch slurm/finetune_crowdfix_dhf1k.sh

#SBATCH --job-name=crowdfix-dhf1k-ft
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=02:00:00
#SBATCH --output=logs/finetune_dhf1k_%j.log

set -euo pipefail

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

cd "${SLURM_SUBMIT_DIR}"

source env.sh
: "${CONTAINER:?Set CONTAINER in env.sh}"
: "${DATA_DIR:?Set DATA_DIR in env.sh}"
: "${SQSH_PATH:?Set SQSH_PATH in env.sh}"
[[ -f "$SQSH_PATH" ]] || { echo "ERROR: Missing sqsh overlay: $SQSH_PATH" >&2; exit 1; }

PRETRAIN_CKPT="checkpoints/dhf1k_pretrain/best.pth"
[[ -f "$PRETRAIN_CKPT" ]] || {
    echo "ERROR: DHF1K pretrain checkpoint not found at ${PRETRAIN_CKPT}" >&2
    echo "Run pretrain_dhf1k.sh first." >&2
    exit 1
}

mkdir -p logs checkpoints/dhf1k_finetune

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB
export MIOPEN_USER_DB_PATH="/tmp/${USER}_${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="/tmp/${USER}_${SLURM_JOB_ID}"
export TORCH_HOME="${SCRATCH}/.torch_hub"

time srun singularity exec -B "$SQSH_PATH":/user-software:image-src=/ "$CONTAINER" \
  /user-software/bin/python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=8 \
  train.py \
    --model                density_swin \
    --data-dir             "$DATA_DIR" \
    --splits               splits.json \
    --checkpoint-dir       checkpoints/dhf1k_finetune \
    --init-from            "$PRETRAIN_CKPT" \
    --epochs               150 \
    --batch-size           2 \
    --lr                   5e-5 \
    --clip-len             8 \
    --freeze-epochs        10 \
    --backbone-lr-scale    0.1 \
    --early-stop-patience  20
