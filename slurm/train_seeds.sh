#!/bin/bash
# Multi-seed training for variance estimation.
# Submits to small-g; each run saves to checkpoints/seed_$SEED/.
#
# Usage:
#   cd <repo-root>
#   sbatch slurm/train_seeds.sh 0   # seed 0
#   sbatch slurm/train_seeds.sh 1   # seed 1
#   sbatch slurm/train_seeds.sh 2   # seed 2
#
# The existing Run-4 checkpoint (checkpoints/best.pth) was trained with
# --seed 42 and acts as the fourth data point for mean±std computation.

#SBATCH --job-name=crowdfix-seed
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=01:30:00
#SBATCH --output=logs/train_seed_%a_%j.log

set -euo pipefail

SEED="${1:?Usage: sbatch train_seeds.sh <seed>}"

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

cd "${SLURM_SUBMIT_DIR}"

source env.sh
: "${CONTAINER:?Set CONTAINER in env.sh}"
: "${DATA_DIR:?Set DATA_DIR in env.sh}"
: "${SQSH_PATH:?Set SQSH_PATH in env.sh}"
[[ -f "$SQSH_PATH" ]] || { echo "ERROR: Missing sqsh overlay: $SQSH_PATH" >&2; exit 1; }

SEED_CKPT_DIR="${CKPT_DIR}/seed_${SEED}"
mkdir -p logs "$SEED_CKPT_DIR"

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB
export MIOPEN_USER_DB_PATH="/tmp/${USER}_${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="/tmp/${USER}_${SLURM_JOB_ID}"
export TORCH_HOME="${SCRATCH}/.torch_hub"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

echo "=== Training seed=${SEED}, checkpoint-dir=${SEED_CKPT_DIR} ==="

time srun singularity exec -B "$SQSH_PATH":/user-software:image-src=/ "$CONTAINER" \
  /user-software/bin/python -m torch.distributed.run \
  --nnodes="$SLURM_JOB_NUM_NODES" \
  --nproc_per_node=8 \
  --rdzv_id="$SLURM_JOB_ID" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  train.py \
    --model                density_swin \
    --data-dir             "$DATA_DIR" \
    --splits               splits.json \
    --checkpoint-dir       "$SEED_CKPT_DIR" \
    --epochs               150 \
    --batch-size           2 \
    --lr                   1e-4 \
    --clip-len             8 \
    --freeze-epochs        20 \
    --backbone-lr-scale    0.1 \
    --early-stop-patience  20 \
    --seed                 "$SEED"
