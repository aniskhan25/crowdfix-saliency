#!/bin/bash
# Full training run for density_swin on small-g (1 node, 8 GCDs).
# 100 epochs at ~40 s/epoch = ~67 min; 3 h gives comfortable headroom.
#
# Usage: cd <repo-root> && sbatch slurm/train_full_density.sh

#SBATCH --job-name=crowdfix-density-train
#SBATCH --account=project_462000131
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=03:00:00
#SBATCH --output=logs/train_full_density_%j.log

set -euo pipefail

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

cd "${SLURM_SUBMIT_DIR}"

source env.sh
: "${CONTAINER:?Set CONTAINER in env.sh}"
: "${DATA_DIR:?Set DATA_DIR in env.sh}"
: "${SQSH_PATH:?Set SQSH_PATH in env.sh}"
[[ -f "$SQSH_PATH" ]] || { echo "ERROR: Missing sqsh overlay: $SQSH_PATH" >&2; exit 1; }

mkdir -p logs checkpoints

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB
export MIOPEN_USER_DB_PATH=/tmp
export MIOPEN_CUSTOM_CACHE_DIR=/tmp
export TORCH_HOME="${SCRATCH}/.torch_hub"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

time srun singularity exec -B "$SQSH_PATH":/user-software:image-src=/ "$CONTAINER" \
  /user-software/bin/python -m torch.distributed.run \
  --nnodes="$SLURM_JOB_NUM_NODES" \
  --nproc_per_node=8 \
  --rdzv_id="$SLURM_JOB_ID" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  train.py \
    --model              density_swin \
    --data-dir           "$DATA_DIR" \
    --splits             splits.json \
    --checkpoint-dir     "$CKPT_DIR" \
    --epochs             100 \
    --batch-size         2 \
    --lr                 1e-4 \
    --clip-len           8 \
    --freeze-epochs      20 \
    --backbone-lr-scale  0.1
