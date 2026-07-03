#!/bin/bash
# Phase 3a: Pretrain VideoSwinSaliency on DHF1K (600 train clips).
# Decoder and saliency head train from scratch; Swin3D-S encoder starts from
# Kinetics-400, frozen for 10 epochs then unfrozen at 0.1× LR.
#
# After this job, run finetune_crowdfix_dhf1k.sh to transfer to CrowdFix.
#
# Usage: cd <repo-root> && sbatch slurm/pretrain_dhf1k.sh

#SBATCH --job-name=crowdfix-dhf1k-pre
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pretrain_dhf1k_%j.log

set -euo pipefail

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

cd "${SLURM_SUBMIT_DIR}"

source env.sh
: "${CONTAINER:?Set CONTAINER in env.sh}"
: "${SQSH_PATH:?Set SQSH_PATH in env.sh}"
[[ -f "$SQSH_PATH" ]] || { echo "ERROR: Missing sqsh overlay: $SQSH_PATH" >&2; exit 1; }

# DHF1K must be at this path; adjust if downloaded elsewhere
DHF1K_DIR="${SCRATCH}/dhf1k-data"
[[ -d "${DHF1K_DIR}/training" ]] || {
    echo "ERROR: DHF1K training split not found at ${DHF1K_DIR}/training" >&2
    echo "Download DHF1K from https://mmcheng.net/videosal/ and extract to ${DHF1K_DIR}" >&2
    exit 1
}

mkdir -p logs checkpoints/dhf1k_pretrain

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB
export MIOPEN_USER_DB_PATH="/tmp/${USER}_${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="/tmp/${USER}_${SLURM_JOB_ID}"
export TORCH_HOME="${SCRATCH}/.torch_hub"

time srun singularity exec -B "$SQSH_PATH":/user-software:image-src=/ "$CONTAINER" \
  /user-software/bin/python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=8 \
  train.py \
    --model                swin \
    --dataset              dhf1k \
    --dhf1k-dir            "$DHF1K_DIR" \
    --checkpoint-dir       checkpoints/dhf1k_pretrain \
    --epochs               60 \
    --batch-size           2 \
    --lr                   1e-4 \
    --clip-len             8 \
    --freeze-epochs        10 \
    --backbone-lr-scale    0.1 \
    --early-stop-patience  15
