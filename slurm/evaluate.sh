#!/bin/bash
#SBATCH --job-name=crowdfix-eval
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_%j.log

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
mkdir -p logs

CKPT="${1:-checkpoints/best.pth}"
MODEL="${2:-density_swin}"

echo "=== Evaluating $CKPT (model=$MODEL) ==="

singularity exec -B "${SQSH_PATH}:/user-software:image-src=/" "${CONTAINER}" \
  /user-software/bin/python evaluate.py \
    --checkpoint "$CKPT" \
    --model      "$MODEL" \
    --data-dir   "$DATA_DIR" \
    --splits     splits.json \
    --batch-size 4 \
    --clip-len   8
