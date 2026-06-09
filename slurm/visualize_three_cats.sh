#!/bin/bash
# Generate qualitative visualizations for one video per density category
# using the run 4 (density_swin) best checkpoint.
#
# Videos: 026 (SP), 004 (DF), 002 (DC) — from the test split.
# Output: results/<video_id>/ (frame PNGs) + results/<video_id>.mp4
#
# Usage: cd <repo-root> && sbatch slurm/visualize_three_cats.sh

#SBATCH --job-name=crowdfix-viz
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=logs/viz_%j.log

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
export MIOPEN_USER_DB_PATH="/tmp/${USER}_${SLURM_JOB_ID}_rank0"
export MIOPEN_CUSTOM_CACHE_DIR="/tmp/${USER}_${SLURM_JOB_ID}_rank0"
mkdir -p "${MIOPEN_USER_DB_PATH}" logs results

run_viz() {
    local video_id=$1
    local density=$2
    echo "=== Visualising video ${video_id} (density=${density}) ==="
    singularity exec -B "${SQSH_PATH}:/user-software:image-src=/" "${CONTAINER}" \
      /user-software/bin/python visualize.py \
        --checkpoint checkpoints/best.pth \
        --model      density_swin \
        --data-dir   "$DATA_DIR" \
        --video-id   "$video_id" \
        --density    "$density" \
        --out        results \
        --clip-len   8
}

run_viz 026 0   # SP — Sparse Pedestrian
run_viz 004 1   # DF — Dense Free-flowing
run_viz 002 2   # DC — Dense Congested
