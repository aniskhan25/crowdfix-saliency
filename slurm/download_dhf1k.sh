#!/bin/bash
# Download and extract DHF1K to $SCRATCH/dhf1k-data/.
# Runs on a CPU node (no GPU needed for download/extraction).
#
# DHF1K: 600 train + 100 val video clips with saliency and fixation maps.
# Source: https://mmcheng.net/videosal/
#
# Usage: cd <repo-root> && sbatch slurm/download_dhf1k.sh

#SBATCH --job-name=download-dhf1k
#SBATCH --account=project_462000131
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/download_dhf1k_%j.log

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source env.sh

DHF1K_DIR="${SCRATCH}/dhf1k-data"
mkdir -p "${DHF1K_DIR}" logs
cd "${DHF1K_DIR}"

echo "=== Downloading DHF1K to ${DHF1K_DIR} ==="

# Primary: direct download from mmcheng.net
# The dataset page is at https://mmcheng.net/videosal/
# If this URL has changed, download manually and place at ${DHF1K_DIR}
ARCHIVE="DHF1K.rar"

if [[ ! -f "${ARCHIVE}" ]]; then
    wget -c --timeout=60 --tries=5 \
        "https://mmcheng.net/wp-content/uploads/2019/07/DHF1K.rar" \
        -O "${ARCHIVE}" \
    || {
        echo "ERROR: wget failed. Try downloading manually:"
        echo "  Visit https://mmcheng.net/videosal/ and place DHF1K.rar at ${DHF1K_DIR}"
        exit 1
    }
else
    echo "Archive already present, skipping download."
fi

echo "=== Extracting ${ARCHIVE} ==="
if command -v unrar &>/dev/null; then
    unrar x -o+ "${ARCHIVE}"
elif command -v 7z &>/dev/null; then
    7z x "${ARCHIVE}" -aoa
else
    # Try via Python rarfile module (may be in the container sqsh)
    module use /appl/local/containers/ai-modules
    module load singularity-AI-bindings
    : "${CONTAINER:?}" "${SQSH_PATH:?}"
    singularity exec -B "${SQSH_PATH}":/user-software:image-src=/ "${CONTAINER}" \
        /user-software/bin/python -c "
import rarfile, sys
rf = rarfile.RarFile('${ARCHIVE}')
rf.extractall('.')
print('Extracted', len(rf.namelist()), 'entries')
" || {
        echo "ERROR: no extraction tool available (tried unrar, 7z, rarfile)."
        echo "Install unrar: module load unrar  OR  extract locally and rsync."
        exit 1
    }
fi

echo "=== Verifying structure ==="
if [[ -d "training" && -d "validation" ]]; then
    NTRAIN=$(ls training/ | wc -l)
    NVAL=$(ls validation/ | wc -l)
    echo "OK: training=${NTRAIN} videos, validation=${NVAL} videos"
else
    # DHF1K sometimes extracts with a top-level subfolder
    SUBDIR=$(find . -maxdepth 2 -name "training" -type d | head -1 | xargs dirname)
    if [[ -n "${SUBDIR}" && "${SUBDIR}" != "." ]]; then
        echo "Moving contents of ${SUBDIR} up one level..."
        mv "${SUBDIR}"/* .
        rmdir "${SUBDIR}" 2>/dev/null || true
    else
        echo "WARNING: expected training/ and validation/ subdirs not found."
        echo "Contents of ${DHF1K_DIR}:"; ls -la
    fi
fi

echo "=== DHF1K download complete: ${DHF1K_DIR} ==="
