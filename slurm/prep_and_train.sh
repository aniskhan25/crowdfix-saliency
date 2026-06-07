#!/bin/bash
# One-shot prep job: download CrowdFix → unzip → extract frames → make splits → launch training.
#
# Usage: cd <repo-root> && sbatch slurm/prep_and_train.sh

#SBATCH --job-name=crowdfix-prep
#SBATCH --account=project_462000131
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=logs/prep_%j.log

set -euo pipefail

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

cd "${SLURM_SUBMIT_DIR}"
source env.sh

: "${CONTAINER:?Set CONTAINER in env.sh}"
: "${DATA_DIR:?Set DATA_DIR in env.sh}"
: "${SQSH_PATH:?Set SQSH_PATH in env.sh}"
[[ -f "$SQSH_PATH" ]] || { echo "ERROR: Missing sqsh overlay: $SQSH_PATH" >&2; exit 1; }

export TORCH_HOME="${SCRATCH}/.torch_hub"
mkdir -p logs "$DATA_DIR"

if command -v lfs &>/dev/null; then
    lfs setstripe -S 4m -c 8 "$DATA_DIR" 2>/dev/null || true
fi

RUN="singularity exec -B ${SQSH_PATH}:/user-software:image-src=/ ${CONTAINER} /user-software/bin/python"

GDRIVE_FOLDER="https://drive.google.com/drive/folders/1mRvkwqJM2ulFYZotV-L-W3j8xJ-mEsvn"

# ── Step 1: Download zips ────────────────────────────────────────────────────
# The folder contains: BinaryMaps.zip, SaliencyMaps.zip, videos.zip
if [[ ! -f "${DATA_DIR}/videos.zip" && ! -d "${DATA_DIR}/Videos" ]]; then
    echo "=== Downloading CrowdFix zips ==="
    $RUN -c "
import gdown
url = '${GDRIVE_FOLDER}'
files = gdown.download_folder(url, output='${DATA_DIR}', quiet=False)
print('Downloaded', len(files) if files else 0, 'files')
"
else
    echo "=== Zips already downloaded, skipping ==="
fi

# ── Step 2: Unzip ────────────────────────────────────────────────────────────
for zip_name in videos.zip SaliencyMaps.zip BinaryMaps.zip; do
    zip_path="${DATA_DIR}/${zip_name}"
    if [[ -f "$zip_path" ]]; then
        echo "=== Unzipping ${zip_name} ==="
        unzip -q -o "$zip_path" -d "$DATA_DIR"
        rm -f "$zip_path"
    fi
done

# Normalize Videos dir name (gdown may produce 'videos' lowercase)
if [[ -d "${DATA_DIR}/videos" && ! -d "${DATA_DIR}/Videos" ]]; then
    mv "${DATA_DIR}/videos" "${DATA_DIR}/Videos"
fi

# ── Step 3: Get CategoryInfo.xlsx ────────────────────────────────────────────
# Valid xlsx (ZIP-based) must be at least 1 KB; re-download if corrupt/missing.
_xlsx="${DATA_DIR}/CategoryInfo.xlsx"
_xlsx_ok() { [[ -f "$_xlsx" ]] && (( $(stat -c%s "$_xlsx" 2>/dev/null || stat -f%z "$_xlsx") > 1024 )); }

if ! _xlsx_ok; then
    echo "=== Downloading CategoryInfo.xlsx from GitHub (master branch) ==="
    curl -sL "https://raw.githubusercontent.com/MemoonaTahira/CrowdFix/master/CategoryInfo.xlsx" \
        -o "$_xlsx"
    _xlsx_ok || { echo "ERROR: CategoryInfo.xlsx download failed ($(stat -c%s "$_xlsx" 2>/dev/null) bytes)"; exit 1; }
    echo "CategoryInfo.xlsx: $(stat -c%s "$_xlsx" 2>/dev/null || stat -f%z "$_xlsx") bytes"
fi

# ── Step 4: Extract frames ──────────────────────────────────────────────────
if [[ ! -d "${DATA_DIR}/Frames" ]] || [[ -z "$(ls -A ${DATA_DIR}/Frames 2>/dev/null)" ]]; then
    echo "=== Extracting frames from ${DATA_DIR}/Videos ==="
    $RUN data/extract_frames.py \
        --videos-dir "${DATA_DIR}/Videos" \
        --out-dir    "${DATA_DIR}/Frames" \
        --fps 30
else
    echo "=== Frames already extracted, skipping ==="
fi

# ── Step 5: Make splits ─────────────────────────────────────────────────────
echo "=== Building splits.json ==="
$RUN data/make_splits.py \
    --category-info "${DATA_DIR}/CategoryInfo.xlsx" \
    --frames-dir    "${DATA_DIR}/Frames" \
    --out           splits.json

echo "=== Prep complete. Contents of ${DATA_DIR}:"
ls -lh "$DATA_DIR"

echo "=== Submitting training job ==="
sbatch slurm/train_full_density.sh
