#!/usr/bin/env bash
# Download CrowdFix dataset from Google Drive to LUMI scratch.
#
# Usage (on LUMI login node):
#   source env.sh
#   bash data/download_data.sh
#
# The Google Drive folder ID is from the CrowdFix README.
# If gdown fails (quota exceeded), use rclone or download locally then rsync:
#   rsync -avz --progress crowdfix-data/ anisrahm@lumi.csc.fi:/scratch/project_462000131/anisrahm/crowdfix-data/

set -euo pipefail

CROWDFIX_GDRIVE_ID="1mRvkwqJM2ulFYZotV-L-W3j8xJ-mEsvn"
DEST="${DATA_DIR:-/scratch/project_462000131/anisrahm/crowdfix-data}"

echo "Downloading CrowdFix to ${DEST}..."
mkdir -p "$DEST"

# Set Lustre striping before writing large files
if command -v lfs &>/dev/null; then
    lfs setstripe -S 4m -c 8 "$DEST"
    echo "Lustre striping set: 4MB stripe, 8 OSTs"
fi

# Try gdown first
if ! command -v gdown &>/dev/null; then
    pip install --user gdown
fi

gdown --folder "$CROWDFIX_GDRIVE_ID" -O "$DEST" || {
    echo ""
    echo "gdown failed (Google Drive quota or link issue)."
    echo "Manual fallback: download the dataset locally, then run:"
    echo "  rsync -avz --progress crowdfix-data/ anisrahm@lumi.csc.fi:${DEST}/"
    exit 1
}

echo "Download complete. Contents:"
ls -lh "$DEST"
