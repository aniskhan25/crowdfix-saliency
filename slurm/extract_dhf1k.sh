#!/bin/bash
# Extract DHF1K video.rar and extract frames from AVI files.
# Runs as a batch job to avoid login-node timeout.
#
# Usage: cd <repo-root> && sbatch slurm/extract_dhf1k.sh

#SBATCH --job-name=extract-dhf1k
#SBATCH --account=project_462000131
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/extract_dhf1k_%j.log

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source env.sh

DHF1K_DIR="${SCRATCH}/dhf1k-data"
UNRAR="${SCRATCH}/rar/unrar"

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings
: "${CONTAINER:?}" "${SQSH_PATH:?}"
PY="singularity exec -B ${SQSH_PATH}:/user-software:image-src=/ ${CONTAINER} /user-software/bin/python"

mkdir -p logs

# ── Step 1: extract video.rar ─────────────────────────────────────────────────
if [[ ! -d "${DHF1K_DIR}/video" ]]; then
    echo "=== Extracting video.rar (3.8GB) ==="
    "${UNRAR}" x -o+ "${DHF1K_DIR}/video.rar" "${DHF1K_DIR}/"
    echo "Extracted $(ls ${DHF1K_DIR}/video/ | wc -l) video files"
else
    echo "video/ already extracted, skipping."
fi

# ── Step 2: extract frames from AVI files ─────────────────────────────────────
echo "=== Extracting frames from AVI files ==="
$PY - << 'PYEOF'
import os, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

video_dir = Path(os.environ["SCRATCH"]) / "dhf1k-data" / "video"
frames_dir = Path(os.environ["SCRATCH"]) / "dhf1k-data" / "frames"
frames_dir.mkdir(exist_ok=True)

videos = sorted(video_dir.glob("*.AVI")) + sorted(video_dir.glob("*.avi"))
print(f"Found {len(videos)} videos", flush=True)

def extract_frames(vid_path):
    vid_id = vid_path.stem  # e.g. "001"
    out_dir = frames_dir / vid_id
    if out_dir.exists() and len(list(out_dir.glob("*.jpg"))) > 0:
        return f"{vid_id}: already done"
    out_dir.mkdir(exist_ok=True)
    cmd = ["ffmpeg", "-i", str(vid_path), "-q:v", "2",
           str(out_dir / "%04d.jpg"), "-loglevel", "error"]
    r = subprocess.run(cmd, capture_output=True)
    n = len(list(out_dir.glob("*.jpg")))
    return f"{vid_id}: {n} frames"

with ThreadPoolExecutor(max_workers=16) as ex:
    for i, res in enumerate(ex.map(extract_frames, videos)):
        if i % 50 == 0:
            print(f"  {i}/{len(videos)}: {res}", flush=True)

print("Frame extraction complete", flush=True)
PYEOF

# ── Step 3: build training/ and validation/ symlink structure ─────────────────
echo "=== Building training/ and validation/ structure ==="
$PY - << 'PYEOF'
import os, shutil
from pathlib import Path

base      = Path(os.environ["SCRATCH"]) / "dhf1k-data"
frames    = base / "frames"
annot     = base / "annotation"
train_dir = base / "training"
val_dir   = base / "validation"
train_dir.mkdir(exist_ok=True)
val_dir.mkdir(exist_ok=True)

all_ids = sorted(d.name for d in annot.iterdir() if d.is_dir())
train_ids = [v for v in all_ids if int(v) <= 600]
val_ids   = [v for v in all_ids if 601 <= int(v) <= 700]

def link_video(vid_id, dest_dir):
    vdir = dest_dir / vid_id
    vdir.mkdir(exist_ok=True)
    # symlink images/
    img_link = vdir / "images"
    if not img_link.exists():
        img_link.symlink_to(frames / vid_id)
    # symlink maps/ and fixation/
    for sub in ("maps", "fixation"):
        link = vdir / sub
        src  = annot / vid_id / sub
        if src.exists() and not link.exists():
            link.symlink_to(src)

for vid in train_ids:
    link_video(vid, train_dir)
for vid in val_ids:
    link_video(vid, val_dir)

print(f"training/: {len(train_ids)} videos, validation/: {len(val_ids)} videos")
PYEOF

echo "=== DHF1K ready at ${DHF1K_DIR} ==="
echo "training/: $(ls ${DHF1K_DIR}/training/ | wc -l) videos"
echo "validation/: $(ls ${DHF1K_DIR}/validation/ | wc -l) videos"
