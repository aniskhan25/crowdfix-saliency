"""
Extract frames from CrowdFix videos (.mp4 or .avi).

Usage:
    python data/extract_frames.py --videos-dir /path/to/Videos --out-dir /path/to/Frames

Outputs: <out-dir>/<video_stem>/frame_%05d.jpg
Frame names match the index convention used by CrowdFix saliency/fixation maps.
"""

import argparse
import subprocess
from pathlib import Path


def extract(video_path: Path, out_dir: Path, fps: int = 30) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "frame_%05d.jpg")
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vf", f"fps={fps}", "-q:v", "2", pattern],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"  WARNING: ffmpeg failed for {video_path.name}: {result.stderr.decode()[:200]}")
        return 0
    return len(list(out_dir.glob("frame_*.jpg")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    out_dir = Path(args.out_dir)
    videos = sorted(
        v for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv")
        for v in videos_dir.glob(ext)
    )

    if not videos:
        raise SystemExit(f"No video files (.mp4/.avi/.mov/.mkv) found in {videos_dir}")

    total_frames = 0
    for i, video in enumerate(videos, 1):
        dest = out_dir / video.stem
        if dest.exists() and any(dest.iterdir()):
            print(f"[{i}/{len(videos)}] {video.stem}: skipped (already extracted)")
            continue
        n = extract(video, dest, fps=args.fps)
        total_frames += n
        print(f"[{i}/{len(videos)}] {video.stem}: {n} frames")

    print(f"\nDone. {total_frames} frames extracted to {out_dir}")


if __name__ == "__main__":
    main()
