"""
Analyse the learned FiLM parameters (γ, β) of a DensitySwinSaliency checkpoint.

Reports:
  - Pairwise cosine similarity between the three class-specific (γ, β) vectors
  - Top-20 channels by |Δγ| between DC and SP (density-discriminative channels)
  - Saves a per-channel |γ| heatmap (3 × 768) to results/film_analysis.png

Usage:
    python scripts/analyze_film.py --checkpoint checkpoints/best.pth

If matplotlib is unavailable the heatmap step is skipped gracefully.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from models.density_swin_saliency import DensitySwinSaliency

DENSITY_NAMES = ["SP", "DF", "DC"]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu")

    model = DensitySwinSaliency(pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Extract the three (γ, β) vectors: shape (3, 1536)
    with torch.no_grad():
        emb_weight = model.density_emb.weight        # (3, 64)
        params = model.film.proj(emb_weight)          # (3, 1536)
    params_np = params.numpy()                        # (3, 1536)
    gamma = params_np[:, :768]                        # (3, 768)  scale offsets
    beta  = params_np[:, 768:]                        # (3, 768)  shifts

    print("=" * 60)
    print("FiLM parameter analysis")
    print("=" * 60)

    # ── Cosine similarities ──────────────────────────────────────────────────
    print("\nPairwise cosine similarity of full (γ, β) vectors:")
    for i in range(3):
        for j in range(i + 1, 3):
            sim = cosine_sim(params_np[i], params_np[j])
            print(f"  {DENSITY_NAMES[i]} vs {DENSITY_NAMES[j]}: {sim:.4f}")

    print("\nPairwise cosine similarity of γ vectors only:")
    for i in range(3):
        for j in range(i + 1, 3):
            sim = cosine_sim(gamma[i], gamma[j])
            print(f"  {DENSITY_NAMES[i]} vs {DENSITY_NAMES[j]}: {sim:.4f}")

    # ── L2 norm of each class's (γ, β) ──────────────────────────────────────
    print("\nL2 norm of (γ, β) per class (larger = stronger modulation):")
    for i in range(3):
        norm = float(np.linalg.norm(params_np[i]))
        gnorm = float(np.linalg.norm(gamma[i]))
        bnorm = float(np.linalg.norm(beta[i]))
        print(f"  {DENSITY_NAMES[i]}: total={norm:.3f}  γ={gnorm:.3f}  β={bnorm:.3f}")

    # ── Top channels by |Δγ| between DC and SP ─────────────────────────────
    delta_gamma = np.abs(gamma[2] - gamma[0])  # |DC - SP|
    top20_idx = np.argsort(delta_gamma)[::-1][:20]
    print("\nTop-20 channels by |γ_DC - γ_SP| (most density-discriminative):")
    print(f"  {'Ch':>4}  {'γ_SP':>8}  {'γ_DF':>8}  {'γ_DC':>8}  {'|DC-SP|':>8}")
    print("  " + "-" * 46)
    for idx in top20_idx:
        print(f"  {idx:>4}  {gamma[0,idx]:>8.4f}  {gamma[1,idx]:>8.4f}"
              f"  {gamma[2,idx]:>8.4f}  {delta_gamma[idx]:>8.4f}")

    # ── Heatmap ──────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "film_analysis.png"

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(18, 5))
        for i, name in enumerate(DENSITY_NAMES):
            axes[i].bar(range(768), np.abs(gamma[i]), width=1.0, linewidth=0)
            axes[i].set_ylabel(f"|γ| {name}", fontsize=9)
            axes[i].set_xlim(0, 768)
            axes[i].tick_params(labelsize=7)
        axes[-1].set_xlabel("Channel index", fontsize=9)
        fig.suptitle("Per-channel |γ| scale factors by density class", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"\nHeatmap saved → {out_path}")
    except ImportError:
        print("\nmatplotlib not available — skipping heatmap.")

    # ── Summary verdict ──────────────────────────────────────────────────────
    sims = [cosine_sim(gamma[i], gamma[j]) for i in range(3) for j in range(i+1, 3)]
    max_sim = max(sims)
    if max_sim > 0.99:
        print("\n[!] Max pairwise γ cosine similarity > 0.99 — conditioning is near-degenerate.")
        print("    The three class embeddings are nearly collinear; FiLM is acting as")
        print("    a near-constant per-class bias rather than structured modulation.")
    else:
        print(f"\n[✓] Max pairwise γ cosine similarity = {max_sim:.4f} < 0.99.")
        print("    The class embeddings are distinguishable; FiLM is applying")
        print("    class-specific channel recalibration.")


if __name__ == "__main__":
    main()
