"""
Evaluation script: compute Rank-1 and Rank-5 retrieval accuracy,
and generate a visual retrieval grid.

Usage:
    python src/evaluation/evaluate.py [--checkpoint outputs/best_model.pth]
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import discover_pairs, split_pairs, RetrievalDataset


def load_or_generate_test_pairs(data_root: str, seed: int) -> list[dict]:
    """Load saved test split if available, otherwise regenerate (with warning)."""
    splits_path = Path("data/splits/test.json")
    if splits_path.exists():
        with open(splits_path) as f:
            test_pairs = json.load(f)
        print(f"  Loaded saved test split from {splits_path} ({len(test_pairs)} pairs)")
        return test_pairs
    else:
        print("  WARNING: No saved test split found — regenerating from scratch.")
        print("           Run training first to save canonical splits.")
        pairs = discover_pairs(data_root)
        splits = split_pairs(pairs, seed=seed)
        return splits["test"]
from src.models.siamese import SiameseNetwork


@torch.no_grad()
def compute_embeddings(
    model: SiameseNetwork,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract embeddings and labels from a dataset."""
    model.eval()
    all_emb = []
    all_labels = []

    for imgs, labels in tqdm(loader, desc="  Embedding", leave=False):
        imgs = imgs.to(device)
        emb = model.get_embedding(imgs)
        all_emb.append(emb.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_emb), np.concatenate(all_labels)


def retrieval_accuracy(
    sketch_emb: np.ndarray,
    sketch_labels: np.ndarray,
    photo_emb: np.ndarray,
    photo_labels: np.ndarray,
    top_k: list[int] = [1, 5, 10],
) -> dict:
    """
    For each sketch, rank all photos by cosine similarity.
    Report Rank-k accuracy.
    """
    # Cosine similarity matrix: (n_sketches, n_photos)
    sim = sketch_emb @ photo_emb.T

    results = {}
    for k in top_k:
        correct = 0
        for i in range(len(sketch_labels)):
            top_indices = np.argsort(-sim[i])[:k]
            if sketch_labels[i] in photo_labels[top_indices]:
                correct += 1
        acc = correct / len(sketch_labels)
        results[f"rank_{k}"] = acc

    return results


def visualize_retrieval(
    test_pairs: list[dict],
    sketch_emb: np.ndarray,
    sketch_labels: np.ndarray,
    photo_emb: np.ndarray,
    photo_labels: np.ndarray,
    output_path: str,
    n_queries: int = 5,
    top_k: int = 5,
):
    """Generate a visual grid: query sketch | top-k retrieved photos."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        print("matplotlib/Pillow not available, skipping visualization")
        return

    sim = sketch_emb @ photo_emb.T

    fig, axes = plt.subplots(n_queries, top_k + 1, figsize=(3 * (top_k + 1), 3 * n_queries))
    if n_queries == 1:
        axes = axes[np.newaxis, :]

    # Pick n_queries evenly spaced sketches
    indices = np.linspace(0, len(test_pairs) - 1, n_queries, dtype=int)

    for row, idx in enumerate(indices):
        # Query sketch
        sketch_path = test_pairs[idx]["sketch"]
        sketch_img = Image.open(sketch_path).convert("RGB")
        axes[row, 0].imshow(sketch_img)
        axes[row, 0].set_title("Query sketch", fontsize=8)
        axes[row, 0].axis("off")

        # Top-k photos
        top_indices = np.argsort(-sim[idx])[:top_k]
        true_label = sketch_labels[idx]

        for col, photo_idx in enumerate(top_indices):
            photo_path = test_pairs[photo_idx]["photo"] if photo_idx < len(test_pairs) else None
            # Find the actual photo for this index
            # We need the photo corresponding to the photo_idx in the sorted pairs
            if photo_idx < len(test_pairs):
                photo_path = test_pairs[photo_idx]["photo"]
            else:
                continue
            photo_img = Image.open(photo_path).convert("RGB")
            axes[row, col + 1].imshow(photo_img)

            is_match = photo_labels[photo_idx] == true_label
            color = "green" if is_match else "red"
            axes[row, col + 1].set_title(
                f"{'✓' if is_match else '✗'} (sim={sim[idx, photo_idx]:.2f})",
                fontsize=8,
                color=color,
            )
            axes[row, col + 1].axis("off")

    plt.suptitle("Sketch → Photo Retrieval (green = correct match)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved retrieval visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate sketch-photo retrieval")
    parser.add_argument("--checkpoint", default="outputs/best_model.pth")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = ckpt.get("args", {})

    # Rebuild model
    embedding_dim = model_args.get("embedding_dim", 256)
    model = SiameseNetwork(embedding_dim=embedding_dim, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded model from epoch {ckpt.get('epoch', '?')}")

    # Data — load the exact saved test split to avoid leakage
    test_pairs = load_or_generate_test_pairs(args.data_root, args.seed)
    print(f"  Test set: {len(test_pairs)} pairs")

    if len(test_pairs) == 0:
        print("ERROR: No test pairs. Check data.")
        sys.exit(1)

    # Build retrieval datasets
    sketch_ds = RetrievalDataset(test_pairs, mode="sketch")
    photo_ds = RetrievalDataset(test_pairs, mode="photo")

    sketch_loader = DataLoader(sketch_ds, batch_size=args.batch_size, shuffle=False)
    photo_loader = DataLoader(photo_ds, batch_size=args.batch_size, shuffle=False)

    # Compute embeddings
    print("Computing sketch embeddings...")
    sketch_emb, sketch_labels = compute_embeddings(model, sketch_loader, device)
    print("Computing photo embeddings...")
    photo_emb, photo_labels = compute_embeddings(model, photo_loader, device)

    # Retrieval accuracy
    print("\nRetrieval accuracy:")
    results = retrieval_accuracy(sketch_emb, sketch_labels, photo_emb, photo_labels)
    for k, v in results.items():
        print(f"  {k}: {v:.4f} ({v * 100:.1f}%)")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Visualization
    vis_path = os.path.join(args.output_dir, "retrieval_visualization.png")
    visualize_retrieval(
        test_pairs, sketch_emb, sketch_labels,
        photo_emb, photo_labels, vis_path,
        n_queries=5, top_k=5,
    )


if __name__ == "__main__":
    main()
