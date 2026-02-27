"""
Training script for the Siamese sketch-photo matching network.

Usage:
    python src/training/train.py [--epochs 20] [--batch-size 32] [--lr 1e-4]
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import (
    discover_pairs,
    split_pairs,
    SiamesePairDataset,
)
from src.models.siamese import SiameseNetwork, ContrastiveLoss


def train_one_epoch(
    model: SiameseNetwork,
    loader: DataLoader,
    criterion: ContrastiveLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for sketches, photos, labels in tqdm(loader, desc="  Train", leave=False):
        sketches = sketches.to(device)
        photos = photos.to(device)
        labels = labels.to(device)

        emb_sketch, emb_photo = model(sketches, photos)
        loss = criterion(emb_sketch, emb_photo, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {"loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def validate(
    model: SiameseNetwork,
    loader: DataLoader,
    criterion: ContrastiveLoss,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for sketches, photos, labels in tqdm(loader, desc="  Val", leave=False):
        sketches = sketches.to(device)
        photos = photos.to(device)
        labels = labels.to(device)

        emb_sketch, emb_photo = model(sketches, photos)
        loss = criterion(emb_sketch, emb_photo, labels)

        total_loss += loss.item()
        n_batches += 1

    return {"loss": total_loss / max(n_batches, 1)}


def main():
    parser = argparse.ArgumentParser(description="Train Siamese sketch-photo matching")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    print("Discovering pairs...")
    pairs = discover_pairs(args.data_root)
    print(f"  Found {len(pairs)} pairs")

    if len(pairs) == 0:
        print("ERROR: No pairs found. Check your data_root path.")
        sys.exit(1)

    splits = split_pairs(pairs, seed=args.seed)
    print(f"  Train: {len(splits['train'])}  Val: {len(splits['val'])}  Test: {len(splits['test'])}")

    # Save splits
    os.makedirs("data/splits", exist_ok=True)
    for name, data in splits.items():
        with open(f"data/splits/{name}.json", "w") as f:
            json.dump(data, f, indent=2)

    train_ds = SiamesePairDataset(splits["train"], train=True, seed=args.seed)
    val_ds = SiamesePairDataset(splits["val"], train=False, seed=args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model
    print("Building model...")
    model = SiameseNetwork(embedding_dim=args.embedding_dim, pretrained=True).to(device)
    criterion = ContrastiveLoss(margin=args.margin)

    # Only optimize parameters that require grad
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"  Trainable params: {trainable_count:,}  Frozen: {frozen_params:,}")

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    history = []

    print(f"\nStarting training for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - t0

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "lr": optimizer.param_groups[0]["lr"],
            "time_s": elapsed,
        }
        history.append(record)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            ckpt_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "args": vars(args),
            }, ckpt_path)
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")

    # Save final model
    final_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "args": vars(args),
    }, final_path)

    # Save history
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
