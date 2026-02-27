"""
Training runner: batch-hard triplet loss for sketch-photo matching.
"""
import os
import sys
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import discover_pairs, split_pairs, TripletBatchDataset
from src.models.siamese import SiameseNetwork, BatchHardTripletLoss


def main():
    data_root = "data/raw"
    output_dir = "outputs"
    epochs = 100
    batch_size = 32      # larger batches = more identities per batch = better mining
    lr = 3e-4
    embedding_dim = 128  # compact embedding to reduce overfitting
    margin = 0.3
    seed = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Data
    print("Discovering pairs...", flush=True)
    pairs = discover_pairs(data_root)
    print(f"  Found {len(pairs)} pairs", flush=True)

    splits = split_pairs(pairs, seed=seed)
    print(f"  Train: {len(splits['train'])}  Val: {len(splits['val'])}  Test: {len(splits['test'])}", flush=True)

    # Save splits so evaluate.py uses the exact same held-out set
    os.makedirs("data/splits", exist_ok=True)
    for split_name, split_data in splits.items():
        with open(f"data/splits/{split_name}.json", "w") as f:
            json.dump(split_data, f, indent=2)
    with open("data/splits/all_pairs.json", "w") as f:
        json.dump(pairs, f, indent=2)
    print("  Saved splits to data/splits/", flush=True)

    # TripletBatchDataset: each pair contributes sketch + photo = 2 items
    train_ds = TripletBatchDataset(splits["train"], train=True, seed=seed)
    val_ds = TripletBatchDataset(splits["val"], train=False, seed=seed)
    print(f"  Train items: {len(train_ds)}  Val items: {len(val_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=min(batch_size, len(val_ds)), shuffle=False, num_workers=0)

    # Model
    print("Building model...", flush=True)
    model = SiameseNetwork(embedding_dim=embedding_dim, pretrained=True).to(device)
    criterion = BatchHardTripletLoss(margin=margin)

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Trainable: {trainable_count:,}  Frozen: {frozen_count:,}", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=5e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float("inf")
    history = []

    print(f"\nTraining for {epochs} epochs...\n", flush=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_ds.set_epoch(epoch)

        # Train
        model.train()
        train_loss = 0.0
        n = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            embeddings = model.get_embedding(images)
            loss = criterion(embeddings, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n += 1
            if (batch_idx + 1) % 5 == 0:
                print(f"  Epoch {epoch} batch {batch_idx+1}/{len(train_loader)} loss={loss.item():.4f}", flush=True)

        train_loss /= max(n, 1)

        # Val
        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                embeddings = model.get_embedding(images)
                loss = criterion(embeddings, labels)
                val_loss += loss.item()
                n += 1
        val_loss /= max(n, 1)

        scheduler.step()
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr_now:.2e} | {elapsed:.1f}s", flush=True)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "time": elapsed})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "args": {"embedding_dim": embedding_dim},
            }, os.path.join(output_dir, "best_model.pth"))
            print(f"  -> Saved best model (val={best_val_loss:.4f})", flush=True)

    # Save final
    torch.save({"epoch": epochs, "model_state_dict": model.state_dict(), "args": {"embedding_dim": embedding_dim}},
               os.path.join(output_dir, "final_model.pth"))
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone! Best val loss: {best_val_loss:.4f}", flush=True)


if __name__ == "__main__":
    main()
