"""
Quick training runner that logs to file (avoids tqdm issues in PowerShell).
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

from src.data.dataset import discover_pairs, split_pairs, SiamesePairDataset
from src.models.siamese import SiameseNetwork, ContrastiveLoss


def main():
    data_root = "data/raw"
    output_dir = "outputs"
    epochs = 5
    batch_size = 8
    lr = 1e-4
    seed = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Data
    print("Discovering pairs...", flush=True)
    pairs = discover_pairs(data_root)
    print(f"  Found {len(pairs)} pairs", flush=True)

    splits = split_pairs(pairs, seed=seed)
    print(f"  Train: {len(splits['train'])}  Val: {len(splits['val'])}  Test: {len(splits['test'])}", flush=True)

    train_ds = SiamesePairDataset(splits["train"], train=True, seed=seed)
    val_ds = SiamesePairDataset(splits["val"], train=False, seed=seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    print("Building model (downloading ResNet-50 weights)...", flush=True)
    model = SiameseNetwork(embedding_dim=256, pretrained=True).to(device)
    criterion = ContrastiveLoss(margin=1.0)

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Trainable: {trainable_count:,}  Frozen: {frozen_count:,}", flush=True)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-5
    )

    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float("inf")
    history = []

    print(f"\nTraining for {epochs} epochs...\n", flush=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        n = 0
        for batch_idx, (sketches, photos, labels) in enumerate(train_loader):
            sketches, photos, labels = sketches.to(device), photos.to(device), labels.to(device)
            emb_s, emb_p = model(sketches, photos)
            loss = criterion(emb_s, emb_p, labels)
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
            for sketches, photos, labels in val_loader:
                sketches, photos, labels = sketches.to(device), photos.to(device), labels.to(device)
                emb_s, emb_p = model(sketches, photos)
                loss = criterion(emb_s, emb_p, labels)
                val_loss += loss.item()
                n += 1
        val_loss /= max(n, 1)

        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | {elapsed:.1f}s", flush=True)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "time": elapsed})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "args": {"embedding_dim": 256},
            }, os.path.join(output_dir, "best_model.pth"))
            print(f"  -> Saved best model (val={best_val_loss:.4f})", flush=True)

    # Save final
    torch.save({"epoch": epochs, "model_state_dict": model.state_dict(), "args": {"embedding_dim": 256}},
               os.path.join(output_dir, "final_model.pth"))
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone! Best val loss: {best_val_loss:.4f}", flush=True)


if __name__ == "__main__":
    main()
