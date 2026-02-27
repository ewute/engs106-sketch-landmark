"""
CUFS Dataset loader for Siamese sketch-photo matching.

The CUFS dataset has multiple sub-folders. We focus on the cleanest pairing:
  - photos/  (188 CUHK student photos, named like f-005-01.jpg)
  - sketches/ (188 CUHK student sketches, various prefixes)

For the AR+XM2VTS part, we use the photo/ folder (jpg files only).

Since 'photos' and 'sketches' aren't name-matched, we pair them by sorted
order within each sub-database — this is the standard convention for CUFS.
"""

import os
import re
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Utility: discover photo-sketch pairs
# ---------------------------------------------------------------------------

def discover_pairs(data_root: str) -> list[dict]:
    """
    Walk the raw CUFS folders and return a list of dicts:
        {"photo": <abs_path>, "sketch": <abs_path>, "identity": <str>, "source": <str>}
    
    Strategy:
      - photos/ and sketches/ are the CUHK student subset (188 each).
        We sort both and pair them 1:1.
      - photo/ contains AR (prefix m-/w-) and XM2VTS (prefix f-) photos.
        sketch/ contains matching .dat landmark files (not usable as images).
        However, for photo→sketch matching from AR/XM2VTS, we'd need the
        original_sketch or cropped_sketch with a mapping.
      
    For this project we start with the CUHK student subset (cleanest pairing).
    """
    data_root = Path(data_root)
    pairs = []

    # --- CUHK Student subset ---
    photos_dir = data_root / "photos"
    sketches_dir = data_root / "sketches"

    if photos_dir.exists() and sketches_dir.exists():
        photo_files = sorted(
            [f for f in photos_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        )
        sketch_files = sorted(
            [f for f in sketches_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        )

        # Pair by sorted order (standard CUFS convention)
        n = min(len(photo_files), len(sketch_files))
        for i in range(n):
            identity = f"cuhk_{i:04d}"
            pairs.append({
                "photo": str(photo_files[i]),
                "sketch": str(sketch_files[i]),
                "identity": identity,
                "source": "CUHK",
            })

    # --- AR subset (from photo/ folder, prefix m- or w-) ---
    photo_dir = data_root / "photo"
    if photo_dir.exists():
        ar_photos = sorted([
            f for f in photo_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
            and (f.name.lower().startswith("m-") or f.name.lower().startswith("w-"))
        ])
        # For AR, try to find matching sketches in sketch/ as .jpg
        sketch_dir = data_root / "sketch"
        if sketch_dir.exists():
            ar_sketches_jpg = sorted([
                f for f in sketch_dir.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
                and (f.name.lower().startswith("m-") or f.name.lower().startswith("w-"))
            ])
            n = min(len(ar_photos), len(ar_sketches_jpg))
            for i in range(n):
                identity = f"ar_{i:04d}"
                pairs.append({
                    "photo": str(ar_photos[i]),
                    "sketch": str(ar_sketches_jpg[i]),
                    "identity": identity,
                    "source": "AR",
                })

        # --- XM2VTS subset (prefix f- in photo/ folder) ---
        xm2_photos = sorted([
            f for f in photo_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
            and f.name.lower().startswith("f-")
        ])
        if sketch_dir.exists():
            xm2_sketches_jpg = sorted([
                f for f in sketch_dir.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
                and f.name.lower().startswith("f-")
            ])
            n = min(len(xm2_photos), len(xm2_sketches_jpg))
            for i in range(n):
                identity = f"xm2_{i:04d}"
                pairs.append({
                    "photo": str(xm2_photos[i]),
                    "sketch": str(xm2_sketches_jpg[i]),
                    "identity": identity,
                    "source": "XM2VTS",
                })

    return pairs


def split_pairs(
    pairs: list[dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Split pairs into train/val/test by identity.
    Returns {"train": [...], "val": [...], "test": [...]}.
    """
    rng = random.Random(seed)

    # Group by identity (sorted for determinism across Python runs)
    identities = sorted({p["identity"] for p in pairs})
    rng.shuffle(identities)

    n = len(identities)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = set(identities[:n_train])
    val_ids = set(identities[n_train : n_train + n_val])
    test_ids = set(identities[n_train + n_val :])

    splits = {"train": [], "val": [], "test": []}
    for p in pairs:
        if p["identity"] in train_ids:
            splits["train"].append(p)
        elif p["identity"] in val_ids:
            splits["val"].append(p)
        else:
            splits["test"].append(p)

    return splits


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

IMG_SIZE = 224  # standard for most pretrained CNNs

# ImageNet stats — must match what ResNet-50 was pretrained with
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms(train: bool = True):
    """Return image transforms for photos and sketches."""
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ---------------------------------------------------------------------------
# Siamese pair dataset
# ---------------------------------------------------------------------------

class SiamesePairDataset(Dataset):
    """
    Generates positive and negative pairs for contrastive learning.

    For each item:
      - 50% chance: return (sketch, photo) from the SAME identity  → label = 1
      - 50% chance: return (sketch, photo) from DIFFERENT identities → label = 0
    """

    def __init__(self, pairs: list[dict], train: bool = True, seed: int = 42):
        self.pairs = pairs
        self.train = train
        self.transform = get_transforms(train)
        self.seed = seed
        self.epoch = 0
        self.rng = random.Random(seed)

        # Index pairs by identity for efficient sampling
        self.id_to_pairs = {}
        for p in pairs:
            self.id_to_pairs.setdefault(p["identity"], []).append(p)
        self.identities = sorted(self.id_to_pairs.keys())  # sorted for determinism

    def set_epoch(self, epoch: int):
        """Re-seed RNG so each epoch sees different positive/negative pairings."""
        self.epoch = epoch
        self.rng = random.Random(self.seed + epoch)

    def __len__(self):
        # Each epoch: iterate through all pairs once
        return len(self.pairs)

    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        return img

    def __getitem__(self, idx):
        anchor = self.pairs[idx]

        if self.rng.random() < 0.5:
            # Positive pair: same identity
            label = 1
            partner = self.rng.choice(self.id_to_pairs[anchor["identity"]])
        else:
            # Negative pair: different identity
            label = 0
            # Semi-hard: pick from a small random pool and let the loss handle it
            neg_candidates = [i for i in self.identities if i != anchor["identity"]]
            neg_id = self.rng.choice(neg_candidates)
            partner = self.rng.choice(self.id_to_pairs[neg_id])

        sketch = self._load_image(anchor["sketch"])
        photo = self._load_image(partner["photo"])

        sketch = self.transform(sketch)
        photo = self.transform(photo)

        return sketch, photo, label


# ---------------------------------------------------------------------------
# Gallery/query dataset for evaluation (retrieval)
# ---------------------------------------------------------------------------

class RetrievalDataset(Dataset):
    """
    For evaluation: loads all photos or all sketches with their identity labels.
    """

    def __init__(self, pairs: list[dict], mode: str = "photo"):
        assert mode in ("photo", "sketch")
        self.pairs = pairs
        self.mode = mode
        self.transform = get_transforms(train=False)

        # Create integer labels
        all_ids = sorted({p["identity"] for p in pairs})
        self.id_to_label = {id_: i for i, id_ in enumerate(all_ids)}

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx):
        p = self.pairs[idx]
        img = self._load_image(p[self.mode])
        img = self.transform(img)
        label = self.id_to_label[p["identity"]]
        return img, label


# ---------------------------------------------------------------------------
# CLI: discover + split + save
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Discover and split CUFS dataset")
    parser.add_argument("--data-root", default="data/raw", help="Path to raw data")
    parser.add_argument("--output-dir", default="data/splits", help="Where to save split JSONs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = discover_pairs(args.data_root)
    print(f"Discovered {len(pairs)} photo-sketch pairs")

    # Print source breakdown
    from collections import Counter
    source_counts = Counter(p["source"] for p in pairs)
    for src, cnt in sorted(source_counts.items()):
        print(f"  {src}: {cnt}")

    splits = split_pairs(pairs, seed=args.seed)
    for split_name, split_pairs_list in splits.items():
        print(f"  {split_name}: {len(split_pairs_list)} pairs")

    os.makedirs(args.output_dir, exist_ok=True)
    for split_name, split_pairs_list in splits.items():
        path = os.path.join(args.output_dir, f"{split_name}.json")
        with open(path, "w") as f:
            json.dump(split_pairs_list, f, indent=2)
        print(f"  Saved {path}")

    # Also save all pairs
    all_path = os.path.join(args.output_dir, "all_pairs.json")
    with open(all_path, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"  Saved {all_path}")
