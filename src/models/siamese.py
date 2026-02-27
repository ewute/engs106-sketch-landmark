"""
Siamese network for sketch-photo matching.

Architecture:
  - Shared backbone: ResNet-50 pretrained on ImageNet.
  - Frozen through layer3; only layer4 + projection head are trainable.
  - Compact projection head to reduce overfitting on small datasets.
  - Supports both contrastive and batch-hard triplet loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EmbeddingNet(nn.Module):
    """
    Feature extractor that maps an image to a fixed-size embedding.
    Uses a pretrained ResNet-50 backbone with the classification head replaced
    by a compact projection layer.
    """

    def __init__(self, embedding_dim: int = 128, pretrained: bool = True):
        super().__init__()

        # Load pretrained ResNet-50
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Remove the final FC layer
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # output: (B, 2048, 1, 1)

        # Compact projection head — small to prevent overfitting on 131 training pairs
        self.projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, embedding_dim),
        )

        # Freeze through layer3; only layer4 + projection are trainable
        self._freeze_early_layers()

    def _freeze_early_layers(self):
        """Freeze conv1, bn1, relu, maxpool, layer1, layer2, layer3.
        Only layer4 (+ avgpool) and the projection head are trainable."""
        children = list(self.features.children())
        # [0]=conv1 [1]=bn1 [2]=relu [3]=maxpool [4]=layer1 [5]=layer2 [6]=layer3 [7]=layer4 [8]=avgpool
        for child in children[:7]:  # freeze through layer3
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.projection(x)
        x = F.normalize(x, p=2, dim=1)  # L2-normalize embeddings
        return x


class SiameseNetwork(nn.Module):
    """
    Siamese network: two inputs processed by the same EmbeddingNet.
    Returns embeddings for both inputs.
    """

    def __init__(self, embedding_dim: int = 128, pretrained: bool = True):
        super().__init__()
        self.embedding_net = EmbeddingNet(embedding_dim, pretrained)

    def forward(self, sketch: torch.Tensor, photo: torch.Tensor):
        emb_sketch = self.embedding_net(sketch)
        emb_photo = self.embedding_net(photo)
        return emb_sketch, emb_photo

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get embedding for a single image."""
        return self.embedding_net(x)


class BatchHardTripletLoss(nn.Module):
    """
    Batch-hard triplet loss (Hermans et al., 2017).

    For each anchor in the batch, finds:
      - hardest positive: the farthest embedding with the same label
      - hardest negative: the closest embedding with a different label
    Then applies: loss = max(0, d(a, p) - d(a, n) + margin)

    This is far more effective than random pair sampling on small datasets
    because every batch yields the most informative gradient signal.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Pairwise distance matrix
        dist = torch.cdist(embeddings, embeddings, p=2)  # (B, B)

        # Masks
        labels = labels.unsqueeze(0)  # (1, B)
        same_id = (labels == labels.T).float()   # (B, B)
        diff_id = 1.0 - same_id

        # For each anchor, hardest positive = max distance among same-id
        # Mask out self (distance 0) and different-id
        pos_dist = dist * same_id
        hardest_pos, _ = pos_dist.max(dim=1)  # (B,)

        # For each anchor, hardest negative = min distance among diff-id
        # Set same-id distances to a large value so they're never picked
        big = dist.max().item() + 1.0
        neg_dist = dist + same_id * big
        hardest_neg, _ = neg_dist.min(dim=1)  # (B,)

        # Triplet loss
        loss = F.relu(hardest_pos - hardest_neg + self.margin)
        # Only count valid triplets (anchors that have both a pos and neg)
        valid = (same_id.sum(dim=1) > 1) & (diff_id.sum(dim=1) > 0)
        if valid.sum() == 0:
            return loss.mean()  # fallback
        return loss[valid].mean()


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss (Hadsell et al., 2006).  Kept for backward compatibility.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        distance = F.pairwise_distance(emb1, emb2)
        label = label.float()

        # Positive pairs: minimize distance
        pos_loss = label * distance.pow(2)
        # Negative pairs: maximize distance (up to margin)
        neg_loss = (1 - label) * F.relu(self.margin - distance).pow(2)

        loss = (pos_loss + neg_loss).mean()
        return loss
