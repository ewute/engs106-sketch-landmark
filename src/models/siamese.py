"""
Siamese network for sketch-photo matching.

Architecture:
  - Shared backbone: ResNet-50 (we use this instead of VGGFace2 since it's
    available out-of-the-box in torchvision; a face-pretrained model can be
    swapped in later).
  - The backbone produces an embedding for each image.
  - Contrastive loss pulls matching pairs close and pushes non-matching apart.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EmbeddingNet(nn.Module):
    """
    Feature extractor that maps an image to a fixed-size embedding.
    Uses a pretrained ResNet-50 backbone with the classification head replaced
    by a projection layer.
    """

    def __init__(self, embedding_dim: int = 256, pretrained: bool = True):
        super().__init__()

        # Load pretrained ResNet-50
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Remove the final FC layer
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # output: (B, 2048, 1, 1)

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, embedding_dim),
        )

        # Freeze early layers (first ~6 blocks), fine-tune later layers
        self._freeze_early_layers()

    def _freeze_early_layers(self):
        """Freeze only conv1, bn1, relu, maxpool, and layer1 (first 5 children).
        This lets layer2, layer3, and layer4 adapt to the sketch-photo domain."""
        children = list(self.features.children())
        # children[0]=conv1, [1]=bn1, [2]=relu, [3]=maxpool, [4]=layer1, [5]=layer2, [6]=layer3, [7]=layer4
        for child in children[:5]:  # freeze through layer1 only
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

    def __init__(self, embedding_dim: int = 256, pretrained: bool = True):
        super().__init__()
        self.embedding_net = EmbeddingNet(embedding_dim, pretrained)

    def forward(self, sketch: torch.Tensor, photo: torch.Tensor):
        emb_sketch = self.embedding_net(sketch)
        emb_photo = self.embedding_net(photo)
        return emb_sketch, emb_photo

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get embedding for a single image."""
        return self.embedding_net(x)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss (Hadsell et al., 2006).
    
    For matching pairs (label=1): loss = d^2
    For non-matching pairs (label=0): loss = max(0, margin - d)^2
    
    where d = euclidean distance between embeddings.
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
