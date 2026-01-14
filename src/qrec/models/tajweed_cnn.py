import torch
import torch.nn as nn
import torch.nn.functional as F


class TajweedCNN(nn.Module):
    """
    Segment-level audio classifier: (B, n_mels, T) -> logits (B, num_classes).
    For binary rule correctness: num_classes=2.
    """
    def __init__(self, n_mels: int = 80, base_channels: int = 64, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(c, 2*c, 3, padding=1), nn.BatchNorm2d(2*c), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(2*c, 4*c, 3, padding=1), nn.BatchNorm2d(4*c), nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(4*c, num_classes)

    def forward(self, mels: torch.Tensor):
        # mels: (B, n_mels, T)
        x = mels.unsqueeze(1)      # (B,1,n_mels,T)
        x = self.net(x)            # (B, 4c, n_mels', T')
        x = x.mean(dim=[2, 3])     # global average pool -> (B, 4c)
        x = self.dropout(x)
        return self.head(x)        # (B, num_classes)
