"""Loss prediction module.

Reference:
Yoo et al. 2019, Learning Loss for Active Learning
https://arxiv.org/abs/1905.03677
"""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LossNet(nn.Module):
    """Predict per-sample training loss from multi-scale backbone features."""

    def __init__(
        self,
        num_channels: Sequence[int] = (64, 128, 256, 512),
        interm_dim: int = 128,
    ):
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Linear(channels, interm_dim) for channels in num_channels
        )
        self.regressor = nn.Linear(len(num_channels) * interm_dim, 1)

    def forward(self, features):
        if len(features) != len(self.projections):
            raise ValueError(
                f"Expected {len(self.projections)} feature maps, got {len(features)}."
            )

        pooled_features = []
        for feature, projection in zip(features, self.projections):
            pooled = F.adaptive_avg_pool2d(feature, output_size=1).flatten(1)
            pooled_features.append(F.relu(projection(pooled)))

        return self.regressor(torch.cat(pooled_features, dim=1))
