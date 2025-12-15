import torch
import torch.nn as nn
from typing import List

class MaskGenerator(nn.Module):
    """
    Mask Generator Module

    This module generates learnable masks from input data.
    Each mask determines which features to mask.

    Features:
    - Generates K different masks (ensemble learning)
    - Soft masks between 0-1 using sigmoid activation
    - Data-dependent masks (different mask for each input)

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        num_masks: Number of masks to generate (K)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_masks: int = 15):
        super(MaskGenerator, self).__init__()

        self.num_masks = num_masks
        self.input_dim = input_dim

        # For each mask, create a separate feature extractor
        # Study: "We employ a mask generator G, which consists of a
        # feature extractor F attached by a sigmoid function"
        self.mask_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, input_dim),
                nn.Sigmoid()  # Soft mask between 0-1
            )
            for _ in range(num_masks)
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Creates K masks from input data.

        Args:
            x: Input data, shape: (batch_size, input_dim)

        Returns:
            masks: K mask list, each shape: (batch_size, input_dim)
        """
        masks = []
        for extractor in self.mask_extractors:
            mask = extractor(x)
            masks.append(mask)
        return masks
