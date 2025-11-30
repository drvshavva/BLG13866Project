"""
Loss functions for anomaly detection with diversity-based masking.

Implements reconstruction loss combined with mask diversity regularization
to encourage learning of diverse feature representations.
"""

from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AnomalyDetectionLoss(nn.Module):
    """
    Combined loss function for masked reconstruction-based anomaly detection.

    Computes reconstruction error (MSE) combined with a diversity regularization
    term that encourages different masks to learn diverse representations.

    Loss = MSE + λ * DiversityLoss

    Attributes:
        num_masks: Number of parallel masks/sub-networks
        diversity_weight: Weight coefficient (λ) for diversity loss term
        diversity_loss_fn: Instance of MaskDiversityLoss
    """

    def __init__(
            self,
            model_config: dict,
            diversity_temperature: float = 0.1
    ):
        """
        Initialize the anomaly detection loss function.

        Args:
            model_config: Configuration dictionary containing:
                - mask_num (int): Number of masks/sub-networks
                - lambda (float): Weight for diversity loss term
            diversity_temperature: Temperature parameter for diversity loss (default: 0.1)
        """
        super().__init__()

        # Extract and validate configuration
        self.num_masks = model_config['mask_num']
        self.diversity_weight = model_config.get('lambda', 1.0)

        if self.num_masks < 1:
            raise ValueError(f"num_masks must be positive, got {self.num_masks}")
        if self.diversity_weight < 0:
            raise ValueError(f"diversity_weight must be non-negative, got {self.diversity_weight}")

        # Initialize diversity loss module
        self.diversity_loss_fn = MaskDiversityLoss(temperature=diversity_temperature)

    def forward(
            self,
            x_input: torch.Tensor,
            x_pred: torch.Tensor,
            masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined reconstruction and diversity loss.

        Args:
            x_input: Original input features of shape (batch_size, feature_dim)
            x_pred: Predicted reconstructions of shape (batch_size, num_masks, feature_dim)
            masks: Mask representations of shape (batch_size, num_masks, mask_dim)

        Returns:
            Tuple containing:
                - total_loss: Combined loss (reconstruction + weighted diversity)
                - reconstruction_loss: Mean reconstruction error across all masks
                - diversity_loss: Mean diversity loss across batch

        Example:
            >>> loss_fn = AnomalyDetectionLoss({'mask_num': 5, 'lambda': 0.1})
            >>> x_in = torch.randn(32, 10)
            >>> x_out = torch.randn(32, 5, 10)
            >>> masks = torch.randn(32, 5, 64)
            >>> total, recon, div = loss_fn(x_in, x_out, masks)
        """
        # Expand input to match prediction shape: (batch, 1, features) -> (batch, num_masks, features)
        x_input_expanded = x_input.unsqueeze(1).expand(-1, self.num_masks, -1)

        # Compute reconstruction loss
        reconstruction_loss = self._compute_reconstruction_loss(x_input_expanded, x_pred)

        # Compute diversity loss
        diversity_loss = self.diversity_loss_fn(masks, eval_mode=False)

        # Combine losses
        total_loss = reconstruction_loss + self.diversity_weight * diversity_loss

        return total_loss, reconstruction_loss, diversity_loss

    def _compute_reconstruction_loss(
            self,
            x_target: torch.Tensor,
            x_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute mean squared error reconstruction loss.

        Args:
            x_target: Target features of shape (batch_size, num_masks, feature_dim)
            x_pred: Predicted features of shape (batch_size, num_masks, feature_dim)

        Returns:
            Scalar tensor containing mean reconstruction loss
        """
        # Compute L2 norm of reconstruction error for each mask
        # Shape: (batch_size, num_masks, feature_dim) -> (batch_size, num_masks)
        reconstruction_error = torch.norm(x_pred - x_target, p=2, dim=2)

        # Average across masks, then across batch
        # Shape: (batch_size, num_masks) -> (batch_size, 1) -> scalar
        mse_per_sample = torch.mean(reconstruction_error, dim=1)
        mean_reconstruction_loss = torch.mean(mse_per_sample)

        return mean_reconstruction_loss

    def compute_anomaly_scores(
            self,
            x_input: torch.Tensor,
            x_pred: torch.Tensor,
            masks: torch.Tensor,
            combine_method: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute anomaly scores for evaluation.

        Args:
            x_input: Original input features
            x_pred: Predicted reconstructions
            masks: Mask representations
            combine_method: Method to combine scores ('mean', 'max', 'min', or 'diverse')

        Returns:
            Anomaly scores of shape (batch_size,)
        """
        # Expand input
        x_input_expanded = x_input.unsqueeze(1).expand(-1, self.num_masks, -1)

        # Compute per-mask reconstruction errors
        reconstruction_error = torch.norm(x_pred - x_input_expanded, p=2, dim=2)

        # Combine based on method
        if combine_method == 'mean':
            scores = torch.mean(reconstruction_error, dim=1)
        elif combine_method == 'max':
            scores = torch.max(reconstruction_error, dim=1)[0]
        elif combine_method == 'min':
            scores = torch.min(reconstruction_error, dim=1)[0]
        elif combine_method == 'diverse':
            # Weight by diversity scores
            diversity_scores = self.diversity_loss_fn(masks, eval_mode=True)
            # Normalize diversity scores to [0, 1] for weighting
            diversity_weights = F.softmax(-diversity_scores.unsqueeze(1), dim=1)
            scores = torch.sum(reconstruction_error * diversity_weights, dim=1)
        else:
            raise ValueError(f"Unknown combine_method: {combine_method}")

        return scores


class MaskDiversityLoss(nn.Module):
    """
    Diversity loss to encourage different masks to learn diverse representations.

    Uses contrastive learning principles to maximize dissimilarity between
    different mask representations. Based on normalized temperature-scaled
    cross entropy (NT-Xent) loss.

    The loss encourages masks to be different from each other by penalizing
    high similarity between mask representations.
    """

    def __init__(self, temperature: float = 0.1):
        """
        Initialize the mask diversity loss.

        Args:
            temperature: Temperature parameter for scaling similarities.
                        Lower values make the loss more sensitive to differences.
                        Typical range: 0.05 - 0.5

        Raises:
            ValueError: If temperature is not positive
        """
        super().__init__()

        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")

        self.temperature = temperature

    def forward(
            self,
            mask_representations: torch.Tensor,
            eval_mode: bool = False
    ) -> torch.Tensor:
        """
        Compute diversity loss or diversity scores.

        Args:
            mask_representations: Mask features of shape (batch_size, num_masks, feature_dim)
            eval_mode: If True, return diversity scores for evaluation.
                      If False, return loss for training.

        Returns:
            If eval_mode=False: Loss tensor of shape (batch_size,)
            If eval_mode=True: Diversity scores of shape (batch_size,)

        The diversity score/loss is computed as:
            1. Normalize mask representations to unit sphere
            2. Compute pairwise cosine similarities
            3. Apply temperature scaling and exponential
            4. Sum similarities for each mask (excluding self-similarity)
            5. Take logarithm and apply scaling
        """
        # Normalize representations to unit vectors (L2 normalization)
        # This makes the dot product equivalent to cosine similarity
        normalized_masks = F.normalize(mask_representations, p=2, dim=-1)

        batch_size, num_masks, feature_dim = normalized_masks.shape

        # Compute pairwise similarity matrix
        # Shape: (batch, num_masks, num_masks)
        similarity_matrix = torch.matmul(
            normalized_masks,
            normalized_masks.transpose(1, 2)
        ) / self.temperature

        # Apply exponential (convert to "probability-like" values)
        similarity_matrix = torch.exp(similarity_matrix)

        # Create mask to exclude self-similarity (diagonal elements)
        # Shape: (1, num_masks, num_masks) broadcast to (batch, num_masks, num_masks)
        eye_mask = torch.eye(num_masks, device=normalized_masks.device, dtype=torch.bool)
        eye_mask = eye_mask.unsqueeze(0).expand(batch_size, -1, -1)

        # Invert mask: True for non-diagonal elements
        non_diagonal_mask = ~eye_mask

        # Select only non-diagonal elements and reshape
        # Shape: (batch, num_masks, num_masks-1)
        similarity_matrix_filtered = similarity_matrix.masked_select(
            non_diagonal_mask
        ).view(batch_size, num_masks, num_masks - 1)

        # Sum similarities for each mask (excluding self)
        # Shape: (batch, num_masks)
        total_similarity = similarity_matrix_filtered.sum(dim=-1)

        # Compute scaled logarithmic loss
        # The scaling factor normalizes the loss based on the number of masks
        K = num_masks - 1  # Number of other masks
        scale_factor = 1.0 / np.abs(K * np.log(1.0 / K))

        # Shape: (batch, num_masks)
        diversity_loss_per_mask = torch.log(total_similarity) * scale_factor

        # Sum across masks to get per-sample diversity measure
        # Shape: (batch,)
        diversity_measure = diversity_loss_per_mask.sum(dim=1)

        if eval_mode:
            # Return as scores (higher = less diverse = more anomalous)
            return diversity_measure
        else:
            # Return as loss (to be minimized during training)
            # Mean across batch is taken in the main loss function
            return torch.mean(diversity_measure)


class ReconstructionLoss(nn.Module):
    """
    Standalone reconstruction loss for masked autoencoding.

    Can be used independently or as a component in more complex loss functions.
    """

    def __init__(self, reduction: str = 'mean', norm_type: int = 2):
        """
        Initialize reconstruction loss.

        Args:
            reduction: Reduction method ('mean', 'sum', or 'none')
            norm_type: Norm type for error computation (1 for L1, 2 for L2)
        """
        super().__init__()
        self.reduction = reduction
        self.norm_type = norm_type

    def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.

        Args:
            predictions: Predicted values
            targets: Target values

        Returns:
            Reconstruction loss
        """
        error = torch.norm(predictions - targets, p=self.norm_type, dim=-1)

        if self.reduction == 'mean':
            return torch.mean(error)
        elif self.reduction == 'sum':
            return torch.sum(error)
        elif self.reduction == 'none':
            return error
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
