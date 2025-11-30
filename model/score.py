"""
Anomaly scoring functions for multi-masked reconstruction models.

Provides various methods to compute anomaly scores based on reconstruction
errors from multiple mask networks.
"""

from typing import Optional, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AnomalyScorer(nn.Module):
    """
    Anomaly scoring module for multi-masked reconstruction models.

    Computes anomaly scores by measuring reconstruction errors between
    original inputs and predictions from multiple mask networks.

    Attributes:
        num_masks: Number of mask networks
        aggregation_method: Method to aggregate scores across masks
        norm_type: Type of norm for error computation (1 or 2)
    """

    def __init__(
            self,
            model_config: dict,
            aggregation_method: Literal['mean', 'max', 'min', 'median', 'weighted'] = 'mean',
            norm_type: int = 2,
            return_per_mask: bool = False
    ):
        """
        Initialize the anomaly scorer.

        Args:
            model_config: Configuration dictionary containing:
                - mask_num (int): Number of mask networks
            aggregation_method: Method to combine scores from multiple masks
                - 'mean': Average reconstruction error across masks
                - 'max': Maximum reconstruction error
                - 'min': Minimum reconstruction error
                - 'median': Median reconstruction error
                - 'weighted': Weighted average (lower errors get higher weights)
            norm_type: Norm type for error computation (1 for L1, 2 for L2/Euclidean)
            return_per_mask: If True, also return per-mask scores

        Example:
            >>> config = {'mask_num': 5}
            >>> scorer = AnomalyScorer(config, aggregation_method='mean')
        """
        super().__init__()

        # Extract configuration
        self.num_masks = model_config['mask_num']
        self.aggregation_method = aggregation_method
        self.norm_type = norm_type
        self.return_per_mask = return_per_mask

        # Validate parameters
        if self.num_masks < 1:
            raise ValueError(f"num_masks must be positive, got {self.num_masks}")

        if norm_type not in [1, 2]:
            raise ValueError(f"norm_type must be 1 or 2, got {norm_type}")

        valid_methods = ['mean', 'max', 'min', 'median', 'weighted']
        if aggregation_method not in valid_methods:
            raise ValueError(
                f"aggregation_method must be one of {valid_methods}, "
                f"got {aggregation_method}"
            )

    def forward(
            self,
            x_input: torch.Tensor,
            x_pred: torch.Tensor,
            mask_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute anomaly scores based on reconstruction errors.

        Args:
            x_input: Original input features of shape (batch_size, feature_dim)
            x_pred: Reconstructed predictions of shape (batch_size, num_masks, feature_dim)
            mask_weights: Optional weights for each mask of shape (batch_size, num_masks)
                         Used only when aggregation_method='weighted'

        Returns:
            If return_per_mask=False:
                Anomaly scores of shape (batch_size, 1)
            If return_per_mask=True:
                Tuple of (aggregated_scores, per_mask_scores)
                where per_mask_scores has shape (batch_size, num_masks)

        Example:
            >>> scorer = AnomalyScorer({'mask_num': 5})
            >>> x_in = torch.randn(32, 10)
            >>> x_out = torch.randn(32, 5, 10)
            >>> scores = scorer(x_in, x_out)
            >>> print(scores.shape)
            torch.Size([32, 1])
        """
        # Validate input shapes
        if x_input.dim() != 2:
            raise ValueError(f"x_input must be 2D, got shape {x_input.shape}")

        if x_pred.dim() != 3:
            raise ValueError(f"x_pred must be 3D, got shape {x_pred.shape}")

        if x_pred.shape[1] != self.num_masks:
            raise ValueError(
                f"x_pred second dimension must be {self.num_masks}, "
                f"got {x_pred.shape[1]}"
            )

        # Compute per-mask reconstruction errors
        per_mask_scores = self._compute_reconstruction_errors(x_input, x_pred)

        # Aggregate scores across masks
        aggregated_scores = self._aggregate_scores(per_mask_scores, mask_weights)

        if self.return_per_mask:
            return aggregated_scores, per_mask_scores
        else:
            return aggregated_scores

    def _compute_reconstruction_errors(
            self,
            x_input: torch.Tensor,
            x_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction error for each mask.

        Args:
            x_input: Original inputs of shape (batch_size, feature_dim)
            x_pred: Predictions of shape (batch_size, num_masks, feature_dim)

        Returns:
            Per-mask errors of shape (batch_size, num_masks)
        """
        # Expand input to match prediction shape
        # (batch_size, feature_dim) -> (batch_size, num_masks, feature_dim)
        x_input_expanded = x_input.unsqueeze(1).expand(-1, self.num_masks, -1)

        # Compute reconstruction error
        reconstruction_error = x_pred - x_input_expanded

        # Compute norm across feature dimension
        # Shape: (batch_size, num_masks, feature_dim) -> (batch_size, num_masks)
        per_mask_scores = torch.norm(reconstruction_error, p=self.norm_type, dim=2)

        return per_mask_scores

    def _aggregate_scores(
            self,
            per_mask_scores: torch.Tensor,
            mask_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Aggregate per-mask scores into final anomaly scores.

        Args:
            per_mask_scores: Scores for each mask of shape (batch_size, num_masks)
            mask_weights: Optional weights of shape (batch_size, num_masks)

        Returns:
            Aggregated scores of shape (batch_size, 1)
        """
        if self.aggregation_method == 'mean':
            scores = torch.mean(per_mask_scores, dim=1, keepdim=True)

        elif self.aggregation_method == 'max':
            scores = torch.max(per_mask_scores, dim=1, keepdim=True)[0]

        elif self.aggregation_method == 'min':
            scores = torch.min(per_mask_scores, dim=1, keepdim=True)[0]

        elif self.aggregation_method == 'median':
            scores = torch.median(per_mask_scores, dim=1, keepdim=True)[0]

        elif self.aggregation_method == 'weighted':
            if mask_weights is None:
                # Use softmax of negative errors as weights
                # Lower error -> higher weight
                weights = F.softmax(-per_mask_scores, dim=1)
            else:
                # Normalize provided weights
                weights = mask_weights / (mask_weights.sum(dim=1, keepdim=True) + 1e-8)

            scores = torch.sum(per_mask_scores * weights, dim=1, keepdim=True)

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

        return scores

    def compute_feature_wise_errors(
            self,
            x_input: torch.Tensor,
            x_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction errors for each feature dimension.

        Useful for identifying which features contribute most to anomaly scores.

        Args:
            x_input: Original inputs of shape (batch_size, feature_dim)
            x_pred: Predictions of shape (batch_size, num_masks, feature_dim)

        Returns:
            Feature-wise errors of shape (batch_size, num_masks, feature_dim)
        """
        x_input_expanded = x_input.unsqueeze(1).expand(-1, self.num_masks, -1)
        feature_wise_errors = torch.abs(x_pred - x_input_expanded)

        return feature_wise_errors

    def get_top_anomalous_features(
            self,
            x_input: torch.Tensor,
            x_pred: torch.Tensor,
            top_k: int = 5
    ) -> torch.Tensor:
        """
        Identify the top-k most anomalous features for each sample.

        Args:
            x_input: Original inputs of shape (batch_size, feature_dim)
            x_pred: Predictions of shape (batch_size, num_masks, feature_dim)
            top_k: Number of top features to return

        Returns:
            Indices of top-k anomalous features of shape (batch_size, top_k)
        """
        # Compute feature-wise errors
        feature_errors = self.compute_feature_wise_errors(x_input, x_pred)

        # Average across masks
        avg_feature_errors = feature_errors.mean(dim=1)  # (batch_size, feature_dim)

        # Get top-k indices
        top_k_indices = torch.topk(avg_feature_errors, k=top_k, dim=1)[1]

        return top_k_indices


class MultiScaleScorer(nn.Module):
    """
    Multi-scale anomaly scorer that combines scores at different granularities.

    Combines:
    1. Global reconstruction error (entire sample)
    2. Local reconstruction errors (per feature)
    3. Mask diversity scores
    """

    def __init__(
            self,
            model_config: dict,
            global_weight: float = 0.5,
            local_weight: float = 0.3,
            diversity_weight: float = 0.2,
            norm_type: int = 2
    ):
        """
        Initialize multi-scale scorer.

        Args:
            model_config: Configuration dictionary
            global_weight: Weight for global reconstruction error
            local_weight: Weight for local (feature-wise) errors
            diversity_weight: Weight for mask diversity score
            norm_type: Norm type for error computation
        """
        super().__init__()

        self.num_masks = model_config['mask_num']
        self.global_weight = global_weight
        self.local_weight = local_weight
        self.diversity_weight = diversity_weight
        self.norm_type = norm_type

        # Validate weights sum to 1
        total_weight = global_weight + local_weight + diversity_weight
        if not np.isclose(total_weight, 1.0):
            raise ValueError(
                f"Weights must sum to 1.0, got {total_weight}. "
                f"Adjust global_weight, local_weight, and diversity_weight."
            )

    def forward(
            self,
            x_input: torch.Tensor,
            x_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute multi-scale anomaly scores.

        Args:
            x_input: Original inputs of shape (batch_size, feature_dim)
            x_pred: Predictions of shape (batch_size, num_masks, feature_dim)

        Returns:
            Multi-scale anomaly scores of shape (batch_size, 1)
        """
        # Global reconstruction error (averaged across masks and features)
        x_input_expanded = x_input.unsqueeze(1).expand(-1, self.num_masks, -1)
        reconstruction_error = x_pred - x_input_expanded
        global_scores = torch.norm(reconstruction_error, p=self.norm_type, dim=2).mean(dim=1)

        # Local (feature-wise) reconstruction error variance
        # High variance indicates inconsistent reconstruction across features
        feature_errors = torch.abs(reconstruction_error).mean(dim=1)  # (batch, features)
        local_scores = feature_errors.std(dim=1)

        # Mask diversity score
        # High variance across masks indicates disagreement (potential anomaly)
        mask_errors = torch.norm(reconstruction_error, p=self.norm_type, dim=2)
        diversity_scores = mask_errors.std(dim=1)

        # Normalize scores to [0, 1] range
        global_scores = self._normalize(global_scores)
        local_scores = self._normalize(local_scores)
        diversity_scores = self._normalize(diversity_scores)

        # Weighted combination
        combined_scores = (
                self.global_weight * global_scores +
                self.local_weight * local_scores +
                self.diversity_weight * diversity_scores
        )

        return combined_scores.unsqueeze(1)

    @staticmethod
    def _normalize(scores: torch.Tensor) -> torch.Tensor:
        """Normalize scores to [0, 1] range using min-max scaling."""
        min_val = scores.min()
        max_val = scores.max()

        if max_val - min_val < 1e-8:
            return torch.zeros_like(scores)

        return (scores - min_val) / (max_val - min_val + 1e-8)
