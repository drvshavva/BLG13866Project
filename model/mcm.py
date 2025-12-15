import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional

from .encoder_decoder import Encoder, Decoder
from .mask_generator import MaskGenerator


class MCM(nn.Module):
    """
    MCM: Masked Cell Modeling Main Model

    That combines all components:
    1. Mask Generator: Generates learnable masks
    2. Encoder: Encodes masked data
    3. Decoder: Reconstructs data

    Training:
    - Reconstruction Loss: Reconstruction from masked data
    - Diversity Loss: Mask diversity regularization

    Test:
    - Average reconstruction loss = Anomaly score

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        latent_dim: Latent space dimension
        num_masks: Number of Masks (K)
        temperature: Diversity loss for temperature scaling
        lambda_div: Diversity loss weight
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 256,
            latent_dim: int = 128,
            num_masks: int = 15,
            temperature: float = 0.1,
            lambda_div: float = 20.0
    ):
        super(MCM, self).__init__()

        self.input_dim = input_dim
        self.num_masks = num_masks
        self.temperature = temperature
        self.lambda_div = lambda_div

        # Model components
        self.mask_generator = MaskGenerator(input_dim, hidden_dim, num_masks)
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Operation flow:
        1. Create K masks with Mask Generator
        2. For each mask: x_masked = x * mask (element-wise multiplication)
        3. For each masked data: z = Encoder(x_masked)
        4. For each latent: x_hat = Decoder(z)

        Args:
            x: Input data, shape: (batch_size, input_dim)

        Returns:
            reconstructions: Reconstructions for K masks
            masks: K masks
        """
        # Step 1: Mask generation
        masks = self.mask_generator(x)

        reconstructions = []
        masked_inputs = []

        for mask in masks:
            # Step 2: Element-wise multiplication
            x_masked = x * mask
            masked_inputs.append(x_masked)

            # Step 3 & 4: Encode and Decode
            z = self.encoder(x_masked)
            x_hat = self.decoder(z)

            reconstructions.append(x_hat)

        return reconstructions, masks, masked_inputs

    def compute_reconstruction_loss(
            self,
            x: torch.Tensor,
            reconstructions: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Reconstruction Loss

        L_rec = (1/K) * Σ ||X̂_k - X||²_F
              = (1/NK) * Σ_i Σ_k ||x̂_k^(i) - x^(i)||²_2

        This is the mean squared error loss over K reconstructions.

        Args:
            x: Original input, shape: (batch_size, input_dim)
            reconstructions: reconstruction loss from K masks

        Returns:
            loss: Average reconstruction loss
        """
        total_loss = 0.0
        for x_hat in reconstructions:
            # L2 norm (Frobenius norm for matrices)
            loss = torch.sum((x_hat - x) ** 2, dim=1)  # for each sample in batch
            total_loss += torch.mean(loss)  # average over batch

        # average over K reconstructions
        return total_loss / len(reconstructions)

    def compute_diversity_loss(self, masks: List[torch.Tensor]) -> torch.Tensor:
        """
        Diversity Loss

        L_div = Σ_i [ ln(Σ_j (exp(<M_i, M_j>/τ) * 1_{i≠j})) * scale ]

        This loss encourages diversity among the K masks.

        Inner product is used because it has a lower bound
        and is suitable for optimization.

        Args:
            masks: Mask list

        Returns:
            loss: Diversity loss
        """
        K = len(masks)

        if K <= 1:
            return torch.tensor(0.0, device=masks[0].device)

        # Scale factor
        M_mean = torch.mean(torch.abs(masks[0]))
        scale = 1.0 / (torch.abs(M_mean) * torch.log(1.0 / (M_mean + 1e-8)) + 1e-8)

        div_loss = 0.0

        for i in range(K):
            inner_sum = 0.0
            for j in range(K):
                if i != j:
                    inner_prod = torch.sum(masks[i] * masks[j], dim=1)
                    inner_sum += torch.exp(inner_prod / self.temperature)

            inner_sum = torch.mean(inner_sum)
            if inner_sum > 0:
                div_loss += torch.log(inner_sum + 1e-8) * scale

        return div_loss

    def compute_total_loss(
            self,
            x: torch.Tensor,
            reconstructions: List[torch.Tensor],
            masks: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Total loss function

        L = L_rec + λ * L_div

        Args:
            x: Original input, shape: (batch_size, input_dim)
            reconstructions: Reconstructed outputs from K masks
            masks: Mask list

        Returns:
            total_loss: Total loss
            rec_loss: Reconstruction loss
            div_loss: Diversity loss
        """
        rec_loss = self.compute_reconstruction_loss(x, reconstructions)
        div_loss = self.compute_diversity_loss(masks)

        total_loss = rec_loss + self.lambda_div * div_loss

        return total_loss, rec_loss, div_loss

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly scores for test data

        In test time, reconstruction error averaged over each sample is used as
        the anomaly score.

        High score = anomaly likelihood
        Logic behind anomaly detection:
        - Normal data conforms to learned correlations during training
        - Anomalies deviate from these correlations
        - Deviation leads to high reconstruction error

        Args:
            x: Test data, shape: (batch_size, input_dim)

        Returns:
            scores: Anomaly scores, shape: (batch_size,)
        """
        self.eval()
        with torch.no_grad():
            x_reconstructed, masks, _ = self.forward(x)

            errors = []
            for x_hat in x_reconstructed:
                error = torch.sum((x_hat - x) ** 2, dim=1)
                errors.append(error)

            errors = torch.stack(errors, dim=0)
            anomaly_scores = torch.mean(errors, dim=0)

        return anomaly_scores

    def get_per_mask_contributions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate contribution per mask (Per-Mask Contribution)

        This function calculates the contribution of each mask to anomaly detection.
        High contribution = Deviation from correlation captured by O mask

        Used for interpretability.

        Args:
            model: Trained MCM model
            x: One test sample, shape: (1, input_dim)

        Returns:
            contributions: Contribution percentage for each mask
            errors: Error for each mask
        """
        self.eval()
        with torch.no_grad():
            x_reconstructed, _, _ = self.forward(x)

            contributions = []
            for x_hat in x_reconstructed:
                error = torch.mean((x_hat - x) ** 2, dim=1)
                contributions.append(error)

            contributions = torch.stack(contributions, dim=1)
            total = torch.sum(contributions, dim=1, keepdim=True)
            contributions = contributions / (total + 1e-8)

        return contributions

    def get_per_feature_contributions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate contribution per feature ( Per-Feature Contribution )

        This function, calculates the contribution of each feature to anomaly detection.
        High contribution = O feature deviates from normal distribution
        Used for interpretability.

        Example: In thyroid data, T3 and TSH features may show high contribution.

        Args:
            model: Trained MCM model
            x: Data, shape: (batch_size, input_dim)

        Returns:
            contributions: Contribution percentage of each feature
            errors: Error for each feature
        """
        self.eval()
        with torch.no_grad():
            x_reconstructed, _, _ = self.forward(x)

            all_errors = []
            for x_hat in x_reconstructed:
                error = (x_hat - x) ** 2
                all_errors.append(error)

            all_errors = torch.stack(all_errors, dim=0)
            feature_errors = torch.mean(all_errors, dim=0)

            total = torch.sum(feature_errors, dim=1, keepdim=True)
            contributions = feature_errors / (total + 1e-8)

        return contributions
