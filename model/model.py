"""
Multi-Masked Contrastive Model (MCM) for Anomaly Detection.

Implements a masked autoencoder architecture with multiple parallel masks,
where each mask learns to attend to different feature subsets for robust
anomaly detection through diverse reconstructions.
"""

from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mask_nets import MaskGenerator


class MaskedCellModelling(nn.Module):
    """
    Masked Cell Modelling for anomaly detection.

    Architecture:
        1. Mask Generator: Creates N diverse soft masks
        2. Encoder: Maps masked features to latent representations
        3. Decoder: Reconstructs features from latent representations

    Each mask creates a different view of the input, encouraging the model
    to learn diverse and robust feature representations.

    Attributes:
        feature_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        latent_dim: Latent representation dimension
        num_masks: Number of parallel masks
        mask_generator: Module for generating feature masks
        encoder: Encoder network
        decoder: Decoder network
    """

    def __init__(self, model_config: dict):
        """
        Initialize the Multi-Masked Contrastive Model.

        Args:
            model_config: Configuration dictionary containing:
                - data_dim (int): Input feature dimension
                - hidden_dim (int): Hidden layer dimension
                - z_dim (int): Latent space dimension
                - mask_num (int): Number of masks
                - en_nlayers (int): Number of encoder layers
                - de_nlayers (int): Number of decoder layers
                - mask_nlayers (int): Number of layers in mask networks
                - device (str, optional): Device to use
                - dropout (float, optional): Dropout rate
                - use_bias (bool, optional): Whether to use bias in layers
                - activation (str, optional): Activation function

        Example:
            >>> config = {
            ...     'data_dim': 10,
            ...     'hidden_dim': 64,
            ...     'z_dim': 32,
            ...     'mask_num': 5,
            ...     'en_nlayers': 3,
            ...     'de_nlayers': 3,
            ...     'mask_nlayers': 2
            ... }
            >>> model = MultiMaskedContrastiveModel(config)
        """
        super().__init__()

        # Extract and store configuration
        self.feature_dim = model_config['data_dim']
        self.hidden_dim = model_config['hidden_dim']
        self.latent_dim = model_config['z_dim']
        self.num_masks = model_config['mask_num']
        self.num_encoder_layers = model_config['en_nlayers']
        self.num_decoder_layers = model_config['de_nlayers']

        # Optional parameters
        self.dropout = model_config.get('dropout', 0.0)
        self.use_bias = model_config.get('use_bias', False)
        self.activation = model_config.get('activation', 'leaky_relu')
        self.device = model_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        # Validate configuration
        self._validate_config()

        # Build model components
        self.mask_generator = self._build_mask_generator(model_config)
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

        # Move to device
        self.to(self.device)

        # Initialize weights
        self.apply(self._init_weights)

    def _validate_config(self):
        """Validate model configuration parameters."""
        if self.feature_dim < 1:
            raise ValueError(f"feature_dim must be positive, got {self.feature_dim}")
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.latent_dim < 1:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")
        if self.num_masks < 1:
            raise ValueError(f"num_masks must be positive, got {self.num_masks}")
        if self.num_encoder_layers < 1:
            raise ValueError(f"num_encoder_layers must be positive, got {self.num_encoder_layers}")
        if self.num_decoder_layers < 1:
            raise ValueError(f"num_decoder_layers must be positive, got {self.num_decoder_layers}")

    def _build_mask_generator(self, config: dict) -> MaskGenerator:
        """
        Build the mask generator module.

        Args:
            config: Model configuration dictionary

        Returns:
            Initialized MaskGenerator
        """
        return MaskGenerator(
            feature_dim=self.feature_dim,
            num_masks=self.num_masks,
            hidden_dim=self.feature_dim,
            num_layers=config.get('mask_nlayers', 2),
            activation='sigmoid',
            use_bias=self.use_bias,
            dropout=self.dropout,
            device=torch.device(self.device)
        )

    def _build_encoder(self) -> nn.Sequential:
        """
        Build the encoder network.

        Maps input features to latent representations through multiple layers.

        Returns:
            Sequential encoder module
        """
        return self._build_mlp(
            input_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.latent_dim,
            num_layers=self.num_encoder_layers,
            activation=self.activation,
            use_bias=self.use_bias,
            dropout=self.dropout,
            output_activation=False
        )

    def _build_decoder(self) -> nn.Sequential:
        """
        Build the decoder network.

        Reconstructs features from latent representations through multiple layers.

        Returns:
            Sequential decoder module
        """
        return self._build_mlp(
            input_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.feature_dim,
            num_layers=self.num_decoder_layers,
            activation=self.activation,
            use_bias=self.use_bias,
            dropout=self.dropout,
            output_activation=False
        )

    @staticmethod
    def _build_mlp(
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            activation: str = 'leaky_relu',
            use_bias: bool = False,
            dropout: float = 0.0,
            output_activation: bool = False
    ) -> nn.Sequential:
        """
        Build a multi-layer perceptron.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of layers
            activation: Activation function name
            use_bias: Whether to use bias
            dropout: Dropout rate
            output_activation: Whether to apply activation to output

        Returns:
            Sequential MLP module
        """
        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}")

        # Get activation function
        activation_fn = MaskedCellModelling._get_activation(activation)

        layers = []
        current_dim = input_dim

        # Hidden layers
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim, bias=use_bias))
            layers.append(activation_fn)

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            current_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(current_dim, output_dim, bias=use_bias))

        if output_activation:
            layers.append(activation_fn)

        return nn.Sequential(*layers)

    @staticmethod
    def _get_activation(activation: str) -> nn.Module:
        """Get activation function module."""
        activations = {
            'relu': nn.ReLU(inplace=True),
            'leaky_relu': nn.LeakyReLU(0.2, inplace=True),
            'elu': nn.ELU(inplace=True),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }

        if activation not in activations:
            raise ValueError(
                f"Unknown activation: {activation}. "
                f"Choose from {list(activations.keys())}"
            )

        return activations[activation]

    @staticmethod
    def _init_weights(module: nn.Module):
        """Initialize weights using Xavier uniform initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
            self,
            x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x: Input features of shape (batch_size, feature_dim)

        Returns:
            Tuple containing:
                - reconstructions: Shape (batch_size, num_masks, feature_dim)
                                  Reconstructed features for each mask
                - latent_codes: Shape (batch_size, num_masks, latent_dim)
                               Latent representations for each mask
                - raw_masks: Shape (batch_size, num_masks, feature_dim)
                            Raw mask values before activation

        Example:
            >>> model = MultiMaskedContrastiveModel(config)
            >>> x = torch.randn(32, 10)
            >>> recon, latent, masks = model(x)
            >>> print(recon.shape, latent.shape, masks.shape)
            torch.Size([32, 5, 10]) torch.Size([32, 5, 32]) torch.Size([32, 5, 10])
        """
        # Generate masked features
        # x_masked: (batch_size, num_masks, feature_dim)
        # raw_masks: (batch_size, num_masks, feature_dim)
        x_masked, raw_masks = self.mask_generator(x, return_masks=True)

        batch_size, num_masks, feature_dim = x_masked.shape

        # Reshape for batch processing through encoder/decoder
        # (batch_size, num_masks, feature_dim) -> (batch_size * num_masks, feature_dim)
        x_masked_flat = x_masked.reshape(batch_size * num_masks, feature_dim)

        # Encode to latent space
        latent_flat = self.encoder(x_masked_flat)

        # Decode back to feature space
        reconstructions_flat = self.decoder(latent_flat)

        # Reshape back to (batch_size, num_masks, dim)
        latent_codes = latent_flat.reshape(batch_size, num_masks, self.latent_dim)
        reconstructions = reconstructions_flat.reshape(batch_size, num_masks, feature_dim)

        return reconstructions, latent_codes, raw_masks

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent space using all masks.

        Args:
            x: Input features of shape (batch_size, feature_dim)

        Returns:
            Latent codes of shape (batch_size, num_masks, latent_dim)
        """
        x_masked, _ = self.mask_generator(x, return_masks=False)
        batch_size, num_masks, feature_dim = x_masked.shape

        x_masked_flat = x_masked.reshape(batch_size * num_masks, feature_dim)
        latent_flat = self.encoder(x_masked_flat)
        latent_codes = latent_flat.reshape(batch_size, num_masks, self.latent_dim)

        return latent_codes

    def decode(self, latent_codes: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representations to feature space.

        Args:
            latent_codes: Latent codes of shape (batch_size, num_masks, latent_dim)
                         or (batch_size * num_masks, latent_dim)

        Returns:
            Reconstructions of shape (batch_size, num_masks, feature_dim)
        """
        original_shape = latent_codes.shape

        if len(original_shape) == 3:
            batch_size, num_masks, latent_dim = original_shape
            latent_flat = latent_codes.reshape(batch_size * num_masks, latent_dim)
        else:
            latent_flat = latent_codes
            batch_size = latent_flat.shape[0] // self.num_masks
            num_masks = self.num_masks

        reconstructions_flat = self.decoder(latent_flat)
        reconstructions = reconstructions_flat.reshape(batch_size, num_masks, self.feature_dim)

        return reconstructions

    def get_anomaly_scores(
            self,
            x: torch.Tensor,
            method: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute anomaly scores for input samples.

        Args:
            x: Input features of shape (batch_size, feature_dim)
            method: Scoring method - 'mean', 'max', 'min', or 'weighted'

        Returns:
            Anomaly scores of shape (batch_size,)

        Example:
            >>> model = MultiMaskedContrastiveModel(config)
            >>> x = torch.randn(32, 10)
            >>> scores = model.get_anomaly_scores(x, method='mean')
            >>> print(scores.shape)
            torch.Size([32])
        """
        with torch.no_grad():
            reconstructions, _, _ = self.forward(x)

            # Compute reconstruction error for each mask
            x_expanded = x.unsqueeze(1).expand(-1, self.num_masks, -1)
            errors = torch.norm(reconstructions - x_expanded, p=2, dim=2)

            # Combine errors based on method
            if method == 'mean':
                scores = torch.mean(errors, dim=1)
            elif method == 'max':
                scores = torch.max(errors, dim=1)[0]
            elif method == 'min':
                scores = torch.min(errors, dim=1)[0]
            elif method == 'weighted':
                # Weight by mask diversity
                weights = F.softmax(-errors, dim=1)
                scores = torch.sum(errors * weights, dim=1)
            else:
                raise ValueError(f"Unknown scoring method: {method}")

            return scores

    def get_mask_statistics(self, x: torch.Tensor) -> dict:
        """
        Get statistics about mask behavior on input data.

        Args:
            x: Input features of shape (batch_size, feature_dim)

        Returns:
            Dictionary containing mask statistics
        """
        with torch.no_grad():
            mask_probs = self.mask_generator.get_mask_probabilities(x)

            stats = {
                'mean': mask_probs.mean(dim=(0, 2)),  # Mean per mask
                'std': mask_probs.std(dim=(0, 2)),  # Std per mask
                'sparsity': (mask_probs < 0.5).float().mean(dim=(0, 2)),  # Sparsity per mask
                'diversity': self.mask_generator.get_mask_diversity(x).mean().item()
            }

            return stats

    def get_latent_statistics(self, x: torch.Tensor) -> dict:
        """
        Get statistics about latent representations.

        Args:
            x: Input features of shape (batch_size, feature_dim)

        Returns:
            Dictionary containing latent space statistics
        """
        with torch.no_grad():
            latent_codes = self.encode(x)

            stats = {
                'mean': latent_codes.mean(dim=(0, 1)),
                'std': latent_codes.std(dim=(0, 1)),
                'norm': torch.norm(latent_codes, p=2, dim=2).mean(dim=1)
            }

            return stats


