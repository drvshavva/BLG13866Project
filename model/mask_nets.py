"""
Multi-mask generator architecture for anomaly detection.

Implements a generator that creates multiple learnable masks, each focusing on
different aspects of the input features. Each mask network learns to selectively
attend to relevant features for reconstruction.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskGenerator(nn.Module):
    """
    Multi-mask generator for feature-level attention in anomaly detection.

    Creates multiple parallel mask networks, each learning to generate soft masks
    that highlight different feature subsets. The masked features are then used
    for reconstruction-based anomaly detection.

    Attributes:
        num_masks: Number of parallel mask networks
        feature_dim: Dimensionality of input features
        mask_networks: ModuleList containing individual mask networks
        activation: Activation function for masks (default: sigmoid)
    """

    def __init__(
            self,
            feature_dim: int,
            num_masks: int,
            hidden_dim: Optional[int] = None,
            num_layers: int = 3,
            activation: str = 'sigmoid',
            use_bias: bool = False,
            dropout: float = 0.0,
            device: Optional[torch.device] = None
    ):
        """
        Initialize the mask generator.

        Args:
            feature_dim: Dimensionality of input features
            num_masks: Number of parallel mask networks to create
            hidden_dim: Hidden layer dimension (default: same as feature_dim)
            num_layers: Number of layers in each mask network
            activation: Activation function for masks ('sigmoid', 'softmax', 'tanh')
            use_bias: Whether to use bias in linear layers
            dropout: Dropout rate (0 for no dropout)
            device: Device to place the model on

        Raises:
            ValueError: If invalid parameters are provided
        """
        super().__init__()

        # Validate parameters
        if feature_dim < 1:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if num_masks < 1:
            raise ValueError(f"num_masks must be positive, got {num_masks}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.feature_dim = feature_dim
        self.num_masks = num_masks
        self.hidden_dim = hidden_dim or feature_dim
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.dropout = dropout

        # Set activation function for masks
        self.activation = self._get_activation(activation)

        # Create mask networks
        self.mask_networks = self._create_mask_networks()

        # Set device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        # Initialize weights
        self._initialize_weights()

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function for masks."""
        activations = {
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'softmax': nn.Softmax(dim=-1),
            'relu': nn.ReLU()
        }

        if activation not in activations:
            raise ValueError(
                f"Unknown activation: {activation}. "
                f"Choose from {list(activations.keys())}"
            )

        return activations[activation]

    def _create_mask_networks(self) -> nn.ModuleList:
        """
        Create multiple parallel mask networks.

        Returns:
            ModuleList containing individual mask networks
        """
        mask_networks = nn.ModuleList([
            MaskNetwork(
                input_dim=self.feature_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.feature_dim,
                num_layers=self.num_layers,
                use_bias=self.use_bias,
                dropout=self.dropout
            )
            for _ in range(self.num_masks)
        ])

        return mask_networks

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for network in self.mask_networks:
            network.apply(self._init_weights_fn)

    @staticmethod
    def _init_weights_fn(module):
        """Weight initialization function."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
            self,
            x: torch.Tensor,
            return_masks: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate masked features using multiple mask networks.

        Args:
            x: Input features of shape (batch_size, feature_dim)
            return_masks: Whether to return raw mask values (before activation)

        Returns:
            Tuple containing:
                - masked_features: Tensor of shape (batch_size, num_masks, feature_dim)
                                  Each slice contains features masked by one network
                - raw_masks: If return_masks=True, tensor of shape (batch_size, num_masks, feature_dim)
                           containing mask logits before activation. None otherwise.

        Example:
            >>> generator = MaskGenerator(feature_dim=10, num_masks=5)
            >>> x = torch.randn(32, 10)
            >>> masked_features, masks = generator(x)
            >>> print(masked_features.shape)  # (32, 5, 10)
        """
        # Ensure correct dtype and device
        x = x.float().to(self.device)

        batch_size = x.shape[0]

        # Pre-allocate tensors for efficiency
        masked_features = torch.empty(
            batch_size, self.num_masks, self.feature_dim,
            dtype=x.dtype, device=x.device
        )

        if return_masks:
            raw_masks_list = []

        # Generate masks and apply them
        for i, mask_network in enumerate(self.mask_networks):
            # Generate mask logits
            mask_logits = mask_network(x)

            if return_masks:
                raw_masks_list.append(mask_logits.unsqueeze(1))

            # Apply activation to get soft mask
            mask_probs = self.activation(mask_logits)

            # Apply mask to input features
            masked_features[:, i] = mask_probs * x

        # Concatenate raw masks if requested
        raw_masks = torch.cat(raw_masks_list, dim=1) if return_masks else None

        return masked_features, raw_masks

    def get_mask_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get mask probabilities without applying them to input.

        Args:
            x: Input features of shape (batch_size, feature_dim)

        Returns:
            Mask probabilities of shape (batch_size, num_masks, feature_dim)
        """
        x = x.float().to(self.device)
        batch_size = x.shape[0]

        mask_probs = torch.empty(
            batch_size, self.num_masks, self.feature_dim,
            dtype=x.dtype, device=x.device
        )

        for i, mask_network in enumerate(self.mask_networks):
            mask_logits = mask_network(x)
            mask_probs[:, i] = self.activation(mask_logits)

        return mask_probs

    def get_mask_diversity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute diversity score for masks (how different they are).

        Higher diversity means masks are attending to different features.

        Args:
            x: Input features of shape (batch_size, feature_dim)

        Returns:
            Diversity scores of shape (batch_size,)
        """
        mask_probs = self.get_mask_probabilities(x)

        # Compute pairwise cosine similarity between masks
        # Normalize masks
        mask_probs_flat = mask_probs.view(mask_probs.shape[0], self.num_masks, -1)
        normalized_masks = F.normalize(mask_probs_flat, p=2, dim=-1)

        # Compute similarity matrix
        similarity = torch.bmm(normalized_masks, normalized_masks.transpose(1, 2))

        # Remove diagonal (self-similarity)
        mask_idx = torch.eye(self.num_masks, device=x.device, dtype=torch.bool)
        similarity = similarity.masked_fill(mask_idx.unsqueeze(0), 0)

        # Average similarity (lower is more diverse)
        avg_similarity = similarity.abs().sum(dim=(1, 2)) / (self.num_masks * (self.num_masks - 1))

        # Return diversity (1 - similarity)
        return 1 - avg_similarity


class MaskNetwork(nn.Module):
    """
    Single mask network for generating feature-level attention masks.

    A feedforward network that learns to generate soft masks for input features.
    Each layer applies a linear transformation followed by ReLU activation,
    except the last layer which outputs raw mask logits.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int = 3,
            use_bias: bool = False,
            dropout: float = 0.0,
            hidden_activation: str = 'relu'
    ):
        """
        Initialize a single mask network.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (typically same as input_dim)
            num_layers: Number of layers (minimum 1)
            use_bias: Whether to use bias in linear layers
            dropout: Dropout rate between layers
            hidden_activation: Activation function for hidden layers
        """
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Build network layers
        layers = []
        current_dim = input_dim

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim, bias=use_bias))
            layers.append(self._get_activation(hidden_activation))

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            current_dim = hidden_dim

        # Output layer (no activation - raw logits)
        layers.append(nn.Linear(current_dim, output_dim, bias=use_bias))

        self.network = nn.Sequential(*layers)

    @staticmethod
    def _get_activation(activation: str) -> nn.Module:
        """Get activation module."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh()
        }
        return activations.get(activation, nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the mask network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Mask logits of shape (batch_size, output_dim)
        """
        return self.network(x)


class Generator(nn.Module):
    """
    Legacy wrapper for MaskGenerator to maintain backward compatibility.

    This class provides the same interface as the original Generator class
    but uses the improved MaskGenerator implementation.
    """

    def __init__(self, model, config: dict):
        """
        Initialize generator with legacy interface.

        Args:
            model: Object with _make_nets method (for compatibility, not used)
            config: Configuration dictionary with keys:
                - data_dim: Feature dimension
                - mask_nlayers: Number of layers per mask network
                - mask_num: Number of mask networks
                - device: Device to use
        """
        super().__init__()

        # Extract configuration
        feature_dim = config['data_dim']
        num_layers = config['mask_nlayers']
        num_masks = config['mask_num']
        device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        # Create modern mask generator
        self.mask_generator = MaskGenerator(
            feature_dim=feature_dim,
            num_masks=num_masks,
            num_layers=num_layers,
            device=torch.device(device)
        )

        # Store for compatibility
        self.masks = self.mask_generator.mask_networks
        self.mask_num = num_masks
        self.device = self.mask_generator.device

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass (legacy interface).

        Args:
            x: Input tensor

        Returns:
            Tuple of (masked_features, raw_masks)
        """
        return self.mask_generator(x, return_masks=True)
