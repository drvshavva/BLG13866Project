import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Encoder Module

    Encodes masked data into a latent space.

    Architecture:
    - 3 layer MLP
    - LeakyReLU activation functions
    - Bottleneck structure

    Args:
        input_dim: Input Dimesion
        hidden_dim: Hidden layer dimension
        latent_dim: Latent space dimension (bottleneck)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, latent_dim: int = 128):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Masked input, shape: (batch_size, input_dim)

        Returns:
            z: Latent space, shape: (batch_size, latent_dim)
        """
        return self.encoder(x)


class Decoder(nn.Module):
    """
    Decoder Module

    Decodes the latent representation back to the original data.
    Symetrically designed with the Encoder.

    Args:
        latent_dim: Latent space dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension (same as input dimension of Encoder)
    """

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256, output_dim: int = None):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent space, shape: (batch_size, latent_dim)

        Returns:
            x_hat: Decoded data, shape: (batch_size, output_dim)
        """
        return self.decoder(z)
