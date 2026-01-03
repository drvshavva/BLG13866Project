import torch
import torch.nn as nn


class AttentionMaskGenerator(nn.Module):
    def __init__(self, input_dim, n_masks=15, n_heads=None, hidden_dim=256):
        super().__init__()
        self.n_masks = n_masks

        if n_heads is None:
            n_heads = self._find_optimal_heads(input_dim)
        else:
            if input_dim % n_heads != 0:
                print(f"Warning: input_dim ({input_dim}) is not divisible by n_heads ({n_heads})")
                n_heads = self._find_optimal_heads(input_dim)
                print(f"Auto-adjusted n_heads to: {n_heads}")

        self.n_heads = n_heads

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=n_heads,
            batch_first=True
        )

        # Mask generators
        self.mask_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, input_dim),
                nn.Sigmoid()
            ) for _ in range(n_masks)
        ])

    def _find_optimal_heads(self, input_dim):
        preferred_heads = [4, 2, 8, 1]

        for heads in preferred_heads:
            if input_dim % heads == 0:
                return heads

        divisors = []
        for i in range(1, input_dim + 1):
            if input_dim % i == 0:
                divisors.append(i)

        for divisor in reversed(divisors):
            if divisor <= input_dim // 2:
                return divisor

        return 1

    def forward(self, x):
        # x: (batch_size, input_dim)
        # Add sequence dimension for attention
        x_seq = x.unsqueeze(1)  # (batch_size, 1, input_dim)

        # Apply self-attention
        attn_out, _ = self.attention(x_seq, x_seq, x_seq)
        attn_out = attn_out.squeeze(1)  # (batch_size, input_dim)

        # Generate masks using attended features
        masks = [net(attn_out) for net in self.mask_networks]
        return torch.stack(masks, dim=0)  # (n_masks, batch_size, input_dim)