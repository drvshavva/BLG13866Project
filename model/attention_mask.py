import torch
import torch.nn as nn


class AttentionMaskGenerator(nn.Module):
    def __init__(self, input_dim, n_masks=15, n_heads=4, hidden_dim=256):
        super().__init__()
        self.n_masks = n_masks
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

    def forward(self, x):
        # x: (batch_size, input_dim)
        x_seq = x.unsqueeze(1)  # (batch_size, 1, input_dim)

        # Apply self-attention
        attn_out, _ = self.attention(x_seq, x_seq, x_seq)
        attn_out = attn_out.squeeze(1)  # (batch_size, input_dim)

        # Generate masks
        masks = [net(attn_out) for net in self.mask_networks]
        return torch.stack(masks, dim=0)  # (n_masks, batch_size, input_dim)