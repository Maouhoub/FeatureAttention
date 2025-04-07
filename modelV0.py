import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.bn(out)
        out += residual
        out = self.relu(out)
        return out

class LightweightModel(nn.Module):
    def __init__(self, n_layers=5, in_channels=3, out_channels=3,
                 upscale_factor=2, attn_heads=4, base_channels=32):
        super().__init__()
        self.base_channels = base_channels
        self.upscale_factor = upscale_factor

        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(base_channels) for _ in range(n_layers)
        ])

        # Multi-head self-attention
        self.mha = nn.MultiheadAttention(
            embed_dim=base_channels,
            num_heads=attn_heads,
            batch_first=True
        )

        # Upsampling layers
        self.upsample = nn.Sequential(
            nn.Conv2d(base_channels, out_channels * (upscale_factor ** 2),
                     kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Feature extraction
        x = self.initial_conv(x)
        x = self.res_blocks(x)

        # Attention mechanism
        batch, C, H, W = x.shape
        x_tokens = x.view(batch, C, H*W).permute(0, 2, 1)  # [B, N, C]
        attn_output, _ = self.mha(x_tokens, x_tokens, x_tokens)
        x = attn_output.permute(0, 2, 1).view(batch, C, H, W)

        # Upsampling
        x = self.upsample(x)
        return x