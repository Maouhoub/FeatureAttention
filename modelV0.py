import argparse

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from model import SRDataset


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

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = args.device if args.device else 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize model
    model = LightweightModel(
        n_layers=args.n_layers,
        upscale_factor=args.scale,
        attn_heads=args.attn_heads,
        base_channels=args.base_channels
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()

    lr_transform = transforms.Compose([
        transforms.Resize((128 // args.scale, 128 // args.scale)),
        transforms.ToTensor()
    ])

    hr_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = SRDataset(args.train_hr_dir, args.train_lr_dir, args.scale, lr_transform, hr_transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        print("epoch : ", epoch)
        model.train()
        train_loss = 0.0
        for lr_imgs, hr_imgs in train_loader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            optimizer.zero_grad()
            output = model(lr_imgs)
            assert output.shape == hr_imgs.shape, f"Shape mismatch! Output: {output.shape}, HR: {hr_imgs.shape}"
            loss = criterion(output, hr_imgs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                output = model(lr_imgs)
                loss = criterion(output, hr_imgs)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    train_model(args)