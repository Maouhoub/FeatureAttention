import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import argparse


class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, upscale_factor, transform_lr=None, transform_hr=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.upscale_factor = upscale_factor
        self.transform_lr = transform_lr
        self.transform_hr = transform_hr
        self.lr_files = sorted(os.listdir(lr_dir))
        self.hr_files = sorted(os.listdir(hr_dir))

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        lr_img = Image.open(os.path.join(self.lr_dir, self.lr_files[idx])).convert("RGB")
        hr_img = Image.open(os.path.join(self.hr_dir, self.hr_files[idx])).convert("RGB")
        if self.transform_lr:
            lr_img = self.transform_lr(lr_img)
        if self.transform_hr:
            hr_img = self.transform_hr(hr_img)
        return lr_img, hr_img


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        return self.conv(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )

    def forward(self, x):
        x_attn = x.permute(1, 0, 2)
        attn_out, _ = self.attn(self.norm1(x_attn), self.norm1(x_attn), self.norm1(x_attn))
        x = x + attn_out.permute(1, 0, 2)
        x = x + self.mlp(self.norm2(x))
        return x


class ViTSR(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64, patch_size=4,
                 num_heads=4, depth=6, mlp_dim=128, upscale_factor=2, num_stages=3):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Initial feature extraction without downsampling
        self.init_feature = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)

        # Transformer encoder
        self.transformer = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_dim)
            for _ in range(depth)
        ])

        # Upsampling stages
        self.stages = nn.ModuleList()
        for _ in range(num_stages):
            stage = nn.ModuleDict({
                'transformer': nn.Sequential(*[
                    TransformerBlock(embed_dim, num_heads, mlp_dim)
                    for _ in range(depth // 2)
                ]),
                'upsample': nn.Sequential(
                    nn.Conv2d(embed_dim, embed_dim * (upscale_factor ** 2), 3, padding=1),
                    nn.PixelShuffle(upscale_factor),
                    nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
                )
            })
            self.stages.append(stage)

        # Final upsampling to target resolution
        self.final_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, in_channels, 3, padding=1)
        )

    def forward(self, x):
        # Initial feature extraction
        x = self.init_feature(x)

        # Transformer processing
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).view(B, H * W, C)
        x = self.transformer(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)

        # Upsampling stages
        for stage in self.stages:
            # Transformer
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).view(B, H * W, C)
            x = stage['transformer'](x)
            x = x.view(B, H, W, C).permute(0, 3, 1, 2)

            # Upsample
            x = stage['upsample'](x)

        # Final convolution to RGB
        return self.final_upsample(x)


def train_model(lr_dir, hr_dir, epochs, patience, in_channels, embed_dim,
                patch_size, num_heads, depth, mlp_dim, upscale_factor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ViTSR(in_channels, embed_dim, patch_size, num_heads, depth,
                  mlp_dim, upscale_factor).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()

    lr_size = 128 // upscale_factor
    hr_size = 128

    transform_lr = transforms.Compose([
        transforms.Resize((lr_size, lr_size)),
        transforms.ToTensor()
    ])

    transform_hr = transforms.Compose([
        transforms.Resize((hr_size, hr_size)),
        transforms.ToTensor()
    ])

    dataset = SRDataset(lr_dir, hr_dir, upscale_factor, transform_lr, transform_hr)
    train_size = int(0.8 * len(dataset))
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

    best_loss = float('inf')
    patience_cnt = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for lr, hr in train_loader:
            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad()
            output = model(lr)
            loss = criterion(output, hr)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lr, hr in val_loader:
                lr, hr = lr.to(device), hr.to(device)
                output = model(lr)
                val_loss += criterion(output, hr).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_cnt = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print("Early stopping triggered")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr_dir", type=str, required=True)
    parser.add_argument("--hr_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--mlp_dim", type=int, default=128)
    parser.add_argument("--upscale_factor", type=int, default=2)
    args = parser.parse_args()

    train_model(
        args.lr_dir,
        args.hr_dir,
        args.epochs,
        args.patience,
        in_channels=3,
        embed_dim=args.embed_dim,
        patch_size=args.patch_size,
        num_heads=args.num_heads,
        depth=args.depth,
        mlp_dim=args.mlp_dim,
        upscale_factor=args.upscale_factor
    )