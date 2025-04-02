import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import argparse


# Custom Dataset
class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, upscale_factor, transform_lr=None, transform_hr=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.upscale_factor = upscale_factor
        self.transform_lr = transform_lr
        self.transform_hr = transform_hr
        self.lr_filenames = sorted(os.listdir(lr_dir))
        self.hr_filenames = sorted(os.listdir(hr_dir))

    def __len__(self):
        return len(self.hr_filenames)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_filenames[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_filenames[idx])
        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")
        if self.transform_lr:
            lr_img = self.transform_lr(lr_img)
        if self.transform_hr:
            hr_img = self.transform_hr(hr_img)
        return lr_img, hr_img


# Feature extraction using CNN (kernel_size = stride = patch_size)
class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.conv(x)


# Transformer Encoder Block
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
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


# Vision Transformer for Super-Resolution
# Vision Transformer for Super-Resolution
class ViTSR(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, num_heads, depth, mlp_dim, upscale_factor):
        super().__init__()
        self.feature_extractor = FeatureExtractor(in_channels, embed_dim, patch_size)
        self.transformer = nn.Sequential(*[TransformerBlock(embed_dim, num_heads, mlp_dim) for _ in range(depth)])
        total_upscale = patch_size * upscale_factor  # Calculate total upscale factor
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * (total_upscale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(total_upscale),
            nn.Conv2d(embed_dim, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)

        x = self.feature_extractor(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)

        x = self.feature_extractor(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)


        x = x.permute(1, 2, 0).reshape(B, C, H, W)
        x = self.upsample(x)
        return x
# Training Setup with Early Stopping
def train_model(lr_dir, hr_dir, num_epochs, patience, in_channels, embed_dim, patch_size, num_heads, depth, mlp_dim,
                upscale_factor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTSR(in_channels, embed_dim, patch_size, num_heads, depth, mlp_dim, upscale_factor).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()

    lr_transform = transforms.Compose([
        transforms.Resize((128 // upscale_factor, 128 // upscale_factor)),
        transforms.ToTensor()
    ])

    hr_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = SRDataset(lr_dir, hr_dir, upscale_factor, lr_transform, hr_transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
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
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr_dir", type=str, required=True, help="Path to low-resolution images")
    parser.add_argument("--hr_dir", type=str, required=True, help="Path to high-resolution images")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--patch_size", type=int, default=4, help="Patch size for feature extraction")
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--depth", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--mlp_dim", type=int, default=128, help="MLP hidden layer dimension")
    parser.add_argument("--upscale_factor", type=int, default=2, help="Upscaling factor")
    args = parser.parse_args()

    train_model(args.lr_dir, args.hr_dir, args.epochs, args.patience, 3, args.embed_dim, args.patch_size,
                args.num_heads, args.depth, args.mlp_dim, args.upscale_factor)
