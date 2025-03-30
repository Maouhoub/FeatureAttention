import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os


# Custom Dataset
class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform = transform
        self.filenames = sorted(os.listdir(lr_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.filenames[idx])
        hr_path = os.path.join(self.hr_dir, self.filenames[idx])
        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")
        if self.transform:
            lr_img = self.transform(lr_img)
            hr_img = self.transform(hr_img)
        return lr_img, hr_img


# Feature extraction using CNN (kernel_size = stride = patch_size)
class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64, patch_size=4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.conv(x)  # Output: (B, embed_dim, H/patch_size, W/patch_size)


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
class ViTSR(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64, patch_size=4, num_heads=4, depth=6, mlp_dim=128, upscale_factor=2):
        super().__init__()
        self.feature_extractor = FeatureExtractor(in_channels, embed_dim, patch_size)
        self.transformer = nn.Sequential(*[TransformerBlock(embed_dim, num_heads, mlp_dim) for _ in range(depth)])
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(embed_dim, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)  # (B, C, H//P, W//P)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # Convert feature maps into tokens (B, H*W, C)
        x = x.permute(1, 0, 2)  # Reshape to (H*W, B, C) for transformer
        x = self.transformer(x)
        x = x.permute(1, 2, 0).reshape(B, C, H, W)  # Reshape back to (B, C, H, W)
        x = self.upsample(x)
        return x


# Training Setup with Early Stopping
def train_model(lr_dir, hr_dir, num_epochs=50, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTSR().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = SRDataset(lr_dir, hr_dir, transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for lr_imgs, hr_imgs in train_loader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            optimizer.zero_grad()
            output = model(lr_imgs)
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
            torch.save(model.state_dict(), "best_model.pth")  # Save best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    train_model("path_to_lr_images", "path_to_hr_images")
