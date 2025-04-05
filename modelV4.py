import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm


# Custom Dataset: Loads HR images from a directory and creates LR images by downsampling
class ImageSRDataset(Dataset):
    def __init__(self, root_dir, hr_size, upscaling_factor):
        self.hr_size = hr_size
        self.upscaling_factor = upscaling_factor
        self.image_paths = list(Path(root_dir).glob("*.png"))
        self.hr_transform = transforms.Compose([
            transforms.Resize((hr_size, hr_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        hr_image = self.hr_transform(img)
        lr_size = self.hr_size // self.upscaling_factor
        lr_image = F.interpolate(hr_image.unsqueeze(0), size=(lr_size, lr_size), mode='bilinear',
                                 align_corners=False).squeeze(0)
        return lr_image, hr_image


# FeatureAttentionBlock: Convolution + Attention + Transformer
class FeatureAttentionBlock(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads):
        super(FeatureAttentionBlock, self).__init__()

        # Convolution layers
        conv_layers = []
        conv_layers.append(nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1))
        conv_layers.append(nn.ReLU(inplace=True))

        self.convs = nn.Sequential(*conv_layers)

        # Multi-head self-attention block
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)

        # Transformer encoder layer for patch tokens
        self.transformer_block = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)

    def forward(self, x):
        # Apply convolutional layers
        x = self.convs(x)
        b, c, h, w = x.shape

        # Prepare tokens for attention: reshape to (sequence length, batch, embed_dim)
        tokens = x.flatten(2).permute(2, 0, 1)

        # Apply attention mechanism
        attn_out, _ = self.attn(tokens, tokens, tokens)

        # Reshape attention output back to feature maps
        attn_out = attn_out.permute(1, 2, 0).view(b, c, h, w)

        # Add skip connection and process with transformer block
        x = x + attn_out
        tokens = x.flatten(2).permute(2, 0, 1)

        # Apply transformer block
        tokens = self.transformer_block(tokens)

        # Reshape back to feature maps
        x = tokens.permute(1, 2, 0).view(b, c, h, w)

        return x


# Feature Attention Super-Resolution Model
class FeatureAttentionSR(nn.Module):
    def __init__(self, in_channels=3, num_blocks=4, embed_dim=96, num_heads=4, upscaling_factor=2):
        super(FeatureAttentionSR, self).__init__()

        # Create multiple FeatureAttentionBlocks
        blocks = []
        for _ in range(num_blocks):
            blocks.append(FeatureAttentionBlock(in_channels, embed_dim, num_heads))
            in_channels = embed_dim  # After the first block, the number of channels becomes embed_dim

        self.blocks = nn.Sequential(*blocks)

        # Reconstruction and upsampling layers
        self.reconstruct = nn.Conv2d(embed_dim, in_channels * (upscaling_factor ** 2), kernel_size=3, padding=1)
        self.upsample = nn.PixelShuffle(upscaling_factor)

    def forward(self, x):
        # Pass input through the sequence of FeatureAttentionBlocks
        x = self.blocks(x)

        # Reconstruction and upscaling
        x = self.reconstruct(x)
        x = self.upsample(x)

        return x


# Training function for one epoch
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for lr_images, hr_images in tqdm(dataloader, desc="Training", leave=False):
        lr_images, hr_images = lr_images.to(device), hr_images.to(device)
        optimizer.zero_grad()
        outputs = model(lr_images)
        loss = criterion(outputs, hr_images)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


# Validation function
def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for lr_images, hr_images in dataloader:
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)
            outputs = model(lr_images)
            loss = criterion(outputs, hr_images)
            val_loss += loss.item()
    return val_loss / len(dataloader)


# Main function: sets up dataset, model, and training loop with early stopping
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create training and validation datasets
    train_dataset = ImageSRDataset(args.train_dir, args.hr_size, args.upscaling_factor)
    val_dataset = ImageSRDataset(args.val_dir, args.hr_size, args.upscaling_factor)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = FeatureAttentionSR(
        in_channels=3,
        num_blocks=args.num_blocks,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        upscaling_factor=args.upscaling_factor
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), args.save_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training HR images")
    parser.add_argument("--val_dir", type=str, required=True, help="Path to validation HR images")
    parser.add_argument("--hr_size", type=int, default=128, help="High-resolution image size")
    parser.add_argument("--upscaling_factor", type=int, default=2, help="Upscaling factor for super-resolution")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--embed_dim", type=int, default=96, help="Embedding dimension for conv and transformer")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of heads in multi-head attention")
    parser.add_argument("--num_blocks", type=int, default=4, help="Number of blocks to apply sequentially")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--save_path", type=str, default="best_model.pth", help="Path to save the best model")

    args = parser.parse_args()
    main(args)
