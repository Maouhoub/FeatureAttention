import os
import argparse
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms


# Define the EfficientSR model
class EfficientSR(nn.Module):
    def __init__(self, in_channels=3, n=4, conv_channels=64,
                 num_heads=8, transformer_layers=4, patch_size=4, upscaling_factor=2):
        super(EfficientSR, self).__init__()
        # 1. Feature extraction using n Conv2D layers
        convs = []
        for i in range(n):
            convs.append(nn.Conv2d(in_channels if i == 0 else conv_channels,
                                   conv_channels, kernel_size=3, padding=1))  # Conv2D layer
            convs.append(nn.ReLU(inplace=True))  # ReLU activation
        self.conv_layers = nn.Sequential(*convs)  # Stack of convolutional layers

        # Skip connection convolution for improved reconstruction
        self.skip_conv = nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1)  # Skip connection

        # 2. Multi-head self-attention on flattened feature maps (tokens)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=conv_channels,
                                                    num_heads=num_heads, batch_first=True)  # Multi-head attention

        # 3. Patch embedding via a Conv2D with stride equal to patch size
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(conv_channels, conv_channels, kernel_size=patch_size,
                                     stride=patch_size)  # Patch embedding
        embedding_dim = conv_channels  # Embedding dimension for transformer

        # Transformer encoder block
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                                   batch_first=True)  # Transformer encoder layer
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)  # Transformer encoder

        # 4. Upsampling using PixelShuffle (efficient upsampling)
        # The conv layer outputs upscaling_factor^2 * in_channels to meet PixelShuffle requirements
        self.upsample_conv = nn.Conv2d(conv_channels, (upscaling_factor ** 2) * in_channels, kernel_size=3,
                                       padding=1)  # Pre-upsample conv
        self.pixel_shuffle = nn.PixelShuffle(upscaling_factor)  # PixelShuffle layer for upsampling

    def forward(self, x):
        # Apply n convolutional layers to extract features
        conv_out = self.conv_layers(x)  # [B, conv_channels, H, W]
        skip = self.skip_conv(conv_out)  # Save skip connection

        B, C, H, W = conv_out.shape
        # Flatten the feature maps into tokens for attention [B, H*W, C]
        tokens = conv_out.view(B, C, H * W).permute(0, 2, 1)  # Flatten and permute

        # Apply multi-head self-attention
        attn_out, _ = self.multihead_attn(tokens, tokens, tokens)  # Self-attention on tokens
        # Reconstruct the attended tokens back to feature maps
        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)  # Reshape to feature maps
        # Add skip connection for enhanced feature reconstruction
        attn_out = attn_out + skip  # Skip connection addition

        # Patch embedding: split feature map into patches
        patches = self.patch_embed(attn_out)  # [B, conv_channels, H_patch, W_patch]
        B, C_new, H_patch, W_patch = patches.shape
        # Flatten patches into a sequence for transformer [B, num_patches, embedding_dim]
        patches = patches.view(B, C_new, -1).permute(0, 2, 1)  # Flatten and permute

        # Pass patches through transformer encoder
        transformer_out = self.transformer(patches)  # Transformer processing
        # Reshape transformer output back to feature maps
        transformer_out = transformer_out.permute(0, 2, 1).view(B, C_new, H_patch, W_patch)  # Reshape

        # Upsample the features using PixelShuffle
        upsampled = self.upsample_conv(transformer_out)  # Convolution before PixelShuffle
        out = self.pixel_shuffle(upsampled)  # PixelShuffle to increase resolution
        return out


# Custom dataset that loads HR images from a directory and creates LR images by downsampling
class ImageSRDataset(Dataset):
    def __init__(self, data_path, hr_size=128, upscaling_factor=2, extensions=['jpg', 'png', 'jpeg']):
        self.data_path = data_path
        self.hr_size = hr_size
        self.lr_size = hr_size // upscaling_factor
        self.upscaling_factor = upscaling_factor
        # List all image files in the directory with the given extensions
        self.image_files = []
        for ext in extensions:
            self.image_files.extend(glob(os.path.join(data_path, f'*.{ext}')))
        # Transformation to resize and convert images to tensor
        self.hr_transform = transforms.Compose([
            transforms.Resize((hr_size, hr_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load the HR image
        img = Image.open(self.image_files[idx]).convert("RGB")
        hr_image = self.hr_transform(img)  # High-resolution image tensor
        # Create LR image by downsampling the HR image
        lr_image = F.interpolate(hr_image.unsqueeze(0), size=(self.lr_size, self.lr_size), mode='bilinear',
                                 align_corners=False).squeeze(0)
        return lr_image, hr_image  # Return LR and HR pair


def main(args):
    # If a dataset path is provided, use the ImageSRDataset; otherwise, use a random dataset.
    if args.data_path:
        dataset = ImageSRDataset(data_path=args.data_path, hr_size=args.hr_size, upscaling_factor=args.upscaling_factor)
        total_samples = len(dataset)
        print(f"Loaded {total_samples} images from {args.data_path}")


    # Split dataset into training and validation sets
    train_size = int(args.train_split * total_samples)
    val_size = total_samples - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Instantiate the model with given parameters
    model = EfficientSR(in_channels=args.in_channels, n=args.n, conv_channels=args.conv_channels,
                        num_heads=args.num_heads, transformer_layers=args.transformer_layers,
                        patch_size=args.patch_size, upscaling_factor=args.upscaling_factor)
    model.train()  # Set model to training mode

    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Variables for early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Training loop with validation and early stopping
    for epoch in range(args.num_epochs):
        model.train()  # Ensure model is in training mode
        running_loss = 0.0
        for i, (lr_images, hr_images) in enumerate(train_loader):
            optimizer.zero_grad()  # Zero the gradients

            outputs = model(lr_images)  # Forward pass
            loss = criterion(outputs, hr_images)  # Compute loss

            loss.backward()  # Backpropagate
            optimizer.step()  # Update parameters

            running_loss += loss.item()  # Accumulate training loss

            if (i + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{args.num_epochs}], Step [{i + 1}/{len(train_loader)}], Training Loss: {running_loss / 10:.4f}")
                running_loss = 0.0

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for lr_images, hr_images in val_loader:
                outputs = model(lr_images)
                loss = criterion(outputs, hr_images)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch [{epoch + 1}/{args.num_epochs}], Validation Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print("Early stopping triggered!")
            break

    print("Training finished!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Efficient Super-Resolution Training")
    # Model hyperparameters
    parser.add_argument('--in_channels', type=int, default=3, help="Number of input channels")
    parser.add_argument('--n', type=int, default=4, help="Number of Conv2D layers")
    parser.add_argument('--conv_channels', type=int, default=64, help="Number of convolutional channels")
    parser.add_argument('--num_heads', type=int, default=8, help="Number of heads in multi-head attention")
    parser.add_argument('--transformer_layers', type=int, default=4, help="Number of transformer encoder layers")
    parser.add_argument('--patch_size', type=int, default=4, help="Patch size for patch embedding")
    parser.add_argument('--upscaling_factor', type=int, default=2, help="Upscaling factor for super-resolution")
    parser.add_argument('--hr_size', type=int, default=128, help="High-resolution image size")

    # Dataset and training parameters
    parser.add_argument('--data_path', type=str, default='',
                        help="Path to training images directory (if empty, use random data)")
    parser.add_argument('--total_samples', type=int, default=1200,
                        help="Total number of samples (used if data_path is not provided)")
    parser.add_argument('--train_split', type=float, default=0.8, help="Fraction of data used for training")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--num_epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument('--patience', type=int, default=5, help="Patience for early stopping")

    args = parser.parse_args()  # Parse command-line arguments

    main(args)
