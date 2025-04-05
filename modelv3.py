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

# Custom Dataset
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
        lr_image = F.interpolate(hr_image.unsqueeze(0), size=(lr_size, lr_size), mode='bilinear', align_corners=False).squeeze(0)
        return lr_image, hr_image

# Model Definition
class FeatureAttentionSR(nn.Module):
    def __init__(self, in_channels=3, num_convs=4, embed_dim=96, num_heads=4, upscaling_factor=2):
        super(FeatureAttentionSR, self).__init__()
        self.convs = nn.Sequential(*[
            nn.Conv2d(in_channels if i == 0 else embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            for i in range(num_convs)
        ])

        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.flatten = nn.Flatten(2)
        self.unflatten = lambda x, h, w: x.view(x.size(0), -1, h, w)

        self.transformer_block = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)

        self.reconstruct = nn.Conv2d(embed_dim, in_channels * (upscaling_factor ** 2), kernel_size=3, padding=1)
        self.upsample = nn.PixelShuffle(upscaling_factor)

    def forward(self, x):
        skip = x
        x = self.convs(x)

        b, c, h, w = x.shape
        tokens = self.flatten(x).permute(2, 0, 1)  # (seq_len, batch, embed_dim)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        attn_out = attn_out.permute(1, 2, 0).view(b, c, h, w)

        x = x + attn_out
        x = x.flatten(2).permute(2, 0, 1)
        x = self.transformer_block(x)
        x = x.permute(1, 2, 0).view(b, c, h, w)

        x = self.reconstruct(x)
        x = self.upsample(x)
        return x

# Training function
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for lr_images, hr_images in tqdm(dataloader):
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

# Main function
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = ImageSRDataset(args.train_dir, args.hr_size, args.upscaling_factor)
    val_dataset = ImageSRDataset(args.val_dir, args.hr_size, args.upscaling_factor)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = FeatureAttentionSR(
        in_channels=3,
        num_convs=args.num_convs,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        upscaling_factor=args.upscaling_factor
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    patience, counter = args.patience, 0

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), args.save_path)
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training HR images")
    parser.add_argument("--val_dir", type=str, required=True, help="Path to validation HR images")
    parser.add_argument("--hr_size", type=int, default=128)
    parser.add_argument("--upscaling_factor", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embed_dim", type=int, default=96)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_convs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--save_path", type=str, default="best_model.pth")

    args = parser.parse_args()

    main(args)
