import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm

from modelV0 import LightweightModel


def parse_args():
    parser = argparse.ArgumentParser(description='Train Lightweight SR Model on DIV2K')
    parser.add_argument('--train_hr_dir', type=str, required=True, help='Path to training HR images')
    parser.add_argument('--train_lr_dir', type=str, required=True, help='Path to training LR images')
    parser.add_argument('--val_hr_dir', type=str, required=True, help='Path to validation HR images')
    parser.add_argument('--val_lr_dir', type=str, required=True, help='Path to validation LR images')
    parser.add_argument('--scale', type=int, default=2, choices=[2, 3, 4], help='Super-resolution scale factor')
    parser.add_argument('--batch_size', type=int, default=16, help='Input batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--n_layers', type=int, default=5, help='Number of residual blocks')
    parser.add_argument('--attn_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--base_channels', type=int, default=32, help='Base number of channels')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--save_path', type=str, default='best_model.pth', help='Path to save best model')
    parser.add_argument('--device', type=str, default=None, help="Device to use ('cuda'/'cpu')")
    return parser.parse_args()


class DIV2KDataset(Dataset):
    def __init__(self, hr_root, lr_root, scale=2):
        self.scale = scale
        self.hr_paths = sorted([os.path.join(hr_root, f) for f in os.listdir(hr_root)])
        self.lr_paths = sorted([os.path.join(lr_root, f) for f in os.listdir(lr_root)])
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
        ])

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr_img = Image.open(self.hr_paths[idx]).convert('RGB')
        lr_img = Image.open(self.lr_paths[idx]).convert('RGB')

        hr_tensor = transforms.ToTensor()(hr_img)
        lr_tensor = transforms.ToTensor()(lr_img)

        # Synchronized augmentation
        if self.transform:
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            hr_tensor = self.transform(hr_tensor)
            torch.manual_seed(seed)
            lr_tensor = self.transform(lr_tensor)

        return lr_tensor, hr_tensor


def train_model(args):
    # Device setup
    device = args.device if args.device else 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize model
    model = LightweightModel(
        n_layers=args.n_layers,
        upscale_factor=args.scale,
        attn_heads=args.attn_heads,
        base_channels=args.base_channels
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Dataset and loaders
    train_dataset = DIV2KDataset(args.train_hr_dir, args.train_lr_dir, args.scale)
    val_dataset = DIV2KDataset(args.val_hr_dir, args.val_lr_dir, args.scale)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    best_val_loss = float('inf')
    no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for lr, hr in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            lr = lr.to(device)
            hr = hr.to(device)

            optimizer.zero_grad()
            outputs = model(lr)
            loss = criterion(outputs, hr)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * lr.size(0)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lr, hr in val_loader:
                lr = lr.to(device)
                hr = hr.to(device)
                val_loss += criterion(model(lr), hr).item() * lr.size(0)

        train_loss /= len(train_dataset)
        val_loss /= len(val_dataset)

        print(f'Epoch {epoch + 1}/{args.epochs}')
        print(f'Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), args.save_path)
            print(f'Model saved to {args.save_path}')
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break


if __name__ == '__main__':
    args = parse_args()
    train_model(args)