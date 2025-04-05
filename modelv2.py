import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

class ConvFeatureExtractor(nn.Module):
    def __init__(self, in_channels, num_features, num_convs, kernel_size, stride=1):
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers.append(
                nn.Conv2d(
                    in_channels if i == 0 else num_features,
                    num_features,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2
                )
            )
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class EfficientSR(nn.Module):
    def __init__(
        self, in_channels, out_channels, num_features,
        num_convs, kernel_size,
        num_tokens, num_heads, num_layers, upsample_scale
    ):
        super().__init__()
        # conv extractor always stride=1
        self.extractor = ConvFeatureExtractor(
            in_channels, num_features, num_convs, kernel_size, stride=1
        )
        self.class_token = nn.Parameter(torch.randn(1, 1, num_features))
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_tokens + 1, num_features)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_features,
            nhead=num_heads,
            dim_feedforward=num_features * 4,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.upsample = nn.PixelShuffle(upsample_scale)
        self.conv_reconstruct = nn.Conv2d(
            num_features // (upsample_scale**2), out_channels,
            kernel_size=3, padding=1
        )

    def forward(self, x):
        B, C, H, W = x.shape
        feat = self.extractor(x)
        B, F, Hf, Wf = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        cls_tokens = self.class_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.size(1), :]
        out = self.transformer(tokens)
        feat_trans = out[:, 1:, :].transpose(1, 2).view(B, F, Hf, Wf)
        up = self.upsample(feat_trans)
        recon = self.conv_reconstruct(up)
        return recon

class SRDataset(Dataset):
    def __init__(self, root_dir, scale, crop_size, transform=None):
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
        self.scale = scale
        self.crop_size = crop_size
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        hr_w, hr_h = img.size
        if hr_w < self.crop_size or hr_h < self.crop_size:
            img = img.resize(
                (max(hr_w, self.crop_size), max(hr_h, self.crop_size)),
                Image.BICUBIC
            )
            hr_w, hr_h = img.size
        i = torch.randint(0, hr_h - self.crop_size + 1, (1,)).item()
        j = torch.randint(0, hr_w - self.crop_size + 1, (1,)).item()
        hr_patch = img.crop((j, i, j + self.crop_size, i + self.crop_size))
        lr_patch = hr_patch.resize(
            (self.crop_size // self.scale, self.crop_size // self.scale),
            Image.BICUBIC
        )
        if self.transform:
            hr_patch = self.transform(hr_patch)
            lr_patch = self.transform(lr_patch)
        return lr_patch, hr_patch


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.ToTensor()
    # split dataset into train/val
    full_dataset = SRDataset(args.data_dir, args.scale, args.crop_size, transform)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = EfficientSR(
        in_channels=3,
        out_channels=3,
        num_features=args.num_features,
        num_convs=args.num_convs,
        kernel_size=args.kernel_size,
        num_tokens=(args.crop_size ** 2),
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        upsample_scale=args.scale
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    best_val = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for lr, hr in train_loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            loss = criterion(sr, hr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * lr.size(0)
        train_loss /= train_size

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lr, hr in val_loader:
                lr, hr = lr.to(device), hr.to(device)
                sr = model(lr)
                loss = criterion(sr, hr)
                val_loss += loss.item() * lr.size(0)
        val_loss /= val_size

        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # early stopping
        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), args.save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}. Best Val Loss: {best_val:.4f}")
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--num_features', type=int, default=64)
    parser.add_argument('--num_convs', type=int, default=3)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience in epochs')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction of data for validation')
    parser.add_argument('--save_path', type=str, default='sr_model.pth')
    args = parser.parse_args()
    train(args)
