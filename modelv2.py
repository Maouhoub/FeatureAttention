import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    dataset = SRDataset(args.data_dir, args.scale, args.crop_size, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

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

    for epoch in range(args.epochs):
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            loss = criterion(sr, hr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), args.save_path)


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
    parser.add_argument('--save_path', type=str, default='sr_model.pth')
    args = parser.parse_args()
    # enforce stride=1 internally
    train(args)
