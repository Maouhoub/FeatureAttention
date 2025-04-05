import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from thop import profile

from modelv2 import EfficientSR


# You need to either import your model class or define it here
# Or copy the class definition


class TestSRDataset(Dataset):
    """Dataset for testing super-resolution models"""

    def __init__(self, root_dir, scale, hr_crop_size, transform=None):
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir)
                      if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        self.scale = scale
        self.hr_crop_size = hr_crop_size
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        hr_img = img.resize((self.hr_crop_size, self.hr_crop_size), Image.BICUBIC)
        lr_size = self.hr_crop_size // self.scale
        lr_img = hr_img.resize((lr_size, lr_size), Image.BICUBIC)
        hr = self.transform(hr_img)
        lr = self.transform(lr_img)
        return lr, hr, os.path.basename(self.files[idx])


def calculate_psnr(sr, hr, max_val=1.0):
    mse = torch.mean((sr - hr) ** 2)
    return 10 * torch.log10((max_val ** 2) / mse) if mse != 0 else float('inf')


def test_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    test_dataset = TestSRDataset(args.test_dir, args.scale, args.crop_size, transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize model
    model = EfficientSR(
        in_channels=3,
        out_channels=3,
        num_features=args.num_features,
        num_convs=args.num_convs,
        kernel_size=args.kernel_size,
        num_tokens=(args.crop_size // args.scale) ** 2,  # LR spatial dimensions
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        upsample_scale=args.scale
    ).to(device)

    # Load trained weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Testing loop
    total_psnr = 0.0
    with torch.no_grad():
        for lr, hr, filename in test_loader:
            lr, hr = lr.to(device), hr.to(device)

            # Timing
            start_time = time.time()
            sr = model(lr)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_time = (time.time() - start_time) * 1000

            # PSNR
            psnr = calculate_psnr(sr, hr)
            total_psnr += psnr.item()

            print(f"{filename[0]} - PSNR: {psnr:.2f} dB, Time: {inference_time:.2f}ms")

    # Metrics
    avg_psnr = total_psnr / len(test_dataset)
    print(f"\nAverage PSNR: {avg_psnr:.2f} dB")

    # FLOPs and parameters
    dummy_input = torch.randn(1, 3, args.crop_size // args.scale, args.crop_size // args.scale).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    print(f"FLOPs: {flops / 1e9:.2f} G")
    print(f"Parameters: {params / 1e6:.2f} M")

    # Average inference time
    print("\nMeasuring inference time...")
    for _ in range(10):  # Warmup
        _ = model(dummy_input)
    start_time = time.time()
    for _ in range(100):
        _ = model(dummy_input)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    avg_time = (time.time() - start_time) / 100 * 1000
    print(f"Average inference time: {avg_time:.2f} ms")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--num_features', type=int, default=64)
    parser.add_argument('--num_convs', type=int, default=3)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=6)
    args = parser.parse_args()

    test_model(args)