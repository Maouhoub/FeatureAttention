import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import argparse

from model import ViTSR


# Load trained model
def load_model(model_path, in_channels, embed_dim, patch_size, num_heads, depth, mlp_dim, upscale_factor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTSR(in_channels, embed_dim, patch_size, num_heads, depth, mlp_dim, upscale_factor).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


# Custom Dataset for Testing
class TestDataset(Dataset):
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


# Compute PSNR and SSIM
def evaluate(model, dataloader, device):
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    with torch.no_grad():
        for lr_imgs, hr_imgs in dataloader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            sr_imgs = model(lr_imgs).clamp(0, 1)  # Ensure values are in [0,1]

            for i in range(lr_imgs.size(0)):
                hr_img_np = hr_imgs[i].permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C)
                sr_img_np = sr_imgs[i].permute(1, 2, 0).cpu().numpy()

                psnr_value = psnr(hr_img_np, sr_img_np, data_range=1.0)
                ssim_value = ssim(hr_img_np, sr_img_np, data_range=1.0, multichannel=True)

                total_psnr += psnr_value
                total_ssim += ssim_value
                count += 1

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    return avg_psnr, avg_ssim


# Testing
def test_model(model_path, lr_dir, hr_dir, upscale_factor):
    model, device = load_model(model_path, 3, 64, 4, 4, 6, 128, upscale_factor)

    lr_transform = transforms.Compose([
        transforms.Resize((128 // upscale_factor, 128 // upscale_factor)),
        transforms.ToTensor()
    ])

    hr_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    test_dataset = TestDataset(lr_dir, hr_dir, upscale_factor, lr_transform, hr_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    evaluate(model, test_loader, device)


# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--lr_dir", type=str, required=True, help="Path to low-resolution images")
    parser.add_argument("--hr_dir", type=str, required=True, help="Path to high-resolution images")
    parser.add_argument("--upscale_factor", type=int, default=2, help="Upscaling factor")
    args = parser.parse_args()

    test_model(args.model_path, args.lr_dir, args.hr_dir, args.upscale_factor)
