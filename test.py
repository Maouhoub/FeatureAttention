import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import time
import numpy as np
from ptflops import get_model_complexity_info

from model import ViTSR


# Add your ViTSR class definition here (from previous code)
# ... [Include the ViTSR class definition from previous code here] ...

def calculate_psnr(output, target):
    mse = torch.mean((output - target) ** 2)
    return 10 * torch.log10(1 / mse)


def evaluate_model(model, lr_dir, hr_dir, upscale_factor, device, repetitions=100):
    # Create transforms
    lr_transform = transforms.Compose([
        transforms.Resize((128 // upscale_factor, 128 // upscale_factor)),
        transforms.ToTensor()
    ])

    hr_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Metric storage
    psnrs = []
    timings = []

    # Warm up GPU
    dummy_input = torch.randn(1, 3, 128 // upscale_factor, 128 // upscale_factor).to(device)
    for _ in range(10):
        _ = model(dummy_input)

    # Process images
    for img_name in os.listdir(lr_dir):
        lr_path = os.path.join(lr_dir, img_name)
        hr_path = os.path.join(hr_dir, img_name).replace("LR", 'HR')

        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")

        lr_tensor = lr_transform(lr_img).unsqueeze(0).to(device)
        hr_tensor = hr_transform(hr_img).unsqueeze(0).to(device)

        # Inference timing
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()

        times = []
        for _ in range(repetitions):
            starter.record()
            with torch.no_grad():
                _ = model(lr_tensor)
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))

        timings.append(np.mean(times[10:]))  # Skip first 10 measurements

        # PSNR calculation
        with torch.no_grad():
            sr_tensor = model(lr_tensor)

        psnr = calculate_psnr(sr_tensor, hr_tensor)
        psnrs.append(psnr.item())

    return np.mean(psnrs), np.mean(timings)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lr_dir", type=str, required=True)
    parser.add_argument("--hr_dir", type=str, required=True)
    parser.add_argument("--patch_size", type=int, required=True)
    parser.add_argument("--embed_dim", type=int, required=True)
    parser.add_argument("--num_heads", type=int, required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--mlp_dim", type=int, required=True)
    parser.add_argument("--upscale_factor", type=int, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = ViTSR(
        in_channels=3,
        embed_dim=args.embed_dim,
        patch_size=args.patch_size,
        num_heads=args.num_heads,
        depth=args.depth,
        mlp_dim=args.mlp_dim,
        upscale_factor=args.upscale_factor
    ).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # Calculate FLOPs and parameters
    input_shape = (3, 128 // args.upscale_factor, 128 // args.upscale_factor)
    macs, params = get_model_complexity_info(model, input_shape, as_strings=False, print_per_layer_stat=False)

    # Calculate metrics
    mean_psnr, mean_time = evaluate_model(model, args.lr_dir, args.hr_dir, args.upscale_factor, device)

    print("\nEvaluation Results:")
    print(f"PSNR: {mean_psnr:.2f} dB")
    print(f"Inference Time: {mean_time:.2f} ms")
    print(f"FLOPs: {macs / 1e9:.2f} GMac")
    print(f"Parameters: {params / 1e6:.2f} M")