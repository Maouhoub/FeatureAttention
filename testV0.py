import argparse
import math
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

# Import the dataset. Ensure SRDataset is defined in model.py (or adjust the path accordingly).
from model import SRDataset
from modelV0 import LightweightModel

# Import the model.
# If your LightweightModel class is defined in a separate module, import it accordingly.
# For this example we assume it is defined in the training script or in another module.
# Otherwise, you can copy the class definition into this file.


try:
    from ptflops import get_model_complexity_info
except ImportError:
    print("ptflops is required to calculate FLOPs. Please install it using: pip install ptflops")
    exit(1)


def calculate_psnr(pred, target):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between a predicted
    image and the ground truth.
    Assumes pixel values are in the [0, 1] range.
    """
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse.item()))
    return psnr


def count_parameters(model):
    """
    Count all trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device:
        device = torch.device(args.device)

    # Instantiate the model with the same parameters used during training.
    model = LightweightModel(
        n_layers=args.n_layers,
        upscale_factor=args.scale,
        attn_heads=args.attn_heads,
        base_channels=args.base_channels
    )
    # Load the trained checkpoint.
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # Define transforms similar to training.
    # The low-resolution images are resized according to scale.
    lr_transform = transforms.Compose([
        transforms.Resize((args.crop_size // args.scale, args.crop_size // args.scale)),
        transforms.ToTensor()
    ])
    hr_transform = transforms.Compose([
        transforms.Resize((args.crop_size, args.crop_size)),
        transforms.ToTensor()
    ])

    # Load the test dataset.
    test_dataset = SRDataset(args.test_hr_dir, args.test_lr_dir, args.scale, lr_transform, hr_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    total_psnr = 0.0
    total_inference_time = 0.0
    total_images = 0

    # Evaluate on the test dataset.
    with torch.no_grad():
        for lr_imgs, hr_imgs in test_loader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            # Measure inference time for the current batch.
            start_time = time.time()
            outputs = model(lr_imgs)
            # If using CUDA, ensure all kernels are finished.
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            total_inference_time += (end_time - start_time)

            # Calculate PSNR for each image in the batch.
            for pred_img, gt_img in zip(outputs, hr_imgs):
                total_psnr += calculate_psnr(pred_img, gt_img)
                total_images += 1

    avg_psnr = total_psnr / total_images if total_images > 0 else 0.0
    avg_inference_time = total_inference_time / len(test_loader)

    # Count total trainable parameters.
    total_params = count_parameters(model)

    # Calculate FLOPs using ptflops.
    # Prepare a dummy input with the dimensions the model expects.
    # Note: We pass the same input shape that the model sees for LR images.
    dummy_input = (3, args.crop_size // args.scale, args.crop_size // args.scale)
    flops, params_info = get_model_complexity_info(
        model, dummy_input, as_strings=True, print_per_layer_stat=False
    )

    print(f"\n=== Test Results ===")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Total trainable parameters: {total_params}")
    print(f"Average inference time per batch: {avg_inference_time:.4f} seconds")
    print(f"Model FLOPs: {flops}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the SR model on a test dataset.")
    parser.add_argument('--test_hr_dir', type=str, required=True,
                        help="Directory containing high-resolution test images")
    parser.add_argument('--test_lr_dir', type=str, required=True,
                        help="Directory containing low-resolution test images")
    parser.add_argument('--scale', type=int, default=2, choices=[2, 3, 4],
                        help="Upscaling factor (should match training)")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_layers', type=int, default=5,
                        help="Number of residual layers (should match training)")
    parser.add_argument('--attn_heads', type=int, default=4,
                        help="Number of attention heads (should match training)")
    parser.add_argument('--base_channels', type=int, default=32,
                        help="Number of base channels (should match training)")
    parser.add_argument('--device', type=str, default=None,
                        help="Device to run inference on (e.g., 'cuda' or 'cpu')")
    parser.add_argument('--crop_size', type=int, default=128,
                        help="Crop size for HR images (should match training)")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to the trained model checkpoint (e.g., best_model.pth)")

    args = parser.parse_args()
    test_model(args)
