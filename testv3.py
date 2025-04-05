import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from PIL import Image
from thop import profile

from modelv3 import FeatureAttentionSR, ImageSRDataset


def test_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = FeatureAttentionSR(
        num_convs=args.num_convs,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        upscaling_factor=args.upscaling_factor
    ).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # Test dataset (same structure as training)
    test_dataset = ImageSRDataset(args.test_dir, args.hr_size, args.upscaling_factor)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 1. Calculate PSNR
    mse = torch.nn.MSELoss()
    psnr_values = []
    with torch.no_grad():
        for lr, hr in test_loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            psnr = 10 * torch.log10(1.0 / mse(sr, hr))
            psnr_values.append(psnr.item())
    avg_psnr = np.mean(psnr_values)

    # 2. Measure Inference Time
    inputs = next(iter(test_loader))[0].to(device)
    times = []
    for _ in range(100):  # Warm-up
        _ = model(inputs)

    for _ in range(100):
        start = time.time()
        _ = model(inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.time() - start)
    avg_time = np.mean(times) * 1000  # Convert to ms

    # 3. Parameter Count
    params = sum(p.numel() for p in model.parameters())

    # 4. Calculate FLOPs
    lr_size = args.hr_size // args.upscaling_factor
    dummy_input = torch.randn(1, 3, lr_size, lr_size).to(device)
    flops, _ = profile(model, inputs=(dummy_input,))

    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average Inference Time: {avg_time:.2f} ms")
    print(f"Parameter Count: {params / 1e6:.2f}M")
    print(f"FLOPs: {flops / 1e9:.2f} G")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--hr_size", type=int, default=128)
    parser.add_argument("--upscaling_factor", type=int, default=2)
    parser.add_argument("--num_convs", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=96)
    parser.add_argument("--num_heads", type=int, default=4)
    args = parser.parse_args()

    test_model(args)