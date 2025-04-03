import torch

# Create a tensor with shape (B, C, H, W)
# For example: 1 image (B=1), 2 channels (C=2), 2 rows (H=2), 3 columns (W=3)
x = torch.tensor([[[[ 1,  2,  3],
                    [ 4,  5,  6]],
                   [[ 7,  8,  9],
                    [10, 11, 12]]]])
print("Original shape (B, C, H, W):", x.shape)
print("Original tensor:\n", x)

# Step 1: Permute dimensions from (B, C, H, W) to (B, H, W, C)
x_permuted = x.permute(0, 2, 3, 1)
print("\nAfter permute (B, H, W, C):", x_permuted.shape)
print(x_permuted)

# Step 2: Reshape tensor from (B, H, W, C) to (B, H*W, C)
B, H, W, C = x_permuted.shape
x_reshaped = x_permuted.reshape(B, H * W, C)
x_reshaped = x_reshaped.permute(1, 0, 2)
print("\nAfter reshape (B, H*W, C):", x_reshaped.shape)
print(x_reshaped)
