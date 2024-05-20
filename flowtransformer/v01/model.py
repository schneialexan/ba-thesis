import torch
import torch.nn as nn
from einops import rearrange

import warnings
warnings.filterwarnings("ignore", category=UserWarning) # enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels=3):
        super(ViT, self).__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2

        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth

        self.patch_conv = nn.Conv2d(in_channels=channels, out_channels=dim, kernel_size=patch_size, stride=patch_size)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim, 
                nhead=heads, 
                dim_feedforward=mlp_dim
                ), 
            num_layers=depth
            )

        self.mlp_head = nn.Sequential(
            nn.ConvTranspose2d(in_channels=dim, out_channels=dim // 2, kernel_size=4, stride=2, padding=1),  # Upsample
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=dim // 2, out_channels=channels, kernel_size=4, stride=2, padding=1),  # Upsample
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=channels, out_channels=channels, kernel_size=4, stride=2, padding=1),  # Upsample
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=channels, out_channels=channels, kernel_size=4, stride=2, padding=1),  # Upsample
        )

    def forward(self, img):
        p = self.patch_conv(img)
        p = rearrange(p, 'b c h w -> b (h w) c')

        p += self.pos_embedding
        p = self.transformer(p)

        # Reshape back to image
        p = rearrange(p, 'b (h w) c -> b c h w', h=(img.shape[2] // self.patch_size), w=(img.shape[3] // self.patch_size))
        
        p = self.mlp_head(p)
        return p

'''

# Usage
from dataset import generate_input  # Assuming generate_input function is in a dataset module
input_img = generate_input(100., 0.05).reshape(1, 3, 128, 128)
input_img = torch.tensor(input_img).float()
print(f'Input image shape: {input_img.shape}')

image_size = input_img.shape[2]
patch_size = 16
dim = 128
depth = 6
heads = 8
mlp_dim = 128

model = ViT(
    image_size=image_size,
    patch_size=patch_size,
    dim=dim,
    depth=depth,
    heads=heads,
    mlp_dim=mlp_dim
)

output = model(input_img)
print(f'Output shape: {output.shape}')

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(10, 5))
for i in range(3):
    ax[i].imshow(output[0, i].detach().numpy(), cmap='viridis')
    ax[i].axis('off')
plt.savefig('output.png')'''