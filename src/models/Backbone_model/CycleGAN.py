# ==============================================================================
# src/models.py
#
# This file defines the neural network architectures for the project, including
# the U-Net Generator and the PatchGAN Discriminator.
# ==============================================================================

import argparse

import torch
import torch.nn as nn

# --- Helper Block for U-Net & Discriminator ---
class ConvBlock(nn.Module):
    """A standard Down-Convolution or Up-Convolution block."""
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        if down:
            self.conv = nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, **kwargs)
        
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

# --- 1. GENERATOR (U-Net Architecture) ---
class UNetGenerator(nn.Module):
    """
    The Generator network based on the U-Net architecture. It uses an
    encoder-decoder structure with skip connections to pass spatial information
    from the downsampling path to the upsampling path, preserving details.
    """
    def __init__(self, img_channels=3, features=64):
        super().__init__()

        # --- Encoder (Downsampling Path) ---
        self.encoder1 = ConvBlock(img_channels, features, kernel_size=4, stride=2, padding=1) # 64x256x256
        self.encoder2 = ConvBlock(features, features * 2, kernel_size=4, stride=2, padding=1) # 128x128x128
        self.encoder3 = ConvBlock(features * 2, features * 4, kernel_size=4, stride=2, padding=1) # 256x64x64
        self.encoder4 = ConvBlock(features * 4, features * 8, kernel_size=4, stride=2, padding=1) # 512x32x32
        self.bottleneck = ConvBlock(features * 8, features * 8, kernel_size=4, stride=2, padding=1) # 512x16x16
        self.bottleneck2 = ConvBlock(features*8, features*8, kernel_size=4, stride=2, padding=1) # 512x8x8

        # --- Decoder (Upsampling Path) ---
        self.up0 = ConvBlock(features*8, features*8, down=False, kernel_size=4, stride=2, padding=1)
        self.up1 = ConvBlock(features * 8 * 2, features * 8, down=False, kernel_size=4, stride=2, padding=1)
        self.up2 = ConvBlock(features * 8 * 2, features * 4, down=False, kernel_size=4, stride=2, padding=1)
        self.up3 = ConvBlock(features * 4 * 2, features * 2, down=False, kernel_size=4, stride=2, padding=1)
        self.up4 = ConvBlock(features * 2 * 2, features, down=False, kernel_size=4, stride=2, padding=1)
        
        # --- Final Output Layer ---
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features * 2, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(), # Tanh squashes output to [-1, 1]
        )

    def forward(self, x):
        # Pass through encoder, saving outputs for skip connections
        d1 = self.encoder1(x)
        d2 = self.encoder2(d1)
        d3 = self.encoder3(d2)
        d4 = self.encoder4(d3)
        b1 = self.bottleneck(d4)
        b2 = self.bottleneck2(b1)

        # Pass through decoder, concatenating skip connections                                                                                                                                                                                                                                                                                                                                                                          
        u0 = self.up0(b2)
        u1 = self.up1(torch.cat([u0, b1], dim=1))
        u2 = self.up2(torch.cat([u1, d4], dim=1))
        u3 = self.up3(torch.cat([u2, d3], dim=1))
        u4 = self.up4(torch.cat([u3, d2], dim=1))
        return self.final(torch.cat([u4, d1], dim=1))

# --- 2. DISCRIMINATOR (PatchGAN) ---
class Discriminator(nn.Module):
    """The PatchGAN Discriminator network (critic)."""
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
            self._discriminator_block(features[0], features[1], stride=2),
            self._discriminator_block(features[1], features[2], stride=2),
            self._discriminator_block(features[2], features[3], stride=1), # Last block has stride 1
            nn.Conv2d(features[3], 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
        )
        
    def _discriminator_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, x):
        return self.model(x)


def _count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CycleGAN model smoke test.")
    parser.add_argument("--channels", type=int, default=3, help="Input image channels.")
    parser.add_argument("--size", type=int, default=256, help="Square image size for the dry run tensor.")
    parser.add_argument("--features", type=int, default=64, help="Base channel width for the U-Net generator.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cpu")
    generator = UNetGenerator(img_channels=args.channels, features=args.features).to(device)
    discriminator = Discriminator(in_channels=args.channels).to(device)

    sample = torch.randn(1, args.channels, args.size, args.size, device=device)

    with torch.no_grad():
        fake = generator(sample)
        disc_out = discriminator(fake)

    print(f"Generator output shape: {tuple(fake.shape)}")
    print(f"Discriminator output shape: {tuple(disc_out.shape)}")
    print(f"Generator parameters: {_count_parameters(generator):,}")
    print(f"Discriminator parameters: {_count_parameters(discriminator):,}")
