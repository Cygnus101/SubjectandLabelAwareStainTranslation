"""CycleGANv3
=================

A variant of the CycleGAN generator / discriminator stack that mirrors the
existing implementation but replaces every transpose-convolution based upsampling
step with an (Upsample → Conv2d) block to reduce checkerboard artifacts.
"""

from __future__ import annotations

import argparse
from typing import Sequence

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Downsampling block with reflection padding + instance norm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        use_act: bool = True,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode="reflect",
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.block(x)


class UpSampleBlock(nn.Module):
    """Upsampling block that avoids transpose convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        use_act: bool = True,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.block(x)


class UNetGenerator(nn.Module):
    """U-Net generator with bilinear upsampling based decoding path."""

    def __init__(self, img_channels: int = 3, features: int = 64) -> None:
        super().__init__()
        self.enc1 = ConvBlock(img_channels, features)
        self.enc2 = ConvBlock(features, features * 2)
        self.enc3 = ConvBlock(features * 2, features * 4)
        self.enc4 = ConvBlock(features * 4, features * 8)
        self.bottleneck1 = ConvBlock(features * 8, features * 8)
        self.bottleneck2 = ConvBlock(features * 8, features * 8)

        self.up0 = UpSampleBlock(features * 8, features * 8)
        self.up1 = UpSampleBlock(features * 8 * 2, features * 8)
        self.up2 = UpSampleBlock(features * 8 * 2, features * 4)
        self.up3 = UpSampleBlock(features * 4 * 2, features * 2)
        self.up4 = UpSampleBlock(features * 2 * 2, features)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(features * 2, img_channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.enc1(x)
        d2 = self.enc2(d1)
        d3 = self.enc3(d2)
        d4 = self.enc4(d3)
        b1 = self.bottleneck1(d4)
        b2 = self.bottleneck2(b1)

        u0 = self.up0(b2)
        u1 = self.up1(torch.cat([u0, b1], dim=1))
        u2 = self.up2(torch.cat([u1, d4], dim=1))
        u3 = self.up3(torch.cat([u2, d3], dim=1))
        u4 = self.up4(torch.cat([u3, d2], dim=1))
        return self.final(torch.cat([u4, d1], dim=1))


class Discriminator(nn.Module):
    """PatchGAN discriminator identical to the original CycleGAN variant."""

    def __init__(self, in_channels: int = 3, features: Sequence[int] = (64, 128, 256, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        in_ch = features[0]
        for idx, out_ch in enumerate(features[1:]):
            stride = 1 if idx == len(features) - 2 else 2
            layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=False, padding_mode="reflect"),
                    nn.InstanceNorm2d(out_ch),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            in_ch = out_ch

        layers.append(nn.Conv2d(in_ch, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.model(x)


def _count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CycleGANv3 smoke test.")
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
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
