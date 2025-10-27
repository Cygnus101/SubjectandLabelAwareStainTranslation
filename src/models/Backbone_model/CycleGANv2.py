# ==============================================================================
# src/models.py  — Basic CycleGAN classes (UNet Generator + PatchGAN Critic)
# ==============================================================================

import argparse
import torch
import torch.nn as nn

# -----------------------------
# Small helpers
# -----------------------------
def kaiming_init(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class ConvNormAct(nn.Module):
    """Conv -> InstanceNorm -> Activation"""
    def __init__(self, in_ch, out_ch, k=4, s=2, p=1, use_act=True, padding_mode="reflect"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False, padding_mode=padding_mode)
        self.norm = nn.InstanceNorm2d(out_ch, affine=False, track_running_stats=False)
        self.act  = nn.ReLU(inplace=True) if use_act else nn.Identity()
    def forward(self, x): return self.act(self.norm(self.conv(x)))

class UpConvNormAct(nn.Module):
    """Upsample (nearest) + Conv -> IN -> ReLU (avoids checkerboard)."""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, use_act=True, padding_mode="zeros"):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False, padding_mode=padding_mode)
        self.norm = nn.InstanceNorm2d(out_ch, affine=False, track_running_stats=False)
        self.act  = nn.ReLU(inplace=True) if use_act else nn.Identity()
    def forward(self, x): 
        x = self.up(x)
        return self.act(self.norm(self.conv(x)))

# -----------------------------
# 1) UNet Generator (basic)
# -----------------------------
class UNetGenerator(nn.Module):
    """
    Basic U-Net: 5 downs / 5 ups with skip connections.
    Tanh output in [-1, 1]. No bells & whistles.
    """
    def __init__(self, img_channels=3, base_ch=64):
        super().__init__()
        # Encoder
        self.e1 = ConvNormAct(img_channels, base_ch,   k=4, s=2, p=1)      # 64
        self.e2 = ConvNormAct(base_ch,     base_ch*2,  k=4, s=2, p=1)      # 128
        self.e3 = ConvNormAct(base_ch*2,   base_ch*4,  k=4, s=2, p=1)      # 256
        self.e4 = ConvNormAct(base_ch*4,   base_ch*8,  k=4, s=2, p=1)      # 512
        self.e5 = ConvNormAct(base_ch*8,   base_ch*8,  k=4, s=2, p=1)      # 512 bottleneck in

        # Decoder (upsample+conv)
        self.u1 = UpConvNormAct(base_ch * 8,                 base_ch * 8)   # 512
        self.u2 = UpConvNormAct(base_ch * (8 + 8),           base_ch * 8)   # 512
        self.u3 = UpConvNormAct(base_ch * (8 + 4),           base_ch * 4)   # 256
        self.u4 = UpConvNormAct(base_ch * (4 + 2),           base_ch * 2)   # 128
        self.u5 = UpConvNormAct(base_ch * (2 + 1),           base_ch)       # 64

        # Final head
        self.head = nn.Sequential(
            nn.Conv2d(base_ch + img_channels, img_channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.Tanh(),
        )

        self.apply(kaiming_init)

    def forward(self, x):
        d1 = self.e1(x)
        d2 = self.e2(d1)
        d3 = self.e3(d2)
        d4 = self.e4(d3)
        d5 = self.e5(d4)

        u1 = self.u1(d5)
        u2 = self.u2(torch.cat([u1, d4], dim=1))
        u3 = self.u3(torch.cat([u2, d3], dim=1))
        u4 = self.u4(torch.cat([u3, d2], dim=1))
        u5 = self.u5(torch.cat([u4, d1], dim=1))
        return self.head(torch.cat([u5, x], dim=1))

# -----------------------------
# 2) PatchGAN Discriminator (Critic, WGAN-ready)
# -----------------------------
class Discriminator(nn.Module):
    """
    Basic 70×70 PatchGAN critic.
    IMPORTANT: no sigmoid at the end (works with LSGAN/hinge/WGAN-GP).
    """
    def __init__(self, in_channels=3, base_ch=64):
        super().__init__()
        layers = []
        # First block: no norm per common practice
        layers += [
            nn.Conv2d(in_channels, base_ch, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        # Subsequent blocks
        ch = base_ch
        for mult, stride in [(2, 2), (4, 2), (8, 1)]:
            layers += [
                nn.Conv2d(ch, base_ch*mult, 4, stride, 1, bias=False, padding_mode="reflect"),
                nn.InstanceNorm2d(base_ch*mult, affine=False, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = base_ch*mult
        # Final conv → 1 channel patch score map
        layers += [nn.Conv2d(ch, 1, 4, 1, 1, padding_mode="reflect")]
        self.model = nn.Sequential(*layers)
        self.apply(kaiming_init)

    def forward(self, x): 
        return self.model(x)  # raw scores

# -----------------------------
# Smoke test
# -----------------------------
def _count_params(m): return sum(p.numel() for p in m.parameters())

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--base_ch", type=int, default=64)
    return p.parse_args()

if __name__ == "__main__":
    args = _parse()
    torch.manual_seed(0)
    x = torch.randn(1, args.channels, args.size, args.size)

    G = UNetGenerator(img_channels=args.channels, base_ch=args.base_ch)
    D = Discriminator(in_channels=args.channels, base_ch=args.base_ch)

    with torch.no_grad():
        y = G(x)
        s = D(y)

    print("G out:", tuple(y.shape))
    print("D out:", tuple(s.shape))
    print("G params:", f"{_count_params(G):,}")
    print("D params:", f"{_count_params(D):,}")
