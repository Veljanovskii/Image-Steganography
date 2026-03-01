import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.block(x)
        x_down = F.max_pool2d(x, 2)
        return x, x_down


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class UNetEncoderResidual(nn.Module):
    """
    Predicts residual delta (3ch). Stego is formed as:
      stego = clamp(cover + eps * tanh(delta), 0..1)
    This strongly preserves cover image => higher PSNR(stego, cover).
    """
    def __init__(self, base: int = 32):
        super().__init__()
        self.d1 = Down(6, base)
        self.d2 = Down(base, base * 2)
        self.d3 = Down(base * 2, base * 4)
        self.d4 = Down(base * 4, base * 8)

        self.bottleneck = ConvBlock(base * 8, base * 16)

        self.u4 = Up(base * 16 + base * 8, base * 8)
        self.u3 = Up(base * 8 + base * 4, base * 4)
        self.u2 = Up(base * 4 + base * 2, base * 2)
        self.u1 = Up(base * 2 + base, base)

        self.out_delta = nn.Conv2d(base, 3, 1)

    def forward(self, cover, secret):
        x = torch.cat([cover, secret], dim=1)  # (B,6,H,W)

        s1, x = self.d1(x)
        s2, x = self.d2(x)
        s3, x = self.d3(x)
        s4, x = self.d4(x)

        x = self.bottleneck(x)

        x = self.u4(x, s4)
        x = self.u3(x, s3)
        x = self.u2(x, s2)
        x = self.u1(x, s1)

        delta = self.out_delta(x)  # unbounded residual
        return delta


class Decoder(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(3, base),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),

            ConvBlock(base * 2, base * 2),

            nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),

            ConvBlock(base, base),
            nn.Conv2d(base, 3, 1),
        )

    def forward(self, stego):
        return torch.sigmoid(self.net(stego))


class StegoNet(nn.Module):
    def __init__(self, base: int = 32, eps: float = 0.03):
        """
        eps controls maximum per-pixel perturbation (in [0,1] space).
        Smaller eps => higher PSNR(stego,cover) but harder to recover secret.
        Good starting values:
          eps=0.03 -> typically PSNR > ~35 dB
          eps=0.05 -> easier recovery but PSNR may drop
        """
        super().__init__()
        self.encoder = UNetEncoderResidual(base=base)
        self.decoder = Decoder(base=base)
        self.eps = eps

    def forward(self, cover, secret):
        delta = self.encoder(cover, secret)
        stego = torch.clamp(cover + self.eps * torch.tanh(delta), 0.0, 1.0)
        recovered = self.decoder(stego)
        return stego, recovered
