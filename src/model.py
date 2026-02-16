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
        # in_ch = channels after concat(skip + upsampled)
        self.block = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class UNetEncoder(nn.Module):
    """
    Takes cover+secret (6ch) -> produces stego (3ch).
    """
    def __init__(self, base: int = 32):
        super().__init__()
        self.d1 = Down(6, base)           # 256 -> 128
        self.d2 = Down(base, base * 2)    # 128 -> 64
        self.d3 = Down(base * 2, base * 4)# 64 -> 32
        self.d4 = Down(base * 4, base * 8)# 32 -> 16

        self.bottleneck = ConvBlock(base * 8, base * 16)

        self.u4 = Up(base * 16 + base * 8, base * 8)
        self.u3 = Up(base * 8 + base * 4, base * 4)
        self.u2 = Up(base * 4 + base * 2, base * 2)
        self.u1 = Up(base * 2 + base, base)

        self.out = nn.Conv2d(base, 3, 1)

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

        stego = torch.sigmoid(self.out(x))  # keep in [0,1]
        return stego


class Decoder(nn.Module):
    """
    Takes stego (3ch) -> recovers secret (3ch).
    Simple CNN (can be upgraded to U-Net too, but this is stable and works well).
    """
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
        secret_hat = torch.sigmoid(self.net(stego))
        return secret_hat


class StegoNet(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        self.encoder = UNetEncoder(base=base)
        self.decoder = Decoder(base=base)

    def forward(self, cover, secret):
        stego = self.encoder(cover, secret)
        recovered = self.decoder(stego)
        return stego, recovered
