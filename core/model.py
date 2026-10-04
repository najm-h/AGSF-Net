from torch import nn
import torch
from torchvision import models


# ==========================================================
# Basic Blocks
# ==========================================================

def conv3x3(in_ch, out_ch, s=1, d=1):
    p = (3 // 2) * d
    return nn.Conv2d(in_ch, out_ch, 3, stride=s, padding=p, dilation=d, bias=True)


class ConvINLReLU(nn.Module):
    def __init__(self, in_ch, out_ch, s=1, d=1, negative_slope=0.1):
        super().__init__()
        self.conv = conv3x3(in_ch, out_ch, s=s, d=d)
        self.inorm = nn.InstanceNorm2d(out_ch, affine=True)
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x):
        return self.act(self.inorm(self.conv(x)))


# ==========================================================
# Multi-Residual Module
# ==========================================================

class MRM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        c1 = ch // 2
        c2 = ch
        c3 = int(1.5 * ch)

        self.b1 = ConvINLReLU(ch, c1, d=1)
        self.b2 = ConvINLReLU(c1, c2, d=1)
        self.b3 = ConvINLReLU(c2, c3, d=1)

        self.merge = ConvINLReLU(c1 + c2 + c3, ch, d=1)
        self.skip = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        f1 = self.b1(x)
        f2 = self.b2(f1)
        f3 = self.b3(f2)
        return self.merge(torch.cat([f1, f2, f3], dim=1)) + self.skip(x)


# ==========================================================
# Downsampling Module
# ==========================================================

class FeatureBlock(nn.Module):
    """MRM wrapper retained to match the released checkpoint keys."""

    def __init__(self, ch):
        super().__init__()
        self.block = MRM(ch)

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.refine = FeatureBlock(out_ch)

    def forward(self, x):
        return self.refine(self.conv(self.pool(x)))


# ==========================================================
# Transpose Convolution Upsampling Module
# ==========================================================

class TCUM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

    def forward(self, x):
        return self.up(x)


# ==========================================================
# Original SOS Feature Reconstruction Module
# ==========================================================

class SOSFM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.refine = MRM(out_ch)
        self.align = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, enc_feat_same_level, up_feat):
        i_n = self.align(enc_feat_same_level)
        fused = i_n + up_feat
        return self.refine(fused) - up_feat


# ==========================================================
# Original Dual-Domain Fusion Module
# ==========================================================

class DualDomainFusion(nn.Module):
    """
    Original DDFM:
    spatial branch + frequency branch with channel-wise magnitude modulation.
    No ASFG gate is used.
    """
    def __init__(self, ch, r=4):
        super().__init__()

        self.spatial = nn.Sequential(
            ConvINLReLU(ch, ch),
            ConvINLReLU(ch, ch)
        )

        mid = max(8, ch // r)

        self.freq_ca = nn.Sequential(
            nn.Conv2d(ch, mid, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid, ch, 1),
            nn.Sigmoid()
        )

        self.fuse = ConvINLReLU(2 * ch, ch)
        self.res = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        orig_dtype = x.dtype
        x32 = x.to(torch.float32)

        # Spatial branch
        x_sp = self.spatial(x32)

        # Frequency branch
        X = torch.fft.fft2(x32, dim=(-2, -1))
        mag = torch.abs(X)
        pha = torch.angle(X)

        w = self.freq_ca(mag.mean(dim=(-2, -1), keepdim=True))
        mag2 = mag * w

        X2 = torch.polar(mag2, pha)
        x_fr = torch.fft.ifft2(X2, dim=(-2, -1)).real

        # Fusion
        out32 = self.fuse(torch.cat([x_sp, x_fr], dim=1))
        out32 = self.res(out32)

        out = out32.to(orig_dtype)

        # Original residual fusion
        return out + x


# ==========================================================
# Convolutional Block Attention Module
# ==========================================================

class ChannelAttention(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        mid = max(8, ch // r)

        self.mlp = nn.Sequential(
            nn.Conv2d(ch, mid, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid, ch, 1, bias=True)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=(2, 3), keepdim=True)
        mx = torch.amax(x, dim=(2, 3), keepdim=True)
        w = self.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * w


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx = torch.amax(x, dim=1, keepdim=True)
        w = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * w


class CBAM(nn.Module):
    """
    CBAM-only attention:
    channel attention + spatial attention.
    """
    def __init__(self, ch):
        super().__init__()
        self.ca = ChannelAttention(ch)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


# Compatibility name used while producing the released checkpoint.
CBAMLite = CBAM


# ==========================================================
# Lightweight Detail Refinement Module
# ==========================================================

class LDRM(nn.Module):
    """
    Lightweight Detail Refinement Module.

    This module is placed before the output head.
    It uses depthwise convolution to refine local details
    with small computational overhead.
    """
    def __init__(self, ch):
        super().__init__()

        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=True)
        self.pw = nn.Conv2d(ch, ch, 1, bias=True)

        self.gate = nn.Sequential(
            nn.Conv2d(ch, ch, 1, bias=True),
            nn.Sigmoid()
        )

        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        detail = self.act(self.pw(self.dw(x)))
        g = self.gate(detail)
        return x + g * detail


# ==========================================================
# AGSF-Net with LDRM
# ==========================================================

class AGSFNet(nn.Module):
    """
    Full AGSF-Net used in the paper.

    Bottleneck order: MRM -> DDFM -> CBAM -> Conv.
    """
    def __init__(
        self,
        base=32,
        in_ch=3,
        out_ch=3,
    ):
        super().__init__()

        c1 = base
        c2 = base * 2
        c3 = base * 4
        c4 = base * 8
        c5 = base * 8

        # Encoder. Attribute names intentionally match the training model so
        # that the official checkpoint can be loaded with strict=True.
        self.enc1_conv = nn.Sequential(
            nn.Conv2d(in_ch, c1, 3, padding=1, bias=True),
            nn.InstanceNorm2d(c1, affine=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.enc1_refine = FeatureBlock(c1)

        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)

        # Attention-guided spatial--frequency bottleneck
        self.bot_mrm = MRM(c4)
        self.bot_ddfm = DualDomainFusion(c4)
        self.bot_cbam = CBAM(c4)
        self.bot_out = nn.Conv2d(c4, c5, 3, padding=1, bias=True)

        # Decoder with original SOSFM
        self.up3 = TCUM(c5, c4)
        self.dec3 = SOSFM(in_ch=c3, out_ch=c4)

        self.up2 = TCUM(c4, c3)
        self.dec2 = SOSFM(in_ch=c2, out_ch=c3)

        self.up1 = TCUM(c3, c2)
        self.dec1 = SOSFM(in_ch=c1, out_ch=c2)

        # Detail refinement before output
        self.ldrm = LDRM(c2)

        # Output head
        self.head = nn.Conv2d(c2, out_ch, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1_conv(x)
        e1 = self.enc1_refine(e1)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)

        # Proposed bottleneck: MRM -> DDFM -> CBAM
        b = self.bot_mrm(e4)
        b = self.bot_ddfm(b)
        b = self.bot_cbam(b)
        b = self.bot_out(b)

        # Decoder
        d3 = self.up3(b)
        d3 = self.dec3(e3, d3)
        d2 = self.up2(d3)
        d2 = self.dec2(e2, d2)
        d1 = self.up1(d2)
        d1 = self.dec1(e1, d1)

        # Detail refinement
        d1 = self.ldrm(d1)

        # Direct output prediction
        y = self.head(d1)
        return torch.tanh(y) * 0.5 + 0.5


# ==========================================================
# Perceptual VGG19 Feature Extractor
# ==========================================================

class PerceptualVGG19(nn.Module):
    def __init__(self):
        super().__init__()

        try:
            weights = models.VGG19_Weights.IMAGENET1K_FEATURES
            vgg = models.vgg19(weights=weights).features
        except Exception:
            vgg = models.vgg19(pretrained=True).features

        for p in vgg.parameters():
            p.requires_grad = False

        self.feat = nn.Sequential(*[vgg[i] for i in range(36)])

        for p in self.feat.parameters():
            p.requires_grad = False

    def forward(self, x):
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        return self.feat((x - mean) / std)
