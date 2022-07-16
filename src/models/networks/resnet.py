# https://github.com/LMescheder/GAN_stability/blob/master/gan_training/models/resnet2.py
import torch
import torch.utils.data
import torch.utils.data.distributed
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1 * dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out


class ResnetDecoderMap(nn.Module):
    def __init__(self, imgsize, nfilter=64, **kwargs):
        super().__init__()
        s0 = self.s0 = imgsize // 32
        nf = self.nf = nfilter
        self.input_dim = 16 * nf * s0 * s0

        # Submodules
        self.resnet_0_0 = ResBlock(16 * nf, 16 * nf)
        self.resnet_0_1 = ResBlock(16 * nf, 16 * nf)

        self.resnet_1_0 = ResBlock(16 * nf, 16 * nf)
        self.resnet_1_1 = ResBlock(16 * nf, 16 * nf)

        self.resnet_2_0 = ResBlock(16 * nf, 8 * nf)
        self.resnet_2_1 = ResBlock(8 * nf, 8 * nf)

        self.resnet_3_0 = ResBlock(8 * nf, 4 * nf)
        self.resnet_3_1 = ResBlock(4 * nf, 4 * nf)

        self.resnet_4_0 = ResBlock(4 * nf, 2 * nf)
        self.resnet_4_1 = ResBlock(2 * nf, 2 * nf)

        self.resnet_5_0 = ResBlock(2 * nf, 1 * nf)
        self.resnet_5_1 = ResBlock(1 * nf, 1 * nf)

        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    def forward(self, z):
        out = z.view(z.shape[0], 16 * self.nf, self.s0, self.s0)
        # (16 * nf, s0, s0)
        out = self.resnet_0_0(out)
        out = self.resnet_0_1(out)

        out = F.interpolate(out, scale_factor=2)
        # (16 * nf, 2 * s0, 2 * s0)
        out = self.resnet_1_0(out)
        out = self.resnet_1_1(out)

        out = F.interpolate(out, scale_factor=2)
        # (16 * nf, 4 * s0, 4 * s0)
        out = self.resnet_2_0(out)
        out = self.resnet_2_1(out)

        out = F.interpolate(out, scale_factor=2)
        # (8 * nf, 8 * s0, 8 * s0)
        out = self.resnet_3_0(out)
        out = self.resnet_3_1(out)

        out = F.interpolate(out, scale_factor=2)
        # (4 * nf, 16 * s0, 16 * s0)
        out = self.resnet_4_0(out)
        out = self.resnet_4_1(out)

        out = F.interpolate(out, scale_factor=2)
        # (2 * nf, 32 * s0, 32 * s0)
        out = self.resnet_5_0(out)
        out = self.resnet_5_1(out)

        out = self.conv_img(actvn(out))
        # (3, 32 * s0, 32 * s0)
        out = torch.tanh(out)

        return out


class ResnetEncoderMap(nn.Module):
    def __init__(self, imgsize, nfilter=64, **kwargs):
        super().__init__()
        s0 = self.s0 = imgsize // 32
        nf = self.nf = nfilter
        self.output_dim = 16 * nf * s0 * s0

        # Submodules
        self.conv_img = nn.Conv2d(3, 1 * nf, 3, padding=1)

        self.resnet_0_0 = ResBlock(1 * nf, 1 * nf)
        self.resnet_0_1 = ResBlock(1 * nf, 2 * nf)

        self.resnet_1_0 = ResBlock(2 * nf, 2 * nf)
        self.resnet_1_1 = ResBlock(2 * nf, 4 * nf)

        self.resnet_2_0 = ResBlock(4 * nf, 4 * nf)
        self.resnet_2_1 = ResBlock(4 * nf, 8 * nf)

        self.resnet_3_0 = ResBlock(8 * nf, 8 * nf)
        self.resnet_3_1 = ResBlock(8 * nf, 16 * nf)

        self.resnet_4_0 = ResBlock(16 * nf, 16 * nf)
        self.resnet_4_1 = ResBlock(16 * nf, 16 * nf)

        self.resnet_5_0 = ResBlock(16 * nf, 16 * nf)
        self.resnet_5_1 = ResBlock(16 * nf, 16 * nf)

    def forward(self, x):
        out = self.conv_img(x)

        out = self.resnet_0_0(out)
        out = self.resnet_0_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_1_0(out)
        out = self.resnet_1_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_2_0(out)
        out = self.resnet_2_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_3_0(out)
        out = self.resnet_3_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_4_0(out)
        out = self.resnet_4_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_5_0(out)
        out = self.resnet_5_1(out)

        out = out.view(x.shape[0], 16 * self.nf * self.s0 * self.s0)

        return out


class CUBSentResnetDecoderMap(nn.Module):
    def __init__(self, nfilter=32, **kwargs):
        super().__init__()
        imgsize = (32, 128)
        s0 = self.s0 = imgsize[0] // 8
        s1 = self.s1 = imgsize[1] // 32

        nf = self.nf = nfilter
        self.input_dim = 16 * nf * s0 * s1

        # Submodules
        self.resnet_0_0 = ResBlock(16 * nf, 16 * nf)
        self.resnet_0_1 = ResBlock(16 * nf, 16 * nf)

        self.resnet_1_0 = ResBlock(16 * nf, 16 * nf)
        self.resnet_1_1 = ResBlock(16 * nf, 16 * nf)

        self.resnet_2_0 = ResBlock(16 * nf, 8 * nf)
        self.resnet_2_1 = ResBlock(8 * nf, 8 * nf)

        self.resnet_3_0 = ResBlock(8 * nf, 4 * nf)
        self.resnet_3_1 = ResBlock(4 * nf, 4 * nf)

        self.resnet_4_0 = ResBlock(4 * nf, 2 * nf)
        self.resnet_4_1 = ResBlock(2 * nf, 2 * nf)

        self.resnet_5_0 = ResBlock(2 * nf, 1 * nf)
        self.resnet_5_1 = ResBlock(1 * nf, 1 * nf)

        self.conv_img = nn.Conv2d(nf, 1, 3, padding=1)

    def forward(self, z):
        out = z.view(z.shape[0], 16 * self.nf, self.s0, self.s1)
        # (16 * nf, 4, 4)

        out = self.resnet_0_0(out)
        out = self.resnet_0_1(out)

        out = F.interpolate(out, scale_factor=(1, 2))
        # (16 * nf, 4, 8)
        out = self.resnet_1_0(out)
        out = self.resnet_1_1(out)

        out = F.interpolate(out, scale_factor=(1, 2))
        # (8 * nf, 4, 16)
        out = self.resnet_2_0(out)
        out = self.resnet_2_1(out)

        out = F.interpolate(out, scale_factor=2)
        # (4 * nf, 8, 32)
        out = self.resnet_3_0(out)
        out = self.resnet_3_1(out)

        out = F.interpolate(out, scale_factor=2)
        # (2 * nf, 16, 64)
        out = self.resnet_4_0(out)
        out = self.resnet_4_1(out)

        out = F.interpolate(out, scale_factor=2)
        # (1 * nf, 32, 128)
        out = self.resnet_5_0(out)
        out = self.resnet_5_1(out)

        out = self.conv_img(actvn(out))
        out = torch.tanh(out)

        return out


class CUBSentResnetEncoderMap(nn.Module):
    def __init__(self, nfilter=32, **kwargs):
        super().__init__()
        imgsize = (32, 128)
        s0 = self.s0 = imgsize[0] // 8
        s1 = self.s1 = imgsize[1] // 32

        nf = self.nf = nfilter
        self.output_dim = 16 * nf * s0 * s1

        # Submodules
        self.conv_img = nn.Conv2d(1, 1 * nf, 3, padding=1)

        self.resnet_0_0 = ResBlock(1 * nf, 1 * nf)
        self.resnet_0_1 = ResBlock(1 * nf, 2 * nf)

        self.resnet_1_0 = ResBlock(2 * nf, 2 * nf)
        self.resnet_1_1 = ResBlock(2 * nf, 4 * nf)

        self.resnet_2_0 = ResBlock(4 * nf, 4 * nf)
        self.resnet_2_1 = ResBlock(4 * nf, 8 * nf)

        self.resnet_3_0 = ResBlock(8 * nf, 8 * nf)
        self.resnet_3_1 = ResBlock(8 * nf, 16 * nf)

        self.resnet_4_0 = ResBlock(16 * nf, 16 * nf)
        self.resnet_4_1 = ResBlock(16 * nf, 16 * nf)

        self.resnet_5_0 = ResBlock(16 * nf, 16 * nf)
        self.resnet_5_1 = ResBlock(16 * nf, 16 * nf)

    def forward(self, x):
        out = self.conv_img(x)
        # (1 * nf, 32, 128)
        out = self.resnet_0_0(out)
        out = self.resnet_0_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        # (2 * nf, 16, 64)
        out = self.resnet_1_0(out)
        out = self.resnet_1_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        # (4 * nf, 8, 32)
        out = self.resnet_2_0(out)
        out = self.resnet_2_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        # (8 * nf, 4, 16)
        out = self.resnet_3_0(out)
        out = self.resnet_3_1(out)

        out = F.avg_pool2d(out, (1, 3), stride=(1, 2), padding=(0, 1))
        # (16 * nf, 4, 8)
        out = self.resnet_4_0(out)
        out = self.resnet_4_1(out)

        out = F.avg_pool2d(out, (1, 3), stride=(1, 2), padding=(0, 1))
        # (16 * nf, 4, 4)
        out = self.resnet_5_0(out)
        out = self.resnet_5_1(out)

        out = out.view(x.shape[0], 16 * self.nf * self.s0 * self.s0)

        return out
