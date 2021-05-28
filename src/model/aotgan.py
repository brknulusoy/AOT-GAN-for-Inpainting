import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from .common import BaseNetwork
from model.blocks import VanillaConv, VanillaDeconv


class InpaintGenerator(BaseNetwork):
    def __init__(self, args):  # 1046
        super(InpaintGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(2, 64, 7),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(True)
        )

        self.middle = nn.Sequential(*[AOTBlock(256, args.rates) for _ in range(args.block_num)])

        self.decoder = nn.Sequential(
            UpConv(256, 128),
            nn.ReLU(True),
            UpConv(128, 64),
            nn.ReLU(True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1)
        )

        self.init_weights()

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.tanh(x)
        return x


class UpConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True))


class AOTBlock(nn.Module):
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                'block{}'.format(str(i).zfill(2)), 
                nn.Sequential(
                    nn.ReflectionPad2d(rate),
                    nn.Conv2d(dim, dim//4, 3, padding=0, dilation=rate),
                    nn.ReLU(True)))
        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out = [self.__getattr__(f'block{str(i).zfill(2)}')(x) for i in range(len(self.rates))]
        out = torch.cat(out, 1)
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask


def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat




# ----- discriminator -----
class Discriminator(BaseNetwork):
    def __init__(self, ):
        super(Discriminator, self).__init__()
        inc = 1
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(inc, 64, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, 4, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, stride=1, padding=1)
        )

        self.init_weights()

    def forward(self, x):
        feat = self.conv(x)
        return feat

class BaseModule(nn.Module):
    def __init__(self, conv_type):
        super().__init__()
        self.conv_type = conv_type
        if conv_type == 'gated':
            self.ConvBlock = GatedConv
            self.DeconvBlock = GatedDeconv
        elif conv_type == 'partial':
            self.ConvBlock = PartialConv
            self.DeconvBlock = PartialDeconv
        elif conv_type == 'vanilla':
            self.ConvBlock = VanillaConv
            self.DeconvBlock = VanillaDeconv

class SNTemporal(BaseModule):
    def __init__(
        self, nc_in, nf=32, norm='SN', use_sigmoid=True, use_bias=True, conv_type='vanilla',
        conv_by='3d'
    ):
        super().__init__(conv_type)
        use_bias = use_bias
        self.use_sigmoid = use_sigmoid

        ######################
        # Convolution layers #
        ######################
        self.conv1 = self.ConvBlock(
            nc_in, nf * 1, kernel_size=(1, 5, 5), stride=(1, 2, 2),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.conv2 = self.ConvBlock(
            nf * 1, nf * 2, kernel_size=(1, 5, 5), stride=(1, 2, 2),
            padding=(1, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.conv3 = self.ConvBlock(
            nf * 2, nf * 4, kernel_size=(1, 5, 5), stride=(1, 2, 2),
            padding=(1, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.conv4 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(1, 5, 5), stride=(1, 2, 2),
            padding=(1, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.conv5 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(1, 5, 5), stride=(1, 2, 2),
            padding=(1, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.conv6 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(1, 5, 5), stride=(1, 2, 2),
            padding=(1, 2, 2), bias=use_bias, norm=None, activation=None,
            conv_by=conv_by
        )
        self.sigmoid = nn.Sigmoid()

    def __call__(self, xs):
        # B, L, C, H, W = xs.shape
        xs_t = torch.transpose(xs, 1, 2)
        c1 = self.conv1(xs_t)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c6 = self.conv6(c5)
        if self.use_sigmoid:
            c6 = torch.sigmoid(c6)
        out = torch.transpose(c6, 1, 2)
        return out