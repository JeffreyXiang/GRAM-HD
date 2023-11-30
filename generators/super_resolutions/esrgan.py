import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, scale_factor=4, up_channels=None, to_rgb_ks=3, use_pixel_shuffle=True, interpolate_mode='bilinear', global_residual=False):
        super(RRDBNet, self).__init__()
        self.scale_factor = scale_factor
        self.use_pixel_shuffle = use_pixel_shuffle
        self.interpolate_mode = interpolate_mode
        self.global_residual = global_residual
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.num_upconvs = int(np.log2(scale_factor))
        if up_channels is None:
            up_channels = [nf] * 2 + [nf // 2] * (self.num_upconvs - 2)
        self.upconvs = nn.ModuleList([])
        if self.use_pixel_shuffle:
            for i in range(self.num_upconvs):
                if i == 0:
                    self.upconvs.append(nn.Conv2d(nf, up_channels[0] * 4, 3, 1, 1, bias=True))
                else:
                    self.upconvs.append(nn.Conv2d(up_channels[i - 1], up_channels[i] * 4, 3, 1, 1, bias=True))
        else:
            for i in range(self.num_upconvs):
                if i == 0:
                    self.upconvs.append(nn.Conv2d(nf, up_channels[0], 3, 1, 1, bias=True))
                else:
                    self.upconvs.append(nn.Conv2d(up_channels[i - 1], up_channels[i], 3, 1, 1, bias=True))
                   
        self.HRconv = nn.Conv2d(up_channels[-1], up_channels[-1], 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(up_channels[-1], out_nc, to_rgb_ks, 1, (to_rgb_ks-1)//2, bias=True)
        if self.global_residual:
            with torch.no_grad():
                self.conv_last.weight *= 0
                self.conv_last.bias *= 0

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        if self.use_pixel_shuffle:
            for i in range(self.num_upconvs):
                fea = self.lrelu(F.pixel_shuffle(self.upconvs[i](fea), 2))
        else:
            for i in range(self.num_upconvs):
                fea = F.interpolate(self.lrelu(self.upconvs[i](fea)), scale_factor=2, mode=self.interpolate_mode)

        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        assert out.shape[-1] == x.shape[-1] * self.scale_factor

        if self.global_residual:
            out = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic') + 0.2 * out

        return out