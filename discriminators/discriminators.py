import numpy as np
from .gram_discriminator import *
from .patchgan import *


class Discriminator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.epoch = 0
        self.step = 0


class GramEncoderDiscriminator(Discriminator):
    def __init__(self, img_size, img_channels=3):
        super().__init__()
        self.img_size = img_size
        self.img_size_log2 = int(np.log2(img_size))

        self.layers = nn.ModuleList([])
        for i in range(13 - self.img_size_log2, 12):
            self.layers.append(ResidualCoordConvBlock(int(min(400, 2**i)), int(min(400, 2**(i+1))), downsample=True))
        self.fromRGB = AdapterBlock(img_channels, int(min(400, 2**(13 - self.img_size_log2))))
        self.final_layer = nn.Conv2d(400, 1 + 2, 2)

    def forward(self, input):
        x = self.fromRGB(input)
        for layer in self.layers:
            x = layer(x)

        x = self.final_layer(x).reshape(x.shape[0], -1)

        prediction = x[..., 0:1]
        position = x[..., 1:]

        return prediction, position


class GramEncoderPatchDiscriminator(Discriminator):
    def __init__(self, img_size, img_channels=3, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.img_size = img_size
        self.img_size_log2 = int(np.log2(img_size))

        self.layers = nn.ModuleList([])
        for i in range(13 - self.img_size_log2, 12):
            self.layers.append(ResidualCoordConvBlock(int(min(400, 2**i)), int(min(400, 2**(i+1))), downsample=True))
        self.fromRGB = AdapterBlock(img_channels, int(min(400, 2**(13 - self.img_size_log2))))
        self.final_layer = nn.Conv2d(400, 1 + 2, 2)
        self.patchgan = NLayerDiscriminator(img_channels, min(64, int(2**(14-self.img_size_log2))), max(3, self.img_size_log2-6), norm_layer=norm_layer)

    def forward(self, input):
        x = self.fromRGB(input)
        for layer in self.layers:
            x = layer(x)

        x = self.final_layer(x).reshape(x.shape[0], -1)

        prediction = x[..., 0:1]
        position = x[..., 1:]

        patch = self.patchgan(input).reshape(x.shape[0], -1)

        return prediction, position, patch
