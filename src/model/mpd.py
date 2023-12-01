# https://arxiv.org/pdf/2010.05646.pdf, 2 HiFi-GAN, Appendix A
import torch
from torch import nn
import torch.nn.functional as F

from src.model.utils import WNConv2d
from src.model.utils import LRELU_SLOPE


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, p):
        super().__init__()

        self.p = p
        layers = []
        kernel = (5, 1)
        stride = (3, 1)
        channels = [32, 64, 128, 512]
        for l in range(4):
            # in_channels = 1 if l == 0 else 2 ** (5 + l - 1)
            # layers.append(WNConv2d(in_channels, 2 ** (5 + l), kernel, stride, padding=(2, 0)))
            in_channels = 1 if l == 0 else channels[l - 1]
            layers.append(WNConv2d(in_channels, channels[l], kernel, stride, padding=(2, 0)))
            layers.append(nn.LeakyReLU(LRELU_SLOPE))
        # layers.append(WNConv2d(256, 1024, kernel, padding=(2, 0)))
        layers.append(WNConv2d(512, 1024, kernel, padding=(2, 0)))
        layers.append(nn.LeakyReLU(LRELU_SLOPE))
        layers.append(WNConv2d(1024, 1, (3, 1), padding=(1, 0)))
        self.layers = nn.ModuleList(layers)

    def forward(self, wav):
        if x.shape[-1] % self.period != 0:
            wav = F.pad(wav, (0, self.p - wav.shape[-1] % self.p), mode="reflect")
        x = wav.reshape(wav.shape[0], wav.shape[1], wav.shape[-1] // self.p, self.p)
        feature_maps = []

        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU):
                feature_maps.append(x)

        feature_maps.append(x)
        x = torch.flatten(x, 1, -1)

        return x, feature_maps
