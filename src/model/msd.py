# https://arxiv.org/pdf/1910.06711.pdf, 2 The MelGAN Model
from torch import nn

from src.model.utils import WNConv1d, SNConv1d
from src.utils import LRELU_SLOPE


class DiscriminatorBlock(nn.Module):
    def __init__(self, channels_list, kernels, strides, groups_list, weight_normalization=True):
        super().__init__()

        layers = []
        for i in range(len(channels_list)):
            in_channels = 1 if i == 0 else channels_list[i - 1]
            NConv1d = WNConv1d if weight_normalization else SNConv1d
            layers.append(NConv1d(in_channels, channels_list[i], kernels[i], strides[i], groups=groups_list[i], padding=kernels[i] // 2))

            if i + 1 < len(channels_list):
                layers.append(nn.LeakyReLU(LRELU_SLOPE))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        feature_maps = []
        for layer in self.layers:
            x = layer(x)
            if type(layer) == type(nn.LeakyReLU()):
                feature_maps.append(x)

        feature_maps.append(x)
        return x, feature_maps


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, channels_list, kernels, strides, groups_list):
        super().__init__()

        blocks = []
        blocks.append(DiscriminatorBlock(channels_list, kernels, strides, groups_list, False))
        blocks += [DiscriminatorBlock(channels_list, kernels, strides, groups_list) for _ in range(2)]
        self.discriminator_blocks = nn.ModuleList(blocks)
        self.downsamples = nn.ModuleList([nn.AvgPool1d(4, 2, 2), nn.AvgPool1d(4, 2, 2)])

    def forward(self, wav):
        feature_maps = []
        x = wav
        for i, disc in enumerate(self.discriminator_blocks):
            x, disc_feature_maps = disc(x)
            feature_maps += disc_feature_maps
            if i + 1 < len(self.discriminator_blocks):
                x = self.downsamples[i](x)
        return x, feature_maps
