# https://arxiv.org/pdf/2010.05646.pdf, 2 HiFi-GAN, Appendix A
from torch import nn

from src.utils.mel_spectrogram import MelSpectrogramConfig
from src.utils import LRELU_SLOPE


class ResidualBlock(nn.Module):
    def __init__(self, channels, k_rn, D_rn):
        super().__init__()

        layers = []
        for m in range(len(D_rn)):
            block = []
            for l in range(len(D_rn[m])):
                block.append(nn.LeakyReLU(LRELU_SLOPE))
                block.append(nn.Conv1d(channels, channels, k_rn, padding="same", dilation=D_rn[m][l]))
            layers.append(nn.Sequential(*block))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for block in self.layers:
            x = x + block(x)
        return x


class MultiReceptiveFieldFusion(nn.Module):
    def __init__(self, channels, k_r, D_r):
        super().__init__()

        self.resblocks = nn.ModuleList([ResidualBlock(channels, k_r[n], D_r[n]) for n in range(len(k_r))])

    def forward(self, x):
        sum_x = 0
        for resblock in self.resblocks:
            sum_x += resblock(x)
        return sum_x / len(self.resblocks)


class Generator(nn.Module):
    def __init__(self, k_u, h_u, k_r, D_r):
        super().__init__()

        kernel = 7
        layers = []
        layers.append(nn.Conv1d(MelSpectrogramConfig.n_mels, h_u, kernel, padding="same"))
        for l in range(len(k_u)):
            layers.append(nn.LeakyReLU(LRELU_SLOPE))
            channels = h_u // (2 ** (l + 1))
            layers.append(
                nn.ConvTranspose1d(
                    in_channels=2 * channels,
                    out_channels=channels,
                    kernel_size=k_u[l],
                    stride=k_u[l] // 2,
                    padding=k_u[l] // 4,
                )
            )
            layers.append(MultiReceptiveFieldFusion(channels, k_r, D_r))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Conv1d(h_u // (2 ** len(k_u)), 1, kernel, padding="same"))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, mel):
        return self.layers(mel)
