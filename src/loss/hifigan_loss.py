import torch
from torch import nn
import torch.nn.functional as F

from src.utils.mel_spectrogram import MelSpectrogram, MelSpectrogramConfig


class HiFiGANLoss(nn.Module):
    def __init__(self, lambda_fm, lambda_mel):
        super().__init__()

        self.lambda_fm = lambda_fm
        self.lambda_mel = lambda_mel
        self.wav2mel = MelSpectrogram(MelSpectrogramConfig)

    # def forward(self, pred, target, disc_pred, disc_target, pred_feature_maps, target_feature_maps, **kwargs):
    #     loss_gen_adv = ((disc_target - 1) ** 2 + disc_pred**2).mean()
    #     loss_fm = self.lambda_fm * F.l1_loss(target_feature_maps, pred_feature_maps)
    #     loss_mel = self.lambda_mel * F.l1_loss(wav2vec(pred), target)

    #     loss_disc = ((disc_pred - 1) ** 2).mean()

    #     return {"loss_gen": loss_gen_adv + loss_fm + loss_mel, "loss_fm": loss_fm, "loss_mel": loss_mel, "loss_disc": loss_disc}

    def disc(self, disc_pred, disc_target, **kwargs):
        disc_loss = 0
        for dt, dp in zip(disc_target, disc_pred):
            disc_loss += torch.mean((dt - 1) ** 2) + torch.mean(dp**2)
        return {"disc_loss": disc_loss}

    def gen(self, mel, pred, disc_pred, pred_feature_maps, target_feature_maps, **kwargs):
        loss_adv = 0
        for dp in disc_pred:
            loss_adv += torch.mean((dp - 1) ** 2)

        loss_mel = self.lambda_mel * F.l1_loss(self.wav2mel(pred), mel)

        loss_fm = 0
        for tfm, pfm in zip(target_feature_maps, pred_feature_maps):
            loss_fm += self.lambda_fm * F.l1_loss(tfm, pfm)

        return {"gen_loss": loss_adv + loss_fm + loss_mel, "loss_adv": loss_adv, "loss_fm": loss_fm, "loss_mel": loss_mel}
