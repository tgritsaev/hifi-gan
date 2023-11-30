from torch import nn
import torch.nn.functional as F

from src.utils.mel_spectrogram import MelSpectrogram
from src.utils.mel_spectrogram import wav2mel


class HiFiGANLoss(nn.Module):
    def __init__(self, lambda_fm, lambda_mel):
        super().__init__()

        self.lambda_fm = lambda_fm
        self.lambda_mel = lambda_mel

    # def forward(self, pred, target, disc_pred, disc_target, pred_feature_maps, target_feature_maps, **kwargs):
    #     loss_gen_adv = ((disc_target - 1) ** 2 + disc_pred**2).mean()
    #     loss_fm = self.lambda_fm * F.l1_loss(target_feature_maps, pred_feature_maps)
    #     loss_mel = self.lambda_mel * F.l1_loss(wav2vec(pred), target)

    #     loss_disc = ((disc_pred - 1) ** 2).mean()

    #     return {"loss_gen": loss_gen_adv + loss_fm + loss_mel, "loss_fm": loss_fm, "loss_mel": loss_mel, "loss_disc": loss_disc}

    def disc(self, disc_pred, disc_target, **kwargs):
        return {"disc_loss": ((disc_target - 1) ** 2 + disc_pred**2).mean()}

    def gen(self, pred, target, disc_pred, pred_feature_maps, target_feature_maps, **kwargs):
        loss_adv = ((disc_pred - 1) ** 2).mean()
        loss_fm = self.lambda_fm * F.l1_loss(target_feature_maps, pred_feature_maps)
        loss_mel = self.lambda_mel * F.l1_loss(wav2mel(pred), target)

        return {"gen_loss": loss_adv + loss_fm + loss_mel, "loss_adv": loss_adv, "loss_fm": loss_fm, "loss_mel": loss_mel}
