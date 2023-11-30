# https://arxiv.org/pdf/2010.05646.pdf
import torch
from torch import nn

from src.model.base_model import BaseModel
from src.model.generator import Generator
from src.model.mpd import MultiPeriodDiscriminator
from src.model.msd import MultiScaleDiscriminator


class HiFiGANModel(BaseModel):
    def __init__(self, generator_args, mpd_periods, msd_args, **batch):
        super().__init__(**batch)

        self.gen = Generator(**generator_args)
        self.mpds = nn.ModuleList([MultiPeriodDiscriminator(p) for p in mpd_periods])
        self.msd = MultiScaleDiscriminator(**msd_args)

    def disc_forward(self, pred, target, **kwargs):
        def get_discriminator_output(wav):
            feature_maps = []
            mpds_out = []

            for mpd in self.mpds:
                mpd_out, mpd_feature_map = mpd(wav)
                feature_maps += mpd_feature_map

                mpds_out.append(mpd_out)

            msd_out, msd_feature_maps = self.msd(pred)
            feature_maps += msd_feature_maps

            return mpds_out + [msd_out], feature_maps

        disc_pred, pred_feature_maps = get_discriminator_output(pred)
        disc_target, target_feature_maps = get_discriminator_output(target)

        return {
            "disc_pred": disc_pred,
            "pred_feature_maps": pred_feature_maps,
            "disc_target": disc_target,
            "target_feature_maps": target_feature_maps,
        }

    def forward(self, mel, target=None, **kwargs):
        pred = self.gen(mel)
        out = {"pred": pred}

        if self.training:
            out.update(self.disc_forward(pred, target))

        print("!!!!!!!!", self.training, out.keys())
        return {"pred": pred}
