# https://arxiv.org/pdf/2010.05646.pdf
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
            out = []
            feature_maps = []

            for mpd in self.mpds:
                mpd_out, mpd_feature_map = mpd(wav)
                out.append(mpd_out)
                feature_maps.extend(mpd_feature_map)

            msd_out, msd_feature_maps = self.msd(pred)
            out.append(msd_out)
            feature_maps.extend(msd_feature_maps)

            return out, feature_maps

        disc_pred, pred_feature_maps = get_discriminator_output(pred)
        disc_target, target_feature_maps = get_discriminator_output(target)

        return {
            "disc_pred": disc_pred,
            "pred_feature_maps": pred_feature_maps,
            "disc_target": disc_target,
            "target_feature_maps": target_feature_maps,
        }

    def forward(self, mel, **kwargs):
        return {"pred": self.gen(mel)}
