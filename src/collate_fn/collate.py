from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.mel_spectrogram import wav2mel


def pad_1D_tensor(inputs):
    def pad_data(x, length):
        x_padded = F.pad(x, (0, length - x.shape[0]))
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = torch.stack([pad_data(x, max_len) for x in inputs])

    return padded


def pad_2D_tensor(inputs, maxlen=None):
    def pad(x, max_len):
        if x.size(0) > max_len:
            raise ValueError("not max_len")

        s = x.size(1)
        x_padded = F.pad(x, (0, 0, 0, max_len - x.size(0)))
        return x_padded[:, :s]

    if maxlen:
        output = torch.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(x.size(0) for x in inputs)
        output = torch.stack([pad(x, max_len) for x in inputs])

    return output


def collate_fn(batch: List[dict]):
    wavs = pad_1D_tensor([item["wav"].squeeze(0) for item in batch]).unsqueeze(1)
    new_length = (wavs.shape[-1] // 256) * 256
    wavs = wavs[..., :new_length]

    # texts = []
    # for i in range(batch[])

    mels = pad_2D_tensor([mel for mel in wav2mel(wavs)]).squeeze(1)
    print(mels.shape)

    return {"target": wavs, "mel": mels}
