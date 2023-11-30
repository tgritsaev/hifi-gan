from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.mel_spectrogram import wav2mel


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(x, (0, length - x.shape[0]), mode="constant", constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_1D_tensor(inputs):
    def pad_data(x, length):
        x_padded = F.pad(x, (0, length - x.shape[0]))
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = torch.stack([pad_data(x, max_len) for x in inputs])

    return padded


def collate_fn(batch: List[dict]):
    wavs = pad_1D_tensor([item["wav"].squeeze(0) for item in batch]).unsqueeze(1)
    new_length = (wavs.shape[-1] // 256) * 256
    wavs = wavs[..., :new_length]

    # texts = []
    # for i in range(batch[])

    mels = wav2mel(wavs).squeeze(1)

    return {"target": wavs, "mel": mels}
