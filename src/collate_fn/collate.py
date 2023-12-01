from typing import List
import torch
import torch.nn.functional as F
from src.utils.mel_spectrogram import MelSpectrogram, MelSpectrogramConfig

wav2mel = MelSpectrogram(MelSpectrogramConfig())


def pad_1D_tensor(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = F.pad(x, (0, length - x.shape[0]), value=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = torch.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D_tensor(inputs, maxlen=None):
    def pad(x, max_len):
        if x.size(-1) > max_len:
            raise ValueError("not max_len")

        x_padded = F.pad(x, (0, max_len - x.size(-1), 0, 0))
        return x_padded

    if maxlen:
        output = torch.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(x.size(-1) for x in inputs)
        output = torch.stack([pad(x, max_len) for x in inputs])

    return output


def collate_fn(batch: List[dict]):
    wavs = torch.stack([item["wav"] for item in batch])
    mels = wav2mel(wavs)
    print("!!!!!", wavs.shape, mels.shape)

    wavs = pad_1D_tensor([item["wav"].squeeze(0) for item in batch]).unsqueeze(1)
    new_length = (wavs.shape[-1] // 256) * 256
    wavs = wavs[..., :new_length]

    # mels = []
    # for i in range(wavs.shape[0]):
    #     mels.append(wav2mel(wavs[i]))
    # mels = pad_2D_tensor(mels).squeeze(1)
    mels = pad_2D_tensor([mel for mel in wav2mel(wavs)]).squeeze(1)
    print("?????", wavs.shape, mels.shape)

    return {"target": wavs, "mel": mels}
