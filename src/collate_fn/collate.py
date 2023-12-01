from typing import List

from src.utils.mel_spectrogram import wav2mel


def collate_fn(batch: List[dict]):
    wavs = [item["wav"] for item in batch]
    # new_length = (wavs.shape[-1] // 256) * 256
    # wavs = wavs[..., :new_length]

    # mels = []
    # for i in range(wavs.shape[0]):
    #     mels.append(wav2mel(wavs[i]))
    # mels = pad_2D_tensor(mels).squeeze(1)
    mels = [mel for mel in wav2mel(wavs)]

    return {"target": wavs, "mel": mels}
