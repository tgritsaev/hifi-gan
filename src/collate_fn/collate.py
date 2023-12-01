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
    batch_mels = wav2mel(wavs)
    mels = [batch_mels[i] for i in range(batch_mels.shape[0])]

    return {"target": wavs, "mel": mels}
