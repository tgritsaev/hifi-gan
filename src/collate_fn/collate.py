from typing import List
import torch
import torch.nn.functional as F
from src.utils.mel_spectrogram import MelSpectrogram, MelSpectrogramConfig

wav2mel = MelSpectrogram(MelSpectrogramConfig())


def collate_fn(batch: List[dict]):
    wavs = torch.stack([item["wav"] for item in batch])
    mels = wav2mel(wavs).squeeze(1)

    return {"target": wavs, "mel": mels}
