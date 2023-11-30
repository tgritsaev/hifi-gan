import os
from pathlib import Path
from tqdm import tqdm

import torchaudio
from torch.utils.data import Dataset


class LJSpeechDataset(Dataset):
    def __init__(self, dir, limit=None, **kwargs):
        dir = Path(dir)

        # self.texts = []
        # with open(dir / "text.txt", "r", encoding="utf-8") as f:
        #     for line in f.readlines()[:limit]:
        #         self.texts.append(line)

        self.wavs_path = []
        for wav_path in tqdm((dir / "wavs").iterdir()):
            self.wavs_path.append(str(wav_path))
        self.wavs_path.sort()
        self.wavs_path = self.wavs_path[:limit]

    def __len__(self):
        return len(self.wavs_path)

    def __getitem__(self, idx):
        wav, _ = torchaudio.load(self.wavs_path[idx])
        # return {"wav": wav, "text": self.texts[idx]}
        return {"wav": wav}
