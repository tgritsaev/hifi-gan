import random
from pathlib import Path
from tqdm import tqdm

import torchaudio
from torch.utils.data import Dataset


class LJSpeechDataset(Dataset):
    def __init__(self, dir, limit=None, max_len=None, **kwargs):
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
        self.max_len = max_len

    def __len__(self):
        return len(self.wavs_path)

    def __getitem__(self, idx):
        wav, _ = torchaudio.load(self.wavs_path[idx])
        if self.max_len:
            start = random.randint(0, max(0, wav.shape[-1] - self.max_len))
            wav = wav[:, start : start + self.max_len]
        return {"wav": wav}
        # return {"wav": wav, "text": self.texts[idx]}
