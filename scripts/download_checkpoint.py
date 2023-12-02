from pathlib import Path
import gdown

CHECKPOINT_LINK = "https://drive.google.com/u/0/uc?id=1T3sbkriX6dnHKLmteSL1TskygGpN7JIa&export=download"
SAVE_PATH = Path("test_model/")


def main():
    SAVE_PATH.mkdir(exist_ok=True, parents=True)
    gdown.download(CHECKPOINT_LINK, str(SAVE_PATH / "wav2mel-checkpoint.pth"))


if __name__ == "__main__":
    main()
