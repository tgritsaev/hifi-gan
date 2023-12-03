from pathlib import Path
import gdown

CHECKPOINT_LINK1 = "https://drive.google.com/u/0/uc?id=1T3sbkriX6dnHKLmteSL1TskygGpN7JIa&export=download"
CHECKPOINT_LINK2 = "https://drive.google.com/u/0/uc?id=1ub99QvO4BHe2cUOrGCsFu9kFwr3rMyUT&export=download"
SAVE_PATH = Path("test_model/")


def main():
    SAVE_PATH.mkdir(exist_ok=True, parents=True)
    gdown.download(CHECKPOINT_LINK2, str(SAVE_PATH / "wav2mel-checkpoint.pth"))


if __name__ == "__main__":
    main()
