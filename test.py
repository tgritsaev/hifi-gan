import argparse
import json
import os
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

import src.datasets
import src.model as module_model
from src.utils import DEFAULT_SR
from src.utils.mel_spectrogram import MelSpectrogram, MelSpectrogramConfig
from src.utils.parse_config import ConfigParser


def main(config, args):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)
    logger.info("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    logger.info("Checkpoint has been loaded.")
    model = model.to(device)
    model.eval()

    wav2vec = MelSpectrogram(MelSpectrogramConfig).to(device)
    dataset = config.init_obj(config["data"]["test"]["datasets"][0], src.datasets, config_parser=config)

    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataset, "inference")):
            batch["mel"] = wav2vec(batch["wav"].to(device))[:, :, 100]
            print(batch["mel"].shape)
            pred = model(batch)["pred"]
            torchaudio.save(f"{args.output_dir}/{i}-audio.wav", pred, sample_rate=DEFAULT_SR)

    logger.info("Audios have been generated.")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-c",
        "--config",
        default="configs/test.json",
        type=str,
        help="Config path.",
    )
    args.add_argument(
        "-cp",
        "--checkpoint",
        default="test_model/wav2mel-checkpoint.pth",
        type=str,
        help="Checkpoint path.",
    )
    args.add_argument(
        "-o",
        "--output_dir",
        default="test_model/results",
        type=str,
        help="Output wavs path.",
    )
    args = args.parse_args()

    model_config = Path(args.config)
    with model_config.open() as fin:
        config = ConfigParser(json.load(fin))

    main(config, args)
