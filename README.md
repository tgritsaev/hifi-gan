# Mel-spectrogram to wav with HiFi-GAN 

1. The model from the article [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/pdf/2010.05646.pdf) is implemented.
2. To familiarize with Multi Scale Discriminator and Feature Maps Loss read [MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis](https://arxiv.org/pdf/1910.06711.pdf).

## Code organization

```shell
├── README.md             <- Top-level README.
├── requirements.txt      <- project requirements.
├── train.py              <- Top-level train code.
├── test.py               <- Top-level test code.
│
├── scripts               <- setup scripts.
│   ├── download_checkpoint.py   <- download my final model.
│   └── download_data.sh         <- download data.
│
└── src                   <- main code directory.
    ├── collate_fn               <- collate_fn function
    ├── datasets                 <- LJSpeech dataset implementaion
    ├── logger                   <- supportive logger
    ├── loss                     <- HiFI-GAN loss implementation 
    ├── model                    <- HiFI-GAN model implementation 
    ├── trainer                  <- general train pipeline
    └── utils                    <- utils


## Installation

1. Use python3.11
```shell
conda create -n hifigan python=3.11 && conda activate hifigan
```
2. Install libraries
```shell
pip3 install -r requirements.txt
```
3. Download data
```shell
bash scripts/download_data.sh
```
4. Download my final HiFi-GAN checkpoint, which was trained with `configs/train.json`
```shell
python3 scripts/download_checkpoint.py
```

## Train 
1. Run for training 
```shell
python3 train.py -c configs/train.json
```
Final model was trained with `configs/train.json` config.

## Test
1. Run for testing
```shell
python3 test.py
```
`test.py` contains such arguments:
* Config path: `-c, --config, default="configs/test.json"`
* Checkpoint path: `-cp, --checkpoint, default="test_model/wav2mel-checkpoint.pth"`
* Output texts path: `-o, --output, test_model/results`

## Wandb

1. [Report](https://wandb.ai/tgritsaev/dla4/reports/Mel-to-wav-with-HiFi-GAN--Vmlldzo2MTUxNDAz). (English only)
2. [Wandb project](https://wandb.ai/tgritsaev/dla4?workspace=user-tgritsaev).

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository. 
