# Mel-spectrogram to wav with HiFi-GAN 

1. The model from the article [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/pdf/2010.05646.pdf) is implemented.
2. To familiarize with Multi Scale Discriminator [MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis](https://arxiv.org/pdf/1910.06711.pdf).

## Installation guide

1. Use python3.9
```shell
conda create -n fastspeech2 python=3.9 && conda activate fastspeech2
```
2. Install libraries
```shell
pip3 install -r requirements.txt
```
3. Download data
```shell
bash scripts/download_data.sh
```
4. Preprocess data: save pitch and energy
```shell
python3 scripts/preprocess_data.py
```
5. Download my final FastSpeech2 checkpoint
```shell
python3 scripts/download_checkpoint.py
```

## Train 
1. Run for training 
```shell
python3 train.py -c configs/train.json
```
Final model was trained with `train.json` config.

## Test
1. Run for testing
```shell
python3 test.py
```
`test.py` include such arguments:
* Config path: `-c, --config, default="configs/test.json"`
* Create multiple audio variants with different length, pitch and energy `-t, --test, default=False`
* Increase or decrease audio speed: `-l, --length-control, default=1`
* Increase or decrease audio pitch: `-p, --pitch-control, default=1`
* Increase or decrease audio energy: `-e, --energy-control, default=1`
* Checkpoint path: `-cp, --checkpoint, default="test_model/tts-checkpoint.pth"`
* Input texts path: `-i, --input, test_model/input.txt`
* Waveglow weights path: `-w, --waveglow, default="waveglow/pretrained_model/waveglow_256channels.pt"`

Results will be saved in the `test_model/results`, you can see example in this folder.

## Wandb Report

[https://api.wandb.ai/links/tgritsaev/rkir8sp9](https://wandb.ai/tgritsaev/dla3_text_to_speech/reports/Text-to-speech-with-FastSpeech2--Vmlldzo2MDU2MjU5?accessToken=4bu09pvt6ik0wpse85z5cnckki7q0pcdfo8aug40gw942v1h1jcf1gtp2vfo8w58) (English only)

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository. 
