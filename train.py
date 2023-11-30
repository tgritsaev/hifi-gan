import argparse
import itertools
import warnings

import numpy as np
import torch

import src.loss as module_loss
import src.model as module_arch
from src.trainer import GANTrainer
from src.utils import prepare_device
from src.utils.object_loading import get_dataloaders
from src.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def get_params_count(model_):
    model_parameters = filter(lambda p: p.requires_grad, model_.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def main(config):
    logger = config.get_logger("train")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture, then print to console

    model = config.init_obj(config["arch"], module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    loss_module = config.init_obj(config["loss"], module_loss).to(device)
    metrics = []  # [config.init_obj(metric_dict, module_metric) for metric_dict in config["metrics"]]

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    gen_trainable_params = filter(lambda p: p.requires_grad, model.gen.parameters())
    gen_optimizer = config.init_obj(config["gen_optimizer"], torch.optim, gen_trainable_params)
    gen_lr_scheduler = config.init_obj(config["gen_lr_scheduler"], torch.optim.lr_scheduler, gen_optimizer)
    logger.info(f"Generator params count: {get_params_count(model.gen)}")

    disc_trainable_params = filter(lambda p: p.requires_grad, itertools.chain(model.msd.parameters(), model.mpds.parameters()))
    disc_optimizer = config.init_obj(config["disc_optimizer"], torch.optim, disc_trainable_params)
    disc_lr_scheduler = config.init_obj(config["disc_lr_scheduler"], torch.optim.lr_scheduler, disc_optimizer)
    logger.info(f"MPDs params count: {get_params_count(model.mpds)}")
    logger.info(f"MSD params count: {get_params_count(model.msd)}")

    trainer = GANTrainer(
        model,
        loss_module,
        metrics,
        gen_optimizer,
        disc_optimizer,
        config=config,
        device=device,
        dataloaders=dataloaders,
        gen_lr_scheduler=gen_lr_scheduler,
        disc_lr_scheduler=disc_lr_scheduler,
        len_epoch=config["trainer"].get("len_epoch", None),
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    config = ConfigParser.from_args(args)
    main(config)
