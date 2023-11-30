from tqdm import tqdm

import torch

from src.trainer.base_trainer import BaseTrainer
from src.utils import inf_loop, MetricTracker
from src.utils import DEFAULT_SR


class GANTrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        gen_optimizer,
        disc_optimizer,
        config,
        device,
        dataloaders,
        gen_lr_scheduler,
        disc_lr_scheduler,
        len_epoch=None,
        skip_oom=True,
    ):
        super().__init__(model, criterion, metrics, None, None, config, device)
        self.skip_oom = skip_oom
        self.config = config

        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.gen_lr_scheduler = gen_lr_scheduler
        self.disc_lr_scheduler = disc_lr_scheduler

        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch

        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}

        self.loss_names = ["disc_loss", "gen_loss", "loss_adv", "loss_fm", "loss_mel"]
        self.train_metrics = MetricTracker(*self.loss_names, "grad_norm")
        self.test_metrics = MetricTracker(*self.loss_names)

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / "checkpoint-epoch{}.pth".format(epoch))
        if not (only_best and save_best):
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the GPU
        """
        for tensor_for_gpu in ["target", "mel"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    @torch.no_grad()
    def _log_predictions(self, examples_to_log=3, **kwargs):
        ...
        # for i, wav in enumerate(wavs):
        #     self.writer.add_audio(f"audio-{i}", wav, sample_rate=DEFAULT_SR)

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        out = self.model(**batch)
        batch.update(out)

        if is_train:
            print("calc loss")
            # discriminator
            self.disc_optimizer.zero_grad()
            batch["pred"].detach()
            batch.update(self.model.disc_forward(**batch))
            disc_loss = self.criterion.disc(**batch)
            batch.update(disc_loss)
            batch["disc_loss"].backward()
            self._clip_grad_norm()
            self.disc_optimizer.step()

            # generator
            batch.update(self.model.disc_forward(**batch))
            self.gen_optimizer.zero_grad()
            gen_loss = self.criterion.gen(**batch)
            batch.update(gen_loss)
            batch["gen_loss"].backward()
            self._clip_grad_norm()
            self.gen_optimizer.step()

            self.train_metrics.update("grad_norm", self.get_grad_norm())

            for loss_name in self.loss_names:
                metrics.update(loss_name, batch[loss_name].item())

        for metric in self.metrics:
            metrics.update(metric.name, metric(**batch))

        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()

        with torch.no_grad():
            for _, batch in tqdm(enumerate(dataloader), desc=part, total=len(dataloader)):
                batch = self.process_batch(batch, False, metrics=self.evaluation_metrics)

            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_predictions(False, **batch)
            # self._log_spectrogram(batch["spectrogram"])
            self._log_scalars(self.evaluation_metrics)

        return self.evaluation_metrics.result()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        bar = tqdm(range(self.len_epoch), desc="train")
        for batch_idx, batch in enumerate(self.train_dataloader):
            try:
                batch = self.process_batch(batch, True, metrics=self.train_metrics)
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug("Train Epoch: {} {} Loss: {:.6f}".format(epoch, self._progress(batch_idx), batch["loss"].item()))
                self.writer.add_scalar("disc learning rate", self.disc_lr_scheduler.get_last_lr()[0])
                self.writer.add_scalar("gen learning rate", self.gen_lr_scheduler.get_last_lr()[0])
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
                bar.update(self.log_step)

            if batch_idx + 1 >= self.len_epoch:
                break

        self.gen_lr_scheduler.step()
        self.disc_lr_scheduler.step()

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        log = last_train_metrics
        return log
