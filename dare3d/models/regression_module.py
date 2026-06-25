""""""
from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric

from dare3d.loggers.regression_logger import RegressionLogger

class RegressionLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        criterion=None,
        compile=False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net", "criterion"])

        self.net = net

        # loss function
        self.criterion = criterion

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.train_head1_angle_loss = MeanMetric()
        self.train_head1_len_loss = MeanMetric()
        self.train_head2_angle_loss = MeanMetric()
        self.train_head2_len_loss = MeanMetric()
        self.val_head1_angle_loss = MeanMetric()
        self.val_head1_len_loss = MeanMetric()
        self.val_head2_angle_loss = MeanMetric()
        self.val_head2_len_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        self.optimizer = self.hparams.optimizer(params=self.net.parameters())
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def reset_metrics(self):
        self.train_loss.reset()
        self.val_loss.reset()
        self.test_loss.reset()

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.reset_metrics()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        x, head1, head2 = x["input"], y['head1'], y['head2']
        outputs = self.forward(x)

        lossesd = {}
        
        if len(head2) == 2:
            head1_angle_loss, head1_len_loss, head2_angle_loss, head2_len_loss = self.criterion(outputs["head1"], outputs["head2"], head1, head2)
            head1_loss = head1_angle_loss + head1_len_loss
            head2_loss = head2_angle_loss + head2_len_loss
            loss = head1_loss + head2_loss
            
            lossesd.update({
                "head1_angle_loss": head1_angle_loss,
                "head1_len_loss": head1_len_loss,
                "head2_angle_loss": head2_angle_loss,
                "head2_len_loss": head2_len_loss
            })
            
        else:
            head1_angle_loss, head1_len_loss = self.criterion(outputs["head1"], head1)
            loss = head1_angle_loss + head1_len_loss
            
            lossesd.update({
                "head1_angle_loss": head1_angle_loss,
                "head1_len_loss": head1_len_loss,
            })

        lossesd["loss"] = loss
        
        return lossesd, x, outputs, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        losses, x, preds, targets = self.model_step(batch)
        
        # update and log metrics
        self.safe_update_metric(self.train_loss, losses["loss"])
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)

        self.safe_update_metric(self.train_head1_angle_loss, losses["head1_angle_loss"])
        self.log("train/head1_angle_loss", self.train_head1_angle_loss, on_step=True, on_epoch=True, prog_bar=True)

        self.safe_update_metric(self.train_head1_len_loss, losses["head1_len_loss"])
        self.log("train/head1_len_loss", self.train_head1_len_loss, on_step=True, on_epoch=True, prog_bar=True)


        if "head2_angle_loss" in losses:
            self.safe_update_metric(self.train_head2_angle_loss, losses["head2_angle_loss"])
            self.log("train/head2_angle_loss", self.train_head2_angle_loss, on_step=False, on_epoch=True, prog_bar=True)

        if "head2_len_loss" in losses:
            self.safe_update_metric(self.train_head2_len_loss, losses["head2_len_loss"])
            self.log("train/head2_len_loss", self.train_head2_len_loss, on_step=False, on_epoch=True, prog_bar=True)


        if self.hparams.scheduler is not None:
            self.log("train/lr", self.scheduler.get_last_lr()[0], on_step=True, prog_bar=True)
        else:
            self.log("train/lr", self.optimizer.param_groups[-1]["lr"])

        # return loss or backpropagation will fail
        return losses["loss"]

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def safe_update_metric(self, metric, value, *args):
        if not torch.isnan(value.sum()):
            metric.update(value, *args)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        losses, x, preds, targets = self.model_step(batch)        

        # update and log metrics

        self.safe_update_metric(self.val_loss, losses["loss"])
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.safe_update_metric(self.val_head1_angle_loss, losses["head1_angle_loss"])
        self.log("val/head1_angle_loss", self.val_head1_angle_loss, on_step=True, prog_bar=True)

        self.safe_update_metric(self.val_head1_len_loss, losses["head1_len_loss"])
        self.log("val/head1_len_loss", self.val_head1_len_loss, on_step=True, prog_bar=True)


        if "head2_angle_loss" in losses:
            self.safe_update_metric(self.val_head2_angle_loss, losses["head2_angle_loss"])
            self.log("val/head2_angle_loss", self.val_head2_angle_loss, on_step=False, on_epoch=True, prog_bar=True)

        if "head2_len_loss" in losses:
            self.safe_update_metric(self.val_head2_len_loss, losses["head2_len_loss"])
            self.log("val/head2_len_loss", self.val_head2_len_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.log_regression_results(x, targets, preds)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, x, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss.update(loss["loss"])
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def log_regression_results(self, x, targets, preds):
        logger: RegressionLogger = self.get_regression_logger()
        if logger is None:
            return
        
        logger.log_3D_images(x, targets, preds, epoch=0)
        
    def get_regression_logger(self):
        for logger in self.loggers:
            if isinstance(logger, RegressionLogger):
                return logger


    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            self.scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def run_inference(self, dataset):
        pass