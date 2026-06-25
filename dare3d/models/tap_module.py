""""""
from typing import Any, Dict, Tuple

import os
import torch
from lightning import LightningModule
from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryAccuracy
from monai.visualize import GradCAM
from dare3d.losses.decorrelation import DecorrelationLoss

class TapModule(LightningModule):
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
        self.train_ce = MeanMetric()
        self.train_decor = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        # self.train_acc = BinaryAccuracy()
        # self.val_acc = BinaryAccuracy()
        # self.test_acc = BinaryAccuracy()
        
        self.optimizer = self.hparams.optimizer(params=self.net.parameters())
        self.decorrelation = DecorrelationLoss()
        
        # self.cam = GradCAM(nn_module=self.net.backbone, target_layers="final_conv")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.train_loss.reset()
        # self.val_acc.reset()
        # self.train_acc.reset()
        # self.test_acc.reset()

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
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        # decor_loss = self.decorrelation(features)
        decor_loss = 0
        return loss, decor_loss, logits, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, decor_loss, preds, targets = self.model_step(batch)
        total_loss = loss + decor_loss

        # preds = torch.sigmoid(preds)

        # update and log metrics
        self.train_loss.update(total_loss)
        # self.train_ce.update(loss)
        # self.train_decor.update(decor_loss)
        # self.train_acc.update(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train/ce", self.train_ce, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train/decorr", self.train_decor, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        x, y = batch
        # cam_results = self.cam(x=x, class_idx=0)
        
        from skimage import io
        import numpy as np
        if batch_idx == 0:
            for i in range(0, x.shape[0]):
                # xs = cam_results[i].detach().cpu().numpy()
                pi = preds[i].detach().cpu().numpy()
                yi = y[i].detach().cpu().numpy()
                xi = x[i].detach().cpu().numpy()

                movie_true = np.concatenate([xi, yi], axis=0)
                movie_pred = np.concatenate([xi, pi], axis=0)
                
                if not os.path.exists("outputs"):
                    os.makedirs("outputs")
                io.imsave(f"outputs/pred_{i}.tif", movie_pred)
                io.imsave(f"outputs/true_{i}.tif", movie_true)

        if self.hparams.scheduler is not None:
            self.log("train/lr", self.scheduler.get_last_lr()[0], on_step=True, prog_bar=True)
        else:
            self.log("train/lr", self.optimizer.param_groups[-1]["lr"])

        # return loss or backpropagation will fail
        return total_loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, loss_decor, preds, targets = self.model_step(batch)        
        # preds = torch.sigmoid(preds)
        total_loss = loss + loss_decor

        # update and log metrics
        self.val_loss.update(total_loss)
        # self.val_acc.update(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
                
    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, loss_decor, preds, targets = self.model_step(batch)
        # preds = torch.sigmoid(preds)

        total_loss = loss + loss_decor

        # update and log metrics
        self.test_loss.update(total_loss)
        # self.test_acc.update(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

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
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def run_inference(self, dataset):
        pass