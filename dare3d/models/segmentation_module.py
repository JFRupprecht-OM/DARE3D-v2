""""""
from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import BinaryJaccardIndex

from dare3d.loggers.segmentation_logger import SegmentationLogger
from dare3d.metrics.object_level import evaluate_at_object_level, compute_recall, compute_precision, compute_fmeasure

class SegmentationLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        criterion=None,
        compile=False,
        scheduler_interval: str = "epoch",
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net", "criterion"])

        self.net = net

        # loss function
        self.criterion = criterion
        self.length_criterion = torch.nn.MSELoss(reduction="none")
        self.length_alpha = 1e-3

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = BinaryJaccardIndex(threshold=0.5)
        self.val_acc = BinaryJaccardIndex(threshold=0.5)
        self.test_acc = BinaryJaccardIndex(threshold=0.5)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        self.train_heatmap_loss = MeanMetric()
        self.val_heatmap_loss = MeanMetric()
        self.test_heatmap_loss = MeanMetric()
        
        self.train_length_loss = MeanMetric()
        self.val_length_loss = MeanMetric()
        self.test_length_loss = MeanMetric()
        
        self.train_rotmat_loss = MeanMetric()
        self.val_rotmat_loss = MeanMetric()
        self.test_rotmat_loss = MeanMetric()
        
        self.true_pos = MeanMetric()
        self.false_pos = MeanMetric()
        self.false_neg = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.optimizer = self.hparams.optimizer(params=self.net.parameters())
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def reset_metrics(self):
        self.train_acc.reset()
        self.val_acc.reset()
        self.test_acc.reset()
        self.train_loss.reset()
        self.val_loss.reset()
        self.test_loss.reset()
        self.train_heatmap_loss.reset()
        self.val_heatmap_loss.reset()
        self.test_heatmap_loss.reset()
        self.train_length_loss.reset()
        self.val_length_loss.reset()
        self.test_length_loss.reset()
        self.train_rotmat_loss.reset()
        self.val_rotmat_loss.reset()
        self.test_rotmat_loss.reset()
        self.true_pos.reset()
        self.false_neg.reset()
        self.false_pos.reset()

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.reset_metrics()
        self.val_acc_best.reset()

    def weight_in_out(self, din, dout, weight):
        din = din * weight
        dout = dout * weight
        return din, dout

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
        x, heatmaps, weights = x["input"], y["heatmaps"], y["weights"]
        outputs = self.forward(x)
        
        predicted_heatmaps = outputs["heatmaps"]
        lossesd = {}
        
        # Compute UNREDUCED criterion
        heatmap_losses = [self.criterion(pred, target) for i, (pred, target), in enumerate(zip(predicted_heatmaps, heatmaps))]

        # Mask loss values
        # Scale by the resolution factor
        heatmap_losses = [(loss * weight) / 2**i for i, (loss, weight) in enumerate(zip(heatmap_losses, weights))]
        # Reduce the loss
        heatmap_losses = [loss.sum() / (weight.sum()+1e-4) for loss, weight in zip(heatmap_losses, weights)]
        # Map losses to dict
        lossesd.update({"heatmap{i}": heatmap_losses[i] for i in range(len(heatmap_losses))})
        heatmap_losses = sum(heatmap_losses)
        lossesd.update({"heatmaps": heatmap_losses})
        
        loss = heatmap_losses
        
        if "length" in y:
            predicted_lengths = outputs["length"]
            true_lengths = y["length"]
            
            # Mask values
            mask = y["length_mask"]
            
            if mask.sum() != 0:
                # Non reduced loss
                length_loss = self.length_criterion(predicted_lengths, true_lengths)
                # Compute sum of errors
                length_loss = (length_loss * mask.float()).sum()
                # Divide by the number of non zeros to get a mean
                length_loss = length_loss / mask.sum()
                
                lossesd.update({"length": length_loss})
                
                loss = loss + length_loss * self.length_alpha
            
        if "rotmat" in y:
            predicted_rotmat = outputs["rotmat"]
            true_rotmat = outputs["rotmat"]
            rot_loss = self.rotmat_criterion(predicted_rotmat, true_rotmat)
            loss += rot_loss            
            lossesd.update({"angle": rot_loss})
        
        lossesd["loss"] = loss
        
        return lossesd, outputs, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        losses, preds, targets = self.model_step(batch)

        binary_targets = torch.where(targets["heatmaps"][0] > 0.5, 1.0, 0)
        
        # update and log metrics
        self.safe_update_metric(self.train_loss, losses["loss"])
        self.log("train/loss", self.train_loss, on_step=True, prog_bar=True)

        self.safe_update_metric(self.train_acc, preds["heatmaps"][0], binary_targets)
        self.log("train/iou", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        self.safe_update_metric(self.train_heatmap_loss, losses["heatmaps"])
        self.log("train/heatmaps", self.train_heatmap_loss, on_step=False, on_epoch=True, prog_bar=True)

        if "length" in losses:
            self.safe_update_metric(self.train_length_loss, losses["length"])
            self.log("train/length", self.train_length_loss, on_step=False, on_epoch=True, prog_bar=True)

        if "rotmat" in losses:
            self.safe_update_metric(self.train_rotmat_loss, losses["rotmat"])
            self.log("train/rotmat", self.train_rotmat_loss, on_step=False, on_epoch=True, prog_bar=True)

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
        losses, preds, targets = self.model_step(batch)        

        # update and log metrics
        binary_targets = torch.where(targets["heatmaps"][0] > 0.5, 1.0, 0)

        self.safe_update_metric(self.val_loss, losses["loss"])
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.safe_update_metric(self.val_acc, preds["heatmaps"][0], binary_targets)
        self.log("val/iou", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        self.safe_update_metric(self.val_heatmap_loss, losses["heatmaps"])
        self.log("val/heatmaps", self.val_heatmap_loss, on_step=False, on_epoch=True, prog_bar=True)

        if "length" in losses:
            self.safe_update_metric(self.val_length_loss, losses["length"])
            self.log("val/length", self.val_length_loss, on_step=False, on_epoch=True, prog_bar=True)

        if "rotmat" in losses:
            self.safe_update_metric(self.val_rotmat_loss, losses["rotmat"])
            self.log("val/rotmat", self.val_rotmat_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        x, _ = batch
        x = x['input']
        self.log_segmentation_results(x, targets["heatmaps"], preds["heatmaps"])

    def log_segmentation_results(self, x, targets, preds):
        logger: SegmentationLogger = self.get_segmentation_logger()
        if logger is None:
            return
        
        # logger.log_3D_images(x, targets, preds, epoch=0)
        
    def get_segmentation_logger(self):
        for logger in self.loggers:
            if isinstance(logger, SegmentationLogger):
                return logger

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best.update(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/iou_best", self.val_acc_best.compute(), prog_bar=True)
        
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss.update(loss["loss"])
        binary_targets = torch.where(targets["heatmaps"][0] > 0.5, 1.0, 0)
        self.test_acc.update(preds["heatmaps"][0], binary_targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/iou", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

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
                    "interval": self.hparams.scheduler_interval,
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def run_inference(self, dataset):
        pass