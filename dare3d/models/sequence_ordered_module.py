""""""
from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import BinaryAccuracy

class SequenceOrderedModule(LightningModule):
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

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = BinaryAccuracy(threshold=0.5)
        self.val_acc = BinaryAccuracy(threshold=0.5)
        self.test_acc = BinaryAccuracy(threshold=0.5)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
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

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()
        self.true_pos.reset()
        self.false_neg.reset()
        self.false_pos.reset()

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
        return loss, logits, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss.update(loss)
        self.train_acc.update(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/iou", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)        

        # update and log metrics
        self.val_loss.update(loss)
        self.val_acc.update(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/iou", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        x, _ = batch
        self.log_segmentation_results(x, targets, preds)
        
        # for i in range(preds.shape[0]):
        #     pred_numpy = torch.nn.functional.sigmoid(preds[i])
        #     pred_numpy = pred_numpy.detach().cpu().numpy()[0]
        #     target_numpy = targets[i].detach().cpu().numpy()[0]
        #     stats = evaluate_at_object_level(pred_numpy, target_numpy, threshold=0.5)
        #     self.true_pos.update(stats["tp"])
        #     self.false_pos.update(stats["fp"])
        #     self.false_neg.update(stats["fn"])    

    def log_segmentation_results(self, x, targets, preds):
        logger: SegmentationLogger = self.get_segmentation_logger()
        if logger is None:
            return
        
        logger.log_3D_images(x, targets, preds, epoch=self.current_epoch)
        
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
        
        # tp = self.true_pos.compute().detach().cpu().numpy()
        # fp = self.false_pos.compute().detach().cpu().numpy()
        # fn = self.false_neg.compute().detach().cpu().numpy()
        
        # precision = compute_precision(tp, fp)
        # recall = compute_recall(tp, fn)
        # fmeasure = compute_fmeasure(recall, precision, fbeta=1.0)
        
        # self.log("val/precision", precision, prog_bar=True)
        # self.log("val/recall", recall, prog_bar=True)
        # self.log("val/fmeasure", fmeasure, prog_bar=True) 

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss.update(loss)
        self.test_acc.update(preds, targets)
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
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def run_inference(self, dataset):
        pass