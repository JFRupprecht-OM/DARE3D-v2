import os
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
from monai.optimizers import LearningRateFinder

from lightning.pytorch.plugins.environments import SLURMEnvironment
SLURMEnvironment.detect = lambda: False

# cuDNN's algorithm selection intermittently segfaults (native 0xC0000005, no CUDA error) during
# 3D-conv training on some stacks — observed on Turing (RTX 5000) + torch 2.2.2 / cuDNN 8.8.
# cudnn.benchmark/deterministic only REDUCE the crash rate; only disabling cuDNN was reliable
# (verified 6/6 runs vs intermittent crashes otherwise). So cuDNN is off by default here for
# dependable training; on a healthy stack (e.g. a newer cuDNN) set env DARE3D_CUDNN=1 to re-enable
# it for speed. (Proper long-term fix: upgrade the CUDA/cuDNN/torch stack.)
if os.environ.get("DARE3D_CUDNN") != "1":
    torch.backends.cudnn.enabled = False

OmegaConf.register_new_resolver("eval", eval)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from dare3d.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
    DisabledSLURMEnvironment,
    ModelOutputWrapper,
    CriterionWrapper
)

log = RankedLogger(__name__, rank_zero_only=True)


def check_cudnn_for_3d() -> None:
    """Fail fast on a cuDNN version that segfaults DARE3D's 3D-conv training.

    Older cuDNN (8.x) intermittently crashes the 3D convolutions with a native access
    violation (0xC0000005, no Python traceback); cuDNN >= 9 (torch >= 2.5) runs clean.
    If the user has made an explicit ``DARE3D_CUDNN`` choice, that wins and this guard
    stays out of the way (consistent with the module-level gate above). Fail-fast (raise)
    rather than auto-disabling, so we never mutate global cuDNN state here.
    """
    if not torch.cuda.is_available():
        return
    cudnn_ver = torch.backends.cudnn.version()
    # ENCODING TRAP — do NOT "simplify" this: cuDNN changed its version integer at 9.0.
    #   pre-9: MAJOR*1000  + MINOR*100 + PATCH  -> 8.9.7 = 8907
    #   9+:    MAJOR*10000 + MINOR*100 + PATCH  -> 9.1.0 = 90100
    # Compare against the integer 90000. Do NOT parse the leading digit.
    if cudnn_ver is not None and cudnn_ver < 90000 and "DARE3D_CUDNN" not in os.environ:
        raise RuntimeError(
            f"cuDNN {cudnn_ver} (< 9.0) intermittently segfaults during DARE3D 3D-convolution "
            "training (native 0xC0000005, no Python traceback). Install a supported stack — "
            "torch>=2.5 (cuDNN>=9), e.g.:\n"
            "    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128\n"
            "Or run anyway on this cuDNN by setting DARE3D_CUDNN explicitly:\n"
            "    DARE3D_CUDNN=0  -> disable cuDNN (stable but slower)\n"
            "    DARE3D_CUDNN=1  -> keep cuDNN ON (faster, but will likely crash on cuDNN < 9)"
        )


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Refuse to run on a cuDNN that segfaults 3D-conv training (unless DARE3D_CUDNN is set).
    check_cudnn_for_3d()

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Fine-tuning: load the base checkpoint into model.net BEFORE the (slow) data/Trainer setup,
    # so a wrong-stage or missing base fails in seconds (stage-aware error).
    if getattr(model, "_finetune_active", False):
        log.info(f"Fine-tuning: loading base weights from {model._ft.get('base_ckpt')}")
        model.load_base()

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger, plugins=DisabledSLURMEnvironment(auto_requeue=False))
        
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("find_lr"):
        datamodule.prepare_data()
        lr_finder = LearningRateFinder(
            # ModelOutputWrapper(model.net, lambda x:  x["heatmaps"][0]),
            ModelOutputWrapper(model.net, lambda x:  x[0]),
                                       model.optimizer,
                                       CriterionWrapper(model.criterion, lambda x: x.mean()),
                                       amp=True)
        lr_finder.range_test(datamodule.train_dataloader(),
                            #  image_extractor=lambda x: x[0]["input"],
                            #  label_extractor=lambda x: x[1]["heatmaps"][0],
                             end_lr=100, start_lr=1e-4, num_iter=50)
        print(f"Steepest gradient: {lr_finder.get_steepest_gradient()}")
        lr_finder.plot() # to inspect the loss-learning rate graph

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")
        
    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    # Fine-tuning: write the provenance sidecar next to the saved checkpoints.
    if getattr(model, "_finetune_active", False):
        ckpt_cb = getattr(trainer, "checkpoint_callback", None)
        ckpt_dir = getattr(ckpt_cb, "dirpath", None)
        if ckpt_dir:
            log.info(f"Fine-tuning: wrote provenance sidecar {model.write_sidecar(ckpt_dir)}")

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    OmegaConf.resolve(cfg)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
