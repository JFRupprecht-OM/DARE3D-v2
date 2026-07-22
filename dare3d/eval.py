import json
import os
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

OmegaConf.register_new_resolver("eval", eval, replace=True)

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

from dare3d.metrics.infer_measure import (infer_and_evaluate_regression,
                                             infer_and_evaluate_segmentation)
from dare3d.models.finetune import load_net_state_dict
from dare3d.utils import (RankedLogger, extras, instantiate_loggers,
                             log_hyperparameters, task_wrapper)


class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

log = RankedLogger(__name__, rank_zero_only=True)

def evaluate_segmentation(cfg: DictConfig):
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    load_net_state_dict(model.net, cfg.ckpt_path, stage="segmentation")

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
        
    mlflow_logger = None
    for logg in logger:
        if isinstance(logg, MLFlowLogger):
            mlflow_logger = logg

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    cfg.trainer.accelerator = cfg.device
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.lightning_test:
        log.info("Starting testing!")
        # The network is already loaded strictly above. Passing the checkpoint here would make
        # Lightning reload version-sensitive criterion and TorchMetrics buffers.
        trainer.test(model=model, datamodule=datamodule, ckpt_path=None)
    model.net.eval()

    output_dir = cfg.model_dir

    device_name = cfg.device
    net = model.net.to(device_name)

    datamodule.data_test.init(preprocess=False)
    datamodule.data_test.make_masks()

    stats, info = infer_and_evaluate_segmentation(datamodule.data_test, 
                                    net,
                                    device=device_name,
                                    crop_size=cfg.crop_size,
                                    batch_size=cfg.data.batch_size,
                                    multithread=cfg.multithread,
                                    threshold=cfg.threshold,
                                    iteration_method=cfg.iteration_method,
                                    distance_mode=cfg.distance_mode,
                                    distance_threshold=cfg.distance_threshold,
                                    min_weighted_prob=cfg.min_weighted_prob,
                                    output_dir=output_dir)
    
    metric_dict = trainer.callback_metrics
    if mlflow_logger is not None:
        mlflow_logger.log_metrics(stats)
    else:
        log.warning("No MLFlow logger configured; skipping segmentation metric logging.")
    return stats, info, output_dir, metric_dict

def evaluate_regression(cfg: DictConfig, info):
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    load_net_state_dict(model.net, cfg.ckpt_path, stage="regression")

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
        
    mlflow_logger = None
    for logg in logger:
        if isinstance(logg, MLFlowLogger):
            mlflow_logger = logg

    model.net.eval()
    
    device_name = cfg.device
    net = model.net.to(device_name)
    output_dir = cfg.model_dir

    datamodule.data_test.init(preprocess=False)
    # datamodule.data_test.rescale()
    datamodule.data_test.pad_images()
    datamodule.data_test._normalize(datamodule.data_test.renorm)

    stats = infer_and_evaluate_regression(dataset=datamodule.data_test,
                                          net=net, 
                                          device=device_name,
                                          info=info,
                                          output_dir=output_dir)
    def format_stats(stats):
        flatten_metrics = {}
        for key in stats.keys():
            for metric_key, metric_value in stats[key].items():
                flatten_metrics[f"{key}/{metric_key}"] = metric_value
        return flatten_metrics
        
    if mlflow_logger is not None:
        mlflow_logger.log_metrics(format_stats(stats))
    else:
        log.warning("No MLFlow logger configured; skipping regression metric logging.")
    return stats

def evaluate(cfg, reg_cfg) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    
    stats = {}
    seg_stats, info, output_dir, metric_dict = evaluate_segmentation(cfg)
    stats["segmentation_results"] = seg_stats
    
    # Eval regression if specified
    if reg_cfg is not None:
        reg_stats = evaluate_regression(reg_cfg, info)
        stats["regression_results"] = reg_stats
    
    print("Final results:")
    pprint(stats)

    output_csv = os.path.join(output_dir, "stats.csv")
    print(f"Scores are written to: {output_csv}")
    with open(output_csv, "w") as file:
        json.dump(stats, file, cls=NumpyFloatValuesEncoder)

# `_target_` paths that older saved train configs may reference, mapped to the
# class's current location, so eval can re-instantiate a model trained on an
# earlier revision of the codebase (the datamodule was renamed).
_LEGACY_TARGETS = {
    "dare3d.data.deletme_datamodule.DeletmeDataModule": "dare3d.data.dare_datamodule.DareDataModule",
}


def _remap_legacy_targets(cfg: DictConfig) -> DictConfig:
    """Rewrite renamed ``_target_`` paths in a loaded train config (in any nested
    node) to their current location, so a model trained on an older revision still
    instantiates. Interpolations are preserved — nothing is resolved here."""
    container = OmegaConf.to_container(cfg, resolve=False)

    def _walk(node):
        if isinstance(node, dict):
            target = node.get("_target_")
            if isinstance(target, str) and target in _LEGACY_TARGETS:
                node["_target_"] = _LEGACY_TARGETS[target]
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for value in node:
                _walk(value)

    _walk(container)
    remapped = OmegaConf.create(container)
    OmegaConf.set_struct(remapped, False)
    return remapped


def load_config(cfg, allow_missing=False):
    hydra_config_path = os.path.join(cfg.model_dir, cfg.hydra_dir, "config.yaml")
    checkpoint_config_path = os.path.join(cfg.model_dir, cfg.ckpt_dir, cfg.ckpt_name)
    
    if not os.path.exists(hydra_config_path):
        if allow_missing:
            return None
        raise ValueError(f"Could not find hydra config file path: {hydra_config_path}")
    if not os.path.exists(checkpoint_config_path):
        if allow_missing:
            return None
        raise ValueError(f"Could not find checkpoint path: {checkpoint_config_path}")
    
    train_cfg = OmegaConf.load(hydra_config_path)
    OmegaConf.set_struct(train_cfg, None)
    train_cfg = _remap_legacy_targets(train_cfg)  # tolerate renamed _target_ paths from older runs
    train_cfg.merge_with(cfg)
    
    cfg = train_cfg

    OmegaConf.resolve(cfg)

    # Edit checkpoint path    
    cfg.ckpt_path = checkpoint_config_path
    return cfg

@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # Look for hydra config
    segmentation_cfg = load_config(cfg.segmentation)
    regression_cfg = load_config(cfg.regression, allow_missing=True)

    extras(segmentation_cfg)
    if regression_cfg is not None:
        extras(regression_cfg)
        regression_cfg.cell_radius = segmentation_cfg.cell_radius
    
    
    evaluate(segmentation_cfg, regression_cfg)

if __name__ == "__main__":
    main()
