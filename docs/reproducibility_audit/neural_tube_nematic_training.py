"""Isolated, resumable neural-tube training for the DARE3D audit.

The production package, released checkpoints, released predictions, and manuscript are
read-only inputs. All checkpoints and logs are written below the audit experiment root.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import lightning as L
import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader

import neural_tube_nematic_experiment as common


SEGMENTATION_ROOT = common.RUN_ROOT / "segmentation_retrained"
REGRESSION_ROOT = common.RUN_ROOT / "regression_nematic_retrained"

SEGMENTATION_PHYSICAL_BATCH = 16
SEGMENTATION_ACCUMULATION = 2
REGRESSION_BATCH = 32


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def stage_root(stage: str) -> Path:
    if stage == "segmentation":
        return SEGMENTATION_ROOT
    if stage == "regression":
        return REGRESSION_ROOT
    raise ValueError(stage)


def stage_command(stage: str) -> str:
    return f"train-{stage}"


def state_dict_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def seed_everything() -> None:
    L.seed_everything(common.SEED, workers=True)
    random.seed(common.SEED)
    np.random.seed(common.SEED)
    torch.manual_seed(common.SEED)
    torch.cuda.manual_seed_all(common.SEED)


def stage_spec(stage: str) -> dict[str, Any]:
    if stage == "segmentation":
        return {
            "root": SEGMENTATION_ROOT,
            "physical_batch_size": SEGMENTATION_PHYSICAL_BATCH,
            "accumulate_grad_batches": SEGMENTATION_ACCUMULATION,
            "effective_batch_size": (
                SEGMENTATION_PHYSICAL_BATCH * SEGMENTATION_ACCUMULATION
            ),
            "monitor": "val/iou",
            "monitor_mode": "max",
            "expected_train_length": 1000,
            "expected_validation_length": 76,
            "expected_train_unique": 100,
            "expected_validation_unique": 76,
            "parameter_count": 114869752,
            "compatibility_deviation": (
                "Physical batch 16 with two-step gradient accumulation preserves "
                "effective batch 32. BatchNorm statistics therefore use 16 rather "
                "than the archived physical batch 32. This was strictly required "
                "because the exact batch allocated 29.04 GB on a 16 GB GPU."
            ),
        }
    if stage == "regression":
        return {
            "root": REGRESSION_ROOT,
            "physical_batch_size": REGRESSION_BATCH,
            "accumulate_grad_batches": 1,
            "effective_batch_size": REGRESSION_BATCH,
            "monitor": "val/loss",
            "monitor_mode": "min",
            "expected_train_length": 2000,
            "expected_validation_length": 2000,
            "expected_train_unique": 222,
            "expected_validation_unique": 123,
            "parameter_count": 1605268,
            "compatibility_deviation": None,
        }
    raise ValueError(stage)


class AuditStateCallback(Callback):
    def __init__(self, stage: str):
        super().__init__()
        self.stage = stage
        self.root = stage_root(stage)
        self.state_path = self.root / "run_state.json"
        self.checkpoint_dir = self.root / "checkpoints"

    def _save(self, trainer: Trainer, status: str) -> None:
        checkpoint_callback = next(
            (
                callback
                for callback in trainer.callbacks
                if isinstance(callback, ModelCheckpoint)
            ),
            None,
        )
        state = {
            "status": status,
            "updated_at": common.now_iso(),
            "stage": self.stage,
            "current_epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "max_epochs": trainer.max_epochs,
            "callback_metrics": {
                key: common.jsonable(value)
                for key, value in trainer.callback_metrics.items()
            },
            "best_model_path": (
                checkpoint_callback.best_model_path
                if checkpoint_callback is not None
                else ""
            ),
            "best_model_score": (
                common.jsonable(checkpoint_callback.best_model_score)
                if checkpoint_callback is not None
                else None
            ),
            "last_checkpoint": common.rel(self.checkpoint_dir / "last.ckpt"),
            "resume_command": (
                f"& '{sys.executable}' '{common.rel(Path(__file__))}' "
                f"{stage_command(self.stage)} --resume"
            ),
        }
        common.atomic_write_json(self.state_path, state)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not trainer.sanity_checking:
            self._save(trainer, "training")

    def on_exception(self, trainer, pl_module, exception) -> None:
        self._save(trainer, "interrupted_or_failed")


def make_model(stage: str, spec: dict[str, Any]):
    if stage == "segmentation":
        return common.make_segmentation_model(
            physical_batch=spec["physical_batch_size"],
            accumulation=spec["accumulate_grad_batches"],
        )
    return common.make_regression_model(corrected_loss=True)


def initialize_data(stage: str, spec: dict[str, Any]):
    train_data = common.make_dataset(stage, "train")
    validation_data = common.make_dataset(stage, "validation")
    train_data.set_augmentations(common.make_augmentation(stage))
    train_data.init()
    validation_data.init()

    actual = {
        "train_length": len(train_data),
        "validation_length": len(validation_data),
        "train_unique": (
            len(train_data.sequences_index)
            if stage == "segmentation"
            else len(train_data.crops)
        ),
        "validation_unique": (
            len(validation_data.sequences_index)
            if stage == "segmentation"
            else len(validation_data.crops)
        ),
    }
    expected = {
        "train_length": spec["expected_train_length"],
        "validation_length": spec["expected_validation_length"],
        "train_unique": spec["expected_train_unique"],
        "validation_unique": spec["expected_validation_unique"],
    }
    if actual != expected:
        raise AssertionError(f"{stage} dataset changed: actual={actual}, expected={expected}")

    batch_size = spec["physical_batch_size"]
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
    )
    return train_data, validation_data, train_loader, validation_loader, actual


def training_config(stage: str, spec: dict[str, Any]) -> dict[str, Any]:
    archived = common.archived_config(stage)
    return {
        "created_at": common.now_iso(),
        "stage": stage,
        "source_commit": common.git_head(),
        "seed": common.SEED,
        "production_source_modified": False,
        "train_movie": common.SPLITS["train"]["movie"],
        "validation_movie": common.SPLITS["validation"]["movie"],
        "test_movie_reserved_until_evaluation": common.SPLITS["test"]["movie"],
        "source_scale_xyz_um": common.VOXEL_SCALE_XYZ_UM,
        "target_scale_um": common.TARGET_SCALE_UM,
        "scale_provenance": "manuscript main.tex:128,333-338; scales.json absent",
        "configured_epochs": common.MAX_EPOCHS,
        "physical_batch_size": spec["physical_batch_size"],
        "accumulate_grad_batches": spec["accumulate_grad_batches"],
        "effective_batch_size": spec["effective_batch_size"],
        "compatibility_deviation": spec["compatibility_deviation"],
        "architecture": common.jsonable(
            common.OmegaConf.to_container(archived.model.net, resolve=True)
        ),
        "optimizer": common.jsonable(
            common.OmegaConf.to_container(archived.model.optimizer, resolve=True)
        ),
        "scheduler": common.jsonable(
            common.OmegaConf.to_container(archived.model.scheduler, resolve=True)
        ),
        "loss": (
            common.jsonable(
                common.OmegaConf.to_container(
                    archived.model.criterion, resolve=True
                )
            )
            if stage == "segmentation"
            else {
                "orientation": (
                    "mean(90*(1-(u_pred dot u_true)^2)); nematic-sign and "
                    "division-axis-roll invariant"
                ),
                "length": "32*mean(abs(normalized predicted - true length))",
                "evaluation_metric": (
                    "degrees(acos(clip(abs(unit predicted axis dot unit true "
                    "axis), 0, 1)))"
                ),
            }
        ),
        "checkpoint_selection": {
            "monitor": spec["monitor"],
            "mode": spec["monitor_mode"],
            "save_top_k": 1,
            "save_last": True,
        },
        "trainer": {
            "min_epochs": 10,
            "max_epochs": common.MAX_EPOCHS,
            "accelerator": "gpu",
            "devices": 1,
            "precision": 32,
            "log_every_n_steps": 5,
            "check_val_every_n_epoch": 1,
            "deterministic": False,
            "detect_anomaly": False,
            "num_sanity_val_steps": 2,
        },
        "gpu": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


def run_training(stage: str, resume: bool) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    spec = stage_spec(stage)
    root = spec["root"]
    checkpoint_dir = root / "checkpoints"
    state_path = root / "run_state.json"
    result_path = root / "training_result.json"
    root.mkdir(parents=True, exist_ok=True)

    if result_path.exists():
        existing_result = json.loads(result_path.read_text(encoding="utf-8"))
        if existing_result.get("status") == "trained":
            print(f"Refusing to overwrite completed training: {result_path}")
            return existing_result

    checkpoints = list(checkpoint_dir.glob("*.ckpt")) if checkpoint_dir.exists() else []
    if checkpoints and not resume:
        raise FileExistsError(
            f"Checkpoints already exist in {checkpoint_dir}; use --resume"
        )
    last_checkpoint = checkpoint_dir / "last.ckpt"
    if resume and not last_checkpoint.is_file():
        raise FileNotFoundError(f"Cannot resume without {last_checkpoint}")

    seed_everything()
    model = make_model(stage, spec)
    parameter_count = sum(parameter.numel() for parameter in model.net.parameters())
    if parameter_count != spec["parameter_count"]:
        raise AssertionError(
            f"Unexpected {stage} parameter count: {parameter_count}"
        )
    process_initial_hash = state_dict_sha256(model.net)
    prior_state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if resume and state_path.exists()
        else {}
    )
    original_initial_hash = prior_state.get(
        "fresh_network_state_sha256", process_initial_hash
    )

    common.atomic_write_json(root / "training_config.json", training_config(stage, spec))
    common.atomic_write_json(
        state_path,
        {
            "status": "initializing_data",
            "updated_at": common.now_iso(),
            "stage": stage,
            "seed": common.SEED,
            "fresh_initialization": not resume,
            "fresh_network_state_sha256": original_initial_hash,
            "current_process_network_state_sha256_before_resume": process_initial_hash,
            "parameter_count": parameter_count,
            "resume_checkpoint": common.rel(last_checkpoint) if resume else None,
        },
    )

    (
        train_data,
        validation_data,
        train_loader,
        validation_loader,
        dataset_counts,
    ) = initialize_data(stage, spec)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="epoch_{epoch:03d}",
        monitor=spec["monitor"],
        mode=spec["monitor_mode"],
        save_last=True,
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(logging_interval="epoch"),
        AuditStateCallback(stage),
    ]
    csv_logger = CSVLogger(
        save_dir=str(root / "training_logs"),
        name="csv",
        version=0,
    )
    tensorboard_logger = TensorBoardLogger(
        save_dir=str(root / "training_logs"),
        name="tensorboard",
        version=0,
        default_hp_metric=False,
    )
    trainer = Trainer(
        default_root_dir=str(root),
        min_epochs=10,
        max_epochs=common.MAX_EPOCHS,
        accelerator="gpu",
        devices=1,
        precision=32,
        log_every_n_steps=5,
        check_val_every_n_epoch=1,
        deterministic=False,
        accumulate_grad_batches=spec["accumulate_grad_batches"],
        detect_anomaly=False,
        callbacks=callbacks,
        logger=[csv_logger, tensorboard_logger],
        num_sanity_val_steps=2,
    )

    started = time.perf_counter()
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
        ckpt_path=str(last_checkpoint) if resume else None,
    )
    elapsed = time.perf_counter() - started
    best_path = Path(checkpoint_callback.best_model_path)
    if not best_path.is_file():
        raise FileNotFoundError(best_path)
    if not last_checkpoint.is_file():
        raise FileNotFoundError(last_checkpoint)

    result = {
        "status": "trained",
        "completed_at": common.now_iso(),
        "stage": stage,
        "source_commit": common.git_head(),
        "seed": common.SEED,
        "fresh_initialization": not resume,
        "fresh_network_state_sha256": original_initial_hash,
        "parameter_count": parameter_count,
        "dataset_counts": dataset_counts,
        "physical_batch_size": spec["physical_batch_size"],
        "accumulate_grad_batches": spec["accumulate_grad_batches"],
        "effective_batch_size": spec["effective_batch_size"],
        "compatibility_deviation": spec["compatibility_deviation"],
        "epochs_completed": trainer.current_epoch,
        "global_step": trainer.global_step,
        "elapsed_seconds": elapsed,
        "best_model_path": common.rel(best_path),
        "best_model_sha256": common.sha256(best_path),
        "best_model_score": common.jsonable(checkpoint_callback.best_model_score),
        "last_model_path": common.rel(last_checkpoint),
        "last_model_sha256": common.sha256(last_checkpoint),
        "training_csv": common.rel(
            root / "training_logs/csv/version_0/metrics.csv"
        ),
        "tensorboard_directory": common.rel(
            root / "training_logs/tensorboard/version_0"
        ),
        "training_log": common.rel(root / "training.log"),
    }
    common.atomic_write_json(result_path, result)
    common.atomic_write_json(state_path, result)
    return result


def dispatch(args) -> dict[str, Any]:
    stage = args.command.removeprefix("train-")
    return run_training(stage, args.resume)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for stage in ("segmentation", "regression"):
        stage_parser = subparsers.add_parser(stage_command(stage))
        stage_parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    stage = args.command.removeprefix("train-")
    root = stage_root(stage)
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "training.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as log_stream:
        tee_out = Tee(sys.stdout, log_stream)
        tee_err = Tee(sys.stderr, log_stream)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(
            tee_err
        ):
            print(
                f"\n[{common.now_iso()}] command={args.command} resume={args.resume}",
                flush=True,
            )
            try:
                result = dispatch(args)
                print(
                    f"{args.command} complete: status={result.get('status')}",
                    flush=True,
                )
            except BaseException:
                failure = {
                    "status": "failed",
                    "updated_at": common.now_iso(),
                    "command": args.command,
                    "resume": args.resume,
                    "traceback": traceback.format_exc(),
                }
                common.atomic_write_json(root / "training_failure.json", failure)
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
