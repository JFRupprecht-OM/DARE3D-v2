"""Isolated nuclei DARE3D retraining with a nematic division-axis loss.

This audit-only experiment preserves the released 9D/SVD regression head,
architecture, optimizer, scheduler, data split, augmentation, and epoch/batch
protocol.  It never edits production source, released checkpoints, or data.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import functools
import hashlib
import io
import json
import logging
import math
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import hydra
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as SciPyRotation
from torch.utils.data import DataLoader

from dare3d.data.components.angles3d import symmetric_orthogonalization
from dare3d.data.components.regress_3dataset import Regress3Dataset
from dare3d.losses.angle3d import (
    angle_len_loss,
    length_loss_wrapper,
    matrix_to_quaternion,
)
from dare3d.models.components.simple_regression_net import RegressionNet
from dare3d.models.regression_module import RegressionLitModule

SOURCE_COMMIT = "fe2b14d732359f2bdaf8b197574ad818899ce123"
DATA_ROOT = REPO / "DARE3d_data_190326/Gastruloid_241025"
RELEASED_MODEL_ROOT = DATA_ROOT / "weights/regression3d_exp10-b"
RELEASED_CONFIG = RELEASED_MODEL_ROOT / ".hydra/config.yaml"
ORIGINAL_CHECKPOINT = RELEASED_MODEL_ROOT / "checkpoints/epoch_098.ckpt"
DATA_MANIFEST = HERE / "evidence/data_manifest_sha256.csv"
EXPERIMENT_ROOT = HERE / "nematic_retraining"
RUN_ROOT = EXPERIMENT_ROOT / "seed_12345"
CHECKPOINT_DIR = RUN_ROOT / "checkpoints"
CONFIG_OUT = RUN_ROOT / "experiment_config.json"
PREFLIGHT_OUT = RUN_ROOT / "preflight.json"
STATE_OUT = RUN_ROOT / "run_state.json"
RUN_LOG = RUN_ROOT / "run.log"
SEED = 12345
BOOTSTRAP_SEED = 20260829

MOVIE_SOURCES = {
    "movie2": {
        "image": DATA_ROOT / "trainingset/movie2/im/movie2.tif",
        "label": DATA_ROOT / "trainingset/movie2/label/movie2.tif",
    },
    "movie3": {
        "image": DATA_ROOT / "trainingset/movie3/im/movie3.tif",
        "label": DATA_ROOT / "trainingset/movie3/label/movie3.tif",
    },
    "movie4": {
        "image": DATA_ROOT / "trainingset/movie4/im/movie4.tif",
        "label": DATA_ROOT / "trainingset/movie4/label/movie4.tif",
    },
}

# Internal spatial order is X,Y,Z after the production zyx reader.
NATIVE_INTERNAL_SHAPES_XYZ = {
    "movie2": (362, 305, 180),
    "movie3": (424, 411, 85),
    "movie4": (405, 394, 152),
}
# These are the exact resized shapes printed by the archived 2024 training log.
LOGGED_TARGET_SHAPES_XYZ = {
    "movie2": (330, 278, 164),
    "movie3": (387, 375, 77),
    "movie4": (405, 394, 152),
}
SPLIT = {
    "train": ("movie3", "movie4"),
    "validation": ("movie2",),
    "test": ("movie2",),
}

OmegaConf.register_new_resolver("eval", eval, replace=True)


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


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        try:
            return rel(value)
        except ValueError:
            return str(value)
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def state_dict_sha256(module: torch.nn.Module) -> str:
    stream = io.BytesIO()
    torch.save(module.state_dict(), stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={REPO.as_posix()}",
            "rev-parse",
            "HEAD",
        ],
        cwd=REPO,
        text=True,
    ).strip()


def tracked_tree_is_clean() -> bool:
    return (
        subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={REPO.as_posix()}",
                "diff",
                "--quiet",
            ],
            cwd=REPO,
            check=False,
        ).returncode
        == 0
        and subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={REPO.as_posix()}",
                "diff",
                "--cached",
                "--quiet",
            ],
            cwd=REPO,
            check=False,
        ).returncode
        == 0
    )


def manifest_hash(path: Path) -> str:
    wanted = rel(path)
    with DATA_MANIFEST.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["path"] == wanted:
                return row["sha256"]
    raise KeyError(f"{wanted} is absent from {DATA_MANIFEST}")


class FrozenMultiMovieRegress3Dataset(Regress3Dataset):
    """Production dataset behavior with explicit, log-frozen movie paths/shapes."""

    def __init__(self, movie_names: tuple[str, ...], role: str):
        self.frozen_movie_names = tuple(movie_names)
        self.role = role
        super().__init__(
            im_folder=str(REPO),
            label_folder=str(REPO),
            input_channels=[-1, 0, 1],
            scale_file=str(RUN_ROOT / "__original_scales_json_is_missing__.json"),
            default_scale=[1.0, 1.0, 1.0],
            target_scale=1.0,
            renorm="min-max",
            order_dim_img="zyx",
            training=role == "train",
            steps_per_epoch=2000,
            load_labels=True,
            time_axis_padding=1 if role == "train" else 0,
            crop_size=32,
            angle_representation="rotation_matrix_SVD",
        )

    def _list_data(self) -> None:
        self.movie_names = list(self.frozen_movie_names)

    def get_movie_scale(self, movie_name: str | None = None):
        if movie_name is None:
            raise ValueError("A movie name is required for frozen geometry")
        native = np.asarray(NATIVE_INTERNAL_SHAPES_XYZ[movie_name], dtype=np.float64)
        target = np.asarray(LOGGED_TARGET_SHAPES_XYZ[movie_name], dtype=np.float64)
        return target / native

    def _compute_target_shape(
        self, im_shape: tuple, movie_name: str | None = None
    ) -> tuple:
        if movie_name is None:
            raise ValueError("A movie name is required for frozen geometry")
        return (im_shape[0], *LOGGED_TARGET_SHAPES_XYZ[movie_name])

    def _load_data(self):
        movies = []
        bipoints = []
        self.original_movies_shape = []
        for movie_name in self.movie_names:
            source = MOVIE_SOURCES[movie_name]
            movie = self._load_image(str(source["image"]))
            movie = self.add_time_padding(movie)
            self.original_movies_shape.append(movie.shape)
            movies.append(movie)
            movie_bipoints = self._load_bipoints(
                str(source["label"].parent), movie_name
            )
            if self.time_axis_padding > 0:
                movie_bipoints = (
                    [[] for _ in range(self.time_axis_padding)] + movie_bipoints
                )
            bipoints.append(movie_bipoints)
        return movies, bipoints


def make_dataset(role: str) -> FrozenMultiMovieRegress3Dataset:
    if role == "train":
        names = SPLIT["train"]
    elif role in {"validation", "test"}:
        names = SPLIT["validation"]
    else:
        raise ValueError(role)
    return FrozenMultiMovieRegress3Dataset(names, "train" if role == "train" else role)


def rotation_axes_from_9d(
    rotation: torch.Tensor, project_prediction: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract the normalized quaternion-vector direction from a 9D rotation."""
    matrix = (
        symmetric_orthogonalization(rotation)
        if project_prediction
        else rotation.reshape(-1, 3, 3)
    )
    quaternion = matrix_to_quaternion(matrix)
    vector = quaternion[..., 1:]
    vector_norm = torch.linalg.vector_norm(vector, dim=-1)
    axis = F.normalize(vector, dim=-1, p=2, eps=1e-8)
    return axis, vector_norm


def nematic_axis_projector_loss(scale_degrees: float = 90.0):
    """Smooth sign/roll-invariant training surrogate: scale*(1-(u.v)^2)."""

    def loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        true_axis, _ = rotation_axes_from_9d(y_true, project_prediction=False)
        predicted_axis, _ = rotation_axes_from_9d(
            y_pred, project_prediction=True
        )
        dot = torch.sum(true_axis * predicted_axis, dim=-1)
        dot = torch.clamp(dot, -1.0, 1.0)
        return torch.mean(scale_degrees * (1.0 - torch.square(dot)))

    return loss


def exact_nematic_axis_error_deg(axis_pred, axis_true) -> float:
    pred = np.asarray(axis_pred, dtype=np.float64)
    true = np.asarray(axis_true, dtype=np.float64)
    pred /= np.linalg.norm(pred)
    true /= np.linalg.norm(true)
    absolute_dot = float(np.clip(abs(np.dot(pred, true)), 0.0, 1.0))
    return float(np.degrees(np.arccos(absolute_dot)))


def matrix_for_axis(axis, rotation_angle_deg: float) -> torch.Tensor:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    matrix = SciPyRotation.from_rotvec(
        axis * math.radians(rotation_angle_deg)
    ).as_matrix()
    return torch.tensor(matrix, dtype=torch.float32).reshape(1, 9)


def synthetic_loss_checks() -> dict[str, Any]:
    loss_fn = nematic_axis_projector_loss()
    x = np.asarray([1.0, 0.0, 0.0])
    y = np.asarray([0.0, 1.0, 0.0])
    arbitrary = np.asarray([0.2, 0.7, 0.4])
    arbitrary /= np.linalg.norm(arbitrary)

    identical = float(loss_fn(matrix_for_axis(x, 60), matrix_for_axis(x, 60)))
    opposite = float(loss_fn(matrix_for_axis(x, 60), matrix_for_axis(-x, 60)))
    orthogonal = float(loss_fn(matrix_for_axis(x, 60), matrix_for_axis(y, 60)))
    roll_invariant = float(
        loss_fn(
            matrix_for_axis(arbitrary, 30),
            matrix_for_axis(arbitrary, 150),
        )
    )

    generator = torch.Generator().manual_seed(SEED)
    target_raw = torch.randn(12, 9, generator=generator)
    target = symmetric_orthogonalization(target_raw).detach().reshape(12, 9)
    prediction = torch.randn(12, 9, generator=generator, requires_grad=True)
    gradient_loss = loss_fn(target, prediction)
    gradient_loss.backward()
    finite_gradient = bool(torch.isfinite(prediction.grad).all())

    exact = {
        "identical_axes_deg": exact_nematic_axis_error_deg(x, x),
        "opposite_axes_deg": exact_nematic_axis_error_deg(x, -x),
        "orthogonal_axes_deg": exact_nematic_axis_error_deg(x, y),
    }
    assertions = {
        "training_loss_identical_is_zero": abs(identical) <= 1e-4,
        "training_loss_opposite_is_zero": abs(opposite) <= 1e-4,
        "training_loss_orthogonal_is_90": abs(orthogonal - 90.0) <= 1e-4,
        "training_loss_roll_invariant": abs(roll_invariant) <= 1e-4,
        "random_batch_gradient_is_finite": finite_gradient,
        "evaluation_identical_is_exact_zero": exact["identical_axes_deg"] == 0.0,
        "evaluation_opposite_is_exact_zero": exact["opposite_axes_deg"] == 0.0,
        "evaluation_orthogonal_is_exact_90": exact["orthogonal_axes_deg"] == 90.0,
    }
    return {
        "training_loss_values": {
            "identical": identical,
            "opposite": opposite,
            "orthogonal": orthogonal,
            "same_axis_different_rotation_angles": roll_invariant,
            "random_gradient_loss": float(gradient_loss.detach()),
            "random_gradient_norm": float(prediction.grad.norm()),
        },
        "evaluation_metric_values": exact,
        "assertions": assertions,
        "all_assertions_pass": all(assertions.values()),
    }


def make_augmentation():
    archived = OmegaConf.load(RELEASED_CONFIG)
    return hydra.utils.instantiate(archived.data.augmentation)


def make_model() -> RegressionLitModule:
    net = RegressionNet(
        input_channels=[-1, 0, 1],
        im_size=32,
        angle_vec_size=9,
        start_filters=16,
        n_stages=5,
    )
    criterion = angle_len_loss(
        len_loss=length_loss_wrapper(crop_size=32),
        angle_loss=nematic_axis_projector_loss(scale_degrees=90.0),
    )
    optimizer = functools.partial(
        torch.optim.AdamW,
        lr=0.001,
        betas=[0.9, 0.999],
        weight_decay=0.0001,
    )
    scheduler = functools.partial(
        torch.optim.lr_scheduler.OneCycleLR,
        max_lr=0.001,
        steps_per_epoch=167,
        epochs=100,
        pct_start=0.1,
    )
    return RegressionLitModule(
        net=net,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        compile=False,
    )


def experiment_config() -> dict[str, Any]:
    sources = {}
    for movie_name, values in MOVIE_SOURCES.items():
        sources[movie_name] = {
            "image": rel(values["image"]),
            "image_sha256": manifest_hash(values["image"]),
            "label": rel(values["label"]),
            "label_sha256": manifest_hash(values["label"]),
            "native_internal_shape_xyz": NATIVE_INTERNAL_SHAPES_XYZ[movie_name],
            "logged_target_shape_xyz": LOGGED_TARGET_SHAPES_XYZ[movie_name],
            "effective_geometry_scale_xyz": (
                np.asarray(LOGGED_TARGET_SHAPES_XYZ[movie_name], dtype=float)
                / np.asarray(NATIVE_INTERNAL_SHAPES_XYZ[movie_name], dtype=float)
            ),
        }
    return {
        "experiment": "nuclei regression retraining with nematic axis loss",
        "created_at": now_iso(),
        "source_commit": SOURCE_COMMIT,
        "production_source_modified": False,
        "released_config": rel(RELEASED_CONFIG),
        "released_config_sha256": manifest_hash(RELEASED_CONFIG),
        "original_checkpoint": rel(ORIGINAL_CHECKPOINT),
        "original_checkpoint_sha256": manifest_hash(ORIGINAL_CHECKPOINT),
        "seed": SEED,
        "bootstrap_seed_for_later_evaluation": BOOTSTRAP_SEED,
        "split": SPLIT,
        "data_sources": sources,
        "missing_provenance": (
            "The original scales.json and historical source/environment are "
            "absent. Exact logged target shapes are reconstructed; effective "
            "per-axis target/native ratios transform labels."
        ),
        "preprocessing": {
            "reader_order": "zyx -> internal T,X,Y,Z",
            "train_time_padding": 1,
            "validation_test_time_padding": 0,
            "crop_size_xyz": [32, 32, 32],
            "input_frames": [-1, 0, 1],
            "normalization": "whole-movie min-max after resizing/crop indexing",
            "train_division_crops_expected": 526,
            "validation_division_crops_expected": 156,
            "samples_per_train_epoch": 2000,
            "samples_per_validation_epoch": 2000,
            "validation_repeats_crops_modulo_dataset_length": True,
            "augmentation": {
                "wrapper_probability": 0.5,
                "flip_probability_when_pipeline_runs": 0.5,
                "3d_rotation_probability_when_pipeline_runs": 0.5,
                "rotation_ranges_radians_xyz": [3.14, 3.14, 3.14],
                "zoom_probability_when_pipeline_runs": 0.5,
                "zoom_range": [0.9, 1.1],
            },
        },
        "orientation_representation": {
            "network_output": "9 unconstrained values",
            "projection": "symmetric SVD orthogonalization to SO(3)",
            "ground_truth": (
                "9D rotation matrix generated from the archived quaternion "
                "encoding of the annotated daughter bipoint"
            ),
            "axis_extraction": (
                "convert matrix to wxyz quaternion and normalize q[x,y,z]"
            ),
            "unchanged_architecture": True,
        },
        "loss": {
            "old": (
                "mean degrees(2*acos(abs(q_pred dot q_true))) after "
                "clipping the quaternion dot to +/-0.9999"
            ),
            "new_training": "mean(90*(1-(u_pred dot u_true)^2))",
            "new_training_name": "scaled nematic projector loss",
            "new_training_range": "[0,90]",
            "new_evaluation": "degrees(acos(abs(u_pred dot u_true)))",
            "rationale": (
                "The smooth training surrogate has exactly the nematic minima, "
                "is sign invariant and insensitive to quaternion rotation "
                "angle about the division axis, while avoiding acos endpoint "
                "gradient singularities. Evaluation uses the requested angle."
            ),
            "length_loss_unchanged": "32*mean(abs(normalized_length_error))",
            "joint_loss": "nematic projector loss + unchanged length loss",
        },
        "architecture": {
            "class": "dare3d.models.components.simple_regression_net.RegressionNet",
            "input_channels": 3,
            "crop_size": 32,
            "angle_output_size": 9,
            "start_filters": 16,
            "stages": 5,
            "expected_parameters": 5897940,
            "length_activation": "sigmoid",
            "angle_activation": "identity",
        },
        "optimization": {
            "fresh_initialization": True,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "betas": [0.9, 0.999],
            "weight_decay": 0.0001,
            "scheduler": "OneCycleLR",
            "scheduler_max_lr": 0.001,
            "scheduler_steps_per_epoch_parameter": 167,
            "scheduler_epochs_parameter": 100,
            "scheduler_pct_start": 0.1,
            "scheduler_step_interval": (
                "epoch, preserving RegressionLitModule historical behavior"
            ),
            "batch_size": 12,
            "max_epochs": 100,
            "min_epochs": 10,
            "precision": 32,
            "accelerator": "single CUDA GPU",
            "deterministic": False,
            "checkpoint_monitor": "val/loss",
            "checkpoint_mode": "min",
            "save_top_k": 1,
            "save_last": True,
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "lightning": L.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
    }


def preflight(include_dataset: bool = False) -> dict[str, Any]:
    required = [RELEASED_CONFIG, ORIGINAL_CHECKPOINT, DATA_MANIFEST]
    required.extend(
        value
        for source in MOVIE_SOURCES.values()
        for value in source.values()
    )
    missing = [str(path) for path in required if not path.is_file()]
    checks = synthetic_loss_checks()
    result = {
        "status": "pass",
        "created_at": now_iso(),
        "source_commit_expected": SOURCE_COMMIT,
        "source_commit_observed": git_head(),
        "tracked_tree_clean": tracked_tree_is_clean(),
        "required_files": {rel(path): path.is_file() for path in required},
        "synthetic_loss_checks": checks,
        "cuda": {
            "available": torch.cuda.is_available(),
            "device": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "memory_total": (
                torch.cuda.get_device_properties(0).total_memory
                if torch.cuda.is_available()
                else None
            ),
        },
    }
    assertions = {
        "source_commit_frozen": result["source_commit_observed"] == SOURCE_COMMIT,
        "tracked_tree_clean": result["tracked_tree_clean"],
        "required_files_present": not missing,
        "synthetic_checks_pass": checks["all_assertions_pass"],
        "cuda_available": result["cuda"]["available"],
    }
    if include_dataset:
        train_data = make_dataset("train")
        validation_data = make_dataset("validation")
        train_data.init()
        validation_data.init()
        train_lengths = [
            float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
            for a, b in train_data.bipoint_crops
        ]
        validation_lengths = [
            float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
            for a, b in validation_data.bipoint_crops
        ]
        result["dataset"] = {
            "train_movie_names": train_data.movie_names,
            "validation_movie_names": validation_data.movie_names,
            "train_unique_crops": len(train_data.crops),
            "validation_unique_crops": len(validation_data.crops),
            "train_loader_length": len(train_data),
            "validation_loader_length": len(validation_data),
            "train_original_internal_shapes": train_data.original_movies_shape,
            "validation_original_internal_shapes": (
                validation_data.original_movies_shape
            ),
            "train_padded_preprocessed_shapes": [
                list(movie.shape) for movie in train_data.movies_im
            ],
            "validation_padded_preprocessed_shapes": [
                list(movie.shape) for movie in validation_data.movies_im
            ],
            "minimum_train_daughter_distance_voxels": min(train_lengths),
            "minimum_validation_daughter_distance_voxels": min(
                validation_lengths
            ),
        }
        assertions.update(
            {
                "train_crop_count_matches_archived_log": (
                    len(train_data.crops) == 526
                ),
                "validation_crop_count_matches_archived_log": (
                    len(validation_data.crops) == 156
                ),
                "train_epoch_sample_count_is_2000": len(train_data) == 2000,
                "validation_epoch_sample_count_is_2000": (
                    len(validation_data) == 2000
                ),
                "all_daughter_axes_are_defined": (
                    min(train_lengths) > 0 and min(validation_lengths) > 0
                ),
            }
        )
    result["assertions"] = assertions
    result["all_assertions_pass"] = all(assertions.values())
    if not result["all_assertions_pass"]:
        result["status"] = "fail"
    atomic_write_json(PREFLIGHT_OUT, result)
    if not result["all_assertions_pass"]:
        failed = [name for name, passed in assertions.items() if not passed]
        raise AssertionError(f"Preflight failed: {failed}")
    return result


class AuditStateCallback(Callback):
    def _save(self, trainer: Trainer, status: str) -> None:
        metrics = {
            key: jsonable(value)
            for key, value in trainer.callback_metrics.items()
        }
        checkpoint_callback = trainer.checkpoint_callback
        state = {
            "status": status,
            "updated_at": now_iso(),
            "current_epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "max_epochs": trainer.max_epochs,
            "metrics": metrics,
            "best_model_path": (
                checkpoint_callback.best_model_path
                if checkpoint_callback is not None
                else ""
            ),
            "best_model_score": (
                jsonable(checkpoint_callback.best_model_score)
                if checkpoint_callback is not None
                else None
            ),
            "last_checkpoint": rel(CHECKPOINT_DIR / "last.ckpt"),
            "resume_command": (
                f"& '{sys.executable}' "
                f"'{rel(Path(__file__))}' train --resume"
            ),
        }
        atomic_write_json(STATE_OUT, state)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not trainer.sanity_checking:
            self._save(trainer, "training")

    def on_exception(self, trainer, pl_module, exception) -> None:
        self._save(trainer, "interrupted_or_failed")


def build_dataloaders(train_data, validation_data):
    train_loader = DataLoader(
        train_data,
        batch_size=12,
        num_workers=0,
        pin_memory=False,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=12,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
    )
    return train_loader, validation_loader


def run_training(resume: bool) -> dict[str, Any]:
    if CHECKPOINT_DIR.exists() and any(CHECKPOINT_DIR.glob("*.ckpt")) and not resume:
        raise FileExistsError(
            f"Checkpoints already exist in {CHECKPOINT_DIR}; use --resume"
        )
    quick = preflight(include_dataset=False)
    L.seed_everything(SEED, workers=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    train_data = make_dataset("train")
    validation_data = make_dataset("validation")
    train_data.set_augmentations(make_augmentation())

    # Preserve historical object-instantiation order: model before data init.
    model = make_model()
    parameter_count = sum(parameter.numel() for parameter in model.net.parameters())
    if parameter_count != 5897940:
        raise AssertionError(f"Unexpected parameter count: {parameter_count}")
    initial_hash = state_dict_sha256(model.net)

    atomic_write_json(
        STATE_OUT,
        {
            "status": "initializing_data",
            "updated_at": now_iso(),
            "seed": SEED,
            "fresh_initialization": not resume,
            "initial_network_state_sha256": initial_hash,
            "parameter_count": parameter_count,
            "quick_preflight": quick["all_assertions_pass"],
        },
    )
    detailed_preflight = preflight(include_dataset=True)

    # The detailed preflight initialized separate datasets; initialize the
    # exact instances attached to this model/run after releasing those objects.
    del detailed_preflight
    train_data.init()
    validation_data.init()
    if len(train_data.crops) != 526 or len(validation_data.crops) != 156:
        raise AssertionError(
            f"Crop counts changed: {len(train_data.crops)}, "
            f"{len(validation_data.crops)}"
        )
    train_loader, validation_loader = build_dataloaders(
        train_data, validation_data
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(CHECKPOINT_DIR),
        filename="epoch_{epoch:03d}",
        monitor="val/loss",
        mode="min",
        save_last=True,
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(logging_interval="epoch"),
        AuditStateCallback(),
    ]
    csv_logger = CSVLogger(
        save_dir=str(RUN_ROOT / "training_logs"),
        name="csv",
        version=0,
    )
    tensorboard_logger = TensorBoardLogger(
        save_dir=str(RUN_ROOT / "training_logs"),
        name="tensorboard",
        version=0,
        default_hp_metric=False,
    )
    trainer = Trainer(
        default_root_dir=str(RUN_ROOT),
        min_epochs=10,
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        precision=32,
        log_every_n_steps=5,
        check_val_every_n_epoch=1,
        deterministic=False,
        accumulate_grad_batches=1,
        detect_anomaly=False,
        callbacks=callbacks,
        logger=[csv_logger, tensorboard_logger],
        num_sanity_val_steps=2,
    )
    started = time.perf_counter()
    resume_path = str(CHECKPOINT_DIR / "last.ckpt") if resume else None
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
        ckpt_path=resume_path,
    )
    elapsed = time.perf_counter() - started
    best_path = Path(checkpoint_callback.best_model_path)
    if not best_path.is_file():
        raise FileNotFoundError(best_path)
    result = {
        "status": "trained",
        "completed_at": now_iso(),
        "source_commit": git_head(),
        "seed": SEED,
        "fresh_initialization": not resume,
        "initial_network_state_sha256": initial_hash,
        "parameter_count": parameter_count,
        "epochs_completed": trainer.current_epoch,
        "global_step": trainer.global_step,
        "elapsed_seconds": elapsed,
        "best_model_path": rel(best_path),
        "best_model_sha256": sha256(best_path),
        "best_model_score": jsonable(checkpoint_callback.best_model_score),
        "last_model_path": rel(CHECKPOINT_DIR / "last.ckpt"),
        "last_model_sha256": sha256(CHECKPOINT_DIR / "last.ckpt"),
        "training_csv": rel(
            RUN_ROOT / "training_logs/csv/version_0/metrics.csv"
        ),
        "tensorboard_directory": rel(
            RUN_ROOT / "training_logs/tensorboard/version_0"
        ),
        "run_log": rel(RUN_LOG),
        "evaluation_command": (
            f"& '{sys.executable}' "
            f"'docs/reproducibility_audit/nematic_retraining_evaluation.py'"
        ),
    }
    atomic_write_json(STATE_OUT, result)
    return result


def dispatch(args) -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(CONFIG_OUT, experiment_config())
    if args.command == "preflight":
        result = preflight(include_dataset=args.initialize_data)
        print(
            f"Preflight complete: {sum(result['assertions'].values())}/"
            f"{len(result['assertions'])} assertions"
        )
    elif args.command == "train":
        result = run_training(resume=args.resume)
        print(json.dumps(jsonable(result), indent=2))
    else:
        raise ValueError(args.command)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--initialize-data", action="store_true")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a", encoding="utf-8", buffering=1) as log_stream:
        tee_out = Tee(sys.stdout, log_stream)
        tee_err = Tee(sys.stderr, log_stream)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(
            tee_err
        ):
            print(f"\n[{now_iso()}] command={args.command}")
            try:
                dispatch(args)
            except Exception:
                failure = {
                    "status": "failed",
                    "updated_at": now_iso(),
                    "command": args.command,
                    "traceback": traceback.format_exc(),
                    "resume_command": (
                        f"& '{sys.executable}' "
                        f"'{rel(Path(__file__))}' train --resume"
                    ),
                }
                atomic_write_json(STATE_OUT, failure)
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
