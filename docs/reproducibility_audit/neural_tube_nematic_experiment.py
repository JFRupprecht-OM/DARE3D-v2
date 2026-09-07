"""Audit-only neural-tube data, geometry, and GPU preflight.

This script never modifies production code, released data, released checkpoints,
or manuscript files. Generated evidence is confined to the reproducibility-audit
directory.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import io
import json
import math
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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import hydra
import lightning as L
import numpy as np
import tifffile
import torch
from lightning.pytorch.utilities.seed import isolate_rng
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import nematic_retraining_experiment as nematic_common
from dare3d.metrics.infer_measure import (
    evaluate_segmentation,
    infer_and_evaluate_segmentation,
)
from dare3d.models.finetune import load_net_state_dict
from dare3d.models.regression_module import RegressionLitModule
from dare3d.models.segmentation_module import SegmentationLitModule

SOURCE_COMMIT = "fe2b14d732359f2bdaf8b197574ad818899ce123"
SEED = 12345
DATA_ROOT = REPO / "DARE3d_data_190326/Neural_tube_160226"
RELEASED_SEGMENTATION_RUN = (
    DATA_ROOT / "weights/segmentation3d_new_set_og/runs/12-01-26"
)
RELEASED_REGRESSION_RUN = (
    DATA_ROOT / "weights/regression3d_new_set_og/runs/12-01-26"
)
SEGMENTATION_CONFIG = RELEASED_SEGMENTATION_RUN / ".hydra/config.yaml"
REGRESSION_CONFIG = RELEASED_REGRESSION_RUN / ".hydra/config.yaml"
RELEASED_SEGMENTATION_CHECKPOINT = (
    RELEASED_SEGMENTATION_RUN / "checkpoints/epoch_057.ckpt"
)
RELEASED_REGRESSION_CHECKPOINT = (
    RELEASED_REGRESSION_RUN / "checkpoints/epoch_147.ckpt"
)
RELEASED_STATS = RELEASED_SEGMENTATION_RUN / "stats.csv"
RELEASED_VALIDATION_PROBABILITY = RELEASED_SEGMENTATION_RUN / "movie_I2.tif"

EXPERIMENT_ROOT = HERE / "neural_tube_nematic_full_pipeline"
RUN_ROOT = EXPERIMENT_ROOT / "seed_12345"
PREFLIGHT_ROOT = RUN_ROOT / "preflight"
INVENTORY_JSON = PREFLIGHT_ROOT / "data_inventory.json"
PREFLIGHT_JSON = PREFLIGHT_ROOT / "preflight.json"
CONFIG_JSON = RUN_ROOT / "experiment_config.json"
RUN_LOG = RUN_ROOT / "run.log"

SPLITS = {
    "train": {
        "manuscript_set": 1,
        "movie": "movie_E",
        "frames": 26,
        "complete_pairs": 226,
        "im_dir": DATA_ROOT / "trainingset/movie1/im",
        "label_dir": DATA_ROOT / "trainingset/movie1/label",
    },
    "validation": {
        "manuscript_set": 2,
        "movie": "movie_I2",
        "frames": 21,
        "complete_pairs": 145,
        "im_dir": DATA_ROOT / "trainingset/movie2/im",
        "label_dir": DATA_ROOT / "trainingset/movie2/label",
    },
    "test": {
        "manuscript_set": 3,
        "movie": "movie_M",
        "frames": 11,
        "complete_pairs": 104,
        "im_dir": DATA_ROOT / "test_input/im",
        "label_dir": DATA_ROOT / "test_input/label",
    },
}

# Production internally swaps on-disk T,Z,Y,X to T,X,Y,Z. The manuscript gives
# 0.208 um in-plane and 1 um between z planes.
VOXEL_SCALE_XYZ_UM = [0.208, 0.208, 1.0]
TARGET_SCALE_UM = 1.0
EXPECTED_TARGET_XYZ = [212, 212, 10]
SEGMENTATION_EFFECTIVE_BATCH = 32
SEGMENTATION_STEPS_PER_EPOCH = 1000
REGRESSION_STEPS_PER_EPOCH = 2000
MAX_EPOCHS = 200

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
    commands = (
        ["git", "-c", f"safe.directory={REPO.as_posix()}", "diff", "--quiet"],
        [
            "git",
            "-c",
            f"safe.directory={REPO.as_posix()}",
            "diff",
            "--cached",
            "--quiet",
        ],
    )
    return all(
        subprocess.run(command, cwd=REPO, check=False).returncode == 0
        for command in commands
    )


def required_files() -> list[Path]:
    files = [
        SEGMENTATION_CONFIG,
        REGRESSION_CONFIG,
        RELEASED_SEGMENTATION_CHECKPOINT,
        RELEASED_REGRESSION_CHECKPOINT,
        RELEASED_STATS,
        RELEASED_VALIDATION_PROBABILITY,
        REPO / "manuscript_280826version/main.tex",
    ]
    for split in SPLITS.values():
        files.extend(
            [
                split["im_dir"] / f"{split['movie']}.tif",
                split["label_dir"] / f"{split['movie']}.tif",
            ]
        )
    return files


def archived_config(stage: str):
    path = (
        SEGMENTATION_CONFIG if stage == "segmentation" else REGRESSION_CONFIG
    )
    cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, False)
    cfg.scale_file = str(PREFLIGHT_ROOT / "__released_scales_json_is_missing__.json")
    cfg.default_scale = list(VOXEL_SCALE_XYZ_UM)
    cfg.target_scale = TARGET_SCALE_UM
    return cfg


def resolved_dataset_node(stage: str, role: str):
    cfg = archived_config(stage)
    node_name = {
        "train": "train_data",
        "validation": "val_data",
        "test": "test_data",
    }[role]
    node = cfg.data[node_name]
    split = SPLITS[role]
    node.im_folder = str(split["im_dir"])
    node.label_folder = str(split["label_dir"])
    node.scale_file = str(PREFLIGHT_ROOT / "__released_scales_json_is_missing__.json")
    node.default_scale = list(VOXEL_SCALE_XYZ_UM)
    node.target_scale = TARGET_SCALE_UM
    node.training = role == "train"
    node.time_axis_padding = 1
    if stage == "segmentation":
        node.sparse_folder = (
            str(split["im_dir"].parent / "weights") if role == "train" else None
        )
    return OmegaConf.create(OmegaConf.to_container(node, resolve=True))


def make_dataset(stage: str, role: str):
    return hydra.utils.instantiate(resolved_dataset_node(stage, role))


def make_augmentation(stage: str):
    cfg = archived_config(stage)
    node = OmegaConf.create(
        OmegaConf.to_container(cfg.data.augmentation, resolve=True)
    )
    return hydra.utils.instantiate(node)


def resolved_model_node(stage: str, physical_batch: int = 32, accumulation: int = 1):
    cfg = archived_config(stage)
    cfg.data.batch_size = physical_batch
    cfg.trainer.accumulate_grad_batches = accumulation
    return OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))


def make_segmentation_model(
    physical_batch: int = 32, accumulation: int = 1
) -> SegmentationLitModule:
    node = resolved_model_node("segmentation", physical_batch, accumulation)
    net = hydra.utils.instantiate(node.net)
    criterion = hydra.utils.instantiate(node.criterion)
    optimizer = hydra.utils.instantiate(node.optimizer)
    scheduler = hydra.utils.instantiate(node.scheduler)
    return SegmentationLitModule(
        net=net,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        compile=False,
        scheduler_interval="step",
    )


def make_regression_model(corrected_loss: bool) -> RegressionLitModule:
    node = resolved_model_node("regression")
    net = hydra.utils.instantiate(node.net)
    optimizer = hydra.utils.instantiate(node.optimizer)
    scheduler = hydra.utils.instantiate(node.scheduler)
    if corrected_loss:
        criterion = nematic_common.angle_len_loss(
            len_loss=nematic_common.length_loss_wrapper(crop_size=32),
            angle_loss=nematic_common.nematic_axis_projector_loss(
                scale_degrees=90.0
            ),
        )
    else:
        criterion = hydra.utils.instantiate(node.criterion)
    return RegressionLitModule(
        net=net,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        compile=False,
    )


def model_record(model) -> dict[str, Any]:
    parameters = list(model.net.parameters())
    return {
        "class": f"{type(model.net).__module__}.{type(model.net).__name__}",
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
        "parameter_bytes": sum(
            parameter.numel() * parameter.element_size()
            for parameter in parameters
        ),
        "state_tensor_count": len(model.net.state_dict()),
    }


def label_inventory(path: Path) -> dict[str, Any]:
    label = tifffile.imread(path)
    per_frame_pairs = []
    orphan_occurrences = []
    for frame_index, frame in enumerate(label):
        values = {int(value) for value in np.unique(frame) if value != 0}
        complete = 0
        for value in values:
            counterpart = value - 1 if value % 2 == 0 else value + 1
            if counterpart not in values:
                orphan_occurrences.append(
                    {
                        "frame": frame_index,
                        "label": value,
                        "missing_counterpart": counterpart,
                    }
                )
            elif value % 2 == 1:
                complete += 1
        per_frame_pairs.append(complete)
    nonzero = np.unique(label)
    nonzero = nonzero[nonzero != 0]
    return {
        "path": rel(path),
        "shape_tzyx": list(label.shape),
        "dtype": str(label.dtype),
        "complete_pairs": int(sum(per_frame_pairs)),
        "complete_pairs_per_frame": per_frame_pairs,
        "orphan_occurrences": orphan_occurrences,
        "global_unique_nonzero_labels": int(len(nonzero)),
        "maximum_label": int(nonzero.max()) if len(nonzero) else 0,
    }


def build_inventory() -> dict[str, Any]:
    files = {}
    splits = {}
    for role, spec in SPLITS.items():
        image = spec["im_dir"] / f"{spec['movie']}.tif"
        label = spec["label_dir"] / f"{spec['movie']}.tif"
        with tifffile.TiffFile(image) as tif:
            series = tif.series[0]
            image_meta = {
                "path": rel(image),
                "shape_tzyx": list(series.shape),
                "dtype": str(series.dtype),
                "axes": series.axes,
                "pages": len(tif.pages),
                "imagej_metadata": tif.imagej_metadata,
                "ome_metadata_present": tif.ome_metadata is not None,
                "voxel_size_metadata_present": bool(
                    tif.imagej_metadata
                    and any(
                        key in tif.imagej_metadata
                        for key in ("spacing", "unit", "xorigin", "yorigin")
                    )
                ),
            }
        label_meta = label_inventory(label)
        splits[role] = {
            "manuscript_set": spec["manuscript_set"],
            "movie": spec["movie"],
            "expected_frames": spec["frames"],
            "expected_complete_pairs": spec["complete_pairs"],
            "image": image_meta,
            "label": label_meta,
        }
        for path in (image, label):
            files[rel(path)] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    for path in (
        SEGMENTATION_CONFIG,
        REGRESSION_CONFIG,
        RELEASED_SEGMENTATION_CHECKPOINT,
        RELEASED_REGRESSION_CHECKPOINT,
        RELEASED_STATS,
        RELEASED_VALIDATION_PROBABILITY,
    ):
        files[rel(path)] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    result = {
        "created_at": now_iso(),
        "source_commit": git_head(),
        "scope": "new neural-tube raw data plus released comparators",
        "splits": splits,
        "files": files,
        "scale_provenance": {
            "scales_json_present": any(DATA_ROOT.rglob("scales.json")),
            "tiff_voxel_metadata_present": any(
                split["image"]["voxel_size_metadata_present"]
                for split in splits.values()
            ),
            "chosen_internal_xyz_um": VOXEL_SCALE_XYZ_UM,
            "target_um": TARGET_SCALE_UM,
            "source": (
                "manuscript main.tex:128 gives 0.208 um in-plane and 1 um "
                "between z planes; main.tex:333-338 requires isotropic 1 um"
            ),
            "limitation": (
                "The original scales.json is absent, so exact historical "
                "byte-level scale provenance remains unavailable."
            ),
        },
    }
    assertions = {}
    for role, split in splits.items():
        assertions[f"{role}_frame_count_matches_manuscript"] = (
            split["image"]["shape_tzyx"][0] == split["expected_frames"]
        )
        assertions[f"{role}_pair_count_matches_manuscript"] = (
            split["label"]["complete_pairs"]
            == split["expected_complete_pairs"]
        )
        assertions[f"{role}_image_label_shapes_match"] = (
            split["image"]["shape_tzyx"] == split["label"]["shape_tzyx"]
        )
    assertions["raw_split_image_hashes_are_distinct"] = (
        len(
            {
                files[split["image"]["path"]]["sha256"]
                for split in splits.values()
            }
        )
        == 3
    )
    result["assertions"] = assertions
    result["all_assertions_pass"] = all(assertions.values())
    atomic_write_json(INVENTORY_JSON, result)
    return result


def dataset_record(dataset, stage: str, role: str) -> dict[str, Any]:
    base = {
        "stage": stage,
        "role": role,
        "movie_names": list(dataset.movie_names),
        "original_internal_shapes_txyz": [
            list(shape) for shape in dataset.original_movies_shape
        ],
        "processed_shapes_txyz": [
            list(movie.shape) for movie in dataset.movies_im
        ],
        "computed_target_shapes_txyz": [
            list(dataset._compute_target_shape(shape, movie_name=name))
            for shape, name in zip(
                dataset.original_movies_shape, dataset.movie_names
            )
        ],
        "length": len(dataset),
        "time_axis_padding_effective": dataset.time_axis_padding,
    }
    if stage == "segmentation":
        base.update(
            {
                "sequence_count": len(dataset.sequences_index),
                "positive_sequence_count": len(dataset.positive_sample_index),
                "negative_sequence_count": len(dataset.negative_sample_index),
                "mask_shapes_txyz": [
                    list(mask.shape) for mask in dataset.movies_masks
                ],
                "sparse_training": dataset.sparse_training,
                "sparse_folder_exists": (
                    Path(dataset.sparse_folder).is_dir()
                    if dataset.sparse_folder is not None
                    else None
                ),
            }
        )
    else:
        vector_norms = [
            float(np.linalg.norm(np.asarray(second) - np.asarray(first)))
            for first, second in dataset.bipoint_crops
        ]
        base.update(
            {
                "unique_crop_count": len(dataset.crops),
                "minimum_daughter_distance_processed_voxels": min(vector_norms),
                "zero_length_axes": sum(value == 0 for value in vector_norms),
            }
        )
    return base


def build_experiment_config() -> dict[str, Any]:
    seg_cfg = archived_config("segmentation")
    reg_cfg = archived_config("regression")
    return {
        "created_at": now_iso(),
        "experiment": (
            "full neural-tube segmentation and nematic-regression retraining"
        ),
        "source_commit": SOURCE_COMMIT,
        "production_source_modified": False,
        "seed": SEED,
        "split": {
            role: {
                "manuscript_set": spec["manuscript_set"],
                "movie": spec["movie"],
            }
            for role, spec in SPLITS.items()
        },
        "scale": {
            "internal_order": "X,Y,Z after production T,Z,Y,X reader swap",
            "source_xyz_um": VOXEL_SCALE_XYZ_UM,
            "target_xyz_um": [TARGET_SCALE_UM] * 3,
            "expected_target_xyz": EXPECTED_TARGET_XYZ,
            "provenance": "manuscript main.tex:128,333-338",
            "original_scales_json": "absent",
        },
        "segmentation": {
            "released_config": rel(SEGMENTATION_CONFIG),
            "architecture": OmegaConf.to_container(
                seg_cfg.model.net, resolve=True
            ),
            "criterion": OmegaConf.to_container(
                seg_cfg.model.criterion, resolve=True
            ),
            "optimizer": OmegaConf.to_container(
                seg_cfg.model.optimizer, resolve=True
            ),
            "scheduler": OmegaConf.to_container(
                seg_cfg.model.scheduler, resolve=True
            ),
            "samples_per_epoch": SEGMENTATION_STEPS_PER_EPOCH,
            "effective_batch_size": SEGMENTATION_EFFECTIVE_BATCH,
            "configured_epochs": MAX_EPOCHS,
            "checkpoint_selection": "maximum validation IoU; top 1 plus last",
            "released_event_log": (
                "ends after epoch 153 although max_epochs is 200; released "
                "best checkpoint is epoch 57"
            ),
        },
        "regression": {
            "released_config": rel(REGRESSION_CONFIG),
            "architecture": OmegaConf.to_container(
                reg_cfg.model.net, resolve=True
            ),
            "optimizer": OmegaConf.to_container(
                reg_cfg.model.optimizer, resolve=True
            ),
            "scheduler": OmegaConf.to_container(
                reg_cfg.model.scheduler, resolve=True
            ),
            "samples_per_epoch": REGRESSION_STEPS_PER_EPOCH,
            "batch_size": 32,
            "epochs": MAX_EPOCHS,
            "checkpoint_selection": "minimum validation total loss; top 1 plus last",
            "orientation_representation": (
                "9D unconstrained matrix, symmetric SVD projection, normalized "
                "wxyz quaternion-vector division axis"
            ),
            "old_orientation_loss": (
                "mean degrees(2*acos(abs(q_pred dot q_true))) with +/-0.9999 clip"
            ),
            "new_orientation_loss": (
                "mean(90*(1-(u_pred dot u_true)^2)); sign and roll invariant"
            ),
            "unchanged_length_loss": (
                "32*mean(abs(normalized predicted minus true length))"
            ),
            "evaluation_metric": (
                "degrees(acos(clip(abs(unit axis dot unit axis),0,1)))"
            ),
        },
        "matching_and_evaluation": {
            "validation_threshold_grid": {
                "probability": [0.1, 0.25, 0.4, 0.55, 0.7],
                "weighted_probability": [0, 0.15, 0.3, 0.45, 0.6, 0.75],
            },
            "temporal_prediction_dilation": "plus/minus one frame",
            "distance_mode": "IoU",
            "minimum_iou": 0.000001,
            "iteration_method": "movie",
            "test_policy": (
                "choose thresholds on movie_I2 once, freeze them, then evaluate "
                "movie_M without tuning"
            ),
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


def run_preflight(initialize_data: bool) -> dict[str, Any]:
    missing = [path for path in required_files() if not path.is_file()]
    if not INVENTORY_JSON.is_file():
        inventory = build_inventory()
    else:
        inventory = json.loads(INVENTORY_JSON.read_text(encoding="utf-8"))
    synthetic = nematic_common.synthetic_loss_checks()
    seg_model = make_segmentation_model()
    reg_model = make_regression_model(corrected_loss=True)
    result = {
        "created_at": now_iso(),
        "source_commit_expected": SOURCE_COMMIT,
        "source_commit_observed": git_head(),
        "tracked_tree_clean": tracked_tree_is_clean(),
        "required_files": {rel(path): path.is_file() for path in required_files()},
        "inventory_assertions_pass": inventory["all_assertions_pass"],
        "synthetic_nematic_checks": synthetic,
        "models": {
            "segmentation": model_record(seg_model),
            "nematic_regression": model_record(reg_model),
        },
        "cuda": {
            "available": torch.cuda.is_available(),
            "device": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "total_memory_bytes": (
                torch.cuda.get_device_properties(0).total_memory
                if torch.cuda.is_available()
                else None
            ),
            "cudnn": torch.backends.cudnn.version(),
        },
        "resolved_dataset_configs": {
            stage: {
                role: OmegaConf.to_container(
                    resolved_dataset_node(stage, role), resolve=True
                )
                for role in SPLITS
            }
            for stage in ("segmentation", "regression")
        },
    }
    assertions = {
        "source_commit_frozen": result["source_commit_observed"] == SOURCE_COMMIT,
        "tracked_production_tree_clean": result["tracked_tree_clean"],
        "required_files_present": not missing,
        "inventory_passes": result["inventory_assertions_pass"],
        "nematic_synthetic_checks_pass": synthetic["all_assertions_pass"],
        "cuda_available": result["cuda"]["available"],
        "segmentation_parameter_count_is_archived": (
            result["models"]["segmentation"]["parameter_count"] == 114869752
        ),
        "regression_parameter_count_is_archived": (
            result["models"]["nematic_regression"]["parameter_count"] == 1605268
        ),
    }
    del seg_model, reg_model
    if initialize_data:
        records = []
        for stage in ("segmentation", "regression"):
            for role in SPLITS:
                print(f"Initializing {stage} {role}", flush=True)
                dataset = make_dataset(stage, role)
                dataset.init()
                records.append(dataset_record(dataset, stage, role))
                del dataset
        result["datasets"] = records
        by_key = {
            (record["stage"], record["role"]): record for record in records
        }
        assertions.update(
            {
                "all_target_shapes_match_manuscript_scale": all(
                    record["computed_target_shapes_txyz"][0][1:]
                    == EXPECTED_TARGET_XYZ
                    for record in records
                ),
                "segmentation_epoch_has_1000_samples": (
                    by_key[("segmentation", "train")]["length"] == 1000
                ),
                "regression_epoch_has_2000_samples": (
                    by_key[("regression", "train")]["length"] == 2000
                ),
                "all_regression_axes_defined": all(
                    by_key[("regression", role)]["zero_length_axes"] == 0
                    for role in SPLITS
                ),
                "splits_use_expected_movies": all(
                    by_key[(stage, role)]["movie_names"]
                    == [SPLITS[role]["movie"]]
                    for stage in ("segmentation", "regression")
                    for role in SPLITS
                ),
            }
        )
    result["assertions"] = assertions
    result["all_assertions_pass"] = all(assertions.values())
    result["status"] = "pass" if result["all_assertions_pass"] else "fail"
    atomic_write_json(PREFLIGHT_JSON, result)
    if not result["all_assertions_pass"]:
        failed = [name for name, passed in assertions.items() if not passed]
        raise AssertionError(f"Preflight failed: {failed}")
    return result


def memory_probe(batch_size: int) -> dict[str, Any]:
    if batch_size < 1 or SEGMENTATION_EFFECTIVE_BATCH % batch_size != 0:
        raise ValueError("Batch size must be a positive divisor of 32")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output = PREFLIGHT_ROOT / f"segmentation_memory_probe_batch_{batch_size}.json"
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        print(f"Refusing to overwrite existing probe: {output}")
        return existing
    accumulation = SEGMENTATION_EFFECTIVE_BATCH // batch_size
    L.seed_everything(SEED, workers=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    record = {
        "status": "running",
        "created_at": now_iso(),
        "physical_batch_size": batch_size,
        "gradient_accumulation_for_effective_batch_32": accumulation,
    }
    atomic_write_json(output, record)
    dataset = None
    model = None
    batch = None
    try:
        dataset = make_dataset("segmentation", "train")
        dataset.set_augmentations(make_augmentation("segmentation"))
        dataset.init()
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            shuffle=True,
        )
        model = make_segmentation_model(batch_size, accumulation).to("cuda")
        model.train()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        batch = next(iter(loader))
        losses, _, _ = model.model_step(batch)
        loss = losses["loss"]
        model.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        model.optimizer.step()
        torch.cuda.synchronize()
        record.update(
            {
                "status": "pass",
                "loss": float(loss.detach().cpu()),
                "elapsed_seconds": time.perf_counter() - started,
                "input_shape": list(batch[0]["input"].shape),
                "target_shape": list(batch[1]["heatmaps"][0].shape),
                "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
                "optimizer_step_completed": True,
                "protocol_note": (
                    "One real augmented forward/backward/AdamW step using the "
                    "archived model and loss."
                ),
            }
        )
    except torch.cuda.OutOfMemoryError as exc:
        record.update(
            {
                "status": "cuda_oom",
                "error": str(exc),
                "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
                "optimizer_step_completed": False,
            }
        )
    except BaseException as exc:
        record.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        del batch, model, dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        record["finished_at"] = now_iso()
        atomic_write_json(output, record)
    print(json.dumps(record, indent=2))
    return record


def probability_comparison(generated_path: Path) -> dict[str, Any]:
    generated = tifffile.memmap(generated_path)
    released = tifffile.memmap(RELEASED_VALIDATION_PROBABILITY)
    result = {
        "generated": rel(generated_path),
        "released": rel(RELEASED_VALIDATION_PROBABILITY),
        "generated_shape": list(generated.shape),
        "released_shape": list(released.shape),
        "generated_dtype": str(generated.dtype),
        "released_dtype": str(released.dtype),
        "generated_sha256": sha256(generated_path),
        "released_sha256": sha256(RELEASED_VALIDATION_PROBABILITY),
    }
    if generated.shape != released.shape:
        result["shape_match"] = False
        return result
    count = 0
    equal = 0
    absolute_sum = 0.0
    squared_sum = 0.0
    maximum = 0.0
    threshold_disagreement = 0
    for frame_index in range(generated.shape[0]):
        first = np.asarray(generated[frame_index], dtype=np.float32)
        second = np.asarray(released[frame_index], dtype=np.float32)
        difference = first - second
        absolute = np.abs(difference)
        count += first.size
        equal += int(np.count_nonzero(first == second))
        absolute_sum += float(np.sum(absolute, dtype=np.float64))
        squared_sum += float(
            np.sum(np.square(difference), dtype=np.float64)
        )
        maximum = max(maximum, float(np.max(absolute)))
        threshold_disagreement += int(
            np.count_nonzero((first >= 0.55) != (second >= 0.55))
        )
    result.update(
        {
            "shape_match": True,
            "value_count": count,
            "exact_value_fraction": equal / count,
            "mae": absolute_sum / count,
            "rmse": math.sqrt(squared_sum / count),
            "maximum_absolute_difference": maximum,
            "binary_disagreement_count_at_0_55": threshold_disagreement,
            "binary_disagreement_fraction_at_0_55": (
                threshold_disagreement / count
            ),
        }
    )
    return result


def replay_released_validation(batch_size: int) -> dict[str, Any]:
    output_dir = PREFLIGHT_ROOT / "released_checkpoint_validation_replay"
    result_path = output_dir / "result.json"
    existing = None
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            print(f"Refusing to overwrite completed replay: {result_path}")
            return existing
    output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    record = {
        "status": "running",
        "created_at": existing.get("created_at", now_iso()) if existing else now_iso(),
        "resumed_after_wrapper_failure": existing is not None,
        "checkpoint": rel(RELEASED_SEGMENTATION_CHECKPOINT),
        "checkpoint_sha256": sha256(RELEASED_SEGMENTATION_CHECKPOINT),
        "scale_xyz_um": VOXEL_SCALE_XYZ_UM,
        "threshold": 0.55,
        "minimum_weighted_probability": 0.15,
        "distance_mode": "iou",
        "minimum_iou": 0.000001,
        "iteration_method": "movie",
        "inference_batch_size": batch_size,
    }
    atomic_write_json(result_path, record)
    dataset = make_dataset("segmentation", "validation")
    dataset.init(preprocess=False)
    dataset.make_masks()
    model = make_segmentation_model()
    loaded = load_net_state_dict(
        model.net,
        str(RELEASED_SEGMENTATION_CHECKPOINT),
        stage="segmentation",
    )
    generated = output_dir / "movie_I2.tif"
    reused_probability = generated.is_file()
    net = model.net.to("cuda")
    net.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if reused_probability:
        y_true_movies = [movie.copy() for movie in dataset.movies_masks]
        for movie in y_true_movies:
            movie[: dataset.n_input_channels - 1] = 0
        disk_order_tzyx = tifffile.memmap(generated)
        internal_txyz = np.swapaxes(disk_order_tzyx, -1, -3)
        stats, info = evaluate_segmentation(
            y_pred_movies=[internal_txyz],
            y_true_movies=y_true_movies,
            movies_names=dataset.movie_names,
            multithread=False,
            threshold=0.55,
            iteration_method="movie",
            distance_mode="iou",
            distance_threshold=0.000001,
            min_weighted_prob=0.15,
            output_dir=str(output_dir),
        )
    else:
        stats, info = infer_and_evaluate_segmentation(
            dataset=dataset,
            model=net,
            device=torch.device("cuda:0"),
            crop_size=128,
            batch_size=batch_size,
            multithread=False,
            threshold=0.55,
            iteration_method="movie",
            distance_mode="iou",
            distance_threshold=0.000001,
            min_weighted_prob=0.15,
            output_dir=str(output_dir),
        )
    torch.cuda.synchronize()
    released_stats = json.loads(RELEASED_STATS.read_text(encoding="utf-8"))[
        "segmentation_results"
    ]
    current = {key: jsonable(value) for key, value in stats.items()}
    def component_count(value: Any) -> int:
        return (
            len(value.get("centroids", []))
            if isinstance(value, dict)
            else 0
        )
    components = [
        {
            "true": component_count(movie["true_ccs_stats"]),
            "predicted": component_count(movie["pred_ccs_stats"]),
            "matched": len(movie["matched_items"]),
        }
        for movie in info
    ]
    keys = ("tp", "fp", "fn", "precision", "recall", "fmeasure")
    agreement = {
        key: (
            current[key] == released_stats[key]
            if key in ("tp", "fp", "fn")
            else math.isclose(
                float(current[key]),
                float(released_stats[key]),
                rel_tol=0,
                abs_tol=1e-15,
            )
        )
        for key in keys
    }
    record.update(
        {
            "status": "complete",
            "finished_at": now_iso(),
            "elapsed_seconds": time.perf_counter() - started,
            "loaded_network_keys": len(loaded),
            "reused_saved_probability_without_reinference": reused_probability,
            "dataset": dataset_record(dataset, "segmentation", "validation"),
            "current_metrics": current,
            "released_metrics": released_stats,
            "metric_exact_agreement": agreement,
            "all_released_metrics_exact": all(agreement.values()),
            "components": components,
            "probability_comparison": probability_comparison(generated),
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            "interpretation": (
                "This is a predeclared geometry validation against the released "
                "movie_I2 output, not threshold tuning."
            ),
        }
    )
    atomic_write_json(result_path, record)
    return record


def dispatch(args) -> Any:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(CONFIG_JSON, build_experiment_config())
    if args.command == "inventory":
        result = build_inventory()
    elif args.command == "preflight":
        result = run_preflight(args.initialize_data)
    elif args.command == "memory-probe":
        result = memory_probe(args.batch_size)
    elif args.command == "replay-released-validation":
        result = replay_released_validation(args.batch_size)
    else:
        raise ValueError(args.command)
    print(
        f"{args.command} complete: status={result.get('status', 'complete')}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inventory")
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--initialize-data", action="store_true")
    memory_parser = subparsers.add_parser("memory-probe")
    memory_parser.add_argument("--batch-size", type=int, required=True)
    replay_parser = subparsers.add_parser("replay-released-validation")
    replay_parser.add_argument("--batch-size", type=int, default=12)
    args = parser.parse_args()

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a", encoding="utf-8", buffering=1) as log_stream:
        tee_out = Tee(sys.stdout, log_stream)
        tee_err = Tee(sys.stderr, log_stream)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(
            tee_err
        ):
            print(f"\n[{now_iso()}] command={args.command}", flush=True)
            try:
                dispatch(args)
            except BaseException:
                failure = {
                    "status": "failed",
                    "updated_at": now_iso(),
                    "command": args.command,
                    "traceback": traceback.format_exc(),
                }
                atomic_write_json(
                    PREFLIGHT_ROOT / f"{args.command}_failure.json", failure
                )
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
