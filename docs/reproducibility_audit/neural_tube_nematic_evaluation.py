"""Independent neural-tube evaluation for the isolated nematic audit experiment.

This module reuses production preprocessing, inference, object matching, and
non-angular metrics, but computes angular summaries independently with both the
released quaternion metric and the corrected nematic-axis metric. It never writes
to production or released-result directories. The released segmentation
checkpoint, detections, matching, and centers are fixed across the original and
retrained regression models.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import io
import json
import math
import sys
import threading
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import neural_tube_nematic_experiment as common
import neural_tube_nematic_training as training
from dare3d.data.components.angles3d import symmetric_orthogonalization
from dare3d.losses.angle3d import matrix_to_quaternion
from dare3d.metrics.infer_measure import CenterList, evaluate_segmentation
from dare3d.metrics.inference import regression_inference, segmentation_inference
from dare3d.models.finetune import load_net_state_dict


EVALUATION_ROOT = common.RUN_ROOT / "evaluation"
EVALUATION_LOG = EVALUATION_ROOT / "evaluation.log"
SEGMENTATION_ROOT = EVALUATION_ROOT / "segmentation"
CONTROLLED_ROOT = EVALUATION_ROOT / "controlled_regression_test"
END_TO_END_ROOT = EVALUATION_ROOT / "end_to_end_test"
GRID_PROBABILITIES = (0.1, 0.25, 0.4, 0.55, 0.7)
GRID_WEIGHTED_PROBABILITIES = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75)
INFERENCE_BATCH_SIZE = 12
REGRESSION_BATCH_SIZE = 32
FIXED_SEGMENTATION_MODEL = "original"
WINDOWS_WORKER_STACK_BYTES = 64 * 1024 * 1024
MODE_ORDER = (
    "all_groundtruth_centers",
    "matched_groundtruth_centers",
    "predicted_centers",
)
PIPELINE_ORDER = (
    "original_regression_fixed_segmentation",
    "retrained_regression_fixed_segmentation",
)


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


def call_with_expanded_windows_stack(function):
    """Run a numerically unchanged call outside Windows' small main-thread stack."""
    if not sys.platform.startswith("win"):
        return function()
    result = []
    failure = []

    def target():
        try:
            result.append(function())
        except BaseException as error:
            failure.append((error, error.__traceback__))

    previous_stack_size = threading.stack_size()
    threading.stack_size(WINDOWS_WORKER_STACK_BYTES)
    try:
        worker = threading.Thread(target=target, name="dare3d-expanded-stack")
        worker.start()
        worker.join()
    finally:
        threading.stack_size(previous_stack_size)
    if failure:
        error, error_traceback = failure[0]
        raise error.with_traceback(error_traceback)
    if len(result) != 1:
        raise RuntimeError("Expanded-stack worker returned no result")
    return result[0]



_production_segmentation_inference = segmentation_inference

def segmentation_inference(*args, **kwargs):
    """Call production inference with a larger Windows worker stack."""
    return call_with_expanded_windows_stack(
        lambda: _production_segmentation_inference(*args, **kwargs)
    )


_production_regression_inference = regression_inference

def regression_inference(*args, **kwargs):
    """Call production inference with a larger Windows worker stack."""
    return call_with_expanded_windows_stack(
        lambda: _production_regression_inference(*args, **kwargs)
    )
def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def trained_checkpoint(stage: str) -> Path:
    root = training.stage_root(stage)
    result_path = root / "training_result.json"
    if not result_path.is_file():
        raise FileNotFoundError(
            f"{stage} training is not complete: missing {result_path}"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "trained":
        raise RuntimeError(f"{stage} training status is {result.get('status')}")
    checkpoint = REPO / result["best_model_path"]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def checkpoint_path(stage: str, model_name: str) -> Path:
    if stage == "segmentation":
        if model_name == FIXED_SEGMENTATION_MODEL:
            return common.RELEASED_SEGMENTATION_CHECKPOINT
    elif stage == "regression":
        if model_name == "original":
            return common.RELEASED_REGRESSION_CHECKPOINT
        if model_name == "retrained":
            return trained_checkpoint("regression")
    raise ValueError((stage, model_name))


def checkpoint_metadata(stage: str, model_name: str) -> dict[str, Any]:
    path = checkpoint_path(stage, model_name)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "path": common.rel(path),
        "sha256": common.sha256(path),
        "epoch": int(checkpoint.get("epoch", -1)),
        "global_step": int(checkpoint.get("global_step", -1)),
        "lightning_version": checkpoint.get("pytorch-lightning_version"),
    }


def probability_path(model_name: str, split: str) -> Path:
    if model_name == "original" and split == "validation":
        return (
            common.PREFLIGHT_ROOT
            / "released_checkpoint_validation_replay"
            / "movie_I2.tif"
        )
    movie = common.SPLITS[split]["movie"]
    return SEGMENTATION_ROOT / model_name / split / f"{movie}.tif"


def segmentation_inference_record(model_name: str, split: str) -> Path:
    return SEGMENTATION_ROOT / model_name / split / "inference.json"


def load_segmentation_net(model_name: str):
    model = common.make_segmentation_model()
    loaded = load_net_state_dict(
        model.net,
        str(checkpoint_path("segmentation", model_name)),
        stage="segmentation",
    )
    net = model.net.to("cuda")
    net.eval()
    return net, len(loaded)


def validate_probability(path: Path, split: str) -> dict[str, Any]:
    expected_shape = (
        common.SPLITS[split]["frames"],
        10,
        1024,
        1024,
    )
    array = tifffile.memmap(path)
    if tuple(array.shape) != expected_shape:
        raise AssertionError(
            f"Probability shape changed: {array.shape} != {expected_shape}"
        )
    record = {
        "path": common.rel(path),
        "sha256": common.sha256(path),
        "shape_tzyx": list(array.shape),
        "dtype": str(array.dtype),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }
    del array
    return record


def run_segmentation_inference(
    model_name: str, split: str, batch_size: int
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError(split)
    output_path = probability_path(model_name, split)
    record_path = segmentation_inference_record(model_name, split)
    if record_path.is_file():
        existing = json.loads(record_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete" and output_path.is_file():
            print(f"Reusing completed inference: {record_path}")
            return existing

    if model_name == "original" and split == "validation":
        replay = json.loads(
            (
                common.PREFLIGHT_ROOT
                / "released_checkpoint_validation_replay"
                / "result.json"
            ).read_text(encoding="utf-8")
        )
        if replay.get("status") != "complete":
            raise RuntimeError("Released validation replay is incomplete")
        record = {
            "status": "complete",
            "completed_at": common.now_iso(),
            "model": model_name,
            "split": split,
            "reused_preflight_inference": True,
            "checkpoint": checkpoint_metadata("segmentation", model_name),
            "probability": validate_probability(output_path, split),
            "inference_batch_size": replay["inference_batch_size"],
            "source_replay_result": common.rel(
                common.PREFLIGHT_ROOT
                / "released_checkpoint_validation_replay"
                / "result.json"
            ),
        }
        common.atomic_write_json(record_path, record)
        return record

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if output_path.exists():
        raise FileExistsError(
            f"Unrecorded probability already exists; inspect before reuse: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = common.make_dataset("segmentation", split)
    dataset.init(preprocess=False)
    dataset.make_masks()
    net, loaded_keys = load_segmentation_net(model_name)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    predictions = segmentation_inference(
        dataset=dataset,
        model=net,
        device=torch.device("cuda:0"),
        crop_size=128,
        batch_size=batch_size,
        output_dir=str(output_path.parent),
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if not output_path.is_file():
        raise FileNotFoundError(output_path)
    record = {
        "status": "complete",
        "completed_at": common.now_iso(),
        "model": model_name,
        "split": split,
        "checkpoint": checkpoint_metadata("segmentation", model_name),
        "loaded_network_keys": loaded_keys,
        "dataset": common.dataset_record(dataset, "segmentation", split),
        "probability": validate_probability(output_path, split),
        "inference_batch_size": batch_size,
        "windows_worker_stack_bytes": WINDOWS_WORKER_STACK_BYTES if sys.platform.startswith("win") else None,
        "windows_stack_compatibility_only": sys.platform.startswith("win"),
        "elapsed_seconds": elapsed,
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    common.atomic_write_json(record_path, record)
    del predictions, dataset, net
    gc.collect()
    torch.cuda.empty_cache()
    return record


def initialized_segmentation_truth(split: str):
    dataset = common.make_dataset("segmentation", split)
    dataset.init(preprocess=False)
    dataset.make_masks()
    truth = [movie.copy() for movie in dataset.movies_masks]
    for movie in truth:
        movie[: dataset.n_input_channels - 1] = 0
    return dataset, truth


def probability_internal(path: Path):
    disk_tzyx = tifffile.memmap(path)
    return disk_tzyx, np.swapaxes(disk_tzyx, -1, -3)


def scalar_list(value) -> list[Any]:
    return [common.jsonable(item) for item in list(value)]


def compact_ccs_stats(stats) -> dict[str, Any]:
    if not isinstance(stats, dict):
        return {}
    return {
        "centroids": [
            np.asarray(item, dtype=np.float64).tolist()
            for item in list(stats["centroids"])
        ],
        "voxel_counts": [int(item) for item in list(stats["voxel_counts"])],
        "values": [int(item) for item in list(stats["values"])],
        "mean_prob": [float(item) for item in list(stats["mean_prob"])],
    }


def compact_matching_info(info) -> list[dict[str, Any]]:
    result = []
    for movie in info:
        result.append(
            {
                "matched_items": [
                    [int(pair[0]), int(pair[1])]
                    for pair in movie["matched_items"]
                ],
                "true_ccs_stats": compact_ccs_stats(movie["true_ccs_stats"]),
                "pred_ccs_stats": compact_ccs_stats(movie["pred_ccs_stats"]),
            }
        )
    return result


def restore_matching_info(records) -> list[dict[str, Any]]:
    restored = []
    for movie in records:
        restored.append(
            {
                "matched_items": [
                    (int(pair[0]), int(pair[1]))
                    for pair in movie["matched_items"]
                ],
                "true_ccs_stats": movie["true_ccs_stats"],
                "pred_ccs_stats": movie["pred_ccs_stats"],
            }
        )
    return restored


def evaluate_probability(
    dataset,
    truth,
    prediction,
    threshold: float,
    weighted_probability: float,
    output_dir: Path | None = None,
):
    return evaluate_segmentation(
        y_pred_movies=[prediction],
        y_true_movies=truth,
        movies_names=dataset.movie_names,
        multithread=False,
        threshold=float(threshold),
        iteration_method="movie",
        distance_mode="iou",
        distance_threshold=0.000001,
        min_weighted_prob=float(weighted_probability),
        output_dir=str(output_dir) if output_dir is not None else None,
    )


def threshold_grid(
    model_name: str,
    dataset,
    truth,
    prediction,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    best = None
    for threshold in GRID_PROBABILITIES:
        for weighted_probability in GRID_WEIGHTED_PROBABILITIES:
            stats, _ = evaluate_probability(
                dataset,
                truth,
                prediction,
                threshold,
                weighted_probability,
            )
            row = {
                "model": model_name,
                "split": "validation",
                "probability_threshold": threshold,
                "minimum_weighted_probability": weighted_probability,
                **{key: common.jsonable(value) for key, value in stats.items()},
            }
            rows.append(row)
            if best is None or float(row["fmeasure"]) > float(best["fmeasure"]):
                best = dict(row)
    if best is None:
        raise AssertionError("Threshold grid was empty")
    return best, rows


def score_segmentation(model_name: str) -> dict[str, Any]:
    if model_name != FIXED_SEGMENTATION_MODEL:
        raise ValueError(
            "Segmentation is fixed to the released original checkpoint"
        )
    root = SEGMENTATION_ROOT / model_name
    result_path = root / "segmentation_evaluation.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            print(f"Reusing completed segmentation scoring: {result_path}")
            return existing

    validation_inference = run_segmentation_inference(
        model_name, "validation", INFERENCE_BATCH_SIZE
    )
    test_inference = run_segmentation_inference(
        model_name, "test", INFERENCE_BATCH_SIZE
    )

    validation_data, validation_truth = initialized_segmentation_truth(
        "validation"
    )
    validation_disk, validation_prediction = probability_internal(
        probability_path(model_name, "validation")
    )
    started = time.perf_counter()
    best, grid_rows = threshold_grid(
        model_name,
        validation_data,
        validation_truth,
        validation_prediction,
    )
    write_csv(root / "validation_threshold_grid.csv", grid_rows)
    selected_threshold = float(best["probability_threshold"])
    selected_weight = float(best["minimum_weighted_probability"])
    validation_output = root / "validation_selected_matching"
    validation_stats, validation_info = evaluate_probability(
        validation_data,
        validation_truth,
        validation_prediction,
        selected_threshold,
        selected_weight,
        output_dir=validation_output,
    )
    del validation_disk, validation_prediction, validation_truth, validation_data
    gc.collect()

    test_data, test_truth = initialized_segmentation_truth("test")
    test_disk, test_prediction = probability_internal(
        probability_path(model_name, "test")
    )
    test_output = root / "test_selected_matching"
    test_stats, test_info = evaluate_probability(
        test_data,
        test_truth,
        test_prediction,
        selected_threshold,
        selected_weight,
        output_dir=test_output,
    )
    compact_test_info = compact_matching_info(test_info)
    common.atomic_write_json(root / "test_matching_info.json", compact_test_info)

    frozen = {}
    if model_name == "original":
        for split, dataset, truth, prediction in (
            ("test", test_data, test_truth, test_prediction),
        ):
            stats, _ = evaluate_probability(
                dataset,
                truth,
                prediction,
                0.55,
                0.15,
            )
            frozen[split] = {
                key: common.jsonable(value) for key, value in stats.items()
            }
        replay = json.loads(
            (
                common.PREFLIGHT_ROOT
                / "released_checkpoint_validation_replay"
                / "result.json"
            ).read_text(encoding="utf-8")
        )
        frozen["validation"] = replay["current_metrics"]

    elapsed = time.perf_counter() - started
    released = json.loads(common.RELEASED_STATS.read_text(encoding="utf-8"))[
        "segmentation_results"
    ]
    result = {
        "status": "complete",
        "completed_at": common.now_iso(),
        "role": "fixed_released_segmentation_for_both_regression_models",
        "segmentation_retrained": False,
        "model": model_name,
        "checkpoint": checkpoint_metadata("segmentation", model_name),
        "threshold_selection_split": "movie_I2 / manuscript set 2",
        "untouched_test_split": "movie_M / manuscript bold set 3",
        "selected_probability_threshold": selected_threshold,
        "selected_minimum_weighted_probability": selected_weight,
        "validation_best_grid_row": best,
        "validation_metrics": {
            key: common.jsonable(value) for key, value in validation_stats.items()
        },
        "test_metrics": {
            key: common.jsonable(value) for key, value in test_stats.items()
        },
        "test_matching_info": common.rel(root / "test_matching_info.json"),
        "threshold_grid": common.rel(root / "validation_threshold_grid.csv"),
        "validation_inference": validation_inference,
        "test_inference": test_inference,
        "original_checkpoint_frozen_threshold_metrics": frozen,
        "released_movie_I2_comparator": released,
        "scoring_elapsed_seconds": elapsed,
        "matching_rules": {
            "temporal_prediction_dilation": "plus/minus one frame",
            "distance_mode": "iou",
            "minimum_iou": 0.000001,
            "iteration_method": "movie",
            "production_object_filter_preserved": True,
        },
    }
    common.atomic_write_json(result_path, result)
    del test_disk, test_prediction, test_truth, test_data, test_info
    gc.collect()
    return result


def quaternion_axis_batch(
    rotation: torch.Tensor, project_prediction: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    matrix = (
        symmetric_orthogonalization(rotation)
        if project_prediction
        else rotation.reshape(-1, 3, 3)
    )
    quaternion = F.normalize(matrix_to_quaternion(matrix), dim=-1, p=2)
    vector = quaternion[..., 1:]
    vector_norm = torch.linalg.vector_norm(vector, dim=-1)
    axis = F.normalize(vector, dim=-1, p=2, eps=1e-8)
    return quaternion, axis, vector_norm


def production_errors_deg(predicted_q, true_q) -> torch.Tensor:
    dot = torch.sum(predicted_q * true_q, dim=-1)
    dot = torch.clamp(dot, -1.0 + 1e-4, 1.0 - 1e-4)
    return torch.rad2deg(2.0 * torch.acos(torch.abs(dot)))


def corrected_errors_deg(
    predicted_axis,
    true_axis,
    predicted_axis_norm,
    true_axis_norm,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = (
        torch.isfinite(predicted_axis_norm)
        & torch.isfinite(true_axis_norm)
        & (predicted_axis_norm > 1e-8)
        & (true_axis_norm > 1e-8)
    )
    dot = torch.sum(predicted_axis * true_axis, dim=-1)
    absolute_dot = torch.clamp(torch.abs(dot), 0.0, 1.0)
    errors = torch.rad2deg(torch.acos(absolute_dot))
    errors = torch.where(valid, errors, torch.full_like(errors, float("nan")))
    return errors, valid


def axis_from_quaternion(quaternion) -> tuple[np.ndarray | None, float]:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion_norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(quaternion_norm) or quaternion_norm <= 1e-12:
        return None, quaternion_norm
    quaternion = quaternion / quaternion_norm
    vector = quaternion[1:]
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-8:
        return None, norm
    return vector / norm, norm


def production_error_numpy(predicted_q, true_q) -> float:
    predicted_q = np.asarray(predicted_q, dtype=np.float64)
    true_q = np.asarray(true_q, dtype=np.float64)
    predicted_q /= np.linalg.norm(predicted_q)
    true_q /= np.linalg.norm(true_q)
    dot = float(np.dot(predicted_q, true_q))
    dot = float(np.clip(dot, -1.0 + 1e-4, 1.0 - 1e-4))
    return float(np.degrees(2.0 * np.arccos(abs(dot))))


def corrected_error_numpy(predicted_q, true_q) -> float:
    predicted_axis, _ = axis_from_quaternion(predicted_q)
    true_axis, _ = axis_from_quaternion(true_q)
    if predicted_axis is None or true_axis is None:
        return float("nan")
    absolute_dot = float(
        np.clip(abs(np.dot(predicted_axis, true_axis)), 0.0, 1.0)
    )
    return float(np.degrees(np.arccos(absolute_dot)))


def describe(values) -> dict[str, Any]:
    nominal = len(values)
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"nominal_n": nominal, "n": 0}
    std = float(np.std(values, dtype=np.float32))
    return {
        "nominal_n": nominal,
        "n": int(len(values)),
        "mean": float(np.mean(values, dtype=np.float32)),
        "median": float(np.median(values)),
        "std_population": std,
        "sem": std / math.sqrt(len(values)),
        "rms": float(np.sqrt(np.mean(np.square(values), dtype=np.float64))),
        "minimum": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(np.max(values)),
    }


def load_regression_net(model_name: str):
    model = common.make_regression_model(corrected_loss=True)
    loaded = load_net_state_dict(
        model.net,
        str(checkpoint_path("regression", model_name)),
        stage="regression",
    )
    net = model.net.to("cuda")
    net.eval()
    return net, len(loaded)


def controlled_rows_for_model(model_name: str, net, dataset):
    unique_count = len(dataset.crops)
    loader = DataLoader(
        Subset(dataset, range(unique_count)),
        batch_size=REGRESSION_BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
    )
    rows = []
    next_index = 0
    with torch.inference_mode():
        for x, y in loader:
            image = x["input"].to("cuda")
            true_rotation = y["head1"]["angle"].to("cuda")
            true_length = y["head1"]["len"].to("cuda").reshape(-1)
            output = net(image)["head1"]
            predicted_rotation = output["angle"]
            predicted_length = output["len"].reshape(-1)
            true_q, true_axis, true_axis_norm = quaternion_axis_batch(
                true_rotation, project_prediction=False
            )
            pred_q, pred_axis, pred_axis_norm = quaternion_axis_batch(
                predicted_rotation, project_prediction=True
            )
            production = production_errors_deg(pred_q, true_q)
            corrected, valid = corrected_errors_deg(
                pred_axis,
                true_axis,
                pred_axis_norm,
                true_axis_norm,
            )
            length_error = 32.0 * torch.abs(predicted_length - true_length)
            batch_count = image.shape[0]
            for local_index in range(batch_count):
                sample_index = next_index + local_index
                crop = dataset.crops[sample_index]
                row = {
                    "scope": "controlled_unique_movie_M_crops",
                    "model": model_name,
                    "sample_index": sample_index,
                    "movie": "movie_M",
                    "frame": int(crop[1].stop - 1),
                    "production_quaternion_error_deg": float(
                        production[local_index].cpu()
                    ),
                    "corrected_nematic_axis_error_deg": float(
                        corrected[local_index].cpu()
                    ),
                    "corrected_axis_defined": bool(valid[local_index].cpu()),
                    "true_length_voxels": float(
                        (32.0 * true_length[local_index]).cpu()
                    ),
                    "predicted_length_voxels": float(
                        (32.0 * predicted_length[local_index]).cpu()
                    ),
                    "absolute_length_error_voxels": float(
                        length_error[local_index].cpu()
                    ),
                    "true_quaternion_vector_norm": float(
                        true_axis_norm[local_index].cpu()
                    ),
                    "predicted_quaternion_vector_norm": float(
                        pred_axis_norm[local_index].cpu()
                    ),
                }
                for axis_name, axis in (
                    ("true_axis", true_axis[local_index]),
                    ("predicted_axis", pred_axis[local_index]),
                ):
                    for coordinate, value in zip(
                        "xyz", axis.detach().cpu().numpy()
                    ):
                        row[f"{axis_name}_{coordinate}"] = float(value)
                rows.append(row)
            next_index += batch_count
    if len(rows) != unique_count:
        raise AssertionError(f"Expected {unique_count} rows, got {len(rows)}")
    return rows

_controlled_rows_for_model_native = controlled_rows_for_model

def controlled_rows_for_model(model_name: str, net, dataset):
    """Evaluate unchanged controlled rows with a larger Windows worker stack."""
    return call_with_expanded_windows_stack(
        lambda: _controlled_rows_for_model_native(model_name, net, dataset)
    )


def summarize_controlled(rows) -> dict[str, Any]:
    by_model = {}
    for model_name in ("original", "retrained"):
        selected = [row for row in rows if row["model"] == model_name]
        by_model[model_name] = {
            "sample_count": len(selected),
            "production_quaternion_error_deg": describe(
                [row["production_quaternion_error_deg"] for row in selected]
            ),
            "corrected_nematic_axis_error_deg": describe(
                [row["corrected_nematic_axis_error_deg"] for row in selected]
            ),
            "absolute_length_error_voxels": describe(
                [row["absolute_length_error_voxels"] for row in selected]
            ),
            "undefined_corrected_axis_count": sum(
                not row["corrected_axis_defined"] for row in selected
            ),
        }
    old = sorted(
        (row for row in rows if row["model"] == "original"),
        key=lambda row: row["sample_index"],
    )
    new = sorted(
        (row for row in rows if row["model"] == "retrained"),
        key=lambda row: row["sample_index"],
    )
    if [row["sample_index"] for row in old] != [
        row["sample_index"] for row in new
    ]:
        raise AssertionError("Controlled event ordering changed")
    differences = np.asarray(
        [
            newer["corrected_nematic_axis_error_deg"]
            - older["corrected_nematic_axis_error_deg"]
            for older, newer in zip(old, new)
            if np.isfinite(older["corrected_nematic_axis_error_deg"])
            and np.isfinite(newer["corrected_nematic_axis_error_deg"])
        ],
        dtype=np.float64,
    )
    length_differences = np.asarray(
        [
            newer["absolute_length_error_voxels"]
            - older["absolute_length_error_voxels"]
            for older, newer in zip(old, new)
        ],
        dtype=np.float64,
    )
    return {
        "by_model": by_model,
        "paired_retrained_minus_original": {
            "corrected_nematic_axis_error_deg": describe(differences.tolist()),
            "absolute_length_error_voxels": describe(
                length_differences.tolist()
            ),
            "negative_difference_is_improvement": True,
        },
    }


def evaluate_controlled_regression() -> dict[str, Any]:
    result_path = CONTROLLED_ROOT / "controlled_regression_evaluation.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            return existing
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    CONTROLLED_ROOT.mkdir(parents=True, exist_ok=True)
    dataset = common.make_dataset("regression", "test")
    dataset.init()
    if len(dataset.crops) != 80:
        raise AssertionError(f"Expected 80 unique test crops, got {len(dataset.crops)}")
    rows = []
    loaded_keys = {}
    started = time.perf_counter()
    for model_name in ("original", "retrained"):
        net, loaded_keys[model_name] = load_regression_net(model_name)
        rows.extend(controlled_rows_for_model(model_name, net, dataset))
        del net
        gc.collect()
        torch.cuda.empty_cache()
    write_csv(CONTROLLED_ROOT / "event_metrics.csv", rows)
    summary = summarize_controlled(rows)
    result = {
        "status": "complete",
        "completed_at": common.now_iso(),
        "scope": "80 unique movie_M test crops after unchanged t>1 filter",
        "dataset": common.dataset_record(dataset, "regression", "test"),
        "checkpoints": {
            model: checkpoint_metadata("regression", model)
            for model in ("original", "retrained")
        },
        "loaded_network_keys": loaded_keys,
        "summary": summary,
        "event_metrics": common.rel(CONTROLLED_ROOT / "event_metrics.csv"),
        "elapsed_seconds": time.perf_counter() - started,
    }
    common.atomic_write_json(result_path, result)
    return result


def pair_rows(pipeline_name: str, mode: str, pairs):
    rows = []
    for pair_index, (true, predicted) in enumerate(pairs):
        evaluated = true is not None and predicted is not None
        row = {
            "scope": "production_end_to_end_movie_M",
            "pipeline": pipeline_name,
            "mode": mode,
            "pair_index": pair_index,
            "reported_pair_count": len(pairs),
            "evaluated": evaluated,
        }
        if not evaluated:
            rows.append(row)
            continue
        true_q = np.asarray(true["rotation"], dtype=np.float64)
        predicted_q = np.asarray(predicted["rotation"], dtype=np.float64)
        true_axis, true_axis_norm = axis_from_quaternion(true_q)
        predicted_axis, predicted_axis_norm = axis_from_quaternion(predicted_q)
        true_length = float(np.asarray(true["length"]).reshape(-1)[0])
        predicted_length = float(np.asarray(predicted["length"]).reshape(-1)[0])
        true_center = np.asarray(true["center"], dtype=np.float64)
        predicted_center = np.asarray(predicted["center"], dtype=np.float64)
        row.update(
            {
                "production_quaternion_error_deg": production_error_numpy(
                    predicted_q, true_q
                ),
                "corrected_nematic_axis_error_deg": corrected_error_numpy(
                    predicted_q, true_q
                ),
                "corrected_axis_defined": (
                    true_axis is not None and predicted_axis is not None
                ),
                "true_length_voxels": true_length,
                "predicted_length_voxels": predicted_length,
                "absolute_length_error_voxels": abs(
                    true_length - predicted_length
                ),
                "center_distance_voxels": float(
                    np.linalg.norm(true_center - predicted_center)
                ),
                "time_distance_frames": float(
                    abs(true_center[1] - predicted_center[1])
                ),
                "true_quaternion_vector_norm": true_axis_norm,
                "predicted_quaternion_vector_norm": predicted_axis_norm,
            }
        )
        rows.append(row)
    return rows


def pairs_for_predictions(center_list: CenterList, predictions):
    return {
        "all_groundtruth_centers": center_list.create_gt_pairs(
            predictions["all_groundtruth_centers"]
        ),
        "matched_groundtruth_centers": center_list.create_matched_gt_pairs(
            predictions["matched_groundtruth_centers"]
        ),
        "predicted_centers": center_list.create_pred_gt_pairs(
            predictions["predicted_centers"]
        ),
    }


def summarize_end_to_end(rows) -> dict[str, Any]:
    result = {}
    for pipeline in PIPELINE_ORDER:
        result[pipeline] = {}
        for mode in MODE_ORDER:
            candidates = [
                row
                for row in rows
                if row["pipeline"] == pipeline and row["mode"] == mode
            ]
            evaluated = [row for row in candidates if row["evaluated"]]
            result[pipeline][mode] = {
                "reported_pair_count": (
                    candidates[0]["reported_pair_count"] if candidates else 0
                ),
                "effective_pair_count": len(evaluated),
                "production_quaternion_error_deg": describe(
                    [
                        row["production_quaternion_error_deg"]
                        for row in evaluated
                    ]
                ),
                "corrected_nematic_axis_error_deg": describe(
                    [
                        row["corrected_nematic_axis_error_deg"]
                        for row in evaluated
                    ]
                ),
                "absolute_length_error_voxels": describe(
                    [
                        row["absolute_length_error_voxels"]
                        for row in evaluated
                    ]
                ),
                "center_distance_voxels": describe(
                    [row["center_distance_voxels"] for row in evaluated]
                ),
                "undefined_corrected_axis_count": sum(
                    not row["corrected_axis_defined"] for row in evaluated
                ),
            }
    return result


def evaluate_end_to_end() -> dict[str, Any]:
    result_path = END_TO_END_ROOT / "end_to_end_evaluation.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "not_evaluable":
            return existing
        if existing.get("status") == "complete":
            return existing
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    END_TO_END_ROOT.mkdir(parents=True, exist_ok=True)
    segmentation_result = score_segmentation(FIXED_SEGMENTATION_MODEL)
    match_path = (
        SEGMENTATION_ROOT
        / FIXED_SEGMENTATION_MODEL
        / "test_matching_info.json"
    )
    matching_records = json.loads(match_path.read_text(encoding="utf-8"))
    test_metrics = segmentation_result["test_metrics"]
    if (
        int(test_metrics["tp"]) == 0
        and int(test_metrics["fp"]) == 0
    ):
        result = {
            "status": "not_evaluable",
            "completed_at": common.now_iso(),
            "scope": "fixed released segmentation on independent movie_M test",
            "reason_code": "fixed_segmentation_produced_zero_detections",
            "reason": (
                "The released segmentation checkpoint produced no candidate "
                "components at the validation-selected unchanged threshold grid; "
                "there are therefore no predicted centers for angular scoring."
            ),
            "segmentation_retrained": False,
            "fixed_segmentation_result": segmentation_result,
            "fixed_test_matching_info": common.rel(match_path),
            "detected_center_count": 0,
            "matched_center_count": 0,
            "annotated_test_event_count": int(test_metrics["fn"]),
            "controlled_evaluable_event_count": 80,
            "validation_probability_maximum": segmentation_result["validation_inference"]["probability"]["maximum"],
            "test_probability_maximum": segmentation_result["test_inference"]["probability"]["maximum"],
            "checkpoints": {
                "fixed_segmentation": segmentation_result["checkpoint"],
                "regression": {
                    model: checkpoint_metadata("regression", model)
                    for model in ("original", "retrained")
                },
            },
            "summary": {},
            "event_metrics": None,
            "production_compatibility_note": (
                "No threshold was lowered and no matching rule was changed to "
                "force an end-to-end comparison. The controlled 80-crop test "
                "remains evaluable at ground-truth division locations."
            ),
        }
        common.atomic_write_json(result_path, result)
        return result

    dataset = common.make_dataset("regression", "test")
    dataset.init(preprocess=False)
    dataset.pad_images()
    dataset._normalize(dataset.renorm)

    rows = []
    center_counts = {}
    loaded_keys = {}
    started = time.perf_counter()
    for model_name, pipeline_name in (
        ("original", PIPELINE_ORDER[0]),
        ("retrained", PIPELINE_ORDER[1]),
    ):
        info = restore_matching_info(matching_records)
        center_list = CenterList(0, info)
        center_counts[pipeline_name] = {
            "all_groundtruth_centers": len(center_list.all_gt_centers),
            "matched_groundtruth_centers": len(center_list.true_centers),
            "predicted_centers": len(center_list.predicted_centers),
        }
        net, loaded_keys[model_name] = load_regression_net(model_name)
        predictions = {}
        centers_by_mode = {
            "all_groundtruth_centers": center_list.all_gt_centers,
            "matched_groundtruth_centers": center_list.true_centers,
            "predicted_centers": center_list.predicted_centers,
        }
        for mode, centers in centers_by_mode.items():
            predictions[mode] = regression_inference(
                dataset,
                net,
                centers,
                torch.device("cuda:0"),
                output_dir=str(END_TO_END_ROOT / pipeline_name / mode),
            )
        center_list.compute_real_rot_len_values(dataset)
        for mode, pairs in pairs_for_predictions(
            center_list, predictions
        ).items():
            rows.extend(pair_rows(pipeline_name, mode, pairs))
        del net, predictions, center_list
        gc.collect()
        torch.cuda.empty_cache()

    centers_are_identical = (
        center_counts[PIPELINE_ORDER[0]] == center_counts[PIPELINE_ORDER[1]]
    )
    if not centers_are_identical:
        raise AssertionError("Fixed segmentation produced different center sets")
    write_csv(END_TO_END_ROOT / "event_metrics.csv", rows)
    result = {
        "status": "complete",
        "completed_at": common.now_iso(),
        "scope": (
            "fixed released segmentation and unchanged production matching; "
            "original versus nematic regression on movie_M"
        ),
        "segmentation_retrained": False,
        "fixed_segmentation_result": segmentation_result,
        "fixed_test_matching_info": common.rel(match_path),
        "checkpoints": {
            "fixed_segmentation": segmentation_result["checkpoint"],
            "regression": {
                model: checkpoint_metadata("regression", model)
                for model in ("original", "retrained")
            },
        },
        "loaded_regression_network_keys": loaded_keys,
        "center_counts": center_counts,
        "center_counts_identical_between_regression_models": centers_are_identical,
        "summary": summarize_end_to_end(rows),
        "event_metrics": common.rel(END_TO_END_ROOT / "event_metrics.csv"),
        "elapsed_seconds": time.perf_counter() - started,
        "production_compatibility_note": (
            "The production end-to-end evaluator initializes regression data "
            "with preprocess=False, pads images, and normalizes without the "
            "training-time isotropic rescale; this inherited behavior is "
            "preserved and reported separately from the controlled 80-crop test."
        ),
    }
    common.atomic_write_json(result_path, result)
    return result


def dispatch(args) -> dict[str, Any]:
    if args.command == "segmentation-infer":
        return run_segmentation_inference(
            args.model, args.split, args.batch_size
        )
    if args.command == "segmentation-score":
        return score_segmentation(args.model)
    if args.command == "controlled-regression":
        return evaluate_controlled_regression()
    if args.command == "end-to-end":
        return evaluate_end_to_end()
    if args.command == "all":
        result = {
            "segmentation": {"fixed_released": score_segmentation(
                FIXED_SEGMENTATION_MODEL
            )},
            "controlled_regression": evaluate_controlled_regression(),
            "end_to_end": evaluate_end_to_end(),
        }
        result["status"] = "complete"
        if result["end_to_end"].get("status") != "complete":
            result["status"] = "complete_with_unavailable_end_to_end"
        common.atomic_write_json(EVALUATION_ROOT / "all_results.json", result)
        return result
    raise ValueError(args.command)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    infer_parser = subparsers.add_parser("segmentation-infer")
    infer_parser.add_argument("--model", choices=(FIXED_SEGMENTATION_MODEL,), required=True)
    infer_parser.add_argument("--split", choices=("validation", "test"), required=True)
    infer_parser.add_argument("--batch-size", type=int, default=INFERENCE_BATCH_SIZE)
    score_parser = subparsers.add_parser("segmentation-score")
    score_parser.add_argument("--model", choices=(FIXED_SEGMENTATION_MODEL,), required=True)
    subparsers.add_parser("controlled-regression")
    subparsers.add_parser("end-to-end")
    subparsers.add_parser("all")
    args = parser.parse_args()

    EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)
    with EVALUATION_LOG.open("a", encoding="utf-8", buffering=1) as log_stream:
        tee_out = Tee(sys.stdout, log_stream)
        tee_err = Tee(sys.stderr, log_stream)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(
            tee_err
        ):
            print(f"\n[{common.now_iso()}] command={args.command}", flush=True)
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
                    "traceback": traceback.format_exc(),
                }
                common.atomic_write_json(
                    EVALUATION_ROOT / f"{args.command}_failure.json",
                    failure,
                )
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
