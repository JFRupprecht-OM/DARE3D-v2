"""Evaluate the isolated nematic-loss retraining against the released model.

The script compares both checkpoints on (1) the 156 unique movie2 holdout
regression crops and (2) all three frozen manuscript center modes.  It writes
only beneath the reproducibility-audit experiment directory.
"""
from __future__ import annotations

import contextlib
import csv
import gc
import hashlib
import io
import json
import math
import re
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
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import nematic_retraining_experiment as experiment
import regression_checkpoint_compatibility as compatibility
from dare3d.data.components.angles3d import symmetric_orthogonalization
from dare3d.losses.angle3d import matrix_to_quaternion
from dare3d.metrics.infer_measure import CenterList
from dare3d.metrics.inference import regression_inference
from dare3d.models.components.simple_regression_net import RegressionNet
from dare3d.models.finetune import load_net_state_dict

RUN_ROOT = experiment.RUN_ROOT
STATE_PATH = experiment.STATE_OUT
ORIGINAL_CHECKPOINT = experiment.ORIGINAL_CHECKPOINT
EVALUATION_LOG = RUN_ROOT / "evaluation.log"
EVALUATION_JSON = RUN_ROOT / "evaluation.json"
SUMMARY_CSV = RUN_ROOT / "evaluation_summary.csv"
CONTROLLED_EVENTS_CSV = RUN_ROOT / "controlled_test_event_metrics.csv"
MANUSCRIPT_EVENTS_CSV = RUN_ROOT / "manuscript_event_metrics.csv"
DISTRIBUTION_CSV = RUN_ROOT / "angular_distribution.csv"
PREDICTION_ROOT = RUN_ROOT / "predictions"
PRIOR_NEMATIC = HERE / "evidence/nematic_axis_metric_comparison.json"
PROVENANCE = HERE / "evidence/provenance.json"
BOOTSTRAP_REPLICATES = 10000
MODE_ORDER = (
    "all_groundtruth_centers",
    "matched_groundtruth_centers",
    "predicted_centers",
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


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


def make_net(checkpoint: Path) -> tuple[RegressionNet, int]:
    net = RegressionNet(
        input_channels=[-1, 0, 1],
        im_size=32,
        angle_vec_size=9,
        start_filters=16,
        n_stages=5,
    )
    loaded = load_net_state_dict(net, str(checkpoint), stage="regression")
    net = net.to("cuda")
    net.eval()
    return net, len(loaded)


def quaternion_axis_batch(
    rotation: torch.Tensor, project_prediction: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    matrix = (
        symmetric_orthogonalization(rotation)
        if project_prediction
        else rotation.reshape(-1, 3, 3)
    )
    quaternion = matrix_to_quaternion(matrix)
    quaternion = F.normalize(quaternion, dim=-1, p=2)
    vector = quaternion[..., 1:]
    vector_norm = torch.linalg.vector_norm(vector, dim=-1)
    axis = F.normalize(vector, dim=-1, p=2, eps=1e-8)
    return quaternion, axis, vector_norm


def production_quaternion_errors_deg(
    predicted_quaternion: torch.Tensor, true_quaternion: torch.Tensor
) -> torch.Tensor:
    dot = torch.sum(predicted_quaternion * true_quaternion, dim=-1)
    dot = torch.clamp(dot, -1.0 + 1e-4, 1.0 - 1e-4)
    return torch.rad2deg(2.0 * torch.acos(torch.abs(dot)))


def corrected_axis_errors_deg(
    predicted_axis: torch.Tensor, true_axis: torch.Tensor
) -> torch.Tensor:
    dot = torch.sum(predicted_axis * true_axis, dim=-1)
    absolute_dot = torch.clamp(torch.abs(dot), 0.0, 1.0)
    return torch.rad2deg(torch.acos(absolute_dot))


def axis_from_quaternion(quaternion) -> tuple[np.ndarray | None, float]:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    vector = quaternion[1:]
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        return None, norm
    return vector / norm, norm


def production_error_numpy(q_pred, q_true) -> float:
    q_pred = np.asarray(q_pred, dtype=np.float64)
    q_true = np.asarray(q_true, dtype=np.float64)
    q_pred /= np.linalg.norm(q_pred)
    q_true /= np.linalg.norm(q_true)
    dot = float(np.dot(q_pred, q_true))
    dot = float(np.clip(dot, -1.0 + 1e-4, 1.0 - 1e-4))
    return float(np.degrees(2.0 * np.arccos(abs(dot))))


def corrected_error_numpy(q_pred, q_true) -> float:
    predicted_axis, _ = axis_from_quaternion(q_pred)
    true_axis, _ = axis_from_quaternion(q_true)
    if predicted_axis is None or true_axis is None:
        return float("nan")
    dot = float(np.dot(predicted_axis, true_axis))
    return float(np.degrees(np.arccos(np.clip(abs(dot), 0.0, 1.0))))


def describe(values) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    std = float(np.std(values, dtype=np.float32))
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values, dtype=np.float32)),
        "median": float(np.median(values)),
        "std_population": std,
        "sem": std / math.sqrt(len(values)),
        "rms": float(np.sqrt(np.mean(np.square(values), dtype=np.float64))),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def paired_effect(
    old_values, new_values, seed_offset: int, unit: str
) -> dict[str, Any]:
    old = np.asarray(old_values, dtype=np.float64)
    new = np.asarray(new_values, dtype=np.float64)
    valid = np.isfinite(old) & np.isfinite(new)
    old = old[valid]
    new = new[valid]
    difference = new - old
    rng = np.random.default_rng(experiment.BOOTSTRAP_SEED + seed_offset)
    indices = rng.integers(
        0, len(difference), size=(BOOTSTRAP_REPLICATES, len(difference))
    )
    resampled = difference[indices]
    mean_distribution = np.mean(resampled, axis=1)
    median_distribution = np.median(resampled, axis=1)
    return {
        "n": int(len(difference)),
        "difference_definition": "retrained minus original; negative is improvement",
        "unit": unit,
        "mean_difference": float(np.mean(difference)),
        "median_difference": float(np.median(difference)),
        "mean_difference_bootstrap_95_ci": [
            float(np.percentile(mean_distribution, 2.5)),
            float(np.percentile(mean_distribution, 97.5)),
        ],
        "median_difference_bootstrap_95_ci": [
            float(np.percentile(median_distribution, 2.5)),
            float(np.percentile(median_distribution, 97.5)),
        ],
        "fraction_retrained_lower": float(np.mean(difference < 0)),
        "fraction_equal": float(np.mean(difference == 0)),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": experiment.BOOTSTRAP_SEED + seed_offset,
        "inference_note": (
            "Descriptive paired bootstrap only: movie2 was also used for "
            "checkpoint selection, so this is not an untouched-test inference."
        ),
    }


def evaluate_controlled_model(
    model_name: str,
    net: RegressionNet,
    dataset,
) -> list[dict[str, Any]]:
    unique_count = len(dataset.crops)
    loader = DataLoader(
        Subset(dataset, range(unique_count)),
        batch_size=12,
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
            current = production_quaternion_errors_deg(pred_q, true_q)
            corrected = corrected_axis_errors_deg(pred_axis, true_axis)
            length_error = 32.0 * torch.abs(predicted_length - true_length)
            true_length_voxels = 32.0 * true_length
            predicted_length_voxels = 32.0 * predicted_length

            batch_size = image.shape[0]
            for batch_index in range(batch_size):
                row = {
                    "scope": "controlled_unique_holdout",
                    "model": model_name,
                    "sample_index": next_index + batch_index,
                    "production_quaternion_error_deg": float(
                        current[batch_index].cpu()
                    ),
                    "corrected_nematic_axis_error_deg": float(
                        corrected[batch_index].cpu()
                    ),
                    "true_length_voxels": float(
                        true_length_voxels[batch_index].cpu()
                    ),
                    "predicted_length_voxels": float(
                        predicted_length_voxels[batch_index].cpu()
                    ),
                    "absolute_length_error_voxels": float(
                        length_error[batch_index].cpu()
                    ),
                    "true_quaternion_vector_norm": float(
                        true_axis_norm[batch_index].cpu()
                    ),
                    "predicted_quaternion_vector_norm": float(
                        pred_axis_norm[batch_index].cpu()
                    ),
                }
                for axis_name, axis in (
                    ("true_axis", true_axis[batch_index]),
                    ("predicted_axis", pred_axis[batch_index]),
                ):
                    values = axis.detach().cpu().numpy()
                    for coordinate, value in zip("xyz", values):
                        row[f"{axis_name}_{coordinate}"] = float(value)
                rows.append(row)
            next_index += batch_size
    if len(rows) != unique_count:
        raise AssertionError(f"Expected {unique_count} rows, got {len(rows)}")
    return rows


def summarize_controlled(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model = {}
    for model_name in ("original_epoch_098", "retrained_nematic"):
        selected = [row for row in rows if row["model"] == model_name]
        by_model[model_name] = {
            "n": len(selected),
            "production_quaternion_error_deg": describe(
                [row["production_quaternion_error_deg"] for row in selected]
            ),
            "corrected_nematic_axis_error_deg": describe(
                [row["corrected_nematic_axis_error_deg"] for row in selected]
            ),
            "absolute_length_error_voxels": describe(
                [row["absolute_length_error_voxels"] for row in selected]
            ),
            "predicted_quaternion_vector_norm": describe(
                [row["predicted_quaternion_vector_norm"] for row in selected]
            ),
        }
    old = [
        row
        for row in rows
        if row["model"] == "original_epoch_098"
    ]
    new = [
        row
        for row in rows
        if row["model"] == "retrained_nematic"
    ]
    old.sort(key=lambda row: row["sample_index"])
    new.sort(key=lambda row: row["sample_index"])
    if [row["sample_index"] for row in old] != [
        row["sample_index"] for row in new
    ]:
        raise AssertionError("Controlled event ordering changed")
    return {
        "by_model": by_model,
        "paired_corrected_axis_effect": paired_effect(
            [row["corrected_nematic_axis_error_deg"] for row in old],
            [row["corrected_nematic_axis_error_deg"] for row in new],
            seed_offset=0,
            unit="degrees",
        ),
        "paired_length_effect": paired_effect(
            [row["absolute_length_error_voxels"] for row in old],
            [row["absolute_length_error_voxels"] for row in new],
            seed_offset=1,
            unit="voxels",
        ),
    }


def save_prediction_npz(
    path: Path, predictions: list[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        centers=np.asarray([item["center"] for item in predictions]),
        lengths=np.asarray([item["length"] for item in predictions]),
        quaternions=np.asarray([item["rotation"] for item in predictions]),
    )


def load_raw_manuscript_dataset():
    cfg = compatibility.frozen_config(ORIGINAL_CHECKPOINT)
    dataset = hydra.utils.instantiate(cfg.data.test_data)
    dataset.init(preprocess=False)
    dataset.pad_images()
    dataset._normalize(dataset.renorm)
    return dataset


def manuscript_pairs(center_list: CenterList, predictions):
    return {
        "all_groundtruth_centers": center_list.create_gt_pairs(
            predictions["all_groundtruth_centers"]
        ),
        "matched_groundtruth_centers": (
            center_list.create_matched_gt_pairs(
                predictions["matched_groundtruth_centers"]
            )
        ),
        "predicted_centers": center_list.create_pred_gt_pairs(
            predictions["predicted_centers"]
        ),
    }


def evaluate_pair_rows(
    model_name: str, mode: str, pairs
) -> list[dict[str, Any]]:
    rows = []
    for pair_index, (true, predicted) in enumerate(pairs):
        row = {
            "scope": "manuscript_frozen_centers",
            "model": model_name,
            "mode": mode,
            "pair_index": pair_index,
            "reported_pair_count": len(pairs),
            "evaluated": true is not None and predicted is not None,
        }
        if true is None or predicted is None:
            rows.append(row)
            continue
        true_q = np.asarray(true["rotation"], dtype=np.float64)
        predicted_q = np.asarray(predicted["rotation"], dtype=np.float64)
        true_axis, true_axis_norm = axis_from_quaternion(true_q)
        predicted_axis, predicted_axis_norm = axis_from_quaternion(predicted_q)
        true_length = float(np.asarray(true["length"]).reshape(-1)[0])
        predicted_length = float(
            np.asarray(predicted["length"]).reshape(-1)[0]
        )
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
        if true_axis is not None:
            for coordinate, value in zip("xyz", true_axis):
                row[f"true_axis_{coordinate}"] = float(value)
        if predicted_axis is not None:
            for coordinate, value in zip("xyz", predicted_axis):
                row[f"predicted_axis_{coordinate}"] = float(value)
        rows.append(row)
    return rows


def summarize_manuscript(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for model_name in ("original_epoch_098", "retrained_nematic"):
        result[model_name] = {}
        for mode in MODE_ORDER:
            selected = [
                row
                for row in rows
                if row["model"] == model_name
                and row["mode"] == mode
                and row["evaluated"]
            ]
            nominal = max(
                row["reported_pair_count"]
                for row in rows
                if row["model"] == model_name and row["mode"] == mode
            )
            result[model_name][mode] = {
                "reported_n": int(nominal),
                "effective_n": len(selected),
                "production_quaternion_error_deg": describe(
                    [
                        row["production_quaternion_error_deg"]
                        for row in selected
                    ]
                ),
                "corrected_nematic_axis_error_deg": describe(
                    [
                        row["corrected_nematic_axis_error_deg"]
                        for row in selected
                    ]
                ),
                "absolute_length_error_voxels": describe(
                    [
                        row["absolute_length_error_voxels"]
                        for row in selected
                    ]
                ),
                "center_distance_voxels": describe(
                    [row["center_distance_voxels"] for row in selected]
                ),
                "predicted_quaternion_vector_norm": describe(
                    [
                        row["predicted_quaternion_vector_norm"]
                        for row in selected
                    ]
                ),
            }

    paired = {}
    for mode_index, mode in enumerate(MODE_ORDER):
        old = [
            row
            for row in rows
            if row["model"] == "original_epoch_098"
            and row["mode"] == mode
            and row["evaluated"]
        ]
        new = [
            row
            for row in rows
            if row["model"] == "retrained_nematic"
            and row["mode"] == mode
            and row["evaluated"]
        ]
        old.sort(key=lambda row: row["pair_index"])
        new.sort(key=lambda row: row["pair_index"])
        if [row["pair_index"] for row in old] != [
            row["pair_index"] for row in new
        ]:
            raise AssertionError(f"Pair order changed for {mode}")
        paired[mode] = {
            "corrected_axis": paired_effect(
                [
                    row["corrected_nematic_axis_error_deg"]
                    for row in old
                ],
                [
                    row["corrected_nematic_axis_error_deg"]
                    for row in new
                ],
                seed_offset=10 + mode_index,
                unit="degrees",
            ),
            "length": paired_effect(
                [row["absolute_length_error_voxels"] for row in old],
                [row["absolute_length_error_voxels"] for row in new],
                seed_offset=20 + mode_index,
                unit="voxels",
            ),
        }
    return {"by_model": result, "paired_effects": paired}


def build_summary_rows(controlled, manuscript):
    rows = []
    scenarios = (
        (
            "1_original_model_original_metric",
            "original_epoch_098",
            "production_quaternion_error_deg",
        ),
        (
            "2_original_model_corrected_nematic_metric",
            "original_epoch_098",
            "corrected_nematic_axis_error_deg",
        ),
        (
            "3_retrained_model_corrected_nematic_metric",
            "retrained_nematic",
            "corrected_nematic_axis_error_deg",
        ),
    )
    for scenario, model, metric in scenarios:
        angular = controlled["by_model"][model][metric]
        length = controlled["by_model"][model]["absolute_length_error_voxels"]
        rows.append(
            {
                "scenario": scenario,
                "scope": "controlled_unique_holdout",
                "mode": "156_unique_groundtruth_crops",
                "reported_n": 156,
                "effective_n": angular["n"],
                "angular_metric": metric,
                **{f"angle_{key}": value for key, value in angular.items()},
                "length_mean_absolute_error_voxels": length["mean"],
                "length_std_population_voxels": length["std_population"],
            }
        )
        for mode in MODE_ORDER:
            details = manuscript["by_model"][model][mode]
            angular = details[metric]
            length = details["absolute_length_error_voxels"]
            rows.append(
                {
                    "scenario": scenario,
                    "scope": "manuscript_frozen_centers",
                    "mode": mode,
                    "reported_n": details["reported_n"],
                    "effective_n": details["effective_n"],
                    "angular_metric": metric,
                    **{
                        f"angle_{key}": value
                        for key, value in angular.items()
                    },
                    "length_mean_absolute_error_voxels": length["mean"],
                    "length_std_population_voxels": (
                        length["std_population"]
                    ),
                }
            )
    return rows


def build_distribution_rows(summary_rows, controlled_rows, manuscript_rows):
    values_by_key = {}
    for summary in summary_rows:
        scenario = summary["scenario"]
        scope = summary["scope"]
        mode = summary["mode"]
        model = (
            "retrained_nematic"
            if scenario.startswith("3_")
            else "original_epoch_098"
        )
        metric = summary["angular_metric"]
        source = (
            controlled_rows
            if scope == "controlled_unique_holdout"
            else manuscript_rows
        )
        selected = [
            row
            for row in source
            if row["model"] == model
            and (scope == "controlled_unique_holdout" or row["mode"] == mode)
            and row.get("evaluated", True)
        ]
        values_by_key[(scenario, scope, mode)] = np.asarray(
            [row[metric] for row in selected], dtype=np.float64
        )

    edges = np.asarray([0, 5, 10, 15, 20, 30, 45, 60, 75, 90.000001])
    thresholds = [5, 10, 15, 20, 30, 45, 60, 90]
    rows = []
    for (scenario, scope, mode), values in values_by_key.items():
        values = values[np.isfinite(values)]
        counts, _ = np.histogram(values, bins=edges)
        for index, count in enumerate(counts):
            rows.append(
                {
                    "scenario": scenario,
                    "scope": scope,
                    "mode": mode,
                    "record_type": "histogram_bin",
                    "lower_deg_inclusive": edges[index],
                    "upper_deg_exclusive": edges[index + 1],
                    "threshold_deg": "",
                    "count": int(count),
                    "fraction": float(count / len(values)),
                    "n": len(values),
                }
            )
        for threshold in thresholds:
            count = int(np.sum(values <= threshold))
            rows.append(
                {
                    "scenario": scenario,
                    "scope": scope,
                    "mode": mode,
                    "record_type": "cdf_threshold",
                    "lower_deg_inclusive": "",
                    "upper_deg_exclusive": "",
                    "threshold_deg": threshold,
                    "count": count,
                    "fraction": float(count / len(values)),
                    "n": len(values),
                }
            )
    return rows


def released_current_stats() -> dict[str, Any]:
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    return provenance["released_stats"]["nuclei_gastruloid"][
        "regression_results"
    ]


def prior_corrected_rows() -> dict[str, dict[str, Any]]:
    prior = json.loads(PRIOR_NEMATIC.read_text(encoding="utf-8"))
    return {
        row["mode"]: row
        for row in prior["summary_rows"]
        if row["dataset"] == "gastruloid_nuclei"
    }


def verify_released_agreement(manuscript) -> dict[str, Any]:
    released = released_current_stats()
    prior = prior_corrected_rows()
    current_differences = []
    corrected_differences = []
    for mode in MODE_ORDER:
        released_key = compatibility.MODE_TO_RELEASED[mode]
        generated = manuscript["by_model"]["original_epoch_098"][mode]
        current_differences.extend(
            [
                abs(
                    generated["production_quaternion_error_deg"]["mean"]
                    - float(released[released_key]["mean_angle_error"])
                ),
                abs(
                    generated["production_quaternion_error_deg"][
                        "std_population"
                    ]
                    - float(released[released_key]["std_angle_error"])
                ),
            ]
        )
        corrected_differences.extend(
            [
                abs(
                    generated["corrected_nematic_axis_error_deg"]["mean"]
                    - float(prior[mode]["corrected_mean_deg"])
                ),
                abs(
                    generated["corrected_nematic_axis_error_deg"][
                        "std_population"
                    ]
                    - float(prior[mode]["corrected_std_population_deg"])
                ),
            ]
        )
    return {
        "maximum_original_current_vs_released_aggregate_difference_deg": max(
            current_differences
        ),
        "maximum_original_corrected_vs_prior_audit_difference_deg": max(
            corrected_differences
        ),
        "original_current_within_0_02_deg": max(current_differences) <= 0.02,
        "original_corrected_within_1e_minus_5_deg": (
            max(corrected_differences) <= 1e-5
        ),
    }


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "path": rel(path),
        "sha256": sha256(path),
        "epoch": int(checkpoint.get("epoch", -1)),
        "global_step": int(checkpoint.get("global_step", -1)),
        "lightning_version": checkpoint.get("pytorch-lightning_version"),
    }


def run_evaluation() -> dict[str, Any]:
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if state.get("status") != "trained":
        raise RuntimeError(
            f"Training is not complete according to {STATE_PATH}: "
            f"{state.get('status')}"
        )
    retrained_checkpoint = REPO / state["best_model_path"]
    if not retrained_checkpoint.is_file():
        raise FileNotFoundError(retrained_checkpoint)
    if git_head() != experiment.SOURCE_COMMIT:
        raise RuntimeError("Source commit changed before evaluation")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    original_net, original_keys = make_net(ORIGINAL_CHECKPOINT)
    retrained_net, retrained_keys = make_net(retrained_checkpoint)
    models = {
        "original_epoch_098": original_net,
        "retrained_nematic": retrained_net,
    }

    print("Initializing controlled 156-crop movie2 holdout")
    controlled_dataset = experiment.make_dataset("test")
    controlled_dataset.init()
    if len(controlled_dataset.crops) != 156:
        raise AssertionError(len(controlled_dataset.crops))
    controlled_rows = []
    for model_name, net in models.items():
        print(f"Controlled evaluation: {model_name}")
        controlled_rows.extend(
            evaluate_controlled_model(model_name, net, controlled_dataset)
        )
    controlled = summarize_controlled(controlled_rows)
    write_csv(CONTROLLED_EVENTS_CSV, controlled_rows)
    del controlled_dataset
    gc.collect()

    print("Initializing frozen raw-coordinate manuscript evaluation")
    raw_dataset = load_raw_manuscript_dataset()
    info, center_summary = compatibility.load_frozen_info()
    center_list = CenterList(0, info)
    predictions_by_model = {}
    PREDICTION_ROOT.mkdir(parents=True, exist_ok=True)
    for model_name, net in models.items():
        predictions = {}
        for mode in MODE_ORDER:
            centers = getattr(
                center_list, compatibility.MODE_TO_CENTERS[mode]
            )
            print(f"Manuscript evaluation: {model_name}, {mode}, N={len(centers)}")
            predictions[mode] = regression_inference(
                raw_dataset, net, centers, "cuda", output_dir=None
            )
            save_prediction_npz(
                PREDICTION_ROOT / f"{model_name}_{mode}.npz",
                predictions[mode],
            )
        predictions_by_model[model_name] = predictions

    center_list.compute_real_rot_len_values(raw_dataset)
    manuscript_rows = []
    for model_name, predictions in predictions_by_model.items():
        pairs_by_mode = manuscript_pairs(center_list, predictions)
        for mode, pairs in pairs_by_mode.items():
            manuscript_rows.extend(
                evaluate_pair_rows(model_name, mode, pairs)
            )
    manuscript = summarize_manuscript(manuscript_rows)
    write_csv(MANUSCRIPT_EVENTS_CSV, manuscript_rows)

    summary_rows = build_summary_rows(controlled, manuscript)
    distribution_rows = build_distribution_rows(
        summary_rows, controlled_rows, manuscript_rows
    )
    write_csv(SUMMARY_CSV, summary_rows)
    write_csv(DISTRIBUTION_CSV, distribution_rows)

    released_agreement = verify_released_agreement(manuscript)
    all_corrected = [
        row["angle_mean"]
        for row in summary_rows
        if row["angular_metric"] == "corrected_nematic_axis_error_deg"
    ]
    all_event_corrected = [
        row["corrected_nematic_axis_error_deg"]
        for row in controlled_rows
    ] + [
        row["corrected_nematic_axis_error_deg"]
        for row in manuscript_rows
        if row["evaluated"]
    ]
    old_distances = [
        row["center_distance_voxels"]
        for row in manuscript_rows
        if row["model"] == "original_epoch_098" and row["evaluated"]
    ]
    new_distances = [
        row["center_distance_voxels"]
        for row in manuscript_rows
        if row["model"] == "retrained_nematic" and row["evaluated"]
    ]
    assertions = {
        "controlled_has_156_events_per_model": (
            len(
                [
                    row
                    for row in controlled_rows
                    if row["model"] == "original_epoch_098"
                ]
            )
            == 156
            and len(
                [
                    row
                    for row in controlled_rows
                    if row["model"] == "retrained_nematic"
                ]
            )
            == 156
        ),
        "original_current_reproduces_release_within_0_02_deg": (
            released_agreement["original_current_within_0_02_deg"]
        ),
        "original_corrected_reproduces_prior_audit_within_1e_minus_5_deg": (
            released_agreement[
                "original_corrected_within_1e_minus_5_deg"
            ]
        ),
        "all_corrected_summary_means_are_in_0_to_90": all(
            0.0 <= value <= 90.0 for value in all_corrected
        ),
        "all_event_corrected_angles_are_finite": all(
            np.isfinite(value) for value in all_event_corrected
        ),
        "all_event_corrected_angles_are_in_0_to_90": all(
            0.0 <= value <= 90.0 + 1e-5
            for value in all_event_corrected
        ),
        "center_distance_metrics_are_unchanged": np.allclose(
            old_distances, new_distances, atol=0.0, rtol=0.0
        ),
        "new_checkpoint_differs_from_original": (
            sha256(retrained_checkpoint) != sha256(ORIGINAL_CHECKPOINT)
        ),
        "network_key_counts_match": original_keys == retrained_keys,
        "tracked_source_tree_remains_clean": (
            experiment.tracked_tree_is_clean()
        ),
    }
    result = {
        "status": "complete",
        "completed_at": now_iso(),
        "source_commit": git_head(),
        "scope": {
            "controlled_holdout": (
                "156 unique movie2 ground-truth crops under reconstructed "
                "archived training preprocessing"
            ),
            "manuscript": (
                "frozen released all-GT, matched-GT, and predicted-center "
                "sets under the prior raw-coordinate compatibility protocol"
            ),
            "split_warning": (
                "movie2 is both validation and test in the released protocol; "
                "the comparison is paired and controlled but not an untouched "
                "external generalization test"
            ),
        },
        "checkpoints": {
            "original": checkpoint_metadata(ORIGINAL_CHECKPOINT),
            "retrained": checkpoint_metadata(retrained_checkpoint),
            "network_keys_loaded": {
                "original": original_keys,
                "retrained": retrained_keys,
            },
        },
        "controlled_unique_holdout": controlled,
        "manuscript_frozen_centers": manuscript,
        "released_and_prior_audit_agreement": released_agreement,
        "frozen_center_counts": center_summary,
        "non_angular_metrics": {
            "detection": (
                "Unchanged by construction: the regression-only experiment "
                "uses frozen segmentation centers/matches, so released "
                "TP=124, FP=1, FN=21, F1=0.9185185."
            ),
            "center_distance": (
                "Unchanged exactly because center predictions and matching "
                "are frozen."
            ),
            "length": (
                "Length head and loss are unchanged but shared features are "
                "jointly trained; paired length effects are reported above."
            ),
        },
        "training_state": state,
        "outputs": {
            "summary_csv": rel(SUMMARY_CSV),
            "controlled_events_csv": rel(CONTROLLED_EVENTS_CSV),
            "manuscript_events_csv": rel(MANUSCRIPT_EVENTS_CSV),
            "distribution_csv": rel(DISTRIBUTION_CSV),
            "predictions": rel(PREDICTION_ROOT),
            "evaluation_log": rel(EVALUATION_LOG),
        },
        "assertions": assertions,
        "all_assertions_pass": all(assertions.values()),
    }
    atomic_write_json(EVALUATION_JSON, result)
    if not result["all_assertions_pass"]:
        failed = [name for name, passed in assertions.items() if not passed]
        raise AssertionError(f"Evaluation assertions failed: {failed}")
    return result


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with EVALUATION_LOG.open("a", encoding="utf-8", buffering=1) as stream:
        with contextlib.redirect_stdout(Tee(sys.stdout, stream)), (
            contextlib.redirect_stderr(Tee(sys.stderr, stream))
        ):
            print(f"\n[{now_iso()}] evaluation start")
            try:
                result = run_evaluation()
                print(
                    f"Evaluation complete in {time.perf_counter() - started:.1f}s: "
                    f"{sum(result['assertions'].values())}/"
                    f"{len(result['assertions'])} assertions"
                )
            except Exception:
                failure = {
                    "status": "failed",
                    "updated_at": now_iso(),
                    "traceback": traceback.format_exc(),
                }
                atomic_write_json(EVALUATION_JSON, failure)
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
