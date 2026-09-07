"""Validate the generic regression preprocessing fix without retraining.

This audit-only program never runs segmentation or training. It reuses frozen
detector centers and existing regression checkpoints, writes every result in
the audit-only output directory, and resumes at phase and inference level.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import io
import json
import math
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

import numpy as np
import torch

import nematic_retraining_evaluation as nuclei_evaluation
import nematic_retraining_experiment as nuclei_experiment
import neural_tube_nematic_evaluation as neural_evaluation
import neural_tube_nematic_experiment as neural_common
import neural_tube_regression_center_replay as neural_replay
import regression_checkpoint_compatibility as nuclei_compatibility
from dare3d.data.components.regression_geometry import TRAINING_CONSISTENT
from dare3d.metrics.infer_measure import CenterList


OUTPUT_ROOT = HERE / "regression_preprocessing_fix_validation"
PREDICTION_ROOT = OUTPUT_ROOT / "predictions"
STATE_PATH = OUTPUT_ROOT / "state.json"
LOG_PATH = OUTPUT_ROOT / "run.log"
CROP_RESULT_PATH = OUTPUT_ROOT / "crop_parity.json"
CROP_CSV_PATH = OUTPUT_ROOT / "crop_parity.csv"
NEURAL_RESULT_PATH = OUTPUT_ROOT / "neural_tube_result.json"
NEURAL_CSV_PATH = OUTPUT_ROOT / "neural_tube_event_metrics.csv"
NUCLEI_RESULT_PATH = OUTPUT_ROOT / "nuclei_result.json"
NUCLEI_CSV_PATH = OUTPUT_ROOT / "nuclei_event_metrics.csv"
RESULT_PATH = OUTPUT_ROOT / "result.json"
REPORT_PATH = OUTPUT_ROOT / "REGRESSION_PREPROCESSING_FIX_VALIDATION.md"

MODE_ORDER = (
    "all_groundtruth_centers",
    "matched_groundtruth_centers",
    "predicted_centers",
)

PIPELINE_FILES = (
    REPO / "dare3d/data/components/abstract_celldataset.py",
    REPO / "dare3d/data/components/regress_3dataset.py",
    REPO / "dare3d/data/components/regression_geometry.py",
    REPO / "dare3d/metrics/inference.py",
    REPO / "dare3d/metrics/infer_measure.py",
)

NEURAL_PROFILES = (
    {
        "id": "released_original_native_0621",
        "model": "original",
        "source_spacing_xyz_um": (0.621, 0.621, 2.0),
        "expected_shape_xyz": (635, 635, 20),
        "provenance": "released regression .hydra/config.yaml",
    },
    {
        "id": "released_original_common_0208",
        "model": "original",
        "source_spacing_xyz_um": (0.208, 0.208, 1.0),
        "expected_shape_xyz": (212, 212, 10),
        "provenance": "common-grid diagnostic using manuscript calibration",
    },
    {
        "id": "nematic_retrained_native_0208",
        "model": "retrained",
        "source_spacing_xyz_um": (0.208, 0.208, 1.0),
        "expected_shape_xyz": (212, 212, 10),
        "provenance": "saved retraining experiment configuration",
    },
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
        return jsonable(value.item())
    if isinstance(value, Path):
        if value.resolve().is_relative_to(REPO.resolve()):
            return rel(value)
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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


def sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def combined_hash(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(rel(path).encode("utf-8"))
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def array_hash(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def describe(values) -> dict[str, Any]:
    values = np.asarray(list(values), dtype=np.float64)
    nominal_n = int(values.size)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"nominal_n": nominal_n, "n": 0}
    standard_deviation = float(np.std(values))
    return {
        "nominal_n": nominal_n,
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std_population": standard_deviation,
        "sem": standard_deviation / math.sqrt(len(values)),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "minimum": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(np.max(values)),
    }


def axis_error_degrees(predicted, true) -> float:
    predicted = np.asarray(predicted, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    predicted_norm = float(np.linalg.norm(predicted))
    true_norm = float(np.linalg.norm(true))
    if predicted_norm <= 1e-8 or true_norm <= 1e-8:
        return float("nan")
    dot = abs(float(np.dot(predicted / predicted_norm, true / true_norm)))
    return float(np.degrees(np.arccos(np.clip(dot, 0.0, 1.0))))


def checkpoint_record(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "path": rel(path),
        "sha256": sha256(path),
        "epoch": int(checkpoint.get("epoch", -1)),
        "global_step": int(checkpoint.get("global_step", -1)),
    }


def prediction_arrays(predictions) -> dict[str, np.ndarray]:
    return {
        "centers": np.asarray(
            [item["center_raw"] for item in predictions], dtype=np.float64
        ),
        "lengths": np.asarray(
            [item["length_regression_voxels"] for item in predictions],
            dtype=np.float64,
        ),
        "quaternions": np.asarray(
            [item["rotation"] for item in predictions], dtype=np.float64
        ),
    }


def predictions_from_arrays(dataset, arrays) -> list[dict[str, Any]]:
    predictions = []
    for center, length, quaternion in zip(
        arrays["centers"], arrays["lengths"], arrays["quaternions"]
    ):
        center = tuple(float(item) for item in center)
        decoded = dataset.decode_prediction(center, float(length), quaternion)
        predictions.append(
            {
                "center": center,
                "center_raw": center,
                "center_regression": dataset.raw_center_to_regression(center),
                "length": np.asarray([length], dtype=np.float32),
                "length_regression_voxels": float(length),
                "rotation": np.asarray(quaternion, dtype=np.float64),
                **decoded,
            }
        )
    return predictions


def run_or_load_predictions(
    cache_name: str,
    dataset,
    net,
    centers,
    checkpoint_path: Path,
    pipeline_hash: str,
):
    PREDICTION_ROOT.mkdir(parents=True, exist_ok=True)
    archive_path = PREDICTION_ROOT / f"{cache_name}.npz"
    metadata_path = PREDICTION_ROOT / f"{cache_name}.json"
    center_array = np.asarray(centers, dtype=np.float64)
    identity = {
        "schema_version": 1,
        "pipeline_hash": pipeline_hash,
        "checkpoint_sha256": sha256(checkpoint_path),
        "centers_sha256": array_hash(center_array),
        "preprocessing_manifest": dataset.preprocessing_manifest(),
    }
    if archive_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("identity") == jsonable(identity):
            with np.load(archive_path) as archive:
                arrays = {
                    name: np.asarray(archive[name]).copy()
                    for name in archive.files
                }
            if np.array_equal(arrays["centers"], center_array):
                metadata["reused_this_run"] = True
                return predictions_from_arrays(dataset, arrays), metadata

    started = time.perf_counter()
    predictions = neural_evaluation.regression_inference(
        dataset, net, centers, "cuda", output_dir=None
    )
    arrays = prediction_arrays(predictions)
    temporary = archive_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(archive_path)
    metadata = {
        "identity": identity,
        "archive": rel(archive_path),
        "archive_sha256": sha256(archive_path),
        "elapsed_seconds": time.perf_counter() - started,
        "n": len(predictions),
        "reused_this_run": False,
        "completed_at": now_iso(),
    }
    atomic_json(metadata_path, metadata)
    return predictions, metadata


def event_rows(scope, profile, model, mode, targets, predictions):
    if len(targets) != len(predictions):
        raise AssertionError(
            (scope, profile, model, mode, len(targets), len(predictions))
        )
    rows = []
    for pair_index, (target, prediction) in enumerate(zip(targets, predictions)):
        row = {
            "scope": scope,
            "profile": profile,
            "model": model,
            "mode": mode,
            "pair_index": pair_index,
            "reported_pair_count": len(targets),
            "evaluated": target is not None,
        }
        for prefix, center in (
            ("inference_raw", prediction["center_raw"]),
            ("inference_regression", prediction["center_regression"]),
        ):
            for coordinate, value in zip("mtxyz", center):
                row[f"{prefix}_{coordinate}"] = float(value)
        if target is not None:
            true_axis = np.asarray(target["axis_physical_xyz"], dtype=np.float64)
            predicted_axis = np.asarray(
                prediction["axis_physical_xyz"], dtype=np.float64
            )
            true_length_regression = float(
                target["length_regression_voxels"]
            )
            predicted_length_regression = float(
                prediction["length_regression_voxels"]
            )
            true_length_physical = float(target["length_physical_um"])
            predicted_length_physical = float(
                prediction["length_physical_um"]
            )
            row.update(
                {
                    "event_id": target["event_id"],
                    "segmentation_event_id": target.get(
                        "segmentation_event_id"
                    ),
                    "annotation_index": int(target["annotation_index"]),
                    "target_candidate_count": int(
                        target["target_candidate_count"]
                    ),
                    "target_center_distance_raw_voxels": float(
                        target["target_center_distance_raw_voxels"]
                    ),
                    "corrected_nematic_axis_error_deg": axis_error_degrees(
                        predicted_axis, true_axis
                    ),
                    "production_quaternion_error_deg":
                        neural_evaluation.production_error_numpy(
                            prediction["rotation"], target["rotation"]
                        ),
                    "true_length_regression_voxels":
                        true_length_regression,
                    "predicted_length_regression_voxels":
                        predicted_length_regression,
                    "absolute_length_error_regression_voxels": abs(
                        predicted_length_regression - true_length_regression
                    ),
                    "true_length_physical_um": true_length_physical,
                    "predicted_length_physical_um": predicted_length_physical,
                    "absolute_length_error_physical_um": abs(
                        predicted_length_physical - true_length_physical
                    ),
                    "center_distance_raw_voxels": float(
                        np.linalg.norm(
                            np.asarray(prediction["center_raw"])[1:]
                            - np.asarray(target["center_raw"])[1:]
                        )
                    ),
                }
            )
            for prefix, axis in (
                ("true_axis_physical", true_axis),
                ("predicted_axis_physical", predicted_axis),
            ):
                for coordinate, value in zip("xyz", axis):
                    row[f"{prefix}_{coordinate}"] = float(value)
        rows.append(row)
    return rows


def summarize_event_rows(
    rows, group_fields=("profile", "model", "mode")
):
    result = {}
    group_keys = sorted(
        {tuple(row[field] for field in group_fields) for row in rows}
    )
    for key in group_keys:
        selected = [
            row
            for row in rows
            if tuple(row[field] for field in group_fields) == key
        ]
        evaluated = [row for row in selected if row["evaluated"]]
        label = "|".join(str(item) for item in key)
        result[label] = {
            "group": dict(zip(group_fields, key)),
            "reported_n": len(selected),
            "effective_n": len(evaluated),
            "missing_target_n": len(selected) - len(evaluated),
            "corrected_nematic_axis_error_deg": describe(
                row["corrected_nematic_axis_error_deg"] for row in evaluated
            ),
            "production_quaternion_error_deg": describe(
                row["production_quaternion_error_deg"] for row in evaluated
            ),
            "absolute_length_error_regression_voxels": describe(
                row["absolute_length_error_regression_voxels"]
                for row in evaluated
            ),
            "absolute_length_error_physical_um": describe(
                row["absolute_length_error_physical_um"]
                for row in evaluated
            ),
            "center_distance_raw_voxels": describe(
                row["center_distance_raw_voxels"] for row in evaluated
            ),
        }
    return result


def make_neural_dataset(scale):
    dataset = neural_common.make_dataset("regression", "validation")
    dataset.default_scale = np.asarray(scale, dtype=np.float64)
    dataset.movies_scale = {}
    return dataset


def enumerate_training_events(dataset):
    events = []
    crop_index = 0
    for movie_index, frames in enumerate(dataset.movies_bipoints_raw):
        for time_index, bipoints in enumerate(frames):
            if (
                time_index <= 1
                or time_index >= dataset.movies_im[movie_index].shape[0]
            ):
                continue
            for annotation_index, bipoint in enumerate(bipoints):
                if crop_index >= len(dataset.crops):
                    raise AssertionError(
                        "More annotations than stored regression crops"
                    )
                events.append(
                    {
                        "crop_index": crop_index,
                        "movie_index": movie_index,
                        "time_index": time_index,
                        "annotation_index": annotation_index,
                        "bipoint_raw": bipoint,
                    }
                )
                crop_index += 1
    if crop_index != len(dataset.crops):
        raise AssertionError((crop_index, len(dataset.crops)))
    return events


def run_crop_parity(pipeline_hash: str, validator_hash: str):
    print("Running training/inference crop-parity validation", flush=True)
    rows = []
    profile_results = {}
    for profile in (NEURAL_PROFILES[0], NEURAL_PROFILES[2]):
        print(f"Crop parity profile={profile['id']}", flush=True)
        training = make_neural_dataset(profile["source_spacing_xyz_um"])
        training.init()
        inference = make_neural_dataset(profile["source_spacing_xyz_um"])
        inference.init_inference(TRAINING_CONSISTENT)
        events = enumerate_training_events(training)
        full_movie_equal = all(
            np.array_equal(left, right)
            for left, right in zip(
                training.movies_im, inference.movies_im
            )
        )
        exact_grid_crop_count = 0
        exact_raw_midpoint_crop_count = 0
        maximum_grid_difference = 0.0
        maximum_raw_midpoint_difference = 0.0
        offsets = []
        for event in events:
            crop_index = event["crop_index"]
            crop_spec = training.crops[crop_index]
            movie_index, time_slice, x_slice, y_slice, z_slice = crop_spec
            training_crop = training.movies_im[movie_index][
                time_slice, x_slice, y_slice, z_slice
            ].astype(np.float32, copy=False)
            grid_center = np.asarray(
                [x_slice.start, y_slice.start, z_slice.start],
                dtype=np.float64,
            )
            center_regression = (
                movie_index,
                event["time_index"],
                *grid_center,
            )
            grid_crop = (
                inference.get_crop_from_center(
                    center_regression, center_space="regression"
                )
                .cpu()
                .numpy()
            )
            p1_raw, p2_raw = event["bipoint_raw"]
            midpoint_raw = 0.5 * (
                np.asarray(p1_raw, dtype=np.float64)
                + np.asarray(p2_raw, dtype=np.float64)
            )
            center_raw = (
                movie_index,
                event["time_index"],
                *midpoint_raw,
            )
            raw_midpoint_crop = (
                inference.get_crop_from_center(
                    center_raw, center_space="raw"
                )
                .cpu()
                .numpy()
            )
            mapped = np.asarray(
                inference.raw_center_to_regression(center_raw)[2:],
                dtype=np.float64,
            )
            mapped_rounded = np.rint(mapped)
            offset = mapped_rounded - grid_center
            grid_difference = float(
                np.max(np.abs(grid_crop - training_crop))
            )
            raw_difference = float(
                np.max(np.abs(raw_midpoint_crop - training_crop))
            )
            grid_equal = np.array_equal(grid_crop, training_crop)
            raw_equal = np.array_equal(raw_midpoint_crop, training_crop)
            exact_grid_crop_count += int(grid_equal)
            exact_raw_midpoint_crop_count += int(raw_equal)
            maximum_grid_difference = max(
                maximum_grid_difference, grid_difference
            )
            maximum_raw_midpoint_difference = max(
                maximum_raw_midpoint_difference, raw_difference
            )
            offsets.append(float(np.linalg.norm(offset)))
            row = {
                "profile": profile["id"],
                "crop_index": crop_index,
                "movie": training.movie_names[movie_index],
                "movie_index": movie_index,
                "time_index": event["time_index"],
                "annotation_index": event["annotation_index"],
                "grid_crop_exact": grid_equal,
                "raw_midpoint_crop_exact": raw_equal,
                "grid_crop_max_abs_difference": grid_difference,
                "raw_midpoint_crop_max_abs_difference": raw_difference,
                "training_crop_sha256": array_hash(training_crop),
                "grid_crop_sha256": array_hash(grid_crop),
                "raw_midpoint_crop_sha256": array_hash(raw_midpoint_crop),
                "mapped_center_offset_norm_regression_voxels": float(
                    np.linalg.norm(offset)
                ),
            }
            for prefix, values in (
                ("raw_midpoint", midpoint_raw),
                ("mapped_regression_center", mapped),
                ("rounded_regression_center", mapped_rounded),
                ("training_grid_center", grid_center),
                ("mapped_minus_training", offset),
            ):
                for coordinate, value in zip("xyz", values):
                    row[f"{prefix}_{coordinate}"] = float(value)
            rows.append(row)
        manifest = inference.preprocessing_manifest()
        observed_shape = tuple(
            manifest["movies"][0]["regression_shape_xyz"]
        )
        profile_results[profile["id"]] = {
            "preprocessing_manifest": manifest,
            "training_crop_count": len(events),
            "full_processed_movie_byte_identical": full_movie_equal,
            "exact_same_grid_crop_count": exact_grid_crop_count,
            "exact_raw_midpoint_crop_count": exact_raw_midpoint_crop_count,
            "maximum_same_grid_crop_abs_difference":
                maximum_grid_difference,
            "maximum_raw_midpoint_crop_abs_difference":
                maximum_raw_midpoint_difference,
            "mapped_center_offset_regression_voxels": describe(offsets),
            "observed_shape_xyz": observed_shape,
            "expected_shape_xyz": profile["expected_shape_xyz"],
            "shape_matches_profile":
                observed_shape == profile["expected_shape_xyz"],
        }
        del training, inference
        gc.collect()
    write_csv(CROP_CSV_PATH, rows)
    assertions = {
        "training_and_inference_full_movies_are_byte_identical": all(
            result["full_processed_movie_byte_identical"]
            for result in profile_results.values()
        ),
        "same_grid_center_crops_are_byte_identical": all(
            result["exact_same_grid_crop_count"]
            == result["training_crop_count"]
            for result in profile_results.values()
        ),
        "raw_midpoint_mapping_differs_by_at_most_one_per_axis": all(
            result["mapped_center_offset_regression_voxels"]["maximum"]
            <= math.sqrt(3.0)
            for result in profile_results.values()
        ),
        "profile_shapes_match": all(
            result["shape_matches_profile"]
            for result in profile_results.values()
        ),
    }
    result = {
        "status": (
            "complete"
            if all(assertions.values())
            else "complete_with_failed_assertions"
        ),
        "completed_at": now_iso(),
        "pipeline_hash": pipeline_hash,
        "validator_hash": validator_hash,
        "profiles": profile_results,
        "assertions": assertions,
        "outputs": {
            "event_csv": rel(CROP_CSV_PATH),
            "event_csv_sha256": sha256(CROP_CSV_PATH),
        },
    }
    atomic_json(CROP_RESULT_PATH, result)
    return result


def load_neural_centers():
    centers = {}
    for mode, path in neural_replay.MODE_SOURCES.items():
        centers[mode], _ = neural_replay.centers_from_archive(path)
    return centers


def neural_summary_key(profile, mode):
    model = next(
        item["model"] for item in NEURAL_PROFILES
        if item["id"] == profile
    )
    return f"{profile}|{model}|{mode}"


def run_neural_validation(pipeline_hash: str, validator_hash: str):
    print("Running cached-center neural-tube validation", flush=True)
    centers = load_neural_centers()
    rows = []
    profiles = {}
    inference_records = {}
    checkpoint_records = {}
    target_identity_ok = True
    all_targets_present = True
    for profile in NEURAL_PROFILES:
        profile_id = profile["id"]
        model_name = profile["model"]
        print(f"Neural profile={profile_id}", flush=True)
        dataset = make_neural_dataset(
            profile["source_spacing_xyz_um"]
        )
        dataset.init_inference(TRAINING_CONSISTENT)
        manifest = dataset.preprocessing_manifest()
        observed_shape = tuple(
            manifest["movies"][0]["regression_shape_xyz"]
        )
        targets_all = dataset.gather_groundtruth_info(
            centers["all_groundtruth_centers"]
        )
        targets_matched = dataset.gather_groundtruth_info(
            centers["matched_groundtruth_centers"]
        )
        targets_by_mode = {
            "all_groundtruth_centers": targets_all,
            "matched_groundtruth_centers": targets_matched,
            "predicted_centers": targets_matched,
        }
        target_identity_ok &= all(
            left is right
            for left, right in zip(
                targets_by_mode["matched_groundtruth_centers"],
                targets_by_mode["predicted_centers"],
            )
        )
        all_targets_present &= not any(
            target is None
            for target in targets_all + targets_matched
        )
        checkpoint_path = neural_evaluation.checkpoint_path(
            "regression", model_name
        )
        checkpoint_records[model_name] = checkpoint_record(
            checkpoint_path
        )
        net, loaded_keys = neural_evaluation.load_regression_net(
            model_name
        )
        profile_records = {}
        for mode in MODE_ORDER:
            print(
                f"  mode={mode}, N={len(centers[mode])}",
                flush=True,
            )
            predictions, record = run_or_load_predictions(
                f"neural_{profile_id}_{mode}",
                dataset,
                net,
                centers[mode],
                checkpoint_path,
                pipeline_hash,
            )
            profile_records[mode] = record
            rows.extend(
                event_rows(
                    "neural_tube_movie_I2_frozen_centers",
                    profile_id,
                    model_name,
                    mode,
                    targets_by_mode[mode],
                    predictions,
                )
            )
        profiles[profile_id] = {
            **profile,
            "source_spacing_xyz_um":
                list(profile["source_spacing_xyz_um"]),
            "expected_shape_xyz": list(profile["expected_shape_xyz"]),
            "observed_shape_xyz": list(observed_shape),
            "shape_matches_profile":
                observed_shape == profile["expected_shape_xyz"],
            "loaded_network_keys": loaded_keys,
            "preprocessing_manifest": manifest,
        }
        inference_records[profile_id] = profile_records
        del net, dataset
        gc.collect()
        torch.cuda.empty_cache()
    write_csv(NEURAL_CSV_PATH, rows)
    summary = summarize_event_rows(rows)
    predicted_mode = "predicted_centers"
    old_common = summary[
        neural_summary_key(
            "released_original_common_0208", predicted_mode
        )
    ]["corrected_nematic_axis_error_deg"]["mean"]
    new_native = summary[
        neural_summary_key(
            "nematic_retrained_native_0208", predicted_mode
        )
    ]["corrected_nematic_axis_error_deg"]["mean"]
    prior_path = (
        neural_evaluation.EVALUATION_ROOT
        / "released_movie_I2_center_replay/result.json"
    )
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    prior_compact = {
        model: {
            mode: prior["summary"][model][mode]
            for mode in MODE_ORDER
        }
        for model in ("original", "retrained")
    }
    prior_retrained_raw_mean = prior_compact["retrained"][
        predicted_mode
    ]["corrected_nematic_axis_error_deg"]["mean"]
    comparisons = {
        "same_0208_grid_predicted_centers": {
            "original_mean_deg": old_common,
            "retrained_mean_deg": new_native,
            "retrained_minus_original_mean_deg":
                new_native - old_common,
            "relative_change_percent":
                100.0 * (new_native - old_common) / old_common,
        },
        "retrained_corrected_path_vs_prior_legacy_raw_predicted_centers": {
            "legacy_raw_mean_deg": prior_retrained_raw_mean,
            "training_consistent_mean_deg": new_native,
            "difference_deg":
                new_native - prior_retrained_raw_mean,
        },
    }
    assertions = {
        "center_counts_unchanged": all(
            len(centers[mode]) == neural_replay.EXPECTED_COUNTS[mode]
            for mode in MODE_ORDER
        ),
        "all_centers_link_to_annotations": all_targets_present,
        "predicted_center_mode_reuses_matched_ground_truth_objects":
            target_identity_ok,
        "all_checkpoint_profiles_have_expected_shape": all(
            profile["shape_matches_profile"]
            for profile in profiles.values()
        ),
        "both_existing_checkpoints_loaded":
            set(checkpoint_records) == {"original", "retrained"},
        "retrained_improves_on_same_0208_grid_at_predicted_centers":
            new_native < old_common,
        "corrected_path_recovers_retrained_improvement_over_legacy_raw":
            new_native < prior_retrained_raw_mean,
    }
    result = {
        "status": (
            "complete"
            if all(assertions.values())
            else "complete_with_failed_assertions"
        ),
        "completed_at": now_iso(),
        "scope":
            "movie_I2 frozen released center sets; segmentation was not rerun",
        "scientific_role":
            "historical validation/reported set, not independent test",
        "pipeline_hash": pipeline_hash,
        "validator_hash": validator_hash,
        "profiles": profiles,
        "checkpoints": checkpoint_records,
        "inference": inference_records,
        "summary": summary,
        "prior_legacy_raw_reference": {
            "path": rel(prior_path),
            "summary": prior_compact,
        },
        "comparisons": comparisons,
        "assertions": assertions,
        "outputs": {
            "event_csv": rel(NEURAL_CSV_PATH),
            "event_csv_sha256": sha256(NEURAL_CSV_PATH),
        },
    }
    atomic_json(NEURAL_RESULT_PATH, result)
    return result


def nuclei_checkpoint_paths():
    state = json.loads(
        (
            nuclei_experiment.RUN_ROOT / "run_state.json"
        ).read_text(encoding="utf-8")
    )
    return {
        "original_epoch_098":
            nuclei_experiment.ORIGINAL_CHECKPOINT,
        "retrained_nematic_epoch_095":
            REPO / state["best_model_path"],
    }


def load_nuclei_legacy_rows():
    path = (
        nuclei_experiment.RUN_ROOT / "manuscript_event_metrics.csv"
    )
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            row["pair_index"] = int(row["pair_index"])
            row["evaluated"] = row["evaluated"].lower() == "true"
            for field in (
                "corrected_nematic_axis_error_deg",
                "absolute_length_error_voxels",
            ):
                row[field] = (
                    float(row[field])
                    if row[field]
                    else float("nan")
                )
            rows.append(row)
    return path, rows


def summarize_legacy_nuclei(rows):
    result = {}
    for model in ("original_epoch_098", "retrained_nematic"):
        for mode in MODE_ORDER:
            selected = [
                row
                for row in rows
                if row["model"] == model and row["mode"] == mode
            ]
            evaluated = [
                row for row in selected if row["evaluated"]
            ]
            result[f"{model}|{mode}"] = {
                "reported_n": len(selected),
                "effective_n": len(evaluated),
                "corrected_nematic_axis_error_deg": describe(
                    row["corrected_nematic_axis_error_deg"]
                    for row in evaluated
                ),
                "absolute_length_error_regression_voxels": describe(
                    row["absolute_length_error_voxels"]
                    for row in evaluated
                ),
            }
    return result


def run_nuclei_validation(pipeline_hash: str, validator_hash: str):
    print("Running nuclei compatibility validation", flush=True)
    dataset = nuclei_experiment.make_dataset("test")
    dataset.init_inference(TRAINING_CONSISTENT)
    info, center_summary = nuclei_compatibility.load_frozen_info()
    center_list = CenterList(0, info)
    center_list.compute_real_rot_len_values(dataset)
    centers_by_mode = {
        "all_groundtruth_centers": center_list.all_gt_centers,
        "matched_groundtruth_centers": center_list.true_centers,
        "predicted_centers": center_list.predicted_centers,
    }
    targets_by_mode = {
        "all_groundtruth_centers": center_list.real_rot_length,
        "matched_groundtruth_centers":
            center_list.real_rot_length_matched,
        "predicted_centers": center_list.real_rot_length_matched,
    }
    target_identity_ok = all(
        left is right
        for left, right in zip(
            targets_by_mode["matched_groundtruth_centers"],
            targets_by_mode["predicted_centers"],
        )
    )
    rows = []
    inference_records = {}
    checkpoints = {}
    for model_name, checkpoint_path in nuclei_checkpoint_paths().items():
        print(f"Nuclei checkpoint={model_name}", flush=True)
        checkpoints[model_name] = checkpoint_record(checkpoint_path)
        net, loaded_keys = nuclei_evaluation.make_net(checkpoint_path)
        model_records = {
            "loaded_network_keys": loaded_keys,
            "modes": {},
        }
        for mode in MODE_ORDER:
            print(
                f"  mode={mode}, N={len(centers_by_mode[mode])}",
                flush=True,
            )
            predictions, record = run_or_load_predictions(
                f"nuclei_{model_name}_{mode}",
                dataset,
                net,
                centers_by_mode[mode],
                checkpoint_path,
                pipeline_hash,
            )
            model_records["modes"][mode] = record
            rows.extend(
                event_rows(
                    "nuclei_movie2_frozen_centers",
                    "training_consistent_log_reconstructed",
                    model_name,
                    mode,
                    targets_by_mode[mode],
                    predictions,
                )
            )
        inference_records[model_name] = model_records
        del net
        gc.collect()
        torch.cuda.empty_cache()

    legacy_path, legacy_rows = load_nuclei_legacy_rows()
    legacy_by_pair = {
        (row["model"], row["mode"], row["pair_index"]): row
        for row in legacy_rows
    }
    model_to_legacy = {
        "original_epoch_098": "original_epoch_098",
        "retrained_nematic_epoch_095": "retrained_nematic",
    }
    for row in rows:
        legacy = legacy_by_pair.get(
            (
                model_to_legacy[row["model"]],
                row["mode"],
                row["pair_index"],
            )
        )
        row["legacy_raw_evaluated"] = bool(
            legacy and legacy["evaluated"]
        )
        if (
            row["evaluated"]
            and legacy
            and legacy["evaluated"]
        ):
            old_angle = legacy[
                "corrected_nematic_axis_error_deg"
            ]
            old_length = legacy["absolute_length_error_voxels"]
            row[
                "legacy_raw_corrected_nematic_axis_error_deg"
            ] = old_angle
            row[
                "training_consistent_minus_legacy_angle_deg"
            ] = row["corrected_nematic_axis_error_deg"] - old_angle
            row[
                "legacy_raw_absolute_length_error_voxels"
            ] = old_length
            row[
                "training_consistent_minus_legacy_length_error_voxels"
            ] = (
                row["absolute_length_error_regression_voxels"]
                - old_length
            )
    write_csv(NUCLEI_CSV_PATH, rows)
    current_summary = summarize_event_rows(rows)
    legacy_summary = summarize_legacy_nuclei(legacy_rows)
    paired = {}
    for model_name, legacy_name in model_to_legacy.items():
        for mode in MODE_ORDER:
            selected = [
                row
                for row in rows
                if row["model"] == model_name
                and row["mode"] == mode
                and row.get("legacy_raw_evaluated")
                and row["evaluated"]
            ]
            paired[f"{model_name}|{mode}"] = {
                "n": len(selected),
                "training_consistent_minus_legacy_angle_deg":
                    describe(
                        row[
                            "training_consistent_minus_legacy_angle_deg"
                        ]
                        for row in selected
                    ),
                "training_consistent_minus_legacy_length_error_voxels":
                    describe(
                        row[
                            "training_consistent_minus_legacy_length_error_voxels"
                        ]
                        for row in selected
                    ),
                "legacy_model_label": legacy_name,
                "negative_difference_is_improvement": True,
            }
    original_predicted = paired[
        "original_epoch_098|predicted_centers"
    ]
    current_counts = {
        mode: sum(
            target is not None for target in targets_by_mode[mode]
        )
        for mode in MODE_ORDER
    }
    legacy_counts = {
        mode: legacy_summary[
            f"original_epoch_098|{mode}"
        ]["effective_n"]
        for mode in MODE_ORDER
    }
    assertions = {
        "detector_center_counts_unchanged": {
            mode: len(centers_by_mode[mode])
            for mode in MODE_ORDER
        }
        == {
            "all_groundtruth_centers": 145,
            "matched_groundtruth_centers": 124,
            "predicted_centers": 124,
        },
        "target_association_effective_counts_unchanged":
            current_counts == legacy_counts,
        "predicted_center_mode_reuses_matched_ground_truth_objects":
            target_identity_ok,
        "both_existing_checkpoints_loaded":
            set(checkpoints)
            == {
                "original_epoch_098",
                "retrained_nematic_epoch_095",
            },
        "original_predicted_center_mean_angle_not_degraded_over_2_deg":
            original_predicted[
                "training_consistent_minus_legacy_angle_deg"
            ]["mean"] < 2.0,
        "original_predicted_center_mean_length_error_not_degraded_over_1_voxel":
            original_predicted[
                "training_consistent_minus_legacy_length_error_voxels"
            ]["mean"] < 1.0,
    }
    result = {
        "status": (
            "complete"
            if all(assertions.values())
            else "complete_with_failed_assertions"
        ),
        "completed_at": now_iso(),
        "scope":
            "nuclei movie2 frozen centers; segmentation and matching were not rerun",
        "geometry_caveat": (
            "Target shape is reconstructed from the archived training log "
            "because historical scales.json is absent; it is not "
            "independently verified calibration."
        ),
        "pipeline_hash": pipeline_hash,
        "validator_hash": validator_hash,
        "center_summary": center_summary,
        "preprocessing_manifest": dataset.preprocessing_manifest(),
        "checkpoints": checkpoints,
        "inference": inference_records,
        "current_summary": current_summary,
        "legacy_raw_reference": {
            "path": rel(legacy_path),
            "sha256": sha256(legacy_path),
            "summary": legacy_summary,
        },
        "paired_differences": paired,
        "target_effective_counts": current_counts,
        "assertions": assertions,
        "outputs": {
            "event_csv": rel(NUCLEI_CSV_PATH),
            "event_csv_sha256": sha256(NUCLEI_CSV_PATH),
        },
    }
    atomic_json(NUCLEI_RESULT_PATH, result)
    del dataset
    gc.collect()
    return result


def format_mean_sd(summary):
    return "{:.3f} +/- {:.3f}".format(
        summary["mean"], summary["std_population"]
    )


def write_report(result):
    crop = result["phases"]["crop_parity"]
    neural = result["phases"]["neural_tube"]
    nuclei = result["phases"]["nuclei"]
    lines = [
        "# Generic regression preprocessing fix: no-retraining validation",
        "",
        "This validation did not run segmentation or training. Detector centers,",
        "matches, annotations, architectures, and checkpoints were frozen; only",
        "regression image-space preparation and coordinate decoding changed.",
        "",
        "## Crop parity",
        "",
        "| Checkpoint profile | Crops | Same-grid exact | Raw-midpoint exact | Maximum mapped-center offset (regression voxels) |",
        "|---|---:|---:|---:|---:|",
    ]
    for profile, values in crop["profiles"].items():
        lines.append(
            f"| {profile} | {values['training_crop_count']} | "
            f"{values['exact_same_grid_crop_count']} | "
            f"{values['exact_raw_midpoint_crop_count']} | "
            f"{values['mapped_center_offset_regression_voxels']['maximum']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The full preprocessed movies and every crop extracted at the same",
            "regression-grid center must be byte-identical. Raw annotation",
            "midpoints can differ by endpoint-rounding quantisation; this is",
            "reported rather than hidden.",
            "",
            "## Neural tube (movie_I2 frozen centers)",
            "",
            "movie_I2 is the historical validation/reported set, not an",
            "independent test.",
            "",
            "| Profile | Center mode | N | Corrected nematic error, mean +/- population SD (deg) | Length MAE (physical units) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for values in neural["summary"].values():
        group = values["group"]
        lines.append(
            f"| {group['profile']} | {group['mode']} | "
            f"{values['effective_n']} | "
            f"{format_mean_sd(values['corrected_nematic_axis_error_deg'])} | "
            f"{values['absolute_length_error_physical_um']['mean']:.3f} |"
        )
    comparison = neural["comparisons"][
        "same_0208_grid_predicted_centers"
    ]
    lines.extend(
        [
            "",
            "At the same saved detector-predicted centers and the same",
            "0.208/0.208/1 regression grid, the retrained-minus-original mean",
            "angular difference is",
            f"{comparison['retrained_minus_original_mean_deg']:.3f} degrees.",
            "",
            "## Nuclei (movie2 frozen centers)",
            "",
            "The training-space geometry is log-reconstructed because the",
            "historical scales.json is unavailable.",
            "",
            "| Checkpoint | Center mode | Legacy raw mean (deg) | Corrected-path mean (deg) | Paired change (deg) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    legacy_model = {
        "original_epoch_098": "original_epoch_098",
        "retrained_nematic_epoch_095": "retrained_nematic",
    }
    for model, old_model in legacy_model.items():
        for mode in MODE_ORDER:
            current = nuclei["current_summary"][
                "training_consistent_log_reconstructed"
                f"|{model}|{mode}"
            ]["corrected_nematic_axis_error_deg"]
            old = nuclei["legacy_raw_reference"]["summary"][
                f"{old_model}|{mode}"
            ]["corrected_nematic_axis_error_deg"]
            delta = nuclei["paired_differences"][
                f"{model}|{mode}"
            ]["training_consistent_minus_legacy_angle_deg"]
            lines.append(
                f"| {model} | {mode} | {old['mean']:.3f} | "
                f"{current['mean']:.3f} | {delta['mean']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Invariants and interpretation",
            "",
            f"- Overall validation status: **{result['status']}**.",
            "- Segmentation probabilities, thresholds, detections, and matching",
            "  were not recomputed; saved center lists are validator inputs.",
            "- A detector-predicted center selects only the regression crop. The",
            "  target is the annotation associated with its matched true event.",
            "- Existing checkpoints load unchanged; no architecture or weight",
            "  conversion is performed.",
            "- Per-event values, preprocessing manifests, checkpoint hashes,",
            "  and resumable prediction caches accompany this report.",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def assemble_result(pipeline_hash: str, validator_hash: str):
    phase_paths = {
        "crop_parity": CROP_RESULT_PATH,
        "neural_tube": NEURAL_RESULT_PATH,
        "nuclei": NUCLEI_RESULT_PATH,
    }
    missing = [
        str(path)
        for path in phase_paths.values()
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            f"Cannot assemble report; missing phase results: {missing}"
        )
    phases = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in phase_paths.items()
    }
    assertions = {
        f"{phase}.{name}": value
        for phase, phase_result in phases.items()
        for name, value in phase_result["assertions"].items()
    }
    result = {
        "status": (
            "complete"
            if all(assertions.values())
            else "complete_with_failed_acceptance_criteria"
        ),
        "completed_at": now_iso(),
        "purpose":
            "no-retraining validation of training-consistent regression inference",
        "pipeline_hash": pipeline_hash,
        "validator_hash": validator_hash,
        "cuda": {
            "available": torch.cuda.is_available(),
            "device": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
        "pipeline_files": {
            rel(path): sha256(path) for path in PIPELINE_FILES
        },
        "phases": phases,
        "acceptance_assertions": assertions,
        "outputs": {
            "report": rel(REPORT_PATH),
            "crop_parity": rel(CROP_CSV_PATH),
            "neural_tube_events": rel(NEURAL_CSV_PATH),
            "nuclei_events": rel(NUCLEI_CSV_PATH),
        },
    }
    atomic_json(RESULT_PATH, result)
    write_report(result)
    result["outputs"]["report_sha256"] = sha256(REPORT_PATH)
    atomic_json(RESULT_PATH, result)
    return result


def phase_is_current(path: Path, validator_hash: str) -> bool:
    if not path.is_file():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        return result.get("validator_hash") == validator_hash
    except (OSError, ValueError):
        return False


def execute(phase: str, force: bool):
    if (
        not torch.cuda.is_available()
        and phase in {"all", "neural", "nuclei"}
    ):
        raise RuntimeError("CUDA is required for checkpoint validation")
    pipeline_hash = combined_hash(PIPELINE_FILES)
    validator_hash = combined_hash(
        PIPELINE_FILES + (Path(__file__).resolve(),)
    )
    phases = (
        ("crop", CROP_RESULT_PATH, run_crop_parity),
        ("neural", NEURAL_RESULT_PATH, run_neural_validation),
        ("nuclei", NUCLEI_RESULT_PATH, run_nuclei_validation),
    )
    selected = (
        {"crop", "neural", "nuclei"}
        if phase == "all"
        else {phase}
    )
    for name, path, function in phases:
        if name not in selected:
            continue
        if not force and phase_is_current(path, validator_hash):
            print(
                f"Reusing complete phase={name}: {rel(path)}",
                flush=True,
            )
            continue
        atomic_json(
            STATE_PATH,
            {
                "status": "running",
                "active_phase": name,
                "updated_at": now_iso(),
                "pipeline_hash": pipeline_hash,
                "validator_hash": validator_hash,
            },
        )
        function(pipeline_hash, validator_hash)
    result = None
    if phase in {"all", "report"}:
        result = assemble_result(pipeline_hash, validator_hash)
    atomic_json(
        STATE_PATH,
        {
            "status": "complete",
            "requested_phase": phase,
            "updated_at": now_iso(),
            "pipeline_hash": pipeline_hash,
            "validator_hash": validator_hash,
            "result": rel(RESULT_PATH) if result else None,
        },
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("all", "crop", "neural", "nuclei", "report"),
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open(
        "a", encoding="utf-8", buffering=1
    ) as log_stream:
        with contextlib.redirect_stdout(
            Tee(sys.stdout, log_stream)
        ), contextlib.redirect_stderr(Tee(sys.stderr, log_stream)):
            print(
                f"\n[{now_iso()}] phase={args.phase} force={args.force}",
                flush=True,
            )
            try:
                result = execute(args.phase, args.force)
                if result:
                    print(
                        json.dumps(
                            {
                                "status": result["status"],
                                "result": rel(RESULT_PATH),
                                "report": rel(REPORT_PATH),
                            },
                            indent=2,
                        )
                    )
            except BaseException:
                atomic_json(
                    STATE_PATH,
                    {
                        "status": "failed",
                        "updated_at": now_iso(),
                        "requested_phase": args.phase,
                        "traceback": traceback.format_exc(),
                    },
                )
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
