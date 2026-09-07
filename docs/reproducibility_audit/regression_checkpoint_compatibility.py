"""Replay released nuclei regression checkpoints on frozen released center sets.

This audit-only script isolates the regression stage.  It uses the exact current-
evaluator centers reconstructed from the archived nuclei probability, executes
both predeclared best/last checkpoints in separate invocations, and writes only
compact raw predictions, error tables, logs, and JSON below the audit directory.
It intentionally does not create the multi-gigabyte display TIFFs.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import inspect
import io
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from dare3d.metrics.infer_measure import CenterList, evaluate_center_pair
from dare3d.metrics.inference import regression_inference
from dare3d.models.finetune import load_net_state_dict

DATA = REPO / "DARE3d_data_190326"
MODEL_DIR = DATA / "Gastruloid_241025/weights/regression3d_exp10-b"
CONFIG = MODEL_DIR / ".hydra/config.yaml"
INPUT_DIR = DATA / "Gastruloid_241025/test_input"
INPUT_TIFF = INPUT_DIR / "movie2.tif"
LABEL_DIR = DATA / "Gastruloid_241025/trainingset/movie2/label"
CENTER_TABLE = HERE / "evidence/nuclei_current_evaluator_objects.csv"
PROVENANCE = HERE / "evidence/provenance.json"
MANIFEST = HERE / "evidence/data_manifest_sha256.csv"
OUTPUT_ROOT = HERE / "checkpoint_runs/nuclei_regression"
SOURCE_COMMIT = "fe2b14d732359f2bdaf8b197574ad818899ce123"
CHECKPOINTS = {
    "best": MODEL_DIR / "checkpoints/epoch_098.ckpt",
    "last": MODEL_DIR / "checkpoints/last.ckpt",
}
RUN_NAMES = {"best": "best_epoch_098", "last": "last_last"}
MODE_TO_CENTERS = {
    "all_groundtruth_centers": "all_gt_centers",
    "matched_groundtruth_centers": "true_centers",
    "predicted_centers": "predicted_centers",
}
MODE_TO_RELEASED = {
    "all_groundtruth_centers": "regression_performance_on_all_groundtruth_centers",
    "matched_groundtruth_centers": "regression_performance_on_matched_groundtruth_centers",
    "predicted_centers": "regression_performance_on_predicted_centers",
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


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return rel(value)
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_head() -> str:
    command = [
        "git",
        "-c",
        f"safe.directory={REPO.as_posix()}",
        "rev-parse",
        "HEAD",
    ]
    return subprocess.check_output(command, cwd=REPO, text=True).strip()


def manifest_sha256(path: Path) -> str | None:
    wanted = rel(path)
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["path"] == wanted:
                return row["sha256"]
    return None


def remap_legacy_targets(cfg):
    container = OmegaConf.to_container(cfg, resolve=False)

    def walk(node):
        if isinstance(node, dict):
            if node.get("_target_") == "dare3d.data.deletme_datamodule.DeletmeDataModule":
                node["_target_"] = "dare3d.data.dare_datamodule.DareDataModule"
            for item in node.values():
                walk(item)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(container)
    result = OmegaConf.create(container)
    OmegaConf.set_struct(result, False)
    return result


def frozen_config(checkpoint: Path):
    cfg = remap_legacy_targets(OmegaConf.load(CONFIG))
    cfg.ckpt_path = str(checkpoint)
    cfg.device = "cuda"
    test = cfg.data.test_data
    test.im_folder = str(INPUT_DIR)
    test.label_folder = str(LABEL_DIR)
    test.scale_file = str(OUTPUT_ROOT / "__released_scales_json_is_missing__.json")
    test.load_labels = True
    test.training = False
    return cfg


def import_origins() -> dict[str, str]:
    functions = {
        "CenterList": CenterList,
        "evaluate_center_pair": evaluate_center_pair,
        "regression_inference": regression_inference,
        "load_net_state_dict": load_net_state_dict,
    }
    origins = {
        name: Path(inspect.getsourcefile(function)).resolve()
        for name, function in functions.items()
    }
    escaped = {name: str(path) for name, path in origins.items() if not path.is_relative_to(REPO)}
    if escaped:
        raise RuntimeError(f"dare3d imports escaped current checkout: {escaped}")
    return {name: rel(path) for name, path in origins.items()}


def load_frozen_info() -> tuple[list[dict], dict]:
    true_rows = {}
    pred_rows = {}
    matches = []
    with CENTER_TABLE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            index = int(row["index"])
            center = [float(row[key]) for key in ("t", "x", "y", "z")]
            if row["kind"] == "true":
                true_rows[index] = center
                if row["matched_index"]:
                    matches.append((index, int(row["matched_index"])))
            elif row["kind"] == "pred":
                pred_rows[index] = center
            else:
                raise RuntimeError(f"Unexpected object kind: {row['kind']}")

    true_centers = [true_rows[index] for index in range(len(true_rows))]
    pred_centers = [pred_rows[index] for index in range(len(pred_rows))]
    matches.sort()
    if len(true_centers) != 145 or len(pred_centers) != 125 or len(matches) != 124:
        raise RuntimeError(
            f"Frozen center counts changed: true={len(true_centers)}, "
            f"pred={len(pred_centers)}, matches={len(matches)}"
        )
    if len({a for a, _ in matches}) != 124 or len({b for _, b in matches}) != 124:
        raise RuntimeError("Frozen matching is not one-to-one")
    info = [
        {
            "matched_items": matches,
            "pred_ccs_stats": {"centroids": np.asarray(pred_centers, dtype=float)},
            "true_ccs_stats": {"centroids": np.asarray(true_centers, dtype=float)},
        }
    ]
    summary = {
        "true_components": len(true_centers),
        "predicted_components": len(pred_centers),
        "matched_components": len(matches),
        "source": rel(CENTER_TABLE),
    }
    return info, summary


def save_predictions(path: Path, predictions: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        centers=np.asarray([item["center"] for item in predictions]),
        lengths=np.asarray([item["length"] for item in predictions]),
        quaternions=np.asarray([item["rotation"] for item in predictions]),
    )


def evaluate_modes(center_list: CenterList, predictions: dict[str, list[dict]]):
    center_list.compute_real_rot_len_values(DATASET)
    pairs = {
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
    stats = {}
    error_rows = []
    effective_counts = {}
    for mode, center_pairs in pairs.items():
        mode_stats, distance, time_distance, angle, length = evaluate_center_pair(center_pairs)
        stats[MODE_TO_RELEASED[mode]] = mode_stats
        effective_counts[mode] = len(angle)
        for index, values in enumerate(zip(distance, time_distance, angle, length)):
            error_rows.append(
                {
                    "mode": mode,
                    "evaluated_index": index,
                    "distance_error_voxels": values[0],
                    "time_distance_error_frames": values[1],
                    "full_quaternion_angle_error_deg": values[2],
                    "length_error_voxels": values[3],
                }
            )
    return stats, effective_counts, error_rows


def write_errors(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "mode",
        "evaluated_index",
        "distance_error_voxels",
        "time_distance_error_frames",
        "full_quaternion_angle_error_deg",
        "length_error_voxels",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compare_stats(generated: dict, released: dict) -> dict:
    comparisons = {}
    all_differences = []
    for mode, released_key in MODE_TO_RELEASED.items():
        generated_stats = generated[released_key]
        released_stats = released[released_key]
        metrics = {}
        for key in (
            "n",
            "mean_angle_error",
            "std_angle_error",
            "mean_length_error",
            "std_length_error",
            "mean_distance_error",
            "std_distance_error",
        ):
            generated_value = float(generated_stats[key])
            released_value = float(released_stats[key])
            difference = abs(generated_value - released_value)
            metrics[key] = {
                "generated": generated_value,
                "released": released_value,
                "absolute_difference": difference,
            }
            all_differences.append(difference)
        comparisons[mode] = metrics
    maximum = max(all_differences)
    return {
        "by_mode": comparisons,
        "maximum_absolute_difference": maximum,
        "all_values_within_1e_minus_6": maximum <= 1e-6,
        "all_values_within_1e_minus_5": maximum <= 1e-5,
        "all_values_within_1e_minus_3": maximum <= 1e-3,
    }


def execute(checkpoint_name: str, run_dir: Path) -> dict:
    global DATASET
    checkpoint = CHECKPOINTS[checkpoint_name]
    cfg = frozen_config(checkpoint)
    info, center_summary = load_frozen_info()
    released = json.loads(PROVENANCE.read_text(encoding="utf-8"))["released_stats"][
        "nuclei_gastruloid"
    ]["regression_results"]

    print(f"Loading dataset from {INPUT_DIR}")
    DATASET = hydra.utils.instantiate(cfg.data.test_data)
    DATASET.init(preprocess=False)
    original_shape = [list(movie.shape) for movie in DATASET.movies_im]
    DATASET.pad_images()
    DATASET._normalize(DATASET.renorm)
    padded_shape = [list(movie.shape) for movie in DATASET.movies_im]

    print(f"Loading regression checkpoint {checkpoint}")
    module = hydra.utils.instantiate(cfg.model)
    loaded_keys = load_net_state_dict(module.net, str(checkpoint), stage="regression")
    net = module.net.to("cuda")
    net.eval()

    center_list = CenterList(0, info)
    predictions = {}
    inference_seconds = {}
    for mode, attribute in MODE_TO_CENTERS.items():
        centers = getattr(center_list, attribute)
        print(f"Running {mode}: {len(centers)} centers")
        started = time.perf_counter()
        predictions[mode] = regression_inference(DATASET, net, centers, "cuda", output_dir=None)
        inference_seconds[mode] = time.perf_counter() - started
        save_predictions(run_dir / f"{mode}_raw_predictions.npz", predictions[mode])

    generated_stats, effective_counts, error_rows = evaluate_modes(center_list, predictions)
    write_errors(run_dir / "event_errors.csv", error_rows)
    comparison = compare_stats(generated_stats, released)
    parameter_count = sum(parameter.numel() for parameter in net.parameters())

    return {
        "status": "complete",
        "phase": "regression_checkpoint_inference_on_frozen_released_centers",
        "source_commit": SOURCE_COMMIT,
        "checkpoint_selection": checkpoint_name,
        "inputs": {
            "config": rel(CONFIG),
            "checkpoint": rel(checkpoint),
            "checkpoint_sha256_from_verified_manifest": manifest_sha256(checkpoint),
            "image": rel(INPUT_TIFF),
            "image_sha256_from_verified_manifest": manifest_sha256(INPUT_TIFF),
            "label_dir": rel(LABEL_DIR),
            "center_table": rel(CENTER_TABLE),
        },
        "frozen_protocol": {
            "center_source": "released nuclei probability replay under current evaluator",
            "center_counts": center_summary,
            "checkpoint_candidates_predeclared": ["best", "last"],
            "dataset_init_preprocess": False,
            "padding_then_normalization": True,
            "normalization": str(DATASET.renorm),
            "spatial_resampling_during_evaluation": False,
            "scale_file_present": False,
            "scale_note": "The current regression evaluation path does not resize when init(preprocess=False); it operates in raw anisotropic voxel coordinates.",
            "display_tiffs_written": False,
        },
        "dataset": {
            "movie_names": list(DATASET.movie_names),
            "original_internal_shapes_txyz": original_shape,
            "padded_internal_shapes_txyz": padded_shape,
            "effective_evaluated_counts": effective_counts,
        },
        "model": {
            "class": f"{type(net).__module__}.{type(net).__name__}",
            "parameter_count": parameter_count,
            "loaded_network_key_count": len(loaded_keys),
            "strict_checkpoint_load": True,
        },
        "generated_regression_results": generated_stats,
        "released_regression_results": released,
        "aggregate_comparison": comparison,
        "inference_seconds_by_mode": inference_seconds,
        "outputs": {
            "raw_prediction_npz": [
                rel(run_dir / f"{mode}_raw_predictions.npz") for mode in MODE_TO_CENTERS
            ],
            "event_errors_csv": rel(run_dir / "event_errors.csv"),
            "run_log": rel(run_dir / "run.log"),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
            "import_origins": import_origins(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=sorted(CHECKPOINTS), required=True)
    args = parser.parse_args()

    if git_head() != SOURCE_COMMIT:
        raise RuntimeError(f"Source commit changed; expected {SOURCE_COMMIT}, found {git_head()}")
    for required in (CONFIG, INPUT_TIFF, LABEL_DIR, CENTER_TABLE, PROVENANCE, MANIFEST):
        if not required.exists():
            raise FileNotFoundError(required)
    checkpoint = CHECKPOINTS[args.checkpoint]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen compatibility run")

    run_dir = OUTPUT_ROOT / RUN_NAMES[args.checkpoint]
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    log_path = run_dir / "run.log"
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    with log_path.open("w", encoding="utf-8") as log_handle:
        stdout = Tee(sys.__stdout__, log_handle)
        stderr = Tee(sys.__stderr__, log_handle)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            print(f"Started UTC: {started_utc}")
            print(f"Arguments: {vars(args)}")
            try:
                result = execute(args.checkpoint, run_dir)
                result["started_utc"] = started_utc
                result["finished_utc"] = datetime.now(timezone.utc).isoformat()
                result["elapsed_seconds"] = time.perf_counter() - started
                result["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
                result["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
                write_json(result_path, result)
                print(json.dumps({
                    "status": result["status"],
                    "checkpoint": args.checkpoint,
                    "maximum_absolute_difference": result["aggregate_comparison"]["maximum_absolute_difference"],
                    "within_1e-5": result["aggregate_comparison"]["all_values_within_1e_minus_5"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "result_json": rel(result_path),
                }, indent=2))
            except Exception as exc:
                failure = {
                    "status": "failed",
                    "source_commit": SOURCE_COMMIT,
                    "checkpoint_selection": args.checkpoint,
                    "started_utc": started_utc,
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": time.perf_counter() - started,
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "traceback": traceback.format_exc(),
                    "run_log": rel(log_path),
                }
                write_json(result_path, failure)
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
