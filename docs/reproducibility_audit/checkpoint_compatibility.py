"""Non-mutating current-code compatibility runs for released nuclei checkpoints.

Outputs are confined to docs/reproducibility_audit/checkpoint_runs/.  The
"log_reconstructed" scale is frozen from the archived training log's movie2
resize (362,305,180 -> 330,278,164), not selected from inference agreement.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path

import hydra
import numpy as np
import tifffile
import torch
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dare3d.metrics.infer_measure import infer_and_evaluate_segmentation
from dare3d.models.finetune import load_net_state_dict

DATA = REPO / "DARE3d_data_190326"
MODEL_DIR = DATA / "Gastruloid_241025/weights/segmentation3d_exp10-b"
CONFIG = MODEL_DIR / ".hydra/config.yaml"
INPUT_DIR = DATA / "Gastruloid_241025/test_input"
INPUT_TIFF = INPUT_DIR / "movie2.tif"
LABEL_DIR = DATA / "Gastruloid_241025/trainingset/movie2/label"
ARCHIVED_RUN = MODEL_DIR / "runs/01-01"
ARCHIVED_PROBABILITY = ARCHIVED_RUN / "movie2.tif"
ARCHIVED_STATS = ARCHIVED_RUN / "stats.csv"
OUTPUT_ROOT = HERE / "checkpoint_runs/nuclei_segmentation"

CHECKPOINTS = {
    "best": MODEL_DIR / "checkpoints/epoch_067.ckpt",
    "last": MODEL_DIR / "checkpoints/last.ckpt",
}
SCALE_MODES = {
    "log_reconstructed": {
        "default_scale": [0.912, 0.912, 0.912],
        "expected_target_txyz": [10, 330, 278, 164],
        "classification": (
            "diagnostic fixed from archived pre-output training-log geometry; "
            "original scales.json provenance remains missing"
        ),
    },
    "bundle_default": {
        "default_scale": [0.621, 0.621, 2.0],
        "expected_target_txyz": [10, 224, 189, 360],
        "classification": (
            "current documented missing-scale fallback from frozen config; "
            "contradicts archived training-log geometry"
        ),
    },
}

THRESHOLD = 0.55
MIN_WEIGHTED_PROBABILITY = 0.15
DISTANCE_MODE = "iou"
DISTANCE_THRESHOLD = 0.000001
ITERATION_METHOD = "movie"
OVERLAP = 0.5

OmegaConf.register_new_resolver("eval", eval, replace=True)


def rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def serial(value):
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=serial) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_manifest_hash(path: Path) -> str | None:
    manifest = HERE / "evidence/data_manifest_sha256.csv"
    wanted = rel(path)
    import csv

    with manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["path"] == wanted:
                return row["sha256"]
    return None


def import_origins() -> dict:
    paths = {
        "infer_and_evaluate_segmentation": Path(
            inspect.getsourcefile(infer_and_evaluate_segmentation)
        ).resolve(),
        "load_net_state_dict": Path(
            inspect.getsourcefile(load_net_state_dict)
        ).resolve(),
    }
    escaped = {name: path for name, path in paths.items() if not path.is_relative_to(REPO)}
    if escaped:
        raise RuntimeError(f"dare3d imports escaped checkout: {escaped}")
    return {name: rel(path) for name, path in paths.items()}


def remap_legacy_targets(cfg):
    container = OmegaConf.to_container(cfg, resolve=False)

    def walk(node):
        if isinstance(node, dict):
            if node.get("_target_") == "dare3d.data.deletme_datamodule.DeletmeDataModule":
                node["_target_"] = "dare3d.data.dare_datamodule.DareDataModule"
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(container)
    result = OmegaConf.create(container)
    OmegaConf.set_struct(result, False)
    return result


def load_frozen_config(checkpoint: Path, scale_mode: str):
    cfg = remap_legacy_targets(OmegaConf.load(CONFIG))
    cfg.ckpt_path = str(checkpoint)
    cfg.device = "cuda"
    test = cfg.data.test_data
    test.im_folder = str(INPUT_DIR)
    test.label_folder = str(LABEL_DIR)
    test.sparse_folder = None
    test.scale_file = str(OUTPUT_ROOT / "__released_scales_json_is_missing__.json")
    test.default_scale = SCALE_MODES[scale_mode]["default_scale"]
    test.target_scale = 1.0
    test.load_labels = True
    return cfg


def build_model(cfg):
    model = hydra.utils.instantiate(cfg.model)
    loaded_keys = load_net_state_dict(
        model.net, cfg.ckpt_path, stage="segmentation"
    )
    return model, loaded_keys


def model_summary(model) -> dict:
    parameters = list(model.net.parameters())
    return {
        "class": f"{type(model.net).__module__}.{type(model.net).__name__}",
        "parameter_count": sum(item.numel() for item in parameters),
        "trainable_parameter_count": sum(
            item.numel() for item in parameters if item.requires_grad
        ),
        "parameter_bytes": sum(item.numel() * item.element_size() for item in parameters),
        "state_tensor_count": len(model.net.state_dict()),
    }


def base_record(args, run_dir: Path) -> dict:
    checkpoint = CHECKPOINTS[args.checkpoint]
    raw = tifffile.memmap(INPUT_TIFF)
    internal = [raw.shape[0], raw.shape[3], raw.shape[2], raw.shape[1]]
    scale = np.asarray(SCALE_MODES[args.scale_mode]["default_scale"])
    target = [internal[0], *list((np.asarray(internal[1:]) * scale).astype(int))]
    return {
        "status": "initializing",
        "scientific_classification": SCALE_MODES[args.scale_mode]["classification"],
        "source_commit": "fe2b14d732359f2bdaf8b197574ad818899ce123",
        "arguments": vars(args),
        "inputs": {
            "config": rel(CONFIG),
            "checkpoint": rel(checkpoint),
            "checkpoint_sha256_from_verified_manifest": checkpoint_manifest_hash(checkpoint),
            "input_tiff": rel(INPUT_TIFF),
            "input_sha256": sha256(INPUT_TIFF),
            "label_dir": rel(LABEL_DIR),
            "archived_probability": rel(ARCHIVED_PROBABILITY),
            "archived_probability_sha256": sha256(ARCHIVED_PROBABILITY),
            "archived_stats": rel(ARCHIVED_STATS),
        },
        "frozen_settings": {
            "threshold": THRESHOLD,
            "min_weighted_probability": MIN_WEIGHTED_PROBABILITY,
            "distance_mode": DISTANCE_MODE,
            "distance_threshold": DISTANCE_THRESHOLD,
            "iteration_method": ITERATION_METHOD,
            "overlap": OVERLAP,
            "crop_size": 128,
            "inference_batch_size": args.batch_size,
            "default_scale_xyz": SCALE_MODES[args.scale_mode]["default_scale"],
            "scale_file": "absent; deliberate nonexistent path triggers frozen default",
            "raw_disk_shape_tzyx": list(raw.shape),
            "raw_internal_shape_txyz": internal,
            "computed_target_shape_txyz": target,
            "expected_target_shape_txyz": SCALE_MODES[args.scale_mode][
                "expected_target_txyz"
            ],
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "imports": import_origins(),
            "run_directory": rel(run_dir),
        },
    }


class Tee:
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


def released_segmentation_stats() -> dict:
    return json.loads(ARCHIVED_STATS.read_text(encoding="utf-8"))[
        "segmentation_results"
    ]


def compare_probabilities(generated_path: Path) -> dict:
    generated = tifffile.memmap(generated_path)
    archived = tifffile.memmap(ARCHIVED_PROBABILITY)
    result = {
        "generated_path": rel(generated_path),
        "generated_sha256": sha256(generated_path),
        "generated_bytes": generated_path.stat().st_size,
        "generated_shape": list(generated.shape),
        "generated_dtype": str(generated.dtype),
        "archived_shape": list(archived.shape),
        "archived_dtype": str(archived.dtype),
        "file_sha256_exact_match": sha256(generated_path) == sha256(ARCHIVED_PROBABILITY),
    }
    if generated.shape != archived.shape:
        result["shape_match"] = False
        return result

    count = 0
    equal = 0
    binary_disagreement = 0
    sum_abs = 0.0
    sum_sq = 0.0
    max_abs = 0.0
    sx = sy = sxx = syy = sxy = 0.0
    frame_rows = []
    for frame_index in range(generated.shape[0]):
        x = np.asarray(generated[frame_index], dtype=np.float32)
        y = np.asarray(archived[frame_index], dtype=np.float32)
        delta = x - y
        absolute = np.abs(delta)
        n = x.size
        frame_rows.append(
            {
                "frame": frame_index,
                "mae": float(np.mean(absolute, dtype=np.float64)),
                "rmse": float(
                    math.sqrt(np.mean(np.square(delta), dtype=np.float64))
                ),
                "max_abs": float(np.max(absolute)),
                "exact_fraction": float(np.count_nonzero(x == y) / n),
                "binary_disagreement_at_0_55": int(
                    np.count_nonzero((x >= THRESHOLD) != (y >= THRESHOLD))
                ),
            }
        )
        count += n
        equal += int(np.count_nonzero(x == y))
        binary_disagreement += int(
            np.count_nonzero((x >= THRESHOLD) != (y >= THRESHOLD))
        )
        sum_abs += float(np.sum(absolute, dtype=np.float64))
        sum_sq += float(np.sum(np.square(delta), dtype=np.float64))
        max_abs = max(max_abs, float(np.max(absolute)))
        sx += float(np.sum(x, dtype=np.float64))
        sy += float(np.sum(y, dtype=np.float64))
        sxx += float(np.sum(np.square(x), dtype=np.float64))
        syy += float(np.sum(np.square(y), dtype=np.float64))
        sxy += float(np.sum(x * y, dtype=np.float64))
    covariance = sxy - sx * sy / count
    variance_x = sxx - sx * sx / count
    variance_y = syy - sy * sy / count
    correlation = covariance / math.sqrt(max(variance_x * variance_y, 1e-300))
    result.update(
        {
            "shape_match": True,
            "array_exact_match": equal == count,
            "value_count": count,
            "exact_value_fraction": equal / count,
            "mae": sum_abs / count,
            "rmse": math.sqrt(sum_sq / count),
            "max_abs": max_abs,
            "pearson_correlation": correlation,
            "binary_disagreement_count_at_0_55": binary_disagreement,
            "binary_disagreement_fraction_at_0_55": binary_disagreement / count,
            "per_frame": frame_rows,
        }
    )
    return result


def extract_info(info) -> list[dict]:
    rows = []
    for movie in info:
        rows.append(
            {
                "true_components": len(movie["true_ccs_stats"]["centroids"]),
                "predicted_components": len(movie["pred_ccs_stats"]["centroids"]),
                "matched_components": len(movie["matched_items"]),
            }
        )
    return rows


def preflight(args, record: dict) -> dict:
    cfg = load_frozen_config(CHECKPOINTS[args.checkpoint], args.scale_mode)
    resolved_test = OmegaConf.to_container(cfg.data.test_data, resolve=True)
    started = time.perf_counter()
    model, loaded_keys = build_model(cfg)
    record.update(
        {
            "status": "complete",
            "phase": "preflight",
            "strict_checkpoint_load": True,
            "loaded_network_key_count": len(loaded_keys),
            "first_loaded_keys": loaded_keys[:10],
            "model": model_summary(model),
            "resolved_test_dataset": resolved_test,
            "elapsed_seconds": time.perf_counter() - started,
            "assertions": {
                "computed_target_matches_frozen_expectation": record[
                    "frozen_settings"
                ]["computed_target_shape_txyz"]
                == record["frozen_settings"]["expected_target_shape_txyz"],
                "checkpoint_manifest_hash_present": record["inputs"][
                    "checkpoint_sha256_from_verified_manifest"
                ]
                is not None,
                "current_checkout_imports": all(
                    not value.startswith("../")
                    for value in record["runtime"]["imports"].values()
                ),
            },
        }
    )
    record["all_assertions_pass"] = all(record["assertions"].values())
    if not record["all_assertions_pass"]:
        raise AssertionError(record["assertions"])
    del model
    return record


def infer(args, record: dict, run_dir: Path) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen compatibility run")
    cfg = load_frozen_config(CHECKPOINTS[args.checkpoint], args.scale_mode)
    dataset = hydra.utils.instantiate(cfg.data.test_data)
    dataset.init(preprocess=False)
    dataset.make_masks()
    actual_target = list(
        dataset._compute_target_shape(
            dataset.original_movies_shape[0], movie_name=dataset.movie_names[0]
        )
    )
    expected_target = record["frozen_settings"]["expected_target_shape_txyz"]
    if actual_target != expected_target:
        raise AssertionError(
            f"Dataset target shape {actual_target} != frozen {expected_target}"
        )

    model, loaded_keys = build_model(cfg)
    device = torch.device("cuda:0")
    net = model.net.to(device)
    net.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    stats, info = infer_and_evaluate_segmentation(
        dataset=dataset,
        model=net,
        device=device,
        crop_size=int(cfg.crop_size),
        batch_size=args.batch_size,
        multithread=False,
        threshold=THRESHOLD,
        iteration_method=ITERATION_METHOD,
        distance_mode=DISTANCE_MODE,
        distance_threshold=DISTANCE_THRESHOLD,
        min_weighted_prob=MIN_WEIGHTED_PROBABILITY,
        output_dir=str(run_dir),
    )
    inference_seconds = time.perf_counter() - started
    generated = run_dir / "movie2.tif"
    if not generated.exists():
        raise RuntimeError(f"Inference returned without probability TIFF: {generated}")

    comparison = compare_probabilities(generated)
    released = released_segmentation_stats()
    current = {key: serial(value) if not isinstance(value, (int, float)) else value for key, value in stats.items()}
    metric_keys = ("tp", "fp", "fn", "precision", "recall", "fmeasure")
    record.update(
        {
            "status": "complete",
            "phase": "full_inference_and_evaluation",
            "strict_checkpoint_load": True,
            "loaded_network_key_count": len(loaded_keys),
            "model": model_summary(model),
            "dataset": {
                "movie_names": list(dataset.movie_names),
                "original_shapes_txyz": [
                    list(shape) for shape in dataset.original_movies_shape
                ],
                "actual_target_shape_txyz": actual_target,
                "groundtruth_mask_shapes_txyz": [
                    list(mask.shape) for mask in dataset.movies_masks
                ],
            },
            "current_evaluator_stats": current,
            "current_evaluator_components": extract_info(info),
            "released_evaluator_stats": released,
            "released_metric_exact_agreement": {
                key: (
                    current[key] == released[key]
                    if key in ("tp", "fp", "fn")
                    else math.isclose(
                        float(current[key]), float(released[key]), rel_tol=0.0, abs_tol=1e-15
                    )
                )
                for key in metric_keys
            },
            "probability_comparison": comparison,
            "elapsed_inference_and_evaluation_seconds": inference_seconds,
            "gpu": {
                "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
            },
        }
    )
    record["released_all_metrics_exact"] = all(
        record["released_metric_exact_agreement"].values()
    )
    return record


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=sorted(CHECKPOINTS), required=True)
    parser.add_argument(
        "--scale-mode", choices=sorted(SCALE_MODES), default="log_reconstructed"
    )
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


def main() -> int:
    args = parse_args()
    checkpoint_stem = CHECKPOINTS[args.checkpoint].stem
    run_dir = (
        OUTPUT_ROOT
        / args.scale_mode
        / f"{args.checkpoint}_{checkpoint_stem}"
        / f"batch_{args.batch_size}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / (
        "preflight.json" if args.preflight_only else "result.json"
    )
    log_path = run_dir / (
        "preflight.log" if args.preflight_only else "run.log"
    )
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        print(
            f"Refusing to overwrite existing {existing.get('status')} result: "
            f"{result_path}"
        )
        return 0 if existing.get("status") == "complete" else 3

    record = base_record(args, run_dir)
    record["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(result_path, record)
    started = time.perf_counter()
    exit_code = 0
    with log_path.open("x", encoding="utf-8") as log_stream:
        tee_out = Tee(sys.__stdout__, log_stream)
        tee_err = Tee(sys.__stderr__, log_stream)
        try:
            with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                print(json.dumps(record, indent=2, default=serial), flush=True)
                if args.preflight_only:
                    record = preflight(args, record)
                else:
                    record = infer(args, record, run_dir)
                record["finished_utc"] = time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                )
                record["total_elapsed_seconds"] = time.perf_counter() - started
                write_json(result_path, record)
                print(json.dumps(record, indent=2, default=serial), flush=True)
        except BaseException as exc:
            exit_code = 1
            record.update(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "finished_utc": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                    "total_elapsed_seconds": time.perf_counter() - started,
                }
            )
            if torch.cuda.is_available():
                try:
                    record["gpu_at_failure"] = {
                        "memory_allocated_bytes": torch.cuda.memory_allocated(0),
                        "memory_reserved_bytes": torch.cuda.memory_reserved(0),
                        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(0),
                        "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(0),
                    }
                except Exception:
                    pass
            write_json(result_path, record)
            traceback.print_exc(file=tee_err)
            print(f"Failure evidence saved to {result_path}", file=tee_err)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
