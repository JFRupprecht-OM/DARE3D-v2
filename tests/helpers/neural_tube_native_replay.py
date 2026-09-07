"""One-shot acceptance replay for the promoted neural segmenter (not training).

Run directly with --output-dir under the checkout's .pytest_tmp, or through the
opt-in slow pytest contract. Existing output directories are never overwritten.
No Napari GUI, regression model, or historical eval output directory is opened.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time


ROOT = Path(__file__).resolve().parents[2]
# Locked before the implementation replay; never auto-adjust to observed output.
LIMITS = {
    "exact_voxel_fraction_min": 0.998,
    "mean_absolute_difference_max": 1e-6,
    "maximum_absolute_difference_max": 0.02,
    "foreground_correlation_min": 0.9999,
}
CHECKPOINT_SHA256 = "8261c70f41c7f0f5184559e98f3c69ad336e5e46edba5bb57e5edbf9b0568f7d"
CONFIG_SHA256 = "0c6aebaafad7cdc43254ec06261679f8f6bfee591bf842f2002b9768d5966761"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    def encode(item):
        if isinstance(item, Path):
            return str(item)
        if hasattr(item, "tolist"):
            return item.tolist()
        raise TypeError(type(item).__name__)

    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, default=encode, allow_nan=False)


def large_stack(function):
    """Windows/Numba connected components need more than the main thread stack."""
    results, errors = [], []

    def run():
        try:
            results.append(function())
        except BaseException:
            errors.append(sys.exc_info())

    previous = threading.stack_size(64 * 1024 * 1024)
    try:
        worker = threading.Thread(target=run)
        worker.start()
        worker.join()
    finally:
        threading.stack_size(previous)
    if errors:
        raise errors[0][1].with_traceback(errors[0][2])
    return results[0]


def replay(output):
    import numpy as np
    import tifffile
    import torch
    from omegaconf import OmegaConf
    from scipy.optimize import linear_sum_assignment
    from unittest.mock import patch

    sys.path.insert(0, str(ROOT))
    from dare3d.metrics import object_level
    from napari_dare3d import _api as api
    from napari_dare3d._release_models import (
        release_model_selection,
        release_segmentation_overlap,
        release_segmentation_scale_mode,
    )

    if not Path(api.__file__).resolve().is_relative_to(ROOT):
        raise RuntimeError("Inference did not import this checkout")
    if not torch.cuda.is_available():
        raise RuntimeError("The explicitly requested acceptance replay requires CUDA")
    torch.set_num_threads(4)
    data = ROOT / "DARE3d_data_190326/Neural_tube_160226"
    legacy = data / "weights/segmentation3d_new_set_og/runs/12-01-26"
    model_dir, checkpoint = release_model_selection(
        "neural_tube", "segmentation", ROOT / "DARE3dv2_Zenodo_040926"
    )
    image_path = data / "trainingset/movie2/im/movie_I2.tif"
    label_path = data / "trainingset/movie2/label/movie_I2.tif"
    archive_path = legacy / "movie_I2.tif"
    scale_path = ROOT / "data/3D/scales.json"
    expected = {
        checkpoint: CHECKPOINT_SHA256,
        legacy / "checkpoints/epoch_057.ckpt": CHECKPOINT_SHA256,
        model_dir / ".hydra/config.yaml": CONFIG_SHA256,
        legacy / ".hydra/config.yaml": CONFIG_SHA256,
        image_path: "b669093406d537164f7487e4740a9e1f19d959031cb0ff86991785d7a6154062",
        label_path: "99ad277d1ea28334b02d6824deddb9b76eeff0e0b449a2656e0bbb255546b4f5",
        archive_path: "8a54807c1ee1748e55038939aead7750ec696d9c08e9f44357076e6d4855546f",
        scale_path: "e206eb6502a5a5907bd919e647be24fc66dce6ea7a6fc1a357856b09fc4021fe",
    }
    before = {str(path): sha256(path) for path in expected}
    if before != {str(path): value for path, value in expected.items()}:
        raise RuntimeError("Release/input/archive identity differs from the locked reference")
    mode = release_segmentation_scale_mode("neural_tube", checkpoint, checkpoint.parent)
    overlap = release_segmentation_overlap("neural_tube", checkpoint, checkpoint.parent)
    if mode != "native" or overlap != 0.5:
        raise RuntimeError("Promoted neural compatibility metadata is not native / 0.5")
    git = ["git", "--no-optional-locks", "-c", "safe.directory=" + ROOT.as_posix()]
    source_files = [
        "napari_dare3d/_api.py", "napari_dare3d/_release_models.py",
        "napari_dare3d/_widget.py", "dare3d/metrics/inference.py",
        "dare3d/metrics/object_level.py",
        "dare3d/data/components/abstract_celldataset.py",
        "dare3d/data/components/seg_3dataset.py",
        "tests/helpers/neural_tube_native_replay.py",
    ]
    record = {
        "limits": LIMITS,
        "asset_hashes_before": before,
        "source_sha256": {p: sha256(ROOT / p) for p in source_files},
        "git_head": subprocess.check_output(git + ["rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": sys.executable,
        "versions": {p: importlib.metadata.version(p) for p in [
            "torch", "monai", "numpy", "scipy", "scikit-image", "tifffile"
        ]},
        "gpu": torch.cuda.get_device_name(),
        "segmentation_scale_mode": mode,
        "overlap": overlap,
        "batch_size": 4,
        "napari_threshold": 0.5,
        "napari_weighted_cutoff": 0.1,
        "regression_disabled": True,
    }
    save(output / "preflight.json", record)
    print("BEGIN one full movie_I2 replay through the modified Napari API", flush=True)
    stack = tifffile.imread(image_path)
    if stack.shape != (21, 10, 1024, 1024):
        raise RuntimeError("Unexpected movie_I2 shape")
    model_loads, inference_calls, dataset_records = [], [], []
    real_build_model = api._build_model
    real_build_dataset = api._build_dataset
    real_inference = api.segmentation_inference
    probability_path = output / "native_movie_I2.tif"

    def guarded_model(cfg, device, *, stage):
        if stage != "segmentation" or Path(cfg.ckpt_path).resolve() != checkpoint.resolve():
            raise RuntimeError("An out-of-scope model was requested")
        model_loads.append(str(cfg.ckpt_path))
        model = real_build_model(cfg, device, stage=stage)
        if model.net.training:
            raise RuntimeError("Network is not in evaluation mode")
        return model

    def recorded_dataset(cfg, stage, *args, **kwargs):
        dataset = real_build_dataset(cfg, stage, *args, **kwargs)
        td = OmegaConf.to_container(cfg.data.test_data, resolve=True)
        target = dataset._compute_target_shape(dataset.movies_im[0].shape, "movie_I2")
        padding = dataset.get_movie_padding(dataset.movies_im[0][:3])
        if tuple(target) != (21, 1024, 1024, 10):
            raise RuntimeError("Napari still resamples the neural input")
        if padding != ((0, 0), (0, 0), (0, 0), (59, 59)):
            raise RuntimeError("Unexpected native-grid patch padding")
        dataset_records.append({"test_data": td, "target_txyz": target, "padding": padding})
        save(output / "effective_dataset.json", dataset_records[-1])
        return dataset

    def recorded_inference(*args, **kwargs):
        inference_calls.append(1)
        predictions = real_inference(*args, **kwargs)
        # Record actual API output; no model prediction is substituted.
        tifffile.imwrite(probability_path, predictions[0], metadata={"axes": "TZYX"})
        return predictions

    started = time.perf_counter()
    with patch.object(api, "_build_model", guarded_model), patch.object(
        api, "_build_dataset", recorded_dataset
    ), patch.object(api, "segmentation_inference", recorded_inference):
        detections = api.infer_stack(
            stack, seg_checkpoint=str(checkpoint), reg_checkpoint=None,
            device="gpu", segmentation_scale_mode=mode, overlap=overlap,
            batch_size=4, threshold=0.5, min_weighted_prob=0.1,
            scale_file=str(scale_path), movie_name="movie_I2",
            progress_cb=lambda stage: print("NAPARI STAGE", stage, flush=True),
        )
    record["api_seconds"] = time.perf_counter() - started
    record["model_loads"] = model_loads
    record["inference_call_count"] = len(inference_calls)
    record["napari_count"] = len(detections)
    record["probability_sha256"] = sha256(probability_path)
    save(output / "napari_detections.json", detections)
    del stack
    gc.collect()
    torch.cuda.empty_cache()

    fresh = tifffile.memmap(probability_path)
    archived = tifffile.memmap(archive_path)
    rows = []
    for t in range(len(fresh)):
        a = fresh[t].astype(np.float32)
        b = archived[t].astype(np.float32)
        difference = np.abs(a - b)
        foreground = (a > 0.1) | (b > 0.1)
        correlation = (
            float(np.corrcoef(a[foreground], b[foreground])[0, 1])
            if np.count_nonzero(foreground) > 2 else None
        )
        rows.append({
            "t": t, "mean_abs": float(difference.mean()),
            "max_abs": float(difference.max()),
            "exact_voxels": int(np.count_nonzero(a == b)), "voxels": int(a.size),
            "half_threshold_disagreements": int(np.count_nonzero((a > 0.5) != (b > 0.5))),
            "foreground_correlation": correlation,
        })
    comparison = {
        "shape_tzyx": list(fresh.shape), "dtype": str(fresh.dtype),
        "first_two_frames_zero": not bool(np.any(fresh[:2])),
        "finite": all(bool(np.isfinite(fresh[t]).all()) for t in range(len(fresh))),
        "exact_voxel_fraction": sum(r["exact_voxels"] for r in rows) / fresh.size,
        "mean_abs": float(np.mean([r["mean_abs"] for r in rows])),
        "max_abs": max(r["max_abs"] for r in rows),
        "per_frame": rows,
    }
    record["probability_comparison"] = comparison
    save(output / "probability_comparison.json", comparison)
    print("Probability comparison complete; scoring the two cached maps once", flush=True)

    cfg = api._load_inference_cfg(
        str(model_dir), str(image_path.parent), "cpu", checkpoint_path=str(checkpoint),
        default_scale=[1, 1, 1], target_scale=1,
    )
    cfg.data.test_data.load_labels = True
    cfg.data.test_data.label_folder = str(label_path.parent)
    dataset = api._build_dataset(cfg, "segmentation")
    dataset.make_masks()
    truth = dataset.movies_masks[0]
    truth[:2] = 0
    eligible = sum(len(pairs) for pairs in dataset.movies_bipoints[0][2:])
    del dataset
    gc.collect()
    scores = {}
    for name, probability in [("archive", archived), ("fresh", fresh)]:
        metrics, info = object_level.evaluate_at_object_level(
            np.swapaxes(probability, -1, -3), truth, 0.55, 0.15,
            distance_mode="iou", distance_threshold=1e-6,
        )
        scores[name] = {
            "metrics": metrics,
            "centers_txyz": np.asarray(info["pred_ccs_stats"]["centroids"]),
            "matched_items": info["matched_items"],
        }
        print(name, metrics, "centers", len(scores[name]["centers_txyz"]), flush=True)
        del info
        gc.collect()
    save(output / "historical_scores.json", scores)
    p, q = scores["fresh"]["centers_txyz"], scores["archive"]["centers_txyz"]
    cost = np.linalg.norm((p[:, None] - q[None]) * np.array([100, 1, 1, 1]), axis=2)
    new_rows, archive_rows = linear_sum_assignment(cost)
    mapping = dict(zip(new_rows.tolist(), archive_rows.tolist()))
    new_pairs = {(int(gt), mapping[int(pred)]) for gt, pred in scores["fresh"]["matched_items"]}
    old_pairs = {tuple(map(int, pair)) for pair in scores["archive"]["matched_items"]}
    expected_metrics = {
        "tp": 114, "fp": 8, "fn": 8, "precision": 114 / 122,
        "recall": 114 / 122, "fmeasure": 114 / 122,
    }
    checks = {
        "one_promoted_model_and_one_inference": model_loads == [str(checkpoint)] and len(inference_calls) == 1,
        "shape_and_dtype": fresh.shape == archived.shape == (21, 10, 1024, 1024) and fresh.dtype == np.float16,
        "finite_and_initial_frames_zero": comparison["finite"] and comparison["first_two_frames_zero"],
        "exact_voxel_fraction": comparison["exact_voxel_fraction"] >= LIMITS["exact_voxel_fraction_min"],
        "mean_absolute_difference": comparison["mean_abs"] <= LIMITS["mean_absolute_difference_max"],
        "maximum_absolute_difference": comparison["max_abs"] <= LIMITS["maximum_absolute_difference_max"],
        "foreground_correlation": all(
            r["foreground_correlation"] is not None
            and r["foreground_correlation"] >= LIMITS["foreground_correlation_min"] for r in rows[2:]
        ),
        "historical_centers": len(p) == len(q) == 122,
        "historical_metrics": all(scores[name]["metrics"] == expected_metrics for name in scores),
        "matched_events": new_pairs == old_pairs and len(new_pairs) == 114,
        "napari_centers": len(detections) == 118,
        "eligible_annotations": eligible == 123,
    }
    record["centroid_max_abs_txyz"] = np.max(np.abs(p[new_rows] - q[archive_rows]), axis=0)
    record["historical_metrics"] = scores["fresh"]["metrics"]
    record["historical_centers"] = len(p)
    record["matched_event_pairs_identical"] = new_pairs == old_pairs
    record["asset_hashes_after"] = {str(path): sha256(path) for path in expected}
    checks["protected_asset_hashes"] = record["asset_hashes_after"] == before
    record["checks"] = checks
    record["passed"] = all(checks.values())
    save(output / "result.json", record)
    print("ACCEPTANCE", json.dumps(checks), flush=True)
    if not record["passed"]:
        raise RuntimeError("Locked acceptance failed; keep results and do not relax limits")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    temporary_root = (ROOT / ".pytest_tmp").resolve()
    if output == temporary_root or not output.is_relative_to(temporary_root):
        parser.error("--output-dir must be a NEW child beneath this checkout's .pytest_tmp")
    output.mkdir(parents=True, exist_ok=False)
    tempfile.tempdir = str(output)
    os.environ["PROJECT_ROOT"] = str(ROOT)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["NUMBA_NUM_THREADS"] = "4"
    for key, folder in [
        ("MPLCONFIGDIR", "mpl_cache"), ("NUMBA_CACHE_DIR", "numba_cache"),
        ("CUDA_CACHE_PATH", "cuda_cache"),
    ]:
        os.environ[key] = str(output / folder)
    large_stack(lambda: replay(output))


if __name__ == "__main__":
    main()
