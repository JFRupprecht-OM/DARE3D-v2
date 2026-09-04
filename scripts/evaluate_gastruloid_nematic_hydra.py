"""Validate and document the Hydra gastruloid nematic retraining candidate.

This evaluator is intentionally regression-only. It compares the new Hydra
checkpoint directly with the validated audit checkpoint on identical crops and
frozen center sets, then proves that protected legacy, audit, segmentation, and
Zenodo assets did not change.
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
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
AUDIT_CODE = REPO / "docs/reproducibility_audit"
for import_root in (REPO, AUDIT_CODE):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

import nematic_retraining_evaluation as audit_evaluation
import nematic_retraining_experiment as audit_experiment
import regression_checkpoint_compatibility as center_compatibility
import regression_preprocessing_fix_validation as current_evaluation
from dare3d.data.components.regression_geometry import LEGACY_RAW, TRAINING_CONSISTENT
from dare3d.metrics.infer_measure import CenterList
from dare3d.metrics.inference import regression_inference


MODEL_ROOT = (
    REPO
    / "DARE3d_data_190326/Gastruloid_241025/weights/"
    "regression3d_nematic_hydra_seed12345"
)
STAGING_ROOT = (
    REPO
    / "DARE3d_data_190326/Gastruloid_241025/training_splits/"
    "regression3d_nematic_seed12345"
)
REFERENCE = (
    REPO
    / "docs/reproducibility_audit/nematic_retraining/seed_12345/"
    "checkpoints/epoch_095.ckpt"
)
REFERENCE_SHA256 = "e3bc5a3ff81497d36606ee4ca57c83395eef5f02b0ffd3cf5f117d0bb1f51ec8"
TRAINING_COMMIT = "9bd5797772056ad9358a0d27d1886f2468aaba56"
EVALUATION_ROOT = MODEL_ROOT / "evaluation"
PREDICTION_ROOT = EVALUATION_ROOT / "predictions"
PROVENANCE_ROOT = MODEL_ROOT / "provenance"
PROTECTED_BEFORE = STAGING_ROOT / "protected_before.json"
BOOTSTRAP_SEED = 20260829
BOOTSTRAP_REPLICATES = 10_000
MODES = (
    "all_groundtruth_centers",
    "matched_groundtruth_centers",
    "predicted_centers",
)
MODE_TO_CENTERS = {
    "all_groundtruth_centers": "all_gt_centers",
    "matched_groundtruth_centers": "true_centers",
    "predicted_centers": "predicted_centers",
}
REFERENCE_CONTROLLED = {
    "mean_deg": 10.892897605895996,
    "p95_deg": 26.08814811706543,
    "length_mae_voxels": 1.502277135848999,
}
REFERENCE_CURRENT_MEANS = {
    "all_groundtruth_centers": 11.736369640423076,
    "matched_groundtruth_centers": 11.010686815585304,
    "predicted_centers": 19.80617943692261,
}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


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
        return jsonable(value.detach().cpu().numpy())
    if isinstance(value, Path):
        return rel(value)
    return value


def write_json(path: Path, value: Any) -> None:
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
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def describe(values) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return {"n": 0}
    std = float(np.std(finite, dtype=np.float32))
    return {
        "n": int(len(finite)),
        "mean": float(np.mean(finite, dtype=np.float32)),
        "median": float(np.median(finite)),
        "std_population": std,
        "sem": std / math.sqrt(len(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p25": float(np.percentile(finite, 25)),
        "p75": float(np.percentile(finite, 75)),
        "p95": float(np.percentile(finite, 95)),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
    }


def paired_bootstrap(reference, candidate) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise ValueError((reference.shape, candidate.shape))
    difference = candidate - reference
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0,
        len(difference),
        size=(BOOTSTRAP_REPLICATES, len(difference)),
    )
    distribution = np.mean(difference[indices], axis=1)
    return {
        "difference": "candidate minus reference; negative is improvement",
        "n": int(len(difference)),
        "mean_difference_deg": float(np.mean(difference)),
        "bootstrap_mean_difference_95_ci_deg": [
            float(np.percentile(distribution, 2.5)),
            float(np.percentile(distribution, 97.5)),
        ],
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
    }


def candidate_checkpoint() -> Path:
    checkpoints = sorted((MODEL_ROOT / "checkpoints").glob("epoch_*.ckpt"))
    if len(checkpoints) != 1:
        raise RuntimeError(
            "Expected exactly one top-ranked epoch checkpoint, found "
            f"{[path.name for path in checkpoints]}"
        )
    return checkpoints[0]


def checkpoint_record(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "path": rel(path),
        "sha256": sha256(path),
        "epoch": int(checkpoint.get("epoch", -1)),
        "global_step": int(checkpoint.get("global_step", -1)),
        "lightning_version": checkpoint.get("pytorch-lightning_version"),
    }


def controlled_evaluation(models) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    print("Initializing the validated 156-crop controlled holdout", flush=True)
    dataset = audit_experiment.make_dataset("test")
    dataset.init()
    if len(dataset.crops) != 156:
        raise AssertionError(f"Controlled crop count changed: {len(dataset.crops)}")

    rows = []
    for name, net in models.items():
        print(f"Controlled evaluation: {name}", flush=True)
        rows.extend(audit_evaluation.evaluate_controlled_model(name, net, dataset))
    del dataset
    gc.collect()

    summaries = {}
    for name in models:
        selected = [row for row in rows if row["model"] == name]
        summaries[name] = {
            "corrected_nematic_axis_error_deg": describe(
                row["corrected_nematic_axis_error_deg"] for row in selected
            ),
            "production_quaternion_error_deg": describe(
                row["production_quaternion_error_deg"] for row in selected
            ),
            "absolute_length_error_voxels": describe(
                row["absolute_length_error_voxels"] for row in selected
            ),
        }

    reference = sorted(
        (row for row in rows if row["model"] == "reference_epoch_095"),
        key=lambda row: row["sample_index"],
    )
    candidate = sorted(
        (row for row in rows if row["model"] == "candidate_hydra"),
        key=lambda row: row["sample_index"],
    )
    if [row["sample_index"] for row in reference] != [
        row["sample_index"] for row in candidate
    ]:
        raise AssertionError("Controlled sample ordering changed")
    paired = paired_bootstrap(
        [row["corrected_nematic_axis_error_deg"] for row in reference],
        [row["corrected_nematic_axis_error_deg"] for row in candidate],
    )
    result = {
        "dataset": (
            "validated audit movie2 holdout; 156 unique, identically ordered "
            "ground-truth crops"
        ),
        "by_model": summaries,
        "paired_candidate_vs_reference": paired,
    }
    write_csv(EVALUATION_ROOT / "controlled_events.csv", rows)
    return result, rows

def save_predictions(path: Path, predictions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "centers_raw": np.asarray(
            [prediction["center_raw"] for prediction in predictions]
        ),
        "centers_regression": np.asarray(
            [prediction["center_regression"] for prediction in predictions]
        ),
        "lengths_regression_voxels": np.asarray(
            [
                prediction["length_regression_voxels"]
                for prediction in predictions
            ]
        ),
        "quaternions_wxyz": np.asarray(
            [prediction["rotation"] for prediction in predictions]
        ),
    }
    for output_name, prediction_name in (
        ("axes_physical_xyz", "axis_physical_xyz"),
        ("lengths_physical_um", "length_physical_um"),
        ("endpoints_raw_xyz", "endpoints_raw_xyz"),
    ):
        if predictions and prediction_name in predictions[0]:
            arrays[output_name] = np.asarray(
                [prediction[prediction_name] for prediction in predictions]
            )
    np.savez_compressed(path, **arrays)


def make_frozen_center_dataset(mode: str):
    dataset = audit_experiment.make_dataset("test")
    dataset.init_inference(mode)
    return dataset


def frozen_center_evaluation(models) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_rows = []
    manifests = {}
    center_summary = None
    target_counts = {}

    for profile in (TRAINING_CONSISTENT, LEGACY_RAW):
        print(f"Initializing frozen centers: {profile}", flush=True)
        dataset = make_frozen_center_dataset(profile)
        info, center_summary = center_compatibility.load_frozen_info()
        center_list = CenterList(0, info)
        center_list.compute_real_rot_len_values(dataset)
        centers_by_mode = {
            mode: getattr(center_list, MODE_TO_CENTERS[mode])
            for mode in MODES
        }
        targets_by_mode = {
            "all_groundtruth_centers": center_list.real_rot_length,
            "matched_groundtruth_centers": center_list.real_rot_length_matched,
            "predicted_centers": center_list.real_rot_length_matched,
        }
        target_counts[profile] = {
            mode: sum(target is not None for target in targets_by_mode[mode])
            for mode in MODES
        }
        manifests[profile] = dataset.preprocessing_manifest()

        for model_name, net in models.items():
            for mode in MODES:
                centers = centers_by_mode[mode]
                print(
                    f"Frozen centers: {profile}, {model_name}, "
                    f"{mode}, N={len(centers)}",
                    flush=True,
                )
                predictions = regression_inference(
                    dataset,
                    net,
                    centers,
                    "cuda",
                    output_dir=None,
                )
                save_predictions(
                    PREDICTION_ROOT
                    / f"{profile}_{model_name}_{mode}.npz",
                    predictions,
                )
                all_rows.extend(
                    current_evaluation.event_rows(
                        "movie2_frozen_centers",
                        profile,
                        model_name,
                        mode,
                        targets_by_mode[mode],
                        predictions,
                    )
                )

        del dataset
        gc.collect()
        torch.cuda.empty_cache()

    write_csv(EVALUATION_ROOT / "frozen_center_events.csv", all_rows)
    return {
        "center_source": (
            "frozen detector centers and matches from the validated "
            "reproducibility audit; segmentation was not rerun"
        ),
        "center_summary": center_summary,
        "target_effective_counts": target_counts,
        "preprocessing_manifests": manifests,
        "summaries": current_evaluation.summarize_event_rows(all_rows),
    }, all_rows

def current_hash_tree(root_relative: str) -> list[dict[str, Any]]:
    root = REPO / root_relative
    return [
        {
            "path": rel(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def protected_asset_verification() -> dict[str, Any]:
    before = json.loads(PROTECTED_BEFORE.read_text(encoding="utf-8"))
    legacy_now = current_hash_tree(before["legacy"]["root"])
    audit_now = current_hash_tree(before["validated_audit"]["root"])
    segmentation_now = []
    for record in before["segmentation_selected_files"]:
        path = REPO / record["path"]
        segmentation_now.append(
            {
                "path": record["path"],
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    zenodo_now = {}
    zenodo_root = REPO / "DARE3dv2_Zenodo_040926"
    for path in sorted(item for item in zenodo_root.rglob("*") if item.is_file()):
        zenodo_now[rel(path)] = {
            "bytes": path.stat().st_size,
            "mtime": path.stat().st_mtime,
        }
    zenodo_before = {
        record["path"]: {
            "bytes": int(record["bytes"]),
            "mtime": datetime.fromisoformat(
                record["last_write_utc"].replace("Z", "+00:00")
            ).timestamp(),
        }
        for record in before["zenodo_tree_metadata"]
    }
    zenodo_unchanged = (
        zenodo_now.keys() == zenodo_before.keys()
        and all(
            zenodo_now[path]["bytes"] == zenodo_before[path]["bytes"]
            and abs(
                zenodo_now[path]["mtime"] - zenodo_before[path]["mtime"]
            )
            < 1e-5
            for path in zenodo_now
        )
    )
    assertions = {
        "candidate_was_absent_at_baseline": not before["candidate_existed"],
        "legacy_regression_folder_byte_identical": (
            legacy_now == before["legacy"]["files"]
        ),
        "validated_audit_run_byte_identical": (
            audit_now == before["validated_audit"]["files"]
        ),
        "selected_segmentation_files_byte_identical": (
            segmentation_now == before["segmentation_selected_files"]
        ),
        "zenodo_tree_paths_sizes_timestamps_identical": zenodo_unchanged,
    }
    PROVENANCE_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROTECTED_BEFORE, PROVENANCE_ROOT / "protected_before.json")
    result = {
        "baseline": rel(PROTECTED_BEFORE),
        "baseline_copy": rel(PROVENANCE_ROOT / "protected_before.json"),
        "assertions": assertions,
        "all_assertions_pass": all(assertions.values()),
    }
    write_json(PROVENANCE_ROOT / "protected_after_verification.json", result)
    return result


def git_output(*arguments: str) -> str:
    return subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={REPO.as_posix()}",
            *arguments,
        ],
        cwd=REPO,
        text=True,
    ).strip()


def native_artifact_verification(checkpoint: Path) -> dict[str, bool]:
    run_roots = sorted((MODEL_ROOT / "runs").glob("*"))
    return {
        "root_hydra_config_exists": (
            MODEL_ROOT / ".hydra/config.yaml"
        ).is_file(),
        "root_hydra_overrides_exists": (
            MODEL_ROOT / ".hydra/overrides.yaml"
        ).is_file(),
        "root_hydra_runtime_exists": (
            MODEL_ROOT / ".hydra/hydra.yaml"
        ).is_file(),
        "selected_checkpoint_exists": checkpoint.is_file(),
        "last_checkpoint_exists": (
            MODEL_ROOT / "checkpoints/last.ckpt"
        ).is_file(),
        "single_native_run_directory": len(run_roots) == 1,
        "tensorboard_events_exist": any(
            (MODEL_ROOT / "runs").rglob("events.out.tfevents.*")
        ),
        "regression_diagnostics_exist": any(
            (MODEL_ROOT / "runs").rglob("regression/*/im_*.tif")
        ),
        "mlflow_database_exists": (MODEL_ROOT / "mlflow.db").is_file(),
        "native_job_log_exists": (
            MODEL_ROOT / "regression3d_nematic_hydra_seed12345.log"
        ).is_file(),
    }


def artifact_manifest() -> list[dict[str, Any]]:
    manifest_path = PROVENANCE_ROOT / "artifact_manifest_sha256.csv"
    rows = [
        {
            "path": path.relative_to(MODEL_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(
            item for item in MODEL_ROOT.rglob("*") if item.is_file()
        )
        if path != manifest_path
    ]
    write_csv(manifest_path, rows)
    return rows

def training_contract() -> dict[str, Any]:
    os.environ.setdefault("PROJECT_ROOT", str(REPO))
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg = OmegaConf.load(MODEL_ROOT / ".hydra/config.yaml")
    model = hydra.utils.instantiate(cfg.model)
    parameter_count = sum(parameter.numel() for parameter in model.net.parameters())
    del model

    sources = {
        "movie2_image": (
            REPO
            / "DARE3d_data_190326/Gastruloid_241025/trainingset/"
            "movie2/im/movie2.tif",
            STAGING_ROOT / "val/im/movie2.tif",
            "b57ef33081e627a9305389b3dff796d9ca8e718e7ead2c4eaee7940733681f81",
        ),
        "movie2_label": (
            REPO
            / "DARE3d_data_190326/Gastruloid_241025/trainingset/"
            "movie2/label/movie2.tif",
            STAGING_ROOT / "val/label/movie2.tif",
            "d079c16070a378026d404f2bf101d017aaae2d76fa836dfbb16e981e88c03456",
        ),
        "movie3_image": (
            REPO
            / "DARE3d_data_190326/Gastruloid_241025/trainingset/"
            "movie3/im/movie3.tif",
            STAGING_ROOT / "train/im/movie3.tif",
            "3c56bd2ed0866eb2749d2373d33bfac6de03850146cb0318cb519113295a4cee",
        ),
        "movie3_label": (
            REPO
            / "DARE3d_data_190326/Gastruloid_241025/trainingset/"
            "movie3/label/movie3.tif",
            STAGING_ROOT / "train/label/movie3.tif",
            "ba6aa1d1db44b5b1133458287c0b5ccc7a83c616cb7a0a74e681b15b116c7db9",
        ),
        "movie4_image": (
            REPO
            / "DARE3d_data_190326/Gastruloid_241025/trainingset/"
            "movie4/im/movie4.tif",
            STAGING_ROOT / "train/im/movie4.tif",
            "cae0d6e069382519e13c19b478a78e0338073eb8ef9ed41274674a8aa2dfecf2",
        ),
        "movie4_label": (
            REPO
            / "DARE3d_data_190326/Gastruloid_241025/trainingset/"
            "movie4/label/movie4.tif",
            STAGING_ROOT / "train/label/movie4.tif",
            "e48254fef509ad6321f2275e81022ad41ae5b2d746f506f834a7f74a939de9f4",
        ),
    }
    source_records = {}
    for name, (source, staged, expected) in sources.items():
        source_hash = sha256(source)
        staged_hash = sha256(staged)
        source_records[name] = {
            "source": rel(source),
            "staged": rel(staged),
            "expected_sha256": expected,
            "source_sha256": source_hash,
            "staged_sha256": staged_hash,
            "same_file": os.path.samefile(source, staged),
            "verified": source_hash == staged_hash == expected,
        }

    assertions = {
        "seed_12345": cfg.seed == 12345,
        "corrected_nematic_loss": (
            cfg.model.criterion.angle_loss._target_
            == "dare3d.losses.angle3d.nematic_axis_projector_loss"
        ),
        "five_stage_16_filter_architecture": (
            cfg.model.net.n_stages == 5
            and cfg.model.net.start_filters == 16
            and parameter_count == 5_897_940
        ),
        "optimizer_matches_reference": (
            cfg.model.optimizer._target_ == "torch.optim.AdamW"
            and cfg.model.optimizer.lr == 0.001
            and list(cfg.model.optimizer.betas) == [0.9, 0.999]
            and cfg.model.optimizer.weight_decay == 0.0001
        ),
        "scheduler_matches_reference": (
            cfg.model.scheduler.steps_per_epoch == 167
            and cfg.model.scheduler.epochs == 100
            and cfg.model.scheduler.pct_start == 0.1
        ),
        "training_protocol_matches_reference": (
            cfg.steps_per_epoch == 2000
            and cfg.data.batch_size == 12
            and cfg.trainer.max_epochs == 100
            and cfg.trainer.precision == 32
        ),
        "preprocessing_matches_reference": (
            list(cfg.input_channels) == [-1, 0, 1]
            and cfg.renorm == "min-max"
            and cfg.order_dim_img == "zyx"
            and cfg.crop_size == 32
            and cfg.representation_mode == "rotation_matrix_SVD"
        ),
        "all_staged_sources_verified": all(
            record["verified"] for record in source_records.values()
        ),
        "reference_checkpoint_hash_verified": sha256(REFERENCE)
        == REFERENCE_SHA256,
    }
    return {
        "parameter_count": parameter_count,
        "source_files": source_records,
        "expected_unique_crops": {"train": 526, "validation": 156},
        "expected_resized_shapes_xyz": {
            "movie2": [330, 278, 164],
            "movie3": [387, 375, 77],
            "movie4": [405, 394, 152],
        },
        "assertions": assertions,
        "all_assertions_pass": all(assertions.values()),
    }


def performance_assertions(controlled, frozen_rows) -> dict[str, bool]:
    reference = controlled["by_model"]["reference_epoch_095"]
    candidate = controlled["by_model"]["candidate_hydra"]
    reference_angle = reference["corrected_nematic_axis_error_deg"]
    candidate_angle = candidate["corrected_nematic_axis_error_deg"]
    candidate_length = candidate["absolute_length_error_voxels"]
    paired = controlled["paired_candidate_vs_reference"]

    assertions = {
        "reference_controlled_mean_anchors": abs(
            reference_angle["mean"] - REFERENCE_CONTROLLED["mean_deg"]
        )
        <= 1e-5,
        "reference_controlled_p95_anchors": abs(
            reference_angle["p95"] - REFERENCE_CONTROLLED["p95_deg"]
        )
        <= 1e-5,
        "reference_controlled_length_anchors": abs(
            reference["absolute_length_error_voxels"]["mean"]
            - REFERENCE_CONTROLLED["length_mae_voxels"]
        )
        <= 1e-5,
        "candidate_controlled_mean_within_one_degree": (
            candidate_angle["mean"]
            <= REFERENCE_CONTROLLED["mean_deg"] + 1.0
        ),
        "paired_bootstrap_upper_bound_below_two_degrees": (
            paired["bootstrap_mean_difference_95_ci_deg"][1] < 2.0
        ),
        "candidate_controlled_p95_within_five_degrees": (
            candidate_angle["p95"]
            <= REFERENCE_CONTROLLED["p95_deg"] + 5.0
        ),
        "candidate_length_mae_within_quarter_voxel": (
            candidate_length["mean"]
            <= REFERENCE_CONTROLLED["length_mae_voxels"] + 0.25
        ),
    }

    for mode, expected in REFERENCE_CURRENT_MEANS.items():
        selected_reference = [
            row
            for row in frozen_rows
            if row["profile"] == TRAINING_CONSISTENT
            and row["model"] == "reference_epoch_095"
            and row["mode"] == mode
            and row["evaluated"]
        ]
        selected_candidate = [
            row
            for row in frozen_rows
            if row["profile"] == TRAINING_CONSISTENT
            and row["model"] == "candidate_hydra"
            and row["mode"] == mode
            and row["evaluated"]
        ]
        reference_mean = describe(
            row["corrected_nematic_axis_error_deg"]
            for row in selected_reference
        )["mean"]
        candidate_mean = describe(
            row["corrected_nematic_axis_error_deg"]
            for row in selected_candidate
        )["mean"]
        assertions[f"reference_{mode}_anchors"] = abs(reference_mean - expected) <= 1e-5
        assertions[f"candidate_{mode}_within_two_degrees"] = (
            candidate_mean <= reference_mean + 2.0
        )

    all_angles = [
        row["corrected_nematic_axis_error_deg"]
        for row in frozen_rows
        if row["evaluated"]
    ]
    assertions["all_frozen_center_angles_finite"] = all(
        np.isfinite(value) for value in all_angles
    )
    assertions["all_frozen_center_angles_in_nematic_range"] = all(
        0.0 <= value <= 90.0 + 1e-5 for value in all_angles
    )
    return assertions

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


def run_evaluation() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the locked candidate evaluation")
    if not PROTECTED_BEFORE.is_file():
        raise FileNotFoundError(PROTECTED_BEFORE)

    checkpoint = candidate_checkpoint()
    contract = training_contract()
    reference_net, reference_keys = audit_evaluation.make_net(REFERENCE)
    candidate_net, candidate_keys = audit_evaluation.make_net(checkpoint)
    models = {
        "reference_epoch_095": reference_net,
        "candidate_hydra": candidate_net,
    }

    controlled, controlled_rows = controlled_evaluation(models)
    frozen, frozen_rows = frozen_center_evaluation(models)
    del reference_net, candidate_net, models
    gc.collect()
    torch.cuda.empty_cache()

    performance = performance_assertions(controlled, frozen_rows)
    controlled_angles = [
        row["corrected_nematic_axis_error_deg"] for row in controlled_rows
    ]
    inference_counts_ok = all(
        len(
            [
                row
                for row in frozen_rows
                if row["profile"] == profile
                and row["model"] == model
                and row["mode"] == mode
            ]
        )
        == expected
        for profile in (TRAINING_CONSISTENT, LEGACY_RAW)
        for model in ("reference_epoch_095", "candidate_hydra")
        for mode, expected in (
            ("all_groundtruth_centers", 145),
            ("matched_groundtruth_centers", 124),
            ("predicted_centers", 124),
        )
    )

    protected = protected_asset_verification()
    native_artifacts = native_artifact_verification(checkpoint)
    checkpoint_data = {
        "reference": checkpoint_record(REFERENCE),
        "candidate": checkpoint_record(checkpoint),
        "loaded_network_keys": {
            "reference": reference_keys,
            "candidate": candidate_keys,
        },
    }
    core_assertions = {
        "reference_checkpoint_is_validated_epoch_095": (
            checkpoint_data["reference"]["sha256"] == REFERENCE_SHA256
            and checkpoint_data["reference"]["epoch"] == 95
        ),
        "candidate_checkpoint_differs_from_reference": (
            checkpoint_data["candidate"]["sha256"] != REFERENCE_SHA256
        ),
        "both_checkpoints_load_all_network_keys": (
            reference_keys == candidate_keys == 63
        ),
        "controlled_angles_are_finite": all(
            np.isfinite(value) for value in controlled_angles
        ),
        "controlled_angles_are_in_nematic_range": all(
            0.0 <= value <= 90.0 + 1e-5
            for value in controlled_angles
        ),
        "one_regression_prediction_per_frozen_center": inference_counts_ok,
        "training_contract_passes": contract["all_assertions_pass"],
        "protected_assets_pass": protected["all_assertions_pass"],
        "native_hydra_artifacts_complete": all(native_artifacts.values()),
    }
    assertions = {
        **core_assertions,
        **performance,
        **{
            f"protected::{name}": passed
            for name, passed in protected["assertions"].items()
        },
        **{
            f"native_artifact::{name}": passed
            for name, passed in native_artifacts.items()
        },
    }
    passed = all(assertions.values())
    result = {
        "status": "complete" if passed else "complete_with_failed_gates",
        "release_suitable": passed,
        "completed_at": now_iso(),
        "acceptance_rule": (
            "One locked Hydra run only; no checkpoint/trajectory hash match "
            "required. Candidate must meet every controlled, paired-bootstrap, "
            "frozen-center, provenance, and loadability gate."
        ),
        "controlled_unique_holdout": controlled,
        "frozen_centers": frozen,
        "checkpoints": checkpoint_data,
        "training_contract": contract,
        "protected_assets": protected,
        "native_artifacts": native_artifacts,
        "assertions": assertions,
        "all_assertions_pass": passed,
    }
    result_path = EVALUATION_ROOT / "result.json"
    write_json(result_path, result)

    manifest_rows = artifact_manifest()
    manifest_path = PROVENANCE_ROOT / "artifact_manifest_sha256.csv"
    run_dirs = sorted((MODEL_ROOT / "runs").glob("*"))
    provenance = {
        "schema_version": 1,
        "created_at": now_iso(),
        "scope": "fresh Hydra retraining of the gastruloid regression model only",
        "training_source_commit": TRAINING_COMMIT,
        "evaluation_source_commit": git_output("rev-parse", "HEAD"),
        "tracked_diff_at_evaluation": git_output(
            "status", "--short", "--untracked-files=no"
        ),
        "training_command": (
            "python dare3d/train.py "
            "experiment=gastruloid_nematic_regression "
            f"model_root={MODEL_ROOT.as_posix()} "
            f"split_root={STAGING_ROOT.as_posix()} "
            "date=20260904T193751Z"
        ),
        "launch_history": [
            {
                "attempt": 1,
                "outcome": "stopped before data/model initialization",
                "reason": (
                    "paths.output_dir did not yet exist for enforce_tags; "
                    "no optimization step ran"
                ),
            },
            {
                "attempt": 2,
                "outcome": "completed native Hydra training and test",
                "change": (
                    "created only the already-declared runs/20260904T193751Z "
                    "directory, then used identical overrides"
                ),
            },
        ],
        "model_root": str(MODEL_ROOT),
        "run_directories": [rel(path) for path in run_dirs],
        "checkpoint": checkpoint_data["candidate"],
        "reference_checkpoint": checkpoint_data["reference"],
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
        },
        "evaluation": {
            "result": rel(result_path),
            "result_sha256": sha256(result_path),
            "release_suitable": passed,
        },
        "artifact_manifest": {
            "path": rel(manifest_path),
            "sha256": sha256(manifest_path),
            "records": len(manifest_rows),
            "exclusions": [
                "the manifest itself",
                "provenance.json, which is written after the manifest",
            ],
        },
        "protected_asset_verification": protected,
        "notes": [
            "No segmentation model was loaded by this evaluator.",
            "No Zenodo, Napari default, legacy model, or validated audit file was written.",
            "The controlled holdout is also the historical validation set, not an independent external test.",
        ],
    }
    write_json(MODEL_ROOT / "provenance.json", provenance)
    if not passed:
        failed = [name for name, value in assertions.items() if not value]
        raise AssertionError(f"Candidate failed release gates: {failed}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = EVALUATION_ROOT / "evaluation.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as stream:
        with contextlib.redirect_stdout(Tee(sys.stdout, stream)):
            with contextlib.redirect_stderr(Tee(sys.stderr, stream)):
                result = run_evaluation()
    print(
        json.dumps(
            {
                "release_suitable": result["release_suitable"],
                "result": rel(EVALUATION_ROOT / "result.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
