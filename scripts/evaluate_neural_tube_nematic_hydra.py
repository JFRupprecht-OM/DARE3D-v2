"""Evaluate, decide, and safely promote the fresh neural-tube Hydra regressor.

The script is regression-only. It never runs segmentation and compares the new
Hydra checkpoint directly with the immutable validated epoch_139 checkpoint.
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
SCRIPTS = REPO / "scripts"
for import_root in (REPO, AUDIT_CODE, SCRIPTS):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

import neural_tube_nematic_evaluation as neural_evaluation
import neural_tube_nematic_experiment as neural_experiment
import neural_tube_regression_center_replay as neural_replay
import prepare_neural_tube_nematic_hydra as preparation
import regression_preprocessing_fix_validation as current_evaluation
from dare3d.data.components.regression_geometry import TRAINING_CONSISTENT
from dare3d.models.finetune import load_net_state_dict


MODEL_ROOT = preparation.MODEL_ROOT
STAGING_ROOT = preparation.STAGING_ROOT
DESTINATION_ROOT = preparation.DESTINATION_ROOT
REFERENCE = (
    REPO
    / "docs/reproducibility_audit/neural_tube_nematic_full_pipeline/"
    "seed_12345/regression_nematic_retrained/checkpoints/epoch_139.ckpt"
)
REFERENCE_SHA256 = "192c2bec48e8c6dc9b78740254a55082b89416e3b986bf243edf60586530483b"
EVALUATION_ROOT = MODEL_ROOT / "evaluation"
PREDICTION_ROOT = EVALUATION_ROOT / "predictions"
PROVENANCE_ROOT = MODEL_ROOT / "provenance"
PROTECTED_BEFORE = STAGING_ROOT / "protected_before.json"
SCALE_TABLE = REPO / "data/3D/scales.json"
BOOTSTRAP_SEED = 20260829
BOOTSTRAP_REPLICATES = 10_000
REFERENCE_LABEL = "reference_epoch_139"
CANDIDATE_LABEL = "candidate_hydra"
MODES = (
    "all_groundtruth_centers",
    "matched_groundtruth_centers",
    "predicted_centers",
)
EXPECTED_CENTER_COUNTS = {
    "all_groundtruth_centers": 122,
    "matched_groundtruth_centers": 114,
    "predicted_centers": 114,
}
REFERENCE_CONTROLLED = {
    "audit_0208": {
        "mean_deg": 18.945465087890625,
        "median_deg": 17.12960720062256,
        "p95_deg": 40.24514503479003,
        "length_mae_voxels": 0.8358930349349976,
    },
    "physical_02076": {
        "mean_deg": 19.546722412109375,
        "median_deg": 16.44617748260498,
        "p95_deg": 45.802453422546385,
        "length_mae_voxels": 0.835491955280304,
    },
}
REFERENCE_FROZEN = {
    "all_groundtruth_centers": 14.202218311301653,
    "matched_groundtruth_centers": 14.2764151161204,
    "predicted_centers": 14.347481341219094,
}
WAIVABLE_ASSERTIONS = frozenset(
    {
        "candidate_audit_0208_mean_within_one_degree",
        "audit_0208_bootstrap_upper_bound_below_two_degrees",
        "candidate_physical_02076_mean_within_one_degree",
        "physical_02076_bootstrap_upper_bound_below_two_degrees",
    }
)


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
            "Expected exactly one validation-ranked epoch checkpoint, found "
            f"{[path.name for path in checkpoints]}"
        )
    return checkpoints[0]


def checkpoint_record(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "path": rel(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "epoch": int(checkpoint.get("epoch", -1)),
        "global_step": int(checkpoint.get("global_step", -1)),
        "lightning_version": checkpoint.get("pytorch-lightning_version"),
    }


def load_regression_net(path: Path):
    model = neural_experiment.make_regression_model(corrected_loss=True)
    loaded = load_net_state_dict(model.net, str(path), stage="regression")
    net = model.net.to("cuda")
    net.eval()
    return net, len(loaded)


def make_controlled_dataset(profile: str):
    dataset = neural_experiment.make_dataset("regression", "test")
    if profile == "physical_02076":
        dataset.scale_file = str(SCALE_TABLE.resolve())
        dataset.movies_scale = dataset.load_movie_scales(dataset.scale_file)
        dataset.require_scale_file = True
    elif profile != "audit_0208":
        raise ValueError(profile)
    dataset.init()
    return dataset


def controlled_evaluation(
    models: dict[str, torch.nn.Module], profile: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    print(f"Controlled movie_M evaluation: {profile}", flush=True)
    dataset = make_controlled_dataset(profile)
    if len(dataset.crops) != 80:
        raise AssertionError(f"Expected 80 unique movie_M crops, got {len(dataset.crops)}")

    rows = []
    for name, net in models.items():
        model_rows = neural_evaluation.controlled_rows_for_model(
            name, net, dataset
        )
        for row in model_rows:
            row["preprocessing_profile"] = profile
        rows.extend(model_rows)

    summaries = {}
    for name in models:
        selected = [row for row in rows if row["model"] == name]
        summaries[name] = {
            "sample_count": len(selected),
            "corrected_nematic_axis_error_deg": describe(
                row["corrected_nematic_axis_error_deg"] for row in selected
            ),
            "production_quaternion_error_deg": describe(
                row["production_quaternion_error_deg"] for row in selected
            ),
            "absolute_length_error_voxels": describe(
                row["absolute_length_error_voxels"] for row in selected
            ),
            "undefined_corrected_axis_count": sum(
                not row["corrected_axis_defined"] for row in selected
            ),
        }

    reference = sorted(
        (row for row in rows if row["model"] == REFERENCE_LABEL),
        key=lambda row: row["sample_index"],
    )
    candidate = sorted(
        (row for row in rows if row["model"] == CANDIDATE_LABEL),
        key=lambda row: row["sample_index"],
    )
    reference_ids = [
        (row["sample_index"], row["frame"], row["movie"]) for row in reference
    ]
    candidate_ids = [
        (row["sample_index"], row["frame"], row["movie"]) for row in candidate
    ]
    if reference_ids != candidate_ids:
        raise AssertionError(f"Controlled event ordering changed for {profile}")

    result = {
        "profile": profile,
        "source_scale_xyz_um": (
            [0.208, 0.208, 1.0]
            if profile == "audit_0208"
            else [0.2076, 0.2076, 1.0]
        ),
        "dataset": neural_experiment.dataset_record(
            dataset, "regression", "test"
        ),
        "by_model": summaries,
        "paired_candidate_vs_reference": paired_bootstrap(
            [row["corrected_nematic_axis_error_deg"] for row in reference],
            [row["corrected_nematic_axis_error_deg"] for row in candidate],
        ),
    }
    del dataset
    gc.collect()
    return result, rows


def save_predictions(path: Path, predictions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        centers_raw=np.asarray(
            [prediction["center_raw"] for prediction in predictions]
        ),
        centers_regression=np.asarray(
            [prediction["center_regression"] for prediction in predictions]
        ),
        lengths_regression_voxels=np.asarray(
            [
                prediction["length_regression_voxels"]
                for prediction in predictions
            ]
        ),
        quaternions_wxyz=np.asarray(
            [prediction["rotation"] for prediction in predictions]
        ),
        axes_physical_xyz=np.asarray(
            [prediction["axis_physical_xyz"] for prediction in predictions]
        ),
        lengths_physical_um=np.asarray(
            [prediction["length_physical_um"] for prediction in predictions]
        ),
    )


def frozen_center_evaluation(
    models: dict[str, torch.nn.Module]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    print("Frozen movie_I2 center evaluation at validated 0.208 geometry", flush=True)
    centers = {
        mode: neural_replay.centers_from_archive(path)[0]
        for mode, path in neural_replay.MODE_SOURCES.items()
    }
    if {mode: len(value) for mode, value in centers.items()} != EXPECTED_CENTER_COUNTS:
        raise AssertionError("Frozen neural center counts changed")

    dataset = current_evaluation.make_neural_dataset((0.208, 0.208, 1.0))
    dataset.init_inference(TRAINING_CONSISTENT)
    targets_all = dataset.gather_groundtruth_info(
        centers["all_groundtruth_centers"]
    )
    targets_matched = dataset.gather_groundtruth_info(
        centers["matched_groundtruth_centers"]
    )
    targets = {
        "all_groundtruth_centers": targets_all,
        "matched_groundtruth_centers": targets_matched,
        "predicted_centers": targets_matched,
    }
    if any(target is None for values in targets.values() for target in values):
        raise AssertionError("A frozen center no longer resolves to an annotation")

    rows = []
    for model_name, net in models.items():
        for mode in MODES:
            print(
                f"Frozen centers: {model_name}, {mode}, N={len(centers[mode])}",
                flush=True,
            )
            predictions = neural_evaluation.regression_inference(
                dataset,
                net,
                centers[mode],
                "cuda",
                output_dir=None,
            )
            if len(predictions) != len(centers[mode]):
                raise AssertionError((model_name, mode, len(predictions)))
            save_predictions(
                PREDICTION_ROOT / f"{model_name}_{mode}.npz",
                predictions,
            )
            rows.extend(
                current_evaluation.event_rows(
                    "neural_tube_movie_I2_frozen_centers",
                    TRAINING_CONSISTENT,
                    model_name,
                    mode,
                    targets[mode],
                    predictions,
                )
            )

    result = {
        "center_source": {
            mode: rel(path) for mode, path in neural_replay.MODE_SOURCES.items()
        },
        "counts": EXPECTED_CENTER_COUNTS,
        "preprocessing_manifest": dataset.preprocessing_manifest(),
        "summaries": current_evaluation.summarize_event_rows(rows),
        "segmentation_rerun": False,
    }
    del dataset
    gc.collect()
    torch.cuda.empty_cache()
    return result, rows


def comparable_hash_records(current, baseline) -> bool:
    def index(records):
        return {
            record["path"]: {
                "bytes": int(record["bytes"]),
                "sha256": record["sha256"],
            }
            for record in records
        }

    return index(current) == index(baseline)


def comparable_metadata(current, baseline) -> bool:
    def index(records):
        return {
            record["path"]: {
                "bytes": int(record["bytes"]),
                "mtime_ns": int(record["mtime_ns"]),
            }
            for record in records
        }

    return index(current) == index(baseline)


def protected_asset_verification() -> dict[str, Any]:
    baseline = json.loads(PROTECTED_BEFORE.read_text(encoding="utf-8"))
    tree_results = {}
    for label, before in baseline["protected_trees"].items():
        current = preparation.hash_tree(REPO / before["root"])
        tree_results[label] = comparable_hash_records(
            current["files"], before["files"]
        )

    current_files = [
        preparation.file_record(REPO / record["path"])
        for record in baseline["protected_files"]
    ]
    source_results = {}
    for name, before in baseline["source_files"].items():
        current = preparation.file_record(REPO / before["path"])
        source_results[name] = (
            current["bytes"] == before["bytes"]
            and current["sha256"] == before["sha256"]
        )

    assertions = {
        "candidate_was_absent_at_baseline": not baseline["candidate_existed"],
        "destination_was_absent_at_baseline": not baseline["destination_existed"],
        "all_protected_trees_byte_identical": all(tree_results.values()),
        "all_release_root_checkpoints_byte_identical": comparable_hash_records(
            current_files, baseline["protected_files"]
        ),
        "all_neural_source_tiffs_byte_identical": all(source_results.values()),
        "complete_zenodo_tree_unchanged_before_promotion": comparable_metadata(
            preparation.zenodo_metadata(), baseline["zenodo_tree_metadata"]
        ),
    }
    PROVENANCE_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROTECTED_BEFORE, PROVENANCE_ROOT / "protected_before.json")
    result = {
        "baseline": rel(PROTECTED_BEFORE),
        "tree_results": tree_results,
        "source_results": source_results,
        "assertions": assertions,
        "all_assertions_pass": all(assertions.values()),
    }
    write_json(PROVENANCE_ROOT / "protected_after_evaluation.json", result)
    return result
def native_artifact_verification(checkpoint: Path) -> dict[str, bool]:
    run_roots = sorted(
        path for path in (MODEL_ROOT / "runs").glob("*") if path.is_dir()
    )
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


def training_contract() -> dict[str, Any]:
    os.environ.setdefault("PROJECT_ROOT", str(REPO))
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg = OmegaConf.load(MODEL_ROOT / ".hydra/config.yaml")
    model = hydra.utils.instantiate(cfg.model)
    parameter_count = sum(parameter.numel() for parameter in model.net.parameters())
    state_tensor_count = len(model.net.state_dict())
    del model

    observed_datasets = {}
    for role, dataset_cfg in (
        ("train", cfg.data.train_data),
        ("validation", cfg.data.val_data),
    ):
        dataset = hydra.utils.instantiate(dataset_cfg)
        dataset.init()
        observed_datasets[role] = {
            "movie_names": list(dataset.movie_names),
            "unique_crops": len(dataset.crops),
            "scale_file": str(dataset.scale_file),
            "scale_file_exists": Path(dataset.scale_file).is_file(),
            "default_scale_xyz_um": [
                float(value) for value in dataset.default_scale
            ],
            "target_scale_um": float(dataset.target_scale),
            "original_internal_shapes_txyz": [
                [int(value) for value in shape]
                for shape in dataset.original_movies_shape
            ],
            "resized_shapes_xyz_before_crop_padding": [
                [
                    int(value)
                    for value in dataset._compute_target_shape(
                        shape, movie_name
                    )[1:]
                ]
                for shape, movie_name in zip(
                    dataset.original_movies_shape,
                    dataset.movie_names,
                )
            ],
            "padded_preprocessed_shapes_txyz": [
                [int(value) for value in movie.shape]
                for movie in dataset.movies_im
            ],
        }
        del dataset
        gc.collect()

    source_records = {}
    for name, spec in preparation.SOURCE_FILES.items():
        source_hash = sha256(spec["source"])
        staged = spec["staged"]
        staged_hash = sha256(staged) if staged is not None else None
        same_file = (
            os.path.samefile(spec["source"], staged)
            if staged is not None
            else None
        )
        verified = source_hash == spec["sha256"]
        if staged is not None:
            verified = (
                verified
                and staged_hash == spec["sha256"]
                and same_file
            )
        source_records[name] = {
            "source": rel(spec["source"]),
            "staged": rel(staged) if staged is not None else None,
            "expected_sha256": spec["sha256"],
            "source_sha256": source_hash,
            "staged_sha256": staged_hash,
            "same_file": same_file,
            "verified": verified,
        }

    assertions = {
        "seed_12345": cfg.seed == 12345,
        "corrected_nematic_loss": (
            cfg.model.criterion.angle_loss._target_
            == "dare3d.losses.angle3d.nematic_axis_projector_loss"
        ),
        "neural_three_stage_32_filter_architecture": (
            cfg.model.net.n_stages == 3
            and cfg.model.net.start_filters == 32
            and parameter_count == 1_605_268
            and state_tensor_count == 41
        ),
        "optimizer_matches_validated_neural_reference": (
            cfg.model.optimizer._target_ == "torch.optim.AdamW"
            and cfg.model.optimizer.lr == 0.001
            and list(cfg.model.optimizer.betas) == [0.9, 0.999]
            and cfg.model.optimizer.weight_decay == 0.0001
        ),
        "scheduler_matches_validated_neural_reference": (
            cfg.model.scheduler.steps_per_epoch == 63
            and cfg.model.scheduler.epochs == 200
            and cfg.model.scheduler.max_lr == 0.001
            and cfg.model.scheduler.pct_start == 0.1
        ),
        "training_protocol_matches_validated_neural_reference": (
            cfg.steps_per_epoch == 2000
            and cfg.data.batch_size == 32
            and cfg.trainer.max_epochs == 200
            and cfg.trainer.precision == 32
            and cfg.trainer.deterministic is False
            and cfg.trainer.accumulate_grad_batches == 1
        ),
        "preprocessing_matches_validated_neural_reference": (
            list(cfg.input_channels) == [-1, 0, 1]
            and list(cfg.default_scale) == [0.208, 0.208, 1.0]
            and cfg.target_scale == 1.0
            and cfg.renorm == "min-max"
            and cfg.order_dim_img == "zyx"
            and cfg.crop_size == 32
            and cfg.time_axis_padding == 1
            and cfg.representation_mode == "rotation_matrix_SVD"
            and not cfg.data.train_data.require_scale_file
            and not cfg.data.val_data.require_scale_file
        ),
        "native_neural_split_and_crop_counts_observed": (
            observed_datasets["train"]["movie_names"] == ["movie_E"]
            and observed_datasets["train"]["unique_crops"] == 222
            and observed_datasets["validation"]["movie_names"]
            == ["movie_I2"]
            and observed_datasets["validation"]["unique_crops"] == 123
        ),
        "native_neural_resized_shapes_observed": (
            observed_datasets["train"][
                "resized_shapes_xyz_before_crop_padding"
            ]
            == [[212, 212, 10]]
            and observed_datasets["validation"][
                "resized_shapes_xyz_before_crop_padding"
            ]
            == [[212, 212, 10]]
            and observed_datasets["train"][
                "padded_preprocessed_shapes_txyz"
            ]
            == [[27, 244, 244, 42]]
            and observed_datasets["validation"][
                "padded_preprocessed_shapes_txyz"
            ]
            == [[21, 244, 244, 42]]
        ),
        "missing_neural_scale_table_is_explicit": (
            not observed_datasets["train"]["scale_file_exists"]
            and not observed_datasets["validation"]["scale_file_exists"]
            and observed_datasets["train"]["default_scale_xyz_um"]
            == [0.208, 0.208, 1.0]
            and observed_datasets["validation"]["default_scale_xyz_um"]
            == [0.208, 0.208, 1.0]
        ),
        "all_genuine_sources_verified": all(
            record["verified"] for record in source_records.values()
        ),
        "reference_checkpoint_hash_verified": (
            sha256(REFERENCE) == REFERENCE_SHA256
        ),
    }
    return {
        "parameter_count": parameter_count,
        "state_tensor_count": state_tensor_count,
        "source_files": source_records,
        "expected_unique_crops": {"train": 222, "validation": 123},
        "expected_resized_shapes_xyz": {
            "movie_E": [212, 212, 10],
            "movie_I2": [212, 212, 10],
        },
        "observed_datasets": observed_datasets,
        "assertions": assertions,
        "all_assertions_pass": all(assertions.values()),
    }


def performance_assertions(
    controlled: dict[str, dict[str, Any]],
    frozen_rows: list[dict[str, Any]],
) -> tuple[dict[str, bool], dict[str, float]]:
    assertions = {}
    mean_deltas = {}
    for profile, expected in REFERENCE_CONTROLLED.items():
        result = controlled[profile]
        reference = result["by_model"][REFERENCE_LABEL]
        candidate = result["by_model"][CANDIDATE_LABEL]
        reference_angle = reference["corrected_nematic_axis_error_deg"]
        candidate_angle = candidate["corrected_nematic_axis_error_deg"]
        candidate_length = candidate["absolute_length_error_voxels"]
        paired = result["paired_candidate_vs_reference"]
        prefix = profile

        assertions[f"reference_{prefix}_mean_anchors"] = (
            abs(reference_angle["mean"] - expected["mean_deg"]) <= 1e-5
        )
        assertions[f"reference_{prefix}_median_anchors"] = (
            abs(reference_angle["median"] - expected["median_deg"]) <= 1e-5
        )
        assertions[f"reference_{prefix}_p95_anchors"] = (
            abs(reference_angle["p95"] - expected["p95_deg"]) <= 1e-5
        )
        assertions[f"reference_{prefix}_length_anchors"] = (
            abs(
                reference["absolute_length_error_voxels"]["mean"]
                - expected["length_mae_voxels"]
            )
            <= 1e-5
        )
        assertions[f"candidate_{prefix}_mean_within_one_degree"] = (
            candidate_angle["mean"] <= expected["mean_deg"] + 1.0
        )
        assertions[f"{prefix}_bootstrap_upper_bound_below_two_degrees"] = (
            paired["bootstrap_mean_difference_95_ci_deg"][1] < 2.0
        )
        assertions[f"candidate_{prefix}_p95_within_five_degrees"] = (
            candidate_angle["p95"] <= expected["p95_deg"] + 5.0
        )
        assertions[f"candidate_{prefix}_length_mae_within_quarter_voxel"] = (
            candidate_length["mean"]
            <= expected["length_mae_voxels"] + 0.25
        )
        mean_deltas[profile] = (
            candidate_angle["mean"] - reference_angle["mean"]
        )

    for mode, expected in REFERENCE_FROZEN.items():
        selected_reference = [
            row
            for row in frozen_rows
            if row["profile"] == TRAINING_CONSISTENT
            and row["model"] == REFERENCE_LABEL
            and row["mode"] == mode
            and row["evaluated"]
        ]
        selected_candidate = [
            row
            for row in frozen_rows
            if row["profile"] == TRAINING_CONSISTENT
            and row["model"] == CANDIDATE_LABEL
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
        assertions[f"reference_frozen_{mode}_anchors"] = (
            abs(reference_mean - expected) <= 1e-5
        )
        assertions[f"candidate_frozen_{mode}_within_two_degrees"] = (
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
    return assertions, mean_deltas


def release_decision(
    assertions: dict[str, bool],
    mean_deltas: dict[str, float],
) -> dict[str, Any]:
    failed = sorted(name for name, passed in assertions.items() if not passed)
    nonwaivable = sorted(set(failed) - WAIVABLE_ASSERTIONS)
    override_delta_passes = all(
        mean_deltas[profile] <= 3.0
        for profile in ("audit_0208", "physical_02076")
    )
    locked_pass = not failed
    override_used = (
        not locked_pass
        and not nonwaivable
        and override_delta_passes
    )
    promotion_authorized = locked_pass or override_used
    if locked_pass:
        outcome = "normal_pass"
    elif override_used:
        outcome = "explicit_user_release_override"
    else:
        outcome = "rejected"
    return {
        "outcome": outcome,
        "locked_criteria_passed": locked_pass,
        "promotion_authorized": promotion_authorized,
        "explicit_release_override_used": override_used,
        "override_rule": (
            "Only the locked +1 degree mean and paired-bootstrap upper-CI "
            "gates are waivable. Both controlled mean degradations must be "
            "<= +3.0 degrees and every other assertion must pass."
        ),
        "controlled_mean_candidate_minus_reference_deg": mean_deltas,
        "controlled_mean_override_limit_deg": 3.0,
        "override_delta_passes": override_delta_passes,
        "failed_locked_assertions": failed,
        "failed_nonwaivable_assertions": nonwaivable,
    }
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


def artifact_manifest() -> list[dict[str, Any]]:
    manifest_path = PROVENANCE_ROOT / "artifact_manifest_sha256.csv"
    excluded = {
        manifest_path,
        MODEL_ROOT / "provenance.json",
    }
    rows = [
        {
            "path": path.relative_to(MODEL_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(
            item for item in MODEL_ROOT.rglob("*") if item.is_file()
        )
        if path not in excluded
    ]
    write_csv(manifest_path, rows)
    return rows


def post_promotion_protected_verification() -> dict[str, Any]:
    baseline = json.loads(PROTECTED_BEFORE.read_text(encoding="utf-8"))
    tree_results = {
        label: comparable_hash_records(
            preparation.hash_tree(REPO / record["root"])["files"],
            record["files"],
        )
        for label, record in baseline["protected_trees"].items()
    }
    root_checkpoints = [
        preparation.file_record(REPO / record["path"])
        for record in baseline["protected_files"]
    ]
    source_results = {}
    for name, before in baseline["source_files"].items():
        current = preparation.file_record(REPO / before["path"])
        source_results[name] = (
            current["bytes"] == before["bytes"]
            and current["sha256"] == before["sha256"]
        )
    assertions = {
        "all_protected_trees_still_byte_identical": all(
            tree_results.values()
        ),
        "all_release_root_checkpoints_still_byte_identical": (
            comparable_hash_records(
                root_checkpoints, baseline["protected_files"]
            )
        ),
        "all_neural_source_tiffs_still_byte_identical": all(
            source_results.values()
        ),
    }
    return {
        "tree_results": tree_results,
        "source_results": source_results,
        "assertions": assertions,
        "all_assertions_pass": all(assertions.values()),
    }


def zenodo_after_promotion_verification() -> dict[str, Any]:
    baseline = json.loads(PROTECTED_BEFORE.read_text(encoding="utf-8"))
    before = {
        record["path"]: {
            "bytes": int(record["bytes"]),
            "mtime_ns": int(record["mtime_ns"]),
        }
        for record in baseline["zenodo_tree_metadata"]
    }
    after_records = preparation.zenodo_metadata()
    after = {
        record["path"]: {
            "bytes": int(record["bytes"]),
            "mtime_ns": int(record["mtime_ns"]),
        }
        for record in after_records
    }
    missing = sorted(set(before) - set(after))
    changed = sorted(
        path
        for path in set(before) & set(after)
        if before[path] != after[path]
    )
    additions = sorted(set(after) - set(before))
    destination_prefix = rel(DESTINATION_ROOT).rstrip("/") + "/"
    additions_scoped = bool(additions) and all(
        path.startswith(destination_prefix) for path in additions
    )
    assertions = {
        "no_preexisting_zenodo_file_removed": not missing,
        "no_preexisting_zenodo_file_size_or_timestamp_changed": not changed,
        "zenodo_additions_exist": bool(additions),
        "all_zenodo_additions_are_inside_new_neural_candidate": (
            additions_scoped
        ),
    }
    return {
        "destination_prefix": destination_prefix,
        "preexisting_file_count": len(before),
        "post_promotion_file_count": len(after),
        "addition_count": len(additions),
        "missing_preexisting_files": missing,
        "changed_preexisting_files": changed,
        "additions": additions,
        "assertions": assertions,
        "all_assertions_pass": all(assertions.values()),
    }


def smoke_test_checkpoint(path: Path) -> dict[str, Any]:
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    with contextlib.redirect_stdout(captured_stdout):
        with contextlib.redirect_stderr(captured_stderr):
            net, loaded_keys = load_regression_net(path)
            dataset = make_controlled_dataset("physical_02076")
            inputs, _ = dataset[0]
            batch = inputs["input"].unsqueeze(0).to("cuda")
            with torch.inference_mode():
                output = net(batch)["head1"]
            angle = output["angle"].detach().cpu()
            length = output["len"].detach().cpu()
    record = {
        "checkpoint": rel(path),
        "sha256": sha256(path),
        "loaded_network_keys": loaded_keys,
        "input_shape": list(batch.shape),
        "angle_shape": list(angle.shape),
        "length_shape": list(length.shape),
        "angle_all_finite": bool(torch.isfinite(angle).all()),
        "length_all_finite": bool(torch.isfinite(length).all()),
        "captured_stdout": captured_stdout.getvalue(),
        "captured_stderr": captured_stderr.getvalue(),
    }
    record["passes"] = (
        loaded_keys == 41
        and record["angle_all_finite"]
        and record["length_all_finite"]
    )
    del net, dataset, batch, output, angle, length
    gc.collect()
    torch.cuda.empty_cache()
    return record


def shared_copy_verification() -> dict[str, Any]:
    records = []
    for source in sorted(
        path for path in MODEL_ROOT.rglob("*") if path.is_file()
    ):
        relative = source.relative_to(MODEL_ROOT)
        destination = DESTINATION_ROOT / relative
        destination_exists = destination.is_file()
        source_hash = sha256(source)
        destination_hash = (
            sha256(destination) if destination_exists else None
        )
        records.append(
            {
                "path": relative.as_posix(),
                "source_bytes": source.stat().st_size,
                "destination_bytes": (
                    destination.stat().st_size
                    if destination_exists
                    else None
                ),
                "source_sha256": source_hash,
                "destination_sha256": destination_hash,
                "byte_identical": (
                    destination_exists
                    and source.stat().st_size == destination.stat().st_size
                    and source_hash == destination_hash
                ),
            }
        )
    return {
        "source_file_count": len(records),
        "all_source_files_present_and_byte_identical": all(
            record["byte_identical"] for record in records
        ),
        "records": records,
    }


def promote(
    checkpoint: Path,
    decision: dict[str, Any],
) -> dict[str, Any]:
    if not decision["promotion_authorized"]:
        raise RuntimeError("Promotion called without a passing release decision")
    if DESTINATION_ROOT.exists():
        raise FileExistsError(
            f"Refusing to overwrite Zenodo destination: {DESTINATION_ROOT}"
        )

    selected = checkpoint_record(checkpoint)
    shutil.copytree(MODEL_ROOT, DESTINATION_ROOT, copy_function=shutil.copy2)
    initial_copy = shared_copy_verification()
    if not initial_copy["all_source_files_present_and_byte_identical"]:
        raise AssertionError("Structured candidate copy is not byte-identical")

    epoch = selected["epoch"]
    alias_name = f"DARE3D_neural_tube_regression_epoch{epoch:03d}.ckpt"
    alias = DESTINATION_ROOT / "checkpoints" / alias_name
    shutil.copy2(checkpoint, alias)
    alias_hash = sha256(alias)
    if alias_hash != selected["sha256"]:
        raise AssertionError("Promoted release alias hash differs from source")

    smoke = smoke_test_checkpoint(alias)
    protected = post_promotion_protected_verification()
    zenodo = zenodo_after_promotion_verification()
    if not smoke["passes"]:
        raise AssertionError("Promoted checkpoint failed inference smoke test")
    if not protected["all_assertions_pass"]:
        raise AssertionError("A protected asset changed during promotion")
    if not zenodo["all_assertions_pass"]:
        raise AssertionError("Zenodo promotion escaped the authorized directory")

    final_shared_copy = shared_copy_verification()
    if not final_shared_copy[
        "all_source_files_present_and_byte_identical"
    ]:
        raise AssertionError("A copied source artifact changed during promotion")

    promotion_record = {
        "schema_version": 1,
        "promoted_at": now_iso(),
        "scope": "neural-tube regression model only",
        "decision": decision,
        "source_model_directory": str(MODEL_ROOT),
        "source_model_directory_retained": MODEL_ROOT.is_dir(),
        "destination_model_directory": str(DESTINATION_ROOT),
        "selected_source_checkpoint": selected,
        "copied_selected_checkpoint": {
            "path": rel(
                DESTINATION_ROOT
                / "checkpoints"
                / checkpoint.name
            ),
            "filename": checkpoint.name,
            "sha256": sha256(
                DESTINATION_ROOT
                / "checkpoints"
                / checkpoint.name
            ),
            "byte_identical_to_source": (
                sha256(
                    DESTINATION_ROOT
                    / "checkpoints"
                    / checkpoint.name
                )
                == selected["sha256"]
            ),
        },
        "release_checkpoint_alias": {
            "path": rel(alias),
            "filename": alias.name,
            "sha256": alias_hash,
            "bytes": alias.stat().st_size,
            "byte_identical_to_selected_source": (
                alias_hash == selected["sha256"]
            ),
        },
        "structured_copy": {
            "source_file_count": final_shared_copy["source_file_count"],
            "all_source_files_present_and_byte_identical": (
                final_shared_copy[
                    "all_source_files_present_and_byte_identical"
                ]
            ),
        },
        "inference_smoke_test": smoke,
        "protected_assets": protected,
        "zenodo_scope_verification": zenodo,
        "explicit_non_actions": [
            "No segmentation checkpoint was loaded, copied, or modified.",
            "No legacy regression model was modified.",
            "No validated audit artifact was modified.",
            "No gastruloid asset was modified.",
            "No Napari configuration was modified.",
            "No pre-existing Zenodo file was modified or removed.",
            "The existing release-root neural epoch_139 checkpoint was retained.",
        ],
    }
    write_json(
        DESTINATION_ROOT / "provenance/release_promotion.json",
        promotion_record,
    )

    final_zenodo = zenodo_after_promotion_verification()
    if not final_zenodo["all_assertions_pass"]:
        raise AssertionError(
            "Final promotion record escaped the authorized Zenodo directory"
        )
    promotion_record["zenodo_scope_verification_after_record"] = final_zenodo
    write_json(
        DESTINATION_ROOT / "provenance/release_promotion.json",
        promotion_record,
    )
    return promotion_record


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
def run_evaluation_and_promotion() -> tuple[dict[str, Any], dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the locked candidate evaluation")
    if not PROTECTED_BEFORE.is_file():
        raise FileNotFoundError(PROTECTED_BEFORE)
    if DESTINATION_ROOT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing destination: {DESTINATION_ROOT}"
        )

    checkpoint = candidate_checkpoint()
    contract = training_contract()
    reference_net, reference_keys = load_regression_net(REFERENCE)
    candidate_net, candidate_keys = load_regression_net(checkpoint)
    models = {
        REFERENCE_LABEL: reference_net,
        CANDIDATE_LABEL: candidate_net,
    }

    controlled = {}
    controlled_rows = []
    for profile in ("audit_0208", "physical_02076"):
        profile_result, profile_rows = controlled_evaluation(models, profile)
        controlled[profile] = profile_result
        controlled_rows.extend(profile_rows)
    write_csv(EVALUATION_ROOT / "controlled_events.csv", controlled_rows)

    frozen, frozen_rows = frozen_center_evaluation(models)
    write_csv(EVALUATION_ROOT / "frozen_center_events.csv", frozen_rows)
    del reference_net, candidate_net, models
    gc.collect()
    torch.cuda.empty_cache()

    performance, mean_deltas = performance_assertions(
        controlled, frozen_rows
    )
    protected = protected_asset_verification()
    native_artifacts = native_artifact_verification(checkpoint)
    legacy_checkpoint = (
        REPO
        / "DARE3d_data_190326/Neural_tube_160226/weights/"
        "regression3d_new_set_og/runs/12-01-26/checkpoints/epoch_147.ckpt"
    )
    checkpoint_data = {
        "historical_legacy_not_used_as_scientific_reference": (
            checkpoint_record(legacy_checkpoint)
        ),
        "validated_scientific_reference": checkpoint_record(REFERENCE),
        "fresh_hydra_candidate": checkpoint_record(checkpoint),
        "loaded_network_keys": {
            REFERENCE_LABEL: reference_keys,
            CANDIDATE_LABEL: candidate_keys,
        },
    }

    controlled_angles = [
        row["corrected_nematic_axis_error_deg"]
        for row in controlled_rows
    ]
    controlled_counts_ok = all(
        len(
            [
                row
                for row in controlled_rows
                if row["preprocessing_profile"] == profile
                and row["model"] == model
            ]
        )
        == 80
        for profile in ("audit_0208", "physical_02076")
        for model in (REFERENCE_LABEL, CANDIDATE_LABEL)
    )
    frozen_counts_ok = all(
        len(
            [
                row
                for row in frozen_rows
                if row["profile"] == TRAINING_CONSISTENT
                and row["model"] == model
                and row["mode"] == mode
            ]
        )
        == expected
        for model in (REFERENCE_LABEL, CANDIDATE_LABEL)
        for mode, expected in EXPECTED_CENTER_COUNTS.items()
    )
    candidate_epoch_from_name = int(checkpoint.stem.rsplit("_", 1)[1])
    core_assertions = {
        "legacy_checkpoint_is_preserved_epoch_147": (
            checkpoint_data[
                "historical_legacy_not_used_as_scientific_reference"
            ]["sha256"]
            == "1457585655bc4e11404e10022e87d2049e8408d201f676b5c7c1546290483a6b"
            and checkpoint_data[
                "historical_legacy_not_used_as_scientific_reference"
            ]["epoch"]
            == 147
        ),
        "reference_checkpoint_is_validated_epoch_139": (
            checkpoint_data["validated_scientific_reference"]["sha256"]
            == REFERENCE_SHA256
            and checkpoint_data["validated_scientific_reference"]["epoch"]
            == 139
        ),
        "candidate_checkpoint_differs_from_existing_models": (
            checkpoint_data["fresh_hydra_candidate"]["sha256"]
            not in {
                checkpoint_data[
                    "historical_legacy_not_used_as_scientific_reference"
                ]["sha256"],
                REFERENCE_SHA256,
            }
        ),
        "candidate_epoch_matches_validation_selected_filename": (
            checkpoint_data["fresh_hydra_candidate"]["epoch"]
            == candidate_epoch_from_name
        ),
        "both_comparison_checkpoints_load_all_network_keys": (
            reference_keys == candidate_keys == 41
        ),
        "controlled_event_count_and_pairing_are_locked": (
            len(controlled_rows) == 320 and controlled_counts_ok
        ),
        "controlled_angles_are_finite": all(
            np.isfinite(value) for value in controlled_angles
        ),
        "controlled_angles_are_in_nematic_range": all(
            0.0 <= value <= 90.0 + 1e-5
            for value in controlled_angles
        ),
        "one_regression_prediction_per_frozen_center": (
            len(frozen_rows) == 700 and frozen_counts_ok
        ),
        "frozen_center_evaluation_did_not_run_segmentation": (
            frozen["segmentation_rerun"] is False
        ),
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
    decision = release_decision(assertions, mean_deltas)
    result = {
        "schema_version": 1,
        "status": "evaluation_complete",
        "completed_at": now_iso(),
        "scope": "fresh neural-tube Hydra regression candidate only",
        "comparison_roles": {
            "historical_legacy": (
                "Preserved provenance comparator only; not the scientific "
                "acceptance reference."
            ),
            "validated_epoch_139": (
                "Immutable scientific reference for paired comparisons."
            ),
            "fresh_hydra_candidate": (
                "New validation-selected checkpoint evaluated for release."
            ),
        },
        "acceptance_rule": {
            "normal": "Every original locked assertion must pass.",
            "override": (
                "Per explicit user authorization, only the listed mean/CI "
                "gates may be overridden when both controlled mean "
                "degradations are <= +3.0 degrees and every nonwaivable "
                "assertion passes."
            ),
            "original_locked_criteria_changed": False,
        },
        "decision": decision,
        "release_accepted": decision["promotion_authorized"],
        "release_accepted_by_locked_criteria": decision[
            "locked_criteria_passed"
        ],
        "controlled_unique_movie_M": controlled,
        "frozen_movie_I2_centers": frozen,
        "checkpoints": checkpoint_data,
        "selection": {
            "criterion": "minimum validation val/loss",
            "save_top_k": 1,
            "selected_checkpoint": rel(checkpoint),
            "test_set_used_for_selection": False,
        },
        "training_contract": contract,
        "protected_assets": protected,
        "native_artifacts": native_artifacts,
        "assertions": assertions,
        "all_original_locked_assertions_pass": all(assertions.values()),
    }
    result_path = EVALUATION_ROOT / "result.json"
    write_json(result_path, result)

    override_path = PROVENANCE_ROOT / "release_acceptance_override.json"
    if decision["explicit_release_override_used"]:
        write_json(
            override_path,
            {
                "schema_version": 1,
                "recorded_at": now_iso(),
                "decision": "explicit_user_release_override",
                "original_locked_criteria_passed": False,
                "original_results_and_thresholds_changed": False,
                "authorization": (
                    "The user explicitly authorized neural-tube promotion "
                    "when controlled mean degradation is no more than +3.0 "
                    "degrees and no other important check fails."
                ),
                "controlled_mean_candidate_minus_reference_deg": (
                    mean_deltas
                ),
                "failed_original_locked_assertions": decision[
                    "failed_locked_assertions"
                ],
                "failed_nonwaivable_assertions": decision[
                    "failed_nonwaivable_assertions"
                ],
                "interpretation": (
                    "This is a documented scientific/release override, not "
                    "a claim that the original locked criteria passed."
                ),
            },
        )

    manifest_rows = artifact_manifest()
    manifest_path = PROVENANCE_ROOT / "artifact_manifest_sha256.csv"
    baseline = json.loads(PROTECTED_BEFORE.read_text(encoding="utf-8"))
    run_dirs = sorted(
        path for path in (MODEL_ROOT / "runs").glob("*") if path.is_dir()
    )
    run_id = run_dirs[0].name
    provenance = {
        "schema_version": 1,
        "created_at": now_iso(),
        "scope": "fresh Hydra retraining of the neural-tube regression model only",
        "training_source_commit": baseline["git"]["head"],
        "evaluation_source_commit": git_output("rev-parse", "HEAD"),
        "tracked_diff_at_evaluation": git_output(
            "status", "--short", "--untracked-files=no"
        ),
        "training_command": (
            "python dare3d/train.py "
            "experiment=neural_tube_nematic_regression "
            f"model_root={MODEL_ROOT.as_posix()} "
            f"split_root={STAGING_ROOT.as_posix()} "
            f"date={run_id}"
        ),
        "model_roles": {
            "historical_legacy": checkpoint_data[
                "historical_legacy_not_used_as_scientific_reference"
            ],
            "validated_scientific_reference": checkpoint_data[
                "validated_scientific_reference"
            ],
            "fresh_hydra_candidate": checkpoint_data[
                "fresh_hydra_candidate"
            ],
        },
        "model_root": str(MODEL_ROOT),
        "run_directories": [rel(path) for path in run_dirs],
        "release_destination": str(DESTINATION_ROOT),
        "checkpoint_selection": result["selection"],
        "decision": decision,
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
            "controlled_events": rel(
                EVALUATION_ROOT / "controlled_events.csv"
            ),
            "frozen_center_events": rel(
                EVALUATION_ROOT / "frozen_center_events.csv"
            ),
        },
        "artifact_manifest": {
            "path": rel(manifest_path),
            "sha256": sha256(manifest_path),
            "records": len(manifest_rows),
            "exclusions": [
                "the manifest itself",
                "provenance.json, written after the manifest to avoid a hash cycle",
            ],
        },
        "protected_asset_verification": protected,
        "notes": [
            "No segmentation model was loaded by training or evaluation.",
            "movie_E trained the model; movie_I2 selected the checkpoint.",
            "movie_M was reserved for paired, event-identical evaluation.",
            "The primary audit-compatible scale is XYZ 0.208,0.208,1 um.",
            "Exact movie_M XYZ 0.2076,0.2076,1 um is a sensitivity profile.",
            "The immutable epoch_139 run is a reference, not a source artifact.",
            "The fresh source run is retained after any release promotion.",
            "No pre-existing Zenodo or Napari path is replaced by promotion.",
        ],
    }
    write_json(MODEL_ROOT / "provenance.json", provenance)

    if not decision["promotion_authorized"]:
        raise AssertionError(
            "Candidate rejected; no promotion performed. Failed assertions: "
            f"{decision['failed_locked_assertions']}"
        )
    promotion_record = promote(checkpoint, decision)
    return result, promotion_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = EVALUATION_ROOT / "evaluation.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as stream:
        with contextlib.redirect_stdout(Tee(sys.stdout, stream)):
            with contextlib.redirect_stderr(Tee(sys.stderr, stream)):
                result, promotion = run_evaluation_and_promotion()
    print(
        json.dumps(
            {
                "decision": result["decision"]["outcome"],
                "locked_criteria_passed": result["decision"][
                    "locked_criteria_passed"
                ],
                "promotion_authorized": result["decision"][
                    "promotion_authorized"
                ],
                "source_checkpoint": promotion[
                    "selected_source_checkpoint"
                ],
                "release_checkpoint_alias": promotion[
                    "release_checkpoint_alias"
                ],
                "destination": str(DESTINATION_ROOT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
