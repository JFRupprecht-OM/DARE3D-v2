"""Replay neural-tube regression at the exact released center locations.

This audit-only experiment bypasses segmentation inference. It evaluates the
released and nematic-retrained regression checkpoints on the same movie_I2
image, labels, preprocessing, and three released center sets. Compact caches
make the experiment resumable and no production files are modified.
"""
from __future__ import annotations

import contextlib
import math
import gc
import io
import json
import sys
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

import numpy as np
import torch

import neural_tube_nematic_evaluation as evaluation
import neural_tube_nematic_experiment as common


RELEASED_RUN = (
    common.DATA_ROOT / "weights/regression3d_new_set_og/runs/12-01-26"
)
OUTPUT_ROOT = evaluation.EVALUATION_ROOT / "released_movie_I2_center_replay"
PREDICTION_ROOT = OUTPUT_ROOT / "predictions"
RESULT_PATH = OUTPUT_ROOT / "result.json"
EVENT_PATH = OUTPUT_ROOT / "event_metrics.csv"
MAPPING_PATH = OUTPUT_ROOT / "target_mapping.csv"
REPORT_PATH = OUTPUT_ROOT / "REGRESSION_CENTER_REPLAY.md"
LOG_PATH = OUTPUT_ROOT / "run.log"
RELEASED_STATS_PATH = (
    common.DATA_ROOT
    / "weights/segmentation3d_new_set_og/runs/12-01-26/stats.csv"
)
MODE_SOURCES = {
    "all_groundtruth_centers": RELEASED_RUN
    / "all_true_centers/raw_predictions.npz",
    "matched_groundtruth_centers": RELEASED_RUN
    / "matched_true_centers/raw_predictions.npz",
    "predicted_centers": RELEASED_RUN / "pred_centers/raw_predictions.npz",
}
MODE_RELEASED_STATS = {
    "all_groundtruth_centers":
        "regression_performance_on_all_groundtruth_centers",
    "matched_groundtruth_centers":
        "regression_performance_on_matched_groundtruth_centers",
    "predicted_centers": "regression_performance_on_predicted_centers",
}
EXPECTED_COUNTS = {
    "all_groundtruth_centers": 122,
    "matched_groundtruth_centers": 114,
    "predicted_centers": 114,
}


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


def centers_from_archive(path: Path):
    with np.load(path) as archive:
        saved = {key: np.asarray(archive[key]).copy() for key in archive.files}
    centers = [
        (
            int(np.rint(row[0])),
            float(row[1]),
            float(row[2]),
            float(row[3]),
            float(row[4]),
        )
        for row in saved["centers"]
    ]
    return centers, saved


def predictions_to_arrays(predictions):
    return {
        "centers": np.asarray([item["center"] for item in predictions]),
        "lengths": np.asarray([item["length"] for item in predictions]),
        "quaternions": np.asarray([item["rotation"] for item in predictions]),
    }


def arrays_to_predictions(arrays):
    return [
        {
            "center": arrays["centers"][index],
            "length": arrays["lengths"][index],
            "rotation": arrays["quaternions"][index],
        }
        for index in range(len(arrays["centers"]))
    ]


def atomic_save_predictions(path: Path, predictions) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **predictions_to_arrays(predictions))
    temporary.replace(path)


def load_prediction_cache(path: Path, expected_centers):
    if not path.is_file():
        return None
    with np.load(path) as archive:
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    if set(arrays) != {"centers", "lengths", "quaternions"}:
        raise RuntimeError(f"Unexpected cache fields: {path}")
    if not np.array_equal(
        arrays["centers"], np.asarray(expected_centers, dtype=np.float64)
    ):
        raise RuntimeError(f"Cached centers do not match released centers: {path}")
    return arrays_to_predictions(arrays)


def run_or_load_predictions(model_name, mode, net, dataset, centers):
    path = PREDICTION_ROOT / f"{model_name}_{mode}.npz"
    cached = load_prediction_cache(path, centers)
    if cached is not None:
        print(f"Reusing {path.relative_to(REPO)}")
        return cached, {
            "cache_reused": True,
            "elapsed_seconds": 0.0,
            "path": common.rel(path),
            "sha256": common.sha256(path),
        }
    print(f"Inferring {model_name}, {mode}, N={len(centers)}")
    started = time.perf_counter()
    predictions = evaluation.regression_inference(
        dataset, net, centers, "cuda", output_dir=None
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    atomic_save_predictions(path, predictions)
    return predictions, {
        "cache_reused": False,
        "elapsed_seconds": elapsed,
        "path": common.rel(path),
        "sha256": common.sha256(path),
    }


def quaternion_replay_comparison(generated, released):
    generated = generated.astype(np.float64)
    released = released.astype(np.float64)
    generated /= np.linalg.norm(generated, axis=1, keepdims=True)
    released /= np.linalg.norm(released, axis=1, keepdims=True)
    dots = np.sum(generated * released, axis=1)
    signs = np.where(dots < 0.0, -1.0, 1.0)[:, None]
    aligned = generated * signs
    angular = np.degrees(
        2.0 * np.arccos(np.clip(np.abs(dots), 0.0, 1.0))
    )
    return {
        "sign_invariant_component_absolute_difference": evaluation.describe(
            np.abs(aligned - released).reshape(-1)
        ),
        "sign_invariant_quaternion_distance_deg": evaluation.describe(angular),
    }


def compare_replay(generated_predictions, released):
    generated = predictions_to_arrays(generated_predictions)
    center_difference = np.abs(
        generated["centers"].astype(np.float64)
        - released["centers"].astype(np.float64)
    )
    length_difference = np.abs(
        generated["lengths"].astype(np.float64)
        - released["lengths"].astype(np.float64)
    )
    result = {
        "n": len(generated_predictions),
        "centers_exactly_equal": bool(
            np.array_equal(generated["centers"], released["centers"])
        ),
        "center_absolute_difference": evaluation.describe(
            center_difference.reshape(-1)
        ),
        "length_absolute_difference_voxels": evaluation.describe(
            length_difference.reshape(-1)
        ),
    }
    result.update(
        quaternion_replay_comparison(
            generated["quaternions"], released["quaternions"]
        )
    )
    return result



def analyze_target_mapping(dataset, centers_by_mode):
    rows = []
    summary = {}
    for mode in (
        "all_groundtruth_centers",
        "matched_groundtruth_centers",
    ):
        selected_keys = []
        selected_distances = []
        missing = 0
        ambiguous = 0
        not_nearest = 0
        for index, source_center in enumerate(centers_by_mode[mode]):
            movie_index = int(np.rint(source_center[0]))
            time_index = int(source_center[1])
            source_xyz = np.asarray(source_center[2:], dtype=np.float64)
            candidates = []
            for annotation_index, bipoint in enumerate(
                dataset.movies_bipoints[movie_index][time_index]
            ):
                first, second = (np.asarray(point) for point in bipoint)
                annotation_xyz = (first + second) / 2.0
                distance = float(np.linalg.norm(source_xyz - annotation_xyz))
                eligible = bool(
                    np.all(np.isclose(source_xyz, annotation_xyz, atol=3))
                )
                candidates.append(
                    (annotation_index, annotation_xyz, distance, eligible)
                )
            eligible = [item for item in candidates if item[3]]
            if not eligible:
                missing += 1
                continue
            selected = eligible[0]
            nearest = min(candidates, key=lambda item: item[2])
            ambiguous += len(eligible) > 1
            not_nearest += selected[0] != nearest[0]
            selected_keys.append((movie_index, time_index, selected[0]))
            selected_distances.append(selected[2])
            row = {
                "mode": mode,
                "source_index": index,
                "movie_index": movie_index,
                "source_time": float(source_center[1]),
                "production_time_index_int_truncation": time_index,
                "candidate_count_at_time": len(candidates),
                "eligible_candidate_count": len(eligible),
                "selected_annotation_index": selected[0],
                "selected_distance_voxels": selected[2],
                "selected_is_nearest": selected[0] == nearest[0],
            }
            for prefix, coordinates in (
                ("source", source_xyz),
                ("selected_annotation", selected[1]),
            ):
                for coordinate, value in zip("xyz", coordinates):
                    row[f"{prefix}_{coordinate}"] = float(value)
            if mode == "matched_groundtruth_centers":
                predicted = centers_by_mode["predicted_centers"][index]
                for coordinate, value in zip("txyz", predicted[1:]):
                    row[f"paired_predicted_center_{coordinate}"] = float(value)
            rows.append(row)
        duplicate_count = len(selected_keys) - len(set(selected_keys))
        summary[mode] = {
            "source_count": len(centers_by_mode[mode]),
            "mapped_count": len(selected_keys),
            "missing_count": missing,
            "ambiguous_eligible_count": ambiguous,
            "selected_not_nearest_count": not_nearest,
            "duplicate_selected_target_count": duplicate_count,
            "fractional_time_center_count": sum(
                center[1] != math.floor(center[1])
                for center in centers_by_mode[mode]
            ),
            "selected_distance_voxels":
                evaluation.describe(selected_distances),
        }
    evaluation.write_csv(MAPPING_PATH, rows)
    return summary
def summarize_rows(rows):
    result = {}
    for model_name in (
        "released_saved_predictions",
        "original",
        "retrained",
    ):
        result[model_name] = {}
        for mode in MODE_SOURCES:
            selected = [
                row for row in rows
                if row["model"] == model_name and row["mode"] == mode
            ]
            result[model_name][mode] = {
                "n": len(selected),
                "production_quaternion_error_deg": evaluation.describe(
                    [row["production_quaternion_error_deg"] for row in selected]
                ),
                "corrected_nematic_axis_error_deg": evaluation.describe(
                    [row["corrected_nematic_axis_error_deg"] for row in selected]
                ),
                "absolute_length_error_voxels": evaluation.describe(
                    [row["absolute_length_error_voxels"] for row in selected]
                ),
                "center_distance_voxels": evaluation.describe(
                    [row["center_distance_voxels"] for row in selected]
                ),
                "undefined_corrected_axis_count": sum(
                    not row["corrected_axis_defined"] for row in selected
                ),
            }
    return result


def paired_model_differences(rows):
    result = {}
    for mode in MODE_SOURCES:
        old = sorted(
            (row for row in rows
             if row["model"] == "original" and row["mode"] == mode),
            key=lambda row: row["pair_index"],
        )
        new = sorted(
            (row for row in rows
             if row["model"] == "retrained" and row["mode"] == mode),
            key=lambda row: row["pair_index"],
        )
        if len(old) != len(new):
            raise AssertionError(f"Model row counts differ for {mode}")
        angular = np.asarray([
            newer["corrected_nematic_axis_error_deg"]
            - older["corrected_nematic_axis_error_deg"]
            for older, newer in zip(old, new)
        ])
        length = np.asarray([
            newer["absolute_length_error_voxels"]
            - older["absolute_length_error_voxels"]
            for older, newer in zip(old, new)
        ])
        result[mode] = {
            "retrained_minus_original_corrected_angle_deg":
                evaluation.describe(angular),
            "fraction_angle_improved": float(np.mean(angular < 0.0)),
            "retrained_minus_original_absolute_length_error_voxels":
                evaluation.describe(length),
            "negative_difference_is_improvement": True,
        }
    return result


def released_stats():
    return json.loads(RELEASED_STATS_PATH.read_text(encoding="utf-8"))[
        "regression_results"
    ]


def aggregate_reproduction(summary, archived, model_name):
    result = {}
    differences = []
    for mode, archived_key in MODE_RELEASED_STATS.items():
        generated = summary[model_name][mode]
        reference = archived[archived_key]
        fields = {
            "mean_angle_error":
                generated["production_quaternion_error_deg"]["mean"],
            "std_angle_error":
                generated["production_quaternion_error_deg"]["std_population"],
            "mean_length_error":
                generated["absolute_length_error_voxels"]["mean"],
            "std_length_error":
                generated["absolute_length_error_voxels"]["std_population"],
            "mean_distance_error":
                generated["center_distance_voxels"]["mean"],
            "std_distance_error":
                generated["center_distance_voxels"]["std_population"],
        }
        by_field = {}
        for field, generated_value in fields.items():
            difference = abs(float(generated_value) - float(reference[field]))
            differences.append(difference)
            by_field[field] = {
                "generated": generated_value,
                "released": reference[field],
                "absolute_difference": difference,
            }
        by_field["n"] = {
            "generated": generated["n"],
            "released": reference["n"],
            "absolute_difference": abs(generated["n"] - reference["n"]),
        }
        result[mode] = by_field
    maximum = max(differences)
    return {
        "by_mode": result,
        "maximum_absolute_difference": maximum,
        "within_1e_minus_5": maximum <= 1e-5,
        "within_1e_minus_3": maximum <= 1e-3,
    }


def write_report(result):
    lines = [
        "# Neural-tube regression replay at released centers",
        "",
        "This evaluation bypasses segmentation inference. movie_I2 is the released",
        "validation/reported set, so this is a historical replay rather than an",
        "independent-test estimate. Predicted-center rows are scored against the",
        "annotation associated with each matched true center.",
        "",
        "| Center mode | N | Original + production | Original + nematic | Retrained + nematic |",
        "|---|---:|---:|---:|---:|",
    ]
    summary = result["summary"]
    for mode in MODE_SOURCES:
        old = summary["original"][mode]
        new = summary["retrained"][mode]
        lines.append(
            "| {} | {} | {:.3f} +/- {:.3f} deg | {:.3f} +/- {:.3f} deg | {:.3f} +/- {:.3f} deg |".format(
                mode.replace("_", " "),
                old["n"],
                old["production_quaternion_error_deg"]["mean"],
                old["production_quaternion_error_deg"]["std_population"],
                old["corrected_nematic_axis_error_deg"]["mean"],
                old["corrected_nematic_axis_error_deg"]["std_population"],
                new["corrected_nematic_axis_error_deg"]["mean"],
                new["corrected_nematic_axis_error_deg"]["std_population"],
            )
        )
    lines.extend([
        "",
        "## Replay checks",
        "",
        "- Saved prediction artifacts, maximum aggregate difference: "
        f"{result['saved_artifact_aggregate_reproduction']['maximum_absolute_difference']:.9g}.",
        "- Checkpoint replay, maximum aggregate difference: "
        f"{result['checkpoint_aggregate_reproduction']['maximum_absolute_difference']:.9g}.",
        "- Checkpoint replay agreement within 0.01: "
        f"{result['checkpoint_aggregate_reproduction']['within_1e_minus_2']}.",
        "- Event-level values and provenance are in event_metrics.csv and result.json.",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def execute():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exact checkpoint replay")
    for path in (*MODE_SOURCES.values(), RELEASED_STATS_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    centers = {}
    released_predictions = {}
    for mode, path in MODE_SOURCES.items():
        centers[mode], released_predictions[mode] = centers_from_archive(path)
        if len(centers[mode]) != EXPECTED_COUNTS[mode]:
            raise AssertionError(
                f"Expected {EXPECTED_COUNTS[mode]} centers for {mode}, "
                f"got {len(centers[mode])}"
            )

    dataset = common.make_dataset("regression", "validation")
    dataset.init(preprocess=False)
    original_shapes = [list(movie.shape) for movie in dataset.movies_im]
    dataset.pad_images()
    dataset._normalize(dataset.renorm)
    padded_shapes = [list(movie.shape) for movie in dataset.movies_im]

    truth_all = dataset.gather_groundtruth_info(
        centers["all_groundtruth_centers"]
    )
    truth_matched = dataset.gather_groundtruth_info(
        centers["matched_groundtruth_centers"]
    )
    if any(item is None for item in truth_all + truth_matched):
        raise RuntimeError(
            "At least one released center could not be linked to an annotation"
        )
    truth_by_mode = {
        "all_groundtruth_centers": truth_all,
        "matched_groundtruth_centers": truth_matched,
        "predicted_centers": truth_matched,
    }
    target_mapping = analyze_target_mapping(dataset, centers)

    all_predictions = {
        "released_saved_predictions": {
            mode: arrays_to_predictions(released_predictions[mode])
            for mode in MODE_SOURCES
        }
    }
    inference = {}
    loaded_keys = {}
    replay = {}
    started = time.perf_counter()
    for model_name in ("original", "retrained"):
        net, loaded_keys[model_name] = evaluation.load_regression_net(model_name)
        all_predictions[model_name] = {}
        inference[model_name] = {}
        for mode in MODE_SOURCES:
            predictions, record = run_or_load_predictions(
                model_name, mode, net, dataset, centers[mode]
            )
            all_predictions[model_name][mode] = predictions
            inference[model_name][mode] = record
            if model_name == "original":
                replay[mode] = compare_replay(
                    predictions, released_predictions[mode]
                )
        del net
        gc.collect()
        torch.cuda.empty_cache()

    rows = []
    for model_name in ("released_saved_predictions", "original", "retrained"):
        for mode in MODE_SOURCES:
            mode_rows = evaluation.pair_rows(
                f"{model_name}_regression_released_centers",
                mode,
                list(zip(truth_by_mode[mode], all_predictions[model_name][mode])),
            )
            for row, truth, prediction in zip(
                mode_rows,
                truth_by_mode[mode],
                all_predictions[model_name][mode],
            ):
                row["scope"] = "movie_I2_released_center_regression_replay"
                row["model"] = model_name
                row["center_source_archive"] = common.rel(MODE_SOURCES[mode])
                for prefix, item in (("target", truth), ("inference", prediction)):
                    center = np.asarray(item["center"], dtype=np.float64)
                    for coordinate, value in zip("mtxyz", center):
                        row[f"{prefix}_center_{coordinate}"] = float(value)
            rows.extend(mode_rows)
    if any(not row["evaluated"] for row in rows):
        raise AssertionError("Unexpected unevaluable regression row")
    evaluation.write_csv(EVENT_PATH, rows)
    summary = summarize_rows(rows)
    archived = released_stats()
    saved_aggregate = aggregate_reproduction(
        summary, archived, "released_saved_predictions"
    )
    checkpoint_aggregate = aggregate_reproduction(
        summary, archived, "original"
    )
    checkpoint_aggregate["within_1e_minus_2"] = (
        checkpoint_aggregate["maximum_absolute_difference"] <= 1e-2
    )

    assertions = {
        "source_commit_unchanged": common.git_head() == common.SOURCE_COMMIT,
        "tracked_tree_clean": common.tracked_tree_is_clean(),
        "center_counts_match_archives": all(
            len(centers[mode]) == EXPECTED_COUNTS[mode]
            for mode in MODE_SOURCES
        ),
        "all_released_centers_linked_to_annotations": True,
        "target_mapping_is_unique_unambiguous_and_nearest": all(
            item["missing_count"] == 0
            and item["ambiguous_eligible_count"] == 0
            and item["selected_not_nearest_count"] == 0
            and item["duplicate_selected_target_count"] == 0
            for item in target_mapping.values()
        ),
        "saved_artifact_aggregate_reproduced_within_1e_minus_5":
            saved_aggregate["within_1e_minus_5"],
        "checkpoint_replay_reproduced_within_1e_minus_2":
            checkpoint_aggregate["within_1e_minus_2"],
    }
    result = {
        "status": (
            "complete" if all(assertions.values())
            else "complete_with_failed_assertions"
        ),
        "completed_at": common.now_iso(),
        "scope":
            "segmentation-free regression replay on released movie_I2 centers",
        "scientific_role":
            "historical validation/reported-set replay, not independent test",
        "target_rule": (
            "Predicted-center rows retain the matched true event annotation as "
            "the regression target; truth is never selected from the predicted "
            "location."
        ),
        "pairing_provenance": {
            "center_construction": "dare3d.metrics.infer_measure.CenterList",
            "target_lookup": (
                "dare3d.data.components.regress_3dataset."
                "Regress3Dataset.gather_groundtruth_info"
            ),
            "inference": "dare3d.metrics.inference.regression_inference",
        },
        "dataset": {
            "role": "validation",
            "movie_names": list(dataset.movie_names),
            "init_preprocess": False,
            "padding_then_normalization": True,
            "normalization": str(dataset.renorm),
            "original_internal_shapes_txyz": original_shapes,
            "padded_internal_shapes_txyz": padded_shapes,
        },
        "center_sources": {
            mode: {
                "path": common.rel(path),
                "sha256": common.sha256(path),
                "n": len(centers[mode]),
            }
            for mode, path in MODE_SOURCES.items()
        },
        "target_mapping": target_mapping,
        "checkpoints": {
            model: evaluation.checkpoint_metadata("regression", model)
            for model in ("original", "retrained")
        },
        "loaded_network_keys": loaded_keys,
        "inference": inference,
        "original_prediction_replay": replay,
        "released_aggregate_stats": archived,
        "saved_artifact_aggregate_reproduction": saved_aggregate,
        "checkpoint_aggregate_reproduction": checkpoint_aggregate,
        "summary": summary,
        "paired_retrained_minus_original":
            paired_model_differences(rows),
        "assertions": assertions,
        "outputs": {
            "event_metrics": common.rel(EVENT_PATH),
            "target_mapping": common.rel(MAPPING_PATH),
            "report": common.rel(REPORT_PATH),
            "run_log": common.rel(LOG_PATH),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    common.atomic_write_json(RESULT_PATH, result)
    write_report(result)
    result["outputs"]["report_sha256"] = common.sha256(REPORT_PATH)
    result["outputs"]["event_metrics_sha256"] = common.sha256(EVENT_PATH)
    result["outputs"]["target_mapping_sha256"] = common.sha256(MAPPING_PATH)
    common.atomic_write_json(RESULT_PATH, result)
    return result


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8", buffering=1) as log_stream:
        stdout = Tee(sys.stdout, log_stream)
        stderr = Tee(sys.stderr, log_stream)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            print(f"\n[{common.now_iso()}] regression center replay", flush=True)
            try:
                result = execute()
                print(json.dumps({
                    "status": result["status"],
                    "checkpoint_maximum_absolute_difference":
                        result["checkpoint_aggregate_reproduction"][
                            "maximum_absolute_difference"
                        ],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "result": common.rel(RESULT_PATH),
                }, indent=2), flush=True)
            except BaseException:
                common.atomic_write_json(
                    OUTPUT_ROOT / "failure.json",
                    {
                        "status": "failed",
                        "updated_at": common.now_iso(),
                        "traceback": traceback.format_exc(),
                    },
                )
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
