"""Recover and audit released DARE3D regression artifacts without inference.

This script treats saved TIFF/NPZ outputs as immutable evidence.  It independently
decodes rendered division axes, reconstructs per-event metrics where possible,
and distinguishes the production full-quaternion error from an unoriented
(nematic) division-axis error.
"""
from __future__ import annotations

import csv
import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage
from skimage.measure import regionprops

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dare3d.data.components.angles3d import get_quaternion
from dare3d.losses.angle3d import quaternion_error

DATA = REPO / "DARE3d_data_190326"
EVIDENCE = HERE / "evidence"
GASTRO_RUN = DATA / "Gastruloid_241025/weights/regression3d_exp10-b/runs/01-01"
GASTRO_STATS = DATA / "Gastruloid_241025/weights/segmentation3d_exp10-b/runs/01-01/stats.csv"
GASTRO_LABEL = DATA / "Gastruloid_241025/trainingset/movie2/label/movie2.tif"
NEURAL_RUN = DATA / "Neural_tube_160226/regression3d_new_set_og/runs/12-01-26"
NEURAL_STATS = DATA / "Neural_tube_160226/segmentation3d_new_set_og/runs/12-01-26/stats.csv"
OBJECTS = EVIDENCE / "nuclei_current_evaluator_objects.csv"
ALIGNMENT = EVIDENCE / "nuclei_saved_visualization_alignment.csv"

REPORT_OUT = EVIDENCE / "regression_artifact_analysis.json"
EVENTS_OUT = EVIDENCE / "regression_event_metrics.csv"
VALIDATION_OUT = EVIDENCE / "regression_raster_decoder_validation.csv"
MISSING_OUT = EVIDENCE / "regression_missing_or_unrecoverable.csv"

MODES = {
    "all_groundtruth_centers": "all_true_centers",
    "matched_groundtruth_centers": "matched_true_centers",
    "predicted_centers": "pred_centers",
}
STATS_KEYS = {mode: f"regression_performance_on_{mode}" for mode in MODES}


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
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=serial) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_stats(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["regression_results"]


def describe(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return {"n": 0}
    return {
        "n": len(a),
        "mean": float(np.mean(a)),
        "std_population": float(np.std(a)),
        "sem_using_population_std": float(np.std(a) / math.sqrt(len(a))),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
    }


def production_stats(values) -> dict:
    a = np.asarray(values)
    return {
        "n_effective": len(a),
        "mean_float32": float(np.mean(a, dtype=np.float32)),
        "std_float32": float(np.std(a, dtype=np.float32)),
    }


def unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def nematic_error(axis_a, axis_b) -> float:
    dot = float(np.dot(unit(axis_a), unit(axis_b)))
    return float(np.degrees(np.arccos(np.clip(abs(dot), 0.0, 1.0))))


def axis_from_quaternion(quaternion) -> np.ndarray:
    q = unit(quaternion)
    sine = math.sqrt(max(0.0, 1.0 - float(q[0]) ** 2))
    if sine < 1e-12:
        raise ValueError("Rendered axis is undefined for an identity quaternion")
    return unit(q[1:] / sine)


def axis_quaternion(axis) -> np.ndarray:
    axis = unit(axis)
    return get_quaternion(-axis, axis)


def load_movie(path: Path):
    return tifffile.memmap(path)


def decode_line(movie_tzyx, center, radius: int = 24) -> dict:
    """Decode the nearest 26-connected rendered line around an MTXYZ center."""
    _, t, x, y, z = np.asarray(center, dtype=float)
    frame = movie_tzyx[int(t)]
    target_zyx = np.asarray([z, y, x], dtype=float)
    rounded = np.rint(target_zyx).astype(int)
    low = np.maximum(rounded - radius, 0)
    high = np.minimum(rounded + radius + 1, frame.shape)
    crop = np.asarray(frame[tuple(slice(a, b) for a, b in zip(low, high))]) > 0
    labels, count = ndimage.label(crop, structure=np.ones((3, 3, 3), dtype=np.uint8))
    candidates = []
    for component in range(1, count + 1):
        local = np.argwhere(labels == component)
        if len(local) < 2:
            continue
        xyz = (local + low)[..., ::-1].astype(np.float64)
        centroid = np.mean(xyz, axis=0)
        centered = xyz - centroid
        _, singular, vh = np.linalg.svd(centered, full_matrices=False)
        axis = unit(vh[0])
        projection = centered @ axis
        first = xyz[int(np.argmin(projection))]
        last = xyz[int(np.argmax(projection))]
        quality = float(singular[0] ** 2 / max(singular[1] ** 2, 1e-12))
        candidates.append(
            {
                "axis": axis,
                "span": float(np.ptp(projection)),
                "midpoint_plus_half_voxel": (first + last) / 2.0 + 0.5,
                "component_voxels": len(xyz),
                "component_center_distance": float(
                    np.linalg.norm(centroid - np.asarray([x, y, z]))
                ),
                "principal_to_second_variance_ratio": quality,
            }
        )
    if not candidates:
        raise RuntimeError(f"No rendered line near center {center}")
    return min(candidates, key=lambda item: item["component_center_distance"])


def mode_summary(rows: list[dict], released: dict) -> dict:
    evaluated = [row for row in rows if row["evaluated"]]
    result = {
        "nominal_n_reported": int(released["n"]),
        "row_count": len(rows),
        "effective_n": len(evaluated),
        "skipped_n": len(rows) - len(evaluated),
        "released_production_full_quaternion": released,
    }
    for name in (
        "nematic_axis_error_deg",
        "approximate_quaternion_error_deg",
        "approximate_length_error_voxels",
        "exact_center_error_voxels",
        "approximate_center_error_voxels",
    ):
        values = [row[name] for row in evaluated if row.get(name) not in (None, "")]
        if values:
            result[name] = describe(values)
            result[name + "_production_float32"] = production_stats(values)
    return result


def analyse_neural(released: dict, event_rows: list[dict], validation_rows: list[dict]):
    groundtruth_path = NEURAL_RUN / "groundtruth/movie_I2.tif"
    groundtruth = load_movie(groundtruth_path)
    summaries = {}
    sources = [groundtruth_path]
    for mode, folder in MODES.items():
        mode_dir = NEURAL_RUN / folder
        npz_path = mode_dir / "raw_predictions.npz"
        render_path = mode_dir / "movie_I2.tif"
        sources.extend([npz_path, render_path])
        saved = np.load(npz_path)
        centers = np.asarray(saved["centers"], dtype=np.float64)
        lengths = np.asarray(saved["lengths"], dtype=np.float64).reshape(-1)
        quaternions = np.asarray(saved["quaternions"], dtype=np.float64)
        render = load_movie(render_path)
        assert len(centers) == len(lengths) == len(quaternions)
        rows = []
        for index, (center, length, quaternion) in enumerate(
            zip(centers, lengths, quaternions)
        ):
            predicted_axis = axis_from_quaternion(quaternion)
            decoded_prediction = decode_line(render, center)
            decoded_truth = decode_truth_line(groundtruth, center)
            decoder_angle = nematic_error(
                predicted_axis, decoded_prediction["axis"]
            )
            span_delta = decoded_prediction["span"] - float(length)
            row = {
                "dataset": "neural_tube_membrane",
                "mode": mode,
                "index": index,
                "evaluated": True,
                "center_t": float(center[1]),
                "center_x": float(center[2]),
                "center_y": float(center[3]),
                "center_z": float(center[4]),
                "nematic_axis_error_deg": nematic_error(
                    decoded_truth["axis"], predicted_axis
                ),
                "approximate_quaternion_error_deg": quaternion_error(
                    axis_quaternion(decoded_truth["axis"]), quaternion
                ),
                "approximate_length_error_voxels": abs(
                    decoded_truth["span"] - float(length)
                ),
                "approximate_center_error_voxels": float(
                    np.linalg.norm(
                        decoded_truth["midpoint_plus_half_voxel"] - center[2:]
                    )
                ),
                "predicted_length_npz": float(length),
                "predicted_raster_span": decoded_prediction["span"],
                "truth_raster_span": decoded_truth["span"],
                "predicted_decode_center_distance": decoded_prediction[
                    "component_center_distance"
                ],
                "predicted_decode_quality_ratio": decoded_prediction[
                    "principal_to_second_variance_ratio"
                ],
                "truth_decode_center_distance": decoded_truth[
                    "component_center_distance"
                ],
                "truth_decode_quality_ratio": decoded_truth[
                    "principal_to_second_variance_ratio"
                ],
                "truth_rendered_time_offset": decoded_truth["rendered_time_offset"],
                "metric_basis": "released NPZ prediction plus independently decoded ground-truth raster",
            }
            rows.append(row)
            event_rows.append(row)
            validation_rows.append(
                {
                    "dataset": "neural_tube_membrane",
                    "mode": mode,
                    "index": index,
                    "quaternion_axis_vs_rendered_axis_deg": decoder_angle,
                    "rendered_span_minus_npz_length": span_delta,
                    "rendered_component_center_distance": decoded_prediction[
                        "component_center_distance"
                    ],
                    "rendered_quality_ratio": decoded_prediction[
                        "principal_to_second_variance_ratio"
                    ],
                }
            )
        released_mode = released[STATS_KEYS[mode]]
        summary = mode_summary(rows, released_mode)
        decoded_angles = [
            item["quaternion_axis_vs_rendered_axis_deg"]
            for item in validation_rows
            if item["mode"] == mode
        ]
        decoded_lengths = [
            item["rendered_span_minus_npz_length"]
            for item in validation_rows
            if item["mode"] == mode
        ]
        summary["raster_decoder_calibration"] = {
            "axis_error_deg": describe(decoded_angles),
            "span_minus_npz_length": describe(decoded_lengths),
        }
        summaries[mode] = summary
    return {
        "classification": "aggregate released-output traceability; not raw-data replay",
        "modes": summaries,
        "source_artifacts": [rel(path) for path in sources],
        "limitations": [
            "Raw neural-tube images and daughter-cell annotations are absent.",
            "The ground-truth axis and length are decoded from a uint8 line raster.",
            "For 37/350 mode events, no line exists at the requested frame; the decoder uses the nearest line at the adjacent all-GT rendered frame, reflecting temporal-centroid rounding and further limiting independence.",
            "The full-quaternion reconstruction is approximate because a line raster discards rotation about the displayed axis.",
        ],
    }


def load_gastruloid_bipoints() -> list[list[tuple[np.ndarray, np.ndarray]]]:
    label_tzyx = tifffile.imread(GASTRO_LABEL)
    result = []
    for frame in label_tzyx:
        props = {int(item.label): np.asarray(item.centroid)[::-1] for item in regionprops(frame)}
        values = set(props)
        pairs = []
        for value in sorted(values):
            if value % 2 == 1 and value + 1 in values:
                pairs.append((props[value], props[value + 1]))
        result.append(pairs)
    return result


def centers_from_detection_evidence() -> dict[str, list[np.ndarray]]:
    objects = read_rows(OBJECTS)
    true = {
        int(row["index"]): np.asarray(
            [0, float(row["t"]), float(row["x"]), float(row["y"]), float(row["z"])],
            dtype=np.float64,
        )
        for row in objects
        if row["kind"] == "true"
    }
    predicted = {
        int(row["index"]): np.asarray(
            [0, float(row["t"]), float(row["x"]), float(row["y"]), float(row["z"])],
            dtype=np.float64,
        )
        for row in objects
        if row["kind"] == "pred"
    }
    alignment = sorted(
        read_rows(ALIGNMENT), key=lambda row: int(row["match_id"])
    )
    pairs = [
        (int(row["replay_true_index"]), int(row["replay_pred_index"]))
        for row in alignment
        if row["saved_row_present"] == "True"
    ]
    return {
        "all_groundtruth_centers": [
            np.asarray([0, *np.rint(true[index][1:])], dtype=np.float64)
            for index in sorted(true)
        ],
        "matched_groundtruth_centers": [true[i] for i, _ in pairs],
        "predicted_centers": [
            np.asarray([0, *np.round(predicted[j][1:] + 1e-9)], dtype=np.float64)
            for _, j in pairs
        ],
    }


def gather_annotation(center, bipoints):
    _, t, x, y, z = center
    for first, second in bipoints[int(t)]:
        actual_center = (first + second) / 2.0
        if np.all(
            np.isclose(
                np.asarray([x, y, z]), actual_center, atol=3
            )
        ):
            return first, second
    return None


def analyse_gastruloid(
    released: dict, event_rows: list[dict], missing_rows: list[dict]
):
    bipoints = load_gastruloid_bipoints()
    centers_by_mode = centers_from_detection_evidence()
    summaries = {}
    sources = [GASTRO_LABEL, OBJECTS, ALIGNMENT]
    for mode, folder in MODES.items():
        render_path = GASTRO_RUN / folder / "movie2.tif"
        sources.append(render_path)
        render = load_movie(render_path)
        rows = []
        for index, center in enumerate(centers_by_mode[mode]):
            decoded = decode_line(render, center)
            annotation_center = (
                centers_by_mode["matched_groundtruth_centers"][index]
                if mode == "predicted_centers"
                else center
            )
            annotation = gather_annotation(annotation_center, bipoints)
            base = {
                "dataset": "gastruloid_nuclei",
                "mode": mode,
                "index": index,
                "center_t": float(center[1]),
                "center_x": float(center[2]),
                "center_y": float(center[3]),
                "center_z": float(center[4]),
                "annotation_lookup_t": float(annotation_center[1]),
                "annotation_lookup_x": float(annotation_center[2]),
                "annotation_lookup_y": float(annotation_center[3]),
                "annotation_lookup_z": float(annotation_center[4]),
                "predicted_raster_span": decoded["span"],
                "predicted_decode_center_distance": decoded[
                    "component_center_distance"
                ],
                "predicted_decode_quality_ratio": decoded[
                    "principal_to_second_variance_ratio"
                ],
                "metric_basis": "released prediction raster plus released raw daughter-label centroids",
            }
            if annotation is None:
                row = {
                    **base,
                    "evaluated": False,
                    "skip_reason": "production gather_groundtruth_info found no label pair within per-axis atol=3",
                }
                missing_rows.append(
                    {
                        "dataset": "gastruloid_nuclei",
                        "mode": mode,
                        "index": index,
                        "center": ",".join(str(float(value)) for value in center),
                        "reason": row["skip_reason"],
                    }
                )
            else:
                first, second = annotation
                true_axis = unit(second - first)
                true_length = float(np.linalg.norm(second - first))
                actual_center = (first + second) / 2.0
                row = {
                    **base,
                    "evaluated": True,
                    "nematic_axis_error_deg": nematic_error(
                        true_axis, decoded["axis"]
                    ),
                    "approximate_length_error_voxels": abs(
                        true_length - decoded["span"]
                    ),
                    "exact_center_error_voxels": float(
                        np.linalg.norm(
                            np.asarray(
                                [0, annotation_center[1], *actual_center]
                            )
                            - center
                        )
                    ),
                    "truth_length_from_label_centroids": true_length,
                    "truth_center_x": float(actual_center[0]),
                    "truth_center_y": float(actual_center[1]),
                    "truth_center_z": float(actual_center[2]),
                    "approximate_quaternion_error_deg": "",
                    "approximate_center_error_voxels": "",
                }
            rows.append(row)
            event_rows.append(row)
        summaries[mode] = mode_summary(rows, released[STATS_KEYS[mode]])
    return {
        "classification": "released-output and raw-annotation reconstruction",
        "modes": summaries,
        "source_artifacts": [rel(path) for path in sources],
        "limitations": [
            "The released Gastruloid regression run has no raw_predictions.npz files.",
            "The saved uint8 line retains an unoriented axis and approximate length, but not the predicted quaternion rotation-angle degree of freedom.",
            "Therefore production full-quaternion angle errors cannot be independently reconstructed event by event for this dataset.",
        ],
    }


def decode_truth_line(movie_tzyx, center) -> dict:
    """Decode at the requested frame, falling back to one adjacent saved frame."""
    try:
        result = decode_line(movie_tzyx, center)
        result["rendered_time_offset"] = 0
        return result
    except RuntimeError:
        candidates = []
        for offset in (-1, 1):
            shifted = np.asarray(center, dtype=np.float64).copy()
            shifted[1] += offset
            if 0 <= shifted[1] < movie_tzyx.shape[0]:
                try:
                    decoded = decode_line(movie_tzyx, shifted)
                    decoded["rendered_time_offset"] = offset
                    candidates.append(decoded)
                except RuntimeError:
                    pass
        if not candidates:
            raise
        return min(candidates, key=lambda item: item["component_center_distance"])


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    angle_origin = Path(inspect.getsourcefile(get_quaternion)).resolve()
    error_origin = Path(inspect.getsourcefile(quaternion_error)).resolve()
    if not angle_origin.is_relative_to(REPO) or not error_origin.is_relative_to(REPO):
        raise RuntimeError(
            f"dare3d import escaped checkout: {angle_origin}, {error_origin}"
        )

    released_gastro = read_stats(GASTRO_STATS)
    released_neural = read_stats(NEURAL_STATS)
    event_rows = []
    validation_rows = []
    missing_rows = []
    neural = analyse_neural(released_neural, event_rows, validation_rows)
    gastro = analyse_gastruloid(released_gastro, event_rows, missing_rows)

    decoder_angles = [
        row["quaternion_axis_vs_rendered_axis_deg"] for row in validation_rows
    ]
    decoder_span_delta = [
        row["rendered_span_minus_npz_length"] for row in validation_rows
    ]
    neural_truth_time_offsets = [
        int(row["truth_rendered_time_offset"])
        for row in event_rows
        if row["dataset"] == "neural_tube_membrane"
    ]
    neural_truth_time_offset_counts = {
        str(offset): neural_truth_time_offsets.count(offset) for offset in (-1, 0, 1)
    }
    missing_by_mode = {
        mode: [
            row["index"]
            for row in missing_rows
            if row["mode"] == mode
        ]
        for mode in MODES
    }

    def independent_mean(dataset, mode):
        return dataset["modes"][mode]["nematic_axis_error_deg"]["mean"]

    def released_mean(dataset, mode):
        return float(
            dataset["modes"][mode][
                "released_production_full_quaternion"
            ]["mean_angle_error"]
        )

    cross_mode = {}
    for name, dataset in (("gastruloid_nuclei", gastro), ("neural_tube_membrane", neural)):
        all_mode = "all_groundtruth_centers"
        matched_mode = "matched_groundtruth_centers"
        predicted_mode = "predicted_centers"
        cross_mode[name] = {
            "released_full_quaternion_predicted_minus_all_mean_deg": (
                released_mean(dataset, predicted_mode)
                - released_mean(dataset, all_mode)
            ),
            "released_full_quaternion_predicted_minus_matched_mean_deg": (
                released_mean(dataset, predicted_mode)
                - released_mean(dataset, matched_mode)
            ),
            "independent_nematic_predicted_minus_all_mean_deg": (
                independent_mean(dataset, predicted_mode)
                - independent_mean(dataset, all_mode)
            ),
            "independent_nematic_predicted_minus_matched_mean_deg": (
                independent_mean(dataset, predicted_mode)
                - independent_mean(dataset, matched_mode)
            ),
            "end_to_end_mode": predicted_mode,
        }

    gastro_all_released = released_gastro[STATS_KEYS["all_groundtruth_centers"]]
    neural_all_released = released_neural[STATS_KEYS["all_groundtruth_centers"]]
    assertions = {
        "imports_resolve_to_current_checkout": True,
        "event_row_count_is_743": len(event_rows) == 743,
        "neural_decoder_validation_count_is_350": len(validation_rows) == 350,
        "neural_truth_time_offsets_total_350": len(neural_truth_time_offsets) == 350,
        "neural_truth_next_frame_fallback_count_is_37": neural_truth_time_offset_counts == {"-1": 0, "0": 313, "1": 37},
        "neural_decoder_axis_mean_below_1_3_deg": np.mean(decoder_angles) < 1.3,
        "neural_decoder_axis_max_below_6_deg": np.max(decoder_angles) < 6.0,
        "neural_decoder_all_quality_ratios_at_least_20": all(
            row["rendered_quality_ratio"] >= 20 for row in validation_rows
        ),
        "neural_mode_counts_match_122_114_114": [
            neural["modes"][mode]["row_count"] for mode in MODES
        ] == [122, 114, 114],
        "neural_all_rows_are_effective": all(
            neural["modes"][mode]["effective_n"]
            == neural["modes"][mode]["row_count"]
            for mode in MODES
        ),
        "gastruloid_mode_counts_match_145_124_124": [
            gastro["modes"][mode]["row_count"] for mode in MODES
        ] == [145, 124, 124],
        "gastruloid_effective_counts_match_140_121_121": [
            gastro["modes"][mode]["effective_n"] for mode in MODES
        ] == [140, 121, 121],
        "gastruloid_missing_indices_match": missing_by_mode
        == {
            "all_groundtruth_centers": [49, 54, 60, 69, 105],
            "matched_groundtruth_centers": [35, 52, 62],
            "predicted_centers": [35, 52, 62],
        },
        "gastruloid_exact_center_mean_replays_released": all(
            abs(
                gastro["modes"][mode]["exact_center_error_voxels_production_float32"][
                    "mean_float32"
                ]
                - released_gastro[STATS_KEYS[mode]]["mean_distance_error"]
            )
            < 1e-5
            for mode in MODES
        ),
        "gastruloid_exact_center_std_replays_released": all(
            abs(
                gastro["modes"][mode]["exact_center_error_voxels_production_float32"][
                    "std_float32"
                ]
                - released_gastro[STATS_KEYS[mode]]["std_distance_error"]
            )
            < 1e-5
            for mode in MODES
        ),
        "gastruloid_prediction_rasters_all_decode": sum(
            row["dataset"] == "gastruloid_nuclei" for row in event_rows
        )
        == 393,
        "gastruloid_raw_npz_absent": not any(GASTRO_RUN.rglob("raw_predictions.npz")),
        "neural_three_raw_npz_present": len(list(NEURAL_RUN.rglob("raw_predictions.npz")))
        == 3,
        "neural_raw_input_and_annotations_absent": not any(
            path
            for path in (DATA / "Neural_tube_160226").rglob("*.tif")
            if "runs" not in path.parts
        ),
    }
    failed = [name for name, passed in assertions.items() if not passed]
    if failed:
        raise AssertionError(f"Audit assertions failed: {failed}")

    report = {
        "classification": "saved regression artifact audit; no inference or retraining",
        "runtime_import_origins": {
            "get_quaternion": rel(angle_origin),
            "quaternion_error": rel(error_origin),
        },
        "metric_definitions": {
            "production_angle_error": (
                "2*acos(abs(dot(q_true,q_pred))) in degrees after quaternion normalization; "
                "dot is clipped to +/-0.9999, so identical quaternions have a 1.620583-degree floor"
            ),
            "independent_axis_error": (
                "acos(abs(dot(unit_axis_true,unit_axis_pred))) in degrees; "
                "an unoriented/nematic 3D division-axis error in [0,90]"
            ),
            "production_reported_n": (
                "evaluate_center_pair reports len(center_pair), although it skips pairs whose "
                "truth or prediction is None before computing every mean and standard deviation"
            ),
            "center_modes": {
                "all_groundtruth_centers": "regressor sampled at every detected ground-truth event center",
                "matched_groundtruth_centers": "regressor sampled at ground-truth centers for detector-matched events",
                "predicted_centers": "regressor sampled at matched detector predictions; the end-to-end mode",
            },
        },
        "raster_decoder_calibration_against_neural_npz": {
            "count": len(validation_rows),
            "axis_error_deg": describe(decoder_angles),
            "rendered_span_minus_npz_length": describe(decoder_span_delta),
            "acceptance_rule": "nearest line component within crop; principal/second variance ratio >=20",
        },
        "neural_groundtruth_raster_time_alignment": {
            "requested_frame": neural_truth_time_offset_counts["0"],
            "previous_frame_fallback": neural_truth_time_offset_counts["-1"],
            "next_frame_fallback": neural_truth_time_offset_counts["1"],
            "interpretation": (
                "The saved ground-truth TIFF was rendered from rounded all-GT centers. "
                "For matched/predicted mode centers, temporal centroids can select the "
                "adjacent annotation frame; raw neural annotations are unavailable."
            ),
        },
        "gastruloid_nuclei": gastro,
        "neural_tube_membrane": neural,
        "cross_mode_comparisons": cross_mode,
        "manuscript_28_degree_claim_check": {
            "gastruloid_released_all_gt_mean_angle_error_deg": gastro_all_released[
                "mean_angle_error"
            ],
            "gastruloid_released_all_gt_std_angle_error_deg": gastro_all_released[
                "std_angle_error"
            ],
            "gastruloid_released_sd_over_sqrt_nominal_n_deg": (
                gastro_all_released["std_angle_error"]
                / math.sqrt(gastro_all_released["n"])
            ),
            "gastruloid_released_sd_over_sqrt_effective_n_deg": (
                gastro_all_released["std_angle_error"] / math.sqrt(140)
            ),
            "neural_released_all_gt_mean_angle_error_deg": neural_all_released[
                "mean_angle_error"
            ],
            "neural_released_all_gt_std_angle_error_deg": neural_all_released[
                "std_angle_error"
            ],
            "neural_released_sd_over_sqrt_n_deg": (
                neural_all_released["std_angle_error"]
                / math.sqrt(neural_all_released["n"])
            ),
            "interpretation": (
                "The manuscript's 28-degree central value aligns with the released population "
                "standard deviations, not the released mean angle errors. Its approximately "
                "2-degree uncertainty aligns with SD/sqrt(n), making the stated mean+/-SEM form "
                "internally inconsistent with the released 3D statistics."
            ),
        },
        "artifact_availability": {
            "gastruloid_raw_predictions_npz_count": len(
                list(GASTRO_RUN.rglob("raw_predictions.npz"))
            ),
            "neural_raw_predictions_npz_count": len(
                list(NEURAL_RUN.rglob("raw_predictions.npz"))
            ),
            "neural_raw_input_or_annotation_tiff_count_outside_runs": len(
                [
                    path
                    for path in (DATA / "Neural_tube_160226").rglob("*.tif")
                    if "runs" not in path.parts
                ]
            ),
        },
        "assertions": assertions,
        "all_assertions_pass": True,
        "outputs": [
            rel(REPORT_OUT),
            rel(EVENTS_OUT),
            rel(VALIDATION_OUT),
            rel(MISSING_OUT),
        ],
        "interpretation_notes": [
            "This phase audits released predictions and renderings; it is not checkpoint inference.",
            "No parameters were changed to improve agreement.",
            "Full-quaternion error is not a scientifically direct substitute for an unoriented division-axis error.",
            "Only predicted_centers evaluates regression at detector-predicted locations.",
        ],
    }

    write_csv(EVENTS_OUT, event_rows)
    write_csv(VALIDATION_OUT, validation_rows)
    write_csv(MISSING_OUT, missing_rows)
    write_json(REPORT_OUT, report)
    print(json.dumps(report, indent=2, default=serial), flush=True)


if __name__ == "__main__":
    main()
