"""Independent, audit-only DARE3D nematic-axis metric comparison.

The released predictions, ground truth, center modes, matching, filtering, and
aggregation scope are held fixed. Only the angular definition changes. This file
writes derived evidence under docs/reproducibility_audit and never imports or
edits the production evaluator.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage
from skimage.measure import regionprops


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "DARE3d_data_190326"
EVIDENCE = HERE / "evidence"

GASTRO_RUN = DATA / "Gastruloid_241025/weights/regression3d_exp10-b/runs/01-01"
GASTRO_STATS = (
    DATA
    / "Gastruloid_241025/weights/segmentation3d_exp10-b/runs/01-01/stats.csv"
)
GASTRO_LABEL = DATA / "Gastruloid_241025/trainingset/movie2/label/movie2.tif"
GASTRO_CHECKPOINT_RUN = HERE / "checkpoint_runs/nuclei_regression/best_epoch_098"

NEURAL_RUN = DATA / "Neural_tube_160226/regression3d_new_set_og/runs/12-01-26"
NEURAL_STATS = (
    DATA
    / "Neural_tube_160226/segmentation3d_new_set_og/runs/12-01-26/stats.csv"
)

OBJECTS = EVIDENCE / "nuclei_current_evaluator_objects.csv"
ALIGNMENT = EVIDENCE / "nuclei_saved_visualization_alignment.csv"
PRIOR_EVENTS = EVIDENCE / "regression_event_metrics.csv"

JSON_OUT = EVIDENCE / "nematic_axis_metric_comparison.json"
SUMMARY_OUT = EVIDENCE / "nematic_axis_metric_comparison.csv"
EVENTS_OUT = EVIDENCE / "nematic_axis_event_metrics.csv"

MODES = {
    "all_groundtruth_centers": "all_true_centers",
    "matched_groundtruth_centers": "matched_true_centers",
    "predicted_centers": "pred_centers",
}
STAT_KEYS = {mode: f"regression_performance_on_{mode}" for mode in MODES}
CHECKPOINT_NPZ = {
    mode: GASTRO_CHECKPOINT_RUN / f"{mode}_raw_predictions.npz" for mode in MODES
}


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def serial(value):
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=serial) + "\n",
        encoding="utf-8",
    )


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def unit_vector(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("A finite, non-zero division-axis vector is required")
    return vector / norm


def nematic_axis_error_deg(axis_pred, axis_true) -> float:
    """acos(abs(unit-axis dot product)) in degrees, with exact-range clipping."""
    dot = float(np.dot(unit_vector(axis_pred), unit_vector(axis_true)))
    absolute_dot = float(np.clip(abs(dot), 0.0, 1.0))
    return float(np.degrees(np.arccos(absolute_dot)))


def division_axis_from_wxyz(quaternion) -> np.ndarray:
    """Recover the axis that current production rendering uses."""
    quaternion = unit_vector(quaternion)
    norm = float(np.linalg.norm(quaternion[1:]))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Quaternion has no defined rotation/division axis")
    return quaternion[1:] / norm


def quaternion_from_axis_angle_wxyz(axis, angle_deg: float) -> np.ndarray:
    axis = unit_vector(axis)
    half_angle = math.radians(angle_deg) / 2.0
    return np.asarray(
        [math.cos(half_angle), *(math.sin(half_angle) * axis)],
        dtype=np.float64,
    )


def canonical_production_quaternion_for_axis(axis) -> np.ndarray:
    """Recreate get_quaternion(-axis, axis) without importing production code."""
    axis = unit_vector(axis)
    points = np.asarray([-axis, axis], dtype=np.float64)
    angles = np.arctan2(points[:, 1], points[:, 0])
    first, second = points[np.argsort(angles)]
    oriented_axis = unit_vector(second - first)
    angle = math.acos(float(np.clip(oriented_axis[0], -1.0, 1.0)))
    return quaternion_from_axis_angle_wxyz(oriented_axis, math.degrees(angle))


def production_quaternion_error_deg(q_pred, q_true) -> float:
    """Frozen current definition, including the +/-0.9999 clipping floor."""
    dot = float(np.dot(unit_vector(q_pred), unit_vector(q_true)))
    dot = float(np.clip(dot, -1.0 + 1e-4, 1.0 - 1e-4))
    return float(np.degrees(2.0 * np.arccos(abs(dot))))


def describe(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"n": 0}
    std = float(np.std(values, dtype=np.float32))
    return {
        "n": int(len(values)),
        "mean_deg": float(np.mean(values, dtype=np.float32)),
        "median_deg": float(np.median(values)),
        "std_population_deg": std,
        "sem_population_deg": std / math.sqrt(len(values)),
        "rms_deg": float(np.sqrt(np.mean(np.square(values), dtype=np.float64))),
        "p95_deg": float(np.percentile(values, 95)),
        "max_deg": float(np.max(values)),
    }


def scalar_difference(current: float, corrected: float) -> dict:
    signed = float(corrected - current)
    relative = float(100.0 * signed / abs(current)) if current != 0 else None
    return {
        "signed_deg": signed,
        "absolute_deg": abs(signed),
        "signed_relative_percent_of_current": relative,
        "absolute_relative_percent_of_current": (
            abs(relative) if relative is not None else None
        ),
    }


def read_released_stats(path: Path) -> dict:
    return read_json(path)["regression_results"]


def decode_line(movie_tzyx, center, radius: int = 24) -> dict:
    """Decode the nearest rendered 26-connected line near an M,T,X,Y,Z center."""
    _, t, x, y, z = np.asarray(center, dtype=np.float64)
    frame = movie_tzyx[int(t)]
    target_zyx = np.asarray([z, y, x], dtype=np.float64)
    rounded = np.rint(target_zyx).astype(int)
    low = np.maximum(rounded - radius, 0)
    high = np.minimum(rounded + radius + 1, frame.shape)
    crop = np.asarray(frame[tuple(slice(a, b) for a, b in zip(low, high))]) > 0
    labels, count = ndimage.label(
        crop, structure=np.ones((3, 3, 3), dtype=np.uint8)
    )
    candidates = []
    for component in range(1, count + 1):
        local = np.argwhere(labels == component)
        if len(local) < 2:
            continue
        xyz = (local + low)[..., ::-1].astype(np.float64)
        centroid = np.mean(xyz, axis=0)
        centered = xyz - centroid
        _, singular, vh = np.linalg.svd(centered, full_matrices=False)
        axis = unit_vector(vh[0])
        candidates.append(
            {
                "axis": axis,
                "component_center_distance": float(
                    np.linalg.norm(centroid - np.asarray([x, y, z]))
                ),
                "principal_to_second_variance_ratio": float(
                    singular[0] ** 2 / max(singular[1] ** 2, 1e-12)
                ),
            }
        )
    if not candidates:
        raise RuntimeError(f"No rendered line near center {center}")
    return min(candidates, key=lambda item: item["component_center_distance"])


def decode_truth_line(movie_tzyx, center) -> dict:
    """Decode at the requested frame, then use the saved audit's +/-1 fallback."""
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
                    result = decode_line(movie_tzyx, shifted)
                    result["rendered_time_offset"] = offset
                    candidates.append(result)
                except RuntimeError:
                    pass
        if not candidates:
            raise
        return min(candidates, key=lambda item: item["component_center_distance"])


def load_gastruloid_bipoints() -> list[list[tuple[np.ndarray, np.ndarray]]]:
    label_tzyx = tifffile.imread(GASTRO_LABEL)
    result = []
    for frame in label_tzyx:
        props = {
            int(item.label): np.asarray(item.centroid, dtype=np.float64)[::-1]
            for item in regionprops(frame)
        }
        values = set(props)
        result.append(
            [
                (props[value], props[value + 1])
                for value in sorted(values)
                if value % 2 == 1 and value + 1 in values
            ]
        )
    return result


def centers_from_detection_evidence() -> dict[str, list[np.ndarray]]:
    objects = read_rows(OBJECTS)
    true = {
        int(row["index"]): np.asarray(
            [0, *(float(row[key]) for key in ("t", "x", "y", "z"))],
            dtype=np.float64,
        )
        for row in objects
        if row["kind"] == "true"
    }
    predicted = {
        int(row["index"]): np.asarray(
            [0, *(float(row[key]) for key in ("t", "x", "y", "z"))],
            dtype=np.float64,
        )
        for row in objects
        if row["kind"] == "pred"
    }
    alignment = sorted(read_rows(ALIGNMENT), key=lambda row: int(row["match_id"]))
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
        if np.all(np.isclose(np.asarray([x, y, z]), actual_center, atol=3)):
            return first, second
    return None


def checkpoint_production_errors() -> dict[str, list[float]]:
    grouped = defaultdict(list)
    for row in read_rows(GASTRO_CHECKPOINT_RUN / "event_errors.csv"):
        grouped[row["mode"]].append(float(row["full_quaternion_angle_error_deg"]))
    return dict(grouped)


def make_summary_row(
    dataset: str,
    mode: str,
    current: dict,
    corrected: dict,
    current_event_proxy: dict,
    current_event_proxy_status: str,
    corrected_basis: str,
) -> dict:
    nominal_n = int(current["n"])
    current_mean = float(current["mean_angle_error"])
    current_std = float(current["std_angle_error"])
    corrected_mean = float(corrected["mean_deg"])
    corrected_std = float(corrected["std_population_deg"])
    current_sem = current_std / math.sqrt(nominal_n)
    corrected_sem = float(corrected["sem_population_deg"])
    mean_difference = scalar_difference(current_mean, corrected_mean)
    std_difference = scalar_difference(current_std, corrected_std)
    sem_difference = scalar_difference(current_sem, corrected_sem)
    median_difference = scalar_difference(
        float(current_event_proxy["median_deg"]),
        float(corrected["median_deg"]),
    )
    rms_difference = scalar_difference(
        float(current_event_proxy["rms_deg"]), float(corrected["rms_deg"])
    )
    return {
        "dataset": dataset,
        "mode": mode,
        "production_reported_n": nominal_n,
        "effective_n": corrected["n"],
        "corrected_basis": corrected_basis,
        "production_mean_deg_exact": current_mean,
        "corrected_mean_deg": corrected_mean,
        "mean_signed_difference_deg": mean_difference["signed_deg"],
        "mean_absolute_difference_deg": mean_difference["absolute_deg"],
        "mean_signed_relative_difference_percent": mean_difference[
            "signed_relative_percent_of_current"
        ],
        "production_median_deg_event_proxy": current_event_proxy["median_deg"],
        "production_median_proxy_status": current_event_proxy_status,
        "corrected_median_deg": corrected["median_deg"],
        "median_proxy_absolute_difference_deg": median_difference["absolute_deg"],
        "production_std_population_deg_exact": current_std,
        "corrected_std_population_deg": corrected_std,
        "std_signed_difference_deg": std_difference["signed_deg"],
        "std_absolute_difference_deg": std_difference["absolute_deg"],
        "std_signed_relative_difference_percent": std_difference[
            "signed_relative_percent_of_current"
        ],
        "production_sem_deg_using_reported_n": current_sem,
        "corrected_sem_deg_using_effective_n": corrected_sem,
        "sem_absolute_difference_deg": sem_difference["absolute_deg"],
        "production_rms_deg_event_proxy": current_event_proxy["rms_deg"],
        "corrected_rms_deg": corrected["rms_deg"],
        "rms_proxy_absolute_difference_deg": rms_difference["absolute_deg"],
        "corrected_p95_deg": corrected["p95_deg"],
        "corrected_max_deg": corrected["max_deg"],
    }


def analyse_gastruloid(event_rows: list[dict]) -> tuple[dict, list[dict]]:
    released = read_released_stats(GASTRO_STATS)
    bipoints = load_gastruloid_bipoints()
    centers_by_mode = centers_from_detection_evidence()
    production_replay = checkpoint_production_errors()
    summaries = {}
    summary_rows = []

    for mode, folder in MODES.items():
        centers = centers_by_mode[mode]
        saved = np.load(CHECKPOINT_NPZ[mode])
        quaternions = np.asarray(saved["quaternions"], dtype=np.float64)
        saved_centers = np.asarray(saved["centers"], dtype=np.float64)
        center_key = lambda value: tuple(
            np.round(np.asarray(value, dtype=np.float64), 8)
        )
        quaternion_by_center = {
            center_key(center): quaternion
            for center, quaternion in zip(saved_centers, quaternions)
        }
        requested_keys = [center_key(center) for center in centers]
        if len(quaternion_by_center) != len(centers) or set(
            quaternion_by_center
        ) != set(requested_keys):
            raise AssertionError(f"Frozen checkpoint center set changed for {mode}")
        quaternions = [quaternion_by_center[key] for key in requested_keys]
        archived_render = tifffile.memmap(GASTRO_RUN / folder / "movie2.tif")

        corrected_raw = []
        corrected_archived = []
        raw_vs_archived = []
        production_values = production_replay[mode]
        truth_by_predicted = (
            dict(
                zip(
                    (center_key(value) for value in centers_by_mode["predicted_centers"]),
                    centers_by_mode["matched_groundtruth_centers"],
                )
            )
            if mode == "predicted_centers"
            else {}
        )
        production_by_center = {}
        production_cursor = 0
        for checkpoint_center in saved_centers:
            lookup_center = truth_by_predicted.get(
                center_key(checkpoint_center), checkpoint_center
            )
            if gather_annotation(lookup_center, bipoints) is not None:
                production_by_center[center_key(checkpoint_center)] = (
                    production_values[production_cursor]
                )
                production_cursor += 1
        if production_cursor != len(production_values):
            raise AssertionError(f"Checkpoint error count changed for {mode}")
        for index, (center, quaternion) in enumerate(zip(centers, quaternions)):
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
                "truth_time_offset": "",
            }
            if annotation is None:
                event_rows.append(
                    {
                        **base,
                        "evaluated": False,
                        "skip_reason": (
                            "production gather_groundtruth_info found no label pair "
                            "within per-axis atol=3"
                        ),
                    }
                )
                continue

            first, second = annotation
            true_axis = unit_vector(second - first)
            predicted_axis = division_axis_from_wxyz(quaternion)
            archived_axis = decode_line(archived_render, center)["axis"]
            corrected_raw_value = nematic_axis_error_deg(
                predicted_axis, true_axis
            )
            corrected_archived_value = nematic_axis_error_deg(
                archived_axis, true_axis
            )
            raw_vs_archived_value = nematic_axis_error_deg(
                predicted_axis, archived_axis
            )
            production_value = production_by_center[center_key(center)]
            corrected_raw.append(corrected_raw_value)
            corrected_archived.append(corrected_archived_value)
            raw_vs_archived.append(raw_vs_archived_value)
            event_rows.append(
                {
                    **base,
                    "evaluated": True,
                    "skip_reason": "",
                    "truth_axis_x": true_axis[0],
                    "truth_axis_y": true_axis[1],
                    "truth_axis_z": true_axis[2],
                    "predicted_axis_x": predicted_axis[0],
                    "predicted_axis_y": predicted_axis[1],
                    "predicted_axis_z": predicted_axis[2],
                    "corrected_nematic_axis_error_deg": corrected_raw_value,
                    "archived_raster_nematic_axis_error_deg": (
                        corrected_archived_value
                    ),
                    "raw_vs_archived_axis_error_deg": raw_vs_archived_value,
                    "production_quaternion_error_deg_event_proxy": (
                        production_value
                    ),
                    "metric_basis": (
                        "epoch_098 raw-quaternion compatibility replay plus "
                        "released daughter-label centroids"
                    ),
                }
            )

        if production_cursor != len(production_values):
            raise AssertionError(f"Checkpoint error count changed for {mode}")
        corrected_description = describe(corrected_raw)
        archived_description = describe(corrected_archived)
        current_event_description = describe(production_values)
        current = released[STAT_KEYS[mode]]
        summary_rows.append(
            make_summary_row(
                "gastruloid_nuclei",
                mode,
                current,
                corrected_description,
                current_event_description,
                (
                    "near-exact epoch_098 checkpoint replay; released median "
                    "was not stored"
                ),
                (
                    "epoch_098 raw quaternion replay; released production "
                    "mean/std reproduced within 0.016 degrees"
                ),
            )
        )
        summaries[mode] = {
            "released_production_full_quaternion": current,
            "production_event_distribution_checkpoint_replay": (
                current_event_description
            ),
            "corrected_nematic_axis_raw_quaternion_replay": (
                corrected_description
            ),
            "corrected_nematic_axis_archived_raster_sensitivity": (
                archived_description
            ),
            "raw_quaternion_vs_archived_raster_axis_disagreement": describe(
                raw_vs_archived
            ),
            "comparison": {
                "mean": scalar_difference(
                    float(current["mean_angle_error"]),
                    corrected_description["mean_deg"],
                ),
                "std_population": scalar_difference(
                    float(current["std_angle_error"]),
                    corrected_description["std_population_deg"],
                ),
            },
            "production_reported_n": int(current["n"]),
            "effective_n": corrected_description["n"],
        }

    return {
        "classification": (
            "Same checkpoint, inputs, center modes, matching, and raw labels. "
            "Original raw quaternions were not released, so primary axes use "
            "the independently verified epoch_098 compatibility replay."
        ),
        "modes": summaries,
        "sources": [
            rel(GASTRO_STATS),
            rel(GASTRO_LABEL),
            rel(OBJECTS),
            rel(ALIGNMENT),
            rel(GASTRO_CHECKPOINT_RUN / "result.json"),
            rel(GASTRO_CHECKPOINT_RUN / "event_errors.csv"),
            *(rel(path) for path in CHECKPOINT_NPZ.values()),
            *(rel(GASTRO_RUN / folder / "movie2.tif") for folder in MODES.values()),
        ],
    }, summary_rows


def analyse_neural(event_rows: list[dict]) -> tuple[dict, list[dict]]:
    released = read_released_stats(NEURAL_STATS)
    groundtruth_path = NEURAL_RUN / "groundtruth/movie_I2.tif"
    groundtruth = tifffile.memmap(groundtruth_path)
    summaries = {}
    summary_rows = []
    time_offsets = []

    for mode, folder in MODES.items():
        npz_path = NEURAL_RUN / folder / "raw_predictions.npz"
        saved = np.load(npz_path)
        centers = np.asarray(saved["centers"], dtype=np.float64)
        quaternions = np.asarray(saved["quaternions"], dtype=np.float64)
        if len(centers) != len(quaternions):
            raise AssertionError(f"Released neural prediction count changed for {mode}")

        corrected_values = []
        production_proxy_values = []
        truth_quality = []
        mode_offsets = []
        for index, (center, quaternion) in enumerate(zip(centers, quaternions)):
            decoded_truth = decode_truth_line(groundtruth, center)
            true_axis = decoded_truth["axis"]
            predicted_axis = division_axis_from_wxyz(quaternion)
            corrected_value = nematic_axis_error_deg(predicted_axis, true_axis)
            production_proxy = production_quaternion_error_deg(
                quaternion, canonical_production_quaternion_for_axis(true_axis)
            )
            offset = int(decoded_truth["rendered_time_offset"])
            corrected_values.append(corrected_value)
            production_proxy_values.append(production_proxy)
            truth_quality.append(
                decoded_truth["principal_to_second_variance_ratio"]
            )
            mode_offsets.append(offset)
            time_offsets.append(offset)
            event_rows.append(
                {
                    "dataset": "neural_tube_membrane",
                    "mode": mode,
                    "index": index,
                    "evaluated": True,
                    "skip_reason": "",
                    "center_t": float(center[1]),
                    "center_x": float(center[2]),
                    "center_y": float(center[3]),
                    "center_z": float(center[4]),
                    "truth_time_offset": offset,
                    "truth_axis_x": true_axis[0],
                    "truth_axis_y": true_axis[1],
                    "truth_axis_z": true_axis[2],
                    "predicted_axis_x": predicted_axis[0],
                    "predicted_axis_y": predicted_axis[1],
                    "predicted_axis_z": predicted_axis[2],
                    "corrected_nematic_axis_error_deg": corrected_value,
                    "archived_raster_nematic_axis_error_deg": "",
                    "raw_vs_archived_axis_error_deg": "",
                    "production_quaternion_error_deg_event_proxy": (
                        production_proxy
                    ),
                    "metric_basis": (
                        "released raw prediction quaternion plus independently "
                        "decoded released ground-truth line raster"
                    ),
                }
            )

        corrected_description = describe(corrected_values)
        current_event_description = describe(production_proxy_values)
        current = released[STAT_KEYS[mode]]
        summary_rows.append(
            make_summary_row(
                "neural_tube_membrane",
                mode,
                current,
                corrected_description,
                current_event_description,
                (
                    "raster-derived proxy; released event-level production "
                    "errors and raw annotations were not stored"
                ),
                "released raw quaternion plus decoded ground-truth line raster",
            )
        )
        summaries[mode] = {
            "released_production_full_quaternion": current,
            "production_event_distribution_raster_proxy": (
                current_event_description
            ),
            "production_proxy_minus_released": {
                "mean_deg": current_event_description["mean_deg"]
                - float(current["mean_angle_error"]),
                "std_population_deg": current_event_description[
                    "std_population_deg"
                ]
                - float(current["std_angle_error"]),
            },
            "corrected_nematic_axis": corrected_description,
            "comparison": {
                "mean": scalar_difference(
                    float(current["mean_angle_error"]),
                    corrected_description["mean_deg"],
                ),
                "std_population": scalar_difference(
                    float(current["std_angle_error"]),
                    corrected_description["std_population_deg"],
                ),
            },
            "production_reported_n": int(current["n"]),
            "effective_n": corrected_description["n"],
            "groundtruth_raster_quality": describe(truth_quality),
            "groundtruth_time_offsets": {
                str(offset): mode_offsets.count(offset) for offset in (-1, 0, 1)
            },
        }

    return {
        "classification": (
            "Released prediction quaternions with independently decoded saved "
            "truth-axis raster. Raw neural images and daughter annotations are absent."
        ),
        "modes": summaries,
        "groundtruth_time_offsets_all_modes": {
            str(offset): time_offsets.count(offset) for offset in (-1, 0, 1)
        },
        "sources": [
            rel(NEURAL_STATS),
            rel(groundtruth_path),
            *(
                rel(NEURAL_RUN / folder / "raw_predictions.npz")
                for folder in MODES.values()
            ),
        ],
    }, summary_rows


def run_synthetic_checks() -> dict:
    identical = nematic_axis_error_deg([1, 0, 0], [1, 0, 0])
    opposite = nematic_axis_error_deg([1, 0, 0], [-1, 0, 0])
    orthogonal = nematic_axis_error_deg([1, 0, 0], [0, 1, 0])
    q30 = quaternion_from_axis_angle_wxyz([0, 0, 1], 30.0)
    q150 = quaternion_from_axis_angle_wxyz([0, 0, 1], 150.0)
    twist_invariant = nematic_axis_error_deg(
        division_axis_from_wxyz(q30), division_axis_from_wxyz(q150)
    )
    production_twist_penalty = production_quaternion_error_deg(q30, q150)
    production_identity_floor = production_quaternion_error_deg(q30, q30)
    repeated = unit_vector([1.0, 1.0, 1.0])
    raw_roundoff_dot = float(np.dot(repeated, repeated))
    exact_clip_result = nematic_axis_error_deg(repeated, repeated)
    assertions = {
        "identical_axes_are_exactly_zero": identical == 0.0,
        "opposite_axes_are_exactly_zero": opposite == 0.0,
        "orthogonal_axes_are_exactly_90": orthogonal == 90.0,
        "rotation_about_axis_is_exactly_invariant": twist_invariant == 0.0,
        "production_metric_penalizes_axis_invariant_rotation": math.isclose(
            production_twist_penalty, 120.0, abs_tol=1e-12
        ),
        "production_metric_has_nonzero_identity_floor": (
            production_identity_floor > 1.6
        ),
        "exact_clip_introduces_no_floor": exact_clip_result == 0.0,
    }
    return {
        "identical_axes_deg": identical,
        "opposite_nematic_axes_deg": opposite,
        "orthogonal_axes_deg": orthogonal,
        "same_axis_different_rotation_angles_axis_error_deg": twist_invariant,
        "same_axis_different_rotation_angles_production_error_deg": (
            production_twist_penalty
        ),
        "production_identical_quaternion_floor_deg": production_identity_floor,
        "floating_point_safety_raw_self_dot": raw_roundoff_dot,
        "floating_point_safety_axis_error_after_exact_interval_clip_deg": (
            exact_clip_result
        ),
        "clip_rule": (
            "Clip only abs(unit-axis dot product) to the exact closed interval [0,1]."
        ),
        "assertions": assertions,
        "all_assertions_pass": all(assertions.values()),
    }


def crosscheck_prior_event_evidence(event_rows: list[dict]) -> dict:
    prior = {
        (row["dataset"], row["mode"], int(row["index"])): row
        for row in read_rows(PRIOR_EVENTS)
    }
    differences = []
    compared = 0
    for row in event_rows:
        key = (row["dataset"], row["mode"], int(row["index"]))
        old = prior[key]
        old_evaluated = old["evaluated"] == "True"
        if bool(row["evaluated"]) != old_evaluated:
            raise AssertionError(f"Prior evaluated state changed for {key}")
        if not old_evaluated:
            continue
        new_value = (
            row["archived_raster_nematic_axis_error_deg"]
            if row["dataset"] == "gastruloid_nuclei"
            else row["corrected_nematic_axis_error_deg"]
        )
        differences.append(
            abs(float(new_value) - float(old["nematic_axis_error_deg"]))
        )
        compared += 1
    maximum = max(differences)
    return {
        "prior_row_count": len(prior),
        "new_row_count": len(event_rows),
        "effective_rows_compared": compared,
        "maximum_absolute_difference_deg": maximum,
        "all_values_within_1e_minus_10_deg": maximum <= 1e-10,
        "interpretation": (
            "The independent implementation exactly cross-checks the earlier "
            "saved-output axis reconstruction. Nuclei primary values additionally "
            "use raw quaternions from the checkpoint compatibility replay."
        ),
    }


def affected_locations() -> dict:
    manuscript = "manuscript_280826version/main.tex"
    return {
        "direct_quantitative_claims": [
            {
                "location": f"{manuscript}:395",
                "section": "Results: 3D nuclei model",
                "claim": "orientation-angle error 28 +/- 2 degrees",
            },
            {
                "location": f"{manuscript}:402",
                "section": "Results: 3D membrane model",
                "claim": "orientation-angle error 28 +/- 2 degrees",
            },
        ],
        "interpretive_claims": [
            {
                "location": f"{manuscript}:79",
                "section": "Abstract",
                "claim": "orientation accuracy approaches annotation uncertainty",
            },
            {
                "location": f"{manuscript}:481",
                "section": "Discussion: 3D nuclei orientation error",
                "claim": "mean error is of the same order as about 20 degrees",
            },
            {
                "location": f"{manuscript}:493-494",
                "section": "Discussion: 3D membrane orientation error",
                "claim": "mean error is of the same order as about 20 degrees",
            },
            {
                "location": f"{manuscript}:501-505",
                "section": "Discussion: 2D vs 3D regression",
                "claim": "3D error is below the 61.2-degree random nematic RMS",
            },
        ],
        "production_and_generated_statistic_provenance": [
            "dare3d/losses/angle3d.py:70-84 (training loss)",
            "dare3d/losses/angle3d.py:106-116 (production evaluation function)",
            "dare3d/metrics/infer_measure.py:372-411 (three-mode aggregation)",
            "dare3d/metrics/infer_measure.py:329-344 (unused error-plot helper)",
            "dare3d/data/components/angles3d.py:265-290 (rendered axis)",
            "dare3d/metrics/inference.py:144-176 (prediction/output)",
            (
                "dare3d/models/regression_module.py:83-107,123-186 "
                "(training/validation angular-loss logs)"
            ),
            "configs/experiment/regression.yaml:43 (checkpoint monitor val/loss)",
            rel(GASTRO_STATS),
            rel(NEURAL_STATS),
            (
                "docs/reproducibility_audit/checkpoint_runs/nuclei_regression/"
                "best_epoch_098/event_errors.csv"
            ),
            (
                "docs/reproducibility_audit/checkpoint_runs/nuclei_regression/"
                "best_epoch_098/result.json"
            ),
        ],
        "not_numerically_affected": [
            (
                f"{manuscript}:178-212, Table dataset:stats_all; supplies "
                "dataset/test provenance and lengths, not angular performance"
            ),
            (
                f"{manuscript}:645-656, Figs. fig:regression and "
                "fig:regression_neuraltube; qualitative axes are unchanged"
            ),
            (
                f"{manuscript}:672-675, Fig. fig:3Dsuccess; qualitative "
                "predictions and detection examples are unchanged"
            ),
            (
                f"{manuscript}:572-594, supplementary movie captions; "
                "displayed axes are unchanged and no aggregate angle is embedded"
            ),
        ],
    }


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    event_rows = []
    synthetic = run_synthetic_checks()
    gastruloid, gastro_rows = analyse_gastruloid(event_rows)
    neural, neural_rows = analyse_neural(event_rows)
    summary_rows = gastro_rows + neural_rows
    crosscheck = crosscheck_prior_event_evidence(event_rows)

    by_key = {(row["dataset"], row["mode"]): row for row in summary_rows}
    nuclei_all = by_key[("gastruloid_nuclei", "all_groundtruth_centers")]
    membrane_all = by_key[
        ("neural_tube_membrane", "all_groundtruth_centers")
    ]
    checkpoint_maximum = read_json(
        GASTRO_CHECKPOINT_RUN / "result.json"
    )["aggregate_comparison"]["maximum_absolute_difference"]
    assertions = {
        "synthetic_checks_pass": synthetic["all_assertions_pass"],
        "six_dataset_mode_summaries": len(summary_rows) == 6,
        "event_row_count_is_743": len(event_rows) == 743,
        "effective_event_count_is_732": (
            sum(bool(row["evaluated"]) for row in event_rows) == 732
        ),
        "nuclei_counts_match_140_121_121": [
            gastruloid["modes"][mode]["effective_n"] for mode in MODES
        ]
        == [140, 121, 121],
        "membrane_counts_match_122_114_114": [
            neural["modes"][mode]["effective_n"] for mode in MODES
        ]
        == [122, 114, 114],
        "prior_axis_evidence_crosscheck_exact": crosscheck[
            "all_values_within_1e_minus_10_deg"
        ],
        "nuclei_checkpoint_production_replay_within_0_016_deg": (
            checkpoint_maximum <= 0.016
        ),
        "neural_truth_fallback_counts_match": neural[
            "groundtruth_time_offsets_all_modes"
        ]
        == {"-1": 0, "0": 313, "1": 37},
        "all_corrected_means_in_nematic_range": all(
            0 <= row["corrected_mean_deg"] <= 90 for row in summary_rows
        ),
        "nuclei_all_gt_corrected_mean_below_20": (
            nuclei_all["corrected_mean_deg"] < 20
        ),
        "membrane_all_gt_corrected_mean_above_30": (
            membrane_all["corrected_mean_deg"] > 30
        ),
        "nuclei_all_gt_rms_below_random_61_2": (
            nuclei_all["corrected_rms_deg"] < 61.2
        ),
        "membrane_all_gt_rms_below_random_61_2": (
            membrane_all["corrected_rms_deg"] < 61.2
        ),
    }
    failed = [name for name, passed in assertions.items() if not passed]
    if failed:
        raise AssertionError(f"Nematic comparison assertions failed: {failed}")

    report = {
        "classification": (
            "Audit-only fixed-prediction angular-metric replacement. No "
            "production file or released artifact was modified."
        ),
        "definitions": {
            "current_production": (
                "2*acos(abs(dot(unit(q_pred),unit(q_true)))) in degrees, "
                "with the un-absolute dot clipped to [-0.9999,0.9999]"
            ),
            "corrected_nematic_axis": (
                "acos(abs(dot(unit(u_pred),unit(u_true)))) in degrees, with "
                "only the absolute scalar product clipped to exact [0,1]"
            ),
            "aggregation": (
                "Same production event selection and skip rules; float32 mean "
                "and population SD; SEM=population SD/sqrt(effective N)."
            ),
        },
        "synthetic_checks": synthetic,
        "gastruloid_nuclei": gastruloid,
        "neural_tube_membrane": neural,
        "summary_rows": summary_rows,
        "manuscript_replacement": {
            "nuclei_all_groundtruth_centers": {
                "current_manuscript_text_deg": "28 +/- 2",
                "corrected_mean_deg": nuclei_all["corrected_mean_deg"],
                "corrected_sem_deg": nuclei_all[
                    "corrected_sem_deg_using_effective_n"
                ],
                "corrected_std_population_deg": nuclei_all[
                    "corrected_std_population_deg"
                ],
                "effective_n": nuclei_all["effective_n"],
                "nominal_n": nuclei_all["production_reported_n"],
            },
            "membrane_all_groundtruth_centers": {
                "current_manuscript_text_deg": "28 +/- 2",
                "corrected_mean_deg": membrane_all["corrected_mean_deg"],
                "corrected_sem_deg": membrane_all[
                    "corrected_sem_deg_using_effective_n"
                ],
                "corrected_std_population_deg": membrane_all[
                    "corrected_std_population_deg"
                ],
                "effective_n": membrane_all["effective_n"],
                "nominal_n": membrane_all["production_reported_n"],
            },
            "interpretation": (
                "The two 28-degree central values match released population "
                "SDs, not means. Corrected mean+/-SEM is the coherent replacement."
            ),
        },
        "affected_locations": affected_locations(),
        "crosscheck_against_prior_saved_output_evidence": crosscheck,
        "limitations": [
            (
                "Original nuclei raw prediction quaternions were not released; "
                "the primary counterfactual uses the verified epoch_098 replay."
            ),
            (
                "Membrane raw daughter annotations are absent; truth axes are "
                "decoded from a uint8 raster, including 37 next-frame fallbacks."
            ),
            (
                "Production medians and event angles were not released. Nuclei "
                "production distribution statistics use a near-exact replay; "
                "membrane production distribution statistics are raster proxies."
            ),
            (
                "The models were trained and checkpoint-selected with the "
                "full-quaternion loss. This fixed-prediction test does not "
                "estimate retraining or checkpoint-selection effects."
            ),
            (
                "No split, anisotropy, matching, reported-N, or manuscript "
                "dataset-identity issue was changed."
            ),
        ],
        "assertions": assertions,
        "all_assertions_pass": True,
        "outputs": [rel(JSON_OUT), rel(SUMMARY_OUT), rel(EVENTS_OUT)],
    }
    write_rows(SUMMARY_OUT, summary_rows)
    write_rows(EVENTS_OUT, event_rows)
    write_json(JSON_OUT, report)
    print(
        f"Nematic-axis comparison complete: {len(assertions)}/{len(assertions)} "
        f"assertions, {len(event_rows)} event rows, {len(summary_rows)} summaries.",
        flush=True,
    )


if __name__ == "__main__":
    main()
