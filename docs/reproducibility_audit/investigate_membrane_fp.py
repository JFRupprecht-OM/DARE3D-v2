"""Investigate the eighth membrane false positive missing from saved renderings.

This independently reconstructs predicted components from the released float16
probability. It reproduces the current evaluator's temporal dilation, threshold,
weighted filtering (including its first-component skip), and marker slicing
without requiring the absent neural ground-truth annotations.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUN = (
    REPO
    / "DARE3d_data_190326/Neural_tube_160226"
    / "segmentation3d_new_set_og/runs/12-01-26"
)
PROBABILITY = RUN / "movie_I2.tif"
PRED_VISUAL = RUN / "movie_I2/movie_I2_pred.tif"
FP_VISUAL = RUN / "movie_I2/movie_I2_fp.tif"
STATS = RUN / "stats.csv"
OBJECT_LEVEL = REPO / "dare3d/metrics/object_level.py"
EVIDENCE = HERE / "evidence"
JSON_OUTPUT = EVIDENCE / "membrane_fp_investigation.json"
CSV_OUTPUT = EVIDENCE / "membrane_predicted_components.csv"

THRESHOLD = 0.55
MIN_WEIGHTED_PROBABILITY = 0.15
MAXIMUM_SIZE = (4.0 / 3.0) * math.pi * (8**3) * 3.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def temporal_dilated_frame(raw: np.ndarray, t: int) -> np.ndarray:
    left = max(0, t - 1)
    right = min(raw.shape[0] - 1, t + 1)
    result = np.maximum(raw[left], raw[t])
    np.maximum(result, raw[right], out=result)
    return result


def saved_events(array: np.ndarray) -> list[dict]:
    events: list[dict] = []
    for t in range(array.shape[0]):
        frame = np.asarray(array[t])
        for value in np.unique(frame):
            value = int(value)
            if value < 2:
                continue
            coords = np.nonzero(frame == value)
            events.append(
                {
                    "t": t,
                    "label": value,
                    "zyx": [float(np.mean(axis)) for axis in coords],
                }
            )
        unmatched, count = ndimage.label(frame == 1)
        for value in range(1, count + 1):
            coords = np.nonzero(unmatched == value)
            events.append(
                {
                    "t": t,
                    "label": 1,
                    "zyx": [float(np.mean(axis)) for axis in coords],
                }
            )
    return events


def production_round(value: float) -> int:
    return int(np.round(value + 1e-9))


def marker_axis(center: float, size: int) -> tuple[int, int, int]:
    rounded = production_round(center)
    start, stop, _ = slice(rounded - 4, rounded + 4).indices(size)
    return rounded, start, stop


def main() -> None:
    started = time.perf_counter()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    released = json.loads(STATS.read_text(encoding="utf-8"))["segmentation_results"]

    raw = tifffile.memmap(PROBABILITY)
    if raw.shape != (21, 10, 1024, 1024) or raw.dtype != np.float16:
        raise AssertionError(f"Unexpected probability geometry: {raw.shape} {raw.dtype}")

    print("Building temporally dilated threshold mask...", flush=True)
    internal_shape = (raw.shape[0], raw.shape[3], raw.shape[2], raw.shape[1])
    binary = np.empty(internal_shape, dtype=np.uint8)
    for t in range(raw.shape[0]):
        binary[t] = np.swapaxes(temporal_dilated_frame(raw, t), -1, -3) >= THRESHOLD

    print("Labelling threshold components...", flush=True)
    labels, initial_components = ndimage.label(binary, output=np.uint16)
    del binary
    gc.collect()
    if initial_components >= np.iinfo(np.uint16).max:
        raise AssertionError("uint16 component label capacity exceeded")

    counts = np.zeros(initial_components + 1, dtype=np.int64)
    probability_sums = np.zeros(initial_components + 1, dtype=np.float64)
    coordinate_sums = np.zeros((4, initial_components + 1), dtype=np.float64)

    print(f"Accumulating statistics for {initial_components} components...", flush=True)
    for t in range(raw.shape[0]):
        coords = np.nonzero(labels[t])
        component_ids = labels[t][coords].astype(np.int64, copy=False)
        frame_counts = np.bincount(component_ids, minlength=initial_components + 1)
        counts += frame_counts
        coordinate_sums[0] += frame_counts * t
        for axis in range(3):
            coordinate_sums[axis + 1] += np.bincount(
                component_ids,
                weights=coords[axis],
                minlength=initial_components + 1,
            )
        dilated_internal = np.swapaxes(
            temporal_dilated_frame(raw, t), -1, -3
        )
        probability_sums += np.bincount(
            component_ids,
            weights=dilated_internal[coords],
            minlength=initial_components + 1,
        )

    mean_probability = np.divide(
        probability_sums,
        counts,
        out=np.zeros_like(probability_sums),
        where=counts != 0,
    )
    weighted_probability = probability_sums / MAXIMUM_SIZE

    keep = weighted_probability >= MIN_WEIGHTED_PROBABILITY
    keep[0] = False
    first_component_bypassed_filter = bool(
        initial_components >= 1 and not keep[1]
    )
    if initial_components >= 1:
        keep[1] = True
    kept_ids = np.flatnonzero(keep)
    filtered_components = int(initial_components - len(kept_ids))

    pred_visual = tifffile.memmap(PRED_VISUAL)
    fp_visual = tifffile.memmap(FP_VISUAL)
    if pred_visual.shape != raw.shape or fp_visual.shape != raw.shape:
        raise AssertionError("Saved visualization geometry differs from probability")

    pred_events = saved_events(pred_visual)
    fp_events = saved_events(fp_visual)
    matched_labels = sorted({event["label"] for event in pred_events if event["label"] >= 2})
    unmatched_events = [event for event in pred_events if event["label"] == 1]

    rows: list[dict] = []
    for component_id in kept_ids:
        centroid_txyz = coordinate_sums[:, component_id] / counts[component_id]
        t = production_round(float(centroid_txyz[0]))
        centroid_zyx = (centroid_txyz[3], centroid_txyz[2], centroid_txyz[1])
        axes = [
            marker_axis(float(value), size)
            for value, size in zip(centroid_zyx, raw.shape[1:])
        ]
        rounded_zyx = [axis[0] for axis in axes]
        starts = [axis[1] for axis in axes]
        stops = [axis[2] for axis in axes]
        lengths = [max(0, stop - start) for start, stop in zip(starts, stops)]
        expected_marker_voxels = int(np.prod(lengths))

        if expected_marker_voxels:
            marker_slice = (
                t,
                slice(starts[0], stops[0]),
                slice(starts[1], stops[1]),
                slice(starts[2], stops[2]),
            )
            pred_block = np.asarray(pred_visual[marker_slice])
            fp_block = np.asarray(fp_visual[marker_slice])
            pred_values, pred_counts = np.unique(
                pred_block[pred_block > 0], return_counts=True
            )
            if len(pred_values):
                dominant_index = int(np.argmax(pred_counts))
                dominant_pred_label = int(pred_values[dominant_index])
                dominant_pred_voxels = int(pred_counts[dominant_index])
            else:
                dominant_pred_label = 0
                dominant_pred_voxels = 0
            saved_fp_voxels = int(np.count_nonzero(fp_block))
        else:
            dominant_pred_label = 0
            dominant_pred_voxels = 0
            saved_fp_voxels = 0

        rows.append(
            {
                "component_id_before_filter": int(component_id),
                "centroid_t": float(centroid_txyz[0]),
                "centroid_z": float(centroid_txyz[3]),
                "centroid_y": float(centroid_txyz[2]),
                "centroid_x": float(centroid_txyz[1]),
                "rounded_t": t,
                "rounded_z": rounded_zyx[0],
                "rounded_y": rounded_zyx[1],
                "rounded_x": rounded_zyx[2],
                "voxel_count": int(counts[component_id]),
                "mean_probability": float(mean_probability[component_id]),
                "weighted_probability": float(weighted_probability[component_id]),
                "was_first_component_filter_exception": bool(component_id == 1),
                "marker_z_start": starts[0],
                "marker_z_stop": stops[0],
                "marker_y_start": starts[1],
                "marker_y_stop": stops[1],
                "marker_x_start": starts[2],
                "marker_x_stop": stops[2],
                "expected_marker_voxels": expected_marker_voxels,
                "saved_pred_dominant_label": dominant_pred_label,
                "saved_pred_dominant_voxels": dominant_pred_voxels,
                "saved_fp_voxels_in_expected_box": saved_fp_voxels,
                "render_visible": bool(dominant_pred_label),
                "classification_from_saved_render": (
                    "matched"
                    if dominant_pred_label >= 2
                    else "false_positive"
                    if dominant_pred_label == 1
                    else "not_rendered"
                ),
            }
        )

    invisible = [row for row in rows if not row["render_visible"]]
    visible_matched = [row for row in rows if row["saved_pred_dominant_label"] >= 2]
    visible_fp = [row for row in rows if row["saved_pred_dominant_label"] == 1]
    empty_slice_components = [row for row in rows if row["expected_marker_voxels"] == 0]
    first_component_rows = [
        row for row in rows if row["was_first_component_filter_exception"]
    ]

    print(
        json.dumps(
            {
                "diagnostic_counts": {
                    "retained": len(rows),
                    "pred_events": len(pred_events),
                    "matched_labels": len(matched_labels),
                    "unmatched_events": len(unmatched_events),
                    "visible_matched": len(visible_matched),
                    "visible_fp": len(visible_fp),
                    "invisible": len(invisible),
                    "empty_slices": len(empty_slice_components),
                },
                "invisible_rows": invisible,
                "empty_slice_rows": empty_slice_components,
            },
            indent=2,
        ),
        flush=True,
    )
    assertions = {
        "released_prediction_total_is_122": int(released["tp"] + released["fp"]) == 122,
        "reconstructed_filtered_prediction_total_is_122": len(rows) == 122,
        "saved_prediction_visual_has_121_events": len(pred_events) == 121,
        "saved_prediction_visual_has_114_matched_ids": len(matched_labels) == 114,
        "saved_prediction_visual_has_7_false_positive_events": len(unmatched_events) == 7,
        "component_to_visual_classification_is_114_plus_7_plus_1": (
            len(visible_matched) == 114 and len(visible_fp) == 7 and len(invisible) == 1
        ),
        "exactly_one_component_has_empty_marker_slice": (
            len(empty_slice_components) == 1 and invisible == empty_slice_components
        ),
        "released_minus_visible_false_positives_is_one": (
            int(released["fp"]) - len(visible_fp) == 1
        ),
        "all_released_matched_ids_are_visible": matched_labels == list(range(2, 116)),
        "saved_fp_visual_contains_7_events": len(fp_events) == 7,
        "all_nonempty_component_markers_are_visible": all(
            row["render_visible"] for row in rows if row["expected_marker_voxels"] > 0
        ),
        "first_component_exception_is_subthreshold_visible_fp": (
            len(first_component_rows) == 1
            and first_component_rows[0]["weighted_probability"] < MIN_WEIGHTED_PROBABILITY
            and first_component_rows[0]["classification_from_saved_render"] == "false_positive"
            and first_component_rows[0]["render_visible"]
        ),
    }
    if not all(assertions.values()):
        failed = [name for name, passed in assertions.items() if not passed]
        raise AssertionError(f"Membrane FP assertions failed: {failed}")

    with CSV_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "scope": (
            "Independent predicted-component reconstruction from the released "
            "membrane probability; ground-truth matching is not rerun because raw "
            "neural annotations are absent."
        ),
        "inputs": {
            "probability": rel(PROBABILITY),
            "probability_sha256": sha256(PROBABILITY),
            "prediction_visual": rel(PRED_VISUAL),
            "prediction_visual_sha256": sha256(PRED_VISUAL),
            "false_positive_visual": rel(FP_VISUAL),
            "false_positive_visual_sha256": sha256(FP_VISUAL),
            "stats": rel(STATS),
            "stats_sha256": sha256(STATS),
            "current_object_level_source": rel(OBJECT_LEVEL),
            "current_object_level_source_sha256": sha256(OBJECT_LEVEL),
            "probability_shape_tzyx": list(raw.shape),
            "evaluation_shape_txyz": list(internal_shape),
            "probability_dtype": str(raw.dtype),
        },
        "frozen_settings": {
            "threshold": THRESHOLD,
            "min_weighted_probability": MIN_WEIGHTED_PROBABILITY,
            "maximum_theoretical_size": MAXIMUM_SIZE,
            "temporal_dilation": "per-frame maximum over t-1, t, t+1",
            "connected_components": (
                "SciPy default direct connectivity in evaluator T,X,Y,Z memory order"
            ),
            "weighted_filter": (
                "component_probability_sum / maximum_theoretical_size >= cutoff; "
                "component label 1 is retained unconditionally to match current code"
            ),
            "marker_rendering": (
                "round centroid, then use un-clipped Python slices center-4:center+4 "
                "on each spatial axis"
            ),
        },
        "released_stats": released,
        "reconstruction": {
            "threshold_components_before_weighted_filter": int(initial_components),
            "components_removed_by_weighted_filter": filtered_components,
            "components_retained": len(rows),
            "first_component_bypassed_filter": first_component_bypassed_filter,
            "first_component_filter_exception": first_component_rows[0],
            "saved_prediction_events": len(pred_events),
            "saved_matched_prediction_ids": len(matched_labels),
            "saved_false_positive_prediction_events": len(unmatched_events),
            "saved_false_positive_visual_events": len(fp_events),
            "visible_reconstructed_matched": len(visible_matched),
            "visible_reconstructed_false_positive": len(visible_fp),
            "invisible_reconstructed_components": len(invisible),
        },
        "invisible_component": invisible[0],
        "interpretation": (
            "The released probability contains 122 retained prediction components, "
            "consistent with 114 TP + 8 FP. All 114 matched markers and seven FP "
            "markers are visible. The remaining component is an FP by exhaustion; "
            "its rounded z=3 makes the un-clipped z marker slice 9:7 empty, so it "
            "is absent from both saved prediction and FP visualizations. The stats "
            "and renderings are therefore consistent: this is a visualization "
            "boundary bug rather than a counting discrepancy. Separately, one visible "
            "FP has weighted score 0.00874 yet is retained only because filtering "
            "skips the first T,X,Y,Z memory-order component."
        ),
        "assertions": assertions,
        "elapsed_seconds": time.perf_counter() - started,
    }
    JSON_OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "complete",
                "assertions_passed": sum(assertions.values()),
                "assertions_total": len(assertions),
                "initial_components": int(initial_components),
                "retained_components": len(rows),
                "visible_matched": len(visible_matched),
                "visible_false_positive": len(visible_fp),
                "invisible_component": invisible[0],
                "json": rel(JSON_OUTPUT),
                "csv": rel(CSV_OUTPUT),
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
