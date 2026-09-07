"""Frozen synthetic checks for DARE3D scientific metric edge cases.

This audit-only script calls the current production evaluator/geometry functions
without modifying them. Tiny synthetic arrays make each behavior independently
inspectable. Results are written only under docs/reproducibility_audit/evidence/.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
from scipy.optimize import linear_sum_assignment

from dare3d.data.components import angles3d as angles3d_module
from dare3d.losses import angle3d as angle3d_module
from dare3d.metrics import object_level as object_level_module
from dare3d.data.components.angles3d import get_points_from_quat
from dare3d.losses.angle3d import quaternion_error
from dare3d.metrics.object_level import (
    connected_components,
    dilate_img,
    evaluate_at_object_level,
    filter_by_object_weighted_prob,
    iterative_matching,
    set_mat_value,
    statistics_optimized,
    wrap_metrics,
)

IMPORT_ORIGINS = {
    "angles3d": Path(angles3d_module.__file__).resolve(),
    "angle3d_loss": Path(angle3d_module.__file__).resolve(),
    "object_level": Path(object_level_module.__file__).resolve(),
}
for import_name, origin in IMPORT_ORIGINS.items():
    if not origin.is_relative_to(REPO):
        raise RuntimeError(f"{import_name} import escaped audit checkout: {origin}")

EVIDENCE = HERE / "evidence"
OUTPUT = EVIDENCE / "scientific_edge_cases.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def clean_metrics(metrics: dict) -> dict:
    return {
        key: int(value) if key in {"tp", "fp", "fn"} else float(value)
        for key, value in metrics.items()
    }


def object_case(pred_t: int, true_t: int, n_frames: int = 5) -> dict:
    shape = (n_frames, 12, 12, 12)
    pred = np.zeros(shape, dtype=np.float32)
    true = np.zeros(shape, dtype=np.uint8)
    pred[pred_t, 6, 6, 6] = 1.0
    true[true_t, 6, 6, 6] = 1
    metrics, info = evaluate_at_object_level(
        pred,
        true,
        threshold=0.5,
        min_weighted_prob=0.0,
        distance_mode="iou",
        distance_threshold=1e-6,
    )
    return {
        "pred_frame": pred_t,
        "true_frame": true_t,
        "metrics": clean_metrics(metrics),
        "matched_items": [list(pair) for pair in info["matched_items"]],
        "iou_matrix": (
            (1.0 - np.asarray(info["distance_matrix"], dtype=float)).tolist()
            if len(info["distance_matrix"])
            else []
        ),
    }


def nematic_error_degrees(quaternion_a: np.ndarray, quaternion_b: np.ndarray) -> float:
    axis_a = np.array(quaternion_a[1:], dtype=float, copy=True)
    axis_b = np.array(quaternion_b[1:], dtype=float, copy=True)
    axis_a /= np.linalg.norm(axis_a)
    axis_b /= np.linalg.norm(axis_b)
    cosine = float(np.clip(abs(np.dot(axis_a, axis_b)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def axis_quaternion(angle_degrees: float, axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    half = math.radians(angle_degrees) / 2.0
    return np.concatenate(([math.cos(half)], math.sin(half) * axis))


def unordered_line_equal(points_a, points_b, atol: float = 1e-7) -> bool:
    a1, a2 = points_a
    b1, b2 = points_b
    direct = np.allclose(a1, b1, atol=atol) and np.allclose(a2, b2, atol=atol)
    reverse = np.allclose(a1, b2, atol=atol) and np.allclose(a2, b1, atol=atol)
    return bool(direct or reverse)


def vector_angle_degrees(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    a = np.asarray(vector_a, dtype=float)
    b = np.asarray(vector_b, dtype=float)
    cosine = abs(float(np.dot(a, b))) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)

    # The production filter skips stats index zero even though background has
    # already been removed from statistics_optimized.
    filter_prob = np.zeros((1, 3, 3, 5), dtype=np.float32)
    filter_prob[0, 1, 1, 0] = 0.1
    filter_prob[0, 1, 1, 4] = 0.1
    filter_binary = (filter_prob > 0).astype(np.uint8)
    filter_labels, filter_n = connected_components(filter_binary, return_N=True)
    filter_stats = statistics_optimized(filter_labels, filter_prob)
    maximum_size = 10.0
    minimum_weight = 0.02
    weights_before = [
        float(count / maximum_size * probability)
        for count, probability in zip(
            filter_stats["voxel_counts"], filter_stats["mean_prob"]
        )
    ]
    filtered = filter_by_object_weighted_prob(
        filter_binary.copy(),
        filter_labels,
        filter_stats,
        minimum_weight,
        maximum_size,
    )
    _, filter_n_after = connected_components(filtered, return_N=True)
    first_component = {
        "components_before": int(filter_n),
        "component_values_in_iteration_order": [
            int(value) for value in filter_stats["values"]
        ],
        "weighted_probabilities": weights_before,
        "minimum_weighted_probability": minimum_weight,
        "components_after": int(filter_n_after),
        "remaining_coordinates": np.argwhere(filtered > 0).tolist(),
        "interpretation": (
            "Both components are below the cutoff, but production code retains "
            "the first foreground component and removes the second."
        ),
    }
    assert filter_n == 2 and filter_n_after == 1
    assert first_component["remaining_coordinates"] == [[0, 1, 1, 0]]

    # Production uses '<', so a component exactly equal to the weighted cutoff
    # is retained. The first entry remains independently protected by the skip.
    equality_mat = np.zeros((1, 1, 1, 3), dtype=np.uint8)
    equality_mat[0, 0, 0, 0] = 1
    equality_mat[0, 0, 0, 2] = 1
    equality_labels = equality_mat.copy()
    equality_labels[0, 0, 0, 2] = 2
    equality_stats = {
        "voxel_counts": [1, 5],
        "mean_prob": [0.1, 0.4],
        "values": [1, 2],
    }
    exact_weight = (5 / 10) * 0.4
    equality_filtered = filter_by_object_weighted_prob(
        equality_mat.copy(),
        equality_labels,
        equality_stats,
        exact_weight,
        10,
    )
    weighted_equality = {
        "weighted_probability": exact_weight,
        "cutoff": exact_weight,
        "second_component_retained": bool(equality_filtered[0, 0, 0, 2]),
        "production_comparison": "weighted_probability < cutoff",
        "manuscript_wording_comparison": "weighted_probability > cutoff",
    }
    assert weighted_equality["second_component_retained"]

    # Evaluation includes values equal to threshold, while predict.py uses >.
    equality_pred = np.zeros((3, 12, 12, 12), dtype=np.float32)
    equality_true = np.zeros_like(equality_pred, dtype=np.uint8)
    equality_pred[1, 6, 6, 6] = 0.5
    equality_true[1, 6, 6, 6] = 1
    equality_metrics, _ = evaluate_at_object_level(
        equality_pred,
        equality_true,
        threshold=0.5,
        min_weighted_prob=0.0,
        distance_mode="iou",
        distance_threshold=1e-6,
    )
    probability_threshold_equality = {
        "value": 0.5,
        "threshold": 0.5,
        "evaluator_greater_equal_selected_voxels": int(
            np.count_nonzero(equality_pred >= 0.5)
        ),
        "predict_strict_greater_selected_voxels": int(
            np.count_nonzero(equality_pred > 0.5)
        ),
        "evaluator_metrics": clean_metrics(equality_metrics),
    }
    assert equality_metrics["tp"] == 1
    assert probability_threshold_equality["predict_strict_greater_selected_voxels"] == 0

    temporal_cases = {
        "same_frame": object_case(2, 2),
        "plus_one_frame": object_case(2, 3),
        "minus_one_frame": object_case(2, 1),
        "two_frames_apart": object_case(2, 4),
        "low_time_boundary_plus_one": object_case(0, 1),
        "high_time_boundary_minus_one": object_case(4, 3),
    }
    assert temporal_cases["same_frame"]["metrics"]["tp"] == 1
    assert temporal_cases["plus_one_frame"]["metrics"]["tp"] == 1
    assert temporal_cases["minus_one_frame"]["metrics"]["tp"] == 1
    assert temporal_cases["two_frames_apart"]["metrics"]["tp"] == 0
    assert temporal_cases["low_time_boundary_plus_one"]["metrics"]["tp"] == 1
    assert temporal_cases["high_time_boundary_minus_one"]["metrics"]["tp"] == 1

    # Two predictions separated by two frames are distinct before dilation but
    # overlap at the intervening frame after +/-1 temporal dilation.
    merge_pred = np.zeros((5, 12, 12, 12), dtype=np.float32)
    merge_pred[1, 6, 6, 6] = 1.0
    merge_pred[3, 6, 6, 6] = 1.0
    _, raw_pred_n = connected_components(merge_pred > 0, return_N=True)
    merge_dilated = dilate_img(merge_pred)
    _, dilated_pred_n = connected_components(merge_dilated > 0, return_N=True)
    merge_true = np.zeros_like(merge_pred, dtype=np.uint8)
    merge_true[1, 6, 6, 6] = 1
    merge_true[3, 6, 6, 6] = 1
    merge_metrics, merge_info = evaluate_at_object_level(
        merge_pred,
        merge_true,
        threshold=0.5,
        min_weighted_prob=0.0,
        distance_mode="iou",
        distance_threshold=1e-6,
    )
    adjacent_true = np.zeros_like(merge_true)
    adjacent_true[1, 6, 6, 6] = 1
    adjacent_true[2, 6, 6, 6] = 1
    _, adjacent_true_n = connected_components(adjacent_true, return_N=True)
    temporal_merging = {
        "prediction_components_before_dilation": int(raw_pred_n),
        "prediction_components_after_dilation": int(dilated_pred_n),
        "two_true_events_two_frames_apart_evaluation": clean_metrics(merge_metrics),
        "matched_items": [list(pair) for pair in merge_info["matched_items"]],
        "conceptual_adjacent_same_location_true_events_labelled_components": int(
            adjacent_true_n
        ),
        "interpretation": (
            "Temporal dilation merges two predictions two frames apart; 4D "
            "connected-component labeling also merges adjacent same-location "
            "ground-truth events."
        ),
    }
    assert raw_pred_n == 2 and dilated_pred_n == 1
    assert merge_metrics["tp"] == 1 and merge_metrics["fn"] == 1
    assert adjacent_true_n == 1

    greedy_matrix = np.asarray([[1.0, 2.0], [2.0, 100.0]])
    greedy_pairs = iterative_matching(greedy_matrix, max_distance=50.0)
    admissible_cost = greedy_matrix.copy()
    admissible_cost[admissible_cost >= 50.0] = 1e6
    rows, columns = linear_sum_assignment(admissible_cost)
    maximum_pairs = [
        (int(row), int(column))
        for row, column in zip(rows, columns)
        if greedy_matrix[row, column] < 50.0
    ]
    greedy_matching = {
        "distance_matrix": greedy_matrix.tolist(),
        "strict_max_distance": 50.0,
        "production_greedy_pairs": [list(pair) for pair in greedy_pairs],
        "production_match_count": len(greedy_pairs),
        "maximum_cardinality_pairs": [list(pair) for pair in maximum_pairs],
        "maximum_cardinality_count": len(maximum_pairs),
    }
    assert len(greedy_pairs) == 1 and len(maximum_pairs) == 2

    strict_distance_boundary = {
        "distance": 10.0,
        "threshold": 10.0,
        "matches_at_equality": [
            list(pair) for pair in iterative_matching(np.asarray([[10.0]]), 10.0)
        ],
        "matches_just_inside": [
            list(pair)
            for pair in iterative_matching(np.asarray([[9.999999]]), 10.0)
        ],
        "production_comparison": "distance < threshold",
    }
    assert not strict_distance_boundary["matches_at_equality"]
    assert strict_distance_boundary["matches_just_inside"] == [[0, 0]]

    empty = np.zeros((1, 8, 8, 8), dtype=np.float32)
    empty_metrics, _ = evaluate_at_object_level(
        empty,
        empty.astype(np.uint8),
        threshold=0.5,
        min_weighted_prob=0.0,
        distance_mode="iou",
        distance_threshold=1e-6,
    )
    only_true = empty.astype(np.uint8)
    only_true[0, 4, 4, 4] = 1
    no_pred_metrics, _ = evaluate_at_object_level(
        empty,
        only_true,
        threshold=0.5,
        min_weighted_prob=0.0,
        distance_mode="iou",
        distance_threshold=1e-6,
    )
    only_pred = empty.copy()
    only_pred[0, 4, 4, 4] = 1
    no_true_metrics, _ = evaluate_at_object_level(
        only_pred,
        empty.astype(np.uint8),
        threshold=0.5,
        min_weighted_prob=0.0,
        distance_mode="iou",
        distance_threshold=1e-6,
    )
    empty_cases = {
        "both_empty": clean_metrics(empty_metrics),
        "prediction_empty": clean_metrics(no_pred_metrics),
        "truth_empty": clean_metrics(no_true_metrics),
        "interpretation": "A both-empty movie receives precision=recall=F1=0.",
    }
    assert empty_metrics["fmeasure"] == 0
    assert no_pred_metrics["fn"] == 1
    assert no_true_metrics["fp"] == 1

    low_border = set_mat_value(
        np.zeros((1, 12, 12, 12), dtype=np.uint16), [0, 2, 6, 6], 1
    )
    interior = set_mat_value(
        np.zeros((1, 12, 12, 12), dtype=np.uint16), [0, 4, 4, 4], 1
    )
    high_border = set_mat_value(
        np.zeros((1, 12, 12, 12), dtype=np.uint16), [0, 11, 11, 11], 1
    )
    border_rendering = {
        "shape_txyz": [1, 12, 12, 12],
        "nominal_marker_voxels": 8**3,
        "center_x2_y6_z6_rendered_voxels": int(np.count_nonzero(low_border)),
        "center_x4_y4_z4_rendered_voxels": int(np.count_nonzero(interior)),
        "center_x11_y11_z11_rendered_voxels": int(np.count_nonzero(high_border)),
        "interpretation": (
            "Negative low-border slices can be empty, while high-border slices "
            "are clipped; displayed FP/FN counts can therefore be incomplete."
        ),
    }
    assert np.count_nonzero(low_border) == 0
    assert np.count_nonzero(interior) == 512
    assert np.count_nonzero(high_border) == 125

    movie_a = clean_metrics(wrap_metrics({"tp": 9, "fp": 0, "fn": 1}))
    movie_b = clean_metrics(wrap_metrics({"tp": 0, "fp": 1, "fn": 0}))
    pooled = clean_metrics(wrap_metrics({"tp": 9, "fp": 1, "fn": 1}))
    macro_f1 = (movie_a["fmeasure"] + movie_b["fmeasure"]) / 2.0
    aggregation = {
        "movie_a": movie_a,
        "movie_b": movie_b,
        "production_style_pooled_micro": pooled,
        "unweighted_macro_f1": macro_f1,
        "difference_micro_minus_macro": pooled["fmeasure"] - macro_f1,
    }
    assert not math.isclose(pooled["fmeasure"], macro_f1)

    axis_y = np.asarray([0.0, 1.0, 0.0])
    q60 = axis_quaternion(60.0, axis_y)
    q120 = axis_quaternion(120.0, axis_y)
    q60_reversed_axis = axis_quaternion(60.0, -axis_y)
    center = np.zeros(3)
    line_q60 = get_points_from_quat(q60.copy(), center, 1.0)
    line_q120 = get_points_from_quat(q120.copy(), center, 1.0)
    line_q60_reversed = get_points_from_quat(
        q60_reversed_axis.copy(), center, 1.0
    )
    quaternion_vs_nematic = {
        "same_rendered_axis_different_quaternion_angle": {
            "quaternion_error_degrees": float(quaternion_error(q60, q120)),
            "nematic_axis_error_degrees": nematic_error_degrees(q60, q120),
            "rendered_unoriented_lines_equal": unordered_line_equal(
                line_q60, line_q120
            ),
        },
        "reversed_equivalent_axis": {
            "quaternion_error_degrees": float(
                quaternion_error(q60, q60_reversed_axis)
            ),
            "nematic_axis_error_degrees": nematic_error_degrees(
                q60, q60_reversed_axis
            ),
            "rendered_unoriented_lines_equal": unordered_line_equal(
                line_q60, line_q60_reversed
            ),
        },
        "identical_quaternion": {
            "quaternion_error_degrees": float(quaternion_error(q60, q60)),
            "expected_exact_geometric_error_degrees": 0.0,
            "cause_of_nonzero_floor": (
                "quaternion_error clips the dot product to at most 0.9999."
            ),
        },
    }
    assert quaternion_vs_nematic[
        "same_rendered_axis_different_quaternion_angle"
    ]["nematic_axis_error_degrees"] == 0.0
    assert quaternion_vs_nematic[
        "same_rendered_axis_different_quaternion_angle"
    ]["quaternion_error_degrees"] > 59.9
    assert quaternion_vs_nematic["reversed_equivalent_axis"][
        "nematic_axis_error_degrees"
    ] == 0.0
    assert quaternion_vs_nematic["reversed_equivalent_axis"][
        "quaternion_error_degrees"
    ] > 119.9
    assert quaternion_vs_nematic["identical_quaternion"][
        "quaternion_error_degrees"
    ] > 1.6

    raw_true_vector = np.asarray([10.0, 0.0, 2.0])
    raw_pred_vector = np.asarray([10.0, 0.0, 0.0])
    illustrative_spacing = np.asarray([0.2, 0.2, 1.0])
    anisotropy = {
        "status": "illustrative diagnostic, not an inferred DARE3D scale",
        "true_vector_raw_xyz": raw_true_vector.tolist(),
        "predicted_vector_raw_xyz": raw_pred_vector.tolist(),
        "illustrative_spacing_xyz": illustrative_spacing.tolist(),
        "raw_voxel_angle_degrees": vector_angle_degrees(
            raw_true_vector, raw_pred_vector
        ),
        "physical_coordinate_angle_degrees": vector_angle_degrees(
            raw_true_vector * illustrative_spacing,
            raw_pred_vector * illustrative_spacing,
        ),
        "interpretation": (
            "Anisotropic voxel spacing can materially change a 3D axis error; "
            "the released missing scales.json prevents the actual conversion."
        ),
    }
    assert anisotropy["raw_voxel_angle_degrees"] < 12
    assert math.isclose(anisotropy["physical_coordinate_angle_degrees"], 45.0)

    source_files = [
        REPO / "dare3d/metrics/object_level.py",
        REPO / "dare3d/losses/angle3d.py",
        REPO / "dare3d/data/components/angles3d.py",
        REPO / "dare3d/predict.py",
    ]
    report = {
        "audit_date": "2026-08-29",
        "scope": (
            "Frozen, tiny synthetic cases against current production functions; "
            "no model, released artifact, or production source was modified."
        ),
        "runtime_import_origins": {
            name: origin.relative_to(REPO).as_posix()
            for name, origin in IMPORT_ORIGINS.items()
        },
        "production_source_sha256": {
            path.relative_to(REPO).as_posix(): sha256_file(path)
            for path in source_files
        },
        "first_component_filtering": first_component,
        "weighted_probability_equality": weighted_equality,
        "probability_threshold_equality": probability_threshold_equality,
        "temporal_window": temporal_cases,
        "temporal_and_4d_component_merging": temporal_merging,
        "greedy_matching_counterexample": greedy_matching,
        "strict_distance_boundary": strict_distance_boundary,
        "empty_movie_metrics": empty_cases,
        "border_rendering": border_rendering,
        "f1_aggregation": aggregation,
        "quaternion_vs_nematic": quaternion_vs_nematic,
        "voxel_anisotropy": anisotropy,
        "assertions": {
            "all_passed": True,
            "count": 24,
        },
    }
    report["audit_script_sha256"] = sha256_file(Path(__file__))
    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT.relative_to(REPO)}")
    print("All 24 frozen edge-case assertions passed.")


if __name__ == "__main__":
    main()

