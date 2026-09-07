"""Replay the released nuclei probability map without running a network.

The production object-level evaluator is called unchanged. Ground-truth masks are
reconstructed from the already-audited annotation-event table using the exact
radius-8, integer-truncated center convention in Segmentation3Dataset. All output
is audit evidence; released files and production source remain read-only.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import tifffile

from dare3d.metrics import object_level as object_level_module
from dare3d.metrics.object_level import evaluate_at_object_level, get_sphere_vol

OBJECT_LEVEL_IMPORT = Path(object_level_module.__file__).resolve()
if not OBJECT_LEVEL_IMPORT.is_relative_to(REPO):
    raise RuntimeError(f"dare3d import escaped audit checkout: {OBJECT_LEVEL_IMPORT}")

DATA = REPO / "DARE3d_data_190326"
EVIDENCE = HERE / "evidence"

RUN = DATA / "Gastruloid_241025/weights/segmentation3d_exp10-b/runs/01-01"
PROBABILITY = RUN / "movie2.tif"
LABEL = DATA / "Gastruloid_241025/trainingset/movie2/label/movie2.tif"
RELEASED_STATS = RUN / "stats.csv"
ANNOTATION_EVENTS = EVIDENCE / "annotation_events.csv"
SAVED_MATCHES = EVIDENCE / "saved_matches_nuclei_gastruloid.csv"
MANIFEST = EVIDENCE / "data_manifest_sha256.csv"

SUMMARY_OUT = EVIDENCE / "nuclei_saved_probability_replay.json"
OBJECTS_OUT = EVIDENCE / "nuclei_current_evaluator_objects.csv"
INDEPENDENT_OUT = EVIDENCE / "nuclei_independent_matches.csv"
ALIGNMENT_OUT = EVIDENCE / "nuclei_saved_visualization_alignment.csv"


def rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def manifest_hash(path: Path) -> str:
    wanted = rel(path)
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["path"] == wanted:
                return row["sha256"]
    raise KeyError(f"Path absent from manifest: {wanted}")


def load_movie2_events() -> list[dict]:
    rows = []
    wanted = rel(LABEL)
    with ANNOTATION_EVENTS.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["file"] != wanted:
                continue
            rows.append(
                {
                    "frame": int(row["frame"]),
                    "center_zyx": np.asarray(
                        [
                            float(row["center_z"]),
                            float(row["center_y"]),
                            float(row["center_x"]),
                        ],
                        dtype=float,
                    ),
                    "eligible": row["eligible_three_frame_input"].lower() == "true",
                }
            )
    if len(rows) != 226:
        raise AssertionError(f"Expected 226 movie2 annotations, found {len(rows)}")
    return rows


def sphere_offsets(radius: int) -> np.ndarray:
    values = np.arange(-radius, radius + 1, dtype=np.int16)
    zz, yy, xx = np.meshgrid(values, values, values, indexing="ij")
    offsets = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)
    return offsets[np.sum(offsets.astype(np.int32) ** 2, axis=1) <= radius**2]


def build_ground_truth(shape_txyz: tuple[int, ...], events: list[dict], radius: int) -> np.ndarray:
    mask = np.zeros(shape_txyz, dtype=np.uint8)
    # Offsets are generated in XYZ order; a sphere is invariant to the names.
    offsets = sphere_offsets(radius)
    spatial_shape = np.asarray(shape_txyz[1:], dtype=int)
    for event in events:
        if not event["eligible"]:
            continue
        # CSV stores disk ZYX. The production loader swaps Z and X to internal XYZ.
        z, y, x = event["center_zyx"]
        center_xyz = np.asarray([x, y, z], dtype=float).astype(int)
        coords = offsets.astype(int) + center_xyz
        valid = np.all((coords >= 0) & (coords < spatial_shape), axis=1)
        coords = coords[valid]
        t = event["frame"]
        mask[t, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    # infer_and_evaluate_segmentation explicitly zeros these frames.
    mask[:2] = 0
    return mask


def maximum_bipartite_matches(
    true_centers: np.ndarray,
    pred_centers: np.ndarray,
    true_indices: list[int],
    pred_indices: list[int],
    allowed,
) -> list[tuple[int, int]]:
    adjacency: dict[int, list[int]] = {}
    for i in true_indices:
        candidates = []
        for j in pred_indices:
            if allowed(i, j):
                distance = float(np.linalg.norm(true_centers[i, 1:] - pred_centers[j, 1:]))
                candidates.append((distance, j))
        adjacency[i] = [j for _, j in sorted(candidates)]

    pred_to_true: dict[int, int] = {}

    def augment(i: int, visited: set[int]) -> bool:
        for j in adjacency[i]:
            if j in visited:
                continue
            visited.add(j)
            if j not in pred_to_true or augment(pred_to_true[j], visited):
                pred_to_true[j] = i
                return True
        return False

    for i in true_indices:
        augment(i, set())
    return sorted((i, j) for j, i in pred_to_true.items())


def centers_for_event_matching(stats: dict) -> np.ndarray:
    values = np.asarray(stats["centroids"], dtype=float)
    rounded = values.copy()
    rounded[:, 0] = np.round(rounded[:, 0] + 1e-9)
    return rounded


def global_maximum_matches(
    true_centers: np.ndarray, pred_centers: np.ndarray, tolerance: float
) -> list[tuple[int, int]]:
    def allowed(i, j):
        dt = abs(true_centers[i, 0] - pred_centers[j, 0])
        spatial = np.linalg.norm(true_centers[i, 1:] - pred_centers[j, 1:])
        return dt <= 1 and spatial <= tolerance

    return maximum_bipartite_matches(
        true_centers,
        pred_centers,
        list(range(len(true_centers))),
        list(range(len(pred_centers))),
        allowed,
    )


def same_frame_priority_matches(
    true_centers: np.ndarray, pred_centers: np.ndarray, tolerance: float
) -> list[tuple[int, int]]:
    spatial = lambda i, j: np.linalg.norm(
        true_centers[i, 1:] - pred_centers[j, 1:]
    )

    same = maximum_bipartite_matches(
        true_centers,
        pred_centers,
        list(range(len(true_centers))),
        list(range(len(pred_centers))),
        lambda i, j: (
            abs(true_centers[i, 0] - pred_centers[j, 0]) == 0
            and spatial(i, j) <= tolerance
        ),
    )
    used_true = {i for i, _ in same}
    used_pred = {j for _, j in same}
    remaining_true = [i for i in range(len(true_centers)) if i not in used_true]
    remaining_pred = [j for j in range(len(pred_centers)) if j not in used_pred]
    adjacent = maximum_bipartite_matches(
        true_centers,
        pred_centers,
        remaining_true,
        remaining_pred,
        lambda i, j: (
            abs(true_centers[i, 0] - pred_centers[j, 0]) == 1
            and spatial(i, j) <= tolerance
        ),
    )
    return sorted(same + adjacent)


def greedy_distance_matches(
    true_centers: np.ndarray,
    pred_centers: np.ndarray,
    tolerance: float,
    same_frame_first: bool,
) -> list[tuple[int, int]]:
    used_true: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int]] = []
    stages = (0, 1) if same_frame_first else (None,)
    for stage in stages:
        candidates = []
        for i, true in enumerate(true_centers):
            if i in used_true:
                continue
            for j, pred in enumerate(pred_centers):
                if j in used_pred:
                    continue
                dt = abs(true[0] - pred[0])
                distance = float(np.linalg.norm(true[1:] - pred[1:]))
                time_allowed = dt <= 1 if stage is None else dt == stage
                if time_allowed and distance <= tolerance:
                    candidates.append((distance, dt, i, j))
        for _, _, i, j in sorted(candidates):
            if i not in used_true and j not in used_pred:
                used_true.add(i)
                used_pred.add(j)
                matches.append((i, j))
    return sorted(matches)


def matching_summary(
    name: str,
    pairs: list[tuple[int, int]],
    true_centers: np.ndarray,
    pred_centers: np.ndarray,
) -> dict:
    spatial = [
        float(np.linalg.norm(true_centers[i, 1:] - pred_centers[j, 1:]))
        for i, j in pairs
    ]
    temporal = [
        float(abs(true_centers[i, 0] - pred_centers[j, 0])) for i, j in pairs
    ]
    tp = len(pairs)
    fp = len(pred_centers) - tp
    fn = len(true_centers) - tp
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {
        "method": name,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "f1": f1,
        "max_spatial_distance_raw_voxels": max(spatial, default=None),
        "max_temporal_centroid_distance_frames": max(temporal, default=None),
        "matches_over_10_raw_voxels": sum(value > 10 for value in spatial),
    }


def independent_match_rows(
    methods: dict[str, list[tuple[int, int]]],
    true_centers: np.ndarray,
    pred_centers: np.ndarray,
) -> list[dict]:
    rows = []
    for method, pairs in methods.items():
        for i, j in pairs:
            true = true_centers[i]
            pred = pred_centers[j]
            rows.append(
                {
                    "method": method,
                    "true_index": i,
                    "pred_index": j,
                    "true_t_rounded": true[0],
                    "pred_t_rounded": pred[0],
                    "dt_frames": abs(true[0] - pred[0]),
                    "spatial_distance_raw_voxels": np.linalg.norm(
                        true[1:] - pred[1:]
                    ),
                    "true_x": true[1],
                    "true_y": true[2],
                    "true_z": true[3],
                    "pred_x": pred[1],
                    "pred_y": pred[2],
                    "pred_z": pred[3],
                }
            )
    return rows


def pipeline_object_rows(info: dict) -> list[dict]:
    true = np.asarray(info["true_ccs_stats"]["centroids"], dtype=float)
    pred = np.asarray(info["pred_ccs_stats"]["centroids"], dtype=float)
    pairs = list(info["matched_items"])
    distance = np.asarray(info["distance_matrix"], dtype=float)
    true_to_pred = {i: j for i, j in pairs}
    matched_pred = {j for _, j in pairs}
    rows = []
    for i, center in enumerate(true):
        j = true_to_pred.get(i)
        row = {
            "kind": "true",
            "index": i,
            "status": "tp" if j is not None else "fn",
            "matched_index": "" if j is None else j,
            "t": center[0],
            "x": center[1],
            "y": center[2],
            "z": center[3],
            "matched_iou": "" if j is None else 1.0 - distance[i, j],
        }
        rows.append(row)
    for j, center in enumerate(pred):
        rows.append(
            {
                "kind": "pred",
                "index": j,
                "status": "tp" if j in matched_pred else "fp",
                "matched_index": next(
                    (i for i, match_j in pairs if match_j == j), ""
                ),
                "t": center[0],
                "x": center[1],
                "y": center[2],
                "z": center[3],
                "matched_iou": next(
                    (1.0 - distance[i, j] for i, match_j in pairs if match_j == j),
                    "",
                ),
                "voxel_count": info["pred_ccs_stats"]["voxel_counts"][j],
                "mean_probability": info["pred_ccs_stats"]["mean_prob"][j],
                "weighted_probability": (
                    info["pred_ccs_stats"]["voxel_counts"][j]
                    / (get_sphere_vol(8) * 3)
                    * info["pred_ccs_stats"]["mean_prob"][j]
                ),
            }
        )
    return rows


def visualization_alignment(info: dict) -> list[dict]:
    saved = {}
    with SAVED_MATCHES.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            saved[int(row["match_id"])] = row

    true = np.asarray(info["true_ccs_stats"]["centroids"], dtype=float)
    pred = np.asarray(info["pred_ccs_stats"]["centroids"], dtype=float)
    distance = np.asarray(info["distance_matrix"], dtype=float)
    rows = []
    for match_id, (i, j) in enumerate(info["matched_items"], start=2):
        row = saved.get(match_id)
        if row is None:
            rows.append({"match_id": match_id, "saved_row_present": False})
            continue
        saved_true = np.asarray(
            [
                float(row["true_t"]),
                float(row["true_x"]),
                float(row["true_y"]),
                float(row["true_z"]),
            ]
        )
        saved_pred = np.asarray(
            [
                float(row["pred_t"]),
                float(row["pred_x"]),
                float(row["pred_y"]),
                float(row["pred_z"]),
            ]
        )
        # set_mat_value draws x-4:x+4, whose voxel centroid is x-0.5.
        saved_true_cube_center = saved_true.copy()
        saved_pred_cube_center = saved_pred.copy()
        saved_true_cube_center[1:] += 0.5
        saved_pred_cube_center[1:] += 0.5
        replay_true_rounded = np.round(true[i] + 1e-9)
        replay_pred_rounded = np.round(pred[j] + 1e-9)
        rows.append(
            {
                "match_id": match_id,
                "saved_row_present": True,
                "replay_true_index": i,
                "replay_pred_index": j,
                "replay_iou": 1.0 - distance[i, j],
                "true_rendered_center_vs_replay_rounded_distance": np.linalg.norm(
                    saved_true_cube_center - replay_true_rounded
                ),
                "pred_rendered_center_vs_replay_rounded_distance": np.linalg.norm(
                    saved_pred_cube_center - replay_pred_rounded
                ),
                "replay_true_t": true[i, 0],
                "replay_pred_t": pred[j, 0],
                "replay_spatial_distance_raw_voxels": np.linalg.norm(
                    true[i, 1:] - pred[j, 1:]
                ),
            }
        )
    return rows


def main() -> None:
    print("Loading released probability TIFF as a read-only memory map...", flush=True)
    probability_tzyx = tifffile.memmap(PROBABILITY, mode="r")
    probability_txyz = np.swapaxes(probability_tzyx, 1, -1)
    if probability_txyz.shape != (10, 362, 305, 180):
        raise AssertionError(f"Unexpected probability shape {probability_txyz.shape}")

    events = load_movie2_events()
    print("Constructing the frozen raw-coordinate ground-truth mask...", flush=True)
    ground_truth = build_ground_truth(probability_txyz.shape, events, radius=8)

    released = json.loads(RELEASED_STATS.read_text(encoding="utf-8"))
    released_seg = released["segmentation_results"]
    threshold = float(released_seg["threshold"])
    min_weighted_prob = float(released_seg["min_weighted_prob"])

    print("Calling the production object-level evaluator unchanged...", flush=True)
    replay, info = evaluate_at_object_level(
        y_pred=probability_txyz,
        y_true=ground_truth,
        threshold=threshold,
        min_weighted_prob=min_weighted_prob,
        movie_name=None,
        output_dir=None,
        distance_mode="iou",
        distance_threshold=0.000001,
    )
    replay["threshold"] = threshold
    replay["min_weighted_prob"] = min_weighted_prob

    true_centers = centers_for_event_matching(info["true_ccs_stats"])
    pred_centers = centers_for_event_matching(info["pred_ccs_stats"])
    methods = {
        "global_maximum_dt1_tol10_raw_voxels": global_maximum_matches(
            true_centers, pred_centers, 10.0
        ),
        "same_frame_priority_maximum_tol10_raw_voxels": same_frame_priority_matches(
            true_centers, pred_centers, 10.0
        ),
        "global_greedy_dt1_tol10_raw_voxels": greedy_distance_matches(
            true_centers, pred_centers, 10.0, same_frame_first=False
        ),
        "same_frame_priority_greedy_tol10_raw_voxels": greedy_distance_matches(
            true_centers, pred_centers, 10.0, same_frame_first=True
        ),
        "global_maximum_dt1_tol16_raw_voxels": global_maximum_matches(
            true_centers, pred_centers, 16.0
        ),
    }

    objects = pipeline_object_rows(info)
    independent = independent_match_rows(methods, true_centers, pred_centers)
    alignment = visualization_alignment(info)
    write_csv(OBJECTS_OUT, objects)
    write_csv(INDEPENDENT_OUT, independent)
    write_csv(ALIGNMENT_OUT, alignment)

    pred_stats = info["pred_ccs_stats"]
    weighted = [
        count / (get_sphere_vol(8) * 3) * prob
        for count, prob in zip(
            pred_stats["voxel_counts"], pred_stats["mean_prob"]
        )
    ]
    match_ious = [
        1.0 - info["distance_matrix"][i, j] for i, j in info["matched_items"]
    ]
    exact_counts = all(
        replay[key] == released_seg[key] for key in ("tp", "fp", "fn")
    )
    exact_metrics = exact_counts and all(
        math.isclose(replay[key], released_seg[key], rel_tol=0.0, abs_tol=1e-15)
        for key in ("precision", "recall", "fmeasure")
    )
    alignment_present = [row for row in alignment if row["saved_row_present"]]
    summary = {
        "classification": (
            "artifact-level current-evaluator replay exact"
            if exact_metrics
            else "artifact-level replay differs"
        ),
        "runtime_import_origin": rel(OBJECT_LEVEL_IMPORT),
        "inputs": {
            "probability_tiff": {
                "path": rel(PROBABILITY),
                "sha256": manifest_hash(PROBABILITY),
                "disk_shape_tzyx": list(probability_tzyx.shape),
                "internal_shape_txyz": list(probability_txyz.shape),
                "dtype": str(probability_txyz.dtype),
                "min": float(np.min(probability_txyz)),
                "max": float(np.max(probability_txyz)),
            },
            "annotation_label": {
                "path": rel(LABEL),
                "sha256": manifest_hash(LABEL),
                "raw_pairs": len(events),
                "three_frame_eligible_pairs": sum(e["eligible"] for e in events),
            },
            "released_stats": {
                "path": rel(RELEASED_STATS),
                "sha256": manifest_hash(RELEASED_STATS),
            },
        },
        "frozen_parameters": {
            "threshold": threshold,
            "min_weighted_prob": min_weighted_prob,
            "distance_mode": "iou",
            "iou_threshold": 0.000001,
            "iteration_method": "movie",
            "radius": 8,
            "first_ground_truth_frames_zeroed": 2,
            "coordinate_space": "raw output voxels after pipeline unscaling",
        },
        "released_segmentation_results": released_seg,
        "replayed_segmentation_results": replay,
        "exact_count_agreement": exact_counts,
        "exact_metric_agreement_1e_minus_15": exact_metrics,
        "components": {
            "true": len(info["true_ccs_stats"]["centroids"]),
            "predicted_after_current_filter": len(pred_stats["centroids"]),
            "matched": len(info["matched_items"]),
            "first_predicted_component_weighted_probability": (
                float(weighted[0]) if weighted else None
            ),
            "predicted_weighted_probability_min": (
                float(min(weighted)) if weighted else None
            ),
            "predicted_weighted_probability_max": (
                float(max(weighted)) if weighted else None
            ),
            "matched_iou_min": float(min(match_ious)) if match_ious else None,
            "matched_iou_max": float(max(match_ious)) if match_ious else None,
        },
        "independent_center_matching": [
            matching_summary(name, pairs, true_centers, pred_centers)
            for name, pairs in methods.items()
        ],
        "saved_visualization_alignment": {
            "saved_match_rows": len(alignment_present),
            "replay_match_rows": len(info["matched_items"]),
            "true_max_rendered_center_difference": max(
                (
                    row["true_rendered_center_vs_replay_rounded_distance"]
                    for row in alignment_present
                ),
                default=None,
            ),
            "pred_max_rendered_center_difference": max(
                (
                    row["pred_rendered_center_vs_replay_rounded_distance"]
                    for row in alignment_present
                ),
                default=None,
            ),
        },
        "interpretation_notes": [
            "This replays a saved probability map, not checkpoint inference.",
            "The production evaluator uses temporal dilation, 4D connected components, and greedy IoU matching.",
            "Independent 10/16-voxel results use raw output coordinates because the missing scales.json prevents a verified physical-coordinate conversion.",
            "The manuscript's stated 10-voxel rule is not equivalent to the frozen IoU evaluator.",
        ],
        "outputs": [
            rel(SUMMARY_OUT),
            rel(OBJECTS_OUT),
            rel(INDEPENDENT_OUT),
            rel(ALIGNMENT_OUT),
        ],
    }
    write_json(SUMMARY_OUT, summary)
    print(json.dumps(summary, indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
