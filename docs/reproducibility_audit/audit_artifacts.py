"""Independent artifact-level audit for the released DARE3D bundle.

The script writes machine-readable evidence beside itself. It never modifies
production code, checkpoints, predictions, or downloaded data.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
from scipy import ndimage
from skimage.measure import regionprops
import tifffile


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "DARE3d_data_190326"
MANUSCRIPT = REPO / "manuscript_280826version"
EVIDENCE = HERE / "evidence"
EVIDENCE.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def file_hash(path: Path, algorithm: str = "sha256", chunk_size: int = 8 << 20) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_json(name: str, value) -> None:
    (EVIDENCE / name).write_text(
        json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def write_csv(name: str, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with (EVIDENCE / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_git(*args: str) -> str:
    command = ["git", "-c", f"safe.directory={REPO.as_posix()}", *args]
    return subprocess.check_output(command, cwd=REPO, text=True).strip()


def build_manifest(root: Path, name: str) -> list[dict]:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rows.append({"path": rel(path), "bytes": path.stat().st_size, "sha256": file_hash(path)})
        print(f"hashed {rel(path)}", flush=True)
    write_csv(name, rows, ["path", "bytes", "sha256"])
    return rows


def source_manifest() -> list[dict]:
    rows = []
    for item in run_git("ls-files").splitlines():
        path = REPO / item
        if path.is_file():
            rows.append(
                {
                    "path": item.replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": file_hash(path),
                }
            )
    write_csv("source_manifest_sha256.csv", rows, ["path", "bytes", "sha256"])
    return rows


def tiff_inventory() -> list[dict]:
    rows = []
    content_tokens = ("/trainingset/", "/test_input/", "/segmentation3d_")
    for path in sorted(DATA.rglob("*.tif")):
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            row = {
                "path": rel(path),
                "shape": "x".join(str(v) for v in series.shape),
                "dtype": str(series.dtype),
                "axes": str(series.axes),
                "pages": len(tif.pages),
                "bytes": path.stat().st_size,
                "min": "",
                "max": "",
            }
        normalized = "/" + rel(path)
        if any(token in normalized for token in content_tokens) and "/regression/0/" not in normalized:
            try:
                array = tifffile.memmap(path)
            except Exception:
                array = tifffile.imread(path)
            row["min"] = float(np.min(array))
            row["max"] = float(np.max(array))
            del array
        rows.append(row)
    write_csv(
        "tiff_inventory.csv",
        rows,
        ["path", "shape", "dtype", "axes", "pages", "bytes", "min", "max"],
    )
    return rows


def frame_regions(frame: np.ndarray) -> dict[int, np.ndarray]:
    return {int(p.label): np.asarray(p.centroid, dtype=float) for p in regionprops(frame)}


def union_find_count(events: list[dict], radius: float = 8.0) -> int:
    """Estimate 4-D components formed by the code's radius-r center spheres."""
    parent = list(range(len(events)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[b] = a

    centers = [np.floor(np.asarray(e["center_zyx"], float)).astype(int) for e in events]
    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            if abs(events[i]["frame"] - events[j]["frame"]) <= 1:
                if np.linalg.norm(centers[i] - centers[j]) <= 2 * radius:
                    union(i, j)
    return len({find(i) for i in range(len(events))})


def analyse_label(path: Path) -> tuple[dict, list[dict]]:
    array = tifffile.imread(path)
    if array.ndim == 3:
        array = array[np.newaxis, ...]
    events = []
    missing = []
    noncontiguous = []
    for frame_index, frame in enumerate(array):
        regions = frame_regions(frame)
        labels = sorted(regions)
        for value in labels:
            counterpart = value - 1 if value % 2 == 0 else value + 1
            if counterpart not in regions:
                missing.append({"frame": frame_index, "label": value, "expected": counterpart})
        for odd in (value for value in labels if value % 2 == 1):
            even = odd + 1
            if even not in regions:
                continue
            p1, p2 = regions[odd], regions[even]
            events.append(
                {
                    "file": rel(path),
                    "frame": frame_index,
                    "odd_label": odd,
                    "even_label": even,
                    "p1_zyx": p1.tolist(),
                    "p2_zyx": p2.tolist(),
                    "center_zyx": ((p1 + p2) / 2).tolist(),
                    "length_voxels": float(np.linalg.norm(p2 - p1)),
                    "eligible_three_frame_input": frame_index >= 2,
                }
            )
        expected = list(range(1, max(labels, default=0) + 1))
        if labels and labels != expected:
            noncontiguous.append(
                {"frame": frame_index, "labels": labels, "expected_contiguous": expected}
            )
    lengths = np.asarray([e["length_voxels"] for e in events], dtype=float)
    eligible = [e for e in events if e["eligible_three_frame_input"]]
    summary = {
        "path": rel(path),
        "shape_tzyx": list(array.shape),
        "dtype": str(array.dtype),
        "frames": int(array.shape[0]),
        "events_all_frames": len(events),
        "events_three_frame_eligible": len(eligible),
        "estimated_4d_center_components_eligible": union_find_count(eligible),
        "missing_counterparts": missing,
        "noncontiguous_label_frames": noncontiguous,
        "length_mean_voxels": float(lengths.mean()) if len(lengths) else None,
        "length_sd_population_voxels": float(lengths.std(ddof=0)) if len(lengths) else None,
        "length_sd_sample_voxels": float(lengths.std(ddof=1)) if len(lengths) > 1 else None,
        "length_sem_voxels": (
            float(lengths.std(ddof=1) / math.sqrt(len(lengths))) if len(lengths) > 1 else None
        ),
    }
    del array
    return summary, events


def annotation_inventory() -> tuple[list[dict], list[dict]]:
    summaries, events = [], []
    for path in sorted(DATA.glob("Gastruloid_241025/trainingset/*/label/*.tif")):
        summary, local = analyse_label(path)
        summaries.append(summary)
        events.extend(local)
        print(f"annotations {rel(path)}: {summary['events_all_frames']} pairs", flush=True)
    write_json("annotation_summary.json", summaries)
    rows = [
        {
            "file": e["file"],
            "frame": e["frame"],
            "odd_label": e["odd_label"],
            "even_label": e["even_label"],
            "center_z": e["center_zyx"][0],
            "center_y": e["center_zyx"][1],
            "center_x": e["center_zyx"][2],
            "length_voxels": e["length_voxels"],
            "eligible_three_frame_input": e["eligible_three_frame_input"],
        }
        for e in events
    ]
    write_csv("annotation_events.csv", rows)
    return summaries, events


def extract_saved_events(array: np.ndarray, prefix: str) -> list[dict]:
    events = []
    for t, frame in enumerate(array):
        for prop in regionprops(frame):
            if int(prop.label) >= 2:
                events.append(
                    {
                        "key": f"match_{int(prop.label)}",
                        "match_id": int(prop.label),
                        "t": t,
                        "zyx": np.asarray(prop.centroid, float),
                        "kind": prefix,
                    }
                )
        unmatched, _ = ndimage.label(frame == 1)
        for prop in regionprops(unmatched):
            events.append(
                {
                    "key": f"{prefix}_unmatched_t{t}_{int(prop.label)}",
                    "match_id": None,
                    "t": t,
                    "zyx": np.asarray(prop.centroid, float),
                    "kind": prefix,
                }
            )
    return events


def maximum_bipartite_matches(
    true_events: list[dict], pred_events: list[dict], tolerance: float
) -> list[tuple[int, int]]:
    adjacency = []
    for true in true_events:
        candidates = []
        for j, pred in enumerate(pred_events):
            dt = abs(true["t"] - pred["t"])
            distance = float(np.linalg.norm(true["zyx"] - pred["zyx"]))
            if dt <= 1 and distance <= tolerance:
                candidates.append((distance, j))
        adjacency.append([j for _, j in sorted(candidates)])
    pred_to_true = {}

    def augment(i: int, visited: set[int]) -> bool:
        for j in adjacency[i]:
            if j in visited:
                continue
            visited.add(j)
            if j not in pred_to_true or augment(pred_to_true[j], visited):
                pred_to_true[j] = i
                return True
        return False

    for i in range(len(true_events)):
        augment(i, set())
    return sorted((i, j) for j, i in pred_to_true.items())


def analyse_saved_detection(case: str, run_dir: Path, movie: str) -> dict:
    true_events = extract_saved_events(
        tifffile.imread(run_dir / movie / f"{movie}_true.tif"), "true"
    )
    pred_events = extract_saved_events(
        tifffile.imread(run_dir / movie / f"{movie}_pred.tif"), "pred"
    )
    true_by_id = {e["match_id"]: e for e in true_events if e["match_id"] is not None}
    pred_by_id = {e["match_id"]: e for e in pred_events if e["match_id"] is not None}
    shared_ids = sorted(set(true_by_id) & set(pred_by_id))
    rows = []
    for match_id in shared_ids:
        true, pred = true_by_id[match_id], pred_by_id[match_id]
        rows.append(
            {
                "case": case,
                "match_id": match_id,
                "true_t": true["t"],
                "pred_t": pred["t"],
                "dt": abs(true["t"] - pred["t"]),
                "true_z": true["zyx"][0],
                "true_y": true["zyx"][1],
                "true_x": true["zyx"][2],
                "pred_z": pred["zyx"][0],
                "pred_y": pred["zyx"][1],
                "pred_x": pred["zyx"][2],
                "spatial_distance_voxels": float(
                    np.linalg.norm(true["zyx"] - pred["zyx"])
                ),
            }
        )
    write_csv(f"saved_matches_{case}.csv", rows)
    return {
        "case": case,
        "true_events_reconstructed": len(true_events),
        "pred_events_reconstructed": len(pred_events),
        "saved_tp_shared_ids": len(shared_ids),
        "saved_fn_unmatched_components": sum(e["match_id"] is None for e in true_events),
        "saved_fp_unmatched_components": sum(e["match_id"] is None for e in pred_events),
        "saved_match_max_dt": max((row["dt"] for row in rows), default=None),
        "saved_match_max_spatial_distance": max(
            (row["spatial_distance_voxels"] for row in rows), default=None
        ),
        "saved_matches_over_10_voxels": sum(
            row["spatial_distance_voxels"] > 10 for row in rows
        ),
        "independent_matches_tolerance_10": len(
            maximum_bipartite_matches(true_events, pred_events, 10.0)
        ),
        "independent_matches_tolerance_16": len(
            maximum_bipartite_matches(true_events, pred_events, 16.0)
        ),
    }


def f1_ci(tp: int, fp: int, fn: int) -> dict:
    """Delta CI under a TP/FP/FN multinomial, with a diagnostic logit CI."""
    n = tp + fp + fn
    p_tp = tp / n
    f1 = 2 * tp / (2 * tp + fp + fn)
    derivative = 2 / (1 + p_tp) ** 2
    se = derivative * math.sqrt(p_tp * (1 - p_tp) / n)
    z = 1.959963984540054
    normal = [max(0.0, f1 - z * se), min(1.0, f1 + z * se)]
    logit = math.log(f1 / (1 - f1))
    logit_se = se / (f1 * (1 - f1))
    invlogit = lambda x: 1 / (1 + math.exp(-x))
    transformed = [invlogit(logit - z * logit_se), invlogit(logit + z * logit_se)]
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "f1": f1,
        "delta_se": se,
        "normal_95_ci": normal,
        "logit_delta_95_ci": transformed,
    }


def uncertainty_simulations(seed: int = 20260828, samples: int = 2_000_000) -> dict:
    rng = np.random.default_rng(seed)

    def nematic_errors(true_vector, p1, p2):
        vectors = (p2 - p1) + true_vector
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        truth = true_vector / np.linalg.norm(true_vector)
        dots = np.clip(np.abs(vectors @ truth), 0.0, 1.0)
        return np.degrees(np.arccos(dots))

    true_nuclei = np.asarray([15.0, 0.0, 0.0])
    nuclei = nematic_errors(
        true_nuclei,
        rng.normal(0, 3, size=(samples, 3)),
        rng.normal(0, 3, size=(samples, 3)),
    )
    scale = np.asarray([0.2, 0.2, 1.0])
    true_membrane = np.asarray([30.0, 0.0, 0.0]) * scale
    membrane = nematic_errors(
        true_membrane,
        rng.normal(0, 2, size=(samples, 3)) * scale,
        rng.normal(0, 2, size=(samples, 3)) * scale,
    )
    return {
        "seed": seed,
        "samples": samples,
        "nuclei_isotropic_length15_sigma3": {
            "mean_deg": float(nuclei.mean()),
            "sd_deg": float(nuclei.std()),
            "median_deg": float(np.median(nuclei)),
            "sem_deg": float(nuclei.std() / math.sqrt(samples)),
        },
        "membrane_scale_0.2_0.2_1_length30_sigma2_inplane": {
            "mean_deg": float(membrane.mean()),
            "sd_deg": float(membrane.std()),
            "median_deg": float(np.median(membrane)),
            "sem_deg": float(membrane.std() / math.sqrt(samples)),
        },
        "random_nematic_3d_rms_deg": math.degrees(math.sqrt(math.pi - 2)),
        "random_nematic_2d_rms_deg": math.degrees(math.pi / math.sqrt(12)),
    }


def npz_inventory() -> list[dict]:
    rows = []
    for path in sorted(DATA.rglob("*.npz")):
        with np.load(path) as archive:
            for key in archive.files:
                value = archive[key]
                rows.append(
                    {
                        "path": rel(path),
                        "key": key,
                        "shape": "x".join(str(v) for v in value.shape),
                        "dtype": str(value.dtype),
                        "finite": bool(np.isfinite(value).all()),
                        "min": float(value.min()) if value.size else None,
                        "max": float(value.max()) if value.size else None,
                    }
                )
    write_csv("npz_inventory.csv", rows)
    return rows


def provenance() -> dict:
    archive = REPO / "DARE3d_data_190326.zip"
    nuclei_stats = json.loads(
        (
            DATA
            / "Gastruloid_241025/weights/segmentation3d_exp10-b/runs/01-01/stats.csv"
        ).read_text()
    )
    membrane_stats = json.loads(
        (
            DATA
            / "Neural_tube_160226/segmentation3d_new_set_og/runs/12-01-26/stats.csv"
        ).read_text()
    )
    return {
        "audit_python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "git_head": run_git("rev-parse", "HEAD"),
        "git_status": run_git("status", "--short", "--untracked-files=all"),
        "zenodo_archive": {
            "path": rel(archive),
            "bytes": archive.stat().st_size,
            "md5": file_hash(archive, "md5"),
            "sha256": file_hash(archive),
        },
        "released_stats": {
            "nuclei_gastruloid": nuclei_stats,
            "membrane_neural_tube": membrane_stats,
        },
    }


def duplicate_groups(manifest: list[dict]) -> list[dict]:
    by_hash = defaultdict(list)
    for row in manifest:
        by_hash[row["sha256"]].append(row["path"])
    return [
        {"sha256": digest, "paths": paths}
        for digest, paths in sorted(by_hash.items())
        if len(paths) > 1
    ]


def main() -> None:
    if not DATA.is_dir():
        raise SystemExit(f"Missing data bundle: {DATA}")
    print("Building SHA-256 manifests...", flush=True)
    data_manifest = build_manifest(DATA, "data_manifest_sha256.csv")
    manuscript_manifest = build_manifest(MANUSCRIPT, "manuscript_manifest_sha256.csv")
    source = source_manifest()
    write_json("duplicate_artifacts.json", duplicate_groups(data_manifest))

    print("Inspecting TIFFs and annotations...", flush=True)
    tiffs = tiff_inventory()
    annotations, events = annotation_inventory()
    npz = npz_inventory()

    gastro_run = (
        DATA / "Gastruloid_241025/weights/segmentation3d_exp10-b/runs/01-01"
    )
    neural_run = (
        DATA
        / "Neural_tube_160226/segmentation3d_new_set_og/runs/12-01-26"
    )
    saved_detection = [
        analyse_saved_detection("nuclei_gastruloid", gastro_run, "movie2"),
        analyse_saved_detection("membrane_neural_tube", neural_run, "movie_I2"),
    ]
    write_json("saved_detection_reconstruction.json", saved_detection)

    arithmetic = {
        "nuclei_gastruloid": f1_ci(124, 1, 21),
        "membrane_neural_tube": f1_ci(114, 8, 8),
        "uncertainty": uncertainty_simulations(),
    }
    write_json("arithmetic_and_uncertainty.json", arithmetic)
    write_json("provenance.json", provenance())

    summary = {
        "data_files": len(data_manifest),
        "manuscript_files": len(manuscript_manifest),
        "tracked_source_files": len(source),
        "tiff_files": len(tiffs),
        "annotation_files": len(annotations),
        "annotation_events": len(events),
        "npz_arrays": len(npz),
        "saved_detection": saved_detection,
        "f1": {
            key: arithmetic[key]
            for key in ("nuclei_gastruloid", "membrane_neural_tube")
        },
    }
    write_json("artifact_audit_summary.json", summary)
    print(json.dumps(summary, indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()

