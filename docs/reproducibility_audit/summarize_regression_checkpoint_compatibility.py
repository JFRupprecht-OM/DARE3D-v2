"""Summarize the two predeclared nuclei regression checkpoint compatibility runs."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUN_ROOT = HERE / "checkpoint_runs/nuclei_regression"
EVIDENCE = HERE / "evidence"
RUNS = {
    "best_epoch_098": RUN_ROOT / "best_epoch_098/result.json",
    "last_last": RUN_ROOT / "last_last/result.json",
}
OUT_JSON = EVIDENCE / "nuclei_regression_checkpoint_compatibility_summary.json"
OUT_CSV = EVIDENCE / "nuclei_regression_checkpoint_compatibility_summary.csv"
MODES = (
    "all_groundtruth_centers",
    "matched_groundtruth_centers",
    "predicted_centers",
)
METRICS = (
    "n",
    "mean_angle_error",
    "std_angle_error",
    "mean_length_error",
    "std_length_error",
    "mean_distance_error",
    "std_distance_error",
)
MODE_TO_KEY = {
    "all_groundtruth_centers": "regression_performance_on_all_groundtruth_centers",
    "matched_groundtruth_centers": "regression_performance_on_matched_groundtruth_centers",
    "predicted_centers": "regression_performance_on_predicted_centers",
}


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def raw_comparison(best_dir: Path, last_dir: Path) -> dict:
    result = {}
    for mode in MODES:
        best = np.load(best_dir / f"{mode}_raw_predictions.npz")
        last = np.load(last_dir / f"{mode}_raw_predictions.npz")
        if not np.array_equal(best["centers"], last["centers"]):
            raise RuntimeError(f"Center arrays differ between best/last for {mode}")
        best_axis = best["quaternions"][:, 1:]
        last_axis = last["quaternions"][:, 1:]
        best_axis = best_axis / np.linalg.norm(best_axis, axis=1, keepdims=True)
        last_axis = last_axis / np.linalg.norm(last_axis, axis=1, keepdims=True)
        dot = np.clip(np.abs(np.sum(best_axis * last_axis, axis=1)), 0.0, 1.0)
        axis_deg = np.degrees(np.arccos(dot))
        length_abs = np.abs(best["lengths"].reshape(-1) - last["lengths"].reshape(-1))
        result[mode] = {
            "n": int(len(axis_deg)),
            "nematic_axis_difference_deg": {
                "mean": float(np.mean(axis_deg)),
                "median": float(np.median(axis_deg)),
                "max": float(np.max(axis_deg)),
            },
            "absolute_length_difference_voxels": {
                "mean": float(np.mean(length_abs)),
                "median": float(np.median(length_abs)),
                "max": float(np.max(length_abs)),
            },
        }
    return result


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    records = {name: load(path) for name, path in RUNS.items()}
    rows = []
    summaries = {}
    for name, record in records.items():
        differences = []
        exact_values = 0
        for mode in MODES:
            key = MODE_TO_KEY[mode]
            generated = record["generated_regression_results"][key]
            released = record["released_regression_results"][key]
            for metric in METRICS:
                generated_value = float(generated[metric])
                released_value = float(released[metric])
                difference = abs(generated_value - released_value)
                differences.append(difference)
                exact_values += difference == 0.0
                rows.append(
                    {
                        "run": name,
                        "checkpoint": record["inputs"]["checkpoint"],
                        "checkpoint_sha256": record["inputs"]["checkpoint_sha256_from_verified_manifest"],
                        "mode": mode,
                        "metric": metric,
                        "generated": generated_value,
                        "released": released_value,
                        "absolute_difference": difference,
                    }
                )
        summaries[name] = {
            "result_json": rel(RUNS[name]),
            "checkpoint": record["inputs"]["checkpoint"],
            "checkpoint_sha256": record["inputs"]["checkpoint_sha256_from_verified_manifest"],
            "status": record["status"],
            "strict_checkpoint_load": record["model"]["strict_checkpoint_load"],
            "center_counts": record["frozen_protocol"]["center_counts"],
            "effective_evaluated_counts": record["dataset"]["effective_evaluated_counts"],
            "maximum_absolute_difference_across_21_values": max(differences),
            "mean_absolute_difference_across_21_values": sum(differences) / len(differences),
            "rms_absolute_difference_across_21_values": math.sqrt(
                sum(value * value for value in differences) / len(differences)
            ),
            "exact_value_count": exact_values,
            "values_within_0_001": sum(value <= 0.001 for value in differences),
            "values_within_0_02": sum(value <= 0.02 for value in differences),
            "values_total": len(differences),
            "generated_results": record["generated_regression_results"],
            "elapsed_seconds": record["elapsed_seconds"],
            "peak_gpu_allocated_bytes": record["peak_gpu_allocated_bytes"],
        }

    raw = raw_comparison(RUN_ROOT / "best_epoch_098", RUN_ROOT / "last_last")
    best = records["best_epoch_098"]
    best_all = best["generated_regression_results"][
        "regression_performance_on_all_groundtruth_centers"
    ]
    best_sem_nominal = float(best_all["std_angle_error"]) / math.sqrt(float(best_all["n"]))
    best_max = summaries["best_epoch_098"]["maximum_absolute_difference_across_21_values"]
    last_max = summaries["last_last"]["maximum_absolute_difference_across_21_values"]
    assertions = {
        "both_predeclared_runs_complete": all(item["status"] == "complete" for item in records.values()),
        "both_checkpoints_load_strictly": all(
            item["model"]["strict_checkpoint_load"] for item in records.values()
        ),
        "both_use_frozen_145_125_124_components": all(
            item["frozen_protocol"]["center_counts"]["true_components"] == 145
            and item["frozen_protocol"]["center_counts"]["predicted_components"] == 125
            and item["frozen_protocol"]["center_counts"]["matched_components"] == 124
            for item in records.values()
        ),
        "both_reproduce_effective_counts_140_121_121": all(
            item["dataset"]["effective_evaluated_counts"]
            == {
                "all_groundtruth_centers": 140,
                "matched_groundtruth_centers": 121,
                "predicted_centers": 121,
            }
            for item in records.values()
        ),
        "best_is_closer_than_last_without_posthoc_checkpoint_search": best_max < last_max,
        "best_all_21_values_are_within_0_02_of_archive": summaries["best_epoch_098"][
            "values_within_0_02"
        ]
        == 21,
        "best_reproduces_headline_28_degree_rounding": round(float(best_all["std_angle_error"]))
        == 28,
        "best_reproduces_headline_2_degree_sem_rounding": round(best_sem_nominal) == 2,
        "best_and_last_raw_predictions_are_not_interchangeable": any(
            value["nematic_axis_difference_deg"]["max"] > 0.1 for value in raw.values()
        ),
    }

    report = {
        "scope": "Current-checkout nuclei regression checkpoint inference on frozen centers reconstructed from the released segmentation probability; no segmentation rerun, display rendering, threshold tuning, or retraining.",
        "selection_rule": "epoch_098.ckpt and last.ckpt were both declared before either regression run. Agreement was not used to select an unplanned checkpoint.",
        "released_reference_source": "docs/reproducibility_audit/evidence/provenance.json",
        "runs": summaries,
        "best_vs_last_raw_prediction_comparison": raw,
        "best_headline_trace": {
            "all_groundtruth_std_angle_error_deg": float(best_all["std_angle_error"]),
            "sd_over_sqrt_nominal_n_deg": best_sem_nominal,
            "manuscript_display": "28 degrees +/- 2 degrees",
            "note": "This numerical trace does not validate the manuscript wording: the central value is a standard deviation, while the +/- term is that same SD divided by sqrt(n).",
        },
        "assertions": assertions,
        "conclusions": {
            "checkpoint_identity": "epoch_098.ckpt is strongly supported: all 21 archived aggregate regression values are within 0.02, whereas last.ckpt has a maximum discrepancy of 1.4716.",
            "aggregate_status": "The validation-selected checkpoint reproduces the archived regression summaries at manuscript headline precision but not bitwise or at full stored precision.",
            "classification": "partially reproducible",
            "remaining_limits": [
                "Archived Gastruloid raw regression predictions are absent, preventing event-wise or bitwise historical-output comparison.",
                "The historical source/software/hardware environment is absent.",
                "The frozen centers inherit the current evaluator and released probability rather than a serialized historical center-pair object.",
                "The released full-quaternion statistic is not the manuscript's scientifically natural nematic division-axis error.",
            ],
        },
    }

    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    OUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    failed = [name for name, passed in assertions.items() if not passed]
    print(
        json.dumps(
            {
                "status": "complete" if not failed else "assertion_failure",
                "assertions_passed": len(assertions) - len(failed),
                "assertions_total": len(assertions),
                "failed": failed,
                "best_max_absolute_difference": best_max,
                "last_max_absolute_difference": last_max,
                "json": rel(OUT_JSON),
                "csv": rel(OUT_CSV),
            },
            indent=2,
        )
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
