"""Freeze the predeclared nuclei checkpoint/scale comparison as audit evidence."""
from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = HERE / "checkpoint_runs/nuclei_segmentation"
EVIDENCE = HERE / "evidence"

RUN_SPECS = (
    ("best_log_reconstructed", "log_reconstructed/best_epoch_067/batch_12/result.json"),
    ("last_log_reconstructed", "log_reconstructed/last_last/batch_12/result.json"),
    ("best_bundle_default", "bundle_default/best_epoch_067/batch_12/result.json"),
)


def load_result(relative_path: str) -> dict:
    path = RUNS / relative_path
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    if result.get("status") != "complete":
        raise RuntimeError(f"Incomplete checkpoint run: {path}")
    return result


def compact(name: str, relative_path: str, result: dict) -> dict:
    stats = result["current_evaluator_stats"]
    probability = result["probability_comparison"]
    return {
        "run": name,
        "result_json": (RUNS / relative_path).relative_to(REPO).as_posix(),
        "checkpoint": result["arguments"]["checkpoint"],
        "checkpoint_sha256": result["inputs"]["checkpoint_sha256_from_verified_manifest"],
        "scale_mode": result["arguments"]["scale_mode"],
        "scale_xyz": result["frozen_settings"]["default_scale_xyz"],
        "target_shape_txyz": result["dataset"]["actual_target_shape_txyz"],
        "tp": stats["tp"],
        "fp": stats["fp"],
        "fn": stats["fn"],
        "precision": stats["precision"],
        "recall": stats["recall"],
        "f1": stats["fmeasure"],
        "released_all_metrics_exact": result["released_all_metrics_exact"],
        "generated_probability_sha256": probability["generated_sha256"],
        "archived_probability_sha256": result["inputs"]["archived_probability_sha256"],
        "probability_sha256_exact": probability["file_sha256_exact_match"],
        "array_exact": probability["array_exact_match"],
        "exact_value_fraction": probability["exact_value_fraction"],
        "mae": probability["mae"],
        "rmse": probability["rmse"],
        "pearson_correlation": probability["pearson_correlation"],
        "binary_disagreements_at_0_55": probability["binary_disagreement_count_at_0_55"],
        "elapsed_inference_and_evaluation_seconds": result["elapsed_inference_and_evaluation_seconds"],
        "peak_gpu_allocated_bytes": result["gpu"]["peak_memory_allocated_bytes"],
    }


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    rows = [compact(name, path, load_result(path)) for name, path in RUN_SPECS]
    by_name = {row["run"]: row for row in rows}

    assertions = {
        "all_three_runs_complete": len(rows) == 3,
        "best_log_exact_released_metrics": by_name["best_log_reconstructed"]["released_all_metrics_exact"],
        "last_log_not_exact_released_metrics": not by_name["last_log_reconstructed"]["released_all_metrics_exact"],
        "bundle_default_not_exact_released_metrics": not by_name["best_bundle_default"]["released_all_metrics_exact"],
        "best_log_near_archived_probability": by_name["best_log_reconstructed"]["pearson_correlation"] > 0.999999,
        "best_log_not_bitwise_archived_probability": not by_name["best_log_reconstructed"]["array_exact"],
        "last_log_materially_differs_from_best_log": by_name["last_log_reconstructed"]["binary_disagreements_at_0_55"] > 1000,
        "bundle_default_materially_breaks_result": by_name["best_bundle_default"]["f1"] < 0.5,
        "scale_modes_have_declared_distinct_geometry": (
            by_name["best_log_reconstructed"]["target_shape_txyz"] == [10, 330, 278, 164]
            and by_name["best_bundle_default"]["target_shape_txyz"] == [10, 224, 189, 360]
        ),
    }
    if not all(assertions.values()):
        failed = [name for name, passed in assertions.items() if not passed]
        raise AssertionError(f"Checkpoint summary assertions failed: {failed}")

    report = {
        "scope": "Current-checkout nuclei segmentation inference; no retraining",
        "selection_rule": (
            "Best and last checkpoints were declared before inference. The log-reconstructed "
            "scale was fixed from archived training-log geometry before output inspection; the "
            "bundle-default scale is the frozen config fallback when scales.json is missing."
        ),
        "released_reference": {"tp": 124, "fp": 1, "fn": 21, "f1": 0.9185185185185186},
        "runs": rows,
        "assertions": assertions,
        "conclusions": {
            "checkpoint_identity": (
                "epoch_067.ckpt is strongly identified as the source checkpoint because it "
                "reproduces all released object metrics under independently reconstructed geometry; "
                "last.ckpt does not."
            ),
            "public_workflow": (
                "The documented missing-scale fallback does not reproduce the release "
                "(32 TP, 5 FP, 113 FN, F1 0.351648)."
            ),
            "bitwise_status": (
                "No generated probability is bitwise identical to the archive. The epoch-067 "
                "log-reconstructed output is nevertheless numerically near-identical and has only "
                "92 threshold disagreements among 198,738,000 voxels."
            ),
            "classification": "partially reproducible",
            "remaining_blockers": [
                "original scales.json",
                "historical source snapshot",
                "historical software/hardware execution environment",
            ],
        },
    }

    json_path = EVIDENCE / "nuclei_checkpoint_compatibility_summary.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    csv_path = EVIDENCE / "nuclei_checkpoint_compatibility_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({
        "status": "complete",
        "assertions_passed": sum(assertions.values()),
        "assertions_total": len(assertions),
        "json": json_path.relative_to(REPO).as_posix(),
        "csv": csv_path.relative_to(REPO).as_posix(),
    }, indent=2))


if __name__ == "__main__":
    main()
