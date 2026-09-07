"""Evaluate both regression checkpoints on training-consistent movie_I2 crops.

This audit-only control complements the exact released-center replay. It uses
the ordinary regression Dataset.init() path, including spatial resampling, and
therefore tests the input domain used during training and checkpoint selection.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch

import neural_tube_nematic_evaluation as evaluation
import neural_tube_nematic_experiment as common


OUTPUT_ROOT = (
    evaluation.EVALUATION_ROOT / "training_consistent_movie_I2_regression"
)
EVENT_PATH = OUTPUT_ROOT / "event_metrics.csv"
RESULT_PATH = OUTPUT_ROOT / "result.json"


def execute():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    dataset = common.make_dataset("regression", "validation")
    dataset.init()
    if len(dataset.crops) != 123:
        raise AssertionError(
            f"Expected 123 movie_I2 training-space crops, got {len(dataset.crops)}"
        )

    rows = []
    loaded_keys = {}
    started = time.perf_counter()
    for model_name in ("original", "retrained"):
        net, loaded_keys[model_name] = evaluation.load_regression_net(model_name)
        model_rows = evaluation.controlled_rows_for_model(
            model_name, net, dataset
        )
        for row in model_rows:
            row["scope"] = "training_consistent_movie_I2_annotation_crops"
            row["movie"] = "movie_I2"
        rows.extend(model_rows)
        del net
        gc.collect()
        torch.cuda.empty_cache()

    evaluation.write_csv(EVENT_PATH, rows)
    summary = evaluation.summarize_controlled(rows)
    result = {
        "status": "complete",
        "completed_at": common.now_iso(),
        "scope": (
            "movie_I2 annotated regression crops after the same spatial "
            "resampling used by training"
        ),
        "scientific_role": (
            "validation/input-domain diagnostic, not independent test and not "
            "the raw-coordinate production-center path"
        ),
        "dataset": common.dataset_record(dataset, "regression", "validation"),
        "checkpoints": {
            model: evaluation.checkpoint_metadata("regression", model)
            for model in ("original", "retrained")
        },
        "loaded_network_keys": loaded_keys,
        "summary": summary,
        "assertions": {
            "source_commit_unchanged":
                common.git_head() == common.SOURCE_COMMIT,
            "tracked_tree_clean": common.tracked_tree_is_clean(),
            "expected_unique_crop_count": len(dataset.crops) == 123,
            "all_corrected_axes_defined": all(
                row["corrected_axis_defined"] for row in rows
            ),
        },
        "outputs": {
            "event_metrics": common.rel(EVENT_PATH),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    common.atomic_write_json(RESULT_PATH, result)
    result["outputs"]["event_metrics_sha256"] = common.sha256(EVENT_PATH)
    common.atomic_write_json(RESULT_PATH, result)
    return result


def main():
    result = execute()
    compact = {
        "status": result["status"],
        "n": result["summary"]["by_model"]["original"]["sample_count"],
        "original_nematic_mean_deg": result["summary"]["by_model"]["original"][
            "corrected_nematic_axis_error_deg"
        ]["mean"],
        "retrained_nematic_mean_deg": result["summary"]["by_model"]["retrained"][
            "corrected_nematic_axis_error_deg"
        ]["mean"],
        "result": common.rel(RESULT_PATH),
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
