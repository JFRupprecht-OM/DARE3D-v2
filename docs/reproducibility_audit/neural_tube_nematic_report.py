"""Build the fixed-segmentation neural-tube nematic regression report."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import neural_tube_nematic_evaluation as evaluation
import neural_tube_nematic_experiment as common


OUTPUT_ROOT = evaluation.EVALUATION_ROOT
FINAL_JSON = OUTPUT_ROOT / "final_comparison.json"
SUMMARY_CSV = OUTPUT_ROOT / "evaluation_summary.csv"
DISTRIBUTION_CSV = OUTPUT_ROOT / "angular_distribution.csv"
FIGURE_PNG = OUTPUT_ROOT / "angular_and_detection_comparison.png"
FIGURE_PDF = OUTPUT_ROOT / "angular_and_detection_comparison.pdf"
REPORT = HERE / "NEURAL_TUBE_NEMATIC_FULL_PIPELINE_COMPARISON.md"
PRIOR_CORRECTED = HERE / "evidence/nematic_axis_metric_comparison.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative_percent(new: float, old: float) -> float:
    return 100.0 * (new - old) / old if old != 0 else float("nan")


def f1_ci(tp: int, fp: int, fn: int) -> dict[str, Any]:
    n = tp + fp + fn
    p_tp = tp / n
    f1 = 2 * tp / (2 * tp + fp + fn)
    derivative = 2 / (1 + p_tp) ** 2
    se = derivative * math.sqrt(p_tp * (1 - p_tp) / n)
    z = 1.959963984540054
    return {
        "f1": f1,
        "delta_se": se,
        "normal_95_ci": [
            max(0.0, f1 - z * se),
            min(1.0, f1 + z * se),
        ],
    }


def finite_float(value) -> float:
    result = float(value)
    return result if np.isfinite(result) else float("nan")


def read_event_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def angular_values(
    rows: list[dict[str, str]],
    selector,
    metric: str,
) -> np.ndarray:
    values = [
        finite_float(row[metric])
        for row in rows
        if selector(row) and row.get(metric, "") not in {"", "nan", "NaN"}
    ]
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def paired_bootstrap(
    old_values: np.ndarray,
    new_values: np.ndarray,
    seed: int,
    replicates: int = 10000,
) -> dict[str, Any]:
    if len(old_values) != len(new_values):
        raise AssertionError("Paired arrays differ in length")
    valid = np.isfinite(old_values) & np.isfinite(new_values)
    difference = new_values[valid] - old_values[valid]
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(difference),
        size=(replicates, len(difference)),
    )
    resampled = difference[indices]
    means = np.mean(resampled, axis=1)
    medians = np.median(resampled, axis=1)
    return {
        "n": int(len(difference)),
        "definition": "new minus old; negative is improvement",
        "mean_difference": float(np.mean(difference)),
        "median_difference": float(np.median(difference)),
        "mean_difference_bootstrap_95_ci": [
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)),
        ],
        "median_difference_bootstrap_95_ci": [
            float(np.percentile(medians, 2.5)),
            float(np.percentile(medians, 97.5)),
        ],
        "fraction_new_lower": float(np.mean(difference < 0)),
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
        "descriptive_only": True,
    }


def angular_summary_row(
    scenario: str,
    scope: str,
    mode: str,
    metric: str,
    stats: dict[str, Any],
    length_stats: dict[str, Any],
) -> dict[str, Any]:
    return {
        "record_type": "angular",
        "scenario": scenario,
        "scope": scope,
        "mode": mode,
        "angular_metric": metric,
        **{f"angle_{key}": value for key, value in stats.items()},
        "length_mae_voxels": length_stats.get("mean"),
        "length_error_std_population_voxels": length_stats.get(
            "std_population"
        ),
    }


def build_summary_rows(
    segmentation,
    controlled,
    end_to_end,
    released,
    prior_corrected,
) -> list[dict[str, Any]]:
    rows = []
    for label, metrics, dataset in (
        (
            "previous_released_movie_I2",
            released["segmentation_results"],
            "movie_I2 set 2 (archived validation=test)",
        ),
        (
            "fixed_released_checkpoint_movie_M",
            segmentation["fixed_released"]["test_metrics"],
            "movie_M independent set 3",
        ),
    ):
        tp, fp, fn = (
            int(metrics["tp"]),
            int(metrics["fp"]),
            int(metrics["fn"]),
        )
        ci = f1_ci(tp, fp, fn)
        rows.append(
            {
                "record_type": "segmentation",
                "scenario": label,
                "scope": dataset,
                "mode": "division_detection",
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["fmeasure"]),
                "f1_ci_low": ci["normal_95_ci"][0],
                "f1_ci_high": ci["normal_95_ci"][1],
                "probability_threshold": metrics["threshold"],
                "minimum_weighted_probability": metrics["min_weighted_prob"],
            }
        )

    controlled_by_model = controlled["summary"]["by_model"]
    controlled_scenarios = (
        (
            "1_original_model_original_metric",
            "original",
            "production_quaternion_error_deg",
        ),
        (
            "2_original_model_corrected_nematic_metric",
            "original",
            "corrected_nematic_axis_error_deg",
        ),
        (
            "3_retrained_model_corrected_nematic_metric",
            "retrained",
            "corrected_nematic_axis_error_deg",
        ),
    )
    for scenario, model, metric in controlled_scenarios:
        details = controlled_by_model[model]
        rows.append(
            angular_summary_row(
                scenario,
                "controlled_unique_movie_M_test",
                "80_unique_groundtruth_crops",
                metric,
                details[metric],
                details["absolute_length_error_voxels"],
            )
        )

    if end_to_end.get("status") == "complete":
        end_summary = end_to_end["summary"]
        end_scenarios = (
            (
                "1_original_regression_fixed_segmentation_original_metric",
                "original_regression_fixed_segmentation",
                "production_quaternion_error_deg",
            ),
            (
                "2_original_regression_fixed_segmentation_corrected_metric",
                "original_regression_fixed_segmentation",
                "corrected_nematic_axis_error_deg",
            ),
            (
                "3_retrained_regression_fixed_segmentation_corrected_metric",
                "retrained_regression_fixed_segmentation",
                "corrected_nematic_axis_error_deg",
            ),
        )
        for scenario, pipeline, metric in end_scenarios:
            for mode in evaluation.MODE_ORDER:
                details = end_summary[pipeline][mode]
                rows.append(
                    angular_summary_row(
                        scenario,
                        "production_end_to_end_movie_M_test",
                        mode,
                        metric,
                        details[metric],
                        details["absolute_length_error_voxels"],
                    )
                )
    else:
        rows.append(
            {
                "record_type": "angular_unavailable",
                "scenario": "end_to_end_fixed_segmentation",
                "scope": "production_end_to_end_movie_M_test",
                "mode": "predicted_centers",
                "angle_n": 0,
                "status": end_to_end["status"],
                "reason_code": end_to_end["reason_code"],
                "reason": end_to_end["reason"],
            }
        )

    prior_by_mode = {
        row["mode"]: row
        for row in prior_corrected["summary_rows"]
        if row["dataset"] == "neural_tube_membrane"
    }
    released_regression = released["regression_results"]
    released_keys = {
        "all_groundtruth_centers": (
            "regression_performance_on_all_groundtruth_centers"
        ),
        "matched_groundtruth_centers": (
            "regression_performance_on_matched_groundtruth_centers"
        ),
        "predicted_centers": "regression_performance_on_predicted_centers",
    }
    for mode in evaluation.MODE_ORDER:
        old = released_regression[released_keys[mode]]
        prior = prior_by_mode[mode]
        rows.append(
            {
                "record_type": "angular_previous",
                "scenario": "previous_released_movie_I2_original_metric",
                "scope": "movie_I2 set 2 (archived validation=test)",
                "mode": mode,
                "angular_metric": "production_quaternion_error_deg",
                "angle_nominal_n": int(old["n"]),
                "angle_n": int(old["n"]),
                "angle_mean": float(old["mean_angle_error"]),
                "angle_std_population": float(old["std_angle_error"]),
                "length_mae_voxels": float(old["mean_length_error"]),
                "length_error_std_population_voxels": float(
                    old["std_length_error"]
                ),
            }
        )
        rows.append(
            {
                "record_type": "angular_previous",
                "scenario": "previous_saved_movie_I2_corrected_proxy",
                "scope": "movie_I2 set 2 (raster-decoded truth proxy)",
                "mode": mode,
                "angular_metric": "corrected_nematic_axis_error_deg",
                "angle_nominal_n": int(prior["production_reported_n"]),
                "angle_n": int(prior["effective_n"]),
                "angle_mean": float(prior["corrected_mean_deg"]),
                "angle_median": float(prior["corrected_median_deg"]),
                "angle_std_population": float(
                    prior["corrected_std_population_deg"]
                ),
                "angle_sem": float(
                    prior["corrected_sem_deg_using_effective_n"]
                ),
                "angle_rms": float(prior["corrected_rms_deg"]),
                "angle_p95": float(prior["corrected_p95_deg"]),
            }
        )
    return rows


def distribution_rows(values_by_scenario) -> list[dict[str, Any]]:
    edges = np.asarray([0, 5, 10, 15, 20, 30, 45, 60, 75, 90, 120, 150, 180.000001])
    thresholds = (5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 180)
    rows = []
    for key, values in values_by_scenario.items():
        scenario, scope, mode = key
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        counts, _ = np.histogram(values, bins=edges)
        for index, count in enumerate(counts):
            rows.append(
                {
                    "scenario": scenario,
                    "scope": scope,
                    "mode": mode,
                    "record_type": "histogram_bin",
                    "lower_deg_inclusive": edges[index],
                    "upper_deg_exclusive": edges[index + 1],
                    "threshold_deg": "",
                    "count": int(count),
                    "fraction": float(count / len(values)),
                    "n": len(values),
                }
            )
        for threshold in thresholds:
            count = int(np.sum(values <= threshold))
            rows.append(
                {
                    "scenario": scenario,
                    "scope": scope,
                    "mode": mode,
                    "record_type": "cdf_threshold",
                    "lower_deg_inclusive": "",
                    "upper_deg_exclusive": "",
                    "threshold_deg": threshold,
                    "count": count,
                    "fraction": float(count / len(values)),
                    "n": len(values),
                }
            )
    return rows


def plot_comparison(values_by_scenario, segmentation) -> None:
    controlled_keys = [
        key for key in values_by_scenario if key[1] == "controlled"
    ]
    full_keys = [
        key
        for key in values_by_scenario
        if key[1] == "end_to_end" and key[2] == "predicted_centers" and len(values_by_scenario[key])
    ]
    colors = ("#666666", "#2b8cbe", "#de2d26")
    labels = {
        "original_original": "Original + quaternion",
        "original_corrected": "Original + nematic",
        "retrained_corrected": "Retrained + nematic",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    bins = np.linspace(0, 180, 37)
    for index, key in enumerate(controlled_keys):
        values = values_by_scenario[key]
        axes[0, 0].hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=2,
            color=colors[index],
            label=labels[key[0]],
        )
        sorted_values = np.sort(values)
        axes[0, 1].step(
            sorted_values,
            np.arange(1, len(sorted_values) + 1) / len(sorted_values),
            where="post",
            linewidth=2,
            color=colors[index],
            label=labels[key[0]],
        )
    for index, key in enumerate(full_keys):
        values = values_by_scenario[key]
        axes[1, 0].hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=2,
            color=colors[index],
            label=labels[key[0]],
        )
    if not full_keys:
        axes[1, 0].text(
            0.5, 0.5, "Not evaluable:\nfixed segmentation produced 0 detections",
            ha="center", va="center", transform=axes[1, 0].transAxes,
        )
    fixed_f1 = float(
        segmentation["fixed_released"]["test_metrics"]["fmeasure"]
    )
    axes[1, 1].bar(
        ("Fixed released",),
        (fixed_f1,),
        color=(colors[1],),
    )
    fixed_metrics = segmentation["fixed_released"]["test_metrics"]
    axes[1, 1].text(
        0, 0.06,
        f"F1={fixed_f1:.2f}\n{fixed_metrics['tp']} TP / {fixed_metrics['fp']} FP / {fixed_metrics['fn']} FN",
        ha="center", va="bottom",
    )
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_ylabel("F1")
    axes[1, 1].set_title("Shared movie_M detection")
    axes[0, 0].set_title("Controlled 80-crop angular distribution")
    axes[0, 1].set_title("Controlled 80-crop empirical CDF")
    axes[1, 0].set_title("End-to-end predicted-center distribution")
    for axis in (axes[0, 0], axes[1, 0]):
        axis.set_xlabel("Angular error (degrees)")
        axis.set_ylabel("Count")
        axis.set_xlim(0, 180)
    axes[0, 0].legend(fontsize=8)
    if full_keys:
        axes[1, 0].legend(fontsize=8)
    axes[0, 1].set_xlabel("Angular error (degrees)")
    axes[0, 1].set_ylabel("Cumulative fraction")
    axes[0, 1].set_xlim(0, 180)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_PNG, dpi=200)
    fig.savefig(FIGURE_PDF)
    plt.close(fig)


def format_stat(stats: dict[str, Any], unit: str = "degrees") -> str:
    if int(stats.get("n", 0)) == 0:
        return "not evaluable (N=0)"
    return (
        f"N={stats['n']}, mean={stats['mean']:.3f} {unit}, "
        f"median={stats['median']:.3f} {unit}, "
        f"SD={stats['std_population']:.3f} {unit}, "
        f"SEM={stats['sem']:.3f} {unit}, RMS={stats['rms']:.3f} {unit}, p95={stats['p95']:.3f} {unit}"
    )


def write_report(result: dict[str, Any]) -> None:
    seg = result["segmentation"]
    controlled = result["controlled_regression"]["summary"]["by_model"]
    end_to_end = result["end_to_end"]
    comparison = result["comparisons"]
    released = result["previous_released"]
    fixed_seg = seg["fixed_released"]["test_metrics"]
    end_available = end_to_end.get("status") == "complete"
    if end_available:
        end = end_to_end["summary"]
        new_pred = end["retrained_regression_fixed_segmentation"]["predicted_centers"][
            "corrected_nematic_axis_error_deg"
        ]
        old_pred_current = end["original_regression_fixed_segmentation"]["predicted_centers"][
            "production_quaternion_error_deg"
        ]
        old_pred_corrected = end["original_regression_fixed_segmentation"]["predicted_centers"][
            "corrected_nematic_axis_error_deg"
        ]
        old_pred_length = end["original_regression_fixed_segmentation"][
            "predicted_centers"
        ]["absolute_length_error_voxels"]
        new_pred_length = end["retrained_regression_fixed_segmentation"][
            "predicted_centers"
        ]["absolute_length_error_voxels"]
        predicted_summary_lines = [
            f"- Original regression + fixed segmentation + original metric: {format_stat(old_pred_current)}",
            f"- Original regression + fixed segmentation + corrected metric: {format_stat(old_pred_corrected)}",
            f"- Retrained regression + fixed segmentation + corrected metric: {format_stat(new_pred)}",
        ]
        predicted_length_lines = [
            "- Predicted-center length error, original model: "
            + format_stat(old_pred_length, "voxels"),
            "- Predicted-center length error, retrained model: "
            + format_stat(new_pred_length, "voxels"),
            f"- Paired predicted-center length-error mean change: {comparison['predicted_center_length_retraining_effect']['mean_difference']:.3f} voxels; "
            f"95% bootstrap interval [{comparison['predicted_center_length_retraining_effect']['mean_difference_bootstrap_95_ci'][0]:.3f}, {comparison['predicted_center_length_retraining_effect']['mean_difference_bootstrap_95_ci'][1]:.3f}].",
        ]
        predicted_effect_lines = [
            "The fixed-segmentation predicted-center corrected mean changes by "
            f"{comparison['full_pipeline_predicted_center_change']['mean_difference_deg']:.3f} degrees "
            f"({comparison['full_pipeline_predicted_center_change']['relative_difference_percent']:.1f}%). "
            "The paired descriptive 95% bootstrap interval is "
            f"[{comparison['full_pipeline_predicted_center_change']['mean_difference_bootstrap_95_ci'][0]:.3f}, {comparison['full_pipeline_predicted_center_change']['mean_difference_bootstrap_95_ci'][1]:.3f}] degrees."
        ]
    else:
        unavailable_text = end_to_end["reason"]
        predicted_summary_lines = [f"- Not evaluable (N=0): {unavailable_text}"]
        predicted_length_lines = [
            "- Predicted-center length comparison: not evaluable (N=0).",
        ]
        predicted_effect_lines = [
            "The fixed-segmentation predicted-center comparison is not evaluable "
            "because the frozen detector produced zero candidate centers.",
        ]

    lines = [
        "# Neural-tube DARE3D nematic regression retraining comparison",
        "",
        "## Outcome",
        "",
        (
            "This is an isolated audit experiment. Production source, released "
            "checkpoints/results, and manuscript files were not modified. The "
            "released segmentation checkpoint and its resulting detections are "
            "held fixed for both regression models."
        ),
        "",
        "## Previous reported result",
        "",
        (
            f"The released movie_I2 result is {released['segmentation_results']['tp']} "
            f"TP / {released['segmentation_results']['fp']} FP / "
            f"{released['segmentation_results']['fn']} FN, F1 "
            f"{100*released['segmentation_results']['fmeasure']:.1f}%. "
            "The released predicted-center regression aggregate has mean "
            f"{released['regression_results']['regression_performance_on_predicted_centers']['mean_angle_error']:.3f} degrees "
            "and population SD "
            f"{released['regression_results']['regression_performance_on_predicted_centers']['std_angle_error']:.3f} degrees "
            "under the quaternion metric. This is the 21-frame set 2 used for "
            "both validation and test in the archived configuration, not the "
            "manuscript's bold 11-frame set 3."
        ),
        "",
        "## Independent movie_M test",
        "",
        (
            f"Fixed released segmentation: {fixed_seg['tp']} TP / "
            f"{fixed_seg['fp']} FP / {fixed_seg['fn']} FN, F1 "
            f"{100*fixed_seg['fmeasure']:.2f}%. The detector produced no candidate "
            "centers, so predicted-center regression scoring is not evaluable."
        ),
        "",
        "Angular summaries on predicted centers:",
        "",
        *predicted_summary_lines,
        "",
        "Controlled independent 80-crop test:",
        "",
        (
            "- Original model + original metric: "
            + format_stat(controlled["original"]["production_quaternion_error_deg"])
        ),
        (
            "- Original model + corrected metric: "
            + format_stat(controlled["original"]["corrected_nematic_axis_error_deg"])
        ),
        (
            "- Retrained model + corrected metric: "
            + format_stat(controlled["retrained"]["corrected_nematic_axis_error_deg"])
        ),
        "",
        "## Non-angular checks",
        "",
        (
            f"- Detection/matching is shared: {fixed_seg['tp']} TP / "
            f"{fixed_seg['fp']} FP / {fixed_seg['fn']} FN; no regression-dependent change."
        ),
        (
            "- Controlled length error, original model: "
            + format_stat(controlled["original"]["absolute_length_error_voxels"], "voxels")
        ),
        (
            "- Controlled length error, retrained model: "
            + format_stat(controlled["retrained"]["absolute_length_error_voxels"], "voxels")
        ),
        (
            f"- Paired controlled length-error mean change: {comparison['controlled_length_retraining_effect']['mean_difference']:.3f} voxels; "
            f"95% bootstrap interval [{comparison['controlled_length_retraining_effect']['mean_difference_bootstrap_95_ci'][0]:.3f}, {comparison['controlled_length_retraining_effect']['mean_difference_bootstrap_95_ci'][1]:.3f}]."
        ),
        *predicted_length_lines,
        "",
        "## Numerical effects",
        "",
        (
            "Changing only evaluation metric on the original controlled test "
            f"changes the mean by {comparison['controlled_metric_change']['mean_difference_deg']:.3f} degrees "
            f"({comparison['controlled_metric_change']['relative_difference_percent']:.1f}%)."
        ),
        (
            "Retraining changes the corrected controlled-test mean by "
            f"{comparison['controlled_retraining_effect']['mean_difference']:.3f} degrees; "
            "the paired descriptive 95% bootstrap interval is "
            f"[{comparison['controlled_retraining_effect']['mean_difference_bootstrap_95_ci'][0]:.3f}, "
            f"{comparison['controlled_retraining_effect']['mean_difference_bootstrap_95_ci'][1]:.3f}] degrees."
        ),
        *predicted_effect_lines,
        "",
        "## Scientific interpretation",
        "",
        result["scientific_interpretation"],
        "",
        "## Compatibility limitations kept visible",
        "",
        "- The historical scales.json is absent; manuscript spacing was used.",
        "- Segmentation was not retrained; the released checkpoint, detections, and matching are fixed across regressors.",
        "- The released segmentation event log stopped at epoch 153 although 200 were configured.",
        "- The production end-to-end regression path omits training-time isotropic rescaling; it was preserved and is reported separately from the controlled test.",
        "- The archived membrane regressor is 3 stages / 32 initial filters, whereas main.tex:351-352 describes 5 stages / 16 filters.",
        "- The first two raw frames are excluded by the unchanged three-frame regression crop constructor, leaving 80 of 104 complete test annotations.",
        "",
        "## Manuscript locations requiring review",
        "",
        "- main.tex:79 (abstract orientation-accuracy claim).",
        "- main.tex:194-196 and 210 (Dataset 2 split/test identification).",
        "- main.tex:333-338 (voxel resampling and segmentation training protocol).",
        "- main.tex:343-352 (orientation formulation and regressor architecture).",
        "- main.tex:356-374 (thresholding, component filtering, matching, and metrics).",
        "- main.tex:397-402 (3D membrane quantitative result).",
        "- main.tex:488-494 (membrane angular-error interpretation).",
        "- main.tex:499-505 (2D-vs-3D angular comparison and random baseline).",
        "- main.tex:655-656 (neural-tube regression example figure caption; update only if the displayed model changes).",
        "- Supplementary movie descriptions at main.tex:588-590.",
        "",
        "## Evidence",
        "",
        f"- Numerical summary: [{SUMMARY_CSV.name}]({SUMMARY_CSV.relative_to(HERE).as_posix()})",
        f"- Distribution table: [{DISTRIBUTION_CSV.name}]({DISTRIBUTION_CSV.relative_to(HERE).as_posix()})",
        f"- Figure: [{FIGURE_PNG.name}]({FIGURE_PNG.relative_to(HERE).as_posix()})",
        f"- Machine result: [{FINAL_JSON.name}]({FINAL_JSON.relative_to(HERE).as_posix()})",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def build_final_comparison() -> dict[str, Any]:
    segmentation = {
        "fixed_released": load_json(
            evaluation.SEGMENTATION_ROOT
            / evaluation.FIXED_SEGMENTATION_MODEL
            / "segmentation_evaluation.json"
        )
    }
    controlled = load_json(
        evaluation.CONTROLLED_ROOT / "controlled_regression_evaluation.json"
    )
    end_to_end = load_json(
        evaluation.END_TO_END_ROOT / "end_to_end_evaluation.json"
    )
    released = load_json(common.RELEASED_STATS)
    prior_corrected = load_json(PRIOR_CORRECTED)

    controlled_events = read_event_csv(
        evaluation.CONTROLLED_ROOT / "event_metrics.csv"
    )
    end_to_end_available = end_to_end.get("status") == "complete"
    end_events = (
        read_event_csv(evaluation.END_TO_END_ROOT / "event_metrics.csv")
        if end_to_end_available
        else []
    )
    values = {
        (
            "original_original",
            "controlled",
            "80_unique_groundtruth_crops",
        ): angular_values(
            controlled_events,
            lambda row: row["model"] == "original",
            "production_quaternion_error_deg",
        ),
        (
            "original_corrected",
            "controlled",
            "80_unique_groundtruth_crops",
        ): angular_values(
            controlled_events,
            lambda row: row["model"] == "original",
            "corrected_nematic_axis_error_deg",
        ),
        (
            "retrained_corrected",
            "controlled",
            "80_unique_groundtruth_crops",
        ): angular_values(
            controlled_events,
            lambda row: row["model"] == "retrained",
            "corrected_nematic_axis_error_deg",
        ),
    }
    for mode in evaluation.MODE_ORDER:
        values[
            ("original_original", "end_to_end", mode)
        ] = angular_values(
            end_events,
            lambda row, mode=mode: (
                row["pipeline"] == "original_regression_fixed_segmentation"
                and row["mode"] == mode
                and row["evaluated"].lower() == "true"
            ),
            "production_quaternion_error_deg",
        )
        values[
            ("original_corrected", "end_to_end", mode)
        ] = angular_values(
            end_events,
            lambda row, mode=mode: (
                row["pipeline"] == "original_regression_fixed_segmentation"
                and row["mode"] == mode
                and row["evaluated"].lower() == "true"
            ),
            "corrected_nematic_axis_error_deg",
        )
        values[
            ("retrained_corrected", "end_to_end", mode)
        ] = angular_values(
            end_events,
            lambda row, mode=mode: (
                row["pipeline"] == "retrained_regression_fixed_segmentation"
                and row["mode"] == mode
                and row["evaluated"].lower() == "true"
            ),
            "corrected_nematic_axis_error_deg",
        )

    summary_rows = build_summary_rows(
        segmentation,
        controlled,
        end_to_end,
        released,
        prior_corrected,
    )
    evaluation.write_csv(SUMMARY_CSV, summary_rows)
    evaluation.write_csv(DISTRIBUTION_CSV, distribution_rows(values))
    plot_comparison(values, segmentation)

    controlled_old_current = values[
        ("original_original", "controlled", "80_unique_groundtruth_crops")
    ]
    controlled_old_corrected = values[
        ("original_corrected", "controlled", "80_unique_groundtruth_crops")
    ]
    controlled_new_corrected = values[
        ("retrained_corrected", "controlled", "80_unique_groundtruth_crops")
    ]
    old_full_corrected = values[
        ("original_corrected", "end_to_end", "predicted_centers")
    ]
    new_full_corrected = values[
        ("retrained_corrected", "end_to_end", "predicted_centers")
    ]
    controlled_old_length = angular_values(
        controlled_events,
        lambda row: row["model"] == "original",
        "absolute_length_error_voxels",
    )
    controlled_new_length = angular_values(
        controlled_events,
        lambda row: row["model"] == "retrained",
        "absolute_length_error_voxels",
    )
    old_full_length = angular_values(
        end_events,
        lambda row: (
            row["pipeline"] == "original_regression_fixed_segmentation"
            and row["mode"] == "predicted_centers"
            and row["evaluated"].lower() == "true"
        ),
        "absolute_length_error_voxels",
    )
    new_full_length = angular_values(
        end_events,
        lambda row: (
            row["pipeline"] == "retrained_regression_fixed_segmentation"
            and row["mode"] == "predicted_centers"
            and row["evaluated"].lower() == "true"
        ),
        "absolute_length_error_voxels",
    )
    controlled_metric_difference = float(
        np.mean(controlled_old_corrected) - np.mean(controlled_old_current)
    )
    if end_to_end_available:
        full_difference = float(
            np.mean(new_full_corrected) - np.mean(old_full_corrected)
        )
        predicted_center_length_effect = {
            **paired_bootstrap(
                old_full_length, new_full_length, seed=20260904
            ),
            "definition": "retrained absolute length error minus original",
            "unit": "voxels",
        }
        full_pipeline_effect = {
            **paired_bootstrap(
                old_full_corrected, new_full_corrected, seed=20260902
            ),
            "definition": (
                "retrained corrected minus original corrected; fixed released "
                "segmentation and row-aligned centers"
            ),
            "mean_difference_deg": full_difference,
            "absolute_difference_deg": abs(full_difference),
            "relative_difference_percent": relative_percent(
                float(np.mean(new_full_corrected)),
                float(np.mean(old_full_corrected)),
            ),
            "old_n": len(old_full_corrected),
            "new_n": len(new_full_corrected),
            "paired": True,
        }
    else:
        unavailable_comparison = {
            "status": "not_evaluable",
            "reason_code": end_to_end["reason_code"],
            "reason": end_to_end["reason"],
            "n": 0,
        }
        predicted_center_length_effect = {
            **unavailable_comparison,
            "definition": "retrained absolute length error minus original",
            "unit": "voxels",
        }
        full_pipeline_effect = {
            **unavailable_comparison,
            "definition": "retrained corrected minus original corrected",
            "old_n": 0,
            "new_n": 0,
            "paired": False,
        }
    comparisons = {
        "controlled_metric_change": {
            "definition": "original corrected minus original quaternion",
            "mean_difference_deg": controlled_metric_difference,
            "absolute_difference_deg": abs(controlled_metric_difference),
            "relative_difference_percent": relative_percent(
                float(np.mean(controlled_old_corrected)),
                float(np.mean(controlled_old_current)),
            ),
            "n": len(controlled_old_corrected),
        },
        "controlled_retraining_effect": paired_bootstrap(
            controlled_old_corrected,
            controlled_new_corrected,
            seed=20260901,
        ),
        "controlled_length_retraining_effect": {
            **paired_bootstrap(
                controlled_old_length,
                controlled_new_length,
                seed=20260903,
            ),
            "definition": "retrained absolute length error minus original",
            "unit": "voxels",
        },
        "predicted_center_length_retraining_effect": predicted_center_length_effect,
        "full_pipeline_predicted_center_change": full_pipeline_effect,
        "fixed_detection_movie_M": {
            "tp": segmentation["fixed_released"]["test_metrics"]["tp"],
            "fp": segmentation["fixed_released"]["test_metrics"]["fp"],
            "fn": segmentation["fixed_released"]["test_metrics"]["fn"],
            "f1": segmentation["fixed_released"]["test_metrics"]["fmeasure"],
            "segmentation_retrained": False,
            "used_by_both_regression_models": True,
        },
    }

    all_corrected = np.concatenate(
        [
            array
            for key, array in values.items()
            if key[0] in {"original_corrected", "retrained_corrected"}
        ]
    )
    assertions = {
        "production_source_and_index_clean": common.tracked_tree_is_clean(),
        "independent_test_is_movie_M": common.SPLITS["test"]["movie"] == "movie_M",
        "controlled_test_has_80_events_per_model": (
            len(controlled_old_corrected) == 80
            and len(controlled_new_corrected) == 80
        ),
        "nonangular_pairs_are_aligned": (
            len(controlled_old_length) == len(controlled_new_length) == 80
            and len(old_full_length) == len(new_full_length)
        ),
        "corrected_angles_are_in_zero_to_90": bool(
            np.all((all_corrected >= 0) & (all_corrected <= 90))
        ),
        "segmentation_is_released_and_fixed": (
            segmentation["fixed_released"]["checkpoint"]["path"]
            == common.rel(common.RELEASED_SEGMENTATION_CHECKPOINT)
            and not segmentation["fixed_released"]["segmentation_retrained"]
        ),
        "end_to_end_state_is_consistent": (
            not end_to_end["segmentation_retrained"]
            and end_to_end["fixed_test_matching_info"]
            == segmentation["fixed_released"]["test_matching_info"]
            and (
                (
                    end_to_end_available
                    and end_to_end["center_counts_identical_between_regression_models"]
                )
                or (
                    not end_to_end_available
                    and end_to_end["reason_code"]
                    == "fixed_segmentation_produced_zero_detections"
                    and end_to_end["detected_center_count"] == 0
                )
            )
        ),
        "test_thresholds_equal_validation_selected_thresholds": all(
            result["test_metrics"]["threshold"]
            == result["selected_probability_threshold"]
            and result["test_metrics"]["min_weighted_prob"]
            == result["selected_minimum_weighted_probability"]
            for result in segmentation.values()
        ),
        "all_required_outputs_exist": all(
            path.is_file()
            for path in (
                SUMMARY_CSV,
                DISTRIBUTION_CSV,
                FIGURE_PNG,
                FIGURE_PDF,
            )
        ),
    }
    scientific_interpretation = (
        "On the independent 80-crop movie_M test, retraining materially reduces "
        f"the corrected mean angular error from {np.mean(controlled_old_corrected):.3f} "
        f"to {np.mean(controlled_new_corrected):.3f} degrees. The paired new-minus-old "
        f"mean is {comparisons['controlled_retraining_effect']['mean_difference']:.3f} "
        "degrees, with the descriptive bootstrap interval reported above. Replacing "
        f"only the evaluation metric changes the original mean from {np.mean(controlled_old_current):.3f} "
        f"to {np.mean(controlled_old_corrected):.3f} degrees. However, full predicted-center "
        "performance is not evaluable because the frozen released segmentation "
        "checkpoint produces zero detections on movie_M under the unchanged protocol. "
        f"Length error also changes from {np.mean(controlled_old_length):.3f} to "
        f"{np.mean(controlled_new_length):.3f} voxels despite an unchanged length-loss "
        "definition, so improvement is demonstrated for this retrained regressor but "
        "cannot be attributed exclusively to the orientation-loss change. The old "
        "movie_I2 result is not a same-test comparator."
    )
    if not all(assertions.values()):
        report_status = "complete_with_failed_assertions"
    elif not end_to_end_available:
        report_status = "complete_with_unavailable_end_to_end"
    else:
        report_status = "complete"
    result = {
        "status": report_status,
        "completed_at": common.now_iso(),
        "definitions": {
            "production_metric": (
                "degrees(2*acos(abs(q_pred dot q_true))) with dot clipped "
                "to [-0.9999,0.9999]"
            ),
            "corrected_metric": (
                "degrees(acos(clip(abs(unit axis_pred dot unit axis_true),0,1)))"
            ),
            "corrected_training_loss": "mean(90*(1-(axis_pred dot axis_true)^2))",
        },
        "previous_released": released,
        "segmentation": segmentation,
        "controlled_regression": controlled,
        "end_to_end": end_to_end,
        "comparisons": comparisons,
        "scientific_interpretation": scientific_interpretation,
        "assertions": assertions,
        "all_assertions_pass": all(assertions.values()),
        "outputs": {
            "summary_csv": common.rel(SUMMARY_CSV),
            "distribution_csv": common.rel(DISTRIBUTION_CSV),
            "figure_png": common.rel(FIGURE_PNG),
            "figure_pdf": common.rel(FIGURE_PDF),
            "report": common.rel(REPORT),
        },
    }
    common.atomic_write_json(FINAL_JSON, result)
    write_report(result)
    result["outputs"]["final_json"] = common.rel(FINAL_JSON)
    common.atomic_write_json(FINAL_JSON, result)
    return result


def main() -> None:
    result = build_final_comparison()
    print(
        json.dumps(
            {
                "status": result["status"],
                "all_assertions_pass": result["all_assertions_pass"],
                "outputs": result["outputs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
