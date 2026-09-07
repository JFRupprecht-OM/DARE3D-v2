"""Build final ledgers and checks from frozen DARE3D audit evidence.

This audit-only script does not run models, retrain, tune parameters, or modify
production source, checkpoints, downloaded data, or released results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EVIDENCE = HERE / "evidence"
REPORT = HERE / "DARE3D_REPRODUCIBILITY_AUDIT_REPORT.md"
MANIFEST = EVIDENCE / "audit_evidence_manifest_sha256.csv"
RETRAINING_RUN = HERE / "nematic_retraining" / "seed_12345"
RETRAINING_EVALUATION = RETRAINING_RUN / "evaluation.json"
RETRAINING_PREFLIGHT = RETRAINING_RUN / "preflight.json"
SOURCE_COMMIT = "fe2b14d732359f2bdaf8b197574ad818899ce123"


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def load_json(name: str) -> Any:
    path = EVIDENCE / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(name: str, value: Any) -> None:
    (EVIDENCE / name).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_csv(name: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty ledger {name}")
    with (EVIDENCE / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


CLAIM_FIELDS = [
    "claim_id", "claim_type", "case", "manuscript_location",
    "manuscript_claim", "expected_value", "unit",
    "dataset_and_frame_scope", "denominator", "split_role",
    "preprocessing", "checkpoint", "prediction_source", "matching_rule",
    "metric_formula", "aggregation", "uncertainty_definition",
    "reproduced_value", "numerical_agreement", "methodological_validity",
    "classification", "evidence", "blocking_issue_or_note",
]


def claim(**values: Any) -> dict[str, Any]:
    row = {field: "" for field in CLAIM_FIELDS}
    row.update(values)
    required = (
        "claim_id", "claim_type", "manuscript_location", "classification", "evidence"
    )
    missing = [field for field in required if not row[field]]
    if missing:
        raise ValueError(f"Claim {row.get('claim_id')!r} lacks {missing}")
    return row


def build_claims(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    arithmetic = inputs["arithmetic"]
    regression = inputs["regression"]
    reg_ckpt = inputs["reg_ckpt"]
    annotations = inputs["annotations"]
    movie2 = next(
        item for item in annotations
        if item["path"].endswith("trainingset/movie2/label/movie2.tif")
    )
    rows: list[dict[str, Any]] = []

    nuclei_data = [
        ("SHAPE", "image/stack size", "(362,305,180)", "pixels", "x,y,z", "x,y,z", "exact"),
        ("FRAMES", "number of frames", "10", "frames", "10", "10", "exact"),
        ("ANNOTATIONS", "annotated divisions", "226", "events", "226", str(movie2["events_all_frames"]), "exact"),
        ("LENGTH-MEAN", "mean division length", "15.3", "raw voxels", "226 daughter pairs", f"{movie2['length_mean_voxels']:.9f}", "display precision"),
        ("LENGTH-SEM", "division-length uncertainty", "0.2", "SEM, raw voxels", "226 daughter pairs", f"{movie2['length_sem_voxels']:.9f}", "display precision"),
    ]
    for suffix, label, expected, unit, denominator, reproduced, agreement in nuclei_data:
        rows.append(claim(
            claim_id=f"DATA-NUC-{suffix}", claim_type="dataset",
            case="Gastruloid nuclei", manuscript_location="main.tex:204,210",
            manuscript_claim=f"Dataset 3 set 1 {label}", expected_value=expected,
            unit=unit,
            dataset_and_frame_scope="Dataset 3 set 1 / movie2, all 10 frames",
            denominator=denominator,
            split_role="bold manuscript test set; config validation=test",
            reproduced_value=reproduced, numerical_agreement=agreement,
            methodological_validity=(
                "valid artifact statistic; split invalid for untouched-test inference"
            ),
            classification="fully reproducible",
            evidence=(
                "evidence/annotation_summary.json; evidence/annotation_events.csv; "
                "evidence/tiff_inventory.csv"
            ),
            blocking_issue_or_note=(
                "Directly recomputable. Detection uses 145 merged eligible objects, "
                "not all 226 annotations."
            ),
        ))

    membrane_data = [
        ("SHAPE", "image/stack size", "(1024,1024,10)", "pixels"),
        ("FRAMES", "number of frames", "11", "frames"),
        ("ANNOTATIONS", "annotated divisions", "104", "events"),
        ("LENGTH-MEAN", "mean division length", "31.3", "raw pixels"),
        ("LENGTH-SEM", "division-length uncertainty", "0.3", "SEM, raw pixels"),
    ]
    for suffix, label, expected, unit in membrane_data:
        rows.append(claim(
            claim_id=f"DATA-MEM-{suffix}", claim_type="dataset",
            case="Neural-tube membrane", manuscript_location="main.tex:196,210",
            manuscript_claim=f"Dataset 2 set 3 {label}", expected_value=expected,
            unit=unit, dataset_and_frame_scope="Dataset 2 set 3, 11 frames",
            denominator="104 annotated divisions" if "LENGTH" in suffix else "",
            split_role="bold manuscript test set",
            reproduced_value=(
                "not independently computable; raw set-3 images and annotations absent"
            ),
            numerical_agreement="not independently checkable",
            methodological_validity="not assessable from released raw data",
            classification="currently not reproducible",
            evidence=(
                "evidence/artifact_audit_summary.json; "
                "evidence/regression_artifact_analysis.json; evidence/tiff_inventory.csv"
            ),
            blocking_issue_or_note=(
                "Release has derived movie_I2 (21 frames, consistent with set 2), "
                "not raw Dataset 2 set 3."
            ),
        ))

    detection = {
        "NUC": {
            "case": "Gastruloid nuclei", "line": "main.tex:395",
            "expected": {
                "TP": 124, "FP": 1, "FN": 21, "F1": "91.8",
                "CI": "87.8--94.6",
            },
            "actual": {
                "TP": 124, "FP": 1, "FN": 21,
                "F1": f"{100*arithmetic['nuclei_gastruloid']['f1']:.10f}",
                "CI": "87.7500--94.6638",
            },
            "denominator": "145 current-evaluator GT components; 125 predictions",
            "checkpoint": "epoch_067 strongly identified; last.ckpt differs",
            "source": "released probability and current-code epoch_067 inference",
            "evidence": (
                "evidence/nuclei_saved_probability_replay.json; "
                "evidence/nuclei_checkpoint_compatibility_summary.json; "
                "evidence/arithmetic_and_uncertainty.json"
            ),
            "note": (
                "Exact released-evaluator/checkpoint aggregate agreement. Stated "
                "10-voxel matcher gives 121 TP, 4 FP, 24 FN, F1 89.63%; validation=test."
            ),
        },
        "MEM": {
            "case": "Neural-tube membrane", "line": "main.tex:402",
            "expected": {
                "TP": 114, "FP": 8, "FN": 8, "F1": "93.4",
                "CI": "89.5--96.0",
            },
            "actual": {
                "TP": 114, "FP": 8, "FN": 8,
                "F1": f"{100*arithmetic['membrane_neural_tube']['f1']:.10f}",
                "CI": "89.4118--96.0075",
            },
            "denominator": "122 released GT components; 122 retained predictions",
            "checkpoint": "not testable from raw input",
            "source": "released stats, probability, and match renderings",
            "evidence": (
                "evidence/membrane_fp_investigation.json; "
                "evidence/saved_detection_reconstruction.json; "
                "evidence/arithmetic_and_uncertainty.json"
            ),
            "note": (
                "Arithmetic/components trace exactly, including hidden eighth FP. "
                "Raw neural labels are absent and released movie conflicts with set 3."
            ),
        },
    }
    formulas = {
        "TP": "one-to-one matched true events",
        "FP": "unmatched predictions",
        "FN": "unmatched ground truth",
        "F1": "2TP/(2TP+FP+FN)",
        "CI": "logit multinomial delta interval from TP/FP/FN",
    }
    units = {
        "TP": "events", "FP": "events", "FN": "events",
        "F1": "percent", "CI": "percent",
    }
    for code, item in detection.items():
        for metric in ("TP", "FP", "FN", "F1", "CI"):
            rows.append(claim(
                claim_id=f"DET-{code}-{metric}", claim_type="detection",
                case=item["case"], manuscript_location=item["line"],
                manuscript_claim=f"{metric} detection result",
                expected_value=str(item["expected"][metric]), unit=units[metric],
                dataset_and_frame_scope="released evaluation movie",
                denominator=item["denominator"], split_role="config validation=test",
                preprocessing="released protocol; original scales.json absent",
                checkpoint=item["checkpoint"], prediction_source=item["source"],
                matching_rule=(
                    "manuscript: centers within 10 voxels and +/-1 frame; released: "
                    "temporal dilation + 4D components + IoU/greedy"
                ),
                metric_formula=formulas[metric],
                aggregation="single-movie component counts",
                uncertainty_definition=(
                    "multinomial logit-delta CI" if metric == "CI" else ""
                ),
                reproduced_value=str(item["actual"][metric]),
                numerical_agreement="exact/display precision under released arithmetic",
                methodological_validity=(
                    "not an independent test and not equivalent to stated matcher"
                ),
                classification="partially reproducible", evidence=item["evidence"],
                blocking_issue_or_note=item["note"],
            ))

    rows.extend([
        claim(
            claim_id="REG-NUC-ORIENTATION", claim_type="regression",
            case="Gastruloid nuclei", manuscript_location="main.tex:395",
            manuscript_claim="orientation-angle error 28 degrees +/- 2 degrees",
            expected_value="28 +/- 2", unit="degrees",
            dataset_and_frame_scope="released nuclei evaluation",
            denominator="reported n=145 all-GT; effective n=140",
            split_role="config validation=test",
            preprocessing="current regression evaluation skips spatial resampling",
            checkpoint="epoch_098 strongly identified",
            prediction_source=(
                "all_groundtruth_centers headline; predicted_centers is end-to-end"
            ),
            matching_rule="all-GT mode does not require detector match",
            metric_formula=(
                "full-quaternion 2*acos(abs(q_true dot q_pred)); not nematic axis error"
            ),
            aggregation="population SD used as central value",
            uncertainty_definition="SD/sqrt(reported n)",
            reproduced_value=(
                f"released mean "
                f"{regression['manuscript_28_degree_claim_check']['gastruloid_released_all_gt_mean_angle_error_deg']:.6f}; "
                f"released SD "
                f"{regression['manuscript_28_degree_claim_check']['gastruloid_released_all_gt_std_angle_error_deg']:.6f}; "
                f"epoch098 SD "
                f"{reg_ckpt['best_headline_trace']['all_groundtruth_std_angle_error_deg']:.6f}; "
                f"SD/sqrt(145) "
                f"{reg_ckpt['best_headline_trace']['sd_over_sqrt_nominal_n_deg']:.6f}"
            ),
            numerical_agreement="display-precision checkpoint agreement",
            methodological_validity=(
                "invalid wording/statistic and non-nematic metric; not end-to-end"
            ),
            classification="partially reproducible",
            evidence=(
                "evidence/regression_artifact_analysis.json; "
                "evidence/nuclei_regression_checkpoint_compatibility_summary.json; "
                "evidence/regression_event_metrics.csv"
            ),
            blocking_issue_or_note=(
                "28 is SD, not mean (25.306); predicted-center mean is 37.47. "
                "Historical raw arrays/scales absent."
            ),
        ),
        claim(
            claim_id="REG-MEM-ORIENTATION", claim_type="regression",
            case="Neural-tube membrane", manuscript_location="main.tex:402",
            manuscript_claim="orientation-angle error 28 degrees +/- 2 degrees",
            expected_value="28 +/- 2", unit="degrees",
            dataset_and_frame_scope="released membrane evaluation",
            denominator="n=122 all-GT centers",
            split_role="config validation=test; released movie conflicts with set",
            preprocessing="raw neural input/annotation unavailable; raster only",
            checkpoint="checkpoint-to-result identity untested",
            prediction_source=(
                "all_groundtruth_centers headline; predicted_centers is end-to-end"
            ),
            matching_rule="released matching IDs only",
            metric_formula=(
                "released full-quaternion; raster permits approximate nematic error"
            ),
            aggregation="population SD used as central value",
            uncertainty_definition="SD/sqrt(n)",
            reproduced_value=(
                f"released mean "
                f"{regression['manuscript_28_degree_claim_check']['neural_released_all_gt_mean_angle_error_deg']:.6f}; "
                f"released SD "
                f"{regression['manuscript_28_degree_claim_check']['neural_released_all_gt_std_angle_error_deg']:.6f}; "
                f"SD/sqrt(122) "
                f"{regression['manuscript_28_degree_claim_check']['neural_released_sd_over_sqrt_n_deg']:.6f}"
            ),
            numerical_agreement="saved-statistic trace only",
            methodological_validity=(
                "invalid wording/statistic and non-nematic metric; not end-to-end"
            ),
            classification="partially reproducible",
            evidence=(
                "evidence/regression_artifact_analysis.json; "
                "evidence/regression_event_metrics.csv; "
                "evidence/regression_raster_decoder_validation.csv"
            ),
            blocking_issue_or_note=(
                "28 is SD, not mean (49.566). Raw annotations absent; "
                "37/350 raster rows need next-frame fallback."
            ),
        ),
        claim(
            claim_id="DERIVED-F1-ABOVE-90", claim_type="derived",
            case="both DARE3D cases", manuscript_location="main.tex:79",
            manuscript_claim="DARE3D F1 scores exceed 90 percent",
            expected_value=">90", unit="percent",
            dataset_and_frame_scope="two released evaluation movies",
            denominator="released component counts",
            split_role="config validation=test",
            prediction_source="released stats; nuclei checkpoint replay",
            matching_rule="released evaluator",
            metric_formula="2TP/(2TP+FP+FN)", aggregation="case-wise",
            reproduced_value="91.8519 nuclei; 93.4426 membrane",
            numerical_agreement="yes for released arithmetic",
            methodological_validity=(
                "not untouched-test evidence; stated matcher not reproduced"
            ),
            classification="partially reproducible",
            evidence=(
                "evidence/arithmetic_and_uncertainty.json; "
                "evidence/nuclei_saved_probability_replay.json; "
                "evidence/membrane_fp_investigation.json"
            ),
            blocking_issue_or_note=(
                "Numerically true for archived counts, but split and object "
                "definitions prevent full status."
            ),
        ),
    ])

    derived = [
        ("BASELINE-2D", "random nematic 2D RMS error", "52.0", f"{arithmetic['uncertainty']['random_nematic_2d_rms_deg']:.10f}", "main.tex:501-505"),
        ("BASELINE-3D", "random nematic 3D RMS error", "61.2", f"{arithmetic['uncertainty']['random_nematic_3d_rms_deg']:.10f}", "main.tex:501-505"),
        ("UNCERTAINTY-NUC", "manual-annotation uncertainty", "approximately 20", f"{arithmetic['uncertainty']['nuclei_isotropic_length15_sigma3']['mean_deg']:.10f}", "main.tex:481"),
        ("UNCERTAINTY-MEM", "manual-annotation uncertainty", "approximately 20", f"{arithmetic['uncertainty']['membrane_scale_0.2_0.2_1_length30_sigma2_inplane']['mean_deg']:.10f}", "main.tex:493"),
    ]
    for suffix, label, expected, reproduced, location in derived:
        rows.append(claim(
            claim_id=f"DERIVED-{suffix}", claim_type="derived",
            case="analytic/simulation", manuscript_location=location,
            manuscript_claim=label, expected_value=expected, unit="degrees",
            dataset_and_frame_scope="stated mathematical assumptions",
            denominator=(
                "2,000,000 fixed-seed samples"
                if "UNCERTAINTY" in suffix else "analytic integral"
            ),
            metric_formula="nematic angle in [0,90 degrees]",
            aggregation="RMS or Monte Carlo mean as stated",
            uncertainty_definition=(
                "fixed seed 20260828" if "UNCERTAINTY" in suffix else "closed form"
            ),
            reproduced_value=reproduced, numerical_agreement="display precision",
            methodological_validity="valid under explicitly stated assumptions",
            classification="fully reproducible",
            evidence="evidence/arithmetic_and_uncertainty.json",
            blocking_issue_or_note=(
                "Verifies the arithmetic/assumption, not empirical model performance."
            ),
        ))

    protocol = [
        ("PROTOCOL-N-FRAMES", "main.tex:334,337", "N=3 consecutive input frames", "3 channels in all four saved configs", "fully reproducible", "valid at configuration level", "Four saved configs."),
        ("PROTOCOL-SEG-CROP", "main.tex:334", "128^3 segmentation crops", "crop_size=128 in both segmentation configs", "fully reproducible", "valid at configuration level", "Four saved configs."),
        ("PROTOCOL-REG-CROP", "main.tex:346", "32^3 regression crops", "crop_size=32 in both regression configs", "fully reproducible", "valid at configuration level", "Four saved configs."),
        ("PROTOCOL-RADIUS", "main.tex:330-331", "target radius r=8 voxels", "cell_radius=8 in both segmentation configs", "fully reproducible", "valid at configuration level", "Four saved configs."),
        ("PROTOCOL-ISOTROPY", "main.tex:334", "resample to isotropic 1 um voxels", "target_scale=1.0, but scales.json absent; regression evaluation skips resampling", "partially reproducible", "coordinate provenance incomplete", "Missing scales materially changes nuclei F1."),
        ("PROTOCOL-PREPROCESS", "main.tex:334,477", "histogram equalization and local contrast enhancement before modeling", "configs use min-max normalization; histogram/contrast are augmentations", "currently not reproducible", "implementation/provenance conflicts with wording", "Historical preprocessing products/source absent."),
        ("PROTOCOL-BALANCE", "main.tex:334", "50% positive and 50% hard-negative crops", "sampling implementation exists; historical crop stream absent", "partially reproducible", "not provenance-verifiable", "Nondeterministic training and no crop ledger."),
        ("PROTOCOL-SEG-ARCH", "main.tex:337", "six stages 32..1024 with residual units", "both segmentation configs match", "fully reproducible", "valid at config/checkpoint level", "Strict nuclei checkpoint load passes."),
        ("PROTOCOL-EPOCHS", "main.tex:338", "200 training epochs", "membrane configs 200; nuclei 100; membrane segmentation last epoch 131", "partially reproducible", "manuscript does not describe nuclei run", "Checkpoint/config conflict across cases."),
        ("PROTOCOL-BATCH", "main.tex:338", "batch size 32", "membrane configs 32; nuclei configs 12", "partially reproducible", "manuscript does not describe nuclei run", "Saved configs are authoritative for replay."),
        ("PROTOCOL-OPTIMIZER", "main.tex:338", "Adam optimizer at initial lr=0.1", "configs/checkpoints use AdamW; segmentation lr=.1, regression lr=.001", "currently not reproducible", "optimizer identity differs", "No manuscript-matching run provenance."),
        ("PROTOCOL-REG-SVD", "main.tex:348", "3x3 rotation output orthogonalized by SVD", "current source/checkpoints implement this", "fully reproducible", "architectural description matches", "Downstream error metric remains problematic."),
        ("PROTOCOL-REG-ARCH", "main.tex:351-352", "five blocks starting at 16 filters", "nuclei matches; membrane has 3 stages starting at 32", "partially reproducible", "case-dependent contradiction", "One architecture is presented for both cases."),
        ("PROTOCOL-THRESHOLD", "main.tex:357,372", "thresholds selected on validation", "selection exists, but val_dir=test_dir for both cases", "partially reproducible", "invalid for untouched-test inference", "Test labels can select parameters."),
        ("PROTOCOL-FILTER", "main.tex:357-372", "retain only if S(O)>tau_S", "evaluator retains equality and skips first foreground component", "currently not reproducible", "implementation differs", "Membrane FP at .008736 survives cutoff .15."),
        ("PROTOCOL-MATCHING", "main.tex:318,372", "one-to-one 10-voxel, +/-1-frame matching", "evaluator uses temporal dilation, 4D components, near-zero IoU, greedy matching", "currently not reproducible", "definition differs and edge cases fail", "Stated nuclei matcher gives 121/4/24, not 124/1/21."),
        ("PROTOCOL-F1", "main.tex:318-320,374", "F1=2TP/(2TP+FN+FP)", "formula exactly reconstructs both displays", "fully reproducible", "valid conditional on event definition", "Object definition remains disputed."),
        ("PROTOCOL-CI", "main.tex:374", "95% multinomial confidence interval", "logit multinomial delta method reproduces both displays", "fully reproducible", "arithmetic valid conditional on event independence", "Single-movie clustering not represented."),
    ]
    for claim_id, location, stated, observed, classification, validity, note in protocol:
        rows.append(claim(
            claim_id=claim_id, claim_type="protocol", case="DARE3D pipeline",
            manuscript_location=location, manuscript_claim=stated,
            expected_value=stated, unit="protocol",
            dataset_and_frame_scope="released nuclei and membrane runs",
            denominator="not applicable",
            split_role="saved train/validation/test config",
            preprocessing="as stated",
            checkpoint="best and last metadata inspected",
            prediction_source="source/config/checkpoint audit",
            matching_rule="not applicable", metric_formula="not applicable",
            aggregation="not applicable", uncertainty_definition="not applicable",
            reproduced_value=observed,
            numerical_agreement="configuration/source comparison",
            methodological_validity=validity, classification=classification,
            evidence=(
                "evidence/checkpoint_metadata.json; "
                "evidence/scientific_edge_cases.json; "
                "evidence/nuclei_checkpoint_compatibility_summary.json; "
                "evidence/workflow_path_checks.json"
            ),
            blocking_issue_or_note=note,
        ))
    if len(rows) != 45:
        raise AssertionError(f"Expected 45 claims, got {len(rows)}")
    return rows


ARTIFACT_FIELDS = [
    "artifact_id", "category", "expected_or_role", "path_or_pattern", "status",
    "integrity_or_observation", "used_for", "evidence", "impact",
]


def build_artifacts(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    p = inputs["provenance"]["zenodo_archive"]
    fallback_f1 = inputs["seg_ckpt"]["runs"][2]["f1"]
    retraining = inputs["nematic_retraining"]
    values = [
        ("ART-SOURCE", "source", "current source snapshot", SOURCE_COMMIT, "present/hash-frozen", "222 tracked files", "current-code checks", "evidence/source_manifest_sha256.csv; evidence/provenance.json", "Historical source absent."),
        ("ART-MANUSCRIPT", "manuscript", "active manuscript", "manuscript_280826version/", "present/hash-frozen", "29 files", "claim extraction", "evidence/manuscript_manifest_sha256.csv", "Untracked but read-only."),
        ("ART-ZIP", "acquisition", "published archive", p["path"], "present/verified", f"bytes={p['bytes']}; md5={p['md5']}; sha256={p['sha256']}", "artifact source", "evidence/provenance.json", "Integrity verified."),
        ("ART-BUNDLE", "data", "extracted bundle", "DARE3d_data_190326/", "present/hash-frozen", "278 files", "all replays", "evidence/data_manifest_sha256.csv; evidence/artifact_audit_summary.json", "Released files unchanged."),
        ("ART-HIST-SOURCE", "provenance", "historical paper source", "not supplied", "absent", "no snapshot", "historical replay", "evidence/artifact_audit_summary.json", "Current compatibility is not historical reproduction."),
        ("ART-HIST-ENV", "environment", "historical lock/container", "not supplied", "absent", "no complete environment", "bitwise replay/retraining", "evidence/environment_report.json", "Checkpoint Lightning versions insufficient."),
        ("ART-SCALES", "data", "scales.json used by four configs", "DARE3d_data_190326/**/scales.json", "absent", "fallback=[.621,.621,2]", "resampling/physical coordinates", "evidence/nuclei_checkpoint_compatibility_summary.json", f"Fallback nuclei F1={fallback_f1:.6f}."),
        ("ART-NUC-RAW", "data", "nuclei test image", "DARE3d_data_190326/Gastruloid_241025/test_input/movie2.tif", "present", "byte-identical to training movie2", "nuclei inference", "evidence/duplicate_artifacts.json", "Split independence not established."),
        ("ART-NUC-LABEL", "annotation", "nuclei daughter annotations", ".../trainingset/movie2/label/movie2.tif", "present", "226 pairs; 145 evaluated components", "nuclei metrics", "evidence/annotation_summary.json", "Exclusion/merging changes denominator."),
        ("ART-MEM-RAW", "data", "raw neural images", "expected outside runs", "absent", "0 raw input TIFFs", "membrane inference/retraining", "evidence/regression_artifact_analysis.json", "Blocks independent replay."),
        ("ART-MEM-LABEL", "annotation", "raw neural daughter labels", "expected outside runs", "absent", "0 annotation TIFFs", "matching/regression/retraining", "evidence/regression_artifact_analysis.json", "Blocks event validation."),
        ("ART-MEM-SET3", "data", "Dataset 2 set 3 (11 frames/104)", "not identifiable", "absent/unresolved", "released movie_I2 has 21 frames", "membrane claims", "evidence/tiff_inventory.csv", "Result/dataset identity conflict."),
        ("ART-NUC-SEG-CONFIG", "configuration", "nuclei segmentation", ".../segmentation3d_exp10-b/.hydra/config.yaml", "present", "100 epochs; batch12; val=test", "segmentation", "evidence/checkpoint_metadata.json", "References missing scales."),
        ("ART-NUC-REG-CONFIG", "configuration", "nuclei regression", ".../regression3d_exp10-b/.hydra/config.yaml", "present", "100 epochs; batch12; 5 stages/16", "regression", "evidence/checkpoint_metadata.json", "References missing scales."),
        ("ART-MEM-SEG-CONFIG", "configuration", "membrane segmentation", ".../segmentation3d_new_set_og/runs/12-01-26/.hydra/config.yaml", "present", "200 epochs; batch32; val=test", "segmentation", "evidence/checkpoint_metadata.json; evidence/workflow_path_checks.json", "Callback paths identify 05-02-26."),
        ("ART-MEM-REG-CONFIG", "configuration", "membrane regression", ".../regression3d_new_set_og/runs/12-01-26/.hydra/config.yaml", "present", "200 epochs; batch32; 3 stages/32", "regression", "evidence/checkpoint_metadata.json", "Architecture differs from manuscript."),
        ("ART-NUC-SEG-CKPTS", "checkpoint", "nuclei segmentation best/last", "{epoch_067,last}.ckpt", "present/distinct", "epochs 67/99; Lightning2.2.5", "inference", "evidence/checkpoint_metadata.json; evidence/nuclei_checkpoint_compatibility_summary.json", "epoch067 identified; non-bitwise."),
        ("ART-NUC-REG-CKPTS", "checkpoint", "nuclei regression best/last", "{epoch_098,last}.ckpt", "present/distinct", "epochs 98/99; Lightning2.2.5", "inference", "evidence/checkpoint_metadata.json; evidence/nuclei_regression_checkpoint_compatibility_summary.json", "epoch098 aggregate identified."),
        ("ART-MEM-SEG-CKPTS", "checkpoint", "membrane segmentation best/last", "{epoch_057,last}.ckpt", "present/distinct", "epochs 57/131; Lightning2.5.6", "inference", "evidence/checkpoint_metadata.json", "Raw input absent; run conflict."),
        ("ART-MEM-REG-CKPTS", "checkpoint", "membrane regression best/last", "{epoch_147,last}.ckpt", "present/distinct", "epochs 147/199; Lightning2.5.6", "inference", "evidence/checkpoint_metadata.json", "Raw images/labels absent."),
        ("ART-NUC-PROB", "prediction", "nuclei probability", ".../runs/01-01/movie2.tif", "present", "exact current evaluator", "artifact/checkpoint replay", "evidence/nuclei_saved_probability_replay.json", "epoch067 near-identical, not bitwise."),
        ("ART-MEM-PROB", "prediction", "membrane probability", ".../runs/12-01-26/movie_I2.tif", "present", "197 components; 122 retained", "component reconstruction", "evidence/membrane_fp_investigation.json", "Raw matching labels absent."),
        ("ART-SAVED-STATS", "result", "segmentation/regression stats", "released runs/**/stats.csv", "present", "headline aggregates traceable", "arithmetic/regression", "evidence/provenance.json; evidence/regression_artifact_analysis.json", "Stats do not prove provenance."),
        ("ART-NUC-REG-RAW", "prediction", "historical nuclei raw regression", "expected NPZ", "absent", "0 NPZ", "event-wise/bitwise comparison", "evidence/regression_artifact_analysis.json", "Only aggregate comparison possible."),
        ("ART-MEM-REG-RAW", "prediction", "membrane raw regression", "released NPZs", "present", "3 arrays/350 predictions", "raster calibration", "evidence/npz_inventory.csv; evidence/regression_artifact_analysis.json", "Truth only as uint8 line."),
        ("ART-LOGS", "provenance", "logs/TensorBoard", "released runs/**", "present/incomplete", "best epoch and geometry clues", "selection/geometry", "evidence/checkpoint_metadata.json", "Not a scale/environment substitute."),
        ("ART-SPLIT", "split", "distinct validation/test", "saved configs", "absent", "test_dir=val_dir in all four", "claim validity", "evidence/checkpoint_metadata.json", "Prevents untouched-test claim."),
        ("ART-NUC-CENTERS", "intermediate", "historical center-pair object", "not supplied", "absent", "reconstructed from released probability", "regression replay", "evidence/nuclei_regression_checkpoint_compatibility_summary.json", "Strong compatibility, not bitwise provenance."),
        ("ART-NEMATIC", "audit result", "fixed-prediction nematic-axis comparison", "docs/reproducibility_audit/evidence/nematic_axis_metric_comparison.*", "present/asserted", "14/14 assertions; 743 event rows", "angular-metric counterfactual", "evidence/nematic_axis_metric_comparison.json; evidence/nematic_axis_metric_comparison.csv; evidence/nematic_axis_event_metrics.csv", "Production source and released artifacts unchanged."),
        ("ART-NEMATIC-RETRAIN", "audit model/result", "fresh nuclei nematic-loss retraining", "docs/reproducibility_audit/nematic_retraining/seed_12345/", "present/asserted", f"100 epochs; best epoch {retraining['checkpoints']['retrained']['epoch']}; 10/10 evaluation assertions", "controlled loss/retraining counterfactual", "nematic_retraining/seed_12345/evaluation.json; NEMATIC_RETRAINING_COMPARISON.md", "Validation=test, reconstructed scale, one seed; not exact historical retraining."),
    ]
    return [dict(zip(ARTIFACT_FIELDS, row)) for row in values]


ACQUISITION_FIELDS = [
    "check_id", "workflow", "check", "status", "observed", "evidence", "impact"
]


def build_acquisition(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    p = inputs["provenance"]["zenodo_archive"]
    values = [
        ("ACQ-01", "manual/CLI", "Zenodo record and named archive resolve", "pass", "DARE3d_data_190326.zip obtained", "evidence/provenance.json", "Artifact publicly obtainable."),
        ("ACQ-02", "CLI", "download/extraction complete", "pass with post-status exception", "278 files extracted before final print", "SESSION_HANDOFF.md:159-164; evidence/data_manifest_sha256.csv", "Data usable despite exception symptom."),
        ("ACQ-03", "integrity", "published MD5", "pass", p["md5"], "evidence/provenance.json", "Archive identity verified."),
        ("ACQ-04", "integrity", "independent SHA-256", "pass", p["sha256"], "evidence/provenance.json", "Audit fingerprint."),
        ("ACQ-05", "CLI", "Windows completion message", "fail", "Unicode arrow causes cp1252 UnicodeEncodeError after extraction", "SESSION_HANDOFF.md:159-164", "Command appears to fail on native Windows."),
        ("ACQ-06", "CLI", "resume partial download", "fail", ".part opened with wb and truncated", "source audit; SESSION_HANDOFF.md", "Advertised resume absent."),
        ("ACQ-07", "CLI", "validate existing archive", "fail", "trusted by size without checksum", "source audit; SESSION_HANDOFF.md", "Same-size corruption may pass."),
        ("ACQ-08", "CLI", "--help behavior", "fail", "parsed as destination", "SESSION_HANDOFF.md", "No conventional CLI help."),
        ("ACQ-09", "layout", "both case roots", "pass", "Gastruloid_241025 and Neural_tube_160226 exist", "evidence/data_manifest_sha256.csv", "Both cases represented."),
        ("ACQ-10", "README", "Gastruloid weights paths", "pass", "config/checkpoint at documented roots", "evidence/workflow_path_checks.json", "Plugin and CLI usable."),
        ("ACQ-11", "README", "neural weights paths", "fail", "documented weights/ layout absent", "evidence/workflow_path_checks.json", "Guide path unusable verbatim."),
        ("ACQ-12", "plugin", "actual four model roots", "pass", "flat Gastruloid and dated neural roots resolve", "evidence/workflow_path_checks.json", "Models locatable manually."),
        ("ACQ-13", "CLI", "neural top-level root", "fail/undocumented workaround", "top-level rejected; runs/12-01-26 accepted", "evidence/workflow_path_checks.json", "Explicit undocumented path needed."),
        ("ACQ-14", "widget", "auto-fill after download", "partial", "works from data parent; fallback one parent high; same-session defaults stale", "evidence/workflow_path_checks.json", "Behavior depends on context/restart."),
        ("ACQ-15", "notebook", "reuse verified root bundle", "fail", "separate package-local demo downloader", "evidence/workflow_path_checks.json", "Duplicate path outside verified workflow."),
        ("ACQ-16", "metadata", "DOI description", "fail", "plugin README calls data DOI a software release", "SESSION_HANDOFF.md", "Misleading provenance wording."),
    ]
    return [dict(zip(ACQUISITION_FIELDS, row)) for row in values]

COMMAND_FIELDS = [
    "command_id", "phase", "command", "working_directory_or_context", "inputs",
    "frozen_overrides", "outcome", "exit_status", "primary_outputs", "notes",
]


def build_commands() -> list[dict[str, Any]]:
    py = r"C:\Users\ruppr\.conda\envs\dare3d-v2.0\python.exe"
    values = [
        ("CMD-01", "acquisition", f"{py} -m napari_dare3d._data .", "repo root", "Zenodo 19113351", "none", "extraction complete; final Unicode print failed", "exception after extraction", "DARE3d_data_190326/", "Released artifacts unchanged."),
        ("CMD-02", "artifact audit", f"{py} docs/reproducibility_audit/audit_artifacts.py", "repo root", "bundle/manuscript/source", "seed 20260828", "pass", "0", "manifests/inventories/arithmetic", "No model execution."),
        ("CMD-03", "checkpoint metadata", f"{py} docs/reproducibility_audit/checkpoint_metadata.py", "repo root", "8 checkpoints", "read-only", "pass", "0", "checkpoint_metadata.json", "Four best/last pairs differ."),
        ("CMD-04", "saved nuclei replay", f"{py} docs/reproducibility_audit/replay_saved_nuclei.py", "repo root", "probability/label/stats", ".55/.15; matcher tolerances frozen", "pass", "0", "nuclei replay JSON/CSVs", "124/1/21 current; 121/4/24 stated."),
        ("CMD-05", "edge cases", f"{py} docs/reproducibility_audit/scientific_edge_cases.py", "repo root", "current production functions", "24 assertions", "pass", "0", "scientific_edge_cases.json", "Non-fatal Numba warning."),
        ("CMD-06", "regression artifacts", f"{py} docs/reproducibility_audit/regression_artifact_analysis.py", "repo root", "stats/rasters/NPZs", "19 assertions", "pass", "0", "regression JSON/CSVs", "Artifact-only."),
        ("CMD-07", "nuclei seg checkpoint", f"{py} docs/reproducibility_audit/checkpoint_compatibility.py --checkpoint best --scale-mode log_reconstructed --batch-size 12", "repo root", "epoch067/movie2", "scale [.912,.912,.912] from log", "pass", "0", "segmentation best run", "Exact objects; non-bitwise probability."),
        ("CMD-08", "nuclei seg checkpoint", f"{py} docs/reproducibility_audit/checkpoint_compatibility.py --checkpoint last --scale-mode log_reconstructed --batch-size 12", "repo root", "last/movie2", "same geometry", "pass", "0", "segmentation last run", "123/4/22; F1 .904412."),
        ("CMD-09", "scale diagnostic", f"{py} docs/reproducibility_audit/checkpoint_compatibility.py --checkpoint best --scale-mode bundle_default --batch-size 12", "repo root", "epoch067/movie2", "fallback [.621,.621,2]", "pass", "0", "segmentation fallback run", "32/5/113; F1 .351648."),
        ("CMD-10", "seg summary", f"{py} docs/reproducibility_audit/summarize_checkpoint_compatibility.py", "repo root", "3 predeclared runs", "9 assertions", "pass", "0", "nuclei checkpoint summary", "No post-hoc search."),
        ("CMD-11", "membrane components", f"{py} docs/reproducibility_audit/investigate_membrane_fp.py", "repo root", "probability/renderings/stats", "T,X,Y,Z order", "pass", "0", "membrane FP evidence", "12/12."),
        ("CMD-12", "workflow/API", f"{py} docs/reproducibility_audit/workflow_path_checks.py", "repo root", "manifest/paths/3 frames", "public defaults", "pass", "0", "workflow evidence", "15/15 and real GPU smoke."),
        ("CMD-13", "nuclei reg checkpoint", f"{py} docs/reproducibility_audit/regression_checkpoint_compatibility.py --checkpoint best", "repo root", "epoch098/frozen centers", "no seg rerun/tuning/resampling", "pass", "0", "regression best run", "21 aggregates within .02."),
        ("CMD-14", "nuclei reg checkpoint", f"{py} docs/reproducibility_audit/regression_checkpoint_compatibility.py --checkpoint last", "repo root", "last/frozen centers", "same protocol", "pass", "0", "regression last run", "Max discrepancy 1.47155."),
        ("CMD-15", "reg summary", f"{py} docs/reproducibility_audit/summarize_regression_checkpoint_compatibility.py", "repo root", "2 predeclared runs", "9 assertions", "pass", "0", "regression checkpoint summary", "epoch098 identified."),
        ("CMD-16", "final package", f"{py} docs/reproducibility_audit/build_final_audit_package.py --hash-manifest", "repo root", "all frozen evidence", "no model/retraining", "pass on finalization", "0", "ledgers/reports/manifest", "Synthesis only."),
        ("CMD-17", "angular metric", f"{py} docs/reproducibility_audit/nematic_axis_metric_comparison.py", "repo root", "same predictions/truth/centers/matches", "only metric changed; no retraining", "pass", "0", "nematic comparison JSON/CSVs/report", "14/14 assertions; 743 event rows."),
        ("CMD-18", "nematic retraining preflight", f"{py} docs/reproducibility_audit/nematic_retraining_experiment.py preflight --initialize-data", "repo root", "movie2/3/4 and archived regression config", "seed 12345; log-reconstructed geometry", "pass", "0", "preflight.json; experiment_config.json", "10/10 assertions; production unchanged."),
        ("CMD-19", "nematic retraining", f"{py} docs/reproducibility_audit/nematic_retraining_experiment.py train", "repo root", "526 train crops; 156 validation crops", "only angular loss changed", "100 epochs complete; best epoch 95", "0", "checkpoint/logs/run_state", "3338.20 seconds; 16700 steps; fresh initialization."),
        ("CMD-20", "paired retraining evaluation", f"{py} docs/reproducibility_audit/nematic_retraining_evaluation.py", "repo root", "original epoch098 and retrained epoch095", "same events/centers/matches; corrected axis metric", "pass", "0", "evaluation JSON/CSVs/predictions/report", "10/10 assertions; all-GT corrected mean 14.894 to 12.575 degrees."),
    ]
    return [dict(zip(COMMAND_FIELDS, row)) for row in values]


TEST_FIELDS = [
    "test_id", "scope", "command_or_check", "outcome", "counts_or_result",
    "evidence", "interpretation",
]


def build_tests() -> list[dict[str, Any]]:
    values = [
        ("TEST-01", "hardware", "CUDA/cuDNN smoke", "pass", "Quadro RTX 5000; cuDNN9.1", "SESSION_HANDOFF.md:263-274", "GPU environment works."),
        ("TEST-02", "production unit", "tests/test_geometry.py", "pass", "all passed", "SESSION_HANDOFF.md:263-274", "Geometry helpers pass."),
        ("TEST-03", "production unit", "tests/test_train_commands.py", "pass", "all passed", "SESSION_HANDOFF.md:263-274", "Command construction passes."),
        ("TEST-04", "checkpoint load", "focused loading", "pass", "5 passed", "SESSION_HANDOFF.md:263-274", "Released state dicts load."),
        ("TEST-05", "config/eval", "focused selection", "partial/fail", "5 passed, 1 failed, 2 errors", "SESSION_HANDOFF.md:263-274", "Stale Hydra fixtures."),
        ("TEST-06", "production suite", "pytest -k 'not slow'", "blocked at collection", "missing pkg_resources", "SESSION_HANDOFF.md:263-274", "Full suite not executable."),
        ("TEST-07", "scientific", "scientific_edge_cases.py", "pass", "24/24", "evidence/scientific_edge_cases.json", "Metric/matching defects frozen."),
        ("TEST-08", "regression artifacts", "regression_artifact_analysis.py", "pass", "19/19", "evidence/regression_artifact_analysis.json", "743 rows and aggregates."),
        ("TEST-09", "nuclei segmentation", "checkpoint summary", "pass", "9/9", "evidence/nuclei_checkpoint_compatibility_summary.json", "Best/last/fallback frozen."),
        ("TEST-10", "membrane components", "FP investigation", "pass", "12/12", "evidence/membrane_fp_investigation.json", "Hidden FP/filter exception."),
        ("TEST-11", "workflow", "workflow_path_checks.py", "pass", "15/15 + GPU smoke", "evidence/workflow_path_checks.json", "Path runs; protocol differs."),
        ("TEST-12", "nuclei regression", "checkpoint summary", "pass", "9/9", "evidence/nuclei_regression_checkpoint_compatibility_summary.json", "epoch098 compatibility."),
        ("TEST-13", "full retraining", "four models + seeds", "not run / blocked", "missing data/scales/split/environment", "evidence/artifact_ledger.csv", "Required for full-pipeline status."),
        ("TEST-14", "membrane checkpoints", "inference/matching", "not run / blocked", "raw images/labels absent", "evidence/artifact_ledger.csv", "Checkpoint identity unproved."),
        ("TEST-15", "angular metric", "nematic_axis_metric_comparison.py", "pass", "14/14 assertions; 732 evaluated pairs", "evidence/nematic_axis_metric_comparison.json", "Fixed-prediction counterfactual with synthetic cases."),
        ("TEST-16", "nematic retraining preflight", "nematic_retraining_experiment.py preflight --initialize-data", "pass", "10/10", "nematic_retraining/seed_12345/preflight.json", "Data, invariance, gradient, CUDA, crop-count, and integrity checks."),
        ("TEST-17", "paired nematic retraining", "nematic_retraining_evaluation.py", "pass", "10/10; 156 controlled pairs", "nematic_retraining/seed_12345/evaluation.json", "Moderate GT-center gain; predicted-center interval spans zero."),
    ]
    return [dict(zip(TEST_FIELDS, row)) for row in values]


ENV_FIELDS = [
    "component", "current_value", "historical_or_checkpoint_value", "status",
    "evidence", "reproducibility_impact",
]


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def run_text(command: list[str]) -> str:
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


def collect_environment(
    inputs: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        import torch
        cuda = {
            "available": bool(torch.cuda.is_available()),
            "torch_cuda": torch.version.cuda,
            "cudnn": str(torch.backends.cudnn.version()),
            "gpu_count": int(torch.cuda.device_count()),
            "gpus": [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ],
        }
    except Exception as error:
        cuda = {"available": False, "error": repr(error)}
    nvidia = run_text([
        "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    ])
    names = [
        "torch", "monai", "lightning", "pytorch-lightning", "numpy", "scipy",
        "scikit-image", "tifffile", "hydra-core", "omegaconf", "dare3d",
    ]
    core = {name: package_version(name) for name in names}
    packages = []
    for dist in sorted(
        metadata.distributions(),
        key=lambda item: (item.metadata.get("Name", "").lower(), item.version),
    ):
        direct_url = dist.read_text("direct_url.json") or ""
        packages.append({
            "name": dist.metadata.get("Name") or "unknown",
            "version": dist.version,
            "location": str(dist.locate_file("")),
            "direct_url_json": direct_url.strip(),
        })
    checkpoint_versions = {
        case: {
            variant: details["pytorch_lightning_version"]
            for variant, details in pair.items()
        }
        for case, pair in inputs["checkpoint"]["checkpoints"].items()
    }
    report = {
        "captured_at": datetime.now().astimezone().isoformat(),
        "source_commit": SOURCE_COMMIT,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "core_packages": core,
        "cuda": cuda,
        "nvidia_smi": nvidia,
        "checkpoint_recorded_lightning_versions": checkpoint_versions,
        "training_deterministic": False,
        "editable_installation_warning": (
            "Editable dare3d points to an older checkout; audit scripts prepend "
            "and verify this checkout."
        ),
        "historical_environment_status": "absent",
        "package_inventory": "evidence/environment_packages.csv",
    }
    values = [
        ("source commit", SOURCE_COMMIT, "historical snapshot absent", "current frozen", "evidence/source_manifest_sha256.csv", "Compatibility is not historical reproduction."),
        ("OS", platform.platform(), "not recorded", "current only", "evidence/environment_report.json", "Bitwise portability unknown."),
        ("Python", platform.python_version(), "not recorded", "current only", "evidence/environment_report.json", "Historical interpreter unknown."),
        ("PyTorch", core["torch"], "not recorded", "current only", "evidence/environment_report.json", "Current compatibility only."),
        ("CUDA", str(cuda.get("torch_cuda", "unknown")), "not recorded", "current only", "evidence/environment_report.json", "GPU numerical variance possible."),
        ("cuDNN", str(cuda.get("cudnn", "unknown")), "not recorded", "current only", "evidence/environment_report.json", "Kernel provenance absent."),
        ("GPU/driver", nvidia, "not recorded", "current only", "evidence/environment_report.json", "Exact hardware unavailable."),
        ("MONAI", core["monai"], "not recorded", "current only", "evidence/environment_report.json", "Transforms version-sensitive."),
        ("Lightning", core["lightning"], "2.2.5 nuclei; 2.5.6 membrane", "mismatch documented", "evidence/checkpoint_metadata.json", "Strict loads pass where tested."),
        ("NumPy", core["numpy"], "not recorded", "current only", "evidence/environment_report.json", "Arithmetic frozen."),
        ("SciPy", core["scipy"], "not recorded", "current only", "evidence/environment_report.json", "Components/statistics version-sensitive."),
        ("scikit-image", core["scikit-image"], "not recorded", "current only", "evidence/environment_report.json", "Historical component behavior unpinned."),
        ("tifffile", core["tifffile"], "not recorded", "current only", "evidence/environment_report.json", "Artifact reads hash-frozen."),
        ("Hydra/OmegaConf", f"{core['hydra-core']} / {core['omegaconf']}", "not recorded", "stale tests", "evidence/test_ledger.csv", "Some tests fail on fixtures."),
        ("determinism", "false in all four configs", "false", "known", "evidence/checkpoint_metadata.json", "Seed variance required for retraining."),
        ("dare3d install", core["dare3d"], "historical package absent", "editable path mismatch mitigated", "evidence/environment_report.json", "Unpinned imports can hit old checkout."),
        ("package inventory", f"{len(packages)} distributions", "not recorded", "current only", "evidence/environment_packages.csv", "Session recovery, not history."),
    ]
    ledger = [dict(zip(ENV_FIELDS, row)) for row in values]
    return report, packages, ledger


def write_manifest() -> int:
    rows = []
    for path in sorted(HERE.rglob("*")):
        if (
            not path.is_file()
            or path == MANIFEST
            or "__pycache__" in path.parts
            or path.suffix == ".pyc"
        ):
            continue
        rows.append({
            "path": rel(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
        if path.stat().st_size >= 100_000_000:
            print(
                f"hashed {rel(path)} ({path.stat().st_size} bytes)",
                flush=True,
            )
    write_csv(MANIFEST.name, rows, ["path", "bytes", "sha256"])
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hash-manifest", action="store_true")
    args = parser.parse_args()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    inputs = {
        "artifact": load_json("artifact_audit_summary.json"),
        "arithmetic": load_json("arithmetic_and_uncertainty.json"),
        "provenance": load_json("provenance.json"),
        "annotations": load_json("annotation_summary.json"),
        "checkpoint": load_json("checkpoint_metadata.json"),
        "replay": load_json("nuclei_saved_probability_replay.json"),
        "scientific": load_json("scientific_edge_cases.json"),
        "regression": load_json("regression_artifact_analysis.json"),
        "seg_ckpt": load_json("nuclei_checkpoint_compatibility_summary.json"),
        "reg_ckpt": load_json(
            "nuclei_regression_checkpoint_compatibility_summary.json"
        ),
        "membrane": load_json("membrane_fp_investigation.json"),
        "workflow": load_json("workflow_path_checks.json"),
        "nematic": load_json("nematic_axis_metric_comparison.json"),
        "nematic_retraining": json.loads(
            RETRAINING_EVALUATION.read_text(encoding="utf-8")
        ),
        "nematic_retraining_preflight": json.loads(
            RETRAINING_PREFLIGHT.read_text(encoding="utf-8")
        ),
    }
    claims = build_claims(inputs)
    artifacts = build_artifacts(inputs)
    acquisition = build_acquisition(inputs)
    commands = build_commands()
    tests = build_tests()
    environment, packages, environment_ledger = collect_environment(inputs)

    write_csv("claim_ledger.csv", claims, CLAIM_FIELDS)
    write_csv("artifact_ledger.csv", artifacts, ARTIFACT_FIELDS)
    write_csv("acquisition_ledger.csv", acquisition, ACQUISITION_FIELDS)
    write_csv("command_ledger.csv", commands, COMMAND_FIELDS)
    write_csv("test_ledger.csv", tests, TEST_FIELDS)
    write_csv("environment_ledger.csv", environment_ledger, ENV_FIELDS)
    write_csv(
        "environment_packages.csv", packages,
        ["name", "version", "location", "direct_url_json"],
    )
    write_json("environment_report.json", environment)

    acquisition_report = {
        "classification": "Artifacts available but guide broken",
        "archive": inputs["provenance"]["zenodo_archive"],
        "checks": dict(Counter(row["status"] for row in acquisition)),
        "conclusion": (
            "Verified artifacts are obtainable after manual recovery, but the "
            "documented Windows workflow and neural layout do not work verbatim."
        ),
        "ledger": "evidence/acquisition_ledger.csv",
    }
    write_json("acquisition_report.json", acquisition_report)

    suites = {
        "scientific_edge_cases": (
            inputs["scientific"]["assertions"]["all_passed"]
        ),
        "regression_artifact_analysis": (
            inputs["regression"]["all_assertions_pass"]
        ),
        "nuclei_segmentation_checkpoint": all(
            inputs["seg_ckpt"]["assertions"].values()
        ),
        "nuclei_regression_checkpoint": all(
            inputs["reg_ckpt"]["assertions"].values()
        ),
        "membrane_components": all(
            inputs["membrane"]["assertions"].values()
        ),
        "workflow_paths": all(inputs["workflow"]["assertions"].values()),
        "nematic_axis_metric": all(
            inputs["nematic"]["assertions"].values()
        ),
        "nematic_retraining_preflight": (
            inputs["nematic_retraining_preflight"]["all_assertions_pass"]
        ),
        "nematic_retraining_evaluation": (
            inputs["nematic_retraining"]["all_assertions_pass"]
        ),
    }
    assertions = {
        "source_commit_frozen": (
            inputs["provenance"]["git_head"] == SOURCE_COMMIT
        ),
        "archive_md5_published": (
            inputs["provenance"]["zenodo_archive"]["md5"]
            == "f8df147bb8249aad44b8ef1509d280dd"
        ),
        "prior_suites_pass": all(suites.values()),
        "45_unique_claims": (
            len(claims) == len({row["claim_id"] for row in claims}) == 45
        ),
        "all_claims_classified": all(
            row["classification"] in {
                "fully reproducible",
                "partially reproducible",
                "currently not reproducible",
                "external/out of scope",
            }
            for row in claims
        ),
        "artifact_ledger_complete": len(artifacts) >= 25,
        "environment_inventory_nonempty": len(packages) >= 20,
        "nuclei_saved_replay_exact": (
            inputs["replay"]["exact_count_agreement"]
            and inputs["replay"]["exact_metric_agreement_1e_minus_15"]
        ),
        "nuclei_seg_best_exact_objects": (
            inputs["seg_ckpt"]["assertions"]["best_log_exact_released_metrics"]
        ),
        "nuclei_reg_best_within_0_02": (
            inputs["reg_ckpt"]["assertions"][
                "best_all_21_values_are_within_0_02_of_archive"
            ]
        ),
        "membrane_122_components": (
            inputs["membrane"]["assertions"][
                "reconstructed_filtered_prediction_total_is_122"
            ]
        ),
        "nematic_axis_comparison": (
            inputs["nematic"]["all_assertions_pass"]
        ),
        "nematic_retraining_preflight": (
            inputs["nematic_retraining_preflight"]["all_assertions_pass"]
        ),
        "nematic_retraining_evaluation": (
            inputs["nematic_retraining"]["all_assertions_pass"]
        ),
    }
    summary = {
        "audit_date": "2026-08-29",
        "source_commit": SOURCE_COMMIT,
        "scope": (
            "Entire current DARE3D pipeline versus active manuscript claims"
        ),
        "overall_classification": "currently not reproducible",
        "reason": (
            "End-to-end reproduction/retraining is blocked by missing neural "
            "raw data/annotations, missing scales and historical source/environment, "
            "non-independent validation/test splits, and metric/protocol discrepancies."
        ),
        "claim_counts": dict(
            Counter(row["classification"] for row in claims)
        ),
        "numerically_strong_partial_results": [
            (
                "Nuclei segmentation epoch_067 reproduces 124/1/21 and F1 "
                ".9185185 under log-reconstructed geometry."
            ),
            (
                "Nuclei regression epoch_098 reproduces all 21 aggregates "
                "within .02."
            ),
            (
                "Membrane probability reconstructs 122 predictions consistent "
                "with 114 TP + 8 FP; matching cannot be rerun."
            ),
            (
                "Fixed-prediction nematic-axis means are 14.89 degrees for "
                "nuclei and 36.63 degrees for membrane in all-GT mode."
            ),
            (
                "Fresh nuclei nematic-loss retraining lowers the corrected "
                "all-GT mean from 14.894 to 12.575 degrees; predicted-center "
                "improvement is not material."
            ),
        ],
        "fully_reproducible_subclaims": [
            row["claim_id"]
            for row in claims
            if row["classification"] == "fully reproducible"
        ],
        "blocked_core_tasks": [
            "independent membrane checkpoint inference/matching",
            "bitwise historical inference",
            "four-model retraining with distinct untouched test data",
        ],
        "prior_suite_assertions": suites,
        "package_assertions": assertions,
        "ledgers": {
            "claims": "evidence/claim_ledger.csv",
            "artifacts": "evidence/artifact_ledger.csv",
            "acquisition": "evidence/acquisition_ledger.csv",
            "commands": "evidence/command_ledger.csv",
            "tests": "evidence/test_ledger.csv",
            "environment": "evidence/environment_ledger.csv",
            "packages": "evidence/environment_packages.csv",
        },
        "report": rel(REPORT),
    }
    write_json("audit_package_summary.json", summary)
    failed = [name for name, passed in assertions.items() if not passed]
    if failed:
        raise AssertionError(f"Final package assertions failed: {failed}")

    manifest_count = 0
    if args.hash_manifest:
        if not REPORT.is_file():
            raise FileNotFoundError(f"Final report missing: {REPORT}")
        manifest_count = write_manifest()
    print(json.dumps({
        "status": "complete",
        "claims": len(claims),
        "claim_counts": summary["claim_counts"],
        "artifacts": len(artifacts),
        "acquisition_checks": len(acquisition),
        "commands": len(commands),
        "tests": len(tests),
        "environment_components": len(environment_ledger),
        "packages": len(packages),
        "assertions": f"{len(assertions)}/{len(assertions)}",
        "manifest_files": manifest_count,
    }, indent=2))


if __name__ == "__main__":
    main()
