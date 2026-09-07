# DARE3D Scientific Reproducibility Audit Plan

## 1. Freeze scope, evidence, and success criteria

- Audit the current code at commit fe2b14d732359f2bdaf8b197574ad818899ce123 together with a hash-identified snapshot of the untracked manuscript directory. Use a clean, immutable checkout and store outputs separately.
- Cover every active DARE3D-derived quantitative claim:
  - Dataset 2, set 3 dimensions, frame count, 104 annotations, and division-length statistics.
  - Membrane model: 114 TP, 8 FP, 8 FN, F1 93.4%, CI 89.5?96.0%, and orientation error \(28^\circ\pm2^\circ\).
  - Dataset 3, set 1 dimensions, frame count, 226 annotations, and division-length statistics.
  - Nuclei model: 124 TP, 1 FP, 21 FN, F1 91.8%, CI 87.8?94.6%, and orientation error \(28^\circ\pm2^\circ\).
  - Derived claims such as F1 above 90%, the 61.2? random-orientation baseline, and approximately 20? annotation uncertainty.
- Catalogue DARE2d-only results as requiring the separate DARE2d repository. Do not use them to judge DARE3D except for explicit cross-method comparisons.
- Use progressively stronger evidence:
  1. Dataset and arithmetic reconstruction.
  2. Recalculation from saved predictions.
  3. Inference from released checkpoints.
  4. Complete retraining from released data.
- Pre-register comparison rules:
  - Checkpoint replay must reproduce TP/FP/FN exactly.
  - F1, CI, dataset, and regression statistics must agree at the manuscript?s displayed precision using the confirmed definitions.
  - Retrained weights need not be byte-identical if the historical workflow was nondeterministic, but the published-seed run must use the exact released protocol without tuning.
  - Leakage, test-set optimization, incompatible metrics, or undocumented exclusions prevent a ?fully reproducible? rating even when numerical values match.

## 2. Audit the documented data-download workflow

- Treat successful data acquisition by a new user as part of reproducibility, not merely a setup convenience.
- In a clean clone with no pre-existing data, follow the root README instructions verbatim:
  - Open Zenodo record 19113351: https://zenodo.org/records/19113351.
  - Download DARE3d_data_190326.zip.
  - Extract it at the repository root.
  - Verify that the documented DARE3d_data_190326/<case>/weights/{segmentation3d_*,regression3d_*} paths actually exist and are accepted by the CLI, notebooks, and plugin.
- Independently test every advertised automated path:
  - python -m napari_dare3d._data <destination>;
  - the napari ?Download DARE3D data? widget;
  - the first-run demo downloader, if it remains part of the public workflow.
- For each path, verify:
  - the Zenodo URL and API record resolve without authentication;
  - the correct published file and version are selected;
  - download size and required free space are communicated;
  - partial downloads, timeouts, interruption, reruns, and existing files behave as documented;
  - extraction finishes without double nesting, stale partial files, or platform-specific junk;
  - the returned/discovered directory exists and contains both Gastruloid_241025 and the neural-tube case;
  - checkpoint directories contain both .hydra/config.yaml and the documented checkpoint;
  - the inference widget auto-fills valid model directories after download;
  - notebook paths and README examples match the extracted names exactly.
- Verify integrity independently. Confirm the published MD5 f8df147bb8249aad44b8ef1509d280dd, generate SHA-256 hashes for the archive and extracted files, and compare the manifests produced by manual and automated downloads. File size alone is not sufficient proof of integrity.
- Check consistency across the root README, plugin README, code constants, notebook defaults, and Zenodo metadata. Record discrepancies such as stale bundle names, case names, paths, DOI descriptions, or unsupported claims about resumability.
- Define acquisition outcomes separately:
  - **Documented download reproducible:** a clean user can follow the guide and obtain a verified, immediately usable layout.
  - **Artifacts available but guide broken:** manual recovery is possible, but one or more documented or automated workflows fail or require undocumented intervention.
  - **Artifacts unavailable:** the required version, archive, or files cannot be obtained or verified.
- Use the verified download produced by this phase as the sole artifact source for subsequent scoring. Do not repair directory layouts silently.

## 3. Build the claim, artifact, and provenance ledger

- Create one ledger row per manuscript value with: manuscript location, dataset/set and frame range, denominator, split role, preprocessing, checkpoint, resolved configuration, prediction source, matching rule, metric formula, aggregation, uncertainty calculation, expected value, and evidence status.
- Resolve two apparent denominator conflicts:
  - Nuclei TP+FN gives 145 evaluated events versus 226 annotations reported for Dataset 3 set 1.
  - Membrane TP+FN gives 122 evaluated events versus 104 annotations reported for Dataset 2 set 3.
  Determine whether these represent different files, frame ranges, temporal merging, exclusions, or manuscript errors. Permit no silent exclusions.
- Inventory images, annotations, sparse weights, scales.json, predictions, checkpoints, saved Hydra configurations, logs, and environment metadata. Tie every checkpoint unambiguously to a case, split, architecture, seed, and selection rule.
- Compare the current source with any archived code in the data release. Distinguish:
  - reproducibility with the historical paper code;
  - compatibility of historical artifacts with the current pipeline.
- Reconstruct the software environment using this precedence: released lock or container metadata, recorded run environment, archived documentation, then current dependency constraints. Record Python, PyTorch, MONAI, Lightning, CUDA, cuDNN, GPU, OS, and deterministic settings.
- Validate each dataset:
  - TIFF axis order, shape, dtype, intensity range, frame count, file pairing, and checksums.
  - Odd/even daughter pairing, missing or duplicated labels, event counts per frame/movie, border events, and length mean, SD, and SEM.
  - Raw versus preprocessed status and the lineage of contrast correction or isotropized copies.
  - Hash-based train/validation/test disjointness, including copied files, derived halves, overlapping frames, and movies from the same experiment.

## 4. Audit scientific definitions before model execution

- Trace the complete data flow: TIFF loading, axis swaps, scale lookup, resampling, normalization, segmentation masks, inference, thresholding, filtering, temporal merging, matching, regression crops, orientation decoding, and aggregation.
- Reconcile released configurations with the manuscript, focusing on:
  - manuscript versus current epoch, batch-size, optimizer, learning-rate, scheduler, and checkpoint-selection settings;
  - the manuscript?s five-block regression CNN versus current selectable architectures;
  - histogram/local-contrast preprocessing versus current normalization and augmentation;
  - current fallback scale [0.621,0.621,2] versus manuscript acquisition scales such as (0.208,0.208,1) ?m.
  Saved run configurations take precedence; missing historical settings must be reported rather than inferred.
- Verify splitting and selection:
  - Dataset 2 set 3 and Dataset 3 set 1 must be absent from training and parameter selection.
  - Validation and test must be distinct at the experimental-movie level.
  - Examine the workflow?s test_dir=val_dir behavior and optional threshold/weighted-probability optimization on evaluated data.
  - Identify whether last.ckpt or a validation-selected epoch generated each claim. If ambiguous, report all released candidates without choosing whichever agrees best.
- Validate detection with an independent evaluator:
  - Define one-to-one object matching, spatial units and tolerance, ?1-frame tolerance, boundary behavior, and ambiguous-assignment optimization.
  - Compare this with current 4D temporal dilation, connected components, IoU/centroid matching, greedy assignment, and size?probability filtering.
  - Test same-frame and ?1-frame matches, out-of-tolerance events, duplicates, adjacent divisions, merged blobs, empty data, border events, the first component, and multi-movie aggregation.
  - Compute pooled micro-F1 and per-movie macro-F1 and identify the manuscript definition.
  - Reproduce the cited multinomial CI independently; report cluster-bootstrap sensitivity separately.
- Validate anisotropy and coordinates:
  - Confirm whether inference and evaluation operate in raw or 1 ?m isotropic coordinates.
  - Ensure masks, centers, matching tolerances, regression crops, lengths, and axes use a consistent coordinate space.
  - Check segmentation rescale/unscale and regression handling independently.
  - Verify (T,Z,Y,X) disk order, internal (T,X,Y,Z) order, displayed coordinates, and scale-vector order with landmarks.
- Validate regression:
  - Independently calculate nematic axis error as \(\arccos(|\hat u_{\mathrm{pred}}\cdot\hat u_{\mathrm{true}}|)\), constrained to 0?90?.
  - Compare it with the current full-quaternion geodesic calculation.
  - Report mean, median, SD, SEM, sample count, and distribution separately, resolving the manuscript?s ambiguous ?standard deviation ? mean ? SEM? wording.
  - Evaluate all ground-truth centers, TP-associated ground-truth centers, and matched predicted centers separately. Identify the manuscript headline mode from provenance.
  - Confirm treatment of ?1-frame detection shifts and anisotropic physical coordinates.
- Keep reference metrics and synthetic tests outside the production pipeline. Do not fix DARE3D during evidence generation.

## 5. Execute in evidence-first order

1. **Artifact-only replay**
   - Recalculate dataset statistics, F1 arithmetic, confidence intervals, random baselines, and annotation-uncertainty simulations.
   - If predictions exist, recompute results with both current and independent evaluators.
   - Produce event-level disagreement tables before running a network.

2. **Released-checkpoint inference**
   - Run nuclei and membrane cases with exact saved configurations and frozen test files.
   - Capture resolved configs, commands, probability maps, components, matches, regression outputs, and environment metadata.
   - Compare historical-code and current-code inference with identical artifacts.
   - Repeat inference on the same device and, where feasible, compare CPU and GPU.
   - Never optimize thresholds or other parameters on test labels.

3. **Full training reproduction**
   - Retraining is unnecessary for dataset statistics, metric definitions, CIs, or frozen-checkpoint performance, but is required to establish full-pipeline reproducibility.
   - Retrain segmentation and regression for both cases using the released split, exact configuration, published seed, checkpoint-selection rule, and adequate hardware.
   - Do not reduce batch size, alter augmentation, change architecture, or substitute checkpoints to accommodate hardware.
   - After the primary published-seed run, execute two additional predeclared seeds to quantify variability. Keep all runs, including failures.
   - Evaluate each paired segmentation/regression run once on the untouched test set, including both ground-truth-center and predicted-center regression.
   - If exact hardware or critical provenance is unavailable, classify that branch as blocked by environment or artifact limitations rather than adapting the protocol.

## 6. Diagnose, report, and classify

- Diagnose mismatches through a fixed sequence: acquisition/integrity ? artifact identity ? split ? preprocessing/scales ? configuration ? checkpoint ? raw output ? filtering/matching ? aggregation/CI ? regression location and angular definition.
- Compare events by movie, time, border proximity, orientation relative to z, depth, component score, temporal offset, and crowding.
- Run threshold, tolerance, scale, checkpoint, or metric sensitivity analyses only after the frozen primary result. Label them diagnostic; never use them to replace the headline result.
- Produce an evidence package containing:
  - acquisition logs and manual/automated download manifests;
  - repository, manuscript, data, and checkpoint hashes;
  - claim and artifact ledgers;
  - environment lock, commands, and resolved configurations;
  - raw predictions and TP/FP/FN matching tables;
  - independent and pipeline metrics;
  - regression distributions and center-mode comparisons;
  - retraining logs and seed-level results.
- Assign each claim and pipeline stage:
  - **Fully reproducible:** documented acquisition works, artifacts and provenance are complete, definitions match, checkpoint replay agrees, and required training is concordant without leakage or tuning.
  - **Partially reproducible:** some evidence reproduces, but acquisition, training, provenance, environment, uncertainty, or definitions remain incomplete.
  - **Currently not reproducible:** required artifacts cannot be obtained or verified, the released protocol cannot execute, or frozen results disagree materially.
  - **External/out of scope:** DARE2d-only claims.
- Report numerical agreement separately from methodological validity. No production API or source-code changes are part of the audit.

