# DARE3D Scientific Reproducibility Audit Report

Finalized: 2026-08-29 (Europe/Paris)

Source under audit: commit fe2b14d732359f2bdaf8b197574ad818899ce123

Manuscript under audit: manuscript_280826version/main.tex and its hash-frozen companion files

Released artifacts: Zenodo record 19113351, DARE3d_data_190326.zip

## Overall assessment

**The entire current DARE3D pipeline is currently not reproducible from the released materials.**

This is a strict end-to-end judgment. It does not mean that the release lacks useful reproducibility evidence. The nuclei result has unusually strong partial support: current-code inference with the validation-selected segmentation checkpoint epoch_067 exactly reproduces the archived object counts and F1 when the missing geometry is reconstructed from the training log, and regression checkpoint epoch_098 reproduces all 21 archived aggregate statistics within 0.02. The saved membrane probability also reconstructs the expected 122 retained prediction components.

Full status is nevertheless precluded by several independent issues:

- Raw neural-tube images and daughter-cell annotations are absent, so membrane matching, checkpoint inference, and training cannot be rerun.
- The scales.json required by every released configuration is absent. Its public fallback does not reproduce the nuclei result.
- Validation and test point to the same data in all four saved configurations, so the reported results are not evidence from an untouched test set.
- The released detection evaluator does not implement the manuscript's stated one-to-one 10-voxel, +/-1-frame center matcher.
- The reported 3D orientation statistic uses full-quaternion error rather than a nematic division-axis error; the manuscript's 28 +/- 2 wording does not describe the stored statistics correctly.
- No historical source snapshot or complete software/hardware environment accompanies the release.
- Exact four-model historical retraining cannot be specified without inventing missing provenance or changing the protocol. A separately labeled nuclei-only nematic-loss experiment is reported below using log-reconstructed geometry.

The claim ledger classifies 17 individual subclaims as fully reproducible, 19 as partially reproducible, and 9 as currently not reproducible. The fully reproducible rows are dataset arithmetic, analytic baselines, uncertainty simulations, or configuration-level protocol facts; they do not override the strict whole-pipeline verdict.

## Classification by audit stage

| Stage | Classification | Strongest evidence | Principal limitation |
|---|---|---|---|
| Data acquisition | Artifacts available but guide broken | Verified 7,048,412,887-byte archive; published MD5 and independent SHA-256 match; 278 extracted files | Windows completion exception, no real resume, size-only reuse, wrong neural paths, and multiple inconsistent download paths |
| Nuclei dataset statistics | Fully reproducible | Movie2 shape, 10 frames, 226 daughter pairs, length 15.32796 and SEM 0.17039 recomputed | This does not validate the evaluation split |
| Membrane test-dataset statistics | Currently not reproducible | Manuscript values are inventoried | Raw Dataset 2 set 3 is absent; released movie_I2 has 21 frames and aligns with set 2 rather than the marked 11-frame set 3 |
| Nuclei saved-prediction evaluation | Partially reproducible | Current evaluator exactly replays 124 TP, 1 FP, 21 FN, F1 0.9185185 | Stated matcher gives 121/4/24; validation=test |
| Nuclei checkpoint inference | Partially reproducible | epoch_067 exactly reproduces object metrics and is nearly identical to archived probability | Original scale provenance and historical environment absent; output not bitwise |
| Membrane saved-prediction evaluation | Partially reproducible | 197 threshold components, 75 removed, 122 retained = 114 TP + 8 FP | Raw ground truth absent, so match assignment cannot be independently rerun |
| Membrane checkpoint inference | Currently not reproducible | Best/last checkpoint metadata and derived outputs exist | Raw inputs/annotations absent and checkpoint/config run identity conflicts |
| Nuclei regression checkpoint | Partially reproducible | epoch_098 reproduces 21 aggregate values within 0.02 | Historical raw predictions absent; centers reconstructed; metric definition scientifically mismatched |
| Membrane regression | Partially reproducible at saved-artifact level; checkpoint replay currently not reproducible | Three NPZ prediction sets and raster outputs permit approximate reconstruction | Raw input and ground truth absent; rendered truth loses quaternion information |
| Full retraining | Currently not reproducible | Configs, checkpoints, and logs inventoried | Missing data/scales/environment, nondeterminism, and no distinct test set |
| Controlled nuclei nematic retraining | Experimental comparison complete | Fresh 100-epoch run; paired corrected-axis mean improves 2.319 degrees in all-GT mode | Validation=test, one seed, reconstructed scale, and no historical environment |
| Public plugin/CLI inference | Executable but not a manuscript reproduction path | Real three-frame GPU segmentation smoke completed with 13 detections | Public defaults differ in checkpoint, overlap, thresholds, scale fallback, and temporal post-processing |

## Headline quantitative results

### Nuclei detection

The archived probability map, evaluated with the current evaluator at probability threshold 0.55 and weighted cutoff 0.15, exactly reproduces:

- TP = 124
- FP = 1
- FN = 21
- F1 = 0.9185185185
- logit multinomial 95% interval = 0.8774999 to 0.9466375, which rounds to the manuscript display

The current-code epoch_067 checkpoint with the independently log-reconstructed scale [0.912, 0.912, 0.912] also reproduces all object metrics exactly. Its probability is extremely close to the archive (Pearson r = 0.9999998905, MAE = 4.512e-7, 92 threshold differences among 198,738,000 voxels), but it is not bitwise identical.

The predeclared last.ckpt comparison gives 123 TP, 4 FP, 22 FN, and F1 0.9044118. The predeclared epoch_067 run using the public missing-scale fallback [0.621, 0.621, 2] gives only 32 TP, 5 FP, 113 FN, and F1 0.3516484. This strongly identifies epoch_067 as the archived checkpoint and demonstrates that the absent scale file is scientifically consequential.

This result remains **partially reproducible**, because the match definition and test protocol are not the manuscript protocol. An independent maximum-cardinality implementation of the stated +/-1-frame and 10-raw-voxel rule gives 121 TP, 4 FP, 24 FN, and F1 0.8962963. Five matches needed a 16-voxel tolerance in the independent center reconstruction, and the current evaluator instead uses temporal dilation, 4D connected components, near-zero IoU eligibility, and greedy assignment.

### Membrane detection

The released counts yield exactly F1 = 0.9344262295 and a logit multinomial interval of 0.8941177 to 0.9600748, reproducing the displayed 93.4% and 89.5--96.0% after rounding conventions.

Independent component reconstruction from the saved probability finds 197 components before weighted filtering, removes 75, and retains 122, exactly matching 114 TP + 8 FP. All 114 matched markers and seven FP markers are visible. The eighth FP is a real retained component whose rounded z position causes an un-clipped 9:7 marker slice, so it is invisible in both saved renderings.

A separate implementation defect also affects the result: the first foreground component in T,X,Y,Z memory order bypasses weighted filtering. In this membrane output, a visible FP with weighted score 0.008736 survives despite the cutoff 0.15.

The membrane claim is **partially reproducible from saved outputs**, but checkpoint-to-result and independent matching are **currently not reproducible** because the raw neural images and annotations are absent. The released 21-frame movie also conflicts with the manuscript's bold 11-frame test set.

### Orientation and length regression

For nuclei, epoch_098 loads strictly and reproduces all 21 archived aggregate values within 0.02 on frozen centers reconstructed from the released segmentation probability. The largest absolute discrepancy is 0.0159645. last.ckpt is materially different and has a maximum discrepancy of 1.47155.

The numerical trace does not validate the manuscript wording:

- Nuclei released all-ground-truth mean angular error is 25.30595 degrees, while its population SD is 28.01793 degrees.
- Membrane released all-ground-truth mean angular error is 49.56607 degrees, while its population SD is 27.10411 degrees.
- The manuscript's central value 28 follows the SD, not the mean.
- The approximately 2-degree term follows that SD divided by the nominal square-root sample count.
- For nuclei, the evaluator reports nominal n values 145/124/124 but actually evaluates 140/121/121 after silently skipping missing pairs.
- Only predicted_centers is end-to-end. Its released full-quaternion mean is about 37.47 degrees for nuclei and 51.37 degrees for membrane, rather than the all-ground-truth headline values.

The production metric is 2 arccos of the absolute quaternion dot product, clipped before evaluation. It is not the scientifically natural nematic-axis error arccos of the absolute unit-axis dot product. Synthetic checks show that identical rendered nematic axes can receive 60- or 120-degree quaternion errors, while even identical quaternions receive a 1.620583-degree floor. The unused rotation around the division axis therefore affects the published number.

Consequently, both 28 +/- 2 claims are **partially reproducible numerically but methodologically invalid as written**.

## Scientific-definition findings

### Train, validation, and test separation

The nuclei configuration trains on movies 3 and 4 and uses movie2 for both validation and test. The membrane configuration trains on movie_E and uses movie_I2 for both validation and test. Threshold selection can therefore use the same labels later reported as test performance. The manuscript presents both as test results.

The nuclei test image is byte-identical to its copy within the released training-set tree. The configured split indicates that it was not one of the two nuclei training movies, but its duplicate placement and validation=test configuration make provenance less clear to a new user.

No reported detection result can receive fully reproducible scientific status without a distinct, hash-identified test set that was not used for checkpoint or threshold selection.

### TP, FP, FN, and matching

The manuscript specifies one-to-one center matching within 10 voxels and +/-1 frame. The current code temporally dilates predictions, labels 4D components, permits matches at IoU above 1e-6, and uses greedy nearest-first assignment. Frozen synthetic cases establish that:

- temporal dilation can merge two nearby predictions into one event;
- adjacent ground-truth events can merge into one 4D object;
- greedy matching can return one match where maximum-cardinality matching returns two;
- a spatial distance exactly equal to the threshold is rejected;
- both-empty movies receive precision, recall, and F1 of zero;
- the first predicted component bypasses weighted filtering;
- evaluate.py and predict.py disagree at threshold equality;
- marker rendering fails at low borders because negative slices are not clipped.

These are definition-level differences, not harmless numerical tolerances.

### F1 aggregation and confidence interval

F1 arithmetic exactly matches 2TP/(2TP+FP+FN). When multiple movies are evaluated, the code pools counts and therefore emits micro-F1; it does not emit an unweighted per-movie macro-F1. A frozen example gives micro-F1 0.9 versus macro-F1 0.473684, showing that the choice can be material. The two headline 3D results each use one released evaluation movie, but the aggregation rule should still be stated explicitly.

The cited confidence intervals are reproducible with a logit multinomial delta calculation from TP/FP/FN. This verifies the reported arithmetic conditional on the event counts; it does not cure split leakage, correlated events within one movie, or mismatched event definitions.

### Voxel anisotropy and coordinate systems

All four configs request target_scale = 1.0 but depend on an absent scales.json. The archived nuclei training log records geometry consistent with [0.912, 0.912, 0.912], while the public fallback is [0.621, 0.621, 2]. The latter collapses nuclei checkpoint F1 from 0.9185 to 0.3516.

Regression evaluation initializes preprocessing as false, pads and min-max normalizes, and does not restore training-time spatial resampling. Regress3Dataset.unscale_prediction is a stub. The membrane manuscript discussion uses anisotropic scaling around (0.2, 0.2, 1) um. It is therefore not possible to establish a single, provenance-backed physical coordinate system for segmentation, matching, crops, lengths, and 3D axes.

A synthetic anisotropy check confirms the scientific risk: for an illustrative spacing, a raw-voxel angular error of 11.31 degrees becomes 45 degrees in physical coordinates. This example demonstrates sensitivity; it is not asserted to be the missing DARE3D scale.

## Data acquisition and public workflow

The archive was downloaded and verified:

- size: 7,048,412,887 bytes
- MD5: f8df147bb8249aad44b8ef1509d280dd
- SHA-256: 7eac84aa75d12cc8431d31d55b707823b4eaf986335207d3fa3f4fbb1fd64004
- extracted files: 278

The acquisition classification is **Artifacts available but guide broken**. Extraction succeeds through the documented CLI, but the final Unicode-arrow print raises a cp1252 error on this native Windows host. The claimed resume behavior truncates the partial file, existing archives are accepted by size without checksum, and --help is interpreted as a destination. Documented Gastruloid weight paths exist; documented neural weights paths do not. The plugin can resolve the actual dated neural roots, whereas the CLI requires explicit run directories. The widget has a source-relative fallback one parent too high and stale import-time defaults after same-session download. The prediction notebook uses another package-local downloader rather than the verified root bundle.

A real headless three-frame segmentation-only plugin smoke ran successfully on the GPU. This proves public-path executability, not manuscript reproduction: the public path defaults to last.ckpt, overlap 0.25, probability >0.5, weighted cutoff 0.1, no temporal dilation, and the missing-scale fallback. The released nuclei result instead used epoch_067, overlap 0.5, probability >=0.55, weighted cutoff 0.15, temporal dilation, and log-reconstructed geometry.

## Environment and test status

The current audit environment is captured in full in the environment ledger and 269-package inventory. Its principal versions are Python 3.10.20, PyTorch 2.5.1+cu121, MONAI 1.3.0, Lightning 2.6.5, NumPy 1.23.4, SciPy 1.13.1, scikit-image 0.24.0, tifffile 2025.5.10, Hydra 1.3.2, and OmegaConf 2.3.1. GPU execution used a Quadro RTX 5000 with 16 GB, driver 581.95, CUDA 12.1, and cuDNN 9.1.

Released checkpoints record Lightning 2.2.5 for nuclei and 2.5.6 for membrane. All saved training configurations set deterministic=false. The current environment's editable dare3d install points to an older checkout; all audit scripts that import DARE3D prepend this repository and verify their import origins.

Focused geometry and training-command tests pass, as do five checkpoint-loading tests. A larger config/evaluation selection gives 5 passed, 1 failed, and 2 errors because of stale Hydra fixtures. Full pytest collection is blocked by an undeclared pkg_resources import. No production fixes were made.

All frozen audit suites pass:

- 24/24 scientific edge-case assertions
- 19/19 regression artifact assertions
- 9/9 nuclei segmentation checkpoint summary assertions
- 9/9 nuclei regression checkpoint summary assertions
- 12/12 membrane component assertions
- 15/15 workflow/path assertions
- 14/14 nematic-axis metric comparison assertions
- 10/10 nematic-retraining preflight assertions
- 10/10 paired nematic-retraining evaluation assertions

## Nematic-axis angular-metric counterfactual

An independent audit-only evaluator changed only the angular definition while
holding predictions, ground truth, splits, filtering, center modes, matching,
and aggregation fixed. Production code and released artifacts were not edited.

- Current all-GT mean/SD: nuclei 25.306/28.018 degrees; membrane
  49.566/27.104 degrees.
- Corrected nematic-axis all-GT mean/SD: nuclei 14.894/12.700 degrees
  (effective N=140); membrane 36.634/22.126 degrees (N=122).
- Corrected predicted-center mean/SD: nuclei 19.672/21.789 degrees
  (effective N=121); membrane 36.032/19.965 degrees (N=114).

The manuscript's two 28 +/- 2 degree statements trace to the released
population SDs rather than the released means. Coherent corrected mean +/- SEM
values are 14.89 +/- 1.07 degrees for nuclei and 36.63 +/- 2.00 degrees for
membrane. Identical and opposite nematic axes return exactly 0 degrees,
orthogonal axes return 90 degrees, and changing only rotation about an axis
leaves the corrected error unchanged. All 14/14 audit assertions pass.

The nuclei annotation-scale conclusion is strengthened, but the membrane
approximately-20-degree claim is not supported. Both corrected all-GT RMS
values remain below the manuscript's 61.2-degree random-axis baseline.
See [the full comparison](NEMATIC_AXIS_METRIC_COMPARISON.md).

## Controlled nuclei nematic-loss retraining

After the fixed-prediction audit, a separately scoped experiment retrained the
nuclei regression model from scratch. It preserved the released 9D/SVD
architecture, train/validation/test assignment, augmentations, optimizer,
scheduler behavior, length loss, epoch count, and checkpoint rule. Only the
orientation objective was replaced by the smooth nematic projector loss
`90*(1-(u_pred dot u_true)^2)`. Exact evaluation used
`acos(abs(u_pred dot u_true))` in degrees.

On the manuscript-compatible all-GT set (effective N=140):

- Original model + original metric: mean/median/SD
  25.307/17.735/28.019 degrees.
- Original model + corrected metric: 14.894/12.342/12.700 degrees.
- Retrained model + corrected metric: 12.575/9.409/11.126 degrees.

Retraining therefore changes the corrected all-GT mean by -2.319 degrees
(-15.6%; paired descriptive bootstrap interval -4.406 to -0.457). On 156
unique reconstructed-preprocessing crops, the change is -2.762 degrees
(-20.2%; interval -4.576 to -1.162). The predicted-center end-to-end change is
only -0.225 degrees with an interval spanning zero. Detection and center metrics
are unchanged, and all length-MAE intervals span zero.

The 100-epoch run, epoch-95 best checkpoint, full logs, event tables,
distributions, predictions, hashes, and 10/10 assertions are saved separately.
Production source and released weights remain unchanged. See
[the full retraining comparison](NEMATIC_RETRAINING_COMPARISON.md).

## Exact full retraining remains blocked

This experiment is not exact historical or full-pipeline retraining. The
original `scales.json` and environment are absent, the logged target geometry
had to be reconstructed, released movie2 is both validation and test, only one
seed was run, and the membrane raw data remain absent. The strict audit plan's
prerequisites are therefore still unmet. To make exact full retraining auditable,
a future release would need:

1. Raw Dataset 2 set 3 images and daughter annotations, plus an unambiguous mapping from manuscript set identifiers to released files.
2. The original scales.json and a coordinate/axis-order specification for every case.
3. Distinct hash-identified training, validation, and untouched test data.
4. The exact historical source snapshot and a complete lockfile or container with CUDA/cuDNN metadata.
5. The checkpoint-selection rule and serialized thresholds chosen only on validation.
6. Event-level historical predictions, matches, and regression arrays for both cases.
7. A corrected scientific specification for center matching, micro versus macro F1, nematic 3D axis error, uncertainty, and predicted-center end-to-end evaluation.
8. A published seed and the planned additional seeds needed to quantify nondeterministic training variability.

## Final classification

The numerical evidence supports these conclusions:

- **Fully reproducible subclaims:** nuclei dataset dimensions/counts/length summary; F1 and CI arithmetic; stated random baselines and annotation-uncertainty simulations; several architecture/crop configuration facts.
- **Partially reproducible:** archived nuclei detection and regression aggregates; saved membrane components and aggregate arithmetic; several configuration claims with incomplete provenance.
- **Currently not reproducible:** raw membrane dataset claims, stated detection matching protocol, several manuscript training/preprocessing claims, membrane checkpoint replay, bitwise historical inference, and exact four-model historical retraining.
- **Entire current pipeline:** **currently not reproducible**.

Numerical agreement and methodological validity are deliberately reported separately. No parameters were changed to force agreement, no production source was modified, and no released data, checkpoint, or result artifact was altered.

## Evidence index

- [Audit package summary](evidence/audit_package_summary.json)
- [Claim ledger](evidence/claim_ledger.csv)
- [Artifact ledger](evidence/artifact_ledger.csv)
- [Acquisition report](evidence/acquisition_report.json)
- [Acquisition ledger](evidence/acquisition_ledger.csv)
- [Command ledger](evidence/command_ledger.csv)
- [Test ledger](evidence/test_ledger.csv)
- [Environment report](evidence/environment_report.json)
- [Environment package inventory](evidence/environment_packages.csv)
- [Nuclei saved-probability replay](evidence/nuclei_saved_probability_replay.json)
- [Nuclei nematic-loss retraining comparison](NEMATIC_RETRAINING_COMPARISON.md)
- [Nematic retraining machine evidence](nematic_retraining/seed_12345/evaluation.json)
- [Nuclei segmentation checkpoint comparison](evidence/nuclei_checkpoint_compatibility_summary.json)
- [Nuclei regression checkpoint comparison](evidence/nuclei_regression_checkpoint_compatibility_summary.json)
- [Regression artifact reconstruction](evidence/regression_artifact_analysis.json)
- [Nematic-axis metric comparison](NEMATIC_AXIS_METRIC_COMPARISON.md)
- [Membrane component investigation](evidence/membrane_fp_investigation.json)
- [Scientific edge cases](evidence/scientific_edge_cases.json)
- [Workflow/path checks](evidence/workflow_path_checks.json)
- [Final audit-file SHA-256 manifest](evidence/audit_evidence_manifest_sha256.csv)
