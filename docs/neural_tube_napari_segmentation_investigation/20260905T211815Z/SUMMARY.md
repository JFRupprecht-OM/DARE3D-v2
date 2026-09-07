# Neural-tube Napari segmentation investigation

Status: diagnostic round complete; no production changes. End-to-end reproduction of the manuscript F1 from the promoted checkpoint remains unresolved.

## Main findings

- The local manuscript neural-tube/membrane result is 93.4% F1 (114 TP / 8 FP / 8 FN); 91.8% is gastruloid/nuclei. The archived movie_I2 probability map reproduces the exact neural counts with the historical evaluator. This verifies archived-output scoring, not its generating checkpoint.
- Napari omits temporal dilation but retains the times-three weighted-volume denominator. Removing dilation alone from the archived evaluation lowers native IoU F1 from 93.44% to 77.78%. Actual current Napari extraction of the same archived map gives 118 centers versus 122 historical. Some F1 decline reflects temporal matching tolerance, not simply fewer detections.
- Fresh promoted epoch057 inference gives 74 I2 centers and 51 M centers with current Napari settings. With identical historical post-processing, fresh I2 gives 79 TP / 13 FP / 43 FN, F1 73.83%, not 114/8/8. The fresh probability-map discrepancy is upstream of filtering.
- Released epoch057 and its bundled legacy copy are byte-identical, but the checkpoints internally reference a 05-02-26 run. The bundled 12-01-26 config/log/TensorBoard show a different history: January 32 steps/epoch and best val/IoU 0.317997843 at epoch 47; selected checkpoint 63 steps/epoch and best val/IoU 0.291719139 at epoch 57. A folder rename alone does not reconcile this. The archived probability producer checkpoint/config/code combination is not verified.
- movie_I2 has no current scales.json entry, no physical TIFF metadata, and no widget-specific fallback. Uncalibrated loading remains unit-scale display. movie_M resolves authoritative XYZ (0.2076,0.2076,1), fallback (0.2,0.2,1). Shared image/point scaling preserves coordinates; segmenter compatibility geometry (0.621,0.621,2) is NOT physical calibration.

## Full-movie replay

Promoted epoch057, saved compatibility geometry, batch size 4. Actual Napari counts use threshold 0.5 and weighted cutoff 0.1. Historical F1 separately uses threshold >=0.55, weighted cutoff 0.15, temporal dilation +/-1, 4D components, radius-8 volume times three, and greedy IoU >1e-6 matching.

| Movie | Overlap | Actual Napari centers | Historical postproc centers | TP / FP / FN | Historical F1 |
|---|---:|---:|---:|---|---:|
| movie_I2 | 0.25 | 74 | 92 | 79 / 13 / 43 | 73.83% |
| movie_I2 | 0.5 | 74 | 92 | 79 / 13 / 43 | 73.83% |
| movie_M | 0.25 | 51 | 72 | 48 / 24 / 32 | 63.16% |
| movie_M | 0.5 | 55 | 71 | 47 / 24 / 33 | 62.25% |

These are diagnostic results, not revised manuscript results. Archived 93.44% is on movie_I2; the manuscript table names movie_M as its held-out set. Dataset/reference provenance needs reconciliation.

## Same archived map: one-factor checks

| Profile | Dilation | Threshold | Weight cutoff | Centers | TP / FP / FN | F1 |
|---|---|---:|---:|---:|---|---:|
| archived | True | 0.55 | 0.15 | 122 | 114 / 8 / 8 | 93.44% |
| no_dilation_only | False | 0.55 | 0.15 | 112 | 91 / 21 / 31 | 77.78% |
| threshold_only | True | 0.5 | 0.15 | 124 | 114 / 10 / 8 | 92.68% |
| weight_only | True | 0.55 | 0.1 | 131 | 116 / 15 / 6 | 91.70% |
| napari_ge | False | 0.5 | 0.1 | 118 | 95 / 23 / 27 | 79.17% |

The napari_ge row uses >=0.5; the actual API uses >0.5. Actual API extraction was separately tested and returned 118 centers on the archived map. A common center-matching rule (10 raw voxels, |dt|<=1, original annotations) gives archived historical F1 93.06%, archived Napari F1 91.29%, and fresh Napari I2 F1 63.96%. Thus native IoU F1 magnifies temporal-profile differences, but the fresh-inference deficit persists under identical event matching.

## Display and preprocessing

- A separate hidden native Napari viewer showed all 122 archived centers across all time/z slices, but at most 16 on any single slice. Changing shared physical scale preserved visibility and zero coordinate/landmark errors. The live user viewer and its actual selected controls were not inspected or modified.
- Input readers/axis conversions and data-versus-Zenodo inputs match exactly. Increasing overlap 0.25 to 0.5 does not improve I2 counts/F1. CPU rather than GPU resampling on the first sequence has 0.99749 correlation with current GPU output and does not recover archive. The existing audit runtime has zero 0.5-threshold voxel disagreements with current runtime on that sequence. Neither short probe excludes every possible full-movie/runtime difference.
- Replacing segmentation compatibility geometry with physical scaling suppresses all detections in first-sequence probes of both movies. This is not a fix for the remaining 74-versus-118 discrepancy. Compatibility activation is exact-path-specific: manually choosing the same-hash native checkpoint selects source mode.
- Offline and Napari both use causal frames (t-2,t-1,t), with no predictions for the first two frames despite saved channel labels [-1,0,1]. Eligible paired annotations: 123 I2 (122 merged mask objects), 80 M (104 paired annotations across all 11 frames). Comparing 51 detections directly with 104 is not the same recall metric.

## Exact promoted segmenter

Checkpoint: C:\Users\ruppr\Documents\Codexsession\DARE3D_280826\DARE3dv2_Zenodo_040926\DARE3D_neural_tube_segmentation_epoch057.ckpt

SHA-256: 8261c70f41c7f0f5184559e98f3c69ad336e5e46edba5bb57e5edbf9b0568f7d

Hydra config: C:\Users\ruppr\Documents\Codexsession\DARE3D_280826\DARE3dv2_Zenodo_040926\Neural_tube_160226\weights\segmentation3d_new_set_og\runs\12-01-26\.hydra\config.yaml

No regression inference was run. No retrained neural segmentation checkpoint was loaded. The bundled legacy last.ckpt was used only for one diagnostic sequence; it did not recover archived output and was not selected or promoted.

## Verification and preservation

36 existing contract tests passed in the audit environment. The live Napari environment lacks pytest; no environment was changed. The known optional pyarrow trace appeared, but tests exited 0. See focused_tests_audit_environment.json.

All 3835 protected baseline files have unchanged sizes/mtimes. 33 selected checkpoint/config/input/code/evidence SHA-256 values were reverified unchanged. See protection_verification.json and protected_tree_final.json. No tracked changes. Existing untracked release, audit, manuscript, handoff files and grep.exe.stackdump were preserved. Only this generated diagnostic directory and an isolated child of .pytest_tmp were added. No commit/push.

## Resume / unresolved questions

1. Obtain genuine 05-02-26 neural segmentation Hydra/config/log/scale provenance and/or the checkpoint that actually produced archived movie_I2.tif; preserve all existing assets.
2. Compare that provenance/config/network to promoted epoch057; replay only the first eligible I2 sequence against the saved archive before another full inference. Do not tune thresholds/scales to manufacture a manuscript match.
3. Once the historical producer is established, rerun fixed full-movie historical evaluation and the same event-identical center metric on I2 and M, changing exactly one verified factor at a time.
4. Separately establish authoritative I2 voxel calibration and the live widget checkpoint/dataset/image name/time window, total count and suspect t/z slice. Never treat (0.621,0.621,2) as physical calibration.
5. Any production change to post-processing/display or release checkpoint/config pairing requires a separate explicit implementation decision.

## Evidence

diagnostic_summary.json indexes conclusions, metrics and next steps. executed_diag_*.json preserves the main diagnostic sources for replay into a new output directory. inventory.json records initial versions/hashes/Git state. Per-profile JSON/TIFF files retain fresh outputs; archived files remain untouched. paired_annotation_center_metrics.json records matched/missed event identities. cpu_preprocessing.json is a preserved incomplete metadata write; use cpu_gpu_preprocessing_complete.json instead.
