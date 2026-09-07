# Exact-scale regression replay

Status: **complete**. This was checkpoint inference/evaluation only. No model was trained, no weights were changed, and segmentation was not rerun.

## Corrected nematic-axis angular results

All SD values are population SD. Deltas are exact-table minus the corresponding prior approximate-scale result.

| Dataset | Model | Center mode | N | Prior mean/median/SD | Exact mean/median/SD | Exact - prior mean/median/SD |
|---|---|---|---:|---:|---:|---:|
| neural_movie_M | original | annotation_crops | 80 | 53.815941/53.434120/22.902840 | 53.441628/53.678970/22.763472 | -0.374313/+0.244850/-0.139368 |
| neural_movie_M | retrained | annotation_crops | 80 | 18.945465/17.129607/11.650493 | 19.546722/16.446177/11.775638 | +0.601257/-0.683430/+0.125145 |
| nuclei_movie2 | original_epoch_098 | all_groundtruth_centers | 140 | 13.670714/10.982744/11.443938 | 11.893218/9.534827/8.378367 | -1.777496/-1.447917/-3.065571 |
| nuclei_movie2 | original_epoch_098 | matched_groundtruth_centers | 121 | 12.840923/10.676945/10.864014 | 11.587284/9.284823/9.284184 | -1.253639/-1.392122/-1.579830 |
| nuclei_movie2 | original_epoch_098 | predicted_centers | 121 | 19.740106/10.977146/21.088224 | 20.133460/12.082413/20.941173 | +0.393354/+1.105267/-0.147052 |
| nuclei_movie2 | retrained_nematic_epoch_095 | all_groundtruth_centers | 140 | 11.736370/9.658257/9.831211 | 10.475700/8.192762/7.999417 | -1.260670/-1.465495/-1.831794 |
| nuclei_movie2 | retrained_nematic_epoch_095 | matched_groundtruth_centers | 121 | 11.010687/9.284841/7.829808 | 10.019182/7.981328/7.214265 | -0.991505/-1.303513/-0.615544 |
| nuclei_movie2 | retrained_nematic_epoch_095 | predicted_centers | 121 | 19.806179/10.038148/23.102825 | 18.915743/9.877305/21.451786 | -0.890437/-0.160842/-1.651039 |

## What changed

- movie_M 0.208 -> 0.2076: 66/104 annotation endpoint pairs and 47/104 derived centers change; among the 80 eligible test events, 47 endpoint pairs and 30 crop centers change. Exactly 30/80 predictions change for each checkpoint, while the other 50 are bit-identical.
- movie2 prior-validator realized shape ratios -> 0.914: 211/226 annotation endpoint pairs and 172/226 derived training-crop centers change; among 156 eligible annotations, 147 endpoint pairs and 126 centers change. On the actual frozen inference centers, changed crops are 107/145 all-GT, 90/124 matched-GT, and 98/124 predicted-center; exactly the same number of predictions changes for each checkpoint, and every unchanged-crop prediction is bit-identical.
- Provenance reconciliation: the scale audit's earlier 194/226 endpoint and 149/226 center counts remain correct for its nominal isotropic 0.912 diagnostic. They are not the comparison baseline for the prior corrected validator, which used the per-axis realized shape ratios recorded in its manifest.
- All integer crop-center shifts are at most one regression-grid voxel. The resampled image target shapes do not change.

## Scientific interpretation

- On exact movie_M geometry, retraining changes the corrected mean by -33.894905 degrees (95% descriptive paired bootstrap interval [-39.21842632749991, -28.505126217040814]). The strong independent-test improvement remains.
- At exact movie2 predicted centers, retraining changes the mean by -1.217717 degrees (interval [-3.186504545240581, 0.790667660884206]). The interval still spans zero, so the prior conclusion of no material predicted-center improvement is unchanged.
- Exact scales refine the reported numbers but do not alter the established regression preprocessing mismatch, the training_consistent correction, target association, checkpoint integrity, or the unavailable movie_M end-to-end result.
- movie_I2 was not assigned movie_M's 0.2076 value: it has no entry in the authoritative table. Its prior explicitly reviewed 0.208 profile remains a separate result, as does the released original checkpoint's native 0.621/0.621/2 profile.

## Protocol and reproducibility

- Source table: `data/3D/scales.json`; movie2 `[0.914, 0.914, 0.914]`, movie_M `[0.2076, 0.2076, 1.0]` in XYZ.
- Preprocessing: `training_consistent` only. `legacy_raw` was not rerun or mixed into these results.
- Metric: `acos(abs(dot(unit predicted axis, unit true axis)))` in degrees.
- Nuclei detector-predicted centers select crops only. Ground truth remains the annotation bound to the matched true component.
- Canonical command: `python docs/reproducibility_audit/exact_scale_regression_replay.py --phase all`
- Executed with `C:\Users\ruppr\.conda\envs\dare3d-v2.0\python.exe` (Python 3.10.20, PyTorch 2.5.1+cu121, CUDA 12.1, Quadro RTX 5000).
- Checkpoints: `DARE3d_data_190326/Neural_tube_160226/weights/regression3d_new_set_og/runs/12-01-26/checkpoints/epoch_147.ckpt` (SHA-256 `1457585655bc4e11404e10022e87d2049e8408d201f676b5c7c1546290483a6b`); `docs/reproducibility_audit/neural_tube_nematic_full_pipeline/seed_12345/regression_nematic_retrained/checkpoints/epoch_139.ckpt` (SHA-256 `192c2bec48e8c6dc9b78740254a55082b89416e3b986bf243edf60586530483b`); `DARE3d_data_190326/Gastruloid_241025/weights/regression3d_exp10-b/checkpoints/epoch_098.ckpt` (SHA-256 `3c453af4af568456cfb8c6b362e8f60c953f246d194cd060229faf5d84044682`); `docs/reproducibility_audit/nematic_retraining/seed_12345/checkpoints/epoch_095.ckpt` (SHA-256 `e3bc5a3ff81497d36606ee4ca57c83395eef5f02b0ffd3cf5f117d0bb1f51ec8`).
- Datasets: `DARE3d_data_190326/Neural_tube_160226/test_input/im/movie_M.tif` (SHA-256 `91f0fd8cd0e14256b51da3247e521d51013c786f792c624c83598dbd827e8c88`); `DARE3d_data_190326/Neural_tube_160226/test_input/label/movie_M.tif` (SHA-256 `d239ac66a09aa1b22e96b4e69a20a1a758dced35d3d818d79c7b38623eacd6df`); `DARE3d_data_190326/Gastruloid_241025/trainingset/movie2/im/movie2.tif` (SHA-256 `b57ef33081e627a9305389b3dff796d9ca8e718e7ead2c4eaee7940733681f81`); `DARE3d_data_190326/Gastruloid_241025/trainingset/movie2/label/movie2.tif` (SHA-256 `d079c16070a378026d404f2bf101d017aaae2d76fa836dfbb16e981e88c03456`).
- The machine-readable `result.json` records exact dataset/checkpoint/table paths, sizes, SHA-256 hashes, prior-result identities, assertions, paired effects, and output hashes.
- `nuclei_result_raw.json` is direct adapter evidence and retains the validator's historical profile-key text; its manifest records exact 0.914. This report and consolidated `result.json` are authoritative for interpretation.
