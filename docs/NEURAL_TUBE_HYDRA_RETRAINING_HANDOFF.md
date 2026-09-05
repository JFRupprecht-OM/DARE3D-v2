# Neural-tube Hydra retraining and promotion handoff

Recorded: 2026-09-05 (Europe/Paris)

## Outcome

The regression-only neural-tube Hydra retraining, paired evaluation, and Zenodo staging promotion are complete. The new candidate passed every original locked release criterion, so promotion is a normal scientific pass. The user-authorized `<= +3.0` degree override was not needed and no override record was created.

No retraining or expensive evaluation should be rerun merely to reproduce this handoff.

## The three model roles

- Historical/legacy regression provenance, preserved and not used as the acceptance reference: `DARE3d_data_190326/Neural_tube_160226/weights/regression3d_new_set_og/`; checkpoint `runs/12-01-26/checkpoints/epoch_147.ckpt`, SHA-256 `1457585655bc4e11404e10022e87d2049e8408d201f676b5c7c1546290483a6b`.
- Immutable validated scientific reference: `docs/reproducibility_audit/neural_tube_nematic_full_pipeline/seed_12345/regression_nematic_retrained/checkpoints/epoch_139.ckpt`, SHA-256 `192c2bec48e8c6dc9b78740254a55082b89416e3b986bf243edf60586530483b`.
- Fresh Hydra candidate and retained training/evaluation source: `DARE3d_data_190326/Neural_tube_160226/weights/regression3d_nematic_hydra_seed12345/`; selected checkpoint `checkpoints/epoch_173.ckpt`, SHA-256 `21e09e08b77800de754acbde12866e8e445b29ed3acd326eff4fde8b82b56a0b`.

The validated `epoch_139.ckpt` audit is a comparator only. It was not copied into, overwritten by, or repurposed as the new Hydra run.

## Training and selection

Run ID: `20260905T121613Z`.

The repository-native Hydra run completed all 200 planned epochs. It used the legacy neural workflow for neural-specific geometry and architecture, and the completed gastruloid workflow for corrected nematic, reproducibility, evaluation, and release machinery:

- Train: `movie_E`; validation and checkpoint selection: `movie_I2`; independent paired test: `movie_M`.
- Verified XYZ scale `0.208,0.208,1`, target scale `1`, temporal channels `[-1,0,1]`, min-max normalization, and `32^3` crops.
- 222 unique train crops and 123 validation crops; 2,000 sampled items per epoch; batch size 32.
- Three-stage RegressionNet with 32 starting filters, 1,605,268 parameters, and 41 checkpoint network keys.
- Corrected nematic-axis objective, AdamW at `0.001`, historical OneCycle epoch scheduling, FP32 CUDA, seed 12345.
- Selection was fixed in advance as the single minimum-`val/loss` checkpoint; `movie_M` was not used for selection.

Hydra selected `epoch_173.ckpt` at global step 10,962. Native restored-checkpoint test loss was `7.236946105957031`. The selected checkpoint is 18,653,046 bytes.

## Paired, event-identical evaluation

Exactly one full downstream evaluation was run after training, comparing the fresh candidate and immutable `epoch_139.ckpt` on the same 80 unique `movie_M` events. Errors are corrected nematic-axis degrees; length is mean absolute error in regression voxels.

| Scale profile | Model | Mean | Median | p95 | Length MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| Audit `0.208,0.208,1` | Candidate | 17.78586 | 15.44728 | 39.72208 | 0.72804 |
| Audit `0.208,0.208,1` | Reference | 18.94547 | 17.12961 | 40.24515 | 0.83589 |
| Physical `0.2076,0.2076,1` | Candidate | 17.82294 | 15.84055 | 35.47125 | 0.74744 |
| Physical `0.2076,0.2076,1` | Reference | 19.54672 | 16.44618 | 45.80245 | 0.83549 |

Paired candidate-minus-reference mean differences and locked 10,000-replicate bootstrap intervals were:

- Audit profile: `-1.15961` degrees, 95% CI `[-2.61478, +0.25699]`.
- Exact physical profile: `-1.72378` degrees, 95% CI `[-3.25682, -0.25898]`.

Negative deltas favor the candidate. Every original locked assertion passed; the recorded decision is `normal_pass`, `locked_criteria_passed: true`, `promotion_authorized: true`, and `explicit_release_override_used: false`.

## Frozen-center evaluation

The evaluator did not rerun segmentation. It replayed the existing frozen `movie_I2` center groups in `training_consistent` mode.

| Frozen center group | n | Candidate mean/median/p95 | Reference mean/median/p95 | Candidate/reference length MAE |
| --- | ---: | --- | --- | --- |
| All ground-truth | 122 | 12.83007 / 10.95510 / 31.63698 | 14.20222 / 12.44475 / 27.63420 | 0.67710 / 0.63774 |
| Matched ground-truth | 114 | 12.48927 / 10.61385 / 30.34433 | 14.27642 / 12.60214 / 27.59697 | 0.65417 / 0.60124 |
| Predicted centers | 114 | 13.25841 / 11.15671 / 28.92807 | 14.34748 / 12.82084 / 25.67144 | 0.69512 / 0.64878 |

The candidate improved mean nematic error in all three comparisons. All frozen-center, p95, length, finite/range, training-contract, native-artifact, checkpoint-load, and protected-asset checks passed.

Machine-readable scientific results remain in the source run at `evaluation/result.json`; event-identical rows and predictions remain under `evaluation/`.

## Promotion

The source training run was retained unchanged as the training/evaluation/provenance authority. Its full structured contents were copied into the new Zenodo staging directory:

`DARE3dv2_Zenodo_040926/Neural_tube_160226/weights/regression3d_nematic_hydra_seed12345/`

The historical release-style alias is:

`DARE3dv2_Zenodo_040926/Neural_tube_160226/weights/regression3d_nematic_hydra_seed12345/checkpoints/DARE3D_neural_tube_regression_epoch173.ckpt`

The source `epoch_173.ckpt`, same-name destination copy, and release alias are all 18,653,046 bytes and all have SHA-256:

`21e09e08b77800de754acbde12866e8e445b29ed3acd326eff4fde8b82b56a0b`

All three are byte-identical. The structured-copy check covered 1,306 source files. The final Zenodo comparison found 1,308 additions, all confined to the new neural candidate directory, with no removed or size/timestamp-changed pre-existing Zenodo file. The release smoke test loaded 41 network keys and produced finite angle `[1,9]` and length `[1,1]` outputs from an input `[1,3,32,32,32]`.

The authoritative promotion record is `DARE3dv2_Zenodo_040926/Neural_tube_160226/weights/regression3d_nematic_hydra_seed12345/provenance/release_promotion.json`.

## Safe recovery history

The first evaluator attempt stopped before checkpoint inference because a vector `target_scale` was converted as a scalar; commit `195d355` corrected only that provenance serialization.

The second attempt completed the sole full scientific evaluation and wrote its immutable result, then stopped after the structured copy and alias were created because the smoke test assumed a Torch tensor where the dataset returned a NumPy array. It did not invalidate the evaluation or checkpoint.

Commit `8d02131` converts the smoke input explicitly with `torch.as_tensor` and adds a resume-only path. That path refused overwrite, verified the completed result and selected checkpoint hash, verified the existing partial structured copy byte-for-byte, ran only the inference smoke/protection checks, and wrote the final promotion record. It did not rerun training or scientific evaluation.

## Protection result

The recorded post-promotion assertions confirm:

- all six genuine neural input TIFFs remain byte-identical;
- all nine protected trees remain byte-identical, including the legacy regression workflow and validated `epoch_139` audit;
- all four pre-existing release-root checkpoints remain byte-identical;
- no segmentation checkpoint was loaded, evaluated, copied, moved, or modified;
- no gastruloid asset, Napari configuration, unrelated model, or pre-existing Zenodo file was changed.

The source-run preflight baseline is `DARE3d_data_190326/Neural_tube_160226/training_splits/regression3d_nematic_seed12345/protected_before.json`.

## Repository changes and verification

Committed implementation:

- `647e3a1` ? `configs/experiment/neural_tube_nematic_regression.yaml`, `scripts/prepare_neural_tube_nematic_hydra.py`, `scripts/evaluate_neural_tube_nematic_hydra.py`, and `tests/test_neural_tube_nematic_hydra.py`.
- `195d355` ? scalar/vector scale-provenance compatibility in the neural evaluator.
- `8d02131` ? safe resume-only promotion finalization and NumPy-to-Torch smoke conversion.

Generated task directories:

- Verified split and protection staging: `DARE3d_data_190326/Neural_tube_160226/training_splits/regression3d_nematic_seed12345/`.
- Retained native Hydra source run, evaluation, and provenance: `DARE3d_data_190326/Neural_tube_160226/weights/regression3d_nematic_hydra_seed12345/`.
- New promoted Zenodo model: `DARE3dv2_Zenodo_040926/Neural_tube_160226/weights/regression3d_nematic_hydra_seed12345/`.

Verification completed:

- Python byte-compilation passed.
- Focused permanent suite passed: 12 tests in `tests/test_neural_tube_nematic_hydra.py` and `tests/test_gastruloid_nematic_hydra.py`, using only the repository `.pytest_tmp` directory.
- A known optional `pyarrow` Windows access-violation trace appeared during import, but pytest completed with exit code 0.
- Promotion inference smoke, source/destination full-tree equality, protected-tree verification, and Zenodo-scope verification all passed.

Pre-existing unrelated untracked audit, manuscript, Zenodo, and handoff material remains preserved. Do not clean, reset, move, or add it indiscriminately.
