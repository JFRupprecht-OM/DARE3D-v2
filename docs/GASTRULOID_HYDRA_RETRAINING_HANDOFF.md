# Gastruloid Hydra retraining handoff

Recorded: 2026-09-05 (Europe/Paris)

## Resume outcome (2026-09-05)

The locked evaluation was resumed from the existing `epoch_098.ckpt`; no retraining occurred.

- Final status: `complete_with_failed_gates`; `release_suitable: false`
- Controlled nematic mean: reference `10.8928976` degrees, candidate `11.8942862` degrees; delta `+1.0013901` degrees exceeded the locked `+1.0`-degree limit
- Paired bootstrap 95% CI for candidate-minus-reference: `[-0.0884531, 2.1155412]` degrees; upper bound exceeded the locked `<2.0`-degree limit
- Controlled p95, length MAE, all three `training_consistent` frozen-center comparisons, finite/range checks, checkpoint loading, training contract, and native-artifact checks passed
- Legacy regression, validated audit, selected segmentation, and the complete Zenodo metadata tree all passed the final protection check
- Focused permanent suite: 40 passed using the single repository `.pytest_tmp`
- No checkpoint was substituted and no Zenodo, Napari, segmentation, legacy, or validated-audit state was changed

## Explicit scientific release override (2026-09-05)

After reviewing the completed locked result, the user explicitly accepted the candidate for release despite its narrow miss. This is a release-decision override only, not a claim that the original criteria were met; `evaluation/result.json`, its thresholds, its failed assertions, and the original `provenance.json` remain unchanged.

- Recorded misses: controlled mean delta `+1.0013901` degrees and paired-bootstrap upper 95% CI `2.1155412` degrees
- All other locked scientific, loadability, training-contract, native-artifact, and protected-asset checks passed
- Accepted source: `regression3d_nematic_hydra_seed12345/checkpoints/epoch_098.ckpt`
- Superseded interim local copy: `DARE3d_data_190326/Gastruloid_241025/weights/regression3d_nematic_hydra_seed12345/checkpoints/DARE3D_gastruloid_regression_epoch098.ckpt`
- Both checkpoint files are 70,823,338 bytes with SHA-256 `202ec55cdde73088a6b67b6abbd072ba25c7a150471cc419295abda4c4806aef`
- Machine-readable override: `regression3d_nematic_hydra_seed12345/provenance/release_acceptance_override.json` (SHA-256 `18470c0d1032bc3897790c2865b87de98fb748deed8d4f2b9303a537b36bbb0d`)

## Corrected Zenodo promotion destination (2026-09-05)

The user corrected the prior destination instruction. The accepted model is now promoted into the Zenodo staging tree; the `DARE3d_data_190326` directory is retained as the immutable training/audit source.

- Training/audit source retained: `DARE3d_data_190326/Gastruloid_241025/weights/regression3d_nematic_hydra_seed12345/`
- Final model directory: `DARE3dv2_Zenodo_040926/Gastruloid_241025/weights/regression3d_nematic_hydra_seed12345/`
- Final checkpoint: `DARE3dv2_Zenodo_040926/Gastruloid_241025/weights/regression3d_nematic_hydra_seed12345/checkpoints/DARE3D_gastruloid_regression_epoch098.ckpt`
- Final filename: `DARE3D_gastruloid_regression_epoch098.ckpt`
- Selected source and final checkpoint are both 70,823,338 bytes with SHA-256 `202ec55cdde73088a6b67b6abbd072ba25c7a150471cc419295abda4c4806aef`
- Full-tree verification before adding the correction record: 274 files and 311,857,654 bytes on each side, with zero path/size/SHA-256 mismatches
- Destination-only correction record: `provenance/release_destination_correction.json` (SHA-256 `0637ba8a85285e061c241973b0e005578e73b8a5dcb3a1c6de90dbcbe9889e9a`)
- The original evaluation and provenance remain unchanged and continue to record `release_suitable: false`; explicit acceptance exists only in the additive override records
- The interim local release-named copy is retained inside the source directory as historical evidence, but it is not the final release destination
- No existing Zenodo file was overwritten; the legacy gastruloid regression directory, every segmentation model, all neural-tube models, the validated audit, and Napari configuration remain unchanged

## Safe-stop state

The user requested a safe stop after the Hydra training had already completed and after the locked evaluator had stopped with an error. No training or evaluation process is intentionally left running. Do not launch another training run when resuming.

The completed candidate is:

`C:\Users\ruppr\Documents\Codexsession\DARE3D_280826\DARE3d_data_190326\Gastruloid_241025\weights\regression3d_nematic_hydra_seed12345`

It is a new, self-contained native Hydra model directory under the exact required gastruloid `weights` location. It contains native `.hydra/`, `checkpoints/`, `runs/` with TensorBoard and regression diagnostics, the Hydra job log, and `mlflow.db`. No artifacts were copied from the legacy model.

## Protected scope

Preserve all of the following exactly as they currently are:

- Validated audit reference: `docs/reproducibility_audit/nematic_retraining/seed_12345/`
- Legacy Hydra provenance: `DARE3d_data_190326/Gastruloid_241025/weights/regression3d_exp10-b/`
- Every segmentation model, checkpoint, configuration, and selected path
- The complete `DARE3dv2_Zenodo_040926/` tree
- Napari code and defaults

This work is gastruloid regression only. Do not modify Zenodo, Napari defaults, the legacy regression directory, or any segmentation model. The new candidate must continue to match the general provenance completeness of `regression3d_exp10-b` through artifacts genuinely generated by this run; never copy legacy artifacts into it.

A pre-run protection snapshot is stored at:

`DARE3d_data_190326/Gastruloid_241025/training_splits/regression3d_nematic_seed12345/protected_before.json`

It records full hashes for the legacy and validated-audit trees, selected segmentation hashes, and path/size/timestamp metadata for the 297-file Zenodo tree. The final evaluator failed before writing its post-run protection result, so byte-level post-verification remains an explicit resume task. Nothing should be reset, cleaned, moved, or deleted.

## Scientific protocol implemented

The committed Hydra experiment maps the validated audit setup into the repository-native pipeline:

- Train: movie3 and movie4; validation/test: movie2
- Repository scale table and observed target shapes: movie2 `330x278x164`, movie3 `387x375x77`, movie4 `405x394x152`
- Three temporal channels `[-1, 0, 1]`, min-max normalization, `32^3` crops
- RegressionNet: five stages, 16 starting filters, 9D rotation output, 5,897,940 parameters
- Corrected nematic loss: `90 * (1 - (u dot v)^2)`, plus the unchanged length loss
- Seed 12345; 2,000 sampled train/validation items per epoch; batch size 12
- AdamW, learning rate 0.001, betas 0.9/0.999, weight decay 0.0001
- OneCycleLR with 167 steps/epoch, 100 epochs, `pct_start=0.1`, preserving the historical epoch scheduler interval
- FP32 CUDA execution; one predeclared run only; checkpoint selection by minimum `val/loss`

The staged split uses verified NTFS hard links to the genuine movie2/movie3/movie4 sources; staged and source SHA-256 values matched before training. Temporary split data is under `DARE3d_data_190326/Gastruloid_241025/training_splits/regression3d_nematic_seed12345/`.

## Training result

Run ID: `20260904T193751Z`

The first invocation stopped before model/data initialization or any optimizer step because the nested Hydra output directory did not yet exist. Commit `7c51454` fixed native creation of that declared directory. The same locked configuration and run ID were then launched and completed all 100 epochs successfully.

Hydra selected and successfully restored:

`DARE3d_data_190326/Gastruloid_241025/weights/regression3d_nematic_hydra_seed12345/checkpoints/epoch_098.ckpt`

- Size: 70,823,338 bytes
- SHA-256: `202ec55cdde73088a6b67b6abbd072ba25c7a150471cc419295abda4c4806aef`
- Native restored-checkpoint test loss: `5.446471214294434`
- `last.ckpt` currently has the same size and SHA-256

This native objective is encouraging but is not the scientific release decision.

## Evaluation state and exact failure

The locked evaluator completed:

- Independent native Hydra reconstruction of 526 train crops and 156 validation crops
- Controlled inference for the candidate and validated `epoch_095.ckpt` on the same 156-event holdout
- Both models on all three frozen center sets in `training_consistent` and diagnostic `legacy_raw` modes
- Compact prediction files and event CSVs under the candidate's `evaluation/` directory

It then failed before aggregating/writing `result.json`, `provenance.json`, and the final protection assertions. The exact error was:

`ValueError: Invalid isoformat string: '2026-08-29T14:19:57.1913951+00:00'`

Python 3.10 accepts at most six fractional-second digits, while the Windows baseline timestamp has seven.

A small provisional, uncommitted change in `scripts/evaluate_gastruloid_nematic_hydra.py` now truncates extra ISO fractional digits before parsing. Its AST passes. The fix has not yet been unit-tested, committed, or used for a successful evaluator rerun. Do not loosen any scientific threshold while resolving this compatibility bug.

Partial evaluation artifacts are useful but provisional. There is no final release-suitability verdict yet.

## Acceptance gates (unchanged)

Compare against validated `epoch_095.ckpt` on identical data and the corrected nematic metric:

- Controlled mean error: candidate no more than 1 degree above reference
- Paired 10,000-replicate bootstrap (seed 20260829): upper 95% bound of the mean difference below 2 degrees
- Controlled p95: candidate no more than 5 degrees above reference
- Length MAE: candidate no more than 0.25 voxel above reference
- Every `training_consistent` frozen-center mean: candidate no more than 2 degrees above reference
- All angular values finite and within 0-90 degrees
- All training-contract, native-artifact, checkpoint-load, and protected-asset assertions pass

If these gates fail, mark the candidate unsuitable. Do not cherry-pick another checkpoint, tune thresholds, or retrain without new user direction.

## Tracked changes and commits

Branch: `main`, currently five commits ahead of the locally recorded `origin/main`.

- `9bd5797` ? complete: Hydra experiment, corrected nematic loss, controlled regression diagnostics, and focused tests
  - `configs/experiment/gastruloid_nematic_regression.yaml`
  - `configs/model/criterion/nematic_angle_len.yaml`
  - `dare3d/losses/angle3d.py`
  - `dare3d/models/regression_module.py`
  - `tests/test_gastruloid_nematic_hydra.py`
- `54b4784` ? complete implementation, validation outcome pending: locked evaluator
  - `scripts/evaluate_gastruloid_nematic_hydra.py`
- `7c51454` ? complete: create nested native Hydra output directories
  - `dare3d/utils/utils.py`
  - `tests/test_gastruloid_nematic_hydra.py`
- `a6597c3` ? complete: path-keyed protected-artifact comparisons
  - `scripts/evaluate_gastruloid_nematic_hydra.py`
- `faa0aba` ? complete: record actually instantiated Hydra split/crop/geometry
  - `scripts/evaluate_gastruloid_nematic_hydra.py`

Uncommitted tracked change:

- `scripts/evaluate_gastruloid_nematic_hydra.py` ? provisional timestamp-parser compatibility fix described above

Generated and intentionally Git-ignored:

- Candidate model directory ? training complete; scientific validation provisional
- Staging/split directory and protection baseline ? complete and required for validation

This handoff file is newly created and untracked until a future commit.

## Tests and checks completed

- `tests/test_gastruloid_nematic_hydra.py`: 6 passed using the repository's single `.pytest_tmp`
- Evaluator AST parse: passed
- Evaluator `--help` import: passed
- Hydra preflight: CUDA available on Quadro RTX 5000; exact split, crop counts, shapes, parameter count, and source hashes confirmed
- Native 100-epoch training and restored best-checkpoint test: passed
- Full focused permanent suite: not yet run
- Final locked evaluation/protection gate: not yet complete because of the timestamp error

A non-fatal Windows optional-import traceback involving `pyarrow` appeared during pytest/MONAI import, but pytest exited successfully.

## Current Git status

Tracked status before creating this handoff:

`## main...origin/main [ahead 5]`
` M scripts/evaluate_gastruloid_nematic_hydra.py`

Untracked material already present and deliberately preserved includes `DARE3D_AUDIT_PROGRESS.md`, `SESSION_HANDOFF.md`, the full `DARE3dv2_Zenodo_040926/` tree, extensive `docs/reproducibility_audit/` evidence and scripts (including the validated run), planning documents, `manuscript_280826version/`, and `grep.exe.stackdump`. Do not add, alter, clean, or remove those unrelated files. Use `git status --short` for the full listing. After this edit, this handoff is also untracked.

No session commit has been pushed; local `main` is five commits ahead of `origin/main`.

## Historical resume instruction (completed 2026-09-05)

The steps below record the completed resume procedure for audit purposes; do not rerun them merely because they remain in this handoff.

Do not retrain. Resume from the existing `epoch_098.ckpt`.

1. Review the uncommitted seven-digit timestamp parser, add a narrowly scoped regression test if appropriate, run the focused test file with `--basetemp=.pytest_tmp`, and commit only the parser/test plus this handoff if they pass.
2. Rerun the unchanged locked evaluator from the repository root:

   `$env:PROJECT_ROOT='C:\Users\ruppr\Documents\Codexsession\DARE3D_280826'; $env:PYTHONUTF8='1'; & 'C:\Users\ruppr\.conda\envs\dare3d-v2.0\python.exe' 'scripts\evaluate_gastruloid_nematic_hydra.py'`

3. Inspect `DARE3d_data_190326/Gastruloid_241025/weights/regression3d_nematic_hydra_seed12345/evaluation/result.json` and `provenance/provenance.json`. Require every scientific, artifact, training-contract, and protected-asset assertion to be true.
4. Run the focused permanent suite with the existing `.pytest_tmp` only:
   `tests/test_gastruloid_nematic_hydra.py tests/test_regression_preprocessing.py tests/test_regression_entrypoint_configuration.py tests/test_napari_release_models.py`
5. Recheck Git status, preserve all unrelated untracked files, and only then push the reviewed commits if still requested/authorized.

Do not modify the candidate checkpoint or replace it with another epoch during resume.
