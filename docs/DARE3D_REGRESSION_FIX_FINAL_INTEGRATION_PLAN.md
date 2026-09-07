# DARE3D regression preprocessing fix: final integration plan

Status: implementation specification for the validated 2026-09-03 correction.

The latest 2026-09-03 sections of `SESSION_HANDOFF.md` and
`DARE3D_AUDIT_PROGRESS.md` are authoritative for scientific conclusions. The
generic correction and its no-retraining validation are complete. Final
integration must preserve that validated behavior while making it part of the
normal source, configuration, packaging, and test surface of DARE3D.

## Public contract

- `training_consistent` is the default for regression prediction, evaluation,
  the headless API, and napari. It resamples each raw movie into the same grid
  used to create regression training crops before extracting an inference crop.
- `legacy_raw` is an explicit compatibility mode for replaying historical
  inference that cropped directly from the raw grid. It is never selected
  implicitly.
- Existing output consumers continue to receive `center`, `rotation`, and
  `length`. New fields identify raw-grid, regression-grid, and physical
  geometry and units explicitly.
- A missing scale file falls back to the saved `default_scale` by default.
  `require_scale_file=true` is the opt-in strict mode and must fail if any
  inference movie lacks an entry.
- Saved absolute paths from a training workstation are not implicit runtime
  dependencies. Prediction and evaluation use a scale path only when the
  current caller supplies it.

## Validated source inventory

The following production files constitute the scientifically validated core.
They must be integrated as normal tracked source and should remain byte-for-byte
unchanged during release wiring unless a newly failing permanent test proves a
correction is unavoidable:

| File | Role | State at start of closeout |
|---|---|---|
| `dare3d/data/components/abstract_celldataset.py` | scale loading and spatial resizing primitives | tracked, modified |
| `dare3d/data/components/regress_3dataset.py` | shared training/inference regression preparation | tracked, modified |
| `dare3d/data/components/regression_geometry.py` | raw/regression/physical transforms and mode validation | required but untracked |
| `dare3d/metrics/infer_measure.py` | evaluation geometry binding | tracked, modified |
| `dare3d/metrics/inference.py` | shared dataset preparation, decode, and durable output | tracked, modified |

The integration call paths and display adapter are:

- `dare3d/predict.py`
- `dare3d/eval.py`
- `dare3d/utils/regression_display.py`
- `napari_dare3d/_api.py`
- `tests/test_geometry.py`
- `tests/test_inference_checkpoint_loading.py`

`tests/test_regression_preprocessing.py` contains the appropriate synthetic
software regression tests but was untracked at the start of closeout. It must
become a permanent tracked test. Scientific validators, experiment logs,
checkpoint replays, comparison tables, and evidence under
`docs/reproducibility_audit/` remain audit material. Production modules and
permanent tests must not import them.

Before implementation, the repository also contained root-level disposable
pytest directories named `.pytest_tmp_assoc_fallback` and
`.pytest_tmp_regression_fix*`, plus `.pytest_cache`. These are local test
scratch data, not validation evidence.

## Ordered implementation work

### 1. Freeze and integrate the validated production core

Files:

- `dare3d/data/components/regression_geometry.py`
- `dare3d/data/components/abstract_celldataset.py`
- `dare3d/data/components/regress_3dataset.py`
- `dare3d/metrics/inference.py`
- `dare3d/metrics/infer_measure.py`
- `dare3d/utils/regression_display.py`

Actions:

1. Track `regression_geometry.py` with the other package modules. Do not copy
   this logic into an audit helper or an entry point.
2. Reuse `RegressionPreprocessingSpec`, `RegressionSpatialTransform`, and
   `normalize_preprocessing_mode` everywhere. Do not introduce a second
   coordinate-transform implementation.
3. Reuse `Regress3Dataset.init_inference` through
   `prepare_regression_dataset`. Training remains on `Regress3Dataset.init`.
4. Preserve raw `MTXYZ`, regression `MTXYZ`, internal `TXYZ`, and disk `TZYX`
   conventions and the documented interpolation, rounding, padding, and
   normalization order.
5. Preserve legacy fields and add explicit fields in the shared inference
   result once, before CLI, evaluator, display, or napari adapters consume it.

Acceptance criteria:

- All six modules above are ordinary package files known to Git.
- No import from `dare3d/` resolves through `docs/`, a notebook, a local data
  directory, or an untracked helper.
- The five validated hashes recorded in the authoritative handoff still match.

### 2. Wire training, prediction, evaluation, and API configuration

Files:

- `configs/data/dataset/regression.yaml`
- `configs/predict.yaml`
- `configs/eval.yaml`
- `dare3d/predict.py`
- `dare3d/eval.py`
- `napari_dare3d/_api.py`
- `napari_dare3d/_widget.py` only if an exposed widget control is deliberately
  added later

Actions:

1. Set the shipped regression dataset default `require_scale_file: false`.
   The class default remains false for old saved Hydra configs.
2. Set `regression.preprocessing_mode: training_consistent` in prediction and
   evaluation configs. Retain `legacy_raw` as a valid explicit override.
3. In `predict.load_config`, distinguish the regression stage, remove the
   saved test dataset scale path, apply current root-level scale overrides,
   and copy `regression.require_scale_file` to the instantiated dataset config.
4. In `eval.load_config`, capture the current regression overrides before
   merging the saved training config. Apply `scale_file_override`, optional
   default/target overrides, and strictness to `data.test_data` before
   resolving interpolations.
5. In `evaluate_regression`, propagate strictness to the instantiated dataset
   and call `prepare_regression_dataset` with the selected mode.
6. In `predict.load_data`, use the same preparation function for regression;
   leave segmentation initialization unchanged.
7. In `_api.infer_stack`, default to `training_consistent`, expose
   `regression_require_scale_file`, and pass strictness only into the
   regression config. Do not add regression-only constructor keys to a
   segmentation dataset.
8. Keep the current widget on the production default. The headless API is the
   explicit advanced interface for historical replay and strict scale checks.

Dependencies:

- Step 1 must be present before these entry points import the shared helpers.
- Scale overrides must be applied before `OmegaConf.resolve` and before Hydra
  instantiation.

Acceptance criteria:

- All four runtime paths select `training_consistent` when no override exists.
- `legacy_raw` reaches `Regress3Dataset.init_inference` unchanged when selected.
- An old saved config containing an absolute scale path does not cause current
  regression inference to read that path.
- Explicit current scale/default/target values win consistently.
- Strict mode fails on missing per-movie metadata; non-strict mode uses the
  checkpoint default scale.
- Segmentation behavior is unchanged.

### 3. Promote permanent software tests

Files:

- `tests/test_regression_preprocessing.py`
- `tests/test_regression_entrypoint_configuration.py`
- `tests/test_inference_checkpoint_loading.py`
- `tests/test_geometry.py`

Required coverage:

- known checkpoint scale profiles and their target shapes;
- raw-to-regression round trips and preservation of movie/time coordinates;
- training and inference crop identity, including a padded border crop;
- raw annotation association and preprocessing manifest contents;
- physical, raw, and regression decoded geometry;
- legacy `center`, `rotation`, and `length` alongside explicit fields;
- invalid preprocessing mode failure;
- strict scale-file failure rather than silent fallback;
- default and explicit legacy mode propagation through prediction,
  evaluation, and napari/API dataset construction;
- removal of saved workstation scale paths and precedence of explicit current
  overrides;
- checkpoint loading remains limited to network weights;
- raw endpoints drive the displayed napari axis.

Rules:

- Tests use generated arrays, temporary configs, and dummy networks only.
- Tests do not use released checkpoints, neural raw data, annotation bundles,
  audit results, workstation paths, GPU availability, or network access.
- Completed scientific validation remains under
  `docs/reproducibility_audit/` and is not made part of routine pytest.

Acceptance criteria:

- The four focused files pass in CPU-only mode with automatic third-party
  pytest plugin loading disabled.
- A fresh checkout can run them without any untracked file.

### 4. Package and automate the release surface

Files:

- `setup.py`
- `MANIFEST.in`
- `.github/workflows/regression-preprocessing.yml`
- `README.md`
- `dare3d/README.md`
- `napari_dare3d/README.md`

Actions:

1. Retain package discovery for `dare3d`, `napari_dare3d`, and the Hydra
   `configs` package. Verify that `regression_geometry.py`, the config YAMLs,
   and the napari manifest appear in built artifacts.
2. Register `predict_command` alongside the existing train/eval console entry
   points, and keep direct `python dare3d/predict.py` execution importable from
   a fresh checkout.
3. Add a focused CPU CI job that builds both distribution formats and runs
   the permanent regression tests.
4. Document the default/replay modes, scale precedence and strictness, legacy
   output compatibility, explicit geometry units, and test/audit boundary.

Acceptance criteria:

- Source and wheel builds succeed without audit assets.
- Installing the wheel into an isolated environment permits imports of the
  geometry module and loading of shipped configs.
- CI invokes only permanent tests and does not retrain or replay scientific
  validation.

### 5. Establish one pytest scratch location

Files:

- `pyproject.toml`
- `.gitignore`
- `scripts/clean_test_artifacts.py`
- `Makefile`

Actions:

1. Configure `--basetemp=.pytest_tmp` and
   `cache_dir=.pytest_tmp/cache`.
2. Ignore only the root `.pytest_tmp/` test area rather than a broad
   `.pytest*` pattern that can conceal unexpected files.
3. Provide `make clean-test`, backed by a repository-root-aware cleanup
   script. The script may remove only direct children named `.pytest_tmp`,
   `.pytest_cache`, or `.pytest_tmp_*`; it must reject paths outside the
   repository and handle links/reparse points without traversing them.
4. First run the cleanup script with `--dry-run`. Verify every listed path is
   disposable pytest output. Then run it with `--include-legacy` to remove the
   old caches and accumulated regression-fix basetemps. Do not copy them into
   the new directory: durable evidence already lives under the audit tree.
5. Subsequent pytest runs may recreate only `.pytest_tmp/`, with its cache
   nested at `.pytest_tmp/cache`; the whole dedicated tree is ignored.

Acceptance criteria:

- No `.pytest_tmp_*` or root `.pytest_cache` remains.
- Routine pytest creates no new root-level scratch directory other than
  `.pytest_tmp/`.
- Git reports none of the pytest scratch contents.

## Verification commands

Run from the repository root in the supported DARE3D environment. On Windows,
set the environment variable using PowerShell syntax.

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest tests/test_regression_preprocessing.py tests/test_regression_entrypoint_configuration.py tests/test_inference_checkpoint_loading.py tests/test_geometry.py
python -m pytest -k 'not slow'
python -m compileall -q dare3d napari_dare3d tests
python -m build --no-isolation
git diff --check
git status --short --untracked-files=all
git ls-files dare3d/data/components/regression_geometry.py tests/test_regression_preprocessing.py tests/test_regression_entrypoint_configuration.py
git grep -n reproducibility_audit -- dare3d napari_dare3d tests
```

The last grep must return no production/test dependency. Inspect the wheel
archive and confirm it contains:

- `dare3d/data/components/regression_geometry.py`;
- `napari_dare3d/napari.yaml`;
- the `configs` YAML tree.

After a selective integration commit, verify committed-only behavior from a
separate path:

```powershell
git clone --no-local . ../DARE3D_regression_fresh_clone
Set-Location ../DARE3D_regression_fresh_clone
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest tests/test_regression_preprocessing.py tests/test_regression_entrypoint_configuration.py tests/test_inference_checkpoint_loading.py tests/test_geometry.py
```

Do not use the working repository's untracked files, Python path, caches, data,
or editable installation for that check.

## Explicit non-goals

Do not:

- retrain any model;
- rerun the completed scientific regression validator or repeat resolved
  historical investigations;
- change checkpoints, raw data, annotations, published metrics, thresholds,
  association policy, or audit conclusions;
- begin the segmentation preprocessing-parity investigation in the same
  integration change;
- special-case neural-tube or gastruloid data in production code;
- import audit helpers into `dare3d/`, `napari_dare3d/`, or `tests/`;
- remove or rewrite audit evidence while cleaning pytest scratch data;
- bulk-stage the working tree, because it contains unrelated local audit and
  data artifacts.

## Expected final Git state

The selective release change contains the production modules and adapters,
the three public configuration files, permanent synthetic tests, focused CI,
documentation, and test-hygiene files listed above. Every required production
or test file is tracked. No runtime or test import depends on an untracked
file, cache, checkpoint, raw dataset, environment-specific absolute path, or
audit-only helper.

Before committing, review an explicit path list and stage only integration
files. Pre-existing local audit/data artifacts may remain untracked, but they
must not be part of the release dependency graph. No `.pytest_tmp_*` directory
or root `.pytest_cache` should remain; the sole ignored test scratch area is
`.pytest_tmp/`.

Once these changes pass focused, non-slow, packaging, hash, and committed-only
checkout checks, the regression preprocessing correction is closed out. The
repository is then ready to move on to the separate segmentation
preprocessing-parity investigation.
