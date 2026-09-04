# DARE3D v2 scale-handling audit

Date: 2026-09-03

## Executive conclusion

The v1 scale table has been restored at the compatibility path
data/3D/scales.json. Its 79 JSON entries are semantically identical to the
public v1 file. The only byte-level difference is one conventional terminal
newline; no number, key, or value was changed.

The file is not passive metadata. DARE3D's shared dataset base class uses a
per-movie XYZ spacing divided by target spacing to resample images and
annotations. If the file or a movie entry is missing, it warns and uses
default_scale. That fallback can materially change training inputs,
segmentation predictions, regression crops, coordinates, axes, and lengths.

The restored table resolves the previously missing nuclei calibration:
movie2 and movie3 are 0.914, 0.914, 0.914 and movie4 is 1, 1, 1. It also
contains movie_M at 0.2076, 0.2076, 1.0. It does not contain movie_E or
movie_I2, for which the configured model default remains the applicable
source unless the caller supplies another scale.

Minimal production changes make the historical path usable on
case-sensitive systems and in both source and wheel installations. The
Napari path now preserves the selected movie name for table lookup and
propagates calibrated layer scale to overlays. No training, checkpoint,
annotation, released result, or completed regression validator evidence was
modified or rerun.

## Scope and evidence

The audit searched production Python, Hydra configuration, scripts,
notebooks, tests, packaging, and plugin code for scales.json, scale,
scales, voxel size, spacing, resolution, pixel size, hard-coded spatial
sizes, isotropic assumptions, and acquisition-metadata readers. It traced
training, evaluation, prediction, segmentation and regression inference,
matching, display/export, and Napari call paths.

The source table was obtained from:

- https://github.com/JFRupprecht-OM/DARE3d/blob/main/data/3D/scales.json
- the same historical file at commit
  fc005b19dba0501a50f448d228c66d295ecd62d8

Integrity results:

| Property | Result |
|---|---|
| Entries | 79 |
| Restored semantic SHA-256, sorted compact JSON | 63b395ea45713ccb329a25451da843de4ed43f0cee516258c407dd1ed0be31cf |
| Public v1 raw-file SHA-256 | d1d14567063a27e0d29912dd480c1027555cf29aa3e9c6d8fdaf02990d200eaf |
| Restored raw-file SHA-256 | e206eb6502a5a5907bd919e647be24fc66dce6ea7a6fc1a357856b09fc4021fe |
| Byte difference | restored file has one terminal LF |
| Numeric differences | none |

## Scale contract in production

AbstractCellDataset is the definitive scale reader.

1. load_movie_scales reads a JSON mapping keyed by TIFF filename stem.
2. get_movie_scale selects the matching entry or default_scale.
3. It divides source XYZ spacing by target_scale to obtain the resize factor.
4. _compute_target_shape applies floor conversion through an integer cast:
   raw XYZ shape multiplied by the resize factor.
5. resize_data resizes complete movies and, when labels are loaded, applies
   the same factor to both daughter endpoints.
6. _resize_bipoints rounds each endpoint before later midpoint, axis, and
   length calculations. Therefore two close scales that produce the same
   image shape can still produce different regression labels and crops.

The JSON values and defaults are XYZ micrometres per source voxel. Image
files are read from TZYX disk order and swapped to TXYZ internally. The
target spacing is normally one micrometre in all three axes.

If the file is absent or unreadable by path, load_movie_scales returns an
empty mapping after a warning. A missing movie key also falls back to
default_scale. Regression can opt into require_scale_file to turn missing
entries into an error; the older/general dataset behavior remains fallback
compatible.

## Execution-path matrix

| Path | How scale is obtained | Missing behavior | Numerical role |
|---|---|---|---|
| Segmentation training | experiment scale_file, then movie key, else default | warning plus default | full image and label resampling; masks, crops, and weights follow the resampled grid |
| Regression training | same shared dataset path | warning plus default | full image and endpoint resampling; crop centers, axes, and lengths change |
| TAP and SegRes training | same shared dataset path | warning plus their configured default | full image/label preprocessing changes |
| Segmentation CLI/API inference | checkpoint/caller dataset settings | default if file or key absent | each temporal sample is resampled before the model; probability is inverse-resized to the raw grid |
| Segmentation evaluation | same inference preprocessing; ground truth is built on the raw grid | default if unavailable | predictions can change; raw-grid matching then evaluates those changed predictions |
| Corrected regression prediction/evaluation | caller override or model default, through init_inference | default, or error with require_scale_file | raw centers are mapped to the training regression grid; crops and decoded geometry change |
| legacy_raw regression replay | explicit legacy mode | scale is not used for input resampling | historical raw-grid model numerics are scale-independent; physical interpretation remains unavailable |
| Napari/API | JSON entry by selected layer/movie stem, then explicit default or calibrated layer, then model default | unit display scale when no physical source is known | model preprocessing uses the resolved source; image/result layers use known physical calibration |
| Object matching | raw voxel masks/centroids | no physical conversion | IoU is raw-grid; centroid distance uses unscaled Euclidean voxel distance |
| TIFF exports | array geometry only | no spacing metadata written | coordinates can be correct, but TIFF does not preserve physical calibration |

The six experiment presets now refer to data/3D/scales.json with the exact
case used by the restored path. The inherited v1 presets used data/3d on a
case-insensitive workstation; that silently fails on a case-sensitive
filesystem even when data/3D/scales.json exists.

Prediction and evaluation intentionally retain explicit portability
controls. A caller can supply scale_file, default_scale, and target_scale.
Regression ignores a stale scale-file path embedded in a checkpoint unless
the current invocation supplies one, while keeping the checkpoint's saved
default. This prevents a training-workstation path from becoming hidden
runtime state. The canonical table is documented as the explicit override.

## Components whose values are scale-sensitive

### Verified numerical dependence

- Full-volume image interpolation and the target shape.
- Label endpoint resampling, including independent endpoint rounding.
- Segmentation masks and sparse weights created from resampled labels.
- Training crop placement and physical field of view.
- Regression-grid conversion of raw detector or annotation centers.
- Daughter-axis direction and length when computed after endpoint scaling.
- Corrected regression output conversion among regression-grid voxels,
  physical XYZ, and raw-grid XYZ.
- Segmentation model probabilities and detections, because the model sees a
  different resampled image when target shape changes.
- Downstream object counts and evaluation scores when scale changes the
  predicted probability field or regression outputs.

### Physical/display dependence

- Napari Image, center, and axis layer calibration.
- Physical axis and length fields in corrected regression outputs.
- Physical interpretation of raw-grid coordinates and crop support.

The TIFF and legacy fields still carry array coordinates. Without scale they
can remain internally registered in voxel space while being physically
mislabelled or displayed with the wrong aspect ratio.

### Unaffected or not directly calibrated

- TIFF byte loading and the TZYX-to-TXYZ axis swap.
- Network architecture and checkpoint parameter loading.
- Quaternion representation conversion as a mathematical operation.
- IoU matching once prediction and ground-truth arrays already share the
  same raw grid.
- The existing centroid-distance matcher, which deliberately measures raw
  voxel distance and does not consult scale.
- Thresholding and connected-component labelling on an already produced
  probability array.
- Random IsotropicScale augmentation. Its name refers to stochastic data
  augmentation, not acquisition calibration.
- The segmentation module's resolution factor, which is a network
  multi-resolution output setting rather than microscope voxel spacing.

Hard-coded cell-radius sphere volumes, component-size filters, crop sizes
such as 128 cubed and 32 cubed, and raw centroid distance thresholds are
voxel-domain assumptions. Resampling to the target grid makes those
approximately physical for the model path, but raw-grid postprocessing and
matching remain anisotropy-unaware.

## Focused effect diagnostics

No model inference or training was rerun for this audit.

For the released nuclei arrays, the restored table gives:

| Movie | Raw internal TXYZ | Restored factor | Target spatial XYZ |
|---|---:|---:|---:|
| movie2 | 10 x 362 x 305 x 180 | 0.914, 0.914, 0.914 | 330 x 278 x 164 |
| movie3 | 28 x 424 x 411 x 85 | 0.914, 0.914, 0.914 | 387 x 375 x 77 |
| movie4 | 28 x 405 x 394 x 152 | 1, 1, 1 | 405 x 394 x 152 |

These spatial shapes independently reproduce the archived training log.
The movie3/movie4 regression training shapes have one additional leading
time frame from the existing temporal-padding rule.

The missing-file default 0.621, 0.621, 2 would instead map movie2 to
224 x 189 x 360. A prior frozen segmentation diagnostic measured F1
0.351648 with that wrong fallback versus 0.918518 with the reconstructed
nuclei scale. This is direct evidence that absence of the table can change
scientific output.

The archived reconstructed 0.912 movie2 factor and restored 0.914 factor
both yield 330 x 278 x 164. Therefore the already completed nuclei
segmentation replay uses the same interpolation target and does not need to
be rerun solely for the 0.912-to-0.914 correction.

Regression labels have a stricter result. Direct annotation diagnostics
showed:

- movie2, 0.914 versus 0.912: 194 of 226 endpoint pairs, 149 crop
  centers, and 166 axis vectors changed after integer rounding; among the
  156 temporally eligible events, the counts were 140, 111, and 119.
  The largest endpoint shift was one target-grid voxel.
- movie_M, 0.2076 versus the audit profile 0.208: 66 of 104 endpoint
  pairs, 47 crop centers, and 54 axis vectors changed; among 80 temporally
  eligible events, the counts were 47, 30, and 37. The largest endpoint
  shift was one target-grid voxel.

Thus same image shape is sufficient for segmentation interpolation parity
but not exact regression-label parity. The completed regression validator
still validates the production preprocessing correction, crop parity under
its declared profiles, and the neural-tube conclusion. Its exact nuclei
numbers are a frozen log-reconstructed protocol, not an exact 0.914 table
replay. Its 0.208 neural protocol likewise must not be silently relabelled as
the exact movie_M table value 0.2076.

## Image-metadata audit

The production loader uses skimage.io.imread, and the Napari direct loader
uses tifffile.imread. Neither extracts physical spacing. No CZI reader or
filename-based spacing parser was found.

Header-only checks of representative released TIFFs found no complete
replacement for the table:

- nuclei movie2: no OME or ImageJ physical metadata and no usable resolution
  unit;
- nuclei movie3: ImageJ axes but no physical spacing;
- nuclei movie4: ImageJ unit says micron and XY resolution is one, but no Z
  spacing;
- neural movie_E, movie_I2, and movie_M: no OME and no usable XYZ physical
  metadata.

Consequently restoring the table is necessary for the named legacy data;
the released TIFF headers cannot reconstruct the same XYZ calibration.

## Napari audit

### Previous behavior

- The plugin did not read scales.json automatically.
- Its direct TIFF loader did not extract physical metadata.
- The headless API always wrote the in-memory stack as movie.tif, so a
  caller-provided table could only select the movie key rather than the
  actual selected movie name.
- Result Points layers did not inherit the Image layer scale.
- Unknown calibration silently remained Napari's unit scale.

This could leave centers voxel-registered but physically miscalibrated, and
could make anisotropic axes or overlays appear with the wrong physical
aspect ratio.

### Corrected behavior

- The advanced scale-file field is prefilled from the source checkout or
  installed environment copy of data/3D/scales.json when available.
- infer_stack accepts movie_name and preserves a filesystem-safe version of
  its stem for the temporary TIFF, enabling the shared dataset reader to
  select the correct JSON entry.
- Physical calibration precedence is matching JSON entry, explicit manual
  default, non-unit Image-layer scale, then a documented movie-specific
  fallback. Saved checkpoint defaults are model geometry, not display metadata.
- XYZ source spacing is converted to Napari ZYX or TZYX scale.
- The resolved calibration is assigned to the Image layer, and center/axis
  result layers receive the same four-dimensional scale.
- Corrected regression output uses raw-grid endpoints for overlays, so
  centers and axes remain registered with the original image grid.

### Legacy neural-tube segmentation compatibility

Physical/display calibration and checkpoint-compatible segmentation geometry
are now deliberately separate. Passing the authoritative movie_M spacing
`XYZ=(0.2076,0.2076,1)` into the promoted legacy neural-tube segmenter resizes
its input to approximately `212x212x10`; its observed probability maximum is
only `0.000437`, so no voxel reaches the unchanged `0.5` threshold. The
documented `XYZ=(0.2,0.2,1)` physical fallback also yields no detections.

The promoted `DARE3D_neural_tube_segmentation_epoch057.ckpt` was trained and
historically inferred with its saved `default_scale=XYZ=(0.621,0.621,2)` and
saved target scale, producing approximately `635x635x20` model inputs. Replaying
that preprocessing with the unchanged widget thresholds produced 51 retained
movie_M centers. This scale is checkpoint preprocessing geometry only; it must
not be interpreted as movie_M physical calibration.

The release preset therefore declares `segmentation_scale_mode=checkpoint_default`
only for that exact promoted checkpoint path. Gastruloid remains in `source`
mode, and a manually substituted checkpoint uses `source` even when its filename
matches. The headless API also defaults to `source` and rejects unknown modes.
In compatibility mode, only segmentation ignores invocation scale overrides.
Corrected `training_consistent` regression still receives the authoritative or
fallback physical scale unchanged, and the Image, center, and axis layers retain
the same physical TZYX calibration.

The post-integration release-asset GPU smoke returned 51 orientations for
the 51 retained centers with `training_consistent` regression. Both result
layers kept the authoritative `(T,Z,Y,X)=(1,1,0.2076,0.2076)` calibration.

The plugin is correctly calibrated when the selected layer name matches a
table entry, the user supplies a manual scale, the input layer already has
a non-unit valid scale, or a documented movie-specific fallback applies. For
an unknown movie with none of those sources, Napari remains at unit display
scale because physical calibration is unknown. The built-in TIFF loader still
does not parse TIFF/CZI spacing; this limitation is explicit rather than hidden.

## v1-to-v2 comparison

The core scale reader, source-over-target resize factor, target-shape
flooring, and endpoint-rounding behavior were retained almost verbatim from
v1. The JSON table was omitted during the v2 repository/data split, while
the inherited configs continued to reference it with lowercase 3d.

V2 added checkpoint-config loading, separate prediction/evaluation entry
points, corrected regression geometry, and a Napari plugin. Those additions
created several scale mechanisms: checkpoint defaults, current-invocation
overrides, the legacy table, and Napari layer scale. They do not supersede
the table for the named legacy movies because the TIFFs lack equivalent
metadata.

The minimal implementation reuses AbstractCellDataset for all model
preprocessing and adds only import-light Napari conversion/resolution
helpers. It does not duplicate image-resampling logic. A later refactor
should centralize source-spacing resolution for CLI, API, and GUI, but that
is separate from restoring the proven legacy behavior.

## Repository and packaging changes

- Added data/3D/scales.json.
- Added narrow gitignore exceptions so this one file can be committed while
  other root data remains ignored.
- Corrected all six experiment scale paths from data/3d to data/3D.
- Added the file to MANIFEST.in and setup.py data_files. The latter places
  it at environment-prefix/data/3D/scales.json for wheel installs.
- Updated README and both primary notebooks to use the exact-case path.
- Added napari_dare3d/_scale.py and targeted API/widget integration.
- Added permanent CPU tests in tests/test_scale_handling.py and a dedicated
  scale-handling CI workflow.
- Kept scientific findings in this audit document rather than embedding
  audit-only data or model runs in the software tests.
- The separately pending regression integration routes Pytest cache/basetemp
  to one ignored .pytest_tmp location and supplies a repository-scoped cleanup
  command for that directory and legacy root .pytest_tmp_* directories.

The source distribution and wheel were built locally. The source archive
contains a path ending data/3D/scales.json; the wheel contains
dare3d-0.0.1.data/data/data/3D/scales.json, which installs to the intended
environment-relative compatibility path.

## Verification

Permanent focused test command:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_scale_handling.py tests/test_regression_preprocessing.py tests/test_regression_entrypoint_configuration.py tests/test_inference_checkpoint_loading.py tests/test_geometry.py

Result: 48 passed in 16.73 seconds.

The scale tests verify:

- exact semantic table identity and representative keys;
- known nuclei target shapes and missing-file fallback geometry;
- equal segmentation target shape for 0.912 and 0.914;
- exact-case paths in all experiment presets;
- source and wheel packaging declarations;
- API movie-stem preservation;
- JSON/manual/layer scale precedence and XYZ-to-Napari conversion;
- anisotropic result-layer propagation and invalid-scale rejection.

Python compilation, notebook JSON parsing, and a headless widget calibration
smoke check also passed. The environment emits a known non-fatal Windows
access-violation traceback while an optional pandas/pyarrow integration is
probed; pytest continues and exits successfully.

A repository-wide non-slow run was also attempted. Modernizing two
test-helper imports from removed pkg_resources APIs to importlib.metadata
allowed complete collection. The result was 69 passed, 10 deselected,
3 failed, and 1 setup error. The four remaining failures are pre-existing
generic Hydra-test fixture drift: the default train fixture lacks
input_channels and the old eval fixture overrides a ckpt_path key no longer
present in the redesigned eval entry point. They are not scale-path failures
and are outside this minimal integration, but a whole-suite green claim is
not made.

Post-verification cleanup removed .pytest_tmp, build, dare3d.egg-info, and
the external temporary v1 comparison clone. No root .pytest_tmp_* directory
remains. Durable audit evidence under docs/reproducibility_audit was not
removed or consolidated with disposable test output.

## Scientific impact

### Verified effects

- Missing table entries fall back to configured defaults in the common
  dataset code.
- The fallback can radically change target geometry; movie2 maps to
  224 x 189 x 360 rather than 330 x 278 x 164 under the common wrong
  fallback.
- A frozen segmentation diagnostic already demonstrated the associated
  F1 change from 0.918518 to 0.351648.
- The restored movie2/movie3/movie4 table entries reproduce archived
  training spatial shapes.
- Close scale changes can alter rounded regression endpoints and derived
  crops even when the resampled image shape is unchanged.
- Pre-fix Napari could not reliably select per-movie table entries and did
  not propagate physical layer scale; the targeted fix corrects both.

### Potential effects requiring provenance checks

- Any v2 training run performed while the configured table was missing and
  whose movie stem is present in the table may have trained on the fallback
  geometry. Such a run needs retraining only if its resolved config/log
  confirms that wrong geometry was actually used.
- Any segmentation inference/evaluation made with a wrong fallback and a
  scale-sensitive movie should be recomputed.
- Any production training_consistent regression result intended as an exact
  v1 table replay should use the exact table scale. Exact nuclei 0.914 and
  movie_M 0.2076 regression numbers should receive a targeted,
  inference-only replay if they will be reported as table-faithful results.
- Unknown movies, movie_E, and movie_I2 require an explicit reviewed source
  scale or the checkpoint default; restoration alone does not establish
  their acquisition spacing.
- Voxel-domain radii and distance thresholds can have anisotropic physical
  meaning on raw grids even when their code is internally consistent.

### Unaffected completed evidence

- Released archived outputs are not changed by adding a repository file.
- The completed regression validator's software assertions, byte-identical
  preprocessing/crop checks under its declared profiles, frozen detections,
  and neural-tube comparison remain valid.
- The nuclei segmentation replay at 0.912 does not need rerunning because
  0.914 produces the identical interpolation target shape.
- Explicit legacy_raw regression replays remain historical raw-grid
  computations and are not numerically changed by the restored file.
- Explicit frozen 0.208 and shape-reconstructed audit protocols remain
  valid as those declared protocols; they must not be relabelled as exact
  table replays.
- Network weights, checkpoint compatibility, annotation pairing, and the
  independent regression preprocessing mismatch finding are unchanged.

## Recompute decision

Do not retrain or rerun completed audit work merely because the table has
been restored.

Recompute only:

1. runs whose resolved provenance confirms that a table-covered movie was
   processed with the wrong missing-file fallback;
2. exact table-faithful regression results that are intended for reporting,
   using 0.914 for nuclei movie2/movie3 and 0.2076 for movie_M;
3. future Napari exports made before this fix if physical calibration, not
   only voxel registration, is scientifically relevant.

Retraining is warranted only for a training run proven to have used wrong
fallback geometry. Otherwise use targeted inference/evaluation.

## Direct answers

1. Why was scales.json needed in v1? It supplied per-movie XYZ acquisition
   spacing so the shared dataset pipeline could transform anisotropic/source
   voxels and labels into the checkpoint target grid.
2. Which v2 components depend on it? All shared-dataset training paths and
   scale-aware segmentation/regression inference paths can use it; Napari
   now does so when a named entry is available.
3. What happened when it was absent? The reader warned and silently used
   default_scale per movie. On case-sensitive systems the old lowercase
   config path behaved as absent even if the uppercase file existed.
4. Could published or audit numbers change? Yes, if their run used the wrong
   fallback in a scale-sensitive path. The known movie2 segmentation
   diagnostic proves the effect. Frozen explicit-profile results remain
   results of those profiles.
5. What should be recomputed? Only provenance-confirmed wrong-fallback runs
   and exact table-faithful regression results needed for reporting, as
   listed above.
6. Does restoration solve the regression preprocessing mismatch? No. That
   mismatch was caused by regression inference skipping training-time
   resampling. The already integrated training_consistent fix solves that;
   the restored table now supplies correct named-movie metadata to it.
7. Is Napari now correctly calibrated? Yes when a table entry, manual scale,
   or calibrated Image layer is available. It cannot infer absent physical
   metadata from the current TIFF/CZI loader and remains unit-scaled when no
   trustworthy source exists.
8. Is retaining this path sufficient? It is necessary for backward
   compatibility and now works in checkouts and wheels. It is not a complete
   long-term design: source-scale resolution, metadata extraction, units,
   and raw-grid physical postprocessing should eventually be centralized
   behind one tested API.
