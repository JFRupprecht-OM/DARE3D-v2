# DARE3D Generic Regression-Inference Preprocessing Fix Plan

Status: core implementation and no-retraining validation completed on
2026-09-03. The design is retained below; numerical evidence is in
\`docs/reproducibility_audit/regression_preprocessing_fix_validation/\`.

## 1. Confirmed root cause

The fault is in the shared regression orchestration, not in segmentation or the
regression network architecture.

Training follows:

```text
raw TIFF (T,Z,Y,X)
-> internal array (T,X,Y,Z)
-> Dataset.init(preprocess=True)
-> full-movie spatial resampling
-> annotation endpoints rescaled
-> 16-voxel spatial padding
-> movie normalization
-> 3x32x32x32 crop
-> generic orientation/length regressor
```

Specifically:

- `DareDataModule.prepare_data()` in `dare3d/data/dare_datamodule.py` calls
  `init()` for training, validation, and test datasets.
- `AbstractCellDataset.init()` in
  `dare3d/data/components/abstract_celldataset.py` calls `resize_data()`,
  regression crop construction, normalization, and postprocessing.
- `resize_data()` resizes each timepoint with first-order
  `skimage.transform.resize` and rescales bipoint annotations.
- `Regress3Dataset.make_crop_all_division()` in
  `dare3d/data/components/regress_3dataset.py` pads the spatial volume and
  builds the 32-cubed crops.

Production regression inference instead follows:

```text
raw TIFF (T,Z,Y,X)
-> internal array (T,X,Y,Z)
-> init(preprocess=False)
-> raw-space padding and normalization only
-> raw detector center used without transformation
-> 3x32x32x32 raw-voxel crop
-> same generic regressor
```

The divergent entry points are:

- `predict.load_data()` in `dare3d/predict.py`;
- `evaluate_regression()` in `dare3d/eval.py`;
- `napari_dare3d._build_dataset()` in `napari_dare3d/_api.py`.

They call `init(preprocess=False)`, after which regression merely pads and
normalizes. `regression_inference()` assumes the supplied center and
`dataset.movies_im` already use the same coordinate system.

Segmentation should remain unchanged. Segmentation inference operates in its
own resampled space, but converts its probability map back to raw dimensions
before connected-component detection. Detector centers are therefore raw-space
`(M,T,X,Y,Z)` coordinates.

### Important checkpoint-provenance distinction

The approximately `212x212x10` neural-tube representation is the audit
retraining profile derived from the manuscript spacing
`(0.208,0.208,1.0) um`. The released original neural-tube regression
checkpoint's archived log instead records training at `635x635x20`, using
the effective configuration `(0.621,0.621,2.0) -> 1.0`.

The generic fix must therefore be checkpoint-specific. Treating
`212x212x10` as a universal neural-tube representation would create another
checkpoint-domain mismatch.

## 2. Preferred generic solution

Introduce an explicit, checkpoint-bound `RegressionPreprocessingSpec` and a
per-movie `RegressionSpatialTransform`.

The specification should record:

- internal axis order;
- raw or effective input spacing and its provenance;
- regression target spacing;
- exact target-shape rule;
- resize implementation and all interpolation parameters;
- normalization and its position relative to padding;
- temporal channels;
- crop size;
- spatial padding;
- center-rounding convention;
- provenance and confidence of every scale value.

Separate two concepts currently conflated by `default_scale`:

1. The input movie's physical voxel spacing.
2. The spatial representation actually used to train a particular checkpoint.

For new checkpoints, save this specification beside and inside the checkpoint.
For legacy checkpoints, construct a versioned sidecar from the archived Hydra
configuration, scale file, and training log. Strict inference must fail with an
actionable error when the training geometry cannot be established; it must not
silently substitute an unrelated default.

Provide an explicit `legacy_raw` mode only for reproducing historical output.

### 2.1 Regression-image preparation

Add a dedicated regression-inference initialization path, for example
`Regress3Dataset.init_inference()`:

1. Load the raw movie and retain its original shape.
2. Convert disk `T,Z,Y,X` to internal `T,X,Y,Z`.
3. Resample the complete movie using the checkpoint's training specification.
4. Apply the same 16-voxel constant spatial padding used in training.
5. Normalize at the same point as training, currently after padding.
6. Do not build annotation-centred training samples.

Initially use the exact training implementation: per-timepoint
`skimage.transform.resize` with `order=1`, `mode="reflect"`,
`clip=True`, and `preserve_range=True`. Record the effective anti-aliasing
behavior and library version. A GPU or crop-local resampler should not replace
this path until numerical crop parity has been demonstrated.

Cache the processed movie by movie hash plus preprocessing-specification hash
so that multiple center groups or checkpoints with the same profile do not
repeat preprocessing.

### 2.2 Coordinate systems and center transformation

All spatial transformations should operate in internal `X,Y,Z` order. Movie
index `M` and time `T` remain unchanged.

Let:

- `s_raw` be raw voxel spacing in micrometres per raw voxel;
- `s_reg` be checkpoint regression-grid spacing;
- `a = s_raw / s_reg` be regression voxels per raw voxel.

The existing training convention is:

```text
p_reg = round(p_raw * a + 1e-9)
N_reg = floor(N_raw * a)
```

The first implementation should preserve that convention because annotations
and existing checkpoints were trained with it. Record the realized
target-shape/raw-shape ratio separately; do not silently replace the legacy
coordinate convention with a different half-pixel affine convention.

For detector centers:

```text
raw detector center (M,T,Xraw,Yraw,Zraw)
-> transform XYZ only
-> regression center (M,T,Xreg,Yreg,Zreg)
-> extract the 32-cubed crop from the processed movie
```

The center passed to crop extraction remains an unpadded regression-space
coordinate. `crop_img_from_center()` already accounts for spatial padding;
adding 16 again would introduce an offset.

Retain the original raw detector center in the prediction record. Store the
regression-space crop center separately. This prevents transformed coordinates
from leaking into segmentation metrics, matching, display, or exported
detections.

For exact training-crop parity at an annotated event, transform both annotated
bipoint endpoints using the existing endpoint rule and take their midpoint,
because that is how the training crop center is constructed. For an arbitrary
detector centroid, transform the center itself and quantize once at crop
extraction.

### 2.3 Orientation and length decoding

Interpret the network output first in regression space:

- `L_reg = 32 * L_normalized`, in regression voxels;
- `u_reg`, the unit division axis in regression-grid XYZ.

For a half-axis vector:

```text
d_reg = (L_reg / 2) * u_reg
d_physical = s_reg * d_reg
d_raw = d_reg / a
```

Export explicit quantities:

- physical length `L_physical = 2 * norm(d_physical)`;
- physical unit axis;
- raw endpoints `center_raw +/- d_raw`;
- raw-coordinate axis and raw-voxel length where needed for visualization;
- regression-space length and axis for checkpoint diagnostics.

For the current isotropic target spacing of `1 um`, a 32-cubed crop has a
nominal `32x32x32 um` field of view. In the neural-tube raw inference path,
the same array currently covers only approximately
`6.66x6.66x32 um`, explaining the severity of the mismatch.

Do not apply anisotropic scaling directly to a quaternion or rotation matrix.
Convert it to the division-axis vector, transform that vector, and renormalize.
If a raw-coordinate quaternion must be retained for compatibility, reconstruct
a canonical roll-free quaternion from the transformed axis. Rotation around
the division axis remains scientifically meaningless.

Replace the unused/stubbed `Regress3Dataset.unscale_prediction()` concept with
this explicit decoder rather than reviving an ambiguous scalar conversion.

### 2.4 Ground-truth separation and event identity

Regression evaluation must use stable event identities:

```text
matched true annotation/event ID
|-- annotated endpoints -> immutable evaluation target
'-- detector center -> crop-selection location only
```

Do not locate ground truth by searching near the detector-predicted center
after resampling. Refactor `CenterList` records to retain:

- movie and time;
- annotation/event index;
- raw annotated center and endpoints;
- raw detector-predicted center;
- matching index.

The same annotated endpoints must be used as the target for matched-true-center
and predicted-center evaluations.

This is required because the current `gather_groundtruth_info()` performs a
proximity lookup. Once annotations and images occupy regression space while
segmentation information remains raw, that lookup becomes unsafe.

## 3. Files and functions likely to require modification

- `dare3d/data/components/abstract_celldataset.py`
  - expose immutable geometry and reusable resampling metadata.
- `dare3d/data/components/regress_3dataset.py`
  - regression inference initialization, coordinate transforms, crop
    extraction, and prediction decoding.
- A new small module such as
  `dare3d/data/components/regression_geometry.py`.
- `dare3d/metrics/inference.py`
  - accept raw centers explicitly, transform only for crop selection, and
    return coordinate- and unit-explicit predictions.
- `dare3d/predict.py` and `dare3d/eval.py`
  - use training-consistent regression initialization while leaving
    segmentation unchanged.
- `dare3d/metrics/infer_measure.py`
  - identity-based target association and coordinate-explicit metrics.
- `dare3d/utils/regression_display.py`
  - draw decoded raw endpoints rather than interpreting regression length as
    raw voxels.
- `napari_dare3d/_api.py`
  - use checkpoint-specific regression geometry and expose raw and physical
    output quantities.
- Prediction/evaluation configuration
  - strict training-consistent mode, explicit legacy mode, and separate
    segmentation/regression scale overrides.
- New geometry, crop-parity, target-association, and integration tests.

## 4. Alternative solutions considered

### 4.1 Crop a physical raw-image region and resample only that crop

This reduces memory and may be faster. However, whole-volume resizing and
crop-local resizing do not necessarily commute. Anti-aliasing, boundary
behavior, normalization, and pixel-center conventions can change the input.
Consider this only after it reproduces the preferred implementation
numerically.

### 4.2 Require callers to pre-resample images and supply regression coordinates

This is simple internally but makes the public API ambiguous, risks double
transformations, and makes CLI and napari use error-prone.

### 4.3 Perform segmentation or connected-component detection in regression space

Reject this because it can change centroids, matching, TP/FP/FN, and
segmentation metrics. Segmentation is outside the fault and must remain fixed.

### 4.4 Make the network spacing-aware or retrain on native-resolution crops

This may be attractive long-term, but requires retraining and does not provide
direct compatibility with released checkpoints.

### 4.5 Transform centers without resampling the regression image

This is insufficient because the physical field of view and image-frequency
content remain wrong.

## 5. Validation strategy without retraining

Reuse existing weights, saved segmentation probabilities, matching records,
and detected centers. Do not rerun training or segmentation.

### 5.1 Geometry and synthetic tests

- Test raw-to-regression and regression-to-raw mappings using impulses and
  monotonic XYZ ramps to detect axis swaps.
- Verify neural-tube `1024x1024x10 -> 212x212x10`, followed by 16 voxels of
  padding per spatial side, for the audit retraining profile.
- Separately verify the released original checkpoint's archived
  `635x635x20` profile.
- Test central, border, half-integer, odd-shaped, and low-Z cases.
- Verify position round trips within the expected quantization tolerance.
- Verify that `M` and `T` never undergo spatial scaling.
- Verify disk `T,Z,Y,X`, internal `T,X,Y,Z`, center `M,T,X,Y,Z`, and
  napari `T,Z,Y,X` conversions independently.

### 5.2 Training/inference crop equivalence

For representative annotated events:

1. Obtain the crop through the ordinary training `Dataset.init()` path.
2. Obtain the crop through the new inference preparation and center-transform
   path using the same preprocessing specification.
3. Compare shape, dtype, normalization range, voxel values, and hashes.

Require identical hashes where both paths call the same implementation.
Otherwise require a predeclared floating-point tolerance and record the maximum
absolute error.

Include central and border events from neural tube and nuclei. For detector
centers, compare against an independently constructed reference crop using the
same transformed center.

### 5.3 Neural-tube stress test

Use cached detector and matching artifacts.

- On `movie_I2`, evaluate all-ground-truth, matched-ground-truth, and the 114
  saved detector-predicted centers.
- For the retrained checkpoint, use its `212x212x10` profile. Matched
  ground-truth results should reproduce the saved training-consistent result.
- Run detector-predicted centers through the corrected transform and determine
  whether the strong retrained-model improvement is retained after the
  localization penalty.
- Compare against the current raw-path retrained predicted-center angular mean
  of approximately `45.9 degrees`; require a clear paired improvement rather
  than a selected subset or changed matching rule.
- Evaluate the released original checkpoint using its own archived
  `635x635x20` profile for deployment compatibility.
- Also evaluate the original checkpoint in the common `212x212x10` space,
  but label this explicitly as a cross-profile scientific diagnostic rather
  than its native checkpoint deployment path.
- Reproduce the existing `movie_M` N=80 controlled, training-consistent
  results without retraining. The unchanged detector produced no movie_M
  candidates, so predicted-center performance there remains not applicable
  unless a valid unchanged-protocol center artifact becomes available.

Report corrected nematic angular error, length error in regression voxels and
physical units, crop-center displacement, and paired event distributions.

### 5.4 Nuclei compatibility test

- Reuse saved nuclei detections and matches.
- Use the archived log-reconstructed preprocessing profile, including raw
  `362x305x180` to historical `330x278x164` for movie2.
- Compare legacy-raw and corrected generic regression paths event by event.
- Predeclare an equivalence margin before inspecting new results, for example
  an upper paired 95% interval below `+2 degrees` for mean nematic error and
  below one regression voxel for length error.
- Classify the result as improved, equivalent, or degraded; do not tune scales
  to obtain equivalence.
- Because the historical nuclei `scales.json` is absent, label this geometry
  as log-reconstructed rather than verified physical calibration.

### 5.5 Target-association and non-regression invariants

- Hash segmentation probability maps, detected centers, matching indices,
  TP/FP/FN counts, and segmentation metrics before and after the change. They
  must be identical.
- Assign every annotation a stable event ID and verify identical target
  endpoint hashes across all-GT, matched-GT, and predicted-center modes.
- Perturb a detector center and confirm that only the crop changes, never the
  associated ground truth.
- Verify that matching counts and evaluated event sets are identical across
  regression checkpoints.
- Overlay decoded raw endpoints on raw images and verify numerical endpoint
  round trips.
- Retain a `legacy_raw` golden-output test that reproduces current predictions
  when explicitly selected.

### 5.6 Acceptance criteria

The corrected path is ready for default use only if:

- training/inference crop parity passes;
- segmentation and matching artifacts remain byte-identical;
- no ground-truth event association changes;
- existing original and retrained checkpoints both load unchanged;
- the neural-tube retrained checkpoint recovers its training-space performance
  at true centers and retains a strong benefit at detector centers;
- nuclei results improve or meet the predeclared equivalence margins;
- output centers, axes, endpoints, and lengths carry explicit coordinate-space
  and unit metadata.

A failed criterion should trigger diagnosis and reporting, not scale,
threshold, matching, or parameter tuning.

## 6. Risks and backward compatibility

- Existing weights remain loadable; this is a preprocessing and decoding
  change, not an architecture change.
- Numerical regression results can legitimately change, especially for neural
  tube and to a lesser extent nuclei.
- Missing legacy scale files make exact checkpoint geometry uncertain. Strict
  mode must stop rather than guess.
- Original and retrained neural-tube checkpoints have different recorded
  preprocessing profiles. Reports must state whether evaluation uses each
  checkpoint's own profile or a common diagnostic profile.
- Historical `length` fields have ambiguous units. Introduce explicit
  `length_regression_voxels`, `length_physical_um`, raw endpoints, and
  coordinate-space metadata. Retain deprecated aliases only with documented
  compatibility behavior.
- Full-movie resampling may increase memory for some anisotropic inputs.
  Optimize or stream only after crop-equivalence tests pass.
- Scikit-image version and implicit anti-aliasing behavior may affect exact
  legacy reproduction and must be captured in the preprocessing manifest.
- A generic fix may change nuclei numbers even though the geometric mismatch
  is smaller. This must be measured and reported rather than assumed harmless.

## 7. Recommended implementation order

1. Freeze preprocessing provenance for each existing checkpoint.
2. Define the preprocessing-specification, spatial-transform, and result
   schemas.
3. Add pure coordinate-transform and prediction-decoding utilities with
   synthetic tests.
4. Add training-equivalent regression inference initialization.
5. Establish crop parity before routing any production caller through it.
6. Route CLI prediction and evaluation through the corrected regression path,
   leaving segmentation untouched.
7. Refactor evaluation to use immutable event identities and raw detector
   centers.
8. Update regression display and napari export to use decoded raw endpoints.
9. Run the cached neural-tube movie_I2 checkpoint/center validation matrix.
10. Reproduce the controlled movie_M regression results.
11. Run the nuclei compatibility analysis.
12. Make training-consistent inference the strict default only after all
    invariants pass; retain explicit `legacy_raw` behavior and publish
    migration notes.
