# DARE3D nematic division-axis metric comparison

## Scope and result

This is an audit-only, fixed-prediction comparison. No production source,
checkpoint, released prediction, annotation, matching decision, threshold,
dataset split, manuscript file, or model parameter was changed. The only
counterfactual change is the angular error definition.

The manuscript's two `28 +/- 2 degrees` statements do not represent the released
mean angular errors. The central value instead tracks the released population
standard deviations: 28.018 degrees for nuclei and 27.104 degrees for membrane.
Using a scientifically appropriate nematic-axis error and expressing the result
coherently as mean +/- SEM on the same all-ground-truth-center samples gives:

- Nuclei: **14.89 +/- 1.07 degrees** (population SD 12.70 degrees,
  effective N=140; production reports nominal N=145).
- Membrane: **36.63 +/- 2.00 degrees** (population SD 22.13 degrees,
  N=122).

Only predicted-center mode is end-to-end. Its corrected mean/SD is
19.67/21.79 degrees for nuclei (effective N=121) and 36.03/19.97 degrees
for membrane (N=114).

## Metric definitions

Current production evaluation:

$$
\theta_q =
2\arccos\left(\left|q_{\rm pred}\cdot q_{\rm true}\right|\right).
$$

The quaternions are normalized, then the un-absolute scalar product is clipped
to `[-0.9999, 0.9999]` in
`dare3d/losses/angle3d.py:106-116`. This evaluates the complete encoded
rotation, including changes that do not change the rendered division axis, and
gives identical quaternions a 1.620583-degree floor.

Corrected audit-only evaluation:

$$
\theta_{\rm axis} =
\arccos\left(
\left|
\hat{\mathbf u}_{\rm pred}\cdot
\hat{\mathbf u}_{\rm true}
\right|
\right).
$$

The axes are normalized, the absolute scalar product is clipped only to the
exact valid interval `[0,1]`, and the result is converted to degrees. The
absolute value enforces the nematic equivalence
$\hat{\mathbf u}\equiv-\hat{\mathbf u}$.

## Side-by-side results

`N` is shown as production-reported/effectively evaluated. Production computes
the means and population SDs after skipping missing pairs, but records the
pre-skip input-pair count as `n`. Relative changes use the current production
value as denominator. Negative relative changes mean that the corrected value
is lower.

| Dataset | Evaluation mode | N | Current mean | Corrected mean | Absolute mean difference (relative change) | Current SD | Corrected SD | Absolute SD difference (relative change) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Nuclei | All GT centers | 145/140 | 25.306 | 14.894 | 10.412 (-41.1%) | 28.018 | 12.700 | 15.318 (-54.7%) |
| Nuclei | Matched GT centers | 124/121 | 26.309 | 14.923 | 11.386 (-43.3%) | 32.020 | 13.518 | 18.502 (-57.8%) |
| Nuclei | Predicted centers (end-to-end) | 124/121 | 37.469 | 19.672 | 17.797 (-47.5%) | 45.579 | 21.789 | 23.791 (-52.2%) |
| Membrane | All GT centers | 122/122 | 49.566 | 36.634 | 12.932 (-26.1%) | 27.104 | 22.126 | 4.978 (-18.4%) |
| Membrane | Matched GT centers | 114/114 | 48.091 | 35.479 | 12.612 (-26.2%) | 25.732 | 21.469 | 4.264 (-16.6%) |
| Membrane | Predicted centers (end-to-end) | 114/114 | 51.367 | 36.032 | 15.335 (-29.9%) | 27.633 | 19.965 | 7.668 (-27.7%) |

All values are degrees. Mean and SD use the same float32 aggregation and
population-SD convention as production.

## Distribution statistics

The released `stats.csv` files do not retain event-level errors, medians, RMS,
percentiles, or maxima. Current nuclei distribution values below are therefore
from the epoch-098 checkpoint replay, whose released mean/SD aggregates are
reproduced to within 0.016 degrees. Current membrane distribution values are
raster-derived proxies because raw daughter annotations are absent. Corrected
values are the primary audit results.

| Dataset/mode | Current median proxy | Corrected median | Current RMS proxy | Corrected RMS | Corrected SEM | Corrected p95 | Corrected max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nuclei/all GT | 17.735 | 12.342 | 37.756 | 19.573 | 1.073 | 36.168 | 83.291 |
| Nuclei/matched GT | 17.488 | 12.195 | 41.442 | 20.135 | 1.229 | 37.935 | 83.291 |
| Nuclei/predicted | 15.203 | 10.106 | 59.024 | 29.355 | 1.981 | 76.651 | 88.875 |
| Membrane/all GT | 40.527 | 32.222 | 57.052 | 42.798 | 2.003 | 83.600 | 89.075 |
| Membrane/matched GT | 40.289 | 29.844 | 55.177 | 41.469 | 2.011 | 83.160 | 89.075 |
| Membrane/predicted | 44.102 | 32.620 | 58.912 | 41.193 | 1.870 | 77.000 | 89.396 |

The manuscript invokes a 61.2-degree RMS baseline for a random 3D nematic
axis. The corrected all-GT RMS remains below that baseline for nuclei
(19.57 degrees) and membrane (42.80 degrees).

## What replaces the manuscript's quoted result

The manuscript says "standard deviation ... 28 +/- 2 degrees" while labeling
the expression as mean +/- SD/sqrt(n). Those descriptions cannot both be true.

The released all-GT statistics establish the provenance:

| Dataset | Released current mean | Released current population SD | SD/sqrt(reported N) | Manuscript text |
|---|---:|---:|---:|---:|
| Nuclei | 25.306 | 28.018 | 2.327 | 28 +/- 2 |
| Membrane | 49.566 | 27.104 | 2.454 | 28 +/- 2 |

Thus, if the intended reporting convention is mean +/- SEM, the coherent
nematic-axis replacements are:

- Nuclei: **14.89 +/- 1.07 degrees**, SD 12.70, effective N=140.
- Membrane: **36.63 +/- 2.00 degrees**, SD 22.13, N=122.

If the intended statistic was instead "population SD +/- an uncertainty on the
SD", the manuscript does not define or compute that uncertainty, and
SD/sqrt(n) is not generally the standard error of the sample SD. The audit does
not invent a replacement for an undefined statistic.

## Synthetic validation and clipping

| Case | Corrected error | Current production behavior |
|---|---:|---:|
| Identical axes | exactly 0 | identical quaternions have a 1.620583-degree floor |
| Opposite vectors for the same nematic axis | exactly 0 | not the evaluated physical equivalence |
| Orthogonal axes | exactly 90 | depends on complete quaternion representation |
| Same axis, quaternion rotation changed from 30 to 150 degrees | exactly 0 | 120 degrees |

A normalized vector's self-dot product was also observed as
`1.0000000000000002` through floating-point roundoff. Clipping
`abs(dot)` to exactly `[0,1]` returned exactly 0 degrees. No epsilon margin is
mathematically necessary, and an epsilon margin would reintroduce an artificial
non-zero floor.

All seven synthetic assertions pass.

## Scientific interpretation

- Nuclei: the all-GT corrected mean falls below the manuscript's approximate
  20-degree annotation-uncertainty estimate. The statement that performance is
  at the annotation scale is strengthened. The end-to-end corrected mean is
  also approximately 20 degrees.
- Membrane: the all-GT corrected mean is 36.63 degrees, not approximately
  20 degrees. On the released evidence, the statement that membrane mean error
  is at the estimated annotation-uncertainty scale is not supported.
- Both DARE3D corrected RMS values remain below the manuscript's 61.2-degree
  random nematic-axis baseline, so the below-random conclusion remains true.
- The corrected 3D errors remain much larger than the manuscript's reported 2D
  angular scale, so the qualitative 2D-versus-3D ranking does not reverse.
- Detection F1, TP/FP/FN, centers, matches, lengths, rendered prediction axes,
  qualitative figures, and movies do not change.

The last point is important: this evaluates the same trained models. The
full-quaternion function is also used as the angular training loss and
contributes to `val/loss`, which selected checkpoints. Retraining with a
nematic loss could change the predictions and checkpoint, but that is a
different experiment and was intentionally not performed here.

## Affected manuscript locations

Direct numerical claims:

- `manuscript_280826version/main.tex:395`, Results: 3D nuclei model.
- `manuscript_280826version/main.tex:402`, Results: 3D membrane model.

Interpretive claims:

- `main.tex:79`, Abstract: orientation accuracy approaches annotation
  uncertainty.
- `main.tex:481`, Discussion: 3D nuclei orientation error.
- `main.tex:493-494`, Discussion: 3D membrane orientation error.
- `main.tex:501-505`, Discussion: 2D versus 3D regression and the random-axis
  RMS comparison.

No numerical DARE3D angular-performance result was found in a table, figure, or
supplementary caption:

- `main.tex:178-212`, Table `dataset:stats_all`, supplies test-set and division
  length provenance but not angular performance.
- `main.tex:645-656`, Figs. `fig:regression` and
  `fig:regression_neuraltube`, are qualitative; their rendered axes do not
  change.
- `main.tex:672-675`, Fig. `fig:3Dsuccess`, is qualitative and detection-focused.
- `main.tex:572-594`, supplementary movie captions, contain displayed
  orientations but no aggregate angular statistic. The underlying axes do not
  change.

## Full provenance

Production source:

- `dare3d/losses/angle3d.py:106-116` defines
  `quaternion_error`.
- `dare3d/metrics/infer_measure.py:372-411` calls it and produces the three
  center-mode mean/SD records.
- `dare3d/data/components/angles3d.py:265-290` extracts the quaternion vector
  part as the rendered axis.
- `dare3d/metrics/inference.py:144-176` converts predictions to quaternions and
  saves/render them.
- `dare3d/losses/angle3d.py:70-84` and
  `dare3d/models/regression_module.py:83-186` show that the same full-rotation
  error also affects training/validation loss.
- `configs/experiment/regression.yaml:43` monitors `val/loss` for checkpoint
  selection.

Released aggregate sources:

- `DARE3d_data_190326/Gastruloid_241025/weights/segmentation3d_exp10-b/runs/01-01/stats.csv`.
- `DARE3d_data_190326/Neural_tube_160226/segmentation3d_new_set_og/runs/12-01-26/stats.csv`.

Corrected nuclei inputs:

- Released daughter-label volume
  `Gastruloid_241025/trainingset/movie2/label/movie2.tif`.
- Frozen production center/match evidence
  `evidence/nuclei_current_evaluator_objects.csv` and
  `evidence/nuclei_saved_visualization_alignment.csv`.
- Epoch-098 raw quaternion compatibility outputs under
  `checkpoint_runs/nuclei_regression/best_epoch_098/`.
- Archived rendered predictions under the three released regression mode
  directories, retained as a sensitivity path.

Corrected membrane inputs:

- Released prediction quaternions in each mode's `raw_predictions.npz`.
- Released `groundtruth/movie_I2.tif` axis raster.

Reproducible outputs from this test:

- `nematic_axis_metric_comparison.py`: independent evaluator and assertions.
- `evidence/nematic_axis_metric_comparison.json`: full definitions, comparisons,
  provenance, limitations, and assertion results.
- `evidence/nematic_axis_metric_comparison.csv`: six side-by-side summary rows.
- `evidence/nematic_axis_event_metrics.csv`: 743 event rows, including explicit
  skipped nuclei events and all axis vectors/errors.
- This report.

The new implementation independently reproduced all 732 earlier saved-output
nematic errors to a maximum absolute difference of
`1.23e-13` degrees.

## Limitations kept separate

- Original nuclei raw prediction quaternions were not released. The primary
  corrected counterfactual therefore uses the independently verified epoch-098
  checkpoint replay. That replay reproduces all six released current-metric
  means/SDs with a maximum discrepancy of 0.015965 degrees. The archived-raster
  sensitivity gives all-GT mean/SD 15.185/12.447 degrees, close to the primary
  14.894/12.700 degrees.
- Neural raw images and daughter annotations are absent. The membrane truth
  axis is decoded from the saved uint8 line raster. In predicted-center mode,
  37 of 114 comparisons require the already documented next-frame fallback.
  The current-metric raster proxy differs from released means by
  0.304-0.333 degrees, quantifying but not eliminating this limitation.
- Released current medians and event-level errors are absent. Current medians
  and RMS values in this report are explicitly labeled replay/proxy values.
- Production reports nominal N=145/124/124 for nuclei while actually evaluating
  140/121/121 angle pairs. Corrected SEM uses effective N.
- Voxel anisotropy, regression-at-GT versus regression-at-predicted-center
  semantics, split leakage, matching behavior, and the membrane dataset identity
  conflict remain as documented in the main audit. None was changed to improve
  agreement.

## Concise conclusion

**Current metric:** `2*acos(|q_pred dot q_true|)` with +/-0.9999 clipping.
The manuscript reports 28 +/- 2 degrees for both DARE3D models, but those values
trace to released population SDs, not released means.

**Corrected metric:** `acos(|u_pred dot u_true|)` with exact `[0,1]` safety
clipping. All-GT mean +/- SEM is 14.89 +/- 1.07 degrees for nuclei and
36.63 +/- 2.00 degrees for membrane.

**Difference:** Relative to released production means, the all-GT mean decreases
by 10.41 degrees (41.1%) for nuclei and 12.93 degrees (26.1%) for membrane.
The end-to-end mean decreases by 17.80 degrees (47.5%) and 15.33 degrees
(29.9%), respectively.

**Scientific interpretation:** The nuclei annotation-limit claim is strengthened;
the membrane approximately-20-degree claim is not supported; both remain better
than the random nematic-axis baseline.

**Affected manuscript locations:** `main.tex:79`, `395`, `402`, `481`,
`493-494`, and `501-505`. No quantitative table, figure, or supplementary
result changes.

**Recommended code change:** Add a unit-axis extractor and a
`nematic_axis_error` helper, then replace only the `quaternion_error` call in
`evaluate_center_pair` for reported evaluation statistics. Clip
`abs(dot(u_pred,u_true))` only to `[0,1]`. Treat changing the training loss and
retraining/checkpoint selection as a separate experiment. No production change
has been applied.
