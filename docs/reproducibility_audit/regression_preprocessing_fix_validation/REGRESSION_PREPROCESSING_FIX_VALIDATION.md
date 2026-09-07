# Generic regression preprocessing fix: no-retraining validation

This validation did not run segmentation or training. Detector centers,
matches, annotations, architectures, and checkpoints were frozen; only
regression image-space preparation and coordinate decoding changed.

## Crop parity

| Checkpoint profile | Crops | Same-grid exact | Raw-midpoint exact | Maximum mapped-center offset (regression voxels) |
|---|---:|---:|---:|---:|
| nematic_retrained_native_0208 | 123 | 123 | 49 | 1.732 |
| released_original_native_0621 | 123 | 123 | 62 | 1.414 |

The full preprocessed movies and every crop extracted at the same
regression-grid center must be byte-identical. Raw annotation
midpoints can differ by endpoint-rounding quantisation; this is
reported rather than hidden.

## Neural tube (movie_I2 frozen centers)

movie_I2 is the historical validation/reported set, not an
independent test.

| Profile | Center mode | N | Corrected nematic error, mean +/- population SD (deg) | Length MAE (physical units) |
|---|---|---:|---:|---:|
| nematic_retrained_native_0208 | all_groundtruth_centers | 122 | 14.202 +/- 7.800 | 0.638 |
| nematic_retrained_native_0208 | matched_groundtruth_centers | 114 | 14.276 +/- 7.567 | 0.601 |
| nematic_retrained_native_0208 | predicted_centers | 114 | 14.347 +/- 7.102 | 0.649 |
| released_original_common_0208 | all_groundtruth_centers | 122 | 47.800 +/- 24.610 | 10.615 |
| released_original_common_0208 | matched_groundtruth_centers | 114 | 47.786 +/- 24.797 | 10.589 |
| released_original_common_0208 | predicted_centers | 114 | 47.351 +/- 24.578 | 10.681 |
| released_original_native_0621 | all_groundtruth_centers | 122 | 19.245 +/- 13.709 | 1.615 |
| released_original_native_0621 | matched_groundtruth_centers | 114 | 19.169 +/- 13.681 | 1.547 |
| released_original_native_0621 | predicted_centers | 114 | 19.293 +/- 14.568 | 1.536 |

At the same saved detector-predicted centers and the same
0.208/0.208/1 regression grid, the retrained-minus-original mean
angular difference is
-33.004 degrees.

## Nuclei (movie2 frozen centers)

The training-space geometry is log-reconstructed because the
historical scales.json is unavailable.

| Checkpoint | Center mode | Legacy raw mean (deg) | Corrected-path mean (deg) | Paired change (deg) |
|---|---|---:|---:|---:|
| original_epoch_098 | all_groundtruth_centers | 14.894 | 13.671 | -1.223 |
| original_epoch_098 | matched_groundtruth_centers | 14.923 | 12.841 | -2.082 |
| original_epoch_098 | predicted_centers | 19.672 | 19.740 | 0.068 |
| retrained_nematic_epoch_095 | all_groundtruth_centers | 12.575 | 11.736 | -0.839 |
| retrained_nematic_epoch_095 | matched_groundtruth_centers | 12.236 | 11.011 | -1.226 |
| retrained_nematic_epoch_095 | predicted_centers | 19.447 | 19.806 | 0.359 |

## Invariants and interpretation

- Overall validation status: **complete**.
- Segmentation probabilities, thresholds, detections, and matching
  were not recomputed; saved center lists are validator inputs.
- A detector-predicted center selects only the regression crop. The
  target is the annotation associated with its matched true event.
- Existing checkpoints load unchanged; no architecture or weight
  conversion is performed.
- Per-event values, preprocessing manifests, checkpoint hashes,
  and resumable prediction caches accompany this report.
