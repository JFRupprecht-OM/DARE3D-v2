# DARE3D nuclei retraining with a nematic-axis loss

## Scope and outcome

This is an isolated audit experiment. The released model, production source,
data, manuscript, matches, and saved results were not changed. A new nuclei
regression model was initialized from scratch and trained for 100 epochs with
the released architecture and protocol as closely as the surviving artifacts
permit. The only intended scientific change was the orientation objective.

On the frozen manuscript all-ground-truth-center evaluation, changing only the
evaluation metric reduces the original model's mean angular error from 25.307
to 14.894 degrees. Retraining with the nematic loss reduces it further to
12.575 degrees. The retraining effect is -2.319 degrees (-15.6%) relative to
the original model under the corrected metric. Its paired descriptive bootstrap
95% interval is -4.406 to -0.457 degrees.

That improvement is real for this frozen comparison but is not an independent
generalization estimate: released `movie2` is both validation and test, the
missing historical `scales.json` had to be reconstructed from logged image
shapes, and only one seed was trained. In predicted-center end-to-end mode, the
retraining effect is only -0.225 degrees with an interval spanning zero.

## Requested three-scenario comparison

These are the manuscript-compatible all-ground-truth-center results. Production
reports nominal N=145, but only 140 pairs have usable ground truth and enter all
angular aggregates. SD is the population SD, matching production aggregation.

| Scenario | Angular metric | N (reported/effective) | Mean | Median | SD | RMS | p95 | Maximum | Length MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1. Original model + original metric | full quaternion | 145/140 | 25.307 | 17.735 | 28.019 | 37.756 | 70.441 | 178.121 | 1.944 voxels |
| 2. Original model + corrected nematic metric | axis | 145/140 | 14.894 | 12.342 | 12.700 | 19.573 | 36.168 | 83.291 | 1.944 voxels |
| 3. Retrained model + corrected nematic metric | axis | 145/140 | 12.575 | 9.409 | 11.126 | 16.791 | 29.255 | 89.940 | 1.896 voxels |

All angular values are degrees. The original-model production replay differs
from the released mean of 25.306 degrees by about 0.001 degrees; all released
aggregate checks remain within the predeclared 0.02-degree tolerance.

The numerical effects on the mean are:

| Counterfactual | Absolute change | Relative change |
|---|---:|---:|
| Correct evaluation metric only: scenario 2 minus scenario 1 | -10.413 degrees | -41.1% |
| Corrected retraining: scenario 3 minus scenario 2 | -2.319 degrees | -15.6% |
| Both changes: scenario 3 minus scenario 1 | -12.732 degrees | -50.3% |

The manuscript's `28 degrees +/- 2 degrees` is not the original mean: 28.018
degrees is the released population SD, and approximately 2 degrees is
SD/sqrt(N). A coherent mean +/- SEM expression for scenario 3 is therefore
12.57 +/- 0.94 degrees (SD 11.13, effective N=140), not 28 +/- 2 degrees.

## Controlled unique holdout

The strongest like-for-like model comparison uses each of the 156 unique
`movie2` daughter pairs once under the reconstructed historical training
preprocessing. It avoids the manuscript evaluator's repeated/missing center
bookkeeping, although it still uses the same data that selected the checkpoint.

| Scenario | Mean | Median | SD | RMS | p95 | Maximum | Length MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original + original metric | 25.730 | 16.863 | 31.288 | 40.508 | 74.830 | 178.643 | 1.565 |
| Original + corrected metric | 13.655 | 11.143 | 11.098 | 17.596 | 31.920 | 89.943 | 1.565 |
| Retrained + corrected metric | 10.893 | 8.768 | 7.820 | 13.409 | 26.088 | 45.090 | 1.502 |

For scenario 3 minus scenario 2, the paired mean difference is -2.762 degrees
(-20.2%), with a descriptive bootstrap 95% interval of -4.576 to -1.162
degrees. The paired median difference is -1.308 degrees (interval -3.061 to
-0.248), and 59.6% of events have lower error after retraining. Length MAE
changes by -0.063 voxels (interval -0.205 to 0.082).

## All frozen manuscript center modes

The same checkpoints were also replayed without changing the production
predictions, ground truth, center sets, filtering, matching, or aggregation.

| Center mode | Effective N | Original + original mean/SD | Original + corrected mean/SD | Retrained + corrected mean/SD | Retraining mean difference (relative) | Paired mean 95% interval |
|---|---:|---:|---:|---:|---:|---:|
| All GT | 140 | 25.307 / 28.019 | 14.894 / 12.700 | 12.575 / 11.126 | -2.319 (-15.6%) | [-4.406, -0.457] |
| Matched GT | 121 | 26.310 / 32.020 | 14.923 / 13.518 | 12.236 / 11.645 | -2.687 (-18.0%) | [-4.765, -0.875] |
| Predicted centers (end-to-end) | 121 | 37.482 / 45.595 | 19.672 / 21.789 | 19.447 / 23.442 | -0.225 (-1.1%) | [-1.809, 1.405] |

The paired intervals are descriptive only because checkpoint selection used the
same movie. The all-GT median difference interval includes zero and only 51.4%
of events improve, indicating that its lower mean is mainly a tail improvement,
not a uniform event-wise shift. Matched-GT behaves similarly. Predicted-center
performance does not materially improve.

## Angular-error distributions

The complete histograms and empirical CDFs are in `angular_distribution.csv`.
For the manuscript all-GT set, the cumulative fractions are:

| Scenario | <=10 degrees | <=20 degrees | <=30 degrees | <=45 degrees |
|---|---:|---:|---:|---:|
| Original + original metric | 22.1% | 57.9% | 77.9% | 88.6% |
| Original + corrected metric | 38.6% | 81.4% | 92.1% | 96.4% |
| Retrained + corrected metric | 54.3% | 80.7% | 95.7% | 98.6% |

For the controlled 156-event set, the corrected retrained model raises the
fraction at or below 10 degrees from 44.2% to 56.4%, at or below 30 degrees from
92.9% to 97.4%, and at or below 45 degrees from 98.1% to 99.4%. Its maximum
drops from 89.94 to 45.09 degrees. In the manuscript raw-coordinate path, one
retrained outlier reaches 89.94 degrees even though p95 improves from 36.17 to
29.26 degrees.

## Orientation representation and loss

The released model does not directly output an axis:

1. The network emits nine unconstrained orientation values.
2. Symmetric SVD orthogonalization projects them to SO(3).
3. Ground truth is a flattened 3x3 rotation matrix derived from the archived
   quaternion encoding of the two annotated daughter positions.
4. Production converts matrices to normalized wxyz quaternions. The normalized
   direction of quaternion components `(x,y,z)` is the rendered division axis.
5. Production training uses
   `2*acos(abs(q_pred dot q_true))` in degrees after clipping to
   `[-0.9999,0.9999]`. This angular term is added to the unchanged length term
   `32*mean(abs(normalized_length_error))`; their sum is `val/loss` and selects
   the checkpoint.

The audit-only training objective replaces only that angular term with

`mean(90 * (1 - (u_pred dot u_true)^2))`.

Here `u_pred` and `u_true` are normalized quaternion-vector axes. Squaring the
dot product enforces `u == -u`; using only axis directions removes sensitivity
to the artificial quaternion rotation angle around that axis. The factor 90
keeps the smooth surrogate on the same 0--90 numerical scale as the physical
axis angle. Exact evaluation remains

`degrees(acos(abs(u_pred dot u_true)))`,

with the absolute dot clipped only to the exact interval `[0,1]`. Directly
optimizing `acos` was not used because its endpoint derivative generated
non-finite gradients in preflight; the smooth projector surrogate has the same
nematic minima.

Synthetic preflight checks all pass: identical and opposite axes give exactly
0 degrees, orthogonal axes give exactly 90 degrees, and changing only the
quaternion rotation angle while holding the division axis fixed leaves both
the loss and evaluation unchanged. Random-batch gradients are finite.

As an expected diagnostic, the retrained model scores 95.925 degrees under the
obsolete full-quaternion metric in all-GT mode. The nematic loss deliberately
does not constrain the irrelevant quaternion rotation angle, so this is not a
physical degradation; it illustrates why that production metric must not be
used to assess a nematic-axis model.

## Frozen training protocol

| Item | Value |
|---|---|
| Seed | 12345 |
| Split | train: movie3 + movie4; validation=test: movie2 |
| Unique crops | 526 train; 156 validation/test |
| Samples per epoch | 2,000 train and 2,000 validation, preserving modulo repetition |
| Input/crop | frames -1,0,+1; 32x32x32; whole-movie min-max |
| Architecture | 3-channel, 5-stage, 16-start-filter RegressionNet; 5,897,940 parameters |
| Optimizer | AdamW, lr 0.001, betas 0.9/0.999, weight decay 1e-4 |
| Scheduler | OneCycleLR, 167 configured steps/epoch, 100 epochs, pct_start 0.1; historically stepped per epoch |
| Augmentation | archived 3D flip, rotation, and zoom settings |
| Runtime | CUDA, precision 32, nondeterministic as archived |

The absent historical `scales.json` was not invented. Exact target shapes were
reconstructed from the archived training log (movie2 330x278x164, movie3
387x375x77, movie4 405x394x152), and the saved config records target/native
ratios and SHA-256 hashes of every image and label.

The fresh run completed 100 epochs and 16,700 optimizer steps in 3,338.20
seconds with no logged NaN/Inf. The validation minimum is epoch 95: total loss
6.099831 = orientation 4.596163 + length 1.503670. The selected checkpoint hash
is `e3bc5a3ff81497d36606ee4ca57c83395eef5f02b0ffd3cf5f117d0bbb1f51ec8`.
The initial random network-state hash is retained in `run_state.json` as evidence
of fresh initialization.

With Lightning 2.6.5, `last.ckpt` is an NTFS hardlink to the most recent saved
top-k checkpoint and therefore also contains epoch 95, not epoch 99. The
epoch-95 named checkpoint is the predeclared best model; `metrics.csv` is the
authoritative complete 100-epoch trace.

## Non-angular metrics

- Detection TP/FP/FN and F1 are unchanged by construction: this is a
  regression-only experiment using frozen centers and matches. The released
  nuclei values remain TP=124, FP=1, FN=21, F1=0.9185185.
- Center distances are bit-for-bit unchanged because neither segmentation nor
  center matching was rerun or modified.
- The length head and length loss are unchanged, but shared features were
  retrained. Mean length-MAE changes are -0.048 voxels in all-GT,
  -0.085 in matched-GT, and -0.050 in predicted-center mode. Every paired
  bootstrap interval spans zero; there is no evidence here of a material
  non-angular regression.

## Provenance and reproducible outputs

Production provenance:

- `dare3d/losses/angle3d.py:18-25` projects the 9D output and calls the
  full-rotation geodesic loss; `:91-103` implements its clipped quaternion form.
- `dare3d/models/regression_module.py:79-107` adds angular and length losses;
  `:165-177` logs the validation quantities; `:246-255` preserves epoch scheduler
  stepping.
- `dare3d/data/components/angles3d.py:242-262` performs SVD projection and
  `:265-290` extracts/renders the quaternion-vector axis.
- The archived exact model/config is
  `DARE3d_data_190326/Gastruloid_241025/weights/regression3d_exp10-b/.hydra/config.yaml`.

Audit-only implementation and evidence:

- `nematic_retraining_experiment.py`: frozen dataset reconstruction, smooth
  nematic loss, preflight, fresh/resumable training, and state recording.
- `nematic_retraining_evaluation.py`: paired original/retrained inference,
  exact corrected metric, production diagnostic, bootstrap, and assertions.
- `nematic_retraining/seed_12345/experiment_config.json`: complete resolved
  protocol, environment, data hashes, representation, and loss definitions.
- `preflight.json`: 10/10 data, invariance, gradient, CUDA, and integrity checks.
- `run_state.json`, `run.log`, `training_logs/`, and `checkpoints/`: full training
  trace and model state.
- `evaluation.json`: complete numerical summaries, paired effects, hashes,
  provenance, and 10/10 evaluation assertions.
- `evaluation_summary.csv`: all requested scenario/mode statistics.
- `controlled_test_event_metrics.csv` and `manuscript_event_metrics.csv`:
  event-level values for independent recalculation.
- `angular_distribution.csv`: fixed-bin histograms and empirical CDFs.
- `predictions/`: compact raw arrays for every checkpoint/evaluation mode.

The evaluation independently replays the original model to within 0.02 degrees
of all released aggregates and to within 0.00001 degrees of the prior corrected
metric audit. Both checkpoints load the same 63 network tensors, differ in
SHA-256, and all 10/10 evaluation assertions pass.

Relevant manuscript locations remain `manuscript_280826version/main.tex:79`
(abstract interpretation), `:395` (nuclei quantitative statement), `:481`
(annotation-uncertainty interpretation), and `:501-505` (3D nematic random
baseline). No quantitative table or figure contains a nuclei angular aggregate.

## Scientific assessment

Retraining materially improves the nuclei regression result when centers are
ground truth or accurately matched: the corrected mean falls by about
2.3--2.8 degrees (16--20%), SD and p95 also fall, and the controlled unique
holdout shows a substantially shorter high-error tail. The corrected all-GT RMS
of 16.79 degrees remains far below the 61.2-degree random nematic baseline and
strengthens the claim that nuclei performance is at the approximately-20-degree
annotation scale.

It does not materially improve the complete end-to-end result on frozen
predicted centers, where the change is -0.225 degrees and the paired interval
spans zero. Localization failures dominate that distribution. The experiment
also cannot establish an unbiased improvement on unseen data because test and
validation are the same released movie, exact historical scale provenance is
missing, training was nondeterministic, and only one seed was run. Parameters
were not tuned to force agreement.

## Concise requested conclusion

1. **Original model + original metric:** full-quaternion
   `2*acos(abs(q_pred dot q_true))`; manuscript-compatible all-GT replay is
   mean 25.307, median 17.735, SD 28.019 degrees (effective N=140). The
   manuscript's 28 +/- 2 wording traces to SD and SD/sqrt(N), not mean +/- SEM.
2. **Original model + corrected nematic metric:**
   `acos(abs(u_pred dot u_true))`; mean 14.894, median 12.342, SD 12.700 degrees.
   This isolates a -10.413-degree (-41.1%) evaluation-metric effect.
3. **Retrained model + corrected nematic metric:** mean 12.575, median 9.409,
   SD 11.126 degrees. Retraining contributes another -2.319 degrees (-15.6%)
   in all-GT mode and -2.762 degrees (-20.2%) on the controlled unique crops.

**Verdict:** the corrected loss materially improves nuclei axis regression at
ground-truth/matched centers in this controlled experiment, but it does not
materially improve end-to-end predicted-center performance and is not yet an
independent test-set generalization result.
