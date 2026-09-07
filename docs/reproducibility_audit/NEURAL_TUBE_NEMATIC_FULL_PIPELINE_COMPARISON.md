# Neural-tube DARE3D nematic regression retraining comparison

## Outcome

This is an isolated audit experiment. Production source, released checkpoints/results, and manuscript files were not modified. The released segmentation checkpoint and its resulting detections are held fixed for both regression models.

## Previous reported result

The released movie_I2 result is 114 TP / 8 FP / 8 FN, F1 93.4%. The released predicted-center regression aggregate has mean 51.367 degrees and population SD 27.633 degrees under the quaternion metric. This is the 21-frame set 2 used for both validation and test in the archived configuration, not the manuscript's bold 11-frame set 3.

## Independent movie_M test

Fixed released segmentation: 0 TP / 0 FP / 80 FN, F1 0.00%. The detector produced no candidate centers, so predicted-center regression scoring is not evaluable.

Angular summaries on predicted centers:

- Not evaluable (N=0): The released segmentation checkpoint produced no candidate components at the validation-selected unchanged threshold grid; there are therefore no predicted centers for angular scoring.

Controlled independent 80-crop test:

- Original model + original metric: N=80, mean=82.465 degrees, median=70.793 degrees, SD=42.503 degrees, SEM=4.752 degrees, RMS=92.774 degrees, p95=160.375 degrees
- Original model + corrected metric: N=80, mean=53.816 degrees, median=53.434 degrees, SD=22.903 degrees, SEM=2.561 degrees, RMS=58.487 degrees, p95=85.752 degrees
- Retrained model + corrected metric: N=80, mean=18.945 degrees, median=17.130 degrees, SD=11.650 degrees, SEM=1.303 degrees, RMS=22.241 degrees, p95=40.245 degrees

## Non-angular checks

- Detection/matching is shared: 0 TP / 0 FP / 80 FN; no regression-dependent change.
- Controlled length error, original model: N=80, mean=10.231 voxels, median=10.252 voxels, SD=0.999 voxels, SEM=0.112 voxels, RMS=10.279 voxels, p95=11.886 voxels
- Controlled length error, retrained model: N=80, mean=0.836 voxels, median=0.732 voxels, SD=0.706 voxels, SEM=0.079 voxels, RMS=1.094 voxels, p95=2.405 voxels
- Paired controlled length-error mean change: -9.395 voxels; 95% bootstrap interval [-9.636, -9.147].
- Predicted-center length comparison: not evaluable (N=0).

## Numerical effects

Changing only evaluation metric on the original controlled test changes the mean by -28.649 degrees (-34.7%).
Retraining changes the corrected controlled-test mean by -34.870 degrees; the paired descriptive 95% bootstrap interval is [-40.106, -29.684] degrees.
The fixed-segmentation predicted-center comparison is not evaluable because the frozen detector produced zero candidate centers.

## Scientific interpretation

On the independent 80-crop movie_M test, retraining materially reduces the corrected mean angular error from 53.816 to 18.945 degrees. The paired new-minus-old mean is -34.870 degrees, with the descriptive bootstrap interval reported above. Replacing only the evaluation metric changes the original mean from 82.465 to 53.816 degrees. However, full predicted-center performance is not evaluable because the frozen released segmentation checkpoint produces zero detections on movie_M under the unchanged protocol. Length error also changes from 10.231 to 0.836 voxels despite an unchanged length-loss definition, so improvement is demonstrated for this retrained regressor but cannot be attributed exclusively to the orientation-loss change. The old movie_I2 result is not a same-test comparator.

## Compatibility limitations kept visible

- The historical scales.json is absent; manuscript spacing was used.
- Segmentation was not retrained; the released checkpoint, detections, and matching are fixed across regressors.
- The released segmentation event log stopped at epoch 153 although 200 were configured.
- The production end-to-end regression path omits training-time isotropic rescaling; it was preserved and is reported separately from the controlled test.
- The archived membrane regressor is 3 stages / 32 initial filters, whereas main.tex:351-352 describes 5 stages / 16 filters.
- The first two raw frames are excluded by the unchanged three-frame regression crop constructor, leaving 80 of 104 complete test annotations.

## Manuscript locations requiring review

- main.tex:79 (abstract orientation-accuracy claim).
- main.tex:194-196 and 210 (Dataset 2 split/test identification).
- main.tex:333-338 (voxel resampling and segmentation training protocol).
- main.tex:343-352 (orientation formulation and regressor architecture).
- main.tex:356-374 (thresholding, component filtering, matching, and metrics).
- main.tex:397-402 (3D membrane quantitative result).
- main.tex:488-494 (membrane angular-error interpretation).
- main.tex:499-505 (2D-vs-3D angular comparison and random baseline).
- main.tex:655-656 (neural-tube regression example figure caption; update only if the displayed model changes).
- Supplementary movie descriptions at main.tex:588-590.

## Evidence

- Numerical summary: [evaluation_summary.csv](neural_tube_nematic_full_pipeline/seed_12345/evaluation/evaluation_summary.csv)
- Distribution table: [angular_distribution.csv](neural_tube_nematic_full_pipeline/seed_12345/evaluation/angular_distribution.csv)
- Figure: [angular_and_detection_comparison.png](neural_tube_nematic_full_pipeline/seed_12345/evaluation/angular_and_detection_comparison.png)
- Machine result: [final_comparison.json](neural_tube_nematic_full_pipeline/seed_12345/evaluation/final_comparison.json)
