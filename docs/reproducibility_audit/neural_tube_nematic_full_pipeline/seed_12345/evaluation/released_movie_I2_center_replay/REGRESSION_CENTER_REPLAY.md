# Neural-tube regression replay at released centers

This evaluation bypasses segmentation inference. movie_I2 is the released
validation/reported set, so this is a historical replay rather than an
independent-test estimate. Predicted-center rows are scored against the
annotation associated with each matched true center.

| Center mode | N | Original + production | Original + nematic | Retrained + nematic |
|---|---:|---:|---:|---:|
| all groundtruth centers | 122 | 49.563 +/- 27.107 deg | 36.680 +/- 22.142 deg | 43.939 +/- 24.870 deg |
| matched groundtruth centers | 114 | 48.087 +/- 25.735 deg | 35.526 +/- 21.488 deg | 44.554 +/- 25.016 deg |
| predicted centers | 114 | 51.362 +/- 27.634 deg | 36.077 +/- 20.005 deg | 45.921 +/- 24.192 deg |

## Replay checks

- Saved prediction artifacts, maximum aggregate difference: 7.62939453e-06.
- Checkpoint replay, maximum aggregate difference: 0.0043258667.
- Checkpoint replay agreement within 0.01: True.
- Event-level values and provenance are in event_metrics.csv and result.json.
