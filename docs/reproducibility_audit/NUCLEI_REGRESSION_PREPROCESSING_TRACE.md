# Nuclei regression preprocessing trace

Date: 2026-09-02

## Conclusion

The nuclei regressor does use spatial resampling before training crops are
constructed. The generic training path is
`DareDataModule.prepare_data() -> Regress3Dataset.init(preprocess=True) ->
resize_data() -> Regress3Dataset.prepare_data() -> pad_images() -> crop
indexing`. Images and daughter-point coordinates are rescaled together.

This is analogous to the neural-tube/membrane training path, but the magnitude
is much smaller: the archived nuclei log records approximately 0.91-fold
resizing in every spatial dimension for movie2 and movie3, while movie4 is
unchanged. The intended post-resampling grid is 1 um isotropic.

The normal prediction and evaluation paths are not symmetric with training.
They call `init(preprocess=False)`, then spatially pad and normalize the raw
array. Regression crops are therefore extracted on the unresampled raw grid.

## Released inputs and dimensions

The TIFF files are stored as T,Z,Y,X and are converted by `Cell3Dataset` to
internal T,X,Y,Z order. Training adds one leading zero time point only to the
training movies; validation/test data do not receive temporal padding.

| Movie | Role in released run | TIFF shape T,Z,Y,X | Internal shape before spatial resize T,X,Y,Z | Logged shape after spatial resize T,X,Y,Z | Actual stored shape after full init and 16-voxel spatial padding |
|---|---|---:|---:|---:|---:|
| movie2 | validation and test | 10,180,305,362 | 10,362,305,180 | 10,330,278,164 | 10,362,310,196 |
| movie3 | training | 28,85,411,424 | 29,424,411,85 | 29,387,375,77 | 29,419,407,109 |
| movie4 | training | 28,152,394,405 | 29,405,394,152 | 29,405,394,152 | 29,437,426,184 |

The post-init padded shapes above were independently materialized by the
audit preflight and match the deterministic 16-voxel padding on each side.

## Voxel-spacing provenance

Exact raw X,Y,Z voxel spacing cannot be recovered from the release:

- the historical per-movie `scales.json` named by the saved run is absent;
- movie2 and movie3 contain no calibrated TIFF spacing metadata;
- movie4 declares an ImageJ unit of micron and unit X/Y resolution, but the
  TIFF resolution unit is NONE and no Z spacing is stored;
- the manuscript states 1 um between acquired Z planes for the gastruloids,
  but does not state their acquired X/Y sampling.

The saved configuration defines scales as source um/voxel, sets
`target_scale=1.0`, and falls back to `[0.621,0.621,2]` if the scale file is
missing. That fallback was not the geometry used in the archived run. The
If `Dataset.init()` is run now without restoring that file, the fallback would
instead produce pre-padding T,X,Y,Z shapes movie2 `10x224x189x360`, movie3
`29x263x255x170`, and movie4 `29x251x244x304`; full post-init shapes would
be `10x256x221x392`, `29x295x287x202`, and `29x283x276x336`,
respectively. Those are the consequences of the public fallback, not the
geometry of the archived training run.

training log instead fixes the target shapes shown above. Target/native shape
ratios are movie2 `[0.911602,0.911475,0.911111]`, movie3
`[0.912736,0.912409,0.905882]`, and movie4 `[1,1,1]`. These ratios are
consistent with near-isotropic approximately 0.91-um source voxels for movie2
and movie3 and 1-um voxels for movie4, but they are geometry evidence, not a
replacement for the missing physical calibration.

Under the code's intended interpretation, the arrays after `resize_data()`
have nominal spacing `[1,1,1] um`. Spatial zero-padding changes dimensions,
not spacing.

## Crop and physical field of view

The base regressor input is three temporal channels and a spatial
`32 x 32 x 32` crop. Because training crops are taken from the nominal
1-um isotropic grid, their nominal voxel-support field of view is
`32 x 32 x 32 um` (31 um between first and last voxel centers). A random
0.9-1.1 zoom augmentation may rescale content, but `keep_size=true` keeps the
network input at 32 cubed.

The order within `init()` is important:

1. load TIFF and change T,Z,Y,X to T,X,Y,Z;
2. spatially resample each timestamp with first-order interpolation;
3. multiply and round daughter coordinates by the same scale;
4. pad every spatial axis by 16 voxels on both sides;
5. build crop indices around each rescaled daughter-pair midpoint;
6. min-max normalize the padded movie;
7. lazily materialize each 3 x 32 x 32 x 32 input in `__getitem__`.

## Prediction/evaluation mismatch

Both `dare3d/predict.py` and `dare3d/eval.py` load regression data with
`init(preprocess=False)`, then call `pad_images()` and min-max normalization.
For movie2 this changes the raw internal array only from
`10 x 362 x 305 x 180` to `10 x 394 x 337 x 212` by padding. It performs no
spatial resize before extracting a 32-raw-voxel crop.

Segmentation inference is different: it dynamically resamples each sequence
for the network and unscales its probability map back to the raw grid. Its
detected centers are consequently raw-grid centers, which the regression stage
uses directly. Thus the released nuclei regression model was trained on
nominal 1-um crops but is normally evaluated/inferred on raw-grid crops.

For movie2, the logged geometry alone suggests an approximately 9% linear
field-of-view mismatch (a 32-raw-voxel crop is approximately 29.2 um if the
missing source scale was about 0.912 um/voxel). This physical value remains an
inference, not a verified calibration. The existence of the preprocessing
mismatch is certain from the control flow and recorded dimensions.

## Primary provenance

- `dare3d/data/dare_datamodule.py:85-98`
- `dare3d/data/components/abstract_celldataset.py:119-138,172-183,480-510`
- `dare3d/data/components/cell_3dataset.py:98-113`
- `dare3d/data/components/regress_3dataset.py:32-74,154-174,210-250`
- `dare3d/eval.py:137-146`
- `dare3d/predict.py:20-35,69-75`
- `dare3d/metrics/inference.py:45-97,123-154`
- archived regression log lines 45-63
- saved regression Hydra config lines 54-102 and 218-232
- `nematic_retraining/seed_12345/preflight.json`
- manuscript `main.tex:132,204-210,333-346`
