# `dare3d` — core framework

The PyTorch Lightning + Hydra framework behind DARE3D: data loading, the segmentation and
regression networks, training, evaluation, and inference. The napari plugin (`napari_dare3d`) is a
thin GUI over this package; everything here also runs headless from the CLI.

## Layout

| Path | Role |
|---|---|
| `data/` | Dataset readers (`data/components/abstract_celldataset.py`, 3D cell datasets), augmentations, Lightning data modules, and `data/visualization/` log viewers. |
| `models/` | Lightning modules + network components (`models/components/`: the 3D multiscale U-Net and the regression CNN). |
| `losses/` | Loss functions (pixel-wise weighted BCE, dice/focal, 3D angle + length, decorrelation, …). |
| `metrics/` | Inference pipelines (`metrics/inference.py`) and object-level stats (`metrics/object_level.py`: connected components, weighted-prob filtering). |
| `loggers/` | Training image loggers (`segmentation_logger.py`, `regression_logger.py`). |
| `tools/`, `utils/` | Dataset/CV-split helpers and general utilities (incl. `utils/io.py`'s `iter_tifs`). |
| `train.py`, `eval.py`, `predict.py` | Hydra entry points: train one stage, evaluate, run inference. |
| `train_eval.py` | High-level wrapper: segmentation → regression → eval (mirrors the retraining notebook / training widget). |

Configuration lives in the repo-root [`configs/`](../configs) Hydra tree (`experiment/`, `model/`,
`data/`, `trainer/`, …).

## Pipeline & output

1. **Segmentation** — a 3D **multiscale U-Net** predicts a per-voxel division-probability heatmap;
   thresholding + connected components + a size×probability filter yield division **centers**.
2. **Regression** — a 3D CNN predicts, per center, the division-axis **orientation** as a unit
   **quaternion** (wxyz; the axis is its normalised imaginary part) and the **axis length** (voxels).

This is a **single-model** pipeline: there is no ensemble or consensus/clustering stage.

## Model input constraints (important)

- Segmentation patch (`crop_size`): **128³**; regression crop: **32³** (see
  `configs/experiment/{segmentation,regression}.yaml`).
- The U-Net uses strides `[2, 2, 2, 2, 2]`, so a patch side must be divisible by **32**.
- Inputs carry temporal context via `input_channels = [-1, 0, 1]` (frames *t-1, t, t+1*); inference
  is per-timepoint with that context.

Because the patch sizes are fixed by the architecture, **do not expose a tile/crop-size knob** to
users — the napari inference widget intentionally omits one.

## Aborting inference

`metrics/inference.py` exposes `segmentation_inference(..., should_stop=None)` and
`regression_inference(..., should_stop=None)`. When the optional `should_stop()` predicate returns
True they raise `InferenceAborted` at the next frame/detection boundary — this is how the plugin's
**Stop** button interrupts a long run. Default `None` keeps the CLI paths (`predict.py`, `eval.py`,
`metrics/infer_measure.py`) unchanged.

## Discovery helper

`utils/io.py::iter_tifs(dir, prefix="")` lists `.tif`/`.tiff` **case-insensitively** (replacing
`glob("*.tif")`, which skips uppercase extensions on Linux/macOS). A behaviourally identical,
import-light copy lives in `napari_dare3d/_io.py` so the plugin never imports torch just to list files.
