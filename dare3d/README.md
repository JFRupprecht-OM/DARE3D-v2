# `dare3d` — core framework

The PyTorch Lightning + Hydra framework behind DARE3D: data loading, the segmentation and
regression networks, training, evaluation, and inference. The napari plugin (`napari_dare3d`) is a
thin GUI over this package; everything here also runs headless from the CLI. See the
[root README](../README.md) for installation and end-to-end usage.

## Project structure

```
dare3d/
├── train.py             # Hydra entry — train ONE stage (experiment=segmentation | regression)
├── eval.py              # Hydra entry — evaluate trained models on a val set
├── predict.py           # Hydra entry — run inference (segmentation [+ regression]) on a movie
├── train_eval.py        # CLI wrapper — segmentation → regression → eval in one command
├── demo.py              # get_path_to_demo_folder() — locate the bundled demo data
│
├── data/                            # dataset + datamodule layer
│   ├── dare_datamodule.py           #   LightningDataModule (train/val/test wiring)
│   ├── augmend_wrapper.py           #   augmentation pipeline wrapper
│   ├── gpu_augmentations.py         #   GPU-side augmentations
│   ├── augmentations/               #   individual transforms (e.g. random_channel_flip)
│   ├── components/                  #   dataset classes + geometry
│   │   ├── abstract_celldataset.py  #     base dataset: movie listing, scales, cropping
│   │   ├── cell_3dataset.py         #     3D dataset; reads (T,Z,Y,X) tif → internal (T,X,Y,Z)
│   │   ├── seg_3dataset.py          #     segmentation datasets
│   │   ├── segres_3dataset.py       #       ('segres' experiment variant)
│   │   ├── regress_3dataset.py      #     regression dataset (crops around detected centers)
│   │   ├── tap_3dataset.py          #     ('tap' experiment variant)
│   │   └── angles3d.py              #     quaternion ↔ axis/angle, representation conversions
│   └── visualization/               #   log/result viewers (display_segmentation_logs.py)
│
├── models/                          # LightningModules + network components
│   ├── segmentation_module.py       #   segmentation training/inference module
│   ├── regression_module.py         #   regression training/inference module
│   ├── sequence_ordered_module.py   #   sequence-ordered variant
│   ├── tap_module.py                #   'tap' experiment variant
│   └── components/                  #   architectures
│       ├── multiscale_unet.py       #     3D multiscale U-Net (default segmentation net)
│       ├── segres_multiscale_unet.py, unet.py, seq_unet.py, swinunetr_wrapper.py
│       ├── simple_regression_net.py #     regression CNN used by train_eval.py
│       ├── regression_net.py        #     alternative regression CNN
│       └── custom.py
│
├── losses/                          # loss functions
│   ├── pixelwise_crossentropy.py    #   pixel-wise weighted BCE (segmentation)
│   ├── angle3d.py                   #   3D division-axis angle + length loss (regression)
│   └── decorrelation.py             #   decorrelation regulariser
│
├── metrics/                         # inference + evaluation
│   ├── inference.py                 #   segmentation_inference / regression_inference
│   │                                #     (+ should_stop / InferenceAborted; see below)
│   ├── object_level.py              #   connected_components, statistics_optimized,
│   │                                #     filter_by_object_weighted_prob, get_sphere_vol
│   └── infer_measure.py             #   end-to-end measurement (GT vs prediction)
│
├── loggers/                         # training image loggers
│   ├── segmentation_logger.py
│   └── regression_logger.py
│
├── tools/                           # dataset utilities
│   ├── generate_sparse_weights.py   #   cylindrical per-voxel weight masks (--annotated_movie)
│   └── generate_cv_split.py         #   cross-validation splits
│
└── utils/                           # framework plumbing
    ├── instantiators.py, logging_utils.py, pylogger.py, rich_utils.py, utils.py
    ├── regression_display.py        #   render division axes from quaternion + length
    ├── image.py                     #   image helpers
    └── io.py                        #   iter_tifs — case-insensitive .tif/.tiff discovery
```

Configuration lives in the repo-root [`configs/`](../configs) Hydra tree (`experiment/`, `model/`
with `net/`/`criterion/`/`optimizer/`/`scheduler/`, `data/`, `trainer/`, `logger/`, `paths/`, …).
Training writes runs to `logs/<task>/runs/<date>/` (`.hydra/config.yaml` + `checkpoints/`).

## Entry points

| Script | What it does | Key inputs → outputs |
|---|---|---|
| `train.py` | Train one stage under Hydra. | `experiment=segmentation\|regression`, `train_dir=…`, `val_dir=…` → `logs/<task>/runs/<date>/` |
| `eval.py` | Evaluate trained models on a val set. | `segmentation.model_dir=…`, `regression.model_dir=…` → metrics |
| `predict.py` | Inference on one `(T,Z,Y,X)` movie. | `segmentation.model_dir`, `regression.model_dir`, `inference_dir` → probability map, axis render, `raw_predictions.npz` |
| `train_eval.py` | Wrap segmentation → regression → eval. | `--set_folder`, `--epoch` (+ flags; see root README) |

A **model directory** is any folder with `.hydra/config.yaml` + `checkpoints/last.ckpt` — produced by
`train.py` and consumed by `eval.py` / `predict.py` and the napari plugin.

## Pipeline & output

1. **Segmentation** — a 3D **multiscale U-Net** (`models/components/multiscale_unet.py`) predicts a
   per-voxel division-probability heatmap; thresholding + connected components
   (`metrics/object_level.py`) + a size×probability filter yield division **centers**.
2. **Regression** — a 3D CNN (`models/components/simple_regression_net.py`) predicts, per center, the
   division-axis **orientation** as a unit **quaternion** (wxyz; the axis is its normalised imaginary
   part) and the **axis length** (voxels).

This is a **single-model** pipeline: there is no ensemble or consensus/clustering stage.

Regression inference defaults to the `training_consistent` preprocessing contract used
to create regression training crops. The historical raw-grid behavior is retained as
the explicit `legacy_raw` replay mode. Legacy result fields (`center`, `rotation`,
`length`) remain available; explicit raw-grid, regression-grid, and physical geometry
fields remove unit ambiguity for new consumers. This contract is shared by
`predict.py`, `eval.py`, and the napari/headless API.

## Coordinate order

Movies are stored on disk as `(T, Z, Y, X)`. `data/components/cell_3dataset.py` swaps them to the
internal `(T, X, Y, Z)` order used throughout the core; detected centers come back as
`(m, t, x, y, z)`. Orientation is a unit quaternion (wxyz), and the axis is the normalised imaginary
part (`data/components/angles3d.py::compute_axis_angle`). The napari plugin reverses the spatial
triple back to napari `(t, z, y, x)` — see `napari_dare3d/_api.py` for the locked mapping.

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
