# `napari_dare3d` — the DARE3D napari plugin

In-process napari plugin that runs DARE3D's 3D inference and (re)training from the GUI. It is a thin
layer over the core `dare3d` package; the heavy work (torch + `dare3d`) is deferred so that merely
loading the plugin (napari plugin discovery) stays cheap.

## Modules

| File | Role |
|---|---|
| `_widget.py` | **Inference** widget (`magic_factory`). Builds the GUI, runs `_api.infer_stack` in a `thread_worker`, and overlays results. Hosts the collapsible *Advanced parameters* toggle and the **Stop** button. |
| `_api.py` | **Headless** inference API (`infer_stack`, `to_layer_data`) — imports no napari. Replicates `dare3d.predict` without the Hydra CLI, and owns the 3D coordinate conventions (below). |
| `_train_widget.py` | **Training** widget (`magic_factory`). Streams `dare3d/train.py` / `eval.py` output live; has its own **Stop** button. |
| `_train.py` | Subprocess driver for training. Import-light (defers `import dare3d`); adds portability fixes the raw scripts lack (`sys.executable`, a local SQLite MLflow store, absolute data overrides). |
| `_data.py` | Zenodo downloader for the demo data/models bundle (`DARE3d_data_190326`, record 19113351). |
| `_io.py` | Tiny stdlib helper `iter_tifs` (case-insensitive `.tif`/`.tiff` discovery). Kept dependency-free so importing it never pulls in torch. |
| `napari.yaml` | Plugin manifest (contributes the inference, training, and data-download widgets). |

## 3D coordinate conventions (locked — see `_api.py`)

- napari feeds a stack as `(T, Z, Y, X)`; a bare `(Z, Y, X)` is a single time frame.
- DARE3D stores movies as `(T, Z, Y, X)` on disk and swaps internally to `(T, X, Y, Z)`. Detected
  centers come back as `(m, t, x, y, z)`; we map a spatial triple `(x, y, z)` to napari `(z, y, x)`
  by **reversing** it (and likewise the axis direction).
- Orientation is a unit **quaternion** (wxyz). The division-axis direction is the **normalised
  imaginary part** of that quaternion (= `dare3d…angles3d.compute_axis_angle`); the drawn segment
  spans `length` voxels centred on the detection.

Any new overlay/save code must reuse these mappings — do **not** reintroduce 2D `(t, y, x)` +
scalar-angle conventions.

## Result layers

- **DARE3D centers** — Points `(N, 4)` `(t, z, y, x)`, red.
- **DARE3D axes** — Points `(M, 4)` (only with a regression model): each axis sampled densely along
  its true 3D length so it is visible crossing every z-slice it traverses (a napari Vectors layer
  cannot render a stick across a spatial slice when a time axis is present).

## Run / Stop

Both widgets run their heavy work off the GUI thread. **Stop** sets a shared flag
(`_INFER_STATE` / `_TRAIN_STATE`) polled by the worker: inference passes `should_stop` into
`infer_stack`, which checks it **between time frames** (segmentation) and **between detections**
(regression) and raises `dare3d.metrics.inference.InferenceAborted`, returning **no** partial layers;
training terminates the subprocess. The inference *Advanced parameters* are collapsed by default and
deliberately expose **no patch/crop-size knob** (the seg model is fixed at 128³, reg at 32³).

## Self-checks (no GPU)

- `python verify_geometry.py` — quaternion→axis + coordinate mapping.
- `python verify_train.py` — training-command construction + path resolution.
