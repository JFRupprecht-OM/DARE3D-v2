# `napari_dare3d` — the DARE3D napari plugin

Run DARE3D's 3D division-axis inference and (re)training from the napari GUI: detect cell-division
centers in a `(T, Z, Y, X)` stack and overlay each division's 3D axis. The package is a thin layer
over the core `dare3d` framework; the torch and `dare3d` imports are deferred, so napari plugin
discovery (importing the manifest) does not load them.

![DARE3D inference widget overlaying detected division centers (red) and axes (cyan) on a demo movie](DARE3Dnapari.png)

## Installation

The plugin ships with the core `dare3d` package and is registered through the same editable install;
there is no separate PyPI package (not yet published — install from source).

```bash
git clone https://github.com/qazi05/DARE3d
cd DARE3d
conda create -n dare3d-v2 python=3.10 -y
conda activate dare3d-v2

# PyTorch first, matching your GPU/CUDA (see https://pytorch.org/get-started/locally/).
# Training requires torch >= 2.5 (cuDNN >= 9); inference also runs CPU-only.
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
pip install "napari[all]"        # napari + a Qt backend
pip install -e .                 # core + this plugin
```

The editable install registers the plugin via its `napari.manifest` entry point, so napari lists
**DARE3D** under *Plugins* with no further step. CPU-only PyTorch runs the full inference pipeline
(segmentation + regression); only training requires a CUDA GPU.

> **cuDNN and 3D convolutions.** torch ≥ 2.5 (cuDNN ≥ 9) is the supported training stack. Older
> cuDNN 8.x (e.g. torch 2.2) can segfault on 3D convolutions; the `DARE3D_CUDNN` environment variable
> overrides the automatic gate — `DARE3D_CUDNN=0` disables cuDNN (stable but slower, for an old
> stack), `DARE3D_CUDNN=1` forces it on (only safe on cuDNN ≥ 9). See `dare3d.train.check_cudnn_for_3d`.

## Quick start

1. Launch napari from a directory containing `DARE3d_data_190326` (or run the download widget below)
   so the model-directory fields pre-fill.
2. Open **Plugins → DARE3D → DARE3D inference**. Select an open image layer or load a `.tif`/`.tiff`
   movie, confirm the segmentation (and optional regression) model directories, then **Run DARE3D**.
3. **Plugins → DARE3D → DARE3D download data** fetches the demo data/models bundle from Zenodo.
4. **Plugins → DARE3D → DARE3D retraining & fine-tuning (beta)** drives training/fine-tuning as a
   subprocess. Retraining and fine-tuning are also available as notebooks under `notebooks/`, the
   recommended path for both.

## Modules

| File | Role |
|---|---|
| `_widget.py` | **Inference** widget (`magic_factory`). Builds the GUI, runs `_api.infer_stack` in a `thread_worker`, and overlays results. Hosts the collapsible *Advanced parameters* toggle and the **Stop** button. |
| `_api.py` | **Headless** inference API (`infer_stack`, `to_layer_data`); imports no napari. Replicates `dare3d.predict` without the Hydra CLI, and owns the 3D coordinate conventions (below). |
| `_train_widget.py` | **Training** widget (`magic_factory`). Streams `dare3d/train.py` / `eval.py` output live; has its own **Stop** button. |
| `_train.py` | Subprocess driver for training. Import-light (defers `import dare3d`); adds portability fixes the raw scripts lack (`sys.executable`, a local SQLite MLflow store, absolute data overrides). |
| `_data.py` | Zenodo downloader for the demo data/models bundle (`DARE3d_data_190326`, record 19113351). |
| `_io.py` | Standalone stdlib helper `iter_tifs` (case-insensitive `.tif`/`.tiff` discovery). Dependency-free, so importing it does not pull in torch. |
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

Both widgets run inference/training off the GUI thread. **Stop** sets a shared flag
(`_INFER_STATE` / `_TRAIN_STATE`) polled by the worker: inference passes `should_stop` into
`infer_stack`, which checks it **between time frames** (segmentation) and **between detections**
(regression) and raises `dare3d.metrics.inference.InferenceAborted`, returning **no** partial layers;
training terminates the subprocess. The inference *Advanced parameters* are collapsed by default and
deliberately expose **no patch/crop-size knob** (the seg model is fixed at 128³, reg at 32³).

## Self-checks (no GPU)

- `python tests/test_geometry.py` — quaternion→axis + coordinate mapping.
- `python tests/test_train_commands.py` — training-command construction + path resolution.

  (Both live in `tests/`, so `pytest` / `make test` also run them.)

## Authors

Romain Karpinski, Alice Gros, Marc Karnat, Qazi Saaheelur Rahaman, Jules Vanaret, Mehdi Saadaoui,
Sham Tlili, and Jean-François Rupprecht.

Affiliation: Aix Marseille Univ, Inserm U1067, CNRS, LAI (UMR 7333), Turing Centre for Living
Systems, Marseille, France.

## Citation

Three distinct references — please cite the ones relevant to your use:

- **Method paper (preprint).** Karpinski *et al.*, *bioRxiv*, 2026 — DOI
  [`10.1101/2024.02.05.578987`](https://doi.org/10.1101/2024.02.05.578987). BibTeX below.
- **Data & pretrained models.** Zenodo record **19113351** (`DARE3d_data_190326.zip`):
  <https://zenodo.org/records/19113351>.
- **Software release.** DOI [`10.5281/zenodo.19113351`](https://doi.org/10.5281/zenodo.19113351)
  (code archive, v2 — same Zenodo record as the data bundle).

A machine-readable [`CITATION.cff`](../CITATION.cff) is provided at the repository root.

```bibtex
@article{Karpinski2024.02.05.578987,
  author       = {Karpinski, Romain and Gros, Alice and Karnat, Marc and Saaheelur Rahaman, Qazi and
                  Vanaret, Jules and Saadaoui, Mehdi and Tlili, Sham and Rupprecht, Jean-Fran{\c c}ois},
  title        = {DARE: Division Axis and Region Estimation from 2D and 3D Time-Lapse Images},
  elocation-id = {2024.02.05.578987},
  year         = {2026},
  doi          = {10.1101/2024.02.05.578987},
  publisher    = {Cold Spring Harbor Laboratory},
  URL          = {https://www.biorxiv.org/content/early/2026/03/27/2024.02.05.578987},
  eprint       = {https://www.biorxiv.org/content/early/2026/03/27/2024.02.05.578987.full.pdf},
  journal      = {bioRxiv}
}
```

## License & acknowledgements

MIT — see [`LICENSE`](../LICENSE).

Funded by the Agence Nationale de la Recherche (ANR-16-CONV-0001, ANR-22-CE30-0021) and the Fondation
pour la Recherche Médicale (FDT202404018538). This work was granted access to the HPC resources of
IDRIS under the allocation AD010314339 made by GENCI.
