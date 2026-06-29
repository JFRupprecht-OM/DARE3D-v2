<div align="center">

# DARE3D — Division Axis and Region Estimation in 3D time-lapse images

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://napari.org"><img alt="napari" src="https://img.shields.io/badge/napari-plugin-blueviolet"></a>

</div>

---

**DARE3D** detects cell divisions in 3D time-lapse image volumes and estimates each division's
**center**, **orientation**, and **axis length**, in two stages:

1. **Segmentation** — detect the center of each division (the barycenter of the two daughter cells).
2. **Regression** — estimate the division-axis orientation and length.

This repository contains **both** the core deep-learning framework (PyTorch Lightning + Hydra)
**and** the **napari plugin** (`napari_dare3d`) that runs it interactively — installed together
by a single `pip install -e .`.

> **What's new in this version.** The headline addition is the **napari plugin** — interactive 3D
> inference and retraining straight from the napari GUI (DARE3D's core was already PyTorch). Input
> stacks are `(T, Z, Y, X)`; each detected division is returned as a **center** plus a **division
> axis** encoded as a unit **quaternion** (the axis is the normalised imaginary part of the
> quaternion) and an **axis length** in voxels.
>
> **Beta features.** Interactive **retraining** — and the upcoming **transfer-learning / fine-tuning**
> mode — ship as **beta** in this version: experimental, with results, defaults, and the API subject
> to change. See *Training & retraining* below.

> **Citation.** If you use DARE3D, please cite the preprint:
> Karpinski *et al.*, *bioRxiv* 2024 — <https://www.biorxiv.org/content/10.1101/2024.02.05.578987v2>
>
> **Authors:** Romain Karpinski, Marc Karnat, Alice Gros, Qazi Saaheelur Rahaman, Jules Vanaret,
> Mehdi Saadaoui, Sham Tlili, and Jean-François Rupprecht.

## Repository contents

```
dare3d/                       # core package: data, models, losses, metrics, train/eval/predict
configs/                      # Hydra configuration tree
napari_dare3d/                # the napari plugin (in-process inference + training widgets)
notebooks/                    # Run_dare3d_Prediction / Run_dare3d_Retraining + data viz/normalisation
scripts/                      # dataset/experiment helpers
tests/                        # unit + integration tests
verify_geometry.py            # plugin geometry self-check (no models/GPU)
verify_train.py               # plugin training-command self-check (no GPU)
```

Per-package internals are documented in [`dare3d/README.md`](dare3d/README.md) (core framework) and
[`napari_dare3d/README.md`](napari_dare3d/README.md) (plugin).

## Installation

**Prerequisites:** Python 3.10 and Conda (recommended). An NVIDIA GPU with **CUDA 11.8+** is needed
for regression and training; CPU-only is fine for **segmentation-only inference**.

```bash
git clone https://github.com/qazi05/DARE3d
cd DARE3d

# 1) conda environment
conda create -n dare3d-v2 python=3.10 -y
conda activate dare3d-v2

# 2) PyTorch first — pick the build that matches your machine:
#    CUDA 12.1:
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#    CUDA 11.8:
#    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#    CPU-only (segmentation inference only — no regression/training):
#    python -m pip install torch torchvision torchaudio

# 3) the rest of the dependencies
pip install -r requirements.txt

# 4) napari + its Qt backend (install explicitly so the GUI backend is present)
pip install "napari[all]"

# 5) the project itself (core + napari plugin)
pip install -e .
```

The editable install registers the napari plugin via its `napari.manifest` entry point, so
**napari lists "DARE3D" under Plugins** with no extra step. (CPU-only PyTorch works for
segmentation inference; **regression and training require a CUDA GPU**.)

> **Import errors after a layout change?** Re-run `pip install -e .` to refresh the editable
> install. For full Hydra tracebacks, set `HYDRA_FULL_ERROR=1` (PowerShell: `$env:HYDRA_FULL_ERROR=1`).

## Models & data

The training/inference dataset (pretrained weights + demo movies) is published on **Zenodo**
([record 19113351](https://zenodo.org/records/19113351): `DARE3d_data_190326.zip`). Unzip it at
the repository root so models resolve as
`DARE3d_data_190326/<case>/weights/{segmentation3d_*,regression3d_*}` (the bundle ships the
**Gastruloid** and **Neural tube** cases). Or click **Download DARE3D data (Zenodo)** in the plugin
(Plugins → DARE3D) to fetch and unzip it automatically into the folder you launch napari from.

A model directory is any folder containing `.hydra/config.yaml` + `checkpoints/last.ckpt`.

## Data format

- **Input movies** are TIFF stacks in disk/napari order `(T, Z, Y, X)` (a bare `(Z, Y, X)` volume is
  treated as a single time frame). Internally DARE3D swaps to `(T, X, Y, Z)`; the napari plugin hides
  this and maps results back to `(t, z, y, x)` for you. `.tif` and `.tiff` are both accepted,
  case-insensitively.
- **Training data** lives under `data/3d/<dataset>/{train,val}/{im,label}/*.tif`, one 4-D movie per
  file. **Labels** encode the daughter-cell pair: first daughter → **odd** instance ids, second →
  **even** ids. An optional `train/weights/` folder supplies per-voxel sparse weighting.
- **Voxel scale** (anisotropy) is read per movie from `data/3d/scales.json` in µm/voxel; if absent,
  the experiment's `default_scale` (e.g. `0.621, 0.621, 2`) is used. The inference widget's
  *default_scale x,y,z* field overrides it.

## Configuration (Hydra)

The CLI (`train.py`, `eval.py`, `predict.py`) is driven by a composable **Hydra** config tree under
[`configs/`](configs) — `experiment/`, `model/` (`net/`, `criterion/`, `optimizer/`, `scheduler/`),
`data/`, `trainer/`, `logger/`, `paths/`, … Any field can be overridden on the command line without
editing files:

```bash
# choose an experiment preset, then override individual fields
python dare3d/train.py experiment=segmentation data.batch_size=8 trainer.max_epochs=100
python dare3d/train.py experiment=regression   model.optimizer.lr=0.001
```

- Prefix a key with `+` to **add** one that isn't in the chosen config (e.g. `+segmentation.model_dir=…`).
- Every run writes its **resolved config** to `<output>/.hydra/config.yaml`; that is exactly what a
  *model directory* carries (`.hydra/config.yaml` + `checkpoints/`), so inference can reload the
  training settings.
- Hydra **multirun** (`-m`) sweeps parameters, e.g. `python dare3d/train.py -m data.batch_size=4,8`.

## Inference

Inference can be run **three ways** — pick whichever fits your workflow:

**1. Terminal (CLI).**

```bash
python dare3d/predict.py \
  +segmentation.model_dir=<seg_model_dir> \
  +regression.model_dir=<reg_model_dir> \
  +inference_dir=<folder_with_one_TZYX_tif> \
  device=gpu
```

Results (probability map, division-axis render, and a `raw_predictions.npz` of centers + rotation
matrices + lengths) are written under the Hydra run directory. Useful overrides:

- **Voxel scale:** `+default_scale=[0.621,0.621,2]` (x,y,z µm) or `+scale_file=data/3d/scales.json`.
- **Detection tuning:** `segmentation.threshold=0.5`, `segmentation.min_weighted_prob=0.1`,
  `segmentation.inference_overlap=0.25`, `segmentation.inference_batch_size=4`.
- **Device:** `device=gpu` (regression needs CUDA) or `device=cpu` (segmentation only).
- Omit `+regression.model_dir` to get **centers only** (no axes).

**2. Notebook.** Open `notebooks/Run_dare3d_Prediction.ipynb` — it sets the model/data paths,
validates them, runs segmentation + regression, and visualises the result.

**3. napari plugin.** Launch `napari`, load a 3D/4D stack `(T, Z, Y, X)` or `(Z, Y, X)`, then
**Plugins → DARE3D → DARE3D inference**. Set the segmentation / regression model-dir fields and
**Run** — it overlays the detected division **centers** and **axes** as napari Points layers.
Uncheck *Analyse whole movie* to process only a `[t_start, t_end]` window, expand **Show advanced
parameters** for the fine-tuning knobs, and use **Stop** to abort a long run. (Regression needs a
CUDA device; segmentation runs on CPU or GPU.)

## Postprocessing

DARE3D is **single-model** — there is no ensemble or consensus step. Division **centers** are
extracted from the segmentation head:

1. The 3D U-Net produces a per-voxel division-probability heatmap.
2. The heatmap is thresholded (`threshold`, default `0.5`) into a binary mask.
3. Connected components are labelled and reduced to their centroids.
4. Each candidate is filtered by **size × probability** (`min_weighted_prob`, default `0.1`) to drop
   weak/small blobs.

If a **regression** model is supplied, each surviving center is cropped and passed through the
regression CNN, which outputs the **division-axis orientation** (a unit quaternion; the axis is its
normalised imaginary part) and the **axis length** in voxels. The plugin draws centers (red) and
axes (cyan) as Points layers.

| Parameter | Default | Effect |
|---|---|---|
| `overlap` | `0.25` | Sliding-window overlap for 3D segmentation (higher = more accurate, slower). |
| `threshold` | `0.5` | Probability cut turning the heatmap into candidate centers. |
| `min_weighted_prob` | `0.1` | Minimum size × probability to keep a center. |
| `batch_size` | `4` | 3D patches inferred at once (lower on GPU OOM). |

## Training & retraining

> **⚠️ Beta feature.** Retraining — and the planned transfer-learning / fine-tuning mode — is
> **experimental**: results, defaults, and the API may change in a future release, and the training
> defaults currently assume a large-memory GPU. For routine use, run **inference** with the released
> models above.

Retraining can be run **two ways** — from the terminal or the notebook. **Training requires a
CUDA GPU.** Data layout: `data/3d/<dataset>/{train,val}/{im,label}/*.tif` (movies are
`(T, Z, Y, X)`; labels encode the daughter pair: first daughter → odd ids, second → even ids).

**1. Terminal (CLI).** The `train_eval.py` wrapper runs segmentation → regression → evaluation:

```bash
python dare3d/train_eval.py --set_folder <dataset> --epoch 50
python dare3d/train_eval.py --set_folder <dataset> --epoch 50 --train_regression False   # seg only
python dare3d/train_eval.py --set_folder <dataset> --batch_size 4 --cell_radius 10
python dare3d/train_eval.py --set_folder <dataset> --eval_only True                       # just evaluate

# …or drive the stages directly through Hydra:
python dare3d/train.py experiment=segmentation train_dir=3d/<dataset>/train val_dir=3d/<dataset>/val ...
python dare3d/train.py experiment=regression   train_dir=3d/<dataset>/train val_dir=3d/<dataset>/val ...
```

`train_eval.py` flags:

| Flag | Default | Meaning |
|---|---|---|
| `--set_folder` | *(required)* | Dataset folder name under `data/3d/`. |
| `--epoch` | *(required)* | Max epochs per stage. |
| `--batch_size` | `32` | 3D patches per step (lower on GPU OOM). |
| `--cell_radius` | `8` | Radius (voxels) of the segmentation target spheres. |
| `--seg_crop_size` | `128` | Segmentation training patch size. |
| `--date` | `01-01` | Run id → `runs/<date>/`. |
| `--threshold` | `None` | Eval segmentation threshold (`None` = auto-search). |
| `--train_segmentation` / `--train_regression` | `True` | Toggle each stage. |
| `--eval_only` | `False` | Skip training, only evaluate existing models. |
| `--overwrite` | `True` | Re-run even if the run directory already exists. |

**2. Notebook.** Open `notebooks/Run_dare3d_Retraining.ipynb` — it validates your dataset layout and
drives segmentation → regression → evaluation, streaming the logs inline.

Outputs land in `logs/<task>/runs/<date>/` (`.hydra/config.yaml` + `checkpoints/`), ready for the
inference step. Runs are logged to a local **MLflow** SQLite store under `logs/` — browse it with
`mlflow ui --backend-store-uri sqlite:///logs/mlflow.db`. (The napari plugin also exposes a
**DARE3D training** widget that wraps these same `train.py` / `eval.py` steps.)

## Data preparation & helper scripts

Optional helpers for building the `data/3d/<dataset>/{train,val}/{im,label}` layout:

```bash
# Split each raw movie into left/right halves (data augmentation; expects movie1..movie3 under <dir>)
python scripts/generate_split_movies.py --data_dir data/3d

# Build a cylindrical per-voxel weight mask between paired daughter centroids -> weights.tif
python dare3d/tools/generate_sparse_weights.py --annotated_movie <label.tif> --radius 4

# Generate cross-validation splits from a per-movie dataset
python dare3d/tools/generate_cv_split.py --help
```

The `scripts/generate_exp*.py` files are experiment-specific dataset generators kept for
reproducibility; adapt one to your own movies rather than running it verbatim.

## Testing & development

```bash
python verify_geometry.py   # quaternion -> axis + coordinate mapping (no napari/models/GPU)
python verify_train.py      # training-command construction + paths (no GPU)
make test                   # unit tests (excludes slow ones)
make test-full              # all tests, including slow ones
make format                 # run pre-commit hooks (formatting/linting)
make clean                  # remove build artefacts and caches
```

## Troubleshooting

- **Qt / napari GUI issues:** `conda install -c conda-forge pyqt`.
- **Full Hydra tracebacks:** set `HYDRA_FULL_ERROR=1`.
- **CUDA out of memory:** lower `data.batch_size`, or `trainer=cpu` for segmentation-only.
- **`ImportError` / plugin not listed in napari:** re-run `pip install -e .` (refreshes the editable
  install and its `napari.manifest` entry point).
- **Demo data won't download:** check your internet connection, or grab the Zenodo bundle manually
  ([record 19113351](https://zenodo.org/records/19113351)) and unzip it at the repo root.
- DARE3D training was developed on Linux/HPC; on some Windows setups the Lightning backward pass
  can crash natively even when inference is fine — train on Linux/HPC and reuse the resulting
  model directory.

## Related projects

- **[DARE2d](https://github.com/JFRupprecht-OM/DARE2d)** — the 2D counterpart: division-axis and
  region estimation in 2D time-lapse images, also shipped with a napari plugin. DARE3D extends the
  idea to 3D volumes, where the division axis is a full 3D orientation (a quaternion) rather than a
  scalar angle.

## Additional resources

- [Hydra](https://hydra.cc/) — configuration management.
- [PyTorch Lightning](https://lightning.ai/) — training framework.
- [MONAI](https://monai.io/) — medical-imaging building blocks (3D U-Net, sliding-window inference).
- [napari](https://napari.org/) — the n-dimensional image viewer the plugin builds on.

## License & attribution

MIT — see [LICENSE](LICENSE). This work was granted access to the HPC resources of IDRIS under the
allocation AD010314339 made by GENCI.
