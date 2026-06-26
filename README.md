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
Run_dare3d_Prediction.ipynb   # inference notebook
Run_dare3d_Retraining.ipynb   # retraining notebook (segmentation -> regression -> eval)
scripts/                      # dataset/experiment helpers
notebooks/                    # data visualisation / normalisation
tests/                        # unit + integration tests
verify_geometry.py            # plugin geometry self-check (no models/GPU)
verify_train.py               # plugin training-command self-check (no GPU)
```

## Installation

```bash
git clone https://github.com/qazi05/DARE3d
cd DARE3d

# 1) conda environment
conda create -n dare3d python=3.10 -y
conda activate dare3d

# 2) PyTorch first (CUDA build; adapt the CUDA version to your machine)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

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

## Models & data

The training/inference dataset (pretrained weights + demo movies) is published on **Zenodo**
([record 17456474](https://zenodo.org/records/17456474): `DARE3d_data_160226.zip`). Unzip it at
the repository root so models resolve as
`DARE3d_data_160226/<case>/weights/{segmentation3d_*,regression3d_*}` (the bundle ships the
**Gastruloid** and **Neural tube** cases). Or click **DARE3D download data** in the plugin
(Plugins → DARE3D) to fetch and unzip it automatically into the folder you launch napari from.

A model directory is any folder containing `.hydra/config.yaml` + `checkpoints/last.ckpt`.

## Inference

Inference can be run **three ways** — pick whichever fits your workflow:

**1. Terminal (CLI).**

```bash
python dare3d/predict.py \
  +segmentation.model_dir=<seg_model_dir> \
  +regression.model_dir=<reg_model_dir> \
  +inference_dir=<folder_with_one_TZYX_tif>
```

**2. Notebook.** Open `Run_dare3d_Prediction.ipynb` — it sets the model/data paths, validates
them, runs segmentation + regression, and visualises the result.

**3. napari plugin.** Launch `napari`, load a 3D/4D stack `(T, Z, Y, X)` or `(Z, Y, X)`, then
**Plugins → DARE3D → DARE3D inference**. Set the segmentation / regression model-dir fields and
**Run** — it overlays the detected division **centers** and **axes** as napari Points layers.
Uncheck *Analyse whole movie* to process only a `[t_start, t_end]` window. (Regression needs a
CUDA device; segmentation runs on CPU or GPU.)

## Training & retraining

Retraining can be run **two ways** — from the terminal or the notebook. **Training requires a
CUDA GPU.** Data layout: `data/3d/<dataset>/{train,val}/{im,label}/*.tif` (movies are
`(T, Z, Y, X)`; labels encode the daughter pair: first daughter → odd ids, second → even ids).

**1. Terminal (CLI).**

```bash
python dare3d/train_eval.py --set_folder <dataset> --epoch 50
# or drive the stages directly:
python dare3d/train.py experiment=segmentation train_dir=3d/<dataset>/train val_dir=3d/<dataset>/val ...
python dare3d/train.py experiment=regression   train_dir=3d/<dataset>/train val_dir=3d/<dataset>/val ...
```

**2. Notebook.** Open `Run_dare3d_Retraining.ipynb` — it validates your dataset layout and drives
segmentation → regression → evaluation, streaming the logs inline.

Outputs land in `logs/<task>/runs/<date>/` (`.hydra/config.yaml` + `checkpoints/`), ready for the
inference step. (The napari plugin also exposes a **DARE3D training** widget that wraps these same
`train.py` / `eval.py` steps.)

## Checks

```bash
python verify_geometry.py   # quaternion -> axis + coordinate mapping (no napari/models/GPU)
python verify_train.py      # training-command construction + paths (no GPU)
make test                   # unit tests (excludes slow ones)
```

## Troubleshooting

- **Qt / napari GUI issues:** `conda install -c conda-forge pyqt`.
- **Full Hydra tracebacks:** set `HYDRA_FULL_ERROR=1`.
- **CUDA out of memory:** lower `data.batch_size`, or `trainer=cpu` for segmentation-only.
- DARE3D training was developed on Linux/HPC; on some Windows setups the Lightning backward pass
  can crash natively even when inference is fine — train on Linux/HPC and reuse the resulting
  model directory.

## License & attribution

MIT — see [LICENSE](LICENSE). This work was granted access to the HPC resources of IDRIS under the
allocation AD010314339 made by GENCI.
