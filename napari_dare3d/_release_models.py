"""Portable model selections for the DARE3D v2 Zenodo release bundle."""

from pathlib import Path
from typing import Optional, Tuple


RELEASE_ROOT_NAME = "DARE3dv2_Zenodo_040926"
DEFAULT_DATASET = "gastruloid"
DATASET_CHOICES = ("gastruloid", "neural_tube")

_PRESETS = {
    "gastruloid": {
        "movie": Path("Gastruloid_241025/test_input/movie2.tif"),
        "segmentation_scale_mode": "source",
        "segmentation": (
            Path("Gastruloid_241025/weights/segmentation3d_exp10-b"),
            "DARE3D_gastruloid_segmentation_epoch067.ckpt",
        ),
        "regression": (
            Path("Gastruloid_241025/weights/regression3d_exp10-b"),
            "DARE3D_gastruloid_regression_epoch095.ckpt",
        ),
    },
    "neural_tube": {
        "movie": Path("Neural_tube_160226/test_input/im/movie_M.tif"),
        "segmentation_scale_mode": "checkpoint_default",
        "segmentation": (
            Path(
                "Neural_tube_160226/weights/segmentation3d_new_set_og/"
                "runs/12-01-26"
            ),
            "DARE3D_neural_tube_segmentation_epoch057.ckpt",
        ),
        "regression": (
            Path(
                "Neural_tube_160226/weights/regression3d_new_set_og/"
                "runs/12-01-26"
            ),
            "DARE3D_neural_tube_regression_epoch139.ckpt",
        ),
    },
}


def find_release_root() -> Optional[Path]:
    """Locate the unpacked v2 Zenodo bundle without workstation-specific paths."""
    candidates = (
        Path.cwd() / RELEASE_ROOT_NAME,
        Path(__file__).resolve().parents[1] / RELEASE_ROOT_NAME,
        Path(__file__).resolve().parents[2] / RELEASE_ROOT_NAME,
    )
    for root in candidates:
        if root.is_dir():
            return root.resolve()
    return None


def release_model_selection(
    dataset: str, stage: str, root: Optional[Path] = None
) -> Tuple[Path, Path]:
    """Return the saved-config directory and explicit promoted checkpoint."""
    if dataset not in _PRESETS:
        raise ValueError(f"Unknown release dataset: {dataset!r}")
    if stage not in ("segmentation", "regression"):
        raise ValueError(f"Unknown model stage: {stage!r}")
    release_root = Path(root) if root is not None else find_release_root()
    if release_root is None:
        return Path(), Path()
    model_rel, checkpoint_name = _PRESETS[dataset][stage]
    return release_root / model_rel, release_root / checkpoint_name


def release_movie_path(dataset: str, root: Optional[Path] = None) -> Path:
    """Return the representative movie shipped for a release dataset."""
    if dataset not in _PRESETS:
        raise ValueError(f"Unknown release dataset: {dataset!r}")
    release_root = Path(root) if root is not None else find_release_root()
    if release_root is None:
        return Path()
    return release_root / _PRESETS[dataset]["movie"]


def release_segmentation_scale_mode(
    dataset: str, checkpoint: Path, root: Optional[Path] = None
) -> str:
    """Return checkpoint compatibility metadata for a release selection.

    The exception is path-specific: a checkpoint manually substituted in the
    widget must keep normal source-scale preprocessing, even if its filename
    happens to match the promoted legacy checkpoint.
    """
    _model_dir, promoted_checkpoint = release_model_selection(
        dataset, "segmentation", root
    )
    if promoted_checkpoint == Path() or not checkpoint:
        return "source"
    if Path(checkpoint).resolve() != promoted_checkpoint.resolve():
        return "source"
    return str(_PRESETS[dataset]["segmentation_scale_mode"])


def model_dir_from_checkpoint(checkpoint: Path, stage: str) -> Path:
    """Infer the saved-config directory for a release or conventional checkpoint."""
    checkpoint = Path(checkpoint).resolve()
    if stage not in ("segmentation", "regression"):
        raise ValueError(f"Unknown model stage: {stage!r}")

    if checkpoint.parent.name == "checkpoints":
        model_dir = checkpoint.parent.parent
        if (model_dir / ".hydra" / "config.yaml").is_file():
            return model_dir

    for dataset in DATASET_CHOICES:
        model_rel, checkpoint_name = _PRESETS[dataset][stage]
        if checkpoint.name == checkpoint_name:
            model_dir = checkpoint.parent / model_rel
            if (model_dir / ".hydra" / "config.yaml").is_file():
                return model_dir

    raise FileNotFoundError(
        "Could not infer a model config directory from checkpoint: "
        f"{checkpoint}. Expected a conventional <model>/checkpoints/*.ckpt path "
        "or a named checkpoint in the DARE3D v2 release root."
    )
