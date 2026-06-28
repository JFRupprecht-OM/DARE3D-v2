"""
Subprocess-based DARE3D training orchestration for the napari plugin — napari-free.

Mirrors ``dare3d/train_eval.py`` (segmentation -> regression -> evaluation) but
RE-USES ``dare3d/train.py`` and ``dare3d/eval.py`` via subprocess with portability
fixes the upstream script lacks:
  - ``sys.executable`` (this env's python) instead of a bare ``"python"``;
  - absolute script paths resolved from the installed ``dare3d`` package;
  - a LOCAL mlflow ``tracking_uri`` (upstream hardcodes another user's path), kept
    because ``eval.py`` requires an MLflow logger;
  - the user's dataset via absolute ``data.*_data.{im,label}_folder`` overrides;
  - a configurable output/logs dir via ``paths.log_dir``.

Training requires a CUDA device. ``dare3d`` must be importable (``pip install -e .``).
"""
import os
import shutil
import subprocess
import sys
from glob import glob
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

from napari_dare3d._io import iter_tifs


def repo_root() -> Path:
    """Root of the editable ``dare3d`` repo (holds ``dare3d/``, ``configs/``)."""
    import dare3d  # deferred: keeps this module import light (no torch)

    return Path(dare3d.__file__).resolve().parents[1]


def _posix(p) -> str:
    return Path(p).resolve().as_posix()


def _mlflow_uri(output_dir) -> str:
    # mlflow >=3 rejects the file:// store ("in maintenance mode"); use a local
    # SQLite backend (its supported local store). Also avoids the upstream
    # hardcoded path. eval.py re-instantiates this logger from the saved config.
    return "sqlite:///" + _posix(Path(output_dir) / "mlflow.db")


def _data_overrides(dataset_dir) -> List[str]:
    """Absolute data-folder overrides (train/val; test = val), forward-slash paths."""
    ds = Path(dataset_dir)

    def f(*parts: str) -> str:
        return _posix(ds.joinpath(*parts))

    return [
        f"data.train_data.im_folder={f('train', 'im')}",
        f"data.train_data.label_folder={f('train', 'label')}",
        f"data.val_data.im_folder={f('val', 'im')}",
        f"data.val_data.label_folder={f('val', 'label')}",
        f"data.test_data.im_folder={f('val', 'im')}",
        f"data.test_data.label_folder={f('val', 'label')}",
    ]


def model_dirs(output_dir, name: str, date: str) -> Dict[str, Path]:
    """Where train.py will write each model_dir (drop-in for the inference widget)."""
    out = Path(output_dir)
    return {
        "segmentation": out / f"segmentation3d_{name}" / "runs" / date,
        "regression": out / f"regression3d_{name}" / "runs" / date,
    }


def seg_command(dataset_dir, output_dir, name, date, epochs, batch_size,
                cell_radius, seg_crop_size) -> List[str]:
    """``dare3d/train.py experiment=segmentation ...`` (mirrors train_eval, fixed)."""
    train_py = _posix(repo_root() / "dare3d" / "train.py")
    return [
        sys.executable, train_py,
        "experiment=segmentation",
        f"task_name=segmentation3d_{name}",
        "trainer.accelerator=gpu",
        f"data.batch_size={int(batch_size)}",
        "data.num_workers=0",
        "steps_per_epoch=1000",
        f"trainer.max_epochs={int(epochs)}",
        "model/criterion=dice_focal",
        "model.optimizer.lr=0.1",
        "model/scheduler=one_cycle_lr",
        "model.scheduler_interval='step'",
        "renorm='min-max'",
        "time_axis_padding=1",
        f"cell_radius={int(cell_radius)}",
        f"date={date}",
        f"crop_size={int(seg_crop_size)}",
        f"paths.log_dir={_posix(output_dir)}",
        f"logger.mlflow.tracking_uri={_mlflow_uri(output_dir)}",
        *_data_overrides(dataset_dir),
        # sparse weights are a segmentation-only, optional input (dataset warns ->
        # all-ones if the folder/movie is absent); regression has no such field.
        f"data.train_data.sparse_folder={_posix(Path(dataset_dir) / 'train' / 'weights')}",
    ]


def reg_command(dataset_dir, output_dir, name, date, epochs, batch_size) -> List[str]:
    """``dare3d/train.py experiment=regression ...`` (mirrors train_eval, fixed)."""
    train_py = _posix(repo_root() / "dare3d" / "train.py")
    return [
        sys.executable, train_py,
        "experiment=regression",
        f"task_name=regression3d_{name}",
        "trainer.accelerator=gpu",
        f"data.batch_size={int(batch_size)}",
        "data.num_workers=0",
        "steps_per_epoch=1000",
        f"trainer.max_epochs={int(epochs)}",
        f"date={date}",
        "model.optimizer.lr=0.001",
        "model/net=simple_regression_net",
        "model.net.n_stages=3",
        "model.net.start_filters=32",
        f"paths.log_dir={_posix(output_dir)}",
        f"logger.mlflow.tracking_uri={_mlflow_uri(output_dir)}",
        *_data_overrides(dataset_dir),
    ]


def _best_ckpt(model_dir) -> Optional[str]:
    """Filename of the ``epoch_*.ckpt`` if present (like train_eval.find_best_model)."""
    for path in glob(_posix(Path(model_dir) / "checkpoints" / "*.ckpt")):
        if "epoch" in Path(path).stem:
            return Path(path).name
    return None


def eval_command(seg_dir, reg_dir, threshold: Optional[float] = None) -> List[str]:
    """``dare3d/eval.py segmentation.model_dir=... regression.model_dir=...``."""
    eval_py = _posix(repo_root() / "dare3d" / "eval.py")
    cmd = [
        sys.executable, eval_py,
        f"segmentation.model_dir={_posix(seg_dir)}",
        f"regression.model_dir={_posix(reg_dir)}",
    ]
    if threshold is not None:  # else use eval.yaml default (0.5)
        cmd.append(f"segmentation.threshold={float(threshold)}")
    best_seg = _best_ckpt(seg_dir)
    if best_seg is not None:
        cmd.append(f"segmentation.ckpt_name={best_seg}")
    best_reg = _best_ckpt(reg_dir)
    if best_reg is not None:
        cmd.append(f"regression.ckpt_name={best_reg}")
    return cmd


def available_movies(dataset_dir) -> List[str]:
    """Movie subfolders (each with ``im/`` + ``label/``) in a raw per-movie dataset."""
    ds = Path(dataset_dir)
    if not ds.is_dir():
        return []
    return sorted(
        p.name for p in ds.iterdir()
        if p.is_dir() and (p / "im").is_dir() and (p / "label").is_dir()
    )


def _resolve_split(dataset_dir, output_dir, train_movies, val_movies) -> Path:
    """Return a dataset dir laid out as ``train/{im,label}`` + ``val/{im,label}``.

    Accepts two layouts:
      * already split: ``<ds>/{train,val}/{im,label}/*.tif`` -> used as-is (movie
        lists ignored);
      * raw per-movie (e.g. the DARE3D trainingset): ``<ds>/<movie>/{im,label}/*.tif``
        where each tif is a whole ``(T,Z,Y,X)`` movie -> consolidated via symlinks
        (names prefixed by movie folder) into a fresh ``<output>/_dare3d_split``,
        using the named ``train_movies`` for training and ``val_movies`` for
        validation/testing. **Any movie folder not listed is excluded.**

    Symlinks (no copy) keep it fast and space-free; created where training runs
    (Linux/WSL supports them without privilege).
    """
    ds = Path(dataset_dir)
    for p in (ds, Path(output_dir)):
        if " " in str(p):
            raise ValueError(f"Path must not contain spaces (Hydra parsing): {p}")

    if (ds / "train" / "im").is_dir() and (ds / "val" / "im").is_dir():
        return ds  # already split

    avail = available_movies(ds)
    if not avail:
        raise FileNotFoundError(
            "Dataset must be either <ds>/{train,val}/{im,label}/*.tif (pre-split) or "
            f"<ds>/<movie>/{{im,label}}/*.tif (per-movie) — neither found in {ds}"
        )
    train_mv = [m.strip() for m in (train_movies or []) if m and m.strip()]
    val_mv = [m.strip() for m in (val_movies or []) if m and m.strip()]
    if not train_mv or not val_mv:
        raise ValueError(
            "Choose which movie folders to train on and which to test on. "
            f"Available: {avail}"
        )
    missing = [m for m in train_mv + val_mv if m not in avail]
    if missing:
        raise FileNotFoundError(f"Movie folder(s) not found: {missing}. Available: {avail}")

    split = Path(output_dir) / "_dare3d_split"
    if split.exists():
        shutil.rmtree(split)
    for sub, mvs in (("train", train_mv), ("val", val_mv)):
        for kind in ("im", "label"):
            (split / sub / kind).mkdir(parents=True, exist_ok=True)
        for m in mvs:
            for kind in ("im", "label"):
                for tif_path in iter_tifs(ds / m / kind):  # case-insensitive .tif/.tiff
                    tif = Path(tif_path)
                    #os.symlink(tif, split / sub / kind / f"{m}_{tif.name}")
                    dst = split / sub / kind / f"{m}_{tif.name}"
                    try:
                        os.symlink(tif, dst)
                    except (OSError, NotImplementedError):
                        shutil.copy2(tif, dst)
    return split


def _stream(cmd: List[str], should_stop: Optional[Callable[[], bool]]) -> Iterator[str]:
    """Run ``cmd``, yield stdout lines live, honour a stop flag, raise on failure."""
    yield f"[DARE3D] $ {' '.join(str(c) for c in cmd[1:4])} ..."
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=str(repo_root()),
    )
    stopped = False
    try:
        for line in proc.stdout:
            yield line.rstrip("\n")
            if should_stop is not None and should_stop():
                proc.terminate()
                stopped = True
                yield "[DARE3D] stop requested — terminating…"
                break
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
    rc = proc.wait()
    if stopped:
        raise RuntimeError("Training stopped by user.")
    if rc != 0:
        raise RuntimeError(f"Step failed (exit {rc}): {Path(cmd[1]).name} {cmd[2]}")


def run_training(
    dataset_dir,
    output_dir,
    name: str,
    date: str,
    epochs: int,
    batch_size: int,
    cell_radius: int = 8,
    seg_crop_size: int = 128,
    train_movies=None,
    val_movies=None,
    train_segmentation: bool = True,
    train_regression: bool = True,
    run_eval: bool = True,
    threshold: Optional[float] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Iterator:
    """Generator: run seg -> reg -> eval as subprocesses, yielding stdout lines.

    ``train_movies``/``val_movies`` name the per-movie subfolders to train and
    test on (ignored if the dataset is already ``train/``+``val/`` split). The
    final yielded value is a dict ``{"segmentation": <dir|None>,
    "regression": <dir|None>}`` of the produced model_dirs (use the LAST yield).
    """
    os.makedirs(output_dir, exist_ok=True)  # so the SQLite mlflow db can be created
    split_dir = _resolve_split(dataset_dir, output_dir, train_movies, val_movies)
    dirs = model_dirs(output_dir, name, date)

    if train_segmentation:
        yield "[DARE3D] === Segmentation training ==="
        yield from _stream(
            seg_command(split_dir, output_dir, name, date, epochs, batch_size,
                        cell_radius, seg_crop_size),
            should_stop,
        )
    if train_regression:
        yield "[DARE3D] === Regression training ==="
        yield from _stream(
            reg_command(split_dir, output_dir, name, date, epochs, batch_size),
            should_stop,
        )
    if run_eval and train_segmentation and train_regression:
        yield "[DARE3D] === Evaluation ==="
        yield from _stream(
            eval_command(dirs["segmentation"], dirs["regression"], threshold),
            should_stop,
        )

    yield {
        "segmentation": str(dirs["segmentation"]) if train_segmentation else None,
        "regression": str(dirs["regression"]) if train_regression else None,
    }
