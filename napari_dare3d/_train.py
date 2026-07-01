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


# --------------------------------------------------------------------------- #
# Fine-tuning (transfer learning) — drives the engine in dare3d/models/finetune.py #
# via experiment=finetune_<stage> + model.finetune.* Hydra overrides.          #
# --------------------------------------------------------------------------- #
def _net_overrides_from_base(base_ckpt, stage: str) -> List[str]:
    """Reproduce the base checkpoint's net architecture as Hydra overrides.

    The engine loads base weights with ``strict=True``, so the fine-tune net must match the
    base exactly. A checkpoint written by ``train.py`` sits in a model_dir with a saved
    ``.hydra/config.yaml``; we read its ``model.net`` and emit ``model/net=<group>`` plus each
    concrete (non-interpolation) net param, so ANY train.py-produced base (widget-trained or a
    published model with a different n_stages/channels) matches. Returns ``[]`` if the sibling
    config is absent (then the experiment's default net is used and a mismatch surfaces as the
    engine's clear stage-aware error).
    """
    try:
        cfg_path = Path(base_ckpt).resolve().parent.parent / ".hydra" / "config.yaml"
        if not cfg_path.is_file():
            return []
        from omegaconf import OmegaConf  # deferred: hydra dep, no torch
        net = OmegaConf.to_container(OmegaConf.load(cfg_path).model.net, resolve=False)
    except Exception:
        return []
    if not isinstance(net, dict):
        return []
    group = {
        "RegressionNet": "simple_regression_net",
        "Regression3dCNN": "regression_net",
        "MultiScaleUNet": "multiscale_unet",
        "SegResMultiScaleUNet": "segres_multiscale_unet",
        "SwinUNETRWrapper": "swinunetr",
    }.get(str(net.get("_target_", "")).split(".")[-1])
    ov: List[str] = [f"model/net={group}"] if group else []
    for k, v in net.items():
        if k == "_target_":
            continue
        if isinstance(v, str) and v.strip().startswith("${"):
            continue  # interpolation (im_size/input_channels/…) -> resolved by the experiment
        if isinstance(v, bool):
            ov.append(f"model.net.{k}={str(v).lower()}")
        elif isinstance(v, (list, tuple)):
            if any(isinstance(x, str) for x in v):
                continue  # structural string lists (e.g. output_names) are CLI-hostile and are
                          # ignored by MultiScaleUNet at downsample_factors=[1] anyway
            ov.append(f"model.net.{k}=[{','.join(str(x) for x in v)}]")
        else:
            ov.append(f"model.net.{k}={v}")
    return ov


def finetune_command(stage: str, dataset_dir, output_dir, name, date, base_ckpt, ft: Dict,
                     epochs: int, batch_size: int, num_workers: int = 2) -> List[str]:
    """``dare3d/train.py experiment=finetune_<stage> ...`` — one fine-tune stage.

    Override names match ``configs/experiment/finetune_<stage>.yaml`` (``model.finetune.*``)
    verbatim. ``stage`` is ``"segmentation"`` or ``"regression"``; ``ft`` carries the Advanced
    GUI params. ``discriminative`` is a GUI convenience: OFF -> ``backbone_lr_mult=1.0`` (uniform
    LR); ON -> the given multiplier. No backend flag (PyTorch-only).
    """
    train_py = _posix(repo_root() / "dare3d" / "train.py")
    disc_mult = ft["backbone_lr_mult"] if ft.get("discriminative", True) else "1.0"
    cmd = [
        sys.executable, train_py,
        f"experiment=finetune_{stage}",
        f"model.finetune.base_ckpt={_posix(base_ckpt)}",
        f"task_name={stage}3d_{name}",
        "trainer.accelerator=gpu",
        f"data.batch_size={int(batch_size)}",
        f"data.num_workers={int(num_workers)}",   # modest (spawn); do not jump to 11
        "data.pin_memory=true",
        f"trainer.max_epochs={int(epochs)}",
        f"date={date}",
        f"paths.log_dir={_posix(output_dir)}",
        f"logger.mlflow.tracking_uri={_mlflow_uri(output_dir)}",
        # --- engine hyperparameters (verbatim model.finetune.* names) ---
        f"model.finetune.freeze_preset={ft['freeze_preset']}",
        f"model.finetune.unfreeze_last_stages={int(ft['unfreeze_last_stages'])}",
        f"model.finetune.bn_mode={ft['bn_mode']}",
        f"model.finetune.ft_lr={ft['ft_lr']}",
        f"model.finetune.backbone_lr_mult={disc_mult}",
        f"model.finetune.weight_decay={ft['weight_decay']}",
        f"model.finetune.lr_schedule={ft['lr_schedule']}",
        f"model.finetune.warmup_epochs={int(ft['warmup_epochs'])}",
        f"model.finetune.grad_clip={ft['grad_clip']}",   # -> trainer.gradient_clip_val (interp)
        f"model.finetune.augment={str(bool(ft.get('augment', True))).lower()}",
        f"model.finetune.augment_strength={ft['augment_strength']}",
        f"model.finetune.patience={int(ft['patience'])}",
        f"model.finetune.seed={int(ft['seed'])}",
        f"seed={int(ft['seed'])}",   # top-level seed drives L.seed_everything
        *_data_overrides(dataset_dir),
        *_net_overrides_from_base(base_ckpt, stage),
    ]
    if ft.get("augment", True):
        # augment_strength = per-sample probability of applying the augmentation pipeline
        # (MonaiAugmentationWrapper.prob). ++ overrides it whether or not the config declares it
        # (monai_augmentation_best sets prob; monai_augmentation_reg relies on the default).
        try:
            aprob = min(1.0, max(0.0, float(ft.get("augment_strength", 0.5))))
        except (TypeError, ValueError):
            aprob = 0.5
        cmd.append(f"++data.augmentation.prob={aprob}")
    else:
        cmd.append("data/augmentation=none")   # disable augmentation entirely
    if stage == "segmentation":
        cmd.append(f"cell_radius={int(ft.get('cell_radius', 8))}")
        cmd.append(f"crop_size={int(ft.get('seg_crop_size', 128))}")
        cmd.append(f"data.train_data.sparse_folder={_posix(Path(dataset_dir) / 'train' / 'weights')}")
    return cmd


def _read_base_geometry(base_ckpt) -> Dict:
    """Geometry the base was trained with, read from its saved ``.hydra/config.yaml`` (preferred
    over hardcoding). Returns ``{}`` if the base has no sibling config (e.g. a bare ``.ckpt``) — the
    caller then skips the soft geometry warnings and leans on the engine's strict-load + a runtime
    shape error. NOTE: a base carries a ``finetune_config.json`` sidecar only if it was ITSELF
    fine-tuned; the *training* geometry always lives in ``.hydra/config.yaml``, so that is the source.
    """
    out: Dict = {}
    try:
        cfg_path = Path(base_ckpt).resolve().parent.parent / ".hydra" / "config.yaml"
        if not cfg_path.is_file():
            return out
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(cfg_path)
        out["crop_size"] = OmegaConf.select(cfg, "crop_size")
        out["cell_radius"] = OmegaConf.select(cfg, "cell_radius")
        strides = OmegaConf.select(cfg, "model.net.strides")
        if strides is not None:
            prod = 1
            for s in OmegaConf.to_container(strides, resolve=False):
                prod *= int(s[0] if isinstance(s, (list, tuple)) else s)
            out["stride_product"] = prod
    except Exception:
        return {}
    return out


def finetune_preflight(stages: List[str], base_ckpts: Dict[str, str], ft: Dict) -> Dict[str, List[str]]:
    """Pre-dispatch check for fine-tune conflicts (see the Advanced-parameter audit). Pure, no side
    effects. Returns ``{"errors", "warnings", "notes"}``:
      - errors   = HARD (crash / wrong-shaped load)  -> caller BLOCKS dispatch;
      - warnings = SOFT (loads/runs but degrades transfer) -> caller asks for confirmation;
      - notes    = information (e.g. base geometry couldn't be read).
    Architecture params (channels/strides/start_filters/n_stages/…) are IMPOSED by the base and are
    inherited automatically by ``_net_overrides_from_base``; a real mismatch there surfaces as the
    engine's strict-load stage-aware error, which is the backstop for this check.
    """
    errors: List[str] = []
    warnings: List[str] = []
    notes: List[str] = []

    # small-float text fields must parse as numbers (Hydra would otherwise fail mid-run)
    for fld in ("ft_lr", "backbone_lr_mult", "weight_decay", "grad_clip", "augment_strength"):
        v = ft.get(fld)
        if v is not None:
            try:
                float(v)
            except (TypeError, ValueError):
                errors.append(f"{fld}={v!r} is not a number.")

    # --- segmentation geometry: the seg net is fully-convolutional, so crop is RUNTIME, not a
    #     weight-shape. It only has a HARD divisibility constraint + SOFT transfer degradation. ---
    if "segmentation" in stages:
        geo = _read_base_geometry(base_ckpts.get("segmentation"))
        crop = int(ft.get("seg_crop_size", 128))
        prod = geo.get("stride_product")
        if prod and crop % prod != 0:
            errors.append(f"seg_crop_size={crop} is not divisible by {prod} (product of the U-Net "
                          f"strides); the down/up path would give mismatched skip-connection shapes "
                          f"and crash. Use a multiple of {prod}.")
        bc = geo.get("crop_size")
        if bc is not None and crop != int(bc):
            warnings.append(f"seg_crop_size={crop} differs from the base's training crop ({bc}). The "
                            f"seg net is fully-convolutional so weights load fine, but the receptive-"
                            f"field/patch mismatch can degrade transfer.")
        if not geo:
            notes.append("segmentation base has no .hydra/config.yaml — cannot verify crop divisibility "
                         "or the base crop/radius; relying on the engine's strict-load + runtime shape check.")
        br = geo.get("cell_radius")
        cr = int(ft.get("cell_radius", 8))
        if br is not None and cr != int(br):
            warnings.append(f"cell_radius={cr} differs from the base's training radius ({br}); this "
                            f"resizes the target spheres vs what the base learned (soft transfer shift).")

    # --- optimization interactions with a frozen pretrained backbone ---
    try:
        bs = int(ft.get("batch_size", 4))
    except (TypeError, ValueError):
        bs = 4
    if str(ft.get("bn_mode")) == "adapt" and bs < 8:
        warnings.append(f"bn_mode='adapt' with batch_size={bs} (<8): frozen BatchNorm3d re-estimates "
                        f"running stats from tiny batches -> noisy/biased normalisation. Prefer "
                        f"bn_mode='frozen', or use a larger batch.")
    try:
        if float(ft.get("ft_lr", 1e-4)) > 1e-3:
            warnings.append(f"ft_lr={ft.get('ft_lr')} is scratch-hot for pretrained weights; fine-tuning "
                            f"usually needs <= 1e-3 so the transferred features are not destroyed.")
    except (TypeError, ValueError):
        pass  # already reported as a parse error above
    preset = str(ft.get("freeze_preset"))
    try:
        uls = int(ft.get("unfreeze_last_stages", 0))
    except (TypeError, ValueError):
        uls = 0
    if uls > 0 and preset == "none":
        warnings.append(f"unfreeze_last_stages={uls} has no effect with freeze_preset='none' (nothing "
                        f"is frozen); use freeze_preset=encoder/encoder_partial to unfreeze only the "
                        f"last stages.")
    # augment_strength is wired to the per-sample augmentation probability (data.augmentation.prob).
    # Heavy augmentation while the backbone is fully frozen shifts the frozen features (which can't
    # adapt) and can hurt transfer -> soft-warn only when it is genuinely heavy (the 0.5 default is
    # fine, so this does not fire on the recommended setup).
    try:
        aprob = float(ft.get("augment_strength", 0.5))
    except (TypeError, ValueError):
        aprob = 0.5
    if bool(ft.get("augment", True)) and aprob > 0.75 and preset == "encoder" and uls == 0:
        warnings.append(f"augment_strength={aprob} (heavy) with a fully-frozen backbone: only the head "
                        f"trains, so frequent augmentation shifts the frozen features and can hurt "
                        f"transfer. Consider a lower strength (~0.5) when training the head only.")
    return {"errors": errors, "warnings": warnings, "notes": notes}


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
        encoding="utf-8", errors="replace", bufsize=1, cwd=str(repo_root()),
        # PYTHONUNBUFFERED/IOENCODING: live, clean UTF-8 stream on Windows (cp1252 would mojibake
        # the progress glyphs / non-ASCII in Lightning's output and stall line buffering).
        env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
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


def run_finetuning(
    dataset_dir,
    output_dir,
    name: str,
    date: str,
    epochs: int,
    batch_size: int,
    *,
    stages: List[str],
    base_ckpts: Dict[str, str],
    ft: Dict,
    train_movies=None,
    val_movies=None,
    num_workers: int = 2,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Iterator:
    """Generator: fine-tune ``stages`` (segmentation before regression), each from its OWN base
    checkpoint, as subprocesses, yielding stdout lines. Mirrors :func:`run_training` but drives
    the fine-tune engine. ``base_ckpts`` maps ``"segmentation"``/``"regression"`` -> base
    ``.ckpt``; ``ft`` carries the Advanced params. Final yielded value is the model_dirs dict.
    """
    os.makedirs(output_dir, exist_ok=True)  # so the SQLite mlflow db can be created
    split_dir = _resolve_split(dataset_dir, output_dir, train_movies, val_movies)
    dirs = model_dirs(output_dir, name, date)

    for stage in [s for s in ("segmentation", "regression") if s in stages]:  # seg -> reg order
        base = base_ckpts.get(stage)
        yield f"[DARE3D] === Fine-tune {stage} (base: {Path(base).name if base else '?'}) ==="
        yield from _stream(
            finetune_command(stage, split_dir, output_dir, name, date, base, ft,
                             epochs, batch_size, num_workers),
            should_stop,
        )

    yield {
        "segmentation": str(dirs["segmentation"]) if "segmentation" in stages else None,
        "regression": str(dirs["regression"]) if "regression" in stages else None,
    }
