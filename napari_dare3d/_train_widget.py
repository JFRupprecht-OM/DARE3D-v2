"""napari widget for DARE3D (re)training and fine-tuning (magicgui ``magic_factory``).

Thin GUI over :mod:`napari_dare3d._train`:
  - **Retrain from scratch** -> :func:`_train.run_training` (seg -> reg -> eval).
  - **Transfer learning — fine-tune** -> :func:`_train.run_finetuning`, which drives the
    fine-tune engine (``experiment=finetune_<stage>`` + ``model.finetune.*`` overrides), each
    stage from its OWN base ``.ckpt``.
Both run DARE3D's own ``train.py`` as subprocesses; the worker streams Lightning's stdout live
and Stop terminates the process. This widget does correct wiring + dispatch only — the engine
(BN policy, freeze, discriminative LR, sidecar) lives in ``dare3d/models/finetune.py``.
"""
import datetime
from pathlib import Path

import napari
from magicgui import magic_factory
from magicgui.widgets import ProgressBar
from napari.qt.threading import thread_worker
from napari.utils import notifications

from napari_dare3d import _train
from napari_dare3d._widget import _data_root, _maybe_dir

#: Shared state (single widget instance in practice): Stop flag + call/stop button refs.
_TRAIN_STATE = {"stop": False, "call_button": None, "stop_button": None}

#: Advanced (fine-tune) controls — collapsed by default, shown only in fine-tune mode.
_ADVANCED = ("freeze_preset", "unfreeze_last_stages", "bn_mode", "ft_lr", "discriminative",
             "backbone_lr_mult", "weight_decay", "lr_schedule", "warmup_epochs", "grad_clip",
             "augment", "augment_strength", "patience", "seed")

_STAGES = {"both": ["segmentation", "regression"],
           "segmentation": ["segmentation"], "regression": ["regression"]}


def _set_running(running: bool) -> None:
    """While training runs, hide the "Run" call button and show Stop in its place."""
    cb = _TRAIN_STATE.get("call_button")
    sb = _TRAIN_STATE.get("stop_button")
    if cb is not None:
        cb.visible = not running
    if sb is not None:
        sb.visible = running


def _default_trainingset() -> Path:
    """Default dataset: the Gastruloid per-movie ``trainingset/`` (if present)."""
    root = _data_root()
    if root is not None:
        p = root / "Gastruloid_241025" / "trainingset"
        if p.is_dir():
            return p
    return Path()


def _set_advanced_visible(widget, show: bool) -> None:
    for name in _ADVANCED:
        w = getattr(widget, name, None)
        if w is not None:
            w.visible = show
    if getattr(widget, "advanced", None) is not None:
        widget.advanced.text = "Hide advanced parameters" if show else "Show advanced parameters"


def _apply_mode(widget) -> None:
    """Show/hide the fine-tune UI. Base pickers appear ONLY in fine-tune mode, by stage;
    the Advanced toggle appears only in fine-tune (collapsed); eval controls are scratch-only."""
    finetune = str(widget.mode.value).startswith("Transfer")
    stage = str(widget.stage.value)
    widget.base_reg.visible = finetune and stage in ("both", "regression")
    widget.base_seg.visible = finetune and stage in ("both", "segmentation")
    if getattr(widget, "advanced", None) is not None:
        widget.advanced.visible = finetune
    widget.run_eval.visible = not finetune       # eval is a scratch-flow step
    widget.threshold.visible = not finetune
    if not finetune:                             # collapse + hide all fine-tune controls
        _set_advanced_visible(widget, False)


def _init_training_widget(widget) -> None:
    """magic_factory hook: wire Stop (replaces Run while active), the Advanced collapse, and
    the Mode/stage-driven visibility of the base pickers."""
    if getattr(widget, "call_button", None) is not None:
        widget.call_button.tooltip = "Start DARE3D training / fine-tuning with the settings above."
    _TRAIN_STATE["call_button"] = getattr(widget, "call_button", None)
    _TRAIN_STATE["stop_button"] = getattr(widget, "stop", None)
    try:
        widget.stop.visible = False  # shown only while a run is active (replaces Run)
        widget.stop.tooltip = "Terminate the running training process."
        widget.stop.changed.connect(lambda *_: _TRAIN_STATE.__setitem__("stop", True))
    except Exception:
        pass

    # Collapsible Advanced parameters (magicgui has no native collapsible -> flip .visible).
    _set_advanced_visible(widget, False)
    if getattr(widget, "advanced", None) is not None:
        widget.advanced.changed.connect(
            lambda *_: _set_advanced_visible(widget, not widget.freeze_preset.visible)
        )

    widget.mode.changed.connect(lambda *_: _apply_mode(widget))
    widget.stage.changed.connect(lambda *_: _apply_mode(widget))
    _apply_mode(widget)   # initial state = scratch (base pickers + Advanced hidden)


def _autofill_inference(viewer, result: dict) -> None:
    """Best-effort: push the produced model dirs into an open inference widget."""
    if viewer is None:
        return
    try:
        for dock in viewer.window._dock_widgets.values():
            inner = getattr(dock.widget(), "_magic_widget", None)
            if inner is not None and hasattr(inner, "seg_model_dir") and hasattr(inner, "reg_model_dir"):
                if result.get("segmentation"):
                    inner.seg_model_dir.value = result["segmentation"]
                if result.get("regression"):
                    inner.reg_model_dir.value = result["regression"]
    except Exception:
        pass


@magic_factory(
    call_button="Run training",
    widget_init=_init_training_widget,
    tooltips=False,
    mode={"widget_type": "ComboBox",
          "choices": ["Retrain from scratch", "Transfer learning — fine-tune"], "label": "Mode",
          "tooltip": "Retrain from scratch, or fine-tune a pretrained DARE3D checkpoint "
                     "(transfer learning). Fine-tune reveals the base pickers + Advanced options."},
    stage={"widget_type": "ComboBox", "choices": ["both", "segmentation", "regression"],
           "label": "Model", "tooltip": "Which stage(s) to (re)train / fine-tune: both, just the "
                                        "segmentation (centre) model, or just the regression (axis) model."},
    base_reg={"widget_type": "FileEdit", "mode": "r", "label": "Regression base (.ckpt)",
              "filter": "*.ckpt",
              "tooltip": "Pretrained REGRESSION checkpoint to fine-tune (a model_dir's "
                         "checkpoints/last.ckpt). Shown for Model = regression or both."},
    base_seg={"widget_type": "FileEdit", "mode": "r", "label": "Segmentation base (.ckpt)",
              "filter": "*.ckpt",
              "tooltip": "Pretrained SEGMENTATION checkpoint to fine-tune (a model_dir's "
                         "checkpoints/last.ckpt). Shown for Model = segmentation or both."},
    dataset_dir={
        "widget_type": "FileEdit", "mode": "d", "label": "Dataset dir",
        "tooltip": "Per-movie dataset <dataset>/<movie>/{im,label}/*.tif (each tif a whole "
                   "T,Z,Y,X movie; labels: odd=first daughter, even=second), or a pre-split "
                   "<dataset>/{train,val}/{im,label}. Default: Gastruloid trainingset.",
    },
    train_movies={"label": "Training movies",
                  "tooltip": "Comma-separated movie subfolders to TRAIN on. Folders not listed "
                             "here or in Test movies are excluded."},
    test_movies={"label": "Test movies",
                 "tooltip": "Comma-separated movie subfolders for validation/testing."},
    output_dir={"widget_type": "FileEdit", "mode": "d", "label": "Output / logs dir",
                "tooltip": "Where model_dirs are written (logs/{task}/runs/{date}/). "
                           "Blank = the dare3d repo's logs/ folder."},
    run_name={"tooltip": "Experiment name (-> task_name <stage>3d_<name>). Blank = dataset folder name."},
    date={"tooltip": "Run date/id, used in the output path runs/<date>/."},
    epochs={"min": 1, "tooltip": "Max training epochs per model. Default: 50."},
    batch_size={"min": 1, "tooltip": "3D patches per step. Lower if you hit GPU out-of-memory. Default: 4."},
    cell_radius={"min": 1, "tooltip": "Radius (voxels) of the segmentation target spheres. Default: 8."},
    seg_crop_size={"min": 16, "tooltip": "3D training patch size for segmentation. Default: 128."},
    run_eval={"tooltip": "Scratch only: after training, run dare3d eval.py on the val set "
                         "(needs both models + GT). Default: on."},
    threshold={"min": 0.0, "max": 1.0, "step": 0.05,
               "tooltip": "Scratch eval: segmentation probability threshold. Default: 0.5."},
    advanced={"widget_type": "PushButton", "text": "Show advanced parameters",
              "tooltip": "Show/hide the fine-tune options (freeze depth, learning rates, "
                         "schedule, BN policy, augmentation, early stopping, seed)."},
    freeze_preset={"widget_type": "ComboBox", "choices": ["encoder", "encoder_partial", "none"],
                   "label": "Freeze preset",
                   "tooltip": "encoder: freeze the backbone, train the decoder/head(s). "
                              "encoder_partial: also unfreeze the last N stages. none: full fine-tune."},
    unfreeze_last_stages={"min": 0, "label": "Unfreeze last N stages",
                          "tooltip": "With encoder_partial, unfreeze the N deepest backbone stages. Default: 0."},
    bn_mode={"widget_type": "ComboBox", "choices": ["frozen", "adapt"], "label": "Frozen-backbone BN",
             "tooltip": "frozen (default): BN running stats + affine fixed (a true freeze). "
                        "adapt: BN re-estimates running stats on the new data (affine stays frozen) "
                        "— for a larger, distribution-shifted fine-tune set."},
    ft_lr={"widget_type": "LineEdit", "label": "Fine-tune LR",
           "tooltip": "Learning rate for the trained (unfrozen) params. Default: 1e-4."},
    discriminative={"label": "Discriminative LR",
                    "tooltip": "Use a lower LR for the unfrozen backbone than the head "
                               "(off -> uniform LR = backbone mult 1.0). Default: on."},
    backbone_lr_mult={"widget_type": "LineEdit", "label": "Backbone LR x",
                      "tooltip": "Unfrozen-backbone LR = Fine-tune LR x this. Default: 0.1."},
    weight_decay={"widget_type": "LineEdit", "label": "Weight decay",
                  "tooltip": "AdamW weight decay. Default: 1e-4."},
    lr_schedule={"widget_type": "ComboBox", "choices": ["warmup_cosine", "cosine"], "label": "LR schedule",
                 "tooltip": "Linear warmup then cosine (default), or cosine (warmup_epochs=0)."},
    warmup_epochs={"min": 0, "label": "Warmup epochs",
                   "tooltip": "Linear LR warmup epochs before cosine. Default: 2."},
    grad_clip={"widget_type": "LineEdit", "label": "Grad clip (max-norm)",
               "tooltip": "Gradient max-norm clipping; 0 = off. Default: 1.0."},
    augment={"label": "Augmentation", "tooltip": "Apply training-data augmentations. Default: on."},
    augment_strength={"widget_type": "LineEdit", "label": "Augment strength",
                      "tooltip": "Recorded in the sidecar (provenance). Default: 1.0."},
    patience={"min": 0, "label": "Early-stop patience",
              "tooltip": "Stop if val metric doesn't improve for N epochs. Default: 10."},
    seed={"label": "Seed", "tooltip": "Random seed (python/numpy/torch/workers). Default: 12345."},
    stop={"widget_type": "PushButton", "text": "Stop"},
    pbar={"label": "progress", "visible": False, "min": 0, "max": 0},
)
def dare3d_training_widget(
    mode: str = "Retrain from scratch",
    stage: str = "both",
    base_reg: Path = Path(),
    base_seg: Path = Path(),
    dataset_dir: Path = _default_trainingset(),
    train_movies: str = "movie3,movie4",
    test_movies: str = "movie2",
    output_dir: Path = Path(),
    run_name: str = "",
    date: str = datetime.date.today().isoformat(),
    epochs: int = 50,
    batch_size: int = 4,
    cell_radius: int = 8,
    seg_crop_size: int = 128,
    run_eval: bool = True,
    threshold: float = 0.5,
    advanced: bool = False,
    freeze_preset: str = "encoder",
    unfreeze_last_stages: int = 0,
    bn_mode: str = "frozen",
    ft_lr: str = "1e-4",
    discriminative: bool = True,
    backbone_lr_mult: str = "0.1",
    weight_decay: str = "1e-4",
    lr_schedule: str = "warmup_cosine",
    warmup_epochs: int = 2,
    grad_clip: str = "1.0",
    augment: bool = True,
    augment_strength: str = "1.0",
    patience: int = 10,
    seed: int = 12345,
    stop: bool = False,
    pbar: ProgressBar = None,
):
    """(Re)train from scratch or fine-tune DARE3D models on a labelled dataset (GPU, subprocess).
    Fine-tune mode loads a pretrained base .ckpt per stage and drives the fine-tune engine.
    Produces model_dirs usable by the inference widget."""
    ds = _maybe_dir(dataset_dir)
    if ds is None:
        notifications.show_warning("DARE3D: choose a valid dataset folder.")
        return
    finetune = str(mode).startswith("Transfer")
    stages = _STAGES[str(stage)]
    tr = [m.strip() for m in train_movies.split(",") if m.strip()]
    te = [m.strip() for m in test_movies.split(",") if m.strip()]
    out = str(output_dir) if str(output_dir) not in ("", ".") else str(_train.repo_root() / "logs")
    name = (run_name or Path(ds).name).strip().replace(" ", "_") or "dataset"

    base_ckpts, ft = {}, {}
    if finetune:
        for s in stages:  # each stage fine-tunes from its OWN base .ckpt
            picker = base_reg if s == "regression" else base_seg
            bp = Path(str(picker))
            if bp.suffix.lower() != ".ckpt" or not bp.is_file():
                notifications.show_warning(
                    f"Fine-tune {s}: choose a valid base checkpoint in the "
                    f"'{s.capitalize()} base (.ckpt)' field.")
                return
            base_ckpts[s] = str(bp)
        ft = dict(freeze_preset=freeze_preset, unfreeze_last_stages=int(unfreeze_last_stages),
                  bn_mode=bn_mode, ft_lr=ft_lr, discriminative=bool(discriminative),
                  backbone_lr_mult=backbone_lr_mult, weight_decay=weight_decay,
                  lr_schedule=lr_schedule, warmup_epochs=int(warmup_epochs), grad_clip=grad_clip,
                  augment=bool(augment), augment_strength=augment_strength, patience=int(patience),
                  seed=int(seed), cell_radius=int(cell_radius), seg_crop_size=int(seg_crop_size))

    _TRAIN_STATE["stop"] = False
    viewer = napari.current_viewer()

    @thread_worker
    def _run():
        from napari_dare3d._train import run_training, run_finetuning

        result: dict = {}
        if finetune:
            gen = run_finetuning(
                dataset_dir=ds, output_dir=out, name=name, date=date,
                epochs=int(epochs), batch_size=int(batch_size),
                stages=stages, base_ckpts=base_ckpts, ft=ft,
                train_movies=tr, val_movies=te,
                should_stop=lambda: _TRAIN_STATE["stop"],
            )
        else:
            gen = run_training(
                dataset_dir=ds, output_dir=out, name=name, date=date,
                epochs=int(epochs), batch_size=int(batch_size),
                cell_radius=int(cell_radius), seg_crop_size=int(seg_crop_size),
                train_movies=tr, val_movies=te,
                train_segmentation=("segmentation" in stages),
                train_regression=("regression" in stages),
                run_eval=run_eval, threshold=float(threshold),
                should_stop=lambda: _TRAIN_STATE["stop"],
            )
        for item in gen:
            if isinstance(item, dict):
                result = item
            else:
                yield item
        return result

    def _on_line(line):
        print(line)
        pbar.label = str(line)[:90]

    def _on_done(result):
        _set_running(False)
        pbar.max, pbar.value = 1, 1
        pbar.label = "DARE3D: done"
        made = ", ".join(f"{k} -> {v}" for k, v in (result or {}).items() if v)
        notifications.show_info(f"DARE3D {'fine-tuning' if finetune else 'training'} finished. {made}")
        _autofill_inference(viewer, result or {})

    def _on_error(exc):
        _set_running(False)
        pbar.max, pbar.value = 1, 1
        pbar.label = "DARE3D: failed"
        notifications.show_error(f"DARE3D {'fine-tuning' if finetune else 'training'} failed: {exc}")

    worker = _run()
    worker.yielded.connect(_on_line)
    worker.returned.connect(_on_done)
    worker.errored.connect(_on_error)
    pbar.max, pbar.value = 0, 0
    pbar.label = "DARE3D: fine-tuning…" if finetune else "DARE3D: training…"
    pbar.visible = True
    _set_running(True)  # hide "Run", show Stop in its place
    worker.start()
    notifications.show_info(f"DARE3D: {'fine-tuning' if finetune else 'training'} started…")
