"""napari widget for DARE3D (re)training (magicgui ``magic_factory``).

Thin GUI over :func:`napari_dare3d._train.run_training`, which runs DARE3D's own
``train.py``/``eval.py`` as subprocesses (seg -> reg -> eval). Because we own the
subprocess read loop, the worker streams Lightning's stdout lines live to the
widget. Heavy work is in the subprocess, so this stays UI-responsive; Stop
terminates the running process.
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

#: Shared stop flag (single widget instance in practice); set by the Stop button.
_TRAIN_STATE = {"stop": False}


def _default_trainingset() -> Path:
    """Default dataset: the Gastruloid per-movie ``trainingset/`` (if present)."""
    root = _data_root()
    if root is not None:
        p = root / "Gastruloid_241025" / "trainingset"
        if p.is_dir():
            return p
    return Path()


def _init_training_widget(widget) -> None:
    """magic_factory hook: tooltip the Run button and wire the Stop button."""
    if getattr(widget, "call_button", None) is not None:
        widget.call_button.tooltip = "Start DARE3D training with the settings above."
    try:
        widget.stop.tooltip = "Terminate the running training process."
        widget.stop.changed.connect(lambda *_: _TRAIN_STATE.__setitem__("stop", True))
    except Exception:
        pass


def _autofill_inference(viewer, result: dict) -> None:
    """Best-effort: push the trained model dirs into an open inference widget."""
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
    dataset_dir={
        "widget_type": "FileEdit", "mode": "d", "label": "Dataset dir",
        "tooltip": "Per-movie dataset <dataset>/<movie>/{im,label}/*.tif (each tif a "
                   "whole T,Z,Y,X movie; labels: odd=first daughter, even=second), or a "
                   "pre-split <dataset>/{train,val}/{im,label}. Default: Gastruloid trainingset.",
    },
    train_movies={
        "label": "Training movies",
        "tooltip": "Comma-separated movie subfolders to TRAIN on (per-movie datasets). "
                   "Folders not listed here or in Test movies are excluded.",
    },
    test_movies={
        "label": "Test movies",
        "tooltip": "Comma-separated movie subfolders for validation/testing.",
    },
    output_dir={
        "widget_type": "FileEdit", "mode": "d", "label": "Output / logs dir",
        "tooltip": "Where model_dirs are written (logs/{task}/runs/{date}/). "
                   "Blank = the dare3d repo's logs/ folder.",
    },
    run_name={"tooltip": "Experiment name (-> task_name segmentation3d_<name>). Blank = dataset folder name."},
    date={"tooltip": "Run date/id, used in the output path runs/<date>/."},
    epochs={"min": 1, "tooltip": "Max training epochs per model. Default: 50."},
    batch_size={"min": 1, "tooltip": "3D patches per step. Lower if you hit GPU out-of-memory. Default: 4."},
    cell_radius={"min": 1, "tooltip": "Radius (voxels) of the segmentation target spheres. Default: 8."},
    seg_crop_size={"min": 16, "tooltip": "3D training patch size for segmentation. Default: 128."},
    train_segmentation={"tooltip": "Train the segmentation (division-center) model. Default: on."},
    train_regression={"tooltip": "Train the regression (division-axis) model. Default: on."},
    run_eval={"tooltip": "After training, run dare3d eval.py on the val set (needs both models + GT). Default: on."},
    threshold={"min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Segmentation probability threshold used at evaluation. Default: 0.5."},
    stop={"widget_type": "PushButton", "text": "Stop"},
    pbar={"label": "progress", "visible": False, "min": 0, "max": 0},
)
def dare3d_training_widget(
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
    train_segmentation: bool = True,
    train_regression: bool = True,
    run_eval: bool = True,
    threshold: float = 0.5,
    stop: bool = False,
    pbar: ProgressBar = None,
):
    """Train DARE3D segmentation/regression models on a labelled dataset (GPU,
    subprocess), then optionally evaluate. Produces model_dirs usable by the
    inference widget."""
    ds = _maybe_dir(dataset_dir)
    if ds is None:
        notifications.show_warning("DARE3D: choose a valid dataset folder.")
        return
    if not (train_segmentation or train_regression):
        notifications.show_warning("DARE3D: enable segmentation and/or regression training.")
        return
    tr = [m.strip() for m in train_movies.split(",") if m.strip()]
    te = [m.strip() for m in test_movies.split(",") if m.strip()]
    out = str(output_dir) if str(output_dir) not in ("", ".") else str(_train.repo_root() / "logs")
    name = (run_name or Path(ds).name).strip().replace(" ", "_") or "dataset"

    _TRAIN_STATE["stop"] = False
    viewer = napari.current_viewer()

    @thread_worker
    def _run():
        from napari_dare3d._train import run_training

        result: dict = {}
        for item in run_training(
            dataset_dir=ds, output_dir=out, name=name, date=date,
            epochs=int(epochs), batch_size=int(batch_size),
            cell_radius=int(cell_radius), seg_crop_size=int(seg_crop_size),
            train_movies=tr, val_movies=te,
            train_segmentation=train_segmentation, train_regression=train_regression,
            run_eval=run_eval, threshold=float(threshold),
            should_stop=lambda: _TRAIN_STATE["stop"],
        ):
            if isinstance(item, dict):
                result = item
            else:
                yield item
        return result

    def _on_line(line):
        print(line)
        pbar.label = str(line)[:90]

    def _on_done(result):
        pbar.max, pbar.value = 1, 1
        pbar.label = "DARE3D: training done"
        made = ", ".join(f"{k} -> {v}" for k, v in (result or {}).items() if v)
        notifications.show_info(f"DARE3D training finished. {made}")
        _autofill_inference(viewer, result or {})

    def _on_error(exc):
        pbar.max, pbar.value = 1, 1
        pbar.label = "DARE3D: training failed"
        notifications.show_error(f"DARE3D training failed: {exc}")

    worker = _run()
    worker.yielded.connect(_on_line)
    worker.returned.connect(_on_done)
    worker.errored.connect(_on_error)
    pbar.max, pbar.value = 0, 0
    pbar.label = "DARE3D: training…"
    pbar.visible = True
    worker.start()
    notifications.show_info("DARE3D: training started…")
