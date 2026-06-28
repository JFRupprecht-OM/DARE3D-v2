"""napari widget for DARE3D inference (magicgui ``magic_factory``).

Thin GUI over :func:`napari_dare3d._api.infer_stack`. The heavy import (torch +
dare3d) is deferred into the worker so merely loading this module (plugin
discovery) stays cheap. Inference runs in a ``thread_worker`` to keep the UI
responsive; results are added back on the main thread.
"""
from pathlib import Path

import napari
import numpy as np
from magicgui import magic_factory
from magicgui.widgets import ProgressBar
from napari.qt.threading import thread_worker
from napari.utils import notifications

from napari_dare3d._io import iter_tifs


#: Name of the Zenodo download holding the demo models / data.
DATA_ROOT_NAME = "DARE3d_data_190326"

#: Shared stop flag (single widget instance in practice); set by the Stop button
#: and polled by ``infer_stack`` between frames to abort a long run.
_INFER_STATE = {"stop": False}

#: Fine-tuning controls hidden behind the "Advanced parameters" toggle (collapsed
#: by default). These are DARE3D's real knobs — note there is NO patch/crop-size
#: knob: the seg model is fixed at 128^3 and reg at 32^3 (U-Net stride-32
#: divisibility), so exposing a size would crash inference (the DARE2d lesson).
_ADVANCED_FIELDS = ("overlap", "batch_size", "threshold", "min_weighted_prob", "default_scale")

#: Short overlay legend shown in the widget (QLabel rich text).
LEGEND_HTML = (
    "<b>Overlay</b><br>"
    "<span style='color:#ff3b30'>&#9679;</span>&nbsp;red dot &mdash; division center<br>"
    "<span style='color:#00e5ff'>&#9679;</span>&nbsp;cyan points &mdash; division axis "
    "(3D orientation &amp; length)"
)


def _data_root():
    """Locate the ``DARE3d_data_190326`` dir that holds ``Gastruloid_241025``.

    Looks first under the current working directory (honours "put the data in the
    current folder"), then relative to this plugin's source — which, as an
    editable install under ``DARE3Dnapariplugin/``, sits next to the data
    regardless of where napari was launched. Tolerant of the common unzip
    double-nesting (``DARE3d_data_190326/DARE3d_data_190326/...``). Returns the
    resolved root, or None.
    """
    bases = [
        Path.cwd() / DATA_ROOT_NAME,
        Path(__file__).resolve().parents[2] / DATA_ROOT_NAME,  # .../DARE3Dnapariplugin/
    ]
    for base in bases:
        for root in (base, base / DATA_ROOT_NAME):
            if (root / "Gastruloid_241025").is_dir():
                return root
    return None


def _default_model_dir(kind: str) -> Path:
    """Default Gastruloid model dir (``segmentation3d_exp10-b`` /
    ``regression3d_exp10-b``), or ``Path()`` if the data isn't found."""
    root = _data_root()
    if root is not None:
        cand = root / "Gastruloid_241025" / "weights" / kind
        if cand.is_dir():
            return cand
    return Path()


def _default_image_path():
    """Path of the Gastruloid ``test_input`` movie to preload, or None."""
    root = _data_root()
    if root is None:
        return None
    test_input = root / "Gastruloid_241025" / "test_input"
    tifs = iter_tifs(test_input)  # case-insensitive .tif/.tiff
    return Path(tifs[0]) if tifs else None


def _set_advanced_visible(widget, visible: bool) -> None:
    """Show/hide the advanced fine-tuning controls and update the toggle label."""
    for field in _ADVANCED_FIELDS:
        ctrl = getattr(widget, field, None)
        if ctrl is not None:
            ctrl.visible = visible
    if getattr(widget, "advanced", None) is not None:
        widget.advanced.text = (
            "Hide advanced parameters" if visible else "Show advanced parameters"
        )


def _init_widget(widget):
    """magic_factory hook: preload the Gastruloid test_input movie into the viewer
    and select it as the input image. Runs once when the widget is created."""
    # The call button is not a function parameter, so set its tooltip here.
    if getattr(widget, "call_button", None) is not None:
        widget.call_button.tooltip = (
            "Run DARE3D inference on the selected image with the settings above."
        )

    # Collapsible "Advanced parameters": hidden by default; the PushButton toggles
    # them. (magicgui has no native collapsible — flip each control's .visible.)
    _set_advanced_visible(widget, False)
    if getattr(widget, "advanced", None) is not None:
        widget.advanced.changed.connect(
            lambda *_: _set_advanced_visible(widget, not widget.overlap.visible)
        )

    # Stop button: set the shared abort flag, polled by infer_stack between frames.
    if getattr(widget, "stop", None) is not None:
        widget.stop.tooltip = "Abort the running inference at the next frame boundary."

        def _request_stop(*_):
            _INFER_STATE["stop"] = True
            if getattr(widget, "pbar", None) is not None:
                widget.pbar.label = "DARE3D: stopping…"

        widget.stop.changed.connect(_request_stop)

    viewer = napari.current_viewer()
    if viewer is None:
        return

    # Show the overlay legend only while DARE3D result layers exist (i.e. after a
    # run has added them); hidden before/during a run and if results are cleared.
    widget.legend.visible = False

    def _update_legend(*_):
        widget.legend.visible = any(
            str(layer.name).startswith("DARE3D") for layer in viewer.layers
        )

    viewer.layers.events.inserted.connect(_update_legend)
    viewer.layers.events.removed.connect(_update_legend)

    path = _default_image_path()
    if path is None:
        return
    name = path.stem
    if name not in viewer.layers:
        try:
            import tifffile

            viewer.add_image(tifffile.imread(str(path)), name=name)
        except Exception as exc:  # never block the widget on a preload hiccup
            notifications.show_warning(f"DARE3D: could not preload {path.name}: {exc}")
            return
    if name in viewer.layers:
        # The image combo's choices are only populated once the widget is docked
        # to the viewer; until then setting .value raises. magicgui then auto-
        # selects this (first) Image layer, so a failure here is harmless.
        try:
            widget.image.value = viewer.layers[name]
        except ValueError:
            pass


def _maybe_dir(p: Path):
    """Return ``str(p)`` if it is a real directory, else None (treat empty as unset)."""
    p = Path(p)
    return str(p) if (str(p) not in ("", ".") and p.is_dir()) else None


def _parse_scale(text: str):
    """Parse 'x,y,z' (or empty) into a list of 3 floats (or None)."""
    text = (text or "").strip()
    if not text:
        return None
    parts = [float(v) for v in text.replace(";", ",").split(",") if v.strip()]
    if len(parts) != 3:
        raise ValueError("default_scale must be three numbers 'x,y,z' (or left blank).")
    return parts


@magic_factory(
    call_button="Run DARE3D",
    widget_init=_init_widget,
    tooltips=False,  # use the explicit per-control "tooltip" options below, not the docstring
    image={"tooltip": "Input stack to analyze (a napari Image layer). 4D (T,Z,Y,X) or 3D (Z,Y,X)."},
    seg_model_dir={
        "widget_type": "FileEdit", "mode": "d", "label": "Segmentation model dir",
        "tooltip": "Segmentation model folder: must contain .hydra/config.yaml + "
                   "checkpoints/last.ckpt. Pre-filled from DARE3d_data_190326 if found.",
    },
    reg_model_dir={
        "widget_type": "FileEdit", "mode": "d", "label": "Regression model dir (optional)",
        "tooltip": "Optional regression model folder. If set, each detection also gets a 3D "
                   "division axis. Requires a CUDA (gpu) device.",
    },
    device={
        "choices": ["gpu", "cpu"], "label": "Device",
        "tooltip": "gpu = CUDA (required for the regression axes); cpu = segmentation centers "
                   "only. Default: gpu.",
    },
    whole_movie={
        "label": "Analyse whole movie",
        "tooltip": "Analyze every time frame. Uncheck to restrict to [Start frame, End frame]. "
                   "Default: on.",
    },
    t_start={
        "min": 0, "label": "Start frame",
        "tooltip": "First frame to analyze (used only when 'Analyse whole movie' is off). "
                   "Default: 0.",
    },
    t_end={
        "min": 0, "label": "End frame (inclusive)",
        "tooltip": "Last frame to analyze, inclusive (used only when 'Analyse whole movie' is "
                   "off). Default: 0.",
    },
    advanced={
        "widget_type": "PushButton", "text": "Show advanced parameters",
        "tooltip": "Show/hide the fine-tuning parameters below (collapsed by default).",
    },
    overlap={
        "min": 0.0, "max": 0.9, "step": 0.05,
        "tooltip": "Sliding-window overlap for 3D segmentation (0-0.9; default 0.25). "
                   "Higher = more accurate, slower.",
    },
    batch_size={
        "tooltip": "Number of 3D patches inferred at once. Lower this if you hit GPU "
                   "out-of-memory. Default: 4.",
    },
    threshold={
        "min": 0.0, "max": 1.0, "step": 0.05,
        "tooltip": "Probability threshold (0-1; default 0.5) to turn the segmentation heatmap "
                   "into division centers.",
    },
    min_weighted_prob={
        "min": 0.0, "max": 1.0, "step": 0.05, "label": "min weighted prob",
        "tooltip": "Minimum size x probability to keep a detected center; filters weak/small "
                   "blobs (default 0.1).",
    },
    default_scale={
        "label": "default_scale x,y,z (blank = from config)",
        "tooltip": "Voxel size x,y,z in microns (e.g. 0.621,0.621,2). "
                   "Blank = use the value saved in the model config.",
    },
    stop={
        "widget_type": "PushButton", "text": "Stop",
        "tooltip": "Abort the running inference at the next frame boundary.",
    },
    legend={"widget_type": "Label", "label": "", "visible": False},  # shown after a run (see _init_widget)
    pbar={"label": "progress", "visible": False, "min": 0, "max": 0},
)
def dare3d_widget(
    image: "napari.layers.Image",
    seg_model_dir: Path = _default_model_dir("segmentation3d_exp10-b"),
    reg_model_dir: Path = _default_model_dir("regression3d_exp10-b"),
    device: str = "gpu",
    whole_movie: bool = True,
    t_start: int = 0,
    t_end: int = 0,
    advanced: bool = False,
    overlap: float = 0.25,
    batch_size: int = 4,
    threshold: float = 0.5,
    min_weighted_prob: float = 0.1,
    default_scale: str = "",
    stop: bool = False,
    legend: str = LEGEND_HTML,
    pbar: ProgressBar = None,
):
    """Run DARE3D segmentation (+ optional regression) on an Image layer and
    overlay detected division centers and axes (napari Points layers)."""
    if image is None:
        notifications.show_warning("DARE3D: select an Image layer first.")
        return
    seg = _maybe_dir(seg_model_dir)
    if seg is None:
        notifications.show_warning("DARE3D: choose a valid segmentation model directory.")
        return
    reg = _maybe_dir(reg_model_dir)
    if reg is None and device == "cpu":
        # seg-only is fine on CPU; this branch only guards the reg+CPU combo below.
        pass
    try:
        ds = _parse_scale(default_scale)
    except ValueError as exc:
        notifications.show_error(str(exc))
        return

    stack = np.asarray(image.data)
    n_t = stack.shape[0] if stack.ndim == 4 else 1
    if whole_movie:
        frames = None
    else:
        if not (0 <= t_start <= t_end < n_t):
            notifications.show_warning(
                f"DARE3D: invalid time range [{t_start}, {t_end}] for a movie of {n_t} frame(s)."
            )
            return
        frames = (t_start, t_end)
    viewer = napari.current_viewer()

    # Arm the Stop button for this run (cleared in case a previous run set it).
    _INFER_STATE["stop"] = False

    @thread_worker
    def _run():
        from napari_dare3d._api import infer_stack  # defer torch/dare3d import

        return infer_stack(
            stack, seg, reg,
            device=device,
            overlap=overlap,
            batch_size=int(batch_size),
            threshold=threshold,
            min_weighted_prob=min_weighted_prob,
            default_scale=ds,
            frames=frames,
            progress_cb=lambda stage: print(f"[DARE3D] {stage}"),
            should_stop=lambda: _INFER_STATE["stop"],
        )

    def _on_done(detections):
        from collections import Counter

        from napari_dare3d._api import to_layer_data

        # Aborted via Stop: infer_stack returns [] and we add no layers.
        if _INFER_STATE["stop"]:
            pbar.max, pbar.value = 1, 1
            pbar.label = "DARE3D: stopped"
            notifications.show_info("DARE3D: inference stopped.")
            return

        for data, kwargs, ltype in to_layer_data(detections):
            getattr(viewer, f"add_{ltype}")(data, **kwargs)
        notifications.show_info(f"DARE3D: {len(detections)} division(s) detected.")
        # Leave the bar in place, filled, as a "done" indicator.
        pbar.max, pbar.value = 1, 1
        pbar.label = f"DARE3D: done — {len(detections)} division(s)"
        if detections:
            # Stay in the 2D slice view (explicit z-stack); just jump the time
            # slider to the busiest frame so detections show up immediately.
            try:
                busiest_t = Counter(
                    int(round(d["center_napari"][0])) for d in detections
                ).most_common(1)[0][0]
                viewer.dims.set_current_step(0, busiest_t)
            except Exception:
                pass

    def _on_error(exc):
        notifications.show_error(f"DARE3D failed: {exc}")
        pbar.max, pbar.value = 1, 1
        pbar.label = "DARE3D: failed"

    worker = _run()
    worker.returned.connect(_on_done)
    worker.errored.connect(_on_error)
    # In-widget progress bar: busy/indeterminate (max=0) while running, then left
    # visible in a filled "done"/"failed" state (set in _on_done/_on_error). Driven
    # only from main-thread signals, so it is GUI-thread safe.
    pbar.max, pbar.value = 0, 0
    pbar.label = "DARE3D: running…"
    pbar.visible = True
    worker.start()
    notifications.show_info("DARE3D: inference started…")


@magic_factory(
    call_button="Download DARE3D data (Zenodo)",
    dest={
        "widget_type": "FileEdit", "mode": "d", "label": "Download into",
        "tooltip": "Folder to download DARE3d_data_190326 into (Zenodo record 19113351, "
                   "~7 GB). Launch napari from here so the model-dir fields auto-fill.",
    },
    pbar={"label": "progress", "visible": False, "min": 0, "max": 0},
)
def dare3d_download_widget(dest: Path = Path.cwd(), pbar: ProgressBar = None):
    """Download the DARE3D demo data + pretrained models from Zenodo (~7 GB) and unzip.

    Saves ``DARE3d_data_190326`` into the chosen folder; the inference widget then
    auto-fills its model-dir fields when napari is launched from there. The download
    runs in a worker thread (napari stays responsive); percent progress prints to the
    terminal.
    """
    from napari_dare3d import _data

    state = {"pct": -5}

    def _progress(done, total):
        if total:
            pct = int(100 * done / total)
            if pct >= state["pct"] + 5:
                state["pct"] = pct
                print(f"[DARE3D] download {pct}%  ({done / 1e9:.2f}/{total / 1e9:.2f} GB)")

    @thread_worker
    def _run():
        return _data.download(dest, progress_cb=_progress,
                              log=lambda m: print(f"[DARE3D] {m}"))

    def _on_done(path):
        pbar.max, pbar.value = 1, 1
        pbar.label = f"downloaded → {Path(path).name}"
        notifications.show_info(f"DARE3D data ready: {path}")

    def _on_error(exc):
        pbar.max, pbar.value = 1, 1
        pbar.label = "download failed"
        notifications.show_error(f"DARE3D download failed: {exc}")

    worker = _run()
    worker.returned.connect(_on_done)
    worker.errored.connect(_on_error)
    pbar.max, pbar.value = 0, 0
    pbar.label = "downloading… (see terminal for %)"
    pbar.visible = True
    worker.start()
    notifications.show_info("DARE3D: download started — progress in the terminal.")
