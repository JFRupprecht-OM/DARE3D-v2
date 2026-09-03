"""napari widget for DARE3D inference (magicgui ``magic_factory``).

Thin GUI over :func:`napari_dare3d._api.infer_stack`. The heavy import (torch +
dare3d) is deferred into the worker so merely loading this module (plugin
discovery) stays cheap. Inference runs in a ``thread_worker`` to keep the UI
responsive; results are added back on the main thread.
"""
import sys
from pathlib import Path

import napari
import numpy as np
from magicgui import magic_factory
from magicgui.widgets import ProgressBar
from napari.qt.threading import thread_worker
from napari.utils import notifications

from napari_dare3d._io import iter_tifs
from napari_dare3d._scale import (
    napari_scale_from_xyz,
    resolve_source_scale_xyz,
    xyz_from_napari_scale,
)


#: Name of the Zenodo download holding the demo models / data.
DATA_ROOT_NAME = "DARE3d_data_190326"

#: Shared run state (single widget instance in practice): ``stop`` is the abort flag
#: (set by the Stop button, polled by ``infer_stack`` between frames); ``call_button``
#: / ``stop_button`` are kept so a running inference can swap Run -> Stop.
_INFER_STATE = {"stop": False, "call_button": None, "stop_button": None}

#: Fine-tuning controls hidden behind the "Advanced parameters" toggle (collapsed
#: by default). These are DARE3D's real knobs — note there is NO patch/crop-size
#: knob: the seg model is fixed at 128^3 and reg at 32^3 (U-Net stride-32
#: divisibility), so exposing a size would crash inference (the DARE2d lesson).
#: ``device`` lives here too (moved out of the main panel to match the DARE2D UX).
_ADVANCED_FIELDS = ("device", "overlap", "batch_size", "threshold", "min_weighted_prob",
                    "scale_file", "default_scale")

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


def _default_scale_file() -> Path:
    """Return the backward-compatible scale table when available."""
    candidates = (
        Path(__file__).resolve().parents[1] / "data" / "3D" / "scales.json",
        Path(sys.prefix) / "data" / "3D" / "scales.json",
    )
    return next((path for path in candidates if path.is_file()), Path())


def _maybe_file(path: Path):
    """Return str(path) for a real file, else None (treat empty as unset)."""
    path = Path(path)
    return str(path) if (str(path) not in ("", ".") and path.is_file()) else None


def _result_layer_scale(image, stack_ndim: int):
    """Map an input Image scale to four-dimensional (T,Z,Y,X) results."""
    scale = tuple(float(value) for value in image.scale)
    if stack_ndim == 3 and len(scale) == 3:
        return (1.0, *scale)
    if stack_ndim == 4 and len(scale) == 4:
        return scale
    return None


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


def _set_running(running: bool) -> None:
    """While a run is active, hide the "Run DARE3D" call button and show Stop in its
    place; restore on completion/abort."""
    cb = _INFER_STATE.get("call_button")
    sb = _INFER_STATE.get("stop_button")
    if cb is not None:
        cb.visible = not running
    if sb is not None:
        sb.visible = running


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

    # Run/Stop: the Stop button is hidden until a run starts, then it replaces the
    # "Run DARE3D" call button (restored when the run finishes/aborts). Clicking it
    # sets the shared abort flag, polled by infer_stack between frames.
    _INFER_STATE["call_button"] = getattr(widget, "call_button", None)
    _INFER_STATE["stop_button"] = getattr(widget, "stop", None)
    if getattr(widget, "stop", None) is not None:
        widget.stop.visible = False  # shown only while a run is active
        widget.stop.tooltip = "Abort the running inference at the next frame boundary."

        def _request_stop(*_):
            _INFER_STATE["stop"] = True
            if getattr(widget, "pbar", None) is not None:
                widget.pbar.label = "DARE3D: stopping…"

        widget.stop.changed.connect(_request_stop)

    viewer = napari.current_viewer()
    if viewer is None:
        return

    # "…or load a movie (.tif)": picking a file loads it into the viewer and selects
    # it as the Run input, so the movie shows immediately (mirrors the DARE2D widget).
    # Extension is matched case-insensitively on the suffix (.tif/.tiff/.TIF/.TIFF),
    # the same test _io.iter_tifs uses; an already-loaded layer of the same name is
    # reused (no duplicates).
    def _show_movie(path):
        if not path:
            return
        p = Path(path)
        if not p.is_file() or p.suffix.lower() not in (".tif", ".tiff"):
            return
        name = p.stem
        if name not in viewer.layers:
            try:
                import tifffile

                viewer.add_image(tifffile.imread(str(p)), name=name)
            except Exception as exc:  # never block on a load hiccup
                notifications.show_warning(f"DARE3D: could not load {p.name}: {exc}")
                return
        try:
            widget.image.value = viewer.layers[name]  # becomes the Run input
        except Exception:
            pass

    if getattr(widget, "movie", None) is not None:
        widget.movie.changed.connect(lambda *_: _show_movie(widget.movie.value))

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
    image={"label": "Image layer (already open)",
           "tooltip": "Run on an Image layer already open in napari. 4D (T,Z,Y,X) or 3D "
                      "(Z,Y,X). Leave empty if you load a movie file below instead."},
    movie={"widget_type": "FileEdit", "mode": "r", "label": "…or load a movie (.tif)",
           "tooltip": "Browse for a (T,Z,Y,X) or (Z,Y,X) .tif/.tiff stack; it loads and "
                      "displays immediately and becomes the Run input. Leave blank to use "
                      "the open Image layer above."},
    seg_model_dir={
        "widget_type": "FileEdit", "mode": "d", "label": "Segmentation model dir",
        "tooltip": "Segmentation model folder: must contain .hydra/config.yaml + "
                   "checkpoints/last.ckpt. Pre-filled from DARE3d_data_190326 if found.",
    },
    reg_model_dir={
        "widget_type": "FileEdit", "mode": "d", "label": "Regression model dir (optional)",
        "tooltip": "Optional regression model folder. If set, each detection also gets a 3D "
                   "division axis. Runs on gpu or cpu (cpu is correct but slower).",
    },
    t_start={
        "min": 0, "label": "Start frame",
        "tooltip": "First frame to analyze (0-based). Default: 0 (with End frame -1 = whole "
                   "movie).",
    },
    t_end={
        "min": -1, "label": "End frame (-1 = end)",
        "tooltip": "Last frame to analyze, inclusive; -1 means the final frame. "
                   "Default: -1.",
    },
    advanced={
        "widget_type": "PushButton", "text": "Show advanced parameters",
        "tooltip": "Show/hide the fine-tuning parameters below (collapsed by default).",
    },
    device={
        "choices": ["gpu", "cpu"], "label": "Device",
        "tooltip": "gpu = CUDA (recommended, much faster); cpu runs the full pipeline "
                   "(segmentation centers + regression axes) too, just slower. Default: gpu.",
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
    scale_file={
        "widget_type": "FileEdit", "mode": "r", "label": "Per-movie scales JSON",
        "tooltip": "Scale table keyed by the selected layer/movie name. Defaults to "
                   "data/3D/scales.json when running from a source checkout; clear it "
                   "to use default_scale or the model config.",
    },
    legend={"widget_type": "Label", "label": "", "visible": False},  # shown after a run (see _init_widget)
    pbar={"label": "progress", "visible": False, "min": 0, "max": 0},
    # Rendered just above the call button so it visually replaces "Run DARE3D" while
    # a run is active (hidden until then; see _init_widget / _set_running).
    stop={
        "widget_type": "PushButton", "text": "Stop",
        "tooltip": "Abort the running inference at the next frame boundary.",
    },
)
def dare3d_widget(
    image: "napari.layers.Image",
    movie: Path = Path(""),
    seg_model_dir: Path = _default_model_dir("segmentation3d_exp10-b"),
    reg_model_dir: Path = _default_model_dir("regression3d_exp10-b"),
    t_start: int = 0,
    t_end: int = -1,
    advanced: bool = False,
    device: str = "gpu",
    overlap: float = 0.25,
    batch_size: int = 4,
    threshold: float = 0.5,
    min_weighted_prob: float = 0.1,
    scale_file: Path = _default_scale_file(),
    default_scale: str = "",
    legend: str = LEGEND_HTML,
    pbar: ProgressBar = None,
    stop: bool = False,
):
    """Run DARE3D segmentation (+ optional regression) on an Image layer and
    overlay detected division centers and axes (napari Points layers)."""
    viewer = napari.current_viewer()
    # Input precedence: the selected Image layer; else the movie file — reusing an
    # already-loaded layer of the same name so picking a movie never double-adds it
    # (the movie picker also loads it on selection; see _init_widget).
    if image is None and str(movie) not in ("", "."):
        p = Path(movie)
        if p.is_file() and p.suffix.lower() in (".tif", ".tiff"):
            name = p.stem
            if viewer is not None and name in viewer.layers:
                image = viewer.layers[name]
            elif viewer is not None:
                import tifffile

                image = viewer.add_image(tifffile.imread(str(p)), name=name)
    if image is None:
        notifications.show_warning("DARE3D: select an Image layer or load a movie (.tif) first.")
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
    sf = _maybe_file(scale_file)
    if str(scale_file) not in ("", ".") and sf is None:
        notifications.show_error(f"Scale file does not exist: {scale_file}")
        return

    stack = np.asarray(image.data)
    try:
        layer_scale_xyz = xyz_from_napari_scale(image.scale, stack.ndim)
        layer_scale_is_calibrated = not np.allclose(
            layer_scale_xyz, (1.0, 1.0, 1.0)
        )
        inference_default_scale = (
            ds
            if ds is not None
            else (layer_scale_xyz if layer_scale_is_calibrated else None)
        )
        source_scale_xyz = resolve_source_scale_xyz(
            image.name, sf, inference_default_scale
        )
        if source_scale_xyz is not None:
            image.scale = napari_scale_from_xyz(source_scale_xyz, stack.ndim)
    except (OSError, TypeError, ValueError) as exc:
        notifications.show_error(f"Invalid spatial calibration: {exc}")
        return
    result_layer_scale = _result_layer_scale(image, stack.ndim)
    n_t = stack.shape[0] if stack.ndim == 4 else 1
    # DARE2D end-frame convention: t_end == -1 means "to the final frame". The whole
    # movie (start 0, end -1) is passed as frames=None; any other window is resolved
    # to inclusive (t0, t1) indices for _api.infer_stack.
    if t_start == 0 and t_end == -1:
        frames = None
    else:
        t_end_res = (n_t - 1) if t_end == -1 else t_end
        if not (0 <= t_start <= t_end_res < n_t):
            notifications.show_warning(
                f"DARE3D: invalid time range [{t_start}, {t_end}] for a movie of {n_t} frame(s)."
            )
            return
        frames = (t_start, t_end_res)

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
            scale_file=sf,
            default_scale=inference_default_scale,
            movie_name=image.name,
            frames=frames,
            progress_cb=lambda stage: print(f"[DARE3D] {stage}"),
            should_stop=lambda: _INFER_STATE["stop"],
        )

    def _on_done(detections):
        from collections import Counter

        from napari_dare3d._api import to_layer_data

        _set_running(False)  # restore the "Run DARE3D" button
        # Aborted via Stop: infer_stack returns [] and we add no layers.
        if _INFER_STATE["stop"]:
            pbar.max, pbar.value = 1, 1
            pbar.label = "DARE3D: stopped"
            notifications.show_info("DARE3D: inference stopped.")
            return

        for data, kwargs, ltype in to_layer_data(
            detections, layer_scale=result_layer_scale
        ):
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
        _set_running(False)  # restore the "Run DARE3D" button
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
    _set_running(True)  # swap "Run DARE3D" -> "Stop" for the duration of the run
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
