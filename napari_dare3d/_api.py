"""
In-process inference API for the DARE3D napari plugin — pure numpy/torch, imports
no napari (so it is headless-testable). It REUSES DARE3D's own inference code
(``dare3d.metrics.inference`` + ``dare3d.metrics.object_level``) and only
re-implements the thin orchestration that ``dare3d.predict`` performs under a
Hydra/CLI context (disk globbing + ``HydraConfig.get()``), which is unavailable
inside a napari session.

Coordinate conventions
----------------------
- napari feeds a stack in ``(T, Z, Y, X)`` order (plane, row, col); a bare 3D
  ``(Z, Y, X)`` stack is treated as a single time frame.
- DARE3D stores movies on disk as ``(T, Z, Y, X)`` and internally swaps to
  ``(T, X, Y, Z)`` (``Cell3Dataset.read_tif_and_order_xyz`` does ``swapaxes(1, -1)``).
  Detected centers come back as ``(m, t, x, y, z)`` in that internal ``(X, Y, Z)``
  order, in *original* (unscaled) voxel coordinates. We map a spatial triple
  ``(x, y, z)`` to napari ``(z, y, x)`` simply by reversing it — and likewise the
  division-axis direction.
- Orientation is a unit quaternion (wxyz). The division-axis direction is the
  normalised imaginary part of that quaternion (= ``angles3d.compute_axis_angle``),
  and the drawn segment spans ``length`` voxels centred on the detection
  (mirrors ``regression_display.get_points_from_quat``).

``dare3d`` must be importable (it is ``pip install -e .`` in this env).
"""
import os
import tempfile
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import hydra
import numpy as np
import tifffile
import torch
from omegaconf import OmegaConf

from dare3d.metrics.inference import (
    InferenceAborted,
    regression_inference,
    segmentation_inference,
)
from dare3d.metrics.object_level import (
    connected_components,
    filter_by_object_weighted_prob,
    get_sphere_vol,
    statistics_optimized,
)

# ``dare3d.predict`` registers this at import time; some net/config nodes use
# ``${eval:...}``. ``replace=True`` keeps re-imports safe in a long napari session.
OmegaConf.register_new_resolver("eval", eval, replace=True)

ProgressCb = Optional[Callable[[str], None]]


# --------------------------------------------------------------------------- #
# Hydra config / model / dataset wiring (replicates dare3d.predict, sans CLI)  #
# --------------------------------------------------------------------------- #
def _torch_device(device: str) -> torch.device:
    return torch.device("cuda:0") if device in ("gpu", "cuda") else torch.device("cpu")


def _resolve_model_dir(model_dir: str, hydra_dir: str = ".hydra") -> str:
    """Return the directory that actually holds ``<hydra_dir>/config.yaml``.

    Accepts either a *flat* model dir (``<model>/`` with ``.hydra/`` + ``checkpoints/``
    directly inside — e.g. the Gastruloid bundle) or one nested under ``runs/<date>/``
    (e.g. the Neural_tube bundle, which keeps the native ``train.py`` output layout).
    If the config is not directly present, descend into the most recent ``runs/<date>/``
    (or single sub-run) that contains it. Falls back to ``model_dir`` unchanged so the
    caller still raises a clear error.
    """
    import glob

    if os.path.exists(os.path.join(model_dir, hydra_dir, "config.yaml")):
        return model_dir
    hits = glob.glob(os.path.join(model_dir, "runs", "*", hydra_dir, "config.yaml"))
    hits += glob.glob(os.path.join(model_dir, "*", hydra_dir, "config.yaml"))
    runs = [os.path.dirname(os.path.dirname(p)) for p in hits if os.path.isfile(p)]
    if runs:
        return max(runs, key=os.path.getmtime)  # most recent training run
    return model_dir


def _history_frames(seg_model_dir: str, hydra_dir: str = ".hydra") -> int:
    """Leading temporal context the segmentation model needs to predict a frame
    (= ``len(input_channels) - 1``; e.g. 2 for input_channels ``[-1, 0, 1]``).

    Frames analysed by DARE3D start at this index, so a requested time window must
    include this many frames of history before its first frame.
    """
    seg_model_dir = _resolve_model_dir(seg_model_dir, hydra_dir)
    cfg = OmegaConf.load(os.path.join(seg_model_dir, hydra_dir, "config.yaml"))
    try:
        return max(0, len(cfg.input_channels) - 1)
    except Exception:
        return 0


def _load_inference_cfg(
    model_dir: str,
    im_folder: str,
    device: str,
    *,
    hydra_dir: str = ".hydra",
    ckpt_dir: str = "checkpoints",
    ckpt_name: str = "last.ckpt",
    scale_file: Optional[str] = None,
    default_scale=None,
    target_scale=None,
):
    """Load the training config saved next to the checkpoint and rewire it for
    in-memory inference.

    Mirrors ``dare3d.predict.load_config`` but pins the scale settings to concrete
    values: in the saved config ``test_data.scale_file`` is ``${scale_file}`` ->
    ``${paths.data_dir}/...``, an interpolation that only resolves under a live
    Hydra run. Everything else (``crop_size``, ``input_channels``, ``renorm``,
    ``cell_radius``, ``representation_mode``, ...) resolves within the saved tree.
    """
    model_dir = _resolve_model_dir(model_dir, hydra_dir)
    hydra_config_path = os.path.join(model_dir, hydra_dir, "config.yaml")
    ckpt_path = os.path.join(model_dir, ckpt_dir, ckpt_name)
    if not os.path.exists(hydra_config_path):
        raise FileNotFoundError(f"Missing training config: {hydra_config_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    cfg = OmegaConf.load(hydra_config_path)
    OmegaConf.set_struct(cfg, False)

    cfg.ckpt_path = ckpt_path
    cfg.device = device

    td = cfg.data.test_data
    td.load_labels = False
    td.im_folder = im_folder
    # label_folder is `${paths.data_dir}/.../label` in the saved config; it is
    # unused here (load_labels=False) but ``instantiate`` still resolves the whole
    # node, so pin it off the ${paths.*}/${oc.env:PROJECT_ROOT} interpolation.
    td.label_folder = im_folder
    # Same rationale for scale_file: a provided file, else a non-existent path so
    # AbstractCellDataset.load_movie_scales falls back to default_scale.
    td.scale_file = scale_file if scale_file else os.path.join(im_folder, "__no_scales__.json")
    if default_scale is not None:
        td.default_scale = list(default_scale) if isinstance(default_scale, (list, tuple)) else default_scale
    if target_scale is not None:
        td.target_scale = target_scale
    return cfg


def _build_model(cfg, torch_device: torch.device):
    """Instantiate the LightningModule and load checkpoint weights (as predict.py)."""
    model = hydra.utils.instantiate(cfg.model)
    state_dict = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state_dict)
    model.net = model.net.to(torch_device)
    model.net.eval()
    return model


def _build_dataset(cfg):
    dataset = hydra.utils.instantiate(cfg.data.test_data)
    dataset.init(preprocess=False)
    return dataset


# --------------------------------------------------------------------------- #
# Inference stages (mirror dare3d.predict.do_segmentation / do_regression)     #
# --------------------------------------------------------------------------- #
def _segment(cfg, torch_device, *, overlap, batch_size, threshold, min_weighted_prob, should_stop=None):
    model = _build_model(cfg, torch_device)
    dataset = _build_dataset(cfg)
    predictions = segmentation_inference(
        dataset, model, torch_device, cfg.crop_size, batch_size, overlap,
        output_dir=None, should_stop=should_stop,
    )

    centers = []
    for m, predicted_movie in enumerate(predictions):
        # predicted_movie is (T, Z, Y, X) -> back to internal (T, X, Y, Z)
        movie = np.swapaxes(predicted_movie, -1, -3)
        binary_pred = (movie > threshold).astype(np.uint8)

        pred_ccs = connected_components(binary_pred, return_N=False)
        stats = statistics_optimized(pred_ccs, movie)

        maximum_size = get_sphere_vol(radius=cfg.cell_radius) * 3
        binary_pred = filter_by_object_weighted_prob(
            binary_pred, pred_ccs, stats, min_weighted_prob, maximum_size
        )

        pred_ccs = connected_components(binary_pred, return_N=False)
        stats = statistics_optimized(pred_ccs, movie)

        for centroid in stats["centroids"]:
            t, x, y, z = centroid
            centers.append((m, t, x, y, z))
    return centers


def _regress(cfg, torch_device, centers, should_stop=None):
    model = _build_model(cfg, torch_device)
    dataset = _build_dataset(cfg)
    dataset.pad_images()
    dataset._normalize(dataset.renorm)
    return regression_inference(
        dataset, model, centers, torch_device, output_dir=None, should_stop=should_stop
    )


# --------------------------------------------------------------------------- #
# Center / prediction -> napari-ready detection dicts                          #
# --------------------------------------------------------------------------- #
def _napari_point(center) -> tuple:
    """``(m, t, x, y, z)`` internal -> napari ``(t, z, y, x)``."""
    _m, t, x, y, z = center
    return (float(t), float(z), float(y), float(x))


def _axis_from_quat(quat) -> np.ndarray:
    """Division-axis unit vector in internal ``(x, y, z)`` order from a wxyz quaternion.

    Equivalent to ``angles3d.compute_axis_angle`` (axis = imaginary part divided by
    ``sin(angle/2) == |imag|``), but guarded against the angle->0 singularity.
    """
    v = np.asarray(quat[1:], dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.zeros(3)
    return v / n


def _center_to_detection(center) -> Dict:
    return {
        "center_internal": tuple(float(v) for v in center),
        "center_napari": _napari_point(center),
    }


def _prediction_to_detection(pred) -> Dict:
    center = pred["center"]  # (m, t, x, y, z)
    length = float(np.asarray(pred["length"]).reshape(-1)[0])
    quat = np.asarray(pred["rotation"], dtype=np.float64)  # wxyz
    axis_xyz = _axis_from_quat(quat)  # internal (x, y, z)
    return {
        "center_internal": tuple(float(v) for v in center),
        "center_napari": _napari_point(center),
        "length": length,
        "axis_napari": tuple(float(v) for v in axis_xyz[::-1]),  # (dz, dy, dx)
        "quaternion": tuple(float(v) for v in quat),
    }


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
def infer_stack(
    stack: np.ndarray,
    seg_model_dir: str,
    reg_model_dir: Optional[str] = None,
    *,
    device: str = "gpu",
    overlap: float = 0.25,
    batch_size: int = 4,
    threshold: float = 0.5,
    min_weighted_prob: float = 0.1,
    scale_file: Optional[str] = None,
    default_scale=None,
    target_scale=None,
    frames: Optional[Tuple[int, int]] = None,
    progress_cb: ProgressCb = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> List[Dict]:
    """Run DARE3D inference on an in-memory stack.

    Args:
        stack: ``(T, Z, Y, X)`` or ``(Z, Y, X)`` array (napari order).
        seg_model_dir: model dir holding ``.hydra/config.yaml`` + ``checkpoints/``.
        reg_model_dir: optional regression model dir; if given, each detection
            also carries orientation (``length``, ``axis_napari``, ``quaternion``).
        device: ``"gpu"``/``"cuda"`` or ``"cpu"``. Regression requires CUDA
            (the upstream ``regression_inference`` has no CPU path).
        overlap, batch_size, threshold, min_weighted_prob: segmentation knobs
            (defaults match ``configs/predict.yaml``).
        scale_file / default_scale / target_scale: optional scale overrides;
            if omitted, the saved training config's values are used.
        frames: optional ``(t_start, t_end)`` INCLUSIVE time window (original frame
            indices) to analyse; ``None`` (default) analyses the whole movie. The
            needed context frames before ``t_start`` are included automatically and
            detection times are returned in original-movie indices.
        progress_cb: optional callback receiving stage strings
            ("segmentation", "regression", "done", "aborted").
        should_stop: optional predicate polled between time frames (segmentation)
            and between detections (regression). When it returns True the run is
            aborted at the next boundary and an EMPTY list is returned (no partial
            results), so the caller adds no layers.

    Returns:
        A list of detection dicts (one per detected division), in napari order.
        Empty if nothing was detected or if ``should_stop`` aborted the run.
    """
    stack = np.asarray(stack)
    if stack.ndim == 3:
        stack = stack[None]  # (Z, Y, X) -> (1, Z, Y, X)
    if stack.ndim != 4:
        raise ValueError(f"Expected a (T,Z,Y,X) or (Z,Y,X) stack, got shape {stack.shape}")
    if stack.dtype == np.float64:
        stack = stack.astype(np.float32)  # avoid the dataset's float64 -> float16 downcast

    # Restrict to the requested time window (keeping the model's context frames).
    offset = 0
    if frames is not None:
        t0, t1 = int(frames[0]), int(frames[1])
        n_t = stack.shape[0]
        if not (0 <= t0 <= t1 < n_t):
            raise ValueError(f"frames {(t0, t1)} out of range for a movie of {n_t} timepoint(s)")
        offset = max(0, t0 - _history_frames(seg_model_dir))
        stack = stack[offset : t1 + 1]

    torch_device = _torch_device(device)
    if reg_model_dir is not None and torch_device.type == "cpu":
        raise RuntimeError(
            "DARE3D regression inference requires a CUDA device "
            "(dare3d.metrics.inference.regression_inference has no CPU path)."
        )

    def report(stage: str) -> None:
        if progress_cb is not None:
            progress_cb(stage)

    scale_kw = dict(scale_file=scale_file, default_scale=default_scale, target_scale=target_scale)

    try:
        with tempfile.TemporaryDirectory(prefix="dare3d_napari_") as tmp:
            # On-disk order must be (T, Z, Y, X) — read_tif_and_order_xyz swaps it to
            # internal (T, X, Y, Z).
            tifffile.imwrite(os.path.join(tmp, "movie.tif"), stack)

            report("segmentation")
            seg_cfg = _load_inference_cfg(seg_model_dir, tmp, device, **scale_kw)
            centers = _segment(
                seg_cfg, torch_device,
                overlap=overlap, batch_size=batch_size,
                threshold=threshold, min_weighted_prob=min_weighted_prob,
                should_stop=should_stop,
            )

            detections = [_center_to_detection(c) for c in centers]

            if reg_model_dir is not None and len(centers) > 0:
                report("regression")
                reg_cfg = _load_inference_cfg(reg_model_dir, tmp, device, **scale_kw)
                preds = _regress(reg_cfg, torch_device, centers, should_stop=should_stop)
                detections = [_prediction_to_detection(p) for p in preds if p is not None]
    except InferenceAborted:
        report("aborted")
        return []

    # Map detection times from sub-stack indices back to original-movie indices.
    if offset:
        for d in detections:
            ci = list(d["center_internal"]); ci[1] += offset; d["center_internal"] = tuple(ci)
            cn = list(d["center_napari"]); cn[0] += offset; d["center_napari"] = tuple(cn)

    report("done")
    return detections


def to_layer_data(
    detections: Sequence[Dict],
    *,
    point_size: float = 12.0,
    axis_point_size: float = 4.0,
) -> List[tuple]:
    """Convert detections into napari ``LayerDataTuple``s (imports no napari).

    Returns ``(data, kwargs, layer_type)`` tuples:
      - Points "DARE3D centers" ``(N, 4)`` = ``(t, z, y, x)`` — division centers.
      - Points "DARE3D axes" ``(M, 4)`` (only when orientation is present): each
        division axis sampled as a dense set of points along its TRUE 3D length
        (centred on the detection, at integer ``t``). Points — not a Vectors layer —
        because a napari Vectors layer cannot render a stick across a spatial slice
        axis when a time axis is also present (its out-of-slice test requires the
        zero time-projection to cover the half-slice distance, which never holds;
        see napari ``layers/vectors/_slice.py``). Sampling makes the axis visible
        crossing every z-slice it traverses while scrolling z in 2D view. The length
        is fixed (the measured division length) and not adjustable.
    """
    if len(detections) == 0:
        return []

    points = np.array([d["center_napari"] for d in detections], dtype=float)
    layers: List[tuple] = []

    point_kwargs = {
        "name": "DARE3D centers",
        "size": point_size,
        "face_color": "red",
        "border_color": "white",  # napari >=0.5 (was edge_color in 0.4.x)
        # NOTE: no out_of_slice_display — with only a handful of time frames and a
        # point size of ~12 it would render every detection on (almost) every time
        # slice, making the detections look identical across all timepoints.
    }
    has_orientation = "axis_napari" in detections[0]
    if has_orientation:
        point_kwargs["features"] = {"length": np.array([d["length"] for d in detections], dtype=float)}
    layers.append((points, point_kwargs, "points"))

    if has_orientation:
        # Sample each axis densely (~2 points / voxel) along its true 3D length,
        # centred on the detection, at integer t. Each sample sits at its own z, so
        # scrolling z in 2D view reveals the axis crossing every z-slice it traverses.
        samples = []
        for d in detections:
            t = float(round(d["center_napari"][0]))
            c = np.asarray(d["center_napari"][1:], dtype=float)  # (z, y, x)
            axis = np.asarray(d["axis_napari"], dtype=float)     # (dz, dy, dx), unit
            length = float(d["length"])
            n = max(2, int(np.ceil(2.0 * length)) + 1)
            for s in np.linspace(-0.5 * length, 0.5 * length, n):
                p = c + s * axis
                samples.append((t, p[0], p[1], p[2]))
        axis_points = np.asarray(samples, dtype=float)
        axis_kwargs = {
            "name": "DARE3D axes",
            "size": axis_point_size,
            "face_color": "cyan",
            "border_color": "cyan",
        }
        layers.append((axis_points, axis_kwargs, "points"))

    return layers
