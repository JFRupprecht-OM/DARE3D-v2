"""Import-light spatial calibration helpers for the napari plugin."""

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def safe_movie_stem(movie_name: str) -> str:
    """Return a filesystem-safe stem while preserving scale-table lookup names."""
    stem = Path(str(movie_name or "")).stem.strip()
    if not stem or stem in {".", ".."}:
        return "movie"
    return "".join("_" if char in '<>:"/\\|?*' else char for char in stem)


def _validated_xyz(values: Sequence[float], source: str) -> tuple[float, float, float]:
    scale = np.asarray(values, dtype=float).reshape(-1)
    if scale.size != 3:
        raise ValueError(f"{source} must contain exactly three XYZ values")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError(f"{source} values must be finite and positive")
    return tuple(float(value) for value in scale)


def resolve_source_scale_xyz(
    movie_name: str,
    scale_file: Optional[str],
    default_scale: Optional[Sequence[float]] = None,
) -> Optional[tuple[float, float, float]]:
    """Resolve source XYZ spacing with the same entry-before-default precedence."""
    if scale_file:
        path = Path(scale_file)
        with path.open(encoding="utf-8") as stream:
            scales = json.load(stream)
        stem = safe_movie_stem(movie_name)
        if stem in scales:
            return _validated_xyz(scales[stem], f"scale entry {stem!r}")
    if default_scale is None:
        return None
    return _validated_xyz(default_scale, "default_scale")


def movie_fallback_scale_xyz(
    movie_name: str,
) -> Optional[tuple[float, float, float]]:
    """Return the documented fallback calibration for known uncalibrated movies."""
    if safe_movie_stem(movie_name).casefold() == "movie_m":
        # Requested TZYX fallback (1, 1, 0.2, 0.2) converted to internal XYZ.
        return (0.2, 0.2, 1.0)
    return None


def napari_scale_from_xyz(
    scale_xyz: Sequence[float], stack_ndim: int
) -> tuple[float, ...]:
    """Convert source XYZ spacing to napari ZYX or TZYX layer scale."""
    x, y, z = _validated_xyz(scale_xyz, "source scale")
    if stack_ndim == 3:
        return (z, y, x)
    if stack_ndim == 4:
        return (1.0, z, y, x)
    raise ValueError(f"Expected a 3D or 4D stack, got {stack_ndim} dimensions")


def xyz_from_napari_scale(
    layer_scale: Sequence[float], stack_ndim: int
) -> tuple[float, float, float]:
    """Convert a napari ZYX/TZYX layer scale to source XYZ spacing."""
    scale = np.asarray(layer_scale, dtype=float).reshape(-1)
    if scale.size != stack_ndim:
        raise ValueError(
            f"Image layer scale has {scale.size} values for a {stack_ndim}D stack"
        )
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("Image layer scale values must be finite and positive")
    z, y, x = scale[-3:]
    return (float(x), float(y), float(z))
