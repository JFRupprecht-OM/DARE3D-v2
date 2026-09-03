"""Coordinate and unit contracts for generic DARE3D regression inference.

The segmentation/detection stage reports centers in the raw internal
(M, T, X, Y, Z) grid. A regression checkpoint, however, was trained on a
spatially resampled (T, X, Y, Z) movie. This module keeps those spaces
explicit and contains no model- or dataset-specific special cases.
"""

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


TRAINING_CONSISTENT = "training_consistent"
LEGACY_RAW = "legacy_raw"
VALID_PREPROCESSING_MODES = (TRAINING_CONSISTENT, LEGACY_RAW)


def normalize_preprocessing_mode(mode: str | None) -> str:
    """Return a validated regression preprocessing mode."""

    value = TRAINING_CONSISTENT if mode is None else str(mode).lower()
    if value not in VALID_PREPROCESSING_MODES:
        raise ValueError(
            f"Unknown regression preprocessing mode {mode!r}; expected one of "
            f"{VALID_PREPROCESSING_MODES}"
        )
    return value


def _float_xyz(value: Sequence[float], name: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain three finite XYZ values, got {value!r}")
    if np.any(array <= 0):
        raise ValueError(f"{name} must be strictly positive, got {value!r}")
    return tuple(float(item) for item in array)


def _int_xyz(value: Sequence[int], name: str) -> tuple[int, int, int]:
    array = np.asarray(value, dtype=np.int64)
    if array.shape != (3,) or np.any(array <= 0):
        raise ValueError(f"{name} must contain three positive XYZ values, got {value!r}")
    return tuple(int(item) for item in array)


@dataclass(frozen=True)
class RegressionPreprocessingSpec:
    """Checkpoint/input preprocessing contract for one movie.

    source_spacing_xyz is the source/effective spacing used by the checkpoint
    configuration. For legacy checkpoints it may be an assumed training scale
    rather than independently verified physical metadata; scale_source makes
    that provenance visible.
    """

    mode: str
    movie_name: str
    source_spacing_xyz: tuple[float, float, float]
    declared_source_spacing_xyz: tuple[float, float, float]
    target_spacing_xyz: tuple[float, float, float]
    scale_factor_xyz: tuple[float, float, float]
    realized_shape_scale_factor_xyz: tuple[float, float, float]
    raw_shape_xyz: tuple[int, int, int]
    regression_shape_xyz: tuple[int, int, int]
    crop_size_xyz: tuple[int, int, int]
    padding_xyz: tuple[int, int, int]
    input_channels: tuple[int, ...]
    normalization: str | None
    scale_source: str
    scale_file: str | None
    internal_order: str = "TXYZ"
    center_order: str = "MTXYZ"
    disk_order: str = "TZYX"
    target_shape_rule: str = "floor(raw_shape * source_spacing / target_spacing)"
    point_transform_rule: str = "round(raw_coordinate * scale_factor + 1e-9)"
    resize_library: str = "skimage.transform.resize"
    resize_order: int = 1
    resize_mode: str = "reflect"
    resize_clip: bool = True
    resize_preserve_range: bool = True
    padding_mode: str = "constant"
    normalization_after_padding: bool = True
    schema_version: int = 1

    @classmethod
    def from_dataset(
        cls,
        dataset: Any,
        movie_index: int,
        mode: str = TRAINING_CONSISTENT,
    ) -> "RegressionPreprocessingSpec":
        mode = normalize_preprocessing_mode(mode)
        movie_name = str(dataset.movie_names[movie_index])
        raw_shape = tuple(int(item) for item in dataset.original_movies_shape[movie_index][1:])

        if movie_name in dataset.movies_scale:
            declared_source_spacing = _float_xyz(
                dataset.movies_scale[movie_name], "movie source spacing"
            )
            scale_source = f"scale_file:{dataset.scale_file}:{movie_name}"
        else:
            declared_source_spacing = _float_xyz(
                dataset.default_scale, "default source spacing"
            )
            scale_source = "default_scale"

        configured_target = _float_xyz(dataset.target_scale, "target spacing")
        if mode == TRAINING_CONSISTENT:
            target_spacing = configured_target
            scale_factor = _float_xyz(
                dataset.get_movie_scale(movie_name), "dataset scale factor"
            )
            regression_shape = tuple(
                int(item)
                for item in dataset._compute_target_shape(
                    dataset.original_movies_shape[movie_index], movie_name
                )[1:]
            )
            declared_scale_factor = np.asarray(
                declared_source_spacing, dtype=np.float64
            ) / np.asarray(target_spacing, dtype=np.float64)
            if not np.allclose(
                scale_factor, declared_scale_factor, rtol=1e-9, atol=1e-12
            ):
                source_spacing = tuple(
                    float(scale * target)
                    for scale, target in zip(scale_factor, target_spacing)
                )
                scale_source = (
                    "dataset.get_movie_scale override:"
                    f"{type(dataset).__module__}.{type(dataset).__name__};"
                    f"declared={scale_source}"
                )
            else:
                source_spacing = declared_source_spacing
        else:
            source_spacing = declared_source_spacing
            target_spacing = source_spacing
            scale_factor = (1.0, 1.0, 1.0)
            regression_shape = raw_shape

        realized_shape_scale_factor = tuple(
            float(target / raw)
            for target, raw in zip(regression_shape, raw_shape)
        )

        crop_size = _int_xyz(dataset.crop_size, "crop size")
        padding = tuple(int(item // 2) for item in crop_size)
        return cls(
            mode=mode,
            movie_name=movie_name,
            declared_source_spacing_xyz=declared_source_spacing,
            source_spacing_xyz=source_spacing,
            target_spacing_xyz=target_spacing,
            realized_shape_scale_factor_xyz=realized_shape_scale_factor,
            scale_factor_xyz=scale_factor,
            raw_shape_xyz=_int_xyz(raw_shape, "raw shape"),
            regression_shape_xyz=_int_xyz(regression_shape, "regression shape"),
            crop_size_xyz=crop_size,
            padding_xyz=padding,
            input_channels=tuple(int(item) for item in dataset.input_channels),
            normalization=dataset.renorm,
            scale_source=scale_source,
            scale_file=getattr(dataset, "scale_file", None),
        )

    @property
    def physical_crop_size_xyz(self) -> tuple[float, float, float]:
        return tuple(
            float(size * spacing)
            for size, spacing in zip(self.crop_size_xyz, self.target_spacing_xyz)
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["physical_crop_size_xyz"] = self.physical_crop_size_xyz
        return result


@dataclass(frozen=True)
class RegressionSpatialTransform:
    """Transform between raw and checkpoint regression grids for one movie."""

    specification: RegressionPreprocessingSpec

    @property
    def scale_factor_xyz(self) -> np.ndarray:
        return np.asarray(self.specification.scale_factor_xyz, dtype=np.float64)

    def raw_point_to_regression(
        self, point_xyz: Sequence[float], *, round_result: bool = False
    ) -> np.ndarray:
        point = np.asarray(point_xyz, dtype=np.float64)
        if point.shape != (3,):
            raise ValueError(f"Expected an XYZ point, got shape {point.shape}")
        transformed = point * self.scale_factor_xyz
        if round_result:
            transformed = np.round(transformed + 1e-9)
        return transformed

    def regression_point_to_raw(self, point_xyz: Sequence[float]) -> np.ndarray:
        point = np.asarray(point_xyz, dtype=np.float64)
        if point.shape != (3,):
            raise ValueError(f"Expected an XYZ point, got shape {point.shape}")
        return point / self.scale_factor_xyz

    def raw_center_to_regression(
        self, center_m_t_xyz: Sequence[float], *, round_result: bool = False
    ) -> tuple[float, float, float, float, float]:
        center = np.asarray(center_m_t_xyz, dtype=np.float64)
        if center.shape != (5,):
            raise ValueError(f"Expected an (M,T,X,Y,Z) center, got {center_m_t_xyz!r}")
        spatial = self.raw_point_to_regression(center[2:], round_result=round_result)
        return tuple(float(item) for item in np.concatenate((center[:2], spatial)))

    def regression_center_to_raw(
        self, center_m_t_xyz: Sequence[float]
    ) -> tuple[float, float, float, float, float]:
        center = np.asarray(center_m_t_xyz, dtype=np.float64)
        if center.shape != (5,):
            raise ValueError(f"Expected an (M,T,X,Y,Z) center, got {center_m_t_xyz!r}")
        spatial = self.regression_point_to_raw(center[2:])
        return tuple(float(item) for item in np.concatenate((center[:2], spatial)))

    @staticmethod
    def normalize_axis(axis_xyz: Sequence[float]) -> np.ndarray:
        axis = np.asarray(axis_xyz, dtype=np.float64)
        if axis.shape != (3,):
            raise ValueError(f"Expected an XYZ axis, got shape {axis.shape}")
        norm = float(np.linalg.norm(axis))
        if not np.isfinite(norm) or norm < 1e-8:
            return np.zeros(3, dtype=np.float64)
        return axis / norm

    def decode_axis_length(
        self,
        center_raw_m_t_xyz: Sequence[float],
        axis_regression_xyz: Sequence[float],
        length_regression_voxels: float,
    ) -> dict[str, Any]:
        """Decode checkpoint-native axis/length into physical and raw geometry."""

        center = np.asarray(center_raw_m_t_xyz, dtype=np.float64)
        if center.shape != (5,):
            raise ValueError(
                f"Expected an (M,T,X,Y,Z) raw center, got {center_raw_m_t_xyz!r}"
            )
        length_regression = float(
            np.asarray(length_regression_voxels, dtype=np.float64).reshape(-1)[0]
        )
        axis_regression = self.normalize_axis(axis_regression_xyz)
        half_regression = 0.5 * length_regression * axis_regression

        target_spacing = np.asarray(
            self.specification.target_spacing_xyz, dtype=np.float64
        )
        half_physical = half_regression * target_spacing
        half_raw = half_regression / self.scale_factor_xyz

        physical_length = float(2.0 * np.linalg.norm(half_physical))
        raw_length = float(2.0 * np.linalg.norm(half_raw))
        physical_axis = self.normalize_axis(half_physical)
        raw_axis = self.normalize_axis(half_raw)
        endpoints_raw = np.stack(
            (center[2:] - half_raw, center[2:] + half_raw), axis=0
        )

        return {
            "center_raw": tuple(float(item) for item in center),
            "center_regression": self.raw_center_to_regression(center),
            "axis_regression_xyz": axis_regression,
            "axis_physical_xyz": physical_axis,
            "axis_raw_xyz": raw_axis,
            "length_regression_voxels": length_regression,
            "length_physical_um": physical_length,
            "length_raw_voxels": raw_length,
            "endpoints_raw_xyz": endpoints_raw,
            "preprocessing_mode": self.specification.mode,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.specification.to_dict()


def axis_from_wxyz_quaternion(quaternion: Sequence[float]) -> np.ndarray:
    """Return DARE3D's encoded division axis from a wxyz quaternion.

    DARE3D stores the division-axis direction in the normalized imaginary
    quaternion component. A zero imaginary component has no defined axis and
    is returned as a zero vector.
    """

    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f"Expected a wxyz quaternion, got shape {quaternion.shape}")
    vector = quaternion[1:]
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return vector / norm
