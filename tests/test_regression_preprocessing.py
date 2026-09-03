"""Regression preprocessing geometry and crop-parity tests."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import dare3d.metrics.inference as inference_module

from dare3d.data.components.regress_3dataset import Regress3Dataset
from dare3d.data.components.regression_geometry import (
    LEGACY_RAW,
    TRAINING_CONSISTENT,
    RegressionPreprocessingSpec,
)


class _MemoryRegressionDataset(Regress3Dataset):
    """Small in-memory dataset that still exercises the production init paths."""

    def __init__(self, movie, bipoints, **kwargs):
        self._test_movie = np.asarray(movie)
        self._test_bipoints = copy.deepcopy(bipoints)
        super().__init__(**kwargs)

    def _list_data(self):
        self.movie_names = ["movie"]

    def _load_data(self):
        movie = self._test_movie.copy()
        self.original_movies_shape = [movie.shape]
        labels = [copy.deepcopy(self._test_bipoints)] if self.load_labels else []
        return [movie], labels


def _movie_and_bipoints():
    movie = np.arange(4 * 12 * 10 * 8, dtype=np.float32).reshape(4, 12, 10, 8)
    bipoints = [
        [],
        [],
        [(np.array([4, 4, 2]), np.array([8, 6, 4]))],
        [],
    ]
    return movie, bipoints


def _dataset(
    *,
    mode=TRAINING_CONSISTENT,
    load_labels=True,
    scale_file=None,
    require_scale_file=False,
):
    movie, bipoints = _movie_and_bipoints()
    return _MemoryRegressionDataset(
        movie,
        bipoints,
        im_folder="unused",
        label_folder="unused",
        input_channels=[-1, 0, 1],
        scale_file=scale_file,
        default_scale=[0.5, 1.0, 2.0],
        target_scale=1.0,
        renorm="min-max",
        training=False,
        steps_per_epoch=-1,
        load_labels=load_labels,
        time_axis_padding=0,
        crop_size=4,
        angle_representation="rotation_matrix_SVD",
        inference_preprocessing=mode,
        require_scale_file=require_scale_file,
    )


@pytest.mark.parametrize(
    ("source_spacing", "expected_shape"),
    [
        ((0.208, 0.208, 1.0), (212, 212, 10)),
        ((0.621, 0.621, 2.0), (635, 635, 20)),
    ],
)
def test_neural_tube_checkpoint_profiles(source_spacing, expected_shape):
    dataset = SimpleNamespace(
        movie_names=["movie"],
        original_movies_shape=[(21, 1024, 1024, 10)],
        movies_scale={},
        default_scale=np.asarray(source_spacing),
        target_scale=np.ones(3),
        crop_size=np.full(3, 32),
        input_channels=[-1, 0, 1],
        renorm="percentile",
        scale_file=None,
    )
    dataset.get_movie_scale = lambda name: np.asarray(source_spacing)
    dataset._compute_target_shape = lambda shape, name: (
        shape[0],
        *tuple((np.asarray(shape[1:]) * np.asarray(source_spacing)).astype(np.int32)),
    )

    spec = RegressionPreprocessingSpec.from_dataset(dataset, 0)

    assert spec.raw_shape_xyz == (1024, 1024, 10)
    assert spec.regression_shape_xyz == expected_shape
    assert spec.padding_xyz == (16, 16, 16)
    assert spec.physical_crop_size_xyz == (32.0, 32.0, 32.0)
    assert spec.center_order == "MTXYZ"
    assert spec.internal_order == "TXYZ"
    assert spec.disk_order == "TZYX"


def test_dataset_scale_override_controls_coordinates_and_is_recorded():
    dataset = SimpleNamespace(
        movie_names=["movie2"],
        original_movies_shape=[(10, 362, 305, 180)],
        movies_scale={},
        default_scale=np.ones(3),
        target_scale=np.ones(3),
        crop_size=np.full(3, 32),
        input_channels=[-1, 0, 1],
        renorm="min-max",
        scale_file=None,
    )
    exact_target = np.array([330, 278, 164])
    dataset.get_movie_scale = lambda name: exact_target / np.array([362, 305, 180])
    dataset._compute_target_shape = lambda shape, name: (shape[0], *exact_target)

    spec = RegressionPreprocessingSpec.from_dataset(dataset, 0)
    transform = dataset.get_movie_scale("movie2")

    assert np.allclose(spec.scale_factor_xyz, transform)
    assert spec.regression_shape_xyz == (330, 278, 164)
    assert "dataset.get_movie_scale override" in spec.scale_source
    assert np.allclose(spec.source_spacing_xyz, transform)
    assert spec.declared_source_spacing_xyz == (1.0, 1.0, 1.0)


def test_raw_regression_transform_preserves_movie_and_time():
    dataset = _dataset()
    dataset.init_inference()
    transform = dataset.get_regression_transform(0)

    raw_center = (0, 2, 6, 5, 3)
    regression_center = transform.raw_center_to_regression(raw_center)

    assert regression_center == (0.0, 2.0, 3.0, 5.0, 6.0)
    assert np.allclose(
        transform.regression_center_to_raw(regression_center), raw_center
    )


def test_prediction_decode_has_explicit_units_and_raw_endpoints():
    dataset = _dataset()
    dataset.init_inference()
    transform = dataset.get_regression_transform(0)
    center = (0, 2, 6, 5, 3)
    axis_regression = np.ones(3) / np.sqrt(3.0)

    decoded = transform.decode_axis_length(center, axis_regression, 6.0)

    assert decoded["length_regression_voxels"] == pytest.approx(6.0)
    assert decoded["length_physical_um"] == pytest.approx(6.0)
    assert np.linalg.norm(decoded["axis_physical_xyz"]) == pytest.approx(1.0)
    assert np.linalg.norm(decoded["axis_raw_xyz"]) == pytest.approx(1.0)
    endpoints = decoded["endpoints_raw_xyz"]
    assert np.allclose(endpoints.mean(axis=0), center[2:])
    half_raw = endpoints[1] - np.asarray(center[2:])
    assert np.allclose(
        half_raw * transform.scale_factor_xyz,
        3.0 * axis_regression,
    )


def test_legacy_raw_mode_is_explicit_identity_replay():
    dataset = _dataset(mode=LEGACY_RAW)
    dataset.init_inference()
    transform = dataset.get_regression_transform(0)

    assert dataset.regression_preprocessing_mode == LEGACY_RAW
    assert transform.specification.regression_shape_xyz == (12, 10, 8)
    assert transform.raw_center_to_regression((0, 2, 6, 5, 3)) == (
        0.0,
        2.0,
        6.0,
        5.0,
        3.0,
    )
    assert dataset.movies_im[0].shape == (4, 16, 14, 12)


def test_training_and_inference_extract_identical_regression_crop():
    training = _dataset()
    inference = _dataset()
    training.init()
    inference.init_inference()

    movie_index, time_slice, x_slice, y_slice, z_slice = training.crops[0]
    training_crop = training.movies_im[movie_index][
        time_slice, x_slice, y_slice, z_slice
    ]

    raw_center = (0, 2, 6, 5, 3)
    inference_crop = inference.get_crop_from_center(raw_center).cpu().numpy()

    assert training_crop.shape == (3, 4, 4, 4)
    assert inference_crop.shape == training_crop.shape
    assert inference_crop.dtype == training_crop.dtype
    assert np.array_equal(inference_crop, training_crop)


def test_border_crop_keeps_training_shape_and_uses_zero_padding():
    dataset = _dataset(load_labels=False)
    dataset.init_inference()

    crop = dataset.get_crop_from_center((0, 2, 0, 0, 0)).cpu().numpy()

    assert crop.shape == (3, 4, 4, 4)
    assert np.all(crop[:, :2, :, :] == 0)
    assert np.all(crop[:, :, :2, :] == 0)
    assert np.all(crop[:, :, :, :2] == 0)


def test_groundtruth_target_is_raw_annotation_bound_and_space_explicit():
    dataset = _dataset()
    dataset.init_inference()
    target = dataset.gather_groundtruth_info([(0, 2, 6, 5, 3)])[0]

    assert target["event_id"] == "movie:2:0"
    assert target["annotation_index"] == 0
    assert target["target_candidate_count"] == 1
    assert target["center"] == (0, 2, 6.0, 5.0, 3.0)
    assert np.array_equal(
        target["annotated_endpoints_raw_xyz"],
        np.array([[4, 4, 2], [8, 6, 4]]),
    )
    assert np.array_equal(
        target["annotated_endpoints_regression_xyz"],
        np.array([[2, 4, 4], [4, 6, 8]]),
    )


def test_invalid_preprocessing_mode_fails_before_inference():
    with pytest.raises(ValueError, match="Unknown regression preprocessing mode"):
        _dataset(mode="not-a-mode")


def test_require_scale_file_rejects_default_scale_fallback():
    dataset = _dataset(require_scale_file=True)

    with pytest.raises(ValueError, match="missing entries"):
        dataset.init_inference()


def test_regression_output_keeps_legacy_and_explicit_geometry(monkeypatch):
    dataset = _dataset(load_labels=False)
    dataset.init_inference()

    class Model:
        def forward(self, _crop):
            return {
                "head1": {
                    "len": torch.tensor([[0.5]], dtype=torch.float32),
                    "angle": torch.eye(3, dtype=torch.float32).reshape(1, 9),
                }
            }

    monkeypatch.setattr(
        inference_module,
        "representation_to_quaternion",
        lambda *args, **kwargs: np.array([1.0, 0.0, 0.0, 0.0]),
    )
    center = (0, 2, 6, 5, 3)

    prediction = inference_module.regression_inference(
        dataset,
        Model(),
        [center],
        "cpu",
    )[0]

    assert prediction["center"] == prediction["center_raw"] == center
    assert {"length", "rotation"} <= prediction.keys()
    assert {
        "center_regression",
        "axis_regression_xyz",
        "axis_physical_xyz",
        "axis_raw_xyz",
        "length_regression_voxels",
        "length_physical_um",
        "length_raw_voxels",
        "endpoints_raw_xyz",
        "preprocessing_mode",
    } <= prediction.keys()
    assert prediction["preprocessing_mode"] == TRAINING_CONSISTENT


def test_manifest_records_exact_preprocessing_contract():
    dataset = _dataset()
    dataset.init_inference()
    manifest = dataset.preprocessing_manifest()
    movie = manifest["movies"][0]

    assert manifest["mode"] == TRAINING_CONSISTENT
    assert movie["raw_shape_xyz"] == (12, 10, 8)
    assert movie["regression_shape_xyz"] == (6, 10, 16)
    assert movie["scale_factor_xyz"] == (0.5, 1.0, 2.0)
    assert movie["resize_order"] == 1
    assert movie["resize_mode"] == "reflect"
    assert movie["normalization_after_padding"] is True
