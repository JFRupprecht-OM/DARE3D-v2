"""Permanent scale-table, resampling-geometry, and napari contract tests."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dare3d.data.components.abstract_celldataset import AbstractCellDataset
from napari_dare3d import _api as napari_api
from napari_dare3d._scale import (
    napari_scale_from_xyz,
    resolve_source_scale_xyz,
    xyz_from_napari_scale,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCALE_FILE = REPOSITORY_ROOT / "data" / "3D" / "scales.json"
SCALE_SEMANTIC_SHA256 = (
    "63b395ea45713ccb329a25451da843de4ed43f0cee516258c407dd1ed0be31cf"
)


def _geometry_dataset(scale_file, default_scale=(0.621, 0.621, 2.0)):
    dataset = object.__new__(AbstractCellDataset)
    dataset.default_scale = np.asarray(default_scale, dtype=float)
    dataset.target_scale = np.ones(3, dtype=float)
    dataset.scale_file = str(scale_file) if scale_file is not None else None
    dataset.scale_file_exists = bool(
        dataset.scale_file and Path(dataset.scale_file).is_file()
    )
    dataset.movies_scale = dataset.load_movie_scales(dataset.scale_file)
    return dataset


def test_recovered_v1_scale_table_is_present_and_semantically_unchanged():
    scales = json.loads(SCALE_FILE.read_text(encoding="utf-8"))
    canonical = json.dumps(scales, sort_keys=True, separators=(",", ":")).encode()

    assert len(scales) == 79
    assert hashlib.sha256(canonical).hexdigest() == SCALE_SEMANTIC_SHA256
    assert scales["movie2"] == [0.914, 0.914, 0.914]
    assert scales["movie3"] == [0.914, 0.914, 0.914]
    assert scales["movie4"] == [1, 1, 1]
    assert scales["movie_M"] == [0.2076, 0.2076, 1.0]
    assert "movie_E" not in scales
    assert "movie_I2" not in scales


def test_recovered_table_reproduces_archived_nuclei_target_shapes():
    dataset = _geometry_dataset(SCALE_FILE)

    assert dataset._compute_target_shape(
        (10, 362, 305, 180), "movie2"
    ) == (10, 330, 278, 164)
    assert dataset._compute_target_shape(
        (28, 424, 411, 85), "movie3"
    ) == (28, 387, 375, 77)
    assert dataset._compute_target_shape(
        (28, 405, 394, 152), "movie4"
    ) == (28, 405, 394, 152)


def test_missing_table_uses_documented_fallback_and_changes_geometry(tmp_path):
    missing = tmp_path / "missing-scales.json"
    dataset = _geometry_dataset(missing)

    assert dataset.movies_scale == {}
    assert dataset._compute_target_shape(
        (10, 362, 305, 180), "movie2"
    ) == (10, 224, 189, 360)


def test_log_reconstructed_and_recovered_movie2_geometry_are_identical():
    reconstructed = _geometry_dataset(
        REPOSITORY_ROOT / "__missing_scales_for_test__.json",
        default_scale=(0.912, 0.912, 0.912),
    )
    recovered = _geometry_dataset(SCALE_FILE)
    shape = (10, 362, 305, 180)

    assert reconstructed._compute_target_shape(shape, "movie2") == (
        recovered._compute_target_shape(shape, "movie2")
    )


@pytest.mark.parametrize(
    "config_name",
    (
        "regression.yaml",
        "segmentation.yaml",
        "segres.yaml",
        "tap.yaml",
        "finetune_regression.yaml",
        "finetune_segmentation.yaml",
    ),
)
def test_experiment_configs_use_case_correct_scale_path(config_name):
    text = (
        REPOSITORY_ROOT / "configs" / "experiment" / config_name
    ).read_text(encoding="utf-8")

    assert "/3D/scales.json" in text
    assert "/3d/scales.json" not in text


def test_distribution_manifest_includes_backward_compatible_scale_path():
    manifest = (REPOSITORY_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    setup = (REPOSITORY_ROOT / "setup.py").read_text(encoding="utf-8")

    assert "include data/3D/scales.json" in manifest.splitlines()
    assert 'data_files=[("data/3D", ["data/3D/scales.json"])]' in setup


def test_napari_api_preserves_movie_stem_for_per_movie_lookup(monkeypatch):
    observed = {}

    def fake_load_config(_model_dir, image_dir, _device, **kwargs):
        observed["tiffs"] = sorted(path.name for path in Path(image_dir).glob("*.tif"))
        observed["scale_file"] = kwargs["scale_file"]
        return SimpleNamespace()

    monkeypatch.setattr(napari_api, "_load_inference_cfg", fake_load_config)
    monkeypatch.setattr(napari_api, "_segment", lambda *args, **kwargs: [])

    detections = napari_api.infer_stack(
        np.zeros((1, 2, 3, 4), dtype=np.uint8),
        "unused-model-dir",
        device="cpu",
        movie_name="movie2.tif",
        scale_file=str(SCALE_FILE),
    )

    assert detections == []
    assert observed["tiffs"] == ["movie2.tif"]
    assert observed["scale_file"] == str(SCALE_FILE)


def test_napari_calibration_resolves_json_manual_and_layer_sources():
    assert resolve_source_scale_xyz("movie_M.tif", str(SCALE_FILE)) == (
        0.2076,
        0.2076,
        1.0,
    )
    assert resolve_source_scale_xyz(
        "unknown.tif", str(SCALE_FILE), (0.4, 0.5, 1.5)
    ) == (0.4, 0.5, 1.5)
    assert xyz_from_napari_scale((1.0, 1.5, 0.5, 0.4), 4) == (
        0.4,
        0.5,
        1.5,
    )
    assert napari_scale_from_xyz((0.4, 0.5, 1.5), 4) == (
        1.0,
        1.5,
        0.5,
        0.4,
    )


def test_napari_result_layers_inherit_anisotropic_image_scale():
    detection = {
        "center_internal": (0, 1, 2, 3, 4),
        "center_napari": (1.0, 4.0, 3.0, 2.0),
        "axis_napari": (1.0, 0.0, 0.0),
        "length": 4.0,
    }
    layer_scale = (1.0, 2.0, 0.621, 0.621)

    layers = napari_api.to_layer_data([detection], layer_scale=layer_scale)

    assert len(layers) == 2
    assert all(layer_kwargs["scale"] == layer_scale for _, layer_kwargs, _ in layers)


@pytest.mark.parametrize(
    "bad_scale",
    ((1.0, 2.0, 3.0), (1.0, 1.0, 0.0, 1.0)),
)
def test_napari_result_layer_scale_is_validated(bad_scale):
    detection = {
        "center_internal": (0, 1, 2, 3, 4),
        "center_napari": (1.0, 4.0, 3.0, 2.0),
    }

    with pytest.raises(ValueError):
        napari_api.to_layer_data([detection], layer_scale=bad_scale)
