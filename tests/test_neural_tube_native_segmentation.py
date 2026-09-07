"""Native-grid neural segmentation contracts and an opt-in release-asset replay."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from dare3d.metrics import inference as core_inference
from napari_dare3d import _api as api
from napari_dare3d import _release_models as release
from napari_dare3d._scale import (
    movie_fallback_scale_xyz,
    napari_scale_from_xyz,
    resolve_source_scale_xyz,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def saved_segmenter(tmp_path):
    model = tmp_path / "segmenter"
    (model / ".hydra").mkdir(parents=True)
    (model / "checkpoints").mkdir()
    (model / "checkpoints" / "last.ckpt").touch()
    cfg = OmegaConf.create({
        "input_channels": [-1, 0, 1],
        "default_scale": [0.621, 0.621, 2.0],
        "target_scale": 1.0,
        "crop_size": 128,
        "cell_radius": 8,
        "data": {"test_data": {
            "_target_": "dare3d.data.components.seg_3dataset.Segmentation3Dataset",
            "im_folder": "old-images",
            "label_folder": "old-labels",
            "scale_file": "old-workstation-scales.json",
            "default_scale": "${default_scale}",
            "target_scale": "${target_scale}",
            "input_channels": "${input_channels}",
            "crop_size": "${crop_size}",
            "radius": "${cell_radius}",
            "renorm": "min-max",
            "order_dim_img": "zyx",
            "training": False,
            "load_labels": True,
            "time_axis_padding": 1,
        }},
    })
    OmegaConf.save(cfg, model / ".hydra" / "config.yaml")
    return model


@pytest.fixture
def stack():
    # Non-square dimensions and distinct intensities expose axis/time mistakes.
    return np.arange(1, 4 * 10 * 8 * 6 + 1, dtype=np.uint16).reshape(4, 10, 8, 6)


@pytest.mark.parametrize("mode", ["source", "checkpoint_default", "native"])
def test_scale_modes_keep_saved_and_physical_settings_separate(
    tmp_path, monkeypatch, saved_segmenter, stack, mode
):
    table = tmp_path / "physical.json"
    table.write_text(json.dumps({"movie_I2": [0.2, 0.3, 4.0]}))
    config_path = saved_segmenter / ".hydra" / "config.yaml"
    original_config = config_path.read_bytes()
    seen = {}

    def inspect_segment(cfg, device, **kwargs):
        seen["cfg"] = cfg
        dataset = api._build_dataset(cfg, "segmentation")
        seen["ratio"] = dataset.get_movie_scale("movie_I2")
        seen["target"] = dataset._compute_target_shape(
            (21, 1024, 1024, 10), "movie_I2"
        )
        seen["table"] = dataset.movies_scale
        assert dataset.time_axis_padding == 0
        np.testing.assert_array_equal(dataset.movies_im[0], np.swapaxes(stack, -1, -3))
        return []

    monkeypatch.setattr(api, "_segment", inspect_segment)
    assert api.infer_stack(
        stack, str(saved_segmenter), device="cpu", movie_name="movie_I2",
        scale_file=str(table), default_scale=[0.4, 0.5, 1.5], target_scale=0.5,
        segmentation_scale_mode=mode,
    ) == []
    expected = {
        "source": ([0.4, 0.6, 8.0], (21, 409, 614, 80)),
        "checkpoint_default": ([0.621, 0.621, 2.0], (21, 635, 635, 20)),
        "native": ([1.0, 1.0, 1.0], (21, 1024, 1024, 10)),
    }
    np.testing.assert_array_equal(seen["ratio"], expected[mode][0])
    assert seen["target"] == expected[mode][1]
    if mode != "source":
        assert seen["table"] == {}
    assert config_path.read_bytes() == original_config


@pytest.mark.parametrize("movie_name", ["movie_I2", "movie_M"])
@pytest.mark.parametrize("with_table", [False, True])
def test_native_geometry_does_not_replace_regression_or_display_calibration(
    tmp_path, monkeypatch, saved_segmenter, stack, movie_name, with_table
):
    table = tmp_path / "physical.json"
    values = {"movie_M": [0.2076, 0.2076, 1.0], "movie_I2": [0.4, 0.5, 1.5]}
    if with_table:
        table.write_text(json.dumps(values))
    supplied_table = str(table) if with_table else None
    fallback = movie_fallback_scale_xyz(movie_name)
    resolved = resolve_source_scale_xyz(movie_name, supplied_table, fallback)
    expected_display = (
        napari_scale_from_xyz(resolved, 4) if resolved is not None else (1, 1, 1, 1)
    )
    centers = [(0, 2, 2, 3, 4), (0, 3, 1, 2, 3)]
    regression_seen = []

    def segment(cfg, device, **kwargs):
        td = cfg.data.test_data
        assert list(td.default_scale) == [1, 1, 1]
        assert td.target_scale == 1
        assert Path(td.scale_file).name == "__no_scales__.json"
        assert not Path(td.scale_file).exists()
        return centers

    def regress(cfg, device, passed_centers, *, preprocessing_mode, **kwargs):
        td = cfg.data.test_data
        assert preprocessing_mode == "training_consistent"
        assert passed_centers is centers
        assert td.target_scale == 0.5  # A conflicting physical target is not discarded.
        assert list(td.default_scale) == (
            list(fallback) if fallback is not None else [0.621, 0.621, 2]
        )
        if with_table:
            assert td.scale_file == str(table)
        assert td.require_scale_file is with_table
        regression_seen.append(cfg)
        return [
            {"center": c, "length": 4, "rotation": np.array([0, 1, 0, 0])}
            for c in centers
        ]

    monkeypatch.setattr(api, "_segment", segment)
    monkeypatch.setattr(api, "_regress", regress)
    detections = api.infer_stack(
        stack, str(saved_segmenter), str(saved_segmenter), device="cpu",
        movie_name=movie_name, segmentation_scale_mode="native",
        scale_file=supplied_table, default_scale=fallback, target_scale=0.5,
        regression_require_scale_file=with_table,
    )
    assert len(regression_seen) == 1
    assert len(detections) == len(centers)
    layers = api.to_layer_data(detections, layer_scale=expected_display)
    assert all(kwargs["scale"] == expected_display for _, kwargs, _ in layers)
    assert detections[0]["center_napari"] == (2, 4, 3, 2)
    if movie_name == "movie_I2" and not with_table:
        assert fallback is None and resolved is None
        assert expected_display == (1, 1, 1, 1)


def test_native_empty_segmentation_does_not_load_regression(monkeypatch, saved_segmenter, stack):
    def forbidden(*args, **kwargs):
        raise AssertionError("Regression must not run without segmentation centers")

    monkeypatch.setattr(api, "_segment", lambda *a, **k: [])
    monkeypatch.setattr(api, "_regress", forbidden)
    assert api.infer_stack(
        stack, str(saved_segmenter), str(saved_segmenter), device="cpu",
        segmentation_scale_mode="native",
    ) == []


def test_native_time_window_retains_history_and_original_coordinates(
    monkeypatch, saved_segmenter, stack
):
    def segment(cfg, device, **kwargs):
        dataset = api._build_dataset(cfg, "segmentation")
        np.testing.assert_array_equal(
            dataset.movies_im[0], np.swapaxes(stack[1:4], -1, -3)
        )
        np.testing.assert_array_equal(dataset.get_movie_scale(), [1, 1, 1])
        return [(0, 2, 2, 3, 4)]

    monkeypatch.setattr(api, "_segment", segment)
    result = api.infer_stack(
        stack, str(saved_segmenter), device="cpu",
        segmentation_scale_mode="native", frames=(3, 3),
    )
    assert result[0]["center_internal"] == (0, 3, 2, 3, 4)
    assert result[0]["center_napari"] == (3, 4, 3, 2)


def test_native_preprocessing_preserves_padding_normalization_and_causal_frames(
    monkeypatch, saved_segmenter, stack
):
    inputs = []
    seen = {}

    def sliding_window(*, inputs, roi_size, sw_batch_size, predictor, overlap, mode, device):
        assert tuple(roi_size) == (128, 128, 128)
        assert sw_batch_size == 4 and overlap == 0.5 and mode == "gaussian"
        seen["inputs"].append(inputs.cpu().numpy()[0].copy())
        return {"heatmap_0": torch.full((1, 1, 128, 128, 128), 2.0, device=device)}

    def segment(cfg, device, **kwargs):
        ds = api._build_dataset(cfg, "segmentation")
        xyz = np.swapaxes(stack, -1, -3)
        pads = ds.get_movie_padding(xyz[:3])
        assert pads == ((0, 0), (61, 61), (60, 60), (59, 59))
        padded = np.pad(xyz[:3], pads)
        expected = (padded / padded.max()).astype(np.float32)
        cpu = ds.preprocess_sample(0, 0, 3)
        tensor = ds.preprocess_sample_gpu(0, 0, 3, torch.device("cpu"))
        np.testing.assert_array_equal(cpu, expected)
        np.testing.assert_array_equal(tensor.numpy(), expected)
        assert cpu.shape == (3, 128, 128, 128)
        np.testing.assert_array_equal(ds.unscale_prediction(padded[2], 0), xyz[2])
        # Run the real time loop, sigmoid, unpadding and TZYX output conversion.
        seen["inputs"] = inputs
        seen["pred"] = core_inference.segmentation_inference(
            ds, None, device, cfg.crop_size, 4, overlap=0.5
        )[0]
        return []

    monkeypatch.setattr(core_inference.monai.inferers, "sliding_window_inference", sliding_window)
    monkeypatch.setattr(api, "_segment", segment)
    api.infer_stack(
        stack, str(saved_segmenter), device="cpu",
        segmentation_scale_mode="native", overlap=0.5,
    )
    assert len(inputs) == 2
    xyz = np.swapaxes(stack, -1, -3)
    for t, actual in enumerate(inputs):
        expected = np.pad(xyz[t:t + 3], ((0, 0), (61, 61), (60, 60), (59, 59)))
        np.testing.assert_array_equal(actual, (expected / expected.max()).astype(np.float32))
    assert seen["pred"].shape == stack.shape
    assert seen["pred"].dtype == np.float16
    assert not np.any(seen["pred"][:2])
    np.testing.assert_array_equal(
        seen["pred"][2:], np.full(stack[2:].shape, np.float16(torch.sigmoid(torch.tensor(2.0))))
    )


class _Signal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def emit(self, value):
        for callback in self.callbacks:
            callback(value)


class _Control:
    def __init__(self, value):
        self._value = value
        self.changed = _Signal()

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        self.changed.emit(value)


def test_widget_overlap_preset_preserves_manual_and_non_neural_choices(tmp_path, monkeypatch):
    # No QApplication or live viewer is created.
    import appdirs

    # Isolate Napari's early translation/settings imports from user preferences.
    monkeypatch.setattr(appdirs, "user_config_dir", lambda *a, **k: str(tmp_path / "napari"))
    monkeypatch.setattr(appdirs, "user_cache_dir", lambda *a, **k: str(tmp_path / "cache"))
    monkeypatch.setenv("NAPARI_CONFIG", str(tmp_path / "napari" / "settings.yaml"))
    from napari_dare3d import _widget as widget_module

    monkeypatch.setattr(release, "find_release_root", lambda: tmp_path)
    _, gastruloid = release.release_model_selection("gastruloid", "segmentation")
    _, neural = release.release_model_selection("neural_tube", "segmentation")
    custom = tmp_path / "custom" / neural.name
    widget = SimpleNamespace(
        dataset=_Control("gastruloid"), seg_checkpoint=_Control(gastruloid),
        overlap=_Control(0.25),
    )
    widget_module._wire_segmentation_overlap_preset(widget)
    assert widget.overlap.value == 0.25
    widget.overlap.value = 0.35
    widget.dataset.value = "neural_tube"
    widget.seg_checkpoint.value = neural
    assert widget.overlap.value == 0.5
    widget.overlap.value = 0.65
    widget.seg_checkpoint.value = neural  # Same selection must not reset manual edits.
    assert widget.overlap.value == 0.65
    widget.seg_checkpoint.value = custom
    assert widget.overlap.value == 0.35
    widget.overlap.value = 0.4
    widget.seg_checkpoint.value = neural
    assert widget.overlap.value == 0.65
    widget.dataset.value = "gastruloid"
    widget.seg_checkpoint.value = gastruloid
    assert widget.overlap.value == 0.4


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("DARE3D_NATIVE_REPLAY") != "1",
    reason="Opt in with DARE3D_NATIVE_REPLAY=1; requires release assets and CUDA.",
)
def test_movie_i2_archived_probability_and_event_replay(tmp_path):
    """Run once in the chosen inference runtime; never benchmark intermediate models."""
    python = os.environ.get("DARE3D_NATIVE_REPLAY_PYTHON", sys.executable)
    output = tmp_path / "replay"
    log = tmp_path / "replay.log"
    command = [
        python, "-B", str(ROOT / "tests/helpers/neural_tube_native_replay.py"),
        "--output-dir", str(output),
    ]
    with log.open("x", encoding="utf-8") as stream:
        completed = subprocess.run(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            timeout=3600, check=False,
        )
    assert completed.returncode == 0, log.read_text(encoding="utf-8", errors="replace")[-8000:]
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["passed"], result["checks"]
