"""Regression preprocessing configuration contracts at public entry points."""

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

import dare3d.eval as eval_module
import dare3d.predict as predict_module
import napari_dare3d._api as api_module


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _saved_model_dir(tmp_path):
    hydra_dir = tmp_path / ".hydra"
    checkpoint_dir = tmp_path / "checkpoints"
    hydra_dir.mkdir(parents=True)
    checkpoint_dir.mkdir()
    OmegaConf.save(
        OmegaConf.create(
            {
                "data": {
                    "test_data": {
                        "_target_": "tests.FakeDataset",
                        "im_folder": "saved-images",
                        "label_folder": "saved-labels",
                        "scale_file": "C:/old-workstation/scales.json",
                        "default_scale": [0.5, 1.0, 2.0],
                        "target_scale": 1.0,
                        "load_labels": True,
                    }
                },
                "model": {"_target_": "tests.FakeModel"},
            }
        ),
        hydra_dir / "config.yaml",
    )
    (checkpoint_dir / "last.ckpt").touch()
    return tmp_path


def _stage_config(model_dir, *, mode="training_consistent", strict=False):
    return OmegaConf.create(
        {
            "model_dir": str(model_dir),
            "hydra_dir": ".hydra",
            "ckpt_dir": "checkpoints",
            "ckpt_name": "last.ckpt",
            "preprocessing_mode": mode,
            "require_scale_file": strict,
        }
    )


def _predict_root(tmp_path, **overrides):
    values = {
        "inference_dir": str(tmp_path / "input"),
        "device": "cpu",
        "scale_file": None,
        "default_scale": None,
        "target_scale": None,
    }
    values.update(overrides)
    return OmegaConf.create(values)


def test_shipped_entrypoint_defaults_are_training_consistent():
    predict_cfg = OmegaConf.load(REPOSITORY_ROOT / "configs" / "predict.yaml")
    eval_cfg = OmegaConf.load(REPOSITORY_ROOT / "configs" / "eval.yaml")
    dataset_cfg = OmegaConf.load(
        REPOSITORY_ROOT / "configs" / "data" / "dataset" / "regression.yaml"
    )

    assert predict_cfg.regression.preprocessing_mode == "training_consistent"
    assert eval_cfg.regression.preprocessing_mode == "training_consistent"
    assert predict_cfg.regression.require_scale_file is False
    assert eval_cfg.regression.require_scale_file is False
    assert dataset_cfg.require_scale_file is False
    api_signature = inspect.signature(api_module.infer_stack)
    assert (
        api_signature.parameters["regression_preprocessing"].default
        == "training_consistent"
    )
    assert api_signature.parameters["regression_require_scale_file"].default is False
    assert (
        api_signature.parameters["segmentation_scale_mode"].default == "source"
    )
    assert api_signature.parameters["seg_checkpoint"].default is None
    assert api_signature.parameters["reg_checkpoint"].default is None
    assert api_signature.parameters["seg_model_dir"].default is None


def test_predict_discards_saved_scale_path_and_uses_saved_default(tmp_path):
    model_dir = _saved_model_dir(tmp_path / "model")
    stage_cfg = _stage_config(model_dir)

    loaded = predict_module.load_config(stage_cfg, _predict_root(tmp_path))

    test_data = loaded.data.test_data
    assert test_data.scale_file is None
    assert list(test_data.default_scale) == [0.5, 1.0, 2.0]
    assert test_data.target_scale == 1.0
    assert test_data.require_scale_file is False
    assert test_data.load_labels is False
    assert loaded.preprocessing_mode == "training_consistent"


def test_predict_applies_explicit_scale_contract_and_legacy_replay(tmp_path):
    model_dir = _saved_model_dir(tmp_path / "model")
    scale_file = tmp_path / "portable-scales.json"
    stage_cfg = _stage_config(model_dir, mode="legacy_raw", strict=True)
    root_cfg = _predict_root(
        tmp_path,
        scale_file=str(scale_file),
        default_scale=[0.2, 0.3, 0.4],
        target_scale=0.5,
    )

    loaded = predict_module.load_config(stage_cfg, root_cfg)

    test_data = loaded.data.test_data
    assert test_data.scale_file == str(scale_file)
    assert list(test_data.default_scale) == [0.2, 0.3, 0.4]
    assert test_data.target_scale == 0.5
    assert test_data.require_scale_file is True
    assert loaded.preprocessing_mode == "legacy_raw"


def test_eval_uses_only_explicit_scale_overrides(tmp_path):
    model_dir = _saved_model_dir(tmp_path / "model")
    stage_cfg = _stage_config(model_dir, strict=True)
    stage_cfg.scale_file_override = "portable/scales.json"
    stage_cfg.default_scale_override = [0.2, 0.3, 0.4]
    stage_cfg.target_scale_override = 0.5

    loaded = eval_module.load_config(stage_cfg)

    test_data = loaded.data.test_data
    assert test_data.scale_file == "portable/scales.json"
    assert list(test_data.default_scale) == [0.2, 0.3, 0.4]
    assert test_data.target_scale == 0.5
    assert test_data.require_scale_file is True
    assert loaded.preprocessing_mode == "training_consistent"


def test_eval_null_override_removes_saved_workstation_scale_path(tmp_path):
    model_dir = _saved_model_dir(tmp_path / "model")
    stage_cfg = _stage_config(model_dir)
    stage_cfg.scale_file_override = None
    stage_cfg.default_scale_override = None
    stage_cfg.target_scale_override = None

    loaded = eval_module.load_config(stage_cfg)

    assert loaded.data.test_data.scale_file is None
    assert list(loaded.data.test_data.default_scale) == [0.5, 1.0, 2.0]


def test_napari_scale_strictness_is_regression_only_and_explicit(tmp_path):
    model_dir = _saved_model_dir(tmp_path / "model")
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    segmentation_cfg = api_module._load_inference_cfg(
        str(model_dir),
        str(image_dir),
        "cpu",
    )
    regression_cfg = api_module._load_inference_cfg(
        str(model_dir),
        str(image_dir),
        "cpu",
        require_scale_file=True,
    )

    assert "require_scale_file" not in segmentation_cfg.data.test_data
    assert regression_cfg.data.test_data.require_scale_file is True
    assert Path(regression_cfg.data.test_data.scale_file).name == "__no_scales__.json"


def test_napari_accepts_an_explicit_checkpoint_outside_model_dir(tmp_path):
    model_dir = _saved_model_dir(tmp_path / "model")
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    promoted = tmp_path / "DARE3D_promoted.ckpt"
    promoted.touch()

    loaded = api_module._load_inference_cfg(
        str(model_dir),
        str(image_dir),
        "cpu",
        checkpoint_path=str(promoted),
    )

    assert Path(loaded.ckpt_path) == promoted


def test_napari_infers_both_model_dirs_from_checkpoint_files(monkeypatch):
    observed = []

    monkeypatch.setattr(
        api_module,
        "model_dir_from_checkpoint",
        lambda _checkpoint, stage: Path(f"resolved-{stage}"),
    )

    def fake_load(model_dir, _image_dir, _device, **kwargs):
        observed.append((model_dir, kwargs["checkpoint_path"]))
        return SimpleNamespace()

    monkeypatch.setattr(api_module, "_load_inference_cfg", fake_load)
    monkeypatch.setattr(
        api_module, "_segment", lambda *args, **kwargs: [(0, 0, 1, 1, 1)]
    )
    monkeypatch.setattr(api_module, "_regress", lambda *args, **kwargs: [None])

    result = api_module.infer_stack(
        np.zeros((1, 2, 2, 2), dtype=np.uint8),
        seg_checkpoint="segmentation.ckpt",
        reg_checkpoint="regression.ckpt",
        device="cpu",
    )

    assert result == []
    assert observed == [
        ("resolved-segmentation", "segmentation.ckpt"),
        ("resolved-regression", "regression.ckpt"),
    ]


def test_napari_checkpoint_default_routes_physical_scale_only_to_regression(
    monkeypatch,
):
    loaded = []
    regression_modes = []

    def fake_load(model_dir, _image_dir, _device, **kwargs):
        loaded.append((model_dir, kwargs))
        return SimpleNamespace(model_dir=model_dir)

    centers = [(0, 0, 1, 1, 1), (0, 0, 2, 2, 2)]

    def fake_regress(
        _cfg, _device, regression_centers, *, preprocessing_mode, **_kwargs
    ):
        regression_modes.append(preprocessing_mode)
        return [
            {
                "center": center,
                "length": 4.0,
                "rotation": np.asarray((1.0, 1.0, 0.0, 0.0)),
            }
            for center in regression_centers
        ]

    monkeypatch.setattr(api_module, "_load_inference_cfg", fake_load)
    monkeypatch.setattr(api_module, "_segment", lambda *args, **kwargs: centers)
    monkeypatch.setattr(api_module, "_regress", fake_regress)

    detections = api_module.infer_stack(
        np.zeros((1, 3, 3, 3), dtype=np.uint8),
        "segmentation-model",
        "regression-model",
        device="cpu",
        scale_file="physical-scales.json",
        default_scale=[0.2076, 0.2076, 1.0],
        target_scale=1.0,
        segmentation_scale_mode="checkpoint_default",
    )

    assert len(detections) == len(centers) == 2
    assert regression_modes == ["training_consistent"]
    segmentation_kwargs = loaded[0][1]
    regression_kwargs = loaded[1][1]
    assert segmentation_kwargs == {"checkpoint_path": None}
    assert regression_kwargs["scale_file"] == "physical-scales.json"
    assert regression_kwargs["default_scale"] == [0.2076, 0.2076, 1.0]
    assert regression_kwargs["target_scale"] == 1.0


def test_napari_zero_segmentation_centers_skip_regression(monkeypatch):
    loaded_model_dirs = []

    def fake_load(model_dir, _image_dir, _device, **_kwargs):
        loaded_model_dirs.append(model_dir)
        return SimpleNamespace()

    def regression_must_not_run(*_args, **_kwargs):
        raise AssertionError("regression ran without segmentation centers")

    monkeypatch.setattr(api_module, "_load_inference_cfg", fake_load)
    monkeypatch.setattr(api_module, "_segment", lambda *args, **kwargs: [])
    monkeypatch.setattr(api_module, "_regress", regression_must_not_run)

    assert api_module.infer_stack(
        np.zeros((1, 2, 2, 2), dtype=np.uint8),
        "segmentation-model",
        "regression-model",
        device="cpu",
    ) == []
    assert loaded_model_dirs == ["segmentation-model"]


def test_napari_rejects_unknown_segmentation_scale_mode():
    with np.testing.assert_raises_regex(
        ValueError, "segmentation_scale_mode must be 'source' or 'checkpoint_default'"
    ):
        api_module.infer_stack(
            np.zeros((1, 2, 2, 2), dtype=np.uint8),
            "segmentation-model",
            device="cpu",
            segmentation_scale_mode="legacy_guess",
        )


def test_napari_checkpoint_config_retains_saved_scale_without_overrides(tmp_path):
    model_dir = _saved_model_dir(tmp_path / "model")
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    loaded = api_module._load_inference_cfg(str(model_dir), str(image_dir), "cpu")

    assert list(loaded.data.test_data.default_scale) == [0.5, 1.0, 2.0]
    assert loaded.data.test_data.target_scale == 1.0
