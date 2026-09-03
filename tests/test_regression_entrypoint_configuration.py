"""Regression preprocessing configuration contracts at public entry points."""

import inspect
from pathlib import Path

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
