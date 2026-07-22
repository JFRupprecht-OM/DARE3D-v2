"""Checkpoint compatibility tests for the inference and evaluation entry points."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

import dare3d.eval as eval_module
import dare3d.predict as predict_module
import napari_dare3d._api as api_module


def _checkpoint(tmp_path, net):
    state_dict = {f"net.{key}": value.clone() for key, value in net.state_dict().items()}
    state_dict["criterion.focal.class_weight"] = torch.ones(1)
    state_dict["train_loss.total"] = torch.tensor(3.0)
    path = tmp_path / "model.ckpt"
    torch.save({"state_dict": state_dict}, path)
    return str(path)


def _assert_same_state(expected, actual):
    for key, value in expected.state_dict().items():
        assert torch.equal(value, actual.state_dict()[key])


class _Dataset:
    renorm = False

    def __init__(self):
        self.initialized = False

    def init(self, preprocess=False):
        self.initialized = not preprocess

    def make_masks(self):
        pass

    def pad_images(self):
        pass

    def _normalize(self, renorm):
        assert renorm is self.renorm


def test_napari_build_model_ignores_non_network_buffers(tmp_path, monkeypatch):
    expected = nn.Linear(3, 2)
    model = SimpleNamespace(net=nn.Linear(3, 2))
    cfg = SimpleNamespace(model=object(), ckpt_path=_checkpoint(tmp_path, expected))
    monkeypatch.setattr(api_module.hydra.utils, "instantiate", lambda _: model)

    loaded = api_module._build_model(cfg, torch.device("cpu"), stage="segmentation")

    assert loaded is model
    assert not model.net.training
    _assert_same_state(expected, model.net)


def test_predict_load_data_ignores_non_network_buffers(tmp_path, monkeypatch):
    expected = nn.Linear(3, 2)
    model = SimpleNamespace(net=nn.Linear(3, 2))
    dataset = _Dataset()
    model_config, data_config = object(), object()
    cfg = SimpleNamespace(
        model=model_config,
        data=SimpleNamespace(test_data=data_config),
        ckpt_path=_checkpoint(tmp_path, expected),
        device="cpu",
    )

    def instantiate(config):
        return model if config is model_config else dataset

    monkeypatch.setattr(predict_module.hydra.utils, "instantiate", instantiate)
    monkeypatch.setattr(
        predict_module.hydra.core.hydra_config.HydraConfig,
        "get",
        classmethod(lambda _: {"runtime": {"output_dir": str(tmp_path)}}),
    )

    loaded_dataset, loaded_model, device, output_dir = predict_module.load_data(
        cfg, stage="regression"
    )

    assert loaded_dataset is dataset
    assert loaded_model is model
    assert dataset.initialized
    assert device == torch.device("cpu")
    assert output_dir == str(tmp_path)
    _assert_same_state(expected, model.net)


class _Trainer:
    def __init__(self):
        self.callback_metrics = {}
        self.test_calls = []

    def test(self, **kwargs):
        self.test_calls.append(kwargs)


def _segmentation_eval_cfg(tmp_path, lightning_test):
    return OmegaConf.create(
        {
            "ckpt_path": "model.ckpt",
            "model_dir": str(tmp_path),
            "model": {"_target_": "tests.Model"},
            "data": {"_target_": "tests.Data", "batch_size": 1},
            "trainer": {"_target_": "tests.Trainer"},
            "logger": None,
            "device": "cpu",
            "lightning_test": lightning_test,
            "crop_size": [4, 4, 4],
            "multithread": False,
            "threshold": 0.5,
            "iteration_method": "movie",
            "distance_mode": "iou",
            "distance_threshold": 1e-6,
            "min_weighted_prob": 0.1,
        }
    )


@pytest.mark.parametrize("lightning_test", [False, True])
def test_segmentation_evaluation_uses_preloaded_network(
    tmp_path, monkeypatch, lightning_test
):
    cfg = _segmentation_eval_cfg(tmp_path, lightning_test)
    dataset = _Dataset()
    datamodule = SimpleNamespace(data_test=dataset)
    model = SimpleNamespace(net=nn.Linear(3, 2))
    trainer = _Trainer()
    load_calls = []

    def instantiate(config, **kwargs):
        target = config.get("_target_")
        return {
            "tests.Data": datamodule,
            "tests.Model": model,
            "tests.Trainer": trainer,
        }[target]

    monkeypatch.setattr(eval_module.hydra.utils, "instantiate", instantiate)
    monkeypatch.setattr(eval_module, "instantiate_loggers", lambda _: [])
    monkeypatch.setattr(
        eval_module,
        "load_net_state_dict",
        lambda net, path, *, stage: load_calls.append((net, path, stage)),
    )
    monkeypatch.setattr(
        eval_module,
        "infer_and_evaluate_segmentation",
        lambda *args, **kwargs: ({}, {}),
    )

    eval_module.evaluate_segmentation(cfg)

    assert load_calls == [(model.net, cfg.ckpt_path, "segmentation")]
    assert len(trainer.test_calls) == int(lightning_test)
    if lightning_test:
        assert trainer.test_calls[0]["ckpt_path"] is None


def test_regression_evaluation_loads_only_network(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {
            "ckpt_path": "model.ckpt",
            "model_dir": str(tmp_path),
            "model": {"_target_": "tests.Model"},
            "data": {"_target_": "tests.Data"},
            "logger": None,
            "device": "cpu",
        }
    )
    dataset = _Dataset()
    datamodule = SimpleNamespace(data_test=dataset)
    model = SimpleNamespace(net=nn.Linear(3, 2))
    load_calls = []

    def instantiate(config, **kwargs):
        return datamodule if config.get("_target_") == "tests.Data" else model

    monkeypatch.setattr(eval_module.hydra.utils, "instantiate", instantiate)
    monkeypatch.setattr(eval_module, "instantiate_loggers", lambda _: [])
    monkeypatch.setattr(
        eval_module,
        "load_net_state_dict",
        lambda net, path, *, stage: load_calls.append((net, path, stage)),
    )
    monkeypatch.setattr(eval_module, "infer_and_evaluate_regression", lambda **kwargs: {})

    eval_module.evaluate_regression(cfg, info={})

    assert load_calls == [(model.net, cfg.ckpt_path, "regression")]
