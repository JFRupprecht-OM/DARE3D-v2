"""Checkpoint compatibility tests for the inference and evaluation entry points."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

import dare3d.eval as eval_module
import dare3d.predict as predict_module
import napari_dare3d._api as api_module
from dare3d.metrics.infer_measure import CenterList


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
        self.preprocessing_mode = None

    def init(self, preprocess=False):
        self.initialized = not preprocess

    def init_inference(self, mode=None):
        self.initialized = True
        self.preprocessing_mode = mode

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
    assert dataset.preprocessing_mode == 'training_consistent'
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

    assert dataset.preprocessing_mode == 'training_consistent'

    assert load_calls == [(model.net, cfg.ckpt_path, "regression")]

@pytest.mark.parametrize('mode', ['training_consistent', 'legacy_raw'])
def test_napari_build_dataset_preserves_regression_mode(monkeypatch, mode):
    dataset = _Dataset()
    data_config = object()
    cfg = SimpleNamespace(data=SimpleNamespace(test_data=data_config))
    monkeypatch.setattr(
        api_module.hydra.utils,
        'instantiate',
        lambda config: dataset if config is data_config else None,
    )

    result = api_module._build_dataset(cfg, 'regression', mode)

    assert result is dataset
    assert dataset.preprocessing_mode == mode


def test_center_list_reuses_annotated_targets_across_movies():
    info = [
        {
            "matched_items": [(0, 0)],
            "pred_ccs_stats": {"centroids": [[2.0, 11.0, 20.0, 30.0]]},
            "true_ccs_stats": {"centroids": [[2.0, 10.0, 20.0, 30.0]]},
        },
        {
            "matched_items": [(0, 0)],
            "pred_ccs_stats": {"centroids": [[3.0, 41.0, 50.0, 60.0]]},
            "true_ccs_stats": {"centroids": [[3.0, 40.0, 50.0, 60.0]]},
        },
        {
            "matched_items": [],
            "pred_ccs_stats": {"centroids": []},
            "true_ccs_stats": {"centroids": [[4.0, 70.0, 80.0, 90.0]]},
        },
    ]
    centers = CenterList(0, info)

    assert centers.all_gt_event_keys == [(0, 0), (1, 0), (2, 0)]
    assert centers.matched_gt_event_keys == [(0, 0), (1, 0)]
    assert list(centers.matched_centers_idx) == [0, 1]
    assert len(centers.all_gt_centers) == 3
    assert len(centers.predicted_centers) == 2

    class Dataset:
        movie_names = ["first", "second", "third"]

        def __init__(self):
            self.calls = 0

        def gather_groundtruth_info(self, groundtruth_centers):
            self.calls += 1
            return [
                {"center": center, "target_number": index}
                for index, center in enumerate(groundtruth_centers)
            ]

    dataset = Dataset()
    centers.compute_real_rot_len_values(dataset)

    assert dataset.calls == 1
    assert centers.real_rot_length_matched[0] is centers.real_rot_length[0]
    assert centers.real_rot_length_matched[1] is centers.real_rot_length[1]
    assert (
        centers.real_rot_length_matched[0]["segmentation_event_id"]
        == "first:component:0"
    )
    predicted_pairs = centers.create_pred_gt_pairs(["pred-first", "pred-second"])
    assert predicted_pairs[0][0] is centers.real_rot_length[0]
    assert predicted_pairs[0][1] == "pred-first"


def test_center_list_uses_true_component_fallback_not_predicted_center():
    info = [
        {
            "matched_items": [(0, 0)],
            "pred_ccs_stats": {
                "centroids": [[7.0, 90.0, 205.0, 61.0]]
            },
            "true_ccs_stats": {
                "centroids": [[7.5, 87.0, 203.5, 59.5]]
            },
        }
    ]
    centers = CenterList(0, info)

    class Dataset:
        movie_names = ["movie"]

        def __init__(self):
            self.calls = []

        def gather_groundtruth_info(self, groundtruth_centers):
            self.calls.append(list(groundtruth_centers))
            return [
                (
                    None
                    if center[1:] == (8, 87, 204, 60)
                    else {"center": center, "event_id": "annotation"}
                )
                for center in groundtruth_centers
            ]

    dataset = Dataset()
    centers.compute_real_rot_len_values(dataset)

    assert dataset.calls == [
        [(0, 8, 87, 204, 60)],
        [(0, 7.5, 87.0, 203.5, 59.5)],
    ]
    assert centers.real_rot_length == [None]
    assert centers.real_rot_length_matched[0]["event_id"] == "annotation"
    assert (
        centers.real_rot_length_matched[0]["segmentation_event_id"]
        == "movie:component:0"
    )
    predicted_pairs = centers.create_pred_gt_pairs(["prediction"])
    assert predicted_pairs == [
        (centers.real_rot_length_matched[0], "prediction")
    ]
