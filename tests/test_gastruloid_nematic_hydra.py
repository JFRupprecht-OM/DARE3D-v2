from pathlib import Path

import hydra
import numpy as np
import pytest
import torch
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from dare3d.losses.angle3d import nematic_axis_projector_loss
from dare3d.utils.utils import extras


def _axis_matrix(axis, angle_degrees):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    matrix = Rotation.from_rotvec(
        axis * np.deg2rad(angle_degrees)
    ).as_matrix()
    return torch.tensor(matrix, dtype=torch.float32).reshape(1, 9)


def _compose_experiment(name):
    GlobalHydra.instance().clear()
    overrides = [
        f"experiment={name}",
        "paths.root_dir=C:/repo",
        "date=20260904T000000Z",
    ]
    if name == "gastruloid_nematic_regression":
        overrides.extend(
            [
                "model_root=C:/candidate",
                "split_root=C:/split",
            ]
        )
    try:
        with initialize(version_base="1.3", config_path="../configs"):
            return compose(
                config_name="train.yaml",
                overrides=overrides,
                return_hydra_config=True,
            )
    finally:
        GlobalHydra.instance().clear()


def test_nematic_projector_loss_is_axis_sign_and_roll_invariant():
    loss = nematic_axis_projector_loss()
    x = [1.0, 0.0, 0.0]
    y = [0.0, 1.0, 0.0]
    arbitrary = [0.2, 0.7, 0.4]

    assert float(loss(_axis_matrix(x, 60), _axis_matrix(x, 60))) == pytest.approx(0, abs=1e-4)
    assert float(loss(_axis_matrix(x, 60), _axis_matrix([-1, 0, 0], 60))) == pytest.approx(0, abs=1e-4)
    assert float(loss(_axis_matrix(x, 60), _axis_matrix(y, 60))) == pytest.approx(90, abs=1e-4)
    assert float(loss(_axis_matrix(arbitrary, 30), _axis_matrix(arbitrary, 150))) == pytest.approx(0, abs=1e-4)


def test_nematic_projector_loss_has_finite_random_gradients():
    generator = torch.Generator().manual_seed(12345)
    target_raw = torch.randn(12, 9, generator=generator)
    from dare3d.data.components.angles3d import symmetric_orthogonalization

    target = symmetric_orthogonalization(target_raw).detach().reshape(12, 9)
    prediction = torch.randn(12, 9, generator=generator, requires_grad=True)
    value = nematic_axis_projector_loss()(target, prediction)
    value.backward()

    assert torch.isfinite(value)
    assert torch.isfinite(prediction.grad).all()


def test_gastruloid_nematic_hydra_config_matches_validated_protocol():
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg = _compose_experiment("gastruloid_nematic_regression")
    HydraConfig().set_config(cfg)

    assert cfg.seed == 12345
    assert cfg.steps_per_epoch == 2000
    assert cfg.data.batch_size == 12
    assert cfg.data.num_workers == 0
    assert cfg.model.net.n_stages == 5
    assert cfg.model.net.start_filters == 16
    assert cfg.model.regression_log_batches_per_epoch == 1
    assert cfg.model.optimizer.lr == pytest.approx(0.001)
    assert cfg.model.scheduler.steps_per_epoch == 167
    assert cfg.model.scheduler.epochs == 100
    assert cfg.trainer.max_epochs == 100
    assert cfg.trainer.precision == 32
    assert cfg.model.criterion.angle_loss._target_.endswith(
        ".nematic_axis_projector_loss"
    )
    assert cfg.callbacks.model_checkpoint.dirpath == "C:/candidate/checkpoints"
    assert cfg.paths.output_dir == "C:/candidate/runs/20260904T000000Z"
    assert cfg.hydra.run.dir == "C:/candidate"
    assert cfg.logger.regression.log_freq == 20
    assert all(
        cfg.data[name].require_scale_file
        for name in ("train_data", "val_data", "test_data")
    )

    model = hydra.utils.instantiate(cfg.model)
    assert sum(parameter.numel() for parameter in model.net.parameters()) == 5_897_940


def test_default_regression_experiment_keeps_historical_loss():
    cfg = _compose_experiment("regression")
    assert cfg.model.criterion.angle_loss._target_.endswith(".svd_loss")


def test_candidate_and_legacy_directories_are_distinct():
    root = Path("DARE3d_data_190326/Gastruloid_241025/weights")
    assert root / "regression3d_nematic_hydra_seed12345" != root / "regression3d_exp10-b"


def test_extras_creates_the_declared_nested_output_directory(tmp_path):
    output_dir = tmp_path / "candidate" / "runs" / "run-id"
    cfg = OmegaConf.create(
        {
            "paths": {"output_dir": str(output_dir)},
            "extras": {
                "ignore_warnings": False,
                "enforce_tags": False,
                "print_config": False,
            },
        }
    )

    extras(cfg)

    assert output_dir.is_dir()
