from pathlib import Path

import hydra
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from scripts.evaluate_neural_tube_nematic_hydra import (
    WAIVABLE_ASSERTIONS,
    release_decision,
)


def _compose_neural_experiment():
    GlobalHydra.instance().clear()
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    try:
        with initialize(version_base="1.3", config_path="../configs"):
            return compose(
                config_name="train.yaml",
                overrides=[
                    "experiment=neural_tube_nematic_regression",
                    "paths.root_dir=C:/repo",
                    "model_root=C:/candidate",
                    "split_root=C:/split",
                    "date=20260905T000000Z",
                ],
                return_hydra_config=True,
            )
    finally:
        GlobalHydra.instance().clear()


def test_neural_hydra_config_reconciles_legacy_domain_and_modern_protocol():
    cfg = _compose_neural_experiment()

    assert cfg.task_name == "regression3d_nematic_hydra_seed12345"
    assert cfg.seed == 12345
    assert cfg.steps_per_epoch == 2000
    assert cfg.data.batch_size == 32
    assert cfg.data.num_workers == 0
    assert cfg.model.net.n_stages == 3
    assert cfg.model.net.start_filters == 32
    assert cfg.model.regression_log_batches_per_epoch == 1
    assert cfg.model.optimizer.lr == pytest.approx(0.001)
    assert cfg.model.scheduler.steps_per_epoch == 63
    assert cfg.model.scheduler.epochs == 200
    assert cfg.trainer.max_epochs == 200
    assert cfg.trainer.deterministic is False
    assert cfg.trainer.precision == 32
    assert list(cfg.default_scale) == pytest.approx([0.208, 0.208, 1.0])
    assert cfg.target_scale == pytest.approx(1.0)
    assert list(cfg.input_channels) == [-1, 0, 1]
    assert cfg.crop_size == 32
    assert cfg.renorm == "min-max"
    assert cfg.representation_mode == "rotation_matrix_SVD"
    assert cfg.model.criterion.angle_loss._target_.endswith(
        ".nematic_axis_projector_loss"
    )
    assert cfg.callbacks.model_checkpoint.dirpath == "C:/candidate/checkpoints"
    assert cfg.paths.output_dir == "C:/candidate/runs/20260905T000000Z"
    assert cfg.hydra.run.dir == "C:/candidate"
    assert all(
        not cfg.data[name].require_scale_file
        for name in ("train_data", "val_data", "test_data")
    )

    model = hydra.utils.instantiate(cfg.model)
    assert sum(parameter.numel() for parameter in model.net.parameters()) == 1_605_268


def test_neural_candidate_is_distinct_from_all_existing_model_roles():
    weights = Path("DARE3d_data_190326/Neural_tube_160226/weights")
    candidate = weights / "regression3d_nematic_hydra_seed12345"
    legacy = weights / "regression3d_new_set_og"
    validated = Path(
        "docs/reproducibility_audit/neural_tube_nematic_full_pipeline/"
        "seed_12345/regression_nematic_retrained/checkpoints/epoch_139.ckpt"
    )

    assert candidate != legacy
    assert candidate not in validated.parents
def _passing_release_assertions():
    return {
        **{name: True for name in WAIVABLE_ASSERTIONS},
        "candidate_audit_0208_p95_within_five_degrees": True,
        "protected_assets_pass": True,
    }


def test_release_decision_reports_a_normal_pass_without_an_override():
    decision = release_decision(
        _passing_release_assertions(),
        {"audit_0208": 0.5, "physical_02076": 0.4},
    )

    assert decision["outcome"] == "normal_pass"
    assert decision["locked_criteria_passed"] is True
    assert decision["promotion_authorized"] is True
    assert decision["explicit_release_override_used"] is False


def test_release_decision_allows_only_explicit_three_degree_override():
    assertions = _passing_release_assertions()
    assertions["candidate_audit_0208_mean_within_one_degree"] = False
    assertions["audit_0208_bootstrap_upper_bound_below_two_degrees"] = False

    decision = release_decision(
        assertions,
        {"audit_0208": 3.0, "physical_02076": 2.9},
    )

    assert decision["outcome"] == "explicit_user_release_override"
    assert decision["locked_criteria_passed"] is False
    assert decision["promotion_authorized"] is True
    assert decision["explicit_release_override_used"] is True


@pytest.mark.parametrize(
    ("mean_deltas", "failed_name"),
    [
        ({"audit_0208": 3.0001, "physical_02076": 2.0}, None),
        (
            {"audit_0208": 2.0, "physical_02076": 2.0},
            "candidate_audit_0208_p95_within_five_degrees",
        ),
    ],
)
def test_release_decision_rejects_excess_delta_or_nonwaivable_failure(
    mean_deltas, failed_name
):
    assertions = _passing_release_assertions()
    assertions["candidate_audit_0208_mean_within_one_degree"] = False
    if failed_name is not None:
        assertions[failed_name] = False

    decision = release_decision(assertions, mean_deltas)

    assert decision["outcome"] == "rejected"
    assert decision["promotion_authorized"] is False
