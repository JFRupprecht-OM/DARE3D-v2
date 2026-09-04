"""Portable Napari release-model preset contracts."""

from pathlib import Path

import pytest

from napari_dare3d._release_models import (
    DATASET_CHOICES,
    RELEASE_ROOT_NAME,
    model_dir_from_checkpoint,
    release_movie_path,
    release_model_selection,
    release_segmentation_scale_mode,
)


@pytest.fixture
def release_root(tmp_path):
    root = tmp_path / RELEASE_ROOT_NAME
    root.mkdir()
    return root


def test_all_four_release_model_selections_are_explicit(release_root):
    expected = {
        ("gastruloid", "segmentation"): (
            "Gastruloid_241025/weights/segmentation3d_exp10-b",
            "DARE3D_gastruloid_segmentation_epoch067.ckpt",
        ),
        ("gastruloid", "regression"): (
            "Gastruloid_241025/weights/regression3d_exp10-b",
            "DARE3D_gastruloid_regression_epoch095.ckpt",
        ),
        ("neural_tube", "segmentation"): (
            "Neural_tube_160226/weights/segmentation3d_new_set_og/runs/12-01-26",
            "DARE3D_neural_tube_segmentation_epoch057.ckpt",
        ),
        ("neural_tube", "regression"): (
            "Neural_tube_160226/weights/regression3d_new_set_og/runs/12-01-26",
            "DARE3D_neural_tube_regression_epoch139.ckpt",
        ),
    }

    assert DATASET_CHOICES == ("gastruloid", "neural_tube")
    for selection, (model_rel, checkpoint_name) in expected.items():
        model_dir, checkpoint = release_model_selection(
            *selection, root=release_root
        )
        assert model_dir == release_root / Path(model_rel)
        assert checkpoint == release_root / checkpoint_name
        assert checkpoint.name != "last.ckpt"


def test_release_movies_are_dataset_specific(release_root):
    assert release_movie_path("gastruloid", release_root) == (
        release_root / "Gastruloid_241025/test_input/movie2.tif"
    )
    assert release_movie_path("neural_tube", release_root) == (
        release_root / "Neural_tube_160226/test_input/im/movie_M.tif"
    )


def test_release_checkpoint_infers_its_saved_config_directory(release_root):
    model_dir, checkpoint = release_model_selection(
        "neural_tube", "regression", release_root
    )
    (model_dir / ".hydra").mkdir(parents=True)
    (model_dir / ".hydra" / "config.yaml").touch()
    checkpoint.touch()

    assert model_dir_from_checkpoint(checkpoint, "regression") == model_dir


def test_conventional_checkpoint_infers_parent_model_directory(tmp_path):
    model_dir = tmp_path / "trained-model"
    checkpoint = model_dir / "checkpoints" / "epoch_012.ckpt"
    checkpoint.parent.mkdir(parents=True)
    (model_dir / ".hydra").mkdir()
    (model_dir / ".hydra" / "config.yaml").touch()
    checkpoint.touch()

    assert model_dir_from_checkpoint(checkpoint, "segmentation") == model_dir


@pytest.mark.parametrize("dataset", ["unknown", "movie2"])
def test_unknown_release_dataset_is_rejected(release_root, dataset):
    with pytest.raises(ValueError, match="Unknown release dataset"):
        release_model_selection(dataset, "segmentation", release_root)


@pytest.mark.parametrize(
    ("dataset", "expected"),
    (("gastruloid", "source"), ("neural_tube", "checkpoint_default")),
)
def test_promoted_release_segmentation_scale_mode_is_checkpoint_specific(
    release_root, dataset, expected
):
    _model_dir, checkpoint = release_model_selection(
        dataset, "segmentation", release_root
    )

    assert (
        release_segmentation_scale_mode(dataset, checkpoint, release_root)
        == expected
    )


def test_custom_checkpoint_does_not_inherit_neural_scale_exception(release_root):
    custom = (
        release_root
        / "custom"
        / "DARE3D_neural_tube_segmentation_epoch057.ckpt"
    )
    assert (
        release_segmentation_scale_mode("neural_tube", custom, release_root) == "source"
    )
