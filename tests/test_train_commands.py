"""Self-check for napari_dare3d._train — command construction + paths (no GPU, no run).

Run:  python tests/test_train_commands.py     # standalone
      pytest tests/test_train_commands.py     # collected with the suite
"""
import sys
from pathlib import Path

from napari_dare3d import _train


def test_repo_and_scripts_resolve():
    root = _train.repo_root()
    assert (root / "dare3d" / "train.py").is_file(), root
    assert (root / "dare3d" / "eval.py").is_file(), root
    assert (root / "configs" / "train.yaml").is_file(), root


def test_seg_command():
    cmd = _train.seg_command(
        dataset_dir="C:/data/myset", output_dir="C:/out", name="myset",
        date="2026-06-20", epochs=50, batch_size=4, cell_radius=8, seg_crop_size=128,
    )
    assert cmd[0] == sys.executable
    assert Path(cmd[1]).name == "train.py"
    joined = " ".join(cmd)
    for token in [
        "experiment=segmentation", "task_name=segmentation3d_myset",
        "trainer.accelerator=gpu", "trainer.max_epochs=50", "data.batch_size=4",
        "data.num_workers=0", "cell_radius=8", "crop_size=128", "date=2026-06-20",
        "model/criterion=dice_focal", "paths.log_dir=C:/out",
        "data.train_data.im_folder=C:/data/myset/train/im",
        "data.val_data.label_folder=C:/data/myset/val/label",
        "data.test_data.im_folder=C:/data/myset/val/im",
    ]:
        assert token in joined, token
    # mlflow override present and LOCAL sqlite (never the upstream hardcoded path)
    assert "logger.mlflow.tracking_uri=sqlite:///C:/out/mlflow.db" in joined
    assert "tlili" not in joined


def test_reg_command():
    cmd = _train.reg_command("C:/data/myset", "C:/out", "myset", "2026-06-20", 50, 4)
    joined = " ".join(cmd)
    for token in [
        "experiment=regression", "task_name=regression3d_myset",
        "model/net=simple_regression_net", "model.optimizer.lr=0.001",
        "trainer.accelerator=gpu", "trainer.max_epochs=50",
        "data.train_data.im_folder=C:/data/myset/train/im",
    ]:
        assert token in joined, token
    assert "tlili" not in joined


def test_eval_command_and_model_dirs():
    dirs = _train.model_dirs("C:/out", "myset", "2026-06-20")
    assert dirs["segmentation"] == Path("C:/out/segmentation3d_myset/runs/2026-06-20")
    assert dirs["regression"] == Path("C:/out/regression3d_myset/runs/2026-06-20")
    cmd = _train.eval_command(dirs["segmentation"], dirs["regression"], threshold=0.5)
    joined = " ".join(cmd)
    assert Path(cmd[1]).name == "eval.py"
    assert "segmentation.model_dir=" in joined and "regression.model_dir=" in joined
    assert "segmentation.threshold=0.5" in joined
    # threshold omitted when None -> eval.yaml default
    assert "threshold" not in " ".join(_train.eval_command(dirs["segmentation"], dirs["regression"]))


def test_split_resolution():
    import os
    import tempfile

    # raw per-movie layout (dirs only; available_movies needs im/ + label/)
    ds = tempfile.mkdtemp(prefix="ds_")
    for mv in ("movieA", "movieB", "movieC"):
        for kind in ("im", "label"):
            os.makedirs(os.path.join(ds, mv, kind))
    assert _train.available_movies(ds) == ["movieA", "movieB", "movieC"]

    out = tempfile.mkdtemp(prefix="out_")
    # space in path -> ValueError (before any fs work)
    try:
        _train._resolve_split("C:/my data/x", out, ["movieA"], ["movieB"]); raise AssertionError
    except ValueError:
        pass
    # neither pre-split nor per-movie -> FileNotFoundError
    try:
        _train._resolve_split(tempfile.mkdtemp(prefix="empty_"), out, ["a"], ["b"]); raise AssertionError
    except FileNotFoundError:
        pass
    # per-movie but no movies chosen -> ValueError
    try:
        _train._resolve_split(ds, out, [], []); raise AssertionError
    except ValueError:
        pass
    # chosen movie missing -> FileNotFoundError
    try:
        _train._resolve_split(ds, out, ["movieA"], ["movieZ"]); raise AssertionError
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    test_repo_and_scripts_resolve()
    test_seg_command()
    test_reg_command()
    test_eval_command_and_model_dirs()
    test_split_resolution()
    print("test_train_commands: ALL OK")
