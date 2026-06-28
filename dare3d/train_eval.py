#!/bin/python
import click
import subprocess
import sys
from glob import glob
import os
from pathlib import Path


def _mlflow_tracking_uri():
    """Local SQLite MLflow store under the repo's ``logs/`` dir.

    Replaces the upstream hardcoded ``file:///C:/Users/<user>/...`` tracking_uri so
    this runs on any machine; SQLite is also the local store mlflow>=3 supports
    (mirrors ``napari_dare3d/_train.py``). ``eval.py`` re-reads it from the config.
    """
    root = Path(__file__).resolve().parents[1]  # repo root (holds dare3d/, logs/)
    db = root / "logs" / "mlflow.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    return "sqlite:///" + db.as_posix()


def segmentation(set, epoch, date, name, batch_size, cell_radius, eval_only, seg_crop_size, overwrite):
    ret_code = 0
    seg_ckpt_path = os.path.join(f"logs/segmentation3d_{name}/runs/{date}")
    
    exp_exists = os.path.exists(seg_ckpt_path)
    # We should not run if overwrite is false and the expirement already exists
    run = (overwrite or not exp_exists)
    
    if not eval_only and run:
        args = [
                sys.executable,
                "dare3d/train.py",
                "experiment=segmentation",
                f"task_name=segmentation3d_{name}",
                f"train_dir=3d/{set}/train",
                f"val_dir=3d/{set}/val",
                f"test_dir=3d/{set}/val",
                "trainer.accelerator=gpu",
                f"data.batch_size={batch_size}",
                "data.num_workers=0",
                "steps_per_epoch=1000",
                f"trainer.max_epochs={epoch}",
                "model/criterion=dice_focal",
                "model.optimizer.lr=0.1",
                "model/scheduler=one_cycle_lr",
                "model.scheduler_interval='step'",
                "renorm='min-max'",
                "time_axis_padding=1",
                f"cell_radius={cell_radius}",
                f"date={date}",
                f"crop_size={seg_crop_size}",
                f"logger.mlflow.tracking_uri={_mlflow_tracking_uri()}",
                ]
        print(args)
        ret_code = subprocess.call(args)
    
    
    return seg_ckpt_path, ret_code != 0

def regression(set, epoch, date, name, batch_size, eval_only, overwrite):
    ret_code = 0

    reg_ckpt_path = os.path.join(f"logs/regression3d_{name}/runs/{date}")
    exp_exists = os.path.exists(reg_ckpt_path)
    # We should not run if overwrite is false and the expirement already exists
    run = (overwrite or not exp_exists)

    if not eval_only and run:
        args = [
            sys.executable,
            "dare3d/train.py",
            f"experiment=regression",
            f"task_name=regression3d_{name}",
            f"train_dir=3d/{set}/train",
            f"val_dir=3d/{set}/val",
            f"trainer.max_epochs={epoch}",
            f"date={date}",
            "steps_per_epoch=1000",
            "model.optimizer.lr=0.001",
            f"data.batch_size={batch_size}",
            "model/net=simple_regression_net",
            "model.net.n_stages=3",
            "model.net.start_filters=32",
            f"logger.mlflow.tracking_uri={_mlflow_tracking_uri()}",
        ]
        
        print(args)
        ret_code = subprocess.call(args)

    return reg_ckpt_path, ret_code != 0

def find_best_model(folder, checkpoint_dir="checkpoints"):
    files = glob(os.path.join(folder, checkpoint_dir, "*.ckpt"))
    
    for file in files:
        name = Path(file).stem
        if "epoch" in name:
            return f"{name}.ckpt"
    return None

def evaluate(seg_path, reg_path, threshold=None):
    args = [
        sys.executable,
        "dare3d/eval.py",
        f"segmentation.model_dir={seg_path}",
        f"regression.model_dir={reg_path}",
        f"segmentation.threshold={'' if threshold is None else threshold}",
    ]
    # Look for best model epoch_{epoch}.ckpt
    best_seg_path = find_best_model(seg_path)
    if best_seg_path is not None:
        args.append(f"segmentation.ckpt_name={best_seg_path}")
    best_reg_path = find_best_model(reg_path)
    if best_reg_path is not None:
        args.append(f"regression.ckpt_name={best_reg_path}")
    
    print(args)
    subprocess.call(args)   

@click.command()
@click.option(
    "--set_folder",
    required=True,
)
@click.option(
    "--epoch",
    required=True,
)
@click.option(
    "--date",
    required=False,
    default="01-01"
)
@click.option(
    "--batch_size",
    required=False,
    default=32
)
@click.option(
    "--cell_radius",
    required=False,
    default=8
)
@click.option(
    "--eval_only",
    required=False,
    default=False
)
@click.option(
    "--overwrite",
    required=False,
    default=True,
    help="If this is True it will run the training even if it already exists. Set it to False to rerun existing experiments"
)
@click.option(
    "--seg_crop_size",
    required=False,
    default=128
)
@click.option(
    "--threshold",
    required=False,
    default=None,
    help="Segmentation threshold. Leave to None to look for the best threshold"
)
@click.option(
    "--train_segmentation",
    required=False,
    default=True
)
@click.option(
    "--train_regression",
    required=False,
    default=True
)
def main(set_folder, epoch, date, batch_size, cell_radius, eval_only, overwrite, seg_crop_size, threshold, train_segmentation, train_regression):
    exp_name = set_folder
    exp_name = exp_name.replace("/", "-")
    exp_name = exp_name.replace("\\", "-")

            
    train_segmentation = (not train_segmentation) or eval_only
    seg_path, fail = segmentation(set_folder, epoch, date, exp_name, batch_size, cell_radius, train_segmentation, seg_crop_size, overwrite)
    if fail:
        raise ValueError("Segmentation training failed...")
    train_regression = (not train_regression) or eval_only
    reg_path, fail = regression(set_folder, epoch, date, exp_name, batch_size, train_regression, overwrite)
    if fail:
        raise ValueError("Regression training failed...")

    evaluate(seg_path, reg_path, threshold)

if __name__ == "__main__":
    main()