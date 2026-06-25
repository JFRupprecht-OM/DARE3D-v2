import shutil
import os
from glob import glob
from skimage import io
from pathlib import Path

from dare3d.tools.generate_sparse_weights import compute_sparse_weights

def copy_to(folders_in, folder_out):
    for folder_in in folders_in:
        shutil.copytree(folder_in, folder_out, dirs_exist_ok=True)

def create_sparse_exp(trains, vals, sparse_trains, radius, output):
    create_exp(trains, vals, output)
    
    weights_folder = os.path.join(output, "train", "weights")
    os.makedirs(weights_folder, exist_ok=True)
    
    for sparse_train in sparse_trains:
        # Look for label
        label_folder = os.path.join(sparse_train, "label")
        labels_paths = glob(os.path.join(label_folder, "*.tif"))
        for label_path in labels_paths:
            label_name = Path(label_path).stem
            weight = compute_sparse_weights(label_path, radius)
            io.imsave(os.path.join(weights_folder, f"{label_name}.tif"), weight)

def create_exp(trains, vals, output):
    os.makedirs(output, exist_ok=True)

    train_folder = os.path.join(output, "train")
    copy_to(trains, train_folder)

    val_folder = os.path.join(output, "val")
    copy_to(vals, val_folder)
