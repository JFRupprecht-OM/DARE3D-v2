import os

import numpy as np
from skimage import io
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only
import torch
from glob import glob
from pathlib import Path

from dare3d.data.components.angles3d import (
    ANGLE_REPRESENTATION,
    draw_line_in_matrix,
    get_points_from_quat,
    representation_to_quaternion,
)


class RegressionLogger(Logger):
    def __init__(self, save_dir, log_freq=1, representation=None):
        self._save_dir = save_dir
        self._log_freq = log_freq
        self.representation = ANGLE_REPRESENTATION.from_string(representation)
    
    @property
    def name(self):
        return "RegressionLogger"

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        pass

    @rank_zero_only
    def get_max_index(self, epoch):
        cdir = os.path.join(self._save_dir, str(epoch))
        if not os.path.exists(cdir):
            os.makedirs(cdir)
            return -1
        
        files = glob(os.path.join(cdir, "im*.tif"))
        # im1.tif -> im1 -> ["im", "1"] -> "1" -> 1
        indices = [int(Path(file).stem.split("_scale")[0].split("im")[-1]) for file in files]
        if len(indices) == 0:
            return -1
        return max(indices)

    @rank_zero_only
    def log_3D_images(self, x, head_true, head_pred, epoch):
        if epoch % self._log_freq != 0:
            return

        cdir = os.path.join(self._save_dir, str(epoch))
        if not os.path.exists(cdir):
            os.makedirs(cdir, exist_ok=True)
        # current_max_index = self.get_max_index(epoch)
        current_max_index = 0

        x = x.detach().cpu().numpy()
        
        if len(head_true["head2"]) == 2:
            raise NotImplementedError()
        else:
            head_true = head_true["head1"]
            head_pred = head_pred["head1"]
            for j in range(x.shape[0]):
                label = np.zeros_like(x[j])
                pred = np.zeros_like(x[j])
                bipoints = np.zeros_like(x[j])

                # Display true division orientation and length
                io.imsave(os.path.join(cdir, f"im_{current_max_index}.tif"), x[j], check_contrast=False)
                self._display(head_true, label, cdir, f"label_{current_max_index}.tif", j)
                self._display(head_pred, pred, cdir, f"pred_{current_max_index}.tif", j, is_pred=True)
                
                # Display bipoint for confirmation
                bp = head_true["bipoint"][j].detach().cpu()
                for k in range(bipoints.shape[0]):
                    bipoints[k, bp[0][0], bp[0][1], bp[0][2]] = 255
                    bipoints[k, bp[1][0], bp[1][1], bp[1][2]] = 255
                io.imsave(os.path.join(cdir, f"bipoints_{current_max_index}.tif"), bipoints, check_contrast=False)
                
                current_max_index += 1

    def _display(self, head, mat, cdir, name, idx, is_pred=False):
        length = head["len"][idx].detach().cpu()
        rot = head["angle"][idx].detach().cpu()
        for k in range(mat.shape[0]):
            mat[k] = self.display_label(length, rot, mat[k], is_pred)
        io.imsave(os.path.join(cdir, name), mat, check_contrast=False)

    def display_label(self, length, rot, matrix, is_pred):
        center = (matrix.shape[0] // 2, matrix.shape[1] // 2, matrix.shape[2] // 2)
        quat = representation_to_quaternion(rot, self.representation, post=is_pred)
        real_length = length * np.min(matrix.shape)
        p1, p2 = get_points_from_quat(quat, center, 0.5 * real_length.numpy())
        matrix = draw_line_in_matrix(matrix, p1, p2, matrix.shape[0], val=255.0)
        return matrix