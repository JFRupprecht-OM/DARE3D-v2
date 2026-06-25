import os
from pathlib import Path
from glob import glob
import numpy as np
from skimage import io
from skimage.transform import rescale
from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning.pytorch.utilities import rank_zero_only
import torch

class SegmentationLogger(Logger):
    def __init__(self, save_dir, log_freq=1):
        self._save_dir = save_dir
        self._log_freq = log_freq
    
    @property
    def name(self):
        return "SegmentationLogger"

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
    def log_3D_images(self, x, y_true, y_pred, epoch):
        if epoch % self._log_freq != 0:
            return

        cdir = os.path.join(self._save_dir, str(epoch))
        # current_max_index = self.get_max_index(epoch)
        current_max_index = 0

        x = x.detach().cpu().numpy()

        # for each scale
        for i in range(len(y_true)):
            c_y_true = y_true[i]
            c_y_pred = y_pred[i]
            c_y_pred = torch.sigmoid(c_y_pred)
            upscale_factor = 2**i

            # for each element in batch
            for j in range(x.shape[0]):
                cj_y_true = torch.unsqueeze(c_y_true[j], dim=0)
                cj_y_pred = torch.unsqueeze(c_y_pred[j], dim=0)

                if upscale_factor > 1:
                    cj_y_true = torch.nn.functional.interpolate(cj_y_true, scale_factor=upscale_factor, mode="trilinear")
                    cj_y_pred = torch.nn.functional.interpolate(cj_y_pred, scale_factor=upscale_factor, mode="trilinear")

                cj_y_true = cj_y_true.detach().cpu().numpy()
                cj_y_pred = cj_y_pred.detach().cpu().numpy()
                    
                current_index = j + current_max_index + 1
                if i == 0:
                    io.imsave(os.path.join(cdir, f"im{current_index}_scale{str(i)}.tif"), x[j].astype(np.float16), check_contrast=False)
                io.imsave(os.path.join(cdir, f"label{current_index}_scale{str(i)}.tif"), cj_y_true.astype(np.float16), check_contrast=False)
                io.imsave(os.path.join(cdir, f"pred{current_index}_scale{str(i)}.tif"), cj_y_pred.astype(np.float16), check_contrast=False)