import logging

import torch
from tqdm import tqdm
from skimage import io
import numpy as np

from dare3d.data.components.abstract_celldataset import AbstractCellDataset

log = logging.getLogger(__name__)


class Cell3Dataset(AbstractCellDataset):
    def __init__(
        self,
        order_dim_img="zyx",
        **kwargs
    ):
        self.order_dim = order_dim_img
        super(Cell3Dataset, self).__init__(**kwargs)

    def compute_window_indices(self, frame_size, size=None, overlap=0.):
        from monai.inferers.utils import _get_scan_interval
        from monai.data.utils import dense_patch_slices
        scan_interval = _get_scan_interval(frame_size, size, 3, [overlap, overlap, overlap])
        slices = dense_patch_slices(frame_size, size, scan_interval, return_slice=True)
        return slices


    def _normalize(self, norm_type):
        log.info(f"Normalizing all movies using renorm={norm_type}")
        for idx, movie in tqdm(enumerate(self.movies_im), total=len(self.movies_im)):
            self.movies_im[idx] = self.normalise(movie, norm_type)
        movie = self.movies_im[0]
        for movie in self.movies_im:
            log.debug(
                f"Normalized movie to shape={movie.shape}, dtype={movie.dtype} min-max: {movie.min()}-{movie.max()}"
            )

    def normalize_percentile(self, movie, min_per=0.5, max_per=99.5, percentile_method=np.percentile, clip_method="numpy"):
        x = movie
        percentile_low = percentile_method(x, min_per)
        percentile_high = percentile_method(x, max_per)
        x = (x - percentile_low) / (percentile_high - percentile_low + 1e-6)
        if clip_method == "torch":
            x = torch.clip(x, min=0, max=percentile_method(x, 99.9))
        else:
            x = np.clip(x, a_min=0, a_max=percentile_method(x, 99.9))
        x = x / x.max()
        movie = x
        return movie
    
    def percentile_torch(self, t: torch.tensor, q: float):
        """
        Return the ``q``-th percentile of the flattened input tensor's data.
        
        CAUTION:
        * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
        * Values are not interpolated, which corresponds to
        ``numpy.percentile(..., interpolation="nearest")``.
        
        :param t: Input tensor.
        :param q: Percentile to compute, which must be between 0 and 100 inclusive.
        :return: Resulting value (scalar).
        """
        # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
        # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
        # so that ``round()`` returns an integer, even if q is a np.float32.
        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        result = t.view(-1).kthvalue(k).values.item()
        return result
    
    def normalise_gpu(self, x, renorm=None):
        if renorm == "percentile":
            return self.normalize_percentile(x, percentile_method=self.percentile_torch, clip_method="torch")
        elif renorm == "min-max":
            return (x-x.min()) / (x.max() - x.min())
        else:
            raise NotImplementedError
    
    def normalise(self, x, renorm=None):
        if renorm == "min-max":
            res = (x - x.min()) / x.ptp()
            return res.astype(np.float32)
        elif renorm == "percentile":
            return self.normalize_percentile(x).astype(np.float32)
        elif renorm == "min-max-z":
            res = x
            # For each frame in the movie
            for i in range(x.shape[0]):
                # Normalize all Z axis
                for z_idx in range(x[i].shape[-1]):
                    res[i, ..., z_idx] = self.normalise((res[i, ..., z_idx]), "min-max")
            return res
        else:
            return x

    def read_tif_and_order_xyz(self, filename, order_dim="zyx"):
        # Read multitif of shape (T, Z, X, Y)
        img_array = io.imread(filename)
        if img_array.dtype == np.float64:
            img_array = img_array.astype(np.float16)
        if order_dim == "zyx":
            img_array = img_array.swapaxes(1, -1)  # (100x, 100y, 36z)
        elif order_dim == "zxy":
            img_array = np.moveaxis(img_array, 1, -1)
        else:
            raise NotImplementedError
        return img_array

    def _load_image(self, path):
        im = self.read_tif_and_order_xyz(path, self.order_dim)
        return im
