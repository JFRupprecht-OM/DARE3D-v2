from typing import Tuple
import numpy as np
import scipy.ndimage as ndi
import torch

def multiscale_decimate_gpu(y: torch.Tensor, decimate: Tuple[int, int]=(1, 2, 2, 2), sigma: float=1.0) -> np.ndarray:
    """Decimate an image by a factor of `decimate` and apply a Gaussian filter with standard deviation `sigma`

    Args:
        y (np.ndarray): Image to be decimated
        decimate (Tuple[int, int], optional): downsampling factor. Defaults to (4, 4).
        sigma (float, optional): standard deviation of the Gaussian filter. Defaults to 1.

    Returns:
        np.ndarray: Decimated image
    """
    if decimate == (1, 1, 1, 1):
        return y
    assert y.ndim == len(decimate)

    # from skimage.measure import block_reduce

    y = torch.nn.functional.max_pool3d(y, decimate[1:])
    # y = block_reduce(y, decimate, np.max)

    # y = 2 * np.pi * sigma**2 * ndi.gaussian_filter(y, sigma)

    y = torch.clip(y, 0, 1)
    # y = np.clip(y, 0, 1)
    # return y.astype(np.float32)
    return y

def multiscale_decimate(y: np.ndarray, decimate: Tuple[int, int]=(1, 2, 2, 2), sigma: float=1.0) -> np.ndarray:
    """Decimate an image by a factor of `decimate` and apply a Gaussian filter with standard deviation `sigma`

    Args:
        y (np.ndarray): Image to be decimated
        decimate (Tuple[int, int], optional): downsampling factor. Defaults to (4, 4).
        sigma (float, optional): standard deviation of the Gaussian filter. Defaults to 1.

    Returns:
        np.ndarray: Decimated image
    """
    if decimate == (1, 1, 1, 1):
        return y
    assert y.ndim == len(decimate)
    from skimage.measure import block_reduce

    y = block_reduce(y, decimate, np.max)
    # y = 2 * np.pi * sigma**2 * ndi.gaussian_filter(y, sigma)
    y = np.clip(y, 0, 1)
    return y.astype(np.float32)