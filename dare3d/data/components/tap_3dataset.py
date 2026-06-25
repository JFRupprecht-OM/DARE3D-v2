import logging
import torch
from random import randrange

import numpy as np

from dare3d.data.components.cell_3dataset import Cell3Dataset

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class Tap3Dataset(Cell3Dataset):
    def __init__(
        self,
        crop_size=32,
        **kwargs
    ):
        self.crop_size = self.set_3dim(crop_size)
        super(Tap3Dataset, self).__init__(**kwargs)

    def add_sequence_indices(self, movie_index, t_start, t_end):
        movie_shape = self.movies_im[movie_index].shape
        windows_indices = self.compute_window_indices(movie_shape[1:],
                                                      size=self.crop_size, overlap=0)
        for slices in windows_indices:
            slices = (slice(t_start, t_end),) + slices
            self.sequences_index.append(
                {
                    "movie_index": movie_index,
                    "start_index": t_start,
                    "end_index": t_end,
                    "window_slice": slices
                }
            )

    def prepare_data(self):
        self._pad_movies()

    def _pad_movies(self):
        for i, movie_im in enumerate(self.movies_im):
            self.movies_im[i] = self._pad_movie(movie_im)
            
        for i, movie_im in enumerate(self.movies_im):
            log.debug(
                f"Padded movie to shape {movie_im.shape}"
            )

    def preprocess_sample(self, movie_index, start, end):
        x = super().preprocess_sample(movie_index, start, end)
        x = self._pad_movie(x)
        x = self.normalise(x, renorm=self.renorm)
        return x

    def preprocess_sample_gpu(self, movie_index, start, end, device):
        x = super().preprocess_sample_gpu(movie_index, start, end, device)
        x = self.pad_movie_gpu(x)
        x = self.normalise_gpu(x, renorm=self.renorm)
        return x

    def _flatten_tuples(self, tupl, reverse=False):
        final = ()
        for sub_tupl in tupl:
            final = final + sub_tupl
        if reverse:
            return tuple(reversed(final))
        return final

    def pad_movie_gpu(self, x):
        target_pad = self.get_movie_padding(x)
        if target_pad is not None:
            target_pad = self._flatten_tuples(target_pad, reverse=True)
            x = torch.nn.functional.pad(x, target_pad, mode="constant", value=0)
        return x

    def get_movie_padding(self, movie):
        img_shape = movie[0].shape

        target_diff = np.array(self.crop_size) - np.array(img_shape)
        target_diff = np.where(target_diff > 0, target_diff, 0)
        if np.any(target_diff > 0):
            pad_x, pad_y, pad_z = target_diff
            target_pad = (
                (0, 0),  # Don't pad time axis
                (pad_x // 2, pad_x - pad_x // 2),
                (pad_y // 2, pad_y - pad_y // 2),
                (pad_z // 2, pad_z - pad_z // 2),
            )
            return target_pad
        return None

    def _pad_movie(self, movie):
        # Whole movie has the same spatial size
        # Find what padding must be used
        target_pad = self.get_movie_padding(movie)        
        if target_pad is not None:
            movie = np.pad(movie, target_pad, mode="constant", constant_values=0)
        return movie

    def set_augmentations(self, augmentations):
        self._augmentations = augmentations

    def random_crop(self, X, target_shape):
        x_diff = X.shape[1] - target_shape[0]
        y_diff = X.shape[2] - target_shape[1]
        z_diff = X.shape[3] - target_shape[2]
        x = randrange(x_diff) if x_diff > 0 else 0
        y = randrange(y_diff) if y_diff > 0 else 0
        z = randrange(z_diff) if z_diff > 0 else 0
        return X[
                :,
                x : x + target_shape[0],
                y : y + target_shape[1],
                z : z + target_shape[2],
            ]

    def __getitem__(self, idx):
        # Get the module of the index in case the max idx has been multiplied 
        idx = idx % len(self.sequences_index)
        
        sample_index = self.sequences_index[idx]
        movie_index, start_idx, end_idx, window_slice = sample_index.values()

        movie = self.movies_im[movie_index]

        # During training we perform a random crop
        if self.training:
            X = movie[start_idx:end_idx]
            X = self.random_crop(X, self.crop_size)
        else: # For a fair eval/val we need to iterate over all possible windows
            X = movie[window_slice]
        
        if self._augmentations:
            X = self._augmentations({"image": X})["image"]
        
        Y = X[-1, ...] - X[-2, ...] # It must predict the difference with the next frame
        Y = Y[None, :]
        X = X[0:3, ...] # Given 3 frames
        return X, Y

    def __next__(self):
        sample = self.__getitem__(self.idx)
        self.idx = (self.idx + 1) % len(self)
        return sample
