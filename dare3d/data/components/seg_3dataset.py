import logging

import torch
import os
from tqdm import tqdm
import numpy as np

from dare3d.data.components.cell_3dataset import Cell3Dataset
from dare3d.utils.image import multiscale_decimate_gpu, multiscale_decimate

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class Segmentation3Dataset(Cell3Dataset):
    def __init__(
        self,
        type_mask="center",
        radius=None,
        val_mask=255,
        crop_size=32,
        mask_geom="sphere",
        downsample_factors=[1],
        sparse_folder=None,
        **kwargs
    ):
        self.type_mask = type_mask
        self.val_mask = val_mask
        self.radius = radius
        self.mask_geom = mask_geom
        self.crop_size = self.set_3dim(crop_size)
        # Positive sample contains at least one pixel of division
        self.positive_sample_index = []
        # Negative sample contains only background pixels
        self.negative_sample_index = []
        self.downsample_factors = [
            (1, t, t, t) for t in downsample_factors
        ]
        self.sparse_training = sparse_folder is not None
        self.sparse_folder = sparse_folder
        super(Segmentation3Dataset, self).__init__(**kwargs)

    def load_sparse_weights(self,):
        if len(self.movies_masks) == 0:
            return []
        
        movies_weights = []
        for i, movie_name in tqdm(enumerate(self.movie_names), total=len(self.movie_names)):
            if self.sparse_folder is None:
                movies_weights.append(np.ones_like(self.movies_masks[i], dtype=np.float32))
                continue

            movie_path = os.path.join(self.sparse_folder, f"{movie_name}.tif")
            # No weights means no sparse training for this movie
            if not (os.path.exists(movie_path) and self.training):
                log.warning(f"Did not find sparse weights for {movie_name} at path {movie_path}. Assuming movie is fully annotated. Sparse weights not used for validation/test")
                movies_weights.append(np.ones_like(self.movies_masks[i], dtype=np.float32))
                continue

            # Load movie weights
            movie_weights = self._load_image(movie_path)
            movie_weights = movie_weights.astype(np.float32)
            log.info(f"Loaded sparse weights for movie: {movie_name}")
            if np.sum(movie_weights) == 0.0:
                log.warning(f"Found sparse weights for movie {movie_name} but all weights are zeros. Please check the weights for this movie")
            
            # Add time padding if necessary
            movie_weights = self.add_time_padding(movie_weights)
    
            # Resize -> interpolate nearest
            target_shape = self._compute_target_shape(movie_weights.shape, movie_name)
            movie_weights = self.resize(movie_weights, target_shape, by_timestamp=False)
    
            # Pad to const zero
            target_pad = self.get_movie_padding(movie_weights)
            if target_pad is not None:
                movie_weights = np.pad(movie_weights, target_pad, mode="constant", constant_values=0.0)

            # Done
            movies_weights.append(movie_weights)
                    
        assert len(movies_weights) == len(self.movies_im)
        for movie_im, movie_weights in zip(self.movies_im, movies_weights):
            assert movie_weights.shape == movie_im.shape, f"Expected movie weights to have the same shape as the movie but found {movie_weights.shape} vs {movie_im.shape}"
        
        return movies_weights

    def post_process_init(self):
        self.sparse_weights = self.load_sparse_weights()

        if len(self.movies_masks) > 0:
            for seq_idx in tqdm(range(len(self.sequences_index)), desc="Separating positive and negative samples..."):
                sample_index = self.sequences_index[seq_idx]
                movie_index, _, end_idx, window_slice = sample_index.values()
                mask = self.movies_masks[movie_index]            
                Y = mask[end_idx -1]
                Y = Y[window_slice[1:]]
                if np.count_nonzero(Y) > 0:
                    self.positive_sample_index.append(seq_idx)
                else:
                    self.negative_sample_index.append(seq_idx)

    def add_sequence_indices(self, movie_index, t_start, t_end):
        movie_shape = self.movies_im[movie_index].shape
        windows_indices = self.compute_window_indices(movie_shape[1:],
                                                      size=self.crop_size, overlap=0)
        log.debug(f"Movie shape = {movie_shape}")
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

    def make_masks(self):
        self.movies_masks = self._construct_all_mask()

    def prepare_data(self):
        if self.load_labels:
            self.movies_masks = self._construct_all_mask()
        self._pad_movies()

    def _construct_all_mask(self):
        movies_masks = []
        for movie_index, movie_bipoints in tqdm(enumerate(self.movies_bipoints), desc="Building masks...", total=len(self.movies_bipoints)):
            movie_masks = []
            for time_index, bipoints in enumerate(movie_bipoints):
                # Get shape X,Y,Z
                mask_shape = self.movies_im[movie_index][time_index].shape
                mask = self.make_a_mask(bipoints, mask_shape)
                if mask.max() > 1:
                    mask = (mask / 255).astype(np.uint8)
                movie_masks.append(mask)
            movies_masks.append(np.array(movie_masks))

        total_frames = sum([movie.shape[0] for movie in movies_masks])
        log.info(
            f"Built {total_frames} masks with type_mask={self.type_mask}, mask_geom={self.mask_geom} and radius={self.radius}"
        )
        for i, movie_masks in enumerate(movies_masks):
            log.debug(
                f" -> movie {i} masks shape: {movie_masks.shape}, dtype {movie_masks.dtype}, min-max: {movie_masks.min()}-{movie_masks.max()} "
            )

        return movies_masks

    def make_a_mask(self, bipoints: list, mask_shape: tuple) -> np.ndarray:  # (dtype=np.uint8)
        mask = np.zeros(mask_shape, dtype=np.uint8)
        if self.mask_geom == "distance":
            from scipy import ndimage
            for bipoint in bipoints:
                p1, p2 = bipoint
                if self.type_mask == "center":
                    sigma = np.sqrt(np.linalg.norm(p1 - p2))
                    center = (np.array(p1) + np.array(p2)) / 2
                    center = center.astype(np.int32)
        
                    cmask = np.ones_like(mask)
                    cmask[center[0], center[1], center[2]] = 0    

                    # Return squared euclidean norm
                    cmask = np.square(ndimage.distance_transform_edt(cmask))
                    
                    cmask = np.exp(-cmask / (2*sigma**2))
                    mask = np.maximum(mask, cmask)
        elif self.mask_geom == "sphere":
            assert isinstance(self.radius, int)
            dimx, dimy, dimz = mask_shape  # Récupérer les dimensions de la matrice
            X, Y, Z = np.ogrid[:dimx, :dimy, :dimz]  # Créer un maillage 3D de coordonnées
            for bipoint in bipoints:
                p1, p2 = bipoint
                if self.type_mask == "center":
                    center = (np.array(p1) + np.array(p2)) / 2
                    center_x, center_y, center_z = center.astype(int)
                    mat_distance = (X - center_x) ** 2 + (Y - center_y) ** 2 + (Z - center_z) ** 2
                    mask[mat_distance <= self.radius**2] = self.val_mask
        return mask

    def draw_line_in_matrix(self, matrix, start, end, num_points, val=1):
        # Generate points along the line using linear interpolation
        t = np.linspace(0, 1, num_points)
        line_points = (1 - t)[:, np.newaxis] * start + t[:, np.newaxis] * end

        for point in line_points:
            x, y, z = point
            x_idx = int(x)
            y_idx = int(y)
            z_idx = int(z)
            matrix[x_idx, y_idx, z_idx] = val
        return matrix

    def _pad_movies(self):
        if self.load_labels:
            for i, (movie_im, movie_masks) in enumerate(zip(self.movies_im, self.movies_masks)):
                self.movies_im[i], self.movies_masks[i] = self._pad_movie_and_masks(
                    movie_im, movie_masks
                )
        else:
            for i, movie_im in enumerate(self.movies_im):
                self.movies_im[i], _ = self._pad_movie_and_masks(movie_im)
            
        for i, movie_im in enumerate(self.movies_im):
            log.debug(
                f"Padded movie to shape {movie_im.shape}"
            )

    def preprocess_sample(self, movie_index, start, end):
        x = super().preprocess_sample(movie_index, start, end)
        x, _ = self._pad_movie_and_masks(x)
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

    def _pad_movie_and_masks(self, movie, masks=None):
        # Whole movie has the same spatial size
        # Find what padding must be used
        target_pad = self.get_movie_padding(movie)        
        if target_pad is not None:
            if masks is not None:
                masks = np.pad(masks, target_pad, mode="constant", constant_values=0)
            movie = np.pad(movie, target_pad, mode="constant", constant_values=0)
        return movie, masks

    def set_augmentations(self, augmentations):
        self._augmentations = augmentations

    def random_crop(self, X, Y, target_shape):
        x_diff = X.shape[1] - target_shape[0]
        y_diff = X.shape[2] - target_shape[1]
        z_diff = X.shape[3] - target_shape[2]
        x = randrange(x_diff) if x_diff > 0 else 0
        y = randrange(y_diff) if y_diff > 0 else 0
        z = randrange(z_diff) if z_diff > 0 else 0
        return (
            X[
                :,
                x : x + target_shape[0],
                y : y + target_shape[1],
                z : z + target_shape[2],
            ],
            Y[
                x : x + target_shape[0],
                y : y + target_shape[1],
                z : z + target_shape[2],
            ],
        )        

    def get_movie_mask_from_idx(self, idx):
        # Get the module of the index in case the max idx has been multiplied 
        seq_idx = idx % len(self.sequences_index)

        # During training we perform a random crop
        if self.training:
            set_ = self.positive_sample_index
            # if not self.sparse_training: # Don't draw negative samples in sparse
            if len(set_) == 0 or (seq_idx % 2 == 1 and len(self.negative_sample_index) > 0):
                set_ = self.negative_sample_index
            seq_idx = np.random.choice(set_)
        
        sample_index = self.sequences_index[seq_idx]
        movie_index, start_idx, end_idx, window_slice = sample_index.values()

        movie = self.movies_im[movie_index]
        mask = self.movies_masks[movie_index]
            
        X = movie[window_slice]
        Y = mask[end_idx -1]
        Y = Y[window_slice[1:]]

        W = None
        if len(self.sparse_weights) > 0:
            W = self.sparse_weights[movie_index]

            W = W[end_idx - 1]
            W = W[window_slice[1:]]
            
        return X, Y, W, sample_index

    def compute_heatmaps(self, Y):
        if isinstance(Y, torch.Tensor):
            return [multiscale_decimate_gpu(Y, factor) for factor in self.downsample_factors]
        return [multiscale_decimate(Y, factor) for factor in self.downsample_factors]

    def build_output(self, input, heatmaps, weight, **kwargs):
        return {"input": input}, {"heatmaps": heatmaps, "weights": weight}

    def __getitem__(self, idx):
        X, Y, W, _ = self.get_movie_mask_from_idx(idx)
        
        Y = np.expand_dims(Y, axis=0)
        if W is not None:
            W = np.expand_dims(W, axis=0)

        if self._augmentations:
            # Augmentation is done on gpu
            augmented = self._augmentations({"image": X, "label":Y, "weight": W})
            X, Y, W = augmented["image"], augmented["label"], augmented["weight"]

        heatmaps = self.compute_heatmaps(Y)
        
        if W is not None:
            W = self.compute_heatmaps(W)
           
        return self.build_output(X, heatmaps, W)

    def __next__(self):
        sample = self.__getitem__(self.idx)
        self.idx = (self.idx + 1) % len(self)
        return sample