"""
Contains an abstract class that can be used to built 2D and 3D cell datasets.
This class provides interfaces that can be both used for segmentation and regression.
"""

import json
import logging
import os
import re
from glob import glob
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import skimage.transform as skt
import torch
from skimage.measure import regionprops
from torch.utils.data import DataLoader
from tqdm import tqdm

from omegaconf import OmegaConf, ListConfig

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def atoi(text: str) -> int:
    """Alphanumerical to integer.

    Args:
        text (str): Text to convert

    Returns:
        int: The text cast as integer if it is and integer.
    """
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]


class AbstractCellDataset(DataLoader):
    def __init__(
        self,
        im_folder: str,
        label_folder: str,
        input_channels: List[int],
        scale_file: str,
        default_scale: Union[float, List[float]],
        target_scale: float,
        renorm: str,
        training: bool = False,
        steps_per_epoch: int = -1,
        load_labels: bool = True,
        time_axis_padding: int = 0,
    ):
        """Create the cell dataset

        You can modulate the input channels to use. At time 't' the index
        is 0 so if you want to use t-1 aswell you can make a list with [-1, 0].
        Only -1, 0, 1 time steps are available.

        Args:
            data_folder (str): The folder where the raw image are stored as .tif
            label_folder (str): The folder where the label are stored as images (.tif) or coordinates (.npy)
            input_channels (List[int]): The index of the input channels to use
            scale (Union[float, List[float]]): The scale used to rescale the bipoints and input images.
            Can either be a single float for all dimensions or a list of floats for each dimension.
            renorm (str): The name of the renormalization strategy: 'min-max'
        """
        self.im_folder = im_folder
        self.label_folder = label_folder
        self.input_channels = input_channels
        self.n_input_channels = len(self.input_channels)
        self.default_scale = self.set_3dim(default_scale)
        self.target_scale = self.set_3dim(target_scale)
        self.movies_scale = self.load_movie_scales(scale_file)
        self._augmentations = None
        self._sample_limit = np.inf
        self.idx = 0
        self.steps_per_epoch = steps_per_epoch
        self.training = training
        self.renorm = renorm
        self.load_labels = load_labels
        # Do not pad validation/test datasets
        self.time_axis_padding = time_axis_padding if training else 0

    def set_3dim(self, value):
        if isinstance(value, int) or isinstance(value, float):
            r = [value, value, value]
        elif isinstance(value, tuple) and len(value) == 3:
            r = list(value)
        elif isinstance(value, list) and len(value) == 3:
            r = list(value)
        elif isinstance(value, ListConfig) and len(value) == 3:
            r = OmegaConf.to_object(value)
        else:
            raise ValueError(
                f"Could not parse {value} of type {type(value)}; Expected [int|tuple(3)|list(3)]"
            )
        return np.array(r)

    def load_movie_scales(self, path):
        if not os.path.exists(path):
            log.warning(
                f"Scale file does not exists at path: {path}. Using default scale {self.default_scale}"
            )
            return {}

        with open(path, "r") as file:
            movies_scale = json.load(file)
        return movies_scale

    def init(self, preprocess=True):
        self.movies_im = []
        self.movies_masks = []
        self.movie_names = []

        # Load data samples paths
        self._list_data()
        # Load images and associated bipoints
        # Data are also rescaled based on the given scale
        self.movies_im, self.movies_bipoints = self._load_data()

        if preprocess:
            self.resize_data()

            self.prepare_data()

            if self.renorm:
                self._normalize(self.renorm)

            self.prepare_data_after_norm()

        # Compute number of sequences that can be made for all movies
        self.sequences_index = []
        self.compute_sequences_index()
        self.total_valid_sequences = len(self.sequences_index)

        # Override the number of sequences (step) in a single epoch
        # In this case the number of steps should be larger that the number of sequences
        if self.training and self.steps_per_epoch > 0:
            assert (
                self.total_valid_sequences <= self.steps_per_epoch
            ), f"Expected the number of steps per epoch to be at least the amount of valid sequences but found {self.steps_per_epoch} vs {self.total_valid_sequences}"
            self.total_valid_sequences = self.steps_per_epoch

        log.info(f"Total number of sequences: {self.total_valid_sequences}")
        self.post_process_init()

    def post_process_init(self):
        pass

    def compute_sequences_index(self):
        movie_start_index = self.n_input_channels - 1
        for i, movie in enumerate(self.movies_im):
            for t in range(movie_start_index, movie.shape[0]):
                start_index = t - movie_start_index
                end_index = t + 1
                self.add_sequence_indices(i, start_index, end_index)

    def add_sequence_indices(self, movie_index, t_start, t_end):
        self.sequences_index.append(
            {"movie_index": movie_index, "start_index": t_start, "end_index": t_end}
        )

    def resize_data(self):
        log.info("Start resizing...")
        for i, (movie, movie_name) in enumerate(zip(self.movies_im, self.movie_names)):
            # Resize the image if needed: based on given scale
            target_shape = self._compute_target_shape(movie.shape, movie_name)
            log.info(f"Movie {movie_name} of shape {movie.shape} => resized to {target_shape}")
            # Resize movie
            self.movies_im[i] = self.resize(movie, target_shape, by_timestamp=True)

            if self.load_labels:
                # Resize bipoints to target size
                self.movies_bipoints[i] = self._resize_bipoints(self.movies_bipoints[i], movie_name)

    def preprocess_sample(self, movie_index, start, end):
        movie = self.movies_im[movie_index]
        movie_name = self.movie_names[movie_index]
        target_im_shape = self._compute_target_shape(movie.shape, movie_name=movie_name)
        target_im_shape = (self.n_input_channels,) + (target_im_shape[1:])
        # Time slice
        x = movie[start:end]
        # Resize to scale
        x = self.resize(x, target_im_shape, by_timestamp=False)
        return x

    def preprocess_sample_gpu(self, movie_index, start, end, device):
        movie = self.movies_im[movie_index]
        movie_name = self.movie_names[movie_index]
        target_im_shape = self._compute_target_shape(movie.shape, movie_name)
        target_im_shape = target_im_shape[1:]
        target_im_shape = tuple(int(s) for s in target_im_shape)
        # Time slice
        x = movie[start:end]
        x = torch.tensor(x.astype(np.float32), device=device)
        x = torch.unsqueeze(x, dim=0)
        x = torch.nn.functional.interpolate(x, size=target_im_shape, mode="trilinear")
        return x[0]

    def prepare_data_after_norm(self) -> None:
        """Simple hook function to format the data after it has been normalized."""
        pass

    def set_sample_limit(self, sample_limit: float) -> None:
        """Set the maximum index to use by the given amount.
        This is useful when you want to use a subset of the data.

        Args:
            sample_limit (float): The quantity to use within range [0,len(self.images)].
            0 is no data and len(self.images) is all data.
        """
        if sample_limit == np.inf:
            sample_limit = len(self.images)
        assert sample_limit >= 0.0 and sample_limit <= len(
            self.images
        ), f"Expected sample limit to be within [0;{len(self.images)}] but found {sample_limit}"
        self.sample_limit = sample_limit

    def resize(self, img: np.ndarray, target_img_shape: tuple, by_timestamp=False) -> np.ndarray:
        """Resize the image to the target shape.
        If the image has already the target shape do nothing.

        Args:
            img (np.ndarray): The image to resize
            target_img_shape (tuple): The target shape of the image

        Returns:
            np.ndarray: The resized image.
        """
        if img.shape == target_img_shape:
            return img

        if by_timestamp:
            new_image = np.zeros(target_img_shape, np.float32)
            for i in range(0, target_img_shape[0]):
                new_image[i] = self.resize(
                    img[i], target_img_shape=target_img_shape[1:], by_timestamp=False
                )
            img = new_image
            return img

        return skt.resize(
            img, target_img_shape, order=1, mode="reflect", cval=0, clip=True, preserve_range=True
        )

    def prepare_data(self) -> None:
        """Hook function to prepare data after loading and before normalization."""
        pass

    def _normalize(self, norm_type: str) -> None:
        """Function where the data must be normalized following the given norm_type

        Args:
            norm_type (str): Normalization strategy to use.
        """
        raise NotImplementedError

    def _load_image(self, path: str) -> np.ndarray:
        """Load a single image from path.

        Args:
            path (str): The path to the image.

        Returns:
            np.ndarray: The image loaded
        """
        raise NotImplementedError

    def unscale_prediction(self, pred, movie_index):
        # Revert all changes on shape and padding added to the image
        original_shape = self.original_movies_shape[movie_index]
        target_shape = self._compute_target_shape(
            original_shape, movie_name=self.movie_names[movie_index]
        )[1:]
        padding_added = np.array(pred.shape) - np.array(target_shape)

        no_pad_index = tuple(
            slice(b // 2, a - (b - b // 2)) for a, b in zip(pred.shape, padding_added)
        )
        # Remove padding
        pred = pred[no_pad_index]

        # Rescale image
        pred = self.resize(pred, original_shape[1:])

        return pred

    def add_time_padding(self, movie):
        if self.time_axis_padding > 0:
            npad = [(0, 0)] * movie.ndim
            npad[0] = (self.time_axis_padding, 0)
            movie = np.pad(movie, pad_width=npad, mode="constant", constant_values=0)
        return movie

    def _load_data(
        self,
    ) -> Tuple[List[np.ndarray], List[Tuple[np.float16, np.float16]]]:
        """Load the image and the bipoints at the same time.
        It loads dynamically the parametrized image channels (concatenated) then resize the image.
        The bipoints are then loaded and also transformed to be in par with the image size.

        Returns:
            Tuple[List[np.ndarray], List[Tuple[np.float16, np.float16]]]: _description_
        """
        movies = []
        bipoints = []

        self.original_movies_shape = []

        log.info("Loading all movies and bipoints...")
        for movie_name in tqdm(self.movie_names, total=len(self.movie_names)):
            # Load raw image
            movie = self._load_image(os.path.join(self.im_folder, f"{movie_name}.tif"))

            # Zero padding time axis
            movie = self.add_time_padding(movie)

            # Save original shape in case we need to get it back
            self.original_movies_shape.append(movie.shape)
            movies.append(movie)

            # Load bipoints labels (from .npy matrix or .tif label matrix)
            if self.load_labels:
                movie_bipoints = self._load_bipoints(self.label_folder, movie_name)

                if self.time_axis_padding > 0:
                    bipoints_padding = [] * self.time_axis_padding
                    movie_bipoints = [bipoints_padding] + movie_bipoints

                bipoints.append(movie_bipoints)

        total_frames = sum([movie.shape[0] for movie in movies])
        log.info(f" LOADED {len(movies)} movies with a total of {total_frames} timestamps")
        for i, (movie_name, movie) in enumerate(zip(self.movie_names, movies)):
            log.debug(
                f" -> movie [{i}:{movie_name}] shape: {movie.shape}, dtype {movie.dtype}, min-max: {movie.min()}-{movie.max()} "
            )

        return movies, bipoints

    def _resize_bipoints(self, bipoints, movie_name):
        # Two dimensionnal bipoints (timeframe, bipoints)
        scale = self.get_movie_scale(movie_name)
        for t in range(len(bipoints)):
            for i in range(len(bipoints[t])):
                p1, p2 = bipoints[t][i]
                p1 = np.round(np.array(p1) * scale + 1e-9)
                p2 = np.round(np.array(p2) * scale + 1e-9)
                bipoints[t][i] = (p1.astype(np.int16), p2.astype(np.int16))
        return bipoints

    def _list_data(self) -> None:
        """List all the image data in the image folder.
        Throws assertion error when there is no images.
        """
        path = os.path.join(self.im_folder, "*.tif")
        img_file_list = glob(path)
        self.movie_names = sorted([Path(x).stem for x in img_file_list], key=natural_keys)
        assert len(self.movie_names) > 0, f"Found no images in {path}"

    def _load_bipoints_from_array(self, folder: str, name: str):
        """Try to load an array {folder}/{name}.npy if it exists

        Args:
            folder (str): The folder where the bipoints file array should be
            name (str): The name of the file to load
        """
        file_path = os.path.join(folder, f"{name}.npy")
        if not os.path.exists(file_path):
            return None

        center_coords = np.load(file_path)
        nb_points = center_coords.shape[0]
        if nb_points % 2 != 0:
            raise ValueError(
                f" file {file_path.split('/')[-1]} contains {nb_points} points. Not a pair number "
            )

        bipoints = []
        # Bipoints are stored in a flat "list"
        for k in range(nb_points // 2):
            # p1_coords = center_coords[2 * k]   dimension [.., .., .. , 8]
            # coordinate[2 * k][1], coordinate[2* k][0], coordinate[2 * k][2]
            # (x,y,z, 8)
            p1 = center_coords[2 * k][:3]
            p2 = center_coords[2 * k + 1][:3]

            p1 = np.round(np.array(p1) + 1e-9)
            p2 = np.round(np.array(p2) + 1e-9)

            bipoints.append((p1.astype(np.int16), p2.astype(np.int16)))

        return bipoints

    def _load_bipoints_from_label(self, folder: str, name: str):
        """Try to load the bipoints coordinates from a label matrix.

        Args:
            folder (str): The folder where the bipoints file array should be
            name (str): The name of the label file to load
        """
        file_path = os.path.join(folder, f"{name}.tif")
        if not os.path.exists(file_path):
            return None
        label_im = self._load_image(file_path)
        # Loop over time dimension
        all_bipoints = []
        for t in range(label_im.shape[0]):
            time_slice = label_im[t]
            bipoints = self._bipoints_from_label_im(time_slice)
            all_bipoints.append(bipoints)

        return all_bipoints

    def _bipoints_from_label_im(self, label):
        unique_values = np.unique(label).tolist()

        # Filter regions that do not have their counterpart
        # 0 is background so 1-2 ; 3-4 etc are valid
        # Pair values go with previous value
        # Unpair values go with next value
        for value in unique_values:
            if value == 0:
                continue
            matching_value = value + 1
            if value % 2 == 0:
                matching_value = value - 1

            # Remove label if its counterpart does no exist
            if matching_value not in unique_values:
                log.warning(f"Label value {value} has missing counterpart value {matching_value}")
                label[label == value] = 0

        regions = regionprops(label)

        points = []
        bipoints = []
        for i, props in enumerate(regions):
            if len(points) == 0:
                points = [props.centroid]
            else:
                points += [props.centroid]
                bipoints.append(points)
                points = []
        return bipoints

    def _load_bipoints(self, folder: str, name: str) -> List[Tuple[np.float16, np.float16]]:
        """Load the bipoints from a bipoints path.
        Attempt to read the bipoints from .npy if it exists else look for a label image

        Args:
            folder (str): The path to the folder that contains the groundtruth.
            name (str): The name of the groundtruth item that can be name.tif or name.npy

        Raises:
            ValueError: The bipoints file is corrupted since len(bipoints)%2 != 0

        Returns:
            List[Tuple[np.float16, np.float16]]: The list of loaded bipoints
        """
        bipoints = self._load_bipoints_from_array(folder, name)
        if bipoints is None:
            bipoints = self._load_bipoints_from_label(folder, name)
            n_bipoints = sum([len(bp) for bp in bipoints])
            if n_bipoints == 0:
                log.info(f"Found {n_bipoints} bipoints for movie {name}")
            if bipoints is None:
                raise ValueError(f"Did not find either .npy or .tif label file in {folder}/{name}")
        return bipoints

    def get_movie_scale(self, movie_name: str = None):
        # Set the movie scale to the default one
        movie_scale = self.default_scale
        if movie_name != None:
            # Attempt to find a scale in the json file if it exists
            if movie_name in self.movies_scale:
                movie_scale = np.array(self.movies_scale[movie_name])
        movie_scale_factor = movie_scale / self.target_scale
        log.debug(f"Movie {movie_name} has to be rescaled by a factor {movie_scale_factor}")
        return movie_scale_factor

    def _compute_target_shape(self, im_shape: tuple, movie_name: str = None) -> tuple:
        """Compute the target shape using the scale parameter.

        Args:
            im_shape (tuple): The current image shape

        Returns:
            tuple: The new target image shape.
        """
        # Shape is : D0, D1, ..., T
        scale = self.get_movie_scale(movie_name)
        log.debug(f"Movie name {movie_name} has a scale factor: {scale}")

        assert len(scale) == len(
            im_shape[1:]
        ), f"Invalid number of scale components vs image dim, found {len(scale)} vs {len(im_shape[1:])}"
        new_shape = scale * im_shape[1:]
        new_shape = new_shape.astype(np.int32)

        return (im_shape[0],) + tuple(new_shape)

    def set_augmentations(self, augmentations: any) -> None:
        """Set the augmentations to use for this dataset.

        Args:
            augmentations (any): The augmentations object from albumentation.
        """
        self._augmentations = augmentations

    def __next__(self) -> List[Tuple[np.ndarray, Tuple[np.float16, np.float16]]]:
        """Retrieve the next sample.

        Returns:
            Tuple[np.ndarray, List[Tuple[np.float16, np.float16]]]: The image and the list of bipoints in the image
        """
        sample = self.__getitem__(self.idx)
        self.idx = (self.idx + 1) % len(self)
        return sample

    def __iter__(self):
        """Return the instance object as iterator.

        Returns:
            AbstractCellDataset: The instance as iterator
        """
        return self

    def __len__(self) -> int:
        """Return the number of samples.

        Returns:
            int: The number of samples.
        """

        return min(self.total_valid_sequences, self._sample_limit)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, List[Tuple[np.float16, np.float16]]]:
        """Return the sample at given index.

        Args:
            idx (int): The index of the sample.

        Returns:
            Tuple[np.ndarray, List[Tuple[np.float16, np.float16]]]: The image and the list of bipoints in the image
        """
        return self.images[idx], self.bipoints[idx]
