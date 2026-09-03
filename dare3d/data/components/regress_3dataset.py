import copy
import logging

import numpy as np
from skimage.measure import regionprops
import scipy

from .angles3d import ANGLE_REPRESENTATION, get_quaternion, quaternion_to_representation
from .regression_geometry import (
    TRAINING_CONSISTENT,
    RegressionPreprocessingSpec,
    RegressionSpatialTransform,
    axis_from_wxyz_quaternion,
    normalize_preprocessing_mode,
)
from .cell_3dataset import Cell3Dataset
from tqdm import tqdm

log = logging.getLogger(__name__)


class Regress3Dataset(Cell3Dataset):
    def __init__(
        self,
        crop_size=32,
        angle_representation="rotation_matrix_GS",
        angle_vec_size=9,
        spatial_shift=0,
        inference_preprocessing=TRAINING_CONSISTENT,
        require_scale_file=False,
        **kwargs
    ):
        if isinstance(crop_size, float) or isinstance(crop_size, int):
            crop_size = [crop_size] * 3
        self.crop_size = np.array(list(crop_size))
        self.half_crop_size = self.crop_size // 2
        self.representation = ANGLE_REPRESENTATION.from_string(angle_representation)
        self.angle_vec_size = angle_vec_size
        self.spatial_shift = spatial_shift
        self.inference_preprocessing = normalize_preprocessing_mode(inference_preprocessing)
        self.require_scale_file = bool(require_scale_file)
        self.regression_inference_ready = False
        super(Regress3Dataset, self).__init__(**kwargs)

    def on_data_loaded(self):
        """Retain raw annotations and construct one spatial map per movie."""
        self.movies_bipoints_raw = copy.deepcopy(self.movies_bipoints)
        self._build_regression_transforms(TRAINING_CONSISTENT)

    def _build_regression_transforms(self, mode):
        mode = normalize_preprocessing_mode(mode)
        if self.require_scale_file:
            missing = [
                name for name in self.movie_names if name not in self.movies_scale
            ]
            if missing:
                raise ValueError(
                    "Regression preprocessing requires per-movie scale metadata; "
                    f"missing entries for {missing} in {self.scale_file!r}"
                )
        self.regression_preprocessing_mode = mode
        self.regression_transforms = [
            RegressionSpatialTransform(
                RegressionPreprocessingSpec.from_dataset(self, index, mode)
            )
            for index in range(len(self.movie_names))
        ]

    def init_inference(self, mode=None):
        """Load and prepare movies in the checkpoint's regression image space.

        Unlike init(), this path does not enumerate annotation-centred samples.
        Input centers remain raw MTXYZ coordinates and are transformed only when
        their regression crop is requested.
        """
        mode = normalize_preprocessing_mode(mode or self.inference_preprocessing)
        self.init(preprocess=False)
        self._build_regression_transforms(mode)
        if mode == TRAINING_CONSISTENT:
            self.resize_data()
        self.pad_images()
        if self.renorm:
            self._normalize(self.renorm)
        self.prepare_data_after_norm()
        self.regression_inference_ready = True

    def preprocessing_manifest(self):
        return {
            "schema_version": 1,
            "mode": self.regression_preprocessing_mode,
            "movies": [transform.to_dict() for transform in self.regression_transforms],
        }

    def get_regression_transform(self, movie_index):
        if not hasattr(self, "regression_transforms"):
            raise RuntimeError(
                "Regression spatial transforms are unavailable; initialize the dataset first"
            )
        return self.regression_transforms[int(movie_index)]

    def raw_center_to_regression(self, center, *, round_result=False):
        return self.get_regression_transform(center[0]).raw_center_to_regression(
            center, round_result=round_result
        )

    def decode_prediction(self, center_raw, length_regression_voxels, quaternion):
        axis = axis_from_wxyz_quaternion(quaternion)
        return self.get_regression_transform(center_raw[0]).decode_axis_length(
            center_raw, axis, length_regression_voxels
        )

    def prepare_data(self):
        self.crops, self.bipoint_crops = self.make_crop_all_division()        

    def pad_images(self):
        # Pad all images
        for movie_index, movies_im in tqdm(enumerate(self.movies_im), desc=f"Padding {len(self.movies_im)} movies...", total=len(self.movies_im)):
            self.movies_im[movie_index] = self.pad_img(self.movies_im[movie_index])

    def make_crop_all_division(self):
        crops = []
        bipoint_crops = []
        no_division_images_index = []
        
        self.pad_images()
        
        for movie_index, movie_bipoints in tqdm(enumerate(self.movies_bipoints), desc="Building crops...", total=len(self.movies_bipoints)):
            for time_index, bipoints in enumerate(movie_bipoints):
                # Get shape X,Y,Z
                movie = self.movies_im[movie_index]
                t_shift = 0
                # for t_shift in range(-1, 2):
                ntime_index = time_index + t_shift
                if ntime_index > 1 and len(bipoints) > 0 and ntime_index < movie.shape[0]:
                    im = movie[ntime_index-2: ntime_index+1]
                    ncrops, nbipoint_crops = self.crop_one_img_division(im, bipoints, movie_index, slice(ntime_index-2, ntime_index+1))
                    crops.extend(ncrops)
                    bipoint_crops.extend(nbipoint_crops)

        log.info(f" CROPED {len(crops)} crops from all images with one division")
        log.info(
            f" ->  {len(no_division_images_index)} images with no divisions :{no_division_images_index}"
        )

        return crops, bipoint_crops

    def crop_img_from_center(self, img, center, return_crop=True):
        center_pad = (center + self.half_crop_size).astype(int)
        start_x, start_y, start_z = center_pad - self.half_crop_size
        end_x, end_y, end_z = center_pad + self.half_crop_size
        if not return_crop:
            return slice(start_x, end_x), slice(start_y, end_y), slice(start_z, end_z)
        crop = img[:, start_x:end_x, start_y:end_y, start_z:end_z]
        return crop

    def pad_img(self, img, padding_border="constant"):
        return np.pad(
            img,
            pad_width=(
                (0, 0),
                (self.half_crop_size[0], self.half_crop_size[0]),
                (self.half_crop_size[1], self.half_crop_size[1]),
                (self.half_crop_size[2], self.half_crop_size[2]),
            ),
            mode=padding_border,
            constant_values=0,
        )

    def unscale_prediction(self, length, rot, center):
        # Remove padding, so shift center
        
        # Apply inverse scaling to length
        
        # Apply inverse scaling to center
        
        return length, rot, center

    def crop_one_img_division(self, img, bipoints, movie_index, time_slice, padding_border="constant"):
        # assert len(img.shape) == 4 and img.shape[-1] == 3

        # img_pad = self.pad_img(img, padding_border)
        img_pad = img
        crops = []
        bipoint_crops = []

        for bipoint in bipoints:
            p1, p2 = bipoint
            center = (p1 + p2) / 2
            p1_center, p2_center = p1 - center, p2 - center

            crop_slice = self.crop_img_from_center(img_pad, center, return_crop=False)

            crops.append((movie_index, time_slice,)+crop_slice)

            new_p1_crop = (p1_center + self.half_crop_size).astype(int)
            new_p2_crop = (p2_center + self.half_crop_size).astype(int)                        

            bipoint_crops.append(tuple([new_p1_crop, new_p2_crop]))

        return crops, bipoint_crops

    def extract_bipoint_from_label(self, Y):
        label, n = scipy.ndimage.label(Y, structure=np.ones((3, 3, 3)))
        if n < 2:
            return None, None
        regions = regionprops(label)
        p1, p2 = np.array(regions[0].centroid), np.array(regions[1].centroid)
        return p1, p2

    def gather_groundtruth_info(self, centers, dist_th=3):
        """Resolve targets in raw space, then express axis/length in regression space.

        A detector-predicted center is never passed here. CenterList calls this
        method only with annotated/true-component centers and reuses the resolved
        target for predicted-center evaluation.
        """
        targets = []
        raw_bipoints = getattr(self, "movies_bipoints_raw", self.movies_bipoints)
        for center in centers:
            m, t, x, y, z = center
            movie_index = int(m)
            time_index = int(t)
            requested_xyz = np.asarray([x, y, z], dtype=np.float64)
            candidates = []
            for annotation_index, bipoint in enumerate(
                raw_bipoints[movie_index][time_index]
            ):
                a_raw, b_raw = (
                    np.asarray(bipoint[0], dtype=np.float64),
                    np.asarray(bipoint[1], dtype=np.float64),
                )
                real_center_raw = (a_raw + b_raw) / 2.0
                if np.all(np.isclose(requested_xyz, real_center_raw, atol=dist_th)):
                    candidates.append(
                        (
                            annotation_index,
                            float(np.linalg.norm(requested_xyz - real_center_raw)),
                            a_raw,
                            b_raw,
                            real_center_raw,
                        )
                    )

            candidate = None
            if candidates:
                # Preserve the historical file-order selection rule. Record the
                # candidate count and distance so ambiguity is never silent.
                annotation_index, distance, a_raw, b_raw, real_center_raw = candidates[0]
                transform = self.get_regression_transform(movie_index)
                a_regression = transform.raw_point_to_regression(
                    a_raw, round_result=True
                )
                b_regression = transform.raw_point_to_regression(
                    b_raw, round_result=True
                )
                rotation = get_quaternion(a_regression, b_regression)
                length = float(np.linalg.norm(a_regression - b_regression))
                decoded = transform.decode_axis_length(
                    (m, t, *real_center_raw),
                    axis_from_wxyz_quaternion(rotation),
                    length,
                )
                candidate = {
                    "length": np.asarray([length], dtype=np.float32),
                    "rotation": rotation,
                    "center": (m, t, *tuple(real_center_raw)),
                    "event_id": (
                        f"{self.movie_names[movie_index]}:"
                        f"{time_index}:{annotation_index}"
                    ),
                    "annotation_index": annotation_index,
                    "target_candidate_count": len(candidates),
                    "target_center_distance_raw_voxels": distance,
                    "annotated_endpoints_raw_xyz": np.stack((a_raw, b_raw)),
                    "annotated_endpoints_regression_xyz": np.stack(
                        (a_regression, b_regression)
                    ),
                    **decoded,
                }

            targets.append(candidate)
            if candidate is None:
                print(
                    f"Failed to find a matching groundtruth center for "
                    f"position {center}"
                )
        assert len(targets) == len(centers)
        return targets

    def get_crop_from_center(self, center, device="cpu", center_space="raw"):
        """Extract the three-frame regression crop around a raw or grid center."""
        import torch

        if center_space == "raw":
            crop_center = self.raw_center_to_regression(center)
        elif center_space == "regression":
            crop_center = tuple(float(item) for item in center)
        else:
            raise ValueError(
                f"Unknown center space {center_space!r}; expected raw or regression"
            )

        m, t, x, y, z = crop_center
        m, t, x, y, z = (
            int(np.rint(m)),
            int(np.rint(t)),
            int(np.rint(x)),
            int(np.rint(y)),
            int(np.rint(z)),
        )
        movie = self.movies_im[m]
        start_time = max(0, t - 2)
        movie = movie[start_time:t + 1]
        if movie.shape[0] < 3:
            n_diff = 3 - movie.shape[0]
            pad_section = np.zeros((n_diff,) + movie.shape[1:], dtype=movie.dtype)
            movie = np.concatenate([pad_section, movie], axis=0)
        assert movie.shape[0] == 3
        movie = torch.tensor(movie.astype(np.float32), device=device)
        crop = self.crop_img_from_center(movie, (x, y, z), return_crop=True)
        expected_shape = (3,) + tuple(int(item) for item in self.crop_size)
        if tuple(crop.shape) != expected_shape:
            raise RuntimeError(
                f"Regression crop at raw center {center} mapped to {crop_center} "
                f"has shape {tuple(crop.shape)}, expected {expected_shape}"
            )
        return crop

    def get_crop_from_center_gpu(self, center, device, center_space="raw"):
        """GPU alias kept for the existing call site."""
        return self.get_crop_from_center(center, device, center_space=center_space)

    def augment_sample(self, X, Y):
        augmented = self._augmentations({"image": X, "label":Y})
        X, Y = augmented["image"], augmented["label"]
        Y = Y.detach().cpu().numpy()
        p1, p2 = self.extract_bipoint_from_label(Y[0])
        if p1 is None or p2 is None:
            return X, None
        return X, (p1, p2)

    def distance_from_bipoint(self, bipoint):
        p1, p2 = bipoint
        vec = p1 - p2
        length = np.linalg.norm(vec)
        length_normalized = length / np.min(self.crop_size)
        return length_normalized

    def compute_rotation(self, bipoint):
        p1, p2 = bipoint
        # Compute quaternion from bipoint
        quat = get_quaternion(p1, p2)
        # Convert it to required representation
        rotation = quaternion_to_representation(quat, self.representation)        
        return rotation.astype(np.float32)

    def draw_bipoint(self, mat, bipoint):
        p1, p2 = bipoint
        mat[tuple(p1)] = 255
        mat[tuple(p2)] = 255
        return mat

    def __getitem__(self, idx):
        idx = idx % len(self.crops)

        crop_slices, bipoint = self.crops[idx], self.bipoint_crops[idx]
        movie_idx, time_slice, x_slice, y_slice, z_slice = crop_slices
        X = self.movies_im[movie_idx][time_slice, x_slice, y_slice, z_slice]

        if self._augmentations:
            label = self.draw_bipoint(np.zeros(X.shape[1:]), bipoint)
            label = np.expand_dims(label, axis=0)
                        
            nx, nbipoint = self.augment_sample(X, label)
            if nbipoint is not None and nx.max() > 0.0:
                X = nx.detach().cpu().numpy()
                bipoint = nbipoint

        distance = self.distance_from_bipoint(bipoint)
        rotation = self.compute_rotation(bipoint)
        
        
        # # DEBUG
        # crop_debug = np.zeros_like(X)
        # new_p1_crop, new_p2_crop = bipoint
        # crop_debug[:, int(new_p1_crop[0]), int(new_p1_crop[1]), int(new_p1_crop[2])] = 1.0
        # crop_debug[:, int(new_p2_crop[0]), int(new_p2_crop[1]), int(new_p2_crop[2])] = 1.0
        
        # from skimage import io
        # p="train" if self.training else "val"
        # io.imsave(f"logs/crop_{p}_{idx}.tif", X)
        # io.imsave(f"logs/crop_bipoints_{p}_{idx}.tif", crop_debug)
        
        Y = {
            "head1": 
                {
                    "len": np.array([distance], dtype=np.float32), 
                    "angle": rotation,
                    "bipoint": np.array(bipoint)}, 
            "head2": {}
            }

        return {"input": X.astype(np.float32)}, Y

    def __len__(self):
        return max(len(self.crops), self.steps_per_epoch)
