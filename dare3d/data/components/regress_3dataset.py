import logging

import numpy as np
from skimage.measure import regionprops
import scipy

from .angles3d import ANGLE_REPRESENTATION, get_quaternion, quaternion_to_representation
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
        **kwargs
    ):
        if isinstance(crop_size, float) or isinstance(crop_size, int):
            crop_size = [crop_size] * 3
        self.crop_size = np.array(list(crop_size))
        self.half_crop_size = self.crop_size // 2
        self.representation = ANGLE_REPRESENTATION.from_string(angle_representation)
        self.angle_vec_size = angle_vec_size
        self.spatial_shift = spatial_shift
        super(Regress3Dataset, self).__init__(**kwargs)

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
        cosin_angles = []
        for center in centers:
            m,t,x,y,z = center
            candidate = None
            for bipoint in self.movies_bipoints[int(m)][int(t)]:
                # Bipoint
                a, b = bipoint
                a = np.asarray(a)
                b = np.asarray(b)

                real_center = (a + b)/2
                if np.all(np.isclose(np.asarray([x,y,z]), real_center, atol=dist_th)):
                    # Compute quaternion from bipoint
                    rot = get_quaternion(a, b)
                    length = self.distance_from_bipoint((a,b)) * np.min(self.crop_size)
                    candidate = {"length": length, "rotation": rot, "center": (m, t)+tuple(real_center)}
                    break
            cosin_angles.append(candidate)
            if candidate is None:
                print(f"/!\ Failed to find a matching groundtruth center for position {center}")
        assert len(cosin_angles) == len(centers)
        return cosin_angles

    def get_crop_from_center_gpu(self, center, device):
        import torch
        m,t,x,y,z = center
        m,t,x,y,z = m,int(np.rint(t)),int(np.rint(x)),int(np.rint(y)),int(np.rint(z))
        movie = self.movies_im[m]
        movie = movie[t-2:t+1]
        if movie.shape[0] < 3:
            n_diff = 3 - movie.shape[0]
            pad_section = np.zeros((n_diff,)+movie.shape[1:], dtype=movie.dtype)
            movie = np.concatenate([pad_section, movie], axis=0)
        assert movie.shape[0] == 3
        movie = torch.tensor(movie.astype(np.float32), device=device)
        crop = self.crop_img_from_center(movie, (x, y, z), return_crop=True)
        return crop

    def get_crop_from_center(self, center):
        m,t,x,y,z = center
        movie = self.movies_im[m]
        movie = movie[int(t)-2:int(t)+1]
        crop = self.crop_img_from_center(movie, (int(x), int(y), int(z)), return_crop=True)
        return crop

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
