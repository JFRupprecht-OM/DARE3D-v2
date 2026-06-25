import numpy as np
import scipy

from dare3d.data.components.seg_3dataset import Segmentation3Dataset
from dare3d.data.components.angles3d import get_quaternion, apply_quaternion, quaternion_to_representation, ANGLE_REPRESENTATION

class SegRes3Dataset(Segmentation3Dataset):
    def __init__(self, base_quat_length=10, angle_representation="rotation_matrix_SVD", **kwargs):
        self.base_quat_length = base_quat_length
        self.representation = ANGLE_REPRESENTATION.from_string(angle_representation)
        super().__init__(**kwargs)
    
    def build_output(self, input, heatmaps, **kwargs):
        x, y = super().build_output(input, heatmaps, **kwargs)
        for key, val in kwargs.items():
            y.update({key: val})
        return x, y
    
    def rotate_scale_point_from_center(self, point, center, quaternion, scale):
        # Vector OA => A-O
        vector = point - center
        # Apply quaternion to vector
        
        rotated = apply_quaternion(quaternion, vector)
        
        # Normalize vector
        rotated = np.linalg.norm(rotated)

        # Scale vector
        rotated = rotated * scale
        
        # Get the extreme point 
        a = rotated + center

        return a
            
    def rotate_scale_bipoint_from_center(self, bipoint, center, quaternion=None, scale=1.0):
        a, b = bipoint
        if quaternion is not None and scale != 1.0:
            a = self.rotate_scale_point_from_center(a, center, quaternion, scale)
            b = self.rotate_scale_point_from_center(b, center, quaternion, scale)
        return a, b
    
    def compute_rotation_length(self, movie_index, time, angle, scale):
        # Mask has shape (W, H, D)
        mask = self.movies_masks[movie_index][time]
        binary_mask = np.zeros_like(mask)
        mask_center = np.array(mask.shape) / 2
        rotation_matrix = np.zeros(((9*2,) + mask.shape))
        length_matrix = np.zeros(((1*2,) + mask.shape))

        bipoints = self.movies_bipoints[movie_index][time]
        new_bipoints = []
        if len(bipoints) != 0:        
            for bipoint in bipoints:
                new_bipoint = self.rotate_scale_bipoint_from_center(bipoint, mask_center, angle, scale)
                new_bipoints.append(new_bipoint)

        X, Y, Z = np.ogrid[:mask.shape[0], :mask.shape[1], :mask.shape[2]]
        for bipoint in new_bipoints:
            a, b = bipoint
            ab = b - a
            center = (a+b) / 2
            
            # Compute the total division length
            division_length = np.linalg.norm(ab)
            
            # Take a radius
            valid_radius = division_length * 0.8
            mat_distance = (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
            
            # Make one in the 
            binary_mask = np.maximum(binary_mask, 
                                     np.where(mat_distance <= valid_radius**2, 1, 0))
            
            indices = zip(*np.where(mat_distance <= valid_radius**2))
            for index in zip(indices):
                x, y, z = index[0]
                pt = np.array([x, y, z])
                rotation_matrix[:9, x, y, z] = self.compute_rotation((a, pt))
                rotation_matrix[9:, x, y, z] = self.compute_rotation((b, pt))
                length_matrix[0, x, y, z] = self.distance_from_bipoint((a, pt))
                length_matrix[1, x, y, z] = self.distance_from_bipoint((b, pt))

        return rotation_matrix, length_matrix, binary_mask
    
    def compute_rotation(self, bipoint):
        p1, p2 = bipoint
        quat = get_quaternion(p1, p2)
        rotation = quaternion_to_representation(quat, self.representation)
        return rotation
        
    def distance_from_bipoint(self, bipoint):
        p1, p2 = bipoint
        vec = p1 - p2
        length = np.linalg.norm(vec)
        length_normalized = length / np.min(self.crop_size)
        return length_normalized

    def extract_angle_scale_from_dummy_mask(self, mask):
        label, n = scipy.ndimage.label(mask[0])
        
        uniques = np.unique(label)
        assert n == 2 # Background and two points
        
        mean_pts = []
        for i in range(uniques.shape[0]):
            unique_val = uniques[i]
            if unique_val == 0:
                continue
            cc = np.array(np.where(label == unique_val))
            centroid = cc.mean(axis=-1)
            mean_pts.append(centroid)
            
        assert len(mean_pts) == 2
        quat = get_quaternion(mean_pts[0], mean_pts[1])
        
        vec_norm = np.linalg.norm(mean_pts[0] - mean_pts[1]) / self.base_quat_length
        
        return quat, vec_norm
    
    def create_dummy_mask(self, shape):
        dummy_mask = np.zeros(shape)
        center = (np.array(shape)/2).astype(np.int32)
        dummy_mask[0, center[1], center[2], center[3]] = 1.0
        dummy_mask[0, center[1] + self.base_quat_length, center[2], center[3]] = 1.0
        return dummy_mask
        
    def __getitem__(self, idx):
        X, Y, sample_index = self.get_movie_mask_from_idx(idx)
        
        movie_index, _, end_idx, window_slice = sample_index.values()
                
        Y = np.expand_dims(Y, axis=0)
        dummy_mask = self.create_dummy_mask(Y.shape)
        
        if self._augmentations:
            # Augmentation is done on gpu
            augmented = self._augmentations({"image": X, "label":Y, "bipoints": dummy_mask})
            X, Y, dummy_mask = augmented["image"], augmented["label"], augmented["bipoints"]

            X = X.detach().cpu().numpy()
            Y = Y.detach().cpu().numpy()
            dummy_mask = dummy_mask.detach().cpu().numpy()

        quat, scale = self.extract_angle_scale_from_dummy_mask(dummy_mask)
        
        rotmat, length, length_mask = self.compute_rotation_length(movie_index, end_idx-1, quat, scale)
        rotmat = rotmat[window_slice[1:]]
        length = length[window_slice[1:]]
        
        rotmat = rotmat.astype(np.float32)
        length = length.astype(np.float32)
        length_mask = length_mask.astype(np.float32)
        
        if np.isnan(rotmat.any()):
            raise ValueError
        if np.isnan(length.any()):
            raise ValueError
        if np.isnan(length_mask.any()):
            raise ValueError

        heatmaps = self.compute_heatmaps(Y)
           
        return self.build_output(X, heatmaps, rotmat=rotmat, length=length, length_mask=length_mask)

    
    
# How to perform augmentation on bipoints

# Create a dummy ground truth with two points at the center
# Perform the augmentation
# Now we retrieve the two points and compute angle + scale
# We get the bipoints for the current index and 

# For each bipoint
    # Put a number in the ground truth mask
    # Augment
    # Retrieve all points with numbers and get the bipoints back
    # Compute the new matrices
    
    
# Output matrix is coupled in its output
## 2xn,W,H,D => 2 x rotation matrix
## 2, W, H, D => 2 x distance
