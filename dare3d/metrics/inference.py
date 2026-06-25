import os

import monai
import numpy as np
import pickle
import json
import torch
from skimage import io
from tqdm import tqdm

from dare3d.data.components.angles3d import representation_to_quaternion
from dare3d.utils.regression_display import display_regression

def monai_model_wrapper(model):
    def f(x):
        o = model(x)
        return {f"heatmap_{i}": o["heatmaps"][i] for i in range(len(o["heatmaps"]))}
    return f

def segmentation_inference(dataset, model, device, crop_size, batch_size, overlap=0.5, output_dir=None):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    movie_start = dataset.n_input_channels - 1
    crop_size = (crop_size, crop_size, crop_size)

    device_name = device if isinstance(device, str) else device.type
    use_cuda = device_name != "cpu"

    print(f"Using {'GPU' if use_cuda else 'CPU'}")
    if output_dir is not None:
        print(f"Prediction will be stored in folder: {output_dir}")

    predictions = []
    # Loop over movies
    for i in tqdm(range(len(dataset.movies_im)), desc="Running segmentation inference..."):
        # T,X,Y,Z
        current_movie = dataset.movies_im[i]
        movie_name = dataset.movie_names[i]

        y_pred_full = None

        # Get all possible sequences for the movie
        for t in tqdm(range(movie_start, current_movie.shape[0]), leave=False, desc=f"Processing movie: {movie_name}"): 

            if use_cuda:
                X = dataset.preprocess_sample_gpu(i, t-movie_start, t+1, device)
            else:
                X = dataset.preprocess_sample(i, t-movie_start, t+1)
                X = torch.tensor(X, device=device)
            X = torch.unsqueeze(X, axis=0)
            
            # Perform inference on current sequence
            with torch.no_grad():
                y_pred = monai.inferers.sliding_window_inference(
                    inputs=X, 
                    roi_size=crop_size, 
                    sw_batch_size=batch_size,
                    predictor=monai_model_wrapper(model),
                    overlap=overlap,
                    mode="gaussian",
                    device=device)

            # initialize the number of output scales using the first prediction
            if y_pred_full is None:
                y_pred_full = np.zeros((len(y_pred),)+current_movie.shape, dtype=np.float16)

            for key, value in y_pred.items():
                # c_y_pred = y_pred[f"heatmap_{k}"]
                if key != "heatmap_0":
                    continue
                c_y_pred = value
                c_y_pred = torch.nn.functional.sigmoid(c_y_pred)
                k = int(key.split("heatmap_")[-1])
                scale_factor = 2**k
                if scale_factor > 1:
                    c_y_pred = torch.nn.functional.interpolate(c_y_pred, scale_factor=scale_factor, mode="trilinear")
                c_y_pred = c_y_pred.detach().cpu().numpy()
                # B, 1, X, Y, Z to X, Y, Z
                c_y_pred = c_y_pred[0, 0].astype(np.float16)
                c_y_pred = dataset.unscale_prediction(c_y_pred, i)
                y_pred_full[k, t] = c_y_pred

        # From N, T, X, Y, Z to N, T, Z, Y, X to get original movie order dim
        y_pred_full = np.swapaxes(y_pred_full, -1, -3)
                                
        if output_dir is not None:
            for i in range(y_pred_full.shape[0]):
                if y_pred_full.shape[0] == 1:
                    output_name = f"{movie_name}.tif"
                else:
                    output_name = f"{movie_name}_scale{i}.tif"
                io.imsave(os.path.join(output_dir, output_name), y_pred_full[i], check_contrast=False)
        
        predictions.append(y_pred_full[0])
    return predictions

def regression_inference(dataset, model, centers, device, output_dir=None):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    device_name = device if isinstance(device, str) else device.type
    use_cuda = device_name != "cpu"

    print(f"Using {'GPU' if use_cuda else 'CPU'}")
    if output_dir is not None:
        print(f"Prediction will be stored in folder: {output_dir}")
    predictions = []
    raw_predictions = []  # Store raw rotation matrices and positions
    
    # Loop over centers
    for i in tqdm(range(len(centers)), desc="Running regression inference..."):
        # Center = (M,T,X,Y,Z) with M the movie index
        center = centers[i]
        
        # Get the crop for the corresponding center
        if use_cuda:
            X = dataset.get_crop_from_center_gpu(center, device)
        else:
            raise NotImplementedError
        X = torch.unsqueeze(X, axis=0)
            
        # Perform inference on current item
        with torch.no_grad():
            y = model.forward(X)
            y = y["head1"]
            # Un-normalize length
            length = y["len"][0].detach().cpu().numpy() * float(np.min(dataset.crop_size))
            
            # Compute rotation to quaternion
            rot = y["angle"][0].detach().cpu().numpy()
            quat = representation_to_quaternion(rot, dataset.representation, post=True)

            # Convert rotation to 3x3 rotation matrix for saving
            from scipy.spatial.transform import Rotation as R
            rotation_matrix = R.from_quat(quat[[1,2,3,0]]).as_matrix()  # Convert wxyz to xyzw for scipy

            # length, rot, center = dataset.unscale_prediction(length, rot, center)
            
            predictions.append({"length": length, "rotation": quat, "center": center})
            
            # Store raw prediction data
            raw_predictions.append({
                "center": center,  # (M,T,X,Y,Z)
                "length": length,
                "rotation_matrix": rotation_matrix,  # 3x3 rotation matrix
                "quaternion": quat,  # quaternion in wxyz format
                "raw_rotation": rot,  # original network output
                "representation": dataset.representation.name if hasattr(dataset.representation, 'name') else str(dataset.representation)
            })

    # Save raw predictions in multiple formats
    if output_dir is not None:
        # Save as numpy archive
        np.savez(os.path.join(output_dir, "raw_predictions.npz"), 
                centers=np.array([pred["center"] for pred in raw_predictions]),
                lengths=np.array([pred["length"] for pred in raw_predictions]),
                rotation_matrices=np.array([pred["rotation_matrix"] for pred in raw_predictions]),
                quaternions=np.array([pred["quaternion"] for pred in raw_predictions]))
        
        # Create visual representation (existing functionality)
        display_regression(predictions, dataset, output_dir)

    return predictions