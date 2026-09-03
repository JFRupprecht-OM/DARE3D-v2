import os

import monai
import numpy as np
import pickle
import json
import torch
from skimage import io
from tqdm import tqdm

from dare3d.data.components.angles3d import representation_to_quaternion
from dare3d.data.components.regression_geometry import (
    TRAINING_CONSISTENT,
    normalize_preprocessing_mode,
)
from dare3d.utils.regression_display import display_regression


class InferenceAborted(Exception):
    """Raised by the inference loops when a ``should_stop`` callback asks to abort.

    Used by the napari plugin's Stop button to interrupt a long run cleanly; the
    caller catches it and discards any partial result.
    """


def monai_model_wrapper(model):
    def f(x):
        o = model(x)
        return {f"heatmap_{i}": o["heatmaps"][i] for i in range(len(o["heatmaps"]))}
    return f

def segmentation_inference(dataset, model, device, crop_size, batch_size, overlap=0.5, output_dir=None, should_stop=None):
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

            if should_stop is not None and should_stop():
                raise InferenceAborted()

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

def prepare_regression_dataset(dataset, mode=TRAINING_CONSISTENT):
    """Initialize a regression dataset without changing segmentation behavior."""
    mode = normalize_preprocessing_mode(mode)
    if hasattr(dataset, "init_inference"):
        dataset.init_inference(mode=mode)
    else:
        # Compatibility for light-weight third-party/test datasets.
        dataset.init(preprocess=False)
        dataset.pad_images()
        if dataset.renorm:
            dataset._normalize(dataset.renorm)
    return dataset


def regression_inference(dataset, model, centers, device, output_dir=None, should_stop=None):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    device_name = device if isinstance(device, str) else device.type
    use_cuda = device_name != "cpu"

    print(f"Using {'GPU' if use_cuda else 'CPU'}")
    if output_dir is not None:
        print(f"Prediction will be stored in folder: {output_dir}")
    predictions = []

    for center_value in tqdm(
        centers, desc="Running regression inference...", total=len(centers)
    ):
        if should_stop is not None and should_stop():
            raise InferenceAborted()

        # Segmentation and evaluation supply raw (M,T,X,Y,Z) centers.
        center_raw = tuple(float(item) for item in center_value)
        if hasattr(dataset, "raw_center_to_regression"):
            center_regression = dataset.raw_center_to_regression(center_raw)
        else:
            center_regression = center_raw

        if use_cuda:
            X = dataset.get_crop_from_center_gpu(center_raw, device)
        else:
            X = dataset.get_crop_from_center(center_raw, device)
        X = torch.unsqueeze(X, axis=0)

        with torch.no_grad():
            y = model.forward(X)["head1"]
            length = (
                y["len"][0].detach().cpu().numpy()
                * float(np.min(dataset.crop_size))
            )
            raw_rotation = y["angle"][0].detach().cpu().numpy()
            quaternion = representation_to_quaternion(
                raw_rotation, dataset.representation, post=True
            )

        from scipy.spatial.transform import Rotation as R

        rotation_matrix = R.from_quat(
            quaternion[[1, 2, 3, 0]]
        ).as_matrix()
        prediction = {
            # Backward-compatible fields. Center remains raw MTXYZ; length is
            # checkpoint-native regression-grid voxels.
            "center": center_raw,
            "length": length,
            "rotation": quaternion,
            # Explicit fields for new callers.
            "center_raw": center_raw,
            "center_regression": center_regression,
            "length_regression_voxels": float(
                np.asarray(length).reshape(-1)[0]
            ),
            "rotation_matrix": rotation_matrix,
            "raw_rotation": raw_rotation,
            "representation": (
                dataset.representation.name
                if hasattr(dataset.representation, "name")
                else str(dataset.representation)
            ),
        }
        if hasattr(dataset, "decode_prediction"):
            prediction.update(
                dataset.decode_prediction(center_raw, length, quaternion)
            )
        predictions.append(prediction)

    if output_dir is not None:
        archive = {
            "centers": np.asarray(
                [prediction["center_raw"] for prediction in predictions],
                dtype=np.float64,
            ),
            "centers_regression": np.asarray(
                [prediction["center_regression"] for prediction in predictions],
                dtype=np.float64,
            ),
            "lengths": np.asarray(
                [prediction["length"] for prediction in predictions]
            ),
            "lengths_regression_voxels": np.asarray(
                [
                    prediction["length_regression_voxels"]
                    for prediction in predictions
                ],
                dtype=np.float64,
            ),
            "rotation_matrices": np.asarray(
                [prediction["rotation_matrix"] for prediction in predictions]
            ),
            "quaternions": np.asarray(
                [prediction["rotation"] for prediction in predictions]
            ),
        }
        optional_arrays = {
            "lengths_physical_um": "length_physical_um",
            "lengths_raw_voxels": "length_raw_voxels",
            "axes_regression_xyz": "axis_regression_xyz",
            "axes_physical_xyz": "axis_physical_xyz",
            "axes_raw_xyz": "axis_raw_xyz",
            "endpoints_raw_xyz": "endpoints_raw_xyz",
        }
        for archive_name, prediction_name in optional_arrays.items():
            if predictions and prediction_name in predictions[0]:
                archive[archive_name] = np.asarray(
                    [prediction[prediction_name] for prediction in predictions]
                )
        np.savez(os.path.join(output_dir, "raw_predictions.npz"), **archive)

        if hasattr(dataset, "preprocessing_manifest"):
            with open(
                os.path.join(output_dir, "regression_preprocessing.json"),
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(dataset.preprocessing_manifest(), file, indent=2)

        display_regression(predictions, dataset, output_dir)

    return predictions
