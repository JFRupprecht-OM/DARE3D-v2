import os

import numpy as np
from skimage import io
from dare3d.data.components.angles3d import (
    draw_line_in_matrix,
    get_points_from_quat,
)

def display_regression(predictions, dataset, output_dir):
    """Draw regression axes in the raw movie grid."""
    os.makedirs(output_dir, exist_ok=True)

    predictions_by_movie = {}
    for prediction in predictions:
        if prediction is None:
            continue
        movie_index = int(prediction["center"][0])
        predictions_by_movie.setdefault(movie_index, []).append(prediction)

    for movie_index in range(len(dataset.movies_im)):
        if hasattr(dataset, "original_movies_shape"):
            mask_shape = tuple(dataset.original_movies_shape[movie_index])
        else:
            mask_shape = tuple(dataset.movies_im[movie_index].shape)
        mask = np.zeros(mask_shape, np.uint8)
        movie_name = dataset.movie_names[movie_index]

        for prediction in predictions_by_movie.get(movie_index, []):
            _, time, x, y, z = prediction["center"]
            if "annotated_endpoints_raw_xyz" in prediction:
                p1, p2 = np.asarray(
                    prediction["annotated_endpoints_raw_xyz"], dtype=np.float64
                )
            elif "endpoints_raw_xyz" in prediction:
                p1, p2 = np.asarray(
                    prediction["endpoints_raw_xyz"], dtype=np.float64
                )
            else:
                length = float(
                    np.asarray(prediction["length"]).reshape(-1)[0]
                )
                quaternion = np.asarray(
                    prediction["rotation"], dtype=np.float64
                ).copy()
                p1, p2 = get_points_from_quat(
                    quaternion, np.asarray([x, y, z]), 0.5 * length
                )
            time = int(np.rint(time))
            if 0 <= time < mask.shape[0]:
                mask[time] = draw_line_in_matrix(
                    mask[time], p1, p2, np.min(mask[time].shape), val=255.0
                )

        # Internal T,X,Y,Z -> disk T,Z,Y,X.
        mask = np.swapaxes(mask, -1, -3)
        io.imsave(
            os.path.join(output_dir, f"{movie_name}.tif"),
            mask,
            check_contrast=False,
        )
