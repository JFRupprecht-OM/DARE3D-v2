import os

import numpy as np
from skimage import io
from dare3d.data.components.angles3d import (
    draw_line_in_matrix,
    get_points_from_quat,
)

def display_regression(predictions, dataset, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    predictions_by_movie = {}
    for i in range(len(predictions)):
        pred = predictions[i]
        if pred is None:
            continue
        m,t,x,y,z = pred["center"]
        
        if m not in predictions_by_movie:
            predictions_by_movie[m] = []
        
        predictions_by_movie[m].append(pred)

    for movie_index in range(len(dataset.movies_im)):
        # Create a mask of the movie
        mask = np.zeros_like(dataset.movies_im[movie_index], np.uint8)
        movie_name = dataset.movie_names[movie_index]

        # Save empty mask
        if movie_index not in predictions_by_movie:
            io.imsave(os.path.join(output_dir, f"{movie_name}.tif"), mask, check_contrast=False)
            continue

        for center in predictions_by_movie[movie_index]:
            length = center["length"]
            quat = center["rotation"]
            _,t,x,y,z = center["center"]
            t,x,y,z = int(t), int(x), int(y), int(z)
            # Draw the prediction
            p1, p2 = get_points_from_quat(quat, np.asarray([x,y,z]), 0.5 * length)
            mask[t] = draw_line_in_matrix(mask[t], p1, p2, np.min(mask[t].shape), val=255.0)
        
        # From T, X, Y, Z to T, Z, Y, X to get original movie order dim
        mask = np.swapaxes(mask, -1, -3)
        io.imsave(os.path.join(output_dir, f"{movie_name}.tif"), mask, check_contrast=False)