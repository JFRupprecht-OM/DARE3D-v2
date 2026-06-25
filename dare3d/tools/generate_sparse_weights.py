"""
Generate matrix weights from sparsely annotated data

Currently supported tubular weights between two daughter cells
"""

import click
import os
import numpy as np
from tqdm import tqdm
from skimage import io
from skimage.measure import regionprops

def get_divisions(label):
    bipoints = []
    for i in tqdm(range(np.max(label)//2), desc="Filtering labels...", leave=False):
        start = i*2 + 1
        end = (i+1)*2
        if np.count_nonzero(label[label==start]) == 0 or\
            np.count_nonzero(label[label==end]) == 0:
                label[label==start] = 0
                label[label==end] = 0

    regions = regionprops(label)
    points = []
    for i, props in tqdm(enumerate(regions), desc="Gathering connected components...", leave=False):
        if len(points) == 0:
            points = [props.centroid]
        else:
            points += [props.centroid]
            bipoints.append(points)
            points = []
    return bipoints

def compute_division_weight(division, mat, radius):
    p1, p2 = division

    def fill_cylinder_matrix(mat, p1, p2, radius):
        # Ensure p1 and p2 are numpy arrays
        p1 = np.array(p1)
        p2 = np.array(p2)

        # Get the vector representing the axis of the cylinder
        axis_vector = p2 - p1

        # Get the length of the cylinder
        cylinder_length = np.linalg.norm(axis_vector)

        # Normalize the axis vector
        axis_vector = axis_vector / cylinder_length

        # Create a grid of points representing the matrix
        grid = np.mgrid[0:mat.shape[0], 0:mat.shape[1], 0:mat.shape[2]]
        points = np.vstack((grid[0].ravel(), grid[1].ravel(), grid[2].ravel())).T

        # Calculate vectors from the cylinder start point to the matrix points
        vectors_to_points = points - p1

        # Project the vectors onto the axis of the cylinder
        projections = np.dot(vectors_to_points, axis_vector)

        # Calculate distances from the points to the axis
        distances_to_axis = np.linalg.norm(vectors_to_points - projections[:, np.newaxis] * axis_vector, axis=1)

        # Find points within the cylinder
        within_cylinder = (0 <= projections) & (projections <= cylinder_length) & (distances_to_axis <= radius)

        # Reshape the result to the original matrix shape
        mat = np.maximum(mat, within_cylinder.reshape(mat.shape))
        return mat
    
    return fill_cylinder_matrix(mat, p1, p2, radius)


def compute_weights(divisions, mat_shape, radius):
    mat_weights = np.zeros((mat_shape)) 
   
    for division in tqdm(divisions, desc="Computing division weights...", leave=False):
        mat_weights = compute_division_weight(division, mat_weights, radius)

    return mat_weights

def compute_sparse_weights(label_path, radius):
    label = io.imread(label_path)
    label_weights = np.zeros_like(label)
    for label_t in tqdm(range(label.shape[0]), desc="Processing movie timestamps..."):
        divisions = get_divisions(label[label_t])
        label_weights[label_t] = compute_weights(divisions, label.shape[1:], radius)
    return label_weights

@click.command()
@click.option(
    "--annotated_movie",
    required=True,
    help="Path to the annotated movie (.tif).",
)
@click.option(
    "--radius",
    required=False,
    default=4,
    help="Radius of the cylindrical shape.",
)
def main(annotated_movie, radius):
    if not os.path.exists(annotated_movie):
        raise ValueError(f"File not found: {annotated_movie}")

    label_weights = compute_sparse_weights(label_path=annotated_movie, radius=radius)
    io.imsave("weights.tif", label_weights)

if __name__ == "__main__":
    main()
