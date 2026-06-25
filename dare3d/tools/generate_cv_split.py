"""
From a given set folder containing
folder/
    im/
        0.tif
    label/
        0.tif
    
Generate all possible combination for a given list of division percent
n_div=0.1

folder/
    10/
        0/
            train/
                im/
                label/
            val/
                im/
                label/
        1/
            ...
        ...
    20/
        ...
"""

import random
from glob import glob
from pathlib import Path
import click
import os
import numpy as np
from skimage import io
from skimage.measure import regionprops

def get_data_pairs(im_folder, label_folder):
    movies_path = glob(os.path.join(im_folder, "*.tif"))

    # First load all movies paired with labels
    data_pairs = []
    for i in range(len(movies_path)):
        movie_path = movies_path[i]
        movie_name = Path(movie_path).stem
        label_path = os.path.join(label_folder, f"{movie_name}.tif")
        if not os.path.exists(label_path):
            raise ValueError(f"Could not find label path: {label_path}")
        data_pairs.append((movie_path,label_path))
    return data_pairs

def load_all_labels(data_pairs):
    return [io.imread(label_path) for _, label_path in data_pairs]

def get_all_divisions_labels(labels, start_index):
    divisions = []
    for i, label in enumerate(labels):
        for t in range(start_index, label.shape[0]):
            divisions += get_divisions(label[t])
    return divisions
        
def get_divisions(label):
    bipoints = []
    for i in range(np.max(label)//2):
        start = i*2 + 1
        end = (i+1)*2
        if np.count_nonzero(label[label==start]) == 0 or\
            np.count_nonzero(label[label==end]) == 0:
                label[label==start] = 0
                label[label==end] = 0

    regions = regionprops(label)
    points = []
    for i, props in enumerate(regions):
        if len(points) == 0:
            points = [props.label]
        else:
            points += [props.label]
            bipoints.append(points)
            points = []
    return bipoints

def get_random_percent(percent, labels, divisions):
    new_labels = [np.zeros_like(label) for label in labels]
    
    # Shuffle divisions
    random.shuffle(divisions)
    
    # Compute percent with current length
    percent_length = int(np.floor((percent / 100) * len(divisions)))
    
    selected_divisions = divisions[:percent_length]
    print(f"Selected {len(selected_divisions)} divisions")

    label = labels[0]
    for division in selected_divisions:
        for label_id in division:
            print(label_id)
            new_labels[0][label == label_id] = label_id
    print(f"Max value : {np.max(new_labels[0])}")
    return new_labels


@click.command()
@click.option(
    "--im_path",
    required=True,
    help="Path to the image movie (.tif).",
)
@click.option(
    "--label_path",
    required=True,
    help="Path to the label movie (.tif).",
)
@click.option(
    "--output",
    required=True,
    help="Output directory path.",
)
@click.option(
    "--start_index",
    required=False,
    default=0,
    help="The start index of the movie. Will ignore labels before the start index.",
)
def main(im_path, label_path, output, start_index):
    # list im,label pairs
    data_pairs = [(im_path, label_path)]

    # Load all labels
    labels = load_all_labels(data_pairs)

    # Get all divisions label
    divisions = get_all_divisions_labels(labels, start_index)
    
    print(divisions)
    print(f"Total number of divisions: {len(divisions)}")
    
    percent_step = 10
    max_repeat = 10
    percents = list(range(percent_step, 101, percent_step))
    
    print(percents)
    label_name = Path(label_path).stem
    im = io.imread(im_path)
    
    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)
    for percent in percents:
        percent_dir = os.path.join(output, str(percent))
        if not os.path.exists(percent_dir):
            os.makedirs(percent_dir, exist_ok=True)    
        current_n_repeat = int(np.ceil(100 / percent))
        print(f"Percent = {percent} ; Repeating {current_n_repeat} times")
        for n in range(current_n_repeat):
            repeat_dir = os.path.join(percent_dir, str(n))
            if not os.path.exists(repeat_dir):
                os.makedirs(repeat_dir, exist_ok=True)
                
            new_labels = get_random_percent(percent, labels, divisions)
            
            train_dir = os.path.join(repeat_dir, "train")
            im_dir = os.path.join(train_dir, "im")
            label_dir = os.path.join(train_dir, "label")
            
            if not os.path.exists(im_dir):
                os.makedirs(im_dir, exist_ok=True)
            if not os.path.exists(label_dir):
                os.makedirs(label_dir, exist_ok=True)
                
            label = new_labels[0]
            io.imsave(os.path.join(label_dir, f"{label_name}.tif"), label)
            io.imsave(os.path.join(im_dir, f"{label_name}.tif"), im)


if __name__ == "__main__":
    main()
