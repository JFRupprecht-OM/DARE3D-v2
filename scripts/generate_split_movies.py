import os
import click
from glob import glob
from pathlib import Path
from skimage import io

from generation_utils import create_exp

def split_movie_label(movie, label):
    half = movie.shape[-1] // 2
    left_movie = movie[..., :half]
    left_label = label[..., :half]
    
    right_movie = movie[..., half:]
    right_label = label[..., half:]
    
    return left_movie, left_label, right_movie, right_label

def split_movies(folder_in, folder_out, name):
    folder_out_a_im = os.path.join(folder_out, f"{name}_a", "im")
    folder_out_a_label = os.path.join(folder_out, f"{name}_a", "label")
    folder_out_b_im = os.path.join(folder_out, f"{name}_b", "im")
    folder_out_b_label = os.path.join(folder_out, f"{name}_b", "label")

    os.makedirs(folder_out_a_im, exist_ok=True)
    os.makedirs(folder_out_a_label, exist_ok=True)
    os.makedirs(folder_out_b_im, exist_ok=True)
    os.makedirs(folder_out_b_label, exist_ok=True)

    # Read each movie
    movies = glob(os.path.join(folder_in, "im", "*.tif"))
    for movie_path in movies:
        movie_name = Path(movie_path).stem
        label_path = os.path.join(folder_in, "label", f"{movie_name}.tif")
        
        # Load image and label
        movie = io.imread(movie_path)
        label = io.imread(label_path)
        
        # Split in half
        movie_a, label_a, movie_b, label_b = split_movie_label(movie, label)
        
        # Write splits in corresponding folders
        io.imsave(os.path.join(folder_out_a_im, f"{movie_name}_a.tif"), movie_a)
        io.imsave(os.path.join(folder_out_a_label, f"{movie_name}_a.tif"), label_a)
        
        io.imsave(os.path.join(folder_out_b_im, f"{movie_name}_b.tif"), movie_b)
        io.imsave(os.path.join(folder_out_b_label, f"{movie_name}_b.tif"), label_b)

@click.command()
@click.option(
    "--data_dir",
    required=True,
    help="Path to the data directory where movies are stored.",
)
def main(data_dir):
    movie_1_path = os.path.join(data_dir, "movie1")
    movie_2_path = os.path.join(data_dir, "movie2")
    movie_3_path = os.path.join(data_dir, "movie3")
    
    split_movies(movie_1_path, data_dir, "movie1")
    split_movies(movie_2_path, data_dir, "movie2")
    split_movies(movie_3_path, data_dir, "movie3")

if __name__ == "__main__":
    main()