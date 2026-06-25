import os
import click

from generation_utils import create_exp

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
    
    exp_dir = os.path.join(data_dir, "exp2")

    # Movie 1 & 2 vs Movie 3
    create_exp(trains=[movie_1_path, movie_2_path], vals=[movie_3_path], output=os.path.join(exp_dir, "a"))
    # Movie 2 & 3 vs Movie 1
    create_exp(trains=[movie_2_path, movie_3_path], vals=[movie_1_path], output=os.path.join(exp_dir, "b"))
    # Movie 1 & 3 vs Movie 2
    create_exp(trains=[movie_1_path, movie_3_path], vals=[movie_2_path], output=os.path.join(exp_dir, "c"))

if __name__ == "__main__":
    main()