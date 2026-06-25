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
    movie_1_a_path = os.path.join(data_dir, "movie1_a")
    movie_1_b_path = os.path.join(data_dir, "movie1_b")
    movie_2_a_path = os.path.join(data_dir, "movie2_a")
    movie_2_b_path = os.path.join(data_dir, "movie2_b")
    movie_3_a_path = os.path.join(data_dir, "movie3_a")
    movie_3_b_path = os.path.join(data_dir, "movie3_b")
    
    exp_dir = os.path.join(data_dir, "exp5")

    # Movie 1_a 2_a 3_a vs Movie 3_b
    create_exp(trains=[movie_1_path, movie_2_path, movie_3_a_path], vals=[movie_3_b_path], output=os.path.join(exp_dir, "a"))
    # Movie 2_a 3_a 1_a vs Movie 1_b
    create_exp(trains=[movie_2_path, movie_3_path, movie_1_a_path], vals=[movie_1_b_path], output=os.path.join(exp_dir, "b"))    
    # Movie 1_a 3_a 2_a vs Movie 2_b
    create_exp(trains=[movie_1_path, movie_3_path, movie_2_a_path], vals=[movie_2_b_path], output=os.path.join(exp_dir, "c"))

if __name__ == "__main__":
    main()