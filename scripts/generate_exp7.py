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
    
    movie_2_path = os.path.join(data_dir, "movie2")
    movie_3_path = os.path.join(data_dir, "movie3")
    movie_4_path = os.path.join(data_dir, "movie4")
    
    exp_dir = os.path.join(data_dir, "exp8")

    # Movie 2 and 3 vs Movie4
    create_exp(trains=[movie_2_path, movie_3_path], vals=[movie_4_path], output=os.path.join(exp_dir, "a"))
    # Movie 3 and 4 vs Movie 2
    create_exp(trains=[movie_3_path, movie_4_path], vals=[movie_2_path], output=os.path.join(exp_dir, "b"))    
    # Movie 4 and 2 vs Movie 3
    create_exp(trains=[movie_4_path, movie_2_path], vals=[movie_3_path], output=os.path.join(exp_dir, "c"))

if __name__ == "__main__":
    main()