from glob import glob
from pathlib import Path
import napari
import click
import os
from skimage import io
import numpy as np

# def display_item(viewer, x, y, pred):
def display_item(viewer, x, pred, label, common):
    # x has shape T, X, Y, Z
    # y has shape 1, X, Y, Z
    viewer.add_image(x, name='im', colormap='magma', contrast_limits=[0.0, 2.0])
    # viewer.add_image(y[0], name='label', opacity=0.5)
    viewer.add_image(pred, name='pred', colormap="red", blending="additive")
    viewer.add_image(label, name='label', colormap="blue", blending="additive")
    viewer.add_image(common, name='common', colormap="green", blending="additive")

@click.command()
@click.option(
    "--log_dir",
    required=True,
    help="Path to the segmentation log directory.",
)
@click.option(
    "--max_display",
    required=False,
    default=10,
    help="Maximum number of segmentation results to display",
)
def main(log_dir, max_display):
    # list im files
    im_files = glob(os.path.join(log_dir, "im*.tif"))

    threshold = 0.2

    for i in range(min(max_display, len(im_files))):
        viewer = napari.Viewer()
        viewer.dims.ndisplay = 3

        im_file = im_files[i]
        index = Path(im_file).stem.split("im")[-1]
        pred_file = os.path.join(log_dir, f"pred{index}.tif")
        label_file = os.path.join(log_dir, f"label{index}.tif")
        
        im = io.imread(im_file)
        pred = io.imread(pred_file)
        label = io.imread(label_file)
        
        pred = (pred * 255).astype(np.uint8)
        label = (label * 255).astype(np.uint8)
        common = (np.logical_and(pred > threshold, label > 0) * 255).astype(np.uint8)
        
        display_item(viewer, im, pred, label, common)

        viewer.dims.current_step = (0, 0,) + viewer.dims.current_step[2:]

        viewer.show()
        napari.run()


if __name__ == "__main__":
    main()
