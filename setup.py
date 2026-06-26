#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="dare3d",
    version="0.0.1",
    description="DARE3D: Division Axis and Region Estimation in 3D time-lapse images (core + napari plugin)",
    author="Romain Karpinski, Marc Karnat, Alice Gros, Qazi Saaheelur Rahaman, "
    "Jules Vanaret, Mehdi Saadaoui, Sham Tlili, Jean-Francois Rupprecht",
    author_email="rupprecht.jf@gmail.com",
    url="https://github.com/qazi05/DARE3d",
    packages=find_packages(),
    include_package_data=True,
    # ship the napari manifest with the plugin package
    package_data={"napari_dare3d": ["napari.yaml"]},
    # console commands (core) + the napari plugin manifest (so napari discovers the plugin)
    entry_points={
        "console_scripts": [
            "train_command = dare3d.train:main",
            "eval_command = dare3d.eval:main",
        ],
        "napari.manifest": [
            "dare3d = napari_dare3d:napari.yaml",
        ],
    },
)
