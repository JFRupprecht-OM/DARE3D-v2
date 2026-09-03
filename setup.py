#!/usr/bin/env python

from setuptools import find_packages, setup

# NOTE: torch / torchvision / torchaudio are intentionally NOT declared here — they
# must be installed separately with the right CUDA build (see the README). Declaring
# them would risk overwriting a user's CUDA wheel with a default PyPI build.
INSTALL_REQUIRES = [
    # numeric / image
    "numpy>=1.23,<1.27",
    "scipy",
    "numba>=0.59,<0.61",
    "scikit-image",
    "tifffile",
    "matplotlib",
    "tqdm",
    # config / CLI / utils
    "hydra-core>=1.3,<1.4",
    "hydra-colorlog>=1.2",
    "hydra-optuna-sweeper>=1.2",
    "omegaconf",
    "rootutils",
    "rich",
    "click",
    # training / models / metrics
    "lightning>=2.0",
    "torchmetrics>=0.11.4",
    "monai>=1.3,<1.4",
    "augmend",
    # loggers
    "mlflow",
    "tensorboard",
    # napari plugin (GUI)
    "napari",
    "magicgui",
]

setup(
    name="dare3d",
    version="0.0.1",
    description="DARE3D: Division Axis and Region Estimation in 3D time-lapse images (core + napari plugin)",
    author="Romain Karpinski, Alice Gros, Marc Karnat, Qazi Saaheelur Rahaman, "
    "Jules Vanaret, Mehdi Saadaoui, Sham Tlili, Jean-Francois Rupprecht",
    author_email="rupprecht.jf@gmail.com",
    url="https://github.com/qazi05/DARE3d",
    python_requires=">=3.10",
    # exclude the test suite from the built distribution
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    # ship the napari manifest with the plugin package, and the Hydra config tree
    # (needed by the train/eval/predict CLIs) which is otherwise dropped from wheels.
    package_data={
        "napari_dare3d": ["napari.yaml"],
        "configs": ["**/*.yaml", "**/*.yml"],
    },
    # Keep the historical project-level path available in wheel installs as
    # <environment>/data/3D/scales.json as well as in source checkouts/sdists.
    data_files=[("data/3D", ["data/3D/scales.json"])],
    install_requires=INSTALL_REQUIRES,
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
