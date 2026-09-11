"""Locate (or download) the DARE3D v2 data + pretrained-model bundle.

Thin wrapper over the shared Zenodo downloader in :mod:`napari_dare3d._data`
(record 22639669, ``DARE3dv2_Zenodo_040926.zip``). Both modules ship in the same
distribution and are stdlib-light, so the dare3d -> napari_dare3d import here does
not pull in napari or torch.
"""
from __future__ import annotations

from pathlib import Path


def get_path_to_demo_folder() -> Path:
    """Return the ``DARE3dv2_Zenodo_040926`` bundle folder, downloading it if absent.

    The bundle is looked up where the napari plugin looks for it (current directory,
    then the repository root). If it is not found, it is downloaded, md5-verified and
    unpacked into the repository root (or the current directory outside a checkout)
    with the same code path as the ``dare3d-download`` command.

    Returns
    -------
    pathlib.Path pointing to the bundle directory (guaranteed to exist).
    """
    from napari_dare3d._data import default_dest, download
    from napari_dare3d._release_models import find_release_root

    found = find_release_root()
    if found is not None:
        print(f"Using existing data in {found}")
        return found
    return download(default_dest())
