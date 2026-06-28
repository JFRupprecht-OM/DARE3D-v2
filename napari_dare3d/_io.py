"""Tiny, import-light filesystem helpers for the napari plugin.

Kept dependency-free (stdlib only) on purpose: ``napari_dare3d._widget`` imports
it, and merely loading the plugin (napari plugin discovery) must not pull in torch
or ``dare3d``. A behaviourally identical copy lives in ``dare3d/utils/io.py`` for
the core package (which already loads torch, so it cannot share this module without
re-introducing that heavy import here).
"""
from pathlib import Path
from typing import List


def iter_tifs(directory, prefix: str = "") -> List[str]:
    """Sorted ``.tif``/``.tiff`` paths in *directory* whose filename starts with
    *prefix*, matched **case-insensitively** on the extension.

    Drop-in replacement for ``glob(os.path.join(directory, prefix + "*.tif"))``,
    which silently skips uppercase ``.TIF``/``.TIFF`` on case-sensitive filesystems
    (Linux/macOS) and misses ``.tiff`` entirely. Returns string paths (like ``glob``).
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    hits = [
        p for p in directory.iterdir()
        if p.is_file()
        and p.suffix.lower() in (".tif", ".tiff")
        and p.name.startswith(prefix)
    ]
    return sorted(str(p) for p in hits)
