"""Small filesystem helpers for the core ``dare3d`` package.

``iter_tifs`` is a behaviourally identical copy of the one in
``napari_dare3d/_io.py``; the plugin keeps its own copy so that loading the napari
plugin never imports ``dare3d`` (and thus torch). This module lives under
``dare3d.utils`` and is free to be imported by core code, which already loads torch.
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
