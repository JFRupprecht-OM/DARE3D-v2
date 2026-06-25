from __future__ import annotations

import shutil
import tempfile
import urllib.request
from importlib import resources
from pathlib import Path
from typing import Iterable

_SENTINELS: set[str] = {".keep", ".gitkeep", ".placeholder"}  # files that *don’t* count as data


def _is_effectively_empty(path: Path, sentinels: Iterable[str]) -> bool:
    """True if folder contains nothing except sentinel files."""
    try:
        return all(p.name in sentinels for p in path.iterdir())
    except FileNotFoundError:
        return True


def get_path_to_demo_folder() -> Path:
    """
    Ensure `dare3d/demo_files` contains data; download if still empty.
    Returns
    -------
    pathlib.Path pointing to the demo directory (guaranteed to exist).
    """
    package = "dare3d"
    subfolder = "demo_files"
    url = "https://zenodo.org/records/19111753/files/DARE3d_data_190326.zip?download=1"
    sentinels = _SENTINELS
    # ── locate a *writable* directory ────────────────────────────────
    try:
        base = resources.files(package)         # works for wheels *and* -e installs
        data_dir = base / subfolder
    except (ModuleNotFoundError, FileNotFoundError):
        tmp_root = Path(tempfile.gettempdir()) / f"{package}_data"
        tmp_root.mkdir(exist_ok=True)
        data_dir = tmp_root

    # ── download if still empty ──────────────────────────────────────
    if _is_effectively_empty(data_dir, sentinels):
        print(f"🔽 First run – downloading data into {data_dir} …")
        data_dir.mkdir(parents=True, exist_ok=True)
        tmp = data_dir / "payload.zip"
        
        urllib.request.urlretrieve(url, tmp)     # <- simple, std-lib only

        shutil.unpack_archive(tmp, data_dir)
        tmp.unlink()                         # remove archive after unpack

        nested = data_dir / "demo_files"
        if nested.is_dir():
            for child in nested.iterdir():
                dest = data_dir / child.name   # move up one level
                if dest.exists():
                    # overwrite files or merge dirs if they already exist
                    if dest.is_dir() and child.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                child.rename(dest)
            nested.rmdir()                     # remove the now-empty wrapper

        print("✅ Data ready")
    else:
        print(f"✅ Using existing data in {data_dir}")

    return data_dir
