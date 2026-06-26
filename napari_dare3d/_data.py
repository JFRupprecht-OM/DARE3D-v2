"""Download the DARE3D demo data + pretrained models from Zenodo (record 19113351).

Stdlib only (``urllib`` + ``zipfile``) so it adds **no** install-time dependency.
Used by the napari "Download DARE3D data" button; also runnable as a script::

    python -m napari_dare3d._data [DEST]
"""
from __future__ import annotations

import json
import sys
import urllib.request
import zipfile
from pathlib import Path

#: Zenodo record holding ``DARE3d_data_190326.zip`` (models + demo movies).
ZENODO_RECORD = "19113351"
DATA_NAME = "DARE3d_data_190326"
_API = "https://zenodo.org/api/records/{}"


def zenodo_files(record_id: str = ZENODO_RECORD):
    """Return the record's file list (each has ``key``, ``size``, ``links.self``)."""
    with urllib.request.urlopen(_API.format(record_id), timeout=60) as resp:
        return json.load(resp)["files"]


def _is_junk(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    return name.startswith("__MACOSX") or base == ".DS_Store" or base.startswith("._")


def _download(url: str, out: Path, total, progress_cb=None) -> None:
    """Stream ``url`` to ``out`` (atomic via a .part temp), reporting bytes done."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    done = 0
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as fh:
        while True:
            buf = resp.read(1 << 20)  # 1 MiB
            if not buf:
                break
            fh.write(buf)
            done += len(buf)
            if progress_cb is not None:
                progress_cb(done, total or 0)
    tmp.replace(out)


def download(dest, progress_cb=None, log=print) -> Path:
    """Download record 19113351 into ``dest`` and unzip it.

    Returns the path to the extracted ``DARE3d_data_190326`` folder. Files already
    present at the expected size are skipped, so a re-run resumes rather than
    re-downloading.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for f in zenodo_files():
        out = dest / f["key"]
        size = f.get("size")
        if out.exists() and (size is None or out.stat().st_size == size):
            log(f"already present: {f['key']}")
        else:
            log(f"downloading {f['key']} ({(size or 0) / 1e9:.2f} GB) — this can take a while…")
            _download(f["links"]["self"], out, size, progress_cb)
        if f["key"].lower().endswith(".zip"):
            log(f"extracting {f['key']}…")
            with zipfile.ZipFile(out) as z:
                for m in z.namelist():
                    if not _is_junk(m):
                        z.extract(m, dest)
    extracted = dest / DATA_NAME
    log(f"done → {extracted}")
    return extracted


if __name__ == "__main__":
    download(sys.argv[1] if len(sys.argv) > 1 else Path.cwd())
