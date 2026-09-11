"""Download, verify and unpack the DARE3D v2 data + pretrained-model bundle from Zenodo.

Zenodo record 22639669 (DOI 10.5281/zenodo.22639669) ships one archive,
``DARE3dv2_Zenodo_040926.zip`` (10.2 GB), which unpacks to the
``DARE3dv2_Zenodo_040926/`` folder (24 GB) that the napari presets, the notebooks
and the README expect at the repository root.

The transfer is stdlib only (``urllib`` + ``zipfile`` + ``hashlib``) so it adds no
install-time dependency; ``click`` (already required by dare3d) is used only by the
command-line entry point. Used by the napari "DARE3D download data" button and as
a CLI::

    dare3d-download [--dest DIR] [--keep-archive] [--dry-run]
    python -m napari_dare3d._data [same options]

Behaviour on re-runs: an existing ``DARE3dv2_Zenodo_040926/`` is never touched; an
existing ``.zip`` is verified and unpacked; a ``.zip.part`` resumes with an HTTP
Range request; a ``.tmp`` staging folder resumes the extraction. The MD5 is
computed while streaming and compared with the checksum pinned below (which is
also cross-checked against the Zenodo API before the transfer starts).
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import click

from napari_dare3d._release_models import RELEASE_ROOT_NAME

#: Zenodo record holding the v2 bundle (a version DOI: its file list is immutable).
ZENODO_RECORD = "22639669"
ZENODO_DOI = "10.5281/zenodo.22639669"
ZENODO_CONCEPT_DOI = "10.5281/zenodo.22639668"
ZENODO_URL = f"https://zenodo.org/records/{ZENODO_RECORD}"

#: Name of the unpacked folder (shared with the napari release presets).
BUNDLE_NAME = RELEASE_ROOT_NAME
ARCHIVE_NAME = BUNDLE_NAME + ".zip"
#: Published archive size / MD5 (Zenodo API ``files[].size`` / ``files[].checksum``).
ARCHIVE_SIZE = 10_212_593_678
ARCHIVE_MD5 = "7f283c1100d07d67ae3aaeb93c4ca90a"
#: Sum of the member sizes (recomputed from the archive before unpacking).
UNPACKED_SIZE = 23_994_519_053

_API = "https://zenodo.org/api/records/{}"
_CONTENT_URL = _API.format(ZENODO_RECORD) + f"/files/{ARCHIVE_NAME}/content"
_CHUNK = 1 << 20  # 1 MiB
_RETRIES = 5
_WINDOWS_MAX_PATH = 260


class DownloadError(RuntimeError):
    """A download, verification or extraction problem with a user-facing message."""


class _IncompleteDownload(DownloadError):
    """The server closed the stream early; retrying resumes from the partial file."""


# --------------------------------------------------------------------------- helpers
def _gb(n: float) -> str:
    return f"{n / 1e9:.2f} GB"


def _open(url: str, headers: Optional[dict] = None, timeout: int = 60):
    """Single HTTP seam (tests replace this)."""
    req = urllib.request.Request(url, headers=headers or {})
    return urllib.request.urlopen(req, timeout=timeout)


def _is_junk(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    return name.startswith("__MACOSX") or base == ".DS_Store" or base.startswith("._")


def _hash_file(path: Path):
    """Return an md5 object fed with ``path`` (streamed in 1 MiB chunks)."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h


def _existing_parent(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if candidate.exists():
            return candidate
    return path


def _free_space(path: Path) -> int:
    return shutil.disk_usage(_existing_parent(Path(path).resolve())).free


def _require_free_space(dest: Path, needed: int) -> None:
    free = _free_space(dest)
    if free < needed:
        raise DownloadError(
            f"not enough free space in {dest}: need about {_gb(needed)}, have {_gb(free)}. "
            "Free up space or pass --dest pointing to a larger drive."
        )


def _max_path_limit() -> Optional[int]:
    """Return the path-length limit to enforce (Windows without long paths), else None."""
    if os.name != "nt":
        return None
    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem"
        )
        with key:
            enabled, _kind = winreg.QueryValueEx(key, "LongPathsEnabled")
        if int(enabled) == 1:
            return None
    except (ImportError, OSError, ValueError, TypeError):
        pass
    return _WINDOWS_MAX_PATH


def _find_project_root(start: Path) -> Optional[Path]:
    for d in (start, *start.parents):
        if (d / ".project-root").is_file():
            return d
    return None


def default_dest() -> Path:
    """Directory that will contain the bundle: the repo root if detectable, else cwd.

    The repository root is recognised by the ``.project-root`` marker (the same
    file ``rootutils`` uses), searched upward from the current directory and then
    at the parent of this package (an editable install). A wheel install with no
    marker falls back to the current directory.
    """
    root = _find_project_root(Path.cwd().resolve())
    if root is None:
        pkg_parent = Path(__file__).resolve().parents[1]
        if (pkg_parent / ".project-root").is_file():
            root = pkg_parent
    return root if root is not None else Path.cwd()


def percent_logger(log: Callable[[str], None] = print, label: str = "download", step: int = 5):
    """Return a ``progress_cb(done, total)`` that logs an ASCII line every ``step`` %."""
    state = {"pct": -step}

    def _cb(done: int, total: int) -> None:
        if not total:
            return
        pct = int(100 * done / total)
        if pct >= state["pct"] + step or (pct == 100 and state["pct"] < 100):
            state["pct"] = pct
            log(f"{label} {pct:3d}%  ({_gb(done)} / {_gb(total)})")

    return _cb


# --------------------------------------------------------------------------- Zenodo API
def zenodo_files(record_id: str = ZENODO_RECORD) -> list:
    """Return the record's file list (each has ``key``, ``size``, ``checksum``, ``links.self``)."""
    with _open(_API.format(record_id)) as resp:
        return json.load(resp)["files"]


def archive_entry(files: list) -> dict:
    """Pick the bundle archive from the API file list and cross-check it with the pins."""
    for entry in files:
        if entry.get("key") != ARCHIVE_NAME:
            continue
        problems = []
        size = entry.get("size")
        checksum = str(entry.get("checksum", "")).lower()
        if size != ARCHIVE_SIZE:
            problems.append(f"size {size} != pinned {ARCHIVE_SIZE}")
        if checksum != "md5:" + ARCHIVE_MD5:
            problems.append(f"checksum {checksum!r} != pinned 'md5:{ARCHIVE_MD5}'")
        if problems:
            raise DownloadError(
                f"Zenodo record {ZENODO_RECORD} disagrees with the constants pinned in "
                f"napari_dare3d/_data.py ({'; '.join(problems)}); refusing to download. "
                "Please report this or update the pinned values."
            )
        return entry
    keys = ", ".join(str(e.get("key")) for e in files) or "none"
    raise DownloadError(
        f"{ARCHIVE_NAME} not found in Zenodo record {ZENODO_RECORD} (files: {keys})."
    )


# --------------------------------------------------------------------------- transfer
def _download(url: str, out: Path, size: int, md5: str, progress_cb=None, log=print) -> None:
    """Stream ``url`` to ``out`` via ``out.part`` (resumable), verifying the md5 on the way."""
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    offset = part.stat().st_size if part.is_file() else 0
    if offset > size:
        log("partial file is larger than the archive; discarding it")
        part.unlink()
        offset = 0

    h = hashlib.md5()
    if offset:
        log(f"resuming at {_gb(offset)}; hashing the existing partial file first...")
        h = _hash_file(part)
    done = offset

    if offset < size:
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        mode = "ab" if offset else "wb"
        with _open(url, headers=headers) as resp:
            status = getattr(resp, "status", None) or resp.getcode()
            if offset and status == 200:
                log("server ignored the Range request; restarting from 0")
                offset, done, mode, h = 0, 0, "wb", hashlib.md5()
            elif offset and status == 206:
                content_range = resp.headers.get("Content-Range", "")
                try:
                    start = int(content_range.split()[1].split("-")[0])
                except (IndexError, ValueError):
                    start = None
                if start != offset:
                    raise DownloadError(
                        f"unexpected Content-Range {content_range!r} for offset {offset}; "
                        f"delete {part} and rerun."
                    )
            with open(part, mode) as fh:
                while True:
                    buf = resp.read(_CHUNK)
                    if not buf:
                        break
                    fh.write(buf)
                    h.update(buf)
                    done += len(buf)
                    if progress_cb is not None:
                        progress_cb(done, size)

    if done != size:
        raise _IncompleteDownload(
            f"incomplete download: {done:,} of {size:,} bytes; rerun to resume."
        )
    digest = h.hexdigest()
    if digest != md5:
        part.unlink(missing_ok=True)
        raise DownloadError(
            f"md5 mismatch for {out.name}: got {digest}, expected {md5}. "
            "The partial file was removed; rerun to download again."
        )
    part.replace(out)


def _retrying_download(url, out, size, md5, progress_cb, log) -> None:
    """Call :func:`_download` up to ``_RETRIES`` times on transient network errors."""
    for attempt in range(1, _RETRIES + 1):
        try:
            _download(url, out, size, md5, progress_cb, log)
            return
        except urllib.error.HTTPError as exc:
            if exc.code != 429 and exc.code < 500:
                raise DownloadError(
                    f"HTTP {exc.code} from Zenodo ({exc.reason}) for {url}."
                ) from exc
            err: Exception = exc
        except (
            _IncompleteDownload,
            urllib.error.URLError,
            http.client.HTTPException,
            TimeoutError,
            ConnectionError,
        ) as exc:
            err = exc
        if attempt == _RETRIES:
            raise DownloadError(
                f"download failed after {_RETRIES} attempts ({err}); "
                "rerun to resume from the partial file."
            ) from err
        delay = 5 * attempt
        log(f"attempt {attempt}/{_RETRIES} failed ({err}); retrying in {delay} s...")
        time.sleep(delay)


def verify_archive(archive: Path, log=print) -> None:
    """Check size then md5 of an archive already on disk."""
    actual = archive.stat().st_size
    if actual != ARCHIVE_SIZE:
        raise DownloadError(
            f"{archive} has {actual:,} bytes, expected {ARCHIVE_SIZE:,}. Delete it, or "
            f"rename it to {archive.name}.part to resume the download, then rerun."
        )
    log(f"verifying md5 of {archive.name} ({_gb(actual)})...")
    digest = _hash_file(archive).hexdigest()
    if digest != ARCHIVE_MD5:
        raise DownloadError(
            f"md5 mismatch for {archive}: got {digest}, expected {ARCHIVE_MD5}. "
            "Delete it and rerun."
        )
    log("checksum OK")


# --------------------------------------------------------------------------- extraction
def _check_layout(members) -> int:
    """Every member must live under ``BUNDLE_NAME/`` with no ``..``; return total bytes."""
    prefix = BUNDLE_NAME + "/"
    total = 0
    for info in members:
        name = info.filename
        if not name.startswith(prefix) or ".." in name.split("/") or "\\" in name:
            raise DownloadError(
                f"unexpected archive member {name!r}: this does not look like "
                f"{ARCHIVE_NAME}. Nothing was extracted."
            )
        if not info.is_dir():
            total += info.file_size
    return total


def extract_archive(archive: Path, dest: Path, log=print) -> Path:
    """Unpack ``archive`` into ``dest/BUNDLE_NAME`` via a staging folder + atomic rename."""
    final = dest / BUNDLE_NAME
    staging = dest / (BUNDLE_NAME + ".tmp")
    prefix = BUNDLE_NAME + "/"
    with zipfile.ZipFile(archive) as z:
        members = [i for i in z.infolist() if not _is_junk(i.filename)]
        total = _check_layout(members)
        already = 0
        if staging.is_dir():
            for info in members:
                target = staging / info.filename[len(prefix) :]
                if not info.is_dir() and target.is_file():
                    if target.stat().st_size == info.file_size:
                        already += info.file_size
        _require_free_space(dest, max(total - already, 0))
        limit = _max_path_limit()
        if limit is not None:
            longest = max(
                (len(str(staging / info.filename[len(prefix) :])) for info in members),
                default=0,
            )
            if longest >= limit:
                raise DownloadError(
                    f"an extracted path would be {longest} characters long (Windows limit "
                    f"{limit - 1}). Clone the repository to a shorter path, pass --dest, or "
                    "enable Windows long paths (LongPathsEnabled) and rerun."
                )
        staging.mkdir(parents=True, exist_ok=True)
        n = len(members)
        step = max(1, n // 10)
        for k, info in enumerate(members, 1):
            rel = info.filename[len(prefix) :]
            if not rel:
                continue
            target = staging / rel
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif not (target.is_file() and target.stat().st_size == info.file_size):
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst, _CHUNK)
            if k % step == 0 or k == n:
                log(f"unpacking {k}/{n} entries")
    try:
        os.replace(staging, final)
    except OSError as exc:
        raise DownloadError(
            f"could not rename {staging} to {final}: {exc}. Close programs using that "
            "folder and rerun."
        ) from exc
    return final


def write_provenance(dest: Path, archive_kept: bool) -> Path:
    """Write ``dest/BUNDLE_NAME.provenance.json`` (sidecar; the bundle itself stays pristine)."""
    path = dest / (BUNDLE_NAME + ".provenance.json")
    payload = {
        "bundle": BUNDLE_NAME,
        "zenodo_record": ZENODO_RECORD,
        "record_url": ZENODO_URL,
        "doi": ZENODO_DOI,
        "concept_doi": ZENODO_CONCEPT_DOI,
        "archive": ARCHIVE_NAME,
        "archive_bytes": ARCHIVE_SIZE,
        "archive_md5": ARCHIVE_MD5,
        "archive_kept": bool(archive_kept),
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": "napari_dare3d._data",
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- orchestration
def describe_plan(dest=None) -> list:
    """Offline description of what :func:`download` would do (for ``--dry-run``)."""
    dest = Path(dest).expanduser() if dest is not None else default_dest()
    final = dest / BUNDLE_NAME
    archive = dest / ARCHIVE_NAME
    part = dest / (ARCHIVE_NAME + ".part")
    staging = dest / (BUNDLE_NAME + ".tmp")
    lines = [
        f"destination: {dest}",
        f"bundle:      {final}",
        f"source:      Zenodo record {ZENODO_RECORD} ({ARCHIVE_NAME}, {_gb(ARCHIVE_SIZE)}, "
        f"md5 {ARCHIVE_MD5})",
    ]
    if final.is_dir():
        lines.append("state:       bundle already present - nothing to do")
        return lines
    if archive.is_file():
        state = f"archive present ({archive.stat().st_size:,} bytes) - will verify and unpack"
        to_download = 0
    elif part.is_file():
        have = part.stat().st_size
        state = f"partial download ({have:,} of {ARCHIVE_SIZE:,} bytes) - will resume"
        to_download = max(ARCHIVE_SIZE - have, 0)
    else:
        state = "fresh - nothing downloaded yet"
        to_download = ARCHIVE_SIZE
    if staging.is_dir():
        state += "; partial extraction found - will resume it"
    free = _free_space(dest)
    needed = to_download + UNPACKED_SIZE
    lines += [
        f"state:       {state}",
        f"to download: {_gb(to_download)}",
        f"unpacked:    {_gb(UNPACKED_SIZE)}",
        f"free space:  {_gb(free)}",
        "verdict:     OK"
        if free >= needed
        else f"verdict:     INSUFFICIENT SPACE (need about {_gb(needed)})",
    ]
    return lines


def download(dest=None, progress_cb=None, log=print, *, keep_archive: bool = False) -> Path:
    """Download (resumable), verify (md5) and unpack the bundle; return the bundle folder.

    ``dest`` is the directory that will contain ``DARE3dv2_Zenodo_040926/`` (default:
    :func:`default_dest`). An existing bundle folder is returned untouched. The
    archive is deleted after a successful extraction unless ``keep_archive``.
    """
    dest = Path(dest).expanduser() if dest is not None else default_dest()
    final = dest / BUNDLE_NAME
    if final.is_dir():
        log(f"already present: {final} (delete this folder to download again)")
        return final
    dest.mkdir(parents=True, exist_ok=True)

    archive = dest / ARCHIVE_NAME
    if archive.is_file():
        verify_archive(archive, log)
    else:
        part = dest / (ARCHIVE_NAME + ".part")
        have = part.stat().st_size if part.is_file() else 0
        _require_free_space(dest, max(ARCHIVE_SIZE - have, 0) + UNPACKED_SIZE)
        log(f"querying Zenodo record {ZENODO_RECORD}...")
        entry = archive_entry(zenodo_files())
        url = (entry.get("links") or {}).get("self") or _CONTENT_URL
        log(
            f"downloading {ARCHIVE_NAME} ({_gb(ARCHIVE_SIZE)}) from {ZENODO_URL} into {dest} "
            "- this can take a while"
        )
        _retrying_download(url, archive, ARCHIVE_SIZE, ARCHIVE_MD5, progress_cb, log)
        log("checksum OK")

    log(f"unpacking {ARCHIVE_NAME} into {final} ({_gb(UNPACKED_SIZE)})...")
    final = extract_archive(archive, dest, log)
    if keep_archive:
        log(f"kept archive: {archive}")
    else:
        archive.unlink()
        log("removed archive (pass --keep-archive to retain it)")
    provenance = write_provenance(dest, keep_archive)
    log(f"provenance: {provenance}")
    log(f"done: {final}")
    return final


# --------------------------------------------------------------------------- CLI
@click.command(name="dare3d-download", context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--dest",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=f"Directory that will contain {BUNDLE_NAME}/ (default: the repository root found "
    "via its .project-root marker, else the current directory).",
)
@click.option(
    "--keep-archive",
    is_flag=True,
    help="Keep the verified .zip after unpacking (default: delete it to free 10 GB).",
)
@click.option("--dry-run", is_flag=True, help="Only report what would happen; no network.")
def main(dest: Optional[Path], keep_archive: bool, dry_run: bool) -> None:
    """Download, verify (md5) and unpack the DARE3D v2 data + pretrained-model bundle.

    Source: Zenodo record 22639669 (DARE3dv2_Zenodo_040926.zip, 10.2 GB; 24 GB unpacked;
    about 34 GB free space needed). Safe to rerun: an existing DARE3dv2_Zenodo_040926/
    folder is left untouched, and an interrupted download or extraction resumes.
    """
    if dry_run:
        for line in describe_plan(dest):
            click.echo(line)
        return
    try:
        download(
            dest,
            progress_cb=percent_logger(click.echo),
            log=click.echo,
            keep_archive=keep_archive,
        )
    except DownloadError as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    main()
