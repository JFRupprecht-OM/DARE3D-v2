"""Offline contract tests for the Zenodo bundle downloader (``napari_dare3d._data``).

All HTTP goes through ``_data._open``, which is replaced by an in-memory fake server
serving the record JSON and a tiny archive with the real top-level folder name. One
opt-in ``slow`` test talks to the live Zenodo API (JSON only, no 10 GB transfer).
"""
import hashlib
import io
import json
import types
import urllib.error
import zipfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from napari_dare3d import _data
from napari_dare3d._release_models import RELEASE_ROOT_NAME, find_release_root

B = _data.BUNDLE_NAME
CONTENT_URL = _data._CONTENT_URL
API_URL = _data._API.format(_data.ZENODO_RECORD)

MEMBERS = {
    f"{B}/DARE3D_gastruloid_segmentation_epoch067.ckpt": bytes(range(256)) * 20,
    f"{B}/Gastruloid_241025/weights/segmentation3d_exp10-b/.hydra/config.yaml": b"model: {}\n",
    f"{B}/Gastruloid_241025/weights/segmentation3d_exp10-b/checkpoints/last.ckpt": b"x" * 3000,
    f"{B}/Gastruloid_241025/test_input/movie2.tif": b"tif" * 100,
}
JUNK = {f"__MACOSX/{B}/._junk": b"junk", f"{B}/.DS_Store": b"ds"}


def _zip_bytes(members, dirs=(f"{B}/Gastruloid_241025/",)):
    buf = io.BytesIO()
    # Stored (uncompressed) so the archive is several KiB: the resume/retry tests rely on
    # the transfer spanning more than one 1 KiB fake-server chunk.
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for d in dirs:
            z.writestr(zipfile.ZipInfo(d), b"")
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


def _rel(name):
    return Path(name[len(B) + 1 :])


class FakeResponse:
    def __init__(self, data, status=200, headers=None, chunk=1024, fail_after=None):
        self._data, self.status, self.headers = data, status, headers or {}
        self._pos, self._chunk, self._fail_after = 0, chunk, fail_after

    def read(self, n=-1):
        if n is None or n < 0:
            buf, self._pos = self._data[self._pos :], len(self._data)
            return buf
        if self._fail_after is not None and self._pos >= self._fail_after:
            raise ConnectionResetError("simulated connection drop")
        end = min(len(self._data), self._pos + min(n, self._chunk))
        buf, self._pos = self._data[self._pos : end], end
        return buf

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeServer:
    def __init__(self, zip_bytes, md5, honour_range=True, fail_after=None, api_checksum=None):
        self.zip, self.md5 = zip_bytes, md5
        self.honour_range, self.fail_after, self.api_checksum = (
            honour_range,
            fail_after,
            api_checksum,
        )
        self.calls = []

    @property
    def content_calls(self):
        return [(u, h) for u, h in self.calls if u == CONTENT_URL]

    def open(self, url, headers=None, timeout=60):
        headers = dict(headers or {})
        self.calls.append((url, headers))
        if url == API_URL:
            payload = {
                "files": [
                    {
                        "key": _data.ARCHIVE_NAME,
                        "size": len(self.zip),
                        "checksum": self.api_checksum or f"md5:{self.md5}",
                        "links": {"self": CONTENT_URL},
                    }
                ]
            }
            return FakeResponse(json.dumps(payload).encode())
        assert url == CONTENT_URL, url
        fail, self.fail_after = self.fail_after, None  # drop the connection once at most
        rng = headers.get("Range")
        if rng and self.honour_range:
            start = int(rng.split("=")[1].rstrip("-"))
            if start >= len(self.zip):
                raise urllib.error.HTTPError(url, 416, "Range Not Satisfiable", {}, None)
            return FakeResponse(
                self.zip[start:],
                status=206,
                fail_after=fail,
                headers={"Content-Range": f"bytes {start}-{len(self.zip) - 1}/{len(self.zip)}"},
            )
        return FakeResponse(self.zip, status=200, fail_after=fail)


@pytest.fixture
def bundle_zip():
    data = _zip_bytes({**MEMBERS, **JUNK})
    return data, hashlib.md5(data).hexdigest()


@pytest.fixture
def pins(monkeypatch, bundle_zip):
    data, md5 = bundle_zip
    monkeypatch.setattr(_data, "ARCHIVE_SIZE", len(data))
    monkeypatch.setattr(_data, "ARCHIVE_MD5", md5)
    monkeypatch.setattr(_data, "UNPACKED_SIZE", sum(len(v) for v in MEMBERS.values()))
    monkeypatch.setattr(
        _data.shutil,
        "disk_usage",
        lambda p: types.SimpleNamespace(total=10**13, used=0, free=10**12),
    )
    monkeypatch.setattr(_data.time, "sleep", lambda s: None)
    monkeypatch.setattr(_data, "_max_path_limit", lambda: None)  # pytest tmp paths are long
    return data, md5


@pytest.fixture
def server(monkeypatch, pins):
    data, md5 = pins
    srv = FakeServer(data, md5)
    monkeypatch.setattr(_data, "_open", srv.open)
    return srv


def _assert_bundle_ok(root: Path):
    assert root.is_dir()
    for name, data in MEMBERS.items():
        assert (root / _rel(name)).read_bytes() == data
    assert not (root / ".DS_Store").exists()
    assert not any("__MACOSX" in p.parts or p.name.startswith("._") for p in root.rglob("*"))


def test_bundle_name_matches_release_presets():
    assert B == RELEASE_ROOT_NAME == "DARE3dv2_Zenodo_040926"
    assert _data.ARCHIVE_NAME == f"{B}.zip"
    assert _data.ZENODO_RECORD.isdigit() and len(_data.ARCHIVE_MD5) == 32
    assert _data.ZENODO_RECORD in _data.ZENODO_DOI and _data.ZENODO_RECORD in CONTENT_URL


def test_fresh_download_verifies_unpacks_and_records_provenance(tmp_path, server, monkeypatch):
    logs = []
    out = _data.download(tmp_path, progress_cb=_data.percent_logger(logs.append), log=logs.append)

    assert out == tmp_path / B
    _assert_bundle_ok(out)
    assert not (tmp_path / _data.ARCHIVE_NAME).exists()
    assert not (tmp_path / f"{_data.ARCHIVE_NAME}.part").exists()
    assert not (tmp_path / f"{B}.tmp").exists()
    prov = json.loads((tmp_path / f"{B}.provenance.json").read_text(encoding="utf-8"))
    assert prov["zenodo_record"] == _data.ZENODO_RECORD
    assert prov["doi"] == _data.ZENODO_DOI and prov["archive_md5"] == server.md5
    assert prov["archive_kept"] is False
    assert any(line.startswith("download 100%") for line in logs)
    assert any("checksum OK" in line for line in logs)
    for line in logs:
        line.encode("ascii")  # Windows cp1252 consoles must never choke on our output

    monkeypatch.chdir(tmp_path)
    assert find_release_root() == out.resolve()


def test_keep_archive_retains_zip(tmp_path, server):
    logs = []
    _data.download(tmp_path, log=logs.append, keep_archive=True)
    archive = tmp_path / _data.ARCHIVE_NAME
    assert archive.is_file() and hashlib.md5(archive.read_bytes()).hexdigest() == server.md5
    prov = json.loads((tmp_path / f"{B}.provenance.json").read_text(encoding="utf-8"))
    assert prov["archive_kept"] is True


def test_md5_mismatch_removes_partial_and_raises(tmp_path, server):
    corrupted = bytearray(server.zip)
    corrupted[len(corrupted) // 2] ^= 0xFF
    server.zip = bytes(corrupted)  # API still reports the pinned (correct) md5/size

    with pytest.raises(_data.DownloadError, match="md5 mismatch"):
        _data.download(tmp_path, log=lambda m: None)
    assert not (tmp_path / f"{_data.ARCHIVE_NAME}.part").exists()
    assert not (tmp_path / _data.ARCHIVE_NAME).exists()
    assert not (tmp_path / B).exists()
    assert len(server.content_calls) == 1  # corruption is not a transient error: no retry


def test_partial_download_resumes_with_range_request(tmp_path, server):
    half = len(server.zip) // 2
    (tmp_path / f"{_data.ARCHIVE_NAME}.part").write_bytes(server.zip[:half])
    logs = []

    out = _data.download(tmp_path, log=logs.append)

    _assert_bundle_ok(out)
    assert len(server.content_calls) == 1
    assert server.content_calls[0][1]["Range"] == f"bytes={half}-"
    assert any("resuming at" in line for line in logs)
    for line in logs:
        line.encode("ascii")


def test_server_ignoring_range_restarts_from_zero(tmp_path, server):
    server.honour_range = False
    half = len(server.zip) // 2
    (tmp_path / f"{_data.ARCHIVE_NAME}.part").write_bytes(server.zip[:half])
    logs = []

    out = _data.download(tmp_path, log=logs.append)

    _assert_bundle_ok(out)
    assert any("server ignored the Range request" in line for line in logs)


def test_connection_drop_is_retried_and_resumed(tmp_path, server):
    server.fail_after = 1500  # 1 KiB chunks -> drop after 2048 bytes were written
    logs = []

    out = _data.download(tmp_path, log=logs.append)

    _assert_bundle_ok(out)
    assert len(server.content_calls) == 2
    assert "Range" not in server.content_calls[0][1]
    assert server.content_calls[1][1]["Range"] == "bytes=2048-"
    assert any(line.startswith("attempt 1/5 failed") for line in logs)


def test_existing_bundle_is_left_untouched_without_network(tmp_path, server):
    root = tmp_path / B
    (root / "Gastruloid_241025").mkdir(parents=True)
    marker = root / "Gastruloid_241025" / "user_file.txt"
    marker.write_text("mine")
    logs = []

    assert _data.download(tmp_path, log=logs.append) == root
    assert server.calls == []
    assert marker.read_text() == "mine"
    assert any(line.startswith("already present") for line in logs)


def test_local_archive_is_verified_and_interrupted_extraction_resumes(tmp_path, pins):
    data, _md5 = pins
    (tmp_path / _data.ARCHIVE_NAME).write_bytes(data)
    staging = tmp_path / f"{B}.tmp"
    truncated = staging / "Gastruloid_241025" / "test_input" / "movie2.tif"
    truncated.parent.mkdir(parents=True)
    truncated.write_bytes(b"tif")  # shorter than the member -> must be rewritten
    kept = (
        staging
        / "Gastruloid_241025"
        / "weights"
        / "segmentation3d_exp10-b"
        / ".hydra"
        / "config.yaml"
    )
    kept.parent.mkdir(parents=True)
    kept.write_bytes(b"MODEL: {}\n")  # same size as the member -> skipped (kept as is)
    logs = []

    out = _data.download(tmp_path, log=logs.append)  # no _open patch: must stay offline

    assert out == tmp_path / B and not staging.exists()
    assert (out / "Gastruloid_241025" / "test_input" / "movie2.tif").read_bytes() == b"tif" * 100
    assert (
        out / _rel(next(k for k in MEMBERS if k.endswith("config.yaml")))
    ).read_bytes() == b"MODEL: {}\n"
    assert not (tmp_path / _data.ARCHIVE_NAME).exists()
    assert any(line.startswith("verifying md5") for line in logs)


def test_wrong_size_local_archive_is_rejected_with_instructions(tmp_path, pins):
    (tmp_path / _data.ARCHIVE_NAME).write_bytes(b"not the archive")
    with pytest.raises(_data.DownloadError, match=r"rename it to .*\.part"):
        _data.download(tmp_path, log=lambda m: None)


@pytest.mark.parametrize(
    "extra_member",
    ["../evil.txt", f"{B}/../evil.txt", f"{B}/sub/../../evil.txt", "other_top_level/x.txt"],
)
def test_unexpected_archive_layout_is_rejected(tmp_path, monkeypatch, extra_member):
    data = _zip_bytes({**MEMBERS, extra_member: b"evil"})
    monkeypatch.setattr(_data, "ARCHIVE_SIZE", len(data))
    monkeypatch.setattr(_data, "ARCHIVE_MD5", hashlib.md5(data).hexdigest())
    monkeypatch.setattr(
        _data.shutil, "disk_usage", lambda p: types.SimpleNamespace(total=1, used=0, free=10**12)
    )
    monkeypatch.setattr(_data, "_max_path_limit", lambda: None)
    archive = tmp_path / _data.ARCHIVE_NAME
    archive.write_bytes(data)

    with pytest.raises(_data.DownloadError, match="unexpected archive member"):
        _data.download(tmp_path, log=lambda m: None)
    assert not (tmp_path / B).exists()
    assert not (tmp_path / f"{B}.tmp").exists()
    assert archive.is_file()  # never deleted on failure
    assert not (tmp_path / "evil.txt").exists()


def test_backslash_member_is_rejected_by_layout_check():
    info = zipfile.ZipInfo("placeholder")
    info.filename = f"{B}\\evil.txt"  # bypass ZipInfo's separator normalisation
    with pytest.raises(_data.DownloadError, match="unexpected archive member"):
        _data._check_layout([info])


def test_windows_path_length_guard_fails_fast(tmp_path, pins, monkeypatch):
    data, _md5 = pins
    (tmp_path / _data.ARCHIVE_NAME).write_bytes(data)
    monkeypatch.setattr(_data, "_max_path_limit", lambda: 40)

    with pytest.raises(_data.DownloadError, match="characters long"):
        _data.download(tmp_path, log=lambda m: None)
    assert not (tmp_path / f"{B}.tmp").exists()
    assert not (tmp_path / B).exists()
    assert (tmp_path / _data.ARCHIVE_NAME).is_file()


def test_api_disagreeing_with_pins_aborts_before_transfer(tmp_path, server):
    server.api_checksum = "md5:" + "0" * 32
    with pytest.raises(_data.DownloadError, match="disagrees with the constants pinned"):
        _data.download(tmp_path, log=lambda m: None)
    assert server.content_calls == []
    assert not (tmp_path / f"{_data.ARCHIVE_NAME}.part").exists()


def test_insufficient_free_space_fails_before_network(tmp_path, server, monkeypatch):
    monkeypatch.setattr(
        _data.shutil, "disk_usage", lambda p: types.SimpleNamespace(total=1, used=0, free=1)
    )
    with pytest.raises(_data.DownloadError, match="not enough free space"):
        _data.download(tmp_path, log=lambda m: None)
    assert server.calls == []


def test_default_dest_prefers_project_root_marker(tmp_path, monkeypatch):
    (tmp_path / ".project-root").touch()
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)
    assert _data.default_dest() == tmp_path.resolve()


def test_default_dest_falls_back_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_data, "__file__", str(tmp_path / "site" / "napari_dare3d" / "_data.py"))
    assert _data.default_dest() == Path.cwd()


def test_cli_help_and_dry_run_are_offline(tmp_path, server):
    runner = CliRunner()

    result = runner.invoke(_data.main, ["--help"])
    assert result.exit_code == 0, result.output
    assert "--dest" in result.output and "--keep-archive" in result.output

    result = runner.invoke(_data.main, ["--dry-run", "--dest", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "fresh" in result.output and "verdict:     OK" in result.output
    assert str(tmp_path / B) in result.output
    assert server.calls == []

    (tmp_path / f"{_data.ARCHIVE_NAME}.part").write_bytes(server.zip[:10])
    result = runner.invoke(_data.main, ["--dry-run", "--dest", str(tmp_path)])
    assert result.exit_code == 0 and "will resume" in result.output
    assert server.calls == []


def test_cli_full_run_and_rerun(tmp_path, server):
    runner = CliRunner()

    result = runner.invoke(_data.main, ["--dest", str(tmp_path)])
    assert result.exit_code == 0, result.output
    _assert_bundle_ok(tmp_path / B)
    assert "removed archive" in result.output and "done:" in result.output
    result.output.encode("ascii")

    calls_before = len(server.calls)
    result = runner.invoke(_data.main, ["--dest", str(tmp_path)])
    assert result.exit_code == 0 and "already present" in result.output
    assert len(server.calls) == calls_before

    result = runner.invoke(_data.main, ["--dry-run", "--dest", str(tmp_path)])
    assert result.exit_code == 0 and "already present" in result.output


def test_cli_reports_download_errors_without_traceback(tmp_path, server):
    server.api_checksum = "md5:" + "0" * 32
    result = runner_result = CliRunner().invoke(_data.main, ["--dest", str(tmp_path)])
    assert runner_result.exit_code == 1
    assert "Error: Zenodo record" in result.output
    assert "Traceback" not in result.output


def test_demo_helper_reuses_existing_bundle_without_downloading(tmp_path, monkeypatch):
    import dare3d.demo

    (tmp_path / B).mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_data, "download", lambda *a, **k: pytest.fail("network call attempted"))

    assert dare3d.demo.get_path_to_demo_folder() == (tmp_path / B).resolve()


@pytest.mark.slow
def test_live_zenodo_record_matches_pinned_constants():
    entry = _data.archive_entry(_data.zenodo_files())
    assert entry["key"] == _data.ARCHIVE_NAME
    assert entry["size"] == _data.ARCHIVE_SIZE
    assert entry["links"]["self"].startswith("https://zenodo.org/")
