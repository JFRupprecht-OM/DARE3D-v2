"""Prepare immutable inputs and protection baselines for neural Hydra training."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO / "DARE3d_data_190326/Neural_tube_160226"
STAGING_ROOT = DATA_ROOT / "training_splits/regression3d_nematic_seed12345"
MODEL_ROOT = DATA_ROOT / "weights/regression3d_nematic_hydra_seed12345"
ZENODO_ROOT = REPO / "DARE3dv2_Zenodo_040926"
DESTINATION_ROOT = (
    ZENODO_ROOT
    / "Neural_tube_160226/weights/regression3d_nematic_hydra_seed12345"
)
BASELINE_PATH = STAGING_ROOT / "protected_before.json"
PREFLIGHT_PATH = STAGING_ROOT / "preflight.json"

SOURCE_FILES = {
    "movie_E_image": {
        "source": DATA_ROOT / "trainingset/movie1/im/movie_E.tif",
        "staged": STAGING_ROOT / "train/im/movie_E.tif",
        "sha256": "bf361c31a40ec129eb1e056ef56e6f05dd60709597d9e60e90657b9cc21a73cb",
    },
    "movie_E_label": {
        "source": DATA_ROOT / "trainingset/movie1/label/movie_E.tif",
        "staged": STAGING_ROOT / "train/label/movie_E.tif",
        "sha256": "c0d8b1109b058cfd85ced9a6c086c563649f7beb53b03f6e53bc9f8113572675",
    },
    "movie_I2_image": {
        "source": DATA_ROOT / "trainingset/movie2/im/movie_I2.tif",
        "staged": STAGING_ROOT / "val/im/movie_I2.tif",
        "sha256": "b669093406d537164f7487e4740a9e1f19d959031cb0ff86991785d7a6154062",
    },
    "movie_I2_label": {
        "source": DATA_ROOT / "trainingset/movie2/label/movie_I2.tif",
        "staged": STAGING_ROOT / "val/label/movie_I2.tif",
        "sha256": "99ad277d1ea28334b02d6824deddb9b76eeff0e0b449a2656e0bbb255546b4f5",
    },
    "movie_M_image": {
        "source": DATA_ROOT / "test_input/im/movie_M.tif",
        "staged": None,
        "sha256": "91f0fd8cd0e14256b51da3247e521d51013c786f792c624c83598dbd827e8c88",
    },
    "movie_M_label": {
        "source": DATA_ROOT / "test_input/label/movie_M.tif",
        "staged": None,
        "sha256": "d239ac66a09aa1b22e96b4e69a20a1a758dced35d3d818d79c7b38623eacd6df",
    },
}

PROTECTED_TREES = {
    "legacy_neural_regression_source": (
        DATA_ROOT / "weights/regression3d_new_set_og"
    ),
    "neural_segmentation_source": (
        DATA_ROOT / "weights/segmentation3d_new_set_og"
    ),
    "validated_neural_audit": (
        REPO
        / "docs/reproducibility_audit/neural_tube_nematic_full_pipeline/seed_12345"
    ),
    "validated_gastruloid_audit": (
        REPO / "docs/reproducibility_audit/nematic_retraining/seed_12345"
    ),
    "gastruloid_source_weights": (
        REPO / "DARE3d_data_190326/Gastruloid_241025/weights"
    ),
    "zenodo_gastruloid_assets": (
        ZENODO_ROOT / "Gastruloid_241025"
    ),
    "zenodo_neural_legacy_regression": (
        ZENODO_ROOT / "Neural_tube_160226/weights/regression3d_new_set_og"
    ),
    "zenodo_neural_segmentation": (
        ZENODO_ROOT / "Neural_tube_160226/weights/segmentation3d_new_set_og"
    ),
    "napari_package": REPO / "napari_dare3d",
}

PROTECTED_FILES = (
    ZENODO_ROOT / "DARE3D_gastruloid_regression_epoch095.ckpt",
    ZENODO_ROOT / "DARE3D_gastruloid_segmentation_epoch067.ckpt",
    ZENODO_ROOT / "DARE3D_neural_tube_regression_epoch139.ckpt",
    ZENODO_ROOT / "DARE3D_neural_tube_segmentation_epoch057.ckpt",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": rel(path),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256(path),
    }


def hash_tree(root: Path) -> dict[str, Any]:
    files = [
        file_record(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]
    return {
        "root": rel(root),
        "file_count": len(files),
        "total_bytes": sum(record["bytes"] for record in files),
        "files": files,
    }


def zenodo_metadata() -> list[dict[str, Any]]:
    return [
        {
            "path": rel(path),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(
            item for item in ZENODO_ROOT.rglob("*") if item.is_file()
        )
    ]


def git_output(*arguments: str) -> str:
    return subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={REPO.as_posix()}",
            *arguments,
        ],
        cwd=REPO,
        text=True,
    ).strip()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    for path, label in (
        (STAGING_ROOT, "staging root"),
        (MODEL_ROOT, "candidate model root"),
        (DESTINATION_ROOT, "Zenodo candidate destination"),
    ):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing {label}: {path}")

    missing = [
        path
        for path in (
            *(record["source"] for record in SOURCE_FILES.values()),
            *PROTECTED_TREES.values(),
            *PROTECTED_FILES,
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Required protected inputs are missing: {missing}")

    print("Verifying genuine neural source TIFFs", flush=True)
    source_records = {}
    for name, spec in SOURCE_FILES.items():
        record = file_record(spec["source"])
        record["expected_sha256"] = spec["sha256"]
        record["verified"] = record["sha256"] == spec["sha256"]
        source_records[name] = record
    if not all(record["verified"] for record in source_records.values()):
        raise AssertionError("A neural source TIFF hash differs from the audited input")

    print("Hashing protected trees", flush=True)
    protected_trees = {
        label: hash_tree(root) for label, root in PROTECTED_TREES.items()
    }
    print("Hashing protected release-root checkpoints", flush=True)
    protected_files = [file_record(path) for path in PROTECTED_FILES]
    print("Inventorying the complete pre-existing Zenodo tree", flush=True)
    zenodo = zenodo_metadata()

    baseline = {
        "schema_version": 2,
        "created_at": now_iso(),
        "candidate_existed": False,
        "destination_existed": False,
        "model_root": rel(MODEL_ROOT),
        "destination_root": rel(DESTINATION_ROOT),
        "source_files": source_records,
        "protected_trees": protected_trees,
        "protected_files": protected_files,
        "zenodo_tree_metadata": zenodo,
        "git": {
            "head": git_output("rev-parse", "HEAD"),
            "tracked_status": git_output(
                "status", "--short", "--untracked-files=no"
            ),
        },
    }

    print("Creating verified NTFS hard-link staging split", flush=True)
    for spec in SOURCE_FILES.values():
        staged = spec["staged"]
        if staged is None:
            continue
        staged.parent.mkdir(parents=True, exist_ok=True)
        os.link(spec["source"], staged)
        if not os.path.samefile(spec["source"], staged):
            raise AssertionError(f"Staged path is not a hard link: {staged}")
        if sha256(staged) != spec["sha256"]:
            raise AssertionError(f"Staged hash mismatch: {staged}")

    write_json(BASELINE_PATH, baseline)
    preflight = {
        "status": "pass",
        "created_at": now_iso(),
        "scope": "fresh neural-tube Hydra regression retraining only",
        "source_files": source_records,
        "staging_root": rel(STAGING_ROOT),
        "model_root": rel(MODEL_ROOT),
        "destination_root": rel(DESTINATION_ROOT),
        "protected_tree_count": len(protected_trees),
        "protected_file_count": len(protected_files),
        "zenodo_file_count": len(zenodo),
        "assertions": {
            "source_hashes_verified": True,
            "staging_hard_links_verified": True,
            "candidate_absent_before_preflight": True,
            "destination_absent_before_preflight": True,
        },
    }
    write_json(PREFLIGHT_PATH, preflight)
    print(json.dumps(preflight, indent=2), flush=True)


if __name__ == "__main__":
    main()
