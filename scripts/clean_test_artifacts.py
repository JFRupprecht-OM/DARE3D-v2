#!/usr/bin/env python
"""Safely remove pytest-generated directories from the repository root."""

from __future__ import annotations

import argparse
import os
import shutil
import stat
from pathlib import Path


REPOSITORY_MARKER = ".project-root"
CURRENT_NAMES = {".pytest_cache", ".pytest_tmp"}
LEGACY_PREFIX = ".pytest_tmp_"


def repository_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    if not (root / REPOSITORY_MARKER).is_file():
        raise RuntimeError(f"Repository marker not found at {root / REPOSITORY_MARKER}")
    return root


def candidate_paths(root: Path, include_legacy: bool) -> list[Path]:
    candidates = [root / name for name in sorted(CURRENT_NAMES)]
    if include_legacy:
        candidates.extend(
            child for child in root.iterdir() if child.name.startswith(LEGACY_PREFIX)
        )
    return sorted(set(candidates), key=lambda path: path.name)


def validate_target(root: Path, target: Path) -> None:
    if target.parent.resolve() != root.resolve():
        raise RuntimeError(f"Refusing to clean outside repository root: {target}")
    if target.name not in CURRENT_NAMES and not target.name.startswith(LEGACY_PREFIX):
        raise RuntimeError(f"Refusing unexpected test-artifact path: {target}")


def is_reparse_point(path: Path) -> bool:
    attributes = getattr(os.lstat(path), "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def remove_target(target: Path) -> None:
    if target.is_symlink() or is_reparse_point(target):
        if target.is_dir():
            os.rmdir(target)
        else:
            target.unlink()
        return
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="also remove direct repository children named .pytest_tmp_*",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list validated targets without removing them",
    )
    args = parser.parse_args()

    root = repository_root()
    for target in candidate_paths(root, args.include_legacy):
        validate_target(root, target)
        if not target.exists() and not target.is_symlink():
            continue
        action = "Would remove" if args.dry_run else "Removing"
        print(f"{action}: {target}")
        if not args.dry_run:
            remove_target(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
