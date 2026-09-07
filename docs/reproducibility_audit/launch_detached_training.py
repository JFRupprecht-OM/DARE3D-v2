"""Launch a long audit training stage as a hidden, detached Windows process."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import neural_tube_nematic_experiment as common
import neural_tube_nematic_training as training


def process_is_running(pid: int) -> bool:
    completed = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0 and f'"{pid}"' in completed.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("segmentation", "regression"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = training.stage_root(args.stage)
    root.mkdir(parents=True, exist_ok=True)
    process_path = root / "launcher_process.json"
    if process_path.exists():
        prior = json.loads(process_path.read_text(encoding="utf-8"))
        prior_pid = int(prior.get("pid", -1))
        if prior.get("status") == "running" and process_is_running(prior_pid):
            raise RuntimeError(f"Training already runs as PID {prior_pid}")

    stdout_path = root / "launcher.stdout.log"
    stderr_path = root / "launcher.stderr.log"
    command = [
        sys.executable,
        str(HERE / "neural_tube_nematic_training.py"),
        training.stage_command(args.stage),
    ]
    if args.resume:
        command.append("--resume")

    creationflags = (
        subprocess.DETACHED_PROCESS
        | subprocess.CREATE_NEW_PROCESS_GROUP
        | subprocess.CREATE_NO_WINDOW
    )
    with stdout_path.open("a", encoding="utf-8", buffering=1) as stdout_stream:
        with stderr_path.open("a", encoding="utf-8", buffering=1) as stderr_stream:
            process = subprocess.Popen(
                command,
                cwd=REPO,
                stdin=subprocess.DEVNULL,
                stdout=stdout_stream,
                stderr=stderr_stream,
                close_fds=True,
                creationflags=creationflags,
            )

    record = {
        "status": "running",
        "started_at": common.now_iso(),
        "pid": process.pid,
        "stage": args.stage,
        "resume": args.resume,
        "command": command,
        "working_directory": str(REPO),
        "stdout": common.rel(stdout_path),
        "stderr": common.rel(stderr_path),
        "authoritative_state": common.rel(root / "run_state.json"),
        "process_check_command": (
            f"Get-Process -Id {process.pid} -ErrorAction SilentlyContinue"
        ),
    }
    common.atomic_write_json(process_path, record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
