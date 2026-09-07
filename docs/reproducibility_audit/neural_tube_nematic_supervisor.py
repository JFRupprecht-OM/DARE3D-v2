"""Strict unattended supervisor for the isolated neural-tube audit pipeline.

The supervisor never resumes or changes a failed training run. It advances to the
next stage only after the preceding detached process has exited successfully and
its durable result passes explicit completion checks.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUN_ROOT = (
    HERE
    / "neural_tube_nematic_full_pipeline"
    / "seed_12345"
)
SEGMENTATION_ROOT = RUN_ROOT / "segmentation_retrained"
REGRESSION_ROOT = RUN_ROOT / "regression_nematic_retrained"
EVALUATION_ROOT = RUN_ROOT / "evaluation"
SUPERVISOR_ROOT = RUN_ROOT / "supervisor"
STATE_PATH = SUPERVISOR_ROOT / "supervisor_state.json"
PROCESS_PATH = SUPERVISOR_ROOT / "supervisor_process.json"
LOG_PATH = SUPERVISOR_ROOT / "supervisor.log"
STDOUT_PATH = SUPERVISOR_ROOT / "launcher.stdout.log"
STDERR_PATH = SUPERVISOR_ROOT / "launcher.stderr.log"
POLL_SECONDS = 60
MAX_EPOCHS = 200

EXPECTED_HASHES = {
    "neural_tube_nematic_training.py": (
        "b305bcf4a37c3203b4848864bd1d0b5b3ac415dad4b0b1a500a97730346490a0"
    ),
    "neural_tube_nematic_evaluation.py": (
        "8a0f423eb947fb5f07a74faf43b4b3468a0e1ab6525f569ad8fae3a32fec3e99"
    ),
    "neural_tube_nematic_report.py": (
        "d15e5bbfb4381a40fc563fc6791e065d40b60f4a220bd057bf4a62158104e451"
    ),
    "launch_detached_training.py": (
        "48c079975a075815041e188919647df19060799729a8954acfc613967aed1322"
    ),
}


class StopAudit(RuntimeError):
    """A strict gate failed and human inspection is required."""


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StopAudit(f"Expected a JSON object in {path}")
    return value


def append_log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8", buffering=1) as stream:
        stream.write(f"[{now_iso()}] {message}\n")


def write_state(status: str, **details: Any) -> None:
    atomic_write_json(
        STATE_PATH,
        {
            "status": status,
            "updated_at": now_iso(),
            "supervisor_pid": os.getpid(),
            "poll_seconds": POLL_SECONDS,
            **details,
        },
    )


def update_process_record(status: str) -> None:
    record = read_json(PROCESS_PATH) or {}
    record.update(
        {
            "status": status,
            "updated_at": now_iso(),
            "pid": int(record.get("pid", os.getpid())),
        }
    )
    atomic_write_json(PROCESS_PATH, record)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_scripts() -> dict[str, str]:
    observed = {}
    mismatches = {}
    for name, expected in EXPECTED_HASHES.items():
        path = HERE / name
        actual = sha256(path) if path.is_file() else "missing"
        observed[name] = actual
        if actual != expected:
            mismatches[name] = {"expected": expected, "actual": actual}
    if mismatches:
        raise StopAudit(
            "Frozen audit script hash mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return observed


def process_is_running(pid: int) -> bool:
    """Check a Windows process without tasklist/CIM privileges."""
    if pid <= 0:
        return False
    synchronize = 0x00100000
    wait_timeout = 0x00000102
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(synchronize, False, pid)
    if not handle:
        return False
    try:
        return kernel32.WaitForSingleObject(handle, 0) == wait_timeout
    finally:
        kernel32.CloseHandle(handle)


def stage_root(stage: str) -> Path:
    if stage == "segmentation":
        return SEGMENTATION_ROOT
    if stage == "regression":
        return REGRESSION_ROOT
    raise ValueError(stage)


def resolve_result_path(value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = REPO / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(REPO.resolve())
    except ValueError as exc:
        raise StopAudit(f"Result path escapes repository: {value}") from exc
    return resolved


def validate_training_result(stage: str, result: dict[str, Any]) -> None:
    if result.get("status") != "trained":
        raise StopAudit(f"{stage} result is not trained: {result.get('status')}")
    if result.get("stage") != stage:
        raise StopAudit(f"{stage} result has wrong stage: {result.get('stage')}")
    epochs = int(result.get("epochs_completed", -1))
    if epochs < MAX_EPOCHS:
        raise StopAudit(
            f"{stage} stopped at {epochs} epochs; expected at least {MAX_EPOCHS}"
        )
    best_value = result.get("best_model_path")
    if not isinstance(best_value, str):
        raise StopAudit(f"{stage} result has no best_model_path")
    best_path = resolve_result_path(best_value)
    if not best_path.is_file():
        raise StopAudit(f"{stage} best checkpoint is missing: {best_path}")


def stage_snapshot(stage: str) -> dict[str, Any]:
    root = stage_root(stage)
    launcher = read_json(root / "launcher_process.json")
    state = read_json(root / "run_state.json")
    result = read_json(root / "training_result.json")
    pid = int((launcher or {}).get("pid", -1))
    return {
        "stage": stage,
        "pid": pid,
        "process_running": process_is_running(pid),
        "launcher": launcher,
        "run_state": state,
        "result_status": None if result is None else result.get("status"),
    }


def wait_for_training(stage: str) -> dict[str, Any]:
    root = stage_root(stage)
    result_path = root / "training_result.json"
    launcher_path = root / "launcher_process.json"
    append_log(f"Waiting for {stage} training")
    while True:
        scripts = verify_frozen_scripts()
        launcher = read_json(launcher_path)
        result = read_json(result_path)
        pid = int((launcher or {}).get("pid", -1))
        running = process_is_running(pid)
        current_state = read_json(root / "run_state.json")
        if result is not None:
            validate_training_result(stage, result)
            if running:
                write_state(
                    f"waiting_for_{stage}_process_exit",
                    frozen_script_hashes=scripts,
                    stage_snapshot=stage_snapshot(stage),
                )
                time.sleep(POLL_SECONDS)
                continue
            append_log(
                f"{stage} training completed at epoch "
                f"{result.get('epochs_completed')} with best "
                f"{result.get('best_model_path')}"
            )
            return result
        if launcher is None:
            raise StopAudit(f"Missing launcher metadata for {stage}")
        if not running:
            failure = read_json(root / "training_failure.json")
            raise StopAudit(
                f"{stage} process PID {pid} stopped without a completed result; "
                f"failure={json.dumps(failure, sort_keys=True)}"
            )
        write_state(
            f"waiting_for_{stage}",
            frozen_script_hashes=scripts,
            stage_snapshot={
                "stage": stage,
                "pid": pid,
                "process_running": True,
                "run_state": current_state,
            },
        )
        time.sleep(POLL_SECONDS)


def launch_regression_if_needed() -> None:
    result = read_json(REGRESSION_ROOT / "training_result.json")
    if result is not None:
        validate_training_result("regression", result)
        append_log("Reusing completed regression training")
        return
    launcher = read_json(REGRESSION_ROOT / "launcher_process.json")
    if launcher is not None:
        pid = int(launcher.get("pid", -1))
        if process_is_running(pid):
            append_log(f"Regression already runs as PID {pid}")
            return
    checkpoints = list((REGRESSION_ROOT / "checkpoints").glob("*.ckpt"))
    if checkpoints:
        raise StopAudit(
            "Regression has checkpoints but no active process or completed "
            "result. Refusing to guess whether to resume."
        )
    write_state(
        "launching_regression",
        segmentation=stage_snapshot("segmentation"),
    )
    command = [
        sys.executable,
        str(HERE / "launch_detached_training.py"),
        "regression",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    append_log(
        "Regression launcher returned "
        f"{completed.returncode}; stdout={completed.stdout!r}; "
        f"stderr={completed.stderr!r}"
    )
    if completed.returncode != 0:
        raise StopAudit(
            f"Regression launcher failed with exit code {completed.returncode}"
        )
    time.sleep(5)
    launcher = read_json(REGRESSION_ROOT / "launcher_process.json")
    pid = int((launcher or {}).get("pid", -1))
    if not process_is_running(pid):
        raise StopAudit(
            f"Regression launcher reported PID {pid}, but it is not running"
        )
    append_log(f"Started regression training as PID {pid}")


def run_foreground(
    label: str,
    command: list[str],
    result_path: Path,
    accepted_statuses: set[str],
) -> dict[str, Any]:
    existing = read_json(result_path)
    if existing is not None and existing.get("status") in accepted_statuses:
        append_log(f"Reusing completed {label}: {relative(result_path)}")
        return existing
    stdout_path = SUPERVISOR_ROOT / f"{label}.stdout.log"
    stderr_path = SUPERVISOR_ROOT / f"{label}.stderr.log"
    write_state(
        f"running_{label}",
        command=command,
        stdout=relative(stdout_path),
        stderr=relative(stderr_path),
    )
    append_log(f"Starting {label}: {command!r}")
    with stdout_path.open("a", encoding="utf-8", buffering=1) as stdout_stream:
        with stderr_path.open("a", encoding="utf-8", buffering=1) as stderr_stream:
            completed = subprocess.run(
                command,
                cwd=REPO,
                stdin=subprocess.DEVNULL,
                stdout=stdout_stream,
                stderr=stderr_stream,
                check=False,
            )
    append_log(f"{label} returned exit code {completed.returncode}")
    if completed.returncode != 0:
        raise StopAudit(
            f"{label} failed with exit code {completed.returncode}; "
            f"inspect {stderr_path}"
        )
    result = read_json(result_path)
    if result is None or result.get("status") not in accepted_statuses:
        raise StopAudit(
            f"{label} did not create an accepted result in {result_path}"
        )
    return result


def monitor() -> int:
    SUPERVISOR_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        scripts = verify_frozen_scripts()
        append_log("Supervisor started; frozen script hashes verified")
        write_state(
            "waiting_for_segmentation",
            frozen_script_hashes=scripts,
            segmentation=stage_snapshot("segmentation"),
        )
        segmentation = wait_for_training("segmentation")
        scripts = verify_frozen_scripts()
        launch_regression_if_needed()
        regression = wait_for_training("regression")
        scripts = verify_frozen_scripts()

        evaluation_path = EVALUATION_ROOT / "all_results.json"
        evaluation = run_foreground(
            "evaluation",
            [
                sys.executable,
                str(HERE / "neural_tube_nematic_evaluation.py"),
                "all",
            ],
            evaluation_path,
            {"complete"},
        )
        scripts = verify_frozen_scripts()

        comparison_path = EVALUATION_ROOT / "final_comparison.json"
        comparison = run_foreground(
            "report",
            [sys.executable, str(HERE / "neural_tube_nematic_report.py")],
            comparison_path,
            {"complete", "complete_with_failed_assertions"},
        )
        final_status = (
            "complete"
            if comparison.get("status") == "complete"
            else "complete_with_failed_assertions"
        )
        write_state(
            final_status,
            frozen_script_hashes=scripts,
            segmentation_result=relative(
                SEGMENTATION_ROOT / "training_result.json"
            ),
            regression_result=relative(REGRESSION_ROOT / "training_result.json"),
            evaluation_result=relative(evaluation_path),
            comparison_result=relative(comparison_path),
            segmentation_status=segmentation.get("status"),
            regression_status=regression.get("status"),
            evaluation_status=evaluation.get("status"),
            comparison_status=comparison.get("status"),
        )
        update_process_record(final_status)
        append_log(f"Supervisor finished with status {final_status}")
        return 0
    except BaseException as exc:
        failure = {
            "status": "blocked",
            "updated_at": now_iso(),
            "supervisor_pid": os.getpid(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "segmentation": stage_snapshot("segmentation"),
            "regression": stage_snapshot("regression"),
        }
        atomic_write_json(STATE_PATH, failure)
        update_process_record("blocked")
        append_log(
            f"Supervisor stopped safely: {type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        return 1


def launch() -> int:
    SUPERVISOR_ROOT.mkdir(parents=True, exist_ok=True)
    prior = read_json(PROCESS_PATH)
    prior_pid = int((prior or {}).get("pid", -1))
    if prior is not None and process_is_running(prior_pid):
        raise StopAudit(f"Supervisor already runs as PID {prior_pid}")
    verify_frozen_scripts()
    command = [sys.executable, str(Path(__file__).resolve()), "monitor"]
    creationflags = (
        subprocess.DETACHED_PROCESS
        | subprocess.CREATE_NEW_PROCESS_GROUP
        | subprocess.CREATE_NO_WINDOW
    )
    with STDOUT_PATH.open("a", encoding="utf-8", buffering=1) as stdout_stream:
        with STDERR_PATH.open("a", encoding="utf-8", buffering=1) as stderr_stream:
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
        "started_at": now_iso(),
        "pid": process.pid,
        "command": command,
        "working_directory": str(REPO),
        "state": relative(STATE_PATH),
        "log": relative(LOG_PATH),
        "stdout": relative(STDOUT_PATH),
        "stderr": relative(STDERR_PATH),
        "process_check_command": (
            f"Get-Process -Id {process.pid} -ErrorAction SilentlyContinue"
        ),
    }
    atomic_write_json(PROCESS_PATH, record)
    print(json.dumps(record, indent=2))
    return 0


def show_status() -> int:
    payload = {
        "process": read_json(PROCESS_PATH),
        "state": read_json(STATE_PATH),
        "segmentation": stage_snapshot("segmentation"),
        "regression": stage_snapshot("regression"),
    }
    print(json.dumps(payload, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("launch", "monitor", "status"))
    args = parser.parse_args()
    if args.command == "launch":
        return launch()
    if args.command == "monitor":
        return monitor()
    return show_status()


if __name__ == "__main__":
    raise SystemExit(main())
