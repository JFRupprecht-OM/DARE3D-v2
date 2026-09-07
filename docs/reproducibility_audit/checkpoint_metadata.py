"""Record metadata for all released DARE3D checkpoints.

This is an audit-only reader. It does not instantiate models or modify checkpoints.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "DARE3d_data_190326"
OUT = HERE / "evidence" / "checkpoint_metadata.json"

RUNS = {
    "nuclei_segmentation": (
        DATA / "Gastruloid_241025/weights/segmentation3d_exp10-b/checkpoints",
        "epoch_067.ckpt",
    ),
    "nuclei_regression": (
        DATA / "Gastruloid_241025/weights/regression3d_exp10-b/checkpoints",
        "epoch_098.ckpt",
    ),
    "membrane_segmentation": (
        DATA
        / "Neural_tube_160226/segmentation3d_new_set_og/runs/12-01-26/checkpoints",
        "epoch_057.ckpt",
    ),
    "membrane_regression": (
        DATA
        / "Neural_tube_160226/regression3d_new_set_og/runs/12-01-26/checkpoints",
        "epoch_147.ckpt",
    ),
}


def rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def scalar(value):
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.detach().cpu().item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None


def summarize(value, depth=0):
    simple = scalar(value)
    if simple is not None or value is None:
        return simple
    if isinstance(value, torch.Tensor):
        return {
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, dict):
        if depth >= 2:
            return {"type": "dict", "keys": [str(k) for k in value.keys()]}
        return {str(k): summarize(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if depth >= 2:
            return {"type": type(value).__name__, "length": len(value)}
        return [summarize(v, depth + 1) for v in value]
    return {"type": type(value).__name__, "repr": repr(value)[:500]}


def load(path: Path):
    return torch.load(path, map_location="cpu", mmap=True, weights_only=False)


def describe(path: Path) -> dict:
    print(f"Loading metadata: {rel(path)}", flush=True)
    checkpoint = load(path)
    state = checkpoint.get("state_dict", {})
    prefix_counts = {}
    dtype_counts = {}
    numel = 0
    nbytes = 0
    for key, tensor in state.items():
        prefix = key.split(".", 1)[0]
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
        dtype_counts[str(tensor.dtype)] = dtype_counts.get(str(tensor.dtype), 0) + 1
        numel += tensor.numel()
        nbytes += tensor.numel() * tensor.element_size()
    callbacks = checkpoint.get("callbacks", {})
    result = {
        "path": rel(path),
        "bytes": path.stat().st_size,
        "top_level_keys": list(checkpoint.keys()),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "pytorch_lightning_version": checkpoint.get("pytorch-lightning_version"),
        "state_dict": {
            "tensor_count": len(state),
            "numel": numel,
            "tensor_bytes": nbytes,
            "dtype_tensor_counts": dtype_counts,
            "prefix_tensor_counts": prefix_counts,
            "first_keys": list(state.keys())[:10],
            "last_keys": list(state.keys())[-10:],
        },
        "optimizer_state_count": len(checkpoint.get("optimizer_states", [])),
        "lr_scheduler_count": len(checkpoint.get("lr_schedulers", [])),
        "callbacks": summarize(callbacks),
        "hyper_parameters": summarize(checkpoint.get("hyper_parameters", {})),
    }
    del checkpoint
    gc.collect()
    return result


def compare(best_path: Path, last_path: Path) -> dict:
    print(f"Comparing states: {best_path.name} vs {last_path.name}", flush=True)
    best = load(best_path)
    last = load(last_path)
    a = best["state_dict"]
    b = last["state_dict"]
    keys_equal = list(a.keys()) == list(b.keys())
    common = sorted(set(a) & set(b))
    shape_mismatches = []
    differing = []
    differing_numel = 0
    for key in common:
        if a[key].shape != b[key].shape or a[key].dtype != b[key].dtype:
            shape_mismatches.append(key)
            continue
        if not torch.equal(a[key], b[key]):
            differing.append(key)
            differing_numel += a[key].numel()
    result = {
        "best_path": rel(best_path),
        "last_path": rel(last_path),
        "best_epoch": best.get("epoch"),
        "last_epoch": last.get("epoch"),
        "best_global_step": best.get("global_step"),
        "last_global_step": last.get("global_step"),
        "ordered_state_keys_equal": keys_equal,
        "best_only_keys": sorted(set(a) - set(b)),
        "last_only_keys": sorted(set(b) - set(a)),
        "shape_or_dtype_mismatch_keys": shape_mismatches,
        "differing_tensor_count": len(differing),
        "differing_numel": differing_numel,
        "state_dict_exactly_equal": (
            keys_equal and not shape_mismatches and not differing
        ),
        "first_differing_keys": differing[:25],
    }
    del a, b, best, last
    gc.collect()
    return result


def main():
    records = {}
    comparisons = {}
    for name, (directory, best_name) in RUNS.items():
        best_path = directory / best_name
        last_path = directory / "last.ckpt"
        records[name] = {
            "best": describe(best_path),
            "last": describe(last_path),
        }
        comparisons[name] = compare(best_path, last_path)
    result = {
        "torch_version": torch.__version__,
        "checkpoints": records,
        "best_vs_last": comparisons,
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(comparisons, indent=2), flush=True)


if __name__ == "__main__":
    main()
