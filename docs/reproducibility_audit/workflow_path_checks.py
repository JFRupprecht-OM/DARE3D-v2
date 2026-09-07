"""Audit documented CLI, notebook, and napari-plugin paths without launching a GUI."""
from __future__ import annotations

import csv
import importlib
import importlib.metadata
import inspect
import json
import os
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np
import tifffile
import torch
from npe2 import PluginManifest
from omegaconf import OmegaConf


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "DARE3d_data_190326"
EVIDENCE = HERE / "evidence"
RUNTIME_TMP = HERE / "runtime_tmp"
JSON_OUTPUT = EVIDENCE / "workflow_path_checks.json"
CSV_OUTPUT = EVIDENCE / "workflow_path_matrix.csv"

GASTRULOID = DATA / "Gastruloid_241025"
NEURAL = DATA / "Neural_tube_160226"
MODEL_ROOTS = {
    "gastruloid_segmentation": GASTRULOID / "weights/segmentation3d_exp10-b",
    "gastruloid_regression": GASTRULOID / "weights/regression3d_exp10-b",
    "neural_segmentation": NEURAL / "segmentation3d_new_set_og",
    "neural_regression": NEURAL / "regression3d_new_set_og",
}
DOCUMENTED_NEURAL_ROOTS = {
    "neural_segmentation": NEURAL / "weights/segmentation3d_new_set_og",
    "neural_regression": NEURAL / "weights/regression3d_new_set_og",
}


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path.resolve())


def install_widget_import_stubs() -> None:
    """Stub GUI-only imports so actual widget path logic can be loaded headlessly."""
    napari = types.ModuleType("napari")
    napari.layers = types.SimpleNamespace(Image=object)
    napari.current_viewer = lambda: None

    napari_qt = types.ModuleType("napari.qt")
    napari_threading = types.ModuleType("napari.qt.threading")
    napari_threading.thread_worker = lambda func: func

    notifications = types.SimpleNamespace(
        show_warning=lambda *args, **kwargs: None,
        show_error=lambda *args, **kwargs: None,
        show_info=lambda *args, **kwargs: None,
    )
    napari_utils = types.ModuleType("napari.utils")
    napari_utils.notifications = notifications

    magicgui = types.ModuleType("magicgui")

    def magic_factory(*args, **kwargs):
        def decorate(func):
            return func

        return decorate

    magicgui.magic_factory = magic_factory
    magicgui_widgets = types.ModuleType("magicgui.widgets")
    magicgui_widgets.ProgressBar = type("ProgressBar", (), {})

    sys.modules.update(
        {
            "napari": napari,
            "napari.qt": napari_qt,
            "napari.qt.threading": napari_threading,
            "napari.utils": napari_utils,
            "magicgui": magicgui,
            "magicgui.widgets": magicgui_widgets,
        }
    )


def cli_config_check(predict_module, section: str, model_dir: Path) -> dict:
    cfg = OmegaConf.load(REPO / "configs/predict.yaml")
    OmegaConf.set_struct(cfg, False)
    cfg.inference_dir = str(GASTRULOID / "test_input")
    cfg.device = "cpu"
    node = cfg[section]
    node.model_dir = str(model_dir)
    try:
        loaded = predict_module.load_config(node, cfg, allow_missing=False)
    except Exception as exc:
        return {
            "accepted": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "checkpoint": None,
        }
    return {
        "accepted": True,
        "error_type": None,
        "error": None,
        "checkpoint": str(Path(loaded.ckpt_path).resolve()),
    }


def matrix_row(interface: str, case: str, role: str, supplied: Path, **extra) -> dict:
    row = {
        "interface": interface,
        "case": case,
        "role": role,
        "supplied_path": rel(supplied),
        "path_exists": supplied.exists(),
        "accepted": extra.pop("accepted", None),
        "resolved_path": extra.pop("resolved_path", None),
        "checkpoint": extra.pop("checkpoint", None),
        "finding": extra.pop("finding", ""),
    }
    if extra:
        row["finding"] += ("; " if row["finding"] else "") + json.dumps(extra, sort_keys=True)
    return row


def main() -> None:
    started = time.perf_counter()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    RUNTIME_TMP.mkdir(parents=True, exist_ok=True)

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    manifest = PluginManifest.from_file(REPO / "napari_dare3d/napari.yaml")
    manifest_entrypoints = [
        {"name": ep.name, "value": ep.value}
        for ep in importlib.metadata.entry_points(group="napari.manifest")
        if "dare3d" in ep.name.lower() or "dare3d" in ep.value.lower()
    ]

    # Import the actual widget module with GUI imports stubbed. Import it first
    # from a no-data working directory to test default binding and source fallback.
    install_widget_import_stubs()
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(dir=RUNTIME_TMP, prefix="widget_path_") as tmp_name:
        tmp_path = Path(tmp_name)
        os.chdir(tmp_path)
        try:
            sys.modules.pop("napari_dare3d._widget", None)
            widget = importlib.import_module("napari_dare3d._widget")
            signature_before = inspect.signature(widget.dare3d_widget)
            seg_default_before = signature_before.parameters["seg_model_dir"].default
            reg_default_before = signature_before.parameters["reg_model_dir"].default
            outside_root = widget._data_root()
            source_fallback_candidate = (
                Path(widget.__file__).resolve().parents[2] / widget.DATA_ROOT_NAME
            )
            correct_repo_relative_candidate = (
                Path(widget.__file__).resolve().parents[1] / widget.DATA_ROOT_NAME
            )

            fake_root = tmp_path / widget.DATA_ROOT_NAME / "Gastruloid_241025"
            fake_seg = fake_root / "weights/segmentation3d_exp10-b"
            fake_reg = fake_root / "weights/regression3d_exp10-b"
            fake_seg.mkdir(parents=True)
            fake_reg.mkdir(parents=True)
            helper_after_download = {
                "root": widget._data_root(),
                "seg": widget._default_model_dir("segmentation3d_exp10-b"),
                "reg": widget._default_model_dir("regression3d_exp10-b"),
            }
            helper_after_download_paths_exist = (
                helper_after_download["seg"].is_dir()
                and helper_after_download["reg"].is_dir()
            )
            signature_after = inspect.signature(widget.dare3d_widget)
            bound_defaults_after_download = {
                "seg": signature_after.parameters["seg_model_dir"].default,
                "reg": signature_after.parameters["reg_model_dir"].default,
            }
        finally:
            os.chdir(original_cwd)

    cwd_root = widget._data_root()
    widget_defaults_at_repo = {
        "seg": widget._default_model_dir("segmentation3d_exp10-b"),
        "reg": widget._default_model_dir("regression3d_exp10-b"),
        "image": widget._default_image_path(),
    }

    from napari_dare3d import _api
    from dare3d import predict as predict_module

    import_origins = {
        "widget": rel(Path(widget.__file__)),
        "api": rel(Path(_api.__file__)),
        "predict": rel(Path(predict_module.__file__)),
    }

    rows: list[dict] = []
    rows.extend(
        [
            matrix_row(
                "root_readme_layout",
                "gastruloid",
                "segmentation",
                MODEL_ROOTS["gastruloid_segmentation"],
                accepted=MODEL_ROOTS["gastruloid_segmentation"].exists(),
                finding="documented <case>/weights layout",
            ),
            matrix_row(
                "root_readme_layout",
                "gastruloid",
                "regression",
                MODEL_ROOTS["gastruloid_regression"],
                accepted=MODEL_ROOTS["gastruloid_regression"].exists(),
                finding="documented <case>/weights layout",
            ),
            matrix_row(
                "root_readme_layout",
                "neural",
                "segmentation",
                DOCUMENTED_NEURAL_ROOTS["neural_segmentation"],
                accepted=False,
                finding="documented <case>/weights layout is absent",
            ),
            matrix_row(
                "root_readme_layout",
                "neural",
                "regression",
                DOCUMENTED_NEURAL_ROOTS["neural_regression"],
                accepted=False,
                finding="documented <case>/weights layout is absent",
            ),
        ]
    )

    api_resolution: dict[str, dict] = {}
    for name, supplied in MODEL_ROOTS.items():
        resolved = Path(_api._resolve_model_dir(str(supplied))).resolve()
        role = "segmentation" if "segmentation" in name else "regression"
        case = "gastruloid" if name.startswith("gastruloid") else "neural"
        config = resolved / ".hydra/config.yaml"
        checkpoint = resolved / "checkpoints/last.ckpt"
        loaded = _api._load_inference_cfg(
            str(supplied),
            str(GASTRULOID / "test_input"),
            "cpu",
        )
        api_resolution[name] = {
            "supplied": rel(supplied),
            "resolved": rel(resolved),
            "config_exists": config.is_file(),
            "last_checkpoint_exists": checkpoint.is_file(),
            "loaded_checkpoint": rel(Path(loaded.ckpt_path)),
            "loaded_default_scale": list(loaded.data.test_data.default_scale),
        }
        rows.append(
            matrix_row(
                "napari_headless_api",
                case,
                role,
                supplied,
                accepted=config.is_file() and checkpoint.is_file(),
                resolved_path=rel(resolved),
                checkpoint=rel(checkpoint),
                finding="API resolves flat or runs/<date> model layouts",
            )
        )

    cli_checks: dict[str, dict] = {}
    cli_specs = {
        "gastruloid_segmentation_flat": (
            "segmentation",
            MODEL_ROOTS["gastruloid_segmentation"],
            "gastruloid",
        ),
        "gastruloid_regression_flat": (
            "regression",
            MODEL_ROOTS["gastruloid_regression"],
            "gastruloid",
        ),
        "neural_segmentation_top_level": (
            "segmentation",
            MODEL_ROOTS["neural_segmentation"],
            "neural",
        ),
        "neural_regression_top_level": (
            "regression",
            MODEL_ROOTS["neural_regression"],
            "neural",
        ),
        "neural_segmentation_resolved_run": (
            "segmentation",
            Path(_api._resolve_model_dir(str(MODEL_ROOTS["neural_segmentation"]))),
            "neural",
        ),
        "neural_regression_resolved_run": (
            "regression",
            Path(_api._resolve_model_dir(str(MODEL_ROOTS["neural_regression"]))),
            "neural",
        ),
    }
    for name, (section, supplied, case) in cli_specs.items():
        result = cli_config_check(predict_module, section, supplied)
        cli_checks[name] = {
            **result,
            "supplied": rel(supplied),
            "checkpoint": rel(Path(result["checkpoint"])) if result["checkpoint"] else None,
        }
        rows.append(
            matrix_row(
                "predict_cli_config_loader",
                case,
                section,
                supplied,
                accepted=result["accepted"],
                checkpoint=cli_checks[name]["checkpoint"],
                finding=(
                    "direct .hydra/config.yaml + checkpoints/last.ckpt required"
                    if result["accepted"]
                    else f"{result['error_type']}: {result['error']}"
                ),
            )
        )

    notebook = json.loads(
        (REPO / "notebooks/Run_dare3d_Prediction.ipynb").read_text(encoding="utf-8")
    )
    notebook_source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    package_demo_dir = REPO / "dare3d/demo_files"
    notebook_check = {
        "uses_get_path_to_demo_folder": "get_path_to_demo_folder()" in notebook_source,
        "package_demo_dir": rel(package_demo_dir),
        "package_demo_dir_exists": package_demo_dir.exists(),
        "verified_root_bundle": rel(DATA),
        "verified_root_bundle_exists": DATA.is_dir(),
        "root_bundle_is_reused": package_demo_dir.resolve() == DATA.resolve(),
        "effect": (
            "Executing the demo cell calls the separate package-local first-run downloader "
            "and does not reuse the verified repository-root bundle."
        ),
    }
    rows.append(
        matrix_row(
            "prediction_notebook",
            "gastruloid",
            "demo_data",
            package_demo_dir,
            accepted=False,
            finding=notebook_check["effect"],
        )
    )

    api_signature = inspect.signature(_api.infer_stack)
    api_defaults = {
        key: api_signature.parameters[key].default
        for key in (
            "device",
            "overlap",
            "batch_size",
            "threshold",
            "min_weighted_prob",
            "default_scale",
        )
    }
    segment_source = inspect.getsource(_api._segment)
    protocol_comparison = {
        "plugin_and_predict_defaults": {
            "checkpoint": "last.ckpt (hard-coded by infer_stack/_load_inference_cfg)",
            "overlap": api_defaults["overlap"],
            "threshold_operator": ">",
            "threshold": api_defaults["threshold"],
            "min_weighted_probability": api_defaults["min_weighted_prob"],
            "temporal_dilation_before_components": "absent",
            "default_scale_override": api_defaults["default_scale"],
            "effective_nuclei_scale_when_scales_json_absent": [0.621, 0.621, 2.0],
        },
        "released_nuclei_evaluation": {
            "identified_checkpoint": "epoch_067.ckpt",
            "overlap": 0.5,
            "threshold_operator": ">=",
            "threshold": 0.55,
            "min_weighted_probability": 0.15,
            "temporal_dilation_before_components": "present (+/-1 frame)",
            "training_log_reconstructed_scale": [0.912, 0.912, 0.912],
            "original_scale_file": "absent",
        },
        "static_source_checks": {
            "plugin_uses_strict_greater_than": "movie > threshold" in segment_source,
            "plugin_segment_omits_dilate_img": "dilate_img" not in segment_source,
        },
        "interpretation": (
            "The plugin/CLI are usable prediction interfaces, but their defaults do not "
            "instantiate the released quantitative-evaluation protocol and cannot be used "
            "as a manuscript-result reproduction command without undocumented overrides "
            "and the missing original scale provenance."
        ),
    }

    # One minimal real API run: three frames are the smallest input that reaches
    # one prediction for the three-channel nuclei model. No GUI and no regression.
    tempfile.tempdir = str(RUNTIME_TMP)
    smoke_stages: list[str] = []
    smoke_started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    movie = tifffile.memmap(GASTRULOID / "test_input/movie2.tif")
    smoke_detections = _api.infer_stack(
        np.asarray(movie[:3]),
        str(MODEL_ROOTS["gastruloid_segmentation"]),
        reg_model_dir=None,
        device="gpu",
        progress_cb=smoke_stages.append,
    )
    smoke_layers = _api.to_layer_data(smoke_detections)
    smoke = {
        "scope": "first three frames, segmentation only, public plugin defaults",
        "input": rel(GASTRULOID / "test_input/movie2.tif"),
        "input_slice_t": [0, 2],
        "input_shape_tzyx": list(movie[:3].shape),
        "model_dir": rel(MODEL_ROOTS["gastruloid_segmentation"]),
        "checkpoint_selected": rel(MODEL_ROOTS["gastruloid_segmentation"] / "checkpoints/last.ckpt"),
        "stages": smoke_stages,
        "detections": len(smoke_detections),
        "detection_records": smoke_detections,
        "layer_count": len(smoke_layers),
        "layer_summaries": [
            {
                "type": layer_type,
                "name": kwargs.get("name"),
                "data_shape": list(np.asarray(layer_data).shape),
            }
            for layer_data, kwargs, layer_type in smoke_layers
        ],
        "elapsed_seconds": time.perf_counter() - smoke_started,
        "peak_gpu_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        ),
    }

    widget_path_check = {
        "import_method": "actual widget source with GUI-only modules stubbed; no GUI launched",
        "manifest_commands": [command.id for command in manifest.contributions.commands],
        "manifest_widgets": [entry.command for entry in manifest.contributions.widgets],
        "installed_manifest_entrypoints": manifest_entrypoints,
        "import_from_no_data_cwd": {
            "cwd_root_result": str(outside_root) if outside_root else None,
            "source_fallback_candidate": rel(source_fallback_candidate),
            "source_fallback_exists": source_fallback_candidate.is_dir(),
            "correct_repo_relative_candidate": rel(correct_repo_relative_candidate),
            "correct_candidate_exists": correct_repo_relative_candidate.is_dir(),
            "bound_seg_default": str(seg_default_before),
            "bound_reg_default": str(reg_default_before),
        },
        "same_session_after_download_fixture": {
            "helper_root": str(helper_after_download["root"]),
            "helper_seg": str(helper_after_download["seg"]),
            "helper_reg": str(helper_after_download["reg"]),
            "bound_seg_default": str(bound_defaults_after_download["seg"]),
            "bound_reg_default": str(bound_defaults_after_download["reg"]),
            "defaults_refresh": (
                bound_defaults_after_download["seg"] != seg_default_before
                or bound_defaults_after_download["reg"] != reg_default_before
            ),
        },
        "from_repository_cwd": {
            "root": rel(cwd_root) if cwd_root else None,
            "seg_default": rel(widget_defaults_at_repo["seg"]),
            "reg_default": rel(widget_defaults_at_repo["reg"]),
            "image_default": rel(widget_defaults_at_repo["image"]),
        },
        "interpretation": (
            "Auto-fill works when napari starts in the repository/data parent. The "
            "source-relative fallback is one parent too high, and defaults are bound "
            "when the widget module imports, so a download in the same session does "
            "not refresh already-bound model fields."
        ),
    }

    assertions = {
        "current_checkout_imports_used": all(
            str((REPO / path).resolve()) == str(Path(origin).resolve())
            for path, origin in (
                ("napari_dare3d/_widget.py", widget.__file__),
                ("napari_dare3d/_api.py", _api.__file__),
                ("dare3d/predict.py", predict_module.__file__),
            )
        ),
        "manifest_has_three_commands_and_widgets": (
            len(manifest.contributions.commands) == 3
            and len(manifest.contributions.widgets) == 3
        ),
        "installed_napari_manifest_entrypoint_present": any(
            item["name"] == "dare3d" for item in manifest_entrypoints
        ),
        "widget_autofill_works_from_repository_cwd": (
            cwd_root is not None
            and widget_defaults_at_repo["seg"] == MODEL_ROOTS["gastruloid_segmentation"]
            and widget_defaults_at_repo["reg"] == MODEL_ROOTS["gastruloid_regression"]
            and widget_defaults_at_repo["image"].is_file()
        ),
        "widget_source_relative_fallback_defect_reproduced": (
            outside_root is None
            and not source_fallback_candidate.is_dir()
            and correct_repo_relative_candidate.is_dir()
        ),
        "widget_defaults_do_not_refresh_after_same_session_download": (
            helper_after_download_paths_exist
            and not widget_path_check["same_session_after_download_fixture"]["defaults_refresh"]
        ),
        "gastruloid_documented_weight_paths_exist": all(
            MODEL_ROOTS[key].is_dir()
            for key in ("gastruloid_segmentation", "gastruloid_regression")
        ),
        "neural_documented_weight_paths_are_absent": all(
            not path.exists() for path in DOCUMENTED_NEURAL_ROOTS.values()
        ),
        "plugin_api_resolves_all_four_actual_model_roots": all(
            value["config_exists"] and value["last_checkpoint_exists"]
            for value in api_resolution.values()
        ),
        "cli_accepts_flat_gastruloid_model_roots": (
            cli_checks["gastruloid_segmentation_flat"]["accepted"]
            and cli_checks["gastruloid_regression_flat"]["accepted"]
        ),
        "cli_rejects_unresolved_neural_top_level_roots": (
            not cli_checks["neural_segmentation_top_level"]["accepted"]
            and not cli_checks["neural_regression_top_level"]["accepted"]
        ),
        "cli_accepts_explicit_neural_run_directories": (
            cli_checks["neural_segmentation_resolved_run"]["accepted"]
            and cli_checks["neural_regression_resolved_run"]["accepted"]
        ),
        "prediction_notebook_does_not_reuse_root_bundle": (
            notebook_check["uses_get_path_to_demo_folder"]
            and not notebook_check["package_demo_dir_exists"]
            and notebook_check["verified_root_bundle_exists"]
            and not notebook_check["root_bundle_is_reused"]
        ),
        "public_plugin_protocol_differs_from_released_protocol": (
            api_defaults["threshold"] == 0.5
            and api_defaults["min_weighted_prob"] == 0.1
            and api_defaults["overlap"] == 0.25
            and "movie > threshold" in segment_source
            and "dilate_img" not in segment_source
        ),
        "headless_three_frame_api_smoke_completed": (
            smoke_stages == ["segmentation", "done"]
            and isinstance(smoke_detections, list)
        ),
    }

    report = {
        "scope": (
            "Read-only path/configuration checks plus one minimal three-frame "
            "segmentation-only plugin API smoke run; no GUI, download, regression, "
            "evaluation, retraining, or released-file mutation."
        ),
        "source_commit": "fe2b14d732359f2bdaf8b197574ad818899ce123",
        "import_origins": import_origins,
        "widget_and_manifest": widget_path_check,
        "model_path_resolution": api_resolution,
        "cli_config_checks": cli_checks,
        "prediction_notebook": notebook_check,
        "protocol_comparison": protocol_comparison,
        "headless_api_smoke": smoke,
        "assertions": assertions,
        "elapsed_seconds": time.perf_counter() - started,
    }

    with CSV_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    JSON_OUTPUT.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")

    failed = [name for name, passed in assertions.items() if not passed]
    print(
        json.dumps(
            {
                "status": "complete" if not failed else "assertion_failure",
                "assertions_passed": sum(assertions.values()),
                "assertions_total": len(assertions),
                "failed": failed,
                "headless_smoke": {
                    "stages": smoke_stages,
                    "detections": len(smoke_detections),
                    "elapsed_seconds": smoke["elapsed_seconds"],
                },
                "json": rel(JSON_OUTPUT),
                "csv": rel(CSV_OUTPUT),
            },
            indent=2,
        ),
        flush=True,
    )
    if failed:
        raise AssertionError(f"Workflow path assertions failed: {failed}")


if __name__ == "__main__":
    main()
