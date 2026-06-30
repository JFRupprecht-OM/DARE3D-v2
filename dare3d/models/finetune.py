"""Fine-tuning / transfer-learning engine for DARE3d (PyTorch-Lightning, 3D, PyTorch-only).

Shared by ``RegressionLitModule`` and ``SegmentationLitModule`` via :class:`FineTuneMixin`.
The engine: load a pretrained checkpoint into ``self.net``, freeze the backbone with a
per-net preset, continue training at a low (optionally discriminative) LR, and write a
provenance sidecar next to the saved checkpoint. The save format is unchanged — the same
Lightning ``last.ckpt`` the inference path already loads.

Five correctness points carried over from the DARE2D reference (see the fine-tuning HANDOFF):
  1. The frozen-BatchNorm policy is enforced in :meth:`FineTuneMixin.train` (the mode toggle),
     NOT an epoch hook — so it survives the mid-epoch eval()->train() re-arm under
     ``val_check_interval < 1.0``. A "frozen" backbone whose BN stays in train() silently
     drifts its running stats; this is the bug the feature exists to kill.
  2. The base is loaded into ``self.net`` with normalized keys (strip the ``net.`` prefix,
     drop torchmetrics/criterion buffers), never into the whole LightningModule.
  3. Freezing uses a per-net preset (encoder/decoder/head), never a flat "unfreeze last N".
  4. SwinUNETR has no BatchNorm3d — callers must guard BN tests with a presence assert.
  5. Determinism is approximate with cuDNN-on; the sidecar records this.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import torch
from torch import nn

try:  # monai layout differs across versions; SkipConnection marks U-Net decoder branches
    from monai.networks.layers.simplelayers import SkipConnection
except Exception:  # pragma: no cover - fallback for other monai layouts
    from monai.networks.layers import SkipConnection
try:
    from monai.networks.nets import UNet as _MonaiUNet
except Exception:  # pragma: no cover
    _MonaiUNet = ()


class FineTuneError(RuntimeError):
    """Raised for a wrong-stage base checkpoint or an unsupported freeze request."""


# --------------------------------------------------------------------------- #
# Base-weight loading (scoped to net, stage-aware)                            #
# --------------------------------------------------------------------------- #
def load_net_state_dict(net: nn.Module, ckpt_path: str, *, stage: Optional[str] = None) -> List[str]:
    """Load a base checkpoint into ``net`` (not the LightningModule).

    Accepts a Lightning ckpt (weights under ``["state_dict"]``), a plain net ``state_dict``,
    or a full module dump. Keys are normalized: take ``["state_dict"]`` if present, keep only
    the ``net.*`` entries and strip that prefix (dropping torchmetrics/criterion buffers), then
    ``net.load_state_dict(strict=True)``. A key/shape mismatch raises a stage-aware
    :class:`FineTuneError` so a wrong-stage pick fails in seconds (before any preprocessing).
    """
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FineTuneError(f"Base checkpoint not found: {ckpt_path!r}")
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    if not isinstance(sd, dict):
        raise FineTuneError(f"Unexpected checkpoint contents in {ckpt_path!r} (no state_dict).")

    net_sd = {k[len("net."):]: v for k, v in sd.items() if k.startswith("net.")}
    if not net_sd:  # maybe it is already a plain net state_dict
        net_sd = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}

    try:
        net.load_state_dict(net_sd, strict=True)
    except RuntimeError as exc:
        other = {"segmentation": "regression", "regression": "segmentation"}.get((stage or "").lower())
        hint = (
            f" This looks like it may not be a {stage.upper()} checkpoint"
            f" (selected stage={stage!r}; did you mean the {other.upper()} model?)."
            if other else ""
        )
        raise FineTuneError(
            f"Base weights do not match this {stage or 'model'}'s network.{hint}\n"
            f"Underlying error: {exc}"
        ) from exc
    return sorted(net_sd.keys())


def base_sha256(ckpt_path: str) -> str:
    h = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Freeze presets (per-net)                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class FreezeResult:
    preset: str
    frozen_bn3d: List[nn.modules.batchnorm._BatchNorm] = field(default_factory=list)
    head_param_ids: Set[int] = field(default_factory=set)
    n_trainable: int = 0
    n_frozen: int = 0
    trainable_modules: List[str] = field(default_factory=list)
    frozen_modules: List[str] = field(default_factory=list)


def _is_multiscale_unet(net: nn.Module) -> bool:
    bb = getattr(net, "backbone", None)
    return (
        hasattr(net, "conv_modules")
        and bb is not None
        and isinstance(bb, _MonaiUNet)
        and isinstance(getattr(bb, "model", None), nn.Sequential)
    )


def _is_regression_head_net(net: nn.Module) -> bool:
    # Both reg nets expose the four division-axis heads; only the backbone differs
    # (RegressionNet -> .blocks ; Regression3dCNN -> .backbone EfficientNet).
    return all(hasattr(net, h) for h in ("head1_len", "head1_angle", "head2_len", "head2_angle"))


def classify_unet_encoder_decoder(unet) -> "tuple[List[nn.Module], List[nn.Module]]":
    """Split a monai ``UNet`` into encoder vs decoder modules.

    monai builds ``unet.model`` recursively as ``Sequential(down, SkipConnection(submodule), up)``.
    The ``down`` at each depth is the encoder; the ``up`` is the decoder; the innermost node
    (no SkipConnection) is the bottleneck (counted as the deepest encoder stage). Returns
    ``(encoder_stages, decoder_stages)`` with encoder ordered shallow -> deep so that
    "unfreeze last K stages" means the K deepest (closest to the bottleneck).
    """
    enc: List[nn.Module] = []
    dec: List[nn.Module] = []

    def walk(node: nn.Module) -> None:
        if (
            isinstance(node, nn.Sequential)
            and len(node) == 3
            and isinstance(node[1], SkipConnection)
        ):
            enc.append(node[0])           # down / encoder (shallow first)
            walk(node[1].submodule)       # deeper level
            dec.append(node[2])           # up / decoder
        else:
            enc.append(node)              # bottleneck (deepest)

    walk(unet.model)
    return enc, dec


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(flag)


def apply_freeze(net: nn.Module, preset: str, unfreeze_last_stages: int = 0) -> FreezeResult:
    """Apply a freeze preset to ``net`` in place and report what is trainable.

    Presets: ``none`` (full fine-tune), ``encoder`` (freeze backbone/encoder, train decoder +
    head(s)), ``encoder_partial`` (also unfreeze the last ``unfreeze_last_stages`` encoder
    stages). Only the two default nets get partial presets; other nets (SwinUNETR, segres,
    unknown) support ``none`` only and otherwise raise :class:`FineTuneError`.
    """
    preset = (preset or "encoder").lower()
    head_modules: List[nn.Module] = []

    if preset == "none":
        _set_requires_grad(net, True)
        head_modules = [net]  # uniform ft_lr for full fine-tune

    elif _is_regression_head_net(net):
        if preset not in ("encoder", "encoder_partial", "backbone"):
            raise FineTuneError(f"Unknown freeze_preset {preset!r} (use none/encoder/encoder_partial).")
        _set_requires_grad(net, False)
        head_modules = [net.head1_len, net.head1_angle, net.head2_len, net.head2_angle]
        for m in head_modules:
            _set_requires_grad(m, True)
        if unfreeze_last_stages > 0:
            # backbone stages differ by class: RegressionNet.blocks vs Regression3dCNN.backbone._blocks
            stages = (list(net.blocks) if hasattr(net, "blocks")
                      else list(getattr(getattr(net, "backbone", None), "_blocks", [])))
            for blk in stages[-unfreeze_last_stages:]:
                _set_requires_grad(blk, True)

    elif _is_multiscale_unet(net):
        if preset not in ("encoder", "encoder_partial", "backbone"):
            raise FineTuneError(f"Unknown freeze_preset {preset!r} (use none/encoder/encoder_partial).")
        enc, dec = classify_unet_encoder_decoder(net.backbone)
        _set_requires_grad(net, False)
        train_mods = list(dec) + list(net.conv_modules)  # decoder + extra multiscale heads
        if unfreeze_last_stages > 0:
            train_mods += enc[-unfreeze_last_stages:]
        for m in train_mods:
            _set_requires_grad(m, True)
        # "head" for discriminative LR: the extra heads + the top output (last decoder stage)
        head_modules = list(net.conv_modules) + (dec[-1:] if dec else [])

    else:
        raise FineTuneError(
            f"Partial freeze is unsupported for net type {type(net).__name__!r} "
            f"(e.g. SwinUNETR / segres). Use freeze_preset=none for a full fine-tune."
        )

    # Guard: a backbone-only freeze on a single-scale seg net would leave nothing to train.
    n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    if n_trainable == 0:
        raise FineTuneError(
            f"freeze_preset={preset!r} leaves no trainable parameters for {type(net).__name__!r}. "
            "Try freeze_preset=encoder_partial (unfreeze_last_stages>0) or none."
        )

    head_param_ids = {id(p) for m in head_modules for p in m.parameters()}
    frozen_bn3d = [
        m for m in net.modules()
        if isinstance(m, nn.BatchNorm3d) and not any(p.requires_grad for p in m.parameters())
    ]
    trainable_names = [n for n, p in net.named_parameters() if p.requires_grad]
    frozen_names = [n for n, p in net.named_parameters() if not p.requires_grad]
    return FreezeResult(
        preset=preset,
        frozen_bn3d=frozen_bn3d,
        head_param_ids=head_param_ids,
        n_trainable=n_trainable,
        n_frozen=sum(p.numel() for p in net.parameters() if not p.requires_grad),
        trainable_modules=trainable_names,
        frozen_modules=frozen_names,
    )


def build_param_groups(net, head_param_ids, ft_lr, backbone_lr_mult, weight_decay):
    """Discriminative AdamW groups: head params at ``ft_lr``, other trainable at
    ``ft_lr * backbone_lr_mult``. Only ``requires_grad`` params are included."""
    head, back = [], []
    for p in net.parameters():
        if not p.requires_grad:
            continue
        (head if id(p) in head_param_ids else back).append(p)
    groups = []
    if head:
        groups.append({"params": head, "lr": ft_lr, "weight_decay": weight_decay})
    if back:
        groups.append({"params": back, "lr": ft_lr * backbone_lr_mult, "weight_decay": weight_decay})
    return groups


def finetune_scheduler(optimizer, warmup_epochs: int, max_epochs: int):
    """Linear warmup -> cosine anneal (epoch-stepped)."""
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    max_epochs = max(1, int(max_epochs))
    if warmup_epochs and warmup_epochs > 0:
        warmup_epochs = min(int(warmup_epochs), max(1, max_epochs - 1))
        warm = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        cos = CosineAnnealingLR(optimizer, T_max=max(1, max_epochs - warmup_epochs))
        return SequentialLR(optimizer, [warm, cos], milestones=[warmup_epochs])
    return CosineAnnealingLR(optimizer, T_max=max_epochs)


# --------------------------------------------------------------------------- #
# Provenance sidecar                                                           #
# --------------------------------------------------------------------------- #
def _git_short_sha() -> Optional[str]:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(
            ["git", "-C", here, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip() or None
    except Exception:
        return None


def write_finetune_sidecar(ckpt_dir: str, ft: dict, *, stage: Optional[str]) -> str:
    """Write ``finetune_config.json`` next to the checkpoints with full provenance."""
    os.makedirs(ckpt_dir, exist_ok=True)
    base = ft.get("base_ckpt")
    try:
        import lightning
        lightning_ver = lightning.__version__
    except Exception:
        lightning_ver = None
    info = {
        "base_model": {
            "path": base,
            "name": os.path.basename(base) if base else None,
            "sha256": base_sha256(base) if base and os.path.exists(base) else None,
            "size_bytes": os.path.getsize(base) if base and os.path.exists(base) else None,
            "stage": stage,
        },
        "hyperparameters": {
            k: ft.get(k) for k in (
                "bn_mode", "freeze_preset", "unfreeze_last_stages", "ft_lr", "backbone_lr_mult",
                "weight_decay", "lr_schedule", "warmup_epochs", "grad_clip", "batch_size",
                "augment", "augment_strength", "patience", "seed",
            )
        },
        "environment": {
            "torch": torch.__version__,
            "cudnn": torch.backends.cudnn.version(),
            "lightning": lightning_ver,
            "DARE3D_CUDNN": os.environ.get("DARE3D_CUDNN"),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "dare3d_git_sha": _git_short_sha(),
        },
        "determinism": "approximate — cuDNN-on 3D convs are not bitwise reproducible; "
                       "run-to-run numerical jitter is expected (Trainer deterministic='warn').",
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    out = os.path.join(ckpt_dir, "finetune_config.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return out


# --------------------------------------------------------------------------- #
# Mixin for the LightningModules                                               #
# --------------------------------------------------------------------------- #
class FineTuneMixin:
    """Adds fine-tuning to a LightningModule. Inherit it FIRST so ``train()`` overrides
    ``LightningModule.train``. Inert unless a ``finetune`` config with a ``base_ckpt`` is given.

    Wiring expected in the host module:
      - ``__init__``: ``self.init_finetune(finetune)``
      - ``setup("fit")``: ``if self._finetune_active: self.apply_finetune_setup()``
      - ``configure_optimizers``: ``if self._finetune_active: return self.finetune_optimizers()``
    """

    def init_finetune(self, finetune: Optional[dict]) -> None:
        self._ft: Optional[dict] = dict(finetune) if finetune else None
        base = (self._ft or {}).get("base_ckpt")
        self._finetune_active: bool = bool(self._ft and base not in (None, "", "???"))
        self._frozen_bn3d: List[nn.Module] = []
        self._ft_head_param_ids: Set[int] = set()
        self._ft_freeze: Optional[FreezeResult] = None

    @property
    def _bn_mode(self) -> str:
        return (self._ft or {}).get("bn_mode", "frozen")

    @property
    def finetune_stage(self) -> Optional[str]:
        return (self._ft or {}).get("stage")

    def load_base(self) -> Optional[List[str]]:
        """Load the base weights into ``self.net`` (fail-fast, stage-aware). Call before fit."""
        if not getattr(self, "_finetune_active", False):
            return None
        return load_net_state_dict(self.net, self._ft["base_ckpt"], stage=self.finetune_stage)

    def apply_finetune_setup(self) -> FreezeResult:
        """Apply the freeze preset, collect frozen BatchNorm3d, and enforce the BN policy now."""
        res = apply_freeze(
            self.net,
            self._ft.get("freeze_preset", "encoder"),
            int(self._ft.get("unfreeze_last_stages", 0) or 0),
        )
        self._ft_freeze = res
        self._frozen_bn3d = res.frozen_bn3d
        self._ft_head_param_ids = res.head_param_ids
        self.train()  # re-apply BN policy immediately (esp. for bn_mode == "frozen")
        return res

    def train(self, mode: bool = True):  # type: ignore[override]
        """Re-arm the frozen-BatchNorm policy on EVERY train()/eval() toggle.

        This is the decisive fix: an ``on_train_epoch_start`` hook misses the mid-epoch
        eval()->train() flip that Lightning does for validation (``val_check_interval < 1.0``),
        which would silently let frozen BN resume drifting its running stats.
        """
        super().train(mode)
        if mode and getattr(self, "_finetune_active", False) and self._bn_mode == "frozen":
            for m in self._frozen_bn3d:
                m.eval()  # running stats AND affine fixed (affine already requires_grad=False)
        # bn_mode == "adapt": leave frozen BN in train() so running stats re-estimate on the
        # new data; their affine stays requires_grad=False (set by apply_freeze) so it can't learn.
        return self

    def finetune_optimizers(self) -> Dict:
        """Discriminative AdamW + warmup->cosine. Used by ``configure_optimizers`` when active."""
        ft = self._ft
        groups = build_param_groups(
            self.net, self._ft_head_param_ids,
            float(ft["ft_lr"]), float(ft.get("backbone_lr_mult", 1.0)),
            float(ft.get("weight_decay", 0.0)),
        )
        optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999))
        self.optimizer = optimizer  # keep training_step's lr logging consistent
        max_epochs = int(getattr(self.trainer, "max_epochs", None) or 1)
        self.scheduler = finetune_scheduler(
            optimizer, int(ft.get("warmup_epochs", 0) or 0), max_epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": self.scheduler, "interval": "epoch", "frequency": 1},
        }

    def write_sidecar(self, ckpt_dir: str) -> Optional[str]:
        if not getattr(self, "_finetune_active", False):
            return None
        return write_finetune_sidecar(ckpt_dir, self._ft, stage=self.finetune_stage)
