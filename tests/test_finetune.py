"""Unit tests for the fine-tune engine (dare3d.models.finetune).

CPU-only, no data bundle, small synthetic nets -> safe for CI / ``make test``. Covers the
HANDOFF §0 invariants: stage-aware base load into ``net``, per-net freeze presets,
discriminative LR groups, and the frozen-BatchNorm3d ``train()`` re-arm (frozen vs adapt).
"""
import tempfile
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from lightning import LightningModule

from dare3d.models.components.multiscale_unet import MultiScaleUNet
from dare3d.models.components.simple_regression_net import RegressionNet
from dare3d.models.finetune import (
    FineTuneError, FineTuneMixin, apply_freeze, build_param_groups,
    classify_unet_encoder_decoder, load_net_state_dict,
)


def small_unet():
    return MultiScaleUNet(
        spatial_dims=3, in_channels=3, out_channels=1, channels=(8, 16, 16),
        strides=(2, 2), norm="batch", num_res_units=1, bias=False, dropout=0.0,
        output_names=[], downsample_factors=[1],
    )


def small_regnet():
    return RegressionNet(input_channels=[-1, 0, 1], im_size=16, n_stages=2, start_filters=4)


class _FTModule(FineTuneMixin, LightningModule):
    """Minimal host so the real FineTuneMixin.train() override is exercised."""
    def __init__(self, net, ft):
        super().__init__()
        self.net = net
        self.init_finetune(ft)


def _ft(bn_mode="frozen", preset="encoder", stages=0):
    return dict(base_ckpt="dummy.ckpt", stage="regression", freeze_preset=preset,
                unfreeze_last_stages=stages, bn_mode=bn_mode, ft_lr=1e-4,
                backbone_lr_mult=0.1, weight_decay=1e-4, warmup_epochs=0)


# --------------------------------------------------------------------------- #
def test_load_net_state_dict_strips_prefix_and_drops_buffers():
    net = small_regnet()
    # a Lightning-style state_dict: net.* weights + foreign torchmetric/criterion buffers
    sd = {f"net.{k}": v for k, v in net.state_dict().items()}
    sd["train_loss.total"] = torch.tensor(3.0)
    sd["criterion.focal.class_weight"] = torch.ones(1)
    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
        torch.save({"state_dict": sd}, f.name)
        path = f.name
    fresh = small_regnet()
    loaded = load_net_state_dict(fresh, path, stage="regression")
    assert all(not k.startswith(("train_loss", "criterion")) for k in loaded)
    for k, v in net.state_dict().items():
        assert torch.equal(fresh.state_dict()[k], v)


def test_load_net_state_dict_stage_aware_error():
    reg = small_regnet()
    sd = {f"net.{k}": v for k, v in small_unet().state_dict().items()}
    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
        torch.save({"state_dict": sd}, f.name)
        path = f.name
    with pytest.raises(FineTuneError):
        load_net_state_dict(reg, path, stage="regression")


def test_classify_unet_encoder_decoder_partitions():
    net = small_unet()
    enc, dec = classify_unet_encoder_decoder(net.backbone)
    assert len(enc) >= 2 and len(dec) >= 1
    enc_ids = {id(p) for m in enc for p in m.parameters()}
    dec_ids = {id(p) for m in dec for p in m.parameters()}
    assert enc_ids and dec_ids and not (enc_ids & dec_ids)  # disjoint partition


def test_freeze_regression_encoder_trains_heads_only():
    net = small_regnet()
    res = apply_freeze(net, "encoder")
    assert all(not p.requires_grad for p in net.blocks.parameters())          # backbone frozen
    assert all(p.requires_grad for p in net.head1_len.parameters())           # heads trainable
    assert res.n_trainable > 0 and len(res.frozen_bn3d) == len(net.blocks)


def test_freeze_regression_partial_unfreezes_last_block():
    net = small_regnet()
    apply_freeze(net, "encoder_partial", unfreeze_last_stages=1)
    assert all(p.requires_grad for p in net.blocks[-1].parameters())
    assert all(not p.requires_grad for p in net.blocks[0].parameters())


def test_freeze_unet_encoder_trains_decoder():
    net = small_unet()
    res = apply_freeze(net, "encoder")
    n_tr = sum(p.numel() for p in net.parameters() if p.requires_grad)
    n_fr = sum(p.numel() for p in net.parameters() if not p.requires_grad)
    assert n_tr > 0 and n_fr > 0 and len(res.frozen_bn3d) > 0


def test_unsupported_net_partial_freeze_raises_but_none_ok():
    weird = nn.Sequential(nn.Conv3d(3, 3, 1))
    with pytest.raises(FineTuneError):
        apply_freeze(weird, "encoder")
    apply_freeze(weird, "none")  # full fine-tune is always allowed
    assert all(p.requires_grad for p in weird.parameters())


def test_discriminative_param_groups():
    net = small_regnet()
    res = apply_freeze(net, "encoder_partial", unfreeze_last_stages=1)
    groups = build_param_groups(net, res.head_param_ids, 1e-4, 0.1, 1e-4)
    lrs = sorted(g["lr"] for g in groups)
    assert lrs == pytest.approx([1e-5, 1e-4])  # backbone at ft_lr*mult, head at ft_lr


def test_bn_policy_frozen_survives_midepoch_rearm():
    net = small_unet()
    m = _FTModule(net, _ft(bn_mode="frozen"))
    res = m.apply_finetune_setup()
    assert res.frozen_bn3d, "test needs BatchNorm3d present"  # §0.4 guard
    fbn = res.frozen_bn3d[0]
    assert not fbn.training                       # eval right after setup
    m.eval(); m.train()                           # the mid-epoch eval()->train() re-arm
    assert not fbn.training                       # still eval (the decisive fix)
    rm = fbn.running_mean.clone()
    with torch.no_grad():
        for _ in range(3):
            m.net(torch.randn(2, 3, 16, 16, 16))
    assert torch.equal(rm, fbn.running_mean)      # running stats frozen


def _optim_module(lr_schedule, warmup_epochs=2, max_epochs=10):
    m = _FTModule(small_regnet(), {**_ft(), "lr_schedule": lr_schedule,
                                   "warmup_epochs": warmup_epochs})
    m.apply_finetune_setup()
    m._trainer = SimpleNamespace(max_epochs=max_epochs)  # finetune_optimizers reads max_epochs
    return m


def test_lr_schedule_selects_scheduler_shape():
    from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR
    sch = _optim_module("warmup_cosine").finetune_optimizers()["lr_scheduler"]["scheduler"]
    assert isinstance(sch, SequentialLR)                  # warmup -> cosine
    # "cosine" must win over a nonzero warmup_epochs: pure cosine, no warmup phase,
    # so the sidecar's recorded lr_schedule always matches what actually ran.
    sch = _optim_module("cosine").finetune_optimizers()["lr_scheduler"]["scheduler"]
    assert isinstance(sch, CosineAnnealingLR)


def test_lr_schedule_invalid_raises():
    with pytest.raises(FineTuneError):
        _optim_module("constant").finetune_optimizers()


def test_bn_policy_adapt_reestimates_stats():
    net = small_unet()
    m = _FTModule(net, _ft(bn_mode="adapt"))
    res = m.apply_finetune_setup()
    fbn = res.frozen_bn3d[0]
    assert fbn.training                            # adapt leaves frozen BN in train()
    assert not fbn.weight.requires_grad            # affine not learnable
    rm = fbn.running_mean.clone()
    m.train()
    with torch.no_grad():
        for _ in range(3):
            m.net(torch.randn(2, 3, 16, 16, 16))
    assert not torch.equal(rm, fbn.running_mean)   # running stats re-estimated


# --------------------------------------------------------------------------- #
# Integration: the invariants above, but under a REAL Trainer.fit rather than a
# hand-simulated eval()->train() flip. This is the test that would catch an
# ordering regression (freeze after configure_optimizers, param groups over
# net.parameters() instead of requires_grad, etc.).
# --------------------------------------------------------------------------- #
class _FitModule(FineTuneMixin, LightningModule):
    """Faithful mirror of the real modules' fit wiring: setup('fit') applies the freeze,
    configure_optimizers delegates to the mixin, training_step drives one backward pass."""
    def __init__(self, net, ft):
        super().__init__()
        self.net = net
        self.init_finetune(ft)

    def setup(self, stage):
        if stage == "fit" and self._finetune_active:
            self.apply_finetune_setup()

    def configure_optimizers(self):
        return self.finetune_optimizers()

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.net(x)["heatmaps"][0]
        return torch.nn.functional.mse_loss(out, y)


class _TinyVolumes(torch.utils.data.Dataset):
    def __init__(self, n=2):
        self.x = torch.randn(n, 3, 16, 16, 16)
        self.y = torch.zeros(n, 1, 16, 16, 16)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def _run_one_fit(bn_mode):
    import lightning as L
    net = small_unet()
    m = _FitModule(net, _ft(bn_mode=bn_mode))
    m.load_base = lambda: None  # no base ckpt in this synthetic test
    dl = torch.utils.data.DataLoader(_TinyVolumes(2), batch_size=2)
    trainer = L.Trainer(accelerator="cpu", max_epochs=1, limit_train_batches=1,
                        num_sanity_val_steps=0, enable_checkpointing=False,
                        logger=False, enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(m, dl)
    return m, net


def test_fit_frozen_bn_stays_eval_and_stats_unchanged():
    """After a real Trainer.fit, frozen BatchNorm3d is in eval AND its running stats are
    untouched — the mid-epoch train() re-arm did not clobber the freeze."""
    m, net = _run_one_fit("frozen")
    fbns = m._frozen_bn3d
    assert fbns, "test needs BatchNorm3d present"
    # Decisive signal: frozen BN is in eval AFTER a real fit (the mid-epoch train() re-arm
    # did not flip it back). Running stats can only drift in train mode, so a frozen BN that
    # never entered train keeps its init defaults (running_mean==0, running_var==1) —
    # asserting that directly proves no stat update happened across the whole fit.
    assert all(not bn.training for bn in fbns)
    assert all(not bn.weight.requires_grad and not bn.bias.requires_grad for bn in fbns)
    for bn in fbns:
        assert torch.count_nonzero(bn.running_mean) == 0        # never updated from init 0
        assert torch.equal(bn.running_var, torch.ones_like(bn.running_var))  # still init 1


def test_fit_optimizer_groups_match_requires_grad_exactly():
    """The optimizer sees EXACTLY the trainable params — no frozen param handed in, no
    trainable param silently omitted."""
    m, net = _run_one_fit("frozen")
    opt = m.optimizers().optimizer if hasattr(m.optimizers(), "optimizer") else m.optimizers()
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    trainable = {id(p) for p in net.parameters() if p.requires_grad}
    assert in_opt == trainable
    assert all(not p.requires_grad for p in net.parameters() if id(p) not in in_opt)
