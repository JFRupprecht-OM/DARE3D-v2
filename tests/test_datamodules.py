"""Unit tests for `DareDataModule` — the real DARE3D datamodule.

Replaces the upstream MNIST template test, which imported a non-existent
`dare3d.data.mnist_datamodule.MNISTDataModule` (ImportError at collection). These
checks exercise the datamodule's dataloader-construction contract with tiny
in-memory datasets, so they need no DARE3D data, no network, and no GPU.
"""
import pytest
import torch
from lightning import LightningDataModule
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from dare3d.data.dare_datamodule import DareDataModule


def _toy_dataset(n: int) -> TensorDataset:
    """A tiny ``(n, 1, 4, 4, 4)`` image / scalar-label dataset standing in for a real one."""
    return TensorDataset(torch.zeros(n, 1, 4, 4, 4), torch.zeros(n))


@pytest.mark.parametrize("batch_size", [2, 4])
def test_dare_datamodule_dataloaders(batch_size: int) -> None:
    """`DareDataModule` builds train/val/test dataloaders with the configured batch
    size, shuffling train but preserving order for val/test.

    :param batch_size: Batch size handed to the datamodule.
    """
    dm = DareDataModule(
        train_data=_toy_dataset(8),
        val_data=_toy_dataset(4),
        test_data=_toy_dataset(4),
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
    )

    # Lightning contract + stored hyperparameters.
    assert isinstance(dm, LightningDataModule)
    assert dm.hparams.batch_size == batch_size
    assert dm.data_train is not None and dm.data_val is not None and dm.data_test is not None

    # Without a trainer, per-device batch size equals the requested batch size.
    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()
    test_dl = dm.test_dataloader()
    for dl in (train_dl, val_dl, test_dl):
        assert isinstance(dl, DataLoader)
        assert dl.batch_size == batch_size

    # Train shuffles; val/test preserve order.
    assert isinstance(train_dl.sampler, RandomSampler)
    assert isinstance(val_dl.sampler, SequentialSampler)
    assert isinstance(test_dl.sampler, SequentialSampler)

    # A batch has the requested size and the toy 3D shape.
    x, y = next(iter(train_dl))
    assert x.shape == (batch_size, 1, 4, 4, 4)
    assert len(y) == batch_size
