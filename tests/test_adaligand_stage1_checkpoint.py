from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from src.train import _load_model_only_checkpoint


class _LifecycleModel(nn.Module):
    """记录 strict 权重恢复后 Lightning lifecycle 是否执行. """

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2))
        self.runtime_value: int | None = None

    def on_load_checkpoint(self, checkpoint: dict[str, object]) -> None:
        self.runtime_value = int(checkpoint["runtime_value"])


def test_model_only_checkpoint_is_strict_and_restores_wrapper_lifecycle(tmp_path: Path) -> None:
    """CPC1→CPC2 只恢复模型与 wrapper runtime, 不恢复优化器或 step. """

    checkpoint_path = tmp_path / "BEST.ckpt"
    torch.save(
        {
            "state_dict": {"weight": torch.tensor([3.0, 4.0])},
            "runtime_value": 19,
            "optimizer_states": [{"forbidden": True}],
            "global_step": 123,
        },
        checkpoint_path,
    )
    model = _LifecycleModel()

    _load_model_only_checkpoint(model, checkpoint_path, verbose=False)

    torch.testing.assert_close(model.weight, torch.tensor([3.0, 4.0]))
    assert model.runtime_value == 19
    assert not hasattr(model, "global_step")


def test_model_only_checkpoint_rejects_missing_or_unexpected_keys(tmp_path: Path) -> None:
    """CPC2 不得以 non-strict 方式吞掉模型结构漂移. """

    checkpoint_path = tmp_path / "bad.ckpt"
    torch.save({"state_dict": {"unexpected": torch.ones(1)}, "runtime_value": 7}, checkpoint_path)
    model = _LifecycleModel()

    with pytest.raises(RuntimeError):
        _load_model_only_checkpoint(model, checkpoint_path, verbose=False)
    assert model.runtime_value is None
