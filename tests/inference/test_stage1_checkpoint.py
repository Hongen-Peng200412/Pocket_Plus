"""正式 checkpoint loader 的 strict wrapper 生命周期测试。"""

from __future__ import annotations

from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("hydra")

from src.inference.checkpoint import load_stage1_wrapper


class _DummyWrapper(torch.nn.Module):
    """记录 `on_load_checkpoint` 是否在 strict 权重恢复后执行。"""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.loaded_runtime_value = None

    def on_load_checkpoint(self, checkpoint):
        self.loaded_runtime_value = checkpoint["runtime_value"]


def test_loader_restores_complete_wrapper_and_lifecycle(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        "model:\n  _target_: ignored.Dummy\ntrain:\n  optimizer: {}\n  scheduler: {}\n",
        encoding="utf-8",
    )
    source = _DummyWrapper()
    source.weight.data.fill_(3.0)
    checkpoint_path = checkpoint_dir / "BEST.ckpt"
    torch.save({"state_dict": source.state_dict(), "runtime_value": 17}, checkpoint_path)

    monkeypatch.setattr("hydra.utils.instantiate", lambda *args, **kwargs: _DummyWrapper())
    restored = load_stage1_wrapper(
        checkpoint_path=checkpoint_path,
        resolved_config_path=None,
        map_location="cpu",
    )
    assert restored.training is False
    assert restored.weight.item() == 3.0
    assert restored.loaded_runtime_value == 17
