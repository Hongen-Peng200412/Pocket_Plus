"""正式 checkpoint loader 的 strict wrapper 生命周期测试. """

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("hydra")

from src.inference import checkpoint as checkpoint_module
from src.inference.checkpoint import (
    load_stage1_wrapper,
    resolve_checkpoint_source_path,
)


class _DummyWrapper(torch.nn.Module):
    """记录 `on_load_checkpoint` 是否在 strict 权重恢复后执行. """

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
        allow_current_workspace_code=True,
    )
    assert restored.training is False
    assert restored.weight.item() == 3.0
    assert restored.loaded_runtime_value == 17


def test_loader_rejects_missing_snapshot_by_default(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "run" / "checkpoints" / "BEST.ckpt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.touch()

    with pytest.raises(FileNotFoundError, match="src_snapshot"):
        load_stage1_wrapper(
            checkpoint_path=checkpoint_path,
            resolved_config_path=None,
            map_location="cpu",
        )


def test_snapshot_path_is_bound_to_checkpoint_run(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "run" / "checkpoints" / "BEST.ckpt"
    source_path = tmp_path / "run" / "src_snapshot" / "src"
    checkpoint_path.parent.mkdir(parents=True)
    source_path.mkdir(parents=True)
    checkpoint_path.touch()
    (source_path / "__init__.py").write_text("", encoding="utf-8")
    for package_name in checkpoint_module._REQUIRED_SNAPSHOT_PACKAGES:
        (source_path / package_name).mkdir()

    assert resolve_checkpoint_source_path(
        checkpoint_path,
        allow_current_workspace_code=False,
    ) == source_path.resolve()


def test_process_rejects_a_second_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = (tmp_path / "first" / "src").resolve()
    second = (tmp_path / "second" / "src").resolve()
    monkeypatch.setattr(checkpoint_module, "_ACTIVE_SNAPSHOT_SOURCE", first)

    with pytest.raises(RuntimeError, match="只能加载一套"):
        checkpoint_module._activate_checkpoint_source(second)


def test_loader_instantiates_wrapper_from_run_snapshot(tmp_path: Path) -> None:
    """实际 Hydra 导入必须使用运行目录的 wrapper, 不能回落到工作区. """
    run_directory = tmp_path / "run"
    checkpoint_path = run_directory / "checkpoints" / "BEST.ckpt"
    source_path = run_directory / "src_snapshot" / "src"
    checkpoint_path.parent.mkdir(parents=True)
    source_path.mkdir(parents=True)
    (source_path / "__init__.py").write_text("", encoding="utf-8")
    for package_name in checkpoint_module._REQUIRED_SNAPSHOT_PACKAGES:
        package_path = source_path / package_name
        package_path.mkdir()
        (package_path / "__init__.py").write_text("", encoding="utf-8")
    (source_path / "wrappers" / "dummy.py").write_text(
        textwrap.dedent(
            """
            import torch

            class SnapshotWrapper(torch.nn.Module):
                def __init__(self, optimizer, scheduler, compile):
                    super().__init__()
                    self.weight = torch.nn.Parameter(torch.zeros(1))
                    self.origin = "snapshot"

                def on_load_checkpoint(self, checkpoint):
                    self.runtime_value = checkpoint["runtime_value"]
            """
        ),
        encoding="utf-8",
    )
    (run_directory / "config.yaml").write_text(
        "model:\n"
        "  _target_: src.wrappers.dummy.SnapshotWrapper\n"
        "train:\n"
        "  optimizer: {}\n"
        "  scheduler: {}\n",
        encoding="utf-8",
    )
    torch.save(
        {"state_dict": {"weight": torch.tensor([5.0])}, "runtime_value": 23},
        checkpoint_path,
    )

    script = textwrap.dedent(
        f"""
        import src.inference.cli
        from src.inference.checkpoint import load_stage1_wrapper

        wrapper = load_stage1_wrapper(
            checkpoint_path={str(checkpoint_path)!r},
            resolved_config_path=None,
            map_location="cpu",
        )
        assert wrapper.origin == "snapshot"
        assert wrapper.weight.item() == 5.0
        assert wrapper.runtime_value == 23
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_cli_import_does_not_preload_snapshot_owned_packages() -> None:
    """导入推理入口时不得提前加载应由训练快照提供的 Dataset。"""
    script = textwrap.dedent(
        """
        import sys
        import src.inference.cli

        assert "src.datasets" not in sys.modules
        assert not any(name.startswith("src.datasets.") for name in sys.modules)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
