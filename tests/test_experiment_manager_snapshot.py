from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import src.utils.experiment_manager as experiment_manager_module
from src.utils.experiment_manager import ExperimentManager


def test_process_rank_prefers_lightning_child_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一 Slurm task 内的第二个 DDP 子进程不能再次写运行快照. """

    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("RANK", "1")
    assert ExperimentManager._resolve_process_rank() == 1
    monkeypatch.delenv("RANK")
    assert ExperimentManager._resolve_process_rank() == 1
    monkeypatch.delenv("LOCAL_RANK")
    assert ExperimentManager._resolve_process_rank() == 0


def test_archive_model_source_copies_complete_src_and_rejects_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pocket_root = tmp_path / "Pocket_Plus"
    source_root = pocket_root / "src"
    (source_root / "model").mkdir(parents=True)
    (source_root / "datasets").mkdir()
    (source_root / "utils").mkdir()
    (source_root / "__init__.py").write_text("", encoding="utf-8")
    (source_root / "model" / "network.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source_root / "datasets" / "sample.py").write_text("VALUE = 2\n", encoding="utf-8")
    fake_module_path = source_root / "utils" / "experiment_manager.py"
    fake_module_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(experiment_manager_module, "__file__", str(fake_module_path))

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text("name: test\n", encoding="utf-8")
    manager = object.__new__(ExperimentManager)
    manager.run_dir = run_dir

    manager._archive_model_source()

    copied = run_dir / "src_snapshot" / "src" / "datasets" / "sample.py"
    assert copied.read_text(encoding="utf-8") == "VALUE = 2\n"
    manifest = json.loads(
        (run_dir / "src_snapshot" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["files"]["src/datasets/sample.py"] == hashlib.sha256(
        copied.read_bytes()
    ).hexdigest()
    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        manager._archive_model_source()
