# -*- coding: utf-8 -*-
"""验证 Find_1 真实受体评估, test_1 派生, 冻结 scored-centered 和训练固定分片编排边界."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess

from omegaconf import OmegaConf
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_find1_shell_uses_official_stages_and_two_gpu_resource_profile() -> None:
    """正式入口应固定 checkpoint、双 GPU 分片、两套评分和 64 CPU 资源值."""

    shell_path = (
        PROJECT_ROOT
        / "训练与运行"
        / "sh"
        / "infer"
        / "find1_real_receptor_evaluation.sh"
    )
    shell_text = shell_path.read_text(encoding="utf-8")
    config = OmegaConf.load(PROJECT_ROOT / "configs" / "inference" / "stage1_v3.yaml")

    assert "训练与运行/sh/infer/stage1_v3.sh" in shell_text
    assert "TOP_epoch_04_score_0.6654.ckpt" in shell_text
    assert 'CUDA_VISIBLE_DEVICES=0 stage1 "$@" --shard-count 2 --shard-index 0' in shell_text
    assert 'CUDA_VISIBLE_DEVICES=1 stage1 "$@" --shard-count 2 --shard-index 1' in shell_text
    assert "--continue-on-blob-exceed" in shell_text
    assert "run_tune 1 basic" in shell_text
    assert "run_tune 2 gaussian" in shell_text
    assert "f1_blobs_basic_macro_selected" in shell_text
    assert "f2_centered_gaussian_macro_selected" in shell_text
    assert "derive_test1.py" not in shell_text
    assert int(config.window.batch_size) == 18
    assert int(config.window.workers) == 26
    assert int(config.window.prefetch_batches) == 26
    assert int(config.centered.batch_size) == 12
    assert int(config.centered.workers) == 26
    assert int(config.blob_workers) == 56
    assert int(config.calibration.workers) == 56


def test_find1_scored_centered_shell_reuses_frozen_parameters() -> None:
    """Stage2/Stage3 入口应复用冻结 F2 语义概率阈值和 Gaussian 参数, 且不调参或评估."""

    shell_path = (
        PROJECT_ROOT
        / "训练与运行"
        / "sh"
        / "infer"
        / "find1_real_receptor_scored_centered.sh"
    )
    shell_text = shell_path.read_text(encoding="utf-8")
    config = OmegaConf.load(PROJECT_ROOT / "configs" / "inference" / "stage1_v3.yaml")

    assert "训练与运行/sh/infer/stage1_v3.sh" in shell_text
    assert "TOP_epoch_04_score_0.6654.ckpt" in shell_text
    assert "stage1_preparation_box_pool_3/split/pdb_split/validation.json" in shell_text
    assert "F2_semantic.json" in shell_text
    assert "F2_gaussian.json" in shell_text
    assert "--semantic-parameters" in shell_text
    assert "--selection-parameters" in shell_text
    assert "--score-only" in shell_text
    assert "--forward-min-voxels 8" in shell_text
    assert "--continue-on-blob-exceed" in shell_text
    assert "--fit-semantic" not in shell_text
    assert " tune " not in shell_text
    assert " evaluate " not in shell_text

    calibration_score = shell_text.index("run_two_gpu_shards centered")
    validation_probability = shell_text.index(
        "run_two_gpu_shards probability", calibration_score
    )
    validation_blobs = shell_text.index("stage1 blobs", validation_probability)
    validation_centered = shell_text.index(
        "run_two_gpu_shards centered", validation_blobs
    )
    validation_score = shell_text.index(
        "run_two_gpu_shards centered", validation_centered + 1
    )
    assert (
        '--pdb-json "${CALIBRATION_JSON}"'
        in shell_text[calibration_score:validation_probability]
    )
    assert (
        '--pdb-json "${VALIDATION_JSON}"'
        in shell_text[validation_probability:validation_blobs]
    )
    assert (
        '--semantic-parameters "${F2_SEMANTIC_PARAMETERS}"'
        in shell_text[validation_blobs:validation_centered]
    )
    assert (
        "--forward-min-voxels 8"
        in shell_text[validation_centered:validation_score]
    )
    assert (
        '--selection-parameters "${F2_GAUSSIAN_PARAMETERS}"'
        in shell_text[validation_score:]
    )
    assert int(config.window.batch_size) == 18
    assert int(config.window.workers) == 26
    assert int(config.centered.batch_size) == 12
    assert int(config.centered.workers) == 26


def test_find1_train_shards_shell_preserves_fixed_sharding_and_frozen_scoring() -> None:
    """训练分片入口应把用户片号 1..50 严格映射到 CLI 下标 0..49, 并保持冻结四阶段流程."""

    shell_path = (
        PROJECT_ROOT
        / "训练与运行"
        / "sh"
        / "infer"
        / "find1_real_receptor_train_shards.sh"
    )
    shell_text = shell_path.read_text(encoding="utf-8")

    assert "stage1_preparation_box_pool_3/split/pdb_split/train.json" in shell_text
    assert "TOP_epoch_04_score_0.6654.ckpt" in shell_text
    assert "F2_semantic.json" in shell_text
    assert "F2_gaussian.json" in shell_text
    assert "--shard-count 50" in shell_text
    assert "gpu0_shard_number - 1" in shell_text
    assert "gpu1_shard_number - 1" in shell_text
    assert 'run_shard_pair "${SHARD_NUMBERS[0]}" "${SHARD_NUMBERS[2]}"' in shell_text
    assert 'run_shard_pair "${SHARD_NUMBERS[1]}" "${SHARD_NUMBERS[3]}"' in shell_text
    assert "--semantic-parameters" in shell_text
    assert "--forward-min-voxels 8" in shell_text
    assert "--continue-on-blob-exceed" in shell_text
    assert "--selection-parameters" in shell_text
    assert "--score-only" in shell_text
    assert "--fit-semantic" not in shell_text
    assert " tune " not in shell_text
    assert " evaluate " not in shell_text
    assert "--overwrite" not in shell_text

    probability = shell_text.index("CUDA_VISIBLE_DEVICES=0 stage1 probability")
    blobs = shell_text.index("stage1 blobs", probability)
    centered = shell_text.index("CUDA_VISIBLE_DEVICES=0 stage1 centered", blobs)
    score_only = shell_text.index("--score-only", centered)
    assert probability < blobs < centered < score_only


def test_find1_train_shards_shell_executes_two_ordered_gpu_sequences(
    tmp_path: Path,
) -> None:
    """伪官方入口应观测到 device 0 的 1 -> 2, device 1 的 3 -> 4, 阶段屏障和失败传播."""

    bash_path = shutil.which("bash")
    if bash_path is None and os.name == "nt":
        for candidate in (
            Path("D:/msys64/usr/bin/bash.exe"),
            Path(os.environ.get("ProgramFiles", "C:/Program Files"))
            / "Git"
            / "bin"
            / "bash.exe",
        ):
            if candidate.is_file():
                bash_path = str(candidate)
                break
    if bash_path is None:
        pytest.skip("当前测试环境没有 bash.")

    fake_project_root = tmp_path / "fake_project"
    fake_entry = (
        fake_project_root / "训练与运行" / "sh" / "infer" / "stage1_v3.sh"
    )
    fake_entry.parent.mkdir(parents=True)
    fake_entry.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s|%s\n' "${CUDA_VISIBLE_DEVICES:-none}" "$*" >> "${CALL_LOG}"
if [[ -n "${FAIL_MATCH:-}" && "$*" == *"${FAIL_MATCH}"* ]]; then
    exit 17
fi
""",
        encoding="utf-8",
        newline="\n",
    )

    shell_path = (
        PROJECT_ROOT
        / "训练与运行"
        / "sh"
        / "infer"
        / "find1_real_receptor_train_shards.sh"
    )
    call_log = tmp_path / "calls.log"
    shell_argument = str(shell_path)
    project_argument = str(fake_project_root)
    log_argument = str(call_log)
    if os.name == "nt":
        cygpath = shutil.which("cygpath")
        if cygpath is None:
            for candidate in (
                Path(bash_path).with_name("cygpath.exe"),
                Path(bash_path).parent.parent / "usr" / "bin" / "cygpath.exe",
            ):
                if candidate.is_file():
                    cygpath = str(candidate)
                    break
        if cygpath is None:
            pytest.skip("Windows bash 测试环境没有 cygpath.")
        shell_argument = subprocess.check_output(
            [cygpath, "-u", str(shell_path)], encoding="utf-8"
        ).strip()
        project_argument = subprocess.check_output(
            [cygpath, "-u", str(fake_project_root)], encoding="utf-8"
        ).strip()
        log_argument = subprocess.check_output(
            [cygpath, "-u", str(call_log)], encoding="utf-8"
        ).strip()

    environment = os.environ.copy()
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    bash_tool_directories = [Path(bash_path).parent]
    sibling_usr_bin = Path(bash_path).parent.parent / "usr" / "bin"
    if sibling_usr_bin.is_dir() and sibling_usr_bin not in bash_tool_directories:
        bash_tool_directories.append(sibling_usr_bin)
    environment["PATH"] = os.pathsep.join(
        [str(path) for path in bash_tool_directories] + [environment["PATH"]]
    )
    environment.update(
        TASK_PROJECT_ROOT=project_argument,
        CALL_LOG=log_argument,
    )
    completed = subprocess.run(
        [bash_path, shell_argument, "1", "2", "3", "4"],
        check=False,
        capture_output=True,
        encoding="utf-8",
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr

    # calls: 按阶段屏障落盘的 16 次官方入口调用; 每项依次为 allocation 内 CUDA device 字符串, Stage1 子命令, 0-based CLI 分片下标和是否启用 score-only. device="none" 表示 CPU blobs, bool 为 True 表示 score-only.
    calls: list[tuple[str, str, str, bool]] = []
    for line in call_log.read_text(encoding="utf-8").splitlines():
        device, arguments_text = line.split("|", maxsplit=1)
        arguments = shlex.split(arguments_text)
        command = arguments[0]
        shard_index = arguments[arguments.index("--shard-index") + 1]
        assert arguments[arguments.index("--shard-count") + 1] == "50"
        assert arguments[arguments.index("--split") + 1] == "train"
        assert arguments[arguments.index("--producer") + 1] == "Find_1"
        assert arguments[arguments.index("--pdb-json") + 1].endswith(
            "/stage1_preparation_box_pool_3/split/pdb_split/train.json"
        )
        if command == "probability":
            assert "TOP_epoch_04_score_0.6654.ckpt" in arguments[
                arguments.index("--checkpoint") + 1
            ]
            assert arguments[arguments.index("--model-code-source") + 1] == (
                "training_snapshot"
            )
        elif command == "blobs":
            assert arguments[arguments.index("--alpha") + 1] == "2"
            assert arguments[arguments.index("--semantic-parameters") + 1].endswith(
                "/Find_1/tuning/F2_semantic.json"
            )
            assert "--fit-semantic" not in arguments
        elif "--score-only" in arguments:
            assert arguments[arguments.index("--selection-parameters") + 1].endswith(
                "/Find_1/tuning/F2_gaussian.json"
            )
            assert "--checkpoint" not in arguments
        else:
            assert arguments[arguments.index("--forward-min-voxels") + 1] == "8"
            assert "--continue-on-blob-exceed" in arguments
            assert "TOP_epoch_04_score_0.6654.ckpt" in arguments[
                arguments.index("--checkpoint") + 1
            ]
        assert "--overwrite" not in arguments
        calls.append((device, command, shard_index, "--score-only" in arguments))

    assert len(calls) == 16
    assert set(calls[0:2]) == {
        ("0", "probability", "0", False),
        ("1", "probability", "2", False),
    }
    assert calls[2:4] == [
        ("none", "blobs", "0", False),
        ("none", "blobs", "2", False),
    ]
    assert set(calls[4:6]) == {
        ("0", "centered", "0", False),
        ("1", "centered", "2", False),
    }
    assert set(calls[6:8]) == {
        ("0", "centered", "0", True),
        ("1", "centered", "2", True),
    }
    assert set(calls[8:10]) == {
        ("0", "probability", "1", False),
        ("1", "probability", "3", False),
    }
    assert calls[10:12] == [
        ("none", "blobs", "1", False),
        ("none", "blobs", "3", False),
    ]
    assert set(calls[12:14]) == {
        ("0", "centered", "1", False),
        ("1", "centered", "3", False),
    }
    assert set(calls[14:16]) == {
        ("0", "centered", "1", True),
        ("1", "centered", "3", True),
    }

    call_log.unlink()
    environment["FAIL_MATCH"] = "--shard-index 2"
    failed = subprocess.run(
        [bash_path, shell_argument, "1", "2", "3", "4"],
        check=False,
        capture_output=True,
        encoding="utf-8",
        env=environment,
    )
    assert failed.returncode != 0
    failed_calls = call_log.read_text(encoding="utf-8").splitlines()
    assert len(failed_calls) == 2
    assert all("|probability " in line for line in failed_calls)
