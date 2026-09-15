# -*- coding: utf-8 -*-
"""验证 Find_1 CryoAtom2 受体正式入口的数据根、阶段顺序和资源契约."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess

from omegaconf import OmegaConf
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_find1_cryoatom2_shell_uses_frozen_scientific_contract() -> None:
    """正式入口应只切换受体数据根, 并保留两套冻结评分流程和 H100 资源参数."""
    shell_path = PROJECT_ROOT / "训练与运行/sh/infer/find1_cryoatom2_receptor_evaluation.sh"
    shell_text = shell_path.read_text(encoding="utf-8")
    config = OmegaConf.load(PROJECT_ROOT / "configs/inference/stage1_v3.yaml")

    assert "训练与运行/sh/infer/stage1_v3.sh" in shell_text
    assert "TOP_epoch_04_score_0.6654.ckpt" in shell_text
    assert "/cryoatom2/calibration/Ori_Data" in shell_text
    assert "/cryoatom2/test_0_chain06/Ori_Data" in shell_text
    assert "/Find_1/CryoAtom2受体/artifacts" in shell_text
    assert "ADALIGAND_DATA_ROOT" in shell_text
    assert "run_tune 1 basic" in shell_text
    assert "run_tune 2 gaussian" in shell_text
    assert "--objective-beta 1" in shell_text
    assert "--forward-min-voxels 8" in shell_text
    assert "--continue-on-blob-exceed" in shell_text
    assert "--score-only" in shell_text
    assert "validation" not in shell_text
    assert "train.json" not in shell_text
    assert int(config.window.batch_size) == 18
    assert int(config.window.workers) == 26
    assert int(config.centered.batch_size) == 12
    assert int(config.centered.workers) == 26
    assert int(config.calibration.workers) == 56


def test_find1_cryoatom2_shell_routes_each_gpu_stage_to_matching_data_root(tmp_path: Path) -> None:
    """伪官方入口应观测 calibration 与 test_0 各自的数据根、双 GPU 分片和单次 calibration score-only."""
    bash_path = shutil.which("bash")
    if bash_path is None and os.name == "nt":
        for candidate in (Path("D:/msys64/usr/bin/bash.exe"), Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"):
            if candidate.is_file():
                bash_path = str(candidate)
                break
    if bash_path is None:
        pytest.skip("当前测试环境没有 bash.")

    fake_project_root = tmp_path / "fake_project"
    fake_entry = fake_project_root / "训练与运行/sh/infer/stage1_v3.sh"
    fake_entry.parent.mkdir(parents=True)
    fake_entry.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nprintf '%s|%s|%s\\n' \"${ADALIGAND_DATA_ROOT:-none}\" \"${CUDA_VISIBLE_DEVICES:-none}\" \"$*\" >> \"${CALL_LOG}\"\n",
        encoding="utf-8",
        newline="\n",
    )

    shell_path = PROJECT_ROOT / "训练与运行/sh/infer/find1_cryoatom2_receptor_evaluation.sh"
    call_log = tmp_path / "calls.log"
    shell_argument = str(shell_path)
    project_argument = str(fake_project_root)
    log_argument = str(call_log)
    if os.name == "nt":
        cygpath = shutil.which("cygpath") or str(Path(bash_path).with_name("cygpath.exe"))
        if not Path(cygpath).is_file():
            pytest.skip("Windows bash 测试环境没有 cygpath.")
        shell_argument = subprocess.check_output([cygpath, "-u", str(shell_path)], encoding="utf-8").strip()
        project_argument = subprocess.check_output([cygpath, "-u", str(fake_project_root)], encoding="utf-8").strip()
        log_argument = subprocess.check_output([cygpath, "-u", str(call_log)], encoding="utf-8").strip()

    environment = os.environ.copy()
    bash_tool_directories = [Path(bash_path).parent]
    sibling_usr_bin = Path(bash_path).parent.parent / "usr/bin"
    if sibling_usr_bin.is_dir():
        bash_tool_directories.append(sibling_usr_bin)
    environment["PATH"] = os.pathsep.join([*(str(path) for path in bash_tool_directories), environment["PATH"]])
    environment.update(TASK_PROJECT_ROOT=project_argument, CALL_LOG=log_argument)
    completed = subprocess.run([bash_path, shell_argument], check=False, capture_output=True, encoding="utf-8", env=environment)
    assert completed.returncode == 0, completed.stderr

    # list[tuple], 18 次官方阶段调用; 三项依次为当前受体数据根、GPU 编号和 Stage1 参数列表.
    calls = []
    for line in call_log.read_text(encoding="utf-8").splitlines():
        data_root, device, arguments_text = line.split("|", maxsplit=2)
        calls.append((data_root, device, shlex.split(arguments_text)))
    assert len(calls) == 18
    gpu_calls = [call for call in calls if call[1] in {"0", "1"}]
    assert len(gpu_calls) == 10
    for data_root, _, arguments in gpu_calls:
        split = arguments[arguments.index("--split") + 1]
        expected_suffix = "/cryoatom2/calibration/Ori_Data" if split == "calibration" else "/cryoatom2/test_0_chain06/Ori_Data"
        assert data_root.endswith(expected_suffix)
    score_only_calls = [call for call in calls if "--score-only" in call[2]]
    assert len(score_only_calls) == 2
    assert {call[1] for call in score_only_calls} == {"0", "1"}
    assert all(call[2][call[2].index("--split") + 1] == "calibration" for call in score_only_calls)
