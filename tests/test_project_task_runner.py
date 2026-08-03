from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_DIRECTORY = Path(
    os.environ.get("PROJECT_TASK_RUNNER_DIRECTORY", PROJECT_ROOT / "训练与运行")
).resolve()


def _find_bash(required_command: str | None = None) -> str:
    candidates = [
        shutil.which("bash"),
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\msys64\usr\bin\bash.exe",
        r"D:\msys64\usr\bin\bash.exe",
    ]
    for candidate in candidates:
        if not candidate or not Path(candidate).is_file():
            continue
        if required_command is not None:
            command_check = subprocess.run(
                [str(candidate), "-lc", f"command -v {required_command}"],
                capture_output=True,
                encoding="utf-8",
            )
            if command_check.returncode != 0:
                continue
        return str(candidate)
    requirement = f"并提供 {required_command}" if required_command else ""
    pytest.skip(f"当前环境没有可用于验证提交脚本{requirement}的 bash")


def _bash_path(bash: str, path: Path) -> str:
    if os.name != "nt":
        return str(path)
    completed = subprocess.run(
        [bash, "-lc", 'cygpath -u "$1"', "_", str(path)],
        check=True,
        capture_output=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _run_submitter(
    bash: str,
    submitter: Path,
    capture_path: Path,
    arguments: list[str],
    *,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    fake_sbatch = capture_path.with_suffix(".sh")
    fake_sbatch.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\nprintf \'%s\\n\' "$@" >"${CAPTURE_PATH}"\n',
        encoding="utf-8",
        newline="\n",
    )
    fake_sbatch.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join(
        (str(Path(bash).parent), environment.get("PATH", ""))
    )
    environment["SBATCH_BIN"] = _bash_path(bash, fake_sbatch)
    environment["CAPTURE_PATH"] = _bash_path(bash, capture_path)
    return subprocess.run(
        [bash, _bash_path(bash, submitter), *arguments],
        cwd=cwd,
        env=environment,
        capture_output=True,
        encoding="utf-8",
    )


def _argument_value(arguments: list[str], name: str) -> str:
    index = arguments.index(name)
    return arguments[index + 1]


def test_copied_runner_uses_parent_directory_as_default_task_root(
    tmp_path: Path,
) -> None:
    """复制到另一项目的“训练与运行”应独立完成正式提交参数组装。"""

    bash = _find_bash()
    copied_project = tmp_path / "AdaLigand"
    copied_runner = copied_project / "训练与运行"
    shutil.copytree(RUNNER_DIRECTORY, copied_runner)
    task_script = copied_runner / "sh" / "smoke.sh"
    task_script.parent.mkdir(exist_ok=True)
    task_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8", newline="\n")
    capture_path = tmp_path / "default-root.arguments"

    completed = _run_submitter(
        bash,
        copied_runner / "submit_task.sh",
        capture_path,
        ["--sh", "smoke.sh", "--resource", "cpu", "--cpus", "1"],
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    submitted = capture_path.read_text(encoding="utf-8").splitlines()
    assert Path(_argument_value(submitted, "--task-root")).name == "AdaLigand"
    assert _argument_value(submitted, "--task") == "训练与运行/sh/smoke.sh"
    assert "--task-mode" not in submitted


def test_absolute_and_relative_task_paths_produce_same_frozen_task(
    tmp_path: Path,
) -> None:
    """同一项目脚本的绝对路径与项目相对路径应得到相同任务契约。"""

    bash = _find_bash()
    copied_project = tmp_path / "AdaLigand"
    copied_runner = copied_project / "训练与运行"
    shutil.copytree(RUNNER_DIRECTORY, copied_runner)
    submitter = copied_runner / "submit_task.sh"
    task_script = copied_runner / "sh" / "smoke.sh"
    task_script.parent.mkdir(exist_ok=True)
    task_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8", newline="\n")
    captured_arguments: list[list[str]] = []
    for index, task_argument in enumerate(
        ("训练与运行/sh/smoke.sh", _bash_path(bash, task_script))
    ):
        capture_path = tmp_path / f"path-{index}.arguments"
        completed = _run_submitter(
            bash,
            submitter,
            capture_path,
            ["--sh", task_argument, "--resource", "cpu", "--cpus", "1"],
            cwd=tmp_path,
        )
        assert completed.returncode == 0, completed.stderr
        captured_arguments.append(
            capture_path.read_text(encoding="utf-8").splitlines()
        )

    assert _argument_value(captured_arguments[0], "--task") == _argument_value(
        captured_arguments[1], "--task"
    )
    assert _argument_value(captured_arguments[0], "--task-root") == _argument_value(
        captured_arguments[1], "--task-root"
    )


def test_submitter_forwards_pre_hold_and_after_hold_without_old_hold(
    tmp_path: Path,
) -> None:
    """提交器应转发两个独立保留开关，并拒绝已经删除的 hold 参数。"""

    bash = _find_bash()
    copied_project = tmp_path / "AdaLigand"
    copied_runner = copied_project / "训练与运行"
    shutil.copytree(RUNNER_DIRECTORY, copied_runner)
    task_script = copied_runner / "sh" / "smoke.sh"
    task_script.parent.mkdir(exist_ok=True)
    task_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8", newline="\n")

    capture_path = tmp_path / "hold-options.arguments"
    completed = _run_submitter(
        bash,
        copied_runner / "submit_task.sh",
        capture_path,
        [
            "--sh",
            "smoke.sh",
            "--resource",
            "cpu",
            "--cpus",
            "1",
            "--pre_hold",
            "--after_hold",
        ],
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    submitted = capture_path.read_text(encoding="utf-8").splitlines()
    assert _argument_value(submitted, "--pre_hold") == "1"
    assert _argument_value(submitted, "--after_hold") == "1"

    rejected = _run_submitter(
        bash,
        copied_runner / "submit_task.sh",
        tmp_path / "old-hold.arguments",
        ["--sh", "smoke.sh", "--resource", "cpu", "--cpus", "1", "--hold"],
        cwd=tmp_path,
    )
    assert rejected.returncode == 2
    assert "未知参数：--hold" in rejected.stderr


def test_task_root_selects_another_project_and_rejects_outside_script(
    tmp_path: Path,
) -> None:
    """完整模式只冻结显式任务根目录内的脚本，不保留外部脚本分支。"""

    bash = _find_bash()
    copied_project = tmp_path / "AdaLigand"
    shutil.copytree(RUNNER_DIRECTORY, copied_project / "训练与运行")
    task_script = copied_project / "训练与运行" / "sh" / "smoke.sh"
    task_script.parent.mkdir(exist_ok=True)
    task_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8", newline="\n")
    submitter = RUNNER_DIRECTORY / "submit_task.sh"
    capture_path = tmp_path / "selected-root.arguments"
    completed = _run_submitter(
        bash,
        submitter,
        capture_path,
        [
            "--task-root",
            _bash_path(bash, copied_project),
            "--sh",
            _bash_path(bash, task_script),
            "--resource",
            "cpu",
            "--cpus",
            "1",
        ],
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    submitted = capture_path.read_text(encoding="utf-8").splitlines()
    assert Path(_argument_value(submitted, "--task-root")).name == "AdaLigand"

    outside_script = tmp_path / "outside.sh"
    outside_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8", newline="\n")
    rejected = _run_submitter(
        bash,
        submitter,
        tmp_path / "rejected.arguments",
        [
            "--task-root",
            _bash_path(bash, copied_project),
            "--sh",
            _bash_path(bash, outside_script),
            "--resource",
            "cpu",
            "--cpus",
            "1",
        ],
        cwd=tmp_path,
    )
    assert rejected.returncode == 2
    assert "请用 --task-root" in rejected.stderr


def test_full_runtime_freezes_copied_project_and_records_generic_stamp(
    tmp_path: Path,
) -> None:
    """完整模式应从复制后的目录建立 release、launch 和统一运行标识。"""

    bash = _find_bash(required_command="rsync")

    copied_project = tmp_path / "AdaLigand"
    copied_runner = copied_project / "训练与运行"
    shutil.copytree(RUNNER_DIRECTORY, copied_runner)
    # 运行组件由 bash 显式解释，不依赖跨 Windows/Linux 复制后保留可执行位。
    (copied_runner / "runtime" / "create_release.sh").chmod(0o644)
    (copied_runner / "runtime" / "create_launch.sh").chmod(0o644)
    task_script = copied_runner / "sh" / "runtime_contract.sh"
    task_script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"${TASK_RUN_STAMP}\" >\"${TEST_OUTPUT}\"\n",
        encoding="utf-8",
        newline="\n",
    )
    feedback_root = tmp_path / "feedback"
    output_path = tmp_path / "task-run-stamp.txt"
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join(
        (str(Path(bash).parent), environment.get("PATH", ""))
    )
    environment.update(
        {
            "SLURM_JOB_ID": "900001",
            "TASK_LOCK_POLL_SECONDS": "0.01",
            "TASK_KILL_POLL_SECONDS": "0.01",
            "TEST_OUTPUT": _bash_path(bash, output_path),
        }
    )

    completed = subprocess.run(
        [
            bash,
            _bash_path(bash, copied_runner / "sbatch" / "task.sbatch"),
            "--task-root",
            _bash_path(bash, copied_project),
            "--feedback-root",
            _bash_path(bash, feedback_root),
            "--task",
            "训练与运行/sh/runtime_contract.sh",
            "--simple",
            "0",
            "--pre_hold",
            "0",
            "--after_hold",
            "0",
            "--resource",
            "cpu",
            "--gpus",
            "0",
            "--nodes",
            "1",
            "--cpus",
            "1",
            "--array",
            "",
        ],
        env=environment,
        capture_output=True,
        encoding="utf-8",
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    task_run_stamp = output_path.read_text(encoding="utf-8").strip()
    launch_json_path = next((feedback_root / "launches" / "900001").glob("*/launch.json"))
    launch = json.loads(launch_json_path.read_text(encoding="utf-8"))
    assert launch["task_run_stamp"] == task_run_stamp
    assert "pocket_run_stamp" not in launch
    assert Path(launch["release_project_root"]).name == "AdaLigand"
    assert launch["task_script"] == "训练与运行/sh/runtime_contract.sh"


def test_simple_runtime_uses_same_task_root_without_release(
    tmp_path: Path,
) -> None:
    """simple 模式应从任务根目录执行脚本，同时不建立正式留证目录。"""

    bash = _find_bash(required_command="setsid")
    copied_project = tmp_path / "AdaLigand"
    copied_runner = copied_project / "训练与运行"
    shutil.copytree(RUNNER_DIRECTORY, copied_runner)
    task_script = copied_runner / "sh" / "simple_contract.sh"
    task_script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "pwd -P >\"${TEST_OUTPUT}\"\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary_home = tmp_path / "home"
    temporary_home.mkdir()
    output_path = tmp_path / "simple-working-directory.txt"
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join(
        (str(Path(bash).parent), environment.get("PATH", ""))
    )
    environment.update(
        {
            "HOME": _bash_path(bash, temporary_home),
            "SLURM_JOB_ID": "900002",
            "TASK_LOCK_POLL_SECONDS": "0.01",
            "TASK_KILL_POLL_SECONDS": "0.01",
            "TEST_OUTPUT": _bash_path(bash, output_path),
        }
    )

    completed = subprocess.run(
        [
            bash,
            _bash_path(bash, copied_runner / "sbatch" / "task.sbatch"),
            "--task-root",
            _bash_path(bash, copied_project),
            "--feedback-root",
            "",
            "--task",
            "训练与运行/sh/simple_contract.sh",
            "--simple",
            "1",
            "--pre_hold",
            "0",
            "--after_hold",
            "0",
            "--resource",
            "cpu",
            "--gpus",
            "0",
            "--nodes",
            "1",
            "--cpus",
            "1",
            "--array",
            "",
        ],
        env=environment,
        capture_output=True,
        encoding="utf-8",
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    working_directory = output_path.read_text(encoding="utf-8").strip()
    assert working_directory.endswith("/AdaLigand")
    assert not (temporary_home / "Feedback").exists()
    assert not (temporary_home / "SIMPLE_RUN" / "after_lock_900002").exists()


def _wait_for_path(path: Path, process: subprocess.Popen[str]) -> None:
    """等待控制文件出现，同时保证模拟 allocation 没有提前退出。"""

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(f"allocation 提前退出: {stdout=} {stderr=}")
        time.sleep(0.02)
    raise AssertionError(f"等待控制文件超时: {path}")


def test_pre_hold_delays_first_run_without_enabling_after_hold(tmp_path: Path) -> None:
    """pre_hold 应只延迟第一次执行，任务完成后仍按默认策略自动释放。"""

    bash = _find_bash(required_command="setsid")
    copied_project = tmp_path / "AdaLigand"
    copied_runner = copied_project / "训练与运行"
    shutil.copytree(RUNNER_DIRECTORY, copied_runner)
    task_script = copied_runner / "sh" / "pre_hold_contract.sh"
    task_script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nprintf done >\"${TEST_OUTPUT}\"\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary_home = tmp_path / "home"
    temporary_home.mkdir()
    output_path = tmp_path / "pre-hold-output.txt"
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join((str(Path(bash).parent), environment.get("PATH", "")))
    environment.update(
        {
            "HOME": _bash_path(bash, temporary_home),
            "SLURM_JOB_ID": "900003",
            "TASK_LOCK_POLL_SECONDS": "0.01",
            "TASK_KILL_POLL_SECONDS": "0.01",
            "TEST_OUTPUT": _bash_path(bash, output_path),
        }
    )
    process = subprocess.Popen(
        [
            bash,
            _bash_path(bash, copied_runner / "sbatch" / "task.sbatch"),
            "--task-root", _bash_path(bash, copied_project),
            "--feedback-root", "",
            "--task", "训练与运行/sh/pre_hold_contract.sh",
            "--simple", "1",
            "--pre_hold", "1",
            "--after_hold", "0",
            "--resource", "cpu",
            "--gpus", "0",
            "--nodes", "1",
            "--cpus", "1",
            "--array", "",
        ],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    pre_lock = temporary_home / "SIMPLE_RUN" / "pre_lock_900003"
    try:
        _wait_for_path(pre_lock, process)
        assert not output_path.exists()
        pre_lock.unlink()
        stdout, stderr = process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    assert process.returncode == 0, (stdout, stderr)
    assert output_path.read_text(encoding="utf-8") == "done"
    assert not (temporary_home / "SIMPLE_RUN" / "try_lock_900003").exists()


def test_after_hold_retains_allocation_after_run(tmp_path: Path) -> None:
    """after_hold 应在任务结束后建立 try_lock，直到人工删除 after_lock。"""

    bash = _find_bash(required_command="setsid")
    copied_project = tmp_path / "AdaLigand"
    copied_runner = copied_project / "训练与运行"
    shutil.copytree(RUNNER_DIRECTORY, copied_runner)
    task_script = copied_runner / "sh" / "after_hold_contract.sh"
    task_script.write_text("#!/usr/bin/env bash\nset -euo pipefail\nexit 0\n", encoding="utf-8", newline="\n")
    temporary_home = tmp_path / "home"
    temporary_home.mkdir()
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join((str(Path(bash).parent), environment.get("PATH", "")))
    environment.update(
        {
            "HOME": _bash_path(bash, temporary_home),
            "SLURM_JOB_ID": "900004",
            "TASK_LOCK_POLL_SECONDS": "0.01",
            "TASK_KILL_POLL_SECONDS": "0.01",
        }
    )
    process = subprocess.Popen(
        [
            bash,
            _bash_path(bash, copied_runner / "sbatch" / "task.sbatch"),
            "--task-root", _bash_path(bash, copied_project),
            "--feedback-root", "",
            "--task", "训练与运行/sh/after_hold_contract.sh",
            "--simple", "1",
            "--pre_hold", "0",
            "--after_hold", "1",
            "--resource", "cpu",
            "--gpus", "0",
            "--nodes", "1",
            "--cpus", "1",
            "--array", "",
        ],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    control_root = temporary_home / "SIMPLE_RUN"
    try_lock = control_root / "try_lock_900004"
    after_lock = control_root / "after_lock_900004"
    try:
        _wait_for_path(try_lock, process)
        assert process.poll() is None
        assert after_lock.is_file()
        after_lock.unlink()
        stdout, stderr = process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    assert process.returncode == 0, (stdout, stderr)
    assert not try_lock.exists()


def test_default_release_preserves_failed_task_exit_code(tmp_path: Path) -> None:
    """未启用 after_hold 时，失败任务应立即释放并成为 Slurm Job 的退出码。"""

    bash = _find_bash(required_command="setsid")
    copied_project = tmp_path / "AdaLigand"
    copied_runner = copied_project / "训练与运行"
    shutil.copytree(RUNNER_DIRECTORY, copied_runner)
    task_script = copied_runner / "sh" / "failure_contract.sh"
    task_script.write_text("#!/usr/bin/env bash\nexit 7\n", encoding="utf-8", newline="\n")
    temporary_home = tmp_path / "home"
    temporary_home.mkdir()
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join((str(Path(bash).parent), environment.get("PATH", "")))
    environment.update(
        {
            "HOME": _bash_path(bash, temporary_home),
            "SLURM_JOB_ID": "900005",
            "TASK_LOCK_POLL_SECONDS": "0.01",
            "TASK_KILL_POLL_SECONDS": "0.01",
        }
    )
    completed = subprocess.run(
        [
            bash,
            _bash_path(bash, copied_runner / "sbatch" / "task.sbatch"),
            "--task-root", _bash_path(bash, copied_project),
            "--feedback-root", "",
            "--task", "训练与运行/sh/failure_contract.sh",
            "--simple", "1",
            "--pre_hold", "0",
            "--after_hold", "0",
            "--resource", "cpu",
            "--gpus", "0",
            "--nodes", "1",
            "--cpus", "1",
            "--array", "",
        ],
        env=environment,
        capture_output=True,
        encoding="utf-8",
        timeout=10,
    )
    assert completed.returncode == 7, completed.stderr
    assert not (temporary_home / "SIMPLE_RUN" / "try_lock_900005").exists()
