from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "训练与运行" / "runtime" / "launch_training_python.sh"


def _find_bash() -> str:
    candidates = (
        shutil.which("bash"),
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\msys64\usr\bin\bash.exe",
        r"D:\msys64\usr\bin\bash.exe",
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    pytest.skip("当前环境没有 bash, 无法验证训练 Python 启动脚本.")


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


def test_training_python_launcher_keeps_single_node_command_and_builds_torchrun(
    tmp_path: Path,
) -> None:
    """单节点保持直接 Python, 跨节点加入完整静态 rendezvous 参数."""

    bash = _find_bash()
    captured = tmp_path / "python-arguments.json"
    fake_python = tmp_path / "python.sh"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "python - \"$@\" <<'PY'\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['CAPTURE_PATH']).write_text(json.dumps(sys.argv[1:]), encoding='utf-8')\n"
        "PY\n",
        encoding="utf-8",
        newline="\n",
    )
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "TASK_PYTHON_BIN": _bash_path(bash, fake_python),
            "CAPTURE_PATH": str(captured),
        }
    )
    single = subprocess.run(
        [bash, _bash_path(bash, LAUNCHER), "src/train.py", "train.devices=2"],
        env=environment,
        capture_output=True,
        encoding="utf-8",
        timeout=20,
    )
    assert single.returncode == 0, single.stderr
    assert json.loads(captured.read_text(encoding="utf-8")) == [
        "-u",
        "src/train.py",
        "train.devices=2",
    ]

    environment.update(
        {
            "TASK_DDP_ENABLED": "1",
            "TASK_NNODES": "2",
            "TASK_GPUS": "2",
            "TASK_DDP_NODE_RANK": "1",
            "TASK_DDP_MASTER_ADDR": "node-a",
            "TASK_DDP_MASTER_PORT": "23456",
        }
    )
    multi = subprocess.run(
        [bash, _bash_path(bash, LAUNCHER), "src/train.py", "train.devices=2"],
        env=environment,
        capture_output=True,
        encoding="utf-8",
        timeout=20,
    )
    assert multi.returncode == 0, multi.stderr
    assert json.loads(captured.read_text(encoding="utf-8")) == [
        "-u",
        "-m",
        "torch.distributed.run",
        "--nnodes=2",
        "--nproc-per-node=2",
        "--rdzv-backend=static",
        "--node-rank=1",
        "--master-addr=node-a",
        "--master-port=23456",
        "src/train.py",
        "train.devices=2",
    ]


def test_four_local_processes_form_one_ddp_process_group(
    tmp_path: Path,
) -> None:
    """四个真实 Python rank 应按跨节点相同的全局 rank 契约完成 all-reduce."""

    torch = pytest.importorskip("torch")
    if not torch.distributed.is_available():
        pytest.skip("当前 PyTorch 没有 distributed 支持.")
    worker_script = tmp_path / "ddp_worker.py"
    worker_script.write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "import torch\n"
        "import torch.distributed as dist\n"
        "dist.init_process_group('gloo')\n"
        "rank = dist.get_rank()\n"
        "value = torch.tensor(float(rank + 1))\n"
        "dist.all_reduce(value)\n"
        "Path(os.environ['DDP_OUTPUT_ROOT']).joinpath(f'rank_{rank}.json').write_text(\n"
        "    json.dumps({'rank': rank, 'world_size': dist.get_world_size(), 'sum': value.item()}),\n"
        "    encoding='utf-8',\n"
        ")\n"
        "dist.destroy_process_group()\n",
        encoding="utf-8",
        newline="\n",
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        master_port = listener.getsockname()[1]
    common_environment = os.environ.copy()
    common_environment.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(master_port),
            "WORLD_SIZE": "4",
            "DDP_OUTPUT_ROOT": str(tmp_path),
            "USE_LIBUV": "0",
        }
    )
    processes = []
    for rank in range(4):
        environment = common_environment.copy()
        environment["RANK"] = str(rank)
        environment["LOCAL_RANK"] = str(rank % 2)
        processes.append(
            subprocess.Popen(
                [sys.executable, str(worker_script)],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                errors="replace",
            )
        )
    try:
        outputs = [process.communicate(timeout=60) for process in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()
    for process, output in zip(processes, outputs, strict=True):
        assert process.returncode == 0, output
    rank_records = [
        json.loads((tmp_path / f"rank_{rank}.json").read_text(encoding="utf-8"))
        for rank in range(4)
    ]
    assert [record["rank"] for record in rank_records] == [0, 1, 2, 3]
    assert {record["world_size"] for record in rank_records} == {4}
    assert {record["sum"] for record in rank_records} == {10.0}
