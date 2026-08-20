# -*- coding: utf-8 -*-
"""运行真实 Stage1 命令并采样 GPU 利用率, 吞吐量和队列等待时间."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

import numpy as np


# ================================================================================================


def main() -> None:
    """执行真实推理命令并发布 GPU/流水线性能事实.

    ``--`` 后的参数原样交给子进程. 运行期间只保留 ``--gpu-index`` 指定设备.
    `samples.csv` 字段:
        - unix_time: float, Unix 秒时间戳.
        - gpu_index: int, 目标 GPU 索引.
        - utilization_percent: float, GPU 利用率百分比.
        - memory_used_mib: float, 已用显存 MiB.

    `summary.json` 字段:
        - command: list[str], 实际子进程命令.
        - run_mode: str, `serial` 或 `pipeline`.
        - gpu_index: int, 目标 GPU 索引.
        - return_code: int, 子进程退出码.
        - wall_seconds: float, 子进程墙钟秒数.
        - sample_interval_seconds: float, GPU 采样周期秒数.
        - gpu_sample_count: int, 有效目标 GPU 样本数.
        - gpu_active_ratio: float, 利用率大于零的样本比例.
        - gpu_utilization_mean_percent: float, 利用率均值百分比.
        - gpu_utilization_p50_percent: float, 利用率 P50 百分比.
        - gpu_utilization_p95_percent: float, 利用率 P95 百分比.
        - window_count: int, 完整图窗口总数.
        - window_per_second: float, 完整图计时口径的窗口吞吐量.
        - full_map_materialize_wait_seconds: float, 完整图物化等待累计秒数.
        - full_map_fusion_wait_seconds: float, 完整图融合等待累计秒数.
        - centered_entry_count: int, centered 条目总数.
        - centered_entry_per_second: float, centered 计时口径的条目吞吐量.
        - centered_materialize_wait_seconds: float, centered 物化等待累计秒数.
        - centered_cpu_arrange_wait_seconds: float, centered CPU 整理等待累计秒数.
        - comparison.summary_path: str, 对照 summary 路径; 未传对照时无此对象.
        - comparison.run_mode: str, 对照运行模式.
        - comparison.wall_speed_ratio: float | null, 对照墙钟除以当前墙钟.
        - comparison.window_throughput_ratio: float | null, 当前窗口吞吐除以对照.
        - comparison.centered_throughput_ratio: float | null, 当前 centered 吞吐除以对照.

    ``nvidia-smi`` 采样失败时 CSV 只保留表头; 比值分母为零时写 ``null``.
    本工具不改写科学 NPZ 或推理配置.
    """

    parser = argparse.ArgumentParser(description="Stage1 V3 GPU utilization benchmark")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-interval-seconds", type=float, required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--producer", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--run-mode", choices=("serial", "pipeline"), required=True)
    parser.add_argument("--comparison-summary")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = arguments.command[1:] if arguments.command[:1] == ["--"] else arguments.command
    if not command:
        parser.error("必须在 -- 后提供真实 Stage1 命令.")

    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()
    process = subprocess.Popen(command)
    samples: list[dict[str, object]] = []
    while process.poll() is None:
        sampled_at = time.time()
        query = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        if query.returncode == 0:
            for line in query.stdout.splitlines():
                gpu_index, utilization, memory_used = [part.strip() for part in line.split(",")]
                if int(gpu_index) != int(arguments.gpu_index):
                    continue
                samples.append(
                    {
                        "unix_time": sampled_at,
                        "gpu_index": int(gpu_index),
                        "utilization_percent": float(utilization),
                        "memory_used_mib": float(memory_used),
                    }
                )
        time.sleep(float(arguments.sample_interval_seconds))
    return_code = int(process.wait())
    wall_seconds = time.perf_counter() - started_at

    sample_path = output_dir / "samples.csv"
    with sample_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "unix_time",
                "gpu_index",
                "utilization_percent",
                "memory_used_mib",
            ),
        )
        writer.writeheader()
        writer.writerows(samples)

    utilization = np.asarray(
        [row["utilization_percent"] for row in samples],
        dtype=np.float64,
    )
    producer_root = (
        Path(arguments.artifact_root) / arguments.producer / arguments.split
    )
    window_count = 0
    full_map_seconds = 0.0
    materialize_wait_seconds = 0.0
    fusion_wait_seconds = 0.0
    centered_entry_count = 0
    centered_seconds = 0.0
    centered_materialize_wait_seconds = 0.0
    centered_cpu_wait_seconds = 0.0
    for geometry_path in producer_root.glob("*/probability/geometry.json"):
        geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
        window_count += int(geometry["window_count"])
    for performance_path in producer_root.glob("*/status/probability/performance.json"):
        payload = json.loads(performance_path.read_text(encoding="utf-8"))
        full_map_seconds += float(payload["wall_seconds"])
        materialize_wait_seconds += float(payload["materialize_wait_seconds"])
        fusion_wait_seconds += float(payload["fusion_wait_seconds"])
    for performance_path in producer_root.glob("*/status/*/performance.json"):
        if performance_path.parent.name == "probability":
            continue
        payload = json.loads(performance_path.read_text(encoding="utf-8"))
        centered_entry_count += int(payload["entry_count"])
        centered_seconds += float(payload["wall_seconds"])
        centered_materialize_wait_seconds += float(payload["materialize_wait_seconds"])
        centered_cpu_wait_seconds += float(payload["cpu_arrange_wait_seconds"])

    summary = {
        "command": command,
        "run_mode": str(arguments.run_mode),
        "gpu_index": int(arguments.gpu_index),
        "return_code": return_code,
        "wall_seconds": wall_seconds,
        "sample_interval_seconds": float(arguments.sample_interval_seconds),
        "gpu_sample_count": int(utilization.size),
        "gpu_active_ratio": float(np.mean(utilization > 0.0)) if utilization.size else 0.0,
        "gpu_utilization_mean_percent": float(np.mean(utilization)) if utilization.size else 0.0,
        "gpu_utilization_p50_percent": float(np.percentile(utilization, 50)) if utilization.size else 0.0,
        "gpu_utilization_p95_percent": float(np.percentile(utilization, 95)) if utilization.size else 0.0,
        "window_count": window_count,
        "window_per_second": float(window_count) / full_map_seconds if full_map_seconds else 0.0,
        "full_map_materialize_wait_seconds": materialize_wait_seconds,
        "full_map_fusion_wait_seconds": fusion_wait_seconds,
        "centered_entry_count": centered_entry_count,
        "centered_entry_per_second": (
            float(centered_entry_count) / centered_seconds if centered_seconds else 0.0
        ),
        "centered_materialize_wait_seconds": centered_materialize_wait_seconds,
        "centered_cpu_arrange_wait_seconds": centered_cpu_wait_seconds,
    }
    if arguments.comparison_summary:
        comparison = json.loads(
            Path(arguments.comparison_summary).read_text(encoding="utf-8")
        )
        summary["comparison"] = {
            "summary_path": str(Path(arguments.comparison_summary)),
            "run_mode": str(comparison["run_mode"]),
            "wall_speed_ratio": (
                float(comparison["wall_seconds"]) / wall_seconds
                if wall_seconds > 0.0
                else None
            ),
            "window_throughput_ratio": (
                float(summary["window_per_second"])
                / float(comparison["window_per_second"])
                if float(comparison["window_per_second"]) > 0.0
                else None
            ),
            "centered_throughput_ratio": (
                float(summary["centered_entry_per_second"])
                / float(comparison["centered_entry_per_second"])
                if float(comparison["centered_entry_per_second"]) > 0.0
                else None
            ),
        }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
