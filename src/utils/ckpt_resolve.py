from __future__ import annotations

import os
from pathlib import Path


def resolve_previous_best_checkpoint_by_job(feedback_root: Path, current_run_dir: Path) -> Path:
    """
    为 `init_from="***"` 解析同一 Slurm job 内上一阶段的 BEST.ckpt。

    输入参数:
        - feedback_root: Path, feedback_plus 根目录; 其下应包含 `logs/*/<tag>____job<id>` run 目录
        - current_run_dir: Path, 当前 run 目录; 必须从候选集中排除

    输出:
        - ckpt_path: Path, 最近一个同 job 历史 run 的 `checkpoints/BEST.ckpt`
    """
    job_id = os.environ["SLURM_JOB_ID"].strip()
    run_suffix = f"____job{job_id}"
    current_resolved = current_run_dir.resolve()
    logs_root = Path(feedback_root) / "logs"
    candidates: list[Path] = []
    for run_dir in logs_root.glob(f"*/*{run_suffix}"):
        if run_dir.resolve() == current_resolved:
            continue
        ckpt_path = run_dir / "checkpoints" / "BEST.ckpt"
        if ckpt_path.is_file():
            candidates.append(ckpt_path)
    if not candidates:
        raise FileNotFoundError(
            "init_from='***' 未找到同一 SLURM_JOB_ID 的上一阶段 BEST.ckpt: "
            f"SLURM_JOB_ID={job_id}, logs_root={logs_root}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)
