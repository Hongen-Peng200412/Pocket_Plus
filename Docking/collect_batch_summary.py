from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from docking_pipeline.config import ServerPaths
from docking_pipeline.io_utils import write_json


def main() -> None:
    """
    汇总 array 分片产生的样本级 docking summary。

    输入参数:
        - run_id: str, pipeline run ID
        - mode: str, `oracle` 或 `realistic`, 决定汇总字段

    输出:
        - None; 写入 `/home/penghongen/分子对接尝试/pipeline_runs/{run_id}/tables/batch_summary.json`
    """
    parser = argparse.ArgumentParser(description="汇总 array docking 分片结果")
    parser.add_argument("--run-id", required=True, help="要汇总的 pipeline run ID")
    parser.add_argument("--mode", choices=("oracle", "realistic"), required=True, help="汇总模式")
    args = parser.parse_args()

    paths = ServerPaths.default()
    run_root = paths.allowed_root / "pipeline_runs" / args.run_id
    summaries = read_sample_summaries(run_root)
    shard_summaries = read_shard_summaries(run_root)
    batch_summary = build_batch_summary(args.run_id, args.mode, summaries, shard_summaries)
    tables_dir = run_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    write_json(tables_dir / "batch_summary.json", batch_summary)
    print(json.dumps(batch_summary, ensure_ascii=False, indent=2))


def read_sample_summaries(run_root: Path) -> list[dict[str, Any]]:
    """读取每个样本的 `audit/summary.json`，并按样本 ID 排序。"""
    summaries: list[dict[str, Any]] = []
    for path in sorted((run_root / "samples").glob("*/audit/summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    return summaries


def read_shard_summaries(run_root: Path) -> list[dict[str, Any]]:
    """读取 array shard 的局部 summary，主要用于审计 array 是否都结束。"""
    shard_dir = run_root / "tables" / "shards"
    if not shard_dir.exists():
        return []
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(shard_dir.glob("*_summary.json"))]


def build_batch_summary(
    run_id: str,
    mode: str,
    summaries: list[dict[str, Any]],
    shard_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """根据样本级 summary 构造与原批处理格式兼容的总表。"""
    common = {
        "run_id": run_id,
        "mode": mode,
        "num_samples": len(summaries),
        "num_ok": sum(item.get("status") == "ok" for item in summaries),
        "num_failed": sum(item.get("status") == "failed" for item in summaries),
        "num_skipped": sum(str(item.get("status", "")).startswith("skipped") for item in summaries),
        "num_jobs": sum(int(item.get("num_jobs", 0)) for item in summaries),
        "num_success": sum(int(item.get("num_success", 0)) for item in summaries),
        "num_shards": len(shard_summaries),
        "samples": summaries,
    }
    if mode == "oracle":
        common["num_variants"] = sum(int(item.get("num_variants", 0)) for item in summaries)
    return common


if __name__ == "__main__":
    main()
