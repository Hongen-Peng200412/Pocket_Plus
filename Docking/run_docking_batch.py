from __future__ import annotations

import argparse
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

from docking_pipeline.config import MatchingOptions, RosettaOptions, ServerPaths
from docking_pipeline.instance_postprocess import InstancePostprocessOptions
from docking_pipeline.io_utils import write_json
from docking_pipeline.runner import run_sample


def main() -> None:
    """
    运行多样本 docking 批处理. 

    输入参数:
        - CLI 参数, 包括 run_id、样本列表、nstruct、并发数和 instance 后处理参数

    输出:
        - None; 批处理结果写入 `/home/penghongen/分子对接尝试/pipeline_runs/{run_id}`
    """
    parser = argparse.ArgumentParser(description="运行 Pocket Plus docking 批处理")
    parser.add_argument("--run-id", required=True, help="批处理运行 ID, 用于隔离输出目录")
    parser.add_argument("--sample-list", help="逗号分隔样本 ID, 或包含样本 ID 的文本文件; 不填则自动发现")
    parser.add_argument("--max-samples", type=int, help="只运行前 N 个样本, 用于小规模验证")
    parser.add_argument("--jobs", type=int, default=1, help="joblib 样本级并发数; 建议不超过 sbatch CPU 核数")
    parser.add_argument("--rosetta-jobs", type=int, default=1, help="每个样本内部同时运行的 Rosetta 子进程数")
    parser.add_argument("--nstruct", type=int, default=2, help="每个 Rosetta job 生成的 decoy 数")
    parser.add_argument("--min-voxels", type=int, default=30, help="instance 最小体素数过滤阈值")
    parser.add_argument("--max-sites-per-sample", type=int, help="每个样本最多进入 docking 的高置信 site 数")
    parser.add_argument("--disable-merge", action="store_true", help="只过滤不合并 instance")
    parser.add_argument("--merge-min-voxel-distance", type=float, default=2.5, help="合并的最近体素距离阈值")
    parser.add_argument("--merge-center-distance", type=float, default=6.0, help="合并的中心距离阈值")
    parser.add_argument("--merge-max-bbox-span-increase", type=float, default=8.0, help="合并后包围盒对角线允许增加量")
    parser.add_argument("--shard-id", help="array 分片 ID; 设置后只写 shard summary, 避免并发覆盖总表")
    parser.add_argument("--plain-assignment", action="store_true", help="使用普通矩形 assignment, 不使用虚拟节点")
    parser.add_argument("--dry-run", action="store_true", help="生成输入和审计, 不执行 Rosetta")
    args = parser.parse_args()

    paths = ServerPaths.default()
    run_root = paths.allowed_root / "pipeline_runs" / args.run_id
    (run_root / "config").mkdir(parents=True, exist_ok=True)
    (run_root / "tables").mkdir(parents=True, exist_ok=True)

    sample_ids = _resolve_sample_ids(paths, args.sample_list)
    if args.max_samples is not None:
        sample_ids = sample_ids[: args.max_samples]
    config_dir = run_root / "config"
    tables_dir = run_root / "tables"
    if args.shard_id:
        config_dir = config_dir / "shards"
        tables_dir = tables_dir / "shards"
    config_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    sample_list_name = "sample_list.txt" if not args.shard_id else f"{args.shard_id}_sample_list.txt"
    (config_dir / sample_list_name).write_text("\n".join(sample_ids) + "\n", encoding="utf-8")

    rosetta_options = replace(RosettaOptions.smoke(), nstruct=args.nstruct)
    instance_options = InstancePostprocessOptions(
        min_voxels=args.min_voxels,
        enable_merge=not args.disable_merge,
        merge_min_voxel_distance=args.merge_min_voxel_distance,
        merge_center_distance=args.merge_center_distance,
        merge_max_bbox_span_increase=args.merge_max_bbox_span_increase,
    )
    config = {
        "run_id": args.run_id,
        "sample_count": len(sample_ids),
        "nstruct": args.nstruct,
        "jobs": args.jobs,
        "rosetta_jobs": args.rosetta_jobs,
        "dry_run": args.dry_run,
        "plain_assignment": args.plain_assignment,
        "shard_id": args.shard_id,
        "max_sites_per_sample": args.max_sites_per_sample,
        "instance_options": instance_options.__dict__,
    }
    config_name = "run_config.json" if not args.shard_id else f"{args.shard_id}_run_config.json"
    write_json(config_dir / config_name, config)

    summaries = _parallel_map(
        [
            {
                "pdb_id": pdb_id,
                "paths": paths,
                "rosetta_options": rosetta_options,
                "matching_options": MatchingOptions.current(),
                "instance_options": instance_options,
                "run_id": args.run_id,
                "use_virtual_nodes": not args.plain_assignment,
                "dry_run": args.dry_run,
                "max_sites_per_sample": args.max_sites_per_sample,
                "rosetta_jobs": args.rosetta_jobs,
            }
            for pdb_id in sample_ids
        ],
        args.jobs,
    )
    batch_summary = {
        "run_id": args.run_id,
        "num_samples": len(summaries),
        "num_ok": sum(item.get("status") == "ok" for item in summaries),
        "num_failed": sum(item.get("status") == "failed" for item in summaries),
        "num_skipped": sum(str(item.get("status", "")).startswith("skipped") for item in summaries),
        "num_jobs": sum(int(item.get("num_jobs", 0)) for item in summaries),
        "num_success": sum(int(item.get("num_success", 0)) for item in summaries),
        "samples": summaries,
    }
    summary_name = "batch_summary.json" if not args.shard_id else f"{args.shard_id}_summary.json"
    write_json(tables_dir / summary_name, batch_summary)
    print(f"写入批处理汇总: {tables_dir / summary_name}")
    if batch_summary["num_failed"] > 0:
        raise SystemExit(1)


def run_one(payload: dict[str, Any]) -> dict[str, Any]:
    """
    执行单个样本并把异常写入样本 audit. 

    输入参数:
        - payload: dict[str, Any], `run_sample` 所需参数

    输出:
        - summary: dict[str, Any], 成功或失败摘要
    """
    pdb_id = str(payload["pdb_id"]).lower()
    paths: ServerPaths = payload["paths"]
    run_id = str(payload["run_id"])
    sample_dir = paths.allowed_root / "pipeline_runs" / run_id / "samples" / pdb_id
    try:
        return run_sample(**payload)
    except Exception as exc:  # noqa: BLE001 - 批处理边界需要捕获并审计单样本失败
        audit_dir = sample_dir / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        error = {
            "pdb_id": pdb_id,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(audit_dir / "error.json", error)
        return error


def _parallel_map(payloads: list[dict[str, Any]], jobs: int) -> list[dict[str, Any]]:
    """使用 joblib 并行执行样本; joblib 不可用时退回顺序执行. """
    if jobs <= 1:
        return [run_one(payload) for payload in payloads]
    try:
        from joblib import Parallel, delayed
    except ImportError:
        return [run_one(payload) for payload in payloads]
    return Parallel(n_jobs=jobs, backend="loky")(delayed(run_one)(payload) for payload in payloads)


def _resolve_sample_ids(paths: ServerPaths, sample_list: str | None) -> list[str]:
    """
    解析样本列表. 

    输入参数:
        - paths: ServerPaths, 服务器路径配置
        - sample_list: str 或 None, 逗号分隔样本、文本文件路径或 None

    输出:
        - sample_ids: list[str], 小写 PDB ID 列表
    """
    if sample_list:
        path = Path(sample_list)
        if path.exists():
            return [line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [item.strip().lower() for item in sample_list.split(",") if item.strip()]
    return sorted(path.name.lower() for path in paths.inference_root.iterdir() if (path / "summary.json").exists())


if __name__ == "__main__":
    main()
