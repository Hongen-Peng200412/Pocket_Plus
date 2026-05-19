from __future__ import annotations

import argparse
import itertools
import traceback
from pathlib import Path
from typing import Any

from docking_pipeline.config import ServerPaths
from docking_pipeline.instance_postprocess import InstancePostprocessOptions, estimate_job_count, postprocess_sites
from docking_pipeline.io_utils import (
    load_instance_label,
    load_sites,
    load_voxel_transform,
    read_ligand_candidates,
    write_json,
)


def main() -> None:
    """
    运行不调用 Rosetta 的 instance 后处理参数预扫描。

    输入参数:
        - CLI 参数, 包括 run_id、样本列表、参数网格和并发数

    输出:
        - None; 预扫描表写入 `/home/penghongen/分子对接尝试/pipeline_runs/{run_id}/tables`
    """
    parser = argparse.ArgumentParser(description="预扫描 instance 后处理参数")
    parser.add_argument("--run-id", required=True, help="预扫描 run ID")
    parser.add_argument("--sample-list", help="逗号分隔样本 ID, 或包含样本 ID 的文本文件; 不填则自动发现")
    parser.add_argument("--max-samples", type=int, help="只扫描前 N 个样本")
    parser.add_argument("--jobs", type=int, default=1, help="joblib 样本级并发数")
    parser.add_argument("--max-sites-per-sample", type=int, help="估计 job 数时每个样本最多保留的高置信 site 数")
    parser.add_argument("--min-voxels-grid", default="10,20,30,50,80", help="最小体素数网格")
    parser.add_argument("--merge-min-distance-grid", default="1.5,2.5,3.5", help="最近体素距离阈值网格")
    parser.add_argument("--merge-center-distance-grid", default="4.0,6.0,8.0", help="中心距离阈值网格")
    parser.add_argument("--merge-bbox-increase-grid", default="6.0,8.0,12.0", help="包围盒对角线增加量阈值网格")
    parser.add_argument("--disable-merge", action="store_true", help="只扫描最小体素过滤")
    args = parser.parse_args()

    paths = ServerPaths.default()
    run_root = paths.allowed_root / "pipeline_runs" / args.run_id
    (run_root / "config").mkdir(parents=True, exist_ok=True)
    (run_root / "tables").mkdir(parents=True, exist_ok=True)

    sample_ids = _resolve_sample_ids(paths, args.sample_list)
    if args.max_samples is not None:
        sample_ids = sample_ids[: args.max_samples]
    (run_root / "config" / "sample_list.txt").write_text("\n".join(sample_ids) + "\n", encoding="utf-8")

    option_grid = _build_option_grid(args)
    write_json(run_root / "config" / "prescan_config.json", {"run_id": args.run_id, "num_options": len(option_grid), "options": [item.__dict__ for item in option_grid]})
    payloads = [{"pdb_id": pdb_id, "paths": paths, "run_id": args.run_id, "options": option_grid, "max_sites_per_sample": args.max_sites_per_sample} for pdb_id in sample_ids]
    rows_nested = _parallel_map(payloads, args.jobs)
    rows = [row for rows_for_sample in rows_nested for row in rows_for_sample]
    write_json(run_root / "tables" / "instance_prescan_rows.json", rows)
    write_json(run_root / "tables" / "instance_prescan_summary.json", _summarize(rows))
    print(f"写入预扫描结果: {run_root / 'tables' / 'instance_prescan_summary.json'}")


def scan_one(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """
    扫描单个样本的参数网格。

    输入参数:
        - payload: dict[str, Any], 包含样本 ID、路径、run_id 和参数列表

    输出:
        - rows: list[dict[str, Any]], 当前样本每个参数组合的一行统计
    """
    pdb_id = str(payload["pdb_id"]).lower()
    paths: ServerPaths = payload["paths"]
    options: list[InstancePostprocessOptions] = payload["options"]
    max_sites_per_sample = payload.get("max_sites_per_sample")
    sample_dir = paths.inference_root / pdb_id
    try:
        sites = load_sites(sample_dir)
        label = load_instance_label(sample_dir)
        origin, voxel_size = load_voxel_transform(sample_dir)
        ligands = read_ligand_candidates(paths.ligand_mapping_csv, pdb_id)
        rows: list[dict[str, Any]] = []
        for index, option in enumerate(options):
            result = postprocess_sites(sites, label, origin, voxel_size, option)
            selected_site_count = len(result.sites) if max_sites_per_sample is None else min(len(result.sites), int(max_sites_per_sample))
            rows.append(
                {
                    "pdb_id": pdb_id,
                    "option_index": index,
                    "status": "ok",
                    "min_voxels": option.min_voxels,
                    "enable_merge": option.enable_merge,
                    "merge_min_voxel_distance": option.merge_min_voxel_distance,
                    "merge_center_distance": option.merge_center_distance,
                    "merge_max_bbox_span_increase": option.merge_max_bbox_span_increase,
                    "num_raw_sites": len(sites),
                    "num_output_sites": len(result.sites),
                    "num_selected_sites": selected_site_count,
                    "max_sites_per_sample": max_sites_per_sample,
                    "num_filtered_sites": len(result.filtered_sites),
                    "num_merge_candidates": len(result.merge_candidates),
                    "num_merges": len(result.merges),
                    "num_ligands": len(ligands),
                    "estimated_jobs": estimate_job_count(selected_site_count, len(ligands), 2),
                }
            )
        return rows
    except Exception as exc:  # noqa: BLE001 - 预扫描边界需要保留失败样本
        return [
            {
                "pdb_id": pdb_id,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        ]


def _build_option_grid(args: argparse.Namespace) -> list[InstancePostprocessOptions]:
    """根据 CLI 参数构造后处理参数网格。"""
    min_voxels_values = [int(value) for value in args.min_voxels_grid.split(",") if value]
    if args.disable_merge:
        return [
            InstancePostprocessOptions(
                min_voxels=min_voxels,
                enable_merge=False,
                merge_min_voxel_distance=0.0,
                merge_center_distance=0.0,
                merge_max_bbox_span_increase=0.0,
            )
            for min_voxels in min_voxels_values
        ]
    min_distance_values = [float(value) for value in args.merge_min_distance_grid.split(",") if value]
    center_distance_values = [float(value) for value in args.merge_center_distance_grid.split(",") if value]
    bbox_values = [float(value) for value in args.merge_bbox_increase_grid.split(",") if value]
    return [
        InstancePostprocessOptions(
            min_voxels=min_voxels,
            enable_merge=True,
            merge_min_voxel_distance=min_distance,
            merge_center_distance=center_distance,
            merge_max_bbox_span_increase=bbox_increase,
        )
        for min_voxels, min_distance, center_distance, bbox_increase in itertools.product(
            min_voxels_values,
            min_distance_values,
            center_distance_values,
            bbox_values,
        )
    ]


def _parallel_map(payloads: list[dict[str, Any]], jobs: int) -> list[list[dict[str, Any]]]:
    """使用 joblib 并行扫描样本; joblib 不可用时退回顺序执行。"""
    if jobs <= 1:
        return [scan_one(payload) for payload in payloads]
    try:
        from joblib import Parallel, delayed
    except ImportError:
        return [scan_one(payload) for payload in payloads]
    return Parallel(n_jobs=jobs, backend="loky")(delayed(scan_one)(payload) for payload in payloads)


def _resolve_sample_ids(paths: ServerPaths, sample_list: str | None) -> list[str]:
    """解析样本列表或从推理根目录自动发现样本。"""
    if sample_list:
        path = Path(sample_list)
        if path.exists():
            return [line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [item.strip().lower() for item in sample_list.split(",") if item.strip()]
    return sorted(path.name.lower() for path in paths.inference_root.iterdir() if (path / "summary.json").exists())


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """按参数组合汇总预扫描统计。"""
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    by_option: dict[int, list[dict[str, Any]]] = {}
    for row in ok_rows:
        by_option.setdefault(int(row["option_index"]), []).append(row)
    option_summaries = []
    for option_index, group in sorted(by_option.items()):
        option_summaries.append(
            {
                "option_index": option_index,
                "num_samples": len(group),
                "total_raw_sites": sum(int(row["num_raw_sites"]) for row in group),
                "total_output_sites": sum(int(row["num_output_sites"]) for row in group),
                "total_selected_sites": sum(int(row["num_selected_sites"]) for row in group),
                "total_filtered_sites": sum(int(row["num_filtered_sites"]) for row in group),
                "total_merges": sum(int(row["num_merges"]) for row in group),
                "total_estimated_jobs": sum(int(row["estimated_jobs"]) for row in group),
                "min_voxels": group[0]["min_voxels"],
                "max_sites_per_sample": group[0]["max_sites_per_sample"],
                "enable_merge": group[0]["enable_merge"],
                "merge_min_voxel_distance": group[0]["merge_min_voxel_distance"],
                "merge_center_distance": group[0]["merge_center_distance"],
                "merge_max_bbox_span_increase": group[0]["merge_max_bbox_span_increase"],
            }
        )
    return {
        "num_rows": len(rows),
        "num_failed_rows": sum(row.get("status") == "failed" for row in rows),
        "options": option_summaries,
    }


if __name__ == "__main__":
    main()
