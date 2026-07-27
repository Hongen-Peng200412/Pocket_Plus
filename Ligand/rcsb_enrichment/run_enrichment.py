from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from joblib import Parallel, delayed

from . import config
from .io_utils import build_summary, load_raw_mapping, make_output_dirs, resolve_pdb_samples, write_mapping_outputs
from .models import MappingRow, SampleRef
from .worker import process_one_pdb


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数. 

    输出:
        - args: argparse.Namespace, run_enrichment 主入口参数
    """

    parser = argparse.ArgumentParser(description="Download and validate RCSB ligand SMILES/mol2 for Make_Data ligands.")
    parser.add_argument("--raw-json", type=str, default=str(config.DEFAULT_RAW_JSON), help="raw.json 路径; 传空字符串表示不过滤。")
    parser.add_argument("--parsed-root", type=Path, default=config.DEFAULT_PARSED_ROOT, help="Make_Data parsed_pdb 根目录。")
    parser.add_argument("--output-root", type=Path, default=config.DEFAULT_OUTPUT_ROOT, help="输出根目录。")
    parser.add_argument("--n-jobs", type=int, default=16, help="joblib 按 PDB 并行的 worker 数。")
    parser.add_argument("--max-retries", type=int, default=3, help="RCSB 下载最大重试次数。")
    parser.add_argument("--retry-sleep", type=float, default=2.0, help="RCSB 下载重试间隔秒数。")
    parser.add_argument("--force-download", action="store_true", help="覆盖已有 RCSB 下载缓存。")
    parser.add_argument("--limit", type=int, default=None, help="debug 时只处理前 N 个 PDB。")
    parser.add_argument("--pdb-id", nargs="*", default=None, help="debug 时只处理指定 PDB ID。")
    parser.add_argument("--array-index", type=int, default=None, help="sbatch array 当前分片编号, 从 0 开始。")
    parser.add_argument("--array-count", type=int, default=None, help="sbatch array 分片总数。")
    return parser.parse_args()


def filter_samples(samples: list[SampleRef], pdb_ids: list[str] | None, limit: int | None, array_index: int | None, array_count: int | None) -> list[SampleRef]:
    """
    根据 debug PDB、limit 和 array 参数筛选样本. 

    输入参数:
        - samples: list[SampleRef], raw.json/parsed_root 求交集后的样本
        - pdb_ids: list[str] | None, 指定 PDB ID 列表
        - limit: int | None, 最大样本数
        - array_index: int | None, 当前 array 分片编号
        - array_count: int | None, array 分片总数

    输出:
        - selected: list[SampleRef], 当前运行实际处理的样本
    """

    selected = samples
    if pdb_ids:
        wanted = {value.upper() for value in pdb_ids}
        selected = [sample for sample in selected if sample.pdb_id_upper in wanted]
    if limit is not None:
        selected = selected[:limit]
    if array_index is not None and array_count is not None:
        selected = [sample for idx, sample in enumerate(selected) if idx % array_count == array_index]
    return selected


def raw_json_stats(raw_mapping) -> tuple[int, int]:
    """
    统计 raw.json 条目数和唯一 PDB 数. 

    输入参数:
        - raw_mapping: list[dict[str, str]] | None, load_raw_mapping 返回值

    输出:
        - raw_json_entries: int, raw.json 原始条目数
        - unique_pdb_ids: int, raw.json 中唯一 PDB 数
    """

    if raw_mapping is None:
        return 0, 0
    pdb_ids = set()
    for item in raw_mapping:
        for pdb_id in item.values():
            pdb_ids.add(str(pdb_id).upper())
    return len(raw_mapping), len(pdb_ids)


def print_summary(summary: dict) -> None:
    """
    打印 sbatch 日志友好的统计摘要. 

    输入参数:
        - summary: dict, build_summary 返回的统计字典

    输出:
        - None; 副作用为向 stdout 打印 JSON 摘要
    """

    print("========== RCSB Ligand Enrichment Summary ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print("====================================================")


def main() -> None:
    """
    RCSB ligand enrichment 统一入口. 

    输出:
        - None; 副作用为下载 RCSB 文件、写出 mapping/summary, 并在 stdout 打印统计摘要
    """

    args = parse_args()
    make_output_dirs(args.output_root)
    raw_mapping = load_raw_mapping(None if args.raw_json.strip() == "" else Path(args.raw_json))
    all_samples, missing_rows = resolve_pdb_samples(raw_mapping, args.parsed_root)
    selected_samples = filter_samples(all_samples, args.pdb_id, args.limit, args.array_index, args.array_count)
    if args.array_index is not None and args.array_count is not None:
        missing_rows = [row for idx, row in enumerate(missing_rows) if idx % args.array_count == args.array_index]
    raw_entries, unique_pdb_ids = raw_json_stats(raw_mapping)

    print(f"[Start] samples before filter: {len(all_samples)}, selected: {len(selected_samples)}")
    results = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(process_one_pdb)(
            sample,
            args.output_root,
            args.force_download,
            args.max_retries,
            args.retry_sleep,
        )
        for sample in selected_samples
    )

    rows: list[MappingRow] = []
    rows.extend(missing_rows)
    for result in results:
        rows.extend(result.rows)

    summary = build_summary(
        rows=rows,
        raw_json_entries=raw_entries,
        unique_pdb_ids_in_raw_json=unique_pdb_ids,
        matched_parsed_pdb_count=len(selected_samples),
        missing_parsed_pdb_count=len(missing_rows),
        output_root=args.output_root,
    )
    summary["created_at"] = datetime.now().isoformat(timespec="seconds")
    summary["array_index"] = args.array_index
    summary["array_count"] = args.array_count

    if args.array_index is None or args.array_count is None:
        write_mapping_outputs(rows, args.output_root, summary)
    else:
        write_mapping_outputs(rows, args.output_root, summary, part_index=args.array_index, part_count=args.array_count)
    print_summary(summary)


if __name__ == "__main__":
    main()
