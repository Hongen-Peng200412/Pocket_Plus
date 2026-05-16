from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .io_utils import build_summary, write_mapping_outputs
from .models import MappingRow


def parse_args() -> argparse.Namespace:
    """
    解析 parts 合并命令行参数。

    输出:
        - args: argparse.Namespace, merge_outputs 参数
    """

    parser = argparse.ArgumentParser(description="Merge RCSB ligand enrichment array part outputs.")
    parser.add_argument("--output-root", type=Path, required=True, help="输出根目录。")
    parser.add_argument("--array-count", type=int, required=True, help="array 分片总数。")
    return parser.parse_args()


def load_part_jsonl(output_root: Path, array_count: int) -> list[MappingRow]:
    """
    读取所有 array part JSONL 并转为 MappingRow。

    输入参数:
        - output_root: Path, 输出根目录
        - array_count: int, array 分片总数

    输出:
        - rows: list[MappingRow], 合并后的 mapping 行
    """

    rows: list[MappingRow] = []
    for index in range(array_count):
        path = output_root / "mapping" / "parts" / f"ligand_mapping_part_{index}_of_{array_count}.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                data["source_file"] = str(path)
                data["source_part_index"] = index
                data["source_part_count"] = array_count
                rows.append(MappingRow(**data))
    return rows


def load_part_summaries(output_root: Path, array_count: int) -> list[dict]:
    """
    读取 array part summary。

    输入参数:
        - output_root: Path, 输出根目录
        - array_count: int, array 分片总数

    输出:
        - summaries: list[dict], 已存在 part summary 的列表
    """

    summaries: list[dict] = []
    for index in range(array_count):
        path = output_root / "reports" / "parts" / f"validation_summary_part_{index}_of_{array_count}.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    return summaries


def main() -> None:
    """
    合并 array parts 输出。

    输出:
        - None; 副作用为写出总 mapping、failed cases 和 validation_summary.json
    """

    args = parse_args()
    rows = load_part_jsonl(args.output_root, args.array_count)
    part_summaries = load_part_summaries(args.output_root, args.array_count)
    if part_summaries:
        raw_json_entries = max(int(item.get("raw_json_entries", 0)) for item in part_summaries)
        unique_pdb_ids = max(int(item.get("unique_pdb_ids_in_raw_json", 0)) for item in part_summaries)
        matched_count = sum(int(item.get("matched_parsed_pdb_count", 0)) for item in part_summaries)
        missing_count = sum(int(item.get("missing_parsed_pdb_count", 0)) for item in part_summaries)
    else:
        raw_json_entries = 0
        unique_pdb_ids = 0
        matched_count = len({row.pdb_id for row in rows if row.pdb_id and row.status != "RAW_JSON_PDB_NOT_IN_PARSED_ROOT"})
        missing_count = sum(1 for row in rows if row.status == "RAW_JSON_PDB_NOT_IN_PARSED_ROOT")
    summary = build_summary(
        rows=rows,
        raw_json_entries=raw_json_entries,
        unique_pdb_ids_in_raw_json=unique_pdb_ids,
        matched_parsed_pdb_count=matched_count,
        missing_parsed_pdb_count=missing_count,
        output_root=args.output_root,
    )
    summary["created_at"] = datetime.now().isoformat(timespec="seconds")
    summary["merged_from_array_count"] = args.array_count
    write_mapping_outputs(rows, args.output_root, summary)
    print("========== Merged RCSB Ligand Enrichment Summary ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print("===========================================================")


if __name__ == "__main__":
    main()
