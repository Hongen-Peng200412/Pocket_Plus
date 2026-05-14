from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .models import MakeDataLigand, MappingRow, SampleRef


NESTED_JSON_FIELDS = {"validation_detail", "download_attempts", "rcsb_nonpoly_scheme_row"}


def normalize_missing(value: Any) -> str:
    """
    统一处理 mmCIF/npz 中的缺失标记。

    输入参数:
        - value: Any, 原始字段值, 可能为 '.', '?', None 或普通字符串

    输出:
        - text: str, 缺失时为空字符串, 其它情况为 strip 后字符串
    """

    text = "" if value is None else str(value).strip()
    if text in {"?", ".", "None", "nan"}:
        return ""
    return text


def normalize_pdb_id(value: str) -> str:
    """
    将 PDB ID 统一为小写。

    输入参数:
        - value: str, 原始 PDB ID

    输出:
        - pdb_id: str, 小写 PDB ID
    """

    return value.strip().lower()


def load_raw_mapping(raw_json_path: Path | None) -> list[dict[str, str]] | None:
    """
    读取 raw.json 外层过滤文件。

    输入参数:
        - raw_json_path: Path | None, raw.json 路径; None 表示不过滤

    输出:
        - mapping: list[dict[str, str]] | None
            - None: 不使用 raw.json 过滤, 扫描 parsed_root 全部样本
            - []: 用户显式给出空过滤集, 处理 0 个样本
            - list[dict[str, str]]: 每项形如 {"emd_63092": "9LHB"}
    """

    if raw_json_path is None:
        return None
    raw_text = str(raw_json_path).strip()
    if not raw_text:
        return None
    data = json.loads(Path(raw_text).read_text(encoding="utf-8-sig"))
    return data


def resolve_pdb_samples(raw_mapping: list[dict[str, str]] | None, parsed_root: Path) -> tuple[list[SampleRef], list[MappingRow]]:
    """
    根据 raw.json 和 parsed_pdb 目录确定待处理 PDB。

    输入参数:
        - raw_mapping: list[dict[str, str]] | None, None 表示不过滤; [] 表示显式空过滤集
        - parsed_root: Path, Make_Data parsed_pdb 根目录

    输出:
        - samples: list[SampleRef], 待处理 PDB 样本列表
        - missing_rows: list[MappingRow], raw.json 中出现但 parsed_pdb 缺失的 PDB 记录
    """

    parsed_dirs = {p.name.upper(): p for p in parsed_root.iterdir() if p.is_dir()}
    if raw_mapping is None:
        samples = [
            SampleRef(emdb_id="", pdb_id=p.name.lower(), pdb_id_upper=p.name.upper(), parsed_dir=p)
            for p in sorted(parsed_root.iterdir(), key=lambda x: x.name.upper())
            if p.is_dir()
        ]
        return samples, []

    emdb_by_pdb: dict[str, list[str]] = {}
    for item in raw_mapping:
        for emdb_id, pdb_id in item.items():
            pdb_upper = str(pdb_id).strip().upper()
            emdb_by_pdb.setdefault(pdb_upper, []).append(str(emdb_id))

    samples: list[SampleRef] = []
    missing_rows: list[MappingRow] = []
    for pdb_upper in sorted(emdb_by_pdb):
        emdb_id = ";".join(sorted(set(emdb_by_pdb[pdb_upper])))
        parsed_dir = parsed_dirs.get(pdb_upper)
        if parsed_dir is None:
            missing_rows.append(
                MappingRow(
                    emdb_id=emdb_id,
                    pdb_id=pdb_upper.lower(),
                    pdb_id_upper=pdb_upper,
                    candidate_id=-1,
                    status=config.RAW_JSON_PDB_NOT_IN_PARSED_ROOT,
                    error_message="raw.json 中存在该 PDB, 但 parsed_pdb 根目录下没有对应子目录。",
                )
            )
            continue
        samples.append(SampleRef(emdb_id=emdb_id, pdb_id=pdb_upper.lower(), pdb_id_upper=pdb_upper, parsed_dir=parsed_dir))
    return samples, missing_rows


def load_make_data_ligands(parsed_dir: Path) -> list[MakeDataLigand]:
    """
    读取一个 parsed_pdb 子目录中的 Make_Data ligand instance。

    输入参数:
        - parsed_dir: Path, 形如 parsed_pdb/{pdb_id} 的目录, 必须包含 labels.npz 和 candidates.npz

    输出:
        - ligands: list[MakeDataLigand], labels.npz 中记录的 ligand instance 列表
    """

    labels = np.load(parsed_dir / "labels.npz", allow_pickle=True)
    candidates = np.load(parsed_dir / "candidates.npz", allow_pickle=True)
    ligand_candidate_ids = labels.get("ligand_candidate_ids", np.array([], dtype=np.int32))
    ligand_class_ids = labels.get("ligand_class_ids", np.array([], dtype=np.int32))
    ligands: list[MakeDataLigand] = []
    pdb_id = parsed_dir.name.lower()

    for candidate_id, class_id in zip(ligand_candidate_ids, ligand_class_ids):
        idx = int(candidate_id)
        ligands.append(
            MakeDataLigand(
                pdb_id=pdb_id,
                candidate_id=idx,
                ligand_class_id=int(class_id),
                ccd_id=str(candidates["resnames"][idx]).upper(),
                chain_id=str(candidates["chain_ids"][idx]),
                res_id=int(candidates["res_ids"][idx]),
                insertion_code=normalize_missing(candidates["insertion_codes"][idx]),
                make_data_heavy_atoms=int(candidates["n_heavy_atoms"][idx]),
                make_data_coords=np.asarray(candidates[f"candidate_coords_{idx}"], dtype=float),
            )
        )
    return ligands


def make_output_dirs(output_root: Path) -> None:
    """
    创建正式输出目录树。

    输入参数:
        - output_root: Path, 输出根目录

    输出:
        - None; 副作用为创建目录
    """

    for sub in [
        "rcsb_full_cif",
        "rcsb_chemcomp_cache",
        "rcsb_ligand_mol2",
        "mapping",
        "mapping/parts",
        "reports",
        "reports/parts",
        "logs",
    ]:
        (output_root / sub).mkdir(parents=True, exist_ok=True)


def row_to_csv_dict(row: MappingRow) -> dict[str, Any]:
    """
    将 MappingRow 转为 CSV 可写出的扁平字典。

    输入参数:
        - row: MappingRow, 单个 ligand mapping 记录

    输出:
        - result: dict[str, Any], 嵌套字段被 JSON 字符串化后的字典
    """

    result = row.to_dict()
    for key in NESTED_JSON_FIELDS:
        result[key] = json.dumps(result.get(key, {}), ensure_ascii=False, sort_keys=True)
    return result


def write_rows_csv(rows: list[MappingRow], path: Path) -> None:
    """
    写出 mapping CSV。

    输入参数:
        - rows: list[MappingRow], 待写出的 mapping 行
        - path: Path, CSV 输出路径

    输出:
        - None; 副作用为写出 CSV 文件
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(MappingRow()).keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row_to_csv_dict(row))


def write_rows_jsonl(rows: list[MappingRow], path: Path) -> None:
    """
    写出 mapping JSONL。

    输入参数:
        - rows: list[MappingRow], 待写出的 mapping 行
        - path: Path, JSONL 输出路径

    输出:
        - None; 副作用为写出 JSONL 文件
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def build_summary(rows: list[MappingRow], raw_json_entries: int, unique_pdb_ids_in_raw_json: int, matched_parsed_pdb_count: int, missing_parsed_pdb_count: int, output_root: Path) -> dict[str, Any]:
    """
    汇总 mapping 结果。

    输入参数:
        - rows: list[MappingRow], 全部 mapping 行
        - raw_json_entries: int, raw.json 原始条目数; 未使用 raw.json 时为 0
        - unique_pdb_ids_in_raw_json: int, raw.json 中唯一 PDB 数; 未使用 raw.json 时为 0
        - matched_parsed_pdb_count: int, 实际参与处理的 PDB 数
        - missing_parsed_pdb_count: int, raw.json 中出现但 parsed_pdb 缺失的 PDB 数
        - output_root: Path, 输出根目录

    输出:
        - summary: dict[str, Any], 可写入 validation_summary.json 的统计字典
    """

    status_counts = Counter(row.status for row in rows)
    dockem_counts = Counter(row.dockem_input_status for row in rows if row.dockem_input_status)
    emerald_counts = Counter(row.emerald_id_input_status for row in rows if row.emerald_id_input_status)
    pocketxmol_counts = Counter(row.pocketxmol_input_status for row in rows if row.pocketxmol_input_status)
    pdb_with_class4 = {row.pdb_id for row in rows if row.ligand_class_id == config.TARGET_LIGAND_CLASS_ID}

    return {
        "raw_json_entries": raw_json_entries,
        "unique_pdb_ids_in_raw_json": unique_pdb_ids_in_raw_json,
        "matched_parsed_pdb_count": matched_parsed_pdb_count,
        "missing_parsed_pdb_count": missing_parsed_pdb_count,
        "pdb_with_class4_ligands": len(pdb_with_class4),
        "total_ligand_records_from_labels": sum(1 for row in rows if row.candidate_id >= 0),
        "class4_candidate_count": sum(1 for row in rows if row.ligand_class_id == config.TARGET_LIGAND_CLASS_ID),
        "skipped_non_small_molecule_count": status_counts.get(config.SKIPPED_NON_SMALL_MOLECULE, 0),
        "status_counts": dict(sorted(status_counts.items())),
        "dockem_input_status_counts": dict(sorted(dockem_counts.items())),
        "emerald_id_input_status_counts": dict(sorted(emerald_counts.items())),
        "pocketxmol_input_status_counts": dict(sorted(pocketxmol_counts.items())),
        "output_root": str(output_root),
    }


def write_mapping_outputs(rows: list[MappingRow], output_root: Path, summary: dict[str, Any], part_index: int | None = None, part_count: int | None = None) -> None:
    """
    写出 mapping、failed_cases 和 summary。

    输入参数:
        - rows: list[MappingRow], 全部 mapping 行
        - output_root: Path, 输出根目录
        - summary: dict[str, Any], 统计摘要
        - part_index: int | None, array 分片编号; None 表示写总文件
        - part_count: int | None, array 分片总数; None 表示写总文件

    输出:
        - None; 副作用为写出 CSV/JSONL/JSON 文件
    """

    make_output_dirs(output_root)
    if part_index is None or part_count is None:
        csv_path = output_root / "mapping" / "ligand_mapping.csv"
        jsonl_path = output_root / "mapping" / "ligand_mapping.jsonl"
        failed_path = output_root / "mapping" / "failed_cases.csv"
        summary_path = output_root / "reports" / "validation_summary.json"
    else:
        suffix = f"part_{part_index}_of_{part_count}"
        csv_path = output_root / "mapping" / "parts" / f"ligand_mapping_{suffix}.csv"
        jsonl_path = output_root / "mapping" / "parts" / f"ligand_mapping_{suffix}.jsonl"
        failed_path = output_root / "mapping" / "parts" / f"failed_cases_{suffix}.csv"
        summary_path = output_root / "reports" / "parts" / f"validation_summary_{suffix}.json"

    for row in rows:
        row.source_file = str(csv_path)
        row.source_part_index = -1 if part_index is None else part_index
        row.source_part_count = -1 if part_count is None else part_count

    failed_rows = [
        row
        for row in rows
        if row.status not in {config.PASS_HIGH, config.SKIPPED_NON_SMALL_MOLECULE}
    ]
    write_rows_csv(rows, csv_path)
    write_rows_jsonl(rows, jsonl_path)
    write_rows_csv(failed_rows, failed_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def read_mapping_rows(output_root: Path) -> list[dict[str, Any]]:
    """
    兼容读取总表或 array parts 表。

    输入参数:
        - output_root: Path, 输出根目录

    输出:
        - rows: list[dict[str, Any]], 从 JSONL 读取的 mapping 行; 优先读取总表, 若不存在则读取 parts
    """

    total_jsonl = output_root / "mapping" / "ligand_mapping.jsonl"
    if total_jsonl.exists():
        files = [total_jsonl]
    else:
        files = sorted((output_root / "mapping" / "parts").glob("ligand_mapping_part_*_of_*.jsonl"))

    rows: list[dict[str, Any]] = []
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    row.setdefault("source_file", str(path))
                    rows.append(row)
    return rows
