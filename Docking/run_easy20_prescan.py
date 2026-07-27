from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

DOCKING_DIR = Path(__file__).resolve().parent
if str(DOCKING_DIR) not in sys.path:
    sys.path.insert(0, str(DOCKING_DIR))

from docking_pipeline.config import ServerPaths
from docking_pipeline.io_utils import (
    load_sites,
    mol2_internal_metals,
    rosetta_ligand_name,
    write_json,
)
from docking_pipeline.records import InferenceSite, LigandCandidate
from docking_pipeline.io_utils import STANDALONE_METAL_CCD


GLYCAN_CCD_IDS = {
    "NAG",
    "NDG",
    "BMA",
    "MAN",
    "FUC",
    "GAL",
    "GLC",
    "BGC",
    "SIA",
    "A2G",
    "XYS",
    "FUL",
    "GCU",
    "GCS",
    "G6D",
    "MMA",
    "RAM",
}


def main() -> None:
    """
    统计 full80 中适合快速验证的 easy20 样本集合. 

    输入参数:
        - CLI 参数, 包括 prescan_run_id、样本数量和保守 instance 合并参数

    输出:
        - None; 统计表写入 `/home/penghongen/分子对接尝试/easy20_prescans/{prescan_run_id}`
    """
    parser = argparse.ArgumentParser(description="统计 Pocket Plus Docking easy20 样本集合")
    parser.add_argument("--prescan-run-id", required=True, help="prescan run ID, 用于隔离输出目录")
    parser.add_argument("--top-n", type=int, default=20, help="每套 easy 集合保留的样本数")
    parser.add_argument("--merge-center-distance", type=float, default=8.0, help="保守合并 instance 的中心距离阈值, 单位 Å")
    parser.add_argument("--large-heavy-atoms", type=int, default=80, help="大 ligand 的 heavy atom 数阈值")
    args = parser.parse_args()

    paths = ServerPaths.default()
    output_dir = paths.allowed_root / "easy20_prescans" / args.prescan_run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = sorted(path.name.lower() for path in paths.inference_root.iterdir() if (path / "summary.json").exists())
    instance_metrics = load_instance_metrics(paths.inference_root)
    ligands_by_sample = load_ligands_by_sample(paths.ligand_mapping_csv, set(sample_ids))
    rows = [
        sample_prescan_row(paths, sample_id, instance_metrics, ligands_by_sample, args.merge_center_distance, args.large_heavy_atoms)
        for sample_id in sample_ids
    ]
    usable_rows = [row for row in rows if row["num_dockable_ligands"] > 0 and row["num_conservative_sites"] > 0]
    easy_by_nk = sorted(usable_rows, key=lambda row: (row["nk"], row["num_dockable_ligands"], row["num_conservative_sites"], row["sample_id"]))[: args.top_n]
    easy_by_instance_f1 = sorted(
        usable_rows,
        key=lambda row: (missing_to_negative_inf(row["instance_f1"]), -row["nk"], row["sample_id"]),
        reverse=True,
    )[: args.top_n]

    write_csv(output_dir / "all_samples.csv", rows)
    write_csv(output_dir / "easy20_by_nk.csv", easy_by_nk)
    write_csv(output_dir / "easy20_by_instance_f1.csv", easy_by_instance_f1)
    write_sample_list(output_dir / "easy20_by_nk.txt", easy_by_nk)
    write_sample_list(output_dir / "easy20_by_instance_f1.txt", easy_by_instance_f1)
    manifest = {
        "prescan_run_id": args.prescan_run_id,
        "top_n": args.top_n,
        "merge_center_distance": args.merge_center_distance,
        "large_heavy_atoms": args.large_heavy_atoms,
        "num_samples": len(rows),
        "num_usable_samples": len(usable_rows),
        "easy20_by_nk": [row["sample_id"] for row in easy_by_nk],
        "easy20_by_instance_f1": [row["sample_id"] for row in easy_by_instance_f1],
        "notes": [
            "num_conservative_sites 使用 min_voxels=1 且仅按中心距离 < merge_center_distance 保守合并 instance。",
            "nk = num_dockable_ligands * num_conservative_sites; 纯金属离子不计入 dockable ligand。",
            "bad_at_tool_ratio_conservative 当前只把明确糖类/支链糖 CCD 计入。",
        ],
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def sample_prescan_row(
    paths: ServerPaths,
    sample_id: str,
    instance_metrics: dict[str, dict[str, Any]],
    ligands_by_sample: dict[str, list[LigandCandidate]],
    merge_center_distance: float,
    large_heavy_atoms: int,
) -> dict[str, Any]:
    """
    构造单个样本的 easy20 统计行. 

    输入参数:
        - paths: ServerPaths, 服务器路径配置
        - sample_id: str, 小写 PDB ID
        - instance_metrics: dict[str, dict[str, Any]], 按样本索引的前置网络指标
        - ligands_by_sample: dict[str, list[LigandCandidate]], 一次性读取的 dockable ligand 候选
        - merge_center_distance: float, instance 中心距离合并阈值
        - large_heavy_atoms: int, 大 ligand 的 heavy atom 数阈值

    输出:
        - row: dict[str, Any], 单样本统计字段
    """
    sample_dir = paths.inference_root / sample_id
    sites = load_sites(sample_dir)
    conservative_sites = center_merge_sites(sites, merge_center_distance)
    ligands = ligands_by_sample.get(sample_id, [])
    num_glycan = sum(ligand.ccd_id.upper() in GLYCAN_CCD_IDS for ligand in ligands)
    num_large = sum(ligand.heavy_atoms > large_heavy_atoms for ligand in ligands)
    num_internal_metal = sum(bool(ligand.internal_metals) for ligand in ligands)
    metrics = instance_metrics.get(sample_id, {})
    n = len(ligands)
    k = len(conservative_sites)
    return {
        "sample_id": sample_id,
        "num_dockable_ligands": n,
        "num_raw_sites": len(sites),
        "num_conservative_sites": k,
        "nk": n * k,
        "instance_precision": numeric_or_blank(metrics.get("instance_precision")),
        "instance_recall": numeric_or_blank(metrics.get("instance_recall")),
        "instance_f1": numeric_or_blank(metrics.get("instance_f1")),
        "voxel_precision": numeric_or_blank(metrics.get("voxel_precision")),
        "voxel_recall": numeric_or_blank(metrics.get("voxel_recall")),
        "voxel_f1": numeric_or_blank(metrics.get("voxel_f1")),
        "num_glycan_like_ligands": num_glycan,
        "bad_at_tool_ratio_conservative": num_glycan / n if n else "",
        "num_large_ligands": num_large,
        "large_ligand_ratio": num_large / n if n else "",
        "num_internal_metal_ligands": num_internal_metal,
        "internal_metal_ratio": num_internal_metal / n if n else "",
        "ligand_labels": ";".join(ligand.label for ligand in ligands),
        "ligand_ccd_ids": ";".join(ligand.ccd_id for ligand in ligands),
        "ligand_heavy_atoms": ";".join(str(ligand.heavy_atoms) for ligand in ligands),
        "ligand_internal_metals": ";".join("+".join(ligand.internal_metals) for ligand in ligands),
    }


def load_ligands_by_sample(mapping_csv: Path, sample_ids: set[str]) -> dict[str, list[LigandCandidate]]:
    """
    一次性读取 ligand mapping, 避免每个样本重复扫描大 CSV. 

    输入参数:
        - mapping_csv: Path, ligand mapping CSV
        - sample_ids: set[str], 需要统计的小写样本 ID 集合

    输出:
        - ligands_by_sample: dict[str, list[LigandCandidate]], key 为小写样本 ID
    """
    rows_by_sample: dict[str, list[dict[str, str]]] = {sample_id: [] for sample_id in sample_ids}
    with mapping_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sample_id = row.get("pdb_id", "").lower()
            if sample_id in rows_by_sample:
                rows_by_sample[sample_id].append(row)

    ligands_by_sample: dict[str, list[LigandCandidate]] = {}
    for sample_id, rows in rows_by_sample.items():
        valid_rows = [row for row in rows if is_dockable_mapping_row(row)]
        ccd_counts: dict[str, int] = {}
        for row in valid_rows:
            ccd_id = row.get("ccd_id", "").upper()
            ccd_counts[ccd_id] = ccd_counts.get(ccd_id, 0) + 1

        ligands: list[LigandCandidate] = []
        for row in valid_rows:
            mol2_path = Path(row.get("native_mol2_path", ""))
            ccd_id = row.get("ccd_id", "").upper()
            ligand_index = len(ligands) + 1
            label = f"{ccd_id}_{ligand_index:02d}" if ccd_counts[ccd_id] > 1 else ccd_id
            ligands.append(
                LigandCandidate(
                    pdb_id=sample_id,
                    ccd_id=ccd_id,
                    label=label,
                    rosetta_name=rosetta_ligand_name(ligand_index),
                    mol2_path=mol2_path,
                    heavy_atoms=int(row.get("mol2_heavy_atoms") or 0),
                    internal_metals=mol2_internal_metals(mol2_path),
                )
            )
        ligands_by_sample[sample_id] = ligands
    return ligands_by_sample


def is_dockable_mapping_row(row: dict[str, str]) -> bool:
    """
    判断 mapping 行是否属于当前 docking 可用 ligand. 

    输入参数:
        - row: dict[str, str], ligand_mapping.csv 的一行

    输出:
        - is_dockable: bool, 是否通过状态、mol2、heavy atom 和纯金属过滤
    """
    mol2_path = Path(row.get("native_mol2_path", ""))
    heavy_atoms = int(row.get("mol2_heavy_atoms") or 0)
    ccd_id = row.get("ccd_id", "").upper()
    return (
        row.get("status") == "PASS_HIGH"
        and row.get("emerald_id_input_status") == "FORMAT_OK_FOR_LIBRARY_ENTRY"
        and mol2_path.exists()
        and heavy_atoms > 1
        and ccd_id not in STANDALONE_METAL_CCD
    )


def center_merge_sites(sites: list[InferenceSite], merge_center_distance: float) -> list[InferenceSite]:
    """
    仅按 instance center 距离合并 site, 不读取 instance label 体素坐标. 

    输入参数:
        - sites: list[InferenceSite], 原始推理 instance 列表
        - merge_center_distance: float, 中心距离小于等于该值时归入同一连通组

    输出:
        - merged_sites: list[InferenceSite], 每个连通组一个保守合并 site
    """
    if len(sites) < 2:
        return list(sites)
    parent = {site.instance_id: site.instance_id for site in sites}

    def find(instance_id: int) -> int:
        while parent[instance_id] != instance_id:
            parent[instance_id] = parent[parent[instance_id]]
            instance_id = parent[instance_id]
        return instance_id

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_index, left in enumerate(sites):
        for right in sites[left_index + 1 :]:
            distance = float(np.linalg.norm(np.asarray(left.center_world_xyz) - np.asarray(right.center_world_xyz)))
            if distance <= merge_center_distance:
                union(left.instance_id, right.instance_id)
    groups: dict[int, list[InferenceSite]] = {}
    for site in sites:
        groups.setdefault(find(site.instance_id), []).append(site)
    merged: list[InferenceSite] = []
    for group in groups.values():
        if len(group) == 1:
            merged.append(group[0])
            continue
        weights = np.asarray([site.voxel_count for site in group], dtype=float)
        centers = np.asarray([site.center_world_xyz for site in group], dtype=float)
        center = tuple(float(value) for value in np.average(centers, axis=0, weights=weights))
        representative = min(site.instance_id for site in group)
        merged.append(
            InferenceSite(
                instance_id=representative,
                center_world_xyz=center,
                score_mean=float(np.average([site.score_mean for site in group], weights=weights)),
                score_max=float(max(site.score_max for site in group)),
                voxel_count=int(sum(site.voxel_count for site in group)),
            )
        )
    return sorted(merged, key=lambda site: site.instance_id)


def load_instance_metrics(inference_root: Path) -> dict[str, dict[str, Any]]:
    """
    从推理输出中读取逐样本前置网络指标. 

    输入参数:
        - inference_root: Path, stage2 推理输出目录

    输出:
        - metrics: dict[str, dict[str, Any]], key 为小写样本 ID
    """
    candidates = [
        inference_root / "per_sample_best_metrics.json",
        inference_root / "per_sample_metrics.json",
        inference_root / "sample_metrics.json",
        inference_root / "metrics_by_sample.json",
        inference_root / "best_per_sample.json",
    ]
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return normalize_metric_mapping(data)
    metrics: dict[str, dict[str, Any]] = {}
    for sample_dir in sorted(path for path in inference_root.iterdir() if path.is_dir()):
        for name in ("summary.json", "metrics.json", "best_summary.json"):
            path = sample_dir / name
            if path.exists():
                metrics[sample_dir.name.lower()] = flatten_metric_dict(json.loads(path.read_text(encoding="utf-8")))
                break
    return metrics


def normalize_metric_mapping(data: Any) -> dict[str, dict[str, Any]]:
    """把不同 JSON 形态统一成 `{sample_id: metrics}`. """
    if isinstance(data, dict):
        if "samples" in data and isinstance(data["samples"], list):
            return normalize_metric_mapping(data["samples"])
        result: dict[str, dict[str, Any]] = {}
        for key, value in data.items():
            if isinstance(value, dict):
                sample_id = str(value.get("sample_id") or value.get("pdb_id") or key).lower()
                result[sample_id] = flatten_metric_dict(value)
        return result
    if isinstance(data, list):
        result = {}
        for item in data:
            if isinstance(item, dict):
                sample_id = str(item.get("sample_id") or item.get("sample_name") or item.get("pdb_id") or item.get("id") or "").lower()
                if sample_id:
                    result[sample_id] = flatten_metric_dict(item)
        return result
    return {}


def flatten_metric_dict(data: dict[str, Any]) -> dict[str, Any]:
    """抽取常用 instance/voxel 指标字段. """
    flat = dict(data)
    for nested_key in ("metrics", "best_metrics", "summary"):
        nested = data.get(nested_key)
        if isinstance(nested, dict):
            flat.update(nested)
    aliases = {
        "instance_precision": ("instance_precision", "avg_instance_precision"),
        "instance_recall": ("instance_recall", "avg_instance_recall"),
        "instance_f1": ("instance_f1", "avg_instance_f1"),
        "voxel_precision": ("voxel_precision", "avg_voxel_precision"),
        "voxel_recall": ("voxel_recall", "avg_voxel_recall"),
        "voxel_f1": ("voxel_f1", "avg_voxel_f1", "avg_voxel_dice"),
    }
    return {target: first_present(flat, names) for target, names in aliases.items()}


def first_present(data: dict[str, Any], names: tuple[str, ...]) -> Any:
    """返回第一个存在的字段值. """
    for name in names:
        if name in data:
            return data[name]
    return ""


def missing_to_negative_inf(value: Any) -> float:
    """排序时把缺失指标放到最后. """
    try:
        return float(value)
    except (TypeError, ValueError):
        return -math.inf


def numeric_or_blank(value: Any) -> float | str:
    """把 JSON 字段转成 float; 缺失或不可转时保留空字符串. """
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出 CSV 表. """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_sample_list(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出样本 ID 文本列表. """
    path.write_text("\n".join(row["sample_id"] for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
