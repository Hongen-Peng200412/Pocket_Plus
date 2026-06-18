from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any


MODEL_SPECS = (
    ("unet_c1", "unet_raw"),
    ("unet_c2", "unet_diff"),
    ("unet_base", "unet_full"),
    ("emb_unet", "emb_unet"),
)

MODE_SPECS = (
    ("stardard", "standard"),
    ("strict", "strict"),
)

DATASET_SPECS = {
    "protein": {
        "title": "蛋白测试集",
        "sample_json": "protein_110.json",
        "emap_dataset": "protein_110",
        "dl_suffix_template": "{source}_{mode}",
        "phenix_suffix_template": "phenix_real_space_diff_map_{mode}_110",
    },
    "nucleic": {
        "title": "核酸测试集",
        "sample_json": "nucleic_40.json",
        "emap_dataset": "nucleic_40",
        "dl_suffix_template": "{source}_{mode}_nucleic_40",
        "phenix_suffix_template": "phenix_real_space_diff_map_{mode}_nucleic_40",
    },
}


def read_json(path: Path) -> Any:
    """
    读取 UTF-8 JSON 文件。

    输入参数:
        - path: Path, JSON 文件路径

    输出:
        - data: Any, JSON 反序列化后的对象
    """
    with path.open("r", encoding="utf-8-sig") as file_obj:
        return json.load(file_obj)


def sample_name_from_pair(pair: dict[str, Any]) -> str:
    """
    从 Pocket Plus 样本条目中恢复样本名。

    输入参数:
        - pair: dict[str, Any], 单个 raw pair 条目

    输出:
        - sample_name: str, 小写 PDB ID 或样本 ID
    """
    if pair.get("sample_name"):
        return str(pair["sample_name"]).lower()
    if pair.get("labels_npz_path"):
        return Path(str(pair["labels_npz_path"])).parent.name.lower()
    return Path(str(pair["cif_path"])).stem.lower()


def emdb_id_from_pair(pair: dict[str, Any]) -> str | None:
    """
    从样本 map 路径中提取 EMDB ID。

    输入参数:
        - pair: dict[str, Any], 单个 raw pair 条目

    输出:
        - emdb_id: str | None, 形如 EMD-1234; 未匹配时返回 None
    """
    match = re.search(r"emd[_-](\d+)", str(pair.get("map_path", "")), re.IGNORECASE)
    if not match:
        return None
    return f"EMD-{match.group(1)}"


def load_sample_meta(sample_json: Path) -> dict[str, dict[str, str | None]]:
    """
    读取样本清单并构造样本 ID 元信息。

    输入参数:
        - sample_json: Path, protein_110.json 或 nucleic_40.json

    输出:
        - meta: dict[str, dict[str, str | None]], 样本名到 PDB/EMDB ID 的映射
    """
    pairs = read_json(sample_json)
    meta: dict[str, dict[str, str | None]] = {}
    for pair in pairs:
        sample_name = sample_name_from_pair(pair)
        meta[sample_name] = {
            "pdb_id": sample_name.upper(),
            "emdb_id": emdb_id_from_pair(pair),
        }
    return meta


def load_per_sample_rows(path: Path) -> list[dict[str, Any]]:
    """
    读取 DL 或 Phenix 的逐样本评估结果。

    输入参数:
        - path: Path, 固定测试输出根目录

    输出:
        - rows: list[dict[str, Any]], 规范化逐样本行
    """
    per_sample_path = path / "per_sample_best_metrics.json"
    if not per_sample_path.exists():
        return []
    rows = read_json(per_sample_path)
    normalized: list[dict[str, Any]] = []
    for row in rows:
        sample_name = str(row.get("sample_name") or "").lower()
        normalized.append(
            {
                "sample_name": sample_name,
                "status": "ok" if row.get("metrics") is not None and row.get("error") in (None, "") else "failed",
                "error": row.get("error"),
                "metrics": row.get("metrics"),
                "source": str(per_sample_path),
            }
        )
    return normalized


def build_dl_entries(eval_root: Path, dataset_key: str) -> list[dict[str, Any]]:
    """
    构造当前数据集的 DL 模型条目。

    输入参数:
        - eval_root: Path, /home/.../EVAL_OUT
        - dataset_key: str, protein 或 nucleic

    输出:
        - entries: list[dict[str, Any]], DL 条目列表
    """
    dataset_spec = DATASET_SPECS[dataset_key]
    entries: list[dict[str, Any]] = []
    for source_name, display_name in MODEL_SPECS:
        for mode_dir, mode_display in MODE_SPECS:
            leaf = dataset_spec["dl_suffix_template"].format(source=source_name, mode=mode_dir)
            path = eval_root / "infer_out" / leaf
            entries.append(
                {
                    "id": f"{dataset_key}_{display_name}_{mode_display}",
                    "name": f"{display_name}({mode_display})",
                    "kind": "DL",
                    "mode": mode_display,
                    "source_name": source_name,
                    "source_path": str(path),
                    "source_exists": path.exists(),
                    "rows": load_per_sample_rows(path),
                }
            )
    return entries


def build_phenix_entries(eval_root: Path, dataset_key: str) -> list[dict[str, Any]]:
    """
    构造当前数据集的 Phenix 条目。

    输入参数:
        - eval_root: Path, /home/.../EVAL_OUT
        - dataset_key: str, protein 或 nucleic

    输出:
        - entries: list[dict[str, Any]], Phenix 条目列表
    """
    dataset_spec = DATASET_SPECS[dataset_key]
    entries: list[dict[str, Any]] = []
    for mode_dir, mode_display in MODE_SPECS:
        leaf = dataset_spec["phenix_suffix_template"].format(mode=mode_dir)
        path = eval_root / "phenix_" / "infer_out" / "baseline" / leaf
        entries.append(
            {
                "id": f"{dataset_key}_phenix_{mode_display}",
                "name": f"phenix({mode_display})",
                "kind": "non-DL",
                "mode": mode_display,
                "source_name": "phenix_real_space_diff_map",
                "source_path": str(path),
                "source_exists": path.exists(),
                "rows": load_per_sample_rows(path),
            }
        )
    return entries


def load_emap_rows(eval_root: Path, dataset_name: str) -> tuple[str, list[dict[str, Any]]]:
    """
    读取 Emap2lig 的逐样本评估结果。

    输入参数:
        - eval_root: Path, /home/.../EVAL_OUT
        - dataset_name: str, protein_110 或 nucleic_40

    输出:
        - result: tuple[str, list[dict[str, Any]]], 来源路径和逐样本行
    """
    path = eval_root / "emap2lig_find_official" / dataset_name / "reports" / "per_sample_results.json"
    rows = read_json(path)
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["sample_name"] = str(item.get("sample_name", "")).lower()
        item["source"] = str(path)
        normalized.append(item)
    return str(path), normalized


def build_snapshot(eval_root: Path, pocket_plus_root: Path) -> dict[str, Any]:
    """
    从服务器评估产物构造规范化结果快照。

    输入参数:
        - eval_root: Path, EVAL_OUT 根目录
        - pocket_plus_root: Path, Pocket Plus 仓库根目录

    输出:
        - snapshot: dict[str, Any], 可供 collect_bibm_results.py 消费的快照
    """
    snapshot: dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "eval_root": str(eval_root),
        "pocket_plus_root": str(pocket_plus_root),
        "datasets": {},
    }
    utils_root = pocket_plus_root / "src" / "inference" / "utils"
    for dataset_key, dataset_spec in DATASET_SPECS.items():
        sample_json = utils_root / str(dataset_spec["sample_json"])
        emap_source_path, emap_rows = load_emap_rows(eval_root, str(dataset_spec["emap_dataset"]))
        snapshot["datasets"][dataset_key] = {
            "title": dataset_spec["title"],
            "sample_json": str(sample_json),
            "sample_meta": load_sample_meta(sample_json),
            "entries": [
                *build_dl_entries(eval_root, dataset_key),
                *build_phenix_entries(eval_root, dataset_key),
            ],
            "emap_source_path": emap_source_path,
            "emap_rows": emap_rows,
        }
    return snapshot


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    输出:
        - args: argparse.Namespace, 命令行参数集合
    """
    parser = argparse.ArgumentParser(description="创建 BIBM 结果汇总快照。")
    parser.add_argument("--eval-root", required=True, help="EVAL_OUT 根目录。")
    parser.add_argument("--pocket-plus-root", required=True, help="Pocket Plus 仓库根目录。")
    parser.add_argument("--output-json", help="可选输出路径; 省略时写到 stdout。")
    return parser.parse_args()


def main() -> None:
    """
    命令行入口。

    输出:
        - None
    """
    args = parse_args()
    snapshot = build_snapshot(Path(args.eval_root), Path(args.pocket_plus_root))
    text = json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
