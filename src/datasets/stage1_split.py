# -*- coding: utf-8 -*-
"""从 Stage G keep-list 冻结 AdaLigand Stage1 的 PDB 级主划分。

阅读主线:
    ``keep_list.jsonl`` 先按 ``pdb_id`` 聚合，随后用稳定哈希排名冻结
    validation 300、calibration 100、train 75% 和 held-out pool。PDB 是不可跨
    split 的身份边界；同一 PDB 的全部 PDB-EMDB pair 始终随该身份一起移动。

输出只承担“谁属于哪个 split”的身份契约。训练 BOX 的 center/bias/context
几何位置由 ``stage1_box_pool.py`` 在 split 冻结后另行生成。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


_SPLIT_SEED = 3407
_TRAIN_FRACTION = 0.75
_VALIDATION_PDB_COUNT = 300
_CALIBRATION_PDB_COUNT = 100
_MIN_GRID_SHAPE_ZYX = (80, 80, 80)


def _atomic_write_text(path: Path, content: str) -> None:
    """
    在目标同目录原子发布 UTF-8 文本或完成标记。

    输入参数:
        - path: Path, 正式输出路径
        - content: str, 要写入的完整文本；空字符串用于完成标记

    输出:
        - None: 临时文件完整写入后原子替换 `path`
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        temp_path.write_text(content, encoding="utf-8")
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def _read_keep_list(path: Path) -> dict[str, list[dict[str, Any]]]:
    """
    读取 Stage G JSONL，并按规范化后的 PDB identity 聚合原始行。

    输入参数:
        - path: Path, Stage G `keep_list.jsonl` 路径

    输出:
        - groups: dict[str,list[dict[str,Any]]], PDB identity 到原始 occurrence 行列表的映射；PDB 内保持文件行序
    """

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} 必须是一条 JSON object。")
            pdb_id = str(row.get("pdb_id", "")).strip().lower()
            if not pdb_id:
                raise ValueError(f"{path}:{line_number} 缺少有效 pdb_id。")
            groups[pdb_id].append(row)
    if not groups:
        raise ValueError(f"Stage G keep-list 为空: {path}。")
    return dict(groups)


def _stable_rank(seed: int, purpose: str, pdb_id: str) -> bytes:
    """
    生成与输入遍历顺序无关的确定性 PDB 排名键。

    输入参数:
        - seed: int, 冻结 split 的基准 seed
        - purpose: str, 区分 `eval` 与 `train` 排名流的用途字符串
        - pdb_id: str, 当前 PDB identity

    输出:
        - digest: bytes, (32,), `sha256(seed|purpose|pdb_id)` 排名键
    """

    payload = f"{int(seed)}|{purpose}|{pdb_id}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def _read_exp_shape(data_root: Path, pdb_id: str) -> tuple[int, int, int]:
    """
    只读取 E1 冻结形状元数据，不实体化完整密度数组。

    输入参数:
        - data_root: Path, A-G 正式数据根目录
        - pdb_id: str, 当前 PDB identity

    输出:
        - shape_zyx: tuple[int,int,int], (3,), `canonical_shape_zyx` 的三个正整数
    """

    exp_path = data_root / "density" / pdb_id / "exp.npz"
    with np.load(exp_path, allow_pickle=False) as data:
        if "canonical_shape_zyx" not in data:
            raise KeyError(f"{exp_path} 缺少 canonical_shape_zyx。")
        shape = np.asarray(data["canonical_shape_zyx"], dtype=np.int64)
    if shape.shape != (3,) or np.any(shape <= 0):
        raise ValueError(f"{exp_path}: canonical_shape_zyx 必须为三个正整数。")
    return tuple(int(value) for value in shape.tolist())


def _flatten_groups(groups: dict[str, list[dict[str, Any]]], pdb_ids: set[str]) -> list[dict[str, Any]]:
    """
    按 PDB 稳定排序展开分组，同时保持每组 Stage G 原始行顺序。

    输入参数:
        - groups: dict[str,list[dict[str,Any]]], PDB 到原始 occurrence 行的映射
        - pdb_ids: set[str], 要展开的 PDB identity 集合

    输出:
        - rows: list[dict[str,Any]], 先按 PDB identity 排序、再按 PDB 内原始行序展开的记录
    """

    return [row for pdb_id in sorted(pdb_ids) for row in groups[pdb_id]]


def freeze_stage1_splits(
    keep_list_path: str | Path,
    data_root: str | Path,
    output_root: str | Path,
    seed: int = _SPLIT_SEED,
    validation_pdb_count: int = _VALIDATION_PDB_COUNT,
    calibration_pdb_count: int = _CALIBRATION_PDB_COUNT,
) -> dict[str, Any]:
    """一次冻结 PDB 不跨界的 train/validation/calibration/held-out 主划分。

    输入参数:
        - keep_list_path: str | Path, Stage G `keep_list.jsonl`；同一 PDB 可含多条 occurrence 行
        - data_root: str | Path, A-G 正式数据根目录，用于读取 `density/{pdb_id}/exp.npz` 的
            ``canonical_shape_zyx``。
        - output_root: str | Path, 本次 split run 的独立输出目录
        - seed: int, 冻结排名 seed；正式值为 3407
        - validation_pdb_count: int, validation 的 PDB 数；正式值为 300
        - calibration_pdb_count: int, calibration 的 PDB 数；正式值为 100

    输出:
        - summary: dict[str,Any], 与 `summary.json` 相同的 PDB/row 计数，包含:
            - `seed`: int, 实际排名 seed
            - `source_pdb_count`: int, keep-list 中唯一 PDB 数
            - `source_row_count`: int, keep-list 原始行数
            - `validation_calibration_shape_checked_pdb_count`: int, eval 选择时检查过 shape 的 PDB 数
            - `short_map_encountered_while_selecting_eval_count`: int, eval 扫描中遇到的短图数
            - `splits`: dict[str,dict[str,int]], 各 split 的 `pdb_count` 与 `row_count`

        train PDB 数严格为
        ``floor(0.75*N)``；validation 与 calibration 只从三轴均不小于 80 的
        PDB 选择。held-out 不做尺寸检查或去冗余。
    """

    keep_path = Path(keep_list_path)
    data_path = Path(data_root)
    root = Path(output_root)
    validation_count = int(validation_pdb_count)
    calibration_count = int(calibration_pdb_count)
    if validation_count < 0 or calibration_count < 0:
        raise ValueError("validation_pdb_count 与 calibration_pdb_count 不能为负数。")

    # dict[str,list[dict]], PDB -> Stage G occurrence 行；同一 PDB 的全部行必须留在同一 split。
    groups = _read_keep_list(keep_path)
    all_pdb_ids = set(groups)
    total_pdb_count = len(all_pdb_ids)
    train_count = math.floor(_TRAIN_FRACTION * total_pdb_count)
    required_eval_count = validation_count + calibration_count
    if train_count + required_eval_count > total_pdb_count:
        raise ValueError(
            "PDB 总数不足以同时满足 train=floor(0.75*N)、validation 和 calibration 固定计数："
            f"N={total_pdb_count}, train={train_count}, validation={validation_count}, "
            f"calibration={calibration_count}。"
        )

    # list[str], (N_pdb,), 由 seed/purpose/PDB 哈希确定的稳定评估候选顺序。
    eval_order = sorted(all_pdb_ids, key=lambda pdb_id: _stable_rank(seed, "eval", pdb_id))
    # list[str], (N_val+N_cal,), 仅收集三轴均可容纳 80³ BOX 的 PDB。
    selected_eval_ids: list[str] = []
    checked_eval_count = 0
    encountered_short_count = 0
    for pdb_id in eval_order:
        checked_eval_count += 1
        shape_zyx = _read_exp_shape(data_path, pdb_id)
        if not all(
            axis >= minimum
            for axis, minimum in zip(shape_zyx, _MIN_GRID_SHAPE_ZYX)
        ):
            encountered_short_count += 1
            continue
        selected_eval_ids.append(pdb_id)
        if len(selected_eval_ids) == required_eval_count:
            break
    if len(selected_eval_ids) < required_eval_count:
        raise ValueError(
            "三轴均不小于 80 的 PDB 不足以冻结 validation/calibration："
            f"eligible={len(selected_eval_ids)}, required={required_eval_count}。"
        )

    # set[str], 固定大小的 validation/calibration PDB 集合；两者互斥。
    validation_ids = set(selected_eval_ids[:validation_count])
    calibration_ids = set(selected_eval_ids[validation_count:required_eval_count])
    remaining_ids = all_pdb_ids.difference(validation_ids, calibration_ids)
    train_order = sorted(remaining_ids, key=lambda pdb_id: _stable_rank(seed, "train", pdb_id))
    train_ids = set(train_order[:train_count])
    held_out_ids = remaining_ids.difference(train_ids)
    split_ids = {
        "train": train_ids,
        "validation": validation_ids,
        "calibration": calibration_ids,
        "held_out_pool": held_out_ids,
    }

    root.mkdir(parents=True, exist_ok=True)
    (root / "_COMPLETE").unlink(missing_ok=True)
    # dict[str,list[dict]], split -> 原始 occurrence 行；PDB 排序稳定，PDB 内行序保持不变。
    split_rows = {name: _flatten_groups(groups, pdb_ids) for name, pdb_ids in split_ids.items()}
    for split_name, rows in split_rows.items():
        _atomic_write_text(
            root / f"{split_name}.json",
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        )

    source_sha256 = hashlib.sha256(keep_path.read_bytes()).hexdigest()
    config = {
        "schema_version": 1,
        "seed": int(seed),
        "train_fraction": _TRAIN_FRACTION,
        "validation_pdb_count": validation_count,
        "calibration_pdb_count": calibration_count,
        "minimum_validation_calibration_shape_zyx": list(_MIN_GRID_SHAPE_ZYX),
        "assignment": "sha256(seed|purpose|pdb_id) ascending",
        "keep_list_path": str(keep_path),
        "keep_list_sha256": source_sha256,
        "held_out_deduplication": False,
    }
    summary: dict[str, Any] = {
        "seed": int(seed),
        "source_pdb_count": total_pdb_count,
        "source_row_count": sum(len(rows) for rows in groups.values()),
        "validation_calibration_shape_checked_pdb_count": checked_eval_count,
        "short_map_encountered_while_selecting_eval_count": encountered_short_count,
        "splits": {
            name: {"pdb_count": len(split_ids[name]), "row_count": len(split_rows[name])}
            for name in split_ids
        },
    }
    _atomic_write_text(root / "config.json", json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    _atomic_write_text(root / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    _atomic_write_text(root / "_COMPLETE", "")
    return summary


def _main() -> None:
    """
    解析 CLI 并冻结 Stage1 主 split。

    输出:
        - None: 写出 split 产物，并把 summary 以 JSON 打印到标准输出
    """

    parser = argparse.ArgumentParser(description="从 Stage G keep-list 冻结 AdaLigand Stage1 主 split。")
    parser.add_argument("--keep-list", required=True, help="Stage G keep_list.jsonl。")
    parser.add_argument("--data-root", required=True, help="A—G 正式数据根目录。")
    parser.add_argument("--output-root", required=True, help="本次 stage1_preparation/split run 目录。")
    parser.add_argument("--seed", type=int, default=_SPLIT_SEED, help="PDB 分组冻结 seed。")
    arguments = parser.parse_args()
    summary = freeze_stage1_splits(
        keep_list_path=arguments.keep_list,
        data_root=arguments.data_root,
        output_root=arguments.output_root,
        seed=arguments.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
