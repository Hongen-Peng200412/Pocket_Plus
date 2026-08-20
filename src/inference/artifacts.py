# -*- coding: utf-8 -*-
"""定义 Stage1 V3 推理产物的固定路径和原子发布边界.

主要入口是 :class:`Stage1ArtifactPaths`, :func:`load_stage1_npz`,
:func:`publish_stage1_artifact` 和 :func:`publish_stage1_json`. 本模块不计算
概率, 连通区域, centered 特征或评估指标.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


_ARTIFACT_RELATIVE_PATHS = {
    "probability": Path("probability/probability_map.npz"),
    "F1_blobs": Path("blobs/F1_blobs.npz"),
    "F3_blobs": Path("blobs/F3_blobs.npz"),
    "F1_basic": Path("centered/F1_basic.npz"),
    "F3_centered": Path("centered/F3_centered.npz"),
}


# ================================================================================================


@dataclass(frozen=True)
class Stage1ArtifactPaths:
    """解析一个 producer, 数据划分和 PDB 的全部正式路径.

    字段:
        - output_root: Path, Stage1 V3 产物根目录.
        - stage1_model_name: str, producer 名称, 例如 `unet_c1` 或 `Find_1`.
        - split: str, PDB 数据划分, 例如 `calibration`,`validation` 或 `train`.
        - pdb_id: str, 小写 PDB 标识, 例如 `7pa9`.
    """

    output_root: Path
    stage1_model_name: str
    split: str
    pdb_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "stage1_model_name", str(self.stage1_model_name).strip())
        object.__setattr__(self, "split", str(self.split).strip().lower())
        object.__setattr__(self, "pdb_id", str(self.pdb_id).strip().lower())

    @property
    def pdb_root(self) -> Path:
        """返回当前数据划分和 PDB 的产物目录."""

        return self.output_root / self.stage1_model_name / self.split / self.pdb_id

    def artifact(self, role: str) -> Path:
        """返回五个正式 PDB 角色之一的 NPZ 路径."""

        return self.pdb_root / _ARTIFACT_RELATIVE_PATHS[str(role)]

    def complete(self, role: str) -> Path:
        """返回当前 PDB 角色的原子完成标记路径."""

        return self.pdb_root / "status" / str(role) / "_COMPLETE"

    @property
    def blob_exceed(self) -> Path:
        """返回 F3 完整模式的候选数量超限 JSON 路径."""

        return self.pdb_root / "status" / "F3_centered" / "_BLOB_EXCEED"

    def performance(self, role: str) -> Path:
        """返回当前 PDB 角色的非科学性能计时 JSON 路径."""

        return self.pdb_root / "status" / str(role) / "performance.json"

def load_stage1_npz(
    path: str | Path,
    fields: Sequence[str] | None,
) -> dict[str, np.ndarray]:
    """读取一个 Stage1 NPZ 中调用者明确消费的字段.

    输入参数:
        - path: str | Path, 目标 NPZ 文件.
        - fields: Sequence[str] | None, 需要解压的字段名; 显式 ``None`` 表示读取全部字段.

    返回值:
        - arrays: dict[str, np.ndarray], 只含 ``fields`` 指定的数组, 或在 ``fields=None`` 时包含全部数组; 不允许 object dtype.
    """

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        names = tuple(archive.files) if fields is None else tuple(fields)
        missing = [name for name in names if name not in archive.files]
        if missing:
            raise KeyError(f"{source}: 缺少字段 {missing}.")
        arrays = {name: np.asarray(archive[name]) for name in names}
    object_fields = [name for name, value in arrays.items() if value.dtype.hasobject]
    if object_fields:
        raise TypeError(f"{source}: 不允许 object dtype 字段 {object_fields}.")
    return arrays


def publish_stage1_artifact(
    path: str | Path,
    arrays: Mapping[str, np.ndarray],
    complete_path: str | Path | None,
) -> None:
    """原子发布一个 NPZ, 可在同次事务后建立角色完成标记.

    输入参数:
        - path: str | Path, 最终 NPZ 路径.
        - arrays: Mapping[str, np.ndarray], 字段名到数值数组的映射; 每个字段必须不是 object dtype.
        - complete_path: str | Path | None, 需要发布的 `_COMPLETE` 路径; calibration 中间文件显式传入 None.

    发布边界:
        - 临时 NPZ 位于最终文件同一目录, 写完后直接调用 `os.replace`; 不重复解压大型正式数组.
        - 完成标记只在最终 NPZ 已替换后建立; 进程提前失败时不会产生完成标记.
        - `_COMPLETE` 保存 `output_role: str` 和 `completed_at_utc: str`; 后者为带时区的 ISO 8601 时间.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if complete_path is not None:
        Path(complete_path).unlink(missing_ok=True)
    values = {str(name): np.asarray(value) for name, value in arrays.items()}
    object_fields = [name for name, value in values.items() if value.dtype.hasobject]
    if object_fields:
        raise TypeError(f"{target}: 不允许 object dtype 字段 {object_fields}.")
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp.npz",
        dir=target.parent,
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **values)
        os.replace(temporary, target)
        if complete_path is not None:
            marker = Path(complete_path)
            publish_stage1_json(
                marker,
                {
                    "output_role": marker.parent.name,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
    finally:
        temporary.unlink(missing_ok=True)


def publish_stage1_json(
    path: str | Path,
    payload: Mapping[str, Any],
) -> None:
    """以 UTF-8 JSON 和同目录原子替换发布配置, 状态或评估汇总."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        json.loads(temporary.read_text(encoding="utf-8"))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def publish_stage1_jsonl(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """以 UTF-8 JSONL 和同目录原子替换发布逐 PDB 汇总."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        lines = [
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True)
            for row in rows
        ]
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
