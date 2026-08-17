# -*- coding: utf-8 -*-
"""解析 Stage1 V3 的 80³ BOX 请求和冻结验证选择。

主要入口是 :class:`Stage1TrainingRequestSet`、:func:`load_validation_selection`
与 :func:`build_request_source`。本模块只解释 V3 ``box_pool`` 中的 PDB 身份、
请求角色和完整图 ZYX 起点；密度、原子与监督数组由 ``stage1_dataset.py`` 读取。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


STAGE1_BOX_SHAPE_ZYX = (80, 80, 80)
BOX_POOL_MANIFEST_FILENAME = "manifest.json"
_VALID_ROLES = {"center", "bias", "context", "sliding", "centered"}


def resolve_stage1_start(
    requested_start_zyx: Sequence[int | float],
    full_shape_zyx: Sequence[int],
    box_shape_zyx: Sequence[int] = STAGE1_BOX_SHAPE_ZYX,
) -> tuple[int, int, int]:
    """把请求起点限制到完整图内，使 BOX 不需要补零。"""

    requested = np.rint(np.asarray(requested_start_zyx, dtype=np.float64)).astype(np.int64)
    full_shape = np.asarray(full_shape_zyx, dtype=np.int64)
    box_shape = np.asarray(box_shape_zyx, dtype=np.int64)
    if requested.shape != (3,) or full_shape.shape != (3,) or box_shape.shape != (3,):
        raise ValueError("BOX 起点、完整图形状和 BOX 形状都必须包含三个 ZYX 分量。")
    if np.any(full_shape < box_shape):
        raise ValueError("完整图三轴必须不小于 Stage1 BOX。")
    return tuple(int(value) for value in np.clip(requested, 0, full_shape - box_shape))


def centered_start_from_sparse_mask(
    sparse_voxel_zyx: np.ndarray,
    full_shape_zyx: Sequence[int],
    box_shape_zyx: Sequence[int] = STAGE1_BOX_SHAPE_ZYX,
) -> tuple[int, int, int]:
    """以 occurrence 非空体素的几何中心计算合法 BOX 起点。"""

    sparse = np.asarray(sparse_voxel_zyx, dtype=np.int64)
    if sparse.ndim != 2 or sparse.shape[1] != 3 or sparse.shape[0] == 0:
        raise ValueError("occurrence 稀疏体素必须是非空 (K,3) ZYX 数组。")
    return centered_start_from_centroid_zyx(
        sparse.astype(np.float64).mean(axis=0),
        full_shape_zyx,
        box_shape_zyx,
    )


def centered_start_from_centroid_zyx(
    centroid_zyx: Sequence[int | float],
    full_shape_zyx: Sequence[int],
    box_shape_zyx: Sequence[int] = STAGE1_BOX_SHAPE_ZYX,
) -> tuple[int, int, int]:
    """把 voxel-index 质心放到 BOX 中心，并返回完整图 ZYX corner 起点。"""

    centroid = np.asarray(centroid_zyx, dtype=np.float64)
    box_shape = np.asarray(box_shape_zyx, dtype=np.int64)
    requested = centroid + 0.5 - box_shape.astype(np.float64) / 2.0
    return resolve_stage1_start(requested, full_shape_zyx, box_shape_zyx)


def _string_array(values: np.ndarray) -> list[str]:
    """把 NPZ bytes 或 Unicode 字段转换为小写字符串列表。"""

    result: list[str] = []
    for value in np.asarray(values).reshape(-1).tolist():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        result.append(str(value).strip().lower())
    return result


def _load_manifest_pool_paths(
    box_pool_root: str | Path,
    split_name: str,
) -> tuple[tuple[str, Path], ...]:
    """读取完成发布的 V3 manifest，并返回指定数据划分的 PDB NPZ 清单。

    ``box_pool_root`` 必须同时包含 ``_COMPLETE`` 与 ``manifest.json``。
    ``manifest.json:splits[split_name]`` 中每项提供小写化后的 ``pdb_id`` 和
    相对于根目录的 ``path``；返回值保持 manifest 顺序，不去重。绝对路径、
    含 ``..`` 的路径或空数据划分会被拒绝。
    """

    root = Path(box_pool_root)
    if not (root / "_COMPLETE").is_file():
        raise FileNotFoundError(f"BOX pool 尚未完成发布：{root}。")
    manifest_path = root / BOX_POOL_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_rows = manifest["splits"][str(split_name).lower()]
    entries: list[tuple[str, Path]] = []
    for record in split_rows:
        pdb_id = str(record["pdb_id"]).strip().lower()
        relative_path = Path(str(record["path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"BOX pool path 必须位于 manifest 根目录内：{relative_path}。")
        entries.append((pdb_id, root / relative_path))
    if not entries:
        raise ValueError(f"{manifest_path}: splits.{split_name} 不能为空。")
    return tuple(entries)


@dataclass(frozen=True)
class _PdbPool:
    """保存一个 PDB 的 occurrence 与完整图 ZYX BOX 起点。

    ``occurrence_id`` 为 int32 ``(O,)``；``center_start_zyx`` 为 int32
    ``(O,3)``；``bias_start_zyx`` 为 int32 ``(O,30,3)``；
    ``context_start_zyx`` 为 int32 ``(C,3)``。前三个数组按 occurrence 位置
    对齐，context 是 PDB 级候选池。
    """

    pdb_id: str
    occurrence_id: np.ndarray
    center_start_zyx: np.ndarray
    bias_start_zyx: np.ndarray
    context_start_zyx: np.ndarray


def _load_pdb_pool(path: Path, expected_pdb_id: str) -> _PdbPool:
    """读取一个 V3 PDB pool，并核对 PDB 身份和四个数组。

    文件字段为字符串标量 ``pdb_id``、int32 ``occurrence_id (O,)``、int32
    ``center_start_zyx (O,3)``、int32 ``bias_start_zyx (O,30,3)`` 与 int32
    ``context_start_zyx (C,3)``。后三个字段是完整图内 80³ BOX 的零基 ZYX
    corner index；前三个数组按 occurrence 位置对齐。
    """

    with np.load(path, allow_pickle=False) as data:
        pdb_id = _string_array(data["pdb_id"])[0]
        occurrence_id = np.asarray(data["occurrence_id"], dtype=np.int32)
        center_start_zyx = np.asarray(data["center_start_zyx"], dtype=np.int32)
        bias_start_zyx = np.asarray(data["bias_start_zyx"], dtype=np.int32)
        context_start_zyx = np.asarray(data["context_start_zyx"], dtype=np.int32)
    if pdb_id != expected_pdb_id:
        raise ValueError(f"{path}: PDB 身份 {pdb_id!r} 与 manifest 不一致。")
    occurrence_count = int(occurrence_id.shape[0])
    if (
        center_start_zyx.shape != (occurrence_count, 3)
        or bias_start_zyx.shape != (occurrence_count, 30, 3)
        or context_start_zyx.ndim != 2
        or context_start_zyx.shape[1] != 3
    ):
        raise ValueError(f"{path}: V3 BOX 起点数组形状不符合契约。")
    return _PdbPool(
        pdb_id,
        occurrence_id,
        center_start_zyx,
        bias_start_zyx,
        context_start_zyx,
    )


@dataclass(frozen=True)
class ResolvedStage1Crop:
    """保存一个已经解析到完整图位置的 Stage1 BOX 请求。

    ``pdb_id`` 是小写 PDB 身份；``box_start_zyx`` 是完整图内 80³ BOX 的
    零基 ZYX corner index；``require_targets`` 决定 Dataset 是否构造监督；
    ``role`` 是 center、bias、context、sliding 或 centered；
    ``occurrence_id`` 是可选真实配体身份；``candidate_index`` 是可选 bias 或
    context 候选位置。
    """

    pdb_id: str
    box_start_zyx: tuple[int, int, int]
    require_targets: bool
    role: str
    occurrence_id: int | None = None
    candidate_index: int | None = None

    def __post_init__(self) -> None:
        pdb_id = str(self.pdb_id).strip().lower()
        start = tuple(int(value) for value in self.box_start_zyx)
        role = str(self.role).strip().lower()
        if not pdb_id or len(start) != 3 or role not in _VALID_ROLES:
            raise ValueError("Stage1 BOX 请求缺少合法 PDB 身份、ZYX 起点或角色。")
        object.__setattr__(self, "pdb_id", pdb_id)
        object.__setattr__(self, "box_start_zyx", start)
        object.__setattr__(self, "role", role)


class Stage1TrainingRequestSet:
    """按训练周期从 V3 train pool 生成全量 0:5:5 BOX 请求。

    ``pool_directory`` 是 ``stage1_preparation_box_pool_3/train``，其父目录
    提供已完成标记、manifest 与 config；``seed`` 与 epoch 共同固定选择结果。
    每个 epoch、每个 PDB 至多无放回选择 50 个 occurrence；每个 occurrence
    无放回选择 5 个 bias 候选，并从 PDB 级 context 池选择 5 个候选，候选
    少于 5 个时有放回采样。请求顺序为 manifest PDB 顺序、随机 occurrence
    顺序、5 个 bias、5 个 context；不生成 center 请求。``set_epoch`` 在 epoch
    变化时重建请求，要求 DataLoader worker 不常驻。
    """

    def __init__(self, pool_directory: str | Path, seed: int) -> None:
        pool_directory = Path(pool_directory)
        self._pools = tuple(
            _load_pdb_pool(path, expected_pdb_id=pdb_id)
            for pdb_id, path in _load_manifest_pool_paths(pool_directory.parent, pool_directory.name)
        )
        config = json.loads((pool_directory.parent / "config.json").read_text(encoding="utf-8"))
        self.entry_ratio = {
            role: int(config["entry_ratio"][role])
            for role in ("center", "bias", "context")
        }
        if self.entry_ratio != {"center": 0, "bias": 5, "context": 5}:
            raise ValueError("Stage1 V3 正式请求比例必须为 center:bias:context = 0:5:5。")
        self.seed = int(seed)
        self.epoch = -1
        self.requests: tuple[ResolvedStage1Crop, ...] = ()
        self.set_epoch(0)

    def _build_epoch_requests(self, epoch: int) -> tuple[ResolvedStage1Crop, ...]:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(epoch)]))
        requests: list[ResolvedStage1Crop] = []
        for pool in self._pools:
            selected_rows = rng.choice(
                int(pool.occurrence_id.shape[0]),
                size=min(50, int(pool.occurrence_id.shape[0])),
                replace=False,
            )
            for occurrence_row in selected_rows.tolist():
                occurrence_id = int(pool.occurrence_id[occurrence_row])
                bias_indices = rng.choice(30, size=5, replace=False)
                for candidate_index in bias_indices.tolist():
                    requests.append(
                        ResolvedStage1Crop(
                            pool.pdb_id,
                            tuple(pool.bias_start_zyx[occurrence_row, candidate_index]),
                            True,
                            "bias",
                            occurrence_id,
                            int(candidate_index),
                        )
                    )
                context_indices = rng.choice(
                    int(pool.context_start_zyx.shape[0]),
                    size=5,
                    replace=int(pool.context_start_zyx.shape[0]) < 5,
                )
                for candidate_index in context_indices.tolist():
                    requests.append(
                        ResolvedStage1Crop(
                            pool.pdb_id,
                            tuple(pool.context_start_zyx[candidate_index]),
                            True,
                            "context",
                            occurrence_id,
                            int(candidate_index),
                        )
                    )
        return tuple(requests)

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch != self.epoch:
            self.requests = self._build_epoch_requests(epoch)
            self.epoch = epoch

    def __len__(self) -> int:
        return len(self.requests)

    def __getitem__(self, index: int) -> ResolvedStage1Crop:
        return self.requests[index]


def load_validation_selection(
    selection_path: str | Path,
    box_pool_root: str | Path,
) -> list[ResolvedStage1Crop]:
    """把 V3 ``validation_selection.npz`` 的索引展开为固定 BOX 请求。

    selection 读取定宽 bytes 或 Unicode ``validation_pdb_id (P,)``；int32
    ``center_pdb_index/center_occurrence_id (N_center,)``；int32
    ``bias_pdb_index/bias_occurrence_id (N_bias,)``；int16
    ``bias_candidate_index (N_bias,)``；以及 int32
    ``context_pdb_index/context_candidate_index (N_context,)``。PDB index 指向
    身份表；occurrence_id 在对应 pool 的 ``occurrence_id (O,)`` 中按值定位；
    candidate index 再索引 bias 的 30 候选轴或 context 的 ``(C,3)`` 数组。
    返回顺序固定为全部 center、全部 bias、全部 context，元素均为
    ``ResolvedStage1Crop``，不会在读取时重新随机选择。
    """

    source = Path(selection_path)
    manifest_paths = dict(_load_manifest_pool_paths(box_pool_root, "validation"))
    with np.load(source, allow_pickle=False) as data:
        pdb_ids = _string_array(data["validation_pdb_id"])
        arrays = {
            field_name: np.asarray(data[field_name])
            for field_name in (
                "center_pdb_index",
                "center_occurrence_id",
                "bias_pdb_index",
                "bias_occurrence_id",
                "bias_candidate_index",
                "context_pdb_index",
                "context_candidate_index",
            )
        }
    pools = {
        pdb_id: _load_pdb_pool(manifest_paths[pdb_id], expected_pdb_id=pdb_id)
        for pdb_id in pdb_ids
    }
    occurrence_rows = {
        pdb_id: {
            int(occurrence_id): row_index
            for row_index, occurrence_id in enumerate(pool.occurrence_id.tolist())
        }
        for pdb_id, pool in pools.items()
    }

    requests: list[ResolvedStage1Crop] = []
    for pdb_index, occurrence_id in zip(
        arrays["center_pdb_index"], arrays["center_occurrence_id"]
    ):
        pdb_id = pdb_ids[int(pdb_index)]
        row_index = occurrence_rows[pdb_id][int(occurrence_id)]
        requests.append(
            ResolvedStage1Crop(
                pdb_id,
                tuple(pools[pdb_id].center_start_zyx[row_index]),
                True,
                "center",
                int(occurrence_id),
            )
        )
    for pdb_index, occurrence_id, candidate_index in zip(
        arrays["bias_pdb_index"],
        arrays["bias_occurrence_id"],
        arrays["bias_candidate_index"],
    ):
        pdb_id = pdb_ids[int(pdb_index)]
        row_index = occurrence_rows[pdb_id][int(occurrence_id)]
        candidate_index = int(candidate_index)
        requests.append(
            ResolvedStage1Crop(
                pdb_id,
                tuple(pools[pdb_id].bias_start_zyx[row_index, candidate_index]),
                True,
                "bias",
                int(occurrence_id),
                candidate_index,
            )
        )
    for pdb_index, candidate_index in zip(
        arrays["context_pdb_index"], arrays["context_candidate_index"]
    ):
        pdb_id = pdb_ids[int(pdb_index)]
        candidate_index = int(candidate_index)
        requests.append(
            ResolvedStage1Crop(
                pdb_id,
                tuple(pools[pdb_id].context_start_zyx[candidate_index]),
                True,
                "context",
                None,
                candidate_index,
            )
        )
    return requests


def load_split_pdb_ids(path: str | Path) -> tuple[str, ...]:
    """读取 V3 split JSON，并按首次出现顺序返回去重的小写 PDB 身份。

    split JSON 以候选配体记录为单位，同一 PDB 可以连续或分散出现多次；推理
    清单只需要 PDB 身份，因此这里保留每个身份的第一次出现。
    """

    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(
        dict.fromkeys(str(record["pdb_id"]).strip().lower() for record in records)
    )


def build_request_source(
    split_file: str | Path,
    mode: str,
    box_pool_root: str | Path | None,
    seed: int,
) -> Stage1TrainingRequestSet | list[ResolvedStage1Crop]:
    """根据请求来源创建 V3 动态训练请求集或固定验证请求列表。

    ``split_file`` 是目录时只允许 ``mode=train``，并用 ``seed`` 创建
    ``Stage1TrainingRequestSet``；文件名为 ``validation_selection.npz`` 时要求
    ``box_pool_root`` 非空，并按该根目录的 validation manifest 展开固定请求。
    其他路径形态不做兼容回退。
    """

    source = Path(split_file)
    if source.is_dir():
        if str(mode).lower() != "train":
            raise ValueError("只有 train 模式允许直接读取 BOX pool 目录。")
        return Stage1TrainingRequestSet(source, seed=seed)
    if source.name == "validation_selection.npz" and box_pool_root is not None:
        return load_validation_selection(source, box_pool_root)
    raise ValueError("Stage1 请求来源必须是 V3 train 目录或 validation_selection.npz。")
