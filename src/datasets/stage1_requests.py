# -*- coding: utf-8 -*-
"""AdaLigand Stage1 的 80³ 请求对象与冻结请求源。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


STAGE1_BOX_SHAPE_ZYX = (80, 80, 80)
BOX_POOL_MANIFEST_FILENAME = "manifest.json"
_VALID_ROLES = {"center", "bias", "context", "sliding", "centered"}


@dataclass(frozen=True)
class ResolvedStage1Crop:
    """表示一个已完成边界解析的 Stage1 80³ 裁剪请求。

    参数:
        pdb_id: 当前整图身份，内部统一为小写。
        box_start_zyx: 合法的 ZYX 整数起点，形状 ``(3,)``。
        require_targets: 是否构造训练或验证监督。
        role: 请求来源，取值为 center/bias/context/sliding/centered。
        occurrence_id: 上游 occurrence ``candidate_id``；context/sliding 可为 ``None``。
        candidate_index: bias/context 候选在对应冻结池中的下标；不适用时为 ``None``。
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
        if not pdb_id:
            raise ValueError("ResolvedStage1Crop.pdb_id 不能为空。")
        if len(start) != 3:
            raise ValueError("ResolvedStage1Crop.box_start_zyx 必须恰有三个 ZYX 分量。")
        if role not in _VALID_ROLES:
            raise ValueError(f"未知 Stage1 请求 role: {role!r}。")
        object.__setattr__(self, "pdb_id", pdb_id)
        object.__setattr__(self, "box_start_zyx", start)
        object.__setattr__(self, "role", role)


def resolve_stage1_start(
    requested_start_zyx: Sequence[int | float],
    full_shape_zyx: Sequence[int],
    box_shape_zyx: Sequence[int] = STAGE1_BOX_SHAPE_ZYX,
) -> tuple[int, int, int]:
    """把请求起点逐轴 clamp 为完整、无补零的合法 BOX 起点。

    参数:
        requested_start_zyx: 请求起点，形状 ``(3,)``；浮点值按最近整数取整。
        full_shape_zyx: 完整密度图形状，形状 ``(3,)``。
        box_shape_zyx: BOX 形状，Stage1 正式值为 ``(80,80,80)``。

    返回:
        合法 ZYX 起点。若完整图任一轴短于 BOX，则直接报错。
    """

    requested = np.rint(np.asarray(requested_start_zyx, dtype=np.float64)).astype(np.int64)
    full_shape = np.asarray(full_shape_zyx, dtype=np.int64)
    box_shape = np.asarray(box_shape_zyx, dtype=np.int64)
    if requested.shape != (3,) or full_shape.shape != (3,) or box_shape.shape != (3,):
        raise ValueError("requested_start_zyx、full_shape_zyx 与 box_shape_zyx 都必须为 (3,)。")
    if np.any(box_shape <= 0):
        raise ValueError(f"box_shape_zyx 必须逐轴为正，实际为 {box_shape.tolist()}。")
    if np.any(full_shape < box_shape):
        raise ValueError(
            "完整图三轴必须不小于 Stage1 BOX："
            f"full_shape_zyx={full_shape.tolist()}, box_shape_zyx={box_shape.tolist()}。"
        )
    resolved = np.clip(requested, 0, full_shape - box_shape)
    return tuple(int(value) for value in resolved.tolist())


def centered_start_from_sparse_mask(
    sparse_voxel_zyx: np.ndarray,
    full_shape_zyx: Sequence[int],
    box_shape_zyx: Sequence[int] = STAGE1_BOX_SHAPE_ZYX,
) -> tuple[int, int, int]:
    """从 occurrence 稀疏体素集合得到唯一的居中起点。

    稀疏坐标表示体素下标，故其几何中心位于 ``index+0.5``；BOX 的物理中心
    位于 ``start+box_shape/2``。两者相减后对整数起点执行最近整数取整，再调用
    :func:`resolve_stage1_start` 处理边界。
    """

    sparse = np.asarray(sparse_voxel_zyx, dtype=np.int64)
    if sparse.ndim != 2 or sparse.shape[1] != 3 or sparse.shape[0] == 0:
        raise ValueError("sparse_voxel_zyx 必须为非空 (K,3) ZYX 整数数组。")
    return centered_start_from_centroid_zyx(
        centroid_zyx=sparse.astype(np.float64).mean(axis=0),
        full_shape_zyx=full_shape_zyx,
        box_shape_zyx=box_shape_zyx,
    )


def centered_start_from_centroid_zyx(
    centroid_zyx: Sequence[int | float],
    full_shape_zyx: Sequence[int],
    box_shape_zyx: Sequence[int] = STAGE1_BOX_SHAPE_ZYX,
) -> tuple[int, int, int]:
    """把 voxel-index 语义的 ZYX 质心转换为唯一合法居中起点。

    参数:
        centroid_zyx: 由体素下标求得的 ZYX 质心，形状 ``(3,)``。它不是 corner
            坐标，因此先逐轴加 ``0.5`` 得到体素几何中心。
        full_shape_zyx: 完整密度图的 ZYX 形状。
        box_shape_zyx: 输出 BOX 的 ZYX 形状，Stage1 正式值为 ``80³``。

    返回:
        ``tuple[int,int,int]`` 合法起点。请求起点按
        ``round(centroid_zyx + 0.5 - box_shape_zyx/2)`` 计算，再交给统一
        :func:`resolve_stage1_start` 做边界 clamp。

    该 helper 同时供 occurrence sparse mask、forest blob 与 centered 推理调用；
    调用方不得再各自实现另一套 ``centroid -> start`` 公式。
    """

    centroid = np.asarray(centroid_zyx, dtype=np.float64)
    box_shape = np.asarray(box_shape_zyx, dtype=np.int64)
    if centroid.shape != (3,):
        raise ValueError("centroid_zyx 必须为 (3,) voxel-index 质心。")
    if not np.isfinite(centroid).all():
        raise ValueError("centroid_zyx 不能包含 NaN/Inf。")
    requested = centroid + 0.5 - box_shape.astype(np.float64) / 2.0
    return resolve_stage1_start(requested, full_shape_zyx, box_shape_zyx)


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    """读取 JSON 列表/``requests`` 对象或 JSONL 请求表。"""

    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                value = json.loads(text)
                if not isinstance(value, dict):
                    raise TypeError(f"{path}:{line_number} 必须是一条 JSON object。")
                rows.append(value)
        return rows

    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("requests")
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise TypeError(f"{path} 必须是 JSON object 列表或包含 requests 列表的 object。")
    return list(value)


def _requests_from_rows(rows: Iterable[dict[str, Any]], default_targets: bool) -> list[ResolvedStage1Crop]:
    """把普通行记录转成不可变请求对象。"""

    requests: list[ResolvedStage1Crop] = []
    for row in rows:
        if "pdb_id" not in row or "box_start_zyx" not in row:
            raise KeyError("每条 Stage1 请求必须包含 pdb_id 与 box_start_zyx。")
        requests.append(
            ResolvedStage1Crop(
                pdb_id=str(row["pdb_id"]),
                box_start_zyx=tuple(row["box_start_zyx"]),
                require_targets=bool(row.get("require_targets", default_targets)),
                role=str(row.get("role", "centered")),
                occurrence_id=None if row.get("occurrence_id") is None else int(row["occurrence_id"]),
                candidate_index=None if row.get("candidate_index") is None else int(row["candidate_index"]),
            )
        )
    return requests


def _string_array(values: np.ndarray) -> list[str]:
    """把 fixed-width bytes/unicode NPZ 字段规范化为小写字符串。"""

    result: list[str] = []
    for value in np.asarray(values).reshape(-1).tolist():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        result.append(str(value).strip().lower())
    return result


def load_flat_requests(path: str | Path, default_targets: bool) -> list[ResolvedStage1Crop]:
    """读取不依赖 BOX pool 的冻结请求文件。"""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return _requests_from_rows(_read_json_rows(source), default_targets)
    if suffix != ".npz":
        raise ValueError(f"Stage1 请求文件仅支持 .json/.jsonl/.npz，实际为 {source}。")
    with np.load(source, allow_pickle=False) as data:
        if "pdb_id" not in data or "box_start_zyx" not in data:
            raise KeyError(f"{source} 缺少 pdb_id 或 box_start_zyx。")
        pdb_ids = _string_array(data["pdb_id"])
        starts = np.asarray(data["box_start_zyx"], dtype=np.int64)
        if starts.shape != (len(pdb_ids), 3):
            raise ValueError(f"{source}: box_start_zyx 形状必须为 ({len(pdb_ids)},3)。")
        roles = _string_array(data["role"]) if "role" in data else ["centered"] * len(pdb_ids)
        occurrence = np.asarray(data["occurrence_id"], dtype=np.int64) if "occurrence_id" in data else None
        candidate = np.asarray(data["candidate_index"], dtype=np.int64) if "candidate_index" in data else None
        targets = np.asarray(data["require_targets"], dtype=bool) if "require_targets" in data else None
    rows = []
    for index, pdb_id in enumerate(pdb_ids):
        rows.append(
            {
                "pdb_id": pdb_id,
                "box_start_zyx": starts[index].tolist(),
                "role": roles[index],
                "occurrence_id": None if occurrence is None else int(occurrence[index]),
                "candidate_index": None if candidate is None else int(candidate[index]),
                "require_targets": default_targets if targets is None else bool(targets[index]),
            }
        )
    return _requests_from_rows(rows, default_targets)


@dataclass(frozen=True)
class _PdbPool:
    """内存中的单 PDB 训练 BOX 池索引。"""

    pdb_id: str
    occurrence_id: np.ndarray
    center_start_zyx: np.ndarray
    bias_start_zyx: np.ndarray
    context_start_zyx: np.ndarray


def _load_pdb_pool(path: Path, expected_pdb_id: str | None = None) -> _PdbPool:
    """读取并严格校验一份 ``box_pool/{split}/{pdb_id}.npz``。"""

    with np.load(path, allow_pickle=False) as data:
        required = {"occurrence_id", "center_start_zyx", "bias_start_zyx", "context_start_zyx"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"{path} 缺少训练池字段: {missing}。")
        occurrence = np.asarray(data["occurrence_id"], dtype=np.int32)
        center = np.asarray(data["center_start_zyx"], dtype=np.int32)
        bias = np.asarray(data["bias_start_zyx"], dtype=np.int32)
        context = np.asarray(data["context_start_zyx"], dtype=np.int32)
        if "pdb_id" in data:
            pdb_values = _string_array(data["pdb_id"])
            if len(pdb_values) != 1:
                raise ValueError(f"{path}: pdb_id 必须为标量。")
            pdb_id = pdb_values[0]
        else:
            pdb_id = path.stem.lower()
    if expected_pdb_id is not None and pdb_id != expected_pdb_id.lower():
        raise ValueError(f"{path}: 文件身份 {pdb_id!r} 与固定 PDB 表 {expected_pdb_id!r} 不一致。")
    n_occ = int(occurrence.shape[0])
    if occurrence.ndim != 1 or center.shape != (n_occ, 3) or bias.shape != (n_occ, 30, 3):
        raise ValueError(
            f"{path}: occurrence/center/bias 形状不符合 [N]、[N,3]、[N,30,3] 契约。"
        )
    if context.ndim != 2 or context.shape[1] != 3:
        raise ValueError(f"{path}: context_start_zyx 必须为 [N_context,3]。")
    if np.unique(occurrence).shape[0] != n_occ:
        raise ValueError(f"{path}: occurrence_id 必须在 PDB 内唯一。")
    return _PdbPool(pdb_id, occurrence, center, bias, context)


def _load_manifest_pool_paths(
    box_pool_root: str | Path,
    split_name: str,
    require_complete: bool = True,
) -> tuple[tuple[str, Path], ...]:
    """从根 manifest 读取一个 split 的精确 PDB pool 清单。

    manifest 是 pool 文件发现的唯一来源；同目录中未列出的历史 NPZ 不会被扫描或消费。
    每个条目的 ``path`` 必须是位于相应 split 子目录内的相对路径。
    """

    root = Path(box_pool_root)
    split_name = str(split_name).strip().lower()
    if require_complete and not (root / "_COMPLETE").is_file():
        raise FileNotFoundError(f"box pool 尚未原子发布完成: {root / '_COMPLETE'}。")
    manifest_path = root / BOX_POOL_MANIFEST_FILENAME
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or int(value.get("schema_version", -1)) != 1:
        raise ValueError(f"{manifest_path}: box pool manifest schema_version 必须为 1。")
    splits = value.get("splits")
    if not isinstance(splits, dict) or not isinstance(splits.get(split_name), list):
        raise KeyError(f"{manifest_path}: 缺少 splits.{split_name} 精确清单。")

    entries: list[tuple[str, Path]] = []
    seen_pdb_ids: set[str] = set()
    for index, row in enumerate(splits[split_name]):
        if not isinstance(row, dict):
            raise TypeError(f"{manifest_path}: splits.{split_name}[{index}] 必须为 object。")
        pdb_id = str(row.get("pdb_id", "")).strip().lower()
        relative_path = Path(str(row.get("path", "")))
        if not pdb_id or pdb_id in seen_pdb_ids:
            raise ValueError(f"{manifest_path}: splits.{split_name} 含空或重复 pdb_id={pdb_id!r}。")
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"{manifest_path}: pool path 必须是无上跳的相对路径: {relative_path}。")
        if not relative_path.parts or relative_path.parts[0].lower() != split_name:
            raise ValueError(
                f"{manifest_path}: {pdb_id} 的 path 必须位于 {split_name}/ 子目录，实际为 {relative_path}。"
            )
        pool_path = root / relative_path
        if not pool_path.is_file():
            raise FileNotFoundError(f"manifest 声明的 BOX pool 不存在: {pool_path}。")
        entries.append((pdb_id, pool_path))
        seen_pdb_ids.add(pdb_id)
    if not entries:
        raise ValueError(f"{manifest_path}: splits.{split_name} 不能为空。")
    return tuple(entries)


class Stage1TrainingRequestSet:
    """按 epoch 从冻结 PDB BOX 池生成 ``1:5:3`` 请求表。

    请求总数在不同 epoch 间保持不变；变化的只有至多 50 个 occurrence、5 个 bias
    以及 3 个 context 的选择。重复起点不会被去重。
    """

    def __init__(self, pool_directory: str | Path, seed: int) -> None:
        pool_dir = Path(pool_directory)
        manifest_entries = _load_manifest_pool_paths(pool_dir.parent, pool_dir.name)
        self._pools = tuple(
            _load_pdb_pool(path, expected_pdb_id=pdb_id)
            for pdb_id, path in manifest_entries
        )
        self.seed = int(seed)
        self.epoch = -1
        self.requests: tuple[ResolvedStage1Crop, ...] = ()
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        """按 ``seed+epoch`` 的稳定随机流重建本 epoch 请求。"""

        epoch = int(epoch)
        if epoch == self.epoch:
            return
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch]))
        requests: list[ResolvedStage1Crop] = []
        for pool in self._pools:
            n_occ = int(pool.occurrence_id.shape[0])
            if n_occ == 0:
                continue
            selected_occ_rows = rng.choice(n_occ, size=min(50, n_occ), replace=False)
            context_count = int(pool.context_start_zyx.shape[0])
            for occ_row in selected_occ_rows.tolist():
                occurrence_id = int(pool.occurrence_id[occ_row])
                requests.append(
                    ResolvedStage1Crop(
                        pool.pdb_id,
                        tuple(pool.center_start_zyx[occ_row].tolist()),
                        True,
                        "center",
                        occurrence_id,
                        None,
                    )
                )
                bias_indices = rng.choice(30, size=5, replace=False)
                for candidate_index in bias_indices.tolist():
                    requests.append(
                        ResolvedStage1Crop(
                            pool.pdb_id,
                            tuple(pool.bias_start_zyx[occ_row, candidate_index].tolist()),
                            True,
                            "bias",
                            occurrence_id,
                            int(candidate_index),
                        )
                    )
                if context_count > 0:
                    context_indices = rng.choice(
                        context_count, size=3, replace=context_count < 3
                    )
                    for candidate_index in context_indices.tolist():
                        requests.append(
                            ResolvedStage1Crop(
                                pool.pdb_id,
                                tuple(pool.context_start_zyx[candidate_index].tolist()),
                                True,
                                "context",
                                occurrence_id,
                                int(candidate_index),
                            )
                        )
        self.requests = tuple(requests)
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.requests)

    def __getitem__(self, index: int) -> ResolvedStage1Crop:
        return self.requests[index]


def load_validation_selection(selection_path: str | Path, box_pool_root: str | Path) -> list[ResolvedStage1Crop]:
    """展开 ``validation_selection.npz`` 对固定 PDB pool 的索引引用。"""

    source = Path(selection_path)
    manifest_paths = dict(_load_manifest_pool_paths(box_pool_root, "validation"))
    with np.load(source, allow_pickle=False) as data:
        required = {
            "validation_pdb_id",
            "center_pdb_index",
            "center_occurrence_id",
            "bias_pdb_index",
            "bias_occurrence_id",
            "bias_candidate_index",
            "context_pdb_index",
            "context_candidate_index",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"{source} 缺少 validation selection 字段: {missing}。")
        pdb_ids = _string_array(data["validation_pdb_id"])
        arrays = {name: np.asarray(data[name]) for name in required if name != "validation_pdb_id"}
    missing_manifest_ids = sorted(set(pdb_ids).difference(manifest_paths))
    if missing_manifest_ids:
        raise KeyError(f"{source}: validation PDB 不在 box pool manifest 中: {missing_manifest_ids[:10]}。")
    pools = {
        pdb_id: _load_pdb_pool(manifest_paths[pdb_id], expected_pdb_id=pdb_id)
        for pdb_id in pdb_ids
    }
    occurrence_rows = {
        pdb_id: {int(occurrence_id): row for row, occurrence_id in enumerate(pool.occurrence_id.tolist())}
        for pdb_id, pool in pools.items()
    }

    requests: list[ResolvedStage1Crop] = []
    for pdb_index, occurrence_id in zip(arrays["center_pdb_index"], arrays["center_occurrence_id"]):
        pdb_id = pdb_ids[int(pdb_index)]
        row = occurrence_rows[pdb_id][int(occurrence_id)]
        requests.append(
            ResolvedStage1Crop(pdb_id, tuple(pools[pdb_id].center_start_zyx[row].tolist()), True, "center", int(occurrence_id))
        )
    for pdb_index, occurrence_id, candidate_index in zip(
        arrays["bias_pdb_index"], arrays["bias_occurrence_id"], arrays["bias_candidate_index"]
    ):
        pdb_id = pdb_ids[int(pdb_index)]
        row = occurrence_rows[pdb_id][int(occurrence_id)]
        candidate_index = int(candidate_index)
        requests.append(
            ResolvedStage1Crop(
                pdb_id,
                tuple(pools[pdb_id].bias_start_zyx[row, candidate_index].tolist()),
                True,
                "bias",
                int(occurrence_id),
                candidate_index,
            )
        )
    for pdb_index, candidate_index in zip(arrays["context_pdb_index"], arrays["context_candidate_index"]):
        pdb_id = pdb_ids[int(pdb_index)]
        candidate_index = int(candidate_index)
        requests.append(
            ResolvedStage1Crop(
                pdb_id,
                tuple(pools[pdb_id].context_start_zyx[candidate_index].tolist()),
                True,
                "context",
                None,
                candidate_index,
            )
        )
    return requests


def build_request_source(
    split_file: str | Path | Sequence[str | Path],
    mode: str,
    box_pool_root: str | Path | None,
    seed: int,
) -> Stage1TrainingRequestSet | list[ResolvedStage1Crop]:
    """根据模式构造训练动态池、固定验证表或普通冻结请求表。"""

    mode = str(mode).lower()
    sources = list(split_file) if isinstance(split_file, (list, tuple)) else [split_file]
    if len(sources) == 1 and Path(sources[0]).is_dir():
        if mode != "train":
            raise ValueError("只有 train 模式允许直接读取 BOX pool 目录。")
        return Stage1TrainingRequestSet(sources[0], seed=seed)
    if len(sources) == 1 and Path(sources[0]).name == "validation_selection.npz":
        if box_pool_root is None:
            raise ValueError("读取 validation_selection.npz 时必须提供 box_pool_root。")
        return load_validation_selection(sources[0], box_pool_root)
    default_targets = mode in {"train", "val", "validation"}
    requests: list[ResolvedStage1Crop] = []
    for source in sources:
        requests.extend(load_flat_requests(source, default_targets=default_targets))
    return requests
