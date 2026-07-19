"""Stage1 数值归档的原子 IO、ragged 校验与 centered 聚合。"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


CENTERED_ROLES: tuple[str, ...] = (
    "F1_centered",
    "CLG_centered",
    "Selected_Refined_Centered",
)
REFINE_STATUS_TO_CODE: dict[str, int] = {
    "success": 0,
    "empty": 1,
    "no_overlap": 2,
    "failed": 3,
}
FIXED_VOXEL_GRID_FIELDS: tuple[str, ...] = (
    "voxel_ds_2",
    "voxel_ds_3",
    "voxel_ds_4",
    "voxel_c4",
)
VOXEL_VALUE_FIELDS: tuple[str, ...] = (
    "voxel_index_local_zyx",
    "centered_probability",
    "voxel_final",
)
AUX_VALUE_FIELDS: tuple[str, ...] = (
    "voxel_aux_index_local_zyx",
    "voxel_aux_probability",
)
P_VALUE_FIELDS: tuple[str, ...] = (
    "P_coord_local_xyz",
    "P_probability",
    "P_feat_L2",
    "P_feat_L3",
    "P_feat_L4",
)
A_VALUE_FIELDS: tuple[str, ...] = (
    "A_global_index",
    "A_coord_local_xyz",
    "A_coord_centered_world",
    "A_probability",
    "A_feat_L1",
    "A_feat_L2",
    "A_feat_L3",
    "A_feat_L4",
)


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """
    在目标目录中完成 JSON 的原子发布。

    输入参数:
        - path: str | Path, 正式 JSON 路径
        - payload: Mapping[str, Any], 可由标准 JSON 编码的内容

    输出:
        - None, 关闭并 fsync 临时文件后以 `os.replace` 发布
    """
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = target_path.with_name(
        f".{target_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_savez_compressed(
    path: str | Path,
    arrays: Mapping[str, np.ndarray],
    validator: Callable[[Mapping[str, np.ndarray]], None] | None = None,
) -> None:
    """
    把纯数值数组压缩写入临时 NPZ，重读校验后原子发布。

    输入参数:
        - path: str | Path, 正式 `.npz` 路径
        - arrays: Mapping[str, np.ndarray], 字段名到无 object dtype 数组的映射
        - validator: Callable | None, 可选的归档级语义校验函数

    输出:
        - None, 仅在临时归档可无 pickle 重读且通过校验后替换正式路径
    """
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {name: np.asarray(value) for name, value in arrays.items()}
    object_fields = [name for name, value in normalized.items() if value.dtype.hasobject]
    if object_fields:
        raise TypeError(f"NPZ 禁止 object dtype，违规字段: {object_fields}")
    if validator is not None:
        validator(normalized)

    temporary_path = target_path.with_name(
        f".{target_path.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp.npz"
    )
    try:
        np.savez_compressed(temporary_path, **normalized)
        # Windows 对只读描述符调用 fsync 会返回 EBADF；以不改内容的读写模式打开。
        with temporary_path.open("rb+") as handle:
            os.fsync(handle.fileno())
        reloaded = load_npz_strict(temporary_path)
        if validator is not None:
            validator(reloaded)
        os.replace(temporary_path, target_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_npz_strict(path: str | Path) -> dict[str, np.ndarray]:
    """
    以 `allow_pickle=False` 完整读出一个 NPZ。

    输入参数:
        - path: str | Path, 待读取的正式或临时 NPZ

    输出:
        - arrays: dict[str, np.ndarray], 已脱离文件句柄的字段副本
    """
    with np.load(Path(path), allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def validate_offsets(
    offsets: np.ndarray,
    value_length: int,
    offsets_name: str,
) -> None:
    """
    校验一个 offsets 数组完整切分指定 value 表。

    输入参数:
        - offsets: np.ndarray, (N+1,), 预期 int64 的半开区间边界
        - value_length: int, 被切分 value 表的第一维长度
        - offsets_name: str, 报错时使用的字段名

    输出:
        - None, 首项、末项、dtype 与单调性均正确时返回
    """
    array = np.asarray(offsets)
    if array.dtype != np.dtype(np.int64) or array.ndim != 1 or array.size == 0:
        raise ValueError(f"{offsets_name} 必须是非空一维 int64 数组")
    if int(array[0]) != 0 or int(array[-1]) != int(value_length):
        raise ValueError(
            f"{offsets_name} 必须从 0 到 {value_length}，实际首末值为 "
            f"{int(array[0])}/{int(array[-1])}"
        )
    if bool(np.any(array[1:] < array[:-1])):
        raise ValueError(f"{offsets_name} 必须单调不减")


def _concatenate_entry_values(
    entries: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    offsets_name: str,
) -> dict[str, np.ndarray]:
    """按 entry 顺序拼接一组共享第一维的 ragged value 字段。"""
    if len(entries) == 0:
        return {offsets_name: np.zeros(1, dtype=np.int64)}
    present = [field for field in fields if field in entries[0]]
    if not present:
        if any(any(field in entry for field in fields) for entry in entries):
            raise ValueError(f"{offsets_name} 对应字段只在部分 entry 出现")
        return {}
    for entry in entries:
        if any((field in entry) != (field in present) for field in fields):
            raise ValueError(f"{offsets_name} 对应字段必须在所有 entry 中保持相同集合")

    lengths: list[int] = []
    values_by_field: dict[str, list[np.ndarray]] = {field: [] for field in present}
    for entry in entries:
        arrays = {field: np.asarray(entry[field]) for field in present}
        first_lengths = {int(value.shape[0]) for value in arrays.values()}
        if len(first_lengths) != 1:
            raise ValueError(f"{offsets_name} 对应 value 表在同一 entry 内长度不一致")
        entry_length = first_lengths.pop()
        lengths.append(entry_length)
        for field, value in arrays.items():
            values_by_field[field].append(value)

    offsets = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(lengths, dtype=np.int64)]
    )
    result: dict[str, np.ndarray] = {offsets_name: offsets}
    for field, values in values_by_field.items():
        exemplar = values[0]
        if int(offsets[-1]) == 0:
            result[field] = np.empty((0, *exemplar.shape[1:]), dtype=exemplar.dtype)
        else:
            result[field] = np.concatenate(values, axis=0)
    return result


def _stack_entry_field(
    entries: Sequence[Mapping[str, Any]],
    field: str,
    dtype: np.dtype[Any],
) -> np.ndarray:
    """把固定 shape 的 entry 字段按第一维堆叠并转换到契约 dtype。"""
    return np.asarray([entry[field] for entry in entries], dtype=dtype)


def pack_centered_entries(
    entries: Sequence[Mapping[str, Any]],
    centered_role: str,
) -> dict[str, np.ndarray]:
    """
    把一个 PDB 的同类 centered entry 聚合为单个数值归档。

    输入参数:
        - entries: Sequence[Mapping[str, Any]], 可变长度，每项包含共同几何/来源字段、
          voxel ragged payload、四张固定 V grid，以及 producer 实际存在的 P/A 字段；
          `CLG_centered` 还包含 candidate 与其 voxel/A membership
        - centered_role: str, 三个正式 centered role 之一

    输出:
        - arrays: dict[str, np.ndarray], 可直接传给 `atomic_savez_compressed` 的聚合字段；
          每个 offsets 字段切分的 value 表由 BOX 契约和本函数常量明确规定
    """
    if centered_role not in CENTERED_ROLES:
        raise ValueError(f"未知 centered_role={centered_role!r}")
    if len(entries) == 0:
        arrays = _empty_centered_archive(centered_role)
        validate_centered_archive(arrays, centered_role)
        return arrays

    arrays: dict[str, np.ndarray] = {
        "contract_version": np.asarray("adaligand_stage1_box_v1"),
        "centered_box_index": np.arange(len(entries), dtype=np.int32),
        "box_start_zyx": _stack_entry_field(entries, "box_start_zyx", np.dtype(np.int32)),
        "box_shape_zyx": _stack_entry_field(entries, "box_shape_zyx", np.dtype(np.uint8)),
        "box_origin_world": _stack_entry_field(entries, "box_origin_world", np.dtype(np.float32)),
        "voxel_size_world": _stack_entry_field(entries, "voxel_size_world", np.dtype(np.float32)),
        "source_tree_id": _stack_entry_field(entries, "source_tree_id", np.dtype(np.int32)),
        "source_node_id": _stack_entry_field(entries, "source_node_id", np.dtype(np.int32)),
        "source_threshold_grid_index": _stack_entry_field(
            entries, "source_threshold_grid_index", np.dtype(np.int32)
        ),
        "source_threshold_value": _stack_entry_field(
            entries, "source_threshold_value", np.dtype(np.float32)
        ),
    }
    arrays.update(_concatenate_entry_values(entries, VOXEL_VALUE_FIELDS, "voxel_offsets"))
    arrays.update(_concatenate_entry_values(entries, AUX_VALUE_FIELDS, "voxel_aux_offsets"))
    arrays.update(_concatenate_entry_values(entries, P_VALUE_FIELDS, "P_offsets"))
    arrays.update(_concatenate_entry_values(entries, A_VALUE_FIELDS, "A_offsets"))

    for field in FIXED_VOXEL_GRID_FIELDS:
        if field in entries[0]:
            if not all(field in entry for entry in entries):
                raise ValueError(f"固定网格字段 {field} 不能只出现在部分 entry")
            arrays[field] = np.stack(
                [np.asarray(entry[field], dtype=np.float16) for entry in entries], axis=0
            )
        elif any(field in entry for entry in entries):
            raise ValueError(f"固定网格字段 {field} 不能只出现在部分 entry")

    if centered_role == "CLG_centered":
        arrays["CLG_id"] = _stack_entry_field(entries, "CLG_id", np.dtype(np.int32))
        arrays["CLG_seed_node_id"] = _stack_entry_field(
            entries, "CLG_seed_node_id", np.dtype(np.int32)
        )
        arrays["CLG_oldest_node_id"] = _stack_entry_field(
            entries, "CLG_oldest_node_id", np.dtype(np.int32)
        )
        arrays.update(
            _concatenate_entry_values(
                entries,
                ("candidate_node_id", "candidate_threshold_grid_index"),
                "candidate_offsets",
            )
        )
        candidate_voxel_rows: list[np.ndarray] = []
        candidate_a_rows: list[np.ndarray] = []
        for entry in entries:
            candidate_count = int(np.asarray(entry["candidate_node_id"]).shape[0])
            voxel_memberships = list(entry["candidate_voxel_membership"])
            if len(voxel_memberships) != candidate_count:
                raise ValueError("candidate_voxel_membership 数量必须等于当前 entry candidate 数")
            candidate_voxel_rows.extend(
                np.asarray(value, dtype=np.int32) for value in voxel_memberships
            )
            if "candidate_A_membership" in entry:
                a_memberships = list(entry["candidate_A_membership"])
                if len(a_memberships) != candidate_count:
                    raise ValueError("candidate_A_membership 数量必须等于当前 entry candidate 数")
                candidate_a_rows.extend(np.asarray(value, dtype=np.int32) for value in a_memberships)
            elif "A_offsets" in arrays:
                raise ValueError("Find 的 CLG_centered 必须提供 candidate_A_membership")
        arrays.update(_pack_nested_membership(candidate_voxel_rows, "candidate_voxel"))
        if candidate_a_rows:
            arrays.update(_pack_nested_membership(candidate_a_rows, "candidate_A"))

    if centered_role == "Selected_Refined_Centered":
        statuses = [str(entry["refine_status"]) for entry in entries]
        arrays["refine_status"] = np.asarray(
            [REFINE_STATUS_TO_CODE[status] for status in statuses], dtype=np.uint8
        )
        arrays["refine_status_names"] = np.asarray(tuple(REFINE_STATUS_TO_CODE), dtype="U10")

    validate_centered_archive(arrays, centered_role)
    return arrays


def _empty_centered_archive(centered_role: str) -> dict[str, np.ndarray]:
    """
    构造没有候选时仍可正式发布的零 entry centered 归档。

    输入参数:
        - centered_role: str，三个正式 centered role 之一

    输出:
        - arrays: dict[str,np.ndarray]，共同 entry 表第一维为 0，全部 ragged offsets
          仅含起始 0；因为没有 entry，Find 的 P/A 模态不需要伪造未知通道宽度
    """
    arrays: dict[str, np.ndarray] = {
        "contract_version": np.asarray("adaligand_stage1_box_v1"),
        "centered_box_index": np.empty(0, dtype=np.int32),
        "box_start_zyx": np.empty((0, 3), dtype=np.int32),
        "box_shape_zyx": np.empty((0, 3), dtype=np.uint8),
        "box_origin_world": np.empty((0, 3), dtype=np.float32),
        "voxel_size_world": np.empty((0, 3), dtype=np.float32),
        "source_tree_id": np.empty(0, dtype=np.int32),
        "source_node_id": np.empty(0, dtype=np.int32),
        "source_threshold_grid_index": np.empty(0, dtype=np.int32),
        "source_threshold_value": np.empty(0, dtype=np.float32),
        "voxel_offsets": np.zeros(1, dtype=np.int64),
        "voxel_index_local_zyx": np.empty((0, 3), dtype=np.int16),
        "centered_probability": np.empty(0, dtype=np.float32),
        "voxel_final": np.empty((0, 48), dtype=np.float16),
        "voxel_aux_offsets": np.zeros(1, dtype=np.int64),
        "voxel_aux_index_local_zyx": np.empty((0, 3), dtype=np.int16),
        "voxel_aux_probability": np.empty(0, dtype=np.float32),
        "voxel_ds_2": np.empty((0, 256, 20, 20, 20), dtype=np.float16),
        "voxel_ds_3": np.empty((0, 256, 10, 10, 10), dtype=np.float16),
        "voxel_ds_4": np.empty((0, 256, 5, 5, 5), dtype=np.float16),
        "voxel_c4": np.empty((0, 256, 5, 5, 5), dtype=np.float16),
    }
    if centered_role == "CLG_centered":
        arrays.update(
            {
                "CLG_id": np.empty(0, dtype=np.int32),
                "CLG_seed_node_id": np.empty(0, dtype=np.int32),
                "CLG_oldest_node_id": np.empty(0, dtype=np.int32),
                "candidate_offsets": np.zeros(1, dtype=np.int64),
                "candidate_node_id": np.empty(0, dtype=np.int32),
                "candidate_threshold_grid_index": np.empty(0, dtype=np.int32),
                "candidate_voxel_offsets": np.zeros(1, dtype=np.int64),
                "candidate_voxel_index": np.empty(0, dtype=np.int32),
            }
        )
    if centered_role == "Selected_Refined_Centered":
        arrays["refine_status"] = np.empty(0, dtype=np.uint8)
        arrays["refine_status_names"] = np.asarray(
            tuple(REFINE_STATUS_TO_CODE), dtype="U10"
        )
    return arrays


def _pack_nested_membership(
    rows: Sequence[np.ndarray],
    prefix: str,
) -> dict[str, np.ndarray]:
    """把逐 candidate 的局部 membership 行编码为 offsets+indices。"""
    lengths = np.asarray([np.asarray(row).size for row in rows], dtype=np.int64)
    offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(lengths)])
    if int(offsets[-1]) == 0:
        indices = np.empty(0, dtype=np.int32)
    else:
        indices = np.concatenate([np.asarray(row, dtype=np.int32).reshape(-1) for row in rows])
    return {f"{prefix}_offsets": offsets, f"{prefix}_index": indices}


def validate_centered_archive(
    arrays: Mapping[str, np.ndarray],
    centered_role: str,
) -> None:
    """
    校验 centered 聚合归档的 dtype、shape 与 ragged 对齐关系。

    输入参数:
        - arrays: Mapping[str, np.ndarray], 解码后的完整 NPZ 字段
        - centered_role: str, 当前归档 role

    输出:
        - None, 契约满足时返回；磁盘边界错误会在最靠近读取处 fail-fast
    """
    required_entry_fields = (
        "centered_box_index",
        "box_start_zyx",
        "box_shape_zyx",
        "box_origin_world",
        "voxel_size_world",
        "source_tree_id",
        "source_node_id",
        "source_threshold_grid_index",
        "source_threshold_value",
    )
    missing = [field for field in required_entry_fields if field not in arrays]
    if missing:
        raise KeyError(f"centered 归档缺少共同字段: {missing}")
    n_entry = int(np.asarray(arrays["centered_box_index"]).shape[0])
    expected_dtypes = {
        "centered_box_index": np.int32,
        "box_start_zyx": np.int32,
        "box_shape_zyx": np.uint8,
        "box_origin_world": np.float32,
        "voxel_size_world": np.float32,
        "source_tree_id": np.int32,
        "source_node_id": np.int32,
        "source_threshold_grid_index": np.int32,
        "source_threshold_value": np.float32,
        "voxel_offsets": np.int64,
        "voxel_index_local_zyx": np.int16,
        "centered_probability": np.float32,
        "voxel_final": np.float16,
        "voxel_aux_offsets": np.int64,
        "voxel_aux_index_local_zyx": np.int16,
        "voxel_aux_probability": np.float32,
        "P_offsets": np.int64,
        "P_coord_local_xyz": np.float32,
        "P_probability": np.float32,
        "P_feat_L2": np.float16,
        "P_feat_L3": np.float16,
        "P_feat_L4": np.float16,
        "A_offsets": np.int64,
        "A_global_index": np.int64,
        "A_coord_local_xyz": np.float32,
        "A_coord_centered_world": np.float32,
        "A_probability": np.float32,
        "A_feat_L1": np.float16,
        "A_feat_L2": np.float16,
        "A_feat_L3": np.float16,
        "A_feat_L4": np.float16,
    }
    for field, dtype in expected_dtypes.items():
        if field in arrays and np.asarray(arrays[field]).dtype != np.dtype(dtype):
            raise ValueError(f"{field}.dtype 必须为 {np.dtype(dtype)}")
    if not np.array_equal(
        np.asarray(arrays["centered_box_index"]), np.arange(n_entry, dtype=np.int32)
    ):
        raise ValueError("centered_box_index 必须是连续 int32 的 0..N_entry-1")
    expected_entry_shapes = {
        "box_start_zyx": (n_entry, 3),
        "box_shape_zyx": (n_entry, 3),
        "box_origin_world": (n_entry, 3),
        "voxel_size_world": (n_entry, 3),
        "source_tree_id": (n_entry,),
        "source_node_id": (n_entry,),
        "source_threshold_grid_index": (n_entry,),
        "source_threshold_value": (n_entry,),
    }
    for field, shape in expected_entry_shapes.items():
        if np.asarray(arrays[field]).shape != shape:
            raise ValueError(f"{field}.shape 必须为 {shape}")
    if not bool(np.all(np.asarray(arrays["box_shape_zyx"]) == 80)):
        raise ValueError("全部 centered BOX 的 box_shape_zyx 必须为 [80,80,80]")

    if "voxel_offsets" not in arrays:
        raise KeyError("centered 归档缺少 voxel_offsets")
    voxel_length = int(np.asarray(arrays.get("voxel_index_local_zyx", np.empty((0, 3)))).shape[0])
    validate_offsets(np.asarray(arrays["voxel_offsets"]), voxel_length, "voxel_offsets")
    if np.asarray(arrays["voxel_offsets"]).shape != (n_entry + 1,):
        raise ValueError("voxel_offsets 长度必须为 N_entry+1")
    for field in ("centered_probability", "voxel_final"):
        if field in arrays and int(np.asarray(arrays[field]).shape[0]) != voxel_length:
            raise ValueError(f"{field} 必须由 voxel_offsets 同步切分")
    if "voxel_index_local_zyx" in arrays:
        voxel_index = np.asarray(arrays["voxel_index_local_zyx"])
        if voxel_index.shape != (voxel_length, 3):
            raise ValueError("voxel_index_local_zyx 必须为 [L_voxel,3]")
        if voxel_index.size and (int(voxel_index.min()) < 0 or int(voxel_index.max()) >= 80):
            raise ValueError("voxel_index_local_zyx 必须位于当前 80³ BOX")
        voxel_offsets = np.asarray(arrays["voxel_offsets"])
        for entry_index in range(n_entry):
            rows = voxel_index[
                int(voxel_offsets[entry_index]) : int(voxel_offsets[entry_index + 1])
            ]
            if rows.size and np.unique(rows, axis=0).shape[0] != rows.shape[0]:
                raise ValueError("同一 centered entry 的权威 voxel index 必须唯一")
    if "voxel_final" in arrays and np.asarray(arrays["voxel_final"]).shape != (voxel_length, 48):
        raise ValueError("voxel_final 必须为 [L_voxel,48]")

    for offsets_name, fields in (
        ("voxel_aux_offsets", AUX_VALUE_FIELDS),
        ("P_offsets", P_VALUE_FIELDS),
        ("A_offsets", A_VALUE_FIELDS),
    ):
        if offsets_name not in arrays:
            continue
        present = [field for field in fields if field in arrays]
        if not present:
            raise ValueError(f"{offsets_name} 存在但没有对应 value 表")
        value_length = int(np.asarray(arrays[present[0]]).shape[0])
        validate_offsets(np.asarray(arrays[offsets_name]), value_length, offsets_name)
        if np.asarray(arrays[offsets_name]).shape != (n_entry + 1,):
            raise ValueError(f"{offsets_name} 长度必须为 N_entry+1")
        for field in present:
            if int(np.asarray(arrays[field]).shape[0]) != value_length:
                raise ValueError(f"{field} 必须由 {offsets_name} 同步切分")

    fixed_shapes = {
        "voxel_ds_2": (n_entry, 256, 20, 20, 20),
        "voxel_ds_3": (n_entry, 256, 10, 10, 10),
        "voxel_ds_4": (n_entry, 256, 5, 5, 5),
        "voxel_c4": (n_entry, 256, 5, 5, 5),
    }
    for field, shape in fixed_shapes.items():
        if field in arrays:
            if np.asarray(arrays[field]).dtype != np.dtype(np.float16):
                raise ValueError(f"{field}.dtype 必须为 float16")
            if np.asarray(arrays[field]).shape != shape:
                raise ValueError(f"{field}.shape 必须为 {shape}")

    if centered_role == "CLG_centered":
        for field in ("CLG_id", "CLG_seed_node_id", "CLG_oldest_node_id"):
            if field not in arrays or np.asarray(arrays[field]).shape != (n_entry,):
                raise ValueError(f"CLG_centered 的 {field} 必须为 [N_entry]")
        candidate_count = int(np.asarray(arrays["candidate_node_id"]).shape[0])
        validate_offsets(
            np.asarray(arrays["candidate_offsets"]), candidate_count, "candidate_offsets"
        )
        if np.asarray(arrays["candidate_offsets"]).shape != (n_entry + 1,):
            raise ValueError("candidate_offsets 长度必须为 N_entry+1")
        if np.asarray(arrays["candidate_threshold_grid_index"]).shape != (candidate_count,):
            raise ValueError("candidate_threshold_grid_index 必须与 candidate_node_id 对齐")
        validate_offsets(
            np.asarray(arrays["candidate_voxel_offsets"]),
            int(np.asarray(arrays["candidate_voxel_index"]).shape[0]),
            "candidate_voxel_offsets",
        )
        if np.asarray(arrays["candidate_voxel_offsets"]).shape != (candidate_count + 1,):
            raise ValueError("candidate_voxel_offsets 长度必须为 N_candidate+1")
        candidate_voxel_index = np.asarray(arrays["candidate_voxel_index"])
        candidate_offsets = np.asarray(arrays["candidate_offsets"])
        candidate_voxel_offsets = np.asarray(arrays["candidate_voxel_offsets"])
        voxel_offsets = np.asarray(arrays["voxel_offsets"])
        for entry_index in range(n_entry):
            for candidate_index in range(
                int(candidate_offsets[entry_index]), int(candidate_offsets[entry_index + 1])
            ):
                rows = candidate_voxel_index[
                    int(candidate_voxel_offsets[candidate_index]) : int(
                        candidate_voxel_offsets[candidate_index + 1]
                    )
                ]
                entry_voxel_count = int(voxel_offsets[entry_index + 1] - voxel_offsets[entry_index])
                if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= entry_voxel_count):
                    raise ValueError("candidate_voxel_index 越过所属 entry 的局部 voxel 段")
                if np.unique(rows).size != rows.size:
                    raise ValueError("同一 candidate 的 candidate_voxel_index 必须唯一")
        if "candidate_A_offsets" in arrays:
            validate_offsets(
                np.asarray(arrays["candidate_A_offsets"]),
                int(np.asarray(arrays["candidate_A_index"]).shape[0]),
                "candidate_A_offsets",
            )
            if np.asarray(arrays["candidate_A_offsets"]).shape != (candidate_count + 1,):
                raise ValueError("candidate_A_offsets 长度必须为 N_candidate+1")
            candidate_a_index = np.asarray(arrays["candidate_A_index"])
            candidate_a_offsets = np.asarray(arrays["candidate_A_offsets"])
            a_offsets = np.asarray(arrays["A_offsets"])
            for entry_index in range(n_entry):
                for candidate_index in range(
                    int(candidate_offsets[entry_index]), int(candidate_offsets[entry_index + 1])
                ):
                    rows = candidate_a_index[
                        int(candidate_a_offsets[candidate_index]) : int(
                            candidate_a_offsets[candidate_index + 1]
                        )
                    ]
                    entry_a_count = int(a_offsets[entry_index + 1] - a_offsets[entry_index])
                    if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= entry_a_count):
                        raise ValueError("candidate_A_index 越过所属 entry 的局部 A 段")
                    if np.unique(rows).size != rows.size:
                        raise ValueError("同一 candidate 的 candidate_A_index 必须唯一")

    if centered_role == "F1_centered":
        forbidden = ("CLG_id", "candidate_offsets", "refine_status")
        if any(field in arrays for field in forbidden):
            raise ValueError("F1_centered 不得伪造 CLG、candidate 或 refine 字段")
    if centered_role == "Selected_Refined_Centered":
        if "refine_status" not in arrays or np.asarray(arrays["refine_status"]).shape != (n_entry,):
            raise ValueError("Selected_Refined_Centered 必须有 [N_entry] refine_status")
        status = np.asarray(arrays["refine_status"])
        if status.dtype != np.dtype(np.uint8) or bool(np.any(status > 3)):
            raise ValueError("refine_status 必须是仅含 0..3 的 uint8")
        voxel_offsets = np.asarray(arrays["voxel_offsets"])
        for entry_index, status_code in enumerate(status):
            voxel_count = int(voxel_offsets[entry_index + 1] - voxel_offsets[entry_index])
            if int(status_code) == REFINE_STATUS_TO_CODE["success"] and voxel_count == 0:
                raise ValueError("refine_status=success 必须具有非空 refined_blob")
            if int(status_code) != REFINE_STATUS_TO_CODE["success"] and voxel_count != 0:
                raise ValueError("非 success Selected entry 不得具有权威 voxel payload")
