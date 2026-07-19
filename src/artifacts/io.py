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
    stage1_model_name: str | None = None,
) -> dict[str, np.ndarray]:
    """
    把一个 PDB 的同类 centered entry 聚合为单个数值归档。

    输入参数:
        - entries: Sequence[Mapping[str, Any]], 可变长度，每项包含共同几何/来源字段、
          voxel ragged payload、四张固定 V grid，以及 producer 实际存在的 P/A 字段；
          `CLG_centered` 还包含 candidate 与其 voxel/A membership；Selected 的固定
          V grid 只随 `refine_status=success` 的 entry 保存
        - centered_role: str, 三个正式 centered role 之一
        - stage1_model_name: str | None, 可选 producer 身份；正式发布必须传入，
          以校验 Find 的 P/A 与 unet_c1 的 V-only 模态

    输出:
        - arrays: dict[str, np.ndarray], 可直接传给 `atomic_savez_compressed` 的聚合字段；
          每个 offsets 字段切分的 value 表由 BOX 契约和本函数常量明确规定
    """
    if centered_role not in CENTERED_ROLES:
        raise ValueError(f"未知 centered_role={centered_role!r}")
    if len(entries) == 0:
        arrays = _empty_centered_archive(centered_role)
        validate_centered_archive(
            arrays, centered_role, stage1_model_name=stage1_model_name
        )
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

    if centered_role == "Selected_Refined_Centered":
        feature_entry_index = np.asarray(
            [
                index
                for index, entry in enumerate(entries)
                if str(entry["refine_status"]) == "success"
            ],
            dtype=np.int32,
        )
        arrays["feature_entry_index"] = feature_entry_index
        feature_entries = [entries[int(index)] for index in feature_entry_index]
        feature_entry_set = set(feature_entry_index.tolist())
        fixed_grid_tail_shapes = {
            "voxel_ds_2": (256, 20, 20, 20),
            "voxel_ds_3": (256, 10, 10, 10),
            "voxel_ds_4": (256, 5, 5, 5),
            "voxel_c4": (256, 5, 5, 5),
        }
        for field, tail_shape in fixed_grid_tail_shapes.items():
            if not all(field in entry for entry in feature_entries):
                raise ValueError(f"Selected success entry 缺少固定网格字段 {field}")
            if any(
                field in entry
                for index, entry in enumerate(entries)
                if index not in feature_entry_set
            ):
                raise ValueError(f"Selected 非 success entry 不得伪造固定网格字段 {field}")
            arrays[field] = (
                np.stack(
                    [np.asarray(entry[field], dtype=np.float16) for entry in feature_entries],
                    axis=0,
                )
                if feature_entries
                else np.empty((0, *tail_shape), dtype=np.float16)
            )
    else:
        for field in FIXED_VOXEL_GRID_FIELDS:
            if not all(field in entry for entry in entries):
                raise ValueError(f"固定网格字段 {field} 必须出现在全部 entry")
            arrays[field] = np.stack(
                [np.asarray(entry[field], dtype=np.float16) for entry in entries], axis=0
            )

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

    validate_centered_archive(
        arrays, centered_role, stage1_model_name=stage1_model_name
    )
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
        arrays["feature_entry_index"] = np.empty(0, dtype=np.int32)
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
    stage1_model_name: str | None = None,
) -> None:
    """
    校验 centered 聚合归档的 dtype、shape 与 ragged 对齐关系。

    输入参数:
        - arrays: Mapping[str, np.ndarray], 解码后的完整 NPZ 字段
        - centered_role: str, 当前归档 role
        - stage1_model_name: str | None, 可选 producer 身份；用于执行模态专属校验

    输出:
        - None, 契约满足时返回；磁盘边界错误会在最靠近读取处 fail-fast
    """
    if centered_role not in CENTERED_ROLES:
        raise ValueError(f"未知 centered_role={centered_role!r}")
    if stage1_model_name is not None and stage1_model_name not in {
        "Find_0",
        "Find_1",
        "unet_c1",
    }:
        raise ValueError(f"未知 stage1_model_name={stage1_model_name!r}")
    required_entry_fields = (
        "contract_version",
        "centered_box_index",
        "box_start_zyx",
        "box_shape_zyx",
        "box_origin_world",
        "voxel_size_world",
        "source_tree_id",
        "source_node_id",
        "source_threshold_grid_index",
        "source_threshold_value",
        "voxel_offsets",
        *VOXEL_VALUE_FIELDS,
        "voxel_aux_offsets",
        *AUX_VALUE_FIELDS,
        *FIXED_VOXEL_GRID_FIELDS,
    )
    if centered_role == "Selected_Refined_Centered":
        required_entry_fields += (
            "feature_entry_index",
            "refine_status",
            "refine_status_names",
        )
    missing = [field for field in required_entry_fields if field not in arrays]
    if missing:
        raise KeyError(f"centered 归档缺少共同字段: {missing}")
    n_entry = int(np.asarray(arrays["centered_box_index"]).shape[0])
    if str(np.asarray(arrays["contract_version"]).item()) != "adaligand_stage1_box_v1":
        raise ValueError("centered 归档 contract_version 不受支持")
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
        "feature_entry_index": np.int32,
        "refine_status": np.uint8,
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
    voxel_length = int(np.asarray(arrays["voxel_index_local_zyx"]).shape[0])
    validate_offsets(np.asarray(arrays["voxel_offsets"]), voxel_length, "voxel_offsets")
    if np.asarray(arrays["voxel_offsets"]).shape != (n_entry + 1,):
        raise ValueError("voxel_offsets 长度必须为 N_entry+1")
    voxel_index = np.asarray(arrays["voxel_index_local_zyx"])
    if voxel_index.shape != (voxel_length, 3):
        raise ValueError("voxel_index_local_zyx 必须为 [L_voxel,3]")
    if np.asarray(arrays["centered_probability"]).shape != (voxel_length,):
        raise ValueError("centered_probability 必须为 [L_voxel]")
    if np.asarray(arrays["voxel_final"]).shape != (voxel_length, 48):
        raise ValueError("voxel_final 必须为 [L_voxel,48]")
    if voxel_index.size and (int(voxel_index.min()) < 0 or int(voxel_index.max()) >= 80):
        raise ValueError("voxel_index_local_zyx 必须位于当前 80³ BOX")
    voxel_offsets = np.asarray(arrays["voxel_offsets"])
    for entry_index in range(n_entry):
        rows = voxel_index[
            int(voxel_offsets[entry_index]) : int(voxel_offsets[entry_index + 1])
        ]
        if rows.size and np.unique(rows, axis=0).shape[0] != rows.shape[0]:
            raise ValueError("同一 centered entry 的权威 voxel index 必须唯一")

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

    aux_length = int(np.asarray(arrays["voxel_aux_probability"]).shape[0])
    if np.asarray(arrays["voxel_aux_index_local_zyx"]).shape != (aux_length, 3):
        raise ValueError("voxel_aux_index_local_zyx 必须为 [L_aux,3]")
    if np.asarray(arrays["voxel_aux_offsets"]).shape != (n_entry + 1,):
        raise ValueError("voxel_aux_offsets 长度必须为 N_entry+1")

    p_fields = ("P_offsets", *P_VALUE_FIELDS)
    a_fields = ("A_offsets", *A_VALUE_FIELDS)
    has_p_or_a = any(field in arrays for field in (*p_fields, *a_fields))
    if stage1_model_name == "unet_c1" and has_p_or_a:
        raise ValueError("unet_c1 centered 归档不得伪造 P/A 字段")
    needs_find_payload = (
        stage1_model_name in {"Find_0", "Find_1"} and n_entry > 0
    )
    if centered_role == "Selected_Refined_Centered" and "refine_status" in arrays:
        needs_find_payload = needs_find_payload and bool(
            np.any(np.asarray(arrays["refine_status"]) == REFINE_STATUS_TO_CODE["success"])
        )
    if needs_find_payload or (stage1_model_name is None and has_p_or_a):
        missing_p_a = [field for field in (*p_fields, *a_fields) if field not in arrays]
        if missing_p_a:
            raise KeyError(f"Find centered 归档缺少 P/A 字段: {missing_p_a}")
    if all(field in arrays for field in p_fields):
        p_length = int(np.asarray(arrays["P_probability"]).shape[0])
        if np.asarray(arrays["P_coord_local_xyz"]).shape != (p_length, 3):
            raise ValueError("P_coord_local_xyz 必须为 [L_P,3]")
        if any(np.asarray(arrays[field]).ndim != 2 for field in ("P_feat_L2", "P_feat_L3", "P_feat_L4")):
            raise ValueError("P_feat_L2/L3/L4 必须为二维 feature 表")
    if all(field in arrays for field in a_fields):
        a_length = int(np.asarray(arrays["A_probability"]).shape[0])
        for field in ("A_coord_local_xyz", "A_coord_centered_world"):
            if np.asarray(arrays[field]).shape != (a_length, 3):
                raise ValueError(f"{field} 必须为 [L_A,3]")
        if np.asarray(arrays["A_global_index"]).shape != (a_length,):
            raise ValueError("A_global_index 必须为 [L_A]")
        if any(
            np.asarray(arrays[field]).ndim != 2
            for field in ("A_feat_L1", "A_feat_L2", "A_feat_L3", "A_feat_L4")
        ):
            raise ValueError("A_feat_L1/L2/L3/L4 必须为二维 feature 表")

    fixed_entry_count = n_entry
    if centered_role == "Selected_Refined_Centered":
        feature_entry_index = np.asarray(arrays["feature_entry_index"])
        if feature_entry_index.ndim != 1:
            raise ValueError("feature_entry_index 必须为一维 int32")
        fixed_entry_count = int(feature_entry_index.size)
    fixed_shapes = {
        "voxel_ds_2": (fixed_entry_count, 256, 20, 20, 20),
        "voxel_ds_3": (fixed_entry_count, 256, 10, 10, 10),
        "voxel_ds_4": (fixed_entry_count, 256, 5, 5, 5),
        "voxel_c4": (fixed_entry_count, 256, 5, 5, 5),
    }
    for field, shape in fixed_shapes.items():
        if np.asarray(arrays[field]).dtype != np.dtype(np.float16):
            raise ValueError(f"{field}.dtype 必须为 float16")
        if np.asarray(arrays[field]).shape != shape:
            raise ValueError(f"{field}.shape 必须为 {shape}")

    if centered_role == "CLG_centered":
        required_clg_fields = (
            "CLG_id",
            "CLG_seed_node_id",
            "CLG_oldest_node_id",
            "candidate_offsets",
            "candidate_node_id",
            "candidate_threshold_grid_index",
            "candidate_voxel_offsets",
            "candidate_voxel_index",
        )
        missing_clg = [field for field in required_clg_fields if field not in arrays]
        if missing_clg:
            raise KeyError(f"CLG_centered 缺少字段: {missing_clg}")
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
        elif stage1_model_name in {"Find_0", "Find_1"} and n_entry > 0:
            raise KeyError("Find CLG_centered 缺少 candidate_A_offsets/candidate_A_index")

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
        expected_feature_entries = np.flatnonzero(
            status == REFINE_STATUS_TO_CODE["success"]
        ).astype(np.int32)
        if not np.array_equal(
            np.asarray(arrays["feature_entry_index"]), expected_feature_entries
        ):
            raise ValueError(
                "feature_entry_index 必须严格等于 refine_status=success 的 entry 行"
            )
        for entry_index, status_code in enumerate(status):
            voxel_count = int(voxel_offsets[entry_index + 1] - voxel_offsets[entry_index])
            if int(status_code) == REFINE_STATUS_TO_CODE["success"] and voxel_count == 0:
                raise ValueError("refine_status=success 必须具有非空 refined_blob")
            if int(status_code) != REFINE_STATUS_TO_CODE["success"] and voxel_count != 0:
                raise ValueError("非 success Selected entry 不得具有权威 voxel payload")
            if int(status_code) != REFINE_STATUS_TO_CODE["success"]:
                for offsets_name in ("voxel_aux_offsets", "P_offsets", "A_offsets"):
                    if offsets_name not in arrays:
                        continue
                    offsets = np.asarray(arrays[offsets_name])
                    if int(offsets[entry_index + 1]) != int(offsets[entry_index]):
                        raise ValueError(
                            f"非 success Selected entry 不得具有 {offsets_name} payload"
                        )
