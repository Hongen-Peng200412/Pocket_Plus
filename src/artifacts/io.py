"""构造、校验并原子发布 Stage1 JSON、纯数值 NPZ 与 centered 聚合归档。

主要入口:
    - `atomic_write_json`: 在目标目录写临时 JSON，完成 flush/fsync 后以 `os.replace` 发布。
    - `atomic_savez_compressed`: 拒绝 object dtype，重读并调用归档校验器后发布压缩 NPZ。
    - `pack_centered_entries`: 把一个 PDB 的同一 centered role 编码成固定 entry 表、ragged value 表和 offsets。
    - `validate_centered_archive`: 校验 centered 字段、数据类型、形状、索引边界、producer 模态和 role 专属关系。

一个 PDB 的每种 centered role 只发布一个 NPZ。长度可变的 voxel、辅助 voxel、P 点、A 点和候选 membership 表由各自的 int64 offsets 同步切分。全部 NPZ 字段都必须能在 `allow_pickle=False` 下读取。
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.stage1_producers import FIND_MODEL_NAMES, STAGE1_MODEL_NAMES


# 三类 centered 聚合归档；顺序与 `OUTPUT_ROLES` 的 centered 子序列一致。
CENTERED_ROLES: tuple[str, ...] = (
    "F1_centered",
    "CLG_centered",
    "Selected_Refined_Centered",
)
# Selected entry 的 uint8 状态编码；只有 `success=0` 允许携带模型特征和非空 refined voxel payload。
REFINE_STATUS_TO_CODE: dict[str, int] = {
    "success": 0,
    "empty": 1,
    "no_overlap": 2,
}
# 由 `voxel_offsets` 同步切分的权威来源/精修 voxel 表；各字段第一维均为 L_voxel。
VOXEL_VALUE_FIELDS: tuple[str, ...] = (
    "voxel_index_local_zyx",
    "centered_probability",
    "voxel_final",
)
# 由 `voxel_aux_offsets` 同步切分的辅助 voxel 表；各字段第一维均为 L_aux。
AUX_VALUE_FIELDS: tuple[str, ...] = (
    "voxel_aux_index_local_zyx",
    "voxel_aux_probability",
)
# 由 `P_offsets` 同步切分的 Find P 点表；坐标、概率和 L2-L3 特征的第一维均为 L_P。
P_VALUE_FIELDS: tuple[str, ...] = (
    "P_coord_local_xyz",
    "P_probability",
    "P_feat_L2",
    "P_feat_L3",
)
# 由 `A_offsets` 同步切分的 Find A 点表；全局编号、两种坐标、概率和 L0-L3 特征的第一维均为 L_A。
A_VALUE_FIELDS: tuple[str, ...] = (
    "A_global_index",
    "A_coord_local_xyz",
    "A_coord_centered_world",
    "A_probability",
    "A_feat_L0",
    "A_feat_L1",
    "A_feat_L2",
    "A_feat_L3",
)

# ------------------ 简单工具函数 ------------------
def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """
    在目标目录中完成 JSON 的原子发布。

    输入参数:
        - path: str | Path, 正式 JSON 路径；父目录不存在时由本函数创建。
        - payload: Mapping[str, Any], 可由标准库 `json` 编码的完整顶层映射。

    输出:
        - None, 关闭并 fsync 临时文件后以 `os.replace` 发布；失败时删除当前函数创建的临时文件。
    """
    # Path, 调用方可见的正式 JSON；只有完整临时文件会替换到此路径。
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
        - path: str | Path, 正式 `.npz` 路径；父目录不存在时由本函数创建。
        - arrays: Mapping[str, np.ndarray], NPZ 字段名到数组的映射；所有数组都必须不含 object dtype。
        - validator: Callable[[Mapping[str, np.ndarray]], None] | None, 可选归档级校验器；写入前和临时归档重读后各调用一次。

    输出:
        - None, 仅在临时归档能以 `allow_pickle=False` 重读且两次校验均通过后替换正式路径。
    """
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    # dict[str, np.ndarray], 与 NPZ 字段一一对应的数组视图；此处统一触发 `np.asarray`，但不复制原数组数值。
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
        - path: str | Path, 待读取的正式或临时 NPZ；object 数组会因 `allow_pickle=False` 直接报错。

    输出:
        - arrays: dict[str, np.ndarray], NPZ 字段名到已脱离文件句柄的数组；字段顺序不属于契约。
    """
    with np.load(Path(path), allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}






# ------------------ 抽象工具函数 ------------------
def _concatenate_entry_values(
    entries: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    offsets_name: str,
) -> dict[str, np.ndarray]:
    """
    按 entry 顺序拼接一组共享第一维的 ragged value 字段, 同时产生 offset 表. 

    输入参数:
        - entries: Sequence[Mapping[str, Any]], 长度 N_entry；每项可包含 `fields` 指定、且在该 entry 内第一维等长的 value 表。
        - fields: Sequence[str], 同一 ragged 模态允许出现的字段名；实际字段集合必须在全部 entry 中一致。
        - offsets_name: str, 输出的 int64 offsets 字段名。

    输出:
        - `{offsets_name}`: int64, (N_entry + 1,), entry i 对应各 value 表的半开区间 `[offsets[i], offsets[i + 1])`，末值为 L_total。
        - 每个实际存在的 `field`: (L_total, ...), 沿第一维按 entry 顺序拼接的 value 表；尾部形状和数据类型沿用首个 entry。

    示例:
        fields = (
            "voxel_index_local_zyx",
            "centered_probability",
            "voxel_final",
        )
        entries = [
            {
                "voxel_index_local_zyx": np.array([[0, 1, 2], [1, 1, 2]], dtype=np.int32),
                "centered_probability": np.array([0.9, 0.8], dtype=np.float32),
                "voxel_final": np.array([[10, 11], [20, 21]], dtype=np.float32),
            },  # 本 entry 有 2 个 voxel
            {
                "voxel_index_local_zyx": np.empty((0, 3), dtype=np.int32),
                "centered_probability": np.empty((0,), dtype=np.float32),
                "voxel_final": np.empty((0, 2), dtype=np.float32),
            },  # 本 entry 有 0 个 voxel
            {
                "voxel_index_local_zyx": np.array([[2, 0, 1]], dtype=np.int32),
                "centered_probability": np.array([0.7], dtype=np.float32),
                "voxel_final": np.array([[30, 31]], dtype=np.float32),
            },  # 本 entry 有 1 个 voxel
        ]
        result = _concatenate_entry_values(entries, fields, "voxel_offsets")

        调用产生一个 `dict[str, np.ndarray]`：
        result["voxel_offsets"]
        # array([0, 2, 2, 3], dtype=int64)
        result["voxel_index_local_zyx"]
        # array([[0, 1, 2], [1, 1, 2], [2, 0, 1]], dtype=int32)
        result["centered_probability"]
        # array([0.9, 0.8, 0.7], dtype=float32)
        result["voxel_final"]
        # array([[10, 11], [20, 21], [30, 31]], dtype=float32)

        `voxel_offsets[i:i + 2]` 给出 entry i 在每个 value 表中的半开区间：entry 0 对应 `[0:2]`，entry 1 对应 `[2:2]`（空区间），entry 2 对应 `[2:3]`。
    """
    if len(entries) == 0:
        # entries=[] 表示没有任何 entry；offsets 仍按 N_entry+1 保持形状 (1,)，唯一的 0 表示总 value 长度为 0。
        # 例如 entries=[] 返回 {"voxel_offsets": np.array([0], dtype=np.int64)}；这不同于有 entry 但全部缺少该字段时返回 {}。
        return {offsets_name: np.zeros(1, dtype=np.int64)}
    # list[str], 当前模态在首个 entry 中实际存在的 value 字段；其余 entry 必须具有完全相同的字段集合。
    present = [field for field in fields if field in entries[0]]
    if not present:
        if any(any(field in entry for field in fields) for entry in entries):
            raise ValueError(f"{offsets_name} 对应字段只在部分 entry 出现")
        return {}
    for entry in entries:
        if any((field in entry) != (field in present) for field in fields):
            raise ValueError(f"{offsets_name} 对应字段必须在所有 entry 中保持相同集合")

    # list[int], 长度 N_entry；第 i 项是 entry i 在全部 `present` value 表中的共同第一维长度。
    lengths: list[int] = []
    # dict[str, list[np.ndarray]], 每个字段按 entry 顺序收集的 value 表，随后统一沿第一维拼接。
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

    for field, values in values_by_field.items():
        exemplar_tail_shape = values[0].shape[1:]
        if any(value.shape[1:] != exemplar_tail_shape for value in values[1:]):
            raise ValueError(f"{field} 在不同 entry 中的尾部形状必须一致")

    # int64, (N_entry + 1,), entry i 对应 `value[offsets[i]:offsets[i + 1]]`；首值为 0，末值为 L_total。
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
    """
    np.asarray([entry[field] for entry in entries], dtype=dtype)

    输入参数:
        - entries: Sequence[Mapping[str, Any]], 长度 N_entry；每项包含尾部形状相同的目标字段。
        - field: str, 要沿新 entry 维堆叠的字段名。
        - dtype: np.dtype, 输出数组的契约数据类型。

    输出:
        - values: np.ndarray, (N_entry, ...), 与 entries 同序的固定尾部形状字段表。
    """
    return np.asarray([entry[field] for entry in entries], dtype=dtype)







# ------------------------------------------------------------- 下面一个是本文件的核心函数 -------------------------------------------------------------
def pack_centered_entries(
    entries: Sequence[Mapping[str, Any]],
    centered_role: str,
    stage1_model_name: str | None = None,
) -> dict[str, np.ndarray]:
    """
    把一个 PDB 的同类 centered entry 聚合为单个数值归档。

    输入参数:
        - entries: Sequence[Mapping[str, Any]], 长度 N_entry；每项包含共同几何、来源身份和变长 voxel 数据；当前模型实际产生 P/A 模态时还包含对应的 P/A 表，`CLG_centered` 还包含候选及其 voxel/A 成员关系。
        - centered_role: str, `F1_centered`、`CLG_centered` 或 `Selected_Refined_Centered`；决定角色专属字段与空条目规则。
        - stage1_model_name: str | None, `STAGE1_MODEL_NAMES` 中的模型来源身份；正式发布必须传入，以校验 Find 的 P/A 模态与 unet_c1 的仅 V 模态。

    输出字段:
        公共 entry 级字段（第一维均为 N_entry）:
        - `centered_box_index`: int32, (N_entry,), 连续的 0..N_entry-1 entry 编号；与全部 entry 级字段第一维对齐。
        - `box_start_zyx`: int32, (N_entry, 3), 每个 80³ BOX 在完整图中的离散 ZYX 起点。
        - `box_shape_zyx`: uint8, (N_entry, 3), 固定为 `[80, 80, 80]` 的 BOX 形状，轴顺序 ZYX。
        - `box_origin_world`: float32, (N_entry, 3), 每个 BOX 起点对应的世界 XYZ 坐标。
        - `voxel_size_world`: float32, (N_entry, 3), 完整图三个世界 XYZ 轴的体素尺寸。
        - `source_tree_id`: int32, (N_entry,), 每个 entry 的来源组件树编号。
        - `source_node_id`: int32, (N_entry,), 每个 entry 在 `source_tree_id` 指定组件树内的局部节点编号；二者共同构成 forest 节点身份。
        - `source_threshold_grid_index`: int32, (N_entry,), 每个来源 forest 节点所在阈值层的网格编号 j，不是体素索引。
        - `source_threshold_value`: float32, (N_entry,), 每个来源 forest 节点的二值化阈值 j/D，不是成员体素概率。

        变长 voxel、P、A 字段：L_voxel、L_aux、L_P、L_A 分别表示拼接后的 voxel、辅助 voxel、P 点和 A 点总行数，C_voxel 与 C_L1..C_L3 分别表示最终 voxel 特征和各层 P/A 特征的通道数；每组 offsets 的第 i 个区间 `[offsets[i], offsets[i + 1])` 对应该 entry 条目的数值行，所有数值字段沿第一维保持对齐；P/A 组仅在当前模型实际产生对应模态时输出。
        - `voxel_offsets`: int64, (N_entry + 1,), `voxel_index_local_zyx`、`centered_probability` 与 `voxel_final` 的 entry 边界。
        - `voxel_index_local_zyx`: int16, (L_voxel, 3), 当前角色权威成员在 80³ BOX 内的局部 voxel 离散坐标，轴顺序 ZYX；不是完整图索引。
        - `centered_probability`: float32, (L_voxel,), 当前 centered 前向经 sigmoid 和模型专属后处理后，按 `voxel_index_local_zyx` 取出的概率；不是原始滑窗完整图概率。
        - `voxel_final`: float16, (L_voxel, C_voxel), 与 `voxel_index_local_zyx` 对齐的最终 voxel 特征；非空归档的 C_voxel 由实际数值表决定，完全空归档使用 `(0, 0)`。
        - `voxel_aux_offsets`: int64, (N_entry + 1,), `voxel_aux_index_local_zyx` 与 `voxel_aux_probability` 的 entry 边界。
        - `voxel_aux_index_local_zyx`: int16, (L_aux, 3), 辅助 voxel 在当前 80³ BOX 内的局部离散索引，轴顺序 ZYX。
        - `voxel_aux_probability`: float32, (L_aux,), 与 `voxel_aux_index_local_zyx` 对齐的辅助 voxel 概率。
        - `P_offsets`: int64, (N_entry + 1,), P 点字段的 entry 边界。
        - `P_coord_local_xyz`: float32, (L_P, 3), Find P 点的 BOX 局部连续 voxel 坐标，轴顺序 XYZ。
        - `P_probability`: float32, (L_P,), 与 P 点坐标对齐的概率。
        - `P_feat_L2`: float16, (L_P, C_L2), P 点对应的 L2 特征。
        - `P_feat_L3`: float16, (L_P, C_L3), P 点对应的 L3 特征。
        - `A_offsets`: int64, (N_entry + 1,), A 点字段的 entry 边界。
        - `A_global_index`: int64, (L_A,), A 点在来源完整图中的全局行号。
        - `A_coord_local_xyz`: float32, (L_A, 3), A 点的 BOX 局部连续 voxel 坐标，轴顺序 XYZ。
        - `A_coord_centered_world`: float32, (L_A, 3), A 点相对 BOX 中心的世界 XYZ 坐标。
        - `A_probability`: float32, (L_A,), 与 A 点坐标对齐的概率。
        - `A_feat_L0`: float32, (L_A, 49), A 点送入 Stage1-Find 点侧嵌入层之前的原始受体特征。
        - `A_feat_L1`: float16, (L_A, C_L1), A 点对应的 L1 特征。
        - `A_feat_L2`: float16, (L_A, C_L2), A 点对应的 L2 特征。
        - `A_feat_L3`: float16, (L_A, C_L3), A 点对应的 L3 特征。

        `CLG_centered` 专属字段（N_candidate 为所有 entry 条目的候选总数；L_candidate_voxel 和 L_candidate_A 分别为所有候选项的 voxel/A 成员索引总数）:
        - `CLG_id`: int32, (N_entry,), 每个 CLG 的编号。
        - `CLG_seed_node_id`: int32, (N_entry,), 每个 CLG 种子节点在 `source_tree_id` 指定 forest 树内的局部编号。
        - `CLG_oldest_node_id`: int32, (N_entry,), 每个 CLG 最早节点在 `source_tree_id` 指定 forest 树内的局部编号。
        - `candidate_offsets`: int64, (N_entry + 1,), entry 的候选边界；entry i 的候选位于 `[candidate_offsets[i], candidate_offsets[i + 1])`。
        - `candidate_node_id`: int32, (N_candidate,), 按 entry 顺序拼接、仅在所属 entry 的 `source_tree_id` 内解释的候选节点编号。
        - `candidate_threshold_grid_index`: int32, (N_candidate,), 与 `candidate_node_id` 对齐的候选冻结阈值网格编号。
        - `candidate_voxel_offsets`: int64, (N_candidate + 1,), 候选项的 voxel 成员边界。
        - `candidate_voxel_index`: int32, (L_candidate_voxel,), 候选项在所属 entry voxel 段内的局部成员行号；既不是 BOX 内坐标，也不是完整图线性索引。
        - `candidate_A_offsets`: int64, (N_candidate + 1,), Find CLG 的候选 A 成员边界；仅在 A 模态存在时输出。
        - `candidate_A_index`: int32, (L_candidate_A,), 候选项在所属 entry A 段内的局部成员行号；仅在 `candidate_A_offsets` 输出时存在。

        `Selected_Refined_Centered` 专属字段:
        - `refine_status`: uint8, (N_entry,), entry 状态码；0/1/2 分别表示 `success`、`empty`、`no_overlap`。
        - `refine_status_names`: Unicode, (3,), 状态码到状态名称的稳定顺序表，顺序为 `success`、`empty`、`no_overlap`。
        对齐与条件规则:
        - Selected 的非 success entry 不得携带任何 voxel/P/A 数值载荷。
        - `unet_c1` 只允许 V 模态，不输出 P/A 字段；Find 模型在存在对应数据载荷时输出完整 P/A 字段组及其 offsets。
    """
    if centered_role not in CENTERED_ROLES:
        raise ValueError(f"未知 centered_role={centered_role!r}")
    if len(entries) == 0:
        arrays = _empty_centered_archive(centered_role)
        validate_centered_archive(arrays, centered_role, stage1_model_name=stage1_model_name)
        return arrays

    # dict[str, np.ndarray], 共同 entry 主表；所有非标量字段第一维 N_entry 与输入 `entries` 完全同序。
    arrays: dict[str, np.ndarray] = {
        "centered_box_index": np.arange(len(entries), dtype=np.int32),
        "box_start_zyx": _stack_entry_field(entries, "box_start_zyx", np.dtype(np.int32)),
        "box_shape_zyx": _stack_entry_field(entries, "box_shape_zyx", np.dtype(np.uint8)),
        "box_origin_world": _stack_entry_field(entries, "box_origin_world", np.dtype(np.float32)),
        "voxel_size_world": _stack_entry_field(entries, "voxel_size_world", np.dtype(np.float32)),
        "source_tree_id": _stack_entry_field(entries, "source_tree_id", np.dtype(np.int32)),
        "source_node_id": _stack_entry_field(entries, "source_node_id", np.dtype(np.int32)),
        "source_threshold_grid_index": _stack_entry_field(entries, "source_threshold_grid_index", np.dtype(np.int32)),
        "source_threshold_value": _stack_entry_field(entries, "source_threshold_value", np.dtype(np.float32)),
    }
    # `voxel_offsets` 同步切分来源或精修 mask 的局部 ZYX 索引、概率和
    # `(N_voxel, C_voxel)` 最终 voxel 特征；C_voxel 直接由实际 entry 字段决定。
    arrays.update(_concatenate_entry_values(entries, VOXEL_VALUE_FIELDS, "voxel_offsets"))
    # `voxel_aux_offsets`、`P_offsets`、`A_offsets` 分别切分辅助 voxel、P 点与
    # A 点的整组逐行对齐值表。
    arrays.update(_concatenate_entry_values(entries, AUX_VALUE_FIELDS, "voxel_aux_offsets"))
    arrays.update(_concatenate_entry_values(entries, P_VALUE_FIELDS, "P_offsets"))
    arrays.update(_concatenate_entry_values(entries, A_VALUE_FIELDS, "A_offsets"))

    if centered_role == "CLG_centered":
        arrays["CLG_id"] = _stack_entry_field(entries, "CLG_id", np.dtype(np.int32))
        arrays["CLG_seed_node_id"] = _stack_entry_field(entries, "CLG_seed_node_id", np.dtype(np.int32))
        arrays["CLG_oldest_node_id"] = _stack_entry_field(entries, "CLG_oldest_node_id", np.dtype(np.int32))
        arrays.update(
            _concatenate_entry_values(
                entries,
                ("candidate_node_id", "candidate_threshold_grid_index"),
                "candidate_offsets",
            )
        )
        # list[int32 array], 长度 N_candidate_total；每项是候选成员在所属 entry 的 `voxel_offsets` 段内的局部行号。
        candidate_voxel_rows: list[np.ndarray] = []
        # list[int32 array], 长度 N_candidate_total；每项是 Find 候选成员在所属 entry 的 `A_offsets` 段内的局部行号。
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
        arrays["refine_status"] = np.asarray([REFINE_STATUS_TO_CODE[status] for status in statuses], dtype=np.uint8)
        arrays["refine_status_names"] = np.asarray(tuple(REFINE_STATUS_TO_CODE), dtype="U10")

    validate_centered_archive(arrays, centered_role, stage1_model_name=stage1_model_name)
    return arrays

def _pack_nested_membership(
    rows: Sequence[np.ndarray],
    prefix: str,
) -> dict[str, np.ndarray]:
    """
    把每个候选项的局部成员行号编码为可切片的 offsets 和 indices。

    输入参数:
        - rows: Sequence[np.ndarray], 长度 N_candidate；列表第 i 项对应第 i 个候选项，可展平为一维局部成员行号；这些行号索引该候选所属 entry 的 voxel 或 A 数值段，而不是拼接后的全局数组。
        - prefix: str, 输出字段前缀，例如 `candidate_voxel` 或 `candidate_A`；函数据此生成 `{prefix}_offsets` 和 `{prefix}_index` 两个字段。

    输出字段:
        - `{prefix}_offsets`: int64, (N_candidate + 1,), 第 i 个候选项的成员位于 `[offsets[i], offsets[i + 1])`；首值为 0，末值为所有成员行号总数 L_membership，空候选项对应相邻且相等的边界值。
        - `{prefix}_index`: int32, (L_membership,), 按候选项顺序拼接的局部成员行号；切片结果仍索引各自所属 entry 的 voxel 或 A 数值段。

    示例:
        rows = (
            np.asarray([2, 5], dtype=np.int32),  # candidate 0 引用所属 entry 的第 2、5 行
            np.empty(0, dtype=np.int32),          # candidate 1 没有成员
            np.asarray([0], dtype=np.int32),      # candidate 2 引用所属 entry 的第 0 行
        )
        result = _pack_nested_membership(rows, "candidate_voxel")
        result["candidate_voxel_offsets"]
        # array([0, 2, 2, 3], dtype=int64)
        result["candidate_voxel_index"]
        # array([2, 5, 0], dtype=int32)

        因此 candidate 0 读取 `candidate_voxel_index[0:2]` 得到 `[2, 5]`，
        candidate 1 读取 `candidate_voxel_index[2:2]` 得到空数组，
        candidate 2 读取 `candidate_voxel_index[2:3]` 得到 `[0]`。
    """
    # int64, (N_candidate,), 每个候选在所属 entry value 段中的局部成员数量。
    lengths = np.asarray([np.asarray(row).size for row in rows], dtype=np.int64)
    # int64, (N_candidate + 1,), candidate i 对应 `indices[offsets[i]:offsets[i + 1]]`；首值为 0，末值为 L_membership。
    offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(lengths)])
    if int(offsets[-1]) == 0:
        indices = np.empty(0, dtype=np.int32)
    else:
        indices = np.concatenate([np.asarray(row, dtype=np.int32).reshape(-1) for row in rows])
    return {f"{prefix}_offsets": offsets, f"{prefix}_index": indices}

def _empty_centered_archive(centered_role: str) -> dict[str, np.ndarray]:
    """
    构造没有候选时仍可正式发布的零 entry centered 归档。

    输入参数:
        - centered_role: str，三个正式 centered role 之一

    输出:
        - arrays: dict[str, np.ndarray], 共同 entry 表第一维为 0，全部已知 ragged offsets 仅含起始 0；因为没有 entry，`voxel_final` 与 Find P/A 模态都不伪造未知通道宽度。
    """
    arrays: dict[str, np.ndarray] = {
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
        "voxel_final": np.empty((0, 0), dtype=np.float16),
        "voxel_aux_offsets": np.zeros(1, dtype=np.int64),
        "voxel_aux_index_local_zyx": np.empty((0, 3), dtype=np.int16),
        "voxel_aux_probability": np.empty(0, dtype=np.float32),
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
        arrays["refine_status_names"] = np.asarray(tuple(REFINE_STATUS_TO_CODE), dtype="U10")
    return arrays







# -------------------------- 检验 --------------------------
def validate_centered_archive(
    arrays: Mapping[str, np.ndarray],
    centered_role: str,
    stage1_model_name: str | None = None,
) -> None:
    """
    校验 centered 聚合归档的 dtype、shape 与 ragged 对齐关系。

    输入参数:
        - arrays: Mapping[str, np.ndarray], 解码后的完整 centered NPZ 字段；各数组必须已经脱离文件句柄。
        - centered_role: str, `F1_centered`、`CLG_centered` 或 `Selected_Refined_Centered`。
        - stage1_model_name: str | None, `STAGE1_MODEL_NAMES` 中的模型来源身份；None 表示只按归档自描述字段判断是否存在 P/A 模态。

    输出:
        - None, 字段集合、数据类型、形状、offsets、坐标索引边界、producer 模态与 role 专属关系全部满足契约时返回；任何磁盘边界错误都会直接抛出。
    """
    if centered_role not in CENTERED_ROLES:
        raise ValueError(f"未知 centered_role={centered_role!r}")
    if stage1_model_name is not None and stage1_model_name not in STAGE1_MODEL_NAMES:
        raise ValueError(f"未知 stage1_model_name={stage1_model_name!r}")
    # tuple[str, ...], 每种 centered role 都必须存在的共同 entry 与 voxel 字段。
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
        "voxel_offsets",
        *VOXEL_VALUE_FIELDS,
        "voxel_aux_offsets",
        *AUX_VALUE_FIELDS,
    )
    if centered_role == "Selected_Refined_Centered":
        required_entry_fields += (
            "refine_status",
            "refine_status_names",
        )
    missing = [field for field in required_entry_fields if field not in arrays]
    if missing:
        raise KeyError(f"centered 归档缺少共同字段: {missing}")
    # int, 共同 entry 表行数；所有 entry-level 数组和主 ragged offsets 均以它为基准。
    n_entry = int(np.asarray(arrays["centered_box_index"]).shape[0])
    # dict[str, dtype], 所有可选和必选字段的盘上数据类型契约；仅校验当前归档实际存在的字段。
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
        "A_offsets": np.int64,
        "A_global_index": np.int64,
        "A_coord_local_xyz": np.float32,
        "A_coord_centered_world": np.float32,
        "A_probability": np.float32,
        "A_feat_L0": np.float32,
        "A_feat_L1": np.float16,
        "A_feat_L2": np.float16,
        "A_feat_L3": np.float16,
        "refine_status": np.uint8,
    }
    for field, dtype in expected_dtypes.items():
        if field in arrays and np.asarray(arrays[field]).dtype != np.dtype(dtype):
            raise ValueError(f"{field}.dtype 必须为 {np.dtype(dtype)}")
    if not np.array_equal(np.asarray(arrays["centered_box_index"]), np.arange(n_entry, dtype=np.int32)):
        raise ValueError("centered_box_index 必须是连续 int32 的 0..N_entry-1")
    # dict[str, tuple[int, ...]], 共同 entry 级字段形状；第一维必须与 `centered_box_index` 的 N_entry 对齐。
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
    # int, 拼接后的权威来源/精修 voxel value 表总行数 L_voxel。
    voxel_length = int(np.asarray(arrays["voxel_index_local_zyx"]).shape[0])
    validate_offsets(np.asarray(arrays["voxel_offsets"]), voxel_length, "voxel_offsets")
    if np.asarray(arrays["voxel_offsets"]).shape != (n_entry + 1,):
        raise ValueError("voxel_offsets 长度必须为 N_entry+1")
    voxel_index = np.asarray(arrays["voxel_index_local_zyx"])
    if voxel_index.shape != (voxel_length, 3):
        raise ValueError("voxel_index_local_zyx 必须为 [L_voxel,3]")
    if np.asarray(arrays["centered_probability"]).shape != (voxel_length,):
        raise ValueError("centered_probability 必须为 [L_voxel]")
    voxel_final = np.asarray(arrays["voxel_final"])
    if voxel_final.ndim != 2 or voxel_final.shape[0] != voxel_length:
        raise ValueError("voxel_final 必须为 [L_voxel,C_voxel]")
    if voxel_length > 0 and voxel_final.shape[1] <= 0:
        raise ValueError("存在 voxel 时，voxel_final 的 C_voxel 必须为正整数")
    if voxel_index.size and (int(voxel_index.min()) < 0 or int(voxel_index.max()) >= 80):
        raise ValueError("voxel_index_local_zyx 必须位于当前 80³ BOX")
    # int64, (N_entry + 1,), 同步切分局部 ZYX 索引、概率与 C_voxel 维 voxel_final 的 entry 边界。
    voxel_offsets = np.asarray(arrays["voxel_offsets"])
    for entry_index in range(n_entry):
        rows = voxel_index[int(voxel_offsets[entry_index]) : int(voxel_offsets[entry_index + 1])]
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
    needs_find_payload = stage1_model_name in FIND_MODEL_NAMES and n_entry > 0
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
        if any(np.asarray(arrays[field]).ndim != 2 for field in ("P_feat_L2", "P_feat_L3")):
            raise ValueError("P_feat_L2/L3 必须为二维 feature 表")
    if all(field in arrays for field in a_fields):
        a_length = int(np.asarray(arrays["A_probability"]).shape[0])
        for field in ("A_coord_local_xyz", "A_coord_centered_world"):
            if np.asarray(arrays[field]).shape != (a_length, 3):
                raise ValueError(f"{field} 必须为 [L_A,3]")
        if np.asarray(arrays["A_global_index"]).shape != (a_length,):
            raise ValueError("A_global_index 必须为 [L_A]")
        if np.asarray(arrays["A_feat_L0"]).shape != (a_length, 49):
            raise ValueError("A_feat_L0 必须为 [L_A,49]")
        if any(
            np.asarray(arrays[field]).ndim != 2
            for field in ("A_feat_L1", "A_feat_L2", "A_feat_L3")
        ):
            raise ValueError("A_feat_L1/L2/L3 必须为二维 feature 表")

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
        # int, 全部 CLG entry 按 `candidate_offsets` 拼接后的候选总数 N_candidate。
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
        elif stage1_model_name in FIND_MODEL_NAMES and n_entry > 0:
            raise KeyError("Find CLG_centered 缺少 candidate_A_offsets/candidate_A_index")

    if centered_role == "F1_centered":
        forbidden = ("CLG_id", "candidate_offsets", "refine_status")
        if any(field in arrays for field in forbidden):
            raise ValueError("F1_centered 不得伪造 CLG、candidate 或 refine 字段")
    if centered_role == "Selected_Refined_Centered":
        if "refine_status" not in arrays or np.asarray(arrays["refine_status"]).shape != (n_entry,):
            raise ValueError("Selected_Refined_Centered 必须有 [N_entry] refine_status")
        status = np.asarray(arrays["refine_status"])
        if status.dtype != np.dtype(np.uint8) or bool(np.any(status > 2)):
            raise ValueError("refine_status 必须是仅含 0..2 的 uint8")
        voxel_offsets = np.asarray(arrays["voxel_offsets"])
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

def validate_offsets(
    offsets: np.ndarray,
    value_length: int,
    offsets_name: str,
) -> None:
    """
    校验一个 offsets 数组完整切分指定 value 表。

    输入参数:
        - offsets: int64, (N_segment + 1,), 对指定 value 表第一维的半开区间边界；首值必须为 0，末值必须为 `value_length`。
        - value_length: int, 被切分 value 表的第一维总长度。
        - offsets_name: str, 报错时使用的具体字段名。

    输出:
        - None, dtype、维数、首末值与单调性全部满足契约时返回。
    """
    # int64, (N_segment + 1,), 对 value 表第一维的半开区间边界；允许相邻边界相等以表示空段。
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
