# -*- coding: utf-8 -*-
"""
=============================================================================
box_geometry — 共享几何工具函数
=============================================================================
从 BoxPointDataset 中提取的、不依赖任何 self 数据成员的纯函数。
训练侧 (BoxPointDataset) 与推断侧 (src/inference/) 均可直接调用。

所有函数的签名、注释与内部逻辑严格保持与原 BoxPointDataset 方法一致，只是将 `self` 和隐式读取的 `self.xxx` 改为显式参数。
"""

import numpy as np


def select_atoms_for_box(
    atom_coords_world: np.ndarray,
    box_origin_world: np.ndarray,
    voxel_size_world: np.ndarray,
    box_shape_zyx: np.ndarray,
    buffer_radius: float,
) -> dict[str, np.ndarray]:
    """
    先选 box 内原子，再从这个BOX额外地扩张 buffer 内原子。

    输入:
        - atom_coords_world: numpy.ndarray, 形如 (N_atom, 3), 表示原子在世界坐标系下的三维坐标，顺序为 (x, y, z)。
        - box_origin_world: numpy.ndarray, 形如 (3,), 表示当前 BOX 左下近角点在世界坐标系下的坐标，顺序为 (x, y, z)。
        - voxel_size_world: numpy.ndarray, 形如 (3,), 表示当前 BOX 中每个体素的世界物理尺寸，顺序为 (x, y, z)。
        - box_shape_zyx: numpy.ndarray, 形如 (3,), 表示当前 BOX 的体素网格大小，顺序为 (Z, Y, X)，对应 (depth, height, width)。
        - buffer_radius: float, 标量, 表示在 box 外部额外扩展接受原子的世界坐标半径。

    输出:
        - 返回一个 dict[str, numpy.ndarray] 字典:
            - "selected_idx": numpy.ndarray, 形如 (N_selected,), int64，表示被选中（包含在 box 或扩展 buffer 内）的原子的原始(全局)索引。
            - "atom_is_in_core_box": numpy.ndarray, 形如 (N_selected,), bool，表示被选中的原子是否位于 box 内(另一部分是buffer外边界)
    """
    box_shape_xyz = box_shape_zyx[[2, 1, 0]].astype(np.float32)   # Z/Y/X -> X/Y/Z
    # numpy.ndarray, (3,), 类型 float32，表示当前 BOX 在世界坐标系下的最大边界坐标 (x_max, y_max, z_max)
    box_max_world = box_origin_world + box_shape_xyz * voxel_size_world

    # core_mask: 落在当前 BOX 内部的原子。
    # numpy.ndarray, (N_atom,), 类型 bool，将会标记哪些原子在当前 BOX 内
    core_mask = np.all(atom_coords_world >= box_origin_world[None, :], axis=1)
    core_mask &= np.all(atom_coords_world < box_max_world[None, :], axis=1)

    buffer_min_world = box_origin_world - float(buffer_radius)
    buffer_max_world = box_max_world + float(buffer_radius)
    # numpy.ndarray, (N_atom,), 类型 bool，将会表示外扩 buffer 最小边界内的原子
    selected_mask = np.all(atom_coords_world >= buffer_min_world[None, :], axis=1)
    selected_mask &= np.all(atom_coords_world < buffer_max_world[None, :], axis=1)

    # numpy.ndarray, (N_selected,), 类型 int64，从 selected_mask 提取出的选中原子的下标数组
    # 当传入一个 1D 布尔数组时，np.where(selected_mask) 返回的是一个元组，例如 (array([1, 4, 5...]),), 其中的第一个元素就是所有值为 True 的元素的下标数组。加上 [0] 把这个下标数组单独提取出来。
    selected_idx = np.where(selected_mask)[0].astype(np.int64)
    # numpy.ndarray, (N_selected,), 类型 bool，对应于选中的原子的是否在 box 内(另一部分是buffer外边界)
    atom_is_in_core_box = core_mask[selected_idx]
    
    return {
        "selected_idx": selected_idx,
        "atom_is_in_core_box": atom_is_in_core_box.astype(bool, copy=False),
    }


def build_atom_coordinates(
    atom_coords_world: np.ndarray,
    selected_idx: np.ndarray,
    box_origin_world: np.ndarray,
    voxel_size_world: np.ndarray,
    box_shape_zyx: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    为选中的原子同时生成三套坐标表示。

    返回的三套坐标分别是:
        - `atom_coord_world`: 世界坐标，顺序 `(x, y, z)`。
        - `atom_coord_local_voxel`: 连续 voxel 坐标，顺序仍是 `(x, y, z)`，数值语义是"距离 BOX (左下)原点多少个 voxel"。
          这里保留的是角点语义：若原子正好位于第 `i/j/k` 个 voxel 的中心，则坐标应为 `(i+0.5, j+0.5, k+0.5)`。
        - `atom_coord_centered_world`: 以 BOX 中心为原点的世界坐标，顺序 `(x, y, z)`。

    输入:
        - atom_coords_world: numpy.ndarray, 形如 (N_atom, 3), 所有原子的世界坐标。
        - selected_idx: numpy.ndarray, 形如 (N_selected,), 选中的原子索引。
        - box_origin_world: numpy.ndarray, 形如 (3,), BOX的原点世界坐标 (x,y,z)。
        - voxel_size_world: numpy.ndarray, 形如 (3,), BOX的世界物理体素大小 (x,y,z)。
        - box_shape_zyx: numpy.ndarray, 形如 (3,), BOX的体素网格大小，顺序 (Z,Y,X)。

    输出:
        - dict[str, numpy.ndarray] 字典:
            - "atom_coord_world": numpy.ndarray, (N_selected, 3), 原子的纯世界坐标 (X,Y,Z)。
            - "atom_coord_local_voxel": numpy.ndarray, (N_selected, 3), 连续的体素网格空间坐标 (X,Y,Z), 采用角点语义。
            - "atom_coord_centered_world": numpy.ndarray, (N_selected, 3), 以 BOX 中心为原点的世界坐标 (X,Y,Z)。
    """
    # numpy.ndarray, (N_selected, 3), float32 类型，选中原子的世界坐标
    selected_coord_world = atom_coords_world[selected_idx].astype(np.float32, copy=False)
    # numpy.ndarray, (N_selected, 3), float32 类型，基于 BOX 原点及 voxel 大小映射后的连续体素坐标(角点语义)
    atom_coord_local_voxel = (
        (selected_coord_world - box_origin_world[None, :]) / voxel_size_world[None, :]
    ).astype(np.float32, copy=False)
    box_shape_xyz = box_shape_zyx[[2, 1, 0]].astype(np.float32)
    # numpy.ndarray, (3,), float32 类型，BOX 的中心世界坐标
    box_center_world = box_origin_world + 0.5 * box_shape_xyz * voxel_size_world
    # numpy.ndarray, (N_selected, 3), float32 类型，每个原子相对于当前 BOX 中心的世界坐标
    atom_coord_centered_world = selected_coord_world - box_center_world[None, :]

    return {
        "atom_coord_world": selected_coord_world,
        "atom_coord_local_voxel": atom_coord_local_voxel,
        "atom_coord_centered_world": atom_coord_centered_world.astype(np.float32, copy=False),
    }


def build_atom_features(
    atom_features_raw: np.ndarray,
    selected_idx: np.ndarray,
) -> np.ndarray:
    """
    构造 atom_feat。
    第一版保持 atom 原始特征不变，不把 `core_flag` 或监督 mask 混入底层 `atom_feat`。

    输入:
        - atom_features_raw: numpy.ndarray, 形如 (N_atom, F_raw), 所有原子的原始特征。
        - selected_idx: numpy.ndarray, 形如 (N_selected,), 被选中原子的下标。

    输出:
        - numpy.ndarray, 形如 (N_selected, F_raw), 最终传入 point branch 的原子级别特征。
    """
    # numpy.ndarray, (N_selected, F_raw), float32 类型，取出被选中原子的原始特征，F_raw 一般为 49
    selected_raw = atom_features_raw[selected_idx].astype(np.float32, copy=False)        # (N_selected, F_raw)
    return selected_raw.astype(np.float32, copy=False)


def build_atom_valid_mask(
    atom_coord_local_voxel: np.ndarray,
    atom_is_in_core_box: np.ndarray,
    box_shape_zyx: np.ndarray,
    valid_crop_margin: float,
) -> np.ndarray:
    """
    构造 atom 监督掩码。

    只有同时满足下面两个条件的 atom 才参与监督:
        1. 位于 core box 内部。
        2. 在连续 voxel 坐标意义下，落在裁掉 `valid_crop_margin` 后的有效监督区域内。

    输入:
        - atom_coord_local_voxel: numpy.ndarray, 形如 (N_selected, 3), atom 的连续局部 voxel 坐标，顺序为 (x, y, z)。
        - atom_is_in_core_box: numpy.ndarray, 形如 (N_selected,), 标记该 atom 是否在 core box 内，True 即在内部。
        - box_shape_zyx: numpy.ndarray, 形如 (3,), BOX 网格大小，顺序为 (Z, Y, X)。
        - valid_crop_margin: float, 标量, 在六个面各无条件裁掉的 voxel 层数。

    输出:
        - numpy.ndarray, 形如 (N_selected,), 类型 bool，表示这些原子在训练时是否作为监督信号纳入 loss 中。
    """
    atom_is_in_core_box = atom_is_in_core_box.astype(bool, copy=False)
    margin = float(valid_crop_margin)
    if atom_coord_local_voxel.shape[0] == 0:
        return atom_is_in_core_box.astype(bool, copy=True)
    if margin <= 0:
        return atom_is_in_core_box.astype(bool, copy=True)

    box_shape_xyz = box_shape_zyx[[2, 1, 0]].astype(np.float32, copy=False)
    inside_lower = atom_coord_local_voxel >= margin
    inside_upper = atom_coord_local_voxel < (box_shape_xyz[None, :] - margin)
    inside_valid_crop = np.all(inside_lower & inside_upper, axis=1)
    return np.logical_and(atom_is_in_core_box, inside_valid_crop).astype(bool, copy=False)


def build_voxel_valid_mask(
    box_shape_zyx: np.ndarray,
    valid_crop_margin: int,
) -> np.ndarray:
    """
    构造 voxel 辅助监督掩码: 在六个面各无条件裁掉 `valid_crop_margin` 个 voxel。

    输入:
        - box_shape_zyx: numpy.ndarray, 形如 (3,), 表示体素数组空间大小信息，顺序为 (depth, height, width)。
        - valid_crop_margin: int, 标量, 在六个面各裁掉的 voxel 层数。

    输出:
        - numpy.ndarray, 形如 (D, H, W), 类型 bool。`True` 表示该位置参与 voxel 监督，`False` 表示该位置被边界裁掉。
    """
    # int 标量类型，分别代表 Z, Y, X 三个维度在体素空间中的格子数量
    depth, height, width = [int(v) for v in box_shape_zyx.tolist()]
    # numpy.ndarray, (D, H, W), 类型 bool，初始化全为 True 的初始体素掩码
    voxel_valid_mask = np.ones((depth, height, width), dtype=bool)
    # int 标量类型，从类的属性中获取需要向内裁剪的边缘层数
    margin = int(valid_crop_margin)
    if margin <= 0:
        return voxel_valid_mask

    # 六个面各裁掉 margin 个 voxel
    voxel_valid_mask[: min(margin, depth), :, :] = False
    voxel_valid_mask[max(depth - margin, 0) :, :, :] = False
    voxel_valid_mask[:, : min(margin, height), :] = False
    voxel_valid_mask[:, max(height - margin, 0) :, :] = False
    voxel_valid_mask[:, :, : min(margin, width)] = False
    voxel_valid_mask[:, :, max(width - margin, 0) :] = False

    return voxel_valid_mask


def build_hardmask_from_atom_coordinates(
    atom_coord_local_voxel: np.ndarray,
    atom_is_in_core_box: np.ndarray,
    box_shape_zyx: np.ndarray,
) -> np.ndarray:
    """
    基于原子几何位置构造 voxel hardmask。

    hardmask 的语义固定为:
        - 只统计 core box 内的原子
        - 某个 voxel 只要落入至少一个 core atom 的 home voxel, 则记为 1
        - 与 `pdb_feature_BOX` 是否存在、是否参与 `voxel_grid` 拼接无关

    输入:
        - atom_coord_local_voxel: numpy.ndarray, 形如 (N_selected, 3), 原子的连续局部 voxel 坐标, 顺序为 (x, y, z), 采用 corner 语义
        - atom_is_in_core_box: numpy.ndarray, 形如 (N_selected,), bool, 标记每个原子是否位于当前 core box 内
        - box_shape_zyx: numpy.ndarray, 形如 (3,), 当前 BOX 的空间大小, 顺序为 (Z, Y, X)

    输出:
        - numpy.ndarray, 形如 (D, H, W), int64, 几何定义的 hardmask, 取值 0 或 1
    """
    # int, int, int, 当前 BOX 的空间尺寸
    depth, height, width = [int(v) for v in np.asarray(box_shape_zyx, dtype=np.int64).tolist()]
    # numpy.ndarray, (D, H, W), int64, 初始化为全零的几何 hardmask
    hardmask = np.zeros((depth, height, width), dtype=np.int64)

    # numpy.ndarray, (N_selected,), bool, 仅保留 core box 内原子
    core_mask = np.asarray(atom_is_in_core_box, dtype=bool)
    if atom_coord_local_voxel.shape[0] == 0 or not np.any(core_mask):
        return hardmask

    # numpy.ndarray, (N_core, 3), float32, core 原子的连续局部 voxel 坐标
    core_coord_local_voxel = np.asarray(atom_coord_local_voxel[core_mask], dtype=np.float32)
    # numpy.ndarray, (N_core, 3), int64, 用 floor 得到原子的 home voxel 索引, 顺序为 (x, y, z)
    voxel_idx_xyz = np.floor(core_coord_local_voxel).astype(np.int64)
    # numpy.ndarray, (3,), int64, BOX 空间大小, 顺序为 (x, y, z)
    box_shape_xyz = np.asarray([width, height, depth], dtype=np.int64)

    # numpy.ndarray, (N_core,), bool, 过滤浮点边界误差或异常坐标导致的越界索引
    valid_mask = np.all(voxel_idx_xyz >= 0, axis=1)
    valid_mask &= voxel_idx_xyz[:, 0] < box_shape_xyz[0]
    valid_mask &= voxel_idx_xyz[:, 1] < box_shape_xyz[1]
    valid_mask &= voxel_idx_xyz[:, 2] < box_shape_xyz[2]
    if not np.any(valid_mask):
        return hardmask

    # numpy.ndarray, (N_valid, 3), int64, 合法的 home voxel 索引
    voxel_idx_xyz = voxel_idx_xyz[valid_mask]
    hardmask[voxel_idx_xyz[:, 2], voxel_idx_xyz[:, 1], voxel_idx_xyz[:, 0]] = 1
    return hardmask


def build_hardmask_from_world_coordinates(
    atom_coords_world: np.ndarray,
    box_origin_world: np.ndarray,
    voxel_size_world: np.ndarray,
    box_shape_zyx: np.ndarray,
) -> np.ndarray:
    """
    基于世界坐标下的原子位置直接构造 voxel hardmask。

    输入:
        - atom_coords_world: numpy.ndarray, 形如 (N_atom, 3), 原子的世界坐标, 顺序为 (x, y, z)
        - box_origin_world: numpy.ndarray, 形如 (3,), 当前 BOX 的世界坐标原点, 顺序为 (x, y, z)
        - voxel_size_world: numpy.ndarray, 形如 (3,), 当前 BOX 的体素尺寸, 顺序为 (x, y, z)
        - box_shape_zyx: numpy.ndarray, 形如 (3,), 当前 BOX 的空间大小, 顺序为 (Z, Y, X)

    输出:
        - numpy.ndarray, 形如 (D, H, W), int64, 几何定义的 hardmask, 取值 0 或 1
    """
    # numpy.ndarray, (N_atom, 3), float32, 原子相对当前 BOX 原点的连续局部 voxel 坐标
    atom_coord_local_voxel = (
        (np.asarray(atom_coords_world, dtype=np.float32) - np.asarray(box_origin_world, dtype=np.float32)[None, :])
        / np.asarray(voxel_size_world, dtype=np.float32)[None, :]
    ).astype(np.float32, copy=False)

    # numpy.ndarray, (3,), float32, BOX 大小, 顺序为 (x, y, z)
    box_shape_xyz = np.asarray(box_shape_zyx, dtype=np.float32)[[2, 1, 0]]
    # numpy.ndarray, (N_atom,), bool, 判断原子是否落在当前 core box 内
    atom_is_in_core_box = np.all(atom_coord_local_voxel >= 0.0, axis=1)
    atom_is_in_core_box &= np.all(atom_coord_local_voxel < box_shape_xyz[None, :], axis=1)

    return build_hardmask_from_atom_coordinates(
        atom_coord_local_voxel=atom_coord_local_voxel,
        atom_is_in_core_box=atom_is_in_core_box,
        box_shape_zyx=np.asarray(box_shape_zyx, dtype=np.int64),
    )


def resolve_emdb_zscore_mask(emdb_z_score, n_emdb_channels: int) -> list[bool]:
    """
    将 emdb_z_score 参数解析为逐通道归一化掩码。

    输入参数:
        - emdb_z_score: bool | int | list[int], 归一化控制
            - false/0: 全部不归一化
            - true/1: 全部归一化
            - list[int]: 逐通道控制(1=归一化, 0=跳过), 长度须等于 n_emdb_channels
        - n_emdb_channels: int, 标量, EMDB 通道总数

    输出:
        - mask: list[bool], 长度 = n_emdb_channels, True=归一化该通道
    """
    if isinstance(emdb_z_score, (bool, int, float)):
        flag = bool(emdb_z_score)
        return [flag] * n_emdb_channels
    if isinstance(emdb_z_score, (list, tuple)):
        if len(emdb_z_score) != n_emdb_channels:
            raise ValueError(
                f"emdb_z_score 的长度 ({len(emdb_z_score)}) "
                f"与 EMDB 通道数 ({n_emdb_channels}) 不一致"
            )
        return [bool(v) for v in emdb_z_score]
    raise TypeError(
        f"emdb_z_score 必须是 bool/int 标量或 list[int], 当前类型: {type(emdb_z_score)}"
    )
