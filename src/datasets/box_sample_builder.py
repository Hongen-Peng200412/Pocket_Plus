# -*- coding: utf-8 -*-
"""
=============================================================================
box_sample_builder — 共享 BOX 样本构建器
=============================================================================
从 BoxPointDataset.__getitem__() 和 parse_input.split_volume_to_boxes() 中提取的公共的："给定一个 BOX 的 voxel 数据 + 全局原子信息 → 标准 sample dict"逻辑。

训练侧 (BoxPointDataset) 与推断侧 (src/inference/) 均调用本模块，从而在结构上强制保证样本契约完全一致。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .box_geometry import (
    build_atom_coordinates,
    build_atom_features,
    build_atom_valid_mask,
    build_hardmask_from_atom_coordinates,
    build_voxel_valid_mask,
    select_atoms_for_box,
)


def _apply_class_mapping(label: np.ndarray, mapping: list[int]) -> np.ndarray:
    """
    将原始类别 ID 映射成新的类别 ID。

    输入参数:
        - label: np.ndarray, 任意形状(通常为 (D, H, W) 或 (N,)), int64, 原始标签
        - mapping: list[int], 映射表, 索引为原始类别 ID, 值为新类别 ID

    输出:
        - np.ndarray, 与 label 同形状, int64, 映射后的标签
    """
    mapped = np.zeros_like(label, dtype=np.int64)
    for old_id, new_id in enumerate(mapping):
        mapped[label == old_id] = int(new_id)
    return mapped




def build_box_point_numpy_sample(
    voxel_grid: np.ndarray,
    voxel_label: np.ndarray,
    atom_coords_world_full: np.ndarray,
    atom_features_raw_full: np.ndarray,
    atom_labels_full: np.ndarray,
    box_origin_world: np.ndarray,
    voxel_size_world: np.ndarray,
    box_shape_zyx: np.ndarray,
    atom_buffer_radius: float,
    valid_crop_margin: int,
    class_mapping: list[int] | None,
) -> dict[str, Any]:
    """
    输入全局数据, 构造与 BoxPointDataset.__getitem__() 完全同构的单样本 numpy dict。

    内部流程:
        1. select_atoms_for_box
        2. build_atom_coordinates
        3. build_hardmask_from_atom_coordinates
        4. build_voxel_valid_mask
        5. build_atom_valid_mask
        6. build_atom_features
        7. 提取 + class_mapping atom_label
        8. 组装 sample dict

    输入参数:
        - voxel_grid: np.ndarray, (C, D, H, W), float32, 已拼接并归一化的体素特征
        - voxel_label: np.ndarray, (D, H, W), int64, 体素标签(推断时传全零占位)
        - atom_coords_world_full: np.ndarray, (N_all, 3), float32, 全局原子坐标 (x,y,z)
        - atom_features_raw_full: np.ndarray, (N_all, F), float32, 全局原子特征
        - atom_labels_full: np.ndarray, (N_all,), int64, 全局原子标签(推断时传全零)
        - box_origin_world: np.ndarray, (3,), float32
        - voxel_size_world: np.ndarray, (3,), float32
        - box_shape_zyx: np.ndarray, (3,), int64
        - atom_buffer_radius: float, 标量, 原子 buffer 半径, 建议值 4.0
        - valid_crop_margin: int, 标量, 边界裁边宽度, 建议值 2
        - class_mapping: list[int] | None, 只作用于 atom_labels_full

    输出:
        - sample: dict[str, Any], 纯 numpy 版本的样本字典, 包含以下字段:
            - "voxel_grid":              np.ndarray, (C, D, H, W), float32, 直接透传输入
            - "voxel_label":             np.ndarray, (D, H, W), int64, 直接透传输入
            - "hardmask":                np.ndarray, (D, H, W), int64, 几何 hardmask
            - "voxel_valid_mask":        np.ndarray, (D, H, W), bool, 裁边后的体素监督区域
            - "box_origin_world":        np.ndarray, (3,), float32, 直接透传输入
            - "voxel_size_world":        np.ndarray, (3,), float32, 直接透传输入
            - "box_shape_zyx":           np.ndarray, (3,), int64, 直接透传输入
            - "atom_coord_world":        np.ndarray, (N, 3), float32, 选中原子的世界坐标
            - "atom_coord_local_voxel":  np.ndarray, (N, 3), float32, 选中原子的体素坐标
            - "atom_coord_centered_world": np.ndarray, (N, 3), float32, 选中原子相对 BOX 中心的世界坐标
            - "atom_feat":               np.ndarray, (N, F), float32, 选中原子的特征
            - "atom_label":              np.ndarray, (N,), int64, 选中原子的标签(经 class_mapping)
            - "atom_is_in_core_box":     np.ndarray, (N,), bool, 标记选中原子是否处于 BOX core 区域
            - "atom_valid_mask":         np.ndarray, (N,), bool, 标记选中原子是否参与损失监督
            - "_selected_idx":           np.ndarray, (N,), int64, 选中原子在全局数组中的索引(推断侧用于获取全局索引; 训练侧可忽略)

        注意: "ligand_dist_map" 不在本函数输出范围内, 它由调用侧
        (BoxPointDataset.__getitem__) 在本函数返回后按需追加。
    """
    # 1. 选择 BOX 内 + buffer 原子
    selected = select_atoms_for_box(
        atom_coords_world=atom_coords_world_full,
        box_origin_world=box_origin_world,
        voxel_size_world=voxel_size_world,
        box_shape_zyx=box_shape_zyx,
        buffer_radius=atom_buffer_radius,
    )
    # np.ndarray, (N_selected,), int64
    selected_idx = selected["selected_idx"]
    # np.ndarray, (N_selected,), bool
    atom_is_in_core_box = selected["atom_is_in_core_box"]

    # 2. 三套坐标
    coord_data = build_atom_coordinates(
        atom_coords_world=atom_coords_world_full,
        selected_idx=selected_idx,
        box_origin_world=box_origin_world,
        voxel_size_world=voxel_size_world,
        box_shape_zyx=box_shape_zyx,
    )

    # 3. hardmask
    hardmask = build_hardmask_from_atom_coordinates(
        atom_coord_local_voxel=coord_data["atom_coord_local_voxel"],
        atom_is_in_core_box=atom_is_in_core_box,
        box_shape_zyx=box_shape_zyx,
    )

    # 4. voxel_valid_mask
    voxel_valid_mask = build_voxel_valid_mask(
        box_shape_zyx=box_shape_zyx,
        valid_crop_margin=valid_crop_margin,
    )

    # 5. atom_valid_mask
    atom_valid_mask = build_atom_valid_mask(
        atom_coord_local_voxel=coord_data["atom_coord_local_voxel"],
        atom_is_in_core_box=atom_is_in_core_box,
        box_shape_zyx=box_shape_zyx,
        valid_crop_margin=float(valid_crop_margin),
    )

    # 6. atom_feat
    atom_feat = build_atom_features(
        atom_features_raw=atom_features_raw_full,
        selected_idx=selected_idx,
    )

    # 7. atom_label + class_mapping
    # np.ndarray, (N_selected,), int64
    atom_label = atom_labels_full[selected_idx].astype(np.int64, copy=False)
    if class_mapping is not None:
        atom_label = _apply_class_mapping(atom_label, class_mapping)

    # 8. 组装
    return {
        "voxel_grid": voxel_grid,
        "voxel_label": voxel_label,
        "hardmask": hardmask,
        "voxel_valid_mask": voxel_valid_mask,
        "box_origin_world": box_origin_world,
        "voxel_size_world": voxel_size_world,
        "box_shape_zyx": box_shape_zyx,
        "atom_coord_world": coord_data["atom_coord_world"],
        "atom_coord_local_voxel": coord_data["atom_coord_local_voxel"],
        "atom_coord_centered_world": coord_data["atom_coord_centered_world"],
        "atom_feat": atom_feat,
        "atom_label": atom_label,
        "atom_is_in_core_box": atom_is_in_core_box,
        "atom_valid_mask": atom_valid_mask,
        # 推断侧使用; 训练侧可忽略
        "_selected_idx": selected_idx,
    }


# ---- dtype 映射表: 字段名 → torch dtype ----
_TENSOR_DTYPE_MAP: dict[str, torch.dtype] = {
    "voxel_grid": torch.float32,
    "voxel_label": torch.int64,
    "hardmask": torch.int64,
    "voxel_valid_mask": torch.bool,
    "box_origin_world": torch.float32,
    "voxel_size_world": torch.float32,
    "box_shape_zyx": torch.int64,
    "atom_coord_world": torch.float32,
    "atom_coord_local_voxel": torch.float32,
    "atom_coord_centered_world": torch.float32,
    "atom_feat": torch.float32,
    "atom_label": torch.int64,
    "atom_is_in_core_box": torch.bool,
    "atom_valid_mask": torch.bool,
    # 可选字段: 由 BoxPointDataset.__getitem__ 追加, 仅在配置了 ligand_dist_BOX 时存在
    "ligand_dist_map": torch.float32,
}


def to_torch_sample(sample_dict: dict[str, Any]) -> dict[str, Any]:
    """
    将 build_box_point_numpy_sample 返回的 numpy dict 转为 torch tensor dict。

    已知 tensor 字段按 _TENSOR_DTYPE_MAP 做显式转换,
    其余字段(元信息 str/int/bool, 以及推断专用 sidecar)原样保留。

    输入参数:
        - sample_dict: dict[str, Any], 含有 numpy 数组和元信息的样本字典

    输出:
        - dict[str, Any], tensor 字段已转为 torch.Tensor, 其余不变
    """
    result: dict[str, Any] = {}
    for key, value in sample_dict.items():
        if key in _TENSOR_DTYPE_MAP:
            result[key] = torch.tensor(value, dtype=_TENSOR_DTYPE_MAP[key])
        else:
            # 元信息(str/int/bool)和推断专用字段(如 _selected_idx)原样保留
            result[key] = value
    return result
