# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any

import torch


# -----------------------------------------------------------------------------
# 本文件负责把 BoxPointDataset 返回的 `list[dict]` 单样本，
# 整理成 stage1 模型/wrapper 直接可消费的 `dict batch`。
#
# 约定:
# 1. voxel 相关字段在一个 batch 内形状固定，因此直接 `torch.stack`
# 2. atom 相关字段是变长的，因此沿第 0 维 `torch.cat`
# 3. 额外生成 `atom_batch_index / atom_counts / atom_offsets` 供后续 point 分支、pack/unpack、按样本还原时使用
# 4. 非 tensor 的元信息(如 sample_name / pdb_id / class_name)保留为 list
# -----------------------------------------------------------------------------


_VOXEL_STACK_FIELDS = (
    "voxel_grid",
    "voxel_label",
    "hardmask",
    "voxel_valid_mask",
    "box_origin_world",
    "voxel_size_world",
    "box_shape_zyx",
)

_ATOM_CONCAT_FIELDS = (
    "atom_coord_world",
    "atom_coord_local_voxel",
    "atom_coord_centered_world",
    "atom_feat",
    "atom_label",
    "atom_is_in_core_box",
    "atom_valid_mask",
)


def box_point_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    将 `list[dict]` 的单样本组装成 stage1 需要的 `dict batch`。

    输入参数 / Args:
        - batch: list[dict[str, Any]], DataLoader 默认传入的一个 batch, 其中每个元素都是 `BoxPointDataset.__getitem__()` 返回的单样本字典.

    输出 / Return:
        - batch_dict: dict[str, Any]
            主要字段数据类型、形状、意义如下:
            1. voxel 部分 (固定大小):
               - "voxel_grid":       torch, (B, C, D, H, W), float32, 拼接后的多通道体素特征网格
               - "voxel_label":      torch, (B, D, H, W),    int64,   拼接后的体素分类真值标签
               - "hardmask":         torch, (B, D, H, W),    int64,   标记该体素是否物理空间真实存在有效结构的掩码
               - "voxel_valid_mask": torch, (B, D, H, W),    bool,    去除边缘 margin 之后参与 loss 计算的核心监督区域掩码
               - "box_origin_world": torch, (B, 3),          float32, 各样本 BOX 在世界坐标系下的基准原点 (x, y, z)
               - "voxel_size_world": torch, (B, 3),          float32, 各样本 BOX 每个体素在世界坐标系下的物理空间大小 (x, y, z)
               - "box_shape_zyx":    torch, (B, 3),          int64,   各样本 BOX 的网格形状，顺序为 (Z, Y, X) 对应 (depth, height, width)

            2. atom 部分 (变长序列展平):
               - "atom_coord_world":          torch, (sumN, 3), float32, 展平后的全部点云在世界坐标系下的坐标 (x, y, z)
               - "atom_coord_local_voxel":    torch, (sumN, 3), float32, 展平后的全部点云的体素级坐标(距离 BOX 左下角原点多少个 voxel)
               - "atom_coord_centered_world": torch, (sumN, 3), float32, 展平后的全部点云相对于各自 BOX 中心原点的世界坐标 (x, y, z)
               - "atom_feat":                 torch, (sumN, F), float32, 展平后的全部原子的特征向量
               - "atom_label":                torch, (sumN,),   int64,   展平后的全部原子的分类真值标签
               - "atom_is_in_core_box":       torch, (sumN,),   bool,    标记每个原子是否处于各自的 BOX 内(另一部分是buffer扩展区域)
               - "atom_valid_mask":           torch, (sumN,),   bool,    标记每个原子在训练期间是否参与损失监督

            3. 索引辅助字段:
               - "atom_batch_index": torch, (sumN,), long,  指明展平的点云序列中，每个点属于当前 batch 内哪个样本 (0 ~ B-1)
               - "atom_counts":      torch, (B,),    long,  当前 batch 内每个样本分别拥有多少个原子
               - "atom_offsets":     torch, (B,),    long,  当前 batch 内每个样本的原子组在总数 (sumN) 的展平序列中的结束偏移位置, 第 i 个样本区间为 [start_i, atom_offsets[i]), 其中 start_i = 0 if i == 0 else atom_offsets[i-1]

            4. 元信息字段:
               - 包括 sample_name / pdb_id / class_name / instance_id 等, 保留为长度为 B 的 Python list, 供调试和输出用

    设计原则:
        - 固定大小的 voxel 字段走 `torch.stack`
        - 变长的 atom 字段走 `torch.cat`
        - 元信息不做 tensor 化，避免后续验证/可视化时再拆回字符串
    """
    if len(batch) == 0:
        raise ValueError("box_point_collate received an empty batch")
    if not all(isinstance(item, dict) for item in batch):
        raise TypeError("box_point_collate expects batch as list[dict]")

    # dict[str, Any], 最终返回给 model / wrapper 的 batch 容器
    batch_dict: dict[str, Any] = {}
    batch_dict.update(_stack_voxel_fields(batch))
    batch_dict.update(_concat_atom_fields(batch))
    batch_dict.update(_collect_meta_fields(batch))
    return batch_dict


def _stack_voxel_fields(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """
    对固定大小的 voxel 字段做 `torch.stack`。

    输入:
        - batch: list[dict], 长度=B

    输出:
        - stacked: dict[str, torch.Tensor], 其中每个字段都把 batch 维放在最前面。

    这里的假设是:
        - 一个 batch 内所有 BOX 的 voxel 相关字段空间尺寸一致(因此适合直接 stack, 不需要 padding)
    """
    stacked: dict[str, torch.Tensor] = {}
    for field_name in _VOXEL_STACK_FIELDS:
        # [item[field_name] for item in batch]:
        #   list[torch.Tensor], 长度=B
        # stack 之后:
        #   - voxel_grid       : (B, C, D, H, W)
        #   - voxel_label      : (B, D, H, W)
        #   - hardmask         : (B, D, H, W)
        #   - voxel_valid_mask : (B, D, H, W)
        #   - box_origin_world : (B, 3)
        #   - voxel_size_world : (B, 3)
        #   - box_shape_zyx    : (B, 3)
        stacked[field_name] = torch.stack([item[field_name] for item in batch], dim=0)
    return stacked


def _concat_atom_fields(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """
    对变长 atom 字段做 `torch.cat`，并构造 batch 内的索引辅助字段。

    因为每个 BOX 内的 atom 数 N 不固定，所以这里不做 padding，而是把整个 batch 的 atom 先拍平成一条长序列:

        sample0 的 atom  +  sample1 的 atom  +  ...  +  sample(B-1) 的 atom

    然后再额外返回:
        - atom_counts:      每个样本各有多少个 atom
        - atom_offsets:     每个样本在扁平序列中的结束偏移位置
        - atom_batch_index: 扁平后的每个 atom 属于哪个样本

    这三者后续常用于按样本聚合或还原回逐样本结构
    """
    # torch.Tensor, (B,), long, 第 i 个元素表示 batch[i] 这个样本里有多少个 atom
    atom_counts = torch.tensor(
        [int(item["atom_label"].shape[0]) for item in batch],
        dtype=torch.long,
    )

    # torch.Tensor, (B,), long, 例如 atom_counts = [5, 3, 0, 7], 则 atom_offsets = [5, 8, 8, 15]
    atom_offsets = torch.cumsum(atom_counts, dim=0)
    batch_size = int(atom_counts.shape[0])
    total_atoms = int(atom_counts.sum().item())

    # torch.Tensor, (sumN,), long, 例如 atom_counts = [2, 3], 则 atom_batch_index = [0, 0, 1, 1, 1]
    atom_batch_index = torch.repeat_interleave(
        torch.arange(batch_size, dtype=torch.long),
        atom_counts,
    )

    concatenated: dict[str, torch.Tensor] = {
        "atom_counts": atom_counts,
        "atom_offsets": atom_offsets,
        "atom_batch_index": atom_batch_index,
    }

    for field_name in _ATOM_CONCAT_FIELDS:
        # 每个 atom 字段都沿第 0 维拼接成扁平形式:
        #   - 坐标类字段: (sumN, 3)
        #   - 特征字段  : (sumN, F_atom)
        #   - 标签/掩码 : (sumN,)
        concatenated[field_name] = _concat_tensor_field(batch=batch, field_name=field_name, total_atoms=total_atoms)

    return concatenated


def _concat_tensor_field(
    batch: list[dict[str, Any]],
    field_name: str,
    total_atoms: int,
) -> torch.Tensor:
    """
    连接单个 atom 字段。

    输入:
        - batch: list[dict], 长度=B
        - field_name: str, 当前要拼接的 atom 字段名
        - total_atoms: int, 当前 batch 内 atom 总数

    输出:
        - concatenated_tensor: torch.Tensor
            若 total_atoms > 0:
                直接返回 `torch.cat(field_list, dim=0)`
            若 total_atoms == 0:
                返回一个 shape 正确但第 0 维为 0 的空 tensor

    这里显式处理 `total_atoms == 0`，是为了避免: 当一个 batch 内所有样本都没有 atom 时 `torch.cat([])` 直接报错
    """
    # list[torch.Tensor], 长度=B, 示例:
    #   - atom_coord_local_voxel:    (Ni, 3)
    #   - atom_feat:                 (Ni, F_atom)
    #   - atom_label / atom_mask:    (Ni,)
    field_list = [item[field_name] for item in batch]
    if total_atoms > 0:
        return torch.cat(field_list, dim=0)

    # 当整个 batch 一个 atom 都没有时，仍然需要返回“形状兼容”的空 tensor，这样后续模型代码就不需要额外判断“这个字段是否存在”。
    template = field_list[0]
    if template.ndim == 1:
        return template.new_empty((0,))
    return template.new_empty((0, template.shape[1]))


def _collect_meta_fields(batch: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """
    将非 tensor 的元信息字段保留为 `list`。

    输入:
        - batch: list[dict], 长度=B

    输出:
        - meta_fields: dict[str, list[Any]]
            例如:
            - "sample_name":  list[str],  长度=B
            - "pdb_id":       list[str],  长度=B
            - "class_name":   list[str],  长度=B
            - "instance_id":  list[int],  长度=B
            - "is_center_box": list[bool], 长度=B

    这样后续如果还想把 `sample_name / pdb_id / class_name / instance_id`, 带到验证、debug 或可视化环节，就不需要再回头改 collate。
    """
    # set[str], 所有“已经明确知道是 tensor 结果”的字段名
    known_tensor_fields = set(_VOXEL_STACK_FIELDS) | set(_ATOM_CONCAT_FIELDS)
    known_tensor_fields.update({"atom_counts", "atom_offsets", "atom_batch_index"})

    # dict[str, list[Any]], 收集所有非 tensor 元信息
    meta_fields: dict[str, list[Any]] = {}
    for field_name in batch[0].keys():
        if field_name in known_tensor_fields:
            continue
        # 把 batch 内同名元信息按样本顺序收集成 list
        meta_fields[field_name] = [item[field_name] for item in batch]
    return meta_fields
