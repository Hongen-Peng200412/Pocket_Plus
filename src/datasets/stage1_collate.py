# -*- coding: utf-8 -*-
"""把 Stage1 单个 80³ BOX 样本拼装成模型使用的批次字典. 

每个 BOX 的稠密体素字段沿批次维堆叠. 不同 BOX 的受体原子数量不同, 原子字段不补齐到相同长度, 而是按样本顺序拼接成 ``(N_atom_total, ...)`` 数组; 
``atom_offsets``、``atom_counts`` 和 ``atom_batch_index`` 共同保存每个 BOX 在该拼接数组中的范围. 
"""

from __future__ import annotations

from typing import Any, Sequence

import torch


_DENSE_STACK_FIELDS = (
    "box_start_zyx",
    "box_shape_zyx",
    "box_origin_world",
    "voxel_size_world",
    "density_input",
    "hardmask",
)
# 这些监督字段只有训练或验证样本才可能存在; 同一批次必须全部存在或全部缺失. 
_OPTIONAL_DENSE_FIELDS = (
    "ligand_area_target",
    "voxel_label",
    "protein_mainchain_target",
    "nucleic_mainchain_target",
    "ligand_inverse_distance_target",
)
# 这些字段的第 0 维都对应同一批次内拼接后的受体原子. 
_ATOM_FIELDS = (
    "atom_global_indices",
    "atom_feat",
    "atom_coord_world",
    "atom_coord_local_voxel",
    "atom_coord_centered_world",
    "atom_is_in_core_box",
)
_OPTIONAL_ATOM_FIELDS = ("atom_label",)


class Stage1BatchCollator:
    """
    拼接数量可变的受体原子字段, 并堆叠固定 80³ 体素字段. 

    前向输入:
        - samples: Sequence[dict[str,Any]], 长度 B, `Stage1Dataset` 物化的同一 producer 单样本字典

    前向输出:
        - batch: dict[str,Any], 包含:
            - pdb_id/request_role/occurrence_id/candidate_index: list, 长度 B, 非张量身份字段
            - 稠密体素字段: torch.Tensor, (B,...), 由 `_DENSE_STACK_FIELDS` 和存在的可选监督字段逐项堆叠
            - atom_counts: torch.Tensor, (B,), int64, 每个 BOX 的原子数; 仅 Find batch 存在
            - atom_offsets: torch.Tensor, (B+1,), int64, 拼接原子表的半开区间边界; 仅 Find batch 存在
            - atom_batch_index: torch.Tensor, (N_A_total,), int64, 每个原子所属 BOX 的批次编号; 仅 Find 批次存在
            - 原子字段: torch.Tensor, (N_A_total,...), 按输入样本顺序沿第 0 维拼接; 仅 Find 批次存在

    外部契约中的 ``atom_offsets`` 为 ``int64[B+1]``, 首项为 0, 末项为 ``N_A_total``; ``atom_counts`` 为每个 BOX 的原子数. 
    模型在进入 PTV3 前才将 offsets 转为其内部使用的 ``B`` 个结束偏移. 
    """
    def __call__(self, samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """
        拼接数量可变的受体原子字段, 并堆叠固定 80³ 体素字段. 

        前向输入:
            - samples: Sequence[dict[str,Any]], 长度 B, `Stage1Dataset` 物化的同一 producer 单样本字典

        前向输出:
            - batch: dict[str,Any], 包含:
                - pdb_id/request_role/occurrence_id/candidate_index: list, 长度 B, 非张量身份字段
                - 稠密体素字段: torch.Tensor, (B,...), 由 `_DENSE_STACK_FIELDS` 和存在的可选监督字段逐项堆叠
                - atom_counts: torch.Tensor, (B,), int64, 每个 BOX 的原子数; 仅 Find batch 存在
                - atom_offsets: torch.Tensor, (B+1,), int64, 拼接原子表的半开区间边界; 仅 Find batch 存在
                - atom_batch_index: torch.Tensor, (N_A_total,), int64, 每个原子所属 BOX 的批次编号; 仅 Find 批次存在
                - 原子字段: torch.Tensor, (N_A_total,...), 按输入样本顺序沿第 0 维拼接; 仅 Find 批次存在

        外部契约中的 ``atom_offsets`` 为 ``int64[B+1]``, 首项为 0, 末项为 ``N_A_total``; ``atom_counts`` 为每个 BOX 的原子数. 
        模型在进入 PTV3 前才将 offsets 转为其内部使用的 ``B`` 个结束偏移. 
        """
        if not samples:
            raise ValueError("Stage1BatchCollator 不能处理空 batch。")
        # dict[str, Any], 批次级非张量身份; 四个列表长度均为 B, 与输入 samples 同序. 
        batch: dict[str, Any] = {
            "pdb_id": [str(sample["pdb_id"]) for sample in samples],
            "request_role": [str(sample["request_role"]) for sample in samples],
            "occurrence_id": [sample.get("occurrence_id") for sample in samples],
            "candidate_index": [sample.get("candidate_index") for sample in samples],
        }
        for field_name in _DENSE_STACK_FIELDS:
            # (B, ...), 把每个 BOX 的同形字段沿新批次维堆叠. 
            batch[field_name] = torch.stack([sample[field_name] for sample in samples], dim=0)

        for field_name in _OPTIONAL_DENSE_FIELDS:
            presence = [field_name in sample for sample in samples]
            if any(presence) and not all(presence):
                raise ValueError(f"同一 Stage1 batch 中 {field_name} 不能部分存在。")
            if all(presence):
                # (B, ...), 监督字段的第 0 维与 batch["pdb_id"] 逐样本对齐. 
                batch[field_name] = torch.stack([sample[field_name] for sample in samples], dim=0)

        has_atoms = ["atom_feat" in sample for sample in samples]
        if any(has_atoms) and not all(has_atoms):
            raise ValueError("同一 Stage1 batch 中不能混合 Find 与 unet_c1 样本。")
        if not all(has_atoms):
            return batch

        # int64, (B,), 每个 BOX 在拼接原子数组中包含的原子数量. 
        counts = torch.as_tensor([int(sample["atom_feat"].shape[0]) for sample in samples], dtype=torch.int64)
        # int64, (B + 1,), 拼接原子数组的半开区间边界; BOX b 对应 [offsets[b], offsets[b + 1]). 
        offsets = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(dim=0)], dim=0)
        batch["atom_counts"] = counts
        batch["atom_offsets"] = offsets
        # int64, (N_A_total,), 每个拼接原子对应的 BOX 批次编号. 
        batch["atom_batch_index"] = torch.repeat_interleave(torch.arange(len(samples), dtype=torch.int64), counts)
        for field_name in _ATOM_FIELDS:
            # (N_A_total, ...), 按 samples 顺序拼接受体原子字段, 不增加补齐原子. 
            batch[field_name] = torch.cat([sample[field_name] for sample in samples], dim=0)
        for field_name in _OPTIONAL_ATOM_FIELDS:
            presence = [field_name in sample for sample in samples]
            if any(presence) and not all(presence):
                raise ValueError(f"同一 Stage1 batch 中 {field_name} 不能部分存在。")
            if all(presence):
                batch[field_name] = torch.cat([sample[field_name] for sample in samples], dim=0)
        return batch
