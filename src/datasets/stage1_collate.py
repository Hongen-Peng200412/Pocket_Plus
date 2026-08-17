# -*- coding: utf-8 -*-
"""把 Stage1 单个 80³ 样本组装成模型使用的 batch 字典。

固定形状的几何、密度、掩码和体素监督沿新批次轴 ``B`` 堆叠；Find producer 的局部受体原子数量逐 BOX 可变，不填充到统一长度，而是按输入样本顺序沿原子轴拼接。``atom_counts``、``atom_offsets`` 和 ``atom_batch_index`` 记录每个 BOX 在拼接表中的范围，模型可据此恢复 ragged 原子集合。
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
# 这些体素监督字段只在目标请求中存在；同一 batch 必须全体存在或全体缺失，避免样本间监督语义混用。
_OPTIONAL_DENSE_FIELDS = (
    "ligand_area_target",
    "voxel_label",
    "protein_mainchain_target",
    "nucleic_mainchain_target",
    "ligand_inverse_distance_target",
)
# 这些字段的第 0 维都对应拼接后的局部受体原子轴，长度为 N_A_total。
_ATOM_FIELDS = (
    "atom_global_indices",
    "atom_feat",
    "atom_is_backbone",
    "atom_coord_world",
    "atom_coord_local_voxel",
    "atom_coord_centered_world",
    "atom_is_in_core_box",
)
_OPTIONAL_ATOM_FIELDS = ("atom_label",)


class Stage1BatchCollator:
    """堆叠固定 80³ 字段并拼接数量可变的 Find 原子字段。

    输入契约:
        - samples: Sequence[dict[str, Any]]；长度为 B 的同一 producer 单样本序列；每个样本来自 ``Stage1Dataset``，固定字段必须是同形 CPU tensor。

    输出字段:
        - ``pdb_id``、``request_role``、``occurrence_id``、``candidate_index``：长度 B 的 Python list，保持输入样本顺序。
        - ``box_start_zyx``、``box_shape_zyx``、``box_origin_world``、``voxel_size_world``、``density_input``、``hardmask``：torch.Tensor；分别沿新批次维堆叠，空间轴仍为 ZYX，结果第 0 维为 B。
        - ``ligand_area_target``、``voxel_label``、``protein_mainchain_target``、``nucleic_mainchain_target``、``ligand_inverse_distance_target``：torch.Tensor；仅当所有样本都具备对应目标时沿新批次维堆叠，结果第 0 维为 B。
        - ``atom_counts``：int64 ``(B,)``；每个 BOX 的局部原子数；仅 Find batch 存在。
        - ``atom_offsets``：int64 ``(B + 1,)``；半开区间边界，首项为 0，末项为 N_A_total；BOX ``b`` 对应 ``[offsets[b], offsets[b + 1])``。
        - ``atom_batch_index``：int64 ``(N_A_total,)``；每个拼接原子的所属 batch index。
        - ``atom_global_indices``、``atom_feat``、``atom_is_backbone``、``atom_coord_world``、``atom_coord_local_voxel``、``atom_coord_centered_world``、``atom_is_in_core_box``：torch.Tensor；沿第 0 维按样本顺序拼接，不增加填充原子，仅 Find batch 存在。

    失败语义:
        - 空输入、可选字段部分存在，或同一 batch 混合 Find 与 ``unet_c1`` 样本时抛出 ``ValueError``；不会静默补空字段。
    """
    def __call__(self, samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """将一组同 producer 单样本转换为一个模型 batch。

        输入参数:
            - samples: Sequence[dict[str, Any]]；长度 B；固定体素字段的空间形状必须一致，Find 原子字段的第 0 维可变但各原子字段必须互相对齐。

        返回值:
            - batch: dict[str, Any]；身份字段保持 Python list；固定字段沿批次维堆叠；Find 原子字段沿原子维拼接并附带 counts、offsets、batch index。

        失败语义:
            - 输入为空、监督字段只在部分样本出现，或 Find 与 ``unet_c1`` 混批时抛出 ``ValueError``。
        """
        if not samples:
            raise ValueError("Stage1BatchCollator 不能处理空 batch。")
        # dict[str, Any]；批次级 Python identity 列表长度均为 B，并与输入 samples 同序。
        batch: dict[str, Any] = {
            "pdb_id": [str(sample["pdb_id"]) for sample in samples],
            "request_role": [str(sample["request_role"]) for sample in samples],
            "occurrence_id": [sample.get("occurrence_id") for sample in samples],
            "candidate_index": [sample.get("candidate_index") for sample in samples],
        }
        for field_name in _DENSE_STACK_FIELDS:
            # torch.Tensor (B, ... )；每个 BOX 的同形固定字段沿新批次维堆叠。
            batch[field_name] = torch.stack([sample[field_name] for sample in samples], dim=0)

        for field_name in _OPTIONAL_DENSE_FIELDS:
            presence = [field_name in sample for sample in samples]
            if any(presence) and not all(presence):
                raise ValueError(f"同一 Stage1 batch 中 {field_name} 不能部分存在。")
            if all(presence):
                # torch.Tensor (B, ... )；监督字段第 0 维与 batch["pdb_id"] 逐样本对齐。
                batch[field_name] = torch.stack([sample[field_name] for sample in samples], dim=0)

        has_atoms = ["atom_feat" in sample for sample in samples]
        if any(has_atoms) and not all(has_atoms):
            raise ValueError("同一 Stage1 batch 中不能混合 Find 与 unet_c1 样本。")
        if not all(has_atoms):
            return batch

        # torch.Tensor int64 (B,)；每个 BOX 在拼接原子轴上包含的局部原子数量。
        counts = torch.as_tensor([int(sample["atom_feat"].shape[0]) for sample in samples], dtype=torch.int64)
        # torch.Tensor int64 (B + 1,)；拼接原子数组的半开区间边界，BOX b 对应 [offsets[b], offsets[b + 1])。
        offsets = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(dim=0)], dim=0)
        batch["atom_counts"] = counts
        batch["atom_offsets"] = offsets
        # torch.Tensor int64 (N_A_total,)；每个拼接原子对应的 BOX 批次编号。
        batch["atom_batch_index"] = torch.repeat_interleave(torch.arange(len(samples), dtype=torch.int64), counts)
        for field_name in _ATOM_FIELDS:
            # torch.Tensor (N_A_total, ...)；按 samples 顺序拼接受体原子字段，不增加填充原子。
            batch[field_name] = torch.cat([sample[field_name] for sample in samples], dim=0)
        for field_name in _OPTIONAL_ATOM_FIELDS:
            presence = [field_name in sample for sample in samples]
            if any(presence) and not all(presence):
                raise ValueError(f"同一 Stage1 batch 中 {field_name} 不能部分存在。")
            if all(presence):
                batch[field_name] = torch.cat([sample[field_name] for sample in samples], dim=0)
        return batch
