# -*- coding: utf-8 -*-
"""AdaLigand Stage1 的 fixed-grid + ragged-atom batch collator。

80³ dense 字段沿 batch 轴堆叠；每个 BOX 数量不同的 receptor atoms 不做 padding，
而是拼成一张 ``(N_A_total, ...)`` 表，再用 ``atom_offsets``、``atom_counts`` 和
``atom_batch_index`` 恢复每个样本的边界。
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
_OPTIONAL_DENSE_FIELDS = (
    "ligand_area_target",
    "voxel_label",
    "protein_mainchain_target",
    "nucleic_mainchain_target",
    "ligand_inverse_distance_target",
)
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
    拼接变长原子表并堆叠固定 80³ 字段。

    前向输入:
        - samples: Sequence[dict[str,Any]], 长度 B，`Stage1Dataset` 物化的同一 producer 单样本字典

    前向输出:
        - batch: dict[str,Any], 包含:
            - `pdb_id/request_role/occurrence_id/candidate_index`: list，长度 B，非张量身份字段
            - dense 字段: torch.Tensor, (B,...), 由 `_DENSE_STACK_FIELDS` 和存在的可选监督字段逐项堆叠
            - `atom_counts`: torch.Tensor, (B,), int64，每个 BOX 的原子数；仅 Find batch 存在
            - `atom_offsets`: torch.Tensor, (B+1,), int64，拼接原子表的半开区间边界；仅 Find batch 存在
            - `atom_batch_index`: torch.Tensor, (N_A_total,), int64，每个原子行所属 BOX；仅 Find batch 存在
            - atom 字段: torch.Tensor, (N_A_total,...), 由各样本变长原子表沿第 0 轴拼接；仅 Find batch 存在

    外部契约中的 ``atom_offsets`` 为 ``int64[B+1]``，首项为 0，末项为
    ``N_A_total``；``atom_counts`` 为每个 BOX 的原子数。模型在进入 PTV3 前
    才将 offsets 转为其内部使用的 ``B`` 个结束偏移。
    """

    def __call__(self, samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """
        把同一 producer 的单样本序列整理成 fixed-grid + ragged-atom batch。

        输入参数:
            - samples: Sequence[dict[str,Any]], 长度 B，字段集合在 batch 内必须一致

        输出:
            - batch: dict[str,Any], fixed-grid 字段以 `(B,...)` 堆叠；Find 原子字段以 `(N_A_total,...)` 拼接并附带 counts/offsets/batch_index
        """
        if not samples:
            raise ValueError("Stage1BatchCollator 不能处理空 batch。")
        # dict[str,Any], batch 级非张量身份；四个列表长度均为 B，与输入 samples 同序。
        batch: dict[str, Any] = {
            "pdb_id": [str(sample["pdb_id"]) for sample in samples],
            "request_role": [str(sample["request_role"]) for sample in samples],
            "occurrence_id": [sample.get("occurrence_id") for sample in samples],
            "candidate_index": [sample.get("candidate_index") for sample in samples],
        }
        for field_name in _DENSE_STACK_FIELDS:
            # torch.Tensor, (B,...), 把每个 BOX 的同形字段沿新 batch 轴堆叠。
            batch[field_name] = torch.stack([sample[field_name] for sample in samples], dim=0)

        for field_name in _OPTIONAL_DENSE_FIELDS:
            presence = [field_name in sample for sample in samples]
            if any(presence) and not all(presence):
                raise ValueError(f"同一 Stage1 batch 中 {field_name} 不能部分存在。")
            if all(presence):
                batch[field_name] = torch.stack([sample[field_name] for sample in samples], dim=0)

        has_atoms = ["atom_feat" in sample for sample in samples]
        if any(has_atoms) and not all(has_atoms):
            raise ValueError("同一 Stage1 batch 中不能混合 Find 与 unet_c1 样本。")
        if not all(has_atoms):
            return batch

        # torch.Tensor[int64], (B,), 每个 BOX 在 ragged 原子总表中贡献的行数。
        counts = torch.as_tensor([int(sample["atom_feat"].shape[0]) for sample in samples], dtype=torch.int64)
        # torch.Tensor[int64], (B+1,), 半开区间边界；样本 b 对应 [offsets[b], offsets[b+1])。
        offsets = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(dim=0)], dim=0)
        batch["atom_counts"] = counts
        batch["atom_offsets"] = offsets
        # torch.Tensor[int64], (N_A_total,), 每一原子行反查其 BOX batch 下标。
        batch["atom_batch_index"] = torch.repeat_interleave(
            torch.arange(len(samples), dtype=torch.int64), counts
        )
        for field_name in _ATOM_FIELDS:
            # torch.Tensor, (N_A_total,...), 按 samples 顺序拼接变长原子字段，不做 padding。
            batch[field_name] = torch.cat([sample[field_name] for sample in samples], dim=0)
        for field_name in _OPTIONAL_ATOM_FIELDS:
            presence = [field_name in sample for sample in samples]
            if any(presence) and not all(presence):
                raise ValueError(f"同一 Stage1 batch 中 {field_name} 不能部分存在。")
            if all(presence):
                batch[field_name] = torch.cat([sample[field_name] for sample in samples], dim=0)
        return batch
