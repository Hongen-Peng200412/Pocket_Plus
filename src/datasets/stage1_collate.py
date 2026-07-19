# -*- coding: utf-8 -*-
"""AdaLigand Stage1 的 dense+ranged-atom batch collator。"""

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
_OPTIONAL_DENSE_FIELDS = ("ligand_area_target", "voxel_label")
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
    """拼接变长原子表并堆叠固定 80³ 字段。

    外部契约中的 ``atom_offsets`` 为 ``int64[B+1]``，首项为 0，末项为
    ``N_A_total``；``atom_counts`` 为每个 BOX 的原子数。模型在进入 PTV3 前
    才将 offsets 转为其内部使用的 ``B`` 个结束偏移。
    """

    def __call__(self, samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("Stage1BatchCollator 不能处理空 batch。")
        batch: dict[str, Any] = {
            "pdb_id": [str(sample["pdb_id"]) for sample in samples],
            "request_role": [str(sample["request_role"]) for sample in samples],
            "occurrence_id": [sample.get("occurrence_id") for sample in samples],
            "candidate_index": [sample.get("candidate_index") for sample in samples],
        }
        for field_name in _DENSE_STACK_FIELDS:
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

        counts = torch.as_tensor([int(sample["atom_feat"].shape[0]) for sample in samples], dtype=torch.int64)
        offsets = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(dim=0)], dim=0)
        batch["atom_counts"] = counts
        batch["atom_offsets"] = offsets
        batch["atom_batch_index"] = torch.repeat_interleave(
            torch.arange(len(samples), dtype=torch.int64), counts
        )
        for field_name in _ATOM_FIELDS:
            batch[field_name] = torch.cat([sample[field_name] for sample in samples], dim=0)
        for field_name in _OPTIONAL_ATOM_FIELDS:
            presence = [field_name in sample for sample in samples]
            if any(presence) and not all(presence):
                raise ValueError(f"同一 Stage1 batch 中 {field_name} 不能部分存在。")
            if all(presence):
                batch[field_name] = torch.cat([sample[field_name] for sample in samples], dim=0)
        return batch
