"""计算 candidate 与 GT occurrence 的非零体素交集基础事实。

这里只保存正交集计数，不在此处决定 IoU、oracle 或最终 selection。candidate 顺序
严格跟随 CLG ragged 表，occurrence 则先形成 PDB 内排序 identity 表供局部 index 回指。
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .structures import CLG


def build_candidate_occurrence_overlap(
    clgs: Sequence[CLG],
    occurrence_voxel_indices: Mapping[int, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    按 `clg.npz` candidate 顺序构造 online oracle 所需的稀疏 overlap。

    输入参数:
        - clgs: Sequence[CLG], 按正式 `CLG_id` 与 candidate 来源顺序排列
        - occurrence_voxel_indices: Mapping[int,np.ndarray], occurrence_id 到
          全图 ZYX C-order 唯一线性 voxel index 的映射

    输出:
        - arrays: dict[str,np.ndarray], 包含:
            - `candidate_occurrence_offsets`: int64, (N_candidate+1,), 同时切分
              `overlap_occurrence_index` 与 `intersection_voxel_count`
            - `overlap_occurrence_index`: int32, (N_overlap,), 指向同文件
              `occurrence_id[N_gt]` 的 PDB 内局部行号 `0..N_gt-1`
            - `intersection_voxel_count`: int32, (N_overlap,), 对应交集体素数
            - `occurrence_id`: int32, (N_gt,), 当前 PDB 的排序 GT identity 表
            - `occurrence_voxel_count`: int32, (N_gt,), 各 GT mask 大小
    """
    # np.ndarray[int32], (N_gt,), 当前 PDB 内排序后的 occurrence identity 本地表。
    occurrence_ids = np.asarray(sorted(int(key) for key in occurrence_voxel_indices), dtype=np.int32)
    # dict[int,np.ndarray[int64]], occurrence_id -> 去重升序的完整图 C-order voxel indices。
    occurrence_sets = {
        int(occurrence_id): np.unique(
            np.asarray(occurrence_voxel_indices[int(occurrence_id)], dtype=np.int64)
        )
        for occurrence_id in occurrence_ids
    }
    # list[ComponentNode], 长度 N_candidate，严格按 CLG 顺序再按 CLG 内候选顺序展开。
    flattened_candidates = [node for clg in clgs for node in clg.candidate_nodes]
    # 两张平行 value 表只记录正交集；offsets 按 candidate 切分它们。
    overlap_indices: list[int] = []
    intersection_counts: list[int] = []
    offsets = [0]
    for candidate in flattened_candidates:
        candidate_indices = candidate.voxel_global_linear_index
        for occurrence_index, occurrence_id in enumerate(occurrence_ids):
            # np.ndarray[int64], 当前 candidate 与 occurrence 的完整图 voxel 交集。
            intersection = np.intersect1d(
                candidate_indices,
                occurrence_sets[int(occurrence_id)],
                assume_unique=True,
            )
            if intersection.size:
                overlap_indices.append(int(occurrence_index))
                intersection_counts.append(int(intersection.size))
        offsets.append(len(overlap_indices))
    return {
        "candidate_occurrence_offsets": np.asarray(offsets, dtype=np.int64),
        "overlap_occurrence_index": np.asarray(overlap_indices, dtype=np.int32),
        "intersection_voxel_count": np.asarray(intersection_counts, dtype=np.int32),
        "occurrence_id": occurrence_ids,
        "occurrence_voxel_count": np.asarray(
            [occurrence_sets[int(occurrence_id)].size for occurrence_id in occurrence_ids],
            dtype=np.int32,
        ),
    }
