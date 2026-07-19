"""计算 candidate 与 GT occurrence 的非零体素交集基础事实。"""

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
    occurrence_ids = np.asarray(sorted(int(key) for key in occurrence_voxel_indices), dtype=np.int32)
    occurrence_sets = {
        int(occurrence_id): np.unique(
            np.asarray(occurrence_voxel_indices[int(occurrence_id)], dtype=np.int64)
        )
        for occurrence_id in occurrence_ids
    }
    flattened_candidates = [node for clg in clgs for node in clg.candidate_nodes]
    overlap_indices: list[int] = []
    intersection_counts: list[int] = []
    offsets = [0]
    for candidate in flattened_candidates:
        candidate_indices = candidate.voxel_global_linear_index
        for occurrence_index, occurrence_id in enumerate(occurrence_ids):
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
