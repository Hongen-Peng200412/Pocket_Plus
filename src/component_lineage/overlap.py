"""计算候选组件与真实配体 occurrence 的非零 voxel 交集基础事实。

主要入口:
    - `build_candidate_occurrence_overlap`: 按 `clg.npz` 的 CLG/candidate 顺序构造 `overlap.npz` 稀疏交集表。

本模块只保存正交集 voxel 数，不计算 IoU、不决定在线 oracle，也不执行最终 selection。candidate 顺序严格跟随 CLG ragged 表；occurrence 先按 `occurrence_id` 排序形成 PDB 内主表，稀疏交集通过局部行号回指该主表。
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
        - clgs: Sequence[CLG], 按正式 `CLG_id` 排列；每个 CLG 内 candidate 顺序与 `clg.npz` 一致。
        - occurrence_voxel_indices: Mapping[int, np.ndarray], occurrence_id 到完整 ZYX 网格 C-order 离散线性 voxel 索引的映射；本函数对每个数组去重并升序。

    输出字段:
        - `candidate_occurrence_offsets`: int64, (N_candidate + 1,), 指向 overlap_occurrence_index 和 intersection_voxel_count, candidate i 的非零 GT 交集位于两张 overlap value 表的 `[offsets[i], offsets[i + 1])`。
        - `overlap_occurrence_index`: int32, (N_overlap,), 指向同归档 `occurrence_id` 第一维的 PDB 内局部行号 `0..N_gt-1`。
        - `intersection_voxel_count`: int32, (N_overlap,), 与 `overlap_occurrence_index` 对齐的相交 voxel 数。
        - `occurrence_id`: int32, (N_gt,), 当前 PDB 所有真实配体的 occurrence_id 。
        - `occurrence_voxel_count`: int32, (N_gt,), 每个 occurrence 去重后的 voxel 数，与 `occurrence_id` 逐 occurrence 对齐。
    """
    # int32, (N_gt,), 当前 PDB 内按数值升序排列的 occurrence identity 主表。
    occurrence_ids = np.asarray(sorted(int(key) for key in occurrence_voxel_indices), dtype=np.int32)
    # dict[int, int64 array], occurrence_id 到去重升序的完整 ZYX 网格 C-order voxel 索引集合。
    occurrence_sets = {
        int(occurrence_id): np.unique(np.asarray(occurrence_voxel_indices[int(occurrence_id)], dtype=np.int64))
        for occurrence_id in occurrence_ids
    }
    # list[ComponentNode], 长度 N_candidate；严格先按 CLG 顺序、再按每个 CLG 的候选顺序展开。
    flattened_candidates = [node for clg in clgs for node in clg.candidate_nodes]
    # 两张平行 value 表只记录相交的情况；`offsets` 按 flattened candidate 顺序同步切分 occurrence 局部行号和交集计数。
    overlap_indices: list[int] = []
    intersection_counts: list[int] = []
    offsets = [0]
    for candidate in flattened_candidates:
        candidate_indices = candidate.voxel_global_linear_index
        for occurrence_index, occurrence_id in enumerate(occurrence_ids):
            # int64, (K_intersection,), 当前 candidate 与 occurrence 在完整图中的共同 C-order voxel 索引；空交集不写入稀疏表。
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
        "occurrence_voxel_count": np.asarray([occurrence_sets[int(occurrence_id)].size for occurrence_id in occurrence_ids], dtype=np.int32),
    }
