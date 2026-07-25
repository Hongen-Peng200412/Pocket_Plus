"""根据基础 overlap 现场生成 Selector 监督。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .antichain_dp import exact_antichain_map


@dataclass(frozen=True)
class OnlineOracle:
    """
    保存一次按当前 lambda_count 现场计算的 CLG oracle。

    字段:
        - candidate_max_iou: torch.Tensor, (N_candidate,), 每个 candidate 对任一 GT occurrence 的最大 IoU
        - selected_candidate_index: torch.Tensor, (N_selected,), oracle 最优反链的 candidate 局部下标
        - is_valid: torch.Tensor, scalar bool，oracle 是否选择非空反链
        - best_score: torch.Tensor, scalar，含计数惩罚的 oracle 最优分数；空解时为 0
    """

    candidate_max_iou: torch.Tensor
    selected_candidate_index: torch.Tensor
    is_valid: torch.Tensor
    best_score: torch.Tensor


def compute_candidate_max_iou(
    candidate_occurrence_offsets: Sequence[int],
    overlap_occurrence_index: Sequence[int],
    intersection_voxel_count: Sequence[int],
    candidate_voxel_count: Sequence[int],
    occurrence_voxel_count: Sequence[int],
) -> np.ndarray:
    """
    从非零 candidate-occurrence 交集计算每个 candidate 的最大 IoU。

    输入参数:
        - candidate_occurrence_offsets: Sequence[int], (N_candidate+1,), 同时切分两个 overlap value 表
        - overlap_occurrence_index: Sequence[int], (N_overlap,), 指向同文件 occurrence_id/occurrence_voxel_count 的局部行
        - intersection_voxel_count: Sequence[int], (N_overlap,), 对应交集体素数
        - candidate_voxel_count: Sequence[int], (N_candidate,), candidate mask 体素数
        - occurrence_voxel_count: Sequence[int], (N_gt,), 对应 GT mask 体素数

    输出:
        - q: np.ndarray, (N_candidate,), float32，每个 candidate 的最大 IoU；无 GT/无交集为 0
    """
    offsets = np.asarray(candidate_occurrence_offsets, dtype=np.int64)
    overlap_indices = np.asarray(overlap_occurrence_index, dtype=np.int64)
    intersections = np.asarray(intersection_voxel_count, dtype=np.float64)
    candidate_counts = np.asarray(candidate_voxel_count, dtype=np.float64)
    gt_counts = np.asarray(occurrence_voxel_count, dtype=np.float64)
    if offsets.shape != (candidate_counts.size + 1,):
        raise ValueError("candidate_occurrence_offsets 长度必须为 N_candidate+1。")
    if offsets[0] != 0 or offsets[-1] != overlap_indices.size or overlap_indices.size != intersections.size:
        raise ValueError("overlap offsets 与 value 表长度不一致。")
    if overlap_indices.size and (overlap_indices.min() < 0 or overlap_indices.max() >= gt_counts.size):
        raise IndexError("overlap_occurrence_index 越过 occurrence 表。")

    result = np.zeros(candidate_counts.size, dtype=np.float32)
    for candidate_index in range(candidate_counts.size):
        begin = int(offsets[candidate_index])
        end = int(offsets[candidate_index + 1])
        best = 0.0
        for row in range(begin, end):
            gt_index = int(overlap_indices[row])
            intersection = float(intersections[row])
            union = float(candidate_counts[candidate_index]) + float(gt_counts[gt_index]) - intersection
            if union <= 0.0:
                raise ValueError("candidate/GT union voxel count 必须为正。")
            best = max(best, intersection / union)
        result[candidate_index] = np.float32(best)
    return result


def build_online_oracle(
    candidate_max_iou: torch.Tensor,
    parent_index: Sequence[int],
    candidate_index_by_node: Sequence[int],
    lambda_count: float,
) -> OnlineOracle:
    """
    按当前 lambda_count 现场求包括空集在内的 GT 最优反链。

    输入参数:
        - candidate_max_iou: torch.Tensor, (N_candidate,), q_i
        - parent_index: Sequence[int], (N_closure,), candidate 最小连接闭包父行
        - candidate_index_by_node: Sequence[int], (N_closure,), 闭包节点到 candidate 行映射
        - lambda_count: float, oracle 计数惩罚

    输出:
        - oracle: OnlineOracle；非空最优分数不大于 0 时稳定选择空集
    """
    q = candidate_max_iou.reshape(-1).to(dtype=torch.float32)
    nonempty_score, selected = exact_antichain_map(
        selection_score=q,
        parent_index=parent_index,
        candidate_index_by_node=candidate_index_by_node,
        lambda_count=lambda_count,
    )
    is_valid = nonempty_score > 0.0
    if not bool(is_valid.item()):
        selected = torch.empty((0,), dtype=torch.long, device=q.device)
        best_score = q.new_zeros(())
    else:
        best_score = nonempty_score
    return OnlineOracle(
        candidate_max_iou=q,
        selected_candidate_index=selected,
        is_valid=is_valid,
        best_score=best_score,
    )
