"""在完整 ZYX voxel 网格上计算 Average Precision、macro AP 与语义 Dice。

主要入口:
    - `average_precision_full_grid`: 对一个 PDB 的所有 voxel 按连续概率排序，计算与 sklearn 一致的阶梯式 Average Precision。
    - `macro_average_precision`: 只对至少含一个真实正 voxel 的 PDB 做等权平均，同时报告有效数和总数。
    - `semantic_dice`: 在冻结概率阈值下计算完整图 TP、FP、FN 与 Dice。

本模块不裁剪有效区域、不移除 Find hardmask voxel，也不对空 GT PDB伪造 AP；调用方必须传入相同完整网格上的概率与真实配体区域并集。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def average_precision_full_grid(
    probability_map: np.ndarray,
    gt_union_mask: np.ndarray,
) -> float | None:
    """
    在完整 `[D,H,W]` 网格上计算阶梯式 Average Precision。

    输入参数:
        - probability_map: numeric, (D, H, W), 完整 ZYX voxel 网格上的连续概率；Find hardmask voxel 仍保留在评估区域内。
        - gt_union_mask: bool, (D, H, W), 同一完整网格上的真实 occurrence 配体区域并集标签。

    输出:
        - average_precision: float | None, 与 sklearn `average_precision_score` 的阶梯积分等价；当前 PDB 没有真实正 voxel 时返回 None，供 macro 聚合跳过。
    """
    # float64, (V,), 按 C-order 展平的完整图连续概率，V=D*H*W。
    score = np.asarray(probability_map, dtype=np.float64).reshape(-1)
    # bool, (V,), 与 `score` 逐 voxel 对齐的真实配体区域并集标签。
    target = np.asarray(gt_union_mask, dtype=np.bool_).reshape(-1)
    if score.shape != target.shape:
        raise ValueError("probability_map 与 gt_union_mask shape 不一致")
    if not bool(np.all(np.isfinite(score))):
        raise ValueError("probability_map 含非有限值")
    positive_count = int(target.sum())
    if positive_count == 0:
        return None
    # int64, (V,), 按概率降序排列的 voxel 行号；稳定排序使同分 voxel 保持原 C-order 顺序。
    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_target = target[order].astype(np.int64, copy=False)
    cumulative_tp = np.cumsum(sorted_target, dtype=np.int64)
    cumulative_total = np.arange(1, sorted_target.size + 1, dtype=np.int64)
    # int64, (N_unique_score,), 每个不同概率分组在排序表中的末索引；只在这些行更新 precision/recall 阶梯。
    boundaries = np.r_[np.flatnonzero(np.diff(sorted_score) != 0.0), sorted_score.size - 1]
    precision = cumulative_tp[boundaries] / cumulative_total[boundaries]
    recall = cumulative_tp[boundaries] / float(positive_count)
    recall_previous = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_previous) * precision))


def macro_average_precision(
    probability_and_target: Iterable[tuple[np.ndarray, np.ndarray]],
) -> dict[str, float | int]:
    """
    对有效 PDB 的完整网格 AP 做 macro 平均。

    输入参数:
        - probability_and_target: Iterable[tuple[np.ndarray, np.ndarray]], 每项是一张 PDB 完整概率图及其同 shape 真实配体区域并集。

    输出字段:
        - `voxel_average_precision_macro`: float, 有效 PDB AP 的等权均值；没有有效 PDB 时为 NaN。
        - `n_valid_pdb`: int, 至少含一个真实正 voxel、因此 AP 有定义的 PDB 数。
        - `n_total_pdb`: int, 输入迭代器实际消费的 PDB 总数。
    """
    values: list[float] = []
    total = 0
    for probability_map, target in probability_and_target:
        total += 1
        value = average_precision_full_grid(probability_map, target)
        if value is not None:
            values.append(value)
    return {
        "voxel_average_precision_macro": float(np.mean(values)) if values else float("nan"),
        "n_valid_pdb": len(values),
        "n_total_pdb": total,
    }


def semantic_dice(
    probability_map: np.ndarray,
    gt_union_mask: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """
    在给定冻结阈值下计算完整图语义 Dice 及 TP/FP/FN。

    输入参数:
        - probability_map: numeric, (D, H, W), 完整 ZYX voxel 网格上的连续概率。
        - gt_union_mask: bool, (D, H, W), 同一完整网格上的真实 occurrence 配体区域并集。
        - threshold: float, 概率二值化阈值；预测正类规则为 `probability >= threshold`。

    输出字段:
        - `dice`: float, `2*tp/(2*tp+fp+fn)`；分母为 0 时返回 0。
        - `tp`: int, 预测与真实标签都为正的完整图 voxel 数。
        - `fp`: int, 预测为正而真实标签为负的完整图 voxel 数。
        - `fn`: int, 预测为负而真实标签为正的完整图 voxel 数。
    """
    prediction = np.asarray(probability_map) >= float(threshold)
    target = np.asarray(gt_union_mask, dtype=np.bool_)
    if prediction.shape != target.shape:
        raise ValueError("probability_map 与 gt_union_mask shape 不一致")
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    denominator = 2 * tp + fp + fn
    return {
        "dice": float(2 * tp / denominator) if denominator else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
