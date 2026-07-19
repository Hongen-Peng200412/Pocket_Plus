"""完整网格 average precision 与语义 Dice。"""

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
        - probability_map: np.ndarray, (D,H,W), 连续概率；Find hardmask voxel 仍在区域内
        - gt_union_mask: np.ndarray, (D,H,W), occurrence ligand-area 并集标签

    输出:
        - average_precision: float | None, 与 sklearn `average_precision_score` 等价；
          当前 PDB 没有 GT 正类时返回 None，供 macro 聚合跳过并报告有效数
    """
    score = np.asarray(probability_map, dtype=np.float64).reshape(-1)
    target = np.asarray(gt_union_mask, dtype=np.bool_).reshape(-1)
    if score.shape != target.shape:
        raise ValueError("probability_map 与 gt_union_mask shape 不一致")
    if not bool(np.all(np.isfinite(score))):
        raise ValueError("probability_map 含非有限值")
    positive_count = int(target.sum())
    if positive_count == 0:
        return None
    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_target = target[order].astype(np.int64, copy=False)
    cumulative_tp = np.cumsum(sorted_target, dtype=np.int64)
    cumulative_total = np.arange(1, sorted_target.size + 1, dtype=np.int64)
    boundaries = np.r_[
        np.flatnonzero(np.diff(sorted_score) != 0.0), sorted_score.size - 1
    ]
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
        - probability_and_target: Iterable[(np.ndarray,np.ndarray)]，每项是一张完整图

    输出:
        - result: dict[str,float|int]，包含:
            - `voxel_average_precision_macro`: float，有效 PDB AP 的等权均值；无有效项为 NaN
            - `n_valid_pdb`: int，至少含一个 GT 正 voxel 的 PDB 数
            - `n_total_pdb`: int，输入 PDB 总数
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
        - probability_map: np.ndarray, (D,H,W), 连续完整图概率
        - gt_union_mask: np.ndarray, (D,H,W), occurrence ligand-area 并集
        - threshold: float, 二值化阈值，规则为 `probability>=threshold`

    输出:
        - result: dict[str,float|int]，包含 `dice`、`tp`、`fp`、`fn`
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
