"""双向 coverage、固定连续 Hungarian 与 top-K instance 指标。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


DEFAULT_COVERAGE_THRESHOLDS: tuple[float, ...] = (0.3, 0.5)
DEFAULT_TOPK_VALUES: tuple[int, ...] = (3, 4, 5)


@dataclass(frozen=True)
class InstanceCounts:
    """
    保存一个或多个 PDB 可直接求 global/micro F1 的 instance 计数。

    输入参数:
        - coverage_thresholds: tuple[float,...], 双向 coverage 阈值顺序
        - n_pred: int, 预测组件总数，precision 分母
        - n_gt: int, GT occurrence 总数，recall 分母
        - coverage_pred_hit: np.ndarray, (T,), int64，各阈值命中任一 GT 的预测数
        - coverage_gt_hit: np.ndarray, (T,), int64，各阈值被任一预测命中的 GT 数
        - one_to_one_tp: np.ndarray, (T,), int64，固定 Hungarian 配对中双向达标数
    """

    coverage_thresholds: tuple[float, ...]
    n_pred: int
    n_gt: int
    coverage_pred_hit: np.ndarray
    coverage_gt_hit: np.ndarray
    one_to_one_tp: np.ndarray

    def metrics(self) -> dict[str, float | int]:
        """
        把计数转换为 coverage 与 one-to-one precision/recall/F1。

        输出:
            - metrics: dict[str,float|int], 含总 instance 数，以及每个阈值的
              `coverage_precision/recall/f1` 与 `one_to_one_precision/recall/f1`
    """
        result: dict[str, float | int] = {
            "n_pred_instances": int(self.n_pred),
            "n_gt_instances": int(self.n_gt),
        }
        for row, threshold in enumerate(self.coverage_thresholds):
            tag = _threshold_tag(threshold)
            coverage_precision = _safe_div(int(self.coverage_pred_hit[row]), self.n_pred)
            coverage_recall = _safe_div(int(self.coverage_gt_hit[row]), self.n_gt)
            one_precision = _safe_div(int(self.one_to_one_tp[row]), self.n_pred)
            one_recall = _safe_div(int(self.one_to_one_tp[row]), self.n_gt)
            result[f"coverage_precision_{tag}"] = coverage_precision
            result[f"coverage_recall_{tag}"] = coverage_recall
            result[f"coverage_f1_{tag}"] = _harmonic_mean(
                coverage_precision, coverage_recall
            )
            result[f"one_to_one_precision_{tag}"] = one_precision
            result[f"one_to_one_recall_{tag}"] = one_recall
            result[f"one_to_one_f1_{tag}"] = _harmonic_mean(one_precision, one_recall)
        return result


def evaluate_instance_labels(
    pred_instance_label: np.ndarray,
    gt_instance_label: np.ndarray,
    coverage_thresholds: Sequence[float] = DEFAULT_COVERAGE_THRESHOLDS,
) -> InstanceCounts:
    """
    对一张 PDB 的预测组件和 GT occurrence 标签图计算正式 instance 计数。

    输入参数:
        - pred_instance_label: np.ndarray, (D,H,W), 正整数为预测组件，0 为背景
        - gt_instance_label: np.ndarray, (D,H,W), 正整数为 GT occurrence，0 为背景
        - coverage_thresholds: Sequence[float], 正式值为 (0.3,0.5)

    输出:
        - counts: InstanceCounts, coverage 允许多对一；one-to-one 先对连续
          `sqrt(c_pred*c_GT)` 做一次固定 Hungarian，再在同一配对上逐阈值计数
    """
    pred_label = np.asarray(pred_instance_label, dtype=np.int64)
    gt_label = np.asarray(gt_instance_label, dtype=np.int64)
    if pred_label.shape != gt_label.shape or pred_label.ndim != 3:
        raise ValueError("预测与 GT instance label 必须是同 shape 的三维数组")
    pred_ids = np.unique(pred_label[pred_label > 0])
    gt_ids = np.unique(gt_label[gt_label > 0])
    pred_sizes = np.asarray([(pred_label == value).sum() for value in pred_ids], dtype=np.int64)
    gt_sizes = np.asarray([(gt_label == value).sum() for value in gt_ids], dtype=np.int64)
    intersections = np.zeros((pred_ids.size, gt_ids.size), dtype=np.int64)
    if pred_ids.size and gt_ids.size:
        both = (pred_label > 0) & (gt_label > 0)
        pred_rows = np.searchsorted(pred_ids, pred_label[both])
        gt_columns = np.searchsorted(gt_ids, gt_label[both])
        flattened = pred_rows * gt_ids.size + gt_columns
        intersections = np.bincount(
            flattened, minlength=pred_ids.size * gt_ids.size
        ).reshape(pred_ids.size, gt_ids.size)
    return evaluate_instance_overlap_counts(
        intersections=intersections,
        pred_sizes=pred_sizes,
        gt_sizes=gt_sizes,
        coverage_thresholds=coverage_thresholds,
    )


def evaluate_instance_masks(
    pred_masks: Sequence[np.ndarray],
    gt_masks: Sequence[np.ndarray],
    coverage_thresholds: Sequence[float] = DEFAULT_COVERAGE_THRESHOLDS,
) -> InstanceCounts:
    """
    直接从互相可重叠的 bool masks 计算正式 instance 计数。

    输入参数:
        - pred_masks: Sequence[np.ndarray], N_pred 张同 shape 预测组件 mask
        - gt_masks: Sequence[np.ndarray], N_gt 张同 shape occurrence GT mask
        - coverage_thresholds: Sequence[float], 双向 coverage 阈值

    输出:
        - counts: InstanceCounts, 与标签图入口相同，但允许 GT masks 彼此重叠
    """
    pred = [np.asarray(mask, dtype=np.bool_) for mask in pred_masks]
    gt = [np.asarray(mask, dtype=np.bool_) for mask in gt_masks]
    shapes = {mask.shape for mask in [*pred, *gt]}
    if len(shapes) > 1:
        raise ValueError("全部 pred/GT masks 必须具有相同 shape")
    pred_sizes = np.asarray([mask.sum() for mask in pred], dtype=np.int64)
    gt_sizes = np.asarray([mask.sum() for mask in gt], dtype=np.int64)
    intersections = np.asarray(
        [[np.logical_and(pred_mask, gt_mask).sum() for gt_mask in gt] for pred_mask in pred],
        dtype=np.int64,
    ).reshape(len(pred), len(gt))
    return evaluate_instance_overlap_counts(
        intersections=intersections,
        pred_sizes=pred_sizes,
        gt_sizes=gt_sizes,
        coverage_thresholds=coverage_thresholds,
    )


def evaluate_instance_overlap_counts(
    intersections: np.ndarray,
    pred_sizes: np.ndarray,
    gt_sizes: np.ndarray,
    coverage_thresholds: Sequence[float],
) -> InstanceCounts:
    """
    从已经汇总的交集与实例体积计算 coverage 和固定 Hungarian 计数。

    输入参数:
        - intersections: np.ndarray, [N_pred,N_gt]，每个预测/GT 对的交集体素数
        - pred_sizes: np.ndarray, [N_pred]，各预测实例体素数
        - gt_sizes: np.ndarray, [N_gt]，各 GT occurrence 体素数
        - coverage_thresholds: Sequence[float]，双向 coverage 阈值；正式值为 (0.3,0.5)

    输出:
        - counts: InstanceCounts，先只按连续 `sqrt(c_pred*c_GT)` 求一次固定 Hungarian，
          再在不改变配对的前提下逐 coverage 阈值计数

    该入口让 overlap.npz 等稀疏基础事实直接复用正式指标，不要求重新物化完整 mask。
    """
    thresholds = tuple(float(value) for value in coverage_thresholds)
    if any(value <= 0.0 or value > 1.0 for value in thresholds):
        raise ValueError("coverage_thresholds 必须位于 (0,1]")
    n_pred, n_gt = int(pred_sizes.size), int(gt_sizes.size)
    if intersections.shape != (n_pred, n_gt):
        raise ValueError("intersections shape 与 pred/GT sizes 不一致")
    pred_cover = np.divide(
        intersections,
        pred_sizes[:, None],
        out=np.zeros((n_pred, n_gt), dtype=np.float64),
        where=pred_sizes[:, None] > 0,
    )
    gt_cover = np.divide(
        intersections,
        gt_sizes[None, :],
        out=np.zeros((n_pred, n_gt), dtype=np.float64),
        where=gt_sizes[None, :] > 0,
    )
    coverage_pred_hit = np.zeros(len(thresholds), dtype=np.int64)
    coverage_gt_hit = np.zeros(len(thresholds), dtype=np.int64)
    one_to_one_tp = np.zeros(len(thresholds), dtype=np.int64)

    if n_pred and n_gt:
        continuous_score = np.sqrt(pred_cover * gt_cover)
        matched_pred, matched_gt = linear_sum_assignment(-continuous_score)
        matched_pred_cover = pred_cover[matched_pred, matched_gt]
        matched_gt_cover = gt_cover[matched_pred, matched_gt]
        for row, threshold in enumerate(thresholds):
            valid_edges = (pred_cover >= threshold) & (gt_cover >= threshold)
            coverage_pred_hit[row] = int(valid_edges.any(axis=1).sum())
            coverage_gt_hit[row] = int(valid_edges.any(axis=0).sum())
            one_to_one_tp[row] = int(
                ((matched_pred_cover >= threshold) & (matched_gt_cover >= threshold)).sum()
            )
    return InstanceCounts(
        coverage_thresholds=thresholds,
        n_pred=n_pred,
        n_gt=n_gt,
        coverage_pred_hit=coverage_pred_hit,
        coverage_gt_hit=coverage_gt_hit,
        one_to_one_tp=one_to_one_tp,
    )


def aggregate_instance_counts(counts: Sequence[InstanceCounts]) -> InstanceCounts:
    """
    对多个 PDB 的计数求和，得到论文表中使用的 global/micro instance 指标。

    输入参数:
        - counts: Sequence[InstanceCounts], 同一 coverage threshold 顺序的逐 PDB 计数

    输出:
        - total: InstanceCounts, 分子和分母均逐 PDB 求和后的计数
    """
    if len(counts) == 0:
        raise ValueError("至少需要一个 PDB 的 InstanceCounts")
    thresholds = counts[0].coverage_thresholds
    if any(item.coverage_thresholds != thresholds for item in counts):
        raise ValueError("聚合的 PDB 必须使用相同 coverage_thresholds")
    return InstanceCounts(
        coverage_thresholds=thresholds,
        n_pred=sum(item.n_pred for item in counts),
        n_gt=sum(item.n_gt for item in counts),
        coverage_pred_hit=np.sum(
            np.stack([item.coverage_pred_hit for item in counts]), axis=0, dtype=np.int64
        ),
        coverage_gt_hit=np.sum(
            np.stack([item.coverage_gt_hit for item in counts]), axis=0, dtype=np.int64
        ),
        one_to_one_tp=np.sum(
            np.stack([item.one_to_one_tp for item in counts]), axis=0, dtype=np.int64
        ),
    )


def evaluate_topk_success(
    pred_masks: Sequence[np.ndarray],
    gt_masks: Sequence[np.ndarray],
    candidate_scores: Sequence[float],
    topk_values: Sequence[int] = DEFAULT_TOPK_VALUES,
    coverage_thresholds: Sequence[float] = DEFAULT_COVERAGE_THRESHOLDS,
) -> dict[str, int]:
    """
    按连续候选质量分取 top-K，并检查是否存在任一双向 coverage 达标 pair。

    输入参数:
        - pred_masks: Sequence[np.ndarray], 与 candidate_scores 同序的预测组件 masks
        - gt_masks: Sequence[np.ndarray], 当前 PDB 全部 occurrence GT masks
        - candidate_scores: Sequence[float], F1 路线为组件内概率均值，Selector 为
          `predicted_max_iou`；不得传 `selection_logit`
        - topk_values: Sequence[int], 正式值为 (3,4,5)
        - coverage_thresholds: Sequence[float], 正式值为 (0.3,0.5)

    输出:
        - result: dict[str,int], `n_topk_eligible_pdb` 在 n_gt>0 时为 1，另含
          每个 K/threshold 的 0/1 success；无预测仍记失败
    """
    if len(pred_masks) != len(candidate_scores):
        raise ValueError("pred_masks 与 candidate_scores 长度必须一致")
    pred = [np.asarray(mask, dtype=np.bool_) for mask in pred_masks]
    gt = [np.asarray(mask, dtype=np.bool_) for mask in gt_masks]
    shapes = {mask.shape for mask in [*pred, *gt]}
    if len(shapes) > 1:
        raise ValueError("全部 pred/GT masks 必须具有相同 shape")
    intersections = np.asarray(
        [[np.logical_and(pred_mask, gt_mask).sum() for gt_mask in gt] for pred_mask in pred],
        dtype=np.int64,
    ).reshape(len(pred), len(gt))
    return evaluate_topk_overlap_counts(
        intersections=intersections,
        pred_sizes=np.asarray([mask.sum() for mask in pred], dtype=np.int64),
        gt_sizes=np.asarray([mask.sum() for mask in gt], dtype=np.int64),
        candidate_scores=candidate_scores,
        topk_values=topk_values,
        coverage_thresholds=coverage_thresholds,
    )


def evaluate_topk_overlap_counts(
    intersections: np.ndarray,
    pred_sizes: np.ndarray,
    gt_sizes: np.ndarray,
    candidate_scores: Sequence[float],
    topk_values: Sequence[int] = DEFAULT_TOPK_VALUES,
    coverage_thresholds: Sequence[float] = DEFAULT_COVERAGE_THRESHOLDS,
) -> dict[str, int]:
    """
    直接从稀疏 overlap 汇总值计算单个 PDB 的 top-K success。

    输入参数:
        - intersections: np.ndarray, [N_pred,N_gt]，预测与 occurrence 的交集体素数
        - pred_sizes: np.ndarray, [N_pred]，预测组件体素数
        - gt_sizes: np.ndarray, [N_gt]，GT occurrence 体素数
        - candidate_scores: Sequence[float]，[N_pred]，连续候选质量分
        - topk_values: Sequence[int]，正式值为 (3,4,5)
        - coverage_thresholds: Sequence[float]，正式值为 (0.3,0.5)

    输出:
        - result: dict[str,int]，每个 K/阈值为单 PDB 0/1；`n_topk_eligible_pdb`
          只由是否存在 GT 决定。该指标不做 Hungarian。
    """
    pred_sizes_array = np.asarray(pred_sizes, dtype=np.int64).reshape(-1)
    gt_sizes_array = np.asarray(gt_sizes, dtype=np.int64).reshape(-1)
    overlap = np.asarray(intersections, dtype=np.int64)
    scores = np.asarray(candidate_scores, dtype=np.float64).reshape(-1)
    n_pred = int(pred_sizes_array.size)
    n_gt = int(gt_sizes_array.size)
    if overlap.shape != (n_pred, n_gt) or scores.shape != (n_pred,):
        raise ValueError("top-K overlap/sizes/scores shape 不一致")
    result: dict[str, int] = {"n_topk_eligible_pdb": int(n_gt > 0)}
    thresholds = tuple(float(value) for value in coverage_thresholds)
    for k in topk_values:
        for threshold in thresholds:
            result[f"top{int(k)}_success_{_threshold_tag(threshold)}"] = 0
    if n_gt == 0 or n_pred == 0:
        return result

    pred_cover = np.divide(
        overlap,
        pred_sizes_array[:, None],
        out=np.zeros((n_pred, n_gt), dtype=np.float64),
        where=pred_sizes_array[:, None] > 0,
    )
    gt_cover = np.divide(
        overlap,
        gt_sizes_array[None, :],
        out=np.zeros((n_pred, n_gt), dtype=np.float64),
        where=gt_sizes_array[None, :] > 0,
    )
    order = np.argsort(-scores, kind="mergesort")
    for k in topk_values:
        selected = order[: int(k)]
        for threshold in thresholds:
            valid = (pred_cover[selected] >= threshold) & (
                gt_cover[selected] >= threshold
            )
            result[f"top{int(k)}_success_{_threshold_tag(threshold)}"] = int(
                bool(np.any(valid))
            )
    return result


def _safe_div(numerator: int, denominator: int) -> float:
    """以 0.0 处理空分母，避免无实例样本产生 NaN。"""
    return float(numerator) / float(denominator) if denominator else 0.0


def _harmonic_mean(left: float, right: float) -> float:
    """返回两个非负比率的调和平均；和为 0 时返回 0。"""
    return 2.0 * left * right / (left + right) if left + right else 0.0


def _threshold_tag(threshold: float) -> str:
    """把 0.3/0.5 等阈值转换为稳定 JSON key 后缀。"""
    return f"{float(threshold):.3f}".rstrip("0").rstrip(".").replace(".", "p")
