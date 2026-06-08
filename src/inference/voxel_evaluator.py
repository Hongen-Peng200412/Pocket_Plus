from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

# tuple[float, ...], 固定覆盖率阈值; instance 匹配与 top-K 命中统一复用, 不参与参数搜索
DEFAULT_COVERAGE_THRESHOLDS: tuple[float, ...] = (0.3, 0.6)
# tuple[int, ...], 固定 top-K 取值; 仅用于 per-sample successful ratio
DEFAULT_TOPK_VALUES: tuple[int, ...] = (3, 4, 5)


def evaluate_voxel_mask(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> dict[str, float | int]:
    """
    评估 voxel 二值预测掩码。

    输入参数:
        - pred_mask: np.ndarray, (D,H,W), 预测二值掩码
        - gt_mask: np.ndarray, (D,H,W), GT 二值掩码

    输出:
        - metrics: dict[str, float | int], voxel 二值评估指标, 包含:
            - "voxel_precision": float, 体素级精确率
            - "voxel_recall": float, 体素级召回率
            - "voxel_f1": float, 体素级 F1
            - "voxel_iou": float, 体素级 IoU
            - "voxel_dice": float, 体素级 Dice
            - "tp": int, 真阳性体素数
            - "fp": int, 假阳性体素数
            - "fn": int, 假阴性体素数
            - "tn": int, 真阴性体素数
    """
    # np.ndarray, (D,H,W), bool, 预测正类掩码
    pred_bool = np.asarray(pred_mask, dtype=bool)
    # np.ndarray, (D,H,W), bool, GT 正类掩码
    gt_bool = np.asarray(gt_mask, dtype=bool)
    if pred_bool.shape != gt_bool.shape:
        raise ValueError(f"pred_mask.shape={pred_bool.shape} 与 gt_mask.shape={gt_bool.shape} 不一致")

    tp = int(np.logical_and(pred_bool, gt_bool).sum())
    fp = int(np.logical_and(pred_bool, ~gt_bool).sum())
    fn = int(np.logical_and(~pred_bool, gt_bool).sum())
    tn = int(np.logical_and(~pred_bool, ~gt_bool).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    iou = _safe_div(tp, tp + fp + fn)
    dice = _safe_div(2 * tp, 2 * tp + fp + fn)
    return {
        "voxel_precision": precision,
        "voxel_recall": recall,
        "voxel_f1": f1,
        "voxel_iou": iou,
        "voxel_dice": dice,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def evaluate_voxel_pr_auc(
    score_map: np.ndarray,
    gt_ligand_mask: np.ndarray,
    hardmask: np.ndarray,
) -> dict[str, float | None]:
    """
    计算体素语义级 PR-AUC(Average Precision, 阶梯式), 阈值无关、只依赖排序。

    口径:
        - 有效区 valid = hardmask==0; 只在有效区上以 score 为分数、gt 为标签。
        - AP = Σ (R_n − R_{n−1})·P_n(前置 R=0,P=1), 在 distinct score 边界取点, 等价 sklearn average_precision_score。
        - AP 对分数的单调变换不变, 故 baseline 的 rank 均衡分数与原始差图分数得到相同 AP, 可与 DL 概率图直接比较。

    输入参数:
        - score_map: np.ndarray, (D,H,W), 体素分数图(DL 概率或 baseline 均衡分数)
        - gt_ligand_mask: np.ndarray, (D,H,W), GT ligand 二值掩码
        - hardmask: np.ndarray, (D,H,W), 受体原子占据掩码(0 为有效区)

    输出:
        - result: dict[str, float | None], {"pr_auc": AP}; 有效区无 GT 正类时 pr_auc 为 None(调用方负责 warn 并不计入 macro)
    """
    # np.ndarray, (D,H,W), bool, 有效区(非受体原子)
    valid_mask = np.asarray(hardmask) == 0
    # np.ndarray, (D,H,W), float64, 体素分数
    score = np.asarray(score_map, dtype=np.float64)
    # np.ndarray, (D,H,W), bool, GT ligand 掩码
    gt_bool = np.asarray(gt_ligand_mask, dtype=bool)
    if not (score.shape == valid_mask.shape == gt_bool.shape):
        raise ValueError(f"shape 不一致: score={score.shape}, hardmask={valid_mask.shape}, gt={gt_bool.shape}")

    # np.ndarray, (N_valid,), float64, 有效区分数
    valid_scores = score[valid_mask]
    # np.ndarray, (N_valid,), int64, 有效区标签(0/1)
    valid_labels = gt_bool[valid_mask].astype(np.int64, copy=False)
    # int, 有效区 GT 正类体素数
    num_positive = int(valid_labels.sum())
    if num_positive == 0:
        return {"pr_auc": None}

    # np.ndarray, (N_valid,), int64, 按分数降序的稳定排序索引
    order = np.argsort(-valid_scores, kind="mergesort")
    # np.ndarray, (N_valid,), float64, 降序后的分数(用于定位 distinct 阈值边界)
    sorted_scores = valid_scores[order]
    # np.ndarray, (N_valid,), int64, 降序后的标签
    sorted_labels = valid_labels[order]
    # np.ndarray, (N_valid,), int64, 累计真阳性
    cum_tp = np.cumsum(sorted_labels)
    # np.ndarray, (N_valid,), int64, 累计假阳性
    cum_fp = np.cumsum(1 - sorted_labels)
    # np.ndarray, (N_thr,), int64, 每个 distinct 分数的最后一个位置索引
    boundary = np.r_[np.where(np.diff(sorted_scores) != 0.0)[0], sorted_scores.shape[0] - 1]  # np.where(...)[0] → 取出"分数发生变化"的那些下标
    # np.ndarray, (N_thr,), int64, 各阈值的 TP / FP
    tp = cum_tp[boundary]
    fp = cum_fp[boundary]
    # np.ndarray, (N_thr,), float64, 各阈值的 precision / recall
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / float(num_positive)
    # np.ndarray, (N_thr+1,), float64, 前置 recall=0 / precision=1 后的曲线点
    recall_curve = np.r_[0.0, recall]
    precision_curve = np.r_[1.0, precision]
    # float, Average Precision
    average_precision = float(np.sum(np.diff(recall_curve) * precision_curve[1:]))
    return {"pr_auc": average_precision}


def build_instance_overlap_stats(
    pred_instance_label: np.ndarray,
    gt_instance_label: np.ndarray,
) -> dict[str, Any]:
    """
    统计预测 instance 与 GT instance 的体素数、重叠数和覆盖率矩阵; 供 global 匹配与 top-K 复用。

    覆盖率定义:
        - pred_cover[i, j] = overlap(pred_i, gt_j) / size(pred_i)
        - gt_cover[i, j]   = overlap(pred_i, gt_j) / size(gt_j)

    输入参数:
        - pred_instance_label: np.ndarray, (D,H,W), 预测 instance 标签, 0 为背景, 允许正标签不连续
        - gt_instance_label: np.ndarray, (D,H,W), GT instance 标签, 0 为背景, 允许正标签不连续

    输出:
        - stats: dict[str, Any], 包含:
            - "pred_ids": np.ndarray, (N_pred,), int64, 预测 instance 原始正标签 ID, 从小到大排
            - "gt_ids": np.ndarray, (N_gt,), int64, GT instance 原始正标签 ID, 从小到大排
            - "pred_sizes": np.ndarray, (N_pred,), int64, 每个预测 instance 的体素数
            - "gt_sizes": np.ndarray, (N_gt,), int64, 每个 GT instance 的体素数
            - "overlap_count_matrix": np.ndarray, (N_pred,N_gt), int64, (i,j) 为 pred_i 与 gt_j 的交叉体素数
            - "pred_cover_matrix": np.ndarray, (N_pred,N_gt), float64, pred_i 被 gt_j 覆盖的比例
            - "gt_cover_matrix": np.ndarray, (N_pred,N_gt), float64, gt_j 被 pred_i 覆盖的比例
    """
    # np.ndarray, (D,H,W), int64, 预测 instance 标签
    pred_label = np.asarray(pred_instance_label, dtype=np.int64)
    # np.ndarray, (D,H,W), int64, GT instance 标签
    gt_label = np.asarray(gt_instance_label, dtype=np.int64)
    if pred_label.shape != gt_label.shape:
        raise ValueError(f"pred_instance_label.shape={pred_label.shape} 与 gt_instance_label.shape={gt_label.shape} 不一致")

    # np.ndarray, (D,H,W), bool, 预测正 instance 区域
    pred_positive_mask = pred_label > 0
    # np.ndarray, (D,H,W), bool, GT 正 instance 区域
    gt_positive_mask = gt_label > 0
    # pred_ids: np.ndarray, (N_pred=预测的instance个数,), int64, 预测 instance 原始正标签 ID(从小到大排)
    # pred_inverse: (pred_pos,) 所有预测正类体素在 pred_ids 中的位置
    pred_ids, pred_inverse = np.unique(pred_label[pred_positive_mask], return_inverse=True)
    # gt_ids: np.ndarray, (N_gt,), int64, GT instance 原始正标签 ID(从小到大排)
    # gt_inverse: (gt_pos,) 所有GT正类体素在 gt_ids 中的位置
    gt_ids, gt_inverse = np.unique(gt_label[gt_positive_mask], return_inverse=True)
    # np.ndarray, (N_pred,), int64, 每个预测 instance 的体素数
    pred_sizes = np.bincount(pred_inverse, minlength=pred_ids.shape[0]).astype(np.int64, copy=False)
    # np.ndarray, (N_gt,), int64, 每个 GT instance 的体素数
    gt_sizes = np.bincount(gt_inverse, minlength=gt_ids.shape[0]).astype(np.int64, copy=False)
    num_pred = int(pred_ids.shape[0])
    num_gt = int(gt_ids.shape[0])

    if num_pred > 0 and num_gt > 0:
        # np.ndarray, (D,H,W), bool, 同时属于预测和 GT instance 的重叠区域
        pair_mask = pred_positive_mask & gt_positive_mask
        if bool(pair_mask.any()):
            # np.ndarray, (N_pair,), int64, 重叠体素对应的预测 instance 压缩索引(从0开始)
            pair_pred_index = np.searchsorted(pred_ids, pred_label[pair_mask])
            # np.ndarray, (N_pair,), int64, 重叠体素对应的 GT instance 压缩索引(从0开始)
            pair_gt_index = np.searchsorted(gt_ids, gt_label[pair_mask])
            # np.ndarray, (N_pair,), int64, 二维重叠矩阵的一维展开索引
            pair_index = pair_pred_index * num_gt + pair_gt_index
            # np.ndarray, (N_pred,N_gt), int64, 预测 instance 与 GT instance 的交叉体素数
            overlap_count_matrix = np.bincount(
                pair_index,
                minlength=num_pred * num_gt,
            ).reshape(num_pred, num_gt)
        else:
            # np.ndarray, (N_pred,N_gt), int64, 无交叉体素时的空重叠矩阵
            overlap_count_matrix = np.zeros((num_pred, num_gt), dtype=np.int64)
        # np.ndarray, (N_pred,N_gt), float64, pred_i 被 gt_j 覆盖的比例
        pred_cover_matrix = overlap_count_matrix / pred_sizes[:, None].astype(np.float64)
        # np.ndarray, (N_pred,N_gt), float64, gt_j 被 pred_i 覆盖的比例
        gt_cover_matrix = overlap_count_matrix / gt_sizes[None, :].astype(np.float64)
    else:
        overlap_count_matrix = np.zeros((num_pred, num_gt), dtype=np.int64)
        pred_cover_matrix = np.zeros((num_pred, num_gt), dtype=np.float64)
        gt_cover_matrix = np.zeros((num_pred, num_gt), dtype=np.float64)

    return {
        "pred_ids": pred_ids,
        "gt_ids": gt_ids,
        "pred_sizes": pred_sizes,
        "gt_sizes": gt_sizes,
        "overlap_count_matrix": overlap_count_matrix,
        "pred_cover_matrix": pred_cover_matrix,
        "gt_cover_matrix": gt_cover_matrix,
    }


def evaluate_global_instance_matching(
    pred_instance_label: np.ndarray,
    gt_instance_label: np.ndarray,
    coverage_thresholds: tuple[float, ...] = DEFAULT_COVERAGE_THRESHOLDS,
) -> dict[str, int]:
    """
    单样本一对一 Hungarian 匹配, 返回每样本 instance 计数, 上层会求和得到 global precision/recall/F1。
    匹配只做一次: 最大化 Σ sqrt(pred_cover * gt_cover); 同一组配对上, 分别按每个覆盖率阈值统计 valid pair 数 (pred_cover>=t 且 gt_cover>=t)。

    输入参数:
        - pred_instance_label: np.ndarray, (D,H,W), 预测 instance 标签, 0 为背景
        - gt_instance_label: np.ndarray, (D,H,W), GT instance 标签, 0 为背景
        - coverage_thresholds: tuple[float, ...], 覆盖率阈值集合, 默认 (0.3, 0.6)

    输出:
        - result: dict[str, int], 包含:
            - "num_pred_instances": int, 预测 instance 总数 (precision 分母累加项)
            - "num_gt_instances": int, GT instance 总数 (recall 分母累加项)
            - "tp_cov{tag}": int, 该覆盖率阈值下成功匹配的实例数; tag 形如 03/06
    """
    # dict[str, Any], 当前样本的 instance overlap 统计
    stats = build_instance_overlap_stats(pred_instance_label, gt_instance_label)

    # np.ndarray, (N_pred,N_gt), float64, pred 覆盖率矩阵
    pred_cover = stats["pred_cover_matrix"]
    # np.ndarray, (N_pred,N_gt), float64, gt 覆盖率矩阵
    gt_cover = stats["gt_cover_matrix"]
    num_pred = int(stats["pred_ids"].shape[0])
    num_gt = int(stats["gt_ids"].shape[0])

    # dict[str, int], 输出计数; 精确匹配 tp 与 loose 两个方向的 tp 先全部置 0
    result: dict[str, int] = {"num_pred_instances": num_pred, "num_gt_instances": num_gt}
    for threshold in coverage_thresholds:
        tag = _coverage_tag(threshold)
        result[f"tp_cov{tag}"] = 0
        result[f"tp_pred_loose_cov{tag}"] = 0
        result[f"tp_gt_loose_cov{tag}"] = 0
    if num_pred == 0 or num_gt == 0:
        return result

    # np.ndarray, (N_pred,N_gt), float64, 一对一匹配分数; 重叠为 0 的对分数为 0
    match_score = np.sqrt(pred_cover * gt_cover)
    # np.ndarray, (M,), int64, Hungarian 匹配的行/列索引, M = min(N_pred, N_gt)
    row_index, col_index = linear_sum_assignment(match_score, maximize=True)
    # np.ndarray, (M,), float64, 匹配对的 pred 覆盖率
    matched_pred_cover = pred_cover[row_index, col_index]
    # np.ndarray, (M,), float64, 匹配对的 gt 覆盖率
    matched_gt_cover = gt_cover[row_index, col_index]
    for threshold in coverage_thresholds:
        tag = _coverage_tag(threshold)
        # np.ndarray, (M,), bool, 该阈值下匹配对是否满足双向覆盖
        valid_pair = (matched_pred_cover >= float(threshold)) & (matched_gt_cover >= float(threshold))
        result[f"tp_cov{tag}"] = int(valid_pair.sum())
        # loose precision 分子: 被任意 GT 覆盖到阈值的预测 instance 数; loose recall 分子: 被任意 pred 覆盖到阈值的 GT instance 数
        result[f"tp_pred_loose_cov{tag}"] = int((pred_cover >= float(threshold)).any(axis=1).sum())
        result[f"tp_gt_loose_cov{tag}"] = int((gt_cover >= float(threshold)).any(axis=0).sum())
    return result


def evaluate_topk_success(
    pred_instance_label: np.ndarray,
    gt_instance_label: np.ndarray,
    candidates: list[Any],
    topk_values: tuple[int, ...] = DEFAULT_TOPK_VALUES,
    coverage_thresholds: tuple[float, ...] = DEFAULT_COVERAGE_THRESHOLDS,
) -> dict[str, int]:
    """
    单样本 top-K 命中: 按候选 score_mean 降序取前 min(K, n) 个预测 blob, 判断是否存在 valid pair(存在一个预测配体和真实配体, 它们的彼此对对方的覆盖率均大于 coverage_thresholds)。

    输入参数:
        - pred_instance_label: np.ndarray, (D,H,W), 预测 instance 标签, 0 为背景; 正 id 与 candidates.instance_id 对应
        - gt_instance_label: np.ndarray, (D,H,W), GT instance 标签, 0 为背景
        - candidates: list[VoxelCandidate], 可变长度, 当前样本预测的 instance 候选; 仅使用 instance_id 与 score_mean
        - topk_values: tuple[int, ...], top-K 取值集合, 默认 (3, 4, 5)
        - coverage_thresholds: tuple[float, ...], 覆盖率阈值集合, 默认 (0.3, 0.6)

    输出:
        - result: dict[str, int], 包含 "top{K}_success_cov{tag}": int(0/1), 每样本是否存在命中的配体
    """
    # dict[str, Any], 当前样本的 instance overlap 统计
    stats = build_instance_overlap_stats(pred_instance_label, gt_instance_label)

    # np.ndarray, (N_pred,), int64, 预测 instance 原始正标签 ID, 从小到大排
    pred_ids = stats["pred_ids"]
    # np.ndarray, (N_pred,N_gt), float64, pred 覆盖率矩阵
    pred_cover = stats["pred_cover_matrix"]
    # np.ndarray, (N_pred,N_gt), float64, gt 覆盖率矩阵
    gt_cover = stats["gt_cover_matrix"]
    num_pred = int(pred_ids.shape[0])
    num_gt = int(stats["gt_ids"].shape[0])

    # dict[str, int], 输出命中标记; 先全部置 0
    result: dict[str, int] = {}
    for k in topk_values:
        for threshold in coverage_thresholds:
            result[f"top{int(k)}_success_cov{_coverage_tag(threshold)}"] = 0
    if num_pred == 0 or num_gt == 0:
        return result

    # list[int], 候选按 score_mean 降序排列的原始 instance id
    sorted_instance_ids = [int(candidate.instance_id) for candidate in sorted(candidates, key=lambda c: float(c.score_mean), reverse=True)]
    for k in topk_values:
        # list[int], 当前 K 取的前 min(K, n) 个候选 id
        topk_ids = sorted_instance_ids[: int(k)]
        if len(topk_ids) == 0:
            continue
        # np.ndarray, (len(topk_ids),), int64, 候选 id 在 pred_ids 中的压缩行索引
        rows = np.searchsorted(pred_ids, np.asarray(topk_ids, dtype=np.int64))
        if not bool(np.all(pred_ids[rows] == np.asarray(topk_ids, dtype=np.int64))):
            raise ValueError("candidates.instance_id 与 pred_instance_label 的正标签不一致")
        for threshold in coverage_thresholds:
            # np.ndarray, (len(topk_ids),N_gt), bool, 前 K 候选与各 GT 是否构成 valid pair
            valid_pair = (pred_cover[rows] >= float(threshold)) & (gt_cover[rows] >= float(threshold))
            if bool(valid_pair.any()):
                result[f"top{int(k)}_success_cov{_coverage_tag(threshold)}"] = 1
    return result









# ----------------------------------------- 工具函数 ------------------------------------------
def _coverage_tag(threshold: float) -> str:
    """
    将覆盖率阈值转成命名标签, 用于指标键名。

    输入参数:
        - threshold: float, 覆盖率阈值, 如 0.3

    输出:
        - tag: str, 两位标签, 如 "03"; 0.6 -> "06"
    """
    return f"{int(round(float(threshold) * 10)):02d}"


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    """
    计算安全除法, 分母为 0 时返回 0.0。

    输入参数:
        - numerator: int | float, 分子
        - denominator: int | float, 分母

    输出:
        - value: float, 除法结果或 0.0
    """
    if float(denominator) == 0.0:
        return 0.0
    return float(numerator) / float(denominator)
