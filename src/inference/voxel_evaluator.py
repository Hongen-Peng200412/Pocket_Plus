from __future__ import annotations

import numpy as np

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


def evaluate_instance_mask(
    pred_instance_label: np.ndarray,
    gt_instance_label: np.ndarray,
    alpha: float,
    beta: float,
) -> dict[str, float | int]:
    """
    按包含率阈值评估 instance 级 precision/recall/F1。

    输入参数:
        - pred_instance_label: np.ndarray, (D,H,W), 预测 instance 标签, 0为背景, 允许正标签不连续
        - gt_instance_label: np.ndarray, (D,H,W), GT instance 标签, 0为背景, 允许正标签不连续
        - alpha: float, 预测 instance 被 GT 覆盖的 precision 判定阈值
        - beta: float, GT instance 被预测覆盖的 recall 判定阈值

    输出:
        - metrics: dict[str, float | int], instance 级评估指标, 包含:
            - "instance_precision": float, 预测 instance 的精确率
            - "instance_recall": float, GT instance 的召回率
            - "instance_f1": float, instance 级 F1
            - "num_pred_instances": int, 预测 instance 总数
            - "num_gt_instances": int, GT instance 总数
            - "pred_instance_tp": int, 满足 precision 阈值的预测 instance 数
            - "gt_instance_hit": int, 满足 recall 阈值的 GT instance 数
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha 必须在 [0,1], 实际为 {alpha}")
    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"beta 必须在 [0,1], 实际为 {beta}")

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
    # pred_ids: np.ndarray, (N_pred,), int64, 预测 instance 原始正标签 ID, 从小到大排; pred_ids[pred_inverse] = 原始输入pred_label[pred_positive_mask]
    pred_ids, pred_inverse = np.unique(pred_label[pred_positive_mask], return_inverse=True)
    # gt_ids: np.ndarray, (N_gt,), int64, GT instance 原始正标签 ID
    gt_ids, gt_inverse = np.unique(gt_label[gt_positive_mask], return_inverse=True)
    # np.ndarray, (N_pred,), int64, 每个预测 instance 的体素数
    pred_sizes = np.bincount(pred_inverse, minlength=pred_ids.shape[0]).astype(np.int64, copy=False)
    # np.ndarray, (N_gt,), int64, 每个 GT instance 的体素数
    gt_sizes = np.bincount(gt_inverse, minlength=gt_ids.shape[0]).astype(np.int64, copy=False)
    num_pred_instances = int(pred_ids.shape[0])
    num_gt_instances = int(gt_ids.shape[0])

    pred_tp = 0
    gt_hit = 0
    if num_pred_instances > 0 and num_gt_instances > 0:
        # np.ndarray, (D,H,W), bool, 同时属于预测和 GT instance 的重叠区域
        pair_mask = pred_positive_mask & gt_positive_mask
        if bool(pair_mask.any()):
            # np.ndarray, (N_pair,), int64, 重叠体素对应的预测 instance 压缩索引(N_pair为pair_mask的非零数)
            pair_pred_index = np.searchsorted(pred_ids, pred_label[pair_mask])
            # np.ndarray, (N_pair,), int64, 重叠体素对应的 GT instance 压缩索引
            pair_gt_index = np.searchsorted(gt_ids, gt_label[pair_mask])
            # np.ndarray, (N_pair,), int64, 二维重叠矩阵的一维展开索引
            pair_index = pair_pred_index * num_gt_instances + pair_gt_index
            # np.ndarray, (N_pred,N_gt), int64, (i,j)表示预测的 instance_i(id第i小的) 与 GT instance_j 的交叉体素数
            overlap_count_matrix = np.bincount(
                pair_index,
                minlength=num_pred_instances * num_gt_instances,
            ).reshape(num_pred_instances, num_gt_instances)
        else:
            # np.ndarray, (N_pred,N_gt), int64, 无交叉体素时的空重叠矩阵
            overlap_count_matrix = np.zeros((num_pred_instances, num_gt_instances), dtype=np.int64)

        # np.ndarray, (N_pred,), float64, 每个预测 instance 被单个 GT instance 覆盖的最大比例
        pred_cover_ratio = overlap_count_matrix.max(axis=1) / pred_sizes.astype(np.float64)
        # np.ndarray, (N_gt,), float64, 每个 GT instance 被单个预测 instance 覆盖的最大比例
        gt_cover_ratio = overlap_count_matrix.max(axis=0) / gt_sizes.astype(np.float64)
        pred_tp = int((pred_cover_ratio >= float(alpha)).sum())
        gt_hit = int((gt_cover_ratio >= float(beta)).sum())

    instance_precision = _safe_div(pred_tp, num_pred_instances)
    instance_recall = _safe_div(gt_hit, num_gt_instances)
    instance_f1 = _safe_div(2.0 * instance_precision * instance_recall, instance_precision + instance_recall)
    return {
        "instance_precision": instance_precision,
        "instance_recall": instance_recall,
        "instance_f1": instance_f1,
        "num_pred_instances": num_pred_instances,
        "num_gt_instances": num_gt_instances,
        "pred_instance_tp": int(pred_tp),
        "gt_instance_hit": int(gt_hit),
    }


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

