"""
evaluator.py - 评估指标模块

提供点云级语义分割的评估指标:
    - Precision (精确率)
    - Recall (召回率)
    - F1-Score (F1 分数)
    - IoU (交并比 / Jaccard Index)
"""
# see me: 目前的整个 pipline, 一律假定为 2 分类

import numpy as np
 

# =============================================================================
# 1. 点云级语义分割评估
# =============================================================================
def _count_hits(
    src_points: np.ndarray,
    tgt_points: np.ndarray,
    dist_threshold: float,
    chunk_size: int = 2048,
) -> int:
    """
    统计 src_points 中被命中的点数（存在任意 tgt_points 距离 <= dist_threshold）。

    输入参数:
        - src_points: np.ndarray, 形状 (N_src, 3), 源点云 (世界坐标, 单位 Å)
        - tgt_points: np.ndarray, 形状 (N_tgt, 3), 目标点云 (世界坐标, 单位 Å)
        - dist_threshold: float, 距离阈值 (Å)
        - chunk_size: int, 分块大小，避免一次性占用过大内存

    输出:
        - hit_count: int, 命中点数量
    """
    if src_points.size == 0 or tgt_points.size == 0:
        return 0

    # float, 命中阈值的平方
    dist2_threshold = float(dist_threshold) ** 2
    # np.ndarray, 形状 (N_src,), bool, 每个源点是否命中
    hit_mask = np.zeros((src_points.shape[0],), dtype=bool)

    for start in range(0, src_points.shape[0], chunk_size):
        end = start + chunk_size
        sub_points = src_points[start:end]
        # np.ndarray, 形状 (M, N_tgt, 3), 点对差值
        diff = sub_points[:, None, :] - tgt_points[None, :, :]
        # np.ndarray, 形状 (M, N_tgt), 平方距离
        dist2 = np.sum(diff * diff, axis=2)
        hit_mask[start:end] = np.any(dist2 <= dist2_threshold, axis=1)

    return int(np.sum(hit_mask))


def semantic_evaluate(
    pred_atom_coords: np.ndarray,
    atom_gt: np.ndarray,
    dist_threshold: float,
) -> dict:
    """
    基于点云进行语义分割评估（预测正类原子坐标 vs GT 正类原子坐标 → 距离命中）。

    输入参数:
        - pred_atom_coords:  np.ndarray, float32, (N_pred, 3), 预测为正类的原子坐标. 由 postprocess 模块产出. 世界坐标, 顺序为xyz, 单位 Å
        - atom_gt:           np.ndarray, float32, (N_gt, 3), 所有正类原子的坐标（Ground Truth）. 世界坐标, 顺序为xyz, 单位 Å
        - dist_threshold:    float, 评估的距离阈值 (Å): 预测正类与真实正类 hit ↔ 距离 <= dist_threshold

    输出:
        - metrics: dict, 包含:
            - "precision":   float, 预测命中率
            - "recall":      float, 真实命中率
            - "f1":          float, F1 分数
            - "iou":         float, IoU 分数（基于 tp/fp/fn 计算）
            - "num_pred":    int,   预测正类原子数
            - "num_gt":      int,   真实正类原子数
            - "hit_pred":    int,   被命中的预测原子数
            - "hit_gt":      int,   被命中的真实原子数

    逻辑:
        - 根据 dist_threshold, 以及 (预测正类点云, 真实正类点云) 确定 hit 的原子数, 进而算各类指标
    """
    # np.ndarray, float32, (N_pred, 3), 预测正类点云
    pred_points = np.asarray(pred_atom_coords, dtype=np.float32).reshape(-1, 3)
    # np.ndarray, float32, (N_gt, 3), GT 正类点云
    gt_points = np.asarray(atom_gt, dtype=np.float32).reshape(-1, 3)

    # int, 预测正类原子数
    num_pred = int(pred_points.shape[0])
    # int, GT 正类原子数
    num_gt = int(gt_points.shape[0])

    # 特殊情况: 没有任何正类且模型也没预测出正类 → 视为完美
    if num_pred == 0 and num_gt == 0:
        return {
            "mode": "point",
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "iou": 1.0,
            "num_pred": 0,
            "num_gt": 0,
            "hit_pred": 0,
            "hit_gt": 0,
        }

    # int, 被命中的预测原子数
    hit_pred = _count_hits(pred_points, gt_points, dist_threshold)
    # int, 被命中的真实原子数
    hit_gt = _count_hits(gt_points, pred_points, dist_threshold)

    precision = hit_pred / num_pred if num_pred > 0 else 0.0
    recall = hit_gt / num_gt if num_gt > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    # 以 hit_pred/ hit_gt 构造 tp / fp / fn
    tp = hit_pred
    fp = num_pred - hit_pred
    fn = num_gt - hit_gt
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "mode": "point",
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "num_pred": num_pred,
        "num_gt": num_gt,
        "hit_pred": int(hit_pred),
        "hit_gt": int(hit_gt),
    }




# =============================================================================
# 2. 预留接口: 实例分割评估
# =============================================================================
def instance_evaluate(
    pred_instances: dict,
    gt_instances: dict,
    **kwargs,
) -> dict:
    """
    [预留接口] 实例分割评估。

    Args:
        - pred_instances: dict, 预测实例信息（格式待定义）
        - gt_instances:   dict, 真实实例信息（格式待定义）

    Returns:
        - metrics: dict, 实例分割评估指标

    Raises:
        NotImplementedError
    """
    raise NotImplementedError(
        "[evaluator] instance_evaluate() 尚未实现。"
    )


# =============================================================================
# 3. 格式化打印
# =============================================================================
def print_metrics(metrics: dict, prefix: str = "") -> None:
    """
    将评估指标格式化打印到控制台。

    # 输入参数:
        - metrics: dict, semantic_evaluate() 的返回值
        - prefix: str, 打印前缀（如样本名）
    """
    if prefix:
        print(f"  {prefix}")
    print(f"    Precision = {metrics['precision']:.4f}")
    print(f"    Recall    = {metrics['recall']:.4f}")
    print(f"    F1        = {metrics['f1']:.4f}")
    print(f"    IoU       = {metrics['iou']:.4f}")

    if metrics.get("mode") == "point":
        print(f"    ( num_pred={metrics['num_pred']}  num_gt={metrics['num_gt']}  "
              f"hit_pred={metrics['hit_pred']}  hit_gt={metrics['hit_gt']} )")
    else:
        print(f"    ( TP={metrics['tp']}  FP={metrics['fp']}  "
              f"FN={metrics['fn']}  TN={metrics['tn']}  "
              f"评估体素数={metrics['num_eval']} )")
