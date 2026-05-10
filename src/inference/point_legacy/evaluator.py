"""
evaluator.py - 评估指标模块

提供点云级与体素级语义分割的评估指标:
    - Precision (精确率)
    - Recall (召回率)
    - F1-Score (F1 分数)
    - IoU (交并比 / Jaccard Index)

体素级评估完全依赖点云级评估的 hit 结果:
    - 一个体素被预测为正类 ⟺ 该体素内存在 ≥1 个被预测为正类的受体原子
    - 一个被预测为正类的体素是 hit ⟺ 该体素内有 ≥1 个预测正类原子在点云评估中处于 hit 状态
"""
# see me: 目前的整个 pipline, 一律假定为 2 分类

import numpy as np
 

# =============================================================================
# 1. 点云级语义分割评估
# =============================================================================
def _compute_hit_mask(
    src_points: np.ndarray,
    tgt_points: np.ndarray,
    dist_threshold: float,
    chunk_size: int = 2048,
) -> np.ndarray:
    """
    计算 src_points 中每个点是否被 tgt_points 命中（存在任意 tgt_points 距离 <= dist_threshold）。

    输入参数:
        - src_points: np.ndarray, (N_src, 3), 源点云 (世界坐标, 单位 Å)
        - tgt_points: np.ndarray, (N_tgt, 3), 目标点云 (世界坐标, 单位 Å)
        - dist_threshold: float, 距离阈值 (Å)
        - chunk_size: int, 分块大小, 避免一次性占用过大内存

    输出:
        - hit_mask: np.ndarray, (N_src,), bool, 每个源点是否命中
    """
    if src_points.size == 0 or tgt_points.size == 0:
        return np.zeros((src_points.shape[0],), dtype=bool)

    # float, 命中阈值的平方
    dist2_threshold = float(dist_threshold) ** 2
    # np.ndarray, (N_src,), bool, 每个源点是否命中
    hit_mask = np.zeros((src_points.shape[0],), dtype=bool)

    for start in range(0, src_points.shape[0], chunk_size):
        end = start + chunk_size
        sub_points = src_points[start:end]
        # np.ndarray, (M, N_tgt, 3), 点对差值
        diff = sub_points[:, None, :] - tgt_points[None, :, :]
        # np.ndarray, (M, N_tgt), 平方距离
        dist2 = np.sum(diff * diff, axis=2)
        hit_mask[start:end] = np.any(dist2 <= dist2_threshold, axis=1)

    return hit_mask


def _count_hits(
    src_points: np.ndarray,
    tgt_points: np.ndarray,
    dist_threshold: float,
    chunk_size: int = 2048,
) -> int:
    """
    统计 src_points 中被命中的点数（存在任意 tgt_points 距离 <= dist_threshold）。

    输入参数:
        - src_points: np.ndarray, (N_src, 3), 源点云 (世界坐标, 单位 Å)
        - tgt_points: np.ndarray, (N_tgt, 3), 目标点云 (世界坐标, 单位 Å)
        - dist_threshold: float, 距离阈值 (Å)
        - chunk_size: int, 分块大小, 避免一次性占用过大内存

    输出:
        - hit_count: int, 命中点数量
    """
    return int(np.sum(_compute_hit_mask(src_points, tgt_points, dist_threshold, chunk_size)))


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
# 2. 体素级语义分割评估
# =============================================================================
def _coords_to_voxel_set(
    coords_world: np.ndarray,
    origin: np.ndarray,
    voxel_size: np.ndarray,
    grid_shape_zyx: np.ndarray,
) -> set:
    """
    将世界坐标映射为 voxel index 元组的集合 (去重)。

    输入参数:
        - coords_world: np.ndarray, (N, 3), 世界坐标 (x, y, z), 单位 Å
        - origin: np.ndarray, (3,), 密度图原点 (x, y, z)
        - voxel_size: np.ndarray, (3,), 体素大小 (x, y, z)
        - grid_shape_zyx: np.ndarray, (3,), 密度图尺寸 (D, H, W)

    输出:
        - voxel_set: set[tuple[int, int, int]], 有效 voxel 索引 (z, y, x) 的集合
    """
    if coords_world.size == 0:
        return set()

    # np.ndarray, (3,), float64
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    voxel_size = np.asarray(voxel_size, dtype=np.float64).reshape(3)
    grid_shape_zyx = np.asarray(grid_shape_zyx, dtype=np.int64).reshape(3)

    # np.ndarray, (N, 3), float64, 连续 voxel 坐标 (x, y, z)
    local_voxel = (np.asarray(coords_world, dtype=np.float64) - origin[None, :]) / voxel_size[None, :]
    # np.ndarray, (N, 3), int64, home voxel 索引 (x, y, z)
    voxel_idx_xyz = np.floor(local_voxel).astype(np.int64)

    # int, int, int, 网格空间上界
    depth, height, width = int(grid_shape_zyx[0]), int(grid_shape_zyx[1]), int(grid_shape_zyx[2])

    # np.ndarray, (N,), bool, 过滤越界索引
    valid_mask = np.all(voxel_idx_xyz >= 0, axis=1)
    valid_mask &= voxel_idx_xyz[:, 0] < width
    valid_mask &= voxel_idx_xyz[:, 1] < height
    valid_mask &= voxel_idx_xyz[:, 2] < depth

    # np.ndarray, (N_valid, 3), int64, 合法 voxel 索引 (x, y, z)
    valid_idx = voxel_idx_xyz[valid_mask]
    # set[tuple], 转为 (z, y, x) 元组集合
    return {(int(row[2]), int(row[1]), int(row[0])) for row in valid_idx}


def voxel_semantic_evaluate(
    pred_atom_coords: np.ndarray,
    atom_gt: np.ndarray,
    dist_threshold: float,
    origin: np.ndarray,
    voxel_size: np.ndarray,
    grid_shape_zyx: np.ndarray,
) -> dict:
    """
    基于点云 hit 结果进行体素级语义分割评估。

    体素正负类别完全依赖于原子正负类别:
        - 一个体素被预测为正类 ⟺ 该体素内存在 ≥1 个被预测为正类的受体原子
        - 一个被预测为正类的体素是 hit ⟺ 该体素内有 ≥1 个预测正类原子在点云评估中处于 hit 状态

    输入参数:
        - pred_atom_coords: np.ndarray, (N_pred, 3), 预测为正类的原子坐标 (由 point_semantic_segment 产出)
        - atom_gt: np.ndarray, (N_gt, 3), GT 正类原子坐标
        - dist_threshold: float, 命中距离阈值 (Å)
        - origin: np.ndarray, (3,), 密度图原点 (x, y, z)
        - voxel_size: np.ndarray, (3,), 体素大小 (x, y, z)
        - grid_shape_zyx: np.ndarray, (3,), 密度图尺寸 (D, H, W)

    输出:
        - metrics: dict, 包含:
            - "mode": "voxel"
            - "precision": float
            - "recall": float
            - "f1": float
            - "iou": float
            - "num_pred": int, 预测正类体素数
            - "num_gt": int, GT 正类体素数
            - "hit_pred": int, 被命中的预测正类体素数
            - "hit_gt": int, 被命中的 GT 正类体素数
    """
    # np.ndarray, (N_pred, 3)
    pred_points = np.asarray(pred_atom_coords, dtype=np.float32).reshape(-1, 3)
    # np.ndarray, (N_gt, 3)
    gt_points = np.asarray(atom_gt, dtype=np.float32).reshape(-1, 3)
    num_pred_atoms = int(pred_points.shape[0])
    num_gt_atoms = int(gt_points.shape[0])

    # 特殊情况: 没有任何正类且模型也没预测出正类 → 完美
    if num_pred_atoms == 0 and num_gt_atoms == 0:
        return {
            "mode": "voxel",
            "precision": 1.0, "recall": 1.0, "f1": 1.0, "iou": 1.0,
            "num_pred": 0, "num_gt": 0, "hit_pred": 0, "hit_gt": 0,
        }

    # ---- 1. 计算原子级 hit mask ----
    # np.ndarray, (N_pred,), bool, 每个预测正类原子是否命中某个 GT 原子
    pred_hit_mask = _compute_hit_mask(pred_points, gt_points, dist_threshold)
    # np.ndarray, (N_gt,), bool, 每个 GT 正类原子是否被某个预测原子命中
    gt_hit_mask = _compute_hit_mask(gt_points, pred_points, dist_threshold)

    # ---- 2. 映射到 voxel 集合 ----
    # set[tuple], 全部预测正类原子对应的 voxel 集合
    pred_voxels = _coords_to_voxel_set(pred_points, origin, voxel_size, grid_shape_zyx)
    # set[tuple], 被命中的预测正类原子对应的 voxel 集合
    hit_pred_voxels = _coords_to_voxel_set(pred_points[pred_hit_mask], origin, voxel_size, grid_shape_zyx)
    # set[tuple], 全部 GT 正类原子对应的 voxel 集合
    gt_voxels = _coords_to_voxel_set(gt_points, origin, voxel_size, grid_shape_zyx)
    # set[tuple], 被命中的 GT 正类原子对应的 voxel 集合
    hit_gt_voxels = _coords_to_voxel_set(gt_points[gt_hit_mask], origin, voxel_size, grid_shape_zyx)

    # ---- 3. 计算指标 ----
    num_pred = len(pred_voxels)
    num_gt = len(gt_voxels)
    hit_pred = len(hit_pred_voxels)
    hit_gt = len(hit_gt_voxels)

    precision = hit_pred / num_pred if num_pred > 0 else 0.0
    recall = hit_gt / num_gt if num_gt > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    tp = hit_pred
    fp = num_pred - hit_pred
    fn = num_gt - hit_gt
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "mode": "voxel",
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "num_pred": num_pred,
        "num_gt": num_gt,
        "hit_pred": hit_pred,
        "hit_gt": hit_gt,
    }




# =============================================================================
# 3. 格式化打印
# =============================================================================
def print_metrics(metrics: dict, prefix: str = "") -> None:
    """
    将评估指标格式化打印到控制台。

    输入参数:
        - metrics: dict, semantic_evaluate() 或 voxel_semantic_evaluate() 的返回值
        - prefix: str, 打印前缀（如样本名）
    """
    mode = metrics.get("mode", "point")
    if prefix:
        print(f"  {prefix}")

    mode_label = "[点云级]" if mode == "point" else "[体素级]"
    print(f"    {mode_label} Precision = {metrics['precision']:.4f}")
    print(f"    {mode_label} Recall    = {metrics['recall']:.4f}")
    print(f"    {mode_label} F1        = {metrics['f1']:.4f}")
    print(f"    {mode_label} IoU       = {metrics['iou']:.4f}")

    if mode in ("point", "voxel"):
        unit = "原子" if mode == "point" else "体素"
        print(f"    ( num_pred_{unit}={metrics['num_pred']}  num_gt_{unit}={metrics['num_gt']}  "
              f"hit_pred={metrics['hit_pred']}  hit_gt={metrics['hit_gt']} )")
    else:
        print(f"    ( TP={metrics.get('tp', '?')}  FP={metrics.get('fp', '?')}  "
              f"FN={metrics.get('fn', '?')}  TN={metrics.get('tn', '?')} )")
