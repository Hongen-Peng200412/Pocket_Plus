"""
postprocess.py - 后处理模块

点云推断后处理:
    1. _compute_per_atom_spatial_weight()、 _compute_per_atom_core_weight: 为每个 BOX 的原子初始化空间权重
    2. merge_box_atom_results(): 合并多 BOX 的原子预测结果
    3. point_semantic_segment(): 原子概率 -> 语义分割（二值化）

旧版体素 -> 原子映射逻辑已迁移至 src/inference/legacy/postprocess_voxel.py
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


# =============================================================================
# 1. BOX 空间权重: 高斯核衰减 + 边缘衰减
# =============================================================================
def _compute_per_atom_spatial_weight(
    atom_coord_local_voxel: np.ndarray,
    box_shape_zyx: np.ndarray,
    voxel_size: np.ndarray,
    window_size: int,
    sigma_ratio: float,
) -> np.ndarray:
    """
    为每个 BOX 内的原子计算以该 BOX 自身中心为原点的高斯衰减权重。

    输入参数:
        - atom_coord_local_voxel: torch.Tensor, (N_box, 3), 原子的连续 voxel 坐标 (x, y, z)
        - box_shape_zyx: torch.Tensor, (3,), BOX 体素形状
        - voxel_size: np.ndarray, (3,), float, 体素大小 (x, y, z), 单位 A
        - window_size: int, 标量, BOX 窗口边长 (voxel)
        - sigma_ratio: float, 标量, 高斯核 sigma 与 BOX 半径之比

    输出:
        - box_spatial_weights: np.ndarray, (N_box,), float32, 该 BOX 内每个原子的空间权重
    """
    n_atom = int(atom_coord_local_voxel.shape[0])
    if n_atom == 0:
        return np.empty((0,), dtype=np.float32)

    voxel_size = np.asarray(voxel_size, dtype=np.float64).reshape(3)
    half_box_world = 0.5 * float(window_size) * voxel_size
    box_radius = float(np.linalg.norm(half_box_world))
    sigma = max(float(sigma_ratio) * box_radius, 1e-6)
    two_sigma2 = 2.0 * sigma * sigma

    box_shape_xyz = np.asarray(box_shape_zyx, dtype=np.float64)[[2, 1, 0]]
    box_center_voxel_xyz = box_shape_xyz * 0.5
    offset_voxel = np.asarray(atom_coord_local_voxel, dtype=np.float64) - box_center_voxel_xyz[np.newaxis, :]
    offset_world = offset_voxel * voxel_size[np.newaxis, :]
    dist2 = np.sum(offset_world * offset_world, axis=1)
    return np.exp(-dist2 / two_sigma2).astype(np.float32)

def _compute_per_atom_core_weight(
    atom_coord_local_voxel: np.ndarray,
    box_shape_zyx: np.ndarray,
    core_offset: int,
    core_decay_mode: str,
) -> np.ndarray:
    """
    计算每个原子基于 core_offset 的额外边界衰减。

    输入参数:
        - atom_coord_local_voxel: np.ndarray, (N, 3), float32, 连续 voxel 坐标 (x, y, z)
        - box_shape_zyx: np.ndarray, (3,), int64, BOX 形状, 顺序 (Z, Y, X)
        - core_offset: int, 标量, 裁边 voxel 数
        - core_decay_mode: str, 衰减模式, 可选 "hard" / "linear" / "none"

    输出:
        - weight: np.ndarray, (N,), float32, 边界衰减权重
    """
    n_atom = int(atom_coord_local_voxel.shape[0])
    if n_atom == 0:
        return np.empty((0,), dtype=np.float32)
    if core_decay_mode == "none":
        return np.ones((n_atom,), dtype=np.float32)
    if core_offset <= 0:
        return np.ones((n_atom,), dtype=np.float32)

    # np.ndarray, (3,), float32, BOX 形状, 顺序 (x, y, z)
    box_shape_xyz = box_shape_zyx[[2, 1, 0]].astype(np.float32)
    # np.ndarray, (N, 3), float32, 到低边界的距离
    dist_lower = atom_coord_local_voxel
    # np.ndarray, (N, 3), float32, 到高边界的距离
    dist_upper = box_shape_xyz[np.newaxis, :] - atom_coord_local_voxel
    # np.ndarray, (N,), float32, 到最近边界的距离
    margin_dist = np.minimum(dist_lower, dist_upper).min(axis=1)

    if core_decay_mode == "hard":
        weight = (margin_dist >= float(core_offset)).astype(np.float32)
    elif core_decay_mode == "linear":
        weight = np.clip(margin_dist / float(core_offset), 0.0, 1.0).astype(np.float32)
    else:
        raise ValueError(f"未知 core_decay_mode: {core_decay_mode}")

    return weight





# =============================================================================
# 2. 多 BOX 原子预测聚合, 变为全局结果
# =============================================================================
def merge_box_atom_results(
    box_results: list,
    total_atom_count: int,
    core_decay_mode: str,
    core_offset: int,
    merge_mode: str,
    voxel_size: np.ndarray,
    window_size: int,
    box_spatial_weight_sigma_ratio: float,
) -> np.ndarray:
    """
    合并多 BOX 的原子预测结果。

    输入参数:
        - box_results: list[dict], 每项包含一个 BOX 的原子级预测结果
            - "global_atom_indices": np.ndarray, (N_box,), int64, 该 BOX 内原子的全局索引
            - "atom_logits": np.ndarray, (N_box,) 或 (N_box, 1), float32, 原始 logits
            - "atom_is_in_core": np.ndarray, (N_box,), bool, 是否在 core BOX 内
            - "atom_coord_local_voxel": np.ndarray, (N_box, 3), float32, 连续 voxel 坐标
            - "box_shape_zyx": np.ndarray, (3,), int64, BOX 体素形状
            - "box_confidence_weight": float, 标量, 该 BOX 的额外置信度权重
        - total_atom_count: int, 标量, 全局原子总数
        - core_decay_mode: str, core 区边界衰减模式, 可选 "hard" / "linear" / "none"
        - core_offset: int, 标量, core 区裁边厚度, 单位 voxel
        - merge_mode: str, 标量, 多 BOX 聚合策略
            - "logit_mean": 先加权平均 logit, 再做 sigmoid
            - "prob_mean": 先将每个 BOX 的 logit 转为 prob, 再加权平均 prob
        - (以下三个参数用于高斯衰减核的生成)
        - voxel_size: np.ndarray, (3,), float32, 体素大小 (x, y, z), 单位 A
        - window_size: int, 标量, BOX 窗口边长 (voxel)
        - box_spatial_weight_sigma_ratio: float, 标量, 高斯核 sigma 与 BOX 半径之比

    输出:
        - atom_probs: np.ndarray, (total_atom_count,), float32, 每个全局原子的最终概率
    """
    if len(box_results) == 0:
        return np.zeros((total_atom_count,), dtype=np.float32)
    if merge_mode not in {"logit_mean", "prob_mean"}:
        raise ValueError(f"未知 merge_mode: {merge_mode}")

    # np.ndarray, (total_atom_count,), float64, 加权数值累加
    value_sum = np.zeros((total_atom_count,), dtype=np.float64)
    # np.ndarray, (total_atom_count,), float64, 权重累加
    weight_sum = np.zeros((total_atom_count,), dtype=np.float64)

    for box_res in box_results:
        # np.ndarray, (N_box,), int64, 当前 BOX 的原子全局索引
        indices = np.asarray(box_res["global_atom_indices"], dtype=np.int64)
        # np.ndarray, (N_box,) 或 (N_box, 1), float64, 当前 BOX 的原始 logits
        logits = np.asarray(box_res["atom_logits"], dtype=np.float64)
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits[:, 0]
        elif logits.ndim != 1:
            raise ValueError(f"Logits shape {logits.shape} is not supported.")

        if merge_mode == "logit_mean":
            # np.ndarray, (N_box,), float64, 待聚合的数值为 logit
            values = logits
        else:
            # np.ndarray, (N_box,), float64, 待聚合的数值为 probability
            values = _sigmoid(logits.astype(np.float32)).astype(np.float64)

        # np.ndarray, (N_box,), float64, BOX 中心高斯空间权重
        spatial_w = _compute_per_atom_spatial_weight(
            atom_coord_local_voxel=np.asarray(box_res["atom_coord_local_voxel"], dtype=np.float32),
            box_shape_zyx=np.asarray(box_res["box_shape_zyx"], dtype=np.int64),
            voxel_size=voxel_size,
            window_size=window_size,
            sigma_ratio=box_spatial_weight_sigma_ratio,
        ).astype(np.float64)

        # float, BOX 级置信度权重
        conf_w = float(box_res.get("box_confidence_weight", 1.0))
        # np.ndarray, (N_box,), float32, core 区边界衰减权重
        core_w = _compute_per_atom_core_weight(
            atom_coord_local_voxel=np.asarray(box_res["atom_coord_local_voxel"], dtype=np.float32),
            box_shape_zyx=np.asarray(box_res["box_shape_zyx"], dtype=np.int64),
            core_offset=core_offset,
            core_decay_mode=core_decay_mode,
        )
        # np.ndarray, (N_box,), float64, 当前 BOX 对原子的最终有效权重
        effective_w = spatial_w * core_w.astype(np.float64) * conf_w

        np.add.at(value_sum, indices, values * effective_w)
        np.add.at(weight_sum, indices, effective_w)

    # np.ndarray, (total_atom_count,), bool, 是否至少被一个 BOX 覆盖
    covered = weight_sum > 0.0
    # np.ndarray, (total_atom_count,), float64, 聚合后的中间结果
    merged_value = np.zeros((total_atom_count,), dtype=np.float64)
    merged_value[covered] = value_sum[covered] / weight_sum[covered]

    if merge_mode == "logit_mean":
        # np.ndarray, (total_atom_count,), float32, 对加权平均 logit 做 sigmoid
        atom_probs = _sigmoid(merged_value.astype(np.float32))
    else:
        # np.ndarray, (total_atom_count,), float32, 加权平均 probability
        atom_probs = np.clip(merged_value, 0.0, 1.0).astype(np.float32)

    atom_probs[~covered] = 0.0
    return atom_probs

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """
    数值稳定的 sigmoid 函数。

    输入参数:
        - x: np.ndarray, 任意形状, logit 数组

    输出:
        - y: np.ndarray, 与 x 同形, sigmoid 概率
    """
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    ).astype(np.float32)





# =============================================================================
# 3. (全局结果)原子概率 -> 语义分割（二值化）
# =============================================================================
def _point_semantic_segment_by_dbscan(
    atom_probs: np.ndarray,
    atom_coords: np.ndarray,
    threshold: float,
    dbscan_eps: float,
    dbscan_min_samples: int,
) -> np.ndarray:
    """
    使用 DBSCAN 对阈值后的正类原子点云做去孤立点后处理。

    输入参数:
        - atom_probs: np.ndarray, (N_atom,), float32, 全部原子的口袋概率
        - atom_coords: np.ndarray, (N_atom, 3), float32, 全部原子的世界坐标 (x, y, z), 单位 A
        - threshold: float, 标量, 概率阈值
        - dbscan_eps: float, 标量, DBSCAN 半径参数 eps, 单位 A
        - dbscan_min_samples: int, 标量, DBSCAN 的核心点最少邻居数

    输出:
        - pred_atom_coords: np.ndarray, (N_pred, 3), float32, DBSCAN 过滤后的正类原子坐标
    """
    if dbscan_eps <= 0.0:
        raise ValueError(f"dbscan_eps 必须 > 0, 当前值为 {dbscan_eps}")
    if dbscan_min_samples <= 0:
        raise ValueError(f"dbscan_min_samples 必须 > 0, 当前值为 {dbscan_min_samples}")

    # np.ndarray, (N_atom,), bool, 阈值后的正类掩码
    positive_mask = atom_probs >= threshold
    # np.ndarray, (N_pos, 3), float32, 正类原子坐标
    positive_coords = atom_coords[positive_mask].astype(np.float32, copy=False)
    n_positive = int(positive_coords.shape[0])
    if n_positive == 0:
        return np.empty((0, 3), dtype=np.float32)
    if n_positive < dbscan_min_samples:
        return np.empty((0, 3), dtype=np.float32)

    # cKDTree, 正类原子坐标的空间索引
    neighbor_tree = cKDTree(positive_coords)
    # list[list[int]], 每个点在 eps 半径内的邻居索引, 含自身
    neighbor_indices = neighbor_tree.query_ball_point(positive_coords, r=float(dbscan_eps))
    # np.ndarray, (N_pos,), int32, 每个点的邻居数, 含自身
    neighbor_count = np.asarray([len(indices) for indices in neighbor_indices], dtype=np.int32)

    # np.ndarray, (N_pos,), int32, 聚类标签; -1 表示噪声点
    labels = np.full((n_positive,), -1, dtype=np.int32)
    # np.ndarray, (N_pos,), bool, 是否已访问
    visited = np.zeros((n_positive,), dtype=bool)
    cluster_id = 0

    for seed_idx in range(n_positive):
        if visited[seed_idx]:
            continue

        visited[seed_idx] = True
        if neighbor_count[seed_idx] < dbscan_min_samples:
            continue

        labels[seed_idx] = cluster_id

        # list[int], 当前聚类的待扩展队列
        queue = list(neighbor_indices[seed_idx])
        # np.ndarray, (N_pos,), bool, 是否已经入队
        in_queue = np.zeros((n_positive,), dtype=bool)
        in_queue[np.asarray(queue, dtype=np.int64)] = True
        head = 0

        while head < len(queue):
            # int, 当前弹出的点索引
            point_idx = int(queue[head])
            head += 1

            if not visited[point_idx]:
                visited[point_idx] = True
                if neighbor_count[point_idx] >= dbscan_min_samples:
                    # np.ndarray, (N_neighbor,), int64, 当前点的全部邻居索引
                    expand_indices = np.asarray(neighbor_indices[point_idx], dtype=np.int64)
                    # np.ndarray, (N_new,), int64, 尚未入队的新邻居索引
                    new_indices = expand_indices[~in_queue[expand_indices]]
                    if new_indices.size > 0:
                        queue.extend(new_indices.tolist())
                        in_queue[new_indices] = True

            if labels[point_idx] == -1:
                labels[point_idx] = cluster_id

        cluster_id += 1

    # np.ndarray, (N_pos,), bool, 非噪声点掩码
    keep_mask = labels >= 0
    return positive_coords[keep_mask].astype(np.float32, copy=False)


def point_semantic_segment(
    atom_probs: np.ndarray,
    atom_coords: np.ndarray,
    threshold: float,
    semantic_segment_method: str,

    dbscan_eps: float,
    dbscan_min_samples: int,
) -> np.ndarray:
    """
    对原子概率做语义分割后处理，返回预测为正类的原子坐标。

    输入参数:
        - atom_probs: np.ndarray, (N_atom,), float32, 每个原子的口袋概率
        - atom_coords: np.ndarray, (N_atom, 3), float32, 原子的世界坐标 (x, y, z), 单位 A
        - threshold: float, 标量, 概率阈值
        - semantic_segment_method: str, 标量, 语义分割后处理方式
            - "threshold": 仅做阈值二值化
            - "dbscan": 先做阈值二值化, 再做 DBSCAN 去孤立点

        - dbscan_eps: float, 标量, DBSCAN 半径参数 eps, 仅在 method="dbscan" 时生效
        - dbscan_min_samples: int, 标量, DBSCAN 最少邻居数, 仅在 method="dbscan" 时生效

    输出:
        - pred_atom_coords: np.ndarray, (N_pred, 3), float32, 最终预测为正类的原子坐标
    """
    if atom_probs.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if atom_probs.shape[0] != atom_coords.shape[0]:
        raise ValueError(
            f"atom_probs 与 atom_coords 长度不一致: "
            f"atom_probs.shape={atom_probs.shape}, atom_coords.shape={atom_coords.shape}"
        )

    if semantic_segment_method == "threshold":
        # np.ndarray, (N_atom,), bool, 正类掩码
        positive_mask = atom_probs >= threshold
        # np.ndarray, (N_pred, 3), float32, 阈值后的正类原子坐标
        return atom_coords[positive_mask].astype(np.float32, copy=False)

    if semantic_segment_method == "dbscan":
        return _point_semantic_segment_by_dbscan(
            atom_probs=atom_probs,
            atom_coords=atom_coords,
            threshold=threshold,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
        )

    raise ValueError(f"未知 semantic_segment_method: {semantic_segment_method}")


# =============================================================================
# 工具函数: 原子坐标 → 体素 mask
# =============================================================================
def build_voxel_mask_from_coords(
    atom_coords_world: np.ndarray,
    origin: np.ndarray,
    voxel_size: np.ndarray,
    grid_shape_zyx: np.ndarray,
) -> np.ndarray:
    """
    将世界坐标下的原子集合映射为 (D, H, W) 的二值体素 mask。

    一个体素被标记为 1, 当且仅当存在 ≥1 个原子的 home voxel 落在该体素位置。

    输入参数:
        - atom_coords_world: np.ndarray, (N, 3), (可能是正类原子, 也可能是所有原子)原子世界坐标 (x, y, z), 单位 Å
        - origin: np.ndarray, (3,), 密度图原点 (x, y, z)
        - voxel_size: np.ndarray, (3,), 体素大小 (x, y, z)
        - grid_shape_zyx: np.ndarray, (3,), 密度图尺寸 (D, H, W)

    输出:
        - mask: np.ndarray, (D, H, W), int64, 取值 0 或 1
    """
    from src.datasets.box_geometry import build_hardmask_from_world_coordinates

    return build_hardmask_from_world_coordinates(
        atom_coords_world=np.asarray(atom_coords_world, dtype=np.float32),
        box_origin_world=np.asarray(origin, dtype=np.float32),
        voxel_size_world=np.asarray(voxel_size, dtype=np.float32),
        box_shape_zyx=np.asarray(grid_shape_zyx, dtype=np.int64),
    )
