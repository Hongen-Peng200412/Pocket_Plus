"""
postprocess.py — 后处理模块

点云推断后处理:
    1. init_box_spatial_weights(): 为每个 BOX 的原子初始化空间权重 (以该 BOX 自身中心为原点的高斯衰减)
    2. merge_box_atom_results(): 合并多 BOX 的原子预测结果 (logit 或 prob 级聚合)
    3. point_semantic_segment(): 对原子概率做阈值二值化

旧版体素→原子映射 (assign_prob_to_atoms) 已迁移至 src/inference/legacy/postprocess_voxel.py
"""

import numpy as np
import math
from typing import Optional


# =============================================================================
# 1. BOX 空间权重: 全BOX的高斯核 * 专门针对边界 core_offset 的额外衰减
# =============================================================================
# 高斯核衰减
def init_box_spatial_weights(
    box_dicts: list,
    voxel_size: np.ndarray,
    window_size: int,
    sigma_ratio: float,
) -> list:
    """
    为每个 BOX 内的原子计算以该 BOX 自身中心为原点的高斯衰减权重。

    每个 BOX 共用同一种高斯核: sigma = sigma_ratio × BOX 半径 (Å)。原子距自己所在 BOX 的中心越近, 权重越高 (≈1); 越靠近 BOX 边缘, 权重越低。

    输入参数:
        - box_dicts: list[dict], split_volume_to_boxes() 返回的 BOX 列表, 每个 dict 含:
            - "atom_coord_local_voxel": torch.Tensor, (N_box, 3), 原子的连续 voxel 坐标 (x, y, z)
            - "box_shape_zyx": torch.Tensor, (3,), BOX 体素形状
        - voxel_size: np.ndarray, (3,), float, 体素大小 (x, y, z), 单位 Å
        - window_size: int, 标量, BOX 窗口边长 (voxel)
        - sigma_ratio: float, 标量, 高斯核 sigma 与 BOX 半径之比, 建议值 0.5
            sigma = sigma_ratio × BOX 中心到角点的距离 (Å)

    输出:
        - box_spatial_weights: list[np.ndarray], 长度与 box_dicts 相同
            每项 np.ndarray, (N_box_atoms,), float32, 该 BOX 内每个原子的空间权重 ∈ (0, 1]
    """
    import torch
    voxel_size = np.asarray(voxel_size, dtype=np.float64).reshape(3)

    # float, BOX 半边长 (voxel) 转为世界坐标后的对角线半径 (Å)
    # 所有 BOX 共享同一个 window_size, 因此 sigma 也是统一的
    half_box_world = 0.5 * float(window_size) * voxel_size  # np.ndarray, (3,)
    box_radius = float(np.linalg.norm(half_box_world))      # float, BOX 中心到角点的距离 (Å)
    sigma = max(sigma_ratio * box_radius, 1e-6)             # float, 高斯核标准差 (Å)
    two_sigma2 = 2.0 * sigma * sigma                        # float, 分母预计算

    box_spatial_weights = []
    for box_dict in box_dicts:
        # 获取原子坐标
        atom_local = box_dict["atom_coord_local_voxel"]
        if isinstance(atom_local, torch.Tensor):
            atom_local = atom_local.numpy()
        # np.ndarray, (N_box, 3), float32, 原子的连续 voxel 坐标 (x, y, z)
        atom_local = np.asarray(atom_local, dtype=np.float64)

        box_shape = box_dict["box_shape_zyx"]
        if isinstance(box_shape, torch.Tensor):
            box_shape = box_shape.numpy()
        # np.ndarray, (3,), float64, BOX 体素形状 (Z, Y, X)
        box_shape = np.asarray(box_shape, dtype=np.float64)
        # np.ndarray, (3,), float64, BOX 中心的 voxel 坐标 (x, y, z)
        box_center_voxel_xyz = box_shape[[2, 1, 0]] * 0.5
        if atom_local.shape[0] == 0:
            box_spatial_weights.append(np.empty(0, dtype=np.float32))
            continue

        # np.ndarray, (N_box, 3), float64, 原子到 BOX 中心的偏移 (voxel, x/y/z)
        offset_voxel = atom_local - box_center_voxel_xyz[np.newaxis, :]
        # np.ndarray, (N_box, 3), float64, 偏移转为世界坐标 (Å)
        offset_world = offset_voxel * voxel_size[np.newaxis, :]
        # np.ndarray, (N_box,), float64, 距离平方
        dist2 = np.sum(offset_world ** 2, axis=1)
        # np.ndarray, (N_box,), float32, 高斯衰减权重
        weights = np.exp(-dist2 / two_sigma2).astype(np.float32)

        box_spatial_weights.append(weights)

    return box_spatial_weights

# core offset 额外衰减
def _compute_per_atom_core_weight(
    atom_coord_local_voxel: np.ndarray,
    atom_is_in_core: np.ndarray,
    box_shape_zyx: np.ndarray,
    core_offset: int,
    core_decay_mode: str,
) -> np.ndarray:
    """
    计算每个原子基于 core_offset 的额外边界衰减。

    输入参数:
        - atom_coord_local_voxel: np.ndarray, (N, 3), float32, 连续 voxel 坐标 (x, y, z)
        - atom_is_in_core: np.ndarray, (N,), bool
        - box_shape_zyx: np.ndarray, (3,), int64
        - core_offset: int, 标量, 裁边 voxel 数
        - core_decay_mode: str, "hard" / "linear" / "none"

    输出:
        - weight: np.ndarray, (N,), float32, 衰减权重 ∈ [0, 1]
    """
    n = atom_coord_local_voxel.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float32)

    if core_decay_mode == "none":
        return np.ones(n, dtype=np.float32)

    if core_offset <= 0:
        return np.ones(n, dtype=np.float32)

    # 计算每个原子到 BOX 六面边界的最短距离 (voxel 单位)
    # atom_coord_local_voxel 是 (x, y, z), box_shape_zyx 是 (Z, Y, X)
    box_shape_xyz = box_shape_zyx[[2, 1, 0]].astype(np.float32)
    # np.ndarray, (N, 3), float32, 到下边界的距离
    dist_lower = atom_coord_local_voxel
    # np.ndarray, (N, 3), float32, 到上边界的距离
    dist_upper = box_shape_xyz[np.newaxis, :] - atom_coord_local_voxel
    # np.ndarray, (N,), float32, 到最近边界的距离
    margin_dist = np.minimum(dist_lower, dist_upper).min(axis=1)

    if core_decay_mode == "hard":
        # 距边界 >= core_offset 的原子权重 1.0, 否则 0.0
        weight = (margin_dist >= float(core_offset)).astype(np.float32)
    elif core_decay_mode == "linear":
        # 距边界 >= core_offset 的原子权重 1.0
        # 距边界 < core_offset 的原子线性衰减到 0
        weight = np.clip(margin_dist / float(core_offset), 0.0, 1.0).astype(np.float32)
    else:
        raise ValueError(f"未知 core_decay_mode: {core_decay_mode}")

    return weight




# =============================================================================
# 2. 多 BOX 原子预测聚合
# =============================================================================
def merge_box_atom_results(
    box_results: list,
    total_atom_count: int,
    core_decay_mode: str,
    core_offset: int,
) -> np.ndarray:
    """
    合并多 BOX 的原子预测结果: 加权平均 logit → sigmoid → prob。

    输入参数:
        - box_results: list[dict], 每项含:
            - "global_atom_indices": np.ndarray, (N_box,), int, 全局原子索引
            - "atom_logits": np.ndarray, (N_box,) 或 (N_box, 1), float32, 原始 logits
            - "atom_is_in_core": np.ndarray, (N_box,), bool, 是否在 core box 内
            - "atom_coord_local_voxel": np.ndarray, (N_box, 3), float32, 连续 voxel 坐标
            - "box_shape_zyx": np.ndarray, (3,), int64, BOX 体素形状
            - "box_spatial_weight": np.ndarray, (N_box,), float32, 以 BOX 中心为原点的高斯衰减权重
            - "box_confidence_weight": float, 样本置信度权重 (预留, 默认 1.0)
        - total_atom_count: int, 标量, 全局原子总数
        - core_decay_mode: str, 核心区衰减方式
            - "hard": 核心区权重 1.0, 非核心区权重 0.0
            - "linear": 从核心区边界到 BOX 边界线性衰减
            - "none": 不做衰减
        - core_offset: int, 标量, 裁边 voxel 数

    输出:
        - atom_probs: np.ndarray, (total_atom_count,), float32, 每个原子的最终概率
    """
    if len(box_results) == 0:
        return np.zeros(total_atom_count, dtype=np.float32)

    # np.ndarray, (total_atom_count,), float64, 加权 logit 累加
    logit_sum = np.zeros(total_atom_count, dtype=np.float64)
    # np.ndarray, (total_atom_count,), float64, 权重累加
    weight_sum = np.zeros(total_atom_count, dtype=np.float64)

    for box_res in box_results:
        indices = box_res["global_atom_indices"]
        logits = box_res["atom_logits"]
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits[:, 0]
        elif logits.ndim != 1:
            raise ValueError(f"Logits shape {logits.shape} is not supported.")
        logits = logits.astype(np.float64)

        # np.ndarray, (N_box,), float32, 以 BOX 中心为原点的高斯空间权重
        spatial_w = box_res.get("box_spatial_weight")
        if spatial_w is None or (isinstance(spatial_w, (int, float)) and spatial_w == 1.0):
            spatial_w = np.ones(len(indices), dtype=np.float32)
        spatial_w = np.asarray(spatial_w, dtype=np.float64)
        # float, 置信度权重 (预留)
        conf_w = float(box_res.get("box_confidence_weight", 1.0))
        # np.ndarray, (N_box,), float32, 核心区衰减权重
        core_w = _compute_per_atom_core_weight(
            atom_coord_local_voxel=box_res["atom_coord_local_voxel"],
            atom_is_in_core=box_res["atom_is_in_core"],
            box_shape_zyx=box_res["box_shape_zyx"],
            core_offset=core_offset,
            core_decay_mode=core_decay_mode,
        )
        # np.ndarray, (N_box,), float64, 每个原子的最终权重 = 空间 × 核心衰减 × 置信度
        effective_w = spatial_w * core_w.astype(np.float64) * conf_w

        np.add.at(logit_sum, indices, logits * effective_w)
        np.add.at(weight_sum, indices, effective_w)

    # 加权均值 logit
    # np.ndarray, (total_atom_count,), bool, 有被覆盖的原子
    covered = weight_sum > 0.0
    # np.ndarray, (total_atom_count,), float64
    mean_logit = np.zeros(total_atom_count, dtype=np.float64)
    mean_logit[covered] = logit_sum[covered] / weight_sum[covered]

    # np.ndarray, (total_atom_count,), float32, sigmoid
    atom_probs = _sigmoid(mean_logit.astype(np.float32))
    # 未被任何 BOX 覆盖到的原子概率设为 0
    atom_probs[~covered] = 0.0
    return atom_probs


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """
    数值稳定的 sigmoid 函数。

    输入参数:
        - x: np.ndarray, 任意形状, logit 值

    输出:
        - np.ndarray, 与 x 同形, sigmoid 概率值
    """
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    ).astype(np.float32)




# =============================================================================
# 3. 原子概率 → 语义分割（二值化）
# =============================================================================
def point_semantic_segment(
    atom_probs: np.ndarray,
    atom_coords: np.ndarray,
    threshold: float,
) -> np.ndarray:   # see me: 这实际是对点云的后处理, 后续需要加入相关机制, 如 DBscan 筛选
    """
    对原子概率做阈值二值化，返回预测为正类的原子坐标。

    输入参数:
        - atom_probs:   np.ndarray, float32, (N_atom,), 每个原子的口袋概率 [0, 1]
        - atom_coords:  np.ndarray, float32, (N_atom, 3), 所有原子的世界坐标 (x, y, z), 单位 Å
        - threshold:    float, 概率阈值，>= threshold → 正类

    输出:
        - pred_atom_coords: np.ndarray, float32, (N_pred, 3), 预测为口袋正类的原子世界坐标
    """
    if atom_probs.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    # np.ndarray, bool, (N_atom,), 正类掩码
    mask = atom_probs >= threshold
    # np.ndarray, float32, (N_pred, 3), 正类原子坐标
    pred_atom_coords = atom_coords[mask].astype(np.float32)
    return pred_atom_coords

