"""
postprocess.py - 后处理模块

从体素概率图 + 原子坐标出发，为每个原子赋予口袋概率，再二值化得到"属于口袋的点"。

流程:
    1. assign_prob_to_atoms(): 结合高斯核距离衰减 × 体素类别权重，为每个原子计算加权概率
    2. point_semantic_segment(): 对原子概率做阈值二值化，返回预测为正类的原子坐标

预留接口: 实例分割（连通域分析等）。
"""

import numpy as np
import math
from typing import Optional


# =============================================================================
# 1. 体素概率 → 原子概率
# =============================================================================
def assign_prob_to_atoms(
    pred_prob: np.ndarray,
    atom_coords: np.ndarray,
    origin: np.ndarray,
    voxel_size: np.ndarray,
    hardmask: np.ndarray,
    radius: float,
    sigma: float,
    cat_weight_home: float,
    cat_weight_has_atom: float,
    cat_weight_no_atom: float,
) -> np.ndarray:
    """
    基于 高斯核距离衰减 × 体素类别权重 为每个原子赋予口袋概率。

    对每个原子 a，在世界坐标半径 radius（Å）内搜索所有体素 v，按以下公式赋概率：
        P(a) = Σ_v [ w_gauss(d_av) · w_cat(v) · P(v) ] / Σ_v [ w_gauss(d_av) · w_cat(v) ]
    其中权重已归一化。当 sigma=0 时退化为 nearest 策略（仅取原子所在体素的概率值）。

    输入参数:
        - pred_prob:            np.ndarray, float32, (D, H, W), 每个体素的正类概率 [0, 1]
        - atom_coords:          np.ndarray, float32, (N_atom, 3), 所有原子的世界坐标 (x, y, z), 单位 Å
        - origin:               np.ndarray, float64, (3,), 密度图原点 (x, y, z), 单位 Å
        - voxel_size:           np.ndarray, float64, (3,), 各轴体素大小 (x, y, z), 单位 Å
        - hardmask:             np.ndarray, int64,   (D, H, W), 1=有原子 0=无原子
        - radius:               float, 搜索半径 (Å)。推荐 2.0
        - sigma:                float, 高斯核标准差 (Å)。推荐 1.0；0 表示 nearest
        - cat_weight_home:      float, 所在体素（原子落入的体素）的类别权重。推荐 0.5
        - cat_weight_has_atom:  float, 含原子体素（hardmask!=0，但非所在体素）的类别权重。推荐 0.35
        - cat_weight_no_atom:   float, 不含原子体素（hardmask==0）的类别权重。推荐 0.15

    输出:
        - atom_probs:           np.ndarray, float32, (N_atom,), 每个原子的口袋概率 [0, 1]
    """
    # int, 原子总数
    N = atom_coords.shape[0]
    if N == 0:
        return np.empty((0,), dtype=np.float32)
    D, H, W = pred_prob.shape

    # ---- σ=0 退化为 nearest：直接取原子所在体素的概率值 ----
    if sigma <= 0.0:
        # np.ndarray, int, (N_atom, 3), 原子对应的体素索引 (x, y, z)
        voxel_ijk = np.floor((atom_coords - origin) / voxel_size).astype(int)
        # np.ndarray, int, (N_atom,), clip 到合法范围
        xi = np.clip(voxel_ijk[:, 0], 0, W - 1)
        yi = np.clip(voxel_ijk[:, 1], 0, H - 1)
        zi = np.clip(voxel_ijk[:, 2], 0, D - 1)
        # np.ndarray, float32, (N_atom,), 直接读取体素概率
        atom_probs = pred_prob[zi, yi, xi].astype(np.float32)
        return atom_probs


    # ---- 一般情况：高斯核 × 类别权重 加权平均（向量化实现） ----
    # float, 高斯核分母预计算: 2 * sigma^2
    two_sigma2 = 2.0 * sigma * sigma
    # np.ndarray, int, (N_atom, 3), 每个原子对应的"所在体素"索引
    home_ijk = np.floor((atom_coords - origin) / voxel_size).astype(int)
    # np.ndarray, int, (N_atom,), clip 到合法范围的所在体素 x 索引
    hx = np.clip(home_ijk[:, 0], 0, W - 1)
    hy = np.clip(home_ijk[:, 1], 0, H - 1)
    hz = np.clip(home_ijk[:, 2], 0, D - 1)

    radius2 = radius * radius
    # np.ndarray, int, (3,), 各轴搜索的体素格数范围（向上取整）
    voxel_radius = np.ceil(radius / voxel_size).astype(int)
    # int, int, int, xyz三个维度的偏移半径
    rx, ry, rz = voxel_radius[0], voxel_radius[1], voxel_radius[2]

    # np.ndarray, int, (2*rx+1,), x 轴的偏移量范围
    dx_range = np.arange(-rx, rx + 1)
    # np.ndarray, int, (2*ry+1,), y 轴的偏移量范围
    dy_range = np.arange(-ry, ry + 1)
    # np.ndarray, int, (2*rz+1,), z 轴的偏移量范围
    dz_range = np.arange(-rz, rz + 1)
    
    # np.ndarray, np.ndarray, np.ndarray, shape 皆为 (2*rz+1, 2*ry+1, 2*rx+1), int, 局部偏移网格
    dz_grid, dy_grid, dx_grid = np.meshgrid(dz_range, dy_range, dx_range, indexing='ij')
    
    # np.ndarray, int, (K,), K=(2*rz+1)*(2*ry+1)*(2*rx+1), 展平后的 z 偏移量
    dz_flat = dz_grid.flatten()
    # np.ndarray, int, (K,), K=(2*rz+1)*(2*ry+1)*(2*rx+1), 展平后的 y 偏移量
    dy_flat = dy_grid.flatten()
    # np.ndarray, int, (K,), K=(2*rz+1)*(2*ry+1)*(2*rx+1), 展平后的 x 偏移量
    dx_flat = dx_grid.flatten()

    # np.ndarray, int, (N_atom, K), 未截断的全局 z 坐标
    Z_unclipped = hz[:, None] + dz_flat
    # np.ndarray, int, (N_atom, K), 未截断的全局 y 坐标
    Y_unclipped = hy[:, None] + dy_flat
    # np.ndarray, int, (N_atom, K), 未截断的全局 x 坐标
    X_unclipped = hx[:, None] + dx_flat

    # np.ndarray, bool, (N_atom, K), 标记偏移后坐标是否在有效网格范围内的掩码
    valid_mask_grid = (Z_unclipped >= 0) & (Z_unclipped < D) & \
                      (Y_unclipped >= 0) & (Y_unclipped < H) & \
                      (X_unclipped >= 0) & (X_unclipped < W)

    # np.ndarray, int, (N_atom, K), 截断在合法范围内的全局 z 坐标
    Z = np.clip(Z_unclipped, 0, D - 1)
    # np.ndarray, int, (N_atom, K), 截断在合法范围内的全局 y 坐标
    Y = np.clip(Y_unclipped, 0, H - 1)
    # np.ndarray, int, (N_atom, K), 截断在合法范围内的全局 x 坐标
    X = np.clip(X_unclipped, 0, W - 1)

    # np.ndarray, float64, (N_atom, K), 网格中心的世界 z 坐标
    cz = origin[2] + (Z + 0.5) * voxel_size[2]
    # np.ndarray, float64, (N_atom, K), 网格中心的世界 y 坐标
    cy = origin[1] + (Y + 0.5) * voxel_size[1]
    # np.ndarray, float64, (N_atom, K), 网格中心的世界 x 坐标
    cx = origin[0] + (X + 0.5) * voxel_size[0]

    # np.ndarray, float32, (N_atom, 1), 原子的世界 z 坐标
    az = atom_coords[:, 2:3]
    # np.ndarray, float32, (N_atom, 1), 原子的世界 y 坐标
    ay = atom_coords[:, 1:2]
    # np.ndarray, float32, (N_atom, 1), 原子的世界 x 坐标
    ax = atom_coords[:, 0:1]

    # np.ndarray, float64, (N_atom, K), 原子到对应局部网格中心距离的平方
    d2 = (ax - cx)**2 + (ay - cy)**2 + (az - cz)**2

    # np.ndarray, bool, (N_atom, K), 距离是否小于搜索半径的掩码
    dist_mask = d2 <= radius2

    # np.ndarray, bool, (N_atom, K), 有效的计算单元，即在网格范围内且满足距离条件
    active_mask = valid_mask_grid & dist_mask

    # np.ndarray, float32, (N_atom, K), 每个被检索体素的预测概率
    P = pred_prob[Z, Y, X]
    # np.ndarray, int64, (N_atom, K), 每个被检索体素是否含原子的硬掩码
    mask_xyz = hardmask[Z, Y, X]

    # np.ndarray, bool, (N_atom, K), 当前体素是否是原子的所在体素
    is_home = (Z == hz[:, None]) & (Y == hy[:, None]) & (X == hx[:, None])
    
    # np.ndarray, float64, (N_atom, K), 体素对应的类别权重，初始化为不含原子的权重
    w_cat = np.full_like(P, cat_weight_no_atom, dtype=np.float64)
    # 修改具有原子的体素权重
    w_cat[mask_xyz != 0] = cat_weight_has_atom
    # 修改所在体素的权重
    w_cat[is_home] = cat_weight_home

    # np.ndarray, float64, (N_atom, K), 经过高斯衰减计算的权重
    w_gauss = np.exp(-d2 / two_sigma2)
    # np.ndarray, float64, (N_atom, K), 综合了高斯、类别和有效性掩码后的最终权重
    w = w_gauss * w_cat * active_mask

    # np.ndarray, float64, (N_atom,), 每个原子的加权概率和
    weighted_sum = np.sum(w * P, axis=1)
    # np.ndarray, float64, (N_atom,), 每个原子的总权重累加和
    weight_sum = np.sum(w, axis=1)

    # np.ndarray, bool, (N_atom,), 权重和是否大于零的判断掩码
    valid_sum_mask = weight_sum > 0.0
    
    # np.ndarray, float32, (N_atom,), 每个原子的最终输出概率
    atom_probs = np.zeros(N, dtype=np.float32)
    # 对具有有效权重的原子正常计算平均概率
    atom_probs[valid_sum_mask] = (weighted_sum[valid_sum_mask] / weight_sum[valid_sum_mask]).astype(np.float32)
    # 处理极端情况: 如果没有体素在搜索范围内，回退取所在体素的概率
    atom_probs[~valid_sum_mask] = pred_prob[hz[~valid_sum_mask], hy[~valid_sum_mask], hx[~valid_sum_mask]].astype(np.float32)

    return atom_probs


# =============================================================================
# 2. 原子概率 → 语义分割（二值化）
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


# =============================================================================
# 3. 预留接口: 实例分割
# =============================================================================
def instance_segment(
    pred_prob: np.ndarray,
    **kwargs,
) -> dict:
    """
    [预留接口] 从概率图执行实例分割后处理。

    将来可实现: 连通域分析、Watershed、HDBSCAN 聚类等。

    Args:
        - pred_prob: np.ndarray, (D, H, W) 或 (C, D, H, W), 概率图

    Returns:
        - result: dict, 实例分割结果（格式待定义）

    Raises:
        NotImplementedError
    """
    raise NotImplementedError(
        "[postprocess] instance_segment() 尚未实现。"
    )
