"""
================================================================================
统一预处理系统 - 局部坐标系 / Unified Preprocessing System - Local Frames
================================================================================

计算每个残基的局部坐标系 (3x3 旋转矩阵):
- 蛋白质: 基于 N-CA-C 三原子
- 核苷酸: 基于 C4'-C1'-N1/N9 三原子

Compute local coordinate frames for each residue.
================================================================================
"""

import numpy as np
from typing import Tuple

from ..parser import ParsedStructure


def compute_local_frame(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray
) -> np.ndarray:
    """
    从三个点构建局部坐标系
    Build local coordinate frame from three points
    
    坐标系定义 / Frame definition:
        - 原点: p2 (中心原子)
        - X 轴: p2 -> p3 方向 (归一化)
        - Y 轴: 垂直于 X 轴，在 p1-p2-p3 平面内
        - Z 轴: X × Y (右手系)
    
    输入参数 / Input:
        - p1: np.ndarray, (3,), float32, 第一个原子坐标 (蛋白: N; 核酸: C4')
        - p2: np.ndarray, (3,), float32, 第二个原子坐标 (蛋白: CA; 核酸: C1')
        - p3: np.ndarray, (3,), float32, 第三个原子坐标 (蛋白: C; 核酸: N1/N9)
    
    输出 / Output:
        - frame: np.ndarray, (3, 3), float32, 局部坐标系旋转矩阵, 每列为一个基向量 [x_axis, y_axis, z_axis]
    """
    # np.ndarray, (3,), float32, 向量 v1 = p2 - p1
    v1 = p2 - p1
    # np.ndarray, (3,), float32, 向量 v2 = p3 - p2
    v2 = p3 - p2
    
    # 归一化 X 轴 (p2 -> p3)
    # float, v2 的模长
    norm_v2 = np.linalg.norm(v2)
    if norm_v2 < 1e-6:
        # 退化情况：返回单位矩阵
        return np.eye(3, dtype=np.float32)
    # np.ndarray, (3,), float32, X 轴单位向量
    x_axis = v2 / norm_v2
    
    # Z 轴 = v1 × v2 (垂直平面)
    # np.ndarray, (3,), float32
    z_axis = np.cross(v1, v2)
    norm_z = np.linalg.norm(z_axis)
    if norm_z < 1e-6:
        # 退化情况：三点共线
        return np.eye(3, dtype=np.float32)
    z_axis = z_axis / norm_z
    
    # Y 轴 = Z × X (确保正交)
    # np.ndarray, (3,), float32
    y_axis = np.cross(z_axis, x_axis)
    
    # 构建旋转矩阵 (列向量为基向量)
    # np.ndarray, (3, 3), float32
    frame = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32)
    
    return frame





def compute_local_frames(
    parsed_data: ParsedStructure
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算所有残基的局部坐标系
    Compute local coordinate frames for all residues
    
    输入参数 / Input:
        - parsed_data: ParsedStructure, 解析后的结构数据
    
    输出 / Output:
        - frames: np.ndarray, (N_res, 3, 3), float32, 局部坐标系矩阵
        - frames_mask: np.ndarray, (N_res,), bool, 有效性掩码 (True 表示骨架完整)
    """
    # int, 残基数量
    n_residues = len(parsed_data.res_names)
    if n_residues == 0:
        return np.zeros((0, 3, 3), dtype=np.float32), np.zeros(0, dtype=bool)
    # np.ndarray, (N_res, 3, 3), float32, 坐标系矩阵
    frames = np.zeros((n_residues, 3, 3), dtype=np.float32)
    # np.ndarray, (N_res,), bool, 有效性掩码
    frames_mask = np.zeros(n_residues, dtype=bool)
    
    for i in range(n_residues):
        # str, 残基类型
        res_type = parsed_data.res_types[i]
        if res_type == 'protein':
            # 蛋白质: N-CA-C
            p1 = parsed_data.backbone_n_coords[i]
            p2 = parsed_data.backbone_ca_coords[i]
            p3 = parsed_data.backbone_c_coords[i]
            # 检查是否有效 (非零坐标)
            if np.allclose(p1, 0) or np.allclose(p2, 0) or np.allclose(p3, 0):
                frames[i] = np.eye(3, dtype=np.float32)
                frames_mask[i] = False
            else:
                frames[i] = compute_local_frame(p1, p2, p3)
                frames_mask[i] = True
        
        else:  # 核苷酸
            # 核苷酸: C4'-C1'-N1/N9
            p1 = parsed_data.backbone_c4p_coords[i]
            p2 = parsed_data.backbone_c1p_coords[i]
            p3 = parsed_data.backbone_n_base_coords[i]
            if np.allclose(p1, 0) or np.allclose(p2, 0) or np.allclose(p3, 0):
                frames[i] = np.eye(3, dtype=np.float32)
                frames_mask[i] = False
            else:
                frames[i] = compute_local_frame(p1, p2, p3)
                frames_mask[i] = True
    
    return frames, frames_mask




def compute_relative_rotations_sparse(
    frames: np.ndarray,
    res_coords: np.ndarray,
    cutoff: float = 40.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算残基间相对旋转 (稀疏格式)
    Compute relative rotations between residues (sparse format)
    
    相对旋转定义 / Definition:
        R_ij = O_i^T @ O_j
        表示从残基 i 到残基 j 的局部旋转————[坐标系j的三个单位向量在坐标系i中的表示]
    
    输入参数 / Input:
        - frames: np.ndarray, (N_res, 3, 3), float32, 局部坐标系
        - res_coords: np.ndarray, (N_res, 3), float32, 残基坐标
        - cutoff: float, 距离截断 (只计算距离小于此值的残基对)
    
    输出 / Output:
        - rot_data: np.ndarray, (N_edges, 9), float32, 展平的相对旋转矩阵
        - row_idx: np.ndarray, (N_edges,), int32, 源节点索引
        - col_idx: np.ndarray, (N_edges,), int32, 目标节点索引
    """
    from scipy.spatial.distance import cdist
    # int, 残基数量
    n_residues = len(res_coords)
    if n_residues == 0:
        return np.zeros((0, 9), dtype=np.float32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    
    # np.ndarray, (N_res, N_res), float32, 距离矩阵
    dist_matrix = cdist(res_coords, res_coords, metric='euclidean')
    # 找出距离小于截断的残基对
    # np.ndarray, (N_edges,), int32
    row_idx, col_idx = np.where((dist_matrix < cutoff) & (dist_matrix > 0))
    # int, 边数量
    n_edges = len(row_idx)

    # np.ndarray, (N_edges, 9), float32
    rot_data = np.zeros((n_edges, 9), dtype=np.float32)
    for k in range(n_edges):
        i = row_idx[k]
        j = col_idx[k]
        # np.ndarray, (3, 3), float32, 相对旋转矩阵
        rel_rot = frames[i].T @ frames[j]
        # 展平为 9 维向量
        rot_data[k] = rel_rot.flatten()
    
    return rot_data, row_idx.astype(np.int32), col_idx.astype(np.int32)
