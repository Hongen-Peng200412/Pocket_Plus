"""
================================================================================
统一预处理系统 - 图构建 / Unified Preprocessing System - Graph Builder
================================================================================

构建原子/残基级别的图结构:
- 距离截断边
- 稀疏邻接矩阵

Build graph structures at atom/residue level.
================================================================================
"""

import numpy as np
from typing import Tuple, Optional
from scipy.spatial import cKDTree
import scipy.sparse as sp

from ..config import GRAPH_CUTOFF, MAX_DISTANCE_CUTOFF
from ..parser import ParsedStructure


def build_graph_edges_sparse(
    coords: np.ndarray,
    cutoff: float = GRAPH_CUTOFF
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    构建距离截断图的边
    Build edges for distance-cutoff graph
    
    输入参数 / Input:
        - coords: np.ndarray, (N, 3), float32, 节点坐标
        - cutoff: float, 距离截断 (默认 10Å)
    
    输出 / Output:
        - row_idx: np.ndarray, (N_edges,), int32, 源节点索引
        - col_idx: np.ndarray, (N_edges,), int32, 目标节点索引
        - distances: np.ndarray, (N_edges,), float32, 边距离
    """
    # int, 节点数量
    n_nodes = coords.shape[0]
    if n_nodes == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)
    # cKDTree, 用于快速范围查询
    tree = cKDTree(coords)
    # 查询所有在 cutoff 范围内的邻居对
    # pairs 是 (N_pairs, 2)，每行是 (i, j) 且 i < j
    pairs = tree.query_pairs(r=cutoff, output_type='ndarray')
    if len(pairs) == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)
    # 转换为双向边, 构造 forward: i -> j 和 backward: j -> i
    row_idx = np.concatenate([pairs[:, 0], pairs[:, 1]]).astype(np.int32)
    col_idx = np.concatenate([pairs[:, 1], pairs[:, 0]]).astype(np.int32)
    # np.ndarray, (N_edges,), float32
    diff = coords[row_idx] - coords[col_idx]
    distances = np.linalg.norm(diff, axis=1).astype(np.float32)
    
    return row_idx, col_idx, distances




def build_sparse_distance_matrix(
    coords: np.ndarray,
    cutoff: float = MAX_DISTANCE_CUTOFF
) -> sp.coo_matrix:
    """
    构建稀疏距离矩阵
    Build sparse distance matrix
    
    输入参数 / Input:
        - coords: np.ndarray, (N, 3), float32, 节点坐标
        - cutoff: float, 距离截断
    
    输出 / Output:
        - dist_sparse: scipy.sparse.coo_matrix, (N, N), 稀疏距离矩阵
    """
    # int, 节点数量
    n_nodes = coords.shape[0]
    if n_nodes == 0:
        return sp.coo_matrix((n_nodes, n_nodes), dtype=np.float32)
    # 获取边
    row_idx, col_idx, distances = build_graph_edges_sparse(coords, cutoff)
    # 构建稀疏矩阵
    dist_sparse = sp.coo_matrix((distances, (row_idx, col_idx)), shape=(n_nodes, n_nodes))
    return dist_sparse



def build_adjacency_matrix(
    coords: np.ndarray,
    cutoff: float = GRAPH_CUTOFF,
    weight_type: str = 'inverse'
) -> sp.coo_matrix:
    """
    构建稀疏邻接矩阵 (默认返回距离矩阵的元素逆)
    Build sparse adjacency matrix
    
    输入参数 / Input:
        - coords: np.ndarray, (N, 3), float32, 节点坐标
        - cutoff: float, 距离截断
        - weight_type: str, 权重类型 ('binary', 'inverse', 'gaussian')
    
    输出 / Output:
        - adj_sparse: scipy.sparse.coo_matrix, (N, N), 稀疏邻接矩阵
    """
    # int, 节点数量
    n_nodes = coords.shape[0]
    if n_nodes == 0:
        return sp.coo_matrix((n_nodes, n_nodes), dtype=np.float32)
    # 获取边
    row_idx, col_idx, distances = build_graph_edges_sparse(coords, cutoff)
    # 计算权重
    if weight_type == 'binary':
        weights = np.ones(len(distances), dtype=np.float32)
    elif weight_type == 'inverse':
        weights = 1.0 / (distances + 1e-6)
    elif weight_type == 'gaussian':
        sigma = cutoff / 3.0
        weights = np.exp(-distances**2 / (2 * sigma**2))
    else:
        weights = distances
    # 构建稀疏矩阵
    adj_sparse = sp.coo_matrix((weights, (row_idx, col_idx)), shape=(n_nodes, n_nodes))
    return adj_sparse



def save_graph_npz(
    parsed_data: ParsedStructure,
    cutoff: float,
    output_path: str,
) -> None:
    """
    保存图结构数据到 .npz 文件 (内存优化版本)
    Save graph structure data to .npz file (memory-optimized)
    
    内存优化 / Memory Optimization:
        - 如果 atom_dist_matrix 为 None，使用 KD-Tree 直接计算边
        - 避免存储完整的 N×N 距离矩阵
    
    输入参数 / Input:
        - parsed_data: ParsedStructure, 解析后的结构数据
        - cutoff: float, 边构建距离截断
        - output_path: str, 输出文件路径
    
    输出文件内容 / Output file contents:
        - edge_row: np.ndarray, (N_edges,), int32, 边源节点
        - edge_col: np.ndarray, (N_edges,), int32, 边目标节点
        - edge_dist: np.ndarray, (N_edges,), float32, 边距离
        - edge_weight: np.ndarray, (N_edges,), float32, 边权重 (距离倒数)
        - num_atoms: int, 原子数量
        - num_residues: int, 残基数量
        - cutoff: float, 距离截断
    """
    # 使用 CKD-Tree 计算边 (大结构)
    row_idx, col_idx, distances = build_graph_edges_sparse(
        parsed_data.atom_coords, cutoff
    )
    
    # 计算权重
    weights = 1.0 / (distances + 1e-6)
    
    np.savez_compressed(
        output_path,
        edge_row=row_idx.astype(np.int32),
        edge_col=col_idx.astype(np.int32),
        edge_dist=distances.astype(np.float32),
        edge_weight=weights.astype(np.float32),   # 默认反比于距离
        num_atoms=len(parsed_data.atom_coords),
        num_residues=len(parsed_data.res_names),
        cutoff=cutoff,
    )

