"""
================================================================================
统一预处理系统 - 原子级特征 / Unified Preprocessing System - Atom Features
================================================================================

计算原子级特征向量 (49 维):
- 元素 One-Hot: 6 维 (C, N, O, S, P, X)
- 残基类型 One-Hot: 25 维 (20 AA + 4 Nucleotides + X)
- 理化性质: 8 维
- 原子质量: 1 维
- 局部密度: 9 维 (距离直方图)

Compute atom-level feature vectors (49 dim).

内存优化 / Memory Optimization:
- 使用 KD-Tree 进行稀疏距离计算，避免 O(N²) 内存消耗
- 对于大型结构 (>10000 原子)，不返回完整距离矩阵
================================================================================
"""

import numpy as np
from typing import Optional, Tuple
from scipy.spatial import cKDTree

from ..config import (
    NUM_ELEMENTS,
    NUM_RESIDUE_TYPES,
    ELEMENT_TO_IDX,
    RESIDUE_TO_IDX,
    ATOM_MASS,
    RESIDUE_PHYSIO,
    DENSITY_BIN_EDGES,
    ATOM_FEATURE_DIM,
    GRAPH_CUTOFF,
    get_element_onehot,
    get_residue_onehot,
    get_residue_physio,
    get_atom_mass,
)
from ..parser import ParsedStructure


# int, 使用完整距离矩阵的原子数量阈值 (超过此值使用稀疏方法)
DENSE_MATRIX_THRESHOLD = 8000  # 8000² × 4 bytes ≈ 256 MB


def compute_local_density_sparse(
    atom_coords: np.ndarray,
    bin_edges: np.ndarray = DENSITY_BIN_EDGES
) -> np.ndarray:
    """
    计算每个原子的局部密度直方图
    Compute local density histogram for each atom
    
    逻辑 / Logic:
        1. 计算原子间距离矩阵
        2. 对每个原子，统计各距离区间内的邻居数量
        3. 归一化得到密度特征
    
    输入参数 / Input:
        - atom_coords: np.ndarray, (N_atoms, 3), float32, 原子坐标
        - bin_edges: np.ndarray, (N_bins+1,), float32, 距离分箱边界
    
    输出 / Output:
        - density: np.ndarray, (N_atoms, N_bins), float32, 密度直方图特征
    """
    # int, 原子数量
    n_atoms = atom_coords.shape[0]
    # int, 分箱数量
    n_bins = len(bin_edges) - 1
    if n_atoms == 0:
        return np.zeros((0, n_bins), dtype=np.float32)
    
    # cKDTree, 用于快速范围查询
    tree = cKDTree(atom_coords)
    # np.ndarray, (N_atoms, N_bins), float32, 密度直方图
    density = np.zeros((n_atoms, n_bins), dtype=np.float32)
    # list[int], 每个原子在各个半径内的邻居数量 (含自身)
    cumulative_counts = np.zeros(n_atoms, dtype=np.int32)
    # list[float], 有限的边界值 (排除可能的最后一个 inf)
    finite_edges = [e for e in bin_edges if e < np.inf]
    # 先统计最大半径内的邻居数量
    max_radius = finite_edges[-1] if finite_edges else 18.0
    # list[list[int]], 每个原子在 max_radius 内的邻居索引
    neighbors_max = tree.query_ball_point(atom_coords, r=max_radius, workers=1)
    # np.ndarray, (N_atoms,), int, 最大半径内的邻居数量
    cumulative_counts = np.array([len(n) for n in neighbors_max], dtype=np.int32)
    
    # 从大到小统计各区间
    prev_counts = cumulative_counts.copy()
    for i in range(n_bins - 1, -1, -1):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        
        if upper >= np.inf:
            density[:, i] = 0
        else:
            if lower > 0:
                # 统计 lower 半径内的邻居
                neighbors_lower = tree.query_ball_point(atom_coords, r=lower, workers=1)
                counts_lower = np.array([len(n) for n in neighbors_lower], dtype=np.int32)
                # 当前区间 = upper 处邻居数 - lower 处邻居数
                density[:, i] = (prev_counts - counts_lower).astype(np.float32)
                prev_counts = counts_lower
            else:
                # lower = 0 的情况，减去 1 (自身)
                density[:, i] = (prev_counts - 1).astype(np.float32)
    
    # 归一化 (使用对数变换平滑)
    density = np.log1p(np.maximum(density, 0))  # 防止负数
    
    return density


def compute_atom_features(
    parsed_data: ParsedStructure,
    compute_density: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    计算原子级特征 
    Compute atom-level features (memory-optimized)
    
    输入参数 / Input:
        - parsed_data: ParsedStructure, 解析后的结构数据
        - compute_density: bool, 是否计算密度特征
    
    输出 / Output:
        - atom_features: np.ndarray, (N_atoms, 49), float32, 原子特征矩阵
    """
    # int, 原子数量
    n_atoms = len(parsed_data.atom_coords)
    
    if n_atoms == 0:
        return np.zeros((0, ATOM_FEATURE_DIM), dtype=np.float32)
    
    # 构建特征矩阵 / Build feature matrix
    # np.ndarray, (N_atoms, 49), float32
    atom_features = np.zeros((n_atoms, ATOM_FEATURE_DIM), dtype=np.float32)
    
    # 特征维度偏移量
    offset_element = 0                                    # 0-5:   元素 One-Hot (6)
    offset_residue = offset_element + NUM_ELEMENTS        # 6-30:  残基类型 One-Hot (25)
    offset_physio = offset_residue + NUM_RESIDUE_TYPES    # 31-38: 理化性质 (8)
    offset_mass = offset_physio + 8                       # 39:    原子质量 (1)
    offset_density = offset_mass + 1                      # 40-48: 密度 (9)
    
    for i in range(n_atoms):
        # str, 元素符号
        element = parsed_data.atom_elements[i]
        # str, 残基名称
        resname = parsed_data.atom_res_names[i]
        
        # 元素 One-Hot
        atom_features[i, offset_element:offset_element + NUM_ELEMENTS] = get_element_onehot(element)
        
        # 残基类型 One-Hot
        atom_features[i, offset_residue:offset_residue + NUM_RESIDUE_TYPES] = get_residue_onehot(resname)
        
        # 理化性质
        atom_features[i, offset_physio:offset_physio + 8] = get_residue_physio(resname)
        
        # 原子质量
        atom_features[i, offset_mass] = get_atom_mass(element)
    
    # 计算密度特征 / Compute density features
    if compute_density:
        # 内部使用稀疏方法计算密度 (内存安全)
        density = compute_local_density_sparse(parsed_data.atom_coords)
        atom_features[:, offset_density:offset_density + 9] = density
    
    return atom_features




def save_atoms_npz(
    parsed_data: ParsedStructure,
    atom_features: np.ndarray,
    output_path: str
) -> None:
    """
    保存原子级数据到 .npz 文件
    Save atom-level data to .npz file
    
    输入参数 / Input:
        - parsed_data: ParsedStructure, 解析后的结构数据
        - atom_features: np.ndarray, (N_atoms, 49), float32, 原子特征
        - output_path: str, 输出文件路径
    
    输出文件内容 / Output file contents:
        - coords: np.ndarray, (N_atoms, 3), float32, 原子坐标
        - features: np.ndarray, (N_atoms, 49), float32, 原子特征
        - elements: np.ndarray, (N_atoms,), str, 元素符号
        - res_indices: np.ndarray, (N_atoms,), int32, 所属残基索引
        - chain_indices: np.ndarray, (N_atoms,), int32, 所属链索引
        - res_names: np.ndarray, (N_atoms,), str, 所属残基名称
        - atom_names: np.ndarray, (N_atoms,), str, 原子名称
    """
    np.savez_compressed(
        output_path,
        coords=parsed_data.atom_coords,
        features=atom_features,
        elements=np.array(parsed_data.atom_elements, dtype=object),
        res_indices=parsed_data.atom_res_indices,
        chain_indices=parsed_data.atom_chain_indices,
        res_names=np.array(parsed_data.atom_res_names, dtype=object),
        atom_names=np.array(parsed_data.atom_names, dtype=object),
    )
