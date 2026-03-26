"""
================================================================================
统一预处理系统 - 残基级特征 / Unified Preprocessing System - Residue Features
================================================================================

计算残基级特征向量 (33 维):
- 残基类型 One-Hot: 25 维 (20 AA + 4 Nucleotides + X)
- 理化性质: 8 维

Compute residue-level feature vectors (33 dim).
================================================================================
"""

import numpy as np
from typing import Tuple

from ..config import (
    NUM_RESIDUE_TYPES,
    RESIDUE_FEATURE_DIM,
    get_residue_onehot,
    get_residue_physio,
)
from ..parser import ParsedStructure


def compute_residue_features(
    parsed_data: ParsedStructure
) -> np.ndarray:
    """
    计算残基级特征
    Compute residue-level features
    
    输入参数 / Input:
        - parsed_data: ParsedStructure, 解析后的结构数据
    
    输出 / Output:
        - residue_features: np.ndarray, (N_res, 33), float32, 残基特征矩阵
    """
    # int, 残基数量
    n_residues = len(parsed_data.res_names)
    
    if n_residues == 0:
        return np.zeros((0, RESIDUE_FEATURE_DIM), dtype=np.float32)
    
    # np.ndarray, (N_res, 33), float32, 预分配特征矩阵
    residue_features = np.zeros((n_residues, RESIDUE_FEATURE_DIM), dtype=np.float32)
    
    # 特征维度偏移量
    offset_type = 0                               # 0-24:  类型 One-Hot (25)
    offset_physio = offset_type + NUM_RESIDUE_TYPES  # 25-32: 理化性质 (8)
    
    for i in range(n_residues):
        # str, 残基名称
        resname = parsed_data.res_names[i]
        
        # 类型 One-Hot
        residue_features[i, offset_type:offset_type + NUM_RESIDUE_TYPES] = get_residue_onehot(resname)
        
        # 理化性质
        residue_features[i, offset_physio:offset_physio + 8] = get_residue_physio(resname)
    
    return residue_features




def save_residues_npz(
    parsed_data: ParsedStructure,
    residue_features: np.ndarray,
    local_frames: np.ndarray,
    frames_mask: np.ndarray,
    output_path: str
) -> None:
    """
    保存残基级数据到 .npz 文件
    Save residue-level data to .npz file
    
    输入参数 / Input:
        - parsed_data: ParsedStructure, 解析后的结构数据
        - residue_features: np.ndarray, (N_res, 33), float32, 残基特征
        - local_frames: np.ndarray, (N_res, 3, 3), float32, 局部坐标系
        - frames_mask: np.ndarray, (N_res,), bool, 坐标系有效性掩码
        - output_path: str, 输出文件路径
    
    输出文件内容 / Output file contents:
        - coords: np.ndarray, (N_res, 3), float32, 残基代表坐标
        - features: np.ndarray, (N_res, 33), float32, 残基特征
        - names: np.ndarray, (N_res,), str, 残基名称
        - types: np.ndarray, (N_res,), str, 残基类型 ('protein' 或 'nucleotide')
        - chain_indices: np.ndarray, (N_res,), int32, 链索引
        - seq_numbers: np.ndarray, (N_res,), int32, 序列号
        - local_frames: np.ndarray, (N_res, 3, 3), float32, 局部坐标系
        - frames_mask: np.ndarray, (N_res,), bool, 坐标系有效性
        - backbone_complete: np.ndarray, (N_res,), bool, 骨架完整性
    """
    np.savez_compressed(
        output_path,
        coords=parsed_data.res_coords,
        features=residue_features,
        names=np.array(parsed_data.res_names, dtype=object),
        types=np.array(parsed_data.res_types, dtype=object),
        chain_indices=parsed_data.res_chain_indices,
        seq_numbers=parsed_data.res_seq_numbers,
        local_frames=local_frames,
        frames_mask=frames_mask,
        backbone_complete=parsed_data.backbone_complete_mask,
    )
