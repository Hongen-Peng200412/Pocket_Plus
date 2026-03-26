"""
================================================================================
统一预处理系统 - 包初始化 / Unified Preprocessing System - Package Init
================================================================================
"""

from .config import (
    # 元素常量
    ALLOWED_ELEMENTS,
    NUM_ELEMENTS,
    ELEMENT_TO_IDX,
    ATOM_MASS,
    # 残基常量
    RESIDUE_TYPES,
    NUM_RESIDUE_TYPES,
    RESIDUE_TO_IDX,
    ALLOWED_RESIDUES,
    # 理化性质
    RESIDUE_PHYSIO,
    # 特征维度
    ATOM_FEATURE_DIM,
    RESIDUE_FEATURE_DIM,
    # 距离阈值
    BINDING_THRESHOLD,
    COVALENT_BOND_THRESHOLD,
    GRAPH_CUTOFF,
    # 辅助函数
    get_residue_onehot,
    get_element_onehot,
    get_residue_physio,
    get_atom_mass,
    is_protein_residue,
    is_nucleotide_residue,
)

__all__ = [
    'ALLOWED_ELEMENTS',
    'NUM_ELEMENTS', 
    'ELEMENT_TO_IDX',
    'ATOM_MASS',
    'RESIDUE_TYPES',
    'NUM_RESIDUE_TYPES',
    'RESIDUE_TO_IDX',
    'ALLOWED_RESIDUES',
    'RESIDUE_PHYSIO',
    'ATOM_FEATURE_DIM',
    'RESIDUE_FEATURE_DIM',
    'BINDING_THRESHOLD',
    'COVALENT_BOND_THRESHOLD',
    'GRAPH_CUTOFF',
    'get_residue_onehot',
    'get_element_onehot',
    'get_residue_physio',
    'get_atom_mass',
    'is_protein_residue',
    'is_nucleotide_residue',
]
