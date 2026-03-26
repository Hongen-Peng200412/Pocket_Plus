"""
================================================================================
统一预处理系统 - 配置文件 / Unified Preprocessing System - Configuration
================================================================================

定义支持 Protein + RNA/DNA 的常量和编码。
Constants and encodings for Protein + RNA/DNA support.

特征维度 / Feature Dimensions:
- 原子特征 (Atom): 49 维 = 元素(6) + 残基类型(25) + 理化性质(8) + 质量(1) + 密度(9)
- 残基特征 (Residue): 33 维 = 类型(25) + 理化性质(8)
================================================================================
"""

import numpy as np
from typing import Dict, List

# ============================================================================
# 元素常量 / Element Constants
# ============================================================================
# list[str], (6,), 允许的重原子元素 (含 P 磷和 X 未知)
# 含义: 系统支持的重原子元素符号列表
ALLOWED_ELEMENTS: List[str] = ['C', 'N', 'O', 'S', 'P', 'X']

# int, (Scalar), 元素类型数量
NUM_ELEMENTS: int = len(ALLOWED_ELEMENTS)  # 6

# dict[str, int], (6,), 元素 -> 索引映射
# 含义: 将元素符号映射到整数索引 (用于 One-Hot 编码)
ELEMENT_TO_IDX: Dict[str, int] = {e: i for i, e in enumerate(ALLOWED_ELEMENTS)}

# dict[str, float], (6,), 原子质量字典
# 含义: 归一化后的原子质量 (除以 32.0，使 S 约为 1.0)
ATOM_MASS: Dict[str, float] = {
    'C': 12.011 / 32.0,   # ~0.375
    'N': 14.007 / 32.0,   # ~0.438
    'O': 15.999 / 32.0,   # ~0.500
    'S': 32.065 / 32.0,   # ~1.002
    'P': 30.974 / 32.0,   # ~0.968 (磷)
    'X': 14.0 / 32.0,     # ~0.438 (未知元素使用氮的质量作为默认)
}





# ============================================================================
# 残基常量 / Residue Constants
# ============================================================================
# list[str], (20,), 标准氨基酸三字母代码
AMINO_ACIDS: List[str] = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 
    'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 
    'SER', 'THR', 'TRP', 'TYR', 'VAL',
]

# list[str], (4,), 标准核苷酸单字母/三字母代码
# 含义: RNA/DNA 核苷酸 (A=腺嘌呤, U=尿嘧啶, C=胞嘧啶, G=鸟嘌呤)
# 注意: DNA 中 T(胸腺嘧啶) 在结构文件中通常表示为 DT，此处统一映射
NUCLEOTIDES: List[str] = ['A', 'U', 'C', 'G']

# list[str], (4,), DNA 特定核苷酸
DNA_NUCLEOTIDES: List[str] = ['DA', 'DT', 'DC', 'DG']

# list[str], (25,), 完整残基类型列表 (20 AA + 4 Nucleotides + X)
# 含义: 用于 One-Hot 编码的有序残基列表
RESIDUE_TYPES: List[str] = AMINO_ACIDS + NUCLEOTIDES + ['X']

# int, (Scalar), 残基类型数量
NUM_RESIDUE_TYPES: int = len(RESIDUE_TYPES)  # 25

# dict[str, int], (25,), 残基 -> 索引映射
RESIDUE_TO_IDX: Dict[str, int] = {r: i for i, r in enumerate(RESIDUE_TYPES)}

# 添加特殊残基映射 / Add special residue mappings, DNA 核苷酸被映射到对应的 RNA 核苷酸，MSE 被映射到 MET(30-->25)
RESIDUE_TO_IDX['MSE'] = RESIDUE_TO_IDX['MET']  # 硒代甲硫氨酸 -> 甲硫氨酸
RESIDUE_TO_IDX['DT'] = RESIDUE_TO_IDX['U']      # 胸腺嘧啶 -> 尿嘧啶 (结构相似)
RESIDUE_TO_IDX['DA'] = RESIDUE_TO_IDX['A']      # DNA 腺嘌呤
RESIDUE_TO_IDX['DC'] = RESIDUE_TO_IDX['C']      # DNA 胞嘧啶
RESIDUE_TO_IDX['DG'] = RESIDUE_TO_IDX['G']      # DNA 鸟嘌呤

# set[str], (N=20+4+4+2,), 所有允许的残基名称 (用于解析时快速查找)
ALLOWED_RESIDUES: set = set(AMINO_ACIDS) | set(NUCLEOTIDES) | set(DNA_NUCLEOTIDES) | {'MSE', 'X'}






# ============================================================================
# 理化性质编码 / Physiochemical Property Encoding
# ============================================================================

# 8 维向量: [Polar(2), Acidity(3), Charge(3)]
# - Polar: [极性, 非极性]
# - Acidity: [酸性, 碱性, 中性]
# - Charge: [正电荷, 负电荷, 无电荷]

# dict[str, list[int]], (N, 8), 氨基酸理化性质字典
AA_PHYSIO: Dict[str, List[int]] = {
    'ALA': [0, 1, 0, 0, 1, 0, 0, 1],  # 非极性, 中性, 无电荷
    'ARG': [1, 0, 0, 1, 0, 1, 0, 0],  # 极性, 碱性, 正电荷
    'ASN': [1, 0, 0, 0, 1, 0, 0, 1],  # 极性, 中性, 无电荷
    'ASP': [1, 0, 1, 0, 0, 0, 1, 0],  # 极性, 酸性, 负电荷
    'CYS': [1, 0, 1, 0, 0, 0, 0, 1],  # 极性, 酸性(弱), 无电荷
    'GLN': [1, 0, 0, 0, 1, 0, 0, 1],  # 极性, 中性, 无电荷
    'GLU': [1, 0, 1, 0, 0, 0, 1, 0],  # 极性, 酸性, 负电荷
    'GLY': [0, 1, 0, 0, 1, 0, 0, 1],  # 非极性, 中性, 无电荷
    'HIS': [1, 0, 0, 1, 0, 0, 0, 1],  # 极性, 碱性(弱), 无电荷(pH7)
    'ILE': [0, 1, 0, 0, 1, 0, 0, 1],  # 非极性, 中性, 无电荷
    'LEU': [0, 1, 0, 0, 1, 0, 0, 1],  # 非极性, 中性, 无电荷
    'LYS': [1, 0, 0, 1, 0, 1, 0, 0],  # 极性, 碱性, 正电荷
    'MET': [0, 1, 0, 0, 1, 0, 0, 1],  # 非极性, 中性, 无电荷
    'PHE': [0, 1, 0, 0, 1, 0, 0, 1],  # 非极性, 中性, 无电荷
    'PRO': [0, 1, 0, 0, 1, 0, 0, 1],  # 非极性, 中性, 无电荷
    'SER': [1, 0, 0, 0, 1, 0, 0, 1],  # 极性, 中性, 无电荷
    'THR': [1, 0, 0, 0, 1, 0, 0, 1],  # 极性, 中性, 无电荷
    'TRP': [0, 1, 0, 0, 1, 0, 0, 1],  # 非极性, 中性, 无电荷
    'TYR': [1, 0, 1, 0, 0, 0, 0, 1],  # 极性, 酸性(弱), 无电荷
    'VAL': [0, 1, 0, 0, 1, 0, 0, 1],  # 非极性, 中性, 无电荷
    'MSE': [0, 1, 0, 0, 1, 0, 0, 1],  # 同 MET
}

# dict[str, list[int]], (N, 8), 核苷酸理化性质字典
# 核苷酸理化性质基于碱基特性:
# - 嘌呤 (A, G): 较大的双环结构
# - 嘧啶 (U, C): 较小的单环结构
# - 氢键供体/受体特性
NUCLEOTIDE_PHYSIO: Dict[str, List[int]] = {
    'A':  [1, 0, 0, 1, 0, 0, 0, 1],  # 腺嘌呤: 极性, 碱性, 无电荷
    'U':  [1, 0, 0, 0, 1, 0, 0, 1],  # 尿嘧啶: 极性, 中性, 无电荷
    'C':  [1, 0, 0, 0, 1, 0, 0, 1],  # 胞嘧啶: 极性, 中性, 无电荷
    'G':  [1, 0, 0, 1, 0, 0, 0, 1],  # 鸟嘌呤: 极性, 碱性, 无电荷
    'DA': [1, 0, 0, 1, 0, 0, 0, 1],  # DNA 腺嘌呤
    'DT': [1, 0, 0, 0, 1, 0, 0, 1],  # DNA 胸腺嘧啶
    'DC': [1, 0, 0, 0, 1, 0, 0, 1],  # DNA 胞嘧啶
    'DG': [1, 0, 0, 1, 0, 0, 0, 1],  # DNA 鸟嘌呤
}

# dict[str, list[int]], (N, 8), 合并的残基理化性质字典
RESIDUE_PHYSIO: Dict[str, List[int]] = {**AA_PHYSIO, **NUCLEOTIDE_PHYSIO}
RESIDUE_PHYSIO['X'] = [0, 0, 0, 0, 1, 0, 0, 1]  # 未知残基: 中性, 无电荷




# ============================================================================
# 特征维度常量 / Feature Dimension Constants
# ============================================================================
# int, (Scalar), 元素 One-Hot 维度
ELEMENT_ONEHOT_DIM: int = 6
# int, (Scalar), 残基类型 One-Hot 维度
RESIDUE_ONEHOT_DIM: int = 25
# int, (Scalar), 理化性质维度
PHYSIO_DIM: int = 8
# int, (Scalar), 原子质量维度
MASS_DIM: int = 1
# int, (Scalar), 局部密度直方图维度 (距离分箱)
DENSITY_DIM: int = 9

# int, (Scalar), 原子特征总维度
ATOM_FEATURE_DIM: int = ELEMENT_ONEHOT_DIM + RESIDUE_ONEHOT_DIM + PHYSIO_DIM + MASS_DIM + DENSITY_DIM  # 49
# int, (Scalar), 残基特征总维度
RESIDUE_FEATURE_DIM: int = RESIDUE_ONEHOT_DIM + PHYSIO_DIM  # 33




# ============================================================================
# 骨架原子定义 / Backbone Atom Definitions
# ============================================================================
# 蛋白质骨架原子 (用于局部坐标系)
PROTEIN_BACKBONE_ATOMS: List[str] = ['N', 'CA', 'C']

# 核苷酸骨架原子 (用于局部坐标系)
# C4' - 糖环碳
# C1' - 糖环碳 (连接碱基)
# N1 (嘧啶) 或 N9 (嘌呤) - 碱基氮原子
NUCLEOTIDE_BACKBONE_ATOMS: List[str] = ["C4'", "C1'", "N1", "N9"]
# 嘌呤碱基 (使用 N9)
PURINES: set = {'A', 'G', 'DA', 'DG'}
# 嘧啶碱基 (使用 N1)
PYRIMIDINES: set = {'U', 'C', 'DT', 'DC'}





# ============================================================================
# 距离阈值 / Distance Thresholds
# ============================================================================
# float, (Scalar), 结合位点距离阈值 (埃)
# 含义: 原子到配体距离 <= 此值时认为属于结合口袋
BINDING_THRESHOLD: float = 4.5

# float, (Scalar), 共价键距离阈值 (埃)
# 含义: 用于判断相邻残基是否共价连接 (C-N 或 O3'-P)
COVALENT_BOND_THRESHOLD: float = 1.8

# float, (Scalar), 图边构建距离截断 (埃)
# 含义: 距离小于此值的原子对之间建立边
GRAPH_CUTOFF: float = 10.0

# float, (Scalar), 最大距离截断 (用于稀疏矩阵)
MAX_DISTANCE_CUTOFF: float = 40.0





# ============================================================================
# 密度直方图分箱边界 / Density Histogram Bin Edges
# ============================================================================
# np.ndarray, (10,), float32, 距离直方图边界
# 含义: 用于计算原子周围的局部密度特征
DENSITY_BIN_EDGES: np.ndarray = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0], dtype=np.float32)






# ============================================================================
# 配体检测排除列表 / Ligand Detection Exclusion List
# ============================================================================
# set[str], (N,), HETATM 排除列表
# 含义: 这些残基名称不被视为配体 (溶剂、缓冲液、结晶添加剂等)
HETATM_EXCLUSION_LIST: set = {
    # 溶剂 / Solvents
    'HOH', 'WAT', 'H2O', 'DOD',
    # 缓冲液和添加剂 / Buffers and Additives
    'GOL', 'EDO', 'MPD', 'PEG', 'ACT', 'TRS', 'BU1', 'TBU', 'DMSO', 'MES',
    # 未知/未解析 / Unknown
    'UNX', 'UNL', 'UNK',
    # 修饰残基 / Modified Residues (常与蛋白主链连接)
    'SEP', 'TPO', 'MSE',
    # 惰性气体 / Noble Gases
    'XE', 'KR',
    # 常见金属离子 (单独处理或排除, 根据需求)
    # 'MG', 'CA', 'ZN', 'FE', 'NA', 'K', 'CL',
}
# set[str], (N,), 经典配体豁免列表
# 含义: 这些残基即使连接到主链，也应被视为配体
CLASSIC_LIGAND_EXEMPTION_LIST: set = {
    'HEM', 'FAD', 'NAD', 'NADP', 'NAP', 'NDP',
    'ATP', 'ADP', 'AMP', 'GTP', 'GDP', 'GMP',
    'FES', 'SF4', 'PLP', 'MGD', 'COA', 'SAM',
}






# ============================================================================
# 辅助函数 / Helper Functions
# ============================================================================

def get_residue_onehot(resname: str) -> np.ndarray:
    """
    获取残基的 One-Hot 编码
    Get One-Hot encoding for a residue
    
    输入参数 / Input:
        - resname: str, 残基名称 (3字母代码)
    
    输出 / Output:
        - np.ndarray, (25,), float32, One-Hot 编码向量
    """
    # np.ndarray, (25,), float32, 全零向量
    onehot = np.zeros(NUM_RESIDUE_TYPES, dtype=np.float32)
    # int, 残基索引 (如果未找到则使用 X 的索引)
    idx = RESIDUE_TO_IDX.get(resname.upper(), RESIDUE_TO_IDX['X'])
    onehot[idx] = 1.0
    return onehot


def get_element_onehot(element: str) -> np.ndarray:
    """
    获取元素的 One-Hot 编码
    Get One-Hot encoding for an element
    
    输入参数 / Input:
        - element: str, 元素符号 (如 'C', 'N', 'O', 'S', 'P')
    
    输出 / Output:
        - np.ndarray, (6,), float32, One-Hot 编码向量
    """
    # np.ndarray, (6,), float32, 全零向量
    onehot = np.zeros(NUM_ELEMENTS, dtype=np.float32)
    # int, 元素索引 (如果未找到则使用 X 的索引)
    idx = ELEMENT_TO_IDX.get(element.upper(), ELEMENT_TO_IDX['X'])
    onehot[idx] = 1.0
    return onehot


def get_residue_physio(resname: str) -> np.ndarray:
    """
    获取残基的理化性质编码
    Get physiochemical property encoding for a residue
    
    输入参数 / Input:
        - resname: str, 残基名称
    
    输出 / Output:
        - np.ndarray, (8,), float32, 理化性质向量
    """
    # list[int], (8,), 理化性质 (如果未找到则使用 X 的编码)
    physio = RESIDUE_PHYSIO.get(resname.upper(), RESIDUE_PHYSIO['X'])
    return np.array(physio, dtype=np.float32)


def get_atom_mass(element: str) -> float:
    """
    获取原子的归一化质量
    Get normalized atomic mass
    
    输入参数 / Input:
        - element: str, 元素符号
    
    输出 / Output:
        - float, 归一化质量
    """
    return ATOM_MASS.get(element.upper(), ATOM_MASS['X'])


def is_protein_residue(resname: str) -> bool:
    """
    判断残基是否为蛋白质氨基酸
    Check if residue is a protein amino acid
    """
    return resname.upper() in set(AMINO_ACIDS) | {'MSE'}


def is_nucleotide_residue(resname: str) -> bool:
    """
    判断残基是否为核苷酸
    Check if residue is a nucleotide
    """
    return resname.upper() in set(NUCLEOTIDES) | set(DNA_NUCLEOTIDES)


def is_purine(resname: str) -> bool:
    """判断核苷酸是否为嘌呤 (使用 N9)"""
    return resname.upper() in PURINES


def is_pyrimidine(resname: str) -> bool:
    """判断核苷酸是否为嘧啶 (使用 N1)"""
    return resname.upper() in PYRIMIDINES
