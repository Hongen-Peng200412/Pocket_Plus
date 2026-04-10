"""
================================================================================
统一预处理系统 - 候选配体解析与属性计算
Unified Preprocessing System - Candidate Ligand Parsing & Attribute Computation
================================================================================

Part 1 核心模块：
  1. 从 PDB/mmCIF 结构中提取 **几乎所有** HETATM 残基作为候选配体
     （排除水分子 + HETATM_EXCLUSION_LIST + 与主链共价连接的 Modified_Residues）
  2. 为每个候选计算一组可扩展的属性，支持下游多种筛选规则

Part 1 core module:
  1. Extract nearly all HETATM residues as candidate ligands
     (excluding water + HETATM_EXCLUSION_LIST + covalently linked Modified_Residues)
  2. Compute an extensible set of attributes per candidate for downstream filtering

当前已实现的属性 / Currently implemented attributes:
  - 基本标识 (candidate_id, resname, chain_id, res_id, ...)
  - 坐标与几何 (coords, center, n_heavy_atoms)
  - 大小: molecular_weight(n_heavy_atoms)
  - 分类标志: is_metal_ion, is_peptide_like, is_nucleotide_like
  - 接触属性: is_covalent, n_contact_receptor_atoms, n_contact_receptor_residues
  - 聚合物链长: polymer_length

后续将可能添加的属性 / Planned future attributes:
  - element_counts, has_only_organic_elements
  - 排除列表命中标志 (AF3, BioLiP, ...)
  - 冗余度 (resname_count_in_structure)
================================================================================
"""



import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from scipy.spatial import cKDTree
from Bio.PDB.Atom import DisorderedAtom

from .config import (
    WATER_RESIDUES,
    HETATM_EXCLUSION_LIST,
    Modified_Residues,
    METAL_ELEMENTS,
    COVALENT_BOND_THRESHOLD,
    is_protein_residue,
    is_nucleotide_residue,
)



def is_connected_to(residue, chain, threshold: float = COVALENT_BOND_THRESHOLD) -> bool:
    """
    判断一个 HETATM 残基是否与主链共价连接
    Check if a HETATM residue is covalently connected to the main chain
    
    逻辑 / Logic:
        1. 检查序列号是否连续
        2. 检查物理距离 (C-N 或 O3'-P) 是否小于阈值
    
    输入参数 / Input:
        - residue: Bio.PDB.Residue, 当前残基
        - chain: Bio.PDB.Chain, 链对象
        - threshold: float, 共价键距离阈值 (默认 1.8Å)
    
    输出 / Output:
        - bool, 是否连接到主链
    """
    
    def _are_ids_adjacent(id1, id2) -> bool:
        """检查两个残基 ID 是否连续"""
        seq1, ins1 = id1[1], id1[2]
        seq2, ins2 = id2[1], id2[2]
        
        diff = seq2 - seq1
        if diff == 1:
            return True
        elif diff == 0:
            if ins1 == ' ' and ins2 == 'A':
                return True
            if ins1 != ' ' and ins2 != ' ' and (ord(ins2) - ord(ins1) == 1):
                return True
        return False
    
    def _is_covalently_bonded(res1, res2, thresh: float) -> bool:
        """检查两个残基之间是否存在共价键"""
        # 肽键: C-N
        if 'C' in res1 and 'N' in res2:
            try:
                diff = res1['C'] - res2['N']
                if diff < thresh:
                    return True
            except Exception:
                pass
        
        # 核酸键: O3'-P
        o3_names = ["O3'", "O3*", "O3"]
        for name in o3_names:
            if name in res1 and 'P' in res2:
                try:
                    diff = res1[name] - res2['P']
                    if diff < thresh:
                        return True
                except Exception:
                    pass
        
        return False
    
    # list, 链中所有残基
    res_list = list(chain.get_residues())
    
    try:
        idx = res_list.index(residue)
    except ValueError:
        return False
    
    # 向前追溯
    curr_idx = idx
    while curr_idx > 0:
        prev_idx = curr_idx - 1
        curr_res = res_list[curr_idx]
        prev_res = res_list[prev_idx]
        
        if not _are_ids_adjacent(prev_res.id, curr_res.id):
            break
        
        if not _is_covalently_bonded(prev_res, curr_res, threshold):
            break
        
        if prev_res.id[0].startswith('H_'):
            curr_idx -= 1
        elif prev_res.id[0] == ' ':
            return True
        else:
            break
    
    # 向后追溯
    curr_idx = idx
    while curr_idx < len(res_list) - 1:
        next_idx = curr_idx + 1
        curr_res = res_list[curr_idx]
        next_res = res_list[next_idx]
        
        if not _are_ids_adjacent(curr_res.id, next_res.id):
            break
        
        if not _is_covalently_bonded(curr_res, next_res, threshold):
            break
        
        if next_res.id[0].startswith('H_'):
            curr_idx += 1
        elif next_res.id[0] == ' ':
            return True
        else:
            break
    
    return False



# ============================================================================
# 数据结构 / Data Structures
# ============================================================================

@dataclass
class LigandCandidate:
    """
    单个候选配体的完整属性记录, 每个 HETATM 残基实例（非水分子）对应一个 LigandCandidate。

    ============================== 当前字段 ==============================

    0. 基本标识 / Basic Identity:
        - candidate_id: int, 全局唯一候选编号 (从0开始, 0-indexed)
        - resname:       str, CCD 残基名 (如 'ATP', 'ZN', 'GOL')
        - chain_id:      str, 链标识符
        - res_id:        int, 残基序号
        - insertion_code: str, 插入码

    0. 坐标与几何 / Coordinates & Geometry:
        - coords:        np.ndarray, (M, 3), float32, 全部重原子坐标
        - center:        np.ndarray, (3,), float32, 重心坐标

    --------------------------------------
    1. 大小 / Size:
        - n_heavy_atoms: int, 重原子数目
        - molecular_weight: float, 配体所有重原子的质量之和 (Da, Biopython atom.mass)

    2. 类别 / Classification Flags (bool → 存储时编码为 0/1):
        - is_metal_ion:       bool, 金属离子 (单原子且元素 ∈ METAL_ELEMENTS)
        - is_peptide_like:    bool, 标准氨基酸类型的 HETATM
        - is_nucleotide_like: bool, 标准核苷酸类型的 HETATM

    3. 接触属性 / Contact Properties:
        - is_covalent:        bool, 与主链共价连接
        - n_contact_receptor_atoms: int, 在 binding_threshold 内接触到的受体重原子数 (Part2, 也就是从候选配体中挑选合格配体时, 由 def compute_contact_attributes 现场计算, 不存入 candidates.npz————因为binding_threshold由用户现场指定)
        - n_contact_receptor_residues: int, 接触受体残基数: 某残基与配体至少 2 次接触才计入 (Part2, 也就是从候选配体中挑选合格配体时, 由 def compute_contact_attributes 现场计算, 不存入 candidates.npz)

    4. 聚合物链长 / Polymer Chain Length:
        - polymer_length: int, 所属连续 HETATM 聚合链的长度 (聚合物=多肽/核酸; 非聚合物=1; 太长会过滤)


    ============================== 保留接口 ==============================
    以下属性在当前版本不计算 (值为 None)，后续版本可能会填充:
        - has_only_organic_elements: Optional[bool], 是否仅含有机元素
        - in_af3_ligand_exclusion:  Optional[bool], AF3 排除列表命中
        - in_af3_ion_list:          Optional[bool], AF3 离子列表命中
        - min_dist_to_receptor:     Optional[float], 最小到受体距离
        - resname_count_in_structure: Optional[int], 同 resname 在结构中出现次数
    """

    # ========================= 基本标识 =========================
    candidate_id: int
    resname: str
    chain_id: str
    res_id: int
    insertion_code: str

    # ========================= 坐标与几何 =========================
    # np.ndarray, (M, 3), float32, 全部重原子坐标
    coords: np.ndarray
    # np.ndarray, (3,), float32, 重心坐标
    center: np.ndarray
    # int, 重原子数目
    n_heavy_atoms: int

    # ========================= 1. 大小 =========================
    # float, 配体所有重原子的质量之和 (Da, 来自 Biopython atom.mass)
    molecular_weight: float = 0.0

    # ========================= 2. 分类标志 =========================
    # bool, 金属离子
    is_metal_ion: bool = False
    # bool, 标准氨基酸类型的 HETATM
    is_peptide_like: bool = False
    # bool, 标准核苷酸类型的 HETATM
    is_nucleotide_like: bool = False

    # ========================= 3. 接触属性 =========================
    # bool, 与主链共价连接
    is_covalent: bool = False
    # int, 在 binding_threshold 内接触到的受体重原子数 (Part 2 就地填充, 不存入 npz)
    n_contact_receptor_atoms: int = 0
    # int, 接触受体残基数: 某残基与配体至少 2 次接触才计入 (Part 2 就地填充, 不存入 npz)
    n_contact_receptor_residues: int = 0

    # ========================= 4. 聚合物链长 =========================
    # int, 所属连续 HETATM 聚合链长度 (非聚合物=1)
    polymer_length: int = 1

    # ========================= 保留接口 (值为 None 表示未计算) =========================
    has_only_organic_elements: Optional[bool] = None
    in_af3_ligand_exclusion: Optional[bool] = None
    in_af3_ion_list: Optional[bool] = None
    min_dist_to_receptor: Optional[float] = None
    resname_count_in_structure: Optional[int] = None



# ---------------------------------------------- 辅助函数 / Helper Functions ----------------------------------------------

def _get_heavy_atom_coords(residue) -> np.ndarray:
    """
    提取一个残基中所有重原子的坐标（跳过氢原子）。

    输入参数 / Input:
        - residue: Bio.PDB.Residue, Biopython 残基对象

    输出 / Output:
        - np.ndarray, (M, 3), float32, 重原子坐标, 若无重原子则返回空数组 shape=(0, 3)
    """
    # list[np.ndarray], 收集每个重原子的坐标
    coords_list = []
    for atom in residue:
        # str, 原子名称
        atom_name = atom.get_name().strip().upper()
        # 跳过氢原子 (以 H 开头或数字+H)
        if atom_name and atom_name[0] == 'H':
            continue
        if len(atom_name) > 1 and atom_name[0].isdigit() and atom_name[1] == 'H':
            continue
        # np.ndarray, (3,), float64, 当前原子坐标
        if isinstance(atom, DisorderedAtom):
            coord = atom.disordered_get_list()[0].get_coord()
        else:
            coord = atom.get_coord()
        coords_list.append(coord)

    if len(coords_list) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.array(coords_list, dtype=np.float32)

def _get_atom_element(atom) -> Optional[str]:
    """
    获取原子的元素符号（大写）。

    输入参数 / Input:
        - atom: Bio.PDB.Atom, Biopython 原子对象

    输出 / Output:
        - str 或 None, 元素符号 (如 'FE', 'ZN', 'C'), 若无法确定则返回 None
    """
    if hasattr(atom, 'element') and atom.element:
        return atom.element.strip().upper()
    return None


def _are_covalently_bonded_pair(res1, res2, threshold: float = COVALENT_BOND_THRESHOLD) -> bool:
    """
    判断两个相邻残基之间是否存在共价键。
    Check if two adjacent residues are covalently bonded.

    检查肽键 (C-N ≤ threshold) 和核酸键 (O3'-P ≤ threshold)。

    输入参数 / Input:
        - res1: Bio.PDB.Residue, 前一个残基
        - res2: Bio.PDB.Residue, 后一个残基
        - threshold: float, 共价键距离阈值 (默认 1.8Å)

    输出 / Output:
        - bool, 是否共价连接
    """
    # 检查序号是否连续
    seq1 = res1.id[1]
    seq2 = res2.id[1]
    diff = seq2 - seq1
    if diff != 0 and diff != 1:
        return False

    # 肽键: C-N
    if 'C' in res1 and 'N' in res2:
        try:
            dist = res1['C'] - res2['N']
            if dist < threshold:
                return True
        except Exception:
            pass

    # 核酸键: O3'-P
    for o3_name in ["O3'", "O3*", "O3"]:
        if o3_name in res1 and 'P' in res2:
            try:
                dist = res1[o3_name] - res2['P']
                if dist < threshold:
                    return True
            except Exception:
                pass

    return False

# ---------------------------------------------- 辅助函数 / Helper Functions ----------------------------------------------













# ========================================================================================================================================
# =============================================================== 属性计算 ================================================================

# 1. 大小
def _compute_molecular_weight(residue) -> float:
    """
    计算一个残基中所有重原子的质量之和。

    输入参数:
        - residue: Bio.PDB.Residue, Biopython 残基对象

    输出:
        - float, 重原子总质量 (Da), 若无重原子则返回 0.0
    """
    # float, 累计质量
    total_mass = 0.0
    for atom in residue:
        # str, 原子名称
        atom_name = atom.get_name().strip().upper()
        # 跳过氢原子
        if atom_name and atom_name[0] == 'H':
            continue
        if len(atom_name) > 1 and atom_name[0].isdigit() and atom_name[1] == 'H':
            continue
        # 对 DisorderedAtom 取第一个选择位
        if isinstance(atom, DisorderedAtom):
            total_mass += atom.disordered_get_list()[0].mass
        else:
            total_mass += atom.mass
    return total_mass


# 2. 类别(核酸和蛋白配体易识别, 当场处理而不加函数)
def _is_single_atom_metal(residue) -> bool:
    """
    判断一个 HETATM 残基是否为单原子金属离子。

    逻辑 / Logic:
        1. 残基中仅含 1 个非氢原子
        2. 该原子的元素符号 ∈ METAL_ELEMENTS

    输入参数 / Input:
        - residue: Bio.PDB.Residue, Biopython 残基对象

    输出 / Output:
        - bool, 是否为单原子金属离子
    """
    # list[Bio.PDB.Atom], 非氢原子列表
    heavy_atoms = []
    for atom in residue:
        atom_name = atom.get_name().strip().upper()
        if atom_name and atom_name[0] == 'H':
            continue
        if len(atom_name) > 1 and atom_name[0].isdigit() and atom_name[1] == 'H':
            continue
        heavy_atoms.append(atom)

    if len(heavy_atoms) != 1:
        return False

    # str 或 None, 唯一重原子的元素符号
    element = _get_atom_element(heavy_atoms[0])
    if element is None:
        return False
    return element in METAL_ELEMENTS




# 3. 接触属性: 只有这个函数不参与存入 candidates.npz, 因为 binding_threshold 由用户现场指定
def compute_contact_attributes(
    candidates: List['LigandCandidate'],
    receptor_coords: np.ndarray,
    receptor_res_indices: np.ndarray,
    threshold: float,
) -> None:
    """
    就地填充每个候选配体的 n_contact_receptor_atoms 和 n_contact_receptor_residues。

    逻辑:
        - 用 cKDTree 对受体原子坐标建树
        - 对每个配体的每个重原子, query_ball_point(threshold) 获取邻近受体原子
        - n_contact_receptor_atoms = 被命中的不重复受体原子数
        - 对每个受体残基, 统计命中原子数; 命中 ≥ 2 的残基计为 binding residue
        - n_contact_receptor_residues = binding residue 数

    输入参数:
        - candidates: list[LigandCandidate], 候选配体列表 (就地修改)
        - receptor_coords: np.ndarray, (N_rec, 3), float32, 受体重原子坐标
        - receptor_res_indices: np.ndarray, (N_rec,), int32, 受体原子→残基索引
        - threshold: float, 接触距离阈值 (Å), 目前将会是 Pocket_Plus\Make_Data\labels\filter_config.py 里面各类配体自带的 binding_threshold 取最大值:, 即= max(r.binding_threshold for r in filter_config.rules)

    输出:
        - None (就地修改 candidates)
    """
    if len(candidates) == 0 or len(receptor_coords) == 0:
        return

    # cKDTree, 受体原子的空间索引
    receptor_tree = cKDTree(receptor_coords)

    for candidate in candidates:
        # np.ndarray, (M, 3), float32, 配体重原子坐标
        lig_coords = candidate.coords
        if lig_coords.shape[0] == 0:
            candidate.n_contact_receptor_atoms = 0
            candidate.n_contact_receptor_residues = 0
            continue

        # list[list[int]], 每个配体原子在 threshold 内的受体原子索引列表
        neighbor_lists = receptor_tree.query_ball_point(lig_coords, r=threshold)

        # set[int], 被命中的不重复受体原子索引
        contacted_atom_indices = set()
        for neighbors in neighbor_lists:
            contacted_atom_indices.update(neighbors)

        # int, 接触受体重原子数
        candidate.n_contact_receptor_atoms = len(contacted_atom_indices)

        # dict[int, int], 受体残基索引 → 被命中的原子计数
        residue_hit_counts: Dict[int, int] = {}
        for atom_idx in contacted_atom_indices:
            # int, 该受体原子所属的残基索引
            res_idx = int(receptor_res_indices[atom_idx])
            residue_hit_counts[res_idx] = residue_hit_counts.get(res_idx, 0) + 1

        # int, binding residue 数 (命中原子数 ≥ 2 的残基)
        candidate.n_contact_receptor_residues = sum(
            1 for count in residue_hit_counts.values() if count >= 2
        )


# 4. 聚合物链长
def _compute_polymer_segments(model) -> Dict[Tuple, int]:
    """
    计算结构中所有 HETATM 聚合物链段的长度。

    逻辑 / Logic:
        对每条链中的残基列表，识别连续的 HETATM 残基链段：
        - 两个相邻 HETATM 残基如果序号连续且共价连接（C-N 或 O3'-P 距离 < 阈值）, 则属于同一链段
        - 每个链段内的残基数即为 polymer_length

    输入参数 / Input:
        - model: Bio.PDB.Model, 结构模型

    输出 / Output:
        - dict[tuple, int], 残基唯一键 (chain_id, res_seq, icode) → polymer_length, 非 HETATM 残基不在字典中
    """
    # dict[tuple, int], 存储每个 HETATM 残基的 polymer_length
    polymer_lengths = {}

    for chain in model:
        # str, 链 ID
        chain_id = chain.get_id()
        # list[Bio.PDB.Residue], 链中所有残基
        res_list = list(chain.get_residues())

        # list[ list[tuple] ], 当前链的所有 HETATM 链段
        # 每个链段是一组连续共价连接的 HETATM 残基键列表
        segments = []
        # list[tuple], 当前正在构建的链段
        current_segment = []

        for i, residue in enumerate(res_list):
            het_flag = residue.id[0]
            if not het_flag.startswith('H_'):
                # 遇到非 HETATM 残基，结束当前链段
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
                continue

            # tuple, 残基唯一键
            res_key = (chain_id, residue.id[1], residue.id[2])

            if not current_segment:
                # 开始新链段
                current_segment = [res_key]
            else:
                # 检查是否与前一个 HETATM 共价连接
                prev_res = res_list[i - 1]
                prev_het = prev_res.id[0]

                if prev_het.startswith('H_') and _are_covalently_bonded_pair(prev_res, residue):
                    # 与前一个 HETATM 共价连接，加入当前链段
                    current_segment.append(res_key)
                else:
                    # 不连接，保存旧链段，开始新链段
                    segments.append(current_segment)
                    current_segment = [res_key]

        # 处理链末尾剩余链段
        if current_segment:
            segments.append(current_segment)

        # 将链段长度写入字典
        for segment in segments:
            # int, 链段长度
            seg_len = len(segment)
            for res_key in segment:
                polymer_lengths[res_key] = seg_len

    return polymer_lengths

# ========================================================================================================================================
# =============================================================== 属性计算 ================================================================












# ============================================================================
# 主函数 / Main Functions(将被 Pocket_Plus\Make_Data\PDB_processor\parser.py 调用以保存候选配体信息 candidates.npz)
# ============================================================================
def find_all_hetatm_candidates(model) -> Tuple[List[LigandCandidate], int]:
    """
    从结构模型中提取所有 HETATM 候选配体，并计算其属性。
    Extract all HETATM candidate ligands from a structure model and compute attributes.

    处理流程 / Pipeline:
        1. 遍历所有链和残基，收集有效 HETATM 候选
           - 排除 WATER_RESIDUES
           - 排除 HETATM_EXCLUSION_LIST（溶剂/缓冲/占位符等）
           - 排除与主链共价连接的 Modified_Residues（修饰受体残基）
        2. 预计算聚合物链段长度
        3. 对每个候选残基：
           a. 提取坐标
           b. 判断金属离子 (单原子 + 金属元素)
           c. 判断标准氨基酸/核苷酸类型
           d. 判断共价连接 (复用 def is_connected_to)
           e. 读取聚合物链段长度

    输入参数 / Input:
        - model: Bio.PDB.Model, Biopython 结构模型 (通常取第一个 model)

    输出 / Output:
        - candidates: list[LigandCandidate], 候选配体列表
        - water_count: int, 被排除的水分子总数
    """

    # =========================================================================
    # 第一步: 预计算聚合物链段长度
    # dict[tuple, int], 残基键 → 聚合链段长度
    polymer_lengths = _compute_polymer_segments(model)


    # =========================================================================
    # 第二步: 遍历所有残基，收集候选
    # list[LigandCandidate], 候选配体列表
    candidates = []
    # int, 全局候选 ID 计数器
    global_id = 0
    # int, 水分子计数
    water_count = 0

    for chain in model:
        # str, 链 ID
        chain_id = chain.get_id()

        for residue in chain:
            het_flag = residue.id[0]
            # 只处理 HETATM 残基
            if not het_flag.startswith('H_'):
                continue
            # str, 残基名 (大写，去空格)
            resname = residue.resname.strip().upper()
            if resname in WATER_RESIDUES:
                water_count += 1
                continue
            # 排除非特异性小分子（溶剂/缓冲/未知组分）
            if resname in HETATM_EXCLUSION_LIST:
                continue
            # 修饰残基若与主链共价连接，则视作受体修饰位点，不作为候选配体
            if resname in Modified_Residues and is_connected_to(residue, chain):
                continue

            # ------ 提取重原子坐标 ------
            # np.ndarray, (M, 3), float32
            coords = _get_heavy_atom_coords(residue)
            # int, 重原子数
            n_heavy = coords.shape[0]
            if n_heavy == 0:
                # 无重原子（全是氢或空残基），跳过
                continue
            # np.ndarray, (3,), float32, 重心
            center = np.mean(coords, axis=0)


            # ------ 分类标志 ------
            # bool, 金属离子
            is_metal = _is_single_atom_metal(residue)
            # bool, 蛋白质修饰/标准氨基酸类 HETATM（已自动做修饰残基映射）
            is_pep = is_protein_residue(resname)
            # bool, 核酸修饰/标准核苷酸类 HETATM（已自动做修饰残基映射）
            is_nuc = is_nucleotide_residue(resname)
            # bool, 与主链共价连接
            is_cov = is_connected_to(residue, chain)


            # ------ 聚合物链长 ------
            # tuple, 残基键
            res_key = (chain_id, residue.id[1], residue.id[2])
            # int, 聚合链段长度 (默认 1)
            poly_len = polymer_lengths.get(res_key, 1)


            # ------ 分子量 ------
            # float, 配体重原子总质量 (Da)
            mw = _compute_molecular_weight(residue)

            # ------ 构造候选 ------
            candidate = LigandCandidate(
                candidate_id=global_id,
                resname=resname,
                chain_id=chain_id,
                res_id=residue.id[1],
                insertion_code=residue.id[2].strip(),
                coords=coords,
                center=center,
                n_heavy_atoms=n_heavy,
                molecular_weight=mw,
                is_metal_ion=is_metal,
                is_peptide_like=is_pep,
                is_nucleotide_like=is_nuc,
                is_covalent=is_cov,
                polymer_length=poly_len,
            )
            candidates.append(candidate)
            global_id += 1

    return candidates, water_count




# 序列化 / Serialization
def save_candidates_npz(
    candidates: List[LigandCandidate],
    water_count: int,
    output_path: str
) -> None:
    """
    将候选配体属性保存到 .npz 文件。
    Save candidate ligand attributes to .npz file.

    输入参数 / Input:
        - candidates: list[LigandCandidate], 候选配体列表
        - water_count: int, 被排除的水分子总数
        - output_path: str, 输出文件路径

    输出文件内容 / Output file contents:
        - n_candidates:       int, 候选数量
        - water_count:        int, 排除的水分子数
        - resnames:           (N,) object, CCD 残基名
        - chain_ids:          (N,) object, 链 ID
        - res_ids:            (N,) int32, 残基序号
        - n_heavy_atoms:      (N,) int32, 重原子数
        - is_metal_ion:       (N,) bool, 金属离子标志
        - is_peptide_like:    (N,) bool, 标准 AA 类 HETATM
        - is_nucleotide_like: (N,) bool, 标准核苷酸类 HETATM
        - is_covalent:        (N,) bool, 共价连接标志
        - polymer_length:     (N,) int32, 聚合物链长
        - centers:            (N, 3) float32, 候选重心
        - candidate_coords_{i}: (M_i, 3) float32, 第 i 个候选原子坐标
    """
    # int, 候选数量
    n = len(candidates)
    # dict, 待保存的键值对
    save_dict = {
        'n_candidates': np.int32(n),
        'water_count': np.int32(water_count),
    }
    if n == 0:
        np.savez_compressed(output_path, **save_dict)
        return

    # ------ 向量化属性 ------
    save_dict['resnames'] = np.array([c.resname for c in candidates], dtype=object)
    save_dict['chain_ids'] = np.array([c.chain_id for c in candidates], dtype=object)
    save_dict['res_ids'] = np.array([c.res_id for c in candidates], dtype=np.int32)
    save_dict['n_heavy_atoms'] = np.array([c.n_heavy_atoms for c in candidates], dtype=np.int32)
    save_dict['is_metal_ion'] = np.array([c.is_metal_ion for c in candidates], dtype=bool)
    save_dict['is_peptide_like'] = np.array([c.is_peptide_like for c in candidates], dtype=bool)
    save_dict['is_nucleotide_like'] = np.array([c.is_nucleotide_like for c in candidates], dtype=bool)
    save_dict['is_covalent'] = np.array([c.is_covalent for c in candidates], dtype=bool)
    save_dict['polymer_length'] = np.array([c.polymer_length for c in candidates], dtype=np.int32)
    save_dict['molecular_weight'] = np.array([c.molecular_weight for c in candidates], dtype=np.float32)
    save_dict['insertion_codes'] = np.array([c.insertion_code for c in candidates], dtype=object)
    save_dict['centers'] = np.array([c.center for c in candidates], dtype=np.float32)

    # ------ 每个候选的原子坐标 ------
    for c in candidates:
        save_dict[f'candidate_coords_{c.candidate_id}'] = c.coords

    np.savez_compressed(output_path, **save_dict)


def load_candidates_npz(path: str) -> Tuple[List[LigandCandidate], int]:
    """
    从 .npz 文件加载候选配体属性。

    输入参数 / Input:
        - path: str, .npz 文件路径

    输出 / Output:
        - candidates: list[LigandCandidate], 候选配体列表
        - water_count: int, 被排除的水分子总数
    """
    data = np.load(path, allow_pickle=True)
    # int, 候选数量
    n = int(data['n_candidates'])
    # int, 水分子数
    water_count = int(data['water_count'])

    if n == 0:
        return [], water_count

    # list[LigandCandidate], 重建候选列表
    candidates = []
    for i in range(n):
        # np.ndarray, (M_i, 3), float32, 第 i 个候选的原子坐标
        coords = data[f'candidate_coords_{i}']
        # float, 分子量 (兼容旧版 npz 不含此字段的情况)
        mw = float(data['molecular_weight'][i]) if 'molecular_weight' in data else 0.0
        candidate = LigandCandidate(
            candidate_id=i,
            resname=str(data['resnames'][i]),
            chain_id=str(data['chain_ids'][i]),
            res_id=int(data['res_ids'][i]),
            insertion_code=str(data['insertion_codes'][i]),
            coords=coords,
            center=data['centers'][i],
            n_heavy_atoms=int(data['n_heavy_atoms'][i]),
            molecular_weight=mw,
            is_metal_ion=bool(data['is_metal_ion'][i]),
            is_peptide_like=bool(data['is_peptide_like'][i]),
            is_nucleotide_like=bool(data['is_nucleotide_like'][i]),
            is_covalent=bool(data['is_covalent'][i]),
            polymer_length=int(data['polymer_length'][i]),
        )
        candidates.append(candidate)

    return candidates, water_count
