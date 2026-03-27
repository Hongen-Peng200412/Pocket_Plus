"""
================================================================================
统一预处理系统 - PDB/mmCIF 解析器 / Unified Preprocessing System - Parser
================================================================================

使用 Biopython 解析 PDB 和 mmCIF 文件，提取原子、残基、骨架和候选配体信息。
Parse PDB and mmCIF files using Biopython, extract atom, residue, backbone,
and candidate ligand info.

支持 / Supports:
1. 蛋白质 (20种标准氨基酸 + 常见修饰残基映射)
2. 核酸 (RNA/DNA + 常见修饰核苷酸映射)
3. 候选配体检测 (由 ligand_candidates.py 的 find_all_hetatm_candidates 完成)
================================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
import warnings

from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Atom import DisorderedAtom

from .config import (
    ALLOWED_ELEMENTS,
    ALLOWED_RESIDUES,
    ELEMENT_TO_IDX,
    RESIDUE_TO_IDX,
    is_protein_residue,
    is_nucleotide_residue,
    is_purine,
    is_pyrimidine,
    Modified_Residues,
    COVALENT_BOND_THRESHOLD,
)
from .error_logger import return_error_info, ErrorType
from .ligand_candidates import LigandCandidate, find_all_hetatm_candidates, is_connected_to


# ============================================================================
# 数据结构 / Data Structures
# ============================================================================

@dataclass
class ParsedStructure:
    """
    解析后的结构数据
    Parsed structure data
    
    包含原子级、残基级、骨架级、候选配体级数据。
    Contains atom-level, residue-level, backbone-level, and candidate ligand data.

    候选配体说明:
        - ligand_candidates 由 find_all_hetatm_candidates() 填充, 但 n_contact_receptor_atoms 和 n_contact_receptor_residues 由 Part 2 的compute_contact_attributes() 在筛选前就地填充 (依赖 binding_threshold)
        - 每个 LigandCandidate 包含: 基本标识、坐标、大小(molecular_weight)、分类标志(is_metal_ion 等)、接触属性(is_covalent)、聚合物链长
    """
    
    # ========================= 原子级数据 / Atom-level data =========================
    # np.ndarray, (N_atoms, 3), float32, 所有重原子的三维坐标
    atom_coords: np.ndarray
    # list[str], (N_atoms,), 每个原子的元素符号
    atom_elements: List[str]
    # np.ndarray, (N_atoms,), int32, 每个原子归属的残基索引 (0 ~ N_res-1)
    atom_res_indices: np.ndarray
    # np.ndarray, (N_atoms,), int32, 每个原子所属的链索引 (0 ~ N_chains-1)
    atom_chain_indices: np.ndarray
    # list[str], (N_atoms,), 每个原子对应的残基名称
    atom_res_names: List[str]
    # list[str], (N_atoms,), 原子名称 (如 'CA', 'N', 'CB')
    atom_names: List[str]
    




    # ========================= 残基级数据 / Residue-level data =========================
    # np.ndarray, (N_res, 3), float32, 残基代表坐标 (蛋白: Cα; 核酸: C4')
    res_coords: np.ndarray = None
    # list[str], (N_res,), 残基名称列表
    res_names: List[str] = field(default_factory=list)
    # list[str], (N_res,), 残基类型 ('protein' 或 'nucleotide')
    res_types: List[str] = field(default_factory=list)
    # np.ndarray, (N_res,), int32, 每个残基所属的链索引 (0 ~ N_chains-1)
    res_chain_indices: np.ndarray = None
    # np.ndarray, (N_res,), int32, 残基在链内的序列号
    res_seq_numbers: np.ndarray = None
    




    # ========================= 骨架数据 / Backbone data =========================
    # 蛋白质骨架 (N, CA, C)
    # np.ndarray, (N_res, 3), float32, 骨架 N 原子坐标
    backbone_n_coords: np.ndarray = None
    # np.ndarray, (N_res, 3), float32, 骨架 CA 原子坐标
    backbone_ca_coords: np.ndarray = None
    # np.ndarray, (N_res, 3), float32, 骨架 C 原子坐标
    backbone_c_coords: np.ndarray = None
    
    # 核苷酸骨架 (C4', C1', N1/N9)
    # np.ndarray, (N_res, 3), float32, C4' 原子坐标
    backbone_c4p_coords: np.ndarray = None
    # np.ndarray, (N_res, 3), float32, C1' 原子坐标
    backbone_c1p_coords: np.ndarray = None
    # np.ndarray, (N_res, 3), float32, N1 或 N9 原子坐标
    backbone_n_base_coords: np.ndarray = None
    
    # np.ndarray, (N_res,), bool, 骨架完整性掩码
    backbone_complete_mask: np.ndarray = None
    


    # ========================= 候选配体数据 / Candidate Ligand data =========================
    # list[LigandCandidate], 全量候选配体列表（仅排除水分子 + HETATM_EXCLUSION_LIST + 共价连接的修饰残基）
    # 每个候选包含: 坐标、大小(n_heavy_atoms, molecular_weight)、分类标志、共价标志、聚合物链长
    # 注: n_contact_receptor_atoms / n_contact_receptor_residues 由 Part 2 就地填充, 不在此处计算
    ligand_candidates: List[LigandCandidate] = field(default_factory=list)
    # int, 被永久排除的水分子总数
    water_count: int = 0
    # int, 候选配体数量（不含水）
    num_candidates: int = 0

    # # ========================= [已弃用] 旧配体数据 / [DEPRECATED] Old ligand data =========================
    # # dict[int, dict], 旧格式配体字典，由 Part 2 筛选后填充
    # # 格式: {global_id: {'coords': np.ndarray, 'resname': str, 'chain_id': str, 'res_id': int}}
    # # 注意: 此字段不再由 parse_structure() 填充，而由 labels/ligand_filter.py 筛选后生成
    # ligand_dict: Dict = field(default_factory=dict)
    # # int, 筛选后的配体数量
    # num_ligands: int = 0



# ============================================================================
# 辅助函数 / Helper Functions
# ============================================================================

def infer_element_from_atom_name(atom_name: str) -> Optional[str]:
    """
    从原子名推断元素类型
    Infer element type from atom name
    
    输入参数 / Input:
        - atom_name: str, 原子名称 (如 'CA', 'N', 'C4'')
    
    输出 / Output:
        - str 或 None, 元素符号 ('C', 'N', 'O', 'S', 'P') 或 None
    """
    # str, 规范化后的原子名
    name = atom_name.strip().upper()
    if not name:
        return None
    # 检查是否为氢原子 (以 H 开头或数字后跟 H)
    if name[0] == 'H':
        return None
    if name[0].isdigit() and len(name) > 1 and name[1] == 'H':
        return None
    # str, 第一个字符
    first_char = name[0]
    # 检查常见元素
    if first_char in ['C', 'N', 'O', 'S', 'P']:
        return first_char
    return None










# ============================================================================
# 主解析函数 / Main Parsing Function
# ============================================================================

def parse_structure(
    file_path: str,
    error_dir: str,
    sample_id: Optional[str] = None,
    require_ligand: bool = True, 
    select_first_model: bool = False
) -> Optional[ParsedStructure]:
    """
    统一解析结构文件 (PDB 或 mmCIF)
    Unified structure file parsing (PDB or mmCIF)
    
    输入参数 / Input:
        - file_path: str, 结构文件路径
        - error_dir: str, 输出目录 (仅用于错误日志)
        - sample_id: str, 样本ID (可选), 一般为pdb_id, 仅用于发消息
        - require_ligand: bool, 是否要求配体存在 (默认 True)
        - select_first_model: structure选择第一个model / 如果一个structure 含有多个model那么直接记入error_log并跳过处理
    
    输出 / Output:
        - ParsedStructure 或 None, 解析结果；失败返回 None
    """
    # 推断 sample_id
    if sample_id is None:
        sample_id = Path(file_path).stem
    # 检查文件存在
    if not Path(file_path).exists():
        return_error_info(file_path, -1, ErrorType.FILE_NOT_FOUND,
                         f"File not found: {file_path}", error_dir, sample_id)
        return None
    # 选择解析器
    file_ext = Path(file_path).suffix.lower()
    if file_ext in ['.pdb']:
        parser = PDBParser(QUIET=True)
    elif file_ext in ['.cif', '.mmcif']:
        parser = MMCIFParser(QUIET=True)
    else:
        return_error_info(file_path, -1, ErrorType.PARSE_ERROR,
                         f"Unsupported file format: {file_ext}", error_dir, sample_id)
        return None
    


    # 解析结构
    try:
        structure = parser.get_structure(sample_id, file_path)
    except Exception as e:
        return_error_info(file_path, -1, ErrorType.PARSE_ERROR,
                         f"Failed to parse structure: {str(e)}", error_dir, sample_id)
        return None
    # 获取 model / Get model from structure
    try:
        model = structure[0]
    except (KeyError, IndexError):
        return_error_info(file_path, -1, ErrorType.EMPTY_STRUCTURE,
                         "No model found in structure", error_dir, sample_id)
        return None

    # 根据 select_first_model 开关决定多 model 处理策略: False时报错跳过
    if not select_first_model:
        # int, structure 中 model 的数量
        num_models = len(list(structure.get_models()))
        if num_models > 1:
            return_error_info(file_path, -1, ErrorType.PARSE_ERROR,
                             f"Structure contains {num_models} models, skipping (set select_first_model=True to use the first model)",
                             error_dir, sample_id)
            return ">1 model"
    


    # =========================================================================
    # 检测候选配体 / Detect candidate ligands (Part 1: 全量解析)
    # =========================================================================
    # 使用 ligand_candidates.py 的全量候选配体系统
    # （排除水分子 + HETATM_EXCLUSION_LIST + 与主链共价连接的 Modified_Residues）
    # 此处计算的属性: coords, center, n_heavy_atoms, molecular_weight, is_metal_ion,
    #                is_peptide_like, is_nucleotide_like, is_covalent, polymer_length
    # 注: n_contact_receptor_atoms / n_contact_receptor_residues 由 Part 2 就地计算
    # list[LigandCandidate], 全量候选列表
    # int, 被排除的水分子总数
    ligand_candidates, water_count = find_all_hetatm_candidates(model)
    
    if require_ligand and len(ligand_candidates) == 0:
        return_error_info(file_path, -1, ErrorType.NO_LIGAND,
                         "No candidate ligand detected after exclusion rules",
                         error_dir, sample_id)
        return None
    


    # =========================================================================
    # 解析原子和残基 / Parse atoms and residues
    # =========================================================================
    # 临时存储
    atom_data = []           # list[dict]
    residue_info = {}        # dict[tuple, dict]
    residue_order = []       # list[tuple]
    chain_to_idx = {}        # dict[str, int]
    
    for chain in model:
        chain_id = chain.get_id()
        if chain_id not in chain_to_idx:
            chain_to_idx[chain_id] = len(chain_to_idx)
        chain_idx = chain_to_idx[chain_id]
        
        for residue in chain:
            het_flag = residue.id[0]  # HETATM 标识
            res_seq = residue.id[1]   # 残基序列号
            i_code = residue.id[2]    # 插入码
            resname = residue.resname.strip().upper()  # 残基名称
            # 默认跳过 HETATM（配体单独处理），
            # 但 Modified_Residues 中的残基若与主链共价连接，则并入受体主链。
            is_modified_receptor = (
                het_flag.startswith('H_')
                and resname in Modified_Residues
                and is_connected_to(residue, chain)
            )
            if het_flag != ' ' and not is_modified_receptor:
                continue
            
            # 检查是否为允许的残基类型
            if not (is_protein_residue(resname) or is_nucleotide_residue(resname)):
                # 跳过而不报错 (可能是修饰残基)
                continue
            
            # 残基唯一标识
            res_key = (chain_id, res_seq, i_code)
            
            # 初始化残基信息
            if res_key not in residue_info:
                residue_info[res_key] = {
                    'name': resname,
                    'chain_idx': chain_idx,
                    'res_seq': res_seq,
                    'type': 'protein' if is_protein_residue(resname) else 'nucleotide',
                    'atom_coords': [],
                    # 蛋白骨架
                    'n_coord': None,
                    'ca_coord': None,
                    'c_coord': None,
                    # 核苷酸骨架
                    'c4p_coord': None,
                    'c1p_coord': None,
                    'n_base_coord': None,  # N1 或 N9
                }
                residue_order.append(res_key)
            

            # 遍历原子
            for atom in residue:
                atom_name = atom.get_name().strip().upper()
                # 获取坐标
                if isinstance(atom, DisorderedAtom):
                    coord = atom.disordered_get_list()[0].get_coord()
                else:
                    coord = atom.get_coord()
                # 推断元素
                element = None
                if hasattr(atom, 'element') and atom.element:
                    element = atom.element.strip().upper()
                    if element == 'SE':
                        element = 'S'  # MSE 处理
                    elif element not in ALLOWED_ELEMENTS:
                        element = infer_element_from_atom_name(atom_name)
                else:
                    element = infer_element_from_atom_name(atom_name)
                # 跳过无法识别的元素 (如氢原子)
                if element is None:
                    continue

                # 记录原子坐标
                residue_info[res_key]['atom_coords'].append(coord.tolist())
                # 记录骨架原子
                if is_protein_residue(resname):
                    if atom_name == 'N':
                        residue_info[res_key]['n_coord'] = coord.tolist()
                    elif atom_name == 'CA':
                        residue_info[res_key]['ca_coord'] = coord.tolist()
                    elif atom_name == 'C':
                        residue_info[res_key]['c_coord'] = coord.tolist()
                else:  # 核苷酸
                    if atom_name == "C4'":
                        residue_info[res_key]['c4p_coord'] = coord.tolist()
                    elif atom_name == "C1'":
                        residue_info[res_key]['c1p_coord'] = coord.tolist()
                    elif atom_name == 'N9' and is_purine(resname):
                        residue_info[res_key]['n_base_coord'] = coord.tolist()
                    elif atom_name == 'N1' and is_pyrimidine(resname):
                        residue_info[res_key]['n_base_coord'] = coord.tolist()
                
                # 记录到原子列表
                res_idx = residue_order.index(res_key)
                atom_data.append({
                    'coords': coord.tolist(),
                    'element': element,
                    'res_idx': res_idx,
                    'chain_idx': chain_idx,
                    'res_name': resname,
                    'atom_name': atom_name,
                })
    


    # =========================================================================
    # 验证数据 / Validate data
    # =========================================================================
    if len(atom_data) == 0:
        return_error_info(file_path, -1, ErrorType.EMPTY_STRUCTURE,
                         "No valid atoms found", error_dir, sample_id)
        return None
    if len(residue_order) == 0:
        return_error_info(file_path, -1, ErrorType.EMPTY_STRUCTURE,
                         "No valid residues found", error_dir, sample_id)
        return None
    


    # =========================================================================
    # 构建输出结构 / Build output structure
    # =========================================================================
    
    n_atoms = len(atom_data)
    n_residues = len(residue_order)
    
    # 原子级数据
    atom_coords = np.array([d['coords'] for d in atom_data], dtype=np.float32)
    atom_elements = [d['element'] for d in atom_data]
    atom_res_indices = np.array([d['res_idx'] for d in atom_data], dtype=np.int32)
    atom_chain_indices = np.array([d['chain_idx'] for d in atom_data], dtype=np.int32)
    atom_res_names = [d['res_name'] for d in atom_data]
    atom_names = [d['atom_name'] for d in atom_data]
    
    # 残基级数据
    res_names = []
    res_types = []
    res_coords = []
    res_chain_indices = []
    res_seq_numbers = []
    
    # 骨架数据
    backbone_n_coords = []
    backbone_ca_coords = []
    backbone_c_coords = []
    backbone_c4p_coords = []
    backbone_c1p_coords = []
    backbone_n_base_coords = []
    backbone_complete_mask = []
    
    for res_key in residue_order:   # 按照 residue_order 的顺序添加
        res = residue_info[res_key]
        res_names.append(res['name'])
        res_types.append(res['type'])
        res_chain_indices.append(res['chain_idx'])
        res_seq_numbers.append(res['res_seq'])
        
        # 代表坐标 + 严格骨架完整性检查
        if res['type'] == 'protein':
            missing = []
            if res['n_coord'] is None:
                missing.append('N')
            if res['ca_coord'] is None:
                missing.append('CA')
            if res['c_coord'] is None:
                missing.append('C')
            if missing:
                return_error_info(
                    file_path,
                    -1,
                    ErrorType.INCOMPLETE_BACKBONE,
                    f"Protein residue {res['name']} at {res_key} missing backbone atoms: {','.join(missing)}",
                    error_dir,
                    sample_id,
                )
                return None
            res_coords.append(res['ca_coord'])
        else:  # 核苷酸
            missing = []
            if res['c4p_coord'] is None:
                missing.append("C4'")
            if res['c1p_coord'] is None:
                missing.append("C1'")
            if res['n_base_coord'] is None:
                missing.append('N1/N9')
            if missing:
                return_error_info(
                    file_path,
                    -1,
                    ErrorType.INCOMPLETE_BACKBONE,
                    f"Nucleotide residue {res['name']} at {res_key} missing backbone atoms: {','.join(missing)}",
                    error_dir,
                    sample_id,
                )
                return None
            res_coords.append(res['c4p_coord'])
        
        # 骨架原子
        # 蛋白
        if res['type'] == 'protein':
            backbone_n_coords.append(res['n_coord'])
            backbone_ca_coords.append(res['ca_coord'])
            backbone_c_coords.append(res['c_coord'])
            # 非核苷酸骨架，填 NaN 以保持 array 形状一致 (N_res, 3)
            backbone_c4p_coords.append([np.nan, np.nan, np.nan])
            backbone_c1p_coords.append([np.nan, np.nan, np.nan])
            backbone_n_base_coords.append([np.nan, np.nan, np.nan])
        elif res['type'] == 'nucleotide':
            backbone_c4p_coords.append(res['c4p_coord'])
            backbone_c1p_coords.append(res['c1p_coord'])
            backbone_n_base_coords.append(res['n_base_coord'])
            # 非蛋白骨架，填 NaN 以保持 array 形状一致 (N_res, 3)
            backbone_n_coords.append([np.nan, np.nan, np.nan])
            backbone_ca_coords.append([np.nan, np.nan, np.nan])
            backbone_c_coords.append([np.nan, np.nan, np.nan])

        # 严格模式下走到这里说明骨架完整
        backbone_complete_mask.append(True)


    result = ParsedStructure(
        # ========================= 原子级数据 / Atom-level data =========================
        atom_coords=atom_coords,                                           # np.ndarray, (N_atoms, 3), float32, 所有重原子的三维坐标 [x, y, z]
        atom_elements=atom_elements,                                       # list[str], (N_atoms,), 每个原子的元素符号 ('C', 'N', 'O', 'S', 'P')
        atom_res_indices=atom_res_indices,                                 # np.ndarray, (N_atoms,), int32, 原子所属的残基全局索引 (0 ~ N_res-1)
        atom_chain_indices=atom_chain_indices,                             # np.ndarray, (N_atoms,), int32, 每个原子所属的链索引 (0 ~ N_chains-1)
        atom_res_names=atom_res_names,                                     # list[str], (N_atoms,), 每个原子对应的残基名称 (3字母代码)
        atom_names=atom_names,                                             # list[str], (N_atoms,), 原子名称 (如 'CA', 'N', 'CB', "C4'")
        
        
        # ========================= 残基级数据 / Residue-level data =========================
        res_coords=np.array(res_coords, dtype=np.float32),                 # np.ndarray, (N_res, 3), float32, 残基代表坐标 (蛋白: Cα; 核酸: C4')
        res_names=res_names,                                               # list[str], (N_res,), 有序的残基名称列表 (3字母代码)
        res_types=res_types,                                               # list[str], (N_res,), 残基类型 ('protein' 或 'nucleotide')
        res_chain_indices=np.array(res_chain_indices, dtype=np.int32),     # np.ndarray, (N_res,), int32, 每个残基所属的链索引
        res_seq_numbers=np.array(res_seq_numbers, dtype=np.int32),         # np.ndarray, (N_res,), int32, 残基在链内的序列号 (来自 PDB 的 resSeq, 链内连续单增), 用于知道同一条链上残基的相对位置
        

        # ========================= 骨架数据 (蛋白) / Backbone data (Protein) =========================
        backbone_n_coords=np.array(backbone_n_coords, dtype=np.float32),   # np.ndarray, (N_res, 3), float32, 骨架 N 原子坐标
        backbone_ca_coords=np.array(backbone_ca_coords, dtype=np.float32), # np.ndarray, (N_res, 3), float32, 骨架 Cα 原子坐标
        backbone_c_coords=np.array(backbone_c_coords, dtype=np.float32),   # np.ndarray, (N_res, 3), float32, 骨架 C 原子坐标 (羰基碳)
        

        # ========================= 骨架数据 (核酸) / Backbone data (Nucleic Acid) =========================
        backbone_c4p_coords=np.array(backbone_c4p_coords, dtype=np.float32),   # np.ndarray, (N_res, 3), float32, C4' 原子坐标
        backbone_c1p_coords=np.array(backbone_c1p_coords, dtype=np.float32),   # np.ndarray, (N_res, 3), float32, C1' 原子坐标
        backbone_n_base_coords=np.array(backbone_n_base_coords, dtype=np.float32), # np.ndarray, (N_res, 3), float32, N1 (嘧啶) 或 N9 (嘌呤) 原子坐标
        backbone_complete_mask=np.array(backbone_complete_mask, dtype=bool),   # np.ndarray, (N_res,), bool, 骨架完整性掩码
        

        # ========================= 候选配体数据 / Candidate ligand data =========================
        ligand_candidates=ligand_candidates,                               # list[LigandCandidate], 全量候选配体（仅排除水分子）
        water_count=water_count,                                           # int, 被排除的水分子总数
        num_candidates=len(ligand_candidates),                             # int, 候选配体数量
        # ligand_dict 和 num_ligands 将由 Part 2 筛选后填充
        # ligand_dict and num_ligands will be populated after Part 2 filtering
    )
    
    return result
