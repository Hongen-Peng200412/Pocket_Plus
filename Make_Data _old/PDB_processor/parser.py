"""
================================================================================
统一预处理系统 - PDB/mmCIF 解析器 / Unified Preprocessing System - Parser
================================================================================

使用 Biopython 解析 PDB 和 mmCIF 文件，提取原子、残基和骨架信息。
Parse PDB and mmCIF files using Biopython, extract atom, residue, and backbone info.

支持 / Supports:
1. 蛋白质 (20种标准氨基酸 + MSE)
2. 核酸 (RNA: A, U, C, G; DNA: DA, DT, DC, DG)
3. 配体检测 (复用 Make_Primary_Label.py 的逻辑)
================================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path

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
    HETATM_EXCLUSION_LIST,
    CLASSIC_LIGAND_EXEMPTION_LIST,
    COVALENT_BOND_THRESHOLD,
)
from .error_logger import return_error_info, ErrorType


# ============================================================================
# 数据结构 / Data Structures
# ============================================================================

@dataclass
class ParsedStructure:
    """
    解析后的结构数据
    Parsed structure data
    
    包含原子级、残基级、配体级数据
    Contains atom-level, residue-level, and ligand-level data
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
    # np.ndarray, (N_res,), int32, 每个残基所属的链索引
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
    


    # ========================= 配体数据 / Ligand data =========================
    # dict[int, dict], 配体字典
    # 格式: {global_id: {'coords': np.ndarray, 'resname': str, 'chain_id': str, 'res_id': int}}
    ligand_dict: Dict = field(default_factory=dict)
    # int, 配体数量
    num_ligands: int = 0



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




def find_ligand_resnames(model) -> Set[str]:
    """
    从结构中识别配体残基名称
    Identify ligand residue names from structure
    
    逻辑 / Logic (复用 Make_Primary_Label.py):
        1. 排除 HETATM_EXCLUSION_LIST 中的残基
        2. 如果连接到主链，只保留经典配体
        3. 如果不连接，除非是'MET'之外的标准残基, 否则保留
    
    输入参数 / Input:
        - model: Bio.PDB.Model, 第一个 model 对象
    
    输出 / Output:
        - set[str], 配体残基名称集合
    """
    # set[str], 标准残基名称
    standard_residues = set([
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
        'DA', 'DC', 'DG', 'DT', 'A', 'C', 'G', 'U', 'HOH', 'WAT'
    ])
    
    # list[tuple], 收集所有 HETATM 实例
    all_hetatm_instances = []
    for chain in model:
        for residue in chain:
            if residue.id[0].startswith('H_'):
                all_hetatm_instances.append((residue, chain))
    
    # set[str], 配体残基名称
    final_ligand_resnames = set()
    
    for residue, chain in all_hetatm_instances:
        resname = residue.resname.strip().upper()
        
        # 排除列表
        if resname in HETATM_EXCLUSION_LIST:
            continue
        
        is_connected = is_connected_to(residue, chain)
        
        if is_connected:
            # 连接到主链: 只保留经典配体
            if resname in CLASSIC_LIGAND_EXEMPTION_LIST:
                final_ligand_resnames.add(resname)
        else:
            # 不连接到主链
            if resname in standard_residues:
                # 标准残基作为配体 (如游离的 MET)
                if resname == 'MET':
                    final_ligand_resnames.add(resname)
                # 短链肽类配体 (长度 <= 25), 按理来说应该加入, 但是目前的基于cryo-EM的建模工具已经可以直接建模它们了(cryAtom),可不加
            else:
                # 非标准残基 (药物、金属等)
                final_ligand_resnames.add(resname)
    
    return final_ligand_resnames




def load_ligand_atoms(model, ligand_names: Set[str]) -> Dict:
    """
    加载配体原子并按独立分子分组
    Load ligand atoms and group by molecule
    
    输入参数 / Input:
        - model: Bio.PDB.Model, 第一个 model 对象
        - ligand_names: set[str], 配体残基名称集合
    
    输出 / Output:
        - dict[int, dict], 配体字典
    """
    ligand_dict = {}
    global_id_counter = 0
    
    for chain in model:
        chain_id = chain.get_id()
        for residue in chain:
            resname = residue.resname.strip().upper()
            het_flag = residue.id[0]
            res_id = residue.id[1]
            
            if not het_flag.startswith('H_'):
                continue
            
            if resname in ligand_names:
                coords_list = []
                for atom in residue:
                    if isinstance(atom, DisorderedAtom):
                        coords_list.append(atom.disordered_get_list()[0].get_coord())
                    else:
                        coords_list.append(atom.get_coord())
                
                if len(coords_list) > 0:
                    ligand_dict[global_id_counter] = {
                        'global_id': global_id_counter,
                        'chain_id': chain_id,
                        'resname': resname,
                        'res_id': res_id,
                        'coords': np.array(coords_list, dtype=np.float32)
                    }
                    global_id_counter += 1
    
    return ligand_dict


# ============================================================================
# 主解析函数 / Main Parsing Function
# ============================================================================

def parse_structure(
    file_path: str,
    output_dir: str,
    sample_id: Optional[str] = None,
    require_ligand: bool = True, 
    select_first_model: bool = False
) -> Optional[ParsedStructure]:
    """
    统一解析结构文件 (PDB 或 mmCIF)
    Unified structure file parsing (PDB or mmCIF)
    
    输入参数 / Input:
        - file_path: str, 结构文件路径
        - output_dir: str, 输出目录 (仅用于错误日志)
        - sample_id: str, 样本ID (可选)
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
                         f"File not found: {file_path}", output_dir, sample_id)
        return None
    # 选择解析器
    file_ext = Path(file_path).suffix.lower()
    if file_ext in ['.pdb']:
        parser = PDBParser(QUIET=True)
    elif file_ext in ['.cif', '.mmcif']:
        parser = MMCIFParser(QUIET=True)
    else:
        return_error_info(file_path, -1, ErrorType.PARSE_ERROR,
                         f"Unsupported file format: {file_ext}", output_dir, sample_id)
        return None
    


    # 解析结构
    try:
        structure = parser.get_structure(sample_id, file_path)
    except Exception as e:
        return_error_info(file_path, -1, ErrorType.PARSE_ERROR,
                         f"Failed to parse structure: {str(e)}", output_dir, sample_id)
        return None
    # 获取 model / Get model from structure
    try:
        model = structure[0]
    except (KeyError, IndexError):
        return_error_info(file_path, -1, ErrorType.EMPTY_STRUCTURE,
                         "No model found in structure", output_dir, sample_id)
        return None

    # 根据 select_first_model 开关决定多 model 处理策略
    if not select_first_model:
        # int, structure 中 model 的数量
        num_models = len(list(structure.get_models()))
        if num_models > 1:
            return_error_info(file_path, -1, ErrorType.PARSE_ERROR,
                             f"Structure contains {num_models} models, skipping (set select_first_model=True to use the first model)",
                             output_dir, sample_id)
            return ">1 model"
    


    # =========================================================================
    # 检测配体 / Detect ligands
    # =========================================================================
    ligand_names = find_ligand_resnames(model)
    ligand_dict = load_ligand_atoms(model, ligand_names)
    
    if require_ligand and len(ligand_dict) == 0:
        return_error_info(file_path, -1, ErrorType.NO_LIGAND,
                         "No ligand detected in structure", output_dir, sample_id)
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
            het_flag = residue.id[0]
            res_seq = residue.id[1]
            i_code = residue.id[2]
            resname = residue.resname.strip().upper()
            
            # 跳过 HETATM (配体单独处理)
            if het_flag != ' ':
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
                         "No valid atoms found", output_dir, sample_id)
        return None
    if len(residue_order) == 0:
        return_error_info(file_path, -1, ErrorType.EMPTY_STRUCTURE,
                         "No valid residues found", output_dir, sample_id)
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
        
        # 代表坐标
        if res['type'] == 'protein':
            if res['ca_coord'] is not None:
                res_coords.append(res['ca_coord'])
            else:
                # 蛋白残基缺少 CA 原子，记录错误并返回 None
                return_error_info(file_path, -1, ErrorType.INCOMPLETE_BACKBONE,
                                 f"Protein residue {res['name']} at {res_key} missing CA atom",
                                 output_dir, sample_id)
                return None
        else:  # 核苷酸
            if res['c4p_coord'] is not None:
                res_coords.append(res['c4p_coord'])
            else:
                # 核苷酸残基缺少 C4' 原子，记录错误并返回 None
                return_error_info(file_path, -1, ErrorType.INCOMPLETE_BACKBONE,
                                 f"Nucleotide residue {res['name']} at {res_key} missing C4' atom",
                                 output_dir, sample_id)
                return None
        
        # 骨架原子
        # 蛋白
        backbone_n_coords.append(res['n_coord'] if res['n_coord'] else [0.0, 0.0, 0.0])
        backbone_ca_coords.append(res['ca_coord'] if res['ca_coord'] else [0.0, 0.0, 0.0])
        backbone_c_coords.append(res['c_coord'] if res['c_coord'] else [0.0, 0.0, 0.0])
        # 核苷酸
        backbone_c4p_coords.append(res['c4p_coord'] if res['c4p_coord'] else [0.0, 0.0, 0.0])
        backbone_c1p_coords.append(res['c1p_coord'] if res['c1p_coord'] else [0.0, 0.0, 0.0])
        backbone_n_base_coords.append(res['n_base_coord'] if res['n_base_coord'] else [0.0, 0.0, 0.0])
        
        # 骨架完整性
        if res['type'] == 'protein':
            is_complete = (res['n_coord'] is not None and 
                          res['ca_coord'] is not None and 
                          res['c_coord'] is not None)
        else:
            is_complete = (res['c4p_coord'] is not None and 
                          res['c1p_coord'] is not None and 
                          res['n_base_coord'] is not None)
        backbone_complete_mask.append(is_complete)


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
        res_seq_numbers=np.array(res_seq_numbers, dtype=np.int32),         # np.ndarray, (N_res,), int32, 残基在链内的序列号 (来自 PDB 的 resSeq)
        
        # ========================= 骨架数据 (蛋白) / Backbone data (Protein) =========================
        backbone_n_coords=np.array(backbone_n_coords, dtype=np.float32),   # np.ndarray, (N_res, 3), float32, 骨架 N 原子坐标
        backbone_ca_coords=np.array(backbone_ca_coords, dtype=np.float32), # np.ndarray, (N_res, 3), float32, 骨架 Cα 原子坐标
        backbone_c_coords=np.array(backbone_c_coords, dtype=np.float32),   # np.ndarray, (N_res, 3), float32, 骨架 C 原子坐标 (羰基碳)
        
        # ========================= 骨架数据 (核酸) / Backbone data (Nucleic Acid) =========================
        backbone_c4p_coords=np.array(backbone_c4p_coords, dtype=np.float32),   # np.ndarray, (N_res, 3), float32, C4' 原子坐标
        backbone_c1p_coords=np.array(backbone_c1p_coords, dtype=np.float32),   # np.ndarray, (N_res, 3), float32, C1' 原子坐标
        backbone_n_base_coords=np.array(backbone_n_base_coords, dtype=np.float32), # np.ndarray, (N_res, 3), float32, N1 (嘧啶) 或 N9 (嘌呤) 原子坐标
        backbone_complete_mask=np.array(backbone_complete_mask, dtype=bool),   # np.ndarray, (N_res,), bool, 骨架完整性掩码
        
        # ========================= 配体数据 / Ligand data =========================
        ligand_dict=ligand_dict,                                           # dict[int, dict], 配体字典 {global_id: {'coords', 'resname', 'chain_id', 'res_id'}}
        num_ligands=len(ligand_dict),                                      # int, 配体数量
    )
    
    return result
