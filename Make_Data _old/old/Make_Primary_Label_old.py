import os
import math
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Atom import DisorderedAtom
import mrcfile
import warnings
from scipy.ndimage import binary_dilation
import mrcfile as mrc
from itertools import product
import sys
sys.path.append('/home/penghongen')
from Ligand.utils.network_tools import *
from Ligand.utils.mrc_tools import *


warnings.filterwarnings("ignore")


# =============================
# 用户可配置部分
# =============================
# 定义每个类别编号、来源类别、具体原子————我们规定, class_id可以不相同，但顺序必须是单调不减的
TARGET_ATOMS = [
    {'class_id': 1, 'category': 'protein', 'atoms': ['CA'], 'include_sidechain': False},  # False表示只要主链CA/N，true则包括侧链
    {'class_id': 3, 'category': 'nucleic', 'atoms': ["C4'"], 'include_sidechain': False},
    {'class_id': 2, 'category': 'nucleic', 'atoms': ["P"], 'include_sidechain': False},
]
Make_Hard_Label, Make_DistanceDecay_Map = True, True  # 是否生成0/1标签和距离衰减图

Target_Voxel_Size = None  # 不重采样
Dilation_Iterations, Structure = [0] * len(TARGET_ATOMS), [None] * len(TARGET_ATOMS)
Use_Cube_Kernel, Kernel_Size = False, [1.8] * len(
    TARGET_ATOMS)  # 邻域截断时使用立方邻域或球形邻域; 邻域大小度量————Use_Cube_Kernel=False,1.8这个组合代表3^3立方体
Overlap_To_Big = -1
Decay_Parteen = 'exp^{L2^2}'

# decay_value = 1 - alpha1 * math.exp(- 0.5 * alpha2 * distance**2) + beta
# f(0.5) = 0, f(1) = 0.5, f(1.5) = 0.8得到下面数值解； 此时 f(1.5 x 1.732) = 0.9062
Alpha1 = [1.1848] * len(TARGET_ATOMS)
Alpha2 = [2.1365] * len(TARGET_ATOMS)
Beta = [-0.0929] * len(TARGET_ATOMS)

grid_num_ALL = []  # 各样本对应的map的网格点总数, 之后形状将会是(N, )
target_count_ALL = []  # 各样本各种类别的原子总数, 之后形状将会是(N, len(TARGET_ATOMS)), (n,i)表示第n个样本的第i种类别的原子总数
grid_count_origin_ALL = []  # 各样本各种类别的网格点总数
grid_count_dilated_ALL = []  # ..网格点总数（膨胀后）

overlap_origin_ALL = []  # ..膨胀前重叠网格点总数
overlap_dilated_ALL = []  # ..膨胀后重叠网格点总数




# 核心函数 load_atoms_dict: 分别提取各类原子(分属蛋白、核酸、配体)的坐标
# ====================================================================================================================================================================
def is_connected_to(residue, chain):
    """
    判断一个 HETATM 残基是否与主链连接（通过编号相邻性追溯）。
    """
    res_id = residue.id[1]
    # 1. 收集所有残基的类型和编号 {ID: ID_Type}
    residue_map = {}
    for r in chain.get_residues():
        residue_map[r.id[1]] = r.id[0]  # r.id[0] 是 ' ' (ATOM) 或 'H_' (HETATM)

    # 2. 向前追溯 (N, N-1, N-2, ...)
    i = res_id
    while i in residue_map and residue_map[i].startswith('H_'):
        # 如果当前残基是 HETATM，检查前一个编号
        i -= 1
    # 检查循环结束时 i 所在的位置：如果 i 存在于 map 中，并且它的类型是标准残基 (' ')，则连接。
    if i in residue_map and residue_map[i] == ' ':
        return True  # 找到连接到主链的路径


    # 3. 向后追溯 (N, N+1, N+2, ...)
    j = res_id
    # 注意：这里从 j = res_id 开始追溯时，我们已经知道 res_id 是 HETATM, 追溯时要跳过 res_id 自身，从 j+1 开始。
    j += 1
    while j in residue_map and residue_map[j].startswith('H_'):
        # 如果当前残基是 HETATM，继续检查下一个编号
        j += 1
    # 检查循环结束时 j 所在的位置：如果 j 存在于 map 中，并且它的类型是标准残基 (' ')，则连接。
    if j in residue_map and residue_map[j] == ' ':
        return True  # 找到连接到主链的路径
    # 4. 都没有找到连接到标准残基的路径
    return False


def find_ligand_resnames(pdb_file_path):
    """
    过程：
        - 遍历所有残基
        - 如果是 HETATM 记录，则进行筛选
        - 筛选规则：
            - 在 HETATM_EXCLUSION_LIST 里面的 HETATM 直接排除(不显眼)
            - 如果 HETATM 连接到主链, 那么除非已知它是经典配体, 否则排除
            - 如果 HETATM 不连接到主链, 那么除非是 'MET' 之外的标准残基, 否则保留

    返回:
        - final_ligand_resnames: 集合, 筛选后的配体残基名称
    """

    if pdb_file_path.endswith(".pdb"):
        parser = PDBParser(QUIET=True)
    else:
        parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file_path)

    # --- 定义常量 ---
    standard_residues = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
        'DA', 'DC', 'DG', 'DT', 'A', 'C', 'G', 'U', 'HOH', 'WAT'
    }
    HETATM_EXCLUSION_LIST = {
        'GOL', 'EDO', 'MPD', 'PEG', 'ACT', 'TRS', 'BU1', 'TBU', 'DMSO', 'MES', 'UNX', 'UNL',
        'SEP', 'TPO', 'MSE', 'UNK', 'XE', 'HOH', 'WAT', 'H2O'
    }
    CLASSIC_LIGAND_EXEMPTION_LIST = {
        'HEM', 'FAD', 'NAD', 'NADP', 'ATP', 'ADP', 'GTP', 'GDP', 'FES', 'SF4', 'PLP', 'MGD'
    }


    # --- 第一次遍历：收集所有 HETATM 实例 ---
    all_hetatm_instances = []
    for model in structure:
        for chain in model:
            for residue in chain:
                is_hetatm_record = residue.id[0].startswith('H_')

                if is_hetatm_record:
                    all_hetatm_instances.append((residue, chain))



    # --- 第二次遍历：对每个 HETATM 实例进行筛选 ---
    final_ligand_resnames = set()
    for residue, chain in all_hetatm_instances:
        resname = residue.resname.strip()
        # 1. 排除 HETATM_EXCLUSION_LIST
        if resname in HETATM_EXCLUSION_LIST:
            continue
        is_connected = is_connected_to(residue, chain)
        if is_connected:
            # HETATM 连接到主链：只保留经典配体豁免
            if resname in CLASSIC_LIGAND_EXEMPTION_LIST:
                final_ligand_resnames.add(resname)
            continue  # 否则排除
        else:  # HETATM 未连接到主链 (游离配体，如 HETATM MET 45)
            # 除非是 'MET' 之外的标准残基, 否则保留
            is_standard = resname in standard_residues
            if is_standard and resname != 'MET':
                continue
            final_ligand_resnames.add(resname)

    return final_ligand_resnames


def load_ligand_atoms(pdb_file_path, ligand_names) -> dict:
    """
    加载配体原子并按独立分子分组，每个配体分配唯一 global_id。
    
    # 输入参数:
        - pdb_file_path: str, PDB/CIF文件路径
        - ligand_names: set, find_ligand_resnames 返回的配体残基名集合
    
    # 输出:
        - ligand_dict: dict[int, dict], 形如:
            {
                0: {
                    'global_id': 0,      # int, 全局ID（与键相同）
                    'chain_id': 'A',     # str, 链ID
                    'resname': 'ATP',    # str, 残基名
                    'res_id': 501,       # int, 残基编号
                    'coords': np.ndarray(K, 3)  # 该配体的所有原子坐标
                },
                1: {...},
                ...
            }
          若无配体，返回空字典 {}
    """
    # 选择解析器
    if pdb_file_path.endswith(".pdb"):
        parser = PDBParser(QUIET=True)
    else:
        parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file_path)
    
    # ligand_dict: dict[int, dict], 存放带ID的配体信息
    ligand_dict = {}
    # global_id_counter: int, 全局配体ID计数器
    global_id_counter = 0

    for model in structure:
        for chain in model:
            # chain_id: str, 当前链的ID
            chain_id = chain.get_id()
            for residue in chain:
                resname = residue.resname.strip()  # str, 残基名
                het_flag = residue.id[0]  # str, 异质标志
                res_id = residue.id[1]  # int, 残基编号
                
                # 1. 检查是否是 HETATM 记录
                is_hetatm_record = het_flag.startswith('H_')
                if not is_hetatm_record:
                    continue  # 跳过所有的 ATOM 记录

                # 2. 检查名称是否在 ligand_names 集合中
                if resname in ligand_names:
                    # coords_list: list, 收集该配体的所有原子坐标
                    coords_list = []
                    for atom in residue:
                        if isinstance(atom, DisorderedAtom):
                            coords_list.append(atom.disordered_get_list()[0].get_coord())
                        else:
                            coords_list.append(atom.get_coord())
                    
                    # 只有当配体有原子时才添加
                    if len(coords_list) > 0:
                        # coords: np.ndarray, 形状 (K, 3), 该配体的原子坐标
                        coords = np.array(coords_list)
                        ligand_dict[global_id_counter] = {
                            'global_id': global_id_counter,
                            'chain_id': chain_id,
                            'resname': resname,
                            'res_id': res_id,
                            'coords': coords
                        }
                        global_id_counter += 1

    return ligand_dict
    

def load_atoms_dict(stu_fn, all_structs=False, quiet=True):
    """
    该函数从给定的结构文件（PDB 或 CIF/mmCIF）中解析原子坐标，把蛋白质、核酸和配体原子分别收集起来, 返回 世界(真实)坐标。
    注意配体原子的识别与加载逻辑单独拎出来了, 在 def find_ligand_resnames(...) 和 def load_ligand_atoms(...) 中; 蛋白核酸的检测逻辑内置于本函数。

    # Inputs:
        - stu_fn: 文件名（字符串），支持以 ".pdb" 或 ".cif"/."mmcif" 结尾的结构文件路径。
        - all_structs: 布尔值，默认 False。若为 False，则只处理结构文件中的第一个 model；若为 True，则处理文件中所有 model（multi-model）。
        - quiet: 布尔值，传入给 Biopython 的解析器以控制消息输出（QUIET）。

    # Outputs:返回一个字典，包含三个键：'protein'、'nucleic'、'ligand'：
        - 'protein': 一个字典，键为原子名 (atom name)，值为 numpy 数组 (N1 x 3) 表示的N1个原子的三维坐标;
        - 'nucleic': 类似 'protein'，但存放核酸（DNA/RNA）原子；
        - 'ligand' : 带有 global_id 的配体字典, 由 load_ligand_atoms(...) 返回。形如:
            {
                0: {'global_id': 0, 'chain_id': 'A', 'resname': 'ATP', 'res_id': 501, 'coords': np.ndarray(K0, 3)},
                1: {'global_id': 1, 'chain_id': 'A', 'resname': 'MG', 'res_id': 502, 'coords': np.ndarray(K1, 3)},
                ...
            }

    最终返回：蛋白质与核酸原子字典(字典的键为原子名(如CA), 值为 numpy 数组); 以及配体字典,它形如
    {
        'protein': {'CA': np.ndarray(N1,3), 'N': np.ndarray(N2,3), ...},
        'nucleic': {'P': np.ndarray(M1,3), "C4'": np.ndarray(M2,3), ...},
        'ligand': {0: {...}, 1: {...}, ...}  # 带ID的配体字典
    }
    """
    if stu_fn.split(".")[-1][:3] == "pdb":
        parser = PDBParser(QUIET=quiet)
    elif stu_fn.split(".")[-1][:3] == "cif":
        parser = MMCIFParser(QUIET=quiet)
    else:
        raise RuntimeError("Unknown type for structure file:", stu_fn[-3:])
    structure = parser.get_structure("structure", stu_fn)   # 解析结构文件，得到 Biopython 的 Structure 对象

    # 如果只需要第一个 model（常见于单模型 PDB），则截取第一个 model
    # all_structs=False 时只保留 structure[0]；all_structs=True 时保留所有 model
    if not all_structs:
        structure = [structure[0]]
    # 预定义蛋白质"标准"残基名集合（其余是配体或非标准残基）
    protein_resnames = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
    }
    # 预定义核酸残基名集合（DNA/RNA 常见的残基名）
    nucleic_resnames = {'DA', 'DC', 'DG', 'DT', 'A', 'C', 'G', 'U'}

    # 准备容器：
    # protein_atoms / nucleic_atoms将会是dict。{key = 原子名 (例如 'CA','N','C', 'O'), value = list of 3D 坐标}
    protein_atoms = {}
    nucleic_atoms = {}

    # 遍历结构的模型、链、残基：
    for model in structure:
        for chain in model:
            for residue in chain:
                # residue.id 的第一个元素通常包含 hetflag 信息。 空格' '：表示该残基是一个标准的氨基酸或核苷酸（对应 ATOM 记录）; W表示水; H代表异质分子(配体,特异氨基酸)
                hetflag = residue.id[0].strip()
                resname = residue.resname.strip()  # 残基名称,例如 'ALA'、'DA' 等
                if hetflag != '':   # 跳过所有 HETATM 类型的残基（包括水、配体、其它异质体; 配体单独加载）
                    continue

                # 如果残基名在蛋白质集合中，则把该残基内的原子加入 protein_atoms
                if resname in protein_resnames:
                    for atom in residue:
                        aname = atom.get_name().strip()
                        # 处理 DisorderedAtom（多重构象）情况：取第一个构象的向量（coordinates）
                        vec = atom.disordered_get_list()[0].get_vector().get_array() if isinstance(atom, DisorderedAtom) else atom.get_vector().get_array()
                        # 把 字典（dictionary）protein_atoms 中以 aname 为键的值当成一个列表（list）来使用,如果键 aname 不存在，就用空列表 [] 作为默认值插入并返回这个新列表
                        protein_atoms.setdefault(aname, []).append(vec)

                # 如果残基名在核酸集合中，则把该残基内的原子加入 nucleic_atoms
                elif resname in nucleic_resnames:
                    for atom in residue:
                        aname = atom.get_name().strip()
                        vec = atom.disordered_get_list()[0].get_vector().get_array() if isinstance(atom, DisorderedAtom) else atom.get_vector().get_array()
                        nucleic_atoms.setdefault(aname, []).append(vec)

                else:
                    # 其他残基（例如配体或非标准残基）在此循环不处理；
                    # 配体将在后面通过 find_ligand_resnames & load_ligand_atoms 读取
                    continue

    # 寻找并读取配体残基名（ligand residue names）
    ligand_names = find_ligand_resnames(stu_fn)
    # 使用专门函数 load_ligand_atoms 来加载配体原子（返回带ID的字典结构）
    ligand_dict = load_ligand_atoms(stu_fn, ligand_names)

    # 最终返回：蛋白质与核酸原子字典（字典的键为原子名(如CA)，值为 numpy 数组），以及配体字典
    return {
        'protein': {k: np.array(v) for k, v in protein_atoms.items()},
        'nucleic': {k: np.array(v) for k, v in nucleic_atoms.items()},
        'ligand': ligand_dict
    }
# ====================================================================================================================================================================






# 对核心函数 load_atoms_dict 的进一步处理: 合并元素 or 返回特定原子类型
# ====================================================================================================================================================================
def infer_atom_type(name_in_pdb):
    """
    根据PDB中的原子名推断原子类型(如CA-->C, OG1-->O, NZ-->N)，支持 #see me  17种元素 + 1种未知元素'X'的识别  
    
    # 输入参数:
        - name_in_pdb: str, PDB中的原子名, 如 'CA', 'CB', 'OG1', 'NZ', 'FE', 'MG' 等
    
    # 输出:
        - atom_type: str, 推断出的元素类型, 如 'C', 'O', 'N', 'S', 'FE', 'MG' 等
    
    # 推断规则:
        1. 去除原子名首尾空格
        2. 优先匹配常见的双字符金属元素 (FE, MG, ZN, CA(钙), MN, CU, NA, K, CL, BR, SE 等)
        3. 特殊情况: 区分氨基酸的 CA(α碳) 与金属钙离子 (根据大小写和上下文)
        4. 若以 C/N/O/S/P/H 开头, 则取首字符作为元素类型
        5. 否则返回原子名的前两个字符(若长度>=2)或原字符
    """
    # name_in_pdb: str, PDB中的原子名
    name = name_in_pdb.strip()  # str, 去除首尾空格后的原子名
    if len(name) == 0:
        return ''
    # 双字符金属/特殊元素列表 (大写形式)
    # 注意: 这些通常作为单独的金属离子出现, 原子名恰好就是元素符号
    two_char_elements = {
        'FE', 'MG', 'ZN', 'MN', 'CU', 'NA', 'CL', 'BR', 'SE', 'CO', 'NI', 'MO', 'CD'
    }

    # 1.首先检查是否是双字符元素 (如 FE, MG 等金属离子), 只有当原子名恰好等于双字符元素时才匹配
    name_upper = name.upper()
    if len(name) == 2 and name_upper in two_char_elements:
        return name_upper
    
    # 2.常见的有机元素: C, N, O, S, P, H, 若原子名以这些字符开头, 则取首字符作为元素类型
    # first_char: str, 原子名的首字符 (大写)
    first_char = name[0].upper()
    if first_char in {'C', 'N', 'O', 'S', 'P', 'H'}:
        return first_char
    
    # 对于其他情况 (如稀有元素), 返回 'X'
    return 'X'


def simplify_atom_dict(atom_dict):
    """
    将 load_atoms_dict 返回的 atom_dict 做简化：在某一大类(如'protein'/'nucleic')下, 将同一元素类型的原子坐标合并。对于配体不做简化，直接保留原结构。
    例如: CA(α碳)、CB(β碳)、C(主链羰基碳) 都会合并到元素类型 'C' 下。
    
    # 输入参数:
        - atom_dict: dict, load_atoms_dict 返回的字典, 形如     
            {
                'protein': {'CA': np.ndarray(N1,3), 'N': np.ndarray(N2,3), ...},
                'nucleic': {'P': np.ndarray(M1,3), "C4'": np.ndarray(M2,3), ...},
                'ligand': {0: {'global_id': 0, 'chain_id': 'A', 'resname': 'ATP', 'res_id': 501, 'coords': np.ndarray(K,3)}, ...}
            }
    
    # 输出:
        - simplified_dict: dict, 简化后的字典, 形如
            {
                'protein': {'C': np.ndarray(N_C,3), 'N': np.ndarray(N_N,3), 'O': np.ndarray(N_O,3), ...},
                'nucleic': {'C': np.ndarray(M_C,3), 'N': np.ndarray(M_N,3), 'O': np.ndarray(M_O,3), 'P': np.ndarray(M_P,3), ...},
                'ligand': {0: {...}, 1: {...}, ...}  # 配体不做简化, 直接保留原结构
            }
    """
    # simplified_dict: dict, 用于存放简化后的结果
    simplified_dict = {}
    
    for category, atoms_data in atom_dict.items():
        # category: str, 类别名 ('protein', 'nucleic', 'ligand')
        # atoms_data: dict, 该类别下的原子数据
        
        if category == 'ligand':
            # 配体直接保留原结构, 不做任何简化
            # atoms_data: dict[int, dict], 带ID的配体字典
            simplified_dict[category] = atoms_data
            continue
        
        # element_coords: dict, 键为元素类型 (如 'C', 'N', 'O'), 值为坐标数组列表
        element_coords = {}
        
        for atom_name, coords in atoms_data.items(): # atom_name: str, PDB中的原子名 (如 'CA', 'CB', 'N', 'O'); coords: np.ndarray, 形状 (K, 3), 该元素对应的所有原子坐标
            # element_type: str, 推断的元素类型 (如 'C', 'N', 'O')
            element_type = infer_atom_type(atom_name)
            if element_type not in element_coords:
                element_coords[element_type] = []
            # 将坐标追加到对应元素类型的列表中
            if coords.size > 0:
                element_coords[element_type].append(coords)
        
        # 将列表中的坐标数组合并为单个数组
        # simplified_atoms: dict, 合并后的原子坐标字典
        simplified_atoms = {}
        for element_type, coords_list in element_coords.items():
            if len(coords_list) > 0:
                # merged_coords: np.ndarray, 形状 (N_total, 3), 合并后的坐标
                merged_coords = np.vstack(coords_list)
                simplified_atoms[element_type] = merged_coords
        simplified_dict[category] = simplified_atoms
    

    return simplified_dict


def extract_certain_atom(atom_dict: dict, 
                         atom_names: str | list[str], 
                         molecule_type: str):
    """
    从 atom_dict 中提取特定类型分子的特定原子坐标。
    
    # 输入参数:
        - atom_dict: dict, load_atoms_dict 或者 simplify_atom_dict 返回的字典
        - atom_names: str 或 list[str], 要提取的原子名称
            - 若为 str: 单个原子名, 如 'CA'
            - 若为 list: 原子名列表, 如 ['CA', 'N', 'O']
            - 特殊值 'ALL': 提取所有原子
        - molecule_type: str, 分子类型，支持以下值:
            - 'protein': 提取蛋白质原子
            - 'nucleic': 提取核酸原子
            - 'ligand' 或 'ligand_all': 提取所有配体的原子
            - 'ligand_0', 'ligand_1', ...: 提取 global_id=N 的特定配体原子
            - 'ALL': 提取所有类型的原子
    
    # 输出:
        - coords: np.ndarray, 形状 (N, 3), 特定原子的坐标数组 (世界坐标, 单位 Å)
            若未找到任何匹配原子, 返回空数组 np.empty((0, 3))
    """
    # 统一 atom_names 为列表形式
    if isinstance(atom_names, str):
        atom_names = [atom_names]
    
    # coords_list: list[np.ndarray], 收集所有匹配原子的坐标
    coords_list = []
    
    # 解析 molecule_type，判断是否涉及配体
    # want_ligand_all: bool, 是否提取所有配体
    # want_ligand_id: int or None, 若指定了特定配体ID则为该ID
    want_ligand_all = False
    want_ligand_id = None
    
    if molecule_type in ('ligand', 'ligand_all'):
        want_ligand_all = True
    elif molecule_type.startswith('ligand_'):
        suffix = molecule_type[7:]  # 去掉 'ligand_' 前缀
        if not suffix.isdigit():
            raise ValueError(f"Invalid ligand_type: {molecule_type}, 你必须输入整数！！")
        else:
            want_ligand_id = int(suffix)
    
    for category, atoms_data in atom_dict.items():
        # category: str, 类别名 ('protein', 'nucleic', 'ligand')
        # atoms_data: dict, 该类别下的原子数据
        
        # ========== 处理配体 ==========
        if category == 'ligand':
            # atoms_data: dict[int, dict], 带ID的配体字典
            # 检查是否需要提取配体
            if molecule_type == 'ALL' or want_ligand_all or want_ligand_id is not None:
                if isinstance(atoms_data, dict):
                    if want_ligand_id is not None:
                        # 提取特定ID的配体
                        if want_ligand_id in atoms_data:
                            ligand_info = atoms_data[want_ligand_id]
                            if 'coords' in ligand_info and ligand_info['coords'].size > 0:
                                coords_list.append(ligand_info['coords'])
                    else:
                        # 提取所有配体 (molecule_type == 'ALL' 或 want_ligand_all)
                        for ground_id, ligand_info in atoms_data.items():
                            if 'coords' in ligand_info and ligand_info['coords'].size > 0:
                                coords_list.append(ligand_info['coords'])
            continue  # 处理完ligand后跳过
        
        # ========== 处理蛋白质/核酸 ==========
        # 检查是否需要处理该类别
        if molecule_type != 'ALL' and category != molecule_type:
            continue
        
        if not isinstance(atoms_data, dict):
            continue
        
        for atom_name, coords in atoms_data.items():
            # atom_name: str, PDB中的原子名 (如 'CA', 'CB', 'N', 'O')
            # coords: np.ndarray, 形状 (K, 3), 该原子名对应的所有原子坐标
            
            # 检查是否匹配
            if 'ALL' in atom_names or atom_name in atom_names:
                if isinstance(coords, np.ndarray) and coords.size > 0:
                    coords_list.append(coords)
    
    # 合并所有收集到的坐标
    if len(coords_list) == 0:
        return np.empty((0, 3))  # 返回空数组
    
    # result_coords: np.ndarray, 形状 (N, 3), 合并后的坐标数组
    result_coords = np.vstack(coords_list)
    return result_coords
# ====================================================================================================================================================================
# (目前为止, 还没用到最开始的配置 TARGET_ATOMS)






# -----------------------------
# 距离衰减计算函数 (软标签/特征图)
# -----------------------------
def apply_distance_decay(coords, shape, origin, voxel_size, kernel_size,
                         alpha1, alpha2, beta, overlap_to_big=-1, decay_parteen='exp^{L2^2}', 
                         use_cube_kernel=True):
    """
    基于原子坐标，计算其在三维体素网格中的距离衰减加权特征图（soft label），
    用于表示每个体素到最近原子的距离影响。
    注意：邻域截断（kernel_size）和距离计算（distance）的单位分别为 体素单位 和 物理单位。

    Inputs:
      - coords (np.ndarray): 原子世界坐标做成的数组，形状为 (N, 3)，按 (x, y, z) 顺序。
      - shape (tuple): 输出特征图的三维形状 (Z, Y, X)。
      - origin (np.ndarray): MRC 原点坐标 (x, y, z)。
      - voxel_size (np.ndarray): 每个体素的尺寸 (x, y, z)，单位为 Å。
      - kernel_size (float): 计算邻域范围（单位为体素坐标, 且相当于半径而非直径）。
      - alpha1, alpha2, beta (float): 距离衰减函数的参数，见下文。
      - overlap_to_big (int): 多原子重叠体素的融合策略：
            1 → 取较大值（max）
           -1 → 取较小值（min）
            0 → 先来后到（first）
      - decay_parteen: 'L1', 'L2', 'exp^L2', 'exp^{L2^2}'  注意 alpha_i 是直接乘到L1或L2距离上, beta是加到最后结果上。
      - use_cube_kernel (bool): 是否使用立方体邻域；False 时限定为球形邻域。

    Outputs (返回):
      - feature_map (np.ndarray): 单通道距离衰减特征图，轴顺序为 (Z, Y, X)，值范围为 [β, α·d+β]。

    关键步骤 (Key steps):
      1. 计算每个原子对应的体素索引 (Z_a, Y_a, X_a)。
      2. 在以该索引为中心、kernel_size.astype(int) 为边长的邻域(球邻域或cube邻域)内遍历体素；
      3. 对每个体素计算其中心坐标, 求L1或L2距离：
      4. 若启用球形邻域 (use_cube_kernel=False)，仅保留kernel_size半径内的体素；超出部分赋值为背景 1.0。

    注意 (Caveats):
      - 原子坐标与 origin/voxel_size 均需使用相同单位（通常为 Å）。
      - 距离衰减按照L1或L2是线性的，若希望更快衰减可可替换为指数型函数。
      - 我们规定 feature_map 的默认背景值为 1（未被任何原子覆盖处）。
    """

    if coords.size == 0:
        return np.ones(shape, dtype=np.float32)

    feature_map = np.ones(shape, dtype=np.float32)
    Z_max, Y_max, X_max = shape
    kernel_int = int(kernel_size)

    for atom_coord in coords:
        indices_f = (atom_coord - origin) / voxel_size
        indices_int = np.floor(indices_f[[2, 1, 0]]).astype(int)
        Z_a, Y_a, X_a = indices_int   # 网格坐标(索引)

        z_min = max(0, Z_a - kernel_int)
        z_max = min(Z_max, Z_a + kernel_int + 1)
        y_min = max(0, Y_a - kernel_int)
        y_max = min(Y_max, Y_a + kernel_int + 1)
        x_min = max(0, X_a - kernel_int)
        x_max = min(X_max, X_a + kernel_int + 1)   # 先界定大概范围

        # Cartesian product of input iterables. Equivalent to nested for-loops.
        for z, y, x in product(range(z_min, z_max), range(y_min, y_max), range(x_min, x_max)):

            if not use_cube_kernel:
                if (z - Z_a)**2 + (y - Y_a)**2 + (x - X_a)**2 > kernel_size**2:
                    continue   # 超出球半径的体素不参与计算

            C_v_x = origin[0] + (x + 0.5) * voxel_size[0]   # 真实坐标，用于计算距离
            C_v_y = origin[1] + (y + 0.5) * voxel_size[1]
            C_v_z = origin[2] + (z + 0.5) * voxel_size[2]
            C_v = np.array([C_v_x, C_v_y, C_v_z])

            if decay_parteen == 'L2':
                distance = np.linalg.norm(atom_coord - C_v)
                decay_value = alpha2 * distance**2 + alpha1 * distance + beta
            elif decay_parteen == 'L1':
                distance = np.sum(np.abs(atom_coord - C_v))
                decay_value = alpha2 * distance**2  + alpha1 * distance + beta
            elif decay_parteen == 'exp^L2':
                distance = np.linalg.norm(atom_coord - C_v)
                decay_value = 1 - alpha1 * np.exp(- 0.5 * alpha2 * distance) + beta
            elif decay_parteen == 'exp^{L2^2}':
                distance = np.linalg.norm(atom_coord - C_v)
                decay_value = 1 - alpha1 * np.exp(- 0.5 * alpha2 * distance**2) + beta
            else:
                raise ValueError('decay_parteen should be L1, L2, exp^L2, exp^{L2^2}')


            current_val = feature_map[z, y, x]   # x,y,z为网格坐标
            if overlap_to_big == 1:
                feature_map[z, y, x] = max(current_val, decay_value)
            elif overlap_to_big == -1:
                feature_map[z, y, x] = min(current_val, decay_value)
            elif overlap_to_big == 0:
                    feature_map[z, y, x] = decay_value
            else:
                raise ValueError('overlap_to_big should be 1, -1, 0')

    return feature_map





# =============================
# 批量处理循环
# =============================
 
def create_multiclass_labels(pdb_file_path, map_file_path=None, npz_file_path=None, class_config=None, target_voxel_size=None,     # 输入
                             npz_output_path_1=None, npz_output_path_2=None, make_hard_label=True, make_DistanceDecay_map=True,    # 输出

                             kernel_size=None,  dilation_iterations=None, structure=None, use_cube_kernel=None,                    # 硬标签配置

                             decay_parteen='exp^{L2^2}', alpha1=None, alpha2=None, beta=None, overlap_to_big=-1,       # distance_map配置
                             
                             map_output_prefix=None, save_each_map=True,                              # 保存硬标签以及distance_map的配置

                             infer_pattern=False           # 推理模式专用, 直接依次return numpy数组形式的硬标签(若有)和distance map(若有)                                
                             ):
    """
    根据 pdb 文件对 map 进行多分类标注（multi-class labeling），返回与 load_map(map)生成的grid（map可能重采样） 大小相同的 npy 标签文件。

    输入:
      - pdb_file_path (str): 结构文件路径（PDB 或 CIF），用于提取原子坐标。
      - map_file_path (str): MRC文件路径，作为标签网格（grid）参考。
      - npz_file_path (str): 经过重采样的map保存的npz，优先级高于map_file_path
      - class_config (dict): 类别配置字典，格式见开头：
            TARGET_ATOMS = [
                # 选取全部主链的做法：如果'atoms': ['ALL'], 'include_sidechain': False是不可以的。因为'atoms'会把所有原子包括侧链原子都选中
                {'class_id': 1, 'category': 'protein', 'atoms': ['N', 'CA', 'C', 'O'], 'include_sidechain': False},    # 蛋白主链原子
                {'class_id': 1, 'category': 'nucleic', 'atoms': ['P', 'O5\'', 'C5\'', 'C4\'', 'C3\'', 'O3\''], 'include_sidechain': False}, 
                {'class_id': 2,'category': 'ligand', 'atoms': ['ALL'], 'include_sidechain': True}     # 配体不区分主链/侧链
            ]
      - target_voxel_size (float or None): 若不为 None，则先对 map（grid）与 origin 做重采样（resample），
            使用 make_model_grid 产生与目标体素大小一致的 grid/voxel_size/origin，然后在该重采样后的格点上打标签。

    保存配置：
      - npz_output_path_1/2 (str): 最终保存多分类标签（label_map）的 .npy 文件路径。1是原始label, 2是distance decay map
      - map_output_prefix (str): 保存每个类别二值 MRC 的文件名前缀。

    硬标签配置：
      - kernel_size (N,): 距离衰减邻域的大小（体素单位）。
      - dilation_iterations (N,): 对每个类别的二值 mask 做膨胀（binary dilation）的迭代次数，默认 1（不膨胀，正常打）。  N = len(class_config)代表每轮打标签的情况
      - structure（N，）: 做hardlabel时碰撞所用的 structure, 若为None则为“与距离直接相连(距离中心为1）的那些体素”
      
    distance_map配置：
      - decay_parteen: 'L1', 'L2', 'exp^L2', 'exp^{L2^2}'
      - alpha1, alpha2,  beta (float): 距离衰减函数的参数。
      - overlap_to_big (int): distance decay map的体素融合策略 (1=max, -1=min, 0=first)。


    可视化：
      - map_output_prefix (str): 保存每个类别二值 MRC 的文件名前缀。（通用）
      - save_each_map (bool): 是否为每个类别保存单独的二值 MRC 文件（True/False）
      
      

    Outputs or save:
      - 保存各类别的二值 MRC（可选）和一个最终的 .npy 多分类标签文件（npy_output_path）
      - infer_pattern=True时：依次return numpy数组形式的硬标签(若有)和distance map(若有)；False时返回统计信息

    关键步骤 (Key steps):
      1. 使用 load_atoms_dict 从结构文件读取蛋白（protein）、核酸（nucleic）和配体（ligand）的原子坐标字典（atoms_dict）。
      2. 使用 load_map_and_origin 读取 MRC 网格（map_data）、体素大小（voxel_size）与原点（origin）。
      3. 若指定 target_voxel_size，则先用 make_model_grid 对 map 做重采样（resample），以目标体素大小为准。
      4. 为每个 class_id 读取要匹配的原子（按 class_config 指定），把这些原子坐标映射到栅格索引（使用 atom2map），构造布尔 mask，
         可选对 mask 做二值膨胀（dilation），并将 mask 写入总的 label_map（后定义类别覆盖前者）。
      5. 可选地把每个类别的二值掩码保存为独立的 MRC（使用 save_map_perfect_copy），并在结束时把整体 label_map 保存为 .npy 文件。

    注意 (Caveats):
      - 函数假设 atoms_dict 中的坐标单位和 map 的 origin/voxel_size 单位一致（例如 Å）。
      - atom2map 返回的索引被 clip 到 grid 范围内以避免越界（np.clip），但这会把越界点强行放到边界，
        如果你想要排除越界的原子，请在映射后筛除越界索引而不是 clip。
      - 当多个类别存在重叠（mask 重叠）时，后面的类别会覆盖先前的类别（label_map[mask] = class_id），
        如果希望不同的优先级规则（例如按类别优先级或按体素距离决定），需要更改赋值策略。
    """

    target_count = [0] * len(class_config)   # 原子总数
    grid_count_origin = [0] * len(class_config)   # 网格点总数
    grid_count_dilated = [0] * len(class_config)   # 网格点总数（膨胀后）

    overlap_origin = [0] * len(class_config)   # 膨胀前重叠网格点总数
    overlap_dilated = [0] * len(class_config)   # 膨胀后重叠网格点总数

    # 读取结构文件中分离好的原子坐标（protein/nucleic/ligand）
    atoms_dict = load_atoms_dict(pdb_file_path)
    if npz_file_path is None:
        map_data, voxel_size, origin = load_map_and_origin(map_file_path)
    else:
        data = np.load(npz_file_path)
        map_data, voxel_size, origin = data['grid'], data['voxel_size'], data['global_origin']

    # 若指定了目标体素大小（target_voxel_size），则对 map 做重采样（make_model_grid 会返回新的 grid/voxel_size/origin）
    if target_voxel_size is not None:
        map_data, voxel_size, origin = make_model_grid(grid=map_data, voxel_size=voxel_size, global_origin=origin, target_voxel_size=target_voxel_size)
    grid_num = map_data.size

    # 准备输出标签数组label_map，shape 与 map_data 一致，类型为 uint8（最多支持 0-255 的类别）
    shape = map_data.shape
    label_map = np.zeros(shape, dtype=np.uint8)
    distance_decay_map = []   # 将会合并为 N C D H W, 这里的C不是len(class_config),而将会合并为不同的 class_id 出现的次数


    for count, info in enumerate(class_config):
        class_id = info['class_id']      # 要打的标签(0,1..)
        category = info['category']      # 'protein' / 'nucleic' / 'ligand'
        atom_list = info['atoms']        # 要匹配的原子名列表

        # 若 atoms_dict 中没有该类别的数据，则跳过（例如配置了 'nucleic' 但 PDB 中无核酸）
        if category not in atoms_dict:
            print(f"[警告] 未找到类别 {category} 的原子数据，跳过。")
            continue

        # 临时收集属于类别要打标签成class_id的所有原子坐标
        coords_list = []
        if category == 'ligand':
            # Ligand 的坐标由专门的 load_ligand_atoms 返回（可能是空）
            coords = atoms_dict['ligand']
            if coords.size > 0:
                coords_list.append(coords)
        else:
            # 对于 protein / nucleic，遍历原子字典 atoms_dict[category]
            include_sidechain = info.get('include_sidechain', False)
            for atom_name, arr in atoms_dict[category].items():
                if atom_list == ['ALL']:
                    coords_list.append(arr)   # 采集该类别下所有原子（all atoms）
                elif atom_name in atom_list:
                    coords_list.append(arr)   # 仅采集指定原子名
                elif include_sidechain:
                    # 如果用户希望包含侧链（sidechain），则匹配以指定前缀开头的原子名
                    # 这里以 C/O/N/S 开头（常见的极性侧链原子），则匹配以 atom_list 中任意原子名为开头的atoms_dict[category]中的原子（注意！！！这里修改了侧链逻辑！！！）
                    if any(atom_name.startswith(prefix) for prefix in ['C', 'O', 'N', 'S']):
                        coords_list.append(arr)

        # 若没有收集到任何坐标，则提示并跳过该类别
        if not coords_list:
            print(f"[提示] 类别 {class_id} ({category}:{atom_list}) 没有原子。")
            continue
        coords_array = np.vstack(coords_list)   # 把多个坐标数组垂直堆叠成 (N,3) 的数组

        
        # ------------------------------------------------ 开始：用原始坐标做distance_decay_map -------------------------------------------------
        if make_DistanceDecay_map is True:
            distance_decay_map_OfThisCount = apply_distance_decay(
                coords_array, shape, origin, voxel_size, kernel_size[count],
                alpha1[count], alpha2[count], beta[count], overlap_to_big=overlap_to_big, decay_parteen=decay_parteen, 
                use_cube_kernel=use_cube_kernel
            )
            distance_decay_map.append(distance_decay_map_OfThisCount)   # 后面(在保存图像后)会整合
        # ------------------------------------------------ 结束：用原始坐标做distance_decay_map -------------------------------------------------






        # 保存第count个原子集合 对应的原子总数、初始网格点数
        target_count[count] = coords_array.shape[0]

        mask = np.zeros(shape, dtype=bool)
        # 将原子坐标映射到栅格索引（atom2map 会返回 (N,3) 的整数索引，顺序为 (z,y,x)）
        indices = atom2map(coords_array, origin, voxel_size)

        if indices.size > 0:
            # np.clip 将索引限制到 [0, shape-1] 范围；这会把越界点放到边界
            # .T进行转置, 吧(N,3=zyx)变为(3,N), 使用 numpy 的高级索引
            mask[tuple(np.clip(indices, 0, np.array(shape) - 1).T)] = True
            grid_count_origin[count] = np.sum(mask)
            overlap_origin[count] = np.sum(mask & (label_map != 0))
        if dilation_iterations and dilation_iterations[count] >= 1.0:
            mask = binary_dilation(mask, iterations=dilation_iterations[count], structure=structure[count]) if structure is not None else binary_dilation(mask, iterations=dilation_iterations[count])

        grid_count_dilated[count] = np.sum(mask)
        overlap_dilated[count] = np.sum(mask & (label_map != 0))
        label_map[mask] = class_id

        # 如果需要，保存每个类别的二值 MRC 文件，方便可视化与检查
        if save_each_map:
            binary_map = np.zeros(shape, dtype=np.float32)
            binary_map[mask] = 1.0
            out_path_1_after = f"count{count}_class{class_id}_{category}_{'_'.join(atom_list)}_Of_Hard_Label.mrc"
            out_path_1 = os.path.join(map_output_prefix, out_path_1_after)
            out_path_2_after = f"count{count}_class{class_id}_{category}_{'_'.join(atom_list)}_Of_Distance_Decay_Map.mrc"
            out_path_2 = os.path.join(map_output_prefix, out_path_2_after)
            # 使用 save_map_perfect_copy 保持原始 MRC 的头信息一致性
            if make_hard_label is True:
                save_map_perfect_copy(out_path_1, binary_map, map_file_path, target_voxel_size)
            if make_DistanceDecay_map is True:
                save_map_perfect_copy(out_path_2, distance_decay_map_OfThisCount, map_file_path, target_voxel_size)


        if count >= 1 and class_config[count-1]['class_id'] == class_config[count]['class_id']:
            distance_decay_map[count-1] = np.minimum(distance_decay_map[count-1], distance_decay_map_OfThisCount)
            distance_decay_map.pop()

        print_mask_stats(mask, f"类别 {class_id} ({category}:{atom_list})")

    if make_hard_label is True and npz_output_path_1 is not None:   # 保存整体多分类标签为 .npy 文件
        atomic_np_savez(npz_output_path_1, grid=label_map, origin=origin, voxel_size=voxel_size)
    if make_DistanceDecay_map is True and npz_output_path_2 is not None:   # 保存距离衰减后的map为 .npy 文件
        distance_decay_map = np.array(distance_decay_map)
        atomic_np_savez(npz_output_path_2, grid=distance_decay_map, origin=origin, voxel_size=voxel_size)
    # print(f"\n✅ 多分类标签已保存: {npy_output_path}")
        
    if infer_pattern and make_hard_label is True and make_DistanceDecay_map is True:
        return label_map, distance_decay_map
    elif infer_pattern and make_hard_label is True:
        return label_map
    elif infer_pattern and make_DistanceDecay_map is True:
        return distance_decay_map

    return target_count, grid_count_origin, grid_count_dilated, overlap_origin, overlap_dilated, grid_num














if __name__ == '__main__':

    csv_file = r"/storage/penghongen/newemdlist(4).csv"
    pdb_folder = r"/storage/chenzhaoyang/cryo_em/PDB/"
    map_folder = r"/storage/penghongen/1.5A_map/"
    label_map_output_folder = r"/storage/penghongen/Ligand/mrcMap_of_label1/" # label做成的map存放的文件夹
    npy_output_folder = r"/storage/penghongen/Ligand/label_1/"

    os.makedirs(label_map_output_folder, exist_ok=True)
    os.makedirs(npy_output_folder, exist_ok=True)
    df = pd.read_csv(csv_file)


    # for count, _, row in enumerate(df.iterrows()):   # 一张CPU
    #     folder_name = row['folder_name']  # 形如 emd_1001
    #     pdb_filename = row['real_file']   # 形如 8wxl.pdb
    #     if pd.isna(pdb_filename) or not pdb_filename.strip():
    #         continue
    #     if not (pdb_filename.endswith(".cif") or pdb_filename.endswith(".pdb")):
    #         pdb_file_path = os.path.join(pdb_folder, pdb_filename + ".cif")   # 完整路径(优先cif)，形如 /storage/chenzhaoyang/cryo_em/PDB/8wxl.cif
    #         if not os.path.exists(pdb_file_path):
    #             pdb_file_path = os.path.join(pdb_folder, pdb_filename + ".pdb")   # 完整路径, 形如 /storage/chenzhaoyang/cryo_em/PDB/8wxl.pdb
    #     else:
    #         pdb_file_path = os.path.join(pdb_folder, pdb_filename)   # 完整路径, 形如 /storage/chenzhaoyang/cryo_em/PDB/8wxl.pdb
        
    #     map_file_path = os.path.join(map_folder, folder_name + ".mrc")   # 完整路径, 形如 /storage/penghongen/1.5A_map/emd_1001.mrc
    #     if not os.path.exists(map_file_path) or not os.path.exists(pdb_file_path):
    #         continue

    #     map_subfolder = os.path.join(label_map_output_folder, folder_name)   
    #     os.makedirs(map_subfolder, exist_ok=True)
    #     # map_output_prefix：保存每个类别二值 MRC 的文件名前缀，形如 /storage/penghongen/Ligand/mrcMap_of_label1/emd_1001/
    #     # 里面要保存的文件将会形如 /storage/penghongen/Ligand/mrcMap_of_label1/emd_1001/emd_1001_class0_protein_CA_N
    #     map_output_prefix = os.path.join(map_subfolder, folder_name)   


    #     npy_filename = f"T-{folder_name}.npy"
    #     npy_output_path = os.path.join(npy_output_folder, npy_filename)
    #     try:
    #         print(f"\n--- 开始处理: {folder_name} ({pdb_filename}) ---")
    #         if count < 100:
    #             target_count, grid_count_origin, grid_count_dilated, overlap_origin, overlap_dilated = create_multiclass_labels(pdb_file_path, map_file_path, map_output_prefix=map_output_prefix, npy_output_path=npy_output_path,
    #                             class_config=TARGET_ATOMS,
    #                             dilation_iterations=DILATION_ITERATIONS, save_each_map=True)
    #         else:
    #             target_count, grid_count_origin, grid_count_dilated, overlap_origin, overlap_dilated = create_multiclass_labels(pdb_file_path, map_file_path, map_output_prefix=None, npy_output_path=npy_output_path,
    #                             class_config=TARGET_ATOMS,
    #                             dilation_iterations=DILATION_ITERATIONS, save_each_map=False)
                
    #         target_count_ALL.append(target_count)
    #         grid_count_origin_ALL.append(grid_count_origin)
    #         grid_count_dilated_ALL.append(grid_count_dilated)
    #         overlap_origin_ALL.append(overlap_origin)
    #         overlap_dilated_ALL.append(overlap_dilated)
    #         # print(f"--- 成功处理: {folder_name} ---")
    #     except Exception as e:
    #         print(f"!!! 错误 处理 {folder_name} 时: {e}")


    # ------------------------------------------------------------ 多CPU时 -----------------------------------------------------
    import sys
    num_tasks = 10
    task_id = int(sys.argv[1])
    assert 0 <= task_id < num_tasks, f"task_id must be in [0, {num_tasks-1}]"
    num_files = len(df)
    start_index = (num_files // num_tasks) * task_id
    end_index = (num_files // num_tasks) * (task_id + 1) if task_id != num_tasks - 1 else num_files

    # 在一开始加上这个就好了
        # if count < start_index or count >= end_index:
        #     continue
    # ------------------------------------------------------------ 多CPU时 -----------------------------------------------------


    for count, (_, row) in enumerate(df.iterrows()):   # 多张 CPU

        if count < start_index or count >= end_index:
            continue


        folder_name = row['folder_name']  # 形如 emd_1001
        pdb_filename = row['real_file']   # 形如 8wxl.pdb
        if pd.isna(pdb_filename) or not pdb_filename.strip():
            continue
        if not (pdb_filename.endswith(".cif") or pdb_filename.endswith(".pdb")):
            pdb_file_path = os.path.join(pdb_folder, pdb_filename + ".cif")   # 完整路径(优先cif)，形如 /storage/chenzhaoyang/cryo_em/PDB/8wxl.cif
            if not os.path.exists(pdb_file_path):
                pdb_file_path = os.path.join(pdb_folder, pdb_filename + ".pdb")   # 完整路径, 形如 /storage/chenzhaoyang/cryo_em/PDB/8wxl.pdb
        else:
            pdb_file_path = os.path.join(pdb_folder, pdb_filename)   # 完整路径, 形如 /storage/chenzhaoyang/cryo_em/PDB/8wxl.pdb
        
        map_file_path = os.path.join(map_folder, folder_name + ".mrc")   # 完整路径, 形如 /storage/penghongen/1.5A_map/emd_1001.mrc
        if not os.path.exists(map_file_path) or not os.path.exists(pdb_file_path):
            continue

        map_subfolder = os.path.join(label_map_output_folder, folder_name)   
        os.makedirs(map_subfolder, exist_ok=True)
        # map_output_prefix：保存每个类别二值 MRC 的文件名前缀，形如 /storage/penghongen/Ligand/mrcMap_of_label1/emd_1001/
        # 里面要保存的文件将会形如 /storage/penghongen/Ligand/mrcMap_of_label1/emd_1001/emd_1001_class0_protein_CA_N
        map_output_prefix = os.path.join(map_subfolder, folder_name)   


        npy_filename = f"T-{folder_name}.npy"
        npy_output_path = os.path.join(npy_output_folder, npy_filename)
        try:
            print(f"\n--- 开始处理: {folder_name} ({pdb_filename}) ---")
            if count < 100:
                target_count, grid_count_origin, grid_count_dilated, overlap_origin, overlap_dilated = create_multiclass_labels(pdb_file_path, map_file_path, map_output_prefix=map_output_prefix, npy_output_path=npy_output_path,
                                class_config=TARGET_ATOMS,
                                dilation_iterations=Dilation_Iterations, save_each_map=True)
            else:
                target_count, grid_count_origin, grid_count_dilated, overlap_origin, overlap_dilated = create_multiclass_labels(pdb_file_path, map_file_path, map_output_prefix=None, npy_output_path=npy_output_path,
                                class_config=TARGET_ATOMS,
                                dilation_iterations=Dilation_Iterations, save_each_map=False)
                
            target_count_ALL.append(target_count)
            grid_count_origin_ALL.append(grid_count_origin)
            grid_count_dilated_ALL.append(grid_count_dilated)
            overlap_origin_ALL.append(overlap_origin)
            overlap_dilated_ALL.append(overlap_dilated)
            # print(f"--- 成功处理: {folder_name} ---")
        except Exception as e:
            print(f"!!! 错误 处理 {folder_name} 时: {e}")





    target_count_ALL = np.array(target_count_ALL)
    grid_count_origin_ALL = np.array(grid_count_origin_ALL)
    grid_count_dilated_ALL = np.array(grid_count_dilated_ALL)
    overlap_origin_ALL = np.array(overlap_origin_ALL)
    overlap_dilated_ALL = np.array(overlap_dilated_ALL)




    atomic_np_save(f"/storage/penghongen/Ligand/label_1_statistics/grid_count_origin_ALL_{task_id}.npy", grid_count_origin_ALL)
    atomic_np_save(f"/storage/penghongen/Ligand/label_1_statistics/grid_count_dilated_ALL_{task_id}.npy", grid_count_dilated_ALL)
    atomic_np_save(f"/storage/penghongen/Ligand/label_1_statistics/overlap_origin_ALL_{task_id}.npy", overlap_origin_ALL)
    atomic_np_save(f"/storage/penghongen/Ligand/label_1_statistics/overlap_dilated_ALL_{task_id}.npy", overlap_dilated_ALL)


    target_count_SUM, grid_count_origin_SUM, grid_count_dilated_SUM, overlap_origin_SUM, overlap_dilated_SUM = target_count_ALL.sum(axis=0), grid_count_origin_ALL.sum(axis=0), grid_count_dilated_ALL.sum(axis=0), overlap_origin_ALL.sum(axis=0), overlap_dilated_ALL.sum(axis=0)
    atomic_np_save(f"/storage/penghongen/Ligand/label_1_statistics/target_count_ALL_{task_id}.npy", target_count_ALL)
    print(
        f"target_count_SUM是{target_count_SUM}\n"
        f"grid_count_origin_SUM是{grid_count_origin_SUM}\n"
        f"grid_count_dilated_SUM是{grid_count_dilated_SUM}\n"
        f"overlap_origin_SUM是{overlap_origin_SUM}\n"
        f"overlap_dilated_SUM是{overlap_dilated_SUM}"
    )







# 工具函数
# ==============================================================================================================
# ==============================================================================================================
def load_map_and_origin(mrc_fn: str, multiply_global_origin: bool = True):
    """
    读取并规整 MRC/CCP4 文件的数据、体素大小 (voxel_size) 与全局原点 (global origin)，
    并把数据轴重新排列为 (z, y, x) 以便后续直接用 grid[z,y,x] 索引。

    Inputs:
      - mrc_fn (str): MRC 文件路径。
      - multiply_global_origin (bool): 是否将 global_origin 从 "像素/栅格坐标" 乘以体素大小 (voxel_size)
        来转换为物理坐标 (通常以 Å 表示)。默认 True。
      - remake_voxel_size(float):将加载后的密度图重采样的分辨率。若为None则保持原有分辨率

    Outputs (返回):
      - grid (np.ndarray): 体密度数组，轴顺序为 (z, y, x)。
      - voxel_size (np.ndarray): 长度为 3 的数组，按 (x, y, z) 顺序表示体素大小（voxel size）。
      - global_origin (np.ndarray): 长度为 3 的数组，按 (x, y, z) 表示地图的全局原点，单位为物理坐标（若 multiply_global_origin=True）。

    关键步骤 (Key steps):
      1. 打开 MRC 文件并读取 voxel_size，如果体素大小非法（<=0）则报错，避免错误的头信息。
      2. 读取 header 中的 mapc/mapr/maps 三个字段（表示数据轴与 XYZ 的映射），以及 nxstart/nystart/nzstart（起始像素偏移）。
      3. 通过将 nxstart 按 mapc/mapr/maps 的轴映射映到对应的坐标上，计算并修正 global_origin。
         （MRC header 中 origin 给出的是相对于 nxstart/nystart/nzstart 的偏移；所以我们需要把 start 累加回去。）
      4. 若 multiply_global_origin 为 True，则把 global_origin（以像素/栅格为单位）乘以 voxel_size，得到物理坐标。
      5. 根据 mapc/mapr/maps 的值对原始 mrc_file.data 进行轴重排（使用 np.moveaxis）以得到统一的 (z,y,x) 排序。

    注意 (Caveats):
      - mapc/mapr/maps 的组合定义了 mrc.data 的维度对应于 X/Y/Z 的哪一维：
        - 例如 (mapc, mapr, maps) == (1,2,3) 表示 data 的第一个维度对应 X，第二维对应 Y，第三维对应 Z，
          在这种情况下，mrc_file.data 已经是 (z,y,x) 还是需要重排取决于具体实现；本函数通过多种排列把最终输出标准化为 (z,y,x)。
      - nxstart/nystart/nzstart 与 origin 的语义在不同 MRC 变体中可能略有差异，本函数通过把 start 累回 origin 的方法尝试获得真正的全局原点（以像素为单位），然后乘以 voxel_size 变为物理坐标。
      - 返回的 global_origin 顺序为 (x,y,z)。在后续将原子坐标映射到栅格时要使用相同的坐标轴顺序并注意与 grid 的轴顺序 (z,y,x) 互换。
    """
    mrc_file = mrc.open(mrc_fn, 'r')
    # 读取 voxel size（体素大小），按原始文件的 x,y,z 字段放入数组
    voxel_size = mrc_file.voxel_size
    voxel_size = np.array([voxel_size.x, voxel_size.y, voxel_size.z])
    # 基本校验：如果体素大小为非正值，说明头信息可能损坏或缺失，抛错以便排查
    if voxel_size[0] <= 0:
        raise RuntimeError(f"Seems like the MRC file: {mrc_fn} does not have a header.")
    # 读取 mapc/mapr/maps（三个整数，指示哪个维度对应 X/Y/Z）
    c = mrc_file.header["mapc"]
    r = mrc_file.header["mapr"]
    s = mrc_file.header["maps"]

    # 读取 header 中的 origin（注意 header.origin 表示相对于 start 的偏移）
    global_origin = mrc_file.header["origin"]
    global_origin = np.array([global_origin.x, global_origin.y, global_origin.z])
    # 读取 nxstart/nystart/nzstart（起始像素偏移，整型）
    nstart = np.array([mrc_file.header["nxstart"], mrc_file.header["nystart"], mrc_file.header["nzstart"]])
    # 将 mapc/mapr/maps 转换为 0-based 索引，以便在后续用来把 nxstart 分配到正确的轴上
    temp1 = [c - 1, r - 1, s - 1]
    temp_start = np.zeros(3)
    # 把 nstart 的值按照 temp1 指定的轴位置放回 temp_start 中
    for index in range(3):
        temp_start[temp1[index]] = nstart[index]
    # origin 在 header 中通常是相对于 nxstart/nystart/nzstart 的偏移，
    # 这里把 start 累加回 origin，得到“全局像素/栅格原点”
    global_origin = global_origin + temp_start
    # 如果需要，将像素/栅格单位的 global_origin 乘以体素大小转换为物理坐标（例如 Å）
    if multiply_global_origin:
        global_origin = global_origin * voxel_size
    # 根据 mapc/mapr/maps 的不同组合，将 mrc_file.data 的轴重排为标准的 (z,y,x)
    # 这些分支覆盖了常见的 6 种轴排列方式（mapc,mapr,maps 的置换）
    if c == 1 and r == 2 and s == 3:   # 快,中,慢
        # 原始顺序已经是 (X, Y, Z) 对应于 (2, 1, 0)；mrc_file.data就已经是 (z,y,x) 了
        grid = mrc_file.data
    elif c == 1 and r == 3 and s == 2:
        # (mapc,mapr,maps) == (1,3,2)
        # 需要把维度按 [2,0,1] 移动到位置 [2,1,0]，使得最终为 (z,y,x)
        grid = np.moveaxis(mrc_file.data, [2, 0, 1], [2, 1, 0])
    elif c == 3 and r == 2 and s == 1:
        # (3,2,1) -> 对应的轴重排 [0,1,2] -> [2,1,0]
        grid = np.moveaxis(mrc_file.data, [0, 1, 2], [2, 1, 0])
    elif c == 3 and r == 1 and s == 2:
        # (3,1,2) -> 重排 [1,0,2] -> [2,1,0]
        grid = np.moveaxis(mrc_file.data, [1, 0, 2], [2, 1, 0])
    elif c == 2 and r == 1 and s == 3:
        # (2,1,3) -> 重排 [1,2,0] -> [2,1,0]
        grid = np.moveaxis(mrc_file.data, [1, 2, 0], [2, 1, 0])
    elif c == 2 and r == 3 and s == 1:
        # (2,3,1) -> 重排 [0,2,1] -> [2,1,0]
        grid = np.moveaxis(mrc_file.data, [0, 2, 1], [2, 1, 0])
    else:
        # 如果遇到未知的轴排列，则抛出错误以便排查（避免产生错误的空间解释）
        raise RuntimeError("MRC file axis arrangement not supported!")
    # 关闭文件并返回结果
    mrc_file.close()
    return (grid, voxel_size, global_origin)


def atom2map(coords, origin, voxel_size, pattern='floor'):
    """
    将原子坐标 (x,y,z) 映射到地图的栅格索引 (z,y,x)。

    Inputs:
      - coords (array-like, shape (N,3)或(3,)): 原子坐标，按 (x, y, z)。
      - origin (array-like, length 3): 地图原点，按 (x, y, z)。
      - voxel_size (array-like, length 3): 体素大小，按 (x, y, z)。
    Outputs:
      - indices (np.ndarray, shape (N,3)或(3,), dtype=int): 返回整数栅格索引，索引顺序为 (z, y, x)，可直接用于 grid[z,y,x]。

    Key steps / 关键步骤:
      1. 若 coords 为空，返回空的 (0,3) 整数数组以避免后续错误。
      2. 使用 (coords - origin) / voxel_size 计算浮点栅格坐标（以 (x,y,z) 列序返回）。
      3. 重排列 (x,y,z) -> (z,y,x) 以匹配 grid 的轴顺序，然后将浮点值 cast 为整数（取整行为为向零截断）。

    Caveats / 注意事项:
      - .astype(int) 会直接截断为整数（例如 2.9 -> 2）；若需要四舍五入，可改为 np.round(...).astype(int)；若需要向下取整请用 np.floor。
      - 返回索引可能越界（<0 或 >= grid.shape），在实际索引前应当进行边界检查或使用 np.clip。
      - 确保 coords, origin, voxel_size 单位一致（例如 Å）。
    """
    # 处理空输入：直接返回空索引数组
    if coords.size == 0:
        return np.empty((0, 3), dtype=int)
    # 计算浮点栅格坐标 (x_index, y_index, z_index)
    grid_indices = (np.asarray(coords) - np.asarray(origin)) / np.asarray(voxel_size)
    # 将 (x,y,z) 列重排为 (z,y,x)，并转为整数索引以便用于 grid[z,y,x]
    if pattern == 'round':
        grid_indices = np.round(grid_indices).astype(int)
    elif pattern == 'floor':
        grid_indices = np.floor(grid_indices).astype(int)

    return grid_indices[..., [2, 1, 0]]


def save_map_perfect_copy(file_path, data, original_map_path, current_voxel_size=None):
    """
    将给定的数据写为新的 MRC 文件，同时保留原 MRC 的头信息（voxel_size, cella, map axes 等）。
    函数会根据 original_map_path 自动计算正确的 origin，支持重采样和非标准轴向的密度图。
    
    【参数详细注释 / Input Parameters】
    Inputs:
      - file_path (str): 
          意义: 新 MRC 文件保存路径（若存在将被覆盖）。
          Meaning: Output MRC file path (will be overwritten if exists).
      - data (np.ndarray): 
          数据类型: numpy.ndarray, dtype 通常为 float32
          形状: (Z, Y, X) 三维数组
          意义: 要写入的密度数据/Mask数据。注意 Numpy 默认顺序是 (Z, Y, X)。
          Meaning: Density/Mask data to write. Note Numpy uses (Z, Y, X) order.
      - original_map_path (str): 
          意义: 原始 MRC 文件路径，用于读取体素大小和计算原点。
          Meaning: Original MRC file path for reading voxel size and computing origin.
      - current_voxel_size (float or None): 
          意义: 如果不为 None，表示目标体素大小 (Target Voxel Size)。
               此时函数会模拟重采样过程计算新的 origin 和 voxel_size。
               如果为 None，则直接使用原始 MRC 的 origin 和 voxel_size。
          Meaning: If not None, the target voxel size. The function will compute
               new origin and voxel_size via resampling simulation.
               If None, uses original MRC's origin and voxel_size directly.

    Outputs:
      - None (函数直接在磁盘写入文件 / Function writes file to disk directly)
    """

    # 1. 始终先从原始 MRC 加载数据、体素大小和原点
    #    Always load original MRC to get grid, voxel_size and origin
    # grid_temp: np.ndarray, (Z, Y, X), 原始 MRC 的数据
    # voxel_size_temp: np.ndarray, (3,), [vx, vy, vz], 原始体素大小
    # origin_temp: np.ndarray, (3,), [ox, oy, oz], 原始物理原点 (已处理非标准轴向)
    grid_temp, voxel_size_temp, origin_temp = load_map(original_map_path)
    
    # 2. 判断是否需要处理重采样逻辑
    #    Determine whether resampling logic is needed
    if current_voxel_size is not None:
        # 需要重采样：模拟重采样过程获取新的 voxel_size 和 origin
        # Resampling needed: simulate resampling to get new voxel_size and origin
        
        # new_vs_array: np.ndarray, (3,), 新的体素大小 [vx, vy, vz]
        # final_origin_xyz: np.ndarray, (3,), 重采样后的新原点 [ox, oy, oz]
        _, new_vs_array, final_origin_xyz = make_model_grid(
            grid_temp, voxel_size_temp, origin_temp,
            target_voxel_size=current_voxel_size
        )
        
        # 读取原始文件的 header 结构以复制格式
        # Read original file's header structure to copy format
        with mrcfile.open(original_map_path, header_only=True, permissive=True) as mrc_temp:
            # original_voxel_size: void, mrcfile 的 voxel_size 结构体
            original_voxel_size = mrc_temp.voxel_size.copy()
            # original_cella: void, mrcfile 的 cella 结构体 (晶胞尺寸)
            original_cella = mrc_temp.header.cella.copy()
            
        # 更新 voxel_size (从 numpy array 更新到 mrc void struct)
        # Update voxel_size (from numpy array to mrc void struct)
        original_voxel_size.x = new_vs_array[0]  # float, x 轴像素大小
        original_voxel_size.y = new_vs_array[1]  # float, y 轴像素大小
        original_voxel_size.z = new_vs_array[2]  # float, z 轴像素大小
        
        # 更新 cella 尺寸 (shape * voxel_size) -> 物理总尺寸
        # Update cella dimensions (shape * voxel_size) -> physical total size
        # data.shape 为 (Z, Y, X)
        original_cella.x = data.shape[2] * new_vs_array[0]  # float, X 轴总长 (Å)
        original_cella.y = data.shape[1] * new_vs_array[1]  # float, Y 轴总长 (Å)
        original_cella.z = data.shape[0] * new_vs_array[2]  # float, Z 轴总长 (Å)
        
    else:
        # 不进行重采样：直接使用原始 MRC 的 origin 和 voxel_size
        # No resampling: use original MRC's origin and voxel_size directly
        final_origin_xyz = origin_temp
        
        # 读取原始文件的关键头信息以便复制
        # Read original file's key header info for copying
        with mrcfile.open(original_map_path, permissive=True) as original_mrc:
            original_voxel_size = original_mrc.voxel_size.copy()
            original_cella = original_mrc.header.cella.copy()
    
    # 3. 强制使用标准轴顺序 (1=X, 2=Y, 3=Z) 对应 (Cols, Rows, Sections)
    #    Force standard axis order (1=X, 2=Y, 3=Z) for (Cols, Rows, Sections)
    #    因为输入的 `data` 是标准的 ZYX Numpy 数组（已通过 load_map 处理非标准轴向），
    #    只有设置为 (1, 2, 3) 才能保证可视化软件正确映射空间 XYZ
    original_map_axes = (1, 2, 3)  # tuple, (mapc, mapr, maps)
            
    # 4. 创建新 MRC 并写入数据与头信息
    #    Create new MRC and write data with header info
    with mrcfile.new(file_path, overwrite=True) as mrc:
        # 写入数据，转换为 float32
        # Write data, convert to float32
        mrc.set_data(data.astype(np.float32))
        
        # 设置新的 origin（header 中保存为 x,y,z）
        # Set new origin (stored as x,y,z in header)
        mrc.header.origin.x = float(final_origin_xyz[0])
        mrc.header.origin.y = float(final_origin_xyz[1])
        mrc.header.origin.z = float(final_origin_xyz[2])
        
        # 将起始索引设置为 0（表示数据从网格 0 起始）
        # Set start indices to 0 (data starts from grid 0)
        mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart = 0, 0, 0
        
        # 恢复/设置 voxel size 与 cella
        # Restore/set voxel size and cella
        mrc.voxel_size = original_voxel_size
        mrc.header.cella = original_cella
        
        # 恢复/强制轴映射信息，确保其他软件读取时轴含义一致
        # Restore/force axis mapping info for consistency with other software
        mrc.header.mapc = original_map_axes[0]  # Index of Col axis (1=x)
        mrc.header.mapr = original_map_axes[1]  # Index of Row axis (2=y)
        mrc.header.maps = original_map_axes[2]  # Index of Sec axis (3=z)
        
        # 更新头部统计（min/max/mean 等）以保持一致性
        # Update header stats (min/max/mean etc.) for consistency
        mrc.update_header_stats()


def print_mask_stats(mask, name):
    num_voxels = np.sum(mask)
    percent = 100 * num_voxels / mask.size
    print(f"{name} 掩码包含体素总数: {num_voxels}，占整体的 {percent:.4f}%")
    return num_voxels, percent
