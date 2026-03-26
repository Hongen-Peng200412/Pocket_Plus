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
from utils import load_map_and_origin, atom2map, vitualiza_grid_VS_map, print_mask_stats


warnings.filterwarnings("ignore")


# 核心函数 load_atoms_dict: 分别提取各类原子(分属蛋白、核酸、配体)的坐标
# ====================================================================================================================================================================
def is_connected_to(residue, chain):
    """
    判断一个 HETATM 残基是否与主链连接（通过在链中的序列相邻性 + 物理距离检测）。
    Determine if a HETATM residue is connected to the main chain (tracing by sequence adjacency AND physical bond distance).

    # 输入参数 / Input:
        - residue: Bio.PDB.Residue, 当前需要检查的残基对象。Current residue object to check.
        - chain: Bio.PDB.Chain, 该残基所在的链对象。The chain object containing the residue.

    # 输出结果 / Output:
        - bool, 若连接到标准残基(Standard Residue)返回 True，否则返回 False。True if connected to a standard residue, else False.
    
    # 逻辑 / Logic:
        - 使用链表索引进行遍历，首先检查断链(Gaps)。Iterate using list indices, checking for gaps.
        - [New] 增加物理距离检测 (<1.8A)。Add physical distance check (<1.8A) to confirm covalent bond.
            - 防止仅因编号连续(如 300->301)但实际游离的配体被误判为连接。Prevent free ligands with sequential numbering from being misclassified.
        - 这可以解决 PDB 插入码 (Insertion Codes, e.g. 10A, 10B) 问题。Resolves Insertion Codes issues.
    """
    
    def _are_ids_adjacent(id1, id2):
        """
        检查两个残基 ID 是否连续。Check if two residue IDs are validly consecutive.
        id tuple: (het_flag, resseq, icode)
        """
        seq1, ins1 = id1[1], id1[2]
        seq2, ins2 = id2[1], id2[2]
        
        diff = seq2 - seq1
        if diff == 1:
            # 序列号增加1，允许任意插入码变化 (通常 10->11, 10A->11 都是合法的)
            # Seq increment by 1. Usually valid regardless of ins code.
            return True
        elif diff == 0:
            # 序列号相同，检查插入码 (10 -> 10A, 10A -> 10B)
            # Same seq, check insertion code progression
            if ins1 == ' ' and ins2 == 'A':
                return True
            if ins1 != ' ' and ins2 != ' ' and (ord(ins2) - ord(ins1) == 1):
                return True
        
        return False

    def _is_covalently_bonded(res1, res2, threshold=1.8):
        """
        检查 res1(前) 和 res2(后) 之间是否存在骨架共价键 (C-N or O3'-P)。
        Check for backbone covalent bond between res1 (prev) and res2 (curr).
        Threshold: 1.8A (Acceptable for C-N ~1.33A, P-O ~1.6A with tolerance).
        """
        # 1. Peptide Bond: res1.C - res2.N
        if 'C' in res1 and 'N' in res2:
            diff = res1['C'] - res2['N']
            if diff < threshold:
                return True
        
        # 2. Nucleic Acid: res1.O3' - res2.P
        # Try common names for O3'
        o3_names = ["O3'", "O3*", "O3"] 
        # Note: PDB atom nams can be tricky. "O3'" is standard in Biopython for RNA/DNA.
        atom_o3 = None
        for name in o3_names:
            if name in res1:
                atom_o3 = res1[name]
                break
        
        if atom_o3 and 'P' in res2:
            diff = atom_o3 - res2['P']
            if diff < threshold:
                return True
                
        return False

    # 1. 获取链中所有残基的列表 (有序)
    res_list = list(chain.get_residues())
    
    try:
        idx = res_list.index(residue)
    except ValueError:
        return False

    # 2. 向前追溯 (Current -> Previous)
    curr_idx = idx
    while curr_idx > 0:
        prev_idx = curr_idx - 1
        curr_res = res_list[curr_idx]
        prev_res = res_list[prev_idx]
        
        # 检查 ID 连续性 / Check ID continuity
        # 注意：这里我们检查 prev -> curr 的连接
        if not _are_ids_adjacent(prev_res.id, curr_res.id):
             # 发现断链(Gap)，停止追溯
             break
        
        # [New] 检查物理连接 / Check Physical Connection
        if not _is_covalently_bonded(prev_res, curr_res):
            # 编号连续但无共价键 -> 视为断开 (游离配体)
            break
        
        # 检查前一个残基类型
        if prev_res.id[0].startswith('H_'):
            curr_idx -= 1 # 继续向前
        elif prev_res.id[0] == ' ':
            return True # 连接到标准残基
        else:
            break # 水或其他

    # 3. 向后追溯 (Current -> Next)
    curr_idx = idx
    while curr_idx < len(res_list) - 1:
        next_idx = curr_idx + 1
        curr_res = res_list[curr_idx]
        next_res = res_list[next_idx]
        
        # Check ID continuity
        if not _are_ids_adjacent(curr_res.id, next_res.id):
            break

        # [New] 检查物理连接 / Check Physical Connection
        if not _is_covalently_bonded(curr_res, next_res):
            break
            
        if next_res.id[0].startswith('H_'):
            curr_idx += 1
        elif next_res.id[0] == ' ':
            return True
        else:
            break

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
            # HETATM connected to the main chain: only retain explicit Classic Ligand Exemptions
            if resname in CLASSIC_LIGAND_EXEMPTION_LIST:
                final_ligand_resnames.add(resname)
            continue  # 否则排除
        else:  # HETATM 未连接到主链 (游离配体) / HETATM not connected to main chain
            
            # 检查是否为标准残基 / Check if it is a standard residue
            is_standard = resname in standard_residues

            if is_standard:
                # 特殊处理：如果是 MET，保留 (原有逻辑) / Keep MET as originally intended
                if resname == 'MET':
                    final_ligand_resnames.add(resname)
                    continue
                
                # 新逻辑：如果是标准残基构成的短链 (Peptide Ligand)，保留————————[肽类配体]
                # New Logic: Keep it if it is a short chain composed of standard residues (Peptide Ligand)
                
                # 计算该残基所在的 HETATM 片段长度 / Calculate length of the HETATM fragment
                # 我们复用 _are_ids_adjacent 逻辑进行计数 / Reuse logic to count connected size
                # 注意：这里我们重新遍历一次链来计算当前residue所属的fragment大小，或者我们可以只做局部搜索
                # Note: We do a local search to find the fragment size.
                
                # 定义内部函数来计算长度 (Efficiency Note: could be optimized but valid for typical PDBs)
                def get_hetatm_fragment_size(start_residue, chain_obj):
                    res_list_inner = list(chain_obj.get_residues())
                    try:
                        start_idx = res_list_inner.index(start_residue)
                    except ValueError:
                        return 0

                    count = 1
                    
                    # 向前搜索 / Trace Backward
                    curr = start_idx
                    while curr > 0:
                        prev = curr - 1
                        curr_r = res_list_inner[curr]
                        prev_r = res_list_inner[prev]
                        
                        # 检查 ID 连续性 / Check ID continuity
                        if not is_connected_to.__code__.co_consts[0] if False else True: # Hack to access helper? No, just redefine simple check or assume simple continuity
                             # 为了简单起见，我们假设ID连续即可 (和 is_connected_to 里的逻辑一致)
                             # For simplicity, duplicate simple ID adjacency check
                             seq1, ins1 = prev_r.id[1], prev_r.id[2]
                             seq2, ins2 = curr_r.id[1], curr_r.id[2]
                             diff = seq2 - seq1
                             adjacent = False
                             if diff == 1: adjacent = True
                             elif diff == 0:
                                 if ins1 == ' ' and ins2 == 'A': adjacent = True
                                 elif ins1 != ' ' and ins2 != ' ' and (ord(ins2)-ord(ins1)==1): adjacent = True
                             
                             if not adjacent: break

                        # 必须也是 HETATM / Must be HETATM
                        if prev_r.id[0].startswith('H_'):
                            count += 1
                            curr -= 1
                        else:
                            break # Hit standard atom or water
                    
                    # 向后搜索 / Trace Forward
                    curr = start_idx
                    while curr < len(res_list_inner) - 1:
                        next_i = curr + 1
                        curr_r = res_list_inner[curr]
                        next_r = res_list_inner[next_i]
                        
                        seq1, ins1 = curr_r.id[1], curr_r.id[2]
                        seq2, ins2 = next_r.id[1], next_r.id[2]
                        diff = seq2 - seq1
                        adjacent = False
                        if diff == 1: adjacent = True
                        elif diff == 0:
                             if ins1 == ' ' and ins2 == 'A': adjacent = True
                             elif ins1 != ' ' and ins2 != ' ' and (ord(ins2)-ord(ins1)==1): adjacent = True
                        
                        if not adjacent: break
                        
                        if next_r.id[0].startswith('H_'):
                            count += 1
                            curr += 1
                        else:
                            break

                    return count

                frag_len = get_hetatm_fragment_size(residue, chain)
                
                # 设定阈值 / Set Threshold
                if frag_len <= 25:
                    final_ligand_resnames.add(resname)
                # else: 长度 > 25，视为长蛋白链的错误标记，忽略 / Ignore as mislabeled protein chain
                
            else:
                # 非标准残基 (e.g. ZN, MG, Drug)，直接保留
                # Non-standard residue, keep it.
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
















# ====================================================================================================================================================================
# Instance Segmentation Labeling Functions / 实例分割标签生成函数
# For Mask3D Integration / 用于 Mask3D 集成
# ====================================================================================================================================================================

def compute_atom_ligand_distances(
    atom_coords: np.ndarray,
    ligand_dict: dict
) -> tuple:
    """
    计算每个原子到每个配体的最近距离
    Compute nearest distance from each atom to each ligand
    
    输入参数 / Input:
        - atom_coords: np.ndarray, (N, 3), 原子坐标（单位：Å）
            N: int, 原子数量
        - ligand_dict: dict, 配体字典，格式为:
            {
                global_id: {'coords': np.ndarray(K, 3), ...},
                ...
            }
    
    输出 / Output:
        - distances: np.ndarray, (N, num_ligands), 它的(i,j)分量代表原子i到 (按照 global_id)编号为 ligand_ids[j] 的配体的最近距离
        - ligand_ids: list[int], 配体 global_id 列表（按顺序）= sorted(ligand_dict.keys()), 注意本脚本生成的ligand_id就是连续单增数组:[0,1,2...], 但这么写是为了兼容性, 即允许ligand_ids乱序
    """
    # int, 原子数量
    num_atoms = atom_coords.shape[0]
    # list[int], 配体 global_id 列表
    ligand_ids = sorted(ligand_dict.keys())
    # int, 配体数量
    num_ligands = len(ligand_ids)
    if num_atoms == 0 or num_ligands == 0:
        return np.empty((num_atoms, num_ligands)), ligand_ids

    # np.ndarray, (N, num_ligands), 距离矩阵
    distances = np.full((num_atoms, num_ligands), np.inf, dtype=np.float32)
    for lig_idx, global_id in enumerate(ligand_ids):
        # np.ndarray, (K, 3), 配体原子坐标
        ligand_coords = ligand_dict[global_id]['coords']
        if ligand_coords.size == 0:
            continue
        # 计算每个原子到该配体所有原子的距离，取最小值
        # np.ndarray, (N, K), 距离矩阵
        # 使用广播计算: atom_coords[:, None, :] - ligand_coords[None, :, :]
        diff = atom_coords[:, None, :] - ligand_coords[None, :, :]  # (N, K, 3)
        # np.ndarray, (N, K), 欧氏距离
        dist_matrix = np.linalg.norm(diff, axis=2)
        # np.ndarray, (N,), 每个原子到该配体的最近距离
        min_dist = np.min(dist_matrix, axis=1)
        
        distances[:, lig_idx] = min_dist
    
    return distances, ligand_ids




def create_instance_labels(
    pdb_file_path: str,
    output_path: str = None,
    threshold: float = 4.5,
    
    include_protein: bool = True,
    include_nucleic: bool = True,
    atom_types: list = None,
) -> dict:
    """
    为 Mask3D 实例分割生成原子级别的标签
    Generate atom-level labels for Mask3D instance segmentation
    
    核心逻辑 / Core Logic:
        1. 加载蛋白质/核酸原子和配体
        2. 计算每个原子到每个配体的最近距离
        3. 如果距离 <= threshold，将原子分配给该配体对应的 pocket instance, 也记录精确距离用于未来软标签训练
    
    输入参数 / Input:
        - pdb_file_path: str, PDB/CIF 文件路径
        - output_path: str, 输出 .npz 文件路径（可选）
        - threshold: float, 距离阈值（Å），默认 4.5, 原子到配体距离 <= threshold 时认为属于该口袋
        - include_protein: bool, 是否包含蛋白质原子，默认 True
        - include_nucleic: bool, 是否包含核酸原子，默认 True
        - atom_types: list[str], 要包含的原子类型（如 ['CA', 'N', 'C', 'O']）, 默认 None 表示包含所有原子
    
    输出 / Output:
        返回字典包含 / Returns dict containing:
        - 'coords': np.ndarray, (N, 3), 原子世界坐标（Å）
        - 'ligand_ids': np.ndarray, (num_ligands,), 配体的全局 ID, 它是恒等映射(即配体的局部id保持为全局id不变; 写上以保持接口灵活罢了)
        - 'instance_ids': np.ndarray, (N,), 每个原子的实例 ID: 0 = 背景（不属于任何口袋）; 实例 ID = ligand_global_id + 1
        - 'distances': np.ndarray, (N, num_ligands), 原子i到配体j的距离, 配体用局部id(配体j的全局ID为 ligand_ids[j])
        - 'nearest_instance': np.ndarray, (N,), 每个原子最近的配体 ID, 用全局id
        - 'nearest_distance': np.ndarray, (N,), 每个原子到最近配体的距离
        - 'ligand_mapping': dict, {instance_id: ligand_global_id} 映射(i --> i-1)
        - 'num_instances': int, 实例数量（不包括背景）
        - 'atom_types': np.ndarray, (N,) of str, 每个原子的类型（如 'CA', 'N'）
        - 'molecule_types': np.ndarray, (N,) of str, 每个原子的分子类型（'protein', 'nucleic'）
        
    保存格式 / Save Format:
        如果提供 output_path，保存为 .npz 文件
    """
    print(f"\n{'='*60}")
    print(f"[create_instance_labels] 开始生成实例标签")
    print(f"[create_instance_labels] PDB: {pdb_file_path}")
    print(f"[create_instance_labels] Threshold: {threshold} Å")
    print(f"{'='*60}")
    
    # ============================================================
    # 1. 加载原子数据 / Load atom data
    # ============================================================
    # dict, 包含 'protein', 'nucleic', 'ligand' 的原子字典
    atoms_dict = load_atoms_dict(pdb_file_path)
    # 提取配体字典
    # dict, 配体字典 {global_id: {'coords': np.ndarray, ...}}
    ligand_dict = atoms_dict.get('ligand', {})
    if not ligand_dict:
        print("[警告] 未找到配体，无法生成实例标签")
        return None
    print(f"[INFO] 找到 {len(ligand_dict)} 个配体")
    for gid, info in ligand_dict.items():
        resname = info.get('resname', 'UNK')
        n_atoms = info['coords'].shape[0] if 'coords' in info else 0
        print(f"  - Ligand {gid}: {resname}, {n_atoms} atoms")
    


    # ============================================================
    # 2. 收集蛋白质/核酸原子 / Collect protein/nucleic atoms
    # ============================================================
    # list[np.ndarray], 坐标列表
    coords_list = []
    # list[str], 原子类型列表
    atom_type_list = []
    # list[str], 分子类型列表
    molecule_type_list = []
    # 处理蛋白质原子
    if include_protein and 'protein' in atoms_dict:
        for aname, coords in atoms_dict['protein'].items():
            if atom_types is not None and aname not in atom_types:
                continue
            if coords.size > 0:
                coords_list.append(coords)
                atom_type_list.extend([aname] * coords.shape[0])
                molecule_type_list.extend(['protein'] * coords.shape[0])
    # 处理核酸原子
    if include_nucleic and 'nucleic' in atoms_dict:
        for aname, coords in atoms_dict['nucleic'].items():
            if atom_types is not None and aname not in atom_types:
                continue
            if coords.size > 0:
                coords_list.append(coords)
                atom_type_list.extend([aname] * coords.shape[0])
                molecule_type_list.extend(['nucleic'] * coords.shape[0])
    if not coords_list:
        print("[警告] 未找到任何蛋白质/核酸原子")
        return None
    # np.ndarray, (N, 3), 所有原子坐标
    all_coords = np.vstack(coords_list)
    # int, 原子总数
    num_atoms = all_coords.shape[0]
    print(f"[INFO] 共 {num_atoms} 个原子")
    



    # ============================================================
    # 3. 计算距离并分配实例 / Compute distances and assign instances
    # ============================================================
    # np.ndarray, (N, num_ligands), 距离矩阵
    # list[int], 配体 ID 列表---[注意本脚本生成的ligand_ids就是连续单增数组:[0,1,2...], 但这么写是为了兼容性, 即允许ligand_ids乱序]
    distances, ligand_ids = compute_atom_ligand_distances(all_coords, ligand_dict)
    # np.ndarray, (N,), 每个原子最近配体的局部索引
    nearest_lig_idx = np.argmin(distances, axis=1)
    # np.ndarray, (N,), 每个原子最近配体的全局索引(global_id)
    nearest_instance = np.array([ligand_ids[idx] for idx in nearest_lig_idx])

    # np.ndarray, (N,), 每个原子到最近配体的距离
    nearest_distance = np.min(distances, axis=1)
    # np.ndarray, (N,), 实例 ID（0 = 背景）, 如果距离 <= threshold，实例 ID = ligand_global_id + 1
    instance_ids = np.zeros(num_atoms, dtype=np.int32)
    for i in range(num_atoms):
        if nearest_distance[i] <= threshold:
            instance_ids[i] = nearest_instance[i] + 1  # +1 使得 0 保留给背景
    
    # 统计 #TODO: 最终, 我们要统计"平均每个PDB的实例数的均值、方差"、"平均每个口袋所含原子数目的均值、方差"
    unique_instances = np.unique(instance_ids)
    num_pocket_atoms = np.sum(instance_ids > 0)
    print(f"[INFO] 口袋原子数: {num_pocket_atoms} / {num_atoms} ({100*num_pocket_atoms/num_atoms:.2f}%)")
    print(f"[INFO] 实例数: {len(unique_instances) - 1} (不含背景)")  # -1 排除背景
    for inst_id in unique_instances:
        if inst_id == 0:
            continue
        count = np.sum(instance_ids == inst_id)
        lig_gid = inst_id - 1
        lig_info = ligand_dict.get(lig_gid, {})
        lig_resname = lig_info.get('resname', 'UNK')
        print(f"  - Instance {inst_id} (Ligand {lig_gid}, {lig_resname}): {count} atoms")
    



    # ============================================================
    # 4. 构建结果字典 / Build result dict
    # ============================================================
    # dict, 实例 ID 到配体 global_id 的映射
    ligand_mapping = {inst_id: inst_id - 1 for inst_id in unique_instances if inst_id >= 1}
    result = {
        'coords': all_coords.astype(np.float32),  # (N, 3), N为原子数
        'ligand_ids': np.array(ligand_ids),  # (num_ligands,), 配体的全局 ID, 就是[0,1,2,..]它是恒等映射(即配体的局部id保持为全局id不变; 写上以保持接口灵活罢了)
        'instance_ids': instance_ids,  # (N,), 0代表背景, 实例 ID = ligand_global_id + 1
        'distances': distances.astype(np.float32),  # (N, num_ligands), 原子i到配体j的距离, 配体用局部id(配体j的全局ID为 ligand_ids[j])
        'nearest_instance': nearest_instance,  # (N,), 配体用全局id
        'nearest_distance': nearest_distance.astype(np.float32),  # (N,)
        'ligand_mapping': ligand_mapping,
        'num_instances': len(unique_instances) - 1,  # 不含背景
        'atom_types': np.array(atom_type_list),  # (N,)
        'molecule_types': np.array(molecule_type_list),  # (N,)
        'threshold': threshold,
    }
    


    # ============================================================
    # 5. 保存结果 / Save results
    # ============================================================
    if output_path is not None:
        np.savez_compressed(
            output_path,
            **{k: v for k, v in result.items() if isinstance(v, np.ndarray)},
            ligand_mapping=np.array(list(ligand_mapping.items())),
            num_instances=result['num_instances'],
            threshold=threshold,
        )
        print(f"[INFO] 实例标签已保存: {output_path}")
    print(f"{'='*60}\n")
    
    return result





def create_instance_masks_for_voxels(
    instance_labels: dict,
    grid_shape: tuple,
    origin: np.ndarray,
    voxel_size: float,
) -> dict:
    """
    将原子级实例标签转换为体素级掩码（用于 Mask3D 训练）
    Convert atom-level instance labels to voxel-level masks (for Mask3D training)
    
    输入参数 / Input:
        - instance_labels: dict, create_instance_labels 的返回值
        - grid_shape: tuple, (D, H, W), 体素网格形状
        - origin: np.ndarray, (3,), 网格原点坐标
        - voxel_size: float, 体素大小（Å）
    
    输出 / Output:
        返回字典包含 / Returns dict containing:
        - 'masks': np.ndarray, (num_instances, D*H*W), 每个实例的二值掩码
        - 'labels': np.ndarray, (num_instances,), 每个实例的类别标签（事实上如果有多种实例, 例如"一般的pocket", "ATP的pocket", 那么就分别标注为0, 1。但目前这里都是 0 = pocket[前景只有一类]）
        - 'instance_to_ligand': dict, 实例索引到配体 global_id 的映射(i -> instance_id-1)
    """
    # np.ndarray, (N, 3), 原子坐标
    coords = instance_labels['coords']
    # np.ndarray, (N,), 实例 ID
    instance_ids = instance_labels['instance_ids']
    # int, 实例数量
    num_instances = instance_labels['num_instances']

    # 转换坐标到体素索引
    # np.ndarray, (N, 3), 体素索引按照 (z, y, x) 的顺序
    voxel_indices = atom2map(coords, origin, voxel_size)
    # 过滤越界索引
    valid_mask = np.all((voxel_indices >= 0) & (voxel_indices < np.array(grid_shape)), axis=1)
    voxel_indices = voxel_indices[valid_mask]   # (N_valid, 3)
    instance_ids_valid = instance_ids[valid_mask]   # (N_valid,)
    

    # list[set], 所有不为背景0的索引,如[3,4,7,..]
    unique_instances = sorted([i for i in np.unique(instance_ids_valid) if i > 0])
    total_voxels = np.prod(grid_shape)
    masks = np.zeros((len(unique_instances), total_voxels), dtype=np.bool_)   # np.ndarray, (num_instances, D*H*W), 掩码张量
    labels = np.zeros(len(unique_instances), dtype=np.int64)  # 都是 pocket 类
    
    for mask_idx, inst_id in enumerate(unique_instances):
        # 找到属于该实例的原子
        atom_mask = instance_ids_valid == inst_id  # (N_valid,)
        inst_voxels = voxel_indices[atom_mask]   # (N_inst_id, 3), N_inst_id 为属于实例inst_id的原子数
        
        # 转换为扁平(一维)索引, (N_inst_id,)
        flat_indices = (
            inst_voxels[:, 0] * grid_shape[1] * grid_shape[2] +
            inst_voxels[:, 1] * grid_shape[2] +
            inst_voxels[:, 2]
        )
        # 设置掩码
        masks[mask_idx, flat_indices] = True
        labels[mask_idx] = 0  # pocket 类
    
    # 实例索引到配体 global_id 的映射
    instance_to_ligand = {i: inst_id - 1 for i, inst_id in enumerate(unique_instances)}
    
    return {
        'masks': masks,
        'labels': labels,
        'instance_to_ligand': instance_to_ligand,
        'unique_instances': unique_instances,
    }



