"""
================================================================================
统一预处理系统 - 实例分割标签 / Unified Preprocessing System - Instance Labels
================================================================================

计算配体结合相关标签 (复用 Make_Primary_Label.py 逻辑):
- 实例 ID (每个配体对应一个 ID)
- 到最近配体的距离
- 结合位点掩码

Compute ligand binding-related labels.
================================================================================
"""

import numpy as np
from typing import Tuple, Dict, Optional
from scipy.spatial.distance import cdist

from ..config import BINDING_THRESHOLD
from ..parser import ParsedStructure
from ..error_logger import return_error_info, ErrorType


def compute_binding_labels(
    parsed_data: ParsedStructure,
    binding_threshold: float = BINDING_THRESHOLD,
    output_dir: Optional[str] = None,
    sample_id: Optional[str] = None,
    require_binding_site: bool = True
) -> Optional[Dict]:
    """
    计算结合位点标签
    Compute binding site labels
    
    输入参数 / Input:
        - parsed_data: ParsedStructure, 解析后的结构数据
        - binding_threshold: float, 结合位点距离阈值 (默认 4.5Å)
        - output_dir: str, 输出目录 (仅用于错误日志)
        - sample_id: str, 样本ID
        - require_binding_site: bool, 是否要求结合位点存在
    
    输出 / Output:
        - dict 或 None, 包含以下键:
            - instance_ids: np.ndarray, (N_atoms,), int32, 每个原子的实例ID (-1 表示非结合位点)
            - ligand_ids: np.ndarray, (N_atoms,), int32, 最近配体ID
            - distances: np.ndarray, (N_atoms,), float32, 到最近配体的距离
            - binding_mask: np.ndarray, (N_atoms,), bool, 结合位点掩码
            - ligand_coords: dict, 配体坐标字典, 就是 parsed_data.ligand_dict
            - pocket_centers: np.ndarray, (N_ligands, 3), float32, 口袋中心坐标
    """
    # int, 原子数量
    n_atoms = len(parsed_data.atom_coords)
    # int, 配体数量
    n_ligands = parsed_data.num_ligands
    if n_atoms == 0:
        return None
    if n_ligands == 0:
        # 无配体: 返回全 -1 标签
        return {
            'instance_ids': np.full(n_atoms, -1, dtype=np.int32),
            'ligand_ids': np.full(n_atoms, -1, dtype=np.int32),
            'distances': np.full(n_atoms, np.inf, dtype=np.float32),
            'binding_mask': np.zeros(n_atoms, dtype=bool),
            'ligand_coords': {},
            'pocket_centers': np.zeros((0, 3), dtype=np.float32),
        }
    

    # =========================================================================
    # 计算原子到每个配体的距离 / Compute distances to each ligand
    # =========================================================================
    # np.ndarray, (N_atoms, 3), float32
    atom_coords = parsed_data.atom_coords
    # np.ndarray, (N_atoms,), float32, 到最近配体原子的距离
    min_distances = np.full(n_atoms, np.inf, dtype=np.float32)
    # np.ndarray, (N_atoms,), int32, 最近配体的全局 ID
    closest_ligand_ids = np.full(n_atoms, -1, dtype=np.int32)
    # dict[int, np.ndarray], 配体坐标 (重心), 如果 require_binding_site 为 False 且没有结合原子则用配体中心作为口袋中心
    ligand_centers = {}
    
    for lig_global_id, lig_info in parsed_data.ligand_dict.items():
        # np.ndarray, (N_lig_atoms, 3), float32, 配体原子坐标
        lig_coords = lig_info['coords']
        # np.ndarray, (3,), float32, 配体重心
        lig_center = np.mean(lig_coords, axis=0)
        ligand_centers[lig_global_id] = lig_center
        
        # np.ndarray, (N_atoms, N_lig_atoms), float32, 原子到配体原子的距离
        dist_to_lig = cdist(atom_coords, lig_coords, metric='euclidean')
        # np.ndarray, (N_atoms,), float32, 到该配体最近原子的距离
        min_dist_to_this_lig = np.min(dist_to_lig, axis=1)
        # 更新最近配体
        closer_mask = min_dist_to_this_lig < min_distances
        min_distances[closer_mask] = min_dist_to_this_lig[closer_mask]
        closest_ligand_ids[closer_mask] = lig_global_id
    

    # =========================================================================
    # 计算结合位点掩码和实例 ID / Compute binding mask and instance IDs
    # =========================================================================
    # np.ndarray, (N_atoms,), bool, 结合位点掩码 (距离 < 阈值)
    binding_mask = min_distances <= binding_threshold
    # 检查是否有结合位点
    if require_binding_site and not np.any(binding_mask):
        if output_dir is not None:
            return_error_info("", -1, ErrorType.NO_BINDING_SITE,
                             f"No atoms within {binding_threshold}Å of any ligand",
                             output_dir, sample_id)
        return None
    # np.ndarray, (N_atoms,), int32, 实例 ID (非结合位点为 -1)
    instance_ids = np.full(n_atoms, -1, dtype=np.int32)
    instance_ids[binding_mask] = closest_ligand_ids[binding_mask]
    

    # =========================================================================
    # 计算口袋中心 / Compute pocket centers
    # =========================================================================
    # list[np.ndarray], 每个配体对应口袋的原子坐标
    pocket_centers = []
    for lig_global_id in sorted(parsed_data.ligand_dict.keys()):
        # 找出属于该配体的口袋原子
        pocket_mask = instance_ids == lig_global_id
        if np.any(pocket_mask):
            pocket_center = np.mean(atom_coords[pocket_mask], axis=0)
        else:
            # 该配体无口袋原子, 回退到配体重心 / No pocket atoms, fall back to ligand center
            pocket_center = ligand_centers[lig_global_id]
        pocket_centers.append(pocket_center)
    # np.ndarray, (N_ligands, 3), float32, 每个配体对应的口袋中心坐标
    pocket_centers = np.array(pocket_centers, dtype=np.float32) if pocket_centers else np.zeros((0, 3), dtype=np.float32)
    
    return {
        'instance_ids': instance_ids,
        'ligand_ids': closest_ligand_ids,
        'distances': min_distances,
        'binding_mask': binding_mask,
        'ligand_coords': parsed_data.ligand_dict,
        'pocket_centers': pocket_centers,
    }





def save_labels_npz(
    parsed_data: ParsedStructure,
    binding_labels: Dict,
    output_path: str
) -> None:
    """
    保存标签数据到 .npz 文件
    Save label data to .npz file
    
    输入参数 / Input:
        - parsed_data: ParsedStructure, 解析后的结构数据
        - binding_labels: dict, 结合位点标签
        - output_path: str, 输出文件路径
    
    输出文件内容 / Output file contents:
        - instance_ids: np.ndarray, (N_atoms,), int32, 实例ID(-1 表示非结合位点)
        - ligand_ids: np.ndarray, (N_atoms,), int32, 最近配体ID
        - distances: np.ndarray, (N_atoms,), float32, 到最近配体距离
        - binding_mask: np.ndarray, (N_atoms,), bool, 结合位点掩码
        - num_ligands: int, 配体数量
        - pocket_centers: np.ndarray, (N_ligands, 3), float32, 口袋中心
        - ligand_resnames: list[str], 配体残基名称
        - ligand_coords_*: (有num_ligands这样的键)np.ndarray, 每个配体的原子坐标
    """
    # 基础标签
    save_dict = {
        'instance_ids': binding_labels['instance_ids'],
        'ligand_ids': binding_labels['ligand_ids'],
        'distances': binding_labels['distances'],
        'binding_mask': binding_labels['binding_mask'],
        'num_ligands': parsed_data.num_ligands,
        'pocket_centers': binding_labels['pocket_centers'],
    }
    
    # 配体信息
    ligand_resnames = []
    for lig_id, lig_info in parsed_data.ligand_dict.items():
        ligand_resnames.append(lig_info['resname'])
        save_dict[f'ligand_coords_{lig_id}'] = lig_info['coords']
    
    save_dict['ligand_resnames'] = np.array(ligand_resnames, dtype=object)
    
    np.savez_compressed(output_path, **save_dict)
