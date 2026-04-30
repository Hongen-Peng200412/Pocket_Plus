# -*- coding: utf-8 -*-
"""
Part 2 操作（标签计算）:
  - 接受 PDB_processor\ligand_candidates.py 产生的 List[LigandCandidate]
  - 经 PDB_processor\ligand_candidates.py 的 compute_contact_attributes() 填充接触属性后,  由 labels\ligand_filter.py 的 filter_and_classify() 筛选
  - 最后对受体原子打标签: 用 candidate_id 作为实例 ID，输出多类别的口袋实例分割标签 + 背景

输出标签语义：
  逐原子字段:
    - instance_ids[i] = candidate_id  → 原子 i 属于距离最近的配体口袋 (独占分配)
    - instance_ids[i] = -1            → 原子 i 是背景（不属于任何口袋）
    - pocket_class_ids[i] = class_id  → 原子 i 所属口袋的类别 (0=背景, 基于最近配体独占分配)
  逐配体字段:
    - ligand_class_ids[j] = class_id              → 第 j 个配体的口袋类别 ID (与 ligand_candidate_ids 按同序对齐, 由 filter_and_classify 直接产出)
    - pocket_atom_indices_{candidate_id}  (K_i,)  → 该配体阈值内的所有结合原子索引 (不受独占约束, 同一原子可出现在多个配体中)
"""
 
import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.spatial.distance import cdist
import sys
from pathlib import Path

# 绝对导入（labels/ 是 Make_Data/ 下的顶层包）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PDB_processor.parser import ParsedStructure
from PDB_processor.ligand_candidates import LigandCandidate
from PDB_processor.error_logger import return_error_info, ErrorType


def compute_binding_labels(
    parsed_data: ParsedStructure,
    selected_candidates: List[LigandCandidate],
    pocket_class_map: Dict[int, Tuple[int, str, float]],
    error_dir: Optional[str] = None,
    sample_id: Optional[str] = None,
    require_binding_site: bool = True,
) -> Optional[Dict]:
    """
    计算多类别实例分割标签。

    输入参数 / Input:
        - parsed_data: ParsedStructure, PDB_processor\parser.py 产生, 解析后的结构数据（提供受体原子坐标 atom_coords）
        - selected_candidates: list[LigandCandidate], 经 filter_and_classify 筛选后的候选列表
        - pocket_class_map: dict[int, tuple[int, str, float]], candidate_id → (class_id, class_name, binding_threshold)
          由 filter_and_classify() 返回
        - error_dir: str 或 None, 错误日志目录
        - sample_id: str 或 None, 样本 ID（仅用于错误日志）
        - require_binding_site: bool, 若为 True 且无结合位点则返回 None

    输出 / Output:
        - dict 或 None, 包含以下键:
            - instance_ids:        np.ndarray, (N_atoms,), int32, 每个原子的实例 ID (= 最近配体的 candidate_id; 背景为 -1; 独占分配)
            - ligand_ids:          np.ndarray, (N_atoms,), int32, 每个原子最近配体的 candidate_id（即使距离很远）
            - distances:           np.ndarray, (N_atoms,), float32, 每个原子到最近配体原子的距离
            - binding_mask:        np.ndarray, (N_atoms,), bool, 任一配体阈值内则为 True (各配体独立阈值的并集)
            - pocket_class_ids:    np.ndarray, (N_atoms,), int32, 口袋类别 ID (0=背景, ≥1=各类口袋; 基于最近配体独占分配)
            - pocket_centers:      np.ndarray, (N_ligands, 3), float32, 每个配体对应口袋的几何中心（按 candidate_id 升序; 基于该配体完整结合原子集合计算）
            - pocket_atom_indices: dict[int, np.ndarray], candidate_id → (K_i,) int32, 每个配体阈值内的所有受体原子索引 (不受独占约束)
    """
    # int, 原子数量
    n_atoms = len(parsed_data.atom_coords)
    # int, 筛选后的配体数量
    n_ligands = len(selected_candidates)

    if n_atoms == 0:
        return None

    if n_ligands == 0:
        # 严格模式：无配体时计入错误日志 + 删掉已有的样本文件夹
        if require_binding_site:
            if error_dir is not None:
                return_error_info(
                    file_path="",
                    line=-1,
                    error_type=ErrorType.NO_BINDING_SITE,
                    error_detail="No selected ligands after filtering.",
                    output_dir=error_dir,
                    sample_id=sample_id,
                )
            return None
        # 非严格模式：无配体时返回全背景(-1)标签
        return {
            'instance_ids':        np.full(n_atoms, -1, dtype=np.int32),
            'ligand_ids':          np.full(n_atoms, -1, dtype=np.int32),
            'distances':           np.full(n_atoms, np.inf, dtype=np.float32),
            'binding_mask':        np.zeros(n_atoms, dtype=bool),
            'pocket_class_ids':    np.zeros(n_atoms, dtype=np.int32),
            'pocket_centers':      np.zeros((0, 3), dtype=np.float32),
            'pocket_atom_indices': {},
        }

    # =========================================================================
    # 计算原子到每个配体的距离 / Compute distances to each ligand
    # =========================================================================
    # np.ndarray, (N_atoms, 3), float32, 蛋白/核酸原子坐标
    atom_coords = parsed_data.atom_coords
    # np.ndarray, (N_atoms,), float32, 到最近配体原子的距离（初始化为无穷大）
    min_distances = np.full(n_atoms, np.inf, dtype=np.float32)
    # np.ndarray, (N_atoms,), int32, 最近配体的 candidate_id（初始化为 -1）
    closest_candidate_ids = np.full(n_atoms, -1, dtype=np.int32)
    # dict[int, np.ndarray], candidate_id → 配体重心 (3,)
    ligand_centers = {}
    # dict[int, np.ndarray], candidate_id → 该配体阈值内的所有受体原子索引 (K_i,) int32
    pocket_atom_indices = {}

    for candidate in selected_candidates:
        # int, 候选配体的全局 ID
        cand_id = candidate.candidate_id
        # np.ndarray, (M, 3), float32, 配体重原子坐标
        lig_coords = candidate.coords
        # np.ndarray, (3,), float32, 配体重心
        lig_center = candidate.center
        ligand_centers[cand_id] = lig_center

        # np.ndarray, (N_atoms, M), float32, 原子到配体各原子的距离矩阵
        dist_matrix = cdist(atom_coords, lig_coords, metric='euclidean')
        # np.ndarray, (N_atoms,), float32, 到该配体最近原子的距离
        min_dist_to_this = np.min(dist_matrix, axis=1)

        # ---- 逐配体独立收集: 该配体阈值内的所有受体原子索引 ----
        # float, 该配体的结合距离阈值
        threshold = pocket_class_map[cand_id][2]
        # np.ndarray, (K_i,), int32, 距该配体 ≤ threshold 的所有受体原子索引 (不受独占约束)
        pocket_atom_indices[cand_id] = np.where(min_dist_to_this <= threshold)[0].astype(np.int32)

        # ---- 更新全局最近配体 (独占分配逻辑, 用于 instance_ids / pocket_class_ids) ----
        # np.ndarray, (N_atoms,), bool, 该配体比当前记录更近的原子掩码
        closer_mask = min_dist_to_this < min_distances
        min_distances[closer_mask] = min_dist_to_this[closer_mask]
        closest_candidate_ids[closer_mask] = cand_id



    # =========================================================================
    # 计算结合位点掩码和实例 ID / Compute binding mask and instance IDs
    # =========================================================================
    # np.ndarray, (N_atoms,), bool, 结合位点掩码 (各配体独立阈值的并集)
    binding_mask = np.zeros(n_atoms, dtype=bool)
    
    # 用逐配体的完整结合原子集合构建并集
    for cand_id, indices in pocket_atom_indices.items():
        binding_mask[indices] = True

    if require_binding_site and not np.any(binding_mask):
        if error_dir is not None:
            return_error_info("", -1, ErrorType.NO_BINDING_SITE,
                              f"No atoms within binding thresholds of any ligand",
                              error_dir, sample_id)
        return None

    # np.ndarray, (N_atoms,), int32, 实例 ID(最近配体的 candidate_id；背景 = -1)
    instance_ids = np.full(n_atoms, -1, dtype=np.int32)
    instance_ids[binding_mask] = closest_candidate_ids[binding_mask]



    # =========================================================================
    # 计算口袋类别 ID / Compute pocket class IDs
    # =========================================================================
    # np.ndarray, (N_atoms,), int32, 口袋类别 (0=背景)
    pocket_class_ids = np.zeros(n_atoms, dtype=np.int32)
    for atom_idx in np.where(binding_mask)[0]:
        # int, 该原子最近配体的 candidate_id
        cand_id = int(closest_candidate_ids[atom_idx])
        if cand_id in pocket_class_map:
            # int, 口袋类别 ID
            pocket_class_ids[atom_idx] = pocket_class_map[cand_id][0]



    # =========================================================================
    # 计算口袋中心 / Compute pocket centers
    # (基于每个配体的完整结合原子集合, 而非独占分配的 instance_ids)
    # =========================================================================
    # list[np.ndarray], 每个配体对应口袋的中心坐标（按 candidate_id 升序）
    pocket_centers = []
    for candidate in sorted(selected_candidates, key=lambda c: c.candidate_id):
        cand_id = candidate.candidate_id
        # np.ndarray, (K_i,), int32, 该配体的完整结合原子索引
        indices = pocket_atom_indices[cand_id]
        if len(indices) > 0:
            # np.ndarray, (3,), float32, 口袋结合原子的几何中心
            pocket_center = np.mean(atom_coords[indices], axis=0)
        else:
            # 回退到配体重心（无结合原子时）
            pocket_center = ligand_centers[cand_id]
        pocket_centers.append(pocket_center)
    # np.ndarray, (N_ligands, 3), float32, 口袋中心坐标
    pocket_centers_arr = (np.array(pocket_centers, dtype=np.float32)
                          if pocket_centers else np.zeros((0, 3), dtype=np.float32))

    return {
        'instance_ids':        instance_ids,
        'ligand_ids':          closest_candidate_ids,
        'distances':           min_distances,
        'binding_mask':        binding_mask,
        'pocket_class_ids':    pocket_class_ids,
        'pocket_centers':      pocket_centers_arr,
        'pocket_atom_indices': pocket_atom_indices,
    }



def save_labels_npz(
    parsed_data: ParsedStructure,
    binding_labels: Dict,
    selected_candidates: List[LigandCandidate],
    pocket_class_names: Dict[int, str],
    pocket_class_map: Dict[int, Tuple[int, str, float]],
    output_path: str,
) -> None:
    """
    保存标签数据到 .npz 文件。

    输入参数 / Input:
        - parsed_data: ParsedStructure, 解析后的结构数据（当前仅用于类型一致性）
        - binding_labels: dict, compute_binding_labels() 的返回值
        - selected_candidates: list[LigandCandidate], 筛选后的候选列表
        - pocket_class_names: dict[int, str], 口袋类别 ID → 名称映射（由 get_pocket_class_name_map() 生成，总是包含 0='background'）
        - pocket_class_map: dict[int, tuple[int, str, float]], candidate_id → (class_id, class_name, binding_threshold), 由 filter_and_classify() 返回
        - output_path: str, 输出 .npz 文件路径

    输出文件内容 / Output file contents:
        - instance_ids:         np.ndarray, (N_atoms,), int32,  实例 ID（= 最近配体的 candidate_id；背景为 -1；独占分配）
        - ligand_ids:           np.ndarray, (N_atoms,), int32,  最近配体的 candidate_id（即使距离很远）
        - distances:            np.ndarray, (N_atoms,), float32, 到最近配体原子的距离
        - binding_mask:         np.ndarray, (N_atoms,), bool, 结合位点掩码 (各配体独立阈值的并集)
        - pocket_class_ids:     np.ndarray, (N_atoms,), int32, 口袋类别 ID（0=背景；基于最近配体独占分配）

        - num_ligands:          int, 筛选后的配体数量
        - pocket_centers:       np.ndarray, (N_ligands, 3), float32, 口袋几何中心（按 candidate_id 升序；基于该配体完整结合原子集合计算）
        - ligand_resnames:      np.ndarray, (N_ligands,), object/str, 配体残基名（按 candidate_id 升序）
        - ligand_candidate_ids: np.ndarray, (N_ligands,), int32, 配体的 candidate_id（按升序，与 ligand_resnames 对齐）
        - ligand_class_ids:     np.ndarray, (N_ligands,), int32, 每个配体的口袋类别 ID（与 ligand_candidate_ids 按同序对齐, 由 filter_and_classify() 直接产出）
        - ligand_coords_{id}:   np.ndarray, (M, 3), float32, 第 id 个候选配体的原子坐标（id = candidate_id）
        - pocket_atom_indices_{id}: np.ndarray, (K_id,), int32, 第 id 个配体阈值内的所有结合原子全局索引 (不受独占约束; 同一原子可出现在多个配体中)
        - pocket_class_name_map: np.ndarray, object/str, 类别映射字符串 "0:background,1:druggable,..."
    """
    # list[LigandCandidate], 按 candidate_id 升序排列
    sorted_candidates = sorted(selected_candidates, key=lambda c: c.candidate_id)
    # int, 配体数量
    n_ligands = len(sorted_candidates)

    # 基础标签
    save_dict = {
        'instance_ids':     binding_labels['instance_ids'],
        'ligand_ids':       binding_labels['ligand_ids'],
        'distances':        binding_labels['distances'],
        'binding_mask':     binding_labels['binding_mask'],
        'pocket_class_ids': binding_labels['pocket_class_ids'],
        'num_ligands':      np.int32(n_ligands),
        'pocket_centers':   binding_labels['pocket_centers'],
    }

    # 配体信息（按 candidate_id 升序）
    # list[str], 配体残基名列表
    ligand_resnames = []
    # list[int], 配体 candidate_id 列表
    ligand_candidate_ids = []
    # list[int], 每个配体的口袋类别 ID 列表（由 filter_and_classify 产出，直接持久化）
    ligand_class_ids = []
    # dict[int, np.ndarray], candidate_id → 该配体阈值内的所有受体原子索引
    pocket_atom_indices = binding_labels['pocket_atom_indices']
    for candidate in sorted_candidates:
        ligand_resnames.append(candidate.resname)
        ligand_candidate_ids.append(candidate.candidate_id)
        # int, 该配体的口袋类别 ID (从 pocket_class_map 直接获取)
        ligand_class_ids.append(pocket_class_map[candidate.candidate_id][0])
        # np.ndarray, (M, 3), float32, 该配体的原子坐标
        save_dict[f'ligand_coords_{candidate.candidate_id}'] = candidate.coords
        # np.ndarray, (K_i,), int32, 该配体口袋的所有结合原子全局索引 (不受独占约束)
        save_dict[f'pocket_atom_indices_{candidate.candidate_id}'] = pocket_atom_indices[candidate.candidate_id]

    save_dict['ligand_resnames'] = np.array(ligand_resnames, dtype=object)
    save_dict['ligand_candidate_ids'] = np.array(ligand_candidate_ids, dtype=np.int32)
    # np.ndarray, (N_ligands,), int32, 每个配体的口袋类别 ID（与 ligand_candidate_ids 对齐）
    save_dict['ligand_class_ids'] = np.array(ligand_class_ids, dtype=np.int32)

    # 口袋类别名称映射（格式: "0:background,1:druggable,2:metal_ion"）
    # str, 类别映射字符串; ','.join(...)：将上述生成的所有小字符串，用逗号 , 连接起来
    class_str = ','.join(f'{k}:{v}' for k, v in sorted(pocket_class_names.items()))  # 字符串
    save_dict['pocket_class_name_map'] = np.array(class_str, dtype=object)

    np.savez_compressed(output_path, **save_dict)
