import os
import numpy as np
import json
import sys
import argparse
import traceback
from joblib import Parallel, delayed
from pathlib import Path
from scipy.spatial import cKDTree

# 将 Make_Data/ 加入 sys.path, 使得 PDB_processor 和本目录下的 utils 可被导入
_BINDER_DIR = Path(__file__).resolve().parent
_POCKET_ROOT = _BINDER_DIR.parent
_PROJECT_ROOT = _POCKET_ROOT.parent

for _p in (str(_PROJECT_ROOT), str(_POCKET_ROOT), str(_BINDER_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Make_Data.process_and_label import get_features_when_infer
from Pocket_Plus.utils.mrc_tools import load_map, make_model_grid
from Pocket_Plus.utils.network_tools import atomic_np_savez


def bind_AtomsLabel_to_EMDB(
    sample_folder_path: str = None,
    emdb_path: str = None, 
    target_voxel_size: float = 1.0,

    output_path: str = None,
    overwrite_existing: bool = False,
):
    """
    将 PDB 文件中的原子标签 (atoms.npz 里面的 'features') 映射到对应的 EMDB 文件, 产生多通道特征张量.
    
    输入参数 / Args:
        - sample_folder_path: str, 由 Pocket_Plus/Make_Data/PDB_processor/run_preprocess.py 产生的对应样本文件夹路径 (里面有四个 .npz). 
                              注意将会读取 sample_folder_path/labels.npz 这个文件.
        - emdb_path: str, EMDB 文件路径.
        - target_voxel_size: float, 重采样后的体素大小 (Å). 若为 None 则保持原有分辨率.
        - output_path: str, 保存路径 (.npz).
        - overwrite_existing: bool, 是否覆盖已有文件.
    
    输出 / Return:
        - label_np: np.ndarray, (1, D, H, W), float32. 
                    即, 创建一个与 EMDB 文件有相同空间维度的张量, 将 PDB 的原子标签填充到对应的位置.
                    如果一个体素处存在一个原子, 它是属于类别 k, 就在 label_np[0, z, y, x] 处标注成 k; 是背景则为0.(注意虽然有极小概率使一个原子同时是2个ligand的口袋, 但由于在数据处理部分'强制'指定了'每个原子的口袋类别只由最近配体决定', 所以不必担心一个原子有多类冲突标签的情况)
    """
    # ---- 读取数据 / Load data ----
    labels_npz = os.path.join(sample_folder_path, 'labels.npz')   # str, 标签文件路径
    atoms_npz  = os.path.join(sample_folder_path, 'atoms.npz')    # str, 原子特征文件路径（需要原子坐标）
    labels_data = np.load(labels_npz)
    
    # np.ndarray, (N_atom,), int32, 每个原子的口袋类别 ID (0=背景, 1,2...=口袋)
    pocket_class_ids = labels_data['pocket_class_ids']
    # np.ndarray, (N_atom, 3), float32, 原子坐标 (X, Y, Z)
    atom_pos_array = np.load(atoms_npz)['coords'] 

    # ---- 加载并可选重采样 EMDB 密度图 / Load and optionally resample EMDB map ----
    map_data, voxel_size, origin = load_map(emdb_path)
    if target_voxel_size is not None:
        map_data, voxel_size, origin = make_model_grid(map_data, voxel_size, origin, target_voxel_size)
    
    # np.ndarray, (1, D, H, W), float32, 单通道语义分割图 (0=背景, k=类别)
    label_np = np.zeros((1, map_data.shape[0], map_data.shape[1], map_data.shape[2]), dtype=np.float32)

    # ---- 世界坐标 → 体素索引 / World coords → voxel indices ----
    # np.ndarray, (N_valid,), bool, 有效原子掩码(非背景)
    valid_mask = pocket_class_ids > 0
    if np.any(valid_mask):
        # np.ndarray, (N_valid, 3), float32, 有效原子的坐标
        valid_coords = atom_pos_array[valid_mask]
        # np.ndarray, (N_valid,), int32, 有效原子的类别 ID
        valid_class_ids = pocket_class_ids[valid_mask]
        
        # --- 固定操作---
        voxel_ijk = np.floor(((valid_coords - origin) / voxel_size)).astype(int)
        x_idx = np.clip(voxel_ijk[:, 0], 0, label_np.shape[3] - 1)
        y_idx = np.clip(voxel_ijk[:, 1], 0, label_np.shape[2] - 1)
        z_idx = np.clip(voxel_ijk[:, 2], 0, label_np.shape[1] - 1)
        
        indices = (np.zeros_like(z_idx), z_idx, y_idx, x_idx)
        # 使用 np.maximum.at 确保如果发生冲突(多原子同一体素), 保留较大的类别 ID
        np.maximum.at(label_np, indices, valid_class_ids.astype(np.float32))

    if output_path is not None:
        atomic_np_savez(output_path, do_not_replace=(not overwrite_existing), grid=label_np, voxel_size=voxel_size, origin=origin)

    return label_np


def bind_LigandMinDist_to_EMDB(
    sample_folder_path: str,
    emdb_path: str,
    target_voxel_size: float,
    output_path: str,
    overwrite_existing: bool,
    num_pocket_classes: int,
):
    """
    为每类 ligand 分别计算体素中心到最近同类 ligand 原子的欧氏距离。

    输入参数:
        - sample_folder_path: str, 含 labels.npz 的样本目录
        - emdb_path: str, EMDB .map 文件路径
        - target_voxel_size: float, 标量, 重采样体素大小 (Å)
        - output_path: str, 保存路径 (.npz)
        - overwrite_existing: bool, 是否覆盖已有文件
        - num_pocket_classes: int, 标量, 口袋类别数 (不含背景)

    输出 (保存到 .npz):
        - grid: np.ndarray, (num_pocket_classes, D, H, W), float32
          通道 i = 第 (i+1) 类 ligand 的距离图; 若某类无 ligand 原子则填 np.inf
    """
    if output_path is not None and os.path.exists(output_path) and not overwrite_existing:
        return

    # ---- 读取 labels.npz ----
    labels_npz = os.path.join(sample_folder_path, 'labels.npz')
    labels_data = np.load(labels_npz, allow_pickle=True)
    # np.ndarray, (N_ligands,), int32, 每个配体的 candidate_id
    ligand_candidate_ids = labels_data['ligand_candidate_ids']
    # np.ndarray, (N_ligands,), int32, 每个配体的口袋类别 ID (与 ligand_candidate_ids 对齐)
    ligand_class_ids = labels_data['ligand_class_ids']


    # ---- 按 class_id 分组 ligand 坐标 ----
    # dict[int, list[np.ndarray]], class_id -> [各配体坐标数组]
    class_ligand_coords: dict[int, list[np.ndarray]] = {}
    for cand_id, cls_id in zip(ligand_candidate_ids, ligand_class_ids):
        cls_id_int = int(cls_id)
        if cls_id_int == 0:
            continue
        key = f'ligand_coords_{int(cand_id)}'
        if key in labels_data:
            class_ligand_coords.setdefault(cls_id_int, []).append(labels_data[key])
    labels_data.close()


    # ---- 加载 EMDB 并重采样 ----
    map_data, voxel_size, origin = load_map(emdb_path)
    if target_voxel_size is not None:
        map_data, voxel_size, origin = make_model_grid(map_data, voxel_size, origin, target_voxel_size)
    # int, 空间维度
    D, H, W = map_data.shape


    # ---- 构造体素中心坐标网格 ----
    # np.ndarray, (D*H*W, 3), float32, 每个体素的世界坐标中心
    z_idx, y_idx, x_idx = np.mgrid[0:D, 0:H, 0:W]
    voxel_centers = np.stack([
        x_idx.ravel() * voxel_size[0] + origin[0] + voxel_size[0] * 0.5,
        y_idx.ravel() * voxel_size[1] + origin[1] + voxel_size[1] * 0.5,
        z_idx.ravel() * voxel_size[2] + origin[2] + voxel_size[2] * 0.5,
    ], axis=1).astype(np.float32)
    # np.ndarray, (num_pocket_classes, D, H, W), float32, 距离图
    dist_grid = np.full((num_pocket_classes, D, H, W), np.inf, dtype=np.float32)

    for cls_id in range(1, num_pocket_classes + 1):
        # int, 通道索引 = class_id - 1
        ch = cls_id - 1
        if cls_id not in class_ligand_coords:
            continue
        # np.ndarray, (N_lig_class, 3), float32, 该类所有 ligand 原子坐标
        all_coords = np.vstack(class_ligand_coords[cls_id]).astype(np.float32)
        if len(all_coords) == 0:
            continue
        # cKDTree, 构建空间索引
        tree = cKDTree(all_coords)
        # np.ndarray, (D*H*W,), float32, 每个体素到最近该类 ligand 原子的距离
        distances, _ = tree.query(voxel_centers, k=1)
        dist_grid[ch] = distances.reshape(D, H, W).astype(np.float32)

    if output_path is not None:
        atomic_np_savez(output_path, do_not_replace=(not overwrite_existing), grid=dist_grid, voxel_size=voxel_size, origin=origin)


def bind_AtomsFeature_to_EMDB(
    sample_folder_path: str=None,                                                                                # 与下面二选一
    pdb_path: str=None, error_dir: str=None, compute_density: bool=True, select_first_model: bool=False,         # 与上面二选一
    pre_parsed_atom_info: dict=None,  # 可选: 直接传入已解析的原子信息 dict (含 'features' 和 'coords'), 跳过重复解析

    emdb_path: str=None, 
    target_voxel_size: float=1.0,
    output_path: str=None, 
    
    add_when_conflict=True,
    overwrite_existing: bool=False, 
    return_atom_pos_array: bool=False
):
    """
    将PDB文件中的原子特征(atoms.npz里面的'features')映射到对应的EMDB文件, 产生特征张量.这个函数强烈依赖于 Pocket_Plus\Make_Data\PDB_processor\run_preprocess.py 的返回结果.
    
    Args:
        - sample_folder_path: 由 Pocket_Plus\Make_Data\PDB_processor\run_preprocess.py 产生的对应样本文件夹路径(里面有四个.npz)。注意将会读取 sample_folder_path/atoms.npz这个文件
        - pdb_path: PDB文件路径. 训练时读取atoms.npz, 推断时输入pdb, 通过 _Plus\Make_Data\PDB_processor\run_preprocess.py 里面的 def get_features_when_infer 读取与atoms.npz等价的特征.
        - pre_parsed_atom_info: dict | None, 可选. 若提供, 直接使用其中的 'features' 和 'coords', 跳过 get_features_when_infer 调用, 避免重复解析 CIF.

        - emdb_path: EMDB文件路径
        - target_voxel_size: 重采样后的体素大小. 若为None则保持原有分辨率

        - output_path: 保存路径
        - add_when_conflict: 一个体素含有多个原子时, 如果 add_when_conflict=True 那么就把它们的特征向量进行累加, 否则后写覆盖 (last-write-wins, 取决于原子数组顺序)
        - overwrite_existing: 如果为 True 则覆盖已有 .npz 文件, 否则跳过 (默认 False)
    
    Return:
        - (feature_np, atom_pos_array): tuple
            - feature_np: np.ndarray, (C, D, H, W), float32. 即, 创建一个与EMDB文件有相同空间维度的张量, 将PDB的原子特征填充到对应的位置.
            - atom_pos_array: np.ndarray, (N_atom, 3), float32. 所有原子的世界坐标 (x, y, z), 单位 Å.
    """
    if pre_parsed_atom_info is not None:
        # 直接使用调用方已解析的原子信息, 保证坐标与特征严格同源
        atom_features_array = pre_parsed_atom_info['features']
        atom_pos_array = pre_parsed_atom_info['coords']
    elif sample_folder_path is not None:
        atoms_npz = os.path.join(sample_folder_path, 'atoms.npz')
        atom_features_array = np.load(atoms_npz)['features']  # np.ndarray, (N_atom, 49), float32, 残基级特征向量 (类型 One-Hot + 理化性质)
        atom_pos_array = np.load(atoms_npz)['coords']  # (N_atom, 3)
    elif pdb_path is not None:
        atom_info, _, _ = get_features_when_infer(input_path=pdb_path,error_dir=error_dir,compute_density=compute_density,select_first_model=select_first_model)
        atom_features_array = atom_info['features']
        atom_pos_array = atom_info['coords']
    else:
        raise ValueError("Either pre_parsed_atom_info, sample_folder_path, or pdb_path must be provided")
    feature_channel = atom_features_array.shape[1]
    
    map_data, voxel_size, origin = load_map(emdb_path)
    if target_voxel_size is not None:
        map_data, voxel_size, origin = make_model_grid(map_data, voxel_size, origin, target_voxel_size)
    feature_np = np.zeros((feature_channel, map_data.shape[0], map_data.shape[1], map_data.shape[2]), dtype=np.float32)   # (C, D, H, W)

    # ------------------------------------------------------------------
    # 将世界坐标转为体素索引
    # atom_pos_array: (N_atom, 3), 列顺序 (x, y, z); origin / voxel_size: (3,), 同为 (x, y, z) 顺序
    # feature_np 空间轴顺序是 (Z, Y, X)，因此索引需要反转
    # ------------------------------------------------------------------
    voxel_ijk = np.floor(((atom_pos_array - origin) / voxel_size)).astype(int)  # np.ndarray, (N_atom, 3), int, 体素索引 (x, y, z)
    x_idx = np.clip(voxel_ijk[:, 0], 0, feature_np.shape[3] - 1)  # np.ndarray, (N_atom,), int, X 轴索引
    y_idx = np.clip(voxel_ijk[:, 1], 0, feature_np.shape[2] - 1)  # np.ndarray, (N_atom,), int, Y 轴索引
    z_idx = np.clip(voxel_ijk[:, 2], 0, feature_np.shape[1] - 1)  # np.ndarray, (N_atom,), int, Z 轴索引
    # feature_np: (C, D, H, W) 即 (C, Z, Y, X)
    # atom_features_array: (N_atom, C)  →  .T → (C, N_atom)
    # 高级索引: feature_np[:, z, y, x] 取出 (C, N_atom) 个位置并赋值
    if not add_when_conflict:
        feature_np[:, z_idx, y_idx, x_idx] = atom_features_array.T  # (C, N_atom) 后写覆盖 (last-write-wins), 同一体素的多个原子仅保留最后一个
    else:
        np.add.at(feature_np, (slice(None), z_idx, y_idx, x_idx), atom_features_array.T)  # (C, N_atom) 使用 np.add.at 实现无缓冲累加, 正确处理多个原子落入同一体素的情况

    if output_path is not None:
        atomic_np_savez(output_path, do_not_replace=(not overwrite_existing), grid=feature_np, voxel_size=voxel_size, origin=origin)

    return feature_np, atom_pos_array if return_atom_pos_array else feature_np





# ------------------------------------------------------------------ #
#  分片工具 / Sharding utility
# ------------------------------------------------------------------ #
def apply_sharding(item_list, part_id, total_parts):
    """
    将 item_list 按 part_id / total_parts 切片, 返回当前分片.
    """
    if total_parts <= 1:
        return item_list
    n = len(item_list)
    shard_size = n // total_parts
    remainder = n % total_parts
    if part_id < remainder:
        start = part_id * (shard_size + 1)
        end = start + shard_size + 1
    else:
        start = remainder * (shard_size + 1) + (part_id - remainder) * shard_size
        end = start + shard_size
    return item_list[start:end]


# ------------------------------------------------------------------ #
#  单样本处理 / Process a single (emdb, pdb) pair
# ------------------------------------------------------------------ #
def process_single_item(
    item,                                # dict, JSON 中的一个条目, 形如 {"emd_50120": "9E01"}
    emdb_folder_path: str = None,        # EMDB文件夹路径, 目前按照 emd_50120.map 的命名格式
    sim_folder_path: str = None,         # 可选, 模拟密度图文件夹路径, 含 .mrc 文件
    sample_root_path: str = None,        # 根据 Pocket_Plus\Make_Data\PDB_processor\run_preprocess.py 解析后的结果, 一个样本一个文件夹, 含有四个.npz
    
    label_output_path: str = None,       # def bind_AtomsLabel_to_EMDB 返回结果保存的文件夹, 用.npz:(grid, origin, voxel_size)
    emdb_output_path: str = None,        # 返回重采样后的密度图做成的.npz(含有grid, origin, voxel_size)
    sim_output_path: str = None,         # 可选, 返回重采样后的模拟密度图做成的.npz(含有grid, origin, voxel_size)
    ligand_dist_output_path: str = None,  # 可选, ligand 距离图输出目录

    num_pocket_classes: int = 0,           # int, 口袋类别数 (不含背景); 0 时不生成距离图
    target_voxel_size=None,
    add_when_conflict=True,
    overwrite_existing: bool = False,
):
    """
    处理一个 (EMDB, PDB) 配对, 返回 (pdb_id, success, error_msg, (sample_voxel_counts, total_voxels)), 
        - sample_voxel_counts 为 dict, 键为口袋类别ID(转换为int), 值为该类别所占体素数量
        - total_voxels 为 int, 所有类别体素总数(包括背景类0)
    """
    emdb_id = list(item.keys())[0]              # str, 形如 'emd_50120'
    pdb_id  = list(item.values())[0].lower()    # str, 形如 '9e01'
    try:
        # str, EMDB 密度图路径
        emdb_path = os.path.join(emdb_folder_path, emdb_id + '.map')
        # str, PDB 样本数据目录
        pdb_sample_folder_path = os.path.join(sample_root_path, pdb_id)

        # str, 标签输出路径
        label_output_path_i = os.path.join(label_output_path, pdb_id + '.npz')
        # str, EMDB 输出路径
        emdb_output_path_i = os.path.join(emdb_output_path, pdb_id + '.npz')

        # -------------------- 标签图 --------------------
        # np.ndarray, (1, D, H, W), float32. 获取绑定后的标签数组
        label_np = bind_AtomsLabel_to_EMDB(
            sample_folder_path=pdb_sample_folder_path,
            emdb_path=emdb_path,
            target_voxel_size=target_voxel_size,
            output_path=label_output_path_i,
            overwrite_existing=overwrite_existing
        )
        # 统计本样本的各个类别的体素数量
        # np.ndarray, (num_unique_classes,), float32, 本样本中包含的各个不同口袋类别ID(含背景0)
        # np.ndarray, (num_unique_classes,), int64, 上述各个类别ID在本样本中所占的体素数量计数
        unique_classes, counts = np.unique(label_np, return_counts=True)
        # dict[int, int], 字典, 键为口袋类别ID(转换为int), 值为该类别所占体素数量
        sample_voxel_counts = {int(c): int(count)   for c, count in zip(unique_classes, counts)}
        # int, 标量, 本样本的总体素数量 (D * H * W)
        total_voxels = label_np.size


        # -------------------- 原始密度图 --------------------
        grid, voxel_size, origin = load_map(emdb_path)
        if target_voxel_size is not None:
            grid, voxel_size, origin = make_model_grid(grid, voxel_size, origin, target_voxel_size)
        atomic_np_savez(emdb_output_path_i, do_not_replace=(not overwrite_existing), grid=grid, voxel_size=voxel_size, origin=origin)


        # -------------------- 模拟密度图 --------------------
        if sim_folder_path is not None and sim_output_path is not None:
            sim_path = os.path.join(sim_folder_path, f"{emdb_id}.mrc")
            sim_output_path_i = os.path.join(sim_output_path, f"{pdb_id}.npz")
            sim_grid, sim_vs, sim_origin = load_map(sim_path)
            if target_voxel_size is not None:
                sim_grid, sim_vs, sim_origin = make_model_grid(sim_grid, sim_vs, sim_origin, target_voxel_size)
            atomic_np_savez(sim_output_path_i, do_not_replace=(not overwrite_existing), grid=sim_grid, voxel_size=sim_vs, origin=sim_origin)

        # -------------------- ligand 距离图 --------------------
        if ligand_dist_output_path is not None and num_pocket_classes > 0:
            ligand_dist_output_path_i = os.path.join(ligand_dist_output_path, pdb_id + '.npz')
            bind_LigandMinDist_to_EMDB(
                sample_folder_path=pdb_sample_folder_path,
                emdb_path=emdb_path,
                target_voxel_size=target_voxel_size,
                output_path=ligand_dist_output_path_i,
                overwrite_existing=overwrite_existing,
                num_pocket_classes=num_pocket_classes,
            )

        return (pdb_id, True, None, (sample_voxel_counts, total_voxels))
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"[FAIL] {pdb_id} ({emdb_id}): {error_msg}")
        traceback.print_exc()
        return (pdb_id, False, error_msg, None)



def save(
    emdb_pdb_json: str, 
    add_when_conflict: bool,
    target_voxel_size: float, 

    emdb_folder_path: str = None,        # EMDB文件夹路径, 目前按照 emd_50120.map 的命名格式
    sample_root_path: str = None,        # 根据 Pocket_classic\Make_Data\PDB_processor\run_preprocess.py 解析后的结果, 一个样本一个文件夹, 含有四个.npz
    sim_folder_path: str = None,


    label_output_path: str = None,       # def bind_AtomsLabel_to_EMDB 返回结果保存的文件夹, 用.npz:(grid, origin, voxel_size)
    emdb_output_path: str = None,        # 返回重采样后的密度图做成的.npz(含有grid, origin, voxel_size)
    sim_output_path: str = None,
    ligand_dist_output_path: str = None,


    num_pocket_classes: int = 0,
    part_id: int = 0,             # 当前分片 ID (0-indexed)
    total_parts: int = 1,         # 总分片数
    n_jobs: int = 1,              # joblib 并行进程数
    overwrite_existing: bool = False,  # 是否覆盖已有 .npz 文件
):
    # ---- 创建输出目录 / Create output directories ----
    os.makedirs(label_output_path,   exist_ok=True)
    os.makedirs(emdb_output_path,    exist_ok=True)
    if sim_output_path is not None:
        os.makedirs(sim_output_path, exist_ok=True)
    if ligand_dist_output_path is not None:
        os.makedirs(ligand_dist_output_path, exist_ok=True)

    # ---- 读取 JSON 并分片 / Load JSON & shard ----
    # dict, 加载 EMDB-PDB 映射文件 / Load EMDB-PDB mapping file
    with open(emdb_pdb_json, 'r') as f:
        emdb_pdb_dict = json.load(f)
    print(f"[Info] JSON 总条目 / Total items: {len(emdb_pdb_dict)}")
    # list[dict], 当前分片的任务列表 / Task list for current shard
    shard = apply_sharding(emdb_pdb_dict, part_id, total_parts)
    print(f"[Info] 当前分片 / Shard {part_id}/{total_parts}: {len(shard)} items")
    if len(shard) == 0:
        print("[Info] 当前分片无任务 / No items in this shard.")
        return

    # ---- 并行处理 / Parallel processing ----
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_item)(
            item,
            emdb_folder_path,
            sample_root_path=sample_root_path,
            label_output_path=label_output_path,
            emdb_output_path=emdb_output_path,
            sim_folder_path=sim_folder_path,
            sim_output_path=sim_output_path,
            ligand_dist_output_path=ligand_dist_output_path,
            num_pocket_classes=num_pocket_classes,
            target_voxel_size=target_voxel_size,
            add_when_conflict=add_when_conflict,
            overwrite_existing=overwrite_existing
        )
        for item in shard
    )

    # ---- 统计 / Summary ----
    n_success = sum(1 for _, ok, _, _ in results if ok)
    n_fail    = len(results) - n_success
    print("=" * 60)
    print(f"[Done] 成功 / Success: {n_success}")
    print(f"[Done] 失败 / Failed:  {n_fail}")
    if n_fail > 0:
        print("Failed samples:")
        for pid, ok, msg, _ in results:
            if not ok:
                print(f"  - {pid}: {msg}")
    print("=" * 60)

    # ---- 详细统计 / Detailed Statistics ----
    if n_success > 0:
        # int, 标量, 累计所有成功样本的最总体素个数之和
        total_voxels_all_samples = 0
        # dict[int, int], 字典, 汇总记录所有样本中各个口袋类别ID的总共占有的体素数量
        total_voxels_per_class = {}
        # dict[int, int], 字典, 汇总记录分别含有每一种口袋类别ID的样本总数
        samples_with_class = {}
        
        for pid, ok, msg, stats in results:
            if ok and stats is not None:
                # 解包获取单个样本的体素统计变量
                sample_voxel_counts, total_voxels = stats
                total_voxels_all_samples += total_voxels
                
                for c_id, count in sample_voxel_counts.items():
                    if c_id == 0:
                        continue # 背景类 / Background
                    if count > 0:
                        samples_with_class[c_id] = samples_with_class.get(c_id, 0) + 1
                        total_voxels_per_class[c_id] = total_voxels_per_class.get(c_id, 0) + count

        # list[int], 数据中实际出现的所有非零类别ID
        all_classes = sorted(total_voxels_per_class.keys())
        # set[int], 集合, 合并所有可能的非零类别ID，为输出打印做准备
        all_classes_set = set(all_classes).union(set(total_voxels_per_class.keys()))
        
        print("=" * 60)
        print("[Statistics] 类别统计 / Class Statistics:")
        for c_id in sorted(list(all_classes_set)):
            if c_id == 0:
                continue
            # float, 标量, 计算含有第 c_id 种口袋的样本数占比
            sample_ratio = samples_with_class.get(c_id, 0) / n_success
            # float, 标量, 计算所有样本中含有第 c_id 种口袋的体素之和除以总所有体素之和
            voxel_ratio = total_voxels_per_class.get(c_id, 0) / total_voxels_all_samples if total_voxels_all_samples > 0 else 0
            
            print(f"  - 第{c_id}种口袋 (Class {c_id}):")
            print(f"      含有第{c_id}种口袋的样本占比: {sample_ratio:.2%} ({samples_with_class.get(c_id, 0)}/{n_success})")
            print(f"      所有样本中含有第{c_id}种口袋原子的体素之和 / 总体素之和: {voxel_ratio:.6%} ({total_voxels_per_class.get(c_id, 0)}/{total_voxels_all_samples})")
        print("=" * 60)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bind PDB atom labels/features to EMDB maps")
    parser.add_argument("--emdb_pdb_json",      type=str, default="/home/penghongen/My_Project/Data/split/3.5_cc_qscore/all.json")

    parser.add_argument("--emdb_folder_path",   type=str, default="/storage/chenzhaoyang/cryo_em/EMDB_3.5_cc")
    parser.add_argument("--sample_root_path",   type=str, default="/home/penghongen/My_Project/Data/parsed_pdb/")
    parser.add_argument("--sim_folder_path",    type=str, default=None, help="可选, 模拟密度图 .mrc 所在目录")


    parser.add_argument("--label_output_path",  type=str, default="/storage/penghongen/Pocket_classic/pdb_label_npz/")
    parser.add_argument("--emdb_output_path",   type=str, default="/storage/penghongen/Pocket_classic/emdb_npz/")
    parser.add_argument("--sim_output_path",    type=str, default=None, help="可选, 重采样后的模拟密度图 .npz 存放目录")
    parser.add_argument("--ligand_dist_output_path", type=str, default=None, help="可选, ligand 距离图输出目录")


    parser.add_argument("--target_voxel_size",  type=float, default=0.7)
    parser.add_argument("--add_when_conflict",  action=argparse.BooleanOptionalAction, default=True,
                        help="一个体素含多个原子时累加特征 (默认开启); 用 --no-add_when_conflict 关闭")
    parser.add_argument("--overwrite_existing",  action="store_true", default=False,
                        help="覆盖输出目录中已有的 .npz 文件 (默认跳过已有文件)")
    parser.add_argument("--num_pocket_classes", type=int, default=4,
                        help="口袋类别数(不含背景), 对应 filter_config 中 rules 的数量")


    parser.add_argument("--part_id",            type=int, default=0,  help="当前分片 ID (0-indexed)")
    parser.add_argument("--total_parts",        type=int, default=1,  help="总分片数")
    parser.add_argument("--n_jobs",             type=int, default=1,  help="并行进程数")
    args = parser.parse_args()

    save(
        emdb_pdb_json=args.emdb_pdb_json,
        add_when_conflict=args.add_when_conflict,
        target_voxel_size=args.target_voxel_size,
        emdb_folder_path=args.emdb_folder_path,
        sample_root_path=args.sample_root_path,
        label_output_path=args.label_output_path,
        emdb_output_path=args.emdb_output_path,
        sim_folder_path=args.sim_folder_path,
        sim_output_path=args.sim_output_path,
        ligand_dist_output_path=args.ligand_dist_output_path,
        num_pocket_classes=args.num_pocket_classes,
        part_id=args.part_id,
        total_parts=args.total_parts,
        n_jobs=args.n_jobs,
        overwrite_existing=args.overwrite_existing
    )


