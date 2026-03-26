import os
import numpy as np
import json
import sys
import argparse
import traceback
from joblib import Parallel, delayed
from pathlib import Path

# 将 Make_Data/ 加入 sys.path, 使得 PDB_processor 和本目录下的 utils 可被导入
sys.path.append('/home/penghongen')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # → Make_Data/
sys.path.insert(0, str(Path(__file__).resolve().parent))          # → processedPDB_EMDB_binder/

from PDB_processor.run_preprocess import get_features_when_infer
from utils.mrc_tools import load_map, make_model_grid, atom2map, atomic_np_savez
from utils.network_tools import clean_temp_file


def bind_AtomsLabel_to_EMDB(
    sample_folder_path: str=None,
    emdb_path: str=None, 
    target_voxel_size: float=1.0,
    output_path: str=None,

    overwrite_existing: bool=False
):
    """
    将PDB文件中的原子标签(atoms.npz里面的'features')映射到对应的EMDB文件, 产生特征张量.
    
    Args:
        - sample_folder_path: 由 Pocket\Make_Data\PDB_processor\run_preprocess.py 产生的对应样本文件夹路径(里面有四个.npz)。注意将会读取 sample_folder_path/labels.npz 这个文件

        - emdb_path: EMDB文件路径
        - target_voxel_size: 重采样后的体素大小. 若为None则保持原有分辨率
        - output_path: 保存路径
    
    Return:
        - label_np: np.ndarray, (1, D, H, W), float32. 即, 创建一个与EMDB文件有相同空间维度的张量, 将PDB的原子标签填充到对应的位置(如果一个体素处存在一个原子, 它是属于任一个口袋, 那么就在label_np的对应位置标注成1. 否则为0).
    """
    # ---- 读取数据 / Load data ----
    labels_npz = os.path.join(sample_folder_path, 'labels.npz')   # str, 标签文件路径
    atoms_npz  = os.path.join(sample_folder_path, 'atoms.npz')    # str, 原子文件路径（需要坐标）
    labels_data = np.load(labels_npz)
    instance_ids = labels_data['instance_ids']    # np.ndarray, (N_atom,), int32, 结合位点实例 ID（-1 为背景）
    atom_pos_array = np.load(atoms_npz)['coords'] # np.ndarray, (N_atom, 3), float32, 原子坐标 (X, Y, Z)

    # ---- 将 instance_ids 转为二值标签 / Convert to binary label ----
    # np.ndarray, (N_atom,), float32, 1.0 = 口袋原子 (instance_id != -1), 0.0 = 背景
    binary_label = (instance_ids != -1).astype(np.float32)

    # ---- 加载并可选重采样 EMDB 密度图 / Load and optionally resample EMDB map ----
    map_data, voxel_size, origin = load_map(emdb_path)
    if target_voxel_size is not None:
        map_data, voxel_size, origin = make_model_grid(map_data, voxel_size, origin, target_voxel_size)
    # np.ndarray, (1, D, H, W), float32, D=Z, H=Y, W=X
    label_np = np.zeros((1, map_data.shape[0], map_data.shape[1], map_data.shape[2]), dtype=np.float32)

    # ---- 世界坐标 → 体素索引 / World coords → voxel indices ----
    # atom_pos_array: (N_atom, 3), 列顺序 (x, y, z); origin / voxel_size: (3,), 同为 (x, y, z)
    voxel_ijk = ((atom_pos_array - origin) / voxel_size).astype(int)  # np.ndarray, (N_atom, 3), int, 体素索引 (x, y, z)
    x_idx = np.clip(voxel_ijk[:, 0], 0, label_np.shape[3] - 1)  # np.ndarray, (N_atom,), int, X 轴索引
    y_idx = np.clip(voxel_ijk[:, 1], 0, label_np.shape[2] - 1)  # np.ndarray, (N_atom,), int, Y 轴索引
    z_idx = np.clip(voxel_ijk[:, 2], 0, label_np.shape[1] - 1)  # np.ndarray, (N_atom,), int, Z 轴索引

    # ---- 填充标签 / Fill labels ----
    # label_np: (1, D, H, W) 即 (1, Z, Y, X)
    # 使用 np.maximum.at: 同一体素多个原子取最大值 (0 或 1), 保证二值标签不超过 1
    np.maximum.at(label_np, (0, z_idx, y_idx, x_idx), binary_label)

    if output_path is not None:
        atomic_np_savez(output_path, do_not_replace=(not overwrite_existing), grid=label_np, voxel_size=voxel_size, origin=origin)

    return label_np


def bind_AtomsFeature_to_EMDB(
    sample_folder_path: str=None,   # 与下面二选一
    pdb_path: str=None,             # 与上面二选一

    emdb_path: str=None, 
    target_voxel_size: float=1.0,
    output_path: str=None, 
    
    add_when_conflict=True,
    overwrite_existing: bool=False
):
    """
    将PDB文件中的原子特征(atoms.npz里面的'features')映射到对应的EMDB文件, 产生特征张量.这个函数强烈依赖于Pocket\Make_Data\PDB_processor\run_preprocess.py 的返回结果.
    
    Args:
        - sample_folder_path: 由 Pocket\Make_Data\PDB_processor\run_preprocess.py 产生的对应样本文件夹路径(里面有四个.npz)。注意将会读取 sample_folder_path/atoms.npz这个文件
        - pdb_path: PDB文件路径. 训练时读取atoms.npz, 推断时输入pdb, 通过 Pocket\Make_Data\PDB_processor\run_preprocess.py 里面的 def get_features_when_infer 读取与atoms.npz等价的特征.

        - emdb_path: EMDB文件路径
        - target_voxel_size: 重采样后的体素大小. 若为None则保持原有分辨率

        - output_path: 保存路径
        - add_when_conflict: 一个体素含有多个原子时, 如果 add_when_conflict=True 那么就把它们的特征向量进行累加, 否则后写覆盖 (last-write-wins, 取决于原子数组顺序)
        - overwrite_existing: 如果为 True 则覆盖已有 .npz 文件, 否则跳过 (默认 False)
    
    Return:
        - feature_np: np.ndarray, (C, D, H, W), float32. 即, 创建一个与EMDB文件有相同空间维度的张量, 将PDB的原子特征填充到对应的位置.
    """
    if sample_folder_path is not None:
        atoms_npz = os.path.join(sample_folder_path, 'atoms.npz')
        atom_features_array = np.load(atoms_npz)['features']  # np.ndarray, (N_atom, 49), float32, 残基级特征向量 (类型 One-Hot + 理化性质)
        atom_pos_array = np.load(atoms_npz)['coords']  # (N_atom, 3)
    elif pdb_path is not None:
        atom_info, _, _ = get_features_when_infer(input_path=pdb_path)
        atom_features_array = atom_info['features']
        atom_pos_array = atom_info['coords']
    else:
        raise ValueError("Either atoms_npz or pdb_path must be provided")
    feature_channel = atom_features_array.shape[1]
    print(f"feature_channel 是{feature_channel}")
    
    map_data, voxel_size, origin = load_map(emdb_path)
    if target_voxel_size is not None:
        map_data, voxel_size, origin = make_model_grid(map_data, voxel_size, origin, target_voxel_size)
    feature_np = np.zeros((feature_channel, map_data.shape[0], map_data.shape[1], map_data.shape[2]), dtype=np.float32)   # (C, D, H, W)

    # ------------------------------------------------------------------
    # 将世界坐标转为体素索引 (批量计算，不使用 atom2map)
    # atom_pos_array: (N_atom, 3), 列顺序 (x, y, z); origin / voxel_size: (3,), 同为 (x, y, z) 顺序
    # feature_np 空间轴顺序是 (Z, Y, X)，因此索引需要反转
    # ------------------------------------------------------------------
    voxel_ijk = ((atom_pos_array - origin) / voxel_size).astype(int)  # np.ndarray, (N_atom, 3), int, 体素索引 (x, y, z)
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

    return feature_np





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
    item,                         # dict, JSON 中的一个条目, 形如 {"emd_50120": "9E01"}

    emdb_folder_path: str,        # EMDB文件夹路径, 目前按照 emd_50120.map 的命名格式
    sample_root_path: str,        # 根据 Pocket_classic\Make_Data\PDB_processor\run_preprocess.py 解析后的结果, 一个样本一个文件夹, 含有四个.npz
    
    feature_output_path: str,     # def bind_AtomsFeature_to_EMDB 返回结果保存的文件夹, 用.npz:(grid, origin, voxel_size)
    label_output_path: str,       # def bind_AtomsLabel_to_EMDB 返回结果保存的文件夹, 用.npz:(grid, origin, voxel_size)
    emdb_output_path: str,        # 返回重采样后的密度图做成的.npz(含有grid, origin, voxel_size)

    target_voxel_size,
    add_when_conflict,
    overwrite_existing: bool = False,
):
    """
    处理一个 (EMDB, PDB) 配对, 返回 (pdb_id, success, error_msg).
    """
    emdb_id = list(item.keys())[0]              # str, 形如 'emd_50120'
    pdb_id  = list(item.values())[0].lower()    # str, 形如 '9e01'
    try:
        emdb_path = os.path.join(emdb_folder_path, emdb_id + '.map')
        pdb_sample_folder_path = os.path.join(sample_root_path, pdb_id)

        label_output_path_i = os.path.join(label_output_path, pdb_id + '.npz')
        feature_output_path_i = os.path.join(feature_output_path, pdb_id + '.npz')
        emdb_output_path_i = os.path.join(emdb_output_path, pdb_id + '.npz')

        bind_AtomsLabel_to_EMDB(
            sample_folder_path=pdb_sample_folder_path,
            emdb_path=emdb_path,
            target_voxel_size=target_voxel_size,
            output_path=label_output_path_i,
            overwrite_existing=overwrite_existing,
        )
        bind_AtomsFeature_to_EMDB(
            sample_folder_path=pdb_sample_folder_path,
            emdb_path=emdb_path,
            target_voxel_size=target_voxel_size,
            output_path=feature_output_path_i,
            add_when_conflict=add_when_conflict,
            overwrite_existing=overwrite_existing,
        )

        grid, voxel_size, origin = load_map(emdb_path)
        if target_voxel_size is not None:
            grid, voxel_size, origin = make_model_grid(grid, voxel_size, origin, target_voxel_size)
        atomic_np_savez(emdb_output_path_i, do_not_replace=(not overwrite_existing), grid=grid, voxel_size=voxel_size, origin=origin)

        return (pdb_id, True, None)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"[FAIL] {pdb_id} ({emdb_id}): {error_msg}")
        traceback.print_exc()
        return (pdb_id, False, error_msg)



def save(
    emdb_pdb_json: str, 
    add_when_conflict: bool,
    target_voxel_size: float, 

    emdb_folder_path: str,        # EMDB文件夹路径, 目前按照 emd_50120.map 的命名格式
    sample_root_path: str,        # 根据 Pocket_classic\Make_Data\PDB_processor\run_preprocess.py 解析后的结果, 一个样本一个文件夹, 含有四个.npz

    feature_output_path: str,     # def bind_AtomsFeature_to_EMDB 返回结果保存的文件夹, 用.npz:(grid, origin, voxel_size)
    label_output_path: str,       # def bind_AtomsLabel_to_EMDB 返回结果保存的文件夹, 用.npz:(grid, origin, voxel_size)
    emdb_output_path: str,        # 返回重采样后的密度图做成的.npz(含有grid, origin, voxel_size)

    part_id: int = 0,             # 当前分片 ID (0-indexed)
    total_parts: int = 1,         # 总分片数
    n_jobs: int = 1,              # joblib 并行进程数
    overwrite_existing: bool = False,  # 是否覆盖已有 .npz 文件
):
    # ---- 创建输出目录 / Create output directories ----
    os.makedirs(feature_output_path, exist_ok=True)
    os.makedirs(label_output_path,   exist_ok=True)
    os.makedirs(emdb_output_path,    exist_ok=True)

    # ---- 读取 JSON 并分片 / Load JSON & shard ----
    with open(emdb_pdb_json, 'r') as f:
        emdb_pdb_dict = json.load(f)
    print(f"[Info] JSON 总条目 / Total items: {len(emdb_pdb_dict)}")
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
            sample_root_path,
            feature_output_path,
            label_output_path,
            emdb_output_path,
            target_voxel_size,
            add_when_conflict,
            overwrite_existing,
        )
        for item in shard
    )

    # ---- 统计 / Summary ----
    n_success = sum(1 for _, ok, _ in results if ok)
    n_fail    = len(results) - n_success
    print("=" * 60)
    print(f"[Done] 成功 / Success: {n_success}")
    print(f"[Done] 失败 / Failed:  {n_fail}")
    if n_fail > 0:
        print("Failed samples:")
        for pid, ok, msg in results:
            if not ok:
                print(f"  - {pid}: {msg}")
    print("=" * 60)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bind PDB atom labels/features to EMDB maps")
    parser.add_argument("--emdb_pdb_json",      type=str, default="/home/penghongen/My_Project/Data/split/3.5_cc_qscore/all.json")
    parser.add_argument("--emdb_folder_path",   type=str, default="/storage/chenzhaoyang/cryo_em/EMDB_3.5_cc")
    parser.add_argument("--sample_root_path",   type=str, default="/home/penghongen/My_Project/Data/parsed_pdb/")
    parser.add_argument("--feature_output_path",type=str, default="/storage/penghongen/Pocket_classic/pdb_feature_npz/")
    parser.add_argument("--label_output_path",  type=str, default="/storage/penghongen/Pocket_classic/pdb_label_npz/")
    parser.add_argument("--emdb_output_path",   type=str, default="/storage/penghongen/Pocket_classic/emdb_npz/")
    parser.add_argument("--target_voxel_size",  type=float, default=0.7)
    parser.add_argument("--add_when_conflict",  action=argparse.BooleanOptionalAction, default=True,
                        help="一个体素含多个原子时累加特征 (默认开启); 用 --no-add_when_conflict 关闭")
    parser.add_argument("--overwrite_existing",  action="store_true", default=False,
                        help="覆盖输出目录中已有的 .npz 文件 (默认跳过已有文件)")

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
        feature_output_path=args.feature_output_path,
        label_output_path=args.label_output_path,
        emdb_output_path=args.emdb_output_path,
        part_id=args.part_id,
        total_parts=args.total_parts,
        n_jobs=args.n_jobs,
        overwrite_existing=args.overwrite_existing,
    )


    clean_temp_file([args.feature_output_path, args.label_output_path, args.emdb_output_path])
