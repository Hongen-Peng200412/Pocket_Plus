import argparse
import gc
import json
import traceback
import numpy as np
import os
import warnings
import sys
sys.path.append('/home/penghongen')
from joblib import Parallel, delayed
from utils.mrc_tools import atomic_np_savez
from utils.network_tools import apply_sharding
warnings.filterwarnings("ignore")



def check_box_relative(
    label_block, center_pos_count, center_inner_pos_count,
    cut_length=12, all_pos_ratio=0.3, center_pos_ratio=0.3
):
    """
    以 "中心小BOX" 的正类体素指标为基准, 检查当前 BOX 是否达到相对阈值.
    若中心小BOX本身没有正类体素 (center_pos_count==0 或 center_inner_pos_count==0), 则降级为绝对检查 (>0).

    输入参数 / Input:
        - label_block:              numpy, (..., D', H', W'), 当前 BOX 的 label 切片 (正类>0)
        - center_pos_count:         int, 中心 BOX 的**整体**正类体素数
        - center_inner_pos_count:   int, 中心 BOX **inner 区域**的正类体素数
        - cut_length:               int, inner 区域裁掉的边缘宽度 (体素数)
        - all_pos_ratio:            float, 当前 BOX 正类总数 / 中心 BOX 正类总数 的最低比例
        - center_pos_ratio:         float, 当前 BOX inner 正类数在总正类中的占比 / 中心 BOX inner 正类数在总正类中的占比 的最低比例

    输出 / Output:
        - bool, True 则保留该 BOX, False 则丢弃
    """
    # int, 当前 BOX 的正类体素总数
    box_pos_count = int(np.sum(label_block > 0))
    # numpy, (..., D'-2*cl, H'-2*cl, W'-2*cl), 当前 BOX 的 inner 区域
    inner = label_block[..., cut_length:-cut_length, cut_length:-cut_length, cut_length:-cut_length]
    # int, 当前 BOX inner 区域的正类体素数
    box_inner_pos_count = int(np.sum(inner > 0))

    # ---- 降级: 中心 BOX 无正类时, 只要当前 BOX 有正类就保留 ----
    if center_pos_count == 0:
        return box_pos_count > 0
    if center_inner_pos_count == 0:
        return box_inner_pos_count > 0

    if box_pos_count == 0:
        return False
    ratio_a = box_pos_count / center_pos_count
    ratio_b = (box_inner_pos_count / box_pos_count) / (center_inner_pos_count / center_pos_count)

    return ratio_a >= all_pos_ratio and ratio_b >= center_pos_ratio



def Split_Data_into_Box_PocketCentered(
    origin, voxel_size, pdb_id: str,
    pocket_centers_world,
    pocket_atom_coords_list,
    instance_ids_per_pocket,
    *grids,

    window_size: int = 72, stride: int = 12,
    r_expand: float = 2.0,
    all_pos_ratio: float = 0.3,
    center_pos_ratio: float = 0.3,
    cut_length: int = 12,

    output_root_folder: str = None, name_list: list = None,
    overwrite_existing: bool = False,
    pocket_atom_counts: list = None,
    ligand_atom_counts: list = None,
):
    """
    以每个结合位点 (pocket) 为中心, 构造大立方体并在其中滑动窗口裁剪.

    输入参数 / Input:
        - origin:                   numpy, (3,), float, 密度图全局坐标原点 (x,y,z), 单位 Å
        - voxel_size:               numpy, (3,), float, 体素尺寸 (dx,dy,dz), 单位 Å
        - pdb_id:                   str, 如 "3q0l", 用于文件命名 (小写)
        - pocket_centers_world:     numpy, (N_pockets, 3), float, 各口袋中心的世界坐标 (x,y,z), 单位 Å
        - pocket_atom_coords_list:  list[numpy], 长度 N_pockets, 每个元素 (M_i, 3) float, 代表第 i 个口袋中所有结合原子的世界坐标
        - instance_ids_per_pocket:  list[int], 长度 N_pockets, 每个口袋对应的配体实例 ID. 仅用于发送消息/文件名
        - *grids:                   一系列 numpy 数组, 每个形状 (..., D, H, W). grids[0] 是 label 用于检验

        - window_size:              int, 滑动窗口边长 (体素数), 默认 72, 它和 r_expand 都最好为偶数, 使中心小BOX位于0_0_0, 便于分阶段训练
        - stride:                   int, 滑动窗口步长 (体素数), 默认 12
        - r_expand:                 float, 大立方体扩展系数, 大立方体边长 = window_size + int(r_expand * envelope_side)
        - all_pos_ratio:            float, 当前 BOX 正类总数 / 中心 BOX 正类总数 的最低比例
        - center_pos_ratio:         float, 当前 BOX inner 正类数在总正类中的占比 / 中心 BOX inner 正类数在总正类中的占比 的最低比例
        - cut_length:               int, inner 区域裁掉的边缘宽度 (体素数)

        - output_root_folder:       str, 输出根文件夹
        - name_list:                list[str], 子文件夹名, name_list[i] 对应 grids[i] 保存的文件夹名
        - overwrite_existing:       bool, 是否覆盖已有 .npz
        - pocket_atom_counts:       list[int] | None, 长度 N_pockets, 每个口袋的蛋白质结合原子数 (用于统计)
        - ligand_atom_counts:       list[int] | None, 长度 N_pockets, 每个口袋对应配体的原子数 (用于统计)

    输出 / Output:
        - saved_count:      int, 累计合格且写入的 BOX 数
        - skipped_count:    int, 因文件已存在而跳过的 BOX 数
        - all_count:        int, 累计检查过的 BOX 数
        - pocket_stats_list: list[dict], 每个口袋的统计信息, 含 ligand_atom_count / pocket_atom_count / envelope_side / box_count
    """
    # int,int,int  体素网格的空间形状
    Z, Y, X = grids[0].shape[-3:]
    if output_root_folder is None:
        raise ValueError("output_root_folder is None")
    # str, 小写 PDB ID
    pdb_id = pdb_id.split(".")[0].lower()
    for i, _ in enumerate(grids):
        os.makedirs(os.path.join(output_root_folder, f'{name_list[i]}'), exist_ok=True)

    saved_count, skipped_count, all_count = 0, 0, 0
    # int, 口袋总数
    n_pockets = len(pocket_centers_world)
    if n_pockets == 0:
        print(f"警告: {pdb_id} 没有检测到口袋, 跳过.")
        return 0, 0, 0, []

    # ================================================
    # 遍历每个口袋 / Iterate over each pocket
    # ================================================
    # list[dict], 每个口袋的统计信息
    pocket_stats_list = []
    for pocket_idx in range(n_pockets):
        # int, 当前口袋的配体实例 ID (global_id)
        inst_id = instance_ids_per_pocket[pocket_idx]
        # numpy, (3,), float, 当前口袋中心的世界坐标 (x,y,z)
        pocket_center_world = pocket_centers_world[pocket_idx]
        # numpy, (M_i, 3), float, 当前口袋的结合原子世界坐标
        pocket_atoms_world = pocket_atom_coords_list[pocket_idx]
        # numpy, (3,), int, 口袋中心体素坐标 (x,y,z 顺序; 向下取整)
        pocket_center_voxel = np.floor( (pocket_center_world - origin) / voxel_size ).astype(int)
        # numpy, (M_i, 3), int, 口袋原子的体素坐标
        pocket_atoms_voxel = np.floor( (pocket_atoms_world - origin) / voxel_size ).astype(int)


        # ---- 计算包络正方体 (envelope cube) ----
        # numpy, (3,), int, 各轴最小体素坐标
        vmin = np.min(pocket_atoms_voxel, axis=0)
        vmax = np.max(pocket_atoms_voxel, axis=0)
        ranges = vmax - vmin
        # int, 包络正方体边长 = max(三轴跨度), 向上取整
        envelope_side = np.max(ranges)


        # ---- 计算大立方体 ----
        large_cube_side = int(window_size + r_expand * envelope_side)
        # int, 大立方体在 x 轴的起始体素 (未裁剪)
        large_start_x = pocket_center_voxel[0] - large_cube_side // 2
        large_start_y = pocket_center_voxel[1] - large_cube_side // 2
        large_start_z = pocket_center_voxel[2] - large_cube_side // 2
        # int, 中心 BOX 在 x 轴的起始体素 (未裁剪)
        center_start_x = pocket_center_voxel[0] - window_size // 2
        center_start_y = pocket_center_voxel[1] - window_size // 2
        center_start_z = pocket_center_voxel[2] - window_size // 2


        # ---- 裁剪到网格边界 ----
        # int, 各轴体素索引范围上限 (不含)
        grid_max_x, grid_max_y, grid_max_z = X, Y, Z
        # int, 裁剪后中心 BOX 起始
        c_sx = max(0, min(center_start_x, grid_max_x - window_size))
        c_sy = max(0, min(center_start_y, grid_max_y - window_size))
        c_sz = max(0, min(center_start_z, grid_max_z - window_size))
        if grid_max_x < window_size or grid_max_y < window_size or grid_max_z < window_size:
            print(f"警告: {pdb_id} pocket {inst_id} 尺寸过小 ({Z}x{Y}x{X}), 无法裁剪 {window_size} 块, 跳过.")
            continue
        # numpy, (..., window_size, window_size, window_size), 中心 BOX 的 label 切片
        center_label_block = grids[0][..., c_sz:c_sz + window_size, c_sy:c_sy + window_size, c_sx:c_sx + window_size]
        # int, 中心 BOX 的正类总体素数
        center_pos_count = int(np.sum(center_label_block > 0))
        center_inner = center_label_block[..., cut_length:-cut_length, cut_length:-cut_length, cut_length:-cut_length]
        center_inner_pos_count = int(np.sum(center_inner > 0))
        print(f"  pocket {inst_id}: envelope={envelope_side}, large_cube={large_cube_side}, "  #NOTE
              f"center_pos={center_pos_count}, center_inner_pos={center_inner_pos_count}")


        # ---- 构造滑动窗口坐标列表 ----
        # int, 大立方体在各轴的合法起始范围
        slide_min_x, slide_min_y, slide_min_z = max(0, large_start_x), max(0, large_start_y), max(0, large_start_z)
        slide_max_x = min(grid_max_x - window_size, large_start_x + large_cube_side - window_size)
        slide_max_y = min(grid_max_y - window_size, large_start_y + large_cube_side - window_size)
        slide_max_z = min(grid_max_z - window_size, large_start_z + large_cube_side - window_size)

        # 确保 slide_max >= slide_min
        slide_max_x = max(slide_max_x, slide_min_x)
        slide_max_y = max(slide_max_y, slide_min_y)
        slide_max_z = max(slide_max_z, slide_min_z)
        def _make_positions(smin, smax, center_s, stride_val):
            """
            生成滑动位置列表. 注意尾部对齐 + 手动使之包含正中心

            输入参数:
                - smin:       int, 合法起始最小值
                - smax:       int, 合法起始最大值
                - center_s:   int, 中心 BOX 的起始位置 (已裁剪)
                - stride_val: int, 步长
            输出:
                - list[int], 排序去重后的滑动起始位置列表
            """
            # list[int], 均匀步长生成的起始位置
            positions = list(range(smin, smax + 1, stride_val))
            # 确保包含 smax (尾部对齐)
            if len(positions) == 0 or positions[-1] < smax:
                positions.append(smax)
            # 确保包含中心起始位置
            if center_s not in positions:
                positions.append(center_s)
            return sorted(set(positions))
        # list[int], 各轴的滑动起始坐标
        pos_x = _make_positions(slide_min_x, slide_max_x, c_sx, stride)
        pos_y = _make_positions(slide_min_y, slide_max_y, c_sy, stride)
        pos_z = _make_positions(slide_min_z, slide_max_z, c_sz, stride)


        pocket_check_count = 0
        pocket_save_count = 0
        # ---- 遍历所有滑动位置 ----
        for zz in pos_z:
            for yy in pos_y:
                for xx in pos_x:

                    all_count += 1
                    pocket_check_count += 1
                    # ---- 计算相对偏移 (BOX 中心 - pocket 中心, 体素单位) ----
                    # int, 当前 BOX 中心在 x 轴的体素坐标
                    box_center_x = xx + window_size // 2
                    box_center_y = yy + window_size // 2
                    box_center_z = zz + window_size // 2
                    # int, 相对偏移 (可负)
                    Rxx = box_center_x - pocket_center_voxel[0]
                    Ryy = box_center_y - pocket_center_voxel[1]
                    Rzz = box_center_z - pocket_center_voxel[2]

                    for num, grid in enumerate(grids):
                        assert grid.shape[-3:] == (Z, Y, X), \
                            f"输入的第 {num} 个图尺寸不匹配, 应为 {Z}x{Y}x{X}, 实际为 {grid.shape[-3:]}"
                        # numpy, (..., window_size, window_size, window_size), 当前 BOX 切片
                        block = grid[..., zz:zz + window_size, yy:yy + window_size, xx:xx + window_size]
                        # float tuple, 当前 BOX 在世界坐标下的范围
                        x_range = (origin[0] + xx * voxel_size[0], origin[0] + (xx + window_size) * voxel_size[0])
                        y_range = (origin[1] + yy * voxel_size[1], origin[1] + (yy + window_size) * voxel_size[1])
                        z_range = (origin[2] + zz * voxel_size[2], origin[2] + (zz + window_size) * voxel_size[2])
                        # tuple(float,float,float), 当前 BOX 原点的世界坐标
                        global_origin = (origin[0] + xx * voxel_size[0],
                                         origin[1] + yy * voxel_size[1],
                                         origin[2] + zz * voxel_size[2])
                        if num == 0:  # label — 用于检验
                            if not check_box_relative(block, center_pos_count, center_inner_pos_count,
                                                      cut_length=cut_length,
                                                      all_pos_ratio=all_pos_ratio,
                                                      center_pos_ratio=center_pos_ratio):
                                break  # 不合格, 跳过所有 grids
                            else:
                                saved_count += 1
                                pocket_save_count += 1

                        # ---- 文件名: {pdb_id}_{instance_id}_{Rxx}_{Ryy}_{Rzz}.npz ----
                        path = os.path.join(
                            output_root_folder,
                            f'{name_list[num]}',
                            f'{pdb_id}_{inst_id}_{Rxx}_{Ryy}_{Rzz}.npz'
                        )
                        do_not_replace = not overwrite_existing
                        if do_not_replace and os.path.exists(path):
                            if num == 0:
                                skipped_count += 1
                            continue

                        atomic_np_savez(path, do_not_replace=False,
                                        grid=block,
                                        x_range=np.array(x_range),
                                        y_range=np.array(y_range),
                                        z_range=np.array(z_range),
                                        voxel_size=np.array(voxel_size),
                                        origin=np.array(global_origin))

        print(f"  pocket {inst_id}: 检查 {pocket_check_count} 个 BOX, 合格 {pocket_save_count} 个")

        # ---- 收集口袋级统计 ----
        pocket_stats_list.append({
            'ligand_atom_count':  ligand_atom_counts[pocket_idx] if ligand_atom_counts is not None else -1,
            'pocket_atom_count':  pocket_atom_counts[pocket_idx] if pocket_atom_counts is not None else -1,
            'envelope_side':      int(envelope_side),
            'box_count':          pocket_save_count,
        })

    actual_written = saved_count - skipped_count
    print(f"--- {pdb_id} 裁剪完成: 遍历 {n_pockets} 个口袋, 检查 {all_count} 个 BOX, "
          f"合格 {saved_count} 个, 实际写入 {actual_written} 个, 因已存在跳过 {skipped_count} 个。---\n")
    return saved_count, skipped_count, all_count, pocket_stats_list




# ------------------------------------------------------------------ #
#  单样本处理函数 (供 joblib 并行调用)
# ------------------------------------------------------------------ #
def _process_one_sample(
    item,
    pdb_label_npz_folder, map_npz_folder, pdb_feature_npz_folder,
    sample_root_path,
    valid_id_set,

    window_size, stride,
    r_expand, all_pos_ratio, center_pos_ratio,
    cut_length,

    output_root_folder, name_list,
    overwrite_existing,
):
    """
    处理单个 EMDB-PDB 样本: 加载三个 .npz 文件(密度图、PDB特征图、PDB标签图) + 原始标签   
        →   调用 Split_Data_into_Box_PocketCentered 做口袋中心裁剪 → 保存 BOX.
    该函数由 joblib.Parallel 在 Split_Datas_into_Box 内部并行调用.

    输入参数 / Input:
        - item:                     dict, 形如 {"emd_47345": "9E01"}, 单条 EMDB-PDB 映射

        - pdb_label_npz_folder:     str, bind_AtomsLabel_to_EMDB 保存的 .npz 母文件夹路径, 含有 .npz:(grid, origin, voxel_size)
        - map_npz_folder:           str, 密度图经 bind.py 重采样(目前为0.7A)后的 .npz 母文件夹路径, 含有 .npz:(grid, origin, voxel_size)
        - pdb_feature_npz_folder:   str, bind_AtomsFeature_to_EMDB 保存的 .npz 母文件夹路径, 含有 .npz:(grid, origin, voxel_size)
        - sample_root_path:         str, PDB_processor/run_preprocess.py 解析后的样本根目录, 子文件夹为 {pdb_id}/, 含 labels.npz 和 atoms.npz
        - valid_id_set:             set[str], 以上所有源文件中都存在的 pdb_id 交集 (小写)

        - window_size:              int, 滑动窗口边长 (体素数)
        - stride:                   int, 滑动窗口步长 (体素数)
        - r_expand:                 float, 大立方体扩展系数
        - all_pos_ratio:            float, a — BOX 正类总数 / 中心 BOX 正类总数  的最低比例
        - center_pos_ratio:         float, b — BOX inner 正类数占比 / 中心 BOX inner 正类数占比  的最低比例
        - cut_length:               int, inner 区域裁掉的边缘宽度

        - output_root_folder:       str, 保存的根文件夹路径
        - name_list:                list[str], 保存的文件夹名列表, name_list[i] 对应 grids[i]
        - overwrite_existing:       bool, 是否覆盖已有 .npz

    返回 / Output:
        - tuple: (pdb_id, saved_count, skipped_count, all_count, success, error_msg, pocket_stats_list)
            - pdb_id:            str, 当前样本的 PDB ID (小写)
            - saved_count:       int, 本样本合格的 BOX 数
            - skipped_count:     int, 本样本因文件已存在而跳过的 BOX 数
            - all_count:         int, 本样本检查过的 BOX 总数
            - success:           bool, 是否处理成功
            - error_msg:         str|None, 失败时的错误信息; 跳过时为 'skipped'; 成功时为 None
            - pocket_stats_list: list[dict], 每个口袋的统计 (ligand_atom_count, pocket_atom_count, envelope_side, box_count)
    """
    # ---- 解析 pdb_id ----
    pdb_id = list(item.values())[0].lower()                       # str, 小写的 PDB ID, 如 "9e01"
    if pdb_id not in valid_id_set:
        return (pdb_id, 0, 0, 0, True, 'skipped', [])

    try:
        map_path = os.path.join(map_npz_folder, f'{pdb_id}.npz')                    # str, 密度图 npz 路径
        pdb_feature_path = os.path.join(pdb_feature_npz_folder, f'{pdb_id}.npz')    # str, PDB 特征图 npz 路径
        pdb_label_path = os.path.join(pdb_label_npz_folder, f'{pdb_id}.npz')        # str, PDB 标签图 npz 路径

        # ---- 加载原始标签 (pocket 信息) ----
        labels_npz_path = os.path.join(sample_root_path, pdb_id, 'labels.npz')      # str, 原始标签文件路径
        atoms_npz_path = os.path.join(sample_root_path, pdb_id, 'atoms.npz')        # str, 原始原子文件路径
        if not os.path.exists(labels_npz_path) or not os.path.exists(atoms_npz_path):
            return (pdb_id, 0, 0, 0, True, 'skipped (no labels/atoms)', [])
        labels_data = np.load(labels_npz_path, allow_pickle=True)
        # numpy, (N_ligands, 3), float32, 各口袋中心世界坐标
        pocket_centers = labels_data['pocket_centers']
        # numpy, (N_atoms,), int32, 每个原子的结合位点实例 ID (-1=非结合)
        instance_ids = labels_data['instance_ids']
        labels_data.close()
        # 重新打开用于后续读取 ligand_coords_* (延迟读取, 避免一次性全部加载)
        labels_data_reopen = np.load(labels_npz_path, allow_pickle=True)
        # numpy, (N_atoms, 3), float32, 原子世界坐标
        atom_coords = np.load(atoms_npz_path)['coords']


        # ---- 提取每个口袋的原子坐标和实例 ID ----
        # numpy, (N_atoms,), bool, 结合位点掩码
        binding_mask = instance_ids != -1
        # numpy, (K,), int32, 所有参与结合的唯一实例 ID (升序排列了)
        unique_inst_ids = np.unique(instance_ids[binding_mask])
        if len(unique_inst_ids) == 0:
            return (pdb_id, 0, 0, 0, True, 'skipped (no binding site)', [])
        # list[numpy], 每个口袋的结合原子世界坐标, 元素形状 (M_i, 3), 长度为有口袋原子的ligand数目 N_pockets_matched
        pocket_atom_coords_list = []
        # list[int], 每个口袋的实例 ID
        instance_ids_per_pocket = []
        # numpy, (N_pockets_matched, 3), 对应的口袋中心 (从 pocket_centers 取)
        pocket_centers_matched = []

        # list[int], 每个口袋的蛋白质结合原子数
        pocket_atom_counts = []
        # list[int], 每个口袋对应的配体原子数
        ligand_atom_counts = []

        for inst_id in unique_inst_ids:
            inst_id_int = int(inst_id)
            # numpy, (M_i, 3), float32, 属于该配体的蛋白质结合原子坐标
            coords_i = atom_coords[instance_ids == inst_id_int]
            if len(coords_i) == 0:
                continue
            pocket_atom_coords_list.append(coords_i)
            instance_ids_per_pocket.append(inst_id_int)
            # int, 蛋白质结合原子数
            pocket_atom_counts.append(len(coords_i))
            # int, 配体原子数 (从 labels.npz 中加载)
            lig_coords_key = f'ligand_coords_{inst_id_int}'
            if lig_coords_key in labels_data_reopen:
                ligand_atom_counts.append(len(labels_data_reopen[lig_coords_key]))
            else:
                ligand_atom_counts.append(-1)  # 未找到配体坐标
            pocket_centers_matched.append(pocket_centers[inst_id_int])
        labels_data_reopen.close()
        # numpy, (N_pockets_matched, 3), float32
        pocket_centers_matched = np.array(pocket_centers_matched, dtype=np.float32)

        # ---- 加载三个体素化 npz ----
        map_item = np.load(map_path)              # NpzFile
        map_grid = map_item['grid'][None, ...]    # numpy, (1,D,H,W), 密度图体素值
        origin_arr = map_item['origin']           # numpy, (3,), 原点
        voxel_size_arr = map_item['voxel_size']   # numpy, (3,), 体素尺寸
        map_item.close()

        pdb_feature_grid = np.load(pdb_feature_path)['grid']   # numpy, (C,D,H,W)
        pdb_label_grid = np.load(pdb_label_path)['grid']       # numpy, (1,D,H,W)

        # ---- 调用口袋中心裁剪 ----
        saved, skipped, checked, pocket_stats = Split_Data_into_Box_PocketCentered(
            origin_arr, voxel_size_arr, pdb_id,
            pocket_centers_matched,
            pocket_atom_coords_list,
            instance_ids_per_pocket,

            pdb_label_grid, map_grid, pdb_feature_grid,
            window_size=window_size, stride=stride,
            r_expand=r_expand,
            all_pos_ratio=all_pos_ratio,
            center_pos_ratio=center_pos_ratio,
            cut_length=cut_length,
            output_root_folder=output_root_folder, name_list=name_list,
            overwrite_existing=overwrite_existing,
            pocket_atom_counts=pocket_atom_counts,
            ligand_atom_counts=ligand_atom_counts,
        )
        del map_grid, pdb_feature_grid, pdb_label_grid, atom_coords, pocket_atom_coords_list
        gc.collect()
        return (pdb_id, saved, skipped, checked, True, None, pocket_stats)
    except Exception as e:
        traceback.print_exc()
        return (pdb_id, 0, 0, 0, False, f"{type(e).__name__}: {e}", [])



def Split_Datas_into_Box(
    emdb_pdb_json: str, 
    pdb_label_npz_folder: str, map_npz_folder: str, pdb_feature_npz_folder: str, 
    sample_root_path: str,

    window_size: int, stride: int, 
    r_expand: float,
    all_pos_ratio: float,
    center_pos_ratio: float,
    cut_length: int,

    output_root_folder: str, name_list: list,
    
    part_id: int = 0, total_parts: int = 1,
    n_jobs: int = 1,
    overwrite_existing: bool = False,
):
    """
    针对 bind.py 的输出文件(密度图、PDB特征图、PDB标签图), 批量应用口袋中心裁剪构造训练用 BOX.
    支持 SLURM 分片 (apply_sharding) + CPU 并行 (joblib.Parallel).

    输入参数 / Input:
        - emdb_pdb_json:          str, .json 映射文件路径, 内容为 List[Dict[str, str]]
        - pdb_label_npz_folder:   str, PDB 标签 npz 母文件夹路径
        - map_npz_folder:         str, 密度图 npz 母文件夹路径
        - pdb_feature_npz_folder: str, PDB 特征 npz 母文件夹路径
        - sample_root_path:       str, PDB_processor/run_preprocess.py 输出的样本根目录, 含 {pdb_id}/labels.npz 和 atoms.npz

        - window_size:            int, 滑动窗口边长 (体素数)
        - stride:                 int, 滑动窗口步长 (体素数)
        - r_expand:               float, 大立方体扩展系数
        - all_pos_ratio:          float, a — BOX 正类总数 / 中心 BOX 正类总数 的最低比例
        - center_pos_ratio:       float, b — BOX inner 正类数 / 中心 BOX inner 正类数 的最低比例
        - cut_length:             int, inner 区域裁掉的边缘宽度

        - output_root_folder:     str, 输出根文件夹路径
        - name_list:              list[str], 输出子文件夹名列表

        - part_id:                int, 当前分片 ID (0-indexed)
        - total_parts:            int, 总分片数
        - n_jobs:                 int, joblib 并行进程数
        - overwrite_existing:     bool, 是否覆盖已有 .npz

    返回 / Output:
        - saved_count:  int, 所有样本累计保存的 BOX 总数
        - all_count:    int, 所有样本累计检查的 BOX 总数
    """
    # ---- 读取 JSON 映射 ----
    with open(emdb_pdb_json, 'r') as f:
        emdb_pdb_data = json.load(f)  # list[dict], 形如 [{"emd_47345": "9E01"}, ...]

    # ---- 分片 / Sharding ----
    emdb_pdb_data = apply_sharding(emdb_pdb_data, part_id, total_parts)
    print(f"[Shard {part_id}/{total_parts}] 本分片样本数 = {len(emdb_pdb_data)}")

    # ---- 构建有效 ID 交集 ----
    map_id_set = {x.split('.')[0].lower() for x in os.listdir(map_npz_folder)}  # 是pdb_id
    pdb_feature_id_set = {x.split('.')[0].lower() for x in os.listdir(pdb_feature_npz_folder)}
    pdb_label_id_set = {x.split('.')[0].lower() for x in os.listdir(pdb_label_npz_folder)}
    # set[str], 还需要 sample_root_path 目录下有对应子文件夹
    sample_id_set = {x.lower() for x in os.listdir(sample_root_path) if os.path.isdir(os.path.join(sample_root_path, x))}
    valid_id_set = map_id_set & pdb_feature_id_set & pdb_label_id_set & sample_id_set

    # ---- joblib 并行 ----
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_process_one_sample)(
            item,
            pdb_label_npz_folder, map_npz_folder, pdb_feature_npz_folder,
            sample_root_path,
            valid_id_set,
            window_size, stride,
            r_expand, all_pos_ratio, center_pos_ratio,
            cut_length,
            output_root_folder, name_list,
            overwrite_existing,
        )
        for item in emdb_pdb_data
    )

    # ---- 汇总统计 ----
    saved_count, skipped_count, all_count, fail_count = 0, 0, 0, 0 
    failed_samples = []
    # list[dict], 所有口袋的统计信息 (跨样本扁平汇总)
    all_pocket_stats = []
    for pdb_id, s, sk, a, ok, err, pstats in results:
        saved_count += s
        skipped_count += sk
        all_count += a
        if not ok:
            fail_count += 1
            failed_samples.append(f"  - {pdb_id}: {err}")
        else:
            all_pocket_stats.extend(pstats)

    if failed_samples:
        print(f"\n[Warning] 失败样本 ({fail_count}):")
        for line in failed_samples:
            print(line)

    actual_written = saved_count - skipped_count
    print(f"\n[统计] 合格块: {saved_count}, 实际写入: {actual_written}, 因已存在跳过: {skipped_count}")

    # ---- 口袋级别统计报告 / Pocket-level statistics report ----
    if all_pocket_stats:
        _print_pocket_stats(all_pocket_stats)

    return saved_count, all_count




def _print_pocket_stats(all_pocket_stats: list):
    """
    打印口袋级别的汇总统计: 均值、方差、百分位数.
    Print pocket-level aggregate statistics: mean, variance, percentiles.

    输入参数 / Input:
        - all_pocket_stats: list[dict], 每个元素包含:
            ligand_atom_count  (int) — 配体原子数
            pocket_atom_count  (int) — 结合口袋蛋白原子数
            envelope_side      (int) — 包络正方体边长 (体素)
            box_count          (int) — 切出的合格 BOX 数
    """
    # int, 口袋总数
    n = len(all_pocket_stats)
    # list[str], 需要统计的字段名
    keys = ['ligand_atom_count', 'pocket_atom_count', 'envelope_side', 'box_count']
    # dict[str, str], 字段名到中文描述的映射
    key_labels = {
        'ligand_atom_count':  '配体原子数 / Ligand Atom Count',
        'pocket_atom_count':  '口袋蛋白原子数 / Pocket Protein Atom Count',
        'envelope_side':      '包络正方体边长(体素) / Envelope Cube Side (voxels)',
        'box_count':          '切出的BOX数 / Qualified BOX Count',
    }
    # list[int], 要计算的百分位列表
    percentile_list = list(range(0, 101, 10))  # [0, 10, 20, ..., 100]

    print(f"\n{'='*70}")
    print(f"  口袋级别统计 / Pocket-Level Statistics  (共 {n} 个口袋)")
    print(f"{'='*70}")

    for key in keys:
        # numpy, (n,), float64, 当前字段的所有值
        vals = np.array([s[key] for s in all_pocket_stats], dtype=np.float64)
        # 跳过全部为 -1 (无效) 的字段
        valid = vals[vals >= 0]
        if len(valid) == 0:
            print(f"\n  [{key_labels[key]}]  — 无有效数据")
            continue
        # float, 均值
        mean_val = np.mean(valid)
        # float, 方差
        var_val = np.var(valid)
        # numpy, (len(percentile_list),), float64, 百分位值
        pct_vals = np.percentile(valid, percentile_list)

        print(f"\n  [{key_labels[key]}]")
        print(f"    均值 Mean    = {mean_val:.2f}")
        print(f"    方差 Var     = {var_val:.2f}")
        pct_strs = [f"P{p}={v:.1f}" for p, v in zip(percentile_list, pct_vals)]
        print(f"    百分位 Pctl  = {', '.join(pct_strs)}")
    print(f"{'='*70}\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="对 bind.py 输出的 npz 做口袋中心裁剪, 生成训练用 BOX")

    # ---- 输入路径 ----
    parser.add_argument("--emdb_pdb_jsons",       type=str, nargs='+',
                        default=["/home/penghongen/My_Project/Data/split/3.5_cc_qscore/all.json"],
                        help="一个或多个 .json 映射文件路径 (空格分隔)")
    parser.add_argument("--pdb_label_npz_folder",  type=str, default="/storage/penghongen/Pocket_classic/pdb_label_npz")
    parser.add_argument("--map_npz_folder",        type=str, default="/storage/penghongen/Pocket_classic/emdb_npz")
    parser.add_argument("--pdb_feature_npz_folder",type=str, default="/storage/penghongen/Pocket_classic/pdb_feature_npz")
    parser.add_argument("--sample_root_path",      type=str, default="/home/penghongen/My_Project/Data/parsed_pdb/",
                        help="PDB_processor/run_preprocess.py 输出的样本根目录 (含 {pdb_id}/labels.npz)")

    # ---- 裁剪参数 ----
    parser.add_argument("--window_size",       type=int,   default=72)
    parser.add_argument("--stride",            type=int,   default=12)
    parser.add_argument("--r_expand",          type=float, default=2.0,
                        help="大立方体扩展系数: large_side = window_size + int(r * envelope_side)")
    parser.add_argument("--all_pos_ratio",     type=float, default=0.9,
                        help="a — BOX 正类总数 / 中心 BOX 正类总数 的最低比例")
    parser.add_argument("--center_pos_ratio",  type=float, default=0.9,
                        help="b — BOX inner 正类数 / 中心 BOX inner 正类数 的最低比例")
    parser.add_argument("--cut_length",        type=int,   default=12,
                        help="inner 区域的边缘裁剪宽度 (体素数)")

    # ---- 输出 ----
    parser.add_argument("--output_root_folder", type=str, default="/storage/penghongen/Pocket_classic/")
    parser.add_argument("--name_list",    type=str, nargs='+', default=["pdb_label_BOX", "emdb_BOX", "pdb_feature_BOX"],
                        help="与 *grids 顺序对应的输出子文件夹名")

    # ---- 分片与并行 (SLURM array job) ----
    parser.add_argument("--part_id",      type=int, default=0,  help="当前分片 ID (0-indexed)")
    parser.add_argument("--total_parts",  type=int, default=1,  help="总分片数")
    parser.add_argument("--n_jobs",       type=int, default=1,  help="joblib 并行进程数")
    parser.add_argument("--overwrite_existing", action="store_true", default=False,
                        help="覆盖输出目录中已有的 .npz 文件 (默认跳过已有文件)")

    args = parser.parse_args()

    total_saved, total_checked = 0, 0
    for emdb_pdb_json in args.emdb_pdb_jsons:
        saved_count, all_count = Split_Datas_into_Box(
            emdb_pdb_json,
            args.pdb_label_npz_folder, args.map_npz_folder, args.pdb_feature_npz_folder,
            args.sample_root_path,
            window_size=args.window_size, stride=args.stride,
            r_expand=args.r_expand,
            all_pos_ratio=args.all_pos_ratio,
            center_pos_ratio=args.center_pos_ratio,
            cut_length=args.cut_length,
            output_root_folder=args.output_root_folder, name_list=args.name_list,
            part_id=args.part_id, total_parts=args.total_parts,
            n_jobs=args.n_jobs,
            overwrite_existing=args.overwrite_existing,
        )
        total_saved += saved_count
        total_checked += all_count
        print(f"[Split_Datas_into_Box] {emdb_pdb_json} 裁剪完成: 检查 {all_count} 个块, 保存 {saved_count} 个。")

    print(f"\n{'='*60}")
    print(f"[Done] 总计检查 {total_checked} 个块, 保存 {total_saved} 个合格块。")
    print(f"{'='*60}")


