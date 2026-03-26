import argparse
import gc
import json
import traceback
import numpy as np
import os
import warnings
import sys
from pathlib import Path
from joblib import Parallel, delayed

_BINDER_DIR = Path(__file__).resolve().parent
_POCKET_ROOT = _BINDER_DIR.parent
_PROJECT_ROOT = _POCKET_ROOT.parent

for _p in (str(_PROJECT_ROOT), str(_POCKET_ROOT), str(_BINDER_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Pocket.utils.network_tools import apply_sharding, atomic_np_savez
warnings.filterwarnings("ignore")



def check_box_relative(
    label_block, center_pos_count, center_inner_pos_count,
    cut_length=12, all_pos_ratio=0.9, center_pos_ratio=0.9
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
    class_ids_per_pocket,    # list[int], 每个口袋的类别 ID
    class_names_per_pocket,  # list[str], 每个口袋的类别名称
    *grids,

    window_size: int = 72, stride: int = 12,
    r_expand: float = 2.0,
    edge_expand: float = 0.4,
    all_pos_ratio: float = 0.9,
    center_pos_ratio: float = 0.9,
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
        - class_ids_per_pocket:     list[int], 长度 N_pocket, 每个口袋的类别 ID(目前为1~4)， 用来构造文件名
        - class_names_per_pocket:   list[str], 长度 N_pocket, 每个口袋的类别名称(如"metal_ion")， 用来构造文件名

        - pocket_centers_world:     numpy, (N_pockets, 3), float, 各口袋中心的世界坐标 (x,y,z), 单位 Å
        - pocket_atom_coords_list:  list[numpy], 长度 N_pockets, 每个元素 (M_i, 3) float, 代表第 i 个口袋中所有结合原子的世界坐标
        - instance_ids_per_pocket:  list[int], 长度 N_pockets, 每个口袋对应的配体实例 ID. 仅用于发送消息/文件名
        - *grids:                   一系列 numpy 数组, 每个形状 (..., D, H, W). grids[0] 是 label 用于检验

        - window_size:              int, 滑动窗口边长 (体素数), 默认 72, 它和 r_expand 都最好为偶数, 使中心小BOX位于0_0_0, 便于分阶段训练
        - stride:                   int, 滑动窗口步长 (体素数), 默认 12
        - r_expand:                 float, 大立方体扩展系数, 大立方体边长 = window_size + int(r_expand * envelope_side)
        - edge_expand:              float, 大立方体基础边长比例, 默认 0.4, 大立方体边长从 window_size * edge_expand 开始计算
        - all_pos_ratio:            float, 当前 BOX 正类总数 / 中心 BOX 正类总数 的最低比例
        - center_pos_ratio:         float, 当前 BOX inner 正类数在总正类中的占比 / 中心 BOX inner 正类数在总正类中的占比 的最低比例
        - cut_length:               int, inner 区域裁掉的边缘宽度 (体素数)

        - output_root_folder:       str, 输出根文件夹
        - name_list:                list[str], 子文件夹名, name_list[i] 对应 grids[i] 保存的文件夹名, 暂时为["pdb_label_BOX", "emdb_BOX", "pdb_feature_BOX"]
        - overwrite_existing:       bool, 是否覆盖已有 .npz
        - pocket_atom_counts:       list[int] | None, 长度 N_pockets, 每个口袋的蛋白质结合原子数 (用于统计)
        - ligand_atom_counts:       list[int] | None, 长度 N_pockets, 每个口袋对应配体的原子数 (用于统计)

    输出 / Output:
        - saved_count:      int, 累计合格且写入的 BOX 数
        - skipped_count:    int, 因文件已存在而跳过的 BOX 数
        - all_count:        int, 累计检查过的 BOX 数
        - pocket_stats_list: list[dict], 每个口袋的统计信息, 含 ligand_atom_count / pocket_atom_count / envelope_x / envelope_y / envelope_z / box_count
        - class_voxel_stats: dict, 各类别正类体素统计, 例如 {class_name: {'pos_voxels': int, 'total_voxels': int}}
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
        return 0, 0, 0, [], {}

    # ================================================
    # 遍历每个口袋 / Iterate over each pocket
    # ================================================
    # list[dict], 每个口袋的统计信息
    pocket_stats_list = []
    # dict, 每个类别的正类体素统计: {class_name: {'pos_voxels': int, 'total_voxels': int}}
    class_voxel_stats = {}
    for pocket_idx in range(n_pockets):
        # int, 当前口袋的配体实例 ID (global_id)
        inst_id = instance_ids_per_pocket[pocket_idx]
        # int, 当前口袋的类别 ID
        class_id = class_ids_per_pocket[pocket_idx]
        # str, 当前口袋的类别名称
        class_name = class_names_per_pocket[pocket_idx]
        
        # numpy, (3,), float, 当前口袋中心的世界坐标 (x,y,z)
        pocket_center_world = pocket_centers_world[pocket_idx]
        # numpy, (M_i, 3), float, 当前口袋的结合原子世界坐标
        pocket_atoms_world = pocket_atom_coords_list[pocket_idx]
        # numpy, (3,), int, 口袋中心体素坐标 (x,y,z 顺序; 向下取整)
        pocket_center_voxel = np.floor( (pocket_center_world - origin) / voxel_size ).astype(int)
        # numpy, (M_i, 3), int, 口袋原子的体素坐标
        pocket_atoms_voxel = np.floor( (pocket_atoms_world - origin) / voxel_size ).astype(int)


        # ---- 计算包络长方体 (envelope cuboid) ----
        # numpy, (3,), int, 各轴最小体素坐标 (顺序: x, y, z)
        vmin = np.min(pocket_atoms_voxel, axis=0)
        # numpy, (3,), int, 各轴最大体素坐标 (顺序: x, y, z)
        vmax = np.max(pocket_atoms_voxel, axis=0)
        # numpy, (3,), int, 包络长方体各轴跨度 (体素数, 顺序: x, y, z); 原先是取 max 变为正方体，现在保留各轴独立跨度
        envelope_rect = (vmax - vmin).astype(int)  # shape: (3,)


        # ---- 计算大长方体 (large cuboid) ----
        # numpy, (3,), int, 大长方体各轴边长: 各轴分别 = window_size*edge_expand + r_expand * envelope_rect[i]
        # 原先用单一 large_cube_side（各轴相同），现在各轴独立，精确覆盖该口袋的实际跨度
        large_rect = np.array([
            int(window_size * edge_expand + r_expand * envelope_rect[0]),  # x 轴边长
            int(window_size * edge_expand + r_expand * envelope_rect[1]),  # y 轴边长
            int(window_size * edge_expand + r_expand * envelope_rect[2]),  # z 轴边长
        ], dtype=int)  # shape: (3,)
        # int, 大长方体在 x 轴的起始体素 (未裁剪)
        large_start_x = pocket_center_voxel[0] - large_rect[0] // 2
        # int, 大长方体在 y 轴的起始体素 (未裁剪)
        large_start_y = pocket_center_voxel[1] - large_rect[1] // 2
        # int, 大长方体在 z 轴的起始体素 (未裁剪)
        large_start_z = pocket_center_voxel[2] - large_rect[2] // 2
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
        # 仅取当前类别对应的通道 / Select channel for current class
        # (1, D, H, W) -> (D, H, W) binary mask
        center_label_mask = (center_label_block[0] == class_id)
        # int, 中心 BOX 的正类总体素数
        center_pos_count = int(np.sum(center_label_mask > 0))
        center_inner = center_label_mask[..., cut_length:-cut_length, cut_length:-cut_length, cut_length:-cut_length]
        center_inner_pos_count = int(np.sum(center_inner > 0))



        # ---- 构造滑动窗口坐标列表 ----
        # int, 大长方体在各轴的合法起始范围
        slide_min_x, slide_min_y, slide_min_z = max(0, large_start_x), max(0, large_start_y), max(0, large_start_z)
        slide_max_x = min(grid_max_x - window_size, large_start_x + large_rect[0] - window_size)
        slide_max_y = min(grid_max_y - window_size, large_start_y + large_rect[1] - window_size)
        slide_max_z = min(grid_max_z - window_size, large_start_z + large_rect[2] - window_size)

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
                    # bool, 是否为中心 BOX (由 _make_positions 明确插入的中心起始位置)
                    # 用起始坐标判断而非 Rxx==0 判断: np.floor 取整可能导致中心 BOX 的偏移量不为 (0,0,0)
                    is_center_box = (xx == c_sx and yy == c_sy and zz == c_sz)

                    for num, grid in enumerate(grids):
                        assert grid.shape[-3:] == (Z, Y, X), \
                            f"输入的第 {num} 个图尺寸不匹配, 应为 {Z}x{Y}x{X}, 实际为 {grid.shape[-3:]}"
                        # numpy, (..., window_size, window_size, window_size), 当前 BOX 切片
                        block = grid[..., zz:zz + window_size, yy:yy + window_size, xx:xx + window_size]

                        # numpy, (3,), int, 尺寸校验：剔除因越界导致非标准形状的BOX
                        if block.shape[-3:] != (window_size, window_size, window_size):
                            break

                        # float tuple, 当前 BOX 在世界坐标下的范围
                        x_range = (origin[0] + xx * voxel_size[0], origin[0] + (xx + window_size) * voxel_size[0])
                        y_range = (origin[1] + yy * voxel_size[1], origin[1] + (yy + window_size) * voxel_size[1])
                        z_range = (origin[2] + zz * voxel_size[2], origin[2] + (zz + window_size) * voxel_size[2])
                        # tuple(float,float,float), 当前 BOX 原点的世界坐标
                        global_origin = (origin[0] + xx * voxel_size[0],
                                         origin[1] + yy * voxel_size[1],
                                         origin[2] + zz * voxel_size[2])
                        if num == 0:  # label — 用于检验
                            # 生成该 class_id 的二值掩码
                            label_mask = (block[0] == class_id)
                            
                            if not check_box_relative(label_mask, center_pos_count, center_inner_pos_count,
                                                      cut_length=cut_length,
                                                      all_pos_ratio=all_pos_ratio,
                                                      center_pos_ratio=center_pos_ratio):
                                break  # 不合格, 跳过所有 grids
                            else:
                                saved_count += 1
                                pocket_save_count += 1
                                # 累加本类别的正类体素统计
                                _cs = class_voxel_stats.setdefault(class_name, {'pos_voxels': 0, 'total_voxels': 0})
                                _cs['pos_voxels'] += int(np.sum(label_mask > 0))
                                _cs['total_voxels'] += label_mask.size
                        
                        # ---- 文件名: {pdb_id}_{instance_id}_{Rxx}_{Ryy}_{Rzz}[_C].npz ----
                        # 中心 BOX 额外加 _C 后缀, 方便下游按需单独提取"中心子集"
                        # 保存到类别子文件夹: output_root/name/class_name/file.npz
                        center_suffix = '_C' if is_center_box else ''
                        path = os.path.join(
                            output_root_folder,
                            f'{name_list[num]}',
                            class_name, # class subfolder
                            f'{pdb_id}_{inst_id}_{Rxx}_{Ryy}_{Rzz}{center_suffix}.npz'
                        )
                        
                        os.makedirs(os.path.dirname(path), exist_ok=True)
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



        # ---- 收集口袋级统计 ----
        pocket_stats_list.append({
            'ligand_atom_count':  ligand_atom_counts[pocket_idx] if ligand_atom_counts is not None else -1,
            'pocket_atom_count':  pocket_atom_counts[pocket_idx] if pocket_atom_counts is not None else -1,
            # 包络长方体三轴各自的体素跨度
            'envelope_x':         int(envelope_rect[0]),  # int, x 轴跨度 (体素数)
            'envelope_y':         int(envelope_rect[1]),  # int, y 轴跨度 (体素数)
            'envelope_z':         int(envelope_rect[2]),  # int, z 轴跨度 (体素数)
            'box_count':          pocket_save_count,
        })

    return saved_count, skipped_count, all_count, pocket_stats_list, class_voxel_stats




# ------------------------------------------------------------------ #
#  单样本处理函数 (供 joblib 并行调用)
# ------------------------------------------------------------------ #
def _process_one_sample(
    item,

    pdb_label_npz_folder, map_npz_folder, pdb_feature_npz_folder,
    sample_root_path,
    valid_id_set,

    window_size, stride,
    r_expand, edge_expand, all_pos_ratio, center_pos_ratio,
    cut_length,

    sample_box_num,
    hardmask_upper_required,

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
        - edge_expand:              float, 大立方体基础边长比例
        - all_pos_ratio:            float, a — BOX 正类总数 / 中心 BOX 正类数总数  的最低比例
        - center_pos_ratio:         float, b — BOX inner 正类数占比 / 中心 BOX inner 正类数占比  的最低比例
        - cut_length:               int, inner 区域裁掉的边缘宽度

        - sample_box_num:           int, 意欲在每张密度图中裁剪的随机box数目(会探查至多十倍), 这些BOX将放在与"metal_ion"等同级别的 "sample_BOX"下
        - hardmask_upper_required:  float, 在随机选取BOX时, 只有存在原子的体素比例(hardmask>0) > hardmask_upper_required 的BOX才会被接受

        - output_root_folder:       str, 保存的根文件夹路径
        - name_list:                list[str], 保存的文件夹名列表, name_list[i] 对应 grids[i]
        - overwrite_existing:       bool, 是否覆盖已有 .npz

    保存的.npz:
        - 命名规范: output_root_folder/{name_list[i]}/{class_name}/{pdb_id}_{inst_id}_{Rxx}_{Ryy}_{Rzz}[_C].npz, 当 随机切box时inst_id=-1
        - 键值对:
            - grid:              numpy.ndarray, (window_size, window_size, window_size), float32, 密度图
            - x_range:           numpy.ndarray, (2,), float32, x 轴范围(世界坐标)
            - y_range:           numpy.ndarray, (2,), float32, y 轴范围
            - z_range:           numpy.ndarray, (2,), float32, z 轴范围
            - voxel_size:        numpy.ndarray, (3,), float32, 体素大小
            - origin:            numpy.ndarray, (3,), float32, 世界坐标系下xyz最小的(左下角近点)坐标

    返回 / Output:
        - tuple: (pdb_id, saved_count, skipped_count, all_count, success, error_msg, pocket_stats_list, class_voxel_stats)
            - pdb_id:            str, 当前样本的 PDB ID (小写)
            - saved_count:       int, 本样本合格的 BOX 数
            - skipped_count:     int, 本样本因文件已存在而跳过的 BOX 数
            - all_count:         int, 本样本检查过的 BOX 总数
            - success:           bool, 是否处理成功
            - error_msg:         str|None, 失败时的错误信息; 跳过时为 'skipped'; 成功时为 None
            - pocket_stats_list: list[dict], 每个口袋的统计 (ligand_atom_count, pocket_atom_count, envelope_side, box_count)
            - class_voxel_stats: dict, 各类别正类体素统计 {class_name: {'pos_voxels': int, 'total_voxels': int}}
    """
    # str, 小写 PDB ID
    pdb_id = list(item.values())[0].lower()
    if pdb_id not in valid_id_set:
        return (pdb_id, 0, 0, 0, True, 'skipped', [], {})

    try:
        map_path = os.path.join(map_npz_folder, f'{pdb_id}.npz')                    # str, 密度图 npz 路径
        pdb_feature_path = os.path.join(pdb_feature_npz_folder, f'{pdb_id}.npz')    # str, PDB 特征图 npz 路径
        pdb_label_path = os.path.join(pdb_label_npz_folder, f'{pdb_id}.npz')        # str, PDB 标签图 npz 路径
        
        labels_npz_path = os.path.join(sample_root_path, pdb_id, 'labels.npz')      # str, 原始标签文件路径
        atoms_npz_path = os.path.join(sample_root_path, pdb_id, 'atoms.npz')        # str, 原始原子文件路径
        if not os.path.exists(labels_npz_path) or not os.path.exists(atoms_npz_path):
            return (pdb_id, 0, 0, 0, True, 'skipped (no labels/atoms)', [], {})


        labels_data = np.load(labels_npz_path, allow_pickle=True)
        # numpy, (N_ligands, 3), float32, 各口袋中心世界坐标
        pocket_centers = labels_data['pocket_centers']
        # numpy, (N_atoms,), int32, 每个原子的结合位点实例 ID (-1=非结合)
        instance_ids = labels_data['instance_ids']
        # numpy, (N_atoms,), int32, 每个原子对应的原始类别 ID
        pocket_class_ids_all = labels_data['pocket_class_ids']
        # str, 类别名称映射字符串, 暂时为{1: 'metal_ion', 2: 'peptide', 3: 'nucleic_acid', 4: 'small_molecule'}
        raw_map = labels_data['pocket_class_name_map']
        if isinstance(raw_map, np.ndarray):
            raw_map = str(raw_map)
        # 解析类别名映射 (注意: 这里的 mapping 是 cid -> cname)
        class_name_dict = {}
        for bit in raw_map.split(','):
            if ':' in bit:
                cid, cname = bit.split(':', 1)
                class_name_dict[int(cid)] = cname.strip()
        # numpy.ndarray, (N_atoms, 3), float32, 重原子世界坐标
        atom_coords = np.load(atoms_npz_path)['coords']


        # ---- 提取每个口袋的信息 ----
        # numpy.ndarray, (N_atoms,), bool, 标记参与结合的原子
        binding_mask = instance_ids != -1
        # numpy.ndarray, (K,), int32, 有效的实例 ID 列表（去重后的 candidate_id）
        unique_inst_ids = np.unique(instance_ids[binding_mask])
        if len(unique_inst_ids) == 0:
            labels_data.close()
            return (pdb_id, 0, 0, 0, True, 'skipped (no binding site)', [], {})

        # numpy.ndarray, (N_ligands,), int32, 每个选中配体的 candidate_id（按升序）
        ligand_candidate_ids = labels_data['ligand_candidate_ids']
        # dict[int, int], candidate_id → pocket_centers 中对应的索引
        cand_id_to_center_idx = {int(cid): i for i, cid in enumerate(ligand_candidate_ids)}

        # list[numpy], 每个口袋的结合原子世界坐标, 元素形状 (M_i, 3), 长度为有口袋原子的ligand数目 N_pockets_matched
        pocket_atom_coords_list = []
        # list[int], (N_pockets_matched,), 每个口袋的实例 ID
        instance_ids_per_pocket = []
        # list[numpy], (N_pockets_matched,), 对应的口袋中心 (从 pocket_centers 取)
        pocket_centers_matched = []

        # list[int], (N_pockets_matched,), 每个口袋的蛋白质结合原子数
        pocket_atom_counts = []
        # list[int], (N_pockets_matched,), 每个口袋对应的配体原子数
        ligand_atom_counts = []
        # list[int], (N_pockets_matched,), 每个口袋的类别 ID
        class_ids_per_pocket = []
        # list[str], (N_pockets_matched,), 每个口袋的类别名称
        class_names_per_pocket = []
        
        for inst_id in unique_inst_ids:
            # int, 当前循环处理的配体全局实例索引(就是candidate_id; 从0开始; 唯一但不连续; 见instance_labels.py)
            inst_id_int = int(inst_id)
            # numpy.ndarray, (N_atoms,), bool, 定位当前实例(口袋)所属原子的掩码
            inst_mask = (instance_ids == inst_id_int)
            # numpy.ndarray, (M_i, 3), float32, 该实例包含的蛋白质结合原子的坐标
            coords_i = atom_coords[inst_mask]
            if len(coords_i) == 0:
                continue
            
            # numpy.ndarray, (M_i,), int32, 这些原子的类别列表
            cids = pocket_class_ids_all[inst_mask]
            # int, 口袋原始 ID(使用原子所属类别的众数作为该口袋的原始类别)
            raw_class_id = int(np.bincount(cids).argmax())
            # 过滤背景类
            if raw_class_id == 0:
                continue
            # str, 类别名称 (优先从 labels.npz 的映射表中取)
            this_class_name = class_name_dict.get(raw_class_id, f"class_{raw_class_id}")
            

            pocket_atom_coords_list.append(coords_i)
            instance_ids_per_pocket.append(inst_id_int)
            class_ids_per_pocket.append(raw_class_id)
            class_names_per_pocket.append(this_class_name)
            pocket_atom_counts.append(len(coords_i))
            
            # 提取配体原子坐标
            lig_coords_key = f'ligand_coords_{inst_id_int}'
            if lig_coords_key in labels_data:
                # numpy.ndarray, (L_i, 3), float32, 配体分子原子的坐标
                lig_coords = labels_data[lig_coords_key]
                ligand_atom_counts.append(len(lig_coords))
            else:
                raise ValueError(f"ligand_coords_{inst_id_int} not found in labels.npz! 检查 instance_labels.py 是否跑了个坏文件.")
            
            # int | None, 该 candidate_id 在 pocket_centers 中的行号（None 代表不存在，理论上不应发生）
            center_idx = cand_id_to_center_idx.get(inst_id_int)
            if center_idx is not None:
                # numpy.ndarray, (3,), float32, 当前口袋中心的世界坐标
                pocket_centers_matched.append(pocket_centers[center_idx])
            else:
                # 如果发生，说明上游输出的 unique_inst_ids 中存在，却没有对应中心点，数据已损坏
                raise ValueError(f"候选配体 ID {inst_id_int} 在 pocket_centers 映射中不存在，请检查上游 labels.npz")

        labels_data.close()
        if len(instance_ids_per_pocket) == 0:
            return (pdb_id, 0, 0, 0, True, 'skipped (all pockets are background)', [], {})

        # numpy.ndarray, (N_filtered, 3), float32, 最终确定的口袋中心点集合
        pocket_centers_matched = np.array(pocket_centers_matched, dtype=np.float32)

        # ---- 加载预先绑定的体素 NPZ 数据 ----
        m_item = np.load(map_path)
        # numpy.ndarray, (1, D, H, W), float32, 密度图全图体素数据 (增加 Channel 维度)
        map_grid = m_item['grid'][None, ...]
        # numpy.ndarray, (3,), float32, 原点 (Å)
        origin_arr = m_item['origin']
        # numpy.ndarray, (3,), float32, 步长/分辨率 (Å)
        voxel_size_arr = m_item['voxel_size']
        m_item.close()

        # numpy.ndarray, (C, D, H, W), float32, 蛋白质特征网格图
        pdb_feature_grid = np.load(pdb_feature_path)['grid']
        # numpy.ndarray, (1, D, H, W), int32/float32, 包含类别 ID 的掩码图
        pdb_label_grid = np.load(pdb_label_path)['grid']

        # ---- 调用核心裁剪逻辑 ----
        saved, skipped, checked, pocket_stats, class_voxel_stats = Split_Data_into_Box_PocketCentered(
            origin_arr, voxel_size_arr, pdb_id,
            pocket_centers_matched,
            pocket_atom_coords_list,
            instance_ids_per_pocket,
            class_ids_per_pocket,
            class_names_per_pocket,

            pdb_label_grid, map_grid, pdb_feature_grid,
            window_size=window_size, stride=stride,
            r_expand=r_expand,
            edge_expand=edge_expand,
            all_pos_ratio=all_pos_ratio,
            center_pos_ratio=center_pos_ratio,
            cut_length=cut_length,
            output_root_folder=output_root_folder, name_list=name_list,
            overwrite_existing=overwrite_existing,
            pocket_atom_counts=pocket_atom_counts,
            ligand_atom_counts=ligand_atom_counts,
        )


        # -------------------------------- random_BOX 随机采样 --------------------------------
        if sample_box_num > 0:
            # int, 体素网格的空间形状
            Z, Y, X = map_grid.shape[-3:]
            # numpy, (D, H, W), bool, hardmask: 特征图中有原子的体素
            hardmask_full = np.any(pdb_feature_grid != 0, axis=0)
            # int, 已保存的 random_BOX 计数
            rand_saved = 0
            # int, 累计 hardmask 非零体素数 / 总体素数
            rand_pos_voxels = 0
            rand_total_voxels = 0
            # list[numpy], 按 name_list 顺序排列的全图网格
            all_grids = [pdb_label_grid, map_grid, pdb_feature_grid]

            # int, 最大尝试次数 = 目标数 * 10
            max_attempts = sample_box_num * 10

            for attempt_idx in range(max_attempts):
                if rand_saved >= sample_box_num:
                    break
                # int, 随机采样窗口起始坐标
                rx = np.random.randint(0, max(1, X - window_size + 1))
                ry = np.random.randint(0, max(1, Y - window_size + 1))
                rz = np.random.randint(0, max(1, Z - window_size + 1))
                # numpy, (window_size, window_size, window_size), bool, 当前 BOX 的 hardmask 切片
                hm_block = hardmask_full[rz:rz + window_size, ry:ry + window_size, rx:rx + window_size]
                if hm_block.shape != (window_size, window_size, window_size):
                    continue
                # float, hardmask 中有原子体素的比例
                hm_ratio = np.sum(hm_block) / hm_block.size
                if hm_ratio < hardmask_upper_required:
                    continue

                # 合格, 保存
                rand_pos_voxels += int(np.sum(hm_block))
                rand_total_voxels += hm_block.size
                for num, grid in enumerate(all_grids):
                    # numpy, (..., window_size, window_size, window_size), 当前 BOX 切片
                    block = grid[..., rz:rz + window_size, ry:ry + window_size, rx:rx + window_size]
                    if block.shape[-3:] != (window_size, window_size, window_size):
                        break
                    # float tuple, 当前 BOX 在世界坐标下的范围
                    x_range = (origin_arr[0] + rx * voxel_size_arr[0], origin_arr[0] + (rx + window_size) * voxel_size_arr[0])
                    y_range = (origin_arr[1] + ry * voxel_size_arr[1], origin_arr[1] + (ry + window_size) * voxel_size_arr[1])
                    z_range = (origin_arr[2] + rz * voxel_size_arr[2], origin_arr[2] + (rz + window_size) * voxel_size_arr[2])
                    # tuple(float,float,float), 当前 BOX 原点的世界坐标
                    global_origin = (origin_arr[0] + rx * voxel_size_arr[0], origin_arr[1] + ry * voxel_size_arr[1], origin_arr[2] + rz * voxel_size_arr[2])
                    # str, 保存路径: output_root / name(如"pdb_feature_BOX") / random_BOX / {pdb_id}_-1_{rx}_{ry}_{rz}.npz
                    path = os.path.join(
                        output_root_folder,
                        name_list[num],
                        'random_BOX',
                        f'{pdb_id}_-1_{rx}_{ry}_{rz}.npz'
                    )
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    if not overwrite_existing and os.path.exists(path):
                        continue
                    atomic_np_savez(path, do_not_replace=False,
                                    grid=block,
                                    x_range=np.array(x_range),
                                    y_range=np.array(y_range),
                                    z_range=np.array(z_range),
                                    voxel_size=np.array(voxel_size_arr),
                                    origin=np.array(global_origin))  

                rand_saved += 1

            saved += rand_saved
            checked += rand_saved
            if rand_total_voxels > 0:
                class_voxel_stats['random_BOX'] = {
                    'pos_voxels': rand_pos_voxels,
                    'total_voxels': rand_total_voxels,
                }

        del map_grid, pdb_feature_grid, pdb_label_grid, atom_coords, pocket_atom_coords_list
        gc.collect()

        return (pdb_id, saved, skipped, checked, True, None, pocket_stats, class_voxel_stats)

    except Exception as e:
        traceback.print_exc()
        return (pdb_id, 0, 0, 0, False, f"{type(e).__name__}: {e}", [], {})



def Split_Datas_into_Box(
    emdb_pdb_json: str, 
    pdb_label_npz_folder: str, map_npz_folder: str, pdb_feature_npz_folder: str, 
    sample_root_path: str,

    window_size: int, stride: int, 
    r_expand: float,
    edge_expand: float,
    all_pos_ratio: float,
    center_pos_ratio: float,
    cut_length: int,

    sample_box_num: int,
    hardmask_upper_required: float,

    output_root_folder: str, name_list: list,
    
    part_id: int = 0, total_parts: int = 1,
    n_jobs: int = 1,
    overwrite_existing: bool = False,
):
    """
    针对 bind.py 的输出文件批量应用口袋中心裁剪.
    支持 SLURM 分片 (apply_sharding) + CPU 并行 (joblib.Parallel).

    输入参数 / Input:
        - emdb_pdb_json:          str, 映射 JSON 路径
        - pdb_label_npz_folder:   str, 标签 NPZ 目录
        - map_npz_folder:         str, 密度图 NPZ 目录
        - pdb_feature_npz_folder: str, 特征图 NPZ 目录
        - sample_root_path:       str, 原始 labels.npz 所在根目录
        - window_size:            int, 块边长
        - stride:                 int, 步长
        - r_expand:               float, 扩展系数
        - edge_expand:            float, 大立方体基础边长比例
        - all_pos_ratio:          float, 过滤阈值 a
        - center_pos_ratio:       float, 过滤阈值 b
        - cut_length:             int, inner 裁剪量
        - output_root_folder:     str, 根输出目录
        - name_list:              list[str], 输出子文件夹名
        - part_id:                int, 分片 ID
        - total_parts:            int, 总分片数
        - n_jobs:                 int, 并行数
        - overwrite_existing:     bool, 是否覆盖
        - sample_box_num:         int, 每张密度图尝试的随机 BOX 数
        - hardmask_upper_required: float, random_BOX 中 hardmask 比例的最低阈值

    返回 / Output:
        - saved_count:  int, 总保存块数
        - all_count:    int, 总检查块数
    """
    # list[dict], 读取 JSON 映射
    with open(emdb_pdb_json, 'r') as f:
        emdb_pdb_data = json.load(f)

    # list[dict], 应用分片
    emdb_pdb_data = apply_sharding(emdb_pdb_data, part_id, total_parts)
    print(f"[Shard {part_id}/{total_parts}] 本分片样本数 = {len(emdb_pdb_data)}")

    # 获取现有文件 ID 交集
    map_id_set = {x.split('.')[0].lower() for x in os.listdir(map_npz_folder)}
    pdb_feature_id_set = {x.split('.')[0].lower() for x in os.listdir(pdb_feature_npz_folder)}
    pdb_label_id_set = {x.split('.')[0].lower() for x in os.listdir(pdb_label_npz_folder)}
    sample_id_set = {x.lower() for x in os.listdir(sample_root_path) if os.path.isdir(os.path.join(sample_root_path, x))}
    # set[str], 同时在四个源目录中存在的 PDB ID
    valid_id_set = map_id_set & pdb_feature_id_set & pdb_label_id_set & sample_id_set

    # joblib.Parallel 运行
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_process_one_sample)(
            item,
            pdb_label_npz_folder, map_npz_folder, pdb_feature_npz_folder,
            sample_root_path,
            valid_id_set,
            window_size, stride,
            r_expand, edge_expand, all_pos_ratio, center_pos_ratio,
            cut_length,
            sample_box_num=sample_box_num,
            hardmask_upper_required=hardmask_upper_required,
            output_root_folder=output_root_folder,
            name_list=name_list,
            overwrite_existing=overwrite_existing,
        )
        for item in emdb_pdb_data
    )

    # ---- 汇总统计 ----
    saved_count, skipped_count, all_count, fail_count = 0, 0, 0, 0 
    failed_samples = []
    # list[dict], 所有口袋的统计信息 (跨样本扁平汇总)
    all_pocket_stats = []
    # dict, 跨样本汇总的类别体素统计: {class_name: {'pos_voxels': int, 'total_voxels': int}}
    agg_class_voxel_stats = {}
    for res in results:
        # tuple, 解包单样本处理结果
        pdb_id, s, sk, a, ok, err, pstats, cvs = res
        # int, 累加各种计数
        saved_count += s
        skipped_count += sk
        all_count += a
        
        if not ok:
            # bool, 标记是否发生异常
            fail_count += 1
            failed_samples.append(f"  - {pdb_id}: {err}")
        else:
            # list[dict], 合并汇总各口袋统计
            all_pocket_stats.extend(pstats)
            # 汇总类别体素统计
            for cname, cdata in cvs.items():
                agg = agg_class_voxel_stats.setdefault(cname, {'pos_voxels': 0, 'total_voxels': 0})
                agg['pos_voxels'] += cdata['pos_voxels']
                agg['total_voxels'] += cdata['total_voxels']

    if failed_samples:
        print(f"\n[Warning] 失败样本 ({fail_count}):")
        for line in failed_samples:
            print(line)

    actual_written = saved_count - skipped_count
    print(f"\n[统计] 合格块: {saved_count}, 实际写入: {actual_written}, 因已存在跳过: {skipped_count}")

    # ---- 口袋级别统计报告 / Pocket-level statistics report ----
    if all_pocket_stats:
        _print_pocket_stats(all_pocket_stats)

    # ---- 类别体素比例统计 / Class voxel ratio statistics ----
    if agg_class_voxel_stats:
        _print_class_voxel_stats(agg_class_voxel_stats)

    return saved_count, all_count




def _print_pocket_stats(all_pocket_stats: list):
    """
    打印口袋级别的汇总统计: 均值、方差、百分位数.
    Print pocket-level aggregate statistics: mean, variance, percentiles.

    输入参数 / Input:
        - all_pocket_stats: list[dict], 每个元素包含:
            ligand_atom_count  (int) — 配体原子数
            pocket_atom_count  (int) — 结合口袋蛋白原子数
            envelope_x         (int) — 包络长方体 x 轴跨度 (体素)
            envelope_y         (int) — 包络长方体 y 轴跨度 (体素)
            envelope_z         (int) — 包络长方体 z 轴跨度 (体素)
            box_count          (int) — 切出的合格 BOX 数
    """
    # int, 口袋总数
    n = len(all_pocket_stats)
    # list[str], 需要统计的字段名 (envelope_side 拆分为三轴)
    keys = ['ligand_atom_count', 'pocket_atom_count', 'envelope_x', 'envelope_y', 'envelope_z', 'box_count']
    # dict[str, str], 字段名到中文描述的映射
    key_labels = {
        'ligand_atom_count':  '配体原子数 / Ligand Atom Count',
        'pocket_atom_count':  '口袋蛋白原子数 / Pocket Protein Atom Count',
        'envelope_x':         '包络长方体 X 跨度(体素) / Envelope Cuboid X (voxels)',
        'envelope_y':         '包络长方体 Y 跨度(体素) / Envelope Cuboid Y (voxels)',
        'envelope_z':         '包络长方体 Z 跨度(体素) / Envelope Cuboid Z (voxels)',
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



def _print_class_voxel_stats(agg_class_voxel_stats: dict):
    """
    打印各类别的正类体素比例统计 (切块完毕后调用一次).

    输入参数 / Input:
        - agg_class_voxel_stats: dict, {class_name: {'pos_voxels': int, 'total_voxels': int}}
            对于普通类别, pos_voxels 是 label==class_id 的体素数
            对于 random_BOX, pos_voxels 是 hardmask!=0 的体素数
    """
    print(f"\n{'='*70}")
    print(f"  类别体素比例统计 / Class Voxel Ratio Statistics")
    print(f"{'='*70}")
    for cname in sorted(agg_class_voxel_stats.keys()):
        cdata = agg_class_voxel_stats[cname]
        # int, 正类体素总数
        pos = cdata['pos_voxels']
        # int, 总体素数
        total = cdata['total_voxels']
        # float, 比例
        ratio = pos / total if total > 0 else 0.0
        if cname == 'random_BOX':
            desc = 'hardmask!=0 体素比例'
        else:
            desc = '正类体素比例'
        print(f"  {cname}: {desc} = {ratio:.6f}  ({pos}/{total})")
    print(f"{'='*70}\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="对 bind.py 输出的 npz 做口袋中心裁剪, 生成训练用 BOX")

    # ---- 输入路径 ----
    parser.add_argument("--emdb_pdb_jsons",       type=str, nargs='+',
                        default=["/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v0/all.json"],
                        help="一个或多个 .json 映射文件路径 (空格分隔)")
    parser.add_argument("--pdb_label_npz_folder",  type=str, default="/storage/penghongen/Pocket_classic/v_0/pdb_label_npz")
    parser.add_argument("--map_npz_folder",        type=str, default="/storage/penghongen/Pocket_classic/v_0/emdb_npz")
    parser.add_argument("--pdb_feature_npz_folder",type=str, default="/storage/penghongen/Pocket_classic/v_0/pdb_feature_npz")
    parser.add_argument("--sample_root_path",      type=str, default="/home/penghongen/My_Project/Data/DATA_v0/parsed_pdb/",
                        help="PDB_processor/run_preprocess.py 输出的样本根目录 (含 {pdb_id}/labels.npz)")

    # ---- 裁剪参数 ----
    parser.add_argument("--window_size",       type=int,   default=72)
    parser.add_argument("--stride",            type=int,   default=28)
    parser.add_argument("--r_expand",          type=float, default=1.0,
                        help="大立方体扩展系数: large_side = window_size*edge_expand + int(r * envelope_side)")
    parser.add_argument("--edge_expand",       type=float, default=0.4,
                        help="大立方体基础边长比例: 默认是 window_size 的 0.4")
    parser.add_argument("--all_pos_ratio",     type=float, default=0.87,
                        help="a — BOX 正类总数 / 中心 BOX 正类总数 的最低比例")
    parser.add_argument("--center_pos_ratio",  type=float, default=0.93,
                        help="b — BOX inner 正类数 / 中心 BOX inner 正类数 的最低比例")
    parser.add_argument("--cut_length",        type=int,   default=12,
                        help="inner 区域的边缘裁剪宽度 (体素数)")

    # ---- 输出 ----
    parser.add_argument("--output_root_folder", type=str, default="/storage/penghongen/Pocket_classic/v_0/")
    parser.add_argument("--sample_box_num",    type=int, default=0,
                        help="每张密度图试图选取的 random_BOX 数 (最多探查 N*10 次; <=0 不采样)")
    parser.add_argument("--hardmask_upper_required", type=float, default=0.1,
                        help="random_BOX 中 hardmask(含有原子的体素)比例的最低阈值")
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
            edge_expand=args.edge_expand,
            all_pos_ratio=args.all_pos_ratio,
            center_pos_ratio=args.center_pos_ratio,
            cut_length=args.cut_length,
            output_root_folder=args.output_root_folder, name_list=args.name_list,
            part_id=args.part_id, total_parts=args.total_parts,
            n_jobs=args.n_jobs,
            overwrite_existing=args.overwrite_existing,
            sample_box_num=args.sample_box_num,
            hardmask_upper_required=args.hardmask_upper_required,
        )
        total_saved += saved_count
        total_checked += all_count
        print(f"[Split_Datas_into_Box] {emdb_pdb_json} 裁剪完成: 检查 {all_count} 个块, 保存 {saved_count} 个。")

    print(f"\n{'='*60}")
    print(f"[Done] 总计检查 {total_checked} 个块, 保存 {total_saved} 个合格块。")
    print(f"{'='*60}")


