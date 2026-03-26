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


def vitualiza_grid_VS_map(file_path, data, original_map_path, current_voxel_size=None):
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
