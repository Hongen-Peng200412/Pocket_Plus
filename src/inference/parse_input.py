"""
parse_input.py - 数据加载与特征整合模块

负责将各种来源的数据整合为模型需要的输入格式。
支持两种加载方式：
  方式1. 从已处理的 .npz 文件（emdb_BOX / pdb_feature_BOX / pdb_label_BOX）加载（load_from_npz_dirs）——————移步到 inference.utils.pipline_for_box.py
  方式2. 从原始 .cif + .map 文件实时提取特征（load_from_raw_cif）

典型数据目录结构 (方式1):
    all_data_path/
    ├── emdb_BOX/            ← 密度图特征 (CDHW)
    │   ├── small_molecule/
    │   │   ├── 9f3f_0_0_0_0_C.npz(具体样本)
    │   │   └── ...
    │   └── metal_ion/ ...
    ├── pdb_feature_BOX/     ← 原子特征 (CDHW)
    │   └── ...
    └── pdb_label_BOX/       ← 标签 (CDHW -> DHW)  [可选]
        └── ... 
"""

import sys
import os
from pathlib import Path

# 将 Pocket/ 根目录和 Pocket/Make_Data/ 加入 sys.path，使得 Make_Data 的子模块可以被导入
_INFERENCE_DIR = Path(__file__).resolve().parent       # Pocket/src/inference/
_SRC_DIR       = _INFERENCE_DIR.parent                 # Pocket/src/
_POCKET_ROOT   = _SRC_DIR.parent                       # Pocket/
_PROJECT_ROOT  = _POCKET_ROOT.parent                   # My_Project/
_MAKE_DATA_DIR = _POCKET_ROOT / "Make_Data"            # Pocket/Make_Data/
_BINDER_DIR    = _POCKET_ROOT / "processedPDB_EMDB_binder"  # Pocket/processedPDB_EMDB_binder/

for _p in [str(_PROJECT_ROOT), str(_POCKET_ROOT), str(_MAKE_DATA_DIR), str(_BINDER_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json
from typing import Optional

import numpy as np





# =============================================================================
# 内部辅助函数
# =============================================================================
def apply_class_mapping(label: np.ndarray, mapping: list) -> np.ndarray:
    """
    将标签中的类别 ID 进行映射

    Args:
        - label:   np.ndarray, (D, H, W), 原始标签
        - mapping: list[int],  映射表, 索引为原始类别 ID, 值为新类别 ID

    Returns:
        - mapped_label: np.ndarray, (D, H, W), 映射后的标签
    """
    mapped = np.zeros_like(label)
    for old_id, new_id in enumerate(mapping):
        mapped[label == old_id] = new_id
    return mapped

def compute_hardmask(grid: np.ndarray, emdb_channels: int) -> np.ndarray:
    """
    计算硬掩膜：判断体素处是否有原子（基于 pdb_feature 通道是否全零）

    原理:
        grid 的前 emdb_channels 个通道是 EMDB 密度图（经 z-score 归一化，全零不代表无原子）;
        剩余通道是 pdb_feature（未归一化，全零 ⟺ 无原子）。因此只需检查 grid[emdb_channels:] 在 channel 轴上是否存在非零值。

    # NOTE：此函数假设 grid 的前 emdb_channels 个通道一定是 EMDB 密度图这要求上游 load_from_npz_dirs() 在拼接 grid_parts 时，含 "emdb" 的文件夹必须排在 data_folder_names 列表的最前面。如果顺序被调换，此处截取将出错，hardmask 会被错误计算，且不会产生任何报错（Silent Fail），直接污染下游评估结果。

    Args:
        - grid:          np.ndarray, (C, D, H, W), float32, 拼接后的完整特征网格
        - emdb_channels: int, EMDB 密度图占据的前若干个通道数

    Returns:
        - hardmask: np.ndarray, (D, H, W), int64, 1=有原子 0=无原子
    """
    # np.ndarray, (pdb_feature_channels, D, H, W)
    pdb_features = grid[emdb_channels:]
    # np.ndarray, (D, H, W), bool
    has_atom = np.any(pdb_features != 0, axis=0)
    # np.ndarray, (D, H, W), int64
    return has_atom.astype(np.int64)




# =============================================================================
# 从原始文件加载特征/标签（无需预处理 BOX）  # FIXME: 未来加入更多密度信息(如差图) 时, 需要更改此函数
# =============================================================================
def load_from_raw_cif(
    cif_path: str,
    map_path: str,
    target_voxel_size: float,
    compute_density: bool,
    select_first_model: bool,
    error_dir: str,
) -> dict:
    """
    从原始 .cif + .map 文件加载体素化特征与属性(hardmask)，返回包含 grid, hardmask, emdb_channels, sample_name, class_folder, voxel_size, origin, atom_coords 的 dict

    Args:
        - cif_path:           str,   原始结构文件路径 (.cif / .pdb)
        - map_path:           str,   对应的 EMDB 密度图路径 (.map / .mrc)，必须提供
        - target_voxel_size:  float, 重采样目标体素大小 (Å)，默认 1.0 与训练时一致
        - compute_density:    bool,  是否计算原子局部密度特征（影响 pdb_feature 通道数）
        - select_first_model: bool,  多模型 CIF 时是否仅取第一个模型
        - error_dir:          str|None, 错误日志目录（传给 get_features_when_infer）

    Returns:
        - result: dict, 与 load_from_npz_dirs() 返回格式一致，包含:
            - "grid":          np.ndarray, float32, (C, D, H, W), 拼接后的完整特征网格
            - "hardmask":      np.ndarray, int64,   (D, H, W),    硬掩膜（1=有原子, 0=无原子）
            - "emdb_channels": int, EMDB 密度图占据的通道数（固定为 1）
            - "sample_name":   str, 样本名（从 cif_path 文件名提取）
            - "class_folder":  str, 固定为 "raw"
            - "voxel_size":    np.ndarray, (3,), float, 重采样后的体素大小 (Å)
            - "origin":        np.ndarray, (3,), float, 密度图全局坐标原点 (Å)
            - "atom_coords":   np.ndarray, (N_atom, 3), float32, 所有原子的世界坐标 (x,y,z), 单位 Å

    内部流程:
        1. 加载并重采样 EMDB 密度图, 对密度图做 z-score 归一化
        2. 调用 bind_AtomsFeature_to_EMDB(pdb_path=...) , 现场生成原子特征并映射到 EMDB 体素网格
        3. 拼接 EMDB 密度图通道 + PDB 特征通道 → grid (C=50, D, H, W)
        4. 计算 hardmask（复用 _compute_hardmask()）
    """
    from Pocket.utils.mrc_tools import load_map, make_model_grid
    from processedPDB_EMDB_binder.bind import bind_AtomsFeature_to_EMDB
    # str, 样本名（不含扩展名）
    sample_name = Path(cif_path).stem


    # ---- 1. 加载 EMDB 密度图并重采样 ----
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"[parse_input.load_from_raw_cif] 密度图文件不存在: {map_path}")
    emdb_grid_raw, voxel_size, origin = load_map(map_path)
    emdb_grid, voxel_size, origin = make_model_grid(
        emdb_grid_raw, voxel_size, origin, target_voxel_size
    )
    # np.ndarray, (1, D, H, W), float32
    emdb_grid = emdb_grid[np.newaxis, ...]
    # np.ndarray, (1, D, H, W), float32
    emdb_normed = (emdb_grid - np.mean(emdb_grid)) / (np.std(emdb_grid) + 1e-8)


    # ---- 2. 从 .cif 提取原子级特征 + 将 PDB 原子特征映射到 EMDB 体素网格 ----
    # (np.ndarray, np.ndarray):
    #   pdb_feature_grid: (C_feat, D', H', W'), float32. 目前 C_feat = 49（当 compute_density=True 时）
    #   atom_coords: (N_atom, 3), float32. 所有原子的世界坐标 (x, y, z), 单位 Å
    pdb_feature_grid, atom_coords = bind_AtomsFeature_to_EMDB(
        pdb_path=cif_path,error_dir=error_dir,compute_density=compute_density,select_first_model=select_first_model,
        emdb_path=map_path,
        target_voxel_size=target_voxel_size,
        output_path=None,          # 不保存文件，仅在内存中使用
        add_when_conflict=True,
        overwrite_existing=False,
        return_atom_pos_array=True,
    )


    # ---- 3. 拼接 EMDB 密度图通道 + PDB 特征通道 ----
    # np.ndarray, (1+C_feat, D', H', W'), float32
    grid = np.concatenate([emdb_normed, pdb_feature_grid], axis=0).astype(np.float32)
    # int, EMDB 密度图占据的通道数（固定为 1）
    emdb_channels = 1


    # ---- 4. 计算 hardmask ----
    # np.ndarray, (D', H', W'), int64, 1=有原子 0=无原子
    hardmask = compute_hardmask(grid, emdb_channels=emdb_channels)


    return {
        "grid":          grid,
        "hardmask":      hardmask,
        "emdb_channels": emdb_channels,
        "sample_name":   sample_name,
        "class_folder":  "raw",
        # 附加几何元数据（用于后续可视化或坐标反算）
        "voxel_size":    voxel_size,
        "origin":        origin,
        "atom_coords":   atom_coords,
    }


def load_gt_from_structure(
    cif_path: str,          # 用于提供受体, 必选
    cif_gt_path: str,       # 用于产生候选配体、挑选合格配体并选中配体的信息, 可选: 传 None 则回退到 cif_path
    map_path: str,
    target_voxel_size: float,
    filter_preset: str,
    class_mapping: list,
    select_first_model: bool,
    error_dir: str,
) -> Optional[dict]:
    """
    从原始 .cif / .pdb 文件提取点云级 Ground Truth 标签:  按 Pocket/Make_Data/labels/filter_config.py 定义的预设读取配体筛选规则，
    对蛋白质原子进行结合位点标注，再将原子级标签映射到 EMDB 体素网格。

    Args:
        - cif_path:           str,         必选: 提供受体 (.cif / .pdb), 一般是AF3或cryoAtom预测的结构(可以为空, 为空时默认为 cif_gt_path), 并在cif_gt_path为空时也提供配体
        - cif_gt_path:        str,         可选: 提供配体 (.cif / .pdb)————用于产生候选配体、挑选合格配体并选中配体的信息. 
        - map_path:           str,         对应的 EMDB 密度图路径 (.map / .mrc)
        - target_voxel_size:  float,       重采样目标体素大小 (Å)，默认 0.7
        - filter_preset:      str,         配体筛选预设名，来自 labels/filter_config.py, 例如 "binary" / "five_class" / "cryoem_broad"
        - class_mapping:      list[int]|None, 标签类别映射表，例如 [0,1,1,1,1] 在5分类（背景0）中将多类合并为二分类
        - select_first_model: bool,        structure选择第一个model / 如果一个structure 含有多个model那么直接记入error_log并跳过处理
        - error_dir:          str|None,    错误日志目录

    输出:
        - gt_data: dict, 包含:
            - "atom_coords": np.ndarray, 形状 (N_atom, 3), 全部原子坐标 (世界坐标, 单位 Å)
            - "atom_gt": np.ndarray, 形状 (N_gt, 3), 正类原子坐标 (世界坐标, 单位 Å)

    内部流程:
        1. parse_structure()         解析 .cif 得到原子坐标和配体候选列表
        2. filter_and_classify()     按 filter_preset 筛选配体并分配口袋类别
        3. compute_binding_labels()  计算每个原子的口袋类别 ID
        4. class_mapping（可选）      对类别做映射
    """
    from Make_Data.PDB_processor.parser import parse_structure
    from Make_Data.labels.ligand_filter import filter_and_classify
    from Make_Data.labels.filter_config import get_filter_preset
    from Make_Data.labels.instance_labels import compute_binding_labels
    cif_gt_path = cif_gt_path if cif_gt_path is not None else cif_path
    sample_name = Path(cif_path).stem

    # ---- 0. 读取配体筛选配置 ----
    # LigandFilterConfig, 按预设名读取口袋分类规则
    filter_config = get_filter_preset(filter_preset)
    if filter_config is None:
        from Make_Data.labels.filter_config import list_filter_preset_names
        available = list_filter_preset_names()
        raise ValueError(
            f"[parse_input.load_gt_from_structure] 未知 filter_preset: '{filter_preset}'\n"
            f"可用预设: {available}"
        )


    # ---- 1. 解析 .cif 结构文件 ----
    # ParsedStructure 或 None，包含原子坐标、配体候选列表等
    parsed_gt_data = parse_structure(   # 解析 GT 结构（提供配体信息）
        cif_gt_path,
        error_dir,
        sample_name,
        require_ligand=False,  # 推理时不强制要求配体存在；若无配体则返回全背景标签
        select_first_model=select_first_model,
    )
    if parsed_gt_data is None:
        raise RuntimeError(
            f"[parse_input.load_gt_from_structure] parse_structure() 失败 (GT结构): {cif_gt_path}\n"
            "请检查 CIF/PDB 文件格式，或查看 error_dir 中的日志。"
        )
    # 解析受体结构（提供蛋白/核酸原子坐标）
    if cif_path == cif_gt_path:
        parsed_data = parsed_gt_data
    else:
        parsed_data = parse_structure(
            cif_path,
            error_dir,
            sample_name,
            require_ligand=False,
            select_first_model=select_first_model,
        )
        if parsed_data is None:
            raise RuntimeError(
                f"[parse_input.load_gt_from_structure] parse_structure() 失败 (受体结构): {cif_path}\n"
                "请检查 CIF/PDB 文件格式，或查看 error_dir 中的日志。"
            )


    # ---- 2. 筛选配体并分配口袋类别 ----
    # list[LigandCandidate], 通过筛选的候选配体
    # dict[int, tuple[int, str]], candidate_id → (class_id, class_name)
    # list, 被排除的候选及原因（此处不使用）
    selected, pocket_class_map, _ = filter_and_classify(
        parsed_gt_data.ligand_candidates, filter_config
    )


    # ---- 3. 计算原子级标签 ----
    # dict 或 None, 含 pocket_class_ids: np.ndarray (N_atoms,) int32
    binding_labels = compute_binding_labels(
        parsed_data,
        selected_candidates=selected,
        pocket_class_map=pocket_class_map,
        error_dir=error_dir,
        sample_id=sample_name,
        require_binding_site=False,  # 无配体时返回全背景，而非 None
    )


    # ---- 4. 可选: 仅做路径校验----
    if map_path is not None and not os.path.exists(map_path):
        raise FileNotFoundError(
            f"[parse_input.load_gt_from_structure] 密度图文件不存在: {map_path}"
        )

    # np.ndarray, 形状 (N_atoms,), int32, 每个原子的口袋类别 ID
    if binding_labels is None or "pocket_class_ids" not in binding_labels:
        pocket_class_ids = np.zeros((parsed_data.atom_coords.shape[0],), dtype=np.int32)
    else:
        # np.ndarray, (N_atoms,), int32, 每个原子的口袋类别 ID
        pocket_class_ids = binding_labels["pocket_class_ids"]  
    if class_mapping is not None:
        # np.ndarray, 形状 (N_atoms,), int32, 映射后的类别 ID
        mapped_ids = np.zeros_like(pocket_class_ids)
        for old_id, new_id in enumerate(class_mapping):
            mapped_ids[pocket_class_ids == old_id] = new_id
        pocket_class_ids = mapped_ids

    # np.ndarray, 形状 (N_atoms, 3), float32, 原子坐标 (X, Y, Z)
    atom_coords = parsed_data.atom_coords.astype(np.float32)
    # np.ndarray, 形状 (N_gt, 3), float32, 正类原子坐标
    atom_gt = atom_coords[pocket_class_ids > 0]

    return {
        "atom_coords": atom_coords,
        "atom_gt": atom_gt.astype(np.float32),
    }
















# =============================================================================
# 点云推断: 原子特征加载
# =============================================================================
def load_atom_features_from_raw(
    cif_path: str,
    compute_density: bool,
    select_first_model: bool,
    error_dir: str,
) -> np.ndarray:
    """
    从原始 .cif/.pdb 文件提取 per-atom 49 维特征向量。

    复用 Make_Data.process_and_label.get_features_when_infer()，与 bind_AtomsFeature_to_EMDB 内部调用的是同一函数。

    输入参数:
        - cif_path: str, 输入结构文件路径 (.cif/.pdb)
        - compute_density: bool, 是否计算原子局部密度特征
        - select_first_model: bool, 多模型时是否仅取第一个模型
        - error_dir: str | None, 错误日志目录

    输出:
        - atom_feat: np.ndarray, (N_atom, 49), float32, per-atom 特征向量
    """
    from Make_Data.process_and_label import get_features_when_infer

    result = get_features_when_infer(
        input_path=cif_path,
        error_dir=error_dir,
        compute_density=compute_density,
        select_first_model=select_first_model,
    )
    if result is None:
        raise RuntimeError(
            f"[parse_input.load_atom_features_from_raw] "
            f"get_features_when_infer() 返回 None, cif_path={cif_path}"
        )

    # tuple[dict, dict, dict], (atoms_dict, residues_dict, graph_dict)
    atoms_dict, _, _ = result
    # np.ndarray, (N_atom, 49), float32, per-atom 特征向量
    atom_feat = atoms_dict["features"].astype(np.float32, copy=False)
    return atom_feat


# =============================================================================
# 点云推断: 整体体积切分为 BOX
# =============================================================================
def _compute_window_starts(dim: int, window_size: int, stride: int) -> list[int]:
    """
    计算单轴滑窗起点，并保证最后一个窗口一定贴到尾部。

    输入参数:
        - dim: int, 当前轴长度
        - window_size: int, 窗口长度
        - stride: int, 滑窗步幅

    输出:
        - starts: list[int], 单轴所有滑窗起点
    """
    if dim <= window_size:
        return [0]

    starts = list(range(0, dim - window_size + 1, stride))
    last = dim - window_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def split_volume_to_boxes(
    grid: np.ndarray,
    atom_coords_world: np.ndarray,
    atom_feat: np.ndarray,
    origin: np.ndarray,
    voxel_size: np.ndarray,
    window_size: int,
    stride: int,
    atom_buffer_radius: float,
    valid_crop_margin: int,
    emdb_channels: int,
) -> list:
    """
    将整体密度图切分为多个训练级 BOX，每个 BOX 组装为与训练 __getitem__ 完全同构的 sample dict。

    滑窗策略与旧版 map_segmentation() 一致, 但额外为每个 BOX 构造原子级字段。

    输入参数:
        - grid: np.ndarray, (C, D, H, W), float32, 完整的多通道体素特征网格
        - atom_coords_world: np.ndarray, (N_atom, 3), float32, 所有原子世界坐标 (x, y, z)
        - atom_feat: np.ndarray, (N_atom, 49), float32, per-atom 特征向量
        - origin: np.ndarray, (3,), float, 密度图原点 (x, y, z)
        - voxel_size: np.ndarray, (3,), float, 体素大小 (x, y, z)
        - window_size: int, 标量, BOX 窗口大小 (voxel)
        - stride: int, 标量, 滑窗步长 (voxel)
        - atom_buffer_radius: float, 标量, 原子 buffer 半径 (Å), 建议值 4.0
        - valid_crop_margin: int, 标量, 监督区域裁边量 (voxel), 建议值 0
        - emdb_channels: int, 标量, EMDB 密度图通道数

    输出:
        - box_dicts: list[dict], 每个 dict 为一个 BOX 的完整推断输入
            每个 dict 包含:
            - 训练 sample dict 同构的所有字段 (torch.Tensor)
            - "global_atom_indices": np.ndarray, (N_box,), int, 该 BOX 选中原子的全局索引, N_box 为该 BOX 决定原子的数量(BOX内+buffer)
            - "box_position_zyx": tuple[int, int, int], 该 BOX 在整体体积中的起始坐标 (z0, y0, x0)
    """
    import torch
    from src.datasets.box_geometry import (
        select_atoms_for_box,
        build_atom_coordinates,
        build_atom_features,
        build_atom_valid_mask,
        build_voxel_valid_mask,
        compute_hardmask as shared_compute_hardmask,
    )

    C, D, H, W = grid.shape
    # np.ndarray, (3,), float64, 体素标量大小
    voxel_size = np.asarray(voxel_size, dtype=np.float64).reshape(3)
    # np.ndarray, (3,), float64, 原点
    origin = np.asarray(origin, dtype=np.float64).reshape(3)

    # 滑窗: 对不足 window_size 的维度进行 pad
    pad_d = max(0, window_size - D)
    pad_h = max(0, window_size - H)
    pad_w = max(0, window_size - W)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        grid = np.pad(grid, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)), mode="constant")
    _, Dp, Hp, Wp = grid.shape

    # 生成滑窗起始坐标，并保证最后一个窗口覆盖到各轴尾部
    z_starts = _compute_window_starts(Dp, window_size, stride)
    y_starts = _compute_window_starts(Hp, window_size, stride)
    x_starts = _compute_window_starts(Wp, window_size, stride)

    box_dicts = []
    box_idx = 0

    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                z1 = min(z0 + window_size, Dp)
                y1 = min(y0 + window_size, Hp)
                x1 = min(x0 + window_size, Wp)

                # np.ndarray, (C, D_box, H_box, W_box), float32, 当前 BOX 的体素子块
                box_grid = grid[:, z0:z1, y0:y1, x0:x1].copy()
                # np.ndarray, (3,), int64, 当前 BOX 的体素形状 (Z, Y, X)
                box_shape_zyx = np.array([z1 - z0, y1 - y0, x1 - x0], dtype=np.int64)
                # np.ndarray, (3,), float64, 当前 BOX 的世界原点 (x, y, z)
                # grid 空间轴是 (Z, Y, X), 而 origin/voxel_size 是 (x, y, z)
                box_origin_world = (origin + np.array([x0, y0, z0], dtype=np.float64) * voxel_size).astype(np.float32)
                # np.ndarray, (3,), float32
                voxel_size_f32 = voxel_size.astype(np.float32)

                # 调用共享几何模块选择原子
                selected = select_atoms_for_box(
                    atom_coords_world=atom_coords_world,
                    box_origin_world=box_origin_world,
                    voxel_size_world=voxel_size_f32,
                    box_shape_zyx=box_shape_zyx,
                    buffer_radius=atom_buffer_radius,
                )
                # np.ndarray, (N_box,), int64, 选中的全局原子索引
                selected_idx = selected["selected_idx"]
                # np.ndarray, (N_box,), bool
                atom_is_in_core_box = selected["atom_is_in_core_box"]

                # 构造三套坐标
                coord_data = build_atom_coordinates(
                    atom_coords_world=atom_coords_world,
                    selected_idx=selected_idx,
                    box_origin_world=box_origin_world,
                    voxel_size_world=voxel_size_f32,
                    box_shape_zyx=box_shape_zyx,
                )

                # 构造特征
                atom_feat_sel = build_atom_features(
                    atom_features_raw=atom_feat,
                    selected_idx=selected_idx,
                )

                # 构造 mask（与训练完全一致）
                atom_valid_mask = build_atom_valid_mask(
                    atom_coord_local_voxel=coord_data["atom_coord_local_voxel"],
                    atom_is_in_core_box=atom_is_in_core_box,
                    box_shape_zyx=box_shape_zyx,
                    valid_crop_margin=float(valid_crop_margin),
                )
                voxel_valid_mask = build_voxel_valid_mask(
                    box_shape_zyx=box_shape_zyx,
                    valid_crop_margin=valid_crop_margin,
                )
                hardmask = shared_compute_hardmask(
                    voxel_grid=box_grid,
                    emdb_channels=emdb_channels,
                )

                # 推断时无真实标签, 占位全零
                n_selected = len(selected_idx)
                # np.ndarray, (N_selected,), int64, 占位标签
                atom_label = np.zeros(n_selected, dtype=np.int64)
                # np.ndarray, (D_box, H_box, W_box), int64, 占位标签
                voxel_label = np.zeros(box_shape_zyx.tolist(), dtype=np.int64)

                # 组装训练同构的 sample dict
                sample_dict = _to_torch_sample_infer(
                    voxel_grid=box_grid,
                    voxel_label=voxel_label,
                    hardmask=hardmask,
                    voxel_valid_mask=voxel_valid_mask,
                    box_origin_world=box_origin_world,
                    voxel_size_world=voxel_size_f32,
                    box_shape_zyx=box_shape_zyx,
                    atom_coord_world=coord_data["atom_coord_world"],
                    atom_coord_local_voxel=coord_data["atom_coord_local_voxel"],
                    atom_coord_centered_world=coord_data["atom_coord_centered_world"],
                    atom_feat=atom_feat_sel,
                    atom_label=atom_label,
                    atom_is_in_core_box=atom_is_in_core_box,
                    atom_valid_mask=atom_valid_mask,
                )
                # 附加元信息
                sample_dict["sample_name"] = f"infer_box_{box_idx}"
                sample_dict["pdb_id"] = "infer"
                sample_dict["class_name"] = "infer"
                sample_dict["instance_id"] = 0
                sample_dict["is_center_box"] = False

                # 推断专用: 全局索引与 BOX 位置
                sample_dict["global_atom_indices"] = selected_idx.copy()
                sample_dict["box_position_zyx"] = (z0, y0, x0)

                box_dicts.append(sample_dict)
                box_idx += 1

    print(f"[parse_input] 切分完成: {len(box_dicts)} 个 BOX "
          f"(grid={grid.shape}, window={window_size}, stride={stride})")
    return box_dicts


def _to_torch_sample_infer(**kwargs) -> dict:
    """
    将 numpy 数组转为 torch.Tensor, 字段和 dtype 与训练侧 BoxPointDataset._to_torch_sample 完全一致。

    输入参数:
        - 见 BoxPointDataset.__getitem__ 返回值的 voxel/atom 字段

    输出:
        - dict[str, torch.Tensor], 训练同构的 sample dict
    """
    import torch
    return {
        "voxel_grid": torch.tensor(kwargs["voxel_grid"], dtype=torch.float32),
        "voxel_label": torch.tensor(kwargs["voxel_label"], dtype=torch.int64),
        "hardmask": torch.tensor(kwargs["hardmask"], dtype=torch.int64),
        "voxel_valid_mask": torch.tensor(kwargs["voxel_valid_mask"], dtype=torch.bool),
        "box_origin_world": torch.tensor(kwargs["box_origin_world"], dtype=torch.float32),
        "voxel_size_world": torch.tensor(kwargs["voxel_size_world"], dtype=torch.float32),
        "box_shape_zyx": torch.tensor(kwargs["box_shape_zyx"], dtype=torch.int64),
        "atom_coord_world": torch.tensor(kwargs["atom_coord_world"], dtype=torch.float32),
        "atom_coord_local_voxel": torch.tensor(kwargs["atom_coord_local_voxel"], dtype=torch.float32),
        "atom_coord_centered_world": torch.tensor(kwargs["atom_coord_centered_world"], dtype=torch.float32),
        "atom_feat": torch.tensor(kwargs["atom_feat"], dtype=torch.float32),
        "atom_label": torch.tensor(kwargs["atom_label"], dtype=torch.int64),
        "atom_is_in_core_box": torch.tensor(kwargs["atom_is_in_core_box"], dtype=torch.bool),
        "atom_valid_mask": torch.tensor(kwargs["atom_valid_mask"], dtype=torch.bool),
    }


def prepare_batched_boxes(
    box_dicts: list,
    batch_size: int,
    device: str,
) -> list:
    """
    将 split_volume_to_boxes() 返回的 BOX 列表按 batch_size 分组,
    对每组调用 box_point_collate() 产出 batch dict 并移至设备。

    输入参数:
        - box_dicts: list[dict], split_volume_to_boxes 的返回值
        - batch_size: int, 标量, 每个 batch 包含的 BOX 数
        - device: str, 目标设备

    输出:
        - batches: list[dict], 每个 dict 为一个 batch, 所有 tensor 已在 device 上
            每个 batch dict 额外包含:
            - "_box_meta": list[dict], 每个 BOX 的元信息 (global_atom_indices 和 box_position_zyx )
    """
    from src.datasets.box_point_collate import box_point_collate
    from src.inference.get_pred import move_batch_to_device

    batches = []
    n_boxes = len(box_dicts)

    for start in range(0, n_boxes, batch_size):
        end = min(start + batch_size, n_boxes)
        group = box_dicts[start:end]

        # 保存推断专用的元信息 (不参与 collate)
        box_meta_list = []
        for box_dict in group:
            box_meta_list.append({
                "global_atom_indices": box_dict.pop("global_atom_indices"),
                "box_position_zyx": box_dict.pop("box_position_zyx"),
            })

        # dict[str, Any], box_point_collate 需要标准字段
        batch_dict = box_point_collate(group)
        batch_dict = move_batch_to_device(batch_dict, device)
        batch_dict["_box_meta"] = box_meta_list

        # 恢复 box_dicts 中的元信息 (避免破坏原始列表)
        for i, box_dict in enumerate(group):
            box_dict["global_atom_indices"] = box_meta_list[i]["global_atom_indices"]
            box_dict["box_position_zyx"] = box_meta_list[i]["box_position_zyx"]

        batches.append(batch_dict)

    return batches
