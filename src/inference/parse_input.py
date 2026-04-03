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
import torch
import numpy as np

from src.datasets.box_geometry import (
    build_hardmask_from_atom_coordinates,
    build_hardmask_from_world_coordinates,
)
from src.datasets.box_sample_builder import build_box_point_numpy_sample, to_torch_sample




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
    从原始 .cif + .map 文件加载体素化特征与属性(hardmask)，返回包含 grid, hardmask, emdb_channels, sample_name, class_folder, voxel_size, origin, atom_coords 的 dict。

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
        4. 基于原子几何位置计算 hardmask
    """
    from Pocket.utils.mrc_tools import load_map, make_model_grid
    from processedPDB_EMDB_binder.bind import bind_AtomsFeature_to_EMDB
    from Make_Data.process_and_label import get_features_when_infer
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


    # ---- 2. 一次性解析原子信息, 保证坐标与特征严格同源 ----
    atom_info_result = get_features_when_infer(
        input_path=cif_path,
        error_dir=error_dir,
        compute_density=compute_density,
        select_first_model=select_first_model,
    )
    # dict, 含 'coords' 和 'features'
    atom_info_dict = atom_info_result[0]
    # np.ndarray, (N_atom, 3), float32, 原子世界坐标
    atom_coords = atom_info_dict['coords'].astype(np.float32, copy=False)
    # np.ndarray, (N_atom, F), float32, per-atom 特征向量
    atom_feat = atom_info_dict['features'].astype(np.float32, copy=False)


    # ---- 3. 将预解析的原子信息传入 binder, 映射到 EMDB 体素网格 ----
    pdb_feature_grid, _ = bind_AtomsFeature_to_EMDB(
        pre_parsed_atom_info=atom_info_dict,
        emdb_path=map_path,
        target_voxel_size=target_voxel_size,
        output_path=None,
        add_when_conflict=True,
        overwrite_existing=False,
        return_atom_pos_array=True,
    )


    # ---- 4. 拼接 EMDB 密度图通道 + PDB 特征通道 ----
    # np.ndarray, (1+C_feat, D', H', W'), float32
    grid = np.concatenate([emdb_normed, pdb_feature_grid], axis=0).astype(np.float32)
    # int, EMDB 密度图占据的通道数（固定为 1）
    emdb_channels = 1



    # ---- 5. 基于原子几何位置计算 hardmask ----
    # np.ndarray, (3,), int64, 当前整图的空间大小, 顺序为 (D, H, W)
    box_shape_zyx = np.asarray(grid.shape[-3:], dtype=np.int64)
    # np.ndarray, (D', H', W'), int64, 1=该体素落入至少一个原子的 home voxel
    hardmask = build_hardmask_from_world_coordinates(
        atom_coords_world=atom_coords,
        box_origin_world=np.asarray(origin, dtype=np.float32).reshape(3),
        voxel_size_world=np.asarray(voxel_size, dtype=np.float32).reshape(3),
        box_shape_zyx=box_shape_zyx,
    )


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
        "atom_feat":     atom_feat,
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
    接受 def load_from_raw_cif 的返回，将整体密度图切分为多个训练级 BOX，每个 BOX 组装为与训练 __getitem__ 完全同构的 sample dict。

    输入参数:
        - grid: np.ndarray, (C, D, H, W), float32, 完整的多通道体素特征网格
        - atom_coords_world: np.ndarray, (N_atom, 3), float32, 所有原子世界坐标 (x, y, z)
        - atom_feat: np.ndarray, (N_atom, 49), float32, per-atom 特征向量
        - origin: np.ndarray, (3,), float, 密度图原点 (x, y, z)
        - voxel_size: np.ndarray, (3,), float, 体素大小 (x, y, z)
        - window_size: int, 标量, BOX 窗口大小 (voxel)
        - stride: int, 标量, 滑窗步长 (voxel)
        - atom_buffer_radius: float, 标量, 原子 buffer 半径 (Å)
        - valid_crop_margin: int, 标量, 监督区域裁边量 (voxel)
        - emdb_channels: int, 标量, 旧接口保留参数; 当前仅用于与历史调用保持签名兼容

    输出:
        - box_dicts: list[dict], 每个 dict 为一个 BOX 的完整推断输入, 包含:
            1. 以下 torch.Tensor 字段 (与训练 sample dict 同构):
                - "voxel_grid":              torch.Tensor, (C, D_box, H_box, W_box), float32
                - "voxel_label":             torch.Tensor, (D_box, H_box, W_box), int64, 全零占位
                - "hardmask":                torch.Tensor, (D_box, H_box, W_box), int64
                - "voxel_valid_mask":         torch.Tensor, (D_box, H_box, W_box), bool
                - "box_origin_world":         torch.Tensor, (3,), float32
                - "voxel_size_world":         torch.Tensor, (3,), float32
                - "box_shape_zyx":            torch.Tensor, (3,), int64
                - "atom_coord_world":         torch.Tensor, (N_box, 3), float32
                - "atom_coord_local_voxel":   torch.Tensor, (N_box, 3), float32
                - "atom_coord_centered_world":torch.Tensor, (N_box, 3), float32
                - "atom_feat":               torch.Tensor, (N_box, F), float32
                - "atom_label":              torch.Tensor, (N_box,), int64, 全零占位
                - "atom_is_in_core_box":      torch.Tensor, (N_box,), bool
                - "atom_valid_mask":          torch.Tensor, (N_box,), bool
            2. 元信息字段 (Python 原生类型):
                - "sample_name":  str, 形如 "infer_box_0"
                - "pdb_id":       str, 固定 "infer"
                - "class_name":   str, 固定 "infer"
                - "instance_id":  int, 固定 0
                - "is_center_box": bool, 固定 False
            3. 推断专用 sidecar 字段:
                - "global_atom_indices": np.ndarray, (N_box,), int64, 该 BOX 选中原子在全局原子数组中的索引
                - "box_position_zyx":   tuple[int, int, int], 该 BOX 在整体体积中的滑窗起始坐标 (z0, y0, x0)
    """
    C, D, H, W = grid.shape
    del emdb_channels  # 与旧接口兼容; 几何 hardmask 不再依赖特征通道布局
    # np.ndarray, (3,), float64, 体素标量大小
    voxel_size = np.asarray(voxel_size, dtype=np.float64).reshape(3)
    # np.ndarray, (3,), float64, 原点
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    # np.ndarray, (N_atom,), int64, 全局原子占位标签(推断时全零)
    atom_labels_placeholder = np.zeros(len(atom_coords_world), dtype=np.int64)

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
                # np.ndarray, (3,), float32, 当前 BOX 的世界原点 (x, y, z)
                box_origin_world = (origin + np.array([x0, y0, z0], dtype=np.float64) * voxel_size).astype(np.float32)
                # np.ndarray, (3,), float32
                voxel_size_f32 = voxel_size.astype(np.float32)
                # np.ndarray, (D_box, H_box, W_box), int64, 占位标签
                voxel_label = np.zeros(box_shape_zyx.tolist(), dtype=np.int64)

                # 调用共享 builder 构建标准样本
                sample_dict = build_box_point_numpy_sample(
                    voxel_grid=box_grid,
                    voxel_label=voxel_label,
                    atom_coords_world_full=atom_coords_world,
                    atom_features_raw_full=atom_feat,
                    atom_labels_full=atom_labels_placeholder,
                    box_origin_world=box_origin_world,
                    voxel_size_world=voxel_size_f32,
                    box_shape_zyx=box_shape_zyx,
                    atom_buffer_radius=atom_buffer_radius,
                    valid_crop_margin=valid_crop_margin,
                    class_mapping=None,
                )

                # 推断专用 sidecar: 全局索引与 BOX 位置
                # np.ndarray, (N_selected,), int64
                selected_idx = sample_dict.pop("_selected_idx")

                # tensor 化
                sample_dict = to_torch_sample(sample_dict)

                # 追加元信息
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




def prepare_batched_boxes(
    box_dicts: list,
    batch_size: int,
    device: str,
) -> list:
    """
    将 split_volume_to_boxes() 返回的 BOX 列表按 batch_size 分组, 对每组调用 box_point_collate() 产出 batch dict 。

    输入参数:
        - box_dicts: list[dict], split_volume_to_boxes 的返回值
        - batch_size: int, 标量, 每个 batch 包含的 BOX 数
        - device: str, 目标设备

    输出:
        - batches: list[dict], 每个 dict 为一个 batch (即一组 batch_size 个样本经 box_point_collate 后的结果), 包含:
            1. box_point_collate 产出的标准字段 (见 box_point_collate 文档)
            2. 额外字段:
                - "atom_global_indices": torch.Tensor, (sumN,), long, 展平后的全局原子索引, 与 atom_coord_world 等长
                - "_box_meta": list[dict], 长度=当前 batch 的 BOX 数, 每个 dict 含:
                    - "box_position_zyx": tuple[int, int, int], 该 BOX 在整体体积中的滑窗起始坐标
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
                "box_position_zyx": box_dict.pop("box_position_zyx"),
            })
        # list[list[int]]: global_atom_indices 先 pop 出来, collate 后再拼成扁平 tensor
        per_box_global_indices = [box_dict.pop("global_atom_indices") for box_dict in group]

        # dict[str, Any], box_point_collate 需要标准字段
        batch_dict = box_point_collate(group)
        total_atoms = int(batch_dict["atom_counts"].sum().item())
        if total_atoms > 0:
            batch_dict["atom_global_indices"] = torch.cat(
                [
                    torch.as_tensor(idx, dtype=torch.long)
                    for idx in per_box_global_indices
                ],
                dim=0,
            )
        else:
            batch_dict["atom_global_indices"] = torch.empty((0,), dtype=torch.long)
        batch_dict = move_batch_to_device(batch_dict, device)
        batch_dict["_box_meta"] = box_meta_list

        # 恢复 box_dicts 中的元信息 (避免破坏原始列表)
        for i, box_dict in enumerate(group):
            box_dict["global_atom_indices"] = per_box_global_indices[i]
            box_dict["box_position_zyx"] = box_meta_list[i]["box_position_zyx"]

        batches.append(batch_dict)

    return batches
