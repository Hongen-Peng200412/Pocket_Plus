"""
parse_input.py - 数据加载与特征整合模块

负责将各种来源的数据整合为模型需要的输入格式。
支持两种加载方式：
  方式1. 从已处理的 .npz 文件（emdb_BOX / pdb_feature_BOX / pdb_label_BOX）加载（load_from_npz_dirs）——————移步到 inference.utils.pipline_for_box.py
  方式2. 从原始 .cif + .map 文件实时提取特征（load_from_raw_cif），无需预处理

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

def compute_hardmask(grid: np.ndarray, emdb_channels: int = 1) -> np.ndarray:
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
# 从原始文件加载特征/标签（无需预处理 BOX）
# =============================================================================
def load_from_raw_cif(
    cif_path: str,
    map_path: str,
    target_voxel_size: float = 1.0,
    compute_density: bool = True,
    select_first_model: bool = False,
    error_dir: str = None,
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
    cif_path: str=None,     # 用于提供蛋白/核酸的骨架结构
    cif_gt_path: str=None,  # 用于产生候选配体、挑选合格配体并选中配体的信息(可以为空, 为空时默认为 cif_path)
    map_path: str=None,
    target_voxel_size: float = None,
    filter_preset: str = None,
    class_mapping: list = None,
    select_first_model: bool = True,
    error_dir: str = None,
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
