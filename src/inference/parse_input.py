"""
parse_input.py - 数据加载与特征整合模块

负责将各种来源的数据整合为模型需要的输入格式。
加载方式： 从原始 .cif + .map 文件实时提取特征（load_from_raw_cif）

评估模式 (eval_mode) 数据流对比:

    Trivial 模式 (eval_mode="trivial"):
        cif_path (真实结构)  ──→ 模型推断 ──→ pred (根据受体预测的结合原子坐标) ———————————————————————————————─┐
                                                                                                            │
        cif_path (真实结构)  ──→ 提取配体 ——————————————────┐                                                │
        cif_path (真实结构)  ──→ 提供受体原子 ──→ compute_binding_labels ——————──→ atom_gt                    │
                                                                                    │                       │
                                                                    semantic_evaluate(pred, atom_gt, dist_threshold)

        cif_gt_path 被完全忽略（即使用户提供也不使用）。
        pred ∈ cif_path, GT ∈ cif_path, 且 cif_path 本身含配体 → 最简单的评估场景

    Easy 模式 (eval_mode="easy", 默认):
        cif_path (预测受体)  ──→ 模型推断 ──→ pred (根据受体预测的结合原子坐标) ———————————————————————————————─┐
                                                                                                            │
        cif_gt_path (真实全复合物) ──→ 提取配体 ————————————─┐                                                │
        cif_path (预测受体)  ──→ 提供受体原子 ──→ compute_binding_labels ——————──→ atom_gt                    │
                                                                                    │                       │
                                                                    semantic_evaluate(pred, atom_gt, dist_threshold)

        pred ∈ cif_path 的原子集, GT 也 ∈ cif_path 的原子集, 两者必然匹配良好 → 降低了难度

    Hard 模式 (eval_mode="hard"):
        cif_path (预测受体)  ──→ 模型推断 ──→ pred (根据受体预测的结合原子坐标) ———————————————————————————————─┐
                                                                                                            │
        cif_gt_path (真实全复合物) ──→ 提取配体 ————————————─┐                                                │
        cif_gt_path (真实全复合物) ──→ 提供受体原子 ──→ compute_binding_labels ──→ atom_gt                    │
                                                                                    │                       │
                                                                    semantic_evaluate(pred, atom_gt, dist_threshold)

        pred ∈ cif_path 的原子集, GT ∈ cif_gt_path 的原子集, 两者是不同坐标系的原子 → 评估更严格

    注意: 当 cif_gt_path 为 None 时 (场景A), easy 与 hard 行为完全一致, 因为 cif_gt_path 回退到 cif_path。
"""

import sys
import os
from concurrent.futures import ThreadPoolExecutor
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

from typing import Any, Optional
import torch
import numpy as np

from src.datasets.box_geometry import build_hardmask_from_world_coordinates
from src.datasets.box_sample_builder import build_box_point_numpy_sample, to_torch_sample
from src.datasets.density_channel_builder import (
    ALL_CHANNEL_NAMES,
    DensityChannelConfig,
    build_density_channels,
)




def _normalize_density_channel_config(
    density_channel_config: dict | DensityChannelConfig,
) -> DensityChannelConfig:
    """
    将推理配置中的 density_channel_config 规范化为 DensityChannelConfig。

    输入参数:
        - density_channel_config: dict 或 DensityChannelConfig, 密度通道配置

    输出:
        - config: DensityChannelConfig, 可直接传给 build_density_channels 的配置
    """
    if isinstance(density_channel_config, DensityChannelConfig):
        return density_channel_config
    if not isinstance(density_channel_config, dict):
        raise TypeError(
            "density_channel_config 必须是 dict 或 DensityChannelConfig, "
            f"实际类型: {type(density_channel_config)}"
        )

    return DensityChannelConfig(
        clip_percentile=tuple(float(v) for v in density_channel_config["clip_percentile"]),
        fit_mask_percentile=float(density_channel_config["fit_mask_percentile"]),
        enabled_channels=[str(v) for v in density_channel_config["enabled_channels"]],
    )


def _resolve_density_channel_names(config: DensityChannelConfig) -> list[str]:
    """
    展开实际启用的密度通道名称。

    输入参数:
        - config: DensityChannelConfig, 密度通道配置

    输出:
        - channel_names: list[str], 可变长度, 实际启用的通道名列表
    """
    if "all" in config.enabled_channels:
        return list(ALL_CHANNEL_NAMES)
    return list(config.enabled_channels)


def _load_and_resample_map(
    map_path: str,
    target_voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    读取并重采样单张 MRC/MAP 密度图。

    输入参数:
        - map_path: str, 密度图路径
        - target_voxel_size: float, 目标体素大小

    输出:
        - resampled_grid: np.ndarray, (D, H, W), float32, 重采样后的密度图
        - voxel_size: np.ndarray, (3,), float32, 重采样后体素大小(x,y,z)
        - origin: np.ndarray, (3,), float32, 重采样后世界坐标原点(x,y,z)
        - original_voxel_size: np.ndarray, (3,), float64, 原始体素大小(x,y,z)
        - original_origin: np.ndarray, (3,), float64, 原始世界坐标原点(x,y,z)
        - original_grid_shape: np.ndarray, (3,), int64, 原始密度图形状(D,H,W)
    """
    from processedPDB_EMDB_binder.utils.mrc_tools import load_map, make_model_grid

    if not os.path.exists(map_path):
        raise FileNotFoundError(f"[parse_input] 密度图文件不存在: {map_path}")

    # np.ndarray, (D0, H0, W0), 原始密度图
    grid_raw, voxel_size_raw, origin_raw = load_map(map_path)
    # np.ndarray, (3,), float64, 原始体素大小(x,y,z)
    original_voxel_size = np.asarray(voxel_size_raw, dtype=np.float64).reshape(3)
    # np.ndarray, (3,), float64, 原始世界坐标原点(x,y,z)
    original_origin = np.asarray(origin_raw, dtype=np.float64).reshape(3)
    # np.ndarray, (3,), int64, 原始密度图形状(D,H,W)
    original_grid_shape = np.asarray(grid_raw.shape, dtype=np.int64).reshape(3)

    # np.ndarray, (D, H, W), 重采样密度图
    resampled_grid, voxel_size, origin = make_model_grid(
        grid_raw,
        voxel_size_raw,
        origin_raw,
        target_voxel_size,
    )
    return (
        np.asarray(resampled_grid, dtype=np.float32),
        np.asarray(voxel_size, dtype=np.float32).reshape(3),
        np.asarray(origin, dtype=np.float32).reshape(3),
        original_voxel_size,
        original_origin,
        original_grid_shape,
    )


# =============================================================================
# 从原始文件加载特征/标签（无需预处理 BOX）
# =============================================================================
def load_from_raw_cif(
    cif_path: str,
    map_path: str,
    sim_map_path: str | None,
    target_voxel_size: float,
    compute_density: bool,
    select_first_model: bool,
    error_dir: str | None,
    density_channel_config: dict | DensityChannelConfig,
) -> dict[str, Any]:
    """
    从 raw CIF、真实密度图和已有模拟密度图构造推理输入。

    输入参数:
        - cif_path: str, 结构文件路径, 用于提取受体原子坐标和 atom feature
        - map_path: str, 真实密度图路径
        - sim_map_path: str | None, 已有模拟密度图路径; 启用 sim/diff/posdiff 通道时必须提供
        - target_voxel_size: float, 重采样目标体素大小
        - compute_density: bool, 是否在 get_features_when_infer 中计算原子局部密度特征
        - select_first_model: bool, 多 model 结构处理策略
        - error_dir: str | None, 结构解析错误输出目录
        - density_channel_config: dict 或 DensityChannelConfig, 密度通道构建配置

    输出:
        - result: dict[str, Any], raw 推理输入字典, 包含:
            - "hardmask": np.ndarray, (D, H, W), int64, 原子 home voxel 占据掩码
            - "sample_name": str, 由 cif_path 文件名解析得到的样本名
            - "class_folder": str, 固定为 "raw", 用于与历史数据侧接口对齐
            - "voxel_size": np.ndarray, (3,), float32, 重采样后体素大小(x,y,z)
            - "origin": np.ndarray, (3,), float32, 重采样后世界坐标原点(x,y,z)
            - "atom_coords": np.ndarray, (N_atom, 3), float32, 原子世界坐标(x,y,z)
            - "atom_feat": np.ndarray, (N_atom, F), float32, 原子特征
            - "resampled_emdb": np.ndarray, (D, H, W), float32, 重采样真实密度
            - "resampled_sim": np.ndarray | None, (D, H, W), float32, 重采样模拟密度; 未启用相关通道时为 None
            - "density_config": DensityChannelConfig, BOX 内 density builder 配置
            - "density_channel_names": list[str], 可变长度, 当前实际启用的密度通道名
            - "full_shape_zyx": tuple[int,int,int], 重采样整图形状(D,H,W)
            - "original_voxel_size": np.ndarray, (3,), float64, 原始体素大小(x,y,z)
            - "original_origin": np.ndarray, (3,), float64, 原始世界坐标原点(x,y,z)
            - "original_grid_shape": np.ndarray, (3,), int64, 原始密度图形状(D,H,W)
    """
    from Make_Data.process_and_label import get_features_when_infer

    if not os.path.exists(cif_path):
        raise FileNotFoundError(f"[parse_input] 结构文件不存在: {cif_path}")

    # str, 样本名(不含扩展名)
    sample_name = Path(cif_path).stem
    # DensityChannelConfig, 密度通道构建配置
    density_config = _normalize_density_channel_config(density_channel_config)
    # list[str], 实际启用的密度通道名
    density_channel_names = _resolve_density_channel_names(density_config)

    (
        exp_raw,
        voxel_size,
        origin,
        original_voxel_size,
        original_origin,
        original_grid_shape,
    ) = _load_and_resample_map(map_path=map_path, target_voxel_size=target_voxel_size)

    # dict, 含 coords/features 的原子信息字典
    atom_info_result = get_features_when_infer(
        input_path=cif_path,
        error_dir=error_dir,
        compute_density=compute_density,
        select_first_model=select_first_model,
    )
    atom_info_dict = atom_info_result[0]
    # np.ndarray, (N_atom, 3), float32, 原子世界坐标(x,y,z)
    atom_coords = atom_info_dict["coords"].astype(np.float32, copy=False)
    # np.ndarray, (N_atom, F), float32, 原子特征
    atom_feat = atom_info_dict["features"].astype(np.float32, copy=False)

    # np.ndarray, (D, H, W), int64, 仅原子落点定义的 hardmask
    hardmask = build_hardmask_from_world_coordinates(
        atom_coords_world=atom_coords,
        box_origin_world=origin,
        voxel_size_world=voxel_size,
        box_shape_zyx=np.asarray(exp_raw.shape, dtype=np.int64),
    )

    # bool, 是否需要读取模拟密度图
    needs_sim = any(name.split("_")[0] in {"sim", "diff", "posdiff"} for name in density_channel_names)
    if needs_sim and sim_map_path is None:
        raise ValueError("density_channel_config 启用了 sim/diff/posdiff 通道, 但 sim_map_path 为 None")

    sim_raw = None
    if needs_sim:
        sim_raw, sim_voxel_size, sim_origin, _, _, _ = _load_and_resample_map(
            map_path=sim_map_path,
            target_voxel_size=target_voxel_size,
        )
        if sim_raw.shape != exp_raw.shape:
            raise ValueError(
                "真实密度图与模拟密度图重采样后 shape 不一致: "
                f"exp={exp_raw.shape}, sim={sim_raw.shape}"
            )
        if not np.allclose(sim_voxel_size, voxel_size, rtol=1e-5, atol=1e-6):
            raise ValueError(
                "真实密度图与模拟密度图重采样后 voxel_size 不一致: "
                f"exp={voxel_size}, sim={sim_voxel_size}"
            )
        if not np.allclose(sim_origin, origin, rtol=1e-5, atol=1e-4):
            raise ValueError(
                "真实密度图与模拟密度图重采样后 origin 不一致: "
                f"exp={origin}, sim={sim_origin}"
            )

    return {
        "hardmask": hardmask,
        "sample_name": sample_name,
        "class_folder": "raw",
        "voxel_size": voxel_size,
        "origin": origin,
        "atom_coords": atom_coords,
        "atom_feat": atom_feat,
        "resampled_emdb": exp_raw,
        "resampled_sim": sim_raw,
        "density_config": density_config,
        "density_channel_names": density_channel_names,
        "full_shape_zyx": tuple(int(v) for v in exp_raw.shape),
        "original_voxel_size": original_voxel_size,
        "original_origin": original_origin,
        "original_grid_shape": original_grid_shape,
    }


def load_gt_from_structure(
    cif_path: str,          # 用于提供受体, 必选
    cif_gt_path: str,       # 用于产生候选配体、挑选合格配体并选中配体的信息, 可选: 传 None 则回退到 cif_path
    filter_preset: str,
    class_mapping: list,
    select_first_model: bool,
    error_dir: str,
    eval_mode: str,         # "easy" 或 "hard"
) -> Optional[dict]:
    """
    从原始 .cif / .pdb 文件提取点   云级 Ground Truth 标签:  按 Pocket/Make_Data/labels/filter_config.py 定义的预设读取配体筛选规则，
    对蛋白质原子进行结合位点标注，再将原子级标签映射到 EMDB 体素网格。

    Args:
        - cif_path:           str,         必选: 提供受体 (.cif / .pdb), 一般是AF3或cryoAtom预测的结构(可以为空, 为空时默认为 cif_gt_path), 并在cif_gt_path为空时也提供配体
        - cif_gt_path:        str,         可选: 提供配体 (.cif / .pdb)————用于产生候选配体、挑选合格配体并选中配体的信息. 
        - filter_preset:      str,         配体筛选预设名，来自 labels/filter_config.py, 例如 "binary" / "five_class" / "cryoem_broad"
        - class_mapping:      list[int]|None, 标签类别映射表，例如 [0,1,1,1,1] 在5分类（背景0）中将多类合并为二分类; 依据训练配置 dataset.class_mapping 决定
        - select_first_model: bool,        structure选择第一个model / 如果一个structure 含有多个model那么直接记入error_log并跳过处理
        - error_dir:          str|None,    错误日志目录
        - eval_mode:          str,         评估模式: "trivial"、"easy" 或 "hard"
            - "trivial": 若提供 cif_gt_path，则完全使用 cif_gt_path 提取配体和受体（否则回退到 cif_path）
            - "easy": 结合位点标注基于 cif_path 的受体原子 (预测结构)
            - "hard": 结合位点标注基于 cif_gt_path 的受体原子 (真实结构)

    输出:
        - gt_data: dict, 包含:
            - "atom_coords": np.ndarray, (N_atom, 3), 标注结构的全部受体原子坐标
            - "atom_gt": np.ndarray, (N_gt, 3), 映射后正类原子坐标
            - "pocket_class_ids": np.ndarray, (N_atom,), 映射后的原子口袋类别 ID
            - "pocket_atom_indices": dict[int, np.ndarray], candidate_id → (K_i,), 每个配体对应的口袋原子索引
            - "ligand_coords": dict[int, np.ndarray], candidate_id → (M_i,3), ligand 原子世界坐标(x,y,z)
            - "ligand_candidate_ids": np.ndarray, (N_ligand,), 通过筛选的配体 candidate_id
            - "ligand_class_ids": np.ndarray, (N_ligand,), 原始配体类别 ID
            - "mapped_ligand_class_ids": np.ndarray, (N_ligand,), 映射后的配体类别 ID

    内部流程:
        1. parse_structure()         解析 .cif 得到原子坐标和配体候选列表
        2. filter_and_classify()     按 filter_preset 筛选配体并分配口袋类别
        3. compute_binding_labels()  计算每个原子的口袋类别 ID
           - easy 模式: 受体原子来自 cif_path (parsed_data)
           - hard 模式: 受体原子来自 cif_gt_path (parsed_gt_data)
        4. class_mapping（可选）      对类别做映射
    """
    from Make_Data.PDB_processor.parser import parse_structure
    from Make_Data.labels.ligand_filter import filter_and_classify
    from Make_Data.labels.filter_config import get_filter_preset
    from Make_Data.labels.instance_labels import compute_binding_labels
    if eval_mode not in ("easy", "hard", "trivial"):
        raise ValueError(f"[parse_input.load_gt_from_structure] eval_mode 必须为 'easy'、'hard' 或 'trivial', 收到: '{eval_mode}'")
    # trivial 模式: 若提供 cif_gt_path，则推理输入、配体筛选与 GT 统一基于真实结构 cif_gt_path
    if eval_mode == "trivial": # 换来换去都一样
        if cif_gt_path:
            cif_path = cif_gt_path
        cif_gt_path = cif_path
    else:
        cif_gt_path = cif_gt_path if cif_gt_path is not None else cif_path
    sample_name = Path(cif_path).stem

    # ---- 0. 读取配体筛选配置 ----
    # LigandFilterConfig, 按预设名读取口袋分类规则
    filter_config = get_filter_preset(filter_preset)
    if filter_config is None:
        from Make_Data.labels.filter_config import list_filter_preset_names
        available = list_filter_preset_names()
        raise ValueError(f"[parse_input.load_gt_from_structure] 未知 filter_preset: '{filter_preset}'\n可用预设: {available}")


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
    # trivial/easy 模式: 受体原子来自 cif_path (parsed_data)
    # hard 模式: 受体原子来自 cif_gt_path (parsed_gt_data)
    # ParsedStructure, 决定结合位点标注所依赖的受体结构
    labeling_structure = parsed_gt_data if eval_mode == "hard" else parsed_data
    # dict 或 None, 含 pocket_class_ids: np.ndarray (N_atoms,) int32
    binding_labels = compute_binding_labels(
        labeling_structure,
        selected_candidates=selected,
        pocket_class_map=pocket_class_map,
        error_dir=error_dir,
        sample_id=sample_name,
        require_binding_site=False,  # 无配体时返回全背景，而非 None
    )

    # np.ndarray, (N_atoms,), int32, 原始口袋类别 ID; 0 表示背景
    if binding_labels is None or "pocket_class_ids" not in binding_labels:
        pocket_class_ids_raw = np.zeros((labeling_structure.atom_coords.shape[0],), dtype=np.int32)
    else:
        pocket_class_ids_raw = binding_labels["pocket_class_ids"].astype(np.int32, copy=False)

    # np.ndarray, (N_atoms,), int32, 映射后的口袋类别 ID; 0 表示背景
    pocket_class_ids = pocket_class_ids_raw.copy()
    if class_mapping is not None:
        pocket_class_ids = np.zeros_like(pocket_class_ids_raw)
        for old_id, new_id in enumerate(class_mapping):
            pocket_class_ids[pocket_class_ids_raw == old_id] = int(new_id)

    # dict[int, np.ndarray], candidate_id → (K_i,), 每个配体对应的口袋原子索引
    if binding_labels is None or "pocket_atom_indices" not in binding_labels:
        pocket_atom_indices = {
            int(candidate.candidate_id): np.empty((0,), dtype=np.int64)
            for candidate in selected
        }
    else:
        pocket_atom_indices = {
            int(candidate_id): np.asarray(indices, dtype=np.int64)
            for candidate_id, indices in binding_labels["pocket_atom_indices"].items()
        }

    # list[int], 可变长度, 通过筛选的配体 candidate_id 列表
    ligand_candidate_id_list = []
    # list[int], 可变长度, 原始配体类别 ID 列表
    ligand_class_id_list = []
    # list[int], 可变长度, 映射后的配体类别 ID 列表
    mapped_ligand_class_id_list = []
    # dict[int, np.ndarray], candidate_id → (M_i,3), ligand 原子世界坐标(x,y,z)
    ligand_coords_map: dict[int, np.ndarray] = {}
    for candidate in sorted(selected, key=lambda v: v.candidate_id):
        # int, 当前配体 candidate_id
        candidate_id = int(candidate.candidate_id)
        # int, 当前配体原始类别 ID
        class_id = int(pocket_class_map[candidate_id][0])
        # int, 当前配体映射后类别 ID
        mapped_class_id = class_id
        if class_mapping is not None:
            if class_id < 0 or class_id >= len(class_mapping):
                raise IndexError(f"ligand_class_id={class_id} 超出 class_mapping 长度 {len(class_mapping)}")
            mapped_class_id = int(class_mapping[class_id])
        ligand_candidate_id_list.append(candidate_id)
        ligand_class_id_list.append(class_id)
        mapped_ligand_class_id_list.append(mapped_class_id)
        ligand_coords_map[candidate_id] = np.asarray(candidate.coords, dtype=np.float32)

    # np.ndarray, (N_atoms, 3), float32, 标注结构的受体原子坐标(x,y,z)
    atom_coords = labeling_structure.atom_coords.astype(np.float32)
    # np.ndarray, (N_gt, 3), float32, 映射后正类原子坐标(x,y,z)
    atom_gt = atom_coords[pocket_class_ids > 0]

    return {
        "atom_coords": atom_coords,
        "atom_gt": atom_gt.astype(np.float32),
        "pocket_class_ids": pocket_class_ids.astype(np.int32, copy=False),
        "pocket_atom_indices": pocket_atom_indices,
        "ligand_coords": ligand_coords_map,
        "ligand_candidate_ids": np.asarray(ligand_candidate_id_list, dtype=np.int64),
        "ligand_class_ids": np.asarray(ligand_class_id_list, dtype=np.int64),
        "mapped_ligand_class_ids": np.asarray(mapped_ligand_class_id_list, dtype=np.int64),
    }














# =============================================================================
# 后续整理： 整体体积切分为 BOX
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
    exp_raw: np.ndarray,
    sim_raw: np.ndarray | None,
    density_config: DensityChannelConfig,
    atom_coords_world: np.ndarray,
    atom_feat: np.ndarray,
    origin: np.ndarray,
    voxel_size: np.ndarray,
    window_size: int,
    stride: int,
    atom_buffer_radius: float,
    valid_crop_margin: int,
    num_box_workers: int = 1,
) -> list[dict[str, Any]]:
    """
    将整图 raw 密度切分为 BOX, 并在每个 BOX 内构造训练同构的 density channel。

    输入参数:
        - exp_raw: np.ndarray, (D,H,W), float32, 重采样真实密度图
        - sim_raw: np.ndarray | None, (D,H,W), float32, 重采样模拟密度图; 仅启用 sim/diff/posdiff 通道时非 None
        - density_config: DensityChannelConfig, BOX 内 density builder 配置
        - atom_coords_world: np.ndarray, (N_atom, 3), float32, 所有原子世界坐标(x,y,z)
        - atom_feat: np.ndarray, (N_atom, F), float32, per-atom 特征向量
        - origin: np.ndarray, (3,), float, 密度图原点(x,y,z)
        - voxel_size: np.ndarray, (3,), float, 体素大小(x,y,z)
        - window_size: int, BOX 窗口大小(voxel)
        - stride: int, 滑窗步长(voxel)
        - atom_buffer_radius: float, 原子 buffer 半径(Å); 依据训练配置 dataset.atom_buffer_radius 决定
        - valid_crop_margin: int, 监督区域裁边量(voxel); 依据训练配置 dataset.valid_crop_margin 决定
        - num_box_workers: int, BOX 构造线程数; 1 表示串行, >1 时保持输出顺序并行构造

    输出:
        - box_dicts: list[dict[str, Any]], 每个 dict 为一个 BOX 的完整推断输入, 包含:
            - 1. 以下 torch.Tensor 字段(与训练 sample dict 同构):
                - "voxel_grid": torch.Tensor, (C, D_box, H_box, W_box), float32, 当前 BOX 的多通道体素特征
                - "voxel_label": torch.Tensor, (D_box, H_box, W_box), int64, 全零占位标签
                - "hardmask": torch.Tensor, (D_box, H_box, W_box), int64, 当前 BOX 的几何 hardmask
                - "voxel_valid_mask": torch.Tensor, (D_box, H_box, W_box), bool, 当前 BOX 的有效监督区域掩码
                - "box_origin_world": torch.Tensor, (3,), float32, 当前 BOX 的世界坐标原点(x,y,z)
                - "voxel_size_world": torch.Tensor, (3,), float32, 当前 BOX 的体素大小(x,y,z)
                - "box_shape_zyx": torch.Tensor, (3,), int64, 当前 BOX 的体素形状(z,y,x)
                - "atom_coord_world": torch.Tensor, (N_box, 3), float32, 选中原子的世界坐标(x,y,z)
                - "atom_coord_local_voxel": torch.Tensor, (N_box, 3), float32, 选中原子的局部体素坐标
                - "atom_coord_centered_world": torch.Tensor, (N_box, 3), float32, 选中原子相对 BOX 中心的世界坐标
                - "atom_feat": torch.Tensor, (N_box, F), float32, 选中原子的特征
                - "atom_label": torch.Tensor, (N_box,), int64, 全零占位标签
                - "atom_is_in_core_box": torch.Tensor, (N_box,), bool, 选中原子是否位于 BOX core 区域
                - "atom_valid_mask": torch.Tensor, (N_box,), bool, 选中原子是否参与监督
            - 2. 元信息字段(Python 原生类型):
                - "sample_name": str, 形如 "infer_box_0"
                - "pdb_id": str, 固定为 "infer"
                - "class_name": str, 固定为 "infer"
                - "instance_id": int, 固定为 0
                - "is_center_box": bool, 固定为 False
            - 3. 推断专用 sidecar 字段:
                - "global_atom_indices": np.ndarray, (N_box,), int64, 当前 BOX 选中原子在全局原子数组中的索引
                - "box_position_zyx": tuple[int, int, int], 当前 BOX 在整图中的滑窗起始坐标(z0, y0, x0)
    """
    exp_raw = np.asarray(exp_raw, dtype=np.float32)
    if sim_raw is not None:
        sim_raw = np.asarray(sim_raw, dtype=np.float32)
    depth, height, width = [int(v) for v in exp_raw.shape]
    # np.ndarray, (3,), float64, 体素大小(x,y,z)
    voxel_size = np.asarray(voxel_size, dtype=np.float64).reshape(3)
    # np.ndarray, (3,), float64, 整图原点(x,y,z)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    # np.ndarray, (N_atom,), int64, 全局原子占位标签(推断时全零)
    atom_labels_placeholder = np.zeros(len(atom_coords_world), dtype=np.int64)

    # int, 三个空间轴末端需要补齐的 voxel 数
    pad_d = max(0, window_size - depth)
    pad_h = max(0, window_size - height)
    pad_w = max(0, window_size - width)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        exp_raw = np.pad(exp_raw, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant")
        if sim_raw is not None:
            sim_raw = np.pad(sim_raw, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant")
    padded_depth, padded_height, padded_width = [int(v) for v in exp_raw.shape]

    # list[int], 三轴滑窗起点; 最后一窗贴到 pad 后体积尾部
    z_starts = _compute_window_starts(padded_depth, window_size, stride)
    y_starts = _compute_window_starts(padded_height, window_size, stride)
    x_starts = _compute_window_starts(padded_width, window_size, stride)

    # np.ndarray, (3,), float32, 体素大小(x,y,z)
    voxel_size_f32 = voxel_size.astype(np.float32)

    box_specs: list[tuple[int, int, int, int]] = []
    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                box_specs.append((len(box_specs), z0, y0, x0))
    worker_count = max(1, int(num_box_workers))

    def build_one_box(box_spec: tuple[int, int, int, int]) -> dict[str, Any]:
        box_idx, z0, y0, x0 = box_spec
        z1 = min(z0 + window_size, padded_depth)
        y1 = min(y0 + window_size, padded_height)
        x1 = min(x0 + window_size, padded_width)

        # np.ndarray, (D_box,H_box,W_box), float32, 当前 BOX 的真实密度子块
        box_exp_raw = exp_raw[z0:z1, y0:y1, x0:x1].copy()
        # np.ndarray | None, (D_box,H_box,W_box), float32, 当前 BOX 的模拟密度子块
        box_sim_raw = None if sim_raw is None else sim_raw[z0:z1, y0:y1, x0:x1].copy()
        # np.ndarray, (3,), int64, 当前 BOX 的体素形状(z,y,x)
        box_shape_zyx = np.asarray(box_exp_raw.shape, dtype=np.int64)
        # np.ndarray, (3,), float32, 当前 BOX 的世界原点(x,y,z)
        box_origin_world = (origin + np.array([x0, y0, z0], dtype=np.float64) * voxel_size).astype(np.float32)
        # np.ndarray, (D_box,H_box,W_box), bool, 当前 BOX 的受体原子落点掩码
        box_receptor_mask = build_hardmask_from_world_coordinates(
            atom_coords_world=atom_coords_world,
            box_origin_world=box_origin_world,
            voxel_size_world=voxel_size_f32,
            box_shape_zyx=box_shape_zyx,
        ).astype(bool)
        # np.ndarray, (C,D_box,H_box,W_box), float32, BOX 内按训练同构方式构造的密度通道
        box_grid = build_density_channels(
            exp_raw=box_exp_raw,
            sim_raw=box_sim_raw,
            config=density_config,
            receptor_mask=box_receptor_mask,
        )
        # np.ndarray, (D_box,H_box,W_box), int64, 占位标签
        voxel_label = np.zeros(box_shape_zyx.tolist(), dtype=np.int64)

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

        # np.ndarray, (N_selected,), int64, 当前 BOX 选中的全局原子索引
        selected_idx = sample_dict.pop("_selected_idx")
        sample_dict = to_torch_sample(sample_dict)

        sample_dict["sample_name"] = f"infer_box_{box_idx}"
        sample_dict["pdb_id"] = "infer"
        sample_dict["class_name"] = "infer"
        sample_dict["instance_id"] = 0
        sample_dict["is_center_box"] = False
        sample_dict["global_atom_indices"] = selected_idx.copy()
        sample_dict["box_position_zyx"] = (z0, y0, x0)
        return sample_dict

    if worker_count == 1 or len(box_specs) <= 1:
        box_dicts = [build_one_box(box_spec) for box_spec in box_specs]
    else:
        box_dicts = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for chunk_start in range(0, len(box_specs), worker_count):
                chunk = box_specs[chunk_start:chunk_start + worker_count]
                futures = [executor.submit(build_one_box, box_spec) for box_spec in chunk]
                box_dicts.extend(future.result() for future in futures)

    print(f"[parse_input] 切分完成: {len(box_dicts)} 个 BOX "
          f"(raw_shape={(depth, height, width)}, padded_shape={exp_raw.shape}, "
          f"window={window_size}, stride={stride}, workers={worker_count})")
    return box_dicts




from typing import Iterator

def prepare_batched_boxes(
    box_dicts: list,
    batch_size: int,
    device: str,
) -> Iterator[dict]:
    """
    将 split_volume_to_boxes() 返回的 BOX 列表按 batch_size 分组, 对每组调用 box_point_collate() 产出 batch dict 。
    使用 yield 生成器返回，防止一次性将所有 batch 移至显存导致 OOM。

    输入参数:
        - box_dicts: list[dict], split_volume_to_boxes 的返回值
        - batch_size: int, 标量, 每个 batch 包含的 BOX 数
        - device: str, 目标设备

    输出 (yield):
        - batch_dict: dict[str, Any], 每个 iter 返回一个 batch, 包含:
            - 1. `box_point_collate()` 产出的标准字段:
                - "voxel_grid": torch.Tensor, (B, C, D_box, H_box, W_box), float32, 拼接后的 BOX 体素特征
                - "voxel_label": torch.Tensor, (B, D_box, H_box, W_box), int64, 拼接后的占位标签
                - "hardmask": torch.Tensor, (B, D_box, H_box, W_box), int64, 拼接后的几何 hardmask
                - "voxel_valid_mask": torch.Tensor, (B, D_box, H_box, W_box), bool, 拼接后的体素有效掩码
                - "box_origin_world": torch.Tensor, (B, 3), float32, 各 BOX 的世界坐标原点
                - "voxel_size_world": torch.Tensor, (B, 3), float32, 各 BOX 的体素大小
                - "box_shape_zyx": torch.Tensor, (B, 3), int64, 各 BOX 的体素形状
                - "atom_coord_world": torch.Tensor, (sumN, 3), float32, 当前 batch 全部原子的世界坐标
                - "atom_coord_local_voxel": torch.Tensor, (sumN, 3), float32, 当前 batch 全部原子的局部体素坐标
                - "atom_coord_centered_world": torch.Tensor, (sumN, 3), float32, 当前 batch 全部原子相对 BOX 中心的世界坐标
                - "atom_feat": torch.Tensor, (sumN, F), float32, 当前 batch 全部原子的特征
                - "atom_label": torch.Tensor, (sumN,), int64, 当前 batch 全部原子的占位标签
                - "atom_is_in_core_box": torch.Tensor, (sumN,), bool, 当前 batch 全部原子的 core 标记
                - "atom_valid_mask": torch.Tensor, (sumN,), bool, 当前 batch 全部原子的有效监督标记
                - "atom_batch_index": torch.Tensor, (sumN,), int64, 展平原子属于 batch 内哪个 BOX
                - "atom_counts": torch.Tensor, (B,), int64, 每个 BOX 的原子数
                - "atom_offsets": torch.Tensor, (B,), int64, 每个 BOX 在展平原子序列中的结束偏移
                - "sample_name": list[str], 长度 B, 每个 BOX 的样本名
                - "pdb_id": list[str], 长度 B, 每个 BOX 的 pdb_id
                - "class_name": list[str], 长度 B, 每个 BOX 的 class_name
                - "instance_id": list[int], 长度 B, 每个 BOX 的 instance_id
                - "is_center_box": list[bool], 长度 B, 每个 BOX 的中心框标记
            - 2. 推断阶段追加字段:
                - "atom_global_indices": torch.Tensor, (sumN,), int64, 展平后全部原子的全局原子索引, 与 atom_coord_world 一一对应
                - "_box_meta": list[dict[str, Any]], 长度 B, 每个 BOX 的推断元信息, 每项包含:
                    - "box_position_zyx": tuple[int, int, int], 当前 BOX 在整图中的滑窗起始坐标(z0, y0, x0)
    """
    from src.datasets.box_point_collate import box_point_collate
    from src.inference.get_pred import move_batch_to_device

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
        # list[list[int]]
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

        yield batch_dict
