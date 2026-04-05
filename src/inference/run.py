"""
run.py — 推断与评估的统一入口 (点云推断版): # see me: 只要不调 stride, 那么保存的的BOX级别的.npz总可以复用

通过 Hydra 管理所有配置参数, 串联 parse_input → get_pred → postprocess → evaluator 四个模块。

支持两种评估场景————只有在 eval_gt=True 时才试图提取 cif_gt_path:
    场景A : cif_path = 真实结构，cif_gt_path = None, 那么后者会回退到 cif_path，即模型输入特征和 GT 标签均来自同一 .cif 文件
    场景B (分离): cif_path = 建模结构（AF3/CryoAtom 等），cif_gt_path = 真实实验解析结构. 模型输入特征来自 cif_path，GT 标签来自 cif_gt_path

启动命令示例:
    # 单样本推断 (可以选择后续再评估也可以不评估)
    python src/inference/run.py --config=infer \\
        +class_folder="small_molecule" +sample_name="9f3f_0_0_0_0_C" \\
        ckpt_path="feedback/logs/.../last.ckpt"

    # 批量推断 + 评估
    python src/inference/run.py --config=infer mode=batch \\
        ckpt_path="feedback/logs/.../last.ckpt"

    # 网格调参搜索
    python src/inference/run.py --config=search_best_param \\
        mode=param_search ckpt_path="..."

支持的运行模式 (mode):
    - "raw_single":       单样本原始文件推断 (直接读取 .cif + .map, 全模型点云推断)
    - "raw_batch":        批量原始文件推断 + 可选评估 + Excel 汇总
    - "raw_param_search": 使用原始数据, 进行网格调参搜索 + Excel 对比

    - "legacy_*":     旧版体素推断入口 (从 legacy/ 加载)

旧版体素推断逻辑已迁移至 src/inference/legacy/run_voxel.py
"""

import os
import sys
import argparse
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, List, Optional, Union
import rootutils

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from pathlib import Path
POCKET_ROOT = Path(__file__).resolve().parent.parent.parent  # Pocket/
if str(POCKET_ROOT) in sys.path:
    sys.path.remove(str(POCKET_ROOT))
sys.path.insert(0, str(POCKET_ROOT))


# --- 导入推断模块 ---
from src.inference.parse_input import (
    load_from_raw_cif, load_gt_from_structure,
    split_volume_to_boxes, prepare_batched_boxes,
)
from src.inference.get_pred import load_model, run_point_inference, move_batch_to_device
from src.inference.postprocess import (
    init_box_spatial_weights, merge_box_atom_results, point_semantic_segment,
)
from src.inference.evaluator import semantic_evaluate, print_metrics
from src.inference.utils.utils import (
    write_batch_excel,
    write_param_search_excel,
    generate_param_grid,
    build_infer_vis_bundle,
)


# =============================================================================
# 可视化安全封装 (复用旧版逻辑)
# =============================================================================
def _build_vis_bundle_safe(
    cfg_dict: dict,
    output_root: str,   # 没用; （可视化根目录由 vis_output_root 显式指定）
    cif_path: str,
    map_path: str,
    cif_gt_path: str,
    pred_atom_coords: np.ndarray,
    prob_threshold: float,
    filter_preset: str,
    class_mapping: list,
    select_first_model: bool,
    pdb_id: str,
) -> dict:
    """
    安全封装可视化生成流程：统一处理开关、输出路径与异常捕获。

    输入参数:
        - cfg_dict: dict, 当前 Hydra 配置的扁平字典
        - output_root: str, 推断主输出目录（可视化根目录由 vis_output_root 显式指定）
        - cif_path: str, 推断用结构文件路径
        - map_path: str, 对应密度图路径
        - cif_gt_path: str | None, 真实结构路径（可为空）
        - pred_atom_coords: np.ndarray | None, (N_pred, 3), 预测为正类的原子点云（世界坐标）
        - prob_threshold: float | None, 预测阈值(仅用于写文件名)
        - filter_preset: str | None, 配体筛选预设
        - class_mapping: list | None, 标签类别映射
        - select_first_model: bool | None, 是否仅使用第一个 model
        - pdb_id: str | None, 样本 ID（为空则自动从路径推断）

    输出:
        - result: dict | None, build_infer_vis_bundle 的返回汇总；失败或关闭时为 None
    """
    # bool, 是否启用可视化
    vis_enable = cfg_dict.get("vis_enable", True)
    if not vis_enable:
        return None

    # str | None, 自定义可视化输出根目录
    vis_output_root = cfg_dict.get("vis_output_root")
    if not vis_output_root:
        print("[Vis] 未设置 vis_output_root，跳过可视化")
        return None
    # see me: vis_subdir, str | None, 如果存在, 那么输出目录就变为 os.path.join(vis_output_root, vis_subdir) 而不是 vis_output_root
    vis_subdir = cfg_dict.get("vis_subdir", None)
    if vis_subdir:
        vis_output_root = os.path.join(vis_output_root, vis_subdir)

    if not cif_path or not map_path:
        print("[Vis] 缺少 cif_path/map_path，跳过可视化")
        return None
    if filter_preset is None:
        print("[Vis] 未设置 filter_preset，跳过可视化")
        return None

    try:
        result = build_infer_vis_bundle(
            output_root=vis_output_root,
            cif_path=cif_path,
            map_path=map_path,
            cif_gt_path=cif_gt_path,
            pred_atom_coords=pred_atom_coords,
            prob_threshold=prob_threshold,
            filter_preset=filter_preset,
            class_mapping=class_mapping,
            pdb_id=pdb_id,
            select_first_model=select_first_model,
        )
        if result and result.get("root_dir"):
            print(f"[Vis] 已生成: {result['root_dir']}")
        return result
    except Exception as e:
        print(f"[Vis] 生成失败: {e}")
        return None




# =============================================================================
# 辅助函数
# =============================================================================
def _get_config_name() -> str:
    """
    从命令行参数中解析 --config 参数，用于指定 Hydra 配置文件名。
    用法: python src/inference/run.py --config=infer
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, help="Hydra config file name (without .yaml)")
    args, _ = parser.parse_known_args()
    # 清理 sys.argv，防止 hydra.main 报错无法识别 --config
    cleaned = []
    skip_next = False
    for arg in sys.argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--config":
            skip_next = True
            continue
        if arg.startswith("--config="):
            continue
        cleaned.append(arg)
    sys.argv = cleaned
    return args.config

_CONFIG_NAME = _get_config_name()


def _unpack_batch_results(batch_dict: dict, outputs: dict[str, Any]) -> list:
    """
    将一个 batch 的模型输出拆分回逐 BOX 的结果列表。

    输入参数:
        - batch_dict: dict, prepare_batched_boxes 产出的 batch dict
            含 _box_meta, atom_global_indices, atom_counts, atom_is_in_core_box, atom_coord_local_voxel, box_shape_zyx
        - outputs: dict[str, Any], 模型前向输出, 至少包含 atom_logits

    输出:
        - box_results: list[dict], 每个 dict 含:
            - "global_atom_indices": np.ndarray, (N_box,), int, 该 BOX 选中原子的全局索引
            - "atom_logits": np.ndarray, (N_box,), float32, 模型预测 logits
            - "atom_is_in_core": np.ndarray, (N_box,), bool
            - "atom_coord_local_voxel": np.ndarray, (N_box, 3), float32
            - "box_shape_zyx": np.ndarray, (3,), int64
            - "box_position_zyx": tuple[int, int, int]
    """
    # list[dict], 每个 BOX 的推断专用元信息
    box_meta_list = batch_dict["_box_meta"]
    # torch.Tensor, (B,), long, 每个 BOX 的原子数
    atom_counts = outputs.get("atom_counts", batch_dict["atom_counts"])
    # torch.Tensor, (sumN,), bool
    atom_is_in_core = outputs.get("atom_is_in_core_box", batch_dict["atom_is_in_core_box"]).cpu().numpy()
    # torch.Tensor, (sumN, 3), float32
    atom_coord_local = outputs.get("atom_coord_local_voxel", batch_dict["atom_coord_local_voxel"]).cpu().numpy()
    atom_global_indices = outputs.get("atom_global_indices", batch_dict.get("atom_global_indices"))
    # torch.Tensor, (B, 3), int64
    box_shapes = batch_dict["box_shape_zyx"].cpu().numpy()

    # np.ndarray, (sumN, C=1) 或 (sumN,), float32
    atom_logits = outputs["atom_logits"]
    logits_np = atom_logits.cpu().numpy()
    if logits_np.ndim == 2:
        if logits_np.shape[1] == 1:
            logits_np = logits_np[:, 0]
        else:
            raise ValueError(f"atom_logits has unexpected shape: {logits_np.shape}")
    if atom_global_indices is None:
        raise RuntimeError("Inference batch is missing atom_global_indices, cannot map logits back to global atoms.")
    atom_global_indices_np = atom_global_indices.cpu().numpy()

    box_results = []
    offset = 0
    for i, count in enumerate(atom_counts.tolist()):
        count = int(count)
        box_results.append({
            "global_atom_indices": atom_global_indices_np[offset:offset + count].copy(),
            "atom_logits": logits_np[offset:offset + count].copy(),
            "atom_is_in_core": atom_is_in_core[offset:offset + count].copy(),
            "atom_coord_local_voxel": atom_coord_local[offset:offset + count].copy(),
            "box_shape_zyx": box_shapes[i].copy(),
            "box_position_zyx": box_meta_list[i]["box_position_zyx"],
        })
        offset += count

    if offset != logits_np.shape[0]:
        raise RuntimeError(
            f"Inference atom slicing mismatch: consumed={offset}, logits={logits_np.shape[0]}, "
            f"atom_counts={tuple(int(v) for v in atom_counts.tolist())}"
        )
    return box_results










# =============================================================================
# 点云推断核心管线
# =============================================================================
def run_raw_point_pipeline(
    model: torch.nn.Module,
    device: torch.device,
    cif_path: str,
    map_path: str,
    cif_gt_path: str,           # 如果为 None 且要求评估, 则回退到 cif_path

    target_voxel_size: float,   # 建议值 1.0
    compute_density: bool,
    select_first_model: bool,
    eval_gt: bool,

    filter_preset: str,         # None 表示不筛选
    class_mapping: list,        # None 表示不做映射
    threshold: float,           # 建议值 0.5
    dist_threshold: float,      # 建议值 3.0

    core_decay_mode: str,       # "hard" / "linear" / "none", 建议值 "linear"
    core_offset: int,           # 建议值 2
    atom_buffer_radius: float,  # 建议值 4.0
    valid_crop_margin: int,     # 建议值 0
    box_spatial_weight_sigma_ratio: float,  # 建议值 0.5
    merge_mode: str,            # "logit_mean" / "prob_mean"
    semantic_segment_method: str,  # "threshold" / "dbscan"
    dbscan_eps: float,          # 建议值 2.0
    dbscan_min_samples: int,    # 建议值 3

    stride: int,                # 建议值 36
    windows_size: int,          # 建议值 80
    batch_size: int,            # 建议值 3
    output_dir: str,            # None 则不保存
    show_progress: bool,
    error_dir: str,             # None 则不记录
) -> dict:
    """
    单样本点云推断完整流水线。
    
    保存结果:(路径如 output_dir / atom_probs.npz ) 
        - atom_probs.npz: 原子概率
        - pred_atom_coords.npz: 预测原子坐标
        - gt_points.npz: GT 原子坐标 (可选)

    输入参数:
        - model: nn.Module, eval 模式的 VolumePointStage1Model
        - device: torch.device, 推断设备
        - cif_path: str, 原始结构文件路径 (.cif/.pdb), 必需
        - map_path: str, 对应的 EMDB 密度图路径, 必需
        - cif_gt_path: str | None, 真实结构路径, 仅用于 GT 标签提取评估: 在GT模式下若为None则回退到 cif_path

        - target_voxel_size: float, 重采样目标体素大小 (Å), 建议值 1.0
        - compute_density: bool, 是否计算原子局部密度特征
        - select_first_model: bool, 多模型 CIF 时是否仅取第一个
        - eval_gt: bool, 是否提取 GT 标签进行评估
        - filter_preset: str, 配体筛选预设名 (用于 GT 提取)
        - class_mapping: list[int] | None, 标签类别映射表
        - threshold: float, 语义分割阈值, 建议值 0.5
        - dist_threshold: float, 点云评估距离阈值 (Å), 建议值 3.0

        - core_decay_mode: str, 核心区衰减方式 ("hard" / "linear" / "none")
        - core_offset: int, 裁边 voxel 数, 建议值 10(这个参数是与 core_decay_mode 配合使用的)
        - atom_buffer_radius: float, 原子 buffer 半径 (Å), 建议与训练对齐
        - valid_crop_margin: int, 监督区域裁边量(注意这是原始训练数据的一部分), 建议与训练对齐
        - box_spatial_weight_sigma_ratio: float, 空间权重高斯核 sigma 与密度图半径的比值, 建议值 0.5
        - merge_mode: str, 多 BOX 聚合方式 ("logit_mean" / "prob_mean")
        - semantic_segment_method: str, 推断后处理方式 ("threshold" / "dbscan")
        - dbscan_eps: float, DBSCAN 半径参数 eps, 建议值 2.0
        - dbscan_min_samples: int, DBSCAN 最少邻居数, 建议值 3

        - stride: int, 滑窗步幅, 建议 40
        - windows_size: int, 滑窗边长, 建议与训练对齐(80)
        - batch_size: int, 推断 batch size, 建议值 2
        - output_dir: str | None, 输出目录
        - show_progress: bool, 是否显示进度条
        - error_dir: str | None, 错误日志目录

    输出:
        - result: dict, 包含:
            - "sample_name": str
            - "atom_probs": np.ndarray, (N_atom,), 每个原子的概率
            - "pred_atom_coords": np.ndarray, (N_pred, 3), 预测正类点云
            - "atom_coords": np.ndarray, (N_atom, 3), 全部原子坐标
            - "metrics": dict | None, 评估指标（eval_gt=True 且结构中有配体时）
            - "error": str | None, 错误信息
            - "all_box_results": list[dict], 缓存用的逐 BOX logits
    """
    sample_name = Path(cif_path).stem
    result = {
        "sample_name":  sample_name,
        "class_folder": "raw",
        "metrics":      None,
        "error":        None,
    }

    try:
        # ---- 1. 加载原始数据 ----
        data = load_from_raw_cif(
            cif_path=cif_path,
            map_path=map_path,
            target_voxel_size=target_voxel_size,
            compute_density=compute_density,
            select_first_model=select_first_model,
            error_dir=error_dir,
        )
        # np.ndarray, float32, (C, D, H, W)
        grid = data["grid"]
        atom_coords = data["atom_coords"]   # np.ndarray, (N_atom, 3)
        origin = data["origin"]             # np.ndarray, (3,)
        voxel_sz = data["voxel_size"]       # np.ndarray, (3,)
        emdb_channels = data["emdb_channels"]  # int
        atom_feat = data["atom_feat"]         # np.ndarray, (N_atom, F)
        if show_progress:
            print(f"  [run] 原子特征: {atom_feat.shape}, 体素网格: {grid.shape}")

        # ---- 2. 可选: 提取 GT 标签————只有在 eval_gt 时才试图提取 ----
        gt_data = None
        if eval_gt:
            # str, 场景A(无 cif_gt_path)时回退到 cif_path、场景B(有 cif_gt_path)时使用真实结构
            _effective_gt_cif = cif_gt_path if cif_gt_path else cif_path
            gt_data = load_gt_from_structure(
                cif_path=cif_path,              # 受体结构（场景B时为预测结构）
                cif_gt_path=_effective_gt_cif,  # 真实结构（含配体信息）；场景A时与 cif_path 相同
                map_path=map_path,
                target_voxel_size=target_voxel_size,
                filter_preset=filter_preset,
                class_mapping=class_mapping,
                select_first_model=select_first_model,
                error_dir=error_dir,
            )


        # ---- 3. 切分 BOX ----
        box_dicts = split_volume_to_boxes(
            grid=grid,
            atom_coords_world=atom_coords,
            atom_feat=atom_feat,
            origin=origin,
            voxel_size=voxel_sz,
            window_size=windows_size,
            stride=stride,
            atom_buffer_radius=atom_buffer_radius,
            valid_crop_margin=valid_crop_margin,
            emdb_channels=emdb_channels,
        )


        # ---- 4. 分 batch 推断 ----
        batched = prepare_batched_boxes(box_dicts, batch_size, device)
        all_box_results = []

        import tqdm as _tqdm
        total_boxes = len(box_dicts)
        pbar = _tqdm.tqdm(
            desc="  点云推断",
            total=total_boxes,
            file=sys.stdout,
            position=0,
            leave=False,
            disable=not show_progress,
        )
        for batch_dict in batched:
            outputs = run_point_inference(model, device, batch_dict)
            batch_results = _unpack_batch_results(batch_dict, outputs)
            all_box_results.extend(batch_results)
            pbar.update(len(batch_results))
        pbar.close()


        # ---- 5. 聚合 ----
        atom_probs = merge_box_atom_results(
            box_results=all_box_results,
            total_atom_count=len(atom_coords),
            core_decay_mode=core_decay_mode,
            core_offset=core_offset,
            merge_mode=merge_mode,
            voxel_size=voxel_sz,
            window_size=windows_size,
            box_spatial_weight_sigma_ratio=box_spatial_weight_sigma_ratio,
        )


        # ---- 6. 二值化 ----
        pred_atom_coords = point_semantic_segment(
            atom_probs=atom_probs,
            atom_coords=atom_coords,
            threshold=threshold,
            semantic_segment_method=semantic_segment_method,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
        )

        result["atom_probs"] = atom_probs
        result["pred_atom_coords"] = pred_atom_coords
        result["atom_coords"] = atom_coords
        result["origin"] = origin
        result["voxel_size"] = voxel_sz
        result["all_box_results"] = all_box_results

        if show_progress:
            print(f"  [run] 预测正类原子: {pred_atom_coords.shape[0]} / {len(atom_coords)}")


        # ---- 7. 评估 (可选) ----
        if gt_data is not None:
            metrics = semantic_evaluate(
                pred_atom_coords=pred_atom_coords,
                atom_gt=gt_data["atom_gt"],
                dist_threshold=dist_threshold,
            )
            result["metrics"] = metrics
            if show_progress:
                print_metrics(metrics, prefix="  ")


        # ---- 8. 保存结果 ----  # NOTE
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            # 保存原子概率
            np.savez_compressed(
                os.path.join(output_dir, "atom_probs.npz"),
                atom_probs=atom_probs,
                atom_coords=atom_coords,
            )
            # 保存预测点云
            np.savez_compressed(
                os.path.join(output_dir, "pred_atom_coords.npz"),
                pred_atom_coords=pred_atom_coords,
            )
            if gt_data is not None:
                np.savez_compressed(
                    os.path.join(output_dir, "gt_points.npz"),
                    atom_gt=gt_data["atom_gt"],
                    atom_coords=gt_data["atom_coords"],
                )
            if show_progress:
                print(f"  [run] 结果已保存: {output_dir}")

    except Exception as e:
        result["error"] = str(e)
        import traceback
        print(f"  ❌ 推断失败: {e}")
        traceback.print_exc()

    return result


def _save_box_logits_cache(
    all_box_results: list,
    save_path: str,
    atom_coords: np.ndarray,
) -> None:
    """
    将逐 BOX 的 logits 和元信息保存为 .npz 文件, 用于参数搜索时跳过模型推断。

    输入参数:
        - all_box_results: list[dict], 每个 BOX 的结果
        - save_path: str, 保存路径
        - atom_coords: np.ndarray, (N_atom, 3), 全部原子坐标
    """
    # 将所有 BOX 信息打包为数组列表
    n_boxes = len(all_box_results)
    save_dict = {
        "n_boxes": np.array(n_boxes, dtype=np.int64),
        "atom_coords": atom_coords,
    }
    for i, br in enumerate(all_box_results):
        save_dict[f"box_{i}_global_indices"] = br["global_atom_indices"]
        save_dict[f"box_{i}_logits"] = br["atom_logits"]
        save_dict[f"box_{i}_is_in_core"] = br["atom_is_in_core"]
        save_dict[f"box_{i}_coord_local"] = br["atom_coord_local_voxel"]
        save_dict[f"box_{i}_shape_zyx"] = br["box_shape_zyx"]
        save_dict[f"box_{i}_confidence_weight"] = np.array(float(br.get("box_confidence_weight", 1.0)), dtype=np.float32)

    np.savez_compressed(save_path, **save_dict)


def _load_box_logits_cache(cache_path: str) -> tuple:
    """
    从 .npz 缓存加载逐 BOX logits。

    输入参数:
        - cache_path: str, .npz 文件路径

    输出:
        - all_box_results: list[dict]
        - atom_coords: np.ndarray, (N_atom, 3)
    """
    with np.load(cache_path, allow_pickle=False) as cache:
        n_boxes = int(cache["n_boxes"])
        atom_coords = cache["atom_coords"]
        all_box_results = []
        for i in range(n_boxes):
            all_box_results.append({
                "global_atom_indices": cache[f"box_{i}_global_indices"],
                "atom_logits": cache[f"box_{i}_logits"],
                "atom_is_in_core": cache[f"box_{i}_is_in_core"],
                "atom_coord_local_voxel": cache[f"box_{i}_coord_local"],
                "box_shape_zyx": cache[f"box_{i}_shape_zyx"],
                "box_confidence_weight": float(cache[f"box_{i}_confidence_weight"]),
            })
    return all_box_results, atom_coords









# =============================================================================
# mode 处理函数
# =============================================================================
def _run_raw_single_mode(cfg_dict, model, device, output_root):
    """mode=raw_single: 单样本原始文件推断 + 可选 GT 评估"""
    cif_path = cfg_dict.get("cif_path")
    map_path = cfg_dict.get("map_path")
    if not cif_path or not map_path:
        raise ValueError(
            "[错误] mode=raw_single 需要指定 cif_path 和 map_path。\n"
            "用法: +cif_path=\"xxx.cif\" +map_path=\"xxx.map\""
        )

    cif_gt_path = cfg_dict.get("cif_gt_path")
    sample_output_dir = os.path.join(output_root, Path(cif_path).stem) if output_root else None

    result = run_raw_point_pipeline(
        model=model, device=device,
        cif_path=cif_path, map_path=map_path, cif_gt_path=cif_gt_path,
        target_voxel_size=cfg_dict["target_voxel_size"],
        compute_density=cfg_dict["compute_density"],
        select_first_model=cfg_dict["select_first_model"],
        eval_gt=cfg_dict["eval_gt"],
        filter_preset=cfg_dict.get("filter_preset"),
        class_mapping=cfg_dict.get("class_mapping"),
        threshold=cfg_dict["threshold"],
        dist_threshold=cfg_dict["dist_threshold"],
        merge_mode=cfg_dict["merge_mode"],
        semantic_segment_method=cfg_dict["semantic_segment_method"],
        dbscan_eps=cfg_dict["dbscan_eps"],
        dbscan_min_samples=cfg_dict["dbscan_min_samples"],
        core_decay_mode=cfg_dict["core_decay_mode"],
        core_offset=cfg_dict["core_offset"],
        atom_buffer_radius=cfg_dict["atom_buffer_radius"],
        valid_crop_margin=cfg_dict["valid_crop_margin"],
        box_spatial_weight_sigma_ratio=cfg_dict["box_spatial_weight_sigma_ratio"],
        stride=cfg_dict["stride"],
        windows_size=cfg_dict["windows_size"],
        batch_size=cfg_dict["batch_size"],
        output_dir=sample_output_dir,
        show_progress=True,
        error_dir=cfg_dict.get("error_dir"),
    )

    # 可视化
    if result.get("error") is None:
        _build_vis_bundle_safe(
            cfg_dict=cfg_dict,
            output_root=output_root,
            cif_path=cif_path,
            map_path=map_path,
            cif_gt_path=cif_gt_path,
            pred_atom_coords=result.get("pred_atom_coords"),
            prob_threshold=cfg_dict["threshold"],
            filter_preset=cfg_dict.get("filter_preset"),
            class_mapping=cfg_dict.get("class_mapping"),
            select_first_model=cfg_dict["select_first_model"],
            pdb_id=None,
        )

    return result


def _run_raw_batch_mode(cfg_dict, model, device, output_root):
    """mode=raw_batch: 批量原始文件推断 + Excel 汇总"""
    from src.inference.utils.yield_json_from_raw_sample import load_raw_pairs

    raw_pairs_json = cfg_dict.get("raw_pairs_json")
    if not raw_pairs_json:
        raise ValueError("[错误] mode=raw_batch 需要指定 raw_pairs_json 路径")

    pairs = load_raw_pairs(raw_pairs_json)
    print(f"[raw_batch] 共 {len(pairs)} 个样本")

    all_results = []
    for i, (cif_p, map_p, cif_gt_p) in enumerate(pairs):
        sname = Path(cif_p).stem
        print(f"\n[{i+1}/{len(pairs)}] {sname}")
        sample_out = os.path.join(output_root, sname) if output_root else None

        r = run_raw_point_pipeline(
            model=model, device=device,
            cif_path=cif_p, map_path=map_p, cif_gt_path=cif_gt_p,
            target_voxel_size=cfg_dict["target_voxel_size"],
            compute_density=cfg_dict["compute_density"],
            select_first_model=cfg_dict["select_first_model"],
            eval_gt=cfg_dict["eval_gt"],
            filter_preset=cfg_dict.get("filter_preset"),
            class_mapping=cfg_dict.get("class_mapping"),
            threshold=cfg_dict["threshold"],
            dist_threshold=cfg_dict["dist_threshold"],
            merge_mode=cfg_dict["merge_mode"],
            semantic_segment_method=cfg_dict["semantic_segment_method"],
            dbscan_eps=cfg_dict["dbscan_eps"],
            dbscan_min_samples=cfg_dict["dbscan_min_samples"],
            core_decay_mode=cfg_dict["core_decay_mode"],
            core_offset=cfg_dict["core_offset"],
            atom_buffer_radius=cfg_dict["atom_buffer_radius"],
            valid_crop_margin=cfg_dict["valid_crop_margin"],
            box_spatial_weight_sigma_ratio=cfg_dict["box_spatial_weight_sigma_ratio"],
            stride=cfg_dict["stride"],
            windows_size=cfg_dict["windows_size"],
            batch_size=cfg_dict["batch_size"],
            output_dir=sample_out,
            show_progress=True,
            error_dir=cfg_dict.get("error_dir"),
        )

        # 可视化
        if r.get("error") is None:
            _build_vis_bundle_safe(
                cfg_dict=cfg_dict, output_root=output_root,
                cif_path=cif_p, map_path=map_p, cif_gt_path=cif_gt_p,
                pred_atom_coords=r.get("pred_atom_coords"),
                prob_threshold=cfg_dict["threshold"],
                filter_preset=cfg_dict.get("filter_preset"),
                class_mapping=cfg_dict.get("class_mapping"),
                select_first_model=cfg_dict["select_first_model"],
                pdb_id=sname,
            )

        # 释放大数组
        for key in ["all_box_results", "atom_probs", "pred_atom_coords", "atom_coords"]:
            r.pop(key, None)
        all_results.append(r)

    # Excel 汇总
    if output_root and cfg_dict.get("eval_gt", False):
        excel_path = write_batch_excel(all_results, output_root)
        print(f"[raw_batch] Excel 汇总已保存: {excel_path}")


def _run_raw_param_search_mode(cfg_dict, model, device, output_root):
    """mode=raw_param_search: 参数搜索, 缓存逐 BOX logits, 只重新做聚合+二值化"""
    from src.inference.utils.yield_json_from_raw_sample import load_raw_pairs

    raw_pairs_json = cfg_dict.get("raw_pairs_json")
    if not raw_pairs_json:
        raise ValueError("[错误] mode=raw_param_search 需要指定 raw_pairs_json 路径")

    pairs = load_raw_pairs(raw_pairs_json)
    print(f"[raw_param_search] 共 {len(pairs)} 个样本")

    # 参数网格
    search_params = cfg_dict.get("search_params", {})
    param_grid = generate_param_grid(search_params)
    if not param_grid:
        raise ValueError("[错误] search_params 为空")
    param_names = list(param_grid[0].keys())
    print(f"[raw_param_search] 参数组合数: {len(param_grid)}")



    # 1. 对每个样本做一次完整推断, 缓存 logits
    npz_root = os.path.join(output_root, "_logits_cache")   # FIXME: 未来可以单独指定缓存文件夹, 共同存放共同读取
    os.makedirs(npz_root, exist_ok=True)
    delete_cache = cfg_dict.get("delete_cache_after_search", False)

    cached_info = []
    for i, (cif_p, map_p, cif_gt_p) in enumerate(pairs):
        sname = Path(cif_p).stem
        npz_path = os.path.join(npz_root, f"{sname}.npz")
        info = {"name": sname, "cif_path": cif_p, "map_path": map_p, "cif_gt_path": cif_gt_p,
                "npz_path": npz_path, "error": None, "gt_data": None}

        if os.path.exists(npz_path):
            print(f"  [{i+1}/{len(pairs)}] {sname} → 已有缓存, 跳过推断")
        else:
            print(f"  [{i+1}/{len(pairs)}] {sname} → 执行推断并缓存 logits")
            r = run_raw_point_pipeline(
                model=model, device=device,
                cif_path=cif_p, map_path=map_p, cif_gt_path=cif_gt_p,
                target_voxel_size=cfg_dict["target_voxel_size"],
                compute_density=cfg_dict["compute_density"],
                select_first_model=cfg_dict["select_first_model"],
                eval_gt=False,
                filter_preset=cfg_dict.get("filter_preset"),
                class_mapping=cfg_dict.get("class_mapping"),
                threshold=cfg_dict["threshold"],
                dist_threshold=cfg_dict["dist_threshold"],
                merge_mode=cfg_dict["merge_mode"],
                semantic_segment_method=cfg_dict["semantic_segment_method"],
                dbscan_eps=cfg_dict["dbscan_eps"],
                dbscan_min_samples=cfg_dict["dbscan_min_samples"],
                core_decay_mode="none", core_offset=0,
                atom_buffer_radius=cfg_dict["atom_buffer_radius"],
                valid_crop_margin=cfg_dict["valid_crop_margin"],
                box_spatial_weight_sigma_ratio=cfg_dict["box_spatial_weight_sigma_ratio"],
                stride=cfg_dict["stride"],
                windows_size=cfg_dict["windows_size"],
                batch_size=cfg_dict["batch_size"],
                output_dir=None,
                show_progress=True,
                error_dir=cfg_dict.get("error_dir"),
            )
            if r["error"] is not None:
                info["error"] = r["error"]
                cached_info.append(info)
                continue
            _save_box_logits_cache(
                all_box_results=r["all_box_results"],
                save_path=npz_path,
                atom_coords=r["atom_coords"],
            )

        if cfg_dict["eval_gt"]:
            try:
                _gt_cif = cif_gt_p if cif_gt_p else cif_p
                gt = load_gt_from_structure(
                    cif_path=cif_p,
                    cif_gt_path=_gt_cif,
                    map_path=map_p,
                    target_voxel_size=cfg_dict["target_voxel_size"],
                    filter_preset=cfg_dict.get("filter_preset"),
                    class_mapping=cfg_dict.get("class_mapping"),
                    select_first_model=cfg_dict["select_first_model"],
                    error_dir=cfg_dict.get("error_dir"),
                )
                info["gt_data"] = gt
            except Exception as e:
                print(f"  ⚠️ GT 提取失败: {e}")
        cached_info.append(info)




    # 2. 对每个参数组合做聚合+评估
    summary = []
    for pi, params in enumerate(param_grid):
        print(f"\n[raw_param_search] 参数 {pi+1}/{len(param_grid)}: {params}")
        sample_metrics = []
        per_sample_records = []

        for info in cached_info:
            if info["error"] is not None:
                continue
            if not os.path.exists(info["npz_path"]):
                continue

            box_results, atom_coords = _load_box_logits_cache(info["npz_path"])

            _core_decay = params.get("core_decay_mode", cfg_dict["core_decay_mode"])
            _core_off = params.get("core_offset", cfg_dict["core_offset"])
            _threshold = params.get("threshold", cfg_dict["threshold"])
            _merge_mode = params.get("merge_mode", cfg_dict["merge_mode"])
            _semantic_segment_method = params.get("semantic_segment_method", cfg_dict["semantic_segment_method"])
            _dbscan_eps = params.get("dbscan_eps", cfg_dict["dbscan_eps"])
            _dbscan_min_samples = params.get("dbscan_min_samples", cfg_dict["dbscan_min_samples"])

            atom_probs = merge_box_atom_results(
                box_results=box_results,
                total_atom_count=len(atom_coords),
                core_decay_mode=_core_decay,
                core_offset=_core_off,
                merge_mode=_merge_mode,
                voxel_size=cfg_dict["target_voxel_size"],
                window_size=cfg_dict["windows_size"],
                box_spatial_weight_sigma_ratio=cfg_dict["box_spatial_weight_sigma_ratio"],
            )
            pred_coords = point_semantic_segment(
                atom_probs=atom_probs,
                atom_coords=atom_coords,
                threshold=_threshold,
                semantic_segment_method=_semantic_segment_method,
                dbscan_eps=_dbscan_eps,
                dbscan_min_samples=_dbscan_min_samples,
            )

            if info.get("gt_data") is not None:
                _dist_thresh = params.get("dist_threshold", cfg_dict["dist_threshold"])
                metrics = semantic_evaluate(
                    pred_atom_coords=pred_coords,
                    atom_gt=info["gt_data"]["atom_gt"],
                    dist_threshold=_dist_thresh,
                )
                metrics["sample_name"] = info["name"]
                sample_metrics.append(metrics)
                per_sample_records.append(
                    {
                        "sample_name": info["name"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "iou": metrics["iou"],
                    }
                )

        if sample_metrics:
            avg_p = float(np.mean([record["precision"] for record in sample_metrics]))
            avg_r = float(np.mean([record["recall"] for record in sample_metrics]))
            avg_f1 = float(np.mean([record["f1"] for record in sample_metrics]))
            avg_iou = float(np.mean([record["iou"] for record in sample_metrics]))
        else:
            avg_p = avg_r = avg_f1 = avg_iou = 0.0

        row = {
            **params,
            "avg_P": avg_p,
            "avg_R": avg_r,
            "avg_F1": avg_f1,
            "avg_IoU": avg_iou,
        }
        if per_sample_records:
            row["_per_sample"] = per_sample_records
        summary.append(row)
        print(
            f"  → avg_P={avg_p:.4f}  avg_R={avg_r:.4f}  "
            f"avg_F1={avg_f1:.4f}  avg_IoU={avg_iou:.4f}"
        )

    # 3. Excel 汇总
    summary.sort(key=lambda row: row["avg_F1"], reverse=True)
    best = summary[0]
    if output_root:
        write_param_search_excel(summary, param_names, output_root)

    print("\n" + "=" * 60)
    print("  [raw_param_search] 🏆 最优参数组合:")
    for key, value in best.items():
        if key == "_per_sample":
            continue
        print(f"    {key}: {round(value, 4) if isinstance(value, float) else value}")
    print("=" * 60)




    # 4. 最优参数下的可视化（复用缓存 logits，不重复跑完整模型）
    if cfg_dict.get("vis_enable", False):
        print("\n[raw_param_search] 开始使用最优参数生成可视化")
        best_threshold = best.get("threshold", cfg_dict["threshold"])
        best_core_decay = best.get("core_decay_mode", cfg_dict["core_decay_mode"])
        best_core_offset = best.get("core_offset", cfg_dict["core_offset"])
        best_merge_mode = best.get("merge_mode", cfg_dict["merge_mode"])
        best_semantic_segment_method = best.get("semantic_segment_method", cfg_dict["semantic_segment_method"])
        best_dbscan_eps = best.get("dbscan_eps", cfg_dict["dbscan_eps"])
        best_dbscan_min_samples = best.get("dbscan_min_samples", cfg_dict["dbscan_min_samples"])

        for i, info in enumerate(cached_info):
            if info["error"] is not None:
                continue
            if not os.path.exists(info["npz_path"]):
                continue

            print(f"[raw_param_search][正在可视化 {i+1}/{len(cached_info)}] {info['name']}")
            box_results, atom_coords = _load_box_logits_cache(info["npz_path"])
            atom_probs = merge_box_atom_results(
                box_results=box_results,
                total_atom_count=len(atom_coords),
                core_decay_mode=best_core_decay,
                core_offset=best_core_offset,
                merge_mode=best_merge_mode,
                voxel_size=cfg_dict["target_voxel_size"],
                window_size=cfg_dict["windows_size"],
                box_spatial_weight_sigma_ratio=cfg_dict["box_spatial_weight_sigma_ratio"],
            )
            pred_coords = point_semantic_segment(
                atom_probs=atom_probs,
                atom_coords=atom_coords,
                threshold=best_threshold,
                semantic_segment_method=best_semantic_segment_method,
                dbscan_eps=best_dbscan_eps,
                dbscan_min_samples=best_dbscan_min_samples,
            )
            _build_vis_bundle_safe(
                cfg_dict=cfg_dict,
                output_root=output_root,
                cif_path=info["cif_path"],
                map_path=info["map_path"],
                cif_gt_path=info["cif_gt_path"],
                pred_atom_coords=pred_coords,
                prob_threshold=best_threshold,
                filter_preset=cfg_dict.get("filter_preset"),
                class_mapping=cfg_dict.get("class_mapping"),
                select_first_model=cfg_dict["select_first_model"],
                pdb_id=info["name"],
            )

    # 5. 清理缓存
    if delete_cache:
        print(f"[raw_param_search] 🧹 清除临时缓存 ({npz_root})")
        for info in cached_info:
            if info["error"] is None and os.path.exists(info["npz_path"]):
                try:
                    os.remove(info["npz_path"])
                except Exception as e:
                    print(f"  ❌ 删除失败: {e}")

    print("\n🎉 参数搜索完毕！\n")


# =============================================================================
# 主入口
# =============================================================================
@hydra.main(version_base="1.3", config_path="../../configs/infer_or_eval", config_name=_CONFIG_NAME)
def main(cfg: DictConfig):
    """
    推断与评估的统一主入口。按 cfg.mode 分流:
        raw_single / raw_batch / raw_param_search → 点云推断
        single / batch / param_search → BOX 推断 (点云路径)
        legacy_* → 旧版体素推断
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    ckpt_path = cfg_dict.get("ckpt_path")
    if not ckpt_path:
        raise ValueError(
            "[错误] 未指定 ckpt_path，请通过命令行传入: "
            "ckpt_path=\"feedback/logs/.../checkpoints/last.ckpt\""
        )

    device_str = cfg_dict.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    mode = cfg_dict["mode"]
    output_root = cfg_dict["output_root"]
    os.makedirs(output_root, exist_ok=True)

    print("=" * 60)
    print(f"  [inference/run] 点云推断与评估")
    print(f"  模式: {mode}  |  配置: {_CONFIG_NAME}.yaml")
    print(f"  设备: {device_str}  |  checkpoint: {ckpt_path}")
    print("=" * 60)

    # 加载模型
    backbone_override = cfg_dict.get("backbone_override", None)
    model = load_model(ckpt_path, device, backbone_override=backbone_override)

    # 按 mode 执行
    if mode == "raw_single":
        _run_raw_single_mode(cfg_dict, model, device, output_root)
    elif mode == "raw_batch":
        _run_raw_batch_mode(cfg_dict, model, device, output_root)
    elif mode == "raw_param_search":
        _run_raw_param_search_mode(cfg_dict, model, device, output_root)




    elif mode.startswith("legacy_"):
        # 旧版体素推断入口
        from src.inference.legacy.run_voxel import main as legacy_main
        print("[run] 使用旧版体素推断入口 (legacy)")
        legacy_main(cfg)
    else:
        raise ValueError(
            f"[错误] 未知 mode: '{mode}'，"
            "请在配置中设置 mode = raw_single / raw_batch / raw_param_search / "
            "single / batch / param_search / legacy_*"
        )



if __name__ == "__main__":
    main()
