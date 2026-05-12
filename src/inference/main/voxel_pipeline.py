from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.datasets.density_channel_builder import ALL_CHANNEL_NAMES
from src.inference.get_pred import get_voxel_pred
from src.inference.parse_input import load_from_raw_cif, split_volume_to_boxes
from src.inference.voxel_evaluator import evaluate_instance_mask, evaluate_voxel_mask
from src.inference.utils.yield_json_from_raw_sample import load_raw_pairs
from src.inference.voxel_gt import load_ligand_gt_from_labels_npz, load_ligand_gt_from_structure
from src.inference.voxel_postprocess import postprocess_ligand_probability_map
from src.inference.voxel_tuning import (
    evaluate_postprocess_params_on_cache_set,
    load_voxel_prediction_cache,
    optimize_postprocess_params,
    save_voxel_prediction_cache,
)
from src.inference.utils.voxel_types import VoxelPredCacheData, VoxelPostprocessResult
from src.inference.utils.utils import build_infer_vis_bundle, write_voxel_batch_excel


# -------------------------------------- single / batch 模式下的推理逻辑 --------------------------------------
def _build_cache_or_forward(
    cfg_dict: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    cache_path: str,
) -> VoxelPredCacheData:
    """
    如果存在缓存文件 cache_path 那就简单返回; 如果不存在就从头推理、保存并返回缓存。

    输入参数:
        - cfg_dict: dict[str, Any], 当前样本配置
        - model: torch.nn.Module, stage1 模型
        - device: torch.device, 推理设备
        - cache_path: str, 当前样本缓存路径; 调用方显式控制缓存命名

    输出:
        - cache_data: VoxelPredCacheData, 概率缓存数据
    """
    if bool(_get_cfg(cfg_dict, "use_cache", True)) and os.path.exists(cache_path):
        return load_voxel_prediction_cache(cache_path)

    # str | None, 当前 forward 实际使用的模拟密度图路径
    effective_sim_map_path = _resolve_effective_sim_map_path(cfg_dict)
    raw_data = load_from_raw_cif(
        cif_path=str(_get_cfg(cfg_dict, "cif_path", True)),
        map_path=str(_get_cfg(cfg_dict, "map_path", True)),
        sim_map_path=effective_sim_map_path,
        target_voxel_size=float(_get_cfg(cfg_dict, "target_voxel_size", True)),
        compute_density=bool(_get_cfg(cfg_dict, "compute_density", True)),
        select_first_model=bool(_get_cfg(cfg_dict, "select_first_model", True)),
        error_dir=_get_cfg(cfg_dict, "error_dir", False),
        density_channel_config=_get_cfg(cfg_dict, "density_channel_config", True),
    )
    box_dicts = split_volume_to_boxes(
        exp_raw=raw_data["resampled_emdb"],
        sim_raw=raw_data["resampled_sim"],
        density_config=raw_data["density_config"],
        atom_coords_world=raw_data["atom_coords"],
        atom_feat=raw_data["atom_feat"],
        origin=raw_data["origin"],
        voxel_size=raw_data["voxel_size"],
        window_size=int(_get_cfg(cfg_dict, "window_size", True)),
        stride=int(_get_cfg(cfg_dict, "stride", True)),
        atom_buffer_radius=float(_get_cfg(cfg_dict, "atom_buffer_radius", True)),
        valid_crop_margin=int(_get_cfg(cfg_dict, "valid_crop_margin", True)),
        num_box_workers=int(_get_cfg(cfg_dict, "num_box_workers", False) or 1),
    )
    pred_data = get_voxel_pred(
        model=model,
        device=device,
        box_dicts=box_dicts,
        full_shape_zyx=tuple(int(v) for v in raw_data["full_shape_zyx"]),
        hardmask=raw_data["hardmask"],
        batch_size=int(_get_cfg(cfg_dict, "batch_size", True)),
        output_heads=tuple(str(v) for v in _get_cfg(cfg_dict, "output_heads", True)),
        merge_mode=str(_get_cfg(cfg_dict, "merge_mode", True)),
        core_offset=int(_get_cfg(cfg_dict, "core_offset", True)),
        gaussian_sigma_ratio=_resolve_gaussian_sigma_ratio(cfg_dict),
        show_progress=bool(_get_cfg(cfg_dict, "show_progress", True)),
    )
    # str, GT 来源类型, 可选 none/labels_npz/structure
    gt_source = _resolve_gt_source(cfg_dict)
    # dict[str, Any], voxel GT 数据; eval_gt=false 时保持空值
    gt_data = {"gt_ligand_mask": None, "gt_instance_label": None}
    if gt_source == "labels_npz":
        gt_data = load_ligand_gt_from_labels_npz(
            labels_npz_path=str(_get_cfg(cfg_dict, "labels_npz_path", True)),
            origin=raw_data["origin"],
            voxel_size=raw_data["voxel_size"],
            grid_shape_zyx=tuple(int(v) for v in raw_data["full_shape_zyx"]),
            class_mapping=_get_cfg(cfg_dict, "class_mapping", False),
            ligand_gt_distance_threshold=float(_get_cfg(cfg_dict, "ligand_gt_distance_threshold", True)),
        )
    elif gt_source == "structure":
        gt_data = load_ligand_gt_from_structure(
            cif_path=str(_get_cfg(cfg_dict, "cif_path", True)),
            cif_gt_path=_get_cfg(cfg_dict, "cif_gt_path", False),
            filter_preset=str(_get_cfg(cfg_dict, "filter_preset", True)),
            class_mapping=_get_cfg(cfg_dict, "class_mapping", False),
            select_first_model=bool(_get_cfg(cfg_dict, "select_first_model", True)),
            error_dir=_get_cfg(cfg_dict, "error_dir", False),
            eval_mode=str(_get_cfg(cfg_dict, "eval_mode", True)),
            origin=raw_data["origin"],
            voxel_size=raw_data["voxel_size"],
            grid_shape_zyx=tuple(int(v) for v in raw_data["full_shape_zyx"]),
            ligand_gt_distance_threshold=float(_get_cfg(cfg_dict, "ligand_gt_distance_threshold", True)),
        )

    meta = {
        "sample_name": _resolve_sample_name(cfg_dict),
        "cache_path": cache_path,
        "cif_path": str(_get_cfg(cfg_dict, "cif_path", True)),
        "map_path": str(_get_cfg(cfg_dict, "map_path", True)),
        "sim_map_path": effective_sim_map_path,
        "cif_gt_path": _get_cfg(cfg_dict, "cif_gt_path", False),
        "labels_npz_path": _get_cfg(cfg_dict, "labels_npz_path", False),
        "gt_source": gt_source,
        "density_channel_names": raw_data["density_channel_names"],
    }
    if bool(_get_cfg(cfg_dict, "save_cache", True)):
        save_voxel_prediction_cache(
            cache_path=cache_path,
            ligand_pred=pred_data["ligand_pred"],
            receptor_pred=pred_data.get("receptor_pred"),
            hardmask=raw_data["hardmask"],
            resampled_emdb=raw_data["resampled_emdb"],
            origin=raw_data["origin"],
            voxel_size=raw_data["voxel_size"],
            meta=meta,
            gt_ligand_mask=gt_data["gt_ligand_mask"],
            gt_instance_label=gt_data["gt_instance_label"],
        )
        return load_voxel_prediction_cache(cache_path)

    return VoxelPredCacheData(
        ligand_pred=pred_data["ligand_pred"],
        receptor_pred=pred_data.get("receptor_pred"),
        hardmask=raw_data["hardmask"],
        resampled_emdb=raw_data["resampled_emdb"],
        origin=raw_data["origin"],
        voxel_size=raw_data["voxel_size"],
        gt_ligand_mask=gt_data["gt_ligand_mask"],
        gt_instance_label=gt_data["gt_instance_label"],
        meta=meta,
    )

def run_voxel_single(
    cfg_dict: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """
    执行单样本 voxel-only ligand 推理。

    输入参数:
        - cfg_dict: dict[str, Any], 当前样本推理配置
        - model: torch.nn.Module, stage1 模型
        - device: torch.device, 推理设备

    输出:
        - result: dict[str, Any], 单样本推理摘要, 包含:
            - "sample_name": str, 当前样本名
            - "output_dir": str, 当前样本输出目录
            - "cache_path": str, 当前样本使用或生成的缓存路径
            - "num_candidates": int, 后处理后保留的 ligand 候选数
            - "metrics": dict[str, Any] | None, 评估指标; 未提供 GT 时为 None
            - "error": None, 单样本成功路径下固定为 None
    """
    if str(cfg_dict.get("mode")) == "voxel_single":
        for key in ("window_size", "stride", "batch_size", "merge_mode", "core_offset"):
            _require_direct_cfg(cfg_dict, key)

    sample_name = _resolve_sample_name(cfg_dict)
    output_dir = os.path.join(str(_get_cfg(cfg_dict, "output_root", True)), sample_name)
    os.makedirs(output_dir, exist_ok=True)

    cache_path = _resolve_cache_path(cfg_dict, sample_name)
    cache_data = _build_cache_or_forward(cfg_dict, model, device, cache_path)
    _save_probability_outputs(output_dir, cache_data)

    post_params = _postprocess_params_from_cfg(cfg_dict)
    post_result = postprocess_ligand_probability_map(
        ligand_pred=cache_data.ligand_pred,
        origin=cache_data.origin,
        voxel_size=cache_data.voxel_size,
        threshold=post_params["threshold"],
        min_component_voxels=post_params["min_component_voxels"],
        filter_strength=post_params["filter_strength"],
        connectivity_policy=post_params["connectivity_policy"],
        sigma_nearby=post_params["sigma_nearby"],
        kernel_nearby=post_params["kernel_nearby"],
        receptor_pred=cache_data.receptor_pred,
        sigma_response=post_params["sigma_response"],
        kernel_response=post_params["kernel_response"],
        score_add=post_params["score_add"],
        score_minus=post_params["score_minus"],
        voxel_score_min=post_params["voxel_score_min"],
        instance_score_min=post_params["instance_score_min"],
    )
    _save_postprocess_outputs(output_dir, post_result)

    metrics: dict[str, Any] | None = None
    if cache_data.gt_ligand_mask is not None and cache_data.gt_instance_label is not None:
        metrics = {}
        metrics.update(evaluate_voxel_mask(post_result.binary_mask_filtered, cache_data.gt_ligand_mask))
        metrics.update(
            evaluate_instance_mask(
                pred_instance_label=post_result.instance_label_filtered,
                gt_instance_label=cache_data.gt_instance_label,
                alpha=float(_get_cfg(cfg_dict, "alpha", True)),
                beta=float(_get_cfg(cfg_dict, "beta", True)),
            )
        )
        _write_json(os.path.join(output_dir, "metrics.json"), metrics)

    if bool(_get_cfg(cfg_dict, "vis_enable", True)) and _get_cfg(cfg_dict, "vis_output_root", False) is not None:
        build_infer_vis_bundle(
            output_root=str(_get_cfg(cfg_dict, "vis_output_root", True)),
            cif_path=str(_get_cfg(cfg_dict, "cif_path", True)),
            map_path=str(_get_cfg(cfg_dict, "map_path", True)),
            cif_gt_path=_get_cfg(cfg_dict, "cif_gt_path", False),
            pred_atom_coords=np.empty((0, 3), dtype=np.float32),
            prob_threshold=post_params["threshold"],
            filter_preset=str(_get_cfg(cfg_dict, "filter_preset", False)) if _get_cfg(cfg_dict, "filter_preset", False) is not None else "five_class",
            class_mapping=_get_cfg(cfg_dict, "class_mapping", False),
            pdb_id=sample_name,
            select_first_model=bool(_get_cfg(cfg_dict, "select_first_model", True)),
            pred_voxel_mask=post_result.binary_mask_filtered,
            resampled_emdb=cache_data.resampled_emdb,
            origin=cache_data.origin,
            voxel_size=cache_data.voxel_size,
            pred_voxel_prob=cache_data.ligand_pred,
            pred_instance_label=post_result.instance_label_filtered,
            write_pred_atom_coords=False,
        )

    result = {
        "sample_name": sample_name,
        "output_dir": output_dir,
        "cache_path": cache_path,
        "num_candidates": int(len(post_result.candidates)),
        "metrics": metrics,
        "error": None,
    }
    _write_json(os.path.join(output_dir, "summary.json"), result)
    return result

def run_voxel_batch(
    cfg_dict: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
) -> list[dict[str, Any]]:
    """
    执行批量 voxel-only ligand 推理。

    输入参数:
        - cfg_dict: dict[str, Any], batch 推理配置
        - model: torch.nn.Module, stage1 模型
        - device: torch.device, 推理设备

    输出:
        - results: list[dict[str, Any]], 每项对应一个样本的执行结果, 可能为两种结构之一:
            - 成功项:
                - "sample_name": str, 样本名
                - "output_dir": str, 样本输出目录
                - "cache_path": str, 样本缓存路径
                - "num_candidates": int, 后处理后保留的 ligand 候选数
                - "metrics": dict[str, Any] | None, 样本评估指标; 未提供 GT 时为 None
                - "error": None, 成功时固定为 None
            - 失败项:
                - "sample_name": str, 样本名; 若 pair 未显式提供则可能为空字符串
                - "error": str, 捕获到的异常信息
    """
    pairs = load_raw_pairs(str(_get_cfg(cfg_dict, "raw_pairs_json", True)))
    results: list[dict[str, Any]] = []
    for pair in pairs:
        sample_cfg = _merge_sample_pair_cfg(cfg_dict, pair)
        sample_cfg["mode"] = "voxel_batch_item"
        try:
            results.append(run_voxel_single(sample_cfg, model, device))
        except Exception as exc:
            error_row = {"sample_name": pair.get("sample_name", ""), "error": str(exc)}
            results.append(error_row)
            if not bool(_get_cfg(cfg_dict, "continue_on_error", True)):
                raise
    write_voxel_batch_excel(results, str(_get_cfg(cfg_dict, "output_root", True)))
    _write_json(os.path.join(str(_get_cfg(cfg_dict, "output_root", True)), "voxel_batch_results.json"), results)
    return results





# -------------------------------------- param_search 模式 --------------------------------------
def _iter_with_progress(iterable: Any, total: int, desc: str, enabled: bool) -> Any:
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise ImportError("show_progress=true 需要安装 tqdm") from exc
    return tqdm(iterable, total=total, desc=desc)


def _collect_or_build_cache_paths(
    cfg_dict: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
) -> list[str]:
    """
    生成(若不存在)或读取(若存在)所有样本的缓存(param_search 阶段使用)。

    输入参数:
        - cfg_dict: dict[str, Any], param_search 配置
        - model: torch.nn.Module, stage1 模型
        - device: torch.device, 推理设备

    输出:
        - cache_paths: list[str], 可变长度, 缓存路径列表
    """
    pairs = load_raw_pairs(str(_get_cfg(cfg_dict, "raw_pairs_json", True)))
    show_progress = bool(_get_cfg(cfg_dict, "show_progress", True))
    cache_paths: list[str] = []
    for pair in _iter_with_progress(pairs, total=len(pairs), desc="voxel cache samples", enabled=show_progress):
        sample_cfg = _merge_sample_pair_cfg(cfg_dict, pair)
        sample_cfg["mode"] = "voxel_param_cache_item"
        sample_cfg["eval_gt"] = True
        sample_cfg["vis_enable"] = False
        sample_cfg["vis_output_root"] = None
        result = run_voxel_single(sample_cfg, model, device)
        cache_paths.append(str(result["cache_path"]))
    return cache_paths

def _save_best_outputs_from_cache(
    cache_paths: list[str],
    best_params: dict[str, Any],
    eval_params: dict[str, Any],
    cfg_dict: dict[str, Any],
    output_root: str,
) -> list[dict[str, Any]]:
    """
    用最优后处理参数对每个缓存样本重新输出 mask、instance、candidate 和 metrics, 并进行评估和可能的可视化。

    输入参数:
        - cache_paths: list[str], 可变长度, 缓存路径列表
        - best_params: dict[str, Any], 最优后处理参数
        - eval_params: dict[str, Any], 评估参数, 仅用了 eval_params["alpha"]、eval_params["beta"]
        - cfg_dict: dict[str, Any], param_search 配置; vis_enable=true 且 vis_output_root 非空时输出最优参数可视化
        - output_root: str, 输出根目录

    输出:
        - results: list[dict[str, Any]], 每个缓存样本在最优参数下的结果摘要, 每项包含:
            - "sample_name": str, 样本名
            - "output_dir": str, 当前样本 best_outputs 目录
            - "cache_path": str, 对应的缓存路径
            - "num_candidates": int, 最优后处理后保留的 ligand 候选数
            - "metrics": dict[str, Any] | None, 当前样本指标; 缓存不含 GT 时为 None
            - "error": None, 当前函数成功路径下固定为 None
    """
    results: list[dict[str, Any]] = []
    best_root = os.path.join(output_root, "best_outputs")
    os.makedirs(best_root, exist_ok=True)
    # bool, 是否在最优参数输出阶段生成可视化
    vis_enabled = bool(_get_cfg(cfg_dict, "vis_enable", True)) and _get_cfg(cfg_dict, "vis_output_root", False) is not None
    for cache_path in cache_paths:
        cache_data = load_voxel_prediction_cache(cache_path)
        sample_name = str(cache_data.meta.get("sample_name", Path(cache_path).stem))
        sample_dir = os.path.join(best_root, sample_name)
        os.makedirs(sample_dir, exist_ok=True)
        post_result = postprocess_ligand_probability_map(
            ligand_pred=cache_data.ligand_pred,
            origin=cache_data.origin,
            voxel_size=cache_data.voxel_size,
            threshold=float(best_params["threshold"]),
            min_component_voxels=int(best_params["min_component_voxels"]),
            filter_strength=str(best_params["filter_strength"]),
            connectivity_policy=str(best_params["connectivity_policy"]),
            sigma_nearby=float(best_params["sigma_nearby"]),
            kernel_nearby=int(best_params["kernel_nearby"]),
            receptor_pred=cache_data.receptor_pred,
            sigma_response=float(best_params["sigma_response"]),
            kernel_response=int(best_params["kernel_response"]),
            score_add=float(best_params["score_add"]),
            score_minus=float(best_params["score_minus"]),
            voxel_score_min=float(best_params["voxel_score_min"]),
            instance_score_min=float(best_params["instance_score_min"]),
        )
        _save_probability_outputs(sample_dir, cache_data)
        _save_postprocess_outputs(sample_dir, post_result)
        metrics: dict[str, Any] | None = None
        if cache_data.gt_ligand_mask is not None and cache_data.gt_instance_label is not None:
            metrics = {}
            metrics.update(evaluate_voxel_mask(post_result.binary_mask_filtered, cache_data.gt_ligand_mask))
            metrics.update(
                evaluate_instance_mask(
                    pred_instance_label=post_result.instance_label_filtered,
                    gt_instance_label=cache_data.gt_instance_label,
                    alpha=float(eval_params["alpha"]),
                    beta=float(eval_params["beta"]),
                )
            )
            _write_json(os.path.join(sample_dir, "metrics.json"), metrics)
        if vis_enabled:
            build_infer_vis_bundle(
                output_root=str(_get_cfg(cfg_dict, "vis_output_root", True)),
                cif_path=str(cache_data.meta["cif_path"]),
                map_path=str(cache_data.meta["map_path"]),
                cif_gt_path=cache_data.meta.get("cif_gt_path"),
                pred_atom_coords=np.empty((0, 3), dtype=np.float32),
                prob_threshold=float(best_params["threshold"]),
                filter_preset=str(_get_cfg(cfg_dict, "filter_preset", True)),
                class_mapping=_get_cfg(cfg_dict, "class_mapping", False),
                pdb_id=sample_name,
                select_first_model=bool(_get_cfg(cfg_dict, "select_first_model", True)),
                pred_voxel_mask=post_result.binary_mask_filtered,
                resampled_emdb=cache_data.resampled_emdb,
                origin=cache_data.origin,
                voxel_size=cache_data.voxel_size,
                pred_voxel_prob=cache_data.ligand_pred,
                pred_instance_label=post_result.instance_label_filtered,
                write_pred_atom_coords=False,
            )
        results.append(
            {
                "sample_name": sample_name,
                "output_dir": sample_dir,
                "cache_path": cache_path,
                "num_candidates": int(len(post_result.candidates)),
                "metrics": metrics,
                "error": None,
            }
        )
    return results


# 主函数
def run_voxel_param_search(
    cfg_dict: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """
    执行 voxel-only 后处理参数搜索。

    输入参数:
        - cfg_dict: dict[str, Any], 参数搜索配置
        - model: torch.nn.Module, stage1 模型
        - device: torch.device, 推理设备

    输出:
        - result: dict[str, Any], 参数搜索总结果, 包含:
            - "cache_paths": list[str], 可变长度, 参与搜索的缓存路径列表
            - "search_result": dict[str, Any], optimize_postprocess_params() 的原始搜索结果, 包含 best_params/best_metrics/history 等字段
            - "best_summary": dict[str, Any], 在最佳参数下对整套缓存样本重新评估得到的汇总指标
            - "best_outputs": list[dict[str, Any]], 每个样本在最佳参数下重新导出的结果摘要
    """
    cache_paths = _collect_or_build_cache_paths(cfg_dict, model, device)
    fixed_postprocess_params = _postprocess_params_from_cfg(cfg_dict)
    search_space = _get_cfg(cfg_dict, "search_space", True)
    eval_params = {
        "alpha": float(_get_cfg(cfg_dict, "alpha", True)),
        "beta": float(_get_cfg(cfg_dict, "beta", True)),
    }
    optimizer_params = {
        "objective_expr": str(_get_cfg(cfg_dict, "objective_expr", True)),
        "fixed_search_params": list(_get_cfg(cfg_dict, "fixed_search_params", True)),
        "max_iter": int(_get_cfg(cfg_dict, "max_iter", True)),
        "popsize": int(_get_cfg(cfg_dict, "popsize", True)),
        "random_seed": int(_get_cfg(cfg_dict, "random_seed", True)),
    }
    cache_data_mode = str(_get_cfg(cfg_dict, "cache_data_mode", True))
    loaded_cache_items = []
    if cache_data_mode == "memory":
        loaded_cache_items = [(cache_path, load_voxel_prediction_cache(cache_path)) for cache_path in cache_paths]
    elif cache_data_mode != "disk":
        raise ValueError(f"未知 cache_data_mode: {cache_data_mode}")

    search_result = optimize_postprocess_params(
        cache_paths=cache_paths,
        loaded_cache_items=loaded_cache_items,
        cache_data_mode=cache_data_mode,
        fixed_postprocess_params=fixed_postprocess_params,
        search_space=search_space,
        search_strategy=str(_get_cfg(cfg_dict, "search_strategy", True)),
        eval_params=eval_params,
        optimizer_params=optimizer_params,
        n_jobs=int(_get_cfg(cfg_dict, "n_jobs", True)),
        show_progress=bool(_get_cfg(cfg_dict, "show_progress", True)),
    )

    output_root = str(_get_cfg(cfg_dict, "output_root", True))
    os.makedirs(output_root, exist_ok=True)
    _write_json(os.path.join(output_root, "best_params.json"), search_result["best_params"])
    _write_param_search_excel(search_result["history"], output_root)
    best_summary = evaluate_postprocess_params_on_cache_set(
        cache_paths=cache_paths,
        postprocess_params=search_result["best_params"],
        eval_params=eval_params,
        n_jobs=int(_get_cfg(cfg_dict, "n_jobs", True)),
    )
    best_outputs = _save_best_outputs_from_cache(
        cache_paths=cache_paths,
        best_params=search_result["best_params"],
        eval_params=eval_params,
        cfg_dict=cfg_dict,
        output_root=output_root,
    )
    _write_json(os.path.join(output_root, "best_summary.json"), best_summary)
    _write_json(os.path.join(output_root, "per_sample_best_metrics.json"), best_outputs)
    return {
        "cache_paths": cache_paths,
        "search_result": search_result,
        "best_summary": best_summary,
        "best_outputs": best_outputs,
    }









# ----------------------------------------------- 纯粹工具函数 ------------------------------------------------
def _json_default(value: Any) -> Any:
    """
    将 numpy 和 dataclass 对象转换为 JSON 可序列化对象。

    输入参数:
        - value: Any, 待转换对象

    输出:
        - converted: Any, JSON 可序列化对象
    """
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"对象不可 JSON 序列化: {type(value)}")

def _write_json(path: str, data: Any) -> str:
    """
    写出 UTF-8 JSON 文件。

    输入参数:
        - path: str, JSON 输出路径
        - data: Any, 可 JSON 序列化对象

    输出:
        - path: str, 写出的路径
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
    return path

def _get_cfg(
    cfg_dict: dict[str, Any],
    key: str,
    required: bool,
) -> Any:
    """
    从推理配置或训练 dataset 配置中读取参数。

    输入参数:
        - cfg_dict: dict[str, Any], Hydra 配置转成的普通 dict
        - key: str, 参数名
        - required: bool, 是否必须存在

    输出:
        - value: Any, 参数值
    """
    if key in cfg_dict and cfg_dict[key] is not None and cfg_dict[key] != "???":
        return cfg_dict[key]
    train_dataset_cfg = cfg_dict.get("_train_dataset_cfg", {})
    if key in train_dataset_cfg and train_dataset_cfg[key] is not None:
        return train_dataset_cfg[key]
    if required:
        raise KeyError(f"缺少必需参数: {key}")
    return None

def _require_direct_cfg(
    cfg_dict: dict[str, Any],
    key: str,
) -> Any:
    """
    从推理配置本身读取必须显式提供的参数, 不回退训练配置。

    输入参数:
        - cfg_dict: dict[str, Any], Hydra 配置转成的普通 dict
        - key: str, 参数名

    输出:
        - value: Any, 参数值
    """
    if key not in cfg_dict or cfg_dict[key] is None or cfg_dict[key] == "???":
        raise KeyError(f"single 模式必须显式传入参数: {key}")
    return cfg_dict[key]

def _merge_sample_pair_cfg(
    cfg_dict: dict[str, Any],
    pair: dict[str, str | None],
) -> dict[str, Any]:
    """
    将存放单个样本的.json路径 pair, 合并到单样本推理配置 cfg_dict 中(从而多样本的推理只需调用单样本推理的函数)。

    输入参数:
        - cfg_dict: dict[str, Any], batch 或 param_search 配置
        - pair: dict[str, str | None], raw_pairs JSON 解析后的单样本字段

    输出:
        - sample_cfg: dict[str, Any], 当前样本配置; pair 中为 None 的可选字段不覆盖全局配置
    """
    # dict[str, Any], 当前样本配置副本
    sample_cfg = dict(cfg_dict)
    for key, value in pair.items():
        if value is not None:
            sample_cfg[key] = value
    return sample_cfg

def _resolve_sample_name(cfg_dict: dict[str, Any]) -> str:
    """
    解析当前样本输出名。

    输入参数:
        - cfg_dict: dict[str, Any], 当前样本配置

    输出:
        - sample_name: str, 当前样本名
    """
    if cfg_dict.get("sample_name") is not None:
        return str(cfg_dict["sample_name"])
    return Path(str(cfg_dict["cif_path"])).stem

def _resolve_cache_path(cfg_dict: dict[str, Any], sample_name: str) -> str:
    """
    解析当前样本缓存路径: return os.path.join(str(_get_cfg(cfg_dict, "cache_root", True)), f"{sample_name}.npz")

    输入参数:
        - cfg_dict: dict[str, Any], 当前样本配置
        - sample_name: str, 当前样本名; 用于默认缓存文件名

    输出:
        - cache_path: str, .npz 缓存路径, 固定为 cache_root/sample_name.npz
    """
    return os.path.join(str(_get_cfg(cfg_dict, "cache_root", True)), f"{sample_name}.npz")




# ----------------------------------------------- 半纯粹工具函数 ------------------------------------------------
def _resolve_density_channel_names_from_cfg(cfg_dict: dict[str, Any]) -> list[str]:
    """
    从配置中展开当前实际启用的密度通道名。

    输入参数:
        - cfg_dict: dict[str, Any], 当前推理配置

    输出:
        - channel_names: list[str], 可变长度, 实际启用的密度通道名
    """
    # dict[str, Any], density channel 配置
    density_channel_config = _get_cfg(cfg_dict, "density_channel_config", True)
    # list[str], 用户配置中的启用通道名
    enabled_channels = [str(v) for v in density_channel_config["enabled_channels"]]
    if "all" in enabled_channels:
        return list(ALL_CHANNEL_NAMES)
    return enabled_channels

def _resolve_effective_sim_map_path(cfg_dict: dict[str, Any]) -> str | None:
    """
    解析当前 forward 可能需要的模拟密度图路径(不需要则返回None)。

    输入参数:
        - cfg_dict: dict[str, Any], 当前推理配置

    输出:
        - sim_map_path: str | None, 仅启用 sim/diff/posdiff 通道时返回路径
    """
    # list[str], 实际启用的密度通道名
    density_channel_names = _resolve_density_channel_names_from_cfg(cfg_dict)
    # bool, 当前模型输入是否需要模拟密度图
    needs_sim = any(name.split("_")[0] in {"sim", "diff", "posdiff"} for name in density_channel_names)
    if not needs_sim:
        return None
    # str | None, 模拟密度图路径; 需要 sim 通道时由 load_from_raw_cif 负责 fail-fast
    return _get_cfg(cfg_dict, "sim_map_path", False)

def _resolve_gt_source(cfg_dict: dict[str, Any]) -> str:
    """
    解析当前样本的 GT 来源类型: "none" / "labels_npz" / "structure"。

    输入参数:
        - cfg_dict: dict[str, Any], 当前推理配置

    输出:
        - gt_source: str, GT 来源类型, 可选 none/labels_npz/structure
    """
    if not bool(_get_cfg(cfg_dict, "eval_gt", True)):
        return "none"
    if _get_cfg(cfg_dict, "labels_npz_path", False) is not None:
        return "labels_npz"
    return "structure"

def _resolve_gaussian_sigma_ratio(cfg_dict: dict[str, Any]) -> float | None:
    """
    从配置中读取 voxel BOX Gaussian 衰减参数。

    输入参数:
        - cfg_dict: dict[str, Any], 当前推理配置

    输出:
        - gaussian_sigma_ratio: float | None, sigma 与最大边长的比例; None 表示不启用 Gaussian 衰减
    """
    if "gaussian_sigma_ratio" not in cfg_dict or cfg_dict["gaussian_sigma_ratio"] in (None, "???"):
        return None
    return float(cfg_dict["gaussian_sigma_ratio"])

def _postprocess_params_from_cfg(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    """
    从配置中提取后处理参数。

    输入参数:
        - cfg_dict: dict[str, Any], 当前推理配置

    输出:
        - params: dict[str, Any], 后处理参数, 包含:
            - "threshold": float, ligand 概率二值化阈值
            - "min_component_voxels": int, 连通域最小体素数
            - "filter_strength": str, 候选过滤强度档位
            - "connectivity_policy": str, 连通域邻接策略
            - "sigma_nearby": float, 邻域平滑 sigma
            - "kernel_nearby": int, 邻域平滑核大小
            - "sigma_response": float, receptor response 平滑 sigma
            - "kernel_response": int, receptor response 平滑核大小
            - "score_add": float, 正向响应加分项
            - "score_minus": float, 负向响应减分项
            - "voxel_score_min": float, 候选体素级分数阈值
            - "instance_score_min": float, 候选实例级分数阈值
    """
    return {
        "threshold": float(_get_cfg(cfg_dict, "threshold", True)),
        "min_component_voxels": int(_get_cfg(cfg_dict, "min_component_voxels", True)),
        "filter_strength": str(_get_cfg(cfg_dict, "filter_strength", True)),
        "connectivity_policy": str(_get_cfg(cfg_dict, "connectivity_policy", True)),
        "sigma_nearby": float(_get_cfg(cfg_dict, "sigma_nearby", True)),
        "kernel_nearby": int(_get_cfg(cfg_dict, "kernel_nearby", True)),
        "sigma_response": float(_get_cfg(cfg_dict, "sigma_response", True)),
        "kernel_response": int(_get_cfg(cfg_dict, "kernel_response", True)),
        "score_add": float(_get_cfg(cfg_dict, "score_add", True)),
        "score_minus": float(_get_cfg(cfg_dict, "score_minus", True)),
        "voxel_score_min": float(_get_cfg(cfg_dict, "voxel_score_min", True)),
        "instance_score_min": float(_get_cfg(cfg_dict, "instance_score_min", True)),
    }




# ----------------------------------------------- 用于保存/写入的的工具函数 ------------------------------------------------
def _save_postprocess_outputs(
    output_dir: str,
    post_result: VoxelPostprocessResult,
) -> None:
    """
    保存后处理输出数组和候选 JSON。

    输入参数:
        - output_dir: str, 当前样本输出目录
        - post_result: VoxelPostprocessResult, 后处理结果对象

    输出:
        - None
    """
    np.savez(
        os.path.join(output_dir, "ligand_mask_filtered.npz"),
        binary_mask_filtered=post_result.binary_mask_filtered.astype(np.bool_),
        score_map=post_result.score_map.astype(np.float32),
    )
    np.savez(
        os.path.join(output_dir, "instance_label_filtered.npz"),
        instance_label_filtered=post_result.instance_label_filtered.astype(np.int32),
    )
    _write_json(
        os.path.join(output_dir, "voxel_candidates.json"),
        [asdict(candidate) for candidate in post_result.candidates],
    )

def _save_probability_outputs(
    output_dir: str,
    cache_data: VoxelPredCacheData,
) -> None:
    """
    依据 cache_data 保存：os.path.join(output_dir, "ligand_pred.npz")、os.path.join(output_dir, "receptor_pred.npz")。

    输入参数:
        - output_dir: str, 当前样本输出目录
        - cache_data: VoxelPredCacheData, 概率缓存数据

    输出:
        - None
    """
    np.savez(
        os.path.join(output_dir, "ligand_pred.npz"),
        ligand_pred=cache_data.ligand_pred.astype(np.float32),
    )
    if cache_data.receptor_pred is not None:
        np.savez(
            os.path.join(output_dir, "receptor_pred.npz"),
            receptor_pred=cache_data.receptor_pred.astype(np.float32),
        )

def _write_param_search_excel(
    history: list[dict[str, Any]],
    output_root: str,
) -> str:
    """
    将参数搜索历史 history 写入 Excel。

    输入参数:
        - history: list[dict[str, Any]], 可变长度, 每组参数的评估结果, 每个条目表示一组参数
        - output_root: str, 输出目录

    输出:
        - excel_path: str, 写出的 Excel 路径
    """
    import openpyxl
    from openpyxl.styles import Font

    os.makedirs(output_root, exist_ok=True)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "ParamSearch"
    headers = [
        "rank",
        "objective_score",
        "avg_num_candidates",
        "avg_voxel_precision",
        "avg_voxel_recall",
        "avg_voxel_f1",
        "avg_voxel_iou",
        "avg_voxel_dice",
        "avg_instance_precision",
        "avg_instance_recall",
        "avg_instance_f1",
        "postprocess_params_json",
    ]
    ws.append(headers)
    for cell in ws[1]:
        cell.font = Font(bold=True)

    sorted_history = sorted(history, key=lambda item: float(item.get("objective_score", 0.0)), reverse=True)
    for rank, item in enumerate(sorted_history, start=1):
        ws.append(
            [
                rank,
                item.get("objective_score", ""),
                item.get("avg_num_candidates", ""),
                item.get("avg_voxel_precision", ""),
                item.get("avg_voxel_recall", ""),
                item.get("avg_voxel_f1", ""),
                item.get("avg_voxel_iou", ""),
                item.get("avg_voxel_dice", ""),
                item.get("avg_instance_precision", ""),
                item.get("avg_instance_recall", ""),
                item.get("avg_instance_f1", ""),
                json.dumps(item.get("postprocess_params", {}), ensure_ascii=False, sort_keys=True, default=_json_default),
            ]
        )
    excel_path = os.path.join(output_root, "param_search_results.xlsx")
    wb.save(excel_path)
    return excel_path

