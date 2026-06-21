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
from src.inference.voxel_evaluator import (
    DEFAULT_COVERAGE_THRESHOLDS,
    DEFAULT_TOPK_VALUES,
    evaluate_global_instance_matching,
    evaluate_topk_success,
    evaluate_voxel_mask,
    evaluate_voxel_pr_auc,
)
from src.inference.utils.yield_json_from_raw_sample import load_raw_pairs
from src.inference.voxel_gt import load_ligand_gt_from_labels_npz, load_ligand_gt_from_structure
from src.inference.voxel_postprocess import postprocess_ligand_probability_map
from src.inference.voxel_tuning import (
    evaluate_postprocess_params_on_cache_set,
    load_voxel_prediction_cache,
    optimize_postprocess_params,
    optimize_postprocess_params_by_class,
    save_voxel_prediction_cache,
    write_best_by_class_csv,
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

    # str, 当前 forward 实际使用的结构路径
    effective_cif_path = _resolve_forward_structure_path(cfg_dict)
    # str | None, 当前 forward 实际使用的模拟密度图路径
    effective_sim_map_path = _resolve_effective_sim_map_path(cfg_dict)
    raw_data = load_from_raw_cif(
        cif_path=effective_cif_path,
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
    gt_data = {
        "gt_ligand_mask": None,
        "gt_instance_label": None,
        "gt_ligand_mask_by_class_id": {},
        "gt_instance_label_by_class_id": {},
        "gt_instance_meta": None,
    }
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
            ligand_structure_path=str(_get_cfg(cfg_dict, "cif_gt_path", True)),
            receptor_structure_path=_resolve_gt_receptor_path(cfg_dict),
            filter_preset=str(_get_cfg(cfg_dict, "filter_preset", True)),
            class_mapping=_get_cfg(cfg_dict, "class_mapping", False),
            select_first_model=bool(_get_cfg(cfg_dict, "select_first_model", True)),
            error_dir=_get_cfg(cfg_dict, "error_dir", False),
            origin=raw_data["origin"],
            voxel_size=raw_data["voxel_size"],
            grid_shape_zyx=tuple(int(v) for v in raw_data["full_shape_zyx"]),
            ligand_gt_distance_threshold=float(_get_cfg(cfg_dict, "ligand_gt_distance_threshold", True)),
        )

    # list[str], (C,), 当前任务类别名; 二分类默认 background/foreground
    class_names = [str(v) for v in (_get_cfg(cfg_dict, "class_names", False) or ["background", "foreground"])]
    # tuple[dict[str, np.ndarray] | None, dict[str, np.ndarray] | None], 类别名粒度的逐类 GT
    gt_ligand_mask_by_class, gt_instance_label_by_class = _gt_id_to_name(gt_data, class_names)
    meta = {
        "sample_name": _resolve_sample_name(cfg_dict),
        "cache_path": cache_path,
        "cif_path": effective_cif_path,
        "map_path": str(_get_cfg(cfg_dict, "map_path", True)),
        "sim_map_path": effective_sim_map_path,
        "cif_gt_path": _get_cfg(cfg_dict, "cif_gt_path", False),
        "labels_npz_path": _get_cfg(cfg_dict, "labels_npz_path", False),
        "gt_source": gt_source,
        "density_channel_names": raw_data["density_channel_names"],
        "class_names": class_names,
        "class_mapping": _get_cfg(cfg_dict, "class_mapping", False),
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
            gt_ligand_mask_by_class=gt_ligand_mask_by_class,
            gt_instance_label_by_class=gt_instance_label_by_class,
            gt_instance_meta=gt_data.get("gt_instance_meta"),
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
        gt_ligand_mask_by_class=gt_ligand_mask_by_class,
        gt_instance_label_by_class=gt_instance_label_by_class,
        gt_instance_meta=gt_data.get("gt_instance_meta"),
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
    post_results_by_class = _postprocess_cache(cache_data, post_params)
    first_class_name, post_result = next(iter(post_results_by_class.items()))
    if first_class_name == "foreground":
        _save_postprocess_outputs(output_dir, post_result)
    else:
        for class_name, class_post_result in post_results_by_class.items():
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            _save_postprocess_outputs(class_dir, class_post_result)

    metrics = _evaluate_post_results(
        post_results_by_class=post_results_by_class,
        cache_data=cache_data,
        eval_params=_eval_params_from_cfg(cfg_dict),
    )
    if metrics is not None:
        _write_json(os.path.join(output_dir, "metrics.json"), metrics)

    if first_class_name == "foreground" and bool(_get_cfg(cfg_dict, "vis_enable", True)) and _get_cfg(cfg_dict, "vis_output_root", False) is not None:
        build_infer_vis_bundle(
            output_root=str(_get_cfg(cfg_dict, "vis_output_root", True)),
            cif_path=str(cache_data.meta["cif_path"]),
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
            extra_map_paths=_resolve_extra_map_paths(cache_data),
        )

    # dict[str, int], 前景类别名到后处理候选数的映射
    num_candidates_by_class = {class_name: int(len(class_result.candidates)) for class_name, class_result in post_results_by_class.items()}
    result = {
        "sample_name": sample_name,
        "output_dir": output_dir,
        "cache_path": cache_path,
        "num_candidates": int(sum(num_candidates_by_class.values())),
        "num_candidates_by_class": num_candidates_by_class,
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
        - eval_params: dict[str, Any], 评估参数, 含 compute_instance_metrics / coverage_thresholds / topk_values
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
        post_results_by_class = _postprocess_cache(cache_data, dict(best_params))
        first_class_name, post_result = next(iter(post_results_by_class.items()))
        _save_probability_outputs(sample_dir, cache_data)
        if first_class_name == "foreground":
            _save_postprocess_outputs(sample_dir, post_result)
        else:
            for class_name, class_post_result in post_results_by_class.items():
                class_dir = os.path.join(sample_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                _save_postprocess_outputs(class_dir, class_post_result)
        metrics = _evaluate_post_results(
            post_results_by_class=post_results_by_class,
            cache_data=cache_data,
            eval_params=eval_params,
        )
        if metrics is not None:
            _write_json(os.path.join(sample_dir, "metrics.json"), metrics)
        if first_class_name == "foreground" and vis_enabled:
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
                extra_map_paths=_resolve_extra_map_paths(cache_data),
            )
        # dict[str, int], 前景类别名到最优参数下候选数的映射
        num_candidates_by_class = {class_name: int(len(class_result.candidates)) for class_name, class_result in post_results_by_class.items()}
        results.append(
            {
                "sample_name": sample_name,
                "output_dir": sample_dir,
                "cache_path": cache_path,
                "num_candidates": int(sum(num_candidates_by_class.values())),
                "num_candidates_by_class": num_candidates_by_class,
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
    eval_params = _eval_params_from_cfg(cfg_dict)
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

    # int, 当前缓存 ligand_pred 的维度; 3 表示二分类, 4 表示多分类 softmax
    pred_ndim = _resolve_cache_prediction_ndim(
        cache_paths=cache_paths,
        loaded_cache_items=loaded_cache_items,
        cache_data_mode=cache_data_mode,
    )
    output_root = str(_get_cfg(cfg_dict, "output_root", True))
    os.makedirs(output_root, exist_ok=True)
    if pred_ndim == 3:
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
        _write_json(os.path.join(output_root, "best_params.json"), search_result["best_params"])
        _write_param_search_excel(search_result["history"], output_root)
        best_summary = evaluate_postprocess_params_on_cache_set(
            cache_paths=cache_paths,
            postprocess_params=search_result["best_params"],
            eval_params=eval_params,
            n_jobs=int(_get_cfg(cfg_dict, "n_jobs", True)),
        )
    elif pred_ndim == 4:
        search_result = optimize_postprocess_params_by_class(
            cache_paths=cache_paths,
            loaded_cache_items=loaded_cache_items,
            cache_data_mode=cache_data_mode,
            fixed_postprocess_params=fixed_postprocess_params,
            search_space=search_space,
            search_space_by_class=dict(_get_cfg(cfg_dict, "search_space_by_class", False) or {}),
            search_strategy=str(_get_cfg(cfg_dict, "search_strategy", True)),
            eval_params=eval_params,
            optimizer_params=optimizer_params,
            n_jobs=int(_get_cfg(cfg_dict, "n_jobs", True)),
            show_progress=bool(_get_cfg(cfg_dict, "show_progress", True)),
        )
        _write_json(os.path.join(output_root, "best_params.json"), search_result["best_params"])
        best_summary = search_result["best_metrics"]
        write_best_by_class_csv(
            best_metrics_by_class=search_result["best_metrics"]["by_class"],
            best_params_by_class=search_result["best_params"]["by_class"],
            output_root=output_root,
        )
    else:
        raise ValueError(f"ligand_pred 维度必须为 3 或 4, 实际为 {pred_ndim}")
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
def _resolve_cache_prediction_ndim(
    cache_paths: list[str],
    loaded_cache_items: list[tuple[str, VoxelPredCacheData]],
    cache_data_mode: str,
) -> int:
    """
    检查整批缓存的 ligand_pred 维度是否一致。

    输入参数:
        - cache_paths: list[str], 可变长度, 缓存路径列表; disk 模式使用
        - loaded_cache_items: list[tuple[str, VoxelPredCacheData]] 或 [], 可变长度, 已加载缓存; memory 模式使用
        - cache_data_mode: str, 缓存读取模式, 可选 disk/memory

    输出:
        - pred_ndim: int, ligand_pred 维度, 只允许 3 或 4
    """
    if cache_data_mode == "memory":
        if len(loaded_cache_items) == 0:
            raise ValueError("memory 模式下 loaded_cache_items 不能为空")
        # list[int], 每个已加载缓存的 ligand_pred.ndim
        ndims = [int(np.asarray(data.ligand_pred).ndim) for _, data in loaded_cache_items]
    elif cache_data_mode == "disk":
        if len(cache_paths) == 0:
            raise ValueError("disk 模式下 cache_paths 不能为空")
        # list[int], 每个磁盘缓存的 ligand_pred.ndim
        ndims = [int(np.asarray(load_voxel_prediction_cache(cache_path).ligand_pred).ndim) for cache_path in cache_paths]
    else:
        raise ValueError(f"未知 cache_data_mode: {cache_data_mode}")
    # set[int], 整批缓存出现过的 ligand_pred 维度集合
    ndim_set = set(ndims)
    if len(ndim_set) != 1:
        raise ValueError(f"整批缓存 ligand_pred 维度不一致: {sorted(ndim_set)}")
    pred_ndim = int(ndims[0])
    if pred_ndim not in {3, 4}:
        raise ValueError(f"ligand_pred 维度必须为 3 或 4, 实际为 {pred_ndim}")
    return pred_ndim


def _average_numeric_metrics_by_class(metrics_by_class: dict[str, dict[str, Any]]) -> dict[str, float]:
    """
    对逐类 metrics 中的数值字段做 macro 平均。

    输入参数:
        - metrics_by_class: dict[str, dict[str, Any]], 前景类别名到该类 metrics 的映射

    输出:
        - macro_metrics: dict[str, float], 数值字段的类别间平均值
    """
    if len(metrics_by_class) == 0:
        raise ValueError("metrics_by_class 不能为空")
    # set[str], 所有类别 metrics 中出现的数值字段名
    metric_names: set[str] = set()
    for class_metrics in metrics_by_class.values():
        for key, value in class_metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                metric_names.add(str(key))
    # dict[str, float], 类别间简单平均后的 macro metrics
    macro_metrics: dict[str, float] = {}
    for metric_name in sorted(metric_names):
        values = [float(class_metrics[metric_name]) for class_metrics in metrics_by_class.values() if metric_name in class_metrics]
        if len(values) > 0:
            macro_metrics[metric_name] = float(np.mean(values))
    return macro_metrics


def _evaluate_single_post_result(
    post_result: VoxelPostprocessResult,
    gt_ligand_mask: np.ndarray,
    gt_instance_label: np.ndarray,
    eval_params: dict[str, Any],
) -> dict[str, Any]:
    """
    单个类别(或二分类前景)的后处理结果进行评估指标: evaluate_voxel_mask, evaluate_global_instance_matching, evaluate_topk_success(后两者可选)

    输入参数:
        - post_result: VoxelPostprocessResult, 当前类别的后处理结果
        - gt_ligand_mask: np.ndarray, (D,H,W), 当前类别 GT ligand 掩码
        - gt_instance_label: np.ndarray, (D,H,W), 当前类别 GT instance 标签
        - eval_params: dict[str, Any], 评估参数, 含 compute_instance_metrics/coverage_thresholds/topk_values

    输出:
        - metrics: dict[str, Any], voxel 指标; compute_instance_metrics=True 时追加 instance 计数与 top-K 0/1
    """
    # dict[str, Any], 当前类别平铺 metrics
    metrics: dict[str, Any] = {}
    metrics.update(evaluate_voxel_mask(post_result.binary_mask_filtered, gt_ligand_mask))
    if bool(eval_params["compute_instance_metrics"]):
        # tuple[float, ...], 覆盖率阈值集合
        coverage_thresholds = tuple(eval_params.get("coverage_thresholds", DEFAULT_COVERAGE_THRESHOLDS))
        # tuple[int, ...], top-K 取值集合
        topk_values = tuple(eval_params.get("topk_values", DEFAULT_TOPK_VALUES))
        metrics.update(
            evaluate_global_instance_matching(
                pred_instance_label=post_result.instance_label_filtered,
                gt_instance_label=gt_instance_label,
                coverage_thresholds=coverage_thresholds,
            )
        )
        metrics.update(
            evaluate_topk_success(
                pred_instance_label=post_result.instance_label_filtered,
                gt_instance_label=gt_instance_label,
                candidates=post_result.candidates,
                topk_values=topk_values,
                coverage_thresholds=coverage_thresholds,
            )
        )
    metrics["num_candidates"] = int(len(post_result.candidates))
    return metrics


def _evaluate_post_results(
    post_results_by_class: dict[str, VoxelPostprocessResult],
    cache_data: VoxelPredCacheData,
    eval_params: dict[str, Any],
) -> dict[str, Any] | None:
    """
    根据二分类或多分类后处理结果计算评估指标。

    输入参数:
        - post_results_by_class: dict[str, VoxelPostprocessResult], 前景类别名到后处理结果的映射
        - cache_data: VoxelPredCacheData, 当前样本缓存数据, 可包含 union GT 和逐类 GT
        - eval_params: dict[str, Any], 评估参数, 含 compute_instance_metrics/coverage_thresholds/topk_values

    输出:
        - metrics: dict[str, Any] | None, 二分类为平铺指标, 多分类为 by_class/macro 嵌套指标; 无 GT 时为 None
    """
    if len(post_results_by_class) == 0:
        raise ValueError("post_results_by_class 不能为空")
    first_class_name, first_post_result = next(iter(post_results_by_class.items()))
    if first_class_name == "foreground":
        if cache_data.gt_ligand_mask is None or cache_data.gt_instance_label is None:
            return None
        metrics = _evaluate_single_post_result(
            post_result=first_post_result,
            gt_ligand_mask=cache_data.gt_ligand_mask,
            gt_instance_label=cache_data.gt_instance_label,
            eval_params=eval_params,
        )
        if bool(eval_params.get("compute_pr_auc", True)) and np.asarray(cache_data.ligand_pred).ndim == 3:
            # dict[str, float | None], test 阶段 per-sample PR-AUC; summary macro 仍由 voxel_tuning 统一汇总
            metrics.update(
                evaluate_voxel_pr_auc(
                    score_map=cache_data.ligand_pred,
                    gt_ligand_mask=cache_data.gt_ligand_mask,
                    hardmask=cache_data.hardmask,
                )
            )
        return metrics

    if cache_data.gt_ligand_mask_by_class is None or cache_data.gt_instance_label_by_class is None:
        if cache_data.gt_ligand_mask is None and cache_data.gt_instance_label is None:
            return None
        raise ValueError("多分类评估需要逐类 GT, 请删除旧缓存后重新构建")
    # dict[str, dict[str, Any]], 前景类别名到该类 metrics 的映射
    metrics_by_class: dict[str, dict[str, Any]] = {}
    for class_name, post_result in post_results_by_class.items():
        if class_name not in cache_data.gt_ligand_mask_by_class or class_name not in cache_data.gt_instance_label_by_class:
            raise ValueError(f"缓存缺少类别 {class_name} 的逐类 GT")
        metrics_by_class[class_name] = _evaluate_single_post_result(
            post_result=post_result,
            gt_ligand_mask=cache_data.gt_ligand_mask_by_class[class_name],
            gt_instance_label=cache_data.gt_instance_label_by_class[class_name],
            eval_params=eval_params,
        )
    return {
        "by_class": metrics_by_class,
        "macro": _average_numeric_metrics_by_class(metrics_by_class),
    }


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

def _gt_id_to_name(
    gt_data: dict[str, Any],
    class_names: list[str],
) -> tuple[dict[str, np.ndarray] | None, dict[str, np.ndarray] | None]:
    """
    输入参数:
        - gt_data: dict[str, Any], load_ligand_gt_from_* 返回的 GT 字典
        - class_names: list[str], (C,), 任务类别名列表, 下标对应 class_id

    输出:
        - gt_ligand_mask_by_class: dict[str, np.ndarray] | None, 前景类别名到 (D,H,W) GT ligand mask 的映射
        - gt_instance_label_by_class: dict[str, np.ndarray] | None, 前景类别名到 (D,H,W) GT instance 标签的映射
    """
    # dict[int, np.ndarray], class_id -> (D,H,W), 类别内 GT ligand mask
    mask_by_class_id = gt_data.get("gt_ligand_mask_by_class_id")
    # dict[int, np.ndarray], class_id -> (D,H,W), 类别内 GT instance 标签
    label_by_class_id = gt_data.get("gt_instance_label_by_class_id")
    if mask_by_class_id is None or label_by_class_id is None:
        if gt_data.get("gt_ligand_mask") is None or gt_data.get("gt_instance_label") is None:
            return None, None
        mask_by_class_id = {}
        label_by_class_id = {}
    if set(mask_by_class_id.keys()) != set(label_by_class_id.keys()):
        raise ValueError("逐类 GT mask 与逐类 GT instance label 的 class_id 不一致")
    if gt_data.get("gt_ligand_mask") is None or gt_data.get("gt_instance_label") is None:
        return None, None
    # tuple[int,int,int], GT 体素网格形状(D,H,W)
    grid_shape_zyx = tuple(int(v) for v in np.asarray(gt_data["gt_ligand_mask"]).shape)
    # dict[str, np.ndarray], 前景类别名 -> (D,H,W), 类别内 GT ligand mask
    gt_ligand_mask_by_class: dict[str, np.ndarray] = {}
    # dict[str, np.ndarray], 前景类别名 -> (D,H,W), 类别内 GT instance 标签
    gt_instance_label_by_class: dict[str, np.ndarray] = {}
    for class_id in range(1, len(class_names)):
        # str, 当前前景类别名
        class_name = str(class_names[class_id])
        if class_id in mask_by_class_id:
            gt_ligand_mask_by_class[class_name] = np.asarray(mask_by_class_id[class_id], dtype=bool)
            gt_instance_label_by_class[class_name] = np.asarray(label_by_class_id[class_id], dtype=np.int32)
        else:
            gt_ligand_mask_by_class[class_name] = np.zeros(grid_shape_zyx, dtype=bool)
            gt_instance_label_by_class[class_name] = np.zeros(grid_shape_zyx, dtype=np.int32)
    return gt_ligand_mask_by_class, gt_instance_label_by_class


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


def _resolve_path_by_cfg_key(
    cfg_dict: dict[str, Any],
    selector_key: str,
) -> str:
    """
    按配置中的字段名( cif_path / cif_gt_path / sim_map_path_cryoatom)选择当前样本路径。

    输入参数:
        - cfg_dict: dict[str, Any], 当前推理配置
        - selector_key: str, 选择器配置名; 其值必须是当前 cfg_dict 中的路径字段名

    输出:
        - path_value: str, 解析出的路径字符串
    """
    # str, 资源字段名, 例如 cif_path / cif_gt_path / sim_map_path_cryoatom
    source_key = str(_get_cfg(cfg_dict, selector_key, True))
    # object, 资源字段值; 必须非空
    path_value = _get_cfg(cfg_dict, source_key, True)
    return str(path_value)

def _resolve_forward_structure_path(cfg_dict: dict[str, Any]) -> str:
    """
    解析当前 forward 实际使用的PDB结构路径。

    输入参数:
        - cfg_dict: dict[str, Any], 当前推理配置; structure_input_source 的值必须是结构路径字段名

    输出:
        - cif_path: str, 实际传给 load_from_raw_cif() 的结构路径
    """
    return _resolve_path_by_cfg_key(cfg_dict, "structure_input_source")

def _resolve_gt_receptor_path(cfg_dict: dict[str, Any]) -> str:
    """
    解析 structure GT 中 compute_binding_labels 使用的受体结构路径。

    输入参数:
        - cfg_dict: dict[str, Any], 当前推理配置; gt_receptor_source 的值必须是结构路径字段名

    输出:
        - receptor_structure_path: str, 受体标注结构路径
    """
    return _resolve_path_by_cfg_key(cfg_dict, "gt_receptor_source")

def _resolve_effective_sim_map_path(cfg_dict: dict[str, Any]) -> str | None:
    """
    解析当前 forward 可能需要的模拟密度图路径(不需要则返回None)。

    输入参数:
        - cfg_dict: dict[str, Any], 当前推理配置; sim_map_source 的值必须是模拟图路径字段名

    输出:
        - sim_map_path: str | None, 仅启用 sim/diff/posdiff 通道时返回路径
    """
    # list[str], 实际启用的密度通道名
    density_channel_names = _resolve_density_channel_names_from_cfg(cfg_dict)
    # bool, 当前模型输入是否需要模拟密度图
    needs_sim = any(name.split("_")[0] in {"sim", "diff", "posdiff"} for name in density_channel_names)
    if not needs_sim:
        return None
    return _resolve_path_by_cfg_key(cfg_dict, "sim_map_source")



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
            - "merge_dist": float, 初步 instance 中心世界坐标合并阈值
    """
    params = {
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
        "merge_dist": float(_get_cfg(cfg_dict, "merge_dist", True)),
    }
    by_class = _get_cfg(cfg_dict, "postprocess_by_class", False)
    if by_class is not None:
        params["by_class"] = dict(by_class)
    return params


def _eval_params_from_cfg(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    """
    从配置中构造评估参数; coverage_thresholds / topk_values 为固定常量, 不参与搜索。

    输入参数:
        - cfg_dict: dict[str, Any], 当前推理配置; compute_instance_metrics 缺省时默认 True

    输出:
        - eval_params: dict[str, Any], 含 compute_instance_metrics / coverage_thresholds / topk_values / compute_pr_auc
    """
    # bool|None, 是否计算 instance/top-K; Stage1 voxel-only 时由调用方显式设 False, 缺省按 True
    compute_instance_metrics = _get_cfg(cfg_dict, "compute_instance_metrics", False)
    # bool|None, 是否计算体素 PR-AUC; 仅 test 阶段显式置 True, 缺省 False(不进搜索循环)
    compute_pr_auc = _get_cfg(cfg_dict, "compute_pr_auc", False)
    return {
        "compute_instance_metrics": True if compute_instance_metrics is None else bool(compute_instance_metrics),
        "coverage_thresholds": DEFAULT_COVERAGE_THRESHOLDS,
        "topk_values": DEFAULT_TOPK_VALUES,
        "compute_pr_auc": False if compute_pr_auc is None else bool(compute_pr_auc),
    }

def _resolve_extra_map_paths(cache_data: VoxelPredCacheData) -> list[str] | None:
    """
    解析可视化时需要额外加载的体素图路径: cache_data.meta.get("raw_diff_map_path") 。

    输入参数:
        - cache_data: VoxelPredCacheData, 当前样本缓存; baseline 缓存的 meta 含 raw_diff_map_path

    输出:
        - extra_map_paths: list[str] | None, 额外 MRC 路径列表; DL 缓存无该字段时为 None
    """
    # str | None, baseline 原始差图的 sidecar MRC 路径; DL 缓存为 None
    raw_diff_map_path = cache_data.meta.get("raw_diff_map_path")
    return [str(raw_diff_map_path)] if raw_diff_map_path else None




# ----------------------------------------------- 用于保存/写入的的工具函数 ------------------------------------------------
def _postprocess_one_probability_map(
    prob_map: np.ndarray,
    receptor_pred: np.ndarray | None,
    origin: np.ndarray,
    voxel_size: np.ndarray,
    post_params: dict[str, Any],
    class_name: str | None = None,
) -> VoxelPostprocessResult:
    """
    对单个类别的一张 ligand 概率图执行 voxel 后处理。

    输入参数:
        - prob_map: np.ndarray, (D, H, W), 当前类别 ligand 概率图
        - receptor_pred: np.ndarray | None, (D, H, W), 当前类别 receptor 概率图或二分类 receptor 概率图
        - origin: np.ndarray, (3,), 体素网格世界坐标原点
        - voxel_size: np.ndarray, (3,), 体素大小
        - post_params: dict[str, Any], 后处理参数字典, 可包含 by_class 子配置
        - class_name: str | None, 当前前景类别名; None 表示不查 by_class 覆盖项

    输出:
        - post_result: VoxelPostprocessResult, 当前类别的后处理结果
    """
    if class_name is not None and "by_class" in post_params:
        # dict[str, Any], 当前类别覆盖后的后处理参数
        class_params = dict(post_params)
        class_params.update(dict(post_params["by_class"].get(class_name, {})))
        post_params = class_params
    return postprocess_ligand_probability_map(
        ligand_pred=prob_map,
        origin=origin,
        voxel_size=voxel_size,
        threshold=float(post_params["threshold"]),
        min_component_voxels=int(post_params["min_component_voxels"]),
        filter_strength=str(post_params["filter_strength"]),
        connectivity_policy=str(post_params["connectivity_policy"]),
        sigma_nearby=float(post_params["sigma_nearby"]),
        kernel_nearby=int(post_params["kernel_nearby"]),
        receptor_pred=receptor_pred,
        sigma_response=float(post_params["sigma_response"]),
        kernel_response=int(post_params["kernel_response"]),
        score_add=float(post_params["score_add"]),
        score_minus=float(post_params["score_minus"]),
        voxel_score_min=float(post_params["voxel_score_min"]),
        instance_score_min=float(post_params["instance_score_min"]),
        merge_dist=float(post_params["merge_dist"]),
    )

def _postprocess_cache(cache_data: VoxelPredCacheData, post_params: dict[str, Any]) -> dict[str, VoxelPostprocessResult]:
    """
    根据缓存中的概率图维度执行二分类或多分类后处理, 返回 VoxelPostprocessResult

    输入参数:
        - cache_data: VoxelPredCacheData, 当前样本的整图预测缓存
        - post_params: dict[str, Any], 后处理参数字典, 可包含 by_class 子配置

    输出:
        - results: dict[str, VoxelPostprocessResult], 类别名到后处理结果的映射; 二分类键为 foreground
    """
    # np.ndarray, (D, H, W) 或 (C, D, H, W), ligand 概率图
    ligand_pred = np.asarray(cache_data.ligand_pred)
    if ligand_pred.ndim == 3:
        return {
            "foreground": _postprocess_one_probability_map(
                prob_map=ligand_pred,
                receptor_pred=cache_data.receptor_pred,
                origin=cache_data.origin,
                voxel_size=cache_data.voxel_size,
                post_params=post_params,
                class_name="foreground",
            )
        }
    if ligand_pred.ndim != 4:
        raise ValueError(f"ligand_pred 必须为 (D,H,W) 或 (C,D,H,W)，实际为 {ligand_pred.shape}")
    # list[str], (C,), 缓存中记录的类别名; 缺失时使用 class_{id} 占位
    class_names = list(cache_data.meta.get("class_names", []))
    # dict[str, VoxelPostprocessResult], 每个前景类别的后处理结果
    results: dict[str, VoxelPostprocessResult] = {}
    for class_id in range(1, ligand_pred.shape[0]):
        # str, 当前前景类别名
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        # np.ndarray | None, (D, H, W), 当前类别 receptor 概率图
        receptor_class = None
        if cache_data.receptor_pred is not None:
            # np.ndarray, (D,H,W) 或 (C,D,H,W), receptor 概率图
            receptor_pred = np.asarray(cache_data.receptor_pred)
            receptor_class = receptor_pred[class_id] if receptor_pred.ndim == 4 else receptor_pred
        results[class_name] = _postprocess_one_probability_map(
            prob_map=ligand_pred[class_id],
            receptor_pred=receptor_class,
            origin=cache_data.origin,
            voxel_size=cache_data.voxel_size,
            post_params=post_params,
            class_name=class_name,
        )
    return results



def _save_postprocess_outputs(
    output_dir: str,
    post_result: VoxelPostprocessResult,
) -> None:
    """
    保存后处理输出(ligand_mask_filtered.npz、instance_label_filtered.npz)和候选 JSON(voxel_candidates.json)。

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
    ligand_pred = cache_data.ligand_pred.astype(np.float32)
    np.savez(
        os.path.join(output_dir, "ligand_pred.npz"),
        ligand_pred=ligand_pred,
    )
    if ligand_pred.ndim == 4:
        class_names = list(cache_data.meta.get("class_names", []))
        for class_id in range(1, ligand_pred.shape[0]):
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            np.savez(os.path.join(class_dir, "ligand_pred.npz"), ligand_pred=ligand_pred[class_id])
    if cache_data.receptor_pred is not None:
        receptor_pred = cache_data.receptor_pred.astype(np.float32)
        np.savez(
            os.path.join(output_dir, "receptor_pred.npz"),
            receptor_pred=receptor_pred,
        )
        if receptor_pred.ndim == 4:
            class_names = list(cache_data.meta.get("class_names", []))
            for class_id in range(1, receptor_pred.shape[0]):
                class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                class_dir = os.path.join(output_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                np.savez(os.path.join(class_dir, "receptor_pred.npz"), receptor_pred=receptor_pred[class_id])



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
        "global_instance_precision_cov03",
        "global_instance_recall_cov03",
        "global_instance_f1_cov03",
        "global_instance_precision_cov06",
        "global_instance_recall_cov06",
        "global_instance_f1_cov06",
        "top3_success_ratio_cov03",
        "top4_success_ratio_cov03",
        "top5_success_ratio_cov03",
        "top3_success_ratio_cov06",
        "top4_success_ratio_cov06",
        "top5_success_ratio_cov06",
        "postprocess_params_json",
    ]
    ws.append(headers)
    for cell in ws[1]:
        cell.font = Font(bold=True)

    # list[str], 指标列名(去掉首列 rank 与末列 postprocess_params_json); Stage1 voxel-only 时 instance/top-K 列留空
    metric_headers = headers[1:-1]
    sorted_history = sorted(history, key=lambda item: float(item.get("objective_score", 0.0)), reverse=True)
    for rank, item in enumerate(sorted_history, start=1):
        # list[Any], 当前行内容: rank + 各指标(缺失留空) + 后处理参数 JSON
        row: list[Any] = [rank]
        row.extend(item.get(name, "") for name in metric_headers)
        row.append(json.dumps(item.get("postprocess_params", {}), ensure_ascii=False, sort_keys=True, default=_json_default))
        ws.append(row)
    excel_path = os.path.join(output_root, "param_search_results.xlsx")
    wb.save(excel_path)
    return excel_path

