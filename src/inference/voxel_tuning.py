from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import differential_evolution
from tqdm import tqdm

from src.inference.voxel_evaluator import evaluate_instance_mask, evaluate_voxel_mask
from src.inference.voxel_postprocess import postprocess_ligand_probability_map
from src.inference.utils.voxel_types import VoxelPredCacheData

# ----------------------------------- 保存/读取逻辑 -------------------------------------
def _json_default(value: Any) -> Any:
    """
    将 numpy/path 等对象转换为可 JSON 序列化对象。

    输入参数:
        - value: Any, 待序列化对象

    输出:
        - converted: Any, JSON 可序列化对象
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"对象不可 JSON 序列化: {type(value)}")

def save_voxel_prediction_cache(
    cache_path: str,
    ligand_pred: np.ndarray,
    receptor_pred: np.ndarray | None,
    hardmask: np.ndarray,
    resampled_emdb: np.ndarray,
    origin: np.ndarray,
    voxel_size: np.ndarray,
    meta: dict[str, Any],
    gt_ligand_mask: np.ndarray | None,
    gt_instance_label: np.ndarray | None,
) -> str:
    """
    保存 voxel-only GPU forward 概率缓存。

    输入参数:
        - cache_path: str, .npz 输出路径; 调用方显式控制缓存命名
        - ligand_pred: np.ndarray, (D,H,W), ligand 概率图
        - receptor_pred: np.ndarray | None, (D,H,W), receptor 概率图
        - hardmask: np.ndarray, (D,H,W), 原子落点掩码
        - resampled_emdb: np.ndarray, (D,H,W), 重采样真实密度
        - origin: np.ndarray, (3,), 世界坐标原点(x,y,z)
        - voxel_size: np.ndarray, (3,), 体素大小(x,y,z)
        - meta: dict[str, Any], 缓存信息, 包含: sample_name, cache_path, cif_path, map_path, sim_map_path, cif_gt_path, label_npz_path, gt_source, density_channel_names
        - gt_ligand_mask: np.ndarray | None, (D,H,W), GT ligand 掩码
        - gt_instance_label: np.ndarray | None, (D,H,W), GT instance 标签

    输出:
        - cache_path: str, 写出的缓存路径
    """
    cache_parent = os.path.dirname(cache_path)
    if cache_parent:
        os.makedirs(cache_parent, exist_ok=True)

    # np.ndarray, (D,H,W), float32 或空数组, receptor 缓存占位
    receptor_array = np.asarray(receptor_pred, dtype=np.float32) if receptor_pred is not None else np.empty((0,), dtype=np.float32)
    # np.ndarray, (D,H,W), bool 或空数组, GT ligand 缓存占位
    gt_mask_array = np.asarray(gt_ligand_mask, dtype=bool) if gt_ligand_mask is not None else np.empty((0,), dtype=bool)
    # np.ndarray, (D,H,W), int32 或空数组, GT instance 缓存占位
    gt_instance_array = np.asarray(gt_instance_label, dtype=np.int32) if gt_instance_label is not None else np.empty((0,), dtype=np.int32)
    meta_json = json.dumps(meta, ensure_ascii=False, sort_keys=True, default=_json_default)

    np.savez(
        cache_path,
        ligand_pred=np.asarray(ligand_pred, dtype=np.float32),
        receptor_pred=receptor_array,
        has_receptor_pred=np.asarray(receptor_pred is not None, dtype=bool),
        hardmask=np.asarray(hardmask, dtype=np.int64),
        resampled_emdb=np.asarray(resampled_emdb, dtype=np.float32),
        origin=np.asarray(origin, dtype=np.float32),
        voxel_size=np.asarray(voxel_size, dtype=np.float32),
        gt_ligand_mask=gt_mask_array,
        has_gt_ligand_mask=np.asarray(gt_ligand_mask is not None, dtype=bool),
        gt_instance_label=gt_instance_array,
        has_gt_instance_label=np.asarray(gt_instance_label is not None, dtype=bool),
        meta_json=np.asarray(meta_json),
    )
    return cache_path

def load_voxel_prediction_cache(cache_path: str) -> VoxelPredCacheData:
    """
    读取 voxel-only GPU forward 概率缓存。

    输入参数:
        - cache_path: str, .npz 缓存路径; 调用方显式控制读取目标

    输出:
        - data: VoxelPredCacheData, 缓存数据对象
    """
    with np.load(cache_path, allow_pickle=False) as data:
        # np.ndarray, (D,H,W), float32, ligand 概率图
        ligand_pred = data["ligand_pred"].astype(np.float32, copy=False)
        # bool, receptor_pred 是否真实存在
        has_receptor_pred = bool(data["has_receptor_pred"].item())
        receptor_pred = data["receptor_pred"].astype(np.float32, copy=False) if has_receptor_pred else None
        has_gt_ligand_mask = bool(data["has_gt_ligand_mask"].item())
        gt_ligand_mask = data["gt_ligand_mask"].astype(bool, copy=False) if has_gt_ligand_mask else None
        has_gt_instance_label = bool(data["has_gt_instance_label"].item())
        gt_instance_label = data["gt_instance_label"].astype(np.int32, copy=False) if has_gt_instance_label else None
        meta = json.loads(str(data["meta_json"].item()))

        return VoxelPredCacheData(
            ligand_pred=ligand_pred,
            receptor_pred=receptor_pred,
            hardmask=data["hardmask"].astype(np.int64, copy=False),
            resampled_emdb=data["resampled_emdb"].astype(np.float32, copy=False),
            origin=data["origin"].astype(np.float32, copy=False),
            voxel_size=data["voxel_size"].astype(np.float32, copy=False),
            gt_ligand_mask=gt_ligand_mask,
            gt_instance_label=gt_instance_label,
            meta=meta,
        )












# ------------------------------------------------------- 使用缓存数据跑 batch推理/调参 -------------------------------------------------------

# ------------------------------------------ 单样本 -------------------------------------------
# 【已memory时】: 用一个样本结果(data), 评估一套后处理参数, 产生这组缓存的 summary
def evaluate_loaded_cached_sample_with_postprocess(
    cache_path: str,
    data: VoxelPredCacheData,
    postprocess_params: dict[str, Any],
    eval_params: dict[str, Any],
) -> dict[str, Any]:
    """
    【已memory时】: 用一个样本结果(data), 评估一套后处理参数, 产生这组缓存的 summary。

    输入参数:
        - cache_path: str, voxel prediction cache 路径, 仅用于标记结果(不读取)
        - data: VoxelPredCacheData, 已加载的 voxel prediction cache 数据
        - postprocess_params: dict[str, Any], 后处理参数
        - eval_params: dict[str, Any], 评估参数, 包含 alpha/beta

    输出:
        - summary: dict[str, Any], 单样本后处理评估结果, 包含:
            - "cache_path": str, 当前缓存路径
            - "num_candidates": int, 后处理后保留的 ligand 候选数
            - "voxel_precision": float, 体素级精确率
            - "voxel_recall": float, 体素级召回率
            - "voxel_f1": float, 体素级 F1
            - "voxel_iou": float, 体素级 IoU
            - "voxel_dice": float, 体素级 Dice
            - "instance_precision": float, instance 级精确率
            - "instance_recall": float, instance 级召回率
            - "instance_f1": float, instance 级 F1
            - "tp": int, 体素级真阳性体素数
            - "fp": int, 体素级假阳性体素数
            - "fn": int, 体素级假阴性体素数
            - "tn": int, 体素级真阴性体素数
            - "num_pred_instances": int, 预测 instance 数
            - "num_gt_instances": int, GT instance 数
            - "pred_instance_tp": int, 满足 precision 阈值的预测 instance 数
            - "gt_instance_hit": int, 满足 recall 阈值的 GT instance 数
    """
    if data.gt_ligand_mask is None or data.gt_instance_label is None:
        raise ValueError(f"缓存缺少 GT 字段, 无法评估: {cache_path}")

    result = postprocess_ligand_probability_map(
        ligand_pred=data.ligand_pred,
        origin=data.origin,
        voxel_size=data.voxel_size,
        threshold=float(postprocess_params["threshold"]),
        min_component_voxels=int(postprocess_params["min_component_voxels"]),
        filter_strength=str(postprocess_params["filter_strength"]),
        connectivity_policy=str(postprocess_params["connectivity_policy"]),
        sigma_nearby=float(postprocess_params["sigma_nearby"]),
        kernel_nearby=int(postprocess_params["kernel_nearby"]),
        receptor_pred=data.receptor_pred,
        sigma_response=float(postprocess_params["sigma_response"]),
        kernel_response=int(postprocess_params["kernel_response"]),
        score_add=float(postprocess_params["score_add"]),
        score_minus=float(postprocess_params["score_minus"]),
        voxel_score_min=float(postprocess_params["voxel_score_min"]),
        instance_score_min=float(postprocess_params["instance_score_min"]),
    )
    voxel_metrics = evaluate_voxel_mask(result.binary_mask_filtered, data.gt_ligand_mask)
    instance_metrics = evaluate_instance_mask(
        pred_instance_label=result.instance_label_filtered,
        gt_instance_label=data.gt_instance_label,
        alpha=float(eval_params["alpha"]),
        beta=float(eval_params["beta"]),
    )
    merged = {"cache_path": cache_path, "num_candidates": int(len(result.candidates))}
    merged.update(voxel_metrics)
    merged.update(instance_metrics)
    return merged

# 【disk】: 加载一个样本结果(data), 评估一套后处理参数, 产生这组缓存的 summary
def evaluate_cached_sample_with_postprocess(
    cache_path: str,
    postprocess_params: dict[str, Any],
    eval_params: dict[str, Any],
) -> dict[str, Any]:
    """
    【disk】: 加载一个样本结果(data), 评估一套后处理参数, 产生这组缓存的 summary。

    输入参数:
        - cache_path: str, voxel prediction cache 路径
        - postprocess_params: dict[str, Any], 后处理参数
        - eval_params: dict[str, Any], 评估参数, 包含 alpha/beta

    输出:
        - metrics: dict[str, Any], 单样本后处理评估结果, 字段同 evaluate_loaded_cached_sample_with_postprocess()
    """
    data = load_voxel_prediction_cache(cache_path)
    return evaluate_loaded_cached_sample_with_postprocess(
        cache_path=cache_path,
        data=data,
        postprocess_params=postprocess_params,
        eval_params=eval_params,
    )







# ------------------------------------------ 多样本(用上面单样本函数) -------------------------------------------
# 用 batch 每个样本产生的 summary, 合成 summarys
def _summarize_postprocess_metrics(
    per_sample: list[dict[str, Any]],
    postprocess_params: dict[str, Any],
) -> dict[str, Any]:
    """
    汇总一组缓存样本的后处理评估结果。

    输入参数:
        - per_sample(summary): list[dict[str, Any]], 每个条目为 evaluate_loaded_cached_sample_with_postprocess 产生的一个样本结果
        - postprocess_params: dict[str, Any], 本次评估使用的完整后处理参数

    输出:
        - summarys: dict[str, Any], 在 per_sample(summary) 的基础上添加逐样本(在 postprocess_params 里每个参数上的)汇总结果
    """
    metric_names = [
        "num_candidates",
        "voxel_precision",
        "voxel_recall",
        "voxel_f1",
        "voxel_iou",
        "voxel_dice",
        "instance_precision",
        "instance_recall",
        "instance_f1",
    ]
    summarys: dict[str, Any] = {"per_sample": per_sample, "postprocess_params": dict(postprocess_params)}
    for metric_name in metric_names:
        values = [float(item[metric_name]) for item in per_sample]
        summarys[f"avg_{metric_name}"] = float(np.mean(values))
    return summarys

# 【已memory时】: 用多个样本结果(datas), 评估一套后处理参数, 产生这组缓存的 summarys
def evaluate_loaded_postprocess_params_on_cache_set(
    loaded_cache_items: list[tuple[str, VoxelPredCacheData]],
    postprocess_params: dict[str, Any],
    eval_params: dict[str, Any],
    n_jobs: int,
) -> dict[str, Any]:
    """
    【已memory时】: 用多个样本结果(datas), 评估一套后处理参数, 产生这组缓存的 summarys。

    输入参数:
        - loaded_cache_items: list[tuple[str, VoxelPredCacheData]], 可变长度, 已加载的缓存路径与数据列表
        - postprocess_params: dict[str, Any], 后处理参数
        - eval_params: dict[str, Any], 评估参数
        - n_jobs: int, joblib 并行 worker 数

    输出:
        - summarys: dict[str, Any], 整套缓存样本的汇总结果, 包含 per_sample/postprocess_params/avg_* 指标
    """
    if len(loaded_cache_items) == 0:
        raise ValueError("loaded_cache_items 不能为空")
    per_sample = Parallel(n_jobs=int(n_jobs), prefer="threads")(
        delayed(evaluate_loaded_cached_sample_with_postprocess)(cache_path, data, postprocess_params, eval_params)
        for cache_path, data in loaded_cache_items
    )
    return _summarize_postprocess_metrics(per_sample, postprocess_params)

# 【disk】: 用多个样本结果(datas), 评估一套后处理参数, 产生这组缓存的 summarys
def evaluate_postprocess_params_on_cache_set(
    cache_paths: list[str],
    postprocess_params: dict[str, Any],
    eval_params: dict[str, Any],
    n_jobs: int,
) -> dict[str, Any]:
    """
    【disk】: 用多个样本结果(datas), 评估一套后处理参数, 产生这组缓存的 summarys。

    输入参数:
        - cache_paths: list[str], 可变长度, 缓存路径列表
        - postprocess_params: dict[str, Any], 后处理参数
        - eval_params: dict[str, Any], 评估参数
        - n_jobs: int, joblib 并行 worker 数

    输出:
        - summarys: dict[str, Any], 整套缓存样本的汇总结果, 包含:
            - "per_sample": list[dict[str, Any]], 可变长度, 每个缓存样本的单样本评估结果
            - "postprocess_params": dict[str, Any], 本次评估使用的完整后处理参数
            - "avg_num_candidates": float, 全样本平均后处理候选数
            - "avg_voxel_precision": float, 全样本平均体素级精确率
            - "avg_voxel_recall": float, 全样本平均体素级召回率
            - "avg_voxel_f1": float, 全样本平均体素级 F1
            - "avg_voxel_iou": float, 全样本平均体素级 IoU
            - "avg_voxel_dice": float, 全样本平均体素级 Dice
            - "avg_instance_precision": float, 全样本平均 instance 级精确率
            - "avg_instance_recall": float, 全样本平均 instance 级召回率
            - "avg_instance_f1": float, 全样本平均 instance 级 F1
    """
    if len(cache_paths) == 0:
        raise ValueError("cache_paths 不能为空")
    # joblib.Parallel(...) 会执行这个 generator 里的每个 delayed task，并把每个 evaluate_cached_sample_with_postprocess(...) 的返回值按输入顺序收集成一个 list
    per_sample = Parallel(n_jobs=int(n_jobs))(
        delayed(evaluate_cached_sample_with_postprocess)(cache_path, postprocess_params, eval_params)
        for cache_path in cache_paths
    )
    return _summarize_postprocess_metrics(per_sample, postprocess_params)

# 用 summarys 算 score
def _score_postprocess_summary(
    summarys: dict[str, Any],
    optimizer_params: dict[str, Any],
) -> float:
    """
    根据单次参数评估 summarys 计算优化目标分数。

    输入参数:
        - summarys: dict[str, Any], evaluate_postprocess_params_on_cache_set() 返回的汇总结果
        - optimizer_params: dict[str, Any], 优化器参数, 包含 objective_expr/fixed_search_params

    输出:
        - score: float, 当前参数组合的优化目标分数; 数值越大越好
    """
    # dict[str, Any], eval 可直接使用的局部变量; summarys 保留完整汇总对象
    variables: dict[str, Any] = {"summarys": summarys}
    for key, value in summarys.items():
        if isinstance(value, (bool, np.bool_)):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            variables[str(key)] = float(value)
    return float(eval(str(optimizer_params["objective_expr"]), {}, variables))







# ------------------------------------------ 总函数 -------------------------------------------
def optimize_postprocess_params(
    cache_paths: list[str],
    loaded_cache_items: list[tuple[str, VoxelPredCacheData]],
    cache_data_mode: str,
    fixed_postprocess_params: dict[str, Any],
    search_space: dict[str, Any],
    search_strategy: str,
    eval_params: dict[str, Any],
    optimizer_params: dict[str, Any],
    n_jobs: int,
    show_progress: bool,
) -> dict[str, Any]:
    """
    已知缓存推理结果 —————— cache_paths("disk", 磁盘查询) 或 loaded_cache_items("memory", 内存读取), 运行后处理程序 + 评估脚本, 得到 result 。

    输入参数:
        - cache_paths: list[str], 可变长度, 缓存路径列表; disk 模式下用于逐轮加载
        - loaded_cache_items: list[tuple[str, VoxelPredCacheData]] | 空列表,  memory 模式下预加载缓存; disk 模式传空列表
        - cache_data_mode: str, 缓存数据读取模式, 可选 disk/memory
        - fixed_postprocess_params: dict[str, Any], 固定后处理参数
        - search_space: dict[str, Any], 描述参数空间。单个条目形如: threshold: {type: float, min: 0.05, max: 0.95, step: 0.10}
        - search_strategy: str, grid 或 differential_evolution
        - eval_params: dict[str, Any], 评估参数
        - optimizer_params: dict[str, Any], 优化器参数, 包含 objective_expr/fixed_search_params
        - n_jobs: int, 并行 worker 数
        - show_progress: bool, 是否显示参数组合搜索进度条

    输出:
        - result: dict[str, Any], 参数搜索结果, 包含:
            - "best_params": dict[str, Any], 搜索得到的最佳完整后处理参数
            - "best_metrics": dict[str, Any], 从最佳 summarys 中抽取出的 `avg_*` 指标和 objective_score
            - "history": list[dict[str, Any]], 可变长度, 按搜索轨迹累积的 summarys 历史
            - "optimizer_fun": float, differential_evolution 的最终目标函数值; 仅该策略下存在
    """
    # list[str], 被强制固定的后处理参数名; 同名 search_space 条目会被忽略
    fixed_search_param_names = [str(name) for name in optimizer_params["fixed_search_params"]]
    # dict[str, Any], 实际参与搜索的参数空间
    active_search_space = {
        str(name): spec
        for name, spec in search_space.items()
        if str(name) not in fixed_search_param_names
    }
    history: list[dict[str, Any]] = []


    def evaluate_params(search_params: dict[str, Any]) -> dict[str, Any]:
        # dict[str, Any], 去掉固定参数后的搜索参数
        active_search_params = {
            str(name): value
            for name, value in search_params.items()
            if str(name) not in fixed_search_param_names
        }
        params = dict(fixed_postprocess_params)
        params.update(active_search_params)
        for name in fixed_search_param_names:
            params[name] = fixed_postprocess_params[name]
        if cache_data_mode == "memory":
            summarys = evaluate_loaded_postprocess_params_on_cache_set(
                loaded_cache_items=loaded_cache_items,
                postprocess_params=params,
                eval_params=eval_params,
                n_jobs=n_jobs,
            )
        else:
            summarys = evaluate_postprocess_params_on_cache_set(
                cache_paths=cache_paths,
                postprocess_params=params,
                eval_params=eval_params,
                n_jobs=n_jobs,
            )
        summarys["objective_score"] = _score_postprocess_summary(summarys, optimizer_params)
        history.append(summarys)
        return summarys



    if search_strategy not in {"grid", "differential_evolution"}:
        raise ValueError(f"未知 search_strategy: {search_strategy}")
    if len(active_search_space) == 0:
        best_summary = evaluate_params({})
        return {
            "best_params": best_summary["postprocess_params"],
            "best_metrics": {k: v for k, v in best_summary.items() if k.startswith("avg_") or k == "objective_score"},
            "history": history,
        }


    if search_strategy == "grid":
        best_summary = None
        grid_params = generate_param_grid(active_search_space)
        grid_iter = tqdm(grid_params, total=len(grid_params), desc="voxel param grid") if show_progress else grid_params
        for search_params in grid_iter:
            summarys = evaluate_params(search_params)
            if best_summary is None or float(summarys["objective_score"]) > float(best_summary["objective_score"]):
                best_summary = summarys
        if best_summary is None:
            raise RuntimeError("grid 搜索未产生任何结果")
        return {
            "best_params": best_summary["postprocess_params"],
            "best_metrics": {k: v for k, v in best_summary.items() if k.startswith("avg_") or k == "objective_score"},
            "history": history,
        }


    if search_strategy == "differential_evolution":
        search_names = [str(name) for name in active_search_space.keys()]
        bounds = _build_de_bounds(search_names, active_search_space)

        def objective(vector: np.ndarray) -> float:
            search_params = _decode_de_vector(vector, search_names, active_search_space)
            summarys = evaluate_params(search_params)
            return -float(summarys["objective_score"])

        de_result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=int(optimizer_params["max_iter"]),
            popsize=int(optimizer_params["popsize"]),
            seed=int(optimizer_params["random_seed"]),
            polish=False,
        )
        best_search_params = _decode_de_vector(de_result.x, search_names, active_search_space)
        best_summary = evaluate_params(best_search_params)
        return {
            "best_params": best_summary["postprocess_params"],
            "best_metrics": {k: v for k, v in best_summary.items() if k.startswith("avg_") or k == "objective_score"},
            "history": history,
            "optimizer_fun": float(de_result.fun),
        }
    
    raise ValueError(f"未知 search_strategy: {search_strategy}")










# ----------------------------------- 使用缓存调参的工具函数 -----------------------------------
# 用于 if search_strategy == "grid"
def generate_param_grid(search_space: dict[str, Any]) -> list[dict[str, Any]]:
    """
    从离散搜索空间生成参数网格。

    输入参数:
        - search_space: dict[str, Any], 描述参数空间。单个条目形如: threshold: {type: float, min: 0.05, max: 0.95, step: 0.10}

    输出:
        - grid: list[dict[str, Any]], 可变长度, 参数组合列表
    """
    names: list[str] = []
    value_lists: list[list[Any]] = []
    for name, spec in search_space.items():
        names.append(str(name))
        if "values" in spec:
            value_lists.append(list(spec["values"]))
        else:
            if spec["type"] == "int":
                step = int(spec["step"])
                value_lists.append(list(range(int(spec["min"]), int(spec["max"]) + 1, step)))
            elif spec["type"] == "float":
                step = float(spec["step"])
                values = []
                current = float(spec["min"])
                while current <= float(spec["max"]) + 1e-12:
                    values.append(float(round(current, 10)))
                    current += step
                value_lists.append(values)
            else:
                raise ValueError(f"grid 搜索不支持的参数类型: {spec['type']}")
    return [dict(zip(names, values)) for values in itertools.product(*value_lists)]


# 用于 if search_strategy == "differential_evolution"
def _build_de_bounds(search_names: list[str], search_space: dict[str, Any]) -> list[tuple[float, float]]:
    """
    构造 differential_evolution 的连续边界。

    输入参数:
        - search_names: list[str], 参数名列表
        - search_space: dict[str, Any], 描述参数空间。单个条目形如: threshold: {type: float, min: 0.05, max: 0.95, step: 0.10}
    输出:
        - bounds: list[tuple[float,float]], 优化器边界
    """
    bounds: list[tuple[float, float]] = []
    for name in search_names:
        spec = search_space[name]
        if "values" in spec:
            bounds.append((0.0, float(len(spec["values"]) - 1)))
        else:
            bounds.append((float(spec["min"]), float(spec["max"])))
    return bounds

def _decode_de_vector(
    vector: np.ndarray,
    search_names: list[str],
    search_space: dict[str, Any],
) -> dict[str, Any]:
    """
    将 differential_evolution 的一组向量形状的参数 vector, 解码为标准形式的参数 params 。

    输入参数:
        - vector: np.ndarray, (P,), 优化器连续参数
        - search_names: list[str], 长度 P, 参数名列表
        - search_space: dict[str, Any], 描述参数空间。单个条目形如: threshold: {type: float, min: 0.05, max: 0.95, step: 0.10}

    输出:
        - params: dict[str, Any], 解码后的参数。单个条目形如: threshold: 0.072
    """
    params: dict[str, Any] = {}
    for idx, name in enumerate(search_names):
        spec = search_space[name]
        value = vector[idx]
        if "values" in spec:
            values = list(spec["values"])
            value_index = int(np.clip(round(value), 0, len(values) - 1))
            params[name] = values[value_index]
        elif spec["type"] == "int":
            params[name] = int(round(value))
        elif spec["type"] == "float":
            params[name] = float(value)
        else:
            raise ValueError(f"不支持的搜索参数类型: {spec['type']}")
    return params


