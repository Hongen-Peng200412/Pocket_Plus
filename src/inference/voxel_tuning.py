from __future__ import annotations

import copy
import csv
import hashlib
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

def build_voxel_cache_key(
    ckpt_path: str,
    cif_path: str,
    map_path: str,
    sim_map_path: str | None,
    forward_params: dict[str, Any],
    gt_source: str,
    labels_npz_path: str | None,
    cif_gt_path: str | None,
    gt_params: dict[str, Any],
) -> str:
    """
    根据 forward 输入与 GT 输入生成 voxel 缓存键。

    输入参数:
        - ckpt_path: str, checkpoint 路径
        - cif_path: str, 结构文件路径
        - map_path: str, 实验密度图路径
        - sim_map_path: str | None, 模拟密度图路径; 未使用模拟图时为 None
        - forward_params: dict[str, Any], 影响 GPU forward 与整图合并的参数
        - gt_source: str, GT 来源类型, 可选 none/labels_npz/structure
        - labels_npz_path: str | None, labels.npz 路径; 非 labels_npz GT 时为 None
        - cif_gt_path: str | None, hard/trivial GT 结构路径; 无额外 GT 结构时为 None
        - gt_params: dict[str, Any], 影响 GT 构造的参数

    输出:
        - cache_key: str, sha256 十六进制缓存键
    """
    # dict[str, Any], 参与缓存键计算的稳定 JSON 载荷
    payload = {
        "ckpt_path": str(ckpt_path),
        "cif_path": str(cif_path),
        "map_path": str(map_path),
        "sim_map_path": None if sim_map_path is None else str(sim_map_path),
        "forward_params": dict(forward_params),
        "gt_source": str(gt_source),
        "labels_npz_path": None if labels_npz_path is None else str(labels_npz_path),
        "cif_gt_path": None if cif_gt_path is None else str(cif_gt_path),
        "gt_params": dict(gt_params),
    }
    # str, 排序后的 JSON 文本, 保证同一输入产生稳定 hash
    payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_json_default)
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


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
    gt_ligand_mask_by_class: dict[str, np.ndarray] | None,
    gt_instance_label_by_class: dict[str, np.ndarray] | None,
    gt_instance_meta: list[dict[str, Any]] | None,
) -> str:
    """
    保存 voxel-only GPU forward 概率缓存。

    输入参数:
        - cache_path: str, .npz 输出路径; 调用方显式控制缓存命名
        - ligand_pred: np.ndarray, (D,H,W) 或 (C,D,H,W), ligand 概率图
        - receptor_pred: np.ndarray | None, (D,H,W) 或 (C,D,H,W), receptor 概率图
        - hardmask: np.ndarray, (D,H,W), 原子落点掩码
        - resampled_emdb: np.ndarray, (D,H,W), 重采样真实密度
        - origin: np.ndarray, (3,), 世界坐标原点(x,y,z)
        - voxel_size: np.ndarray, (3,), 体素大小(x,y,z)
        - meta: dict[str, Any], 缓存信息, 包含 sample_name、cache_path、class_names 等上下文
        - gt_ligand_mask: np.ndarray | None, (D,H,W), union GT ligand 掩码
        - gt_instance_label: np.ndarray | None, (D,H,W), union GT instance 标签
        - gt_ligand_mask_by_class: dict[str, np.ndarray] | None, 前景类别名到 (D,H,W) GT ligand mask 的映射
        - gt_instance_label_by_class: dict[str, np.ndarray] | None, 前景类别名到 (D,H,W) GT instance 标签的映射
        - gt_instance_meta: list[dict[str, Any]] | None, GT instance 来源元信息列表

    输出:
        - cache_path: str, 写出的缓存路径
    """
    cache_parent = os.path.dirname(cache_path)
    if cache_parent:
        os.makedirs(cache_parent, exist_ok=True)

    # np.ndarray, (D,H,W) 或 (C,D,H,W), float32 或空数组, receptor 缓存占位
    receptor_array = np.asarray(receptor_pred, dtype=np.float32) if receptor_pred is not None else np.empty((0,), dtype=np.float32)
    # np.ndarray, (D,H,W), bool 或空数组, union GT ligand 缓存占位
    gt_mask_array = np.asarray(gt_ligand_mask, dtype=bool) if gt_ligand_mask is not None else np.empty((0,), dtype=bool)
    # np.ndarray, (D,H,W), int32 或空数组, union GT instance 缓存占位
    gt_instance_array = np.asarray(gt_instance_label, dtype=np.int32) if gt_instance_label is not None else np.empty((0,), dtype=np.int32)
    # list[str], 可变长度, 逐类 GT 的前景类别名列表
    gt_class_names = list(gt_ligand_mask_by_class.keys()) if gt_ligand_mask_by_class is not None else []
    if gt_ligand_mask_by_class is not None and gt_instance_label_by_class is None:
        raise ValueError("gt_ligand_mask_by_class 存在时必须同时提供 gt_instance_label_by_class")
    if gt_instance_label_by_class is not None and gt_ligand_mask_by_class is None:
        raise ValueError("gt_instance_label_by_class 存在时必须同时提供 gt_ligand_mask_by_class")
    if gt_instance_label_by_class is not None and set(gt_instance_label_by_class.keys()) != set(gt_class_names):
        raise ValueError("gt_ligand_mask_by_class 与 gt_instance_label_by_class 的类别名不一致")

    # dict[str, np.ndarray], 写入 npz 的逐类 GT 数组字段
    gt_by_class_arrays: dict[str, np.ndarray] = {}
    for class_index, class_name in enumerate(gt_class_names):
        gt_by_class_arrays[f"gt_ligand_mask_class_{class_index}"] = np.asarray(gt_ligand_mask_by_class[class_name], dtype=bool)
        gt_by_class_arrays[f"gt_instance_label_class_{class_index}"] = np.asarray(gt_instance_label_by_class[class_name], dtype=np.int32)

    meta_json = json.dumps(meta, ensure_ascii=False, sort_keys=True, default=_json_default)
    gt_class_names_json = json.dumps(gt_class_names, ensure_ascii=False, sort_keys=True, default=_json_default)
    gt_instance_meta_json = json.dumps([] if gt_instance_meta is None else gt_instance_meta, ensure_ascii=False, sort_keys=True, default=_json_default)

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
        has_gt_by_class=np.asarray(gt_ligand_mask_by_class is not None, dtype=bool),
        gt_class_names_json=np.asarray(gt_class_names_json),
        gt_instance_meta_json=np.asarray(gt_instance_meta_json),
        meta_json=np.asarray(meta_json),
        **gt_by_class_arrays,
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
        # np.ndarray, (D,H,W) 或 (C,D,H,W), float32, ligand 概率图
        ligand_pred = data["ligand_pred"].astype(np.float32, copy=False)
        # bool, receptor_pred 是否真实存在
        has_receptor_pred = bool(data["has_receptor_pred"].item())
        # np.ndarray | None, (D,H,W) 或 (C,D,H,W), float32, receptor 概率图
        receptor_pred = data["receptor_pred"].astype(np.float32, copy=False) if has_receptor_pred else None
        # bool, union GT ligand mask 是否真实存在
        has_gt_ligand_mask = bool(data["has_gt_ligand_mask"].item())
        # np.ndarray | None, (D,H,W), bool, union GT ligand mask
        gt_ligand_mask = data["gt_ligand_mask"].astype(bool, copy=False) if has_gt_ligand_mask else None
        # bool, union GT instance label 是否真实存在
        has_gt_instance_label = bool(data["has_gt_instance_label"].item())
        # np.ndarray | None, (D,H,W), int32, union GT instance 标签
        gt_instance_label = data["gt_instance_label"].astype(np.int32, copy=False) if has_gt_instance_label else None
        # dict[str, Any], 缓存上下文信息
        meta = json.loads(str(data["meta_json"].item()))
        # bool, 缓存是否包含逐类 GT 字段
        has_gt_by_class = "has_gt_by_class" in data.files and bool(data["has_gt_by_class"].item())
        # dict[str, np.ndarray] | None, 前景类别名到 (D,H,W) GT ligand mask 的映射
        gt_ligand_mask_by_class = None
        # dict[str, np.ndarray] | None, 前景类别名到 (D,H,W) GT instance 标签的映射
        gt_instance_label_by_class = None
        if has_gt_by_class:
            # list[str], 可变长度, 逐类 GT 的前景类别名列表
            gt_class_names = json.loads(str(data["gt_class_names_json"].item()))
            gt_ligand_mask_by_class = {}
            gt_instance_label_by_class = {}
            for class_index, class_name in enumerate(gt_class_names):
                gt_ligand_mask_by_class[str(class_name)] = data[f"gt_ligand_mask_class_{class_index}"].astype(bool, copy=False)
                gt_instance_label_by_class[str(class_name)] = data[f"gt_instance_label_class_{class_index}"].astype(np.int32, copy=False)
        # list[dict[str, Any]] | None, GT instance 来源元信息列表
        gt_instance_meta = None
        if "gt_instance_meta_json" in data.files:
            gt_instance_meta = json.loads(str(data["gt_instance_meta_json"].item()))

        return VoxelPredCacheData(
            ligand_pred=ligand_pred,
            receptor_pred=receptor_pred,
            hardmask=data["hardmask"].astype(np.int64, copy=False),
            resampled_emdb=data["resampled_emdb"].astype(np.float32, copy=False),
            origin=data["origin"].astype(np.float32, copy=False),
            voxel_size=data["voxel_size"].astype(np.float32, copy=False),
            gt_ligand_mask=gt_ligand_mask,
            gt_instance_label=gt_instance_label,
            gt_ligand_mask_by_class=gt_ligand_mask_by_class,
            gt_instance_label_by_class=gt_instance_label_by_class,
            gt_instance_meta=gt_instance_meta,
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
        merge_dist=float(postprocess_params["merge_dist"]),
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


def get_foreground_class_names_from_cache(data: VoxelPredCacheData) -> list[str]:
    """
    从缓存数据中解析参与逐类评估的前景类别名。

    输入参数:
        - data: VoxelPredCacheData, voxel prediction cache 数据, ligand_pred 为 (D,H,W) 或 (C,D,H,W)

    输出:
        - class_names: list[str], 可变长度, 前景类别名列表; 二分类固定为 ["foreground"]
    """
    # np.ndarray, (D,H,W) 或 (C,D,H,W), ligand 概率图
    ligand_pred = np.asarray(data.ligand_pred)
    if ligand_pred.ndim == 3:
        return ["foreground"]
    if ligand_pred.ndim != 4:
        raise ValueError(f"ligand_pred 必须为 (D,H,W) 或 (C,D,H,W), 实际为 {ligand_pred.shape}")
    # list[str], (C,), softmax 任务类别名, 需要包含 background
    class_names = [str(name) for name in data.meta.get("class_names", [])]
    if len(class_names) < ligand_pred.shape[0]:
        raise ValueError(f"缓存 class_names 长度 {len(class_names)} 小于 ligand_pred 通道数 {ligand_pred.shape[0]}")
    return class_names[1:ligand_pred.shape[0]]


def select_class_cache_view(
    data: VoxelPredCacheData,
    class_name: str,
) -> VoxelPredCacheData:
    """
    从多分类缓存中切出单个前景类别的 3D 评估视图。

    输入参数:
        - data: VoxelPredCacheData, voxel prediction cache 数据, ligand_pred 为 (D,H,W) 或 (C,D,H,W)
        - class_name: str, 要切出的前景类别名; 二分类缓存只能使用 foreground

    输出:
        - class_data: VoxelPredCacheData, 单类别缓存视图, ligand_pred 为 (D,H,W)
    """
    # np.ndarray, (D,H,W) 或 (C,D,H,W), ligand 概率图
    ligand_pred = np.asarray(data.ligand_pred)
    if ligand_pred.ndim == 3:
        if class_name != "foreground":
            raise ValueError(f"二分类缓存只支持 foreground, 实际请求 {class_name}")
        # dict[str, Any], 单前景视图的缓存上下文
        meta = dict(data.meta)
        meta["selected_class_name"] = "foreground"
        meta["selected_class_id"] = 1
        return VoxelPredCacheData(
            ligand_pred=data.ligand_pred,
            receptor_pred=data.receptor_pred,
            hardmask=data.hardmask,
            resampled_emdb=data.resampled_emdb,
            origin=data.origin,
            voxel_size=data.voxel_size,
            gt_ligand_mask=data.gt_ligand_mask,
            gt_instance_label=data.gt_instance_label,
            gt_ligand_mask_by_class=data.gt_ligand_mask_by_class,
            gt_instance_label_by_class=data.gt_instance_label_by_class,
            gt_instance_meta=data.gt_instance_meta,
            meta=meta,
        )
    if ligand_pred.ndim != 4:
        raise ValueError(f"ligand_pred 必须为 (D,H,W) 或 (C,D,H,W), 实际为 {ligand_pred.shape}")
    # list[str], (C,), softmax 任务类别名, 需要包含 background
    class_names = [str(name) for name in data.meta.get("class_names", [])]
    if len(class_names) < ligand_pred.shape[0]:
        raise ValueError(f"缓存 class_names 长度 {len(class_names)} 小于 ligand_pred 通道数 {ligand_pred.shape[0]}")
    if class_name not in class_names[1:ligand_pred.shape[0]]:
        raise ValueError(f"类别 {class_name} 不在前景类别列表 {class_names[1:ligand_pred.shape[0]]} 中")
    if data.gt_ligand_mask_by_class is None or data.gt_instance_label_by_class is None:
        raise ValueError("多分类缓存缺少逐类 GT, 请删除旧缓存后重新构建")
    if class_name not in data.gt_ligand_mask_by_class or class_name not in data.gt_instance_label_by_class:
        raise ValueError(f"多分类缓存缺少类别 {class_name} 的逐类 GT")
    # int, 当前前景类别在 softmax 通道中的类别 ID
    class_id = class_names.index(class_name)
    # np.ndarray | None, (D,H,W), 当前类别 receptor 概率图
    receptor_class = None
    if data.receptor_pred is not None:
        # np.ndarray, (D,H,W) 或 (C,D,H,W), receptor 概率图
        receptor_pred = np.asarray(data.receptor_pred)
        if receptor_pred.ndim == 3:
            receptor_class = receptor_pred
        elif receptor_pred.ndim == 4:
            receptor_class = receptor_pred[class_id]
        else:
            raise ValueError(f"receptor_pred 必须为 (D,H,W) 或 (C,D,H,W), 实际为 {receptor_pred.shape}")
    # dict[str, Any], 单类别视图的缓存上下文
    meta = dict(data.meta)
    meta["selected_class_name"] = class_name
    meta["selected_class_id"] = class_id
    return VoxelPredCacheData(
        ligand_pred=ligand_pred[class_id],
        receptor_pred=receptor_class,
        hardmask=data.hardmask,
        resampled_emdb=data.resampled_emdb,
        origin=data.origin,
        voxel_size=data.voxel_size,
        gt_ligand_mask=data.gt_ligand_mask_by_class[class_name],
        gt_instance_label=data.gt_instance_label_by_class[class_name],
        gt_ligand_mask_by_class=data.gt_ligand_mask_by_class,
        gt_instance_label_by_class=data.gt_instance_label_by_class,
        gt_instance_meta=data.gt_instance_meta,
        meta=meta,
    )


def evaluate_loaded_cached_sample_for_class_with_postprocess(
    cache_path: str,
    data: VoxelPredCacheData,
    class_name: str,
    postprocess_params: dict[str, Any],
    eval_params: dict[str, Any],
) -> dict[str, Any]:
    """
    对已加载缓存的单个前景类别执行后处理并评估。

    输入参数:
        - cache_path: str, voxel prediction cache 路径, 仅用于标记结果
        - data: VoxelPredCacheData, 已加载的 voxel prediction cache 数据
        - class_name: str, 当前评估的前景类别名
        - postprocess_params: dict[str, Any], 后处理参数
        - eval_params: dict[str, Any], 评估参数, 包含 alpha/beta

    输出:
        - metrics: dict[str, Any], 单类别单样本后处理评估结果, 包含 class_name/class_id/cache_path
    """
    # VoxelPredCacheData, 单类别 3D 评估视图
    class_data = select_class_cache_view(data, class_name)
    metrics = evaluate_loaded_cached_sample_with_postprocess(
        cache_path=cache_path,
        data=class_data,
        postprocess_params=postprocess_params,
        eval_params=eval_params,
    )
    metrics["class_name"] = class_name
    metrics["class_id"] = int(class_data.meta["selected_class_id"])
    return metrics


def evaluate_cached_sample_for_class_with_postprocess(
    cache_path: str,
    class_name: str,
    postprocess_params: dict[str, Any],
    eval_params: dict[str, Any],
) -> dict[str, Any]:
    """
    从磁盘读取缓存后对单个前景类别执行后处理并评估。

    输入参数:
        - cache_path: str, voxel prediction cache 路径
        - class_name: str, 当前评估的前景类别名
        - postprocess_params: dict[str, Any], 后处理参数
        - eval_params: dict[str, Any], 评估参数, 包含 alpha/beta

    输出:
        - metrics: dict[str, Any], 单类别单样本后处理评估结果
    """
    # VoxelPredCacheData, 从磁盘读取的缓存数据
    data = load_voxel_prediction_cache(cache_path)
    return evaluate_loaded_cached_sample_for_class_with_postprocess(
        cache_path=cache_path,
        data=data,
        class_name=class_name,
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


def evaluate_postprocess_params_on_cache_set_for_class(
    cache_paths: list[str],
    loaded_cache_items: list[tuple[str, VoxelPredCacheData]],
    cache_data_mode: str,
    class_name: str,
    postprocess_params: dict[str, Any],
    eval_params: dict[str, Any],
    n_jobs: int,
) -> dict[str, Any]:
    """
    对整套缓存的单个前景类别执行后处理参数评估。

    输入参数:
        - cache_paths: list[str], 可变长度, 缓存路径列表; disk 模式使用
        - loaded_cache_items: list[tuple[str, VoxelPredCacheData]], 可变长度, 已加载缓存; memory 模式使用
        - cache_data_mode: str, 缓存读取模式, 可选 disk/memory
        - class_name: str, 当前评估的前景类别名
        - postprocess_params: dict[str, Any], 后处理参数
        - eval_params: dict[str, Any], 评估参数, 包含 alpha/beta
        - n_jobs: int, joblib 并行 worker 数

    输出:
        - summarys: dict[str, Any], 单类别整套缓存评估汇总, 包含 avg_* 指标和 class_name
    """
    if cache_data_mode == "memory":
        if len(loaded_cache_items) == 0:
            raise ValueError("memory 模式下 loaded_cache_items 不能为空")
        # list[dict[str, Any]], 每个缓存样本在当前类别下的评估结果
        per_sample = Parallel(n_jobs=int(n_jobs), prefer="threads")(
            delayed(evaluate_loaded_cached_sample_for_class_with_postprocess)(cache_path, data, class_name, postprocess_params, eval_params)
            for cache_path, data in loaded_cache_items
        )
    elif cache_data_mode == "disk":
        if len(cache_paths) == 0:
            raise ValueError("disk 模式下 cache_paths 不能为空")
        # list[dict[str, Any]], 每个缓存样本在当前类别下的评估结果
        per_sample = Parallel(n_jobs=int(n_jobs))(
            delayed(evaluate_cached_sample_for_class_with_postprocess)(cache_path, class_name, postprocess_params, eval_params)
            for cache_path in cache_paths
        )
    else:
        raise ValueError(f"未知 cache_data_mode: {cache_data_mode}")
    summarys = _summarize_postprocess_metrics(per_sample, postprocess_params)
    summarys["class_name"] = class_name
    return summarys


def _macro_average_best_metrics(best_metrics_by_class: dict[str, dict[str, Any]]) -> dict[str, float]:
    """
    对逐类最优指标做 macro 平均。

    输入参数:
        - best_metrics_by_class: dict[str, dict[str, Any]], 类别名到该类 best_metrics 的映射

    输出:
        - macro_metrics: dict[str, float], 各 avg_* 指标和 objective_score 的跨类别平均值
    """
    if len(best_metrics_by_class) == 0:
        raise ValueError("best_metrics_by_class 不能为空")
    # set[str], 所有类别共有的数值指标名
    metric_names: set[str] = set()
    for class_metrics in best_metrics_by_class.values():
        for key, value in class_metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                metric_names.add(str(key))
    # dict[str, float], 跨类别简单平均后的 macro 指标
    macro_metrics: dict[str, float] = {}
    for metric_name in sorted(metric_names):
        values = [float(class_metrics[metric_name]) for class_metrics in best_metrics_by_class.values() if metric_name in class_metrics]
        if len(values) > 0:
            macro_metrics[metric_name] = float(np.mean(values))
    return macro_metrics


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










def _optimize_postprocess_params_for_class(
    cache_paths: list[str],
    loaded_cache_items: list[tuple[str, VoxelPredCacheData]],
    cache_data_mode: str,
    class_name: str,
    fixed_postprocess_params: dict[str, Any],
    search_space: dict[str, Any],
    search_strategy: str,
    eval_params: dict[str, Any],
    optimizer_params: dict[str, Any],
    n_jobs: int,
    show_progress: bool,
) -> dict[str, Any]:
    """
    对单个前景类别独立搜索后处理参数。

    输入参数:
        - cache_paths: list[str], 可变长度, 缓存路径列表; disk 模式下用于逐轮加载
        - loaded_cache_items: list[tuple[str, VoxelPredCacheData]], 可变长度, memory 模式下预加载缓存
        - cache_data_mode: str, 缓存数据读取模式, 可选 disk/memory
        - class_name: str, 当前独立搜索的前景类别名
        - fixed_postprocess_params: dict[str, Any], 固定后处理参数
        - search_space: dict[str, Any], 当前类别的参数搜索空间
        - search_strategy: str, 搜索策略, 可选 grid/differential_evolution
        - eval_params: dict[str, Any], 评估参数, 包含 alpha/beta
        - optimizer_params: dict[str, Any], 优化器参数, 包含 objective_expr/fixed_search_params
        - n_jobs: int, 并行 worker 数
        - show_progress: bool, 是否显示参数组合搜索进度条

    输出:
        - result: dict[str, Any], 当前类别参数搜索结果, 包含 best_params/best_metrics/history
    """
    # list[str], 被强制固定的后处理参数名; 同名 search_space 条目会被忽略
    fixed_search_param_names = [str(name) for name in optimizer_params["fixed_search_params"]]
    # dict[str, Any], 当前类别实际参与搜索的参数空间
    active_search_space = {
        str(name): spec
        for name, spec in search_space.items()
        if str(name) not in fixed_search_param_names
    }
    # list[dict[str, Any]], 当前类别按搜索轨迹累积的评估汇总
    history: list[dict[str, Any]] = []

    def evaluate_params(search_params: dict[str, Any]) -> dict[str, Any]:
        """
        评估当前类别的一组搜索参数。

        输入参数:
            - search_params: dict[str, Any], 当前类别本轮搜索给出的参数子集

        输出:
            - summarys: dict[str, Any], 当前类别在该参数下的整套缓存评估汇总
        """
        # dict[str, Any], 去掉固定参数后的搜索参数
        active_search_params = {
            str(name): value
            for name, value in search_params.items()
            if str(name) not in fixed_search_param_names
        }
        # dict[str, Any], 当前类别完整后处理参数
        params = dict(fixed_postprocess_params)
        params.update(active_search_params)
        for name in fixed_search_param_names:
            params[name] = fixed_postprocess_params[name]
        summarys = evaluate_postprocess_params_on_cache_set_for_class(
            cache_paths=cache_paths,
            loaded_cache_items=loaded_cache_items,
            cache_data_mode=cache_data_mode,
            class_name=class_name,
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
        # list[dict[str, Any]], 当前类别的离散参数组合
        grid_params = generate_param_grid(active_search_space)
        # str, tqdm 显示用的当前类别搜索名称
        grid_desc = f"voxel param grid [{class_name}]"
        grid_iter = tqdm(grid_params, total=len(grid_params), desc=grid_desc) if show_progress else grid_params
        for search_params in grid_iter:
            summarys = evaluate_params(search_params)
            if best_summary is None or float(summarys["objective_score"]) > float(best_summary["objective_score"]):
                best_summary = summarys
        if best_summary is None:
            raise RuntimeError(f"类别 {class_name} 的 grid 搜索未产生任何结果")
        return {
            "best_params": best_summary["postprocess_params"],
            "best_metrics": {k: v for k, v in best_summary.items() if k.startswith("avg_") or k == "objective_score"},
            "history": history,
        }
    # list[str], 当前类别 differential_evolution 的搜索参数名列表
    search_names = [str(name) for name in active_search_space.keys()]
    # list[tuple[float, float]], 当前类别 differential_evolution 的连续边界
    bounds = _build_de_bounds(search_names, active_search_space)

    def objective(vector: np.ndarray) -> float:
        """
        将 differential_evolution 向量解码后返回负 objective 分数。

        输入参数:
            - vector: np.ndarray, (P,), 当前优化器参数向量

        输出:
            - score: float, 负的 objective_score, 供 scipy 最小化
        """
        # dict[str, Any], 当前优化器向量解码后的搜索参数
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
    # dict[str, Any], differential_evolution 最终向量对应的搜索参数
    best_search_params = _decode_de_vector(de_result.x, search_names, active_search_space)
    best_summary = evaluate_params(best_search_params)
    return {
        "best_params": best_summary["postprocess_params"],
        "best_metrics": {k: v for k, v in best_summary.items() if k.startswith("avg_") or k == "objective_score"},
        "history": history,
        "optimizer_fun": float(de_result.fun),
    }


def optimize_postprocess_params_by_class(
    cache_paths: list[str],
    loaded_cache_items: list[tuple[str, VoxelPredCacheData]],
    cache_data_mode: str,
    fixed_postprocess_params: dict[str, Any],
    search_space: dict[str, Any],
    search_space_by_class: dict[str, dict[str, Any]],
    search_strategy: str,
    eval_params: dict[str, Any],
    optimizer_params: dict[str, Any],
    n_jobs: int,
    show_progress: bool,
) -> dict[str, Any]:
    """
    对多分类缓存的每个前景类别独立搜索后处理参数。

    输入参数:
        - cache_paths: list[str], 可变长度, 缓存路径列表; disk 模式下用于逐轮加载
        - loaded_cache_items: list[tuple[str, VoxelPredCacheData]], 可变长度, memory 模式下预加载缓存
        - cache_data_mode: str, 缓存数据读取模式, 可选 disk/memory
        - fixed_postprocess_params: dict[str, Any], 固定后处理参数
        - search_space: dict[str, Any], 全局参数搜索空间
        - search_space_by_class: dict[str, dict[str, Any]], 类别名到局部搜索空间覆盖项的映射
        - search_strategy: str, 搜索策略, 可选 grid/differential_evolution
        - eval_params: dict[str, Any], 评估参数, 包含 alpha/beta
        - optimizer_params: dict[str, Any], 优化器参数, 包含 objective_expr/fixed_search_params
        - n_jobs: int, 并行 worker 数
        - show_progress: bool, 是否显示参数组合搜索进度条

    输出:
        - result: dict[str, Any], 逐类参数搜索结果, 包含 by_class best_params、by_class/macro best_metrics 和 by_class history
    """
    if cache_data_mode == "memory":
        if len(loaded_cache_items) == 0:
            raise ValueError("memory 模式下 loaded_cache_items 不能为空")
        # VoxelPredCacheData, 用于解析类别名的第一条缓存
        first_data = loaded_cache_items[0][1]
    elif cache_data_mode == "disk":
        if len(cache_paths) == 0:
            raise ValueError("disk 模式下 cache_paths 不能为空")
        first_data = load_voxel_prediction_cache(cache_paths[0])
    else:
        raise ValueError(f"未知 cache_data_mode: {cache_data_mode}")
    # list[str], 可变长度, 不含 background 的前景类别名
    class_names = get_foreground_class_names_from_cache(first_data)
    if class_names == ["foreground"]:
        raise ValueError("optimize_postprocess_params_by_class 只接受多分类缓存")

    # dict[str, dict[str, Any]], class_name -> 该类完整 best params
    best_params_by_class: dict[str, dict[str, Any]] = {}
    # dict[str, dict[str, Any]], class_name -> 该类 best metrics
    best_metrics_by_class: dict[str, dict[str, Any]] = {}
    # dict[str, list[dict[str, Any]]], class_name -> 该类搜索历史
    history_by_class: dict[str, list[dict[str, Any]]] = {}
    for class_name in class_names:
        # dict[str, Any], 当前类别的搜索空间, 由全局 search_space 加类别覆盖项得到
        class_search_space = copy.deepcopy(search_space)
        class_search_space.update(copy.deepcopy(search_space_by_class.get(class_name, {})))
        class_result = _optimize_postprocess_params_for_class(
            cache_paths=cache_paths,
            loaded_cache_items=loaded_cache_items,
            cache_data_mode=cache_data_mode,
            class_name=class_name,
            fixed_postprocess_params=fixed_postprocess_params,
            search_space=class_search_space,
            search_strategy=search_strategy,
            eval_params=eval_params,
            optimizer_params=optimizer_params,
            n_jobs=n_jobs,
            show_progress=show_progress,
        )
        best_params_by_class[class_name] = class_result["best_params"]
        best_metrics_by_class[class_name] = class_result["best_metrics"]
        history_by_class[class_name] = class_result["history"]

    return {
        "best_params": {"by_class": best_params_by_class},
        "best_metrics": {
            "by_class": best_metrics_by_class,
            "macro": _macro_average_best_metrics(best_metrics_by_class),
        },
        "history": {"by_class": history_by_class},
    }


def write_best_by_class_csv(
    best_metrics_by_class: dict[str, dict[str, Any]],
    best_params_by_class: dict[str, dict[str, Any]],
    output_root: str,
) -> str:
    """
    写出多分类逐类最优参数与指标的 CSV 汇总。

    输入参数:
        - best_metrics_by_class: dict[str, dict[str, Any]], 类别名到该类 best metrics 的映射
        - best_params_by_class: dict[str, dict[str, Any]], 类别名到该类 best params 的映射
        - output_root: str, 输出目录

    输出:
        - csv_path: str, 写出的 best_by_class_summary.csv 路径
    """
    os.makedirs(output_root, exist_ok=True)
    # str, 逐类最优结果 CSV 输出路径
    csv_path = os.path.join(output_root, "best_by_class_summary.csv")
    # list[str], CSV 列名, 便于人工快速查看各类别最优结果
    fieldnames = [
        "class_name",
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
        "best_params_json",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for class_name, best_metrics in best_metrics_by_class.items():
            # dict[str, Any], 当前类别一行 CSV 内容
            row = {"class_name": class_name}
            for field_name in fieldnames[1:-1]:
                row[field_name] = best_metrics.get(field_name, "")
            row["best_params_json"] = json.dumps(best_params_by_class[class_name], ensure_ascii=False, sort_keys=True, default=_json_default)
            writer.writerow(row)
    return csv_path


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


