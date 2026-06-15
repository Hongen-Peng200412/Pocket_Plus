from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[3]
project_root = str(PROJECT_ROOT)
if project_root in sys.path:
    sys.path.remove(project_root)
sys.path.insert(0, project_root)

from processedPDB_EMDB_binder.utils.mrc_tools import load_map
from src.inference.parse_input import load_from_raw_cif
from src.inference.utils.yield_json_from_raw_sample import load_raw_pairs
from src.inference.voxel_evaluator import (
    DEFAULT_COVERAGE_THRESHOLDS,
    DEFAULT_TOPK_VALUES,
    evaluate_global_instance_matching,
    evaluate_topk_success,
    evaluate_voxel_mask,
    evaluate_voxel_pr_auc,
)
from src.inference.voxel_gt import load_ligand_gt_from_labels_npz
from src.inference.voxel_postprocess import _build_candidates, _filter_small_components


# ============================ 加载、对齐逻辑 ============================ 
def _read_mrc_grid(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    读取 MRC/MAP 文件并返回 Pocket Plus 约定的网格几何。

    输入参数:
        - path: str 或 Path, MRC/MAP 文件路径

    输出:
        - result: tuple[np.ndarray, np.ndarray, np.ndarray], 包含:
            - grid: np.ndarray, (D,H,W), 体素网格
            - voxel_size: np.ndarray, (3,), 体素大小(x,y,z)
            - origin: np.ndarray, (3,), 世界坐标原点(x,y,z)
    """
    grid, voxel_size, origin = load_map(str(path))
    return (
        np.asarray(grid),
        np.asarray(voxel_size, dtype=np.float32).reshape(3),
        np.asarray(origin, dtype=np.float32).reshape(3),
    )


def _same_grid(
    source_shape: tuple[int, int, int],
    source_origin: np.ndarray,
    source_voxel: np.ndarray,
    target_shape: tuple[int, int, int],
    target_origin: np.ndarray,
    target_voxel: np.ndarray,
) -> bool:
    """
    判断两个体素网格是否可直接逐元素对齐。

    输入参数:
        - source_shape: tuple[int,int,int], 源网格形状(D,H,W)
        - source_origin: np.ndarray, (3,), 源网格世界坐标原点(x,y,z)
        - source_voxel: np.ndarray, (3,), 源网格体素大小(x,y,z)
        - target_shape: tuple[int,int,int], 目标网格形状(D,H,W)
        - target_origin: np.ndarray, (3,), 目标网格世界坐标原点(x,y,z)
        - target_voxel: np.ndarray, (3,), 目标网格体素大小(x,y,z)

    输出:
        - is_same: bool, shape/origin/voxel_size 是否一致
    """
    return (
        tuple(int(v) for v in source_shape) == tuple(int(v) for v in target_shape)
        and np.allclose(source_origin, target_origin, rtol=1e-5, atol=1e-4)
        and np.allclose(source_voxel, target_voxel, rtol=1e-5, atol=1e-6)
    )


def _map_to_target_grid(
    source_grid: np.ndarray,
    source_origin: np.ndarray,
    source_voxel: np.ndarray,
    target_shape: tuple[int, int, int],
    target_origin: np.ndarray,
    target_voxel: np.ndarray,
    order: int,
    chunk_depth: int,
) -> np.ndarray:
    """
    将源网格按世界坐标映射到目标网格。

    输入参数:
        - source_grid: np.ndarray, (D,H,W), 源网格
        - source_origin: np.ndarray, (3,), 源网格世界坐标原点(x,y,z)
        - source_voxel: np.ndarray, (3,), 源网格体素大小(x,y,z)
        - target_shape: tuple[int,int,int], 目标网格形状(D,H,W)
        - target_origin: np.ndarray, (3,), 目标网格世界坐标原点(x,y,z)
        - target_voxel: np.ndarray, (3,), 目标网格体素大小(x,y,z)
        - order: int, scipy.ndimage.map_coordinates 插值阶数; mask/label 用 0, score 用 1
        - chunk_depth: int, z 轴分块深度, 用于降低峰值内存

    输出:
        - mapped: np.ndarray, (D,H,W), 映射到目标网格后的数组
    """
    target_depth, target_height, target_width = tuple(int(v) for v in target_shape)
    mapped = np.zeros(target_shape, dtype=np.float32)
    y_idx, x_idx = np.mgrid[0:target_height, 0:target_width]
    for z_start in range(0, target_depth, int(chunk_depth)):
        z_end = min(target_depth, z_start + int(chunk_depth))
        z_idx = np.arange(z_start, z_end, dtype=np.float32)[:, None, None]
        src_z = (target_origin[2] + (z_idx + 0.5) * target_voxel[2] - source_origin[2]) / source_voxel[2] - 0.5
        src_y = (target_origin[1] + (y_idx[None, :, :] + 0.5) * target_voxel[1] - source_origin[1]) / source_voxel[1] - 0.5
        src_x = (target_origin[0] + (x_idx[None, :, :] + 0.5) * target_voxel[0] - source_origin[0]) / source_voxel[0] - 0.5
        # np.ndarray, (Z,H,W), float32, 目标体素中心在源网格中的 z/y/x 浮点索引
        coords = np.broadcast_arrays(src_z, src_y, src_x)
        mapped[z_start:z_end] = ndimage.map_coordinates(
            np.asarray(source_grid, dtype=np.float32),
            coords,
            order=int(order),
            mode="constant",
            cval=0.0,
        )
    return mapped


def _build_eval_grid(pair: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """
    构建 Pocket Plus 标准评估网格和 labels.npz GT。

    输入参数:
        - pair: dict[str, Any], 单个样本配置, 需要 cif_gt_path / map_path / labels_npz_path
        - args: argparse.Namespace, CLI 参数集合

    输出:
        - result: dict[str, Any], 包含:
            - "origin": np.ndarray, (3,), 评估网格世界坐标原点(x,y,z)
            - "voxel_size": np.ndarray, (3,), 评估网格体素大小(x,y,z)
            - "shape": tuple[int,int,int], 评估网格形状(D,H,W)
            - "hardmask": np.ndarray, (D,H,W), 原子占据硬掩码
            - "gt_ligand_mask": np.ndarray, (D,H,W), union GT ligand mask
            - "gt_instance_label": np.ndarray, (D,H,W), union GT instance 标签
    """
    raw_data = load_from_raw_cif(
        cif_path=str(pair["cif_gt_path"]),
        map_path=str(pair["map_path"]),
        sim_map_path=None,
        target_voxel_size=float(args.target_voxel_size),
        compute_density=False,
        select_first_model=True,
        error_dir=None,
        density_channel_config={
            "clip_percentile": (0.001, 0.999),
            "fit_mask_percentile": 0.003,
            "enabled_channels": ["exp_clipnorm_nopost"],
        },
    )
    gt_data = load_ligand_gt_from_labels_npz(
        labels_npz_path=str(pair["labels_npz_path"]),
        origin=raw_data["origin"],
        voxel_size=raw_data["voxel_size"],
        grid_shape_zyx=tuple(int(v) for v in raw_data["full_shape_zyx"]),
        class_mapping=None,
        ligand_gt_distance_threshold=float(args.ligand_gt_distance_threshold),
    )
    return {
        "origin": raw_data["origin"],
        "voxel_size": raw_data["voxel_size"],
        "shape": tuple(int(v) for v in raw_data["full_shape_zyx"]),
        "hardmask": raw_data["hardmask"],
        "gt_ligand_mask": gt_data["gt_ligand_mask"],
        "gt_instance_label": gt_data["gt_instance_label"],
    }





# ============================  评估逻辑 ============================ 
def _empty_metrics(eval_grid: dict[str, Any]) -> dict[str, Any]:
    """
    按空预测构造失败样本指标。

    输入参数:
        - eval_grid: dict[str, Any], 评估网格和 GT 数据

    输出:
        - metrics: dict[str, Any], 空预测在当前 GT 上的完整评估指标
    """
    empty_mask = np.zeros(eval_grid["shape"], dtype=bool)
    empty_label = np.zeros(eval_grid["shape"], dtype=np.int32)
    metrics = _evaluate_arrays(
        pred_mask=empty_mask,
        pred_instance_label=empty_label,
        score_map=np.zeros(eval_grid["shape"], dtype=np.float32),
        eval_grid=eval_grid,
        candidates=[],
    )
    metrics["pr_auc"] = 0.0
    return metrics


def _evaluate_arrays(
    pred_mask: np.ndarray,
    pred_instance_label: np.ndarray,
    score_map: np.ndarray,
    eval_grid: dict[str, Any],
    candidates: list[Any],
) -> dict[str, Any]:
    """
    调用 Pocket Plus 现有 metric 函数评估单样本预测。

    输入参数:
        - pred_mask: np.ndarray, (D,H,W), bool, 预测 ligand mask
        - pred_instance_label: np.ndarray, (D,H,W), int32, 预测 instance 标签
        - score_map: np.ndarray, (D,H,W), float32, 官方 score map
        - eval_grid: dict[str, Any], 评估网格和 GT 数据
        - candidates: list[Any], 可变长度, 与 pred_instance_label 对齐的预测的候选实例

    输出:
        - metrics: dict[str, Any], 单样本评估指标
    """
    metrics: dict[str, Any] = {}
    metrics.update(evaluate_voxel_mask(pred_mask, eval_grid["gt_ligand_mask"]))
    metrics.update(
        evaluate_global_instance_matching(
            pred_instance_label=pred_instance_label,
            gt_instance_label=eval_grid["gt_instance_label"],
            coverage_thresholds=DEFAULT_COVERAGE_THRESHOLDS,
        )
    )
    metrics.update(
        evaluate_topk_success(
            pred_instance_label=pred_instance_label,
            gt_instance_label=eval_grid["gt_instance_label"],
            candidates=candidates,
            topk_values=DEFAULT_TOPK_VALUES,
            coverage_thresholds=DEFAULT_COVERAGE_THRESHOLDS,
        )
    )
    metrics.update(
        evaluate_voxel_pr_auc(
            score_map=score_map,
            gt_ligand_mask=eval_grid["gt_ligand_mask"],
            hardmask=eval_grid["hardmask"],
        )
    )
    metrics["num_candidates"] = int(len(candidates))
    return metrics


def evaluate_one_sample(pair: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """
    评估单个 Emap2lig-Find 官方输出样本。

    输入参数:
        - pair: dict[str, Any], 单个样本路径字典, 使用以下 key:
            - "sample_name": str, 可选, 样本名; 缺失时退回 cif_path 的 stem
            - "cif_path": str, 样本 cif 路径; 仅在 sample_name 缺失时用于取 stem
            - "cif_gt_path": str, GT 结构 cif 路径, 用于构建评估网格
            - "map_path": str, 密度图路径, 用于构建评估网格
            - "labels_npz_path": str, labels.npz 路径, 用于构建 voxel/instance GT
        - args: argparse.Namespace, CLI 参数集合, 包括: target_voxel_size、ligand_gt_distance_threshold 等

    输出:
        - row: dict[str, Any], 单样本状态和指标摘要, 包含:
            - "sample_name": str, 样本名
            - "status": str, 样本状态, "ok" 或 "failed_empty_prediction"
            - "error": str | None, 失败原因; 成功时为 None
            - "metrics": dict[str, Any], 单样本评估指标
            - "grid_alignment": dict[str, Any], 仅 status == "ok" 时存在, 网格对齐信息, 包含:
                - "mask_direct": bool, mask 是否与评估网格直接逐元素对齐
                - "score_direct": bool, score map 是否与评估网格直接逐元素对齐
                - "emap_mask_shape": tuple[int,int,int], Emap2lig mask 原始形状(D,H,W)
                - "eval_shape": tuple[int,int,int], 评估网格形状(D,H,W)
    """
    sample_name = str(pair.get("sample_name") or Path(str(pair["cif_path"])).stem)
    try:
        sample_out = Path(args.output_root) / args.dataset_name / "pocketplus_eval_inputs" / sample_name
        sample_out.mkdir(parents=True, exist_ok=True)
        eval_grid = _build_eval_grid(pair, args)

        find_dir = Path(args.raw_find_root) / sample_name
        status_path = find_dir / "status.json"
        find_status = {"status": "missing_status", "error": "status.json not found"}
        if status_path.exists():
            with open(status_path, "r", encoding="utf-8") as file_obj:
                find_status = json.load(file_obj)

        if find_status.get("status") != "ok":
            print(
                f"[evaluate_emap2lig_find] sample={sample_name} 使用空预测回退: "
                f"Find status={find_status.get('status')}, error={find_status.get('error')}",
                flush=True,
            )
            metrics = _empty_metrics(eval_grid)
            row = {
                "sample_name": sample_name,
                "status": "failed_empty_prediction",
                "error": find_status.get("error"),
                "metrics": metrics,
            }
        else:
            # 0/1 掩码
            mask_grid, mask_voxel, mask_origin = _read_mrc_grid(find_dir / "find_maps" / "ligand_mask.mrc")
            # 每个体素的原始 ligand 概率
            score_grid, score_voxel, score_origin = _read_mrc_grid(find_dir / "find_maps" / "ligand.mrc")
            same_mask_grid = _same_grid(mask_grid.shape, mask_origin, mask_voxel, eval_grid["shape"], eval_grid["origin"], eval_grid["voxel_size"])
            same_score_grid = _same_grid(score_grid.shape, score_origin, score_voxel, eval_grid["shape"], eval_grid["origin"], eval_grid["voxel_size"])
            if same_mask_grid:
                pred_mask = np.asarray(mask_grid > 0, dtype=bool)
            else:
                # 注意, 这是 0/1 的, 后面按照原版做27-连通域分析
                pred_mask = _map_to_target_grid(
                    source_grid=mask_grid,
                    source_origin=mask_origin,
                    source_voxel=mask_voxel,
                    target_shape=eval_grid["shape"],
                    target_origin=eval_grid["origin"],
                    target_voxel=eval_grid["voxel_size"],
                    order=0,
                    chunk_depth=int(args.chunk_depth),
                ) > 0.5
            if same_score_grid:
                score_map = np.asarray(score_grid, dtype=np.float32)
            else:
                score_map = _map_to_target_grid(
                    source_grid=score_grid,
                    source_origin=score_origin,
                    source_voxel=score_voxel,
                    target_shape=eval_grid["shape"],
                    target_origin=eval_grid["origin"],
                    target_voxel=eval_grid["voxel_size"],
                    order=1,
                    chunk_depth=int(args.chunk_depth),
                ).astype(np.float32, copy=False)

            pred_instance_label, _ = ndimage.label(pred_mask, structure=ndimage.generate_binary_structure(3, 3))
            pred_instance_label = _filter_small_components(pred_instance_label.astype(np.int32, copy=False), 32)  # 就是官方 <32 体素过滤(emap2lig main.py:356)
            # 对齐到 eval_grid(我们自己的坐标), 没问题 
            pred_mask = pred_instance_label > 0
            candidates = _build_candidates(pred_instance_label, score_map, eval_grid["origin"], eval_grid["voxel_size"])
            metrics = _evaluate_arrays(pred_mask, pred_instance_label, score_map, eval_grid, candidates)
            np.savez_compressed(sample_out / "instance_label.npz", instance_label=pred_instance_label)
            np.savez_compressed(sample_out / "score_map.npz", score_map=score_map.astype(np.float32))
            _write_json(sample_out / "voxel_candidates.json", [asdict(candidate) for candidate in candidates])
            row = {
                "sample_name": sample_name,
                "status": "ok",
                "error": None,
                "metrics": metrics,
                "grid_alignment": {
                    "mask_direct": bool(same_mask_grid),
                    "score_direct": bool(same_score_grid),
                    "emap_mask_shape": tuple(int(v) for v in mask_grid.shape),  # Find 的 shape
                    "eval_shape": eval_grid["shape"],  # Pocket Plus 的 shape
                },
            }
    except Exception as exc:
        print(
            f"[evaluate_emap2lig_find] sample={sample_name} 评估异常, 使用可记录失败状态: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        row = {
            "sample_name": sample_name,
            "status": "failed_empty_prediction",
            "error": f"{type(exc).__name__}: {exc}",
        }
        if "eval_grid" in locals():
            row["metrics"] = _empty_metrics(eval_grid)

    if "sample_out" not in locals():
        sample_out = Path(args.output_root) / args.dataset_name / "pocketplus_eval_inputs" / sample_name
        sample_out.mkdir(parents=True, exist_ok=True)
    if "metrics" in row:
        _write_json(sample_out / "metrics.json", row["metrics"])
    _write_json(sample_out / "status.json", {k: v for k, v in row.items() if k != "metrics"})
    return row




# ===================================== 工具函数 ===================================== 
def _json_default(value: Any) -> Any:
    """
    将 numpy/path/dataclass 对象转成 JSON 可序列化对象。

    输入参数:
        - value: Any, 待序列化对象

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


def _write_json(path: str | Path, data: Any) -> None:
    """
    写出 UTF-8 JSON。

    输入参数:
        - path: str 或 Path, 输出路径
        - data: Any, 可 JSON 序列化对象

    输出:
        - None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, ensure_ascii=False, indent=2, default=_json_default)


def _safe_div(numerator: float, denominator: float) -> float:
    """
    安全除法。

    输入参数:
        - numerator: float, 分子
        - denominator: float, 分母

    输出:
        - value: float, 分母为 0 时返回 0.0
    """
    if float(denominator) == 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    汇总单样本指标。

    输入参数:
        - rows: list[dict[str, Any]], 每项包含 metrics 字段

    输出:
        - summary: dict[str, Any], 数据集级汇总指标
    """
    metrics_rows = [row["metrics"] for row in rows if "metrics" in row]
    summary: dict[str, Any] = {
        "num_samples": int(len(rows)),
        "num_ok": int(sum(1 for row in rows if row.get("status") == "ok")),
        "num_failed": int(sum(1 for row in rows if row.get("status") != "ok")),
        "num_with_metrics": int(len(metrics_rows)),
        "num_without_metrics": int(sum(1 for row in rows if "metrics" not in row)),
    }
    if summary["num_without_metrics"] > 0:
        print(
            f"[evaluate_emap2lig_find] 汇总发现 {summary['num_without_metrics']} 个样本缺少 metrics, "
            "通常表示评估网格或 GT 构建失败; 这些样本不会参与可计算指标聚合。",
            flush=True,
        )
    if not metrics_rows:
        return summary
    for key in ("voxel_precision", "voxel_recall", "voxel_f1", "voxel_iou", "voxel_dice", "num_candidates"):
        values = [float(item[key]) for item in metrics_rows if key in item]
        summary[f"avg_{key}"] = float(np.mean(values)) if values else 0.0
    pr_auc_values = [float(item["pr_auc"]) for item in metrics_rows if item.get("pr_auc") is not None]
    summary["pr_auc_macro"] = float(np.mean(pr_auc_values)) if pr_auc_values else None

    sum_num_pred = sum(int(item["num_pred_instances"]) for item in metrics_rows)
    sum_num_gt = sum(int(item["num_gt_instances"]) for item in metrics_rows)
    for threshold in DEFAULT_COVERAGE_THRESHOLDS:
        tag = f"{int(round(float(threshold) * 10)):02d}"
        sum_tp = sum(int(item[f"tp_cov{tag}"]) for item in metrics_rows)
        precision = _safe_div(sum_tp, sum_num_pred)
        recall = _safe_div(sum_tp, sum_num_gt)
        summary[f"global_instance_precision_cov{tag}"] = precision
        summary[f"global_instance_recall_cov{tag}"] = recall
        summary[f"global_instance_f1_cov{tag}"] = _safe_div(2.0 * precision * recall, precision + recall)
        if f"tp_pred_loose_cov{tag}" in metrics_rows[0]:
            sum_tp_pred_loose = sum(int(item[f"tp_pred_loose_cov{tag}"]) for item in metrics_rows)
            sum_tp_gt_loose = sum(int(item[f"tp_gt_loose_cov{tag}"]) for item in metrics_rows)
            precision_loose = _safe_div(sum_tp_pred_loose, sum_num_pred)
            recall_loose = _safe_div(sum_tp_gt_loose, sum_num_gt)
            summary[f"global_instance_precision_loose_cov{tag}"] = precision_loose
            summary[f"global_instance_recall_loose_cov{tag}"] = recall_loose
            summary[f"global_instance_f1_loose_cov{tag}"] = _safe_div(2.0 * precision_loose * recall_loose, precision_loose + recall_loose)
        for topk in DEFAULT_TOPK_VALUES:
            key = f"top{int(topk)}_success_cov{tag}"
            summary[key.replace("_success_cov", "_success_ratio_cov")] = _safe_div(
                sum(int(item[key]) for item in metrics_rows),
                len(metrics_rows),
            )
    return summary


def write_summary_csv(summary: dict[str, Any], path: str | Path) -> None:
    """
    将 summary 写成单行 CSV。

    输入参数:
        - summary: dict[str, Any], 数据集级指标
        - path: str 或 Path, CSV 输出路径

    输出:
        - None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def run_dataset(args: argparse.Namespace) -> dict[str, Any]:
    """
    评估一个数据集的 Emap2lig-Find 官方输出。

    输入参数:
        - args: argparse.Namespace, CLI 参数集合

    输出:
        - summary: dict[str, Any], 数据集级汇总指标
    """
    pairs = load_raw_pairs(str(args.raw_pairs_json))
    eval_n_jobs = int(args.eval_n_jobs)
    if eval_n_jobs <= 1:
        rows = [evaluate_one_sample(pair, args) for pair in pairs]
    else:
        from joblib import Parallel, delayed

        print(
            f"[evaluate_emap2lig_find] dataset={args.dataset_name} "
            f"samples={len(pairs)} eval_n_jobs={eval_n_jobs} eval_backend={args.eval_backend}",
            flush=True,
        )
        rows = Parallel(n_jobs=eval_n_jobs, backend=str(args.eval_backend))(
            delayed(evaluate_one_sample)(pair, args) for pair in pairs
        )
    reports_root = Path(args.output_root) / args.dataset_name / "reports"
    summary = summarize(rows)
    _write_json(reports_root / "per_sample_results.json", rows)
    _write_json(reports_root / "summary.json", summary)
    write_summary_csv(summary, reports_root / "summary.csv")
    return summary


def parse_args() -> argparse.Namespace: 
    """
    解析命令行参数。

    输出:
        - args: argparse.Namespace, 运行参数
    """
    parser = argparse.ArgumentParser(description="评估 Emap2lig-Find 官方输出")
    parser.add_argument("--raw-pairs-json", required=True, help="Pocket Plus 样本列表 JSON")
    parser.add_argument("--dataset-name", required=True, choices=["protein_110", "nucleic_40"], help="数据集名称")
    parser.add_argument("--raw-find-root", required=True, help="Emap2lig raw_find_outputs 目录")
    parser.add_argument("--output-root", required=True, help="评估输出根目录")
    parser.add_argument("--target-voxel-size", type=float, default=1.0, help="Pocket Plus 评估网格目标体素大小")
    parser.add_argument("--ligand-gt-distance-threshold", type=float, default=1.7, help="labels.npz 构造 voxel GT 的距离阈值")
    parser.add_argument("--chunk-depth", type=int, default=256, help="坐标映射时的 z 轴分块深度")
    parser.add_argument("--eval-n-jobs", type=int, default=1, help="样本级并行 worker 数; 1 表示串行")
    parser.add_argument("--eval-backend", default="loky", help="joblib backend; 默认 loky")
    return parser.parse_args()


if __name__ == "__main__":
    run_dataset(parse_args())
