"""在 calibration scores 上冻结 Selector 的 CLG 门控阈值 tau_G。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf

from src.artifacts import Stage1ArtifactPaths, atomic_write_json, load_npz_strict
from src.evaluation import (
    DEFAULT_COVERAGE_THRESHOLDS,
    InstanceCounts,
    aggregate_instance_counts,
    evaluate_instance_overlap_counts,
)

from .structured.antichain_dp import build_candidate_tree_closure, exact_antichain_map


@dataclass(frozen=True)
class _PdbCalibrationTable:
    """
    保存一个 PDB 已解码 MAP candidates 的稠密 instance 计数基础表。

    输入参数:
        - pdb_id: str, 当前 PDB 身份
        - gate_probability: np.ndarray, (N_prediction,), 每个去重预测 candidate 所属 CLG 的 p_G
        - pred_sizes: np.ndarray, (N_prediction,), 每个预测 candidate 的 voxel 数
        - intersections: np.ndarray, (N_prediction,N_gt), candidate 与 GT occurrence 的交集 voxel 数
        - gt_sizes: np.ndarray, (N_gt,), 每个 GT occurrence 的 voxel 数
    """

    pdb_id: str
    gate_probability: np.ndarray
    pred_sizes: np.ndarray
    intersections: np.ndarray
    gt_sizes: np.ndarray


def _selected_candidate_rows(
    scores: dict[str, np.ndarray],
    forest: dict[str, np.ndarray],
    clg: dict[str, np.ndarray],
    lambda_count: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    为全部 CLG 各解一次非空预测 MAP，并返回绝对 candidate 行及其 p_G。

    输入参数:
        - scores: dict[str,np.ndarray], 与 clg.npz 同序的 scores.npz
        - forest: dict[str,np.ndarray], 当前 PDB forest.npz
        - clg: dict[str,np.ndarray], 当前 PDB clg.npz
        - lambda_count: float, 预测反链节点计数惩罚

    输出:
        - candidate_rows: np.ndarray, (N_pred,), clg candidate value 表绝对行
        - gate_probability: np.ndarray, (N_pred,), 每个预测所属 CLG 的 p_G
    """
    if not np.array_equal(scores["CLG_id"], clg["CLG_id"]):
        raise ValueError("calibration scores.CLG_id 必须与来源 clg.npz 完全同序。")
    if not np.array_equal(scores["candidate_offsets"], clg["candidate_offsets"]):
        raise ValueError("calibration scores.candidate_offsets 必须逐元素复制来源 clg.npz。")
    probabilities = np.asarray(scores["CLG_valid_probability"], dtype=np.float32)
    if probabilities.shape != np.asarray(clg["CLG_id"]).shape or not np.isfinite(probabilities).all():
        raise ValueError("CLG_valid_probability 必须与 CLG 对齐且全部有限。")

    forest_tree_id = np.asarray(forest["tree_id"], dtype=np.int64)
    forest_node_id = np.asarray(forest["node_id"], dtype=np.int64)
    forest_parent_id = np.asarray(forest["parent_node_id"], dtype=np.int64)
    candidate_offsets = np.asarray(clg["candidate_offsets"], dtype=np.int64)
    candidate_node_id_all = np.asarray(clg["candidate_node_id"], dtype=np.int64)
    selection_logit = np.asarray(scores["selection_logit"], dtype=np.float32)
    if selection_logit.shape != candidate_node_id_all.shape or not np.isfinite(selection_logit).all():
        raise ValueError("selection_logit 必须与 candidate value 表对齐且全部有限。")

    selected_rows: list[int] = []
    selected_probabilities: list[float] = []
    for clg_row in range(probabilities.size):
        begin, end = int(candidate_offsets[clg_row]), int(candidate_offsets[clg_row + 1])
        candidate_node_id = candidate_node_id_all[begin:end]
        tree_rows = np.flatnonzero(forest_tree_id == int(clg["tree_id"][clg_row]))
        closure = build_candidate_tree_closure(
            node_id=forest_node_id[tree_rows].tolist(),
            parent_node_id=forest_parent_id[tree_rows].tolist(),
            candidate_node_id=candidate_node_id.tolist(),
        )
        _, selected_local = exact_antichain_map(
            selection_score=torch.from_numpy(selection_logit[begin:end]),
            parent_index=closure.parent_index,
            candidate_index_by_node=closure.candidate_index_by_node,
            lambda_count=float(lambda_count),
        )
        for local_index in sorted(int(value) for value in selected_local.tolist()):
            selected_rows.append(begin + local_index)
            selected_probabilities.append(float(probabilities[clg_row]))
    return (
        np.asarray(selected_rows, dtype=np.int64),
        np.asarray(selected_probabilities, dtype=np.float32),
    )


def _build_pdb_calibration_table(
    pdb_id: str,
    scores_path: Path,
    paths: Stage1ArtifactPaths,
    lambda_count: float,
) -> _PdbCalibrationTable:
    """
    从 scores/forest/clg/overlap 构造一个 PDB 的 MAP candidate×GT 交集表。

    输入参数:
        - pdb_id: str, 当前 PDB 身份
        - scores_path: Path, 当前 Selector run 的 PDB scores.npz
        - paths: Stage1ArtifactPaths, 当前 PDB forest、CLG 与 overlap artifact 路径
        - lambda_count: float, 预测反链的 candidate 计数惩罚

    输出:
        - table: _PdbCalibrationTable, 按去重 candidate 行组织的校准输入表
    """
    scores = load_npz_strict(scores_path)
    forest = load_npz_strict(paths.forest_npz)
    clg = load_npz_strict(paths.clg_npz)
    overlap = load_npz_strict(paths.overlap_npz)
    candidate_rows, gate_probability = _selected_candidate_rows(
        scores=scores,
        forest=forest,
        clg=clg,
        lambda_count=lambda_count,
    )

    occurrence_sizes = np.asarray(overlap["occurrence_voxel_count"], dtype=np.int64)
    overlap_offsets = np.asarray(overlap["candidate_occurrence_offsets"], dtype=np.int64)
    overlap_occurrence = np.asarray(overlap["overlap_occurrence_index"], dtype=np.int64)
    overlap_count = np.asarray(overlap["intersection_voxel_count"], dtype=np.int64)
    candidate_count = int(np.asarray(clg["candidate_node_id"]).size)
    if overlap_offsets.shape != (candidate_count + 1,):
        raise ValueError("candidate_occurrence_offsets 必须逐 candidate 切分 overlap value 表。")
    if overlap_occurrence.shape != overlap_count.shape:
        raise ValueError("overlap_occurrence_index 与 intersection_voxel_count 必须逐行对齐。")

    forest_rows = {
        (int(tree_id), int(node_id)): row
        for row, (tree_id, node_id) in enumerate(
            zip(forest["tree_id"], forest["node_id"], strict=True)
        )
    }
    clg_offsets = np.asarray(clg["candidate_offsets"], dtype=np.int64)
    clg_row_by_candidate = np.searchsorted(clg_offsets[1:], candidate_rows, side="right")
    unique_rows: list[int] = []
    unique_gate_probability: list[float] = []
    unique_clg_rows: list[int] = []
    prediction_by_identity: dict[tuple[int, int], int] = {}
    for candidate_row, clg_row, probability in zip(
        candidate_rows.tolist(),
        clg_row_by_candidate.tolist(),
        gate_probability.tolist(),
        strict=True,
    ):
        identity = (
            int(clg["tree_id"][clg_row]),
            int(clg["candidate_node_id"][candidate_row]),
        )
        existing = prediction_by_identity.get(identity)
        if existing is not None:
            unique_gate_probability[existing] = max(
                unique_gate_probability[existing], float(probability)
            )
            continue
        prediction_by_identity[identity] = len(unique_rows)
        unique_rows.append(int(candidate_row))
        unique_clg_rows.append(int(clg_row))
        unique_gate_probability.append(float(probability))

    pred_sizes = np.empty(len(unique_rows), dtype=np.int64)
    intersections = np.zeros((len(unique_rows), occurrence_sizes.size), dtype=np.int64)
    for prediction_row, (candidate_row, clg_row) in enumerate(
        zip(unique_rows, unique_clg_rows, strict=True)
    ):
        identity = (
            int(clg["tree_id"][clg_row]),
            int(clg["candidate_node_id"][candidate_row]),
        )
        forest_row = forest_rows[identity]
        pred_sizes[prediction_row] = int(forest["voxel_count"][forest_row])
        begin = int(overlap_offsets[candidate_row])
        end = int(overlap_offsets[candidate_row + 1])
        occurrence_rows = overlap_occurrence[begin:end]
        if occurrence_rows.size and (
            int(occurrence_rows.min()) < 0 or int(occurrence_rows.max()) >= occurrence_sizes.size
        ):
            raise ValueError("overlap_occurrence_index 越过 occurrence_voxel_count 本地表。")
        np.add.at(intersections[prediction_row], occurrence_rows, overlap_count[begin:end])

    return _PdbCalibrationTable(
        pdb_id=str(pdb_id),
        gate_probability=np.asarray(unique_gate_probability, dtype=np.float32),
        pred_sizes=pred_sizes,
        intersections=intersections,
        gt_sizes=occurrence_sizes,
    )


def _metric_summary(counts: InstanceCounts) -> dict[str, float | int]:
    """
    从正式 InstanceCounts 提取完整 global 指标并计算四项均值 M_instance。

    输入参数:
        - counts: InstanceCounts, 已聚合的 instance overlap 计数

    输出:
        - metrics: dict[str,float|int], `InstanceCounts.metrics()` 字段及 `M_instance`
    """
    metrics = counts.metrics()
    component_names = (
        "coverage_f1_0p3",
        "coverage_f1_0p5",
        "one_to_one_f1_0p3",
        "one_to_one_f1_0p5",
    )
    metrics["M_instance"] = float(np.mean([float(metrics[name]) for name in component_names]))
    return metrics


def _evaluate_tau(
    tables: Sequence[_PdbCalibrationTable],
    tau_g: float,
) -> tuple[dict[str, float | int], dict[str, float]]:
    """
    在一个 tau_G 下计算全体 PDB global/micro 指标和逐 PDB macro 诊断。

    输入参数:
        - tables: Sequence[_PdbCalibrationTable], 各 PDB 已解码 candidate 计数表
        - tau_g: float, CLG 门控阈值

    输出:
        - result: tuple[dict[str,float|int],dict[str,float]], 依次为 global 指标和逐 PDB macro 指标
    """
    per_pdb_counts = []
    per_pdb_metrics: list[dict[str, float | int]] = []
    for table in tables:
        selected = table.gate_probability >= float(tau_g)
        counts = evaluate_instance_overlap_counts(
            intersections=table.intersections[selected],
            pred_sizes=table.pred_sizes[selected],
            gt_sizes=table.gt_sizes,
            coverage_thresholds=DEFAULT_COVERAGE_THRESHOLDS,
        )
        per_pdb_counts.append(counts)
        per_pdb_metrics.append(_metric_summary(counts))
    global_metrics = _metric_summary(aggregate_instance_counts(per_pdb_counts))
    macro_names = (
        "coverage_f1_0p3",
        "coverage_f1_0p5",
        "one_to_one_f1_0p3",
        "one_to_one_f1_0p5",
        "M_instance",
    )
    macro_metrics = {
        name: float(np.mean([float(metrics[name]) for metrics in per_pdb_metrics]))
        for name in macro_names
    }
    return global_metrics, macro_metrics


def calibrate_tau_g(
    selector_run_dir: str | Path,
    stage1_outputs_root: str | Path,
    input_clg_list_path: str | Path,
    stage1_model_name: str,
    split: str = "calibration",
) -> dict[str, Any]:
    """
    扫描实际 p_G，按 global/micro M_instance 的首个最大值冻结 tau_G。

    输入参数:
        - selector_run_dir: str | Path, 当前 Selector run 独立输出目录
        - stage1_outputs_root: str | Path, Stage1 producer 正式输出根
        - input_clg_list_path: str | Path, 当前 run 冻结的 input_CLG_list.json
        - stage1_model_name: str, `STAGE1_MODEL_NAMES` 中的 producer 身份
        - split: str, 校正 split；正式为 calibration

    输出:
        - payload: dict[str,Any], 同时原子发布为 run 根目录 calibration.json
    """
    frozen = json.loads(Path(input_clg_list_path).read_text(encoding="utf-8"))
    if frozen.get("stage1_model_name") != stage1_model_name:
        raise ValueError("input_CLG_list.json 与请求的 stage1_model_name 不一致。")
    pdb_ids_by_split = frozen.get("pdb_ids_by_split")
    if not isinstance(pdb_ids_by_split, dict) or split not in pdb_ids_by_split:
        raise ValueError("input_CLG_list.json 缺少 calibration split 的 PDB inventory。")
    pdb_ids = [str(value) for value in pdb_ids_by_split[split]]
    if not pdb_ids:
        raise ValueError(f"冻结清单中没有 split={split!r} 的 Selector calibration PDB。")

    run_dir = Path(selector_run_dir)
    resolved_config_path = run_dir / "resolved_config.yaml"
    if not resolved_config_path.is_file():
        raise FileNotFoundError(
            f"Selector calibration 必须读取训练 run 的 resolved config: {resolved_config_path}"
        )
    resolved_config = OmegaConf.to_container(
        OmegaConf.load(resolved_config_path), resolve=True
    )
    if not isinstance(resolved_config, dict):
        raise TypeError("Selector resolved_config.yaml 必须解析为 mapping。")
    if str(resolved_config.get("stage1_model_name")) != str(stage1_model_name):
        raise ValueError("Selector resolved config 与 calibration producer 不一致。")
    lambda_count = float(resolved_config["data"]["lambda_count"])
    tables = tuple(
        _build_pdb_calibration_table(
            pdb_id=pdb_id,
            scores_path=run_dir / split / pdb_id / "scores.npz",
            paths=Stage1ArtifactPaths(
                output_root=Path(stage1_outputs_root),
                stage1_model_name=stage1_model_name,
                split=split,
                pdb_id=pdb_id,
            ),
            lambda_count=lambda_count,
        )
        for pdb_id in pdb_ids
    )
    all_probabilities = np.concatenate([table.gate_probability for table in tables])
    if all_probabilities.size == 0 or not np.isfinite(all_probabilities).all():
        raise ValueError("calibration 没有可扫描的有限 CLG_valid_probability。")
    tau_values = np.unique(all_probabilities.astype(np.float32, copy=False))

    curve: list[dict[str, Any]] = []
    for tau_g in tau_values.tolist():
        global_metrics, macro_metrics = _evaluate_tau(tables, float(tau_g))
        curve.append(
            {
                "tau_G": float(tau_g),
                "global": global_metrics,
                "macro_diagnostic": macro_metrics,
            }
        )
    # tau_values 已升序；np.argmax 的首项行为即项目约定的 first maximum。
    best_index = int(np.argmax(np.asarray([row["global"]["M_instance"] for row in curve])))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "stage1_model_name": str(stage1_model_name),
        "split": str(split),
        "calibration_fitted": True,
        "lambda_count": float(lambda_count),
        "coverage_thresholds": list(DEFAULT_COVERAGE_THRESHOLDS),
        "scan_definition": "ascending_unique_actual_CLG_valid_probability",
        "pdb_count": len(tables),
        "tau_G": float(curve[best_index]["tau_G"]),
        "best_curve_index": best_index,
        "metrics": curve[best_index]["global"],
        "macro_diagnostic": curve[best_index]["macro_diagnostic"],
        "curve": curve,
    }
    atomic_write_json(run_dir / "calibration.json", payload)
    return payload
