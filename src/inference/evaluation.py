# -*- coding: utf-8 -*-
"""计算 Stage1 候选与真实 ligand occurrence 的语义和实例指标.

主要入口 :func:`evaluate_centered_pdb` 生成逐 PDB 交集矩阵和计数,
:func:`aggregate_stage1_metrics` 生成跨 PDB micro 与 PDB 等权 macro 报告.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class PdbEvaluation:
    """保存一个 PDB 的候选交集事实和各指标计数.

    ``intersections`` 是 int64 ``(N_pred,N_gt)``, 行对应按 score 稳定降序
    排列且被 ``selected`` 保留的候选, 列对应 occurrence identity 升序.
    coverage 命中允许多对多; one-to-one 使用一次连续双向覆盖 Hungarian 配对.
    """

    pdb_id: str
    occurrence_id: np.ndarray
    source_blob_index: np.ndarray
    candidate_score: np.ndarray
    intersections: np.ndarray
    pred_sizes: np.ndarray
    gt_sizes: np.ndarray
    semantic_tp: int
    semantic_fp: int
    semantic_fn: int
    coverage_pred_hit: np.ndarray
    coverage_gt_hit: np.ndarray
    one_to_one_tp: np.ndarray
    topk_success: np.ndarray


# ================================================================================================


def evaluation_arrays(evaluation: PdbEvaluation) -> dict[str, np.ndarray]:
    """把一个逐 PDB 评估结果转换为可直接发布的 NPZ 字段."""

    return {
        "occurrence_id": evaluation.occurrence_id.astype(np.int32, copy=False),
        "source_blob_index": evaluation.source_blob_index.astype(np.int32, copy=False),
        "candidate_score": evaluation.candidate_score.astype(np.float32, copy=False),
        "intersections": evaluation.intersections.astype(np.int64, copy=False),
        "pred_sizes": evaluation.pred_sizes.astype(np.int64, copy=False),
        "gt_sizes": evaluation.gt_sizes.astype(np.int64, copy=False),
        "semantic_tp": np.asarray(evaluation.semantic_tp, dtype=np.int64),
        "semantic_fp": np.asarray(evaluation.semantic_fp, dtype=np.int64),
        "semantic_fn": np.asarray(evaluation.semantic_fn, dtype=np.int64),
        "coverage_pred_hit": evaluation.coverage_pred_hit.astype(np.int64, copy=False),
        "coverage_gt_hit": evaluation.coverage_gt_hit.astype(np.int64, copy=False),
        "one_to_one_tp": evaluation.one_to_one_tp.astype(np.int64, copy=False),
        "topk_success": evaluation.topk_success.astype(np.int64, copy=False),
    }


def load_occurrence_voxels(
    ligand_area_path: str | Path,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[int, int, int]]:
    """读取 schema-v3 ``ligand_area.npz`` 的稀疏 occurrence ZYX 坐标.

    返回的 ``occurrence_id`` 为 int32 ``(N_gt,)``; ``voxel_rows`` 与其逐项
    对齐, 每项为 int32 ``(K_gt,3)``; ``full_shape_zyx`` 是完整图 ZYX 形状.
    """

    path = Path(ligand_area_path)
    with np.load(path, allow_pickle=False) as archive:
        shape = tuple(int(value) for value in archive["grid_shape_zyx"])
        names = sorted(
            (
                name
                for name in archive.files
                if name.startswith("mask_") and name[5:].isdigit()
            ),
            key=lambda name: int(name[5:]),
        )
        occurrence_id = np.asarray([int(name[5:]) for name in names], dtype=np.int32)
        rows = tuple(np.asarray(archive[name], dtype=np.int32) for name in names)
    return occurrence_id, rows, shape


def evaluate_centered_pdb(
    pdb_id: str,
    centered: Mapping[str, np.ndarray],
    occurrence_id: np.ndarray,
    occurrence_voxel_zyx: Sequence[np.ndarray],
    full_shape_zyx: Sequence[int],
    coverage_thresholds: Sequence[float],
    topk_values: Sequence[int],
) -> PdbEvaluation:
    """从 centered 稀疏体素和 occurrence 稀疏体素计算一个 PDB 的事实表."""

    selected = np.flatnonzero(np.asarray(centered["selected"], dtype=np.bool_))
    score = np.asarray(centered["score"], dtype=np.float64)[selected]
    order = np.argsort(-score, kind="stable")
    selected = selected[order]
    score = score[order]
    voxel_offsets = np.asarray(centered["voxel_offsets"], dtype=np.int64)
    local_rows = np.asarray(centered["voxel_index_local_zyx"], dtype=np.int64)
    box_starts = np.asarray(centered["box_start_zyx"], dtype=np.int64)
    shape = tuple(int(value) for value in full_shape_zyx)

    pred_linear: list[np.ndarray] = []
    for entry_index in selected.tolist():
        begin = int(voxel_offsets[entry_index])
        end = int(voxel_offsets[entry_index + 1])
        global_zyx = local_rows[begin:end] + box_starts[entry_index][None, :]
        pred_linear.append(np.ravel_multi_index(global_zyx.T, shape))
    gt_linear = [
        np.ravel_multi_index(np.asarray(rows, dtype=np.int64).T, shape)
        for rows in occurrence_voxel_zyx
    ]
    intersections = np.asarray(
        [
            [np.intersect1d(pred, gt, assume_unique=False).size for gt in gt_linear]
            for pred in pred_linear
        ],
        dtype=np.int64,
    ).reshape(len(pred_linear), len(gt_linear))
    pred_sizes = np.asarray([rows.size for rows in pred_linear], dtype=np.int64)
    gt_sizes = np.asarray([rows.size for rows in gt_linear], dtype=np.int64)

    pred_union = np.unique(np.concatenate(pred_linear)) if pred_linear else np.empty(0, dtype=np.int64)
    gt_union = np.unique(np.concatenate(gt_linear)) if gt_linear else np.empty(0, dtype=np.int64)
    semantic_tp = int(np.intersect1d(pred_union, gt_union, assume_unique=True).size)
    semantic_fp = int(pred_union.size - semantic_tp)
    semantic_fn = int(gt_union.size - semantic_tp)

    pred_cover = np.divide(
        intersections,
        pred_sizes[:, None],
        out=np.zeros(intersections.shape, dtype=np.float64),
        where=pred_sizes[:, None] > 0,
    )
    gt_cover = np.divide(
        intersections,
        gt_sizes[None, :],
        out=np.zeros(intersections.shape, dtype=np.float64),
        where=gt_sizes[None, :] > 0,
    )
    thresholds = tuple(float(value) for value in coverage_thresholds)
    coverage_pred_hit = np.zeros(len(thresholds), dtype=np.int64)
    coverage_gt_hit = np.zeros(len(thresholds), dtype=np.int64)
    one_to_one_tp = np.zeros(len(thresholds), dtype=np.int64)
    if pred_linear and gt_linear:
        matched_pred, matched_gt = linear_sum_assignment(-np.sqrt(pred_cover * gt_cover))
        for row, threshold in enumerate(thresholds):
            valid = (pred_cover >= threshold) & (gt_cover >= threshold)
            coverage_pred_hit[row] = int(valid.any(axis=1).sum())
            coverage_gt_hit[row] = int(valid.any(axis=0).sum())
            one_to_one_tp[row] = int(
                (
                    (pred_cover[matched_pred, matched_gt] >= threshold)
                    & (gt_cover[matched_pred, matched_gt] >= threshold)
                ).sum()
            )

    topk_success = np.zeros((len(topk_values), len(thresholds)), dtype=np.int64)
    for topk_row, topk in enumerate(topk_values):
        for threshold_row, threshold in enumerate(thresholds):
            valid = (
                (pred_cover[: int(topk)] >= threshold)
                & (gt_cover[: int(topk)] >= threshold)
            )
            topk_success[topk_row, threshold_row] = int(bool(valid.any()))
    return PdbEvaluation(
        pdb_id=str(pdb_id).lower(),
        occurrence_id=np.asarray(occurrence_id, dtype=np.int32),
        source_blob_index=np.asarray(centered["source_blob_index"], dtype=np.int32)[selected],
        candidate_score=score.astype(np.float32),
        intersections=intersections,
        pred_sizes=pred_sizes,
        gt_sizes=gt_sizes,
        semantic_tp=semantic_tp,
        semantic_fp=semantic_fp,
        semantic_fn=semantic_fn,
        coverage_pred_hit=coverage_pred_hit,
        coverage_gt_hit=coverage_gt_hit,
        one_to_one_tp=one_to_one_tp,
        topk_success=topk_success,
    )


def aggregate_stage1_metrics(
    evaluations: Sequence[PdbEvaluation],
    coverage_thresholds: Sequence[float],
    topk_values: Sequence[int],
) -> dict[str, object]:
    """汇总语义 F1/F2、coverage、one-to-one 与 top-K micro/macro 指标."""

    thresholds = tuple(float(value) for value in coverage_thresholds)
    semantic = np.asarray(
        [
            [item.semantic_tp, item.semantic_fp, item.semantic_fn]
            for item in evaluations
        ],
        dtype=np.int64,
    )
    if semantic.size == 0:
        semantic = np.zeros((0, 3), dtype=np.int64)
    total_tp, total_fp, total_fn = semantic.sum(axis=0, dtype=np.int64).tolist()
    semantic_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    semantic_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    report: dict[str, object] = {
        "pdb_count": len(evaluations),
        "semantic_tp": int(total_tp),
        "semantic_fp": int(total_fp),
        "semantic_fn": int(total_fn),
    }
    for beta in (1.0, 2.0):
        beta2 = beta * beta
        denominator = beta2 * semantic_precision + semantic_recall
        report[f"semantic_micro_f{int(beta)}"] = (
            (1.0 + beta2) * semantic_precision * semantic_recall / denominator
            if denominator
            else 0.0
        )

    pred_total = sum(int(item.pred_sizes.size) for item in evaluations)
    gt_total = sum(int(item.gt_sizes.size) for item in evaluations)
    coverage_pred = (
        np.sum(np.stack([item.coverage_pred_hit for item in evaluations]), axis=0)
        if evaluations
        else np.zeros(len(thresholds), dtype=np.int64)
    )
    coverage_gt = (
        np.sum(np.stack([item.coverage_gt_hit for item in evaluations]), axis=0)
        if evaluations
        else np.zeros(len(thresholds), dtype=np.int64)
    )
    one_to_one = (
        np.sum(np.stack([item.one_to_one_tp for item in evaluations]), axis=0)
        if evaluations
        else np.zeros(len(thresholds), dtype=np.int64)
    )
    for row, threshold in enumerate(thresholds):
        tag = f"{threshold:.3f}".rstrip("0").rstrip(".").replace(".", "p")
        for name, pred_hit, gt_hit in (
            ("coverage", int(coverage_pred[row]), int(coverage_gt[row])),
            ("one_to_one", int(one_to_one[row]), int(one_to_one[row])),
        ):
            precision = pred_hit / pred_total if pred_total else 0.0
            recall = gt_hit / gt_total if gt_total else 0.0
            report[f"{name}_micro_precision_{tag}"] = precision
            report[f"{name}_micro_recall_{tag}"] = recall
            for beta in (1.0, 2.0):
                beta2 = beta * beta
                denominator = beta2 * precision + recall
                report[f"{name}_micro_f{int(beta)}_{tag}"] = (
                    (1.0 + beta2) * precision * recall / denominator
                    if denominator
                    else 0.0
                )

            per_pdb_f1: list[float] = []
            per_pdb_f2: list[float] = []
            for item in evaluations:
                if name == "coverage":
                    local_pred_hit = int(item.coverage_pred_hit[row])
                    local_gt_hit = int(item.coverage_gt_hit[row])
                else:
                    local_pred_hit = local_gt_hit = int(item.one_to_one_tp[row])
                local_precision = local_pred_hit / item.pred_sizes.size if item.pred_sizes.size else 0.0
                local_recall = local_gt_hit / item.gt_sizes.size if item.gt_sizes.size else 0.0
                for beta, target in ((1.0, per_pdb_f1), (2.0, per_pdb_f2)):
                    beta2 = beta * beta
                    denominator = beta2 * local_precision + local_recall
                    target.append(
                        (1.0 + beta2) * local_precision * local_recall / denominator
                        if denominator
                        else 0.0
                    )
            report[f"{name}_macro_f1_{tag}"] = float(np.mean(per_pdb_f1)) if per_pdb_f1 else 0.0
            report[f"{name}_macro_f2_{tag}"] = float(np.mean(per_pdb_f2)) if per_pdb_f2 else 0.0

    eligible_pdb = sum(int(item.gt_sizes.size > 0) for item in evaluations)
    report["topk_eligible_pdb_count"] = eligible_pdb
    for topk_row, topk in enumerate(topk_values):
        successes = (
            np.sum(
                np.stack([item.topk_success[topk_row] for item in evaluations]),
                axis=0,
            )
            if evaluations
            else np.zeros(len(thresholds), dtype=np.int64)
        )
        for threshold_row, threshold in enumerate(thresholds):
            tag = f"{threshold:.3f}".rstrip("0").rstrip(".").replace(".", "p")
            report[f"top{int(topk)}_success_count_{tag}"] = int(successes[threshold_row])
            report[f"top{int(topk)}_success_ratio_{tag}"] = (
                float(successes[threshold_row]) / eligible_pdb if eligible_pdb else 0.0
            )
    return report
