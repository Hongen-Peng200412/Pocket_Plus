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
    """保存一个 PDB 的候选, 真实 occurrence 和指标身份事实.

    字段:
        - pdb_id: str, 小写 PDB identity.
        - occurrence_id: int32 ``(N_gt,)``, 真实 occurrence identity.
        - source_blob_index: int32 ``(N_pred,)``, 按最终 score 稳定降序的来源 blob identity.
        - candidate_score: float32 ``(N_pred,)``, 与候选行对齐的冻结分数.
        - candidate_selected: bool ``(N_pred,)``, 是否达到冻结分数与体素数门槛.
        - intersections: int64 ``(N_pred, N_gt)``, 候选与 occurrence 的体素交集数.
        - pred_sizes: int64 ``(N_pred,)``, 每个候选的稀疏体素数.
        - gt_sizes: int64 ``(N_gt,)``, 每个真实 occurrence 的稀疏体素数.
        - candidate_semantic_tp: int64 ``(N_pred,)``, 每个候选与真实并集的交集数.
        - semantic_tp: int, 所有已选候选并集与真实并集的交集体素数.
        - semantic_fp: int, 已选候选并集落在真实并集外的体素数.
        - semantic_fn: int, 真实并集未被已选候选并集覆盖的体素数.
        - coverage_thresholds: float32 ``(N_threshold,)``, 评估阈值轴.
        - topk_values: int32 ``(N_topk,)``, top-K 轴.
        - coverage_pred_hit_mask: bool ``(N_threshold, N_pred)``, 多对多预测侧命中.
        - coverage_gt_hit_mask: bool ``(N_threshold, N_gt)``, 多对多真实侧命中.
        - one_to_one_match_offsets: int64 ``(N_threshold+1,)``, 同时切分 `one_to_one_match_pred_index` 和 `one_to_one_match_gt_index`.
        - coverage_pred_hit: int64 ``(N_threshold,)``, 每个阈值命中的候选数.
        - coverage_gt_hit: int64 ``(N_threshold,)``, 每个阈值命中的 occurrence 数.
        - one_to_one_match_pred_index: int32 ``(L_match,)``, 各阈值最大匹配的候选列身份值表.
        - one_to_one_match_gt_index: int32 ``(L_match,)``, 与前项对齐的 occurrence 列身份值表.
        - one_to_one_tp: int64 ``(N_threshold,)``, 每个阈值的一对一匹配数.
        - topk_success: bool ``(N_topk, N_threshold)``, 每组 top-K 与覆盖阈值是否至少命中一个 occurrence.
        - topk_winning_candidate_rank: int32 ``(N_topk, N_threshold)``, 首个获胜候选名次, 未命中为 ``-1``.
        - topk_winning_occurrence_index: int32 同形状, 对应 occurrence 列号, 未命中为 ``-1``.

    所有候选轴都与 ``source_blob_index`` 对齐; 阈值轴与
    ``coverage_thresholds`` 对齐, top-K 轴与 ``topk_values`` 对齐.
    """

    pdb_id: str
    occurrence_id: np.ndarray
    source_blob_index: np.ndarray
    candidate_score: np.ndarray
    candidate_selected: np.ndarray
    intersections: np.ndarray
    pred_sizes: np.ndarray
    gt_sizes: np.ndarray
    candidate_semantic_tp: np.ndarray
    semantic_tp: int
    semantic_fp: int
    semantic_fn: int
    coverage_thresholds: np.ndarray
    topk_values: np.ndarray
    coverage_pred_hit_mask: np.ndarray
    coverage_gt_hit_mask: np.ndarray
    coverage_pred_hit: np.ndarray
    coverage_gt_hit: np.ndarray
    one_to_one_match_offsets: np.ndarray
    one_to_one_match_pred_index: np.ndarray
    one_to_one_match_gt_index: np.ndarray
    one_to_one_tp: np.ndarray
    topk_success: np.ndarray
    topk_winning_candidate_rank: np.ndarray
    topk_winning_occurrence_index: np.ndarray


def load_occurrence_voxels(
    ligand_area_path: str | Path,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[int, int, int]]:
    """读取 schema-v3 ``ligand_area.npz`` 的稀疏 occurrence ZYX 坐标.

    返回的 ``occurrence_id`` 为 int32 ``(N_gt,)``; ``voxel_rows`` 与其逐项
    对齐, 每项为 int32 ``(K_gt, 3)``; ``full_shape_zyx`` 是完整图 ZYX 形状.
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


# ================================================================================================


def evaluate_centered_pdb(
    pdb_id: str,
    centered: Mapping[str, np.ndarray],
    occurrence_id: np.ndarray,
    occurrence_voxel_zyx: Sequence[np.ndarray],
    full_shape_zyx: Sequence[int],
    coverage_thresholds: Sequence[float],
    topk_values: Sequence[int],
) -> PdbEvaluation:
    """计算一个 PDB 的候选与真实 occurrence 逐项事实.

    ``centered`` 中的 ``selected`` 决定参与评估的候选. 候选按 ``score``
    降序稳定排列. ``intersections`` 为 ``int64 (N_pred, N_gt)``. 双向覆盖
    命中掩码分别为 ``bool (N_threshold, N_pred)`` 和
    ``bool (N_threshold, N_gt)``. 一对一匹配使用 offsets 切分每个覆盖阈值的
    候选行号和 occurrence 列号. top-K 获胜身份使用排序后的候选名次和真实
    occurrence 列号, 没有命中时为 ``-1``.
    """

    score = np.asarray(centered["score"], dtype=np.float64)
    order = np.argsort(-score, kind="stable")
    candidate_selected = np.asarray(centered["selected"], dtype=np.bool_)[order]
    score = score[order]
    voxel_offsets = np.asarray(centered["voxel_offsets"], dtype=np.int64)
    local_rows = np.asarray(centered["voxel_index_local_zyx"], dtype=np.int64)
    box_starts = np.asarray(centered["box_start_zyx"], dtype=np.int64)
    shape = tuple(int(value) for value in full_shape_zyx)

    pred_linear: list[np.ndarray] = []
    for entry_index in order.tolist():
        begin = int(voxel_offsets[entry_index])
        end = int(voxel_offsets[entry_index + 1])
        global_zyx = local_rows[begin:end] + box_starts[entry_index][None, :]
        pred_linear.append(np.unique(np.ravel_multi_index(global_zyx.T, shape)))
    gt_linear = [
        np.unique(np.ravel_multi_index(np.asarray(rows, dtype=np.int64).T, shape))
        for rows in occurrence_voxel_zyx
    ]
    # int64 (N_pred, N_gt), 行列身份由 score 排序候选与 occurrence_id 固定.
    intersections = np.asarray(
        [
            [np.intersect1d(pred, gt, assume_unique=True).size for gt in gt_linear]
            for pred in pred_linear
        ],
        dtype=np.int64,
    ).reshape(len(pred_linear), len(gt_linear))
    pred_sizes = np.asarray([rows.size for rows in pred_linear], dtype=np.int64)
    gt_sizes = np.asarray([rows.size for rows in gt_linear], dtype=np.int64)

    selected_rows = np.flatnonzero(candidate_selected)
    selected_pred_linear = [pred_linear[index] for index in selected_rows.tolist()]
    pred_union = (
        np.unique(np.concatenate(selected_pred_linear))
        if selected_pred_linear
        else np.empty(0, dtype=np.int64)
    )
    gt_union = np.unique(np.concatenate(gt_linear)) if gt_linear else np.empty(0, dtype=np.int64)
    candidate_semantic_tp = np.asarray(
        [np.intersect1d(pred, gt_union, assume_unique=True).size for pred in pred_linear],
        dtype=np.int64,
    )
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
    coverage_pred_hit_mask = np.zeros(
        (len(thresholds), len(pred_linear)), dtype=np.bool_
    )
    coverage_gt_hit_mask = np.zeros(
        (len(thresholds), len(gt_linear)), dtype=np.bool_
    )
    match_pred_rows: list[np.ndarray] = []
    match_gt_rows: list[np.ndarray] = []
    for threshold_row, threshold in enumerate(thresholds):
        # bool (N_pred, N_gt), 两个方向的覆盖率都达到同一阈值才形成有效边.
        valid = (pred_cover >= threshold) & (gt_cover >= threshold)
        coverage_pred_hit_mask[threshold_row] = valid.any(axis=1)
        selected_valid = valid[selected_rows]
        coverage_gt_hit_mask[threshold_row] = selected_valid.any(axis=0)
        if selected_valid.size:
            matched_pred, matched_gt = linear_sum_assignment(
                -selected_valid.astype(np.int8)
            )
            keep = selected_valid[matched_pred, matched_gt]
            match_pred_rows.append(selected_rows[matched_pred[keep]].astype(np.int32))
            match_gt_rows.append(matched_gt[keep].astype(np.int32))
        else:
            match_pred_rows.append(np.empty(0, dtype=np.int32))
            match_gt_rows.append(np.empty(0, dtype=np.int32))
    one_to_one_tp = np.asarray(
        [rows.size for rows in match_pred_rows], dtype=np.int64
    )
    match_offsets = np.concatenate(
        (
            np.zeros(1, dtype=np.int64),
            np.cumsum(one_to_one_tp, dtype=np.int64),
        )
    )
    topk_success = np.zeros((len(topk_values), len(thresholds)), dtype=np.int64)
    topk_winning_candidate_rank = np.full(
        topk_success.shape, -1, dtype=np.int32
    )
    topk_winning_occurrence_index = np.full(
        topk_success.shape, -1, dtype=np.int32
    )
    for topk_row, topk in enumerate(topk_values):
        top_candidate_rows = selected_rows[: int(topk)]
        for threshold_row, threshold in enumerate(thresholds):
            valid = (
                (pred_cover[top_candidate_rows] >= threshold)
                & (gt_cover[top_candidate_rows] >= threshold)
            )
            winners = np.argwhere(valid)
            if winners.size:
                topk_success[topk_row, threshold_row] = 1
                topk_winning_candidate_rank[topk_row, threshold_row] = winners[0, 0]
                topk_winning_occurrence_index[topk_row, threshold_row] = winners[0, 1]
    return PdbEvaluation(
        pdb_id=str(pdb_id).lower(),
        occurrence_id=np.asarray(occurrence_id, dtype=np.int32),
        source_blob_index=np.asarray(centered["source_blob_index"], dtype=np.int32)[order],
        candidate_score=score.astype(np.float32),
        candidate_selected=candidate_selected,
        intersections=intersections,
        pred_sizes=pred_sizes,
        gt_sizes=gt_sizes,
        candidate_semantic_tp=candidate_semantic_tp,
        semantic_tp=semantic_tp,
        semantic_fp=semantic_fp,
        semantic_fn=semantic_fn,
        coverage_thresholds=np.asarray(thresholds, dtype=np.float32),
        topk_values=np.asarray(topk_values, dtype=np.int32),
        coverage_pred_hit_mask=coverage_pred_hit_mask,
        coverage_gt_hit_mask=coverage_gt_hit_mask,
        coverage_pred_hit=coverage_pred_hit_mask[:, selected_rows].sum(
            axis=1, dtype=np.int64
        ),
        coverage_gt_hit=coverage_gt_hit_mask.sum(axis=1, dtype=np.int64),
        one_to_one_match_offsets=match_offsets,
        one_to_one_match_pred_index=(
            np.concatenate(match_pred_rows)
            if match_pred_rows
            else np.empty(0, dtype=np.int32)
        ),
        one_to_one_match_gt_index=(
            np.concatenate(match_gt_rows)
            if match_gt_rows
            else np.empty(0, dtype=np.int32)
        ),
        one_to_one_tp=one_to_one_tp,
        topk_success=topk_success,
        topk_winning_candidate_rank=topk_winning_candidate_rank,
        topk_winning_occurrence_index=topk_winning_occurrence_index,
    )


def aggregate_stage1_metrics(
    evaluations: Sequence[PdbEvaluation],
    coverage_thresholds: Sequence[float],
    topk_values: Sequence[int],
) -> dict[str, object]:
    """汇总语义, coverage, one-to-one 与 top-K 指标.

    输入 ``evaluations`` 中每个 PDB 权重相同. 返回 JSON 可序列化映射, 包含
    语义 F1/F2 的 micro 与 PDB 等权 macro; coverage/one-to-one 在每个显式
    阈值上的 precision, recall, micro F1/F2 与 macro F1/F2; top-3/4/5
    成功数和成功比例. 单个 PDB 或全局分母为零时对应指标为 ``0.0``.
    """

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
        per_pdb = []
        for item in evaluations:
            numerator = (1.0 + beta2) * float(item.semantic_tp)
            local_denominator = (
                numerator + beta2 * float(item.semantic_fn) + float(item.semantic_fp)
            )
            per_pdb.append(numerator / local_denominator if local_denominator else 0.0)
        report[f"semantic_macro_f{int(beta)}"] = (
            float(np.mean(per_pdb)) if per_pdb else 0.0
        )

    pred_total = sum(int(item.candidate_selected.sum()) for item in evaluations)
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
                local_pred_count = int(item.candidate_selected.sum())
                local_precision = local_pred_hit / local_pred_count if local_pred_count else 0.0
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
