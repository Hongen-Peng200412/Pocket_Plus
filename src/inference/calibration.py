# -*- coding: utf-8 -*-
"""冻结 Stage1 V3 的语义阈值和 centered 选择参数.

语义阈值使用完整图直方图. centered 校准先把每个候选与真实 occurrence 的
交集压成小型事实表, 再执行来源均值阈值搜索或 Find Gaussian 三阶段搜索.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .evaluation import PdbEvaluation, evaluate_centered_pdb
from .scoring import build_gaussian_distance_table, sum_gaussian_atom_terms


@dataclass(frozen=True)
class CenteredCalibrationFacts:
    """保存一个 PDB 的小型 centered 校准事实.

    字段:
        - evaluation: PdbEvaluation, 候选行保持 centered 条目顺序的评估事实.
        - source_probability_mean: float32 ``(N_candidate,)``, 来源 blob 平均概率.
        - voxel_count: int64 ``(N_candidate,)``, 每个来源 blob 的体素数.
        - atom_offsets: int64 ``(N_candidate+1,)``, 切分下列两个 A 原子值表.
        - atom_distance: float32 ``(N_atom,)``, A 原子到来源 blob 的最近世界距离; 超过 5 Å 为 ``Inf``.
        - atom_probability: float32 ``(N_atom,)``, 与距离逐项对齐的 A 原子概率.

    非 Find producer 使用一个零 offsets 和两个空值表. 大型 V/A/P 特征和
    48³ 稠密数组不进入本对象.
    """

    evaluation: PdbEvaluation
    source_probability_mean: np.ndarray
    voxel_count: np.ndarray
    atom_offsets: np.ndarray
    atom_distance: np.ndarray
    atom_probability: np.ndarray


# ================================================================================================


def calibrate_semantic_thresholds(
    probability_and_target: Iterable[tuple[np.ndarray, np.ndarray]],
    denominator: int,
    betas: Sequence[float],
) -> dict[str, object]:
    """按 calibration 全集的 micro TP/FP/FN 冻结 F-beta 阈值.

    输入可以是一次性迭代器, 因而调用者能够逐 PDB 解压完整概率图而不在内存
    中保留整个 calibration 集.

    返回字段:
        - denominator: int, 阈值网格分母.
        - positive_voxel_count: int, calibration 全集真实配体体素数.
        - negative_voxel_count: int, calibration 全集真实背景体素数.
        - thresholds.<F-beta>.grid_index: int, 首个最优阈值的整数网格位置.
        - thresholds.<F-beta>.value: float, `grid_index / denominator` 概率阈值.
        - thresholds.<F-beta>.micro_f_beta: float, calibration 全集最优 micro F-beta.
        - thresholds.<F-beta>.tp: int, 最优阈值的 micro TP.
        - thresholds.<F-beta>.fp: int, 最优阈值的 micro FP.
        - thresholds.<F-beta>.fn: int, 最优阈值的 micro FN.
        - scan.denominator: int32 标量, 与顶层分母相同.
        - scan.beta_values: float64 ``(N_beta,)``, 与输入 beta 顺序一致.
        - scan.threshold_grid_index: int32 ``(denominator+1,)``, 阈值整数轴.
        - scan.f_beta_curve: float64 ``(N_beta, denominator+1)``, beta 轴与阈值轴组成的完整曲线.
        - scan.tp: int64 ``(denominator+1,)``, 每个阈值的 micro TP.
        - scan.fp: int64 ``(denominator+1,)``, 每个阈值的 micro FP.
        - scan.fn: int64 ``(denominator+1,)``, 每个阈值的 micro FN.

    JSON 摘要发布前把 ``scan`` 分离为独立 NPZ.
    """

    positive_histogram = np.zeros(int(denominator) + 1, dtype=np.int64)
    negative_histogram = np.zeros(int(denominator) + 1, dtype=np.int64)
    for probability, target in probability_and_target:
        values = np.asarray(probability, dtype=np.float64)
        truth = np.asarray(target, dtype=np.bool_)
        bins = np.floor(np.clip(values, 0.0, 1.0) * int(denominator)).astype(np.int64)
        positive_histogram += np.bincount(
            bins[truth], minlength=int(denominator) + 1
        )
        negative_histogram += np.bincount(
            bins[~truth], minlength=int(denominator) + 1
        )
    tp = np.cumsum(positive_histogram[::-1], dtype=np.int64)[::-1]
    fp = np.cumsum(negative_histogram[::-1], dtype=np.int64)[::-1]
    fn = int(positive_histogram.sum()) - tp
    beta_values = np.asarray(tuple(float(beta) for beta in betas), dtype=np.float64)
    curves = np.empty((beta_values.size, int(denominator) + 1), dtype=np.float64)
    thresholds: dict[str, object] = {}
    for row, beta in enumerate(beta_values):
        beta2 = float(beta) ** 2
        numerator = (1.0 + beta2) * tp.astype(np.float64)
        metric_denominator = numerator + beta2 * fn + fp
        curves[row] = np.divide(
            numerator,
            metric_denominator,
            out=np.zeros_like(numerator),
            where=metric_denominator > 0,
        )
        grid_index = int(np.argmax(curves[row]))
        thresholds[f"F{int(beta)}"] = {
            "grid_index": grid_index,
            "value": float(grid_index) / float(denominator),
            "micro_f_beta": float(curves[row, grid_index]),
            "tp": int(tp[grid_index]),
            "fp": int(fp[grid_index]),
            "fn": int(fn[grid_index]),
        }
    return {
        "denominator": int(denominator),
        "positive_voxel_count": int(positive_histogram.sum()),
        "negative_voxel_count": int(negative_histogram.sum()),
        "thresholds": thresholds,
        "scan": {
            "denominator": np.asarray(denominator, dtype=np.int32),
            "beta_values": beta_values,
            "threshold_grid_index": np.arange(
                int(denominator) + 1, dtype=np.int32
            ),
            "f_beta_curve": curves,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        },
    }


def tune_centered_selection(
    centered_items: Iterable[tuple[str, Mapping[str, np.ndarray]]],
    ground_truth_by_pdb: Mapping[
        str, tuple[np.ndarray, Sequence[np.ndarray], Sequence[int]]
    ],
    score_mode: str,
    score_parameter_grid: Mapping[str, Sequence[float]] | None,
    refinement_multipliers: Mapping[str, Sequence[float]] | None,
    min_voxel_values: Sequence[int],
    objective_beta: float,
    coverage_thresholds: Sequence[float],
    topk_values: Sequence[int],
) -> dict[str, object]:
    """按冻结顺序搜索 centered 分数与最小体素数.

    来源均值模式先在最小 ``min_voxels`` 下扫描实际出现的 float32 分数, 然后
    冻结分数阈值并扫描全部最小体素数. Find Gaussian 模式第一阶段扫描 tau,
    两个 lambda 和 ``gauss_score_min`` 的显式粗网格; 第二阶段固定 tau, 对
    第一阶段三个其余参数应用显式乘数; 第三阶段冻结 Gaussian 参数并只扫描
    ``min_voxels``. 目标是 semantic, coverage@0.3 和 one-to-one@0.3 三个
    micro F-beta 之和. 完全并列时保留配置顺序中先出现的值.

    返回字段:
        - objective: float, 最终最小体素数对应的三项 micro F-beta 之和.
        - objective_beta: float, F1 basic 为 1, F3 centered 为 2.
        - score_mode: str, `source_mean` 或 `find_gaussian`.
        - score_parameters.tau_angstrom: float, Gaussian 距离标准差; 来源均值模式无此字段.
        - score_parameters.lambda_positive: float, Gaussian 正项系数; 来源均值模式无此字段.
        - score_parameters.lambda_negative: float, Gaussian 负项系数; 来源均值模式无此字段.
        - score_threshold: float, 冻结分数下限, 包含端点.
        - min_voxels: int, 冻结的来源 blob 最小体素数, 包含端点.
        - stages.score_threshold.objective: float, 来源均值实际分数扫描的最优目标值.
        - stages.score_threshold.score_threshold: float, 来源均值实际分数扫描的最优阈值.
        - stages.coarse.objective: float, Gaussian 粗网格最优目标值.
        - stages.coarse.tau_angstrom: float, Gaussian 粗网格最优距离标准差.
        - stages.coarse.lambda_positive: float, Gaussian 粗网格最优正项系数.
        - stages.coarse.lambda_negative: float, Gaussian 粗网格最优负项系数.
        - stages.coarse.score_threshold: float, Gaussian 粗网格最优分数阈值.
        - stages.refined.objective: float, Gaussian 细网格最优目标值.
        - stages.refined.tau_angstrom: float, Gaussian 细网格固定距离标准差.
        - stages.refined.lambda_positive: float, Gaussian 细网格最优正项系数.
        - stages.refined.lambda_negative: float, Gaussian 细网格最优负项系数.
        - stages.refined.score_threshold: float, Gaussian 细网格最优分数阈值.
        - stages.min_voxels.objective: float, 最终最小体素数对应的目标值.
        - stages.min_voxels.min_voxels: int, 两种模式最终冻结的最小体素数.
    """

    facts_by_pdb: dict[str, CenteredCalibrationFacts] = {}
    for pdb_id, centered in centered_items:
        candidate_count = np.asarray(centered["source_blob_index"]).size
        all_selected = {name: np.asarray(value) for name, value in centered.items()}
        all_selected["selected"] = np.ones(candidate_count, dtype=np.bool_)
        all_selected["score"] = -np.arange(candidate_count, dtype=np.float32)
        occurrence_id, occurrence_rows, full_shape = ground_truth_by_pdb[pdb_id]
        evaluation = evaluate_centered_pdb(
            pdb_id=pdb_id,
            centered=all_selected,
            occurrence_id=occurrence_id,
            occurrence_voxel_zyx=occurrence_rows,
            full_shape_zyx=full_shape,
            coverage_thresholds=coverage_thresholds,
            topk_values=topk_values,
        )
        if score_mode == "find_gaussian":
            atom_table = build_gaussian_distance_table(centered)
            atom_offsets = atom_table["A_offsets"]
            atom_distance = atom_table["A_distance_to_source"]
            atom_probability = atom_table["A_probability"]
        else:
            atom_offsets = np.zeros(1, dtype=np.int64)
            atom_distance = np.empty(0, dtype=np.float32)
            atom_probability = np.empty(0, dtype=np.float32)
        facts_by_pdb[pdb_id] = CenteredCalibrationFacts(
            evaluation=evaluation,
            source_probability_mean=np.asarray(
                centered["source_probability_mean"], dtype=np.float32
            ),
            voxel_count=np.diff(
                np.asarray(centered["voxel_offsets"], dtype=np.int64)
            ),
            atom_offsets=atom_offsets,
            atom_distance=atom_distance,
            atom_probability=atom_probability,
        )

    initial_min_voxels = min(int(value) for value in min_voxel_values)
    if score_mode == "source_mean":
        scores = {
            pdb_id: facts.source_probability_mean
            for pdb_id, facts in facts_by_pdb.items()
        }
        first_stage = _scan_actual_score_thresholds(
            facts_by_pdb, scores, initial_min_voxels, objective_beta
        )
        score_parameters: dict[str, float] = {}
        score_threshold = float(first_stage["score_threshold"])
        stages: dict[str, object] = {"score_threshold": first_stage}
    else:
        coarse_best: dict[str, object] | None = None
        if score_parameter_grid is None:
            raise ValueError("find_gaussian 缺少 score_parameter_grid.")
        for tau in score_parameter_grid["tau_angstrom"]:
            terms = _gaussian_terms(facts_by_pdb, float(tau))
            for lambda_positive in score_parameter_grid["lambda_positive"]:
                for lambda_negative in score_parameter_grid["lambda_negative"]:
                    scores = {
                        pdb_id: (
                            facts_by_pdb[pdb_id].source_probability_mean
                            + np.float32(lambda_positive) * values[0]
                            - np.float32(lambda_negative) * values[1]
                        )
                        for pdb_id, values in terms.items()
                    }
                    for score_min in score_parameter_grid["gauss_score_min"]:
                        objective = _selection_objective(
                            facts_by_pdb,
                            scores,
                            float(score_min),
                            initial_min_voxels,
                            objective_beta,
                        )
                        if coarse_best is None or objective > float(coarse_best["objective"]):
                            coarse_best = {
                                "objective": objective,
                                "tau_angstrom": float(tau),
                                "lambda_positive": float(lambda_positive),
                                "lambda_negative": float(lambda_negative),
                                "score_threshold": float(score_min),
                            }
        if coarse_best is None or refinement_multipliers is None:
            raise ValueError("find_gaussian 粗网格或第二阶段乘数为空.")
        tau = float(coarse_best["tau_angstrom"])
        terms = _gaussian_terms(facts_by_pdb, tau)
        refined_best: dict[str, object] | None = None
        for positive_multiplier in refinement_multipliers["lambda"]:
            lambda_positive = float(coarse_best["lambda_positive"]) * float(
                positive_multiplier
            )
            for negative_multiplier in refinement_multipliers["lambda"]:
                lambda_negative = float(coarse_best["lambda_negative"]) * float(
                    negative_multiplier
                )
                scores = {
                    pdb_id: (
                        facts_by_pdb[pdb_id].source_probability_mean
                        + np.float32(lambda_positive) * values[0]
                        - np.float32(lambda_negative) * values[1]
                    )
                    for pdb_id, values in terms.items()
                }
                for threshold_multiplier in refinement_multipliers["score_threshold"]:
                    score_threshold = float(coarse_best["score_threshold"]) * float(
                        threshold_multiplier
                    )
                    objective = _selection_objective(
                        facts_by_pdb,
                        scores,
                        score_threshold,
                        initial_min_voxels,
                        objective_beta,
                    )
                    if refined_best is None or objective > float(refined_best["objective"]):
                        refined_best = {
                            "objective": objective,
                            "tau_angstrom": tau,
                            "lambda_positive": lambda_positive,
                            "lambda_negative": lambda_negative,
                            "score_threshold": score_threshold,
                        }
        if refined_best is None:
            raise RuntimeError("find_gaussian 第二阶段没有产生参数组合.")
        score_parameters = {
            "tau_angstrom": float(refined_best["tau_angstrom"]),
            "lambda_positive": float(refined_best["lambda_positive"]),
            "lambda_negative": float(refined_best["lambda_negative"]),
        }
        score_threshold = float(refined_best["score_threshold"])
        scores = {
            pdb_id: (
                facts_by_pdb[pdb_id].source_probability_mean
                + np.float32(score_parameters["lambda_positive"]) * values[0]
                - np.float32(score_parameters["lambda_negative"]) * values[1]
            )
            for pdb_id, values in terms.items()
        }
        stages = {"coarse": coarse_best, "refined": refined_best}

    minimum_best: dict[str, object] | None = None
    for min_voxels in min_voxel_values:
        objective = _selection_objective(
            facts_by_pdb,
            scores,
            score_threshold,
            int(min_voxels),
            objective_beta,
        )
        if minimum_best is None or objective > float(minimum_best["objective"]):
            minimum_best = {
                "objective": objective,
                "min_voxels": int(min_voxels),
            }
    if minimum_best is None:
        raise RuntimeError("min_voxel_values 没有产生参数组合.")
    stages["min_voxels"] = minimum_best
    return {
        "objective": float(minimum_best["objective"]),
        "objective_beta": float(objective_beta),
        "score_mode": score_mode,
        "score_parameters": score_parameters,
        "score_threshold": score_threshold,
        "min_voxels": int(minimum_best["min_voxels"]),
        "stages": stages,
    }


def _gaussian_terms(
    facts_by_pdb: Mapping[str, CenteredCalibrationFacts],
    tau_angstrom: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """为一个 tau 计算各候选的正负 Gaussian 和."""

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for pdb_id, facts in facts_by_pdb.items():
        result[pdb_id] = sum_gaussian_atom_terms(
            facts.atom_offsets,
            facts.atom_distance,
            facts.atom_probability,
            tau_angstrom,
        )
    return result


def _scan_actual_score_thresholds(
    facts_by_pdb: Mapping[str, CenteredCalibrationFacts],
    scores_by_pdb: Mapping[str, np.ndarray],
    min_voxels: int,
    beta: float,
) -> dict[str, float]:
    """按全局实际分数降序增量搜索来源均值阈值.

    每个候选只加入一次. semantic 与 coverage 使用累计计数; one-to-one@0.3
    对当前 PDB 的匹配执行一次增广, 不会对每个阈值重新计算全部交集或 Hungarian.
    """

    rows = [
        (float(score), pdb_id, index)
        for pdb_id, scores in scores_by_pdb.items()
        for index, score in enumerate(scores.tolist())
        if facts_by_pdb[pdb_id].voxel_count[index] >= int(min_voxels)
    ]
    rows.sort(key=lambda item: (-item[0], item[1], item[2]))
    if not rows:
        return {"objective": 0.0, "score_threshold": 0.0}
    semantic_gt = sum(
        facts.evaluation.semantic_tp + facts.evaluation.semantic_fn
        for facts in facts_by_pdb.values()
    )
    gt_count = sum(
        facts.evaluation.gt_sizes.size for facts in facts_by_pdb.values()
    )
    semantic_tp = semantic_fp = selected_count = 0
    coverage_pred_hit = coverage_gt_hit = one_to_one_tp = 0
    adjacency_by_pdb: dict[str, np.ndarray] = {}
    covered_gt_by_pdb: dict[str, np.ndarray] = {}
    matched_gt_by_pdb: dict[str, np.ndarray] = {}
    for pdb_id, facts in facts_by_pdb.items():
        evaluation = facts.evaluation
        pred_cover = np.divide(
            evaluation.intersections,
            evaluation.pred_sizes[:, None],
            out=np.zeros(evaluation.intersections.shape, dtype=np.float64),
            where=evaluation.pred_sizes[:, None] > 0,
        )
        gt_cover = np.divide(
            evaluation.intersections,
            evaluation.gt_sizes[None, :],
            out=np.zeros(evaluation.intersections.shape, dtype=np.float64),
            where=evaluation.gt_sizes[None, :] > 0,
        )
        adjacency_by_pdb[pdb_id] = (pred_cover >= 0.3) & (gt_cover >= 0.3)
        covered_gt_by_pdb[pdb_id] = np.zeros(
            evaluation.gt_sizes.size, dtype=np.bool_
        )
        matched_gt_by_pdb[pdb_id] = np.full(
            evaluation.gt_sizes.size, -1, dtype=np.int32
        )

    empty_threshold = float(
        np.nextafter(np.float32(rows[0][0]), np.float32(np.inf))
    )
    best = {"objective": 0.0, "score_threshold": empty_threshold}
    offset = 0
    while offset < len(rows):
        threshold = rows[offset][0]
        end = offset + 1
        while end < len(rows) and rows[end][0] == threshold:
            end += 1
        for _, pdb_id, candidate_index in rows[offset:end]:
            facts = facts_by_pdb[pdb_id]
            evaluation = facts.evaluation
            semantic_tp += int(evaluation.candidate_semantic_tp[candidate_index])
            semantic_fp += int(
                evaluation.pred_sizes[candidate_index]
                - evaluation.candidate_semantic_tp[candidate_index]
            )
            selected_count += 1
            adjacency = adjacency_by_pdb[pdb_id]
            neighbors = np.flatnonzero(adjacency[candidate_index])
            coverage_pred_hit += int(neighbors.size > 0)
            covered_gt = covered_gt_by_pdb[pdb_id]
            coverage_gt_hit += int((~covered_gt[neighbors]).sum())
            covered_gt[neighbors] = True
            matched_gt = matched_gt_by_pdb[pdb_id]

            def augment(candidate: int, seen_gt: np.ndarray) -> bool:
                for gt_index in np.flatnonzero(adjacency[candidate]).tolist():
                    if seen_gt[gt_index]:
                        continue
                    seen_gt[gt_index] = True
                    previous = int(matched_gt[gt_index])
                    if previous < 0 or augment(previous, seen_gt):
                        matched_gt[gt_index] = candidate
                        return True
                return False

            if augment(candidate_index, np.zeros(matched_gt.size, dtype=np.bool_)):
                one_to_one_tp += 1
        objective = (
            _f_beta_from_counts(
                semantic_tp,
                semantic_fp,
                semantic_gt - semantic_tp,
                beta,
            )
            + _f_beta_from_precision_recall_counts(
                coverage_pred_hit,
                selected_count,
                coverage_gt_hit,
                gt_count,
                beta,
            )
            + _f_beta_from_precision_recall_counts(
                one_to_one_tp,
                selected_count,
                one_to_one_tp,
                gt_count,
                beta,
            )
        )
        if objective > float(best["objective"]):
            best = {"objective": objective, "score_threshold": threshold}
        offset = end
    return best


def _selection_objective(
    facts_by_pdb: Mapping[str, CenteredCalibrationFacts],
    scores_by_pdb: Mapping[str, np.ndarray],
    score_threshold: float,
    min_voxels: int,
    beta: float,
) -> float:
    """计算一个冻结选择组合的三个 micro F-beta 之和."""

    semantic_tp = 0
    semantic_fp = 0
    semantic_gt = 0
    selected_count = 0
    gt_count = 0
    coverage_pred_hit = 0
    coverage_gt_hit = 0
    one_to_one_tp = 0
    for pdb_id, facts in facts_by_pdb.items():
        selected = np.flatnonzero(
            (np.asarray(scores_by_pdb[pdb_id]) >= np.float32(score_threshold))
            & (facts.voxel_count >= int(min_voxels))
        )
        evaluation = facts.evaluation
        semantic_tp += int(evaluation.candidate_semantic_tp[selected].sum())
        semantic_fp += int(
            (
                evaluation.pred_sizes[selected]
                - evaluation.candidate_semantic_tp[selected]
            ).sum()
        )
        semantic_gt += int(evaluation.semantic_tp + evaluation.semantic_fn)
        selected_count += int(selected.size)
        gt_count += int(evaluation.gt_sizes.size)
        pred_cover = np.divide(
            evaluation.intersections[selected],
            evaluation.pred_sizes[selected, None],
            out=np.zeros((selected.size, evaluation.gt_sizes.size), dtype=np.float64),
            where=evaluation.pred_sizes[selected, None] > 0,
        )
        gt_cover = np.divide(
            evaluation.intersections[selected],
            evaluation.gt_sizes[None, :],
            out=np.zeros((selected.size, evaluation.gt_sizes.size), dtype=np.float64),
            where=evaluation.gt_sizes[None, :] > 0,
        )
        adjacency = (pred_cover >= 0.3) & (gt_cover >= 0.3)
        coverage_pred_hit += int(adjacency.any(axis=1).sum())
        coverage_gt_hit += int(adjacency.any(axis=0).sum())
        if adjacency.size:
            matched_pred, matched_gt = linear_sum_assignment(-adjacency.astype(np.int8))
            one_to_one_tp += int(adjacency[matched_pred, matched_gt].sum())
    semantic_fn = semantic_gt - semantic_tp
    return (
        _f_beta_from_counts(semantic_tp, semantic_fp, semantic_fn, beta)
        + _f_beta_from_precision_recall_counts(
            coverage_pred_hit, selected_count, coverage_gt_hit, gt_count, beta
        )
        + _f_beta_from_precision_recall_counts(
            one_to_one_tp, selected_count, one_to_one_tp, gt_count, beta
        )
    )


def _f_beta_from_counts(tp: int, fp: int, fn: int, beta: float) -> float:
    """从 TP, FP, FN 计算零分母为零的 F-beta."""

    beta2 = float(beta) ** 2
    numerator = (1.0 + beta2) * float(tp)
    denominator = numerator + beta2 * float(fn) + float(fp)
    return numerator / denominator if denominator else 0.0


def _f_beta_from_precision_recall_counts(
    precision_hit: int,
    precision_total: int,
    recall_hit: int,
    recall_total: int,
    beta: float,
) -> float:
    """从不同精确率与召回率计数计算 F-beta."""

    precision = precision_hit / precision_total if precision_total else 0.0
    recall = recall_hit / recall_total if recall_total else 0.0
    beta2 = float(beta) ** 2
    denominator = beta2 * precision + recall
    return (
        (1.0 + beta2) * precision * recall / denominator
        if denominator
        else 0.0
    )
