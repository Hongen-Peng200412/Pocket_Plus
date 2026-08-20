# -*- coding: utf-8 -*-
"""冻结 Stage1 V3 的语义阈值和 centered 选择参数.

语义阈值由完整图概率与 occurrence 并集的全局直方图确定. centered 选择调参
复用正式评分和评估函数, 对显式参数表、最小体素数与实际候选分数逐项搜索.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .evaluation import (
    PdbEvaluation,
    aggregate_stage1_metrics,
    evaluate_centered_pdb,
)
from .scoring import score_centered_candidates, select_centered_candidates


# ================================================================================================


def calibrate_semantic_thresholds(
    probability_and_target: Sequence[tuple[np.ndarray, np.ndarray]],
    denominator: int,
    betas: Sequence[float],
) -> dict[str, object]:
    """按完整 calibration 集的 micro TP/FP/FN 冻结各 F-beta 阈值.

    概率先量化为 ``floor(probability * denominator)`` 的整数层. 阈值层 ``j``
    表示物理阈值 ``j / denominator``; 同一 F-beta 最大值并列时取较小的 ``j``.
    """

    positive_histogram = np.zeros(int(denominator) + 1, dtype=np.int64)
    negative_histogram = np.zeros(int(denominator) + 1, dtype=np.int64)
    for probability, target in probability_and_target:
        values = np.asarray(probability, dtype=np.float64)
        truth = np.asarray(target, dtype=np.bool_)
        bins = np.floor(np.clip(values, 0.0, 1.0) * int(denominator)).astype(np.int64)
        positive_histogram += np.bincount(
            bins[truth],
            minlength=int(denominator) + 1,
        )
        negative_histogram += np.bincount(
            bins[~truth],
            minlength=int(denominator) + 1,
        )
    tp = np.cumsum(positive_histogram[::-1], dtype=np.int64)[::-1]
    fp = np.cumsum(negative_histogram[::-1], dtype=np.int64)[::-1]
    fn = int(positive_histogram.sum()) - tp
    result: dict[str, object] = {
        "denominator": int(denominator),
        "positive_voxel_count": int(positive_histogram.sum()),
        "negative_voxel_count": int(negative_histogram.sum()),
        "thresholds": {},
    }
    thresholds: dict[str, object] = {}
    for beta in betas:
        beta2 = float(beta) ** 2
        numerator = (1.0 + beta2) * tp.astype(np.float64)
        metric_denominator = numerator + beta2 * fn + fp
        curve = np.divide(
            numerator,
            metric_denominator,
            out=np.zeros_like(numerator),
            where=metric_denominator > 0,
        )
        grid_index = int(np.argmax(curve))
        thresholds[f"F{int(beta)}"] = {
            "grid_index": grid_index,
            "value": float(grid_index) / float(denominator),
            "micro_f_beta": float(curve[grid_index]),
            "tp": int(tp[grid_index]),
            "fp": int(fp[grid_index]),
            "fn": int(fn[grid_index]),
        }
    result["thresholds"] = thresholds
    return result


def tune_centered_selection(
    centered_by_pdb: Mapping[str, Mapping[str, np.ndarray]],
    ground_truth_by_pdb: Mapping[
        str,
        tuple[np.ndarray, Sequence[np.ndarray], Sequence[int]],
    ],
    score_mode: str,
    score_parameter_rows: Sequence[Mapping[str, float]],
    min_voxel_values: Sequence[int],
    objective_beta: float,
    coverage_thresholds: Sequence[float],
    topk_values: Sequence[int],
) -> dict[str, object]:
    """搜索评分参数、最小体素数和实际候选分数阈值.

    每组评分参数都先为全部 PDB 计算固定分数. 对每个最小体素数, 阈值候选是
    当前仍合格候选的实际 float32 分数及一个高于最大分数的空选择边界. 目标值
    是 semantic、coverage@0.3 和 one-to-one@0.3 三个 micro F-beta 之和.
    参数表和最小体素数按调用者顺序遍历; 完全并列时保留先出现的组合.
    """

    best: dict[str, object] | None = None
    beta_tag = int(objective_beta)
    coverage_tag = (
        f"{float(coverage_thresholds[0]):.3f}"
        .rstrip("0")
        .rstrip(".")
        .replace(".", "p")
    )
    for parameter_row in score_parameter_rows:
        scores = {
            pdb_id: score_centered_candidates(
                centered,
                score_mode=score_mode,
                score_parameters=parameter_row,
            )
            for pdb_id, centered in centered_by_pdb.items()
        }
        for min_voxels in min_voxel_values:
            eligible_scores = [
                values[
                    np.diff(np.asarray(centered_by_pdb[pdb_id]["voxel_offsets"], dtype=np.int64))
                    >= int(min_voxels)
                ]
                for pdb_id, values in scores.items()
            ]
            nonempty = [values for values in eligible_scores if values.size]
            if nonempty:
                thresholds = np.unique(np.concatenate(nonempty).astype(np.float32))
                thresholds = np.concatenate(
                    (
                        thresholds,
                        np.asarray(
                            [np.nextafter(thresholds[-1], np.float32(np.inf))],
                            dtype=np.float32,
                        ),
                    )
                )
            else:
                thresholds = np.asarray([0.0], dtype=np.float32)
            for threshold in thresholds.tolist():
                evaluations: list[PdbEvaluation] = []
                for pdb_id, centered in centered_by_pdb.items():
                    selected = select_centered_candidates(
                        centered,
                        scores[pdb_id],
                        score_threshold=float(threshold),
                        min_voxels=int(min_voxels),
                    )
                    occurrence_id, occurrence_rows, full_shape = ground_truth_by_pdb[pdb_id]
                    evaluations.append(
                        evaluate_centered_pdb(
                            pdb_id=pdb_id,
                            centered=selected,
                            occurrence_id=occurrence_id,
                            occurrence_voxel_zyx=occurrence_rows,
                            full_shape_zyx=full_shape,
                            coverage_thresholds=coverage_thresholds,
                            topk_values=topk_values,
                        )
                    )
                metrics = aggregate_stage1_metrics(
                    evaluations,
                    coverage_thresholds=coverage_thresholds,
                    topk_values=topk_values,
                )
                objective = (
                    float(metrics[f"semantic_micro_f{beta_tag}"])
                    + float(metrics[f"coverage_micro_f{beta_tag}_{coverage_tag}"])
                    + float(metrics[f"one_to_one_micro_f{beta_tag}_{coverage_tag}"])
                )
                if best is None or objective > float(best["objective"]):
                    best = {
                        "objective": objective,
                        "objective_beta": float(objective_beta),
                        "score_mode": score_mode,
                        "score_parameters": dict(parameter_row),
                        "min_voxels": int(min_voxels),
                        "score_threshold": float(threshold),
                        "metrics": metrics,
                    }
    if best is None:
        raise RuntimeError("centered calibration 没有产生候选参数组合.")
    return best
