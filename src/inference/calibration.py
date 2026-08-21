# -*- coding: utf-8 -*-
"""冻结单个 F-alpha 语义阈值和 centered 选择参数.

主要入口 :func:`calibrate_semantic_thresholds` 返回单个 alpha 的语义阈值摘要与完整扫描数组,
:func:`tune_centered_selection` 返回 basic 或 Gaussian 评分参数, 分数阈值
和最小体素数. centered 校准先把候选与真实 occurrence 的交集压成小型事实表,
再执行来源均值阈值搜索或 Find Gaussian 三阶段搜索.
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
        - evaluation: PdbEvaluation, 候选轴保持 centered 条目顺序的评估事实.
        - source_probability_mean: float32 ``(N_candidate,)``, 来源 blob 平均概率.
        - voxel_count: int64 ``(N_candidate,)``, 每个来源 blob 的体素数.
        - prefilter_eligible: bool ``(N_candidate,)``, True 表示候选达到 tune 前固定的来源 blob 体素数下限.
        - atom_offsets: int64 ``(N_candidate+1,)``, 以半开区间同时切分 `atom_distance` 和 `atom_probability`; 首值为 0, 末值为 N_atom.
        - atom_distance: float32 ``(N_atom,)``, A 原子到来源 blob 的最近世界距离; 超过 5 Å 为 ``Inf``.
        - atom_probability: float32 ``(N_atom,)``, 与距离逐项对齐的 A 原子概率.

    非 Find producer 使用一个零 `atom_offsets` 以及空的 `atom_distance` 和
    `atom_probability`. 大型 V/A/P 特征和
    48³ 稠密数组不进入本对象.
    """

    evaluation: PdbEvaluation
    source_probability_mean: np.ndarray
    voxel_count: np.ndarray
    prefilter_eligible: np.ndarray
    atom_offsets: np.ndarray
    atom_distance: np.ndarray
    atom_probability: np.ndarray


def _f_beta_from_counts(tp: int, fp: int, fn: int, beta: float) -> float:
    """从 TP, FP, FN 计算 F-beta; 没有正例和预测时返回 0.0."""

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
    """从两组命中数与总数计算 F-beta; 任一零分母对应的比率取 0.0."""

    precision = precision_hit / precision_total if precision_total else 0.0
    recall = recall_hit / recall_total if recall_total else 0.0
    beta2 = float(beta) ** 2
    denominator = beta2 * precision + recall
    return (1.0 + beta2) * precision * recall / denominator if denominator else 0.0


def _gaussian_terms(
    facts_by_pdb: Mapping[str, CenteredCalibrationFacts],
    tau_angstrom: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """为一个 tau 计算每个 PDB 的候选 Gaussian 正项和负项.

    输入参数:
        - facts_by_pdb: 以小写 PDB 标识为键的校准事实; 每个值携带同一 PDB 的 A 原子距离和概率.
        - tau_angstrom: float, Gaussian 距离标准差, 单位 Å.

    返回值:
        - result: 以小写 PDB 标识为键的映射; 每个值的第一个 float32 `(N_candidate,)` 数组是 Gaussian 正项, 第二个同形数组是 Gaussian 负项.

    `result[pdb_id][0]` 正项与 `result[pdb_id][1]` 负项的候选轴都与
    `evaluation.source_blob_index` 对齐.
    """

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for pdb_id, facts in facts_by_pdb.items():
        result[pdb_id] = sum_gaussian_atom_terms(
            facts.atom_offsets,
            facts.atom_distance,
            facts.atom_probability,
            tau_angstrom,
        )
    return result


def _augment_one_to_one_match(
    candidate_index: int,
    adjacency: np.ndarray,
    matched_gt: np.ndarray,
    seen_gt: np.ndarray,
) -> bool:
    """为一个候选寻找一条一对一匹配增广路.

    输入参数:
        - candidate_index: int, 当前 PDB 候选轴下标.
        - adjacency: bool `(N_candidate, N_gt)`, True 表示候选与 occurrence 同时达到双向 0.3 覆盖.
        - matched_gt: int32 `(N_gt,)`, occurrence 当前匹配的候选下标; -1 表示尚未匹配, 成功时原位更新.
        - seen_gt: bool `(N_gt,)`, 本次增广搜索已经访问的 occurrence.

    返回值:
        - augmented: bool, 是否为当前候选找到增广路并增加一个匹配.

    递归只沿当前搜索尚未访问的 occurrence 展开, 因此不会重复进入同一节点.
    """

    for gt_index in np.flatnonzero(adjacency[candidate_index]).tolist():
        if seen_gt[gt_index]:
            continue
        seen_gt[gt_index] = True
        previous = int(matched_gt[gt_index])
        if previous < 0 or _augment_one_to_one_match(
            previous,
            adjacency,
            matched_gt,
            seen_gt,
        ):
            matched_gt[gt_index] = candidate_index
            return True
    return False


def _scan_actual_score_thresholds(
    facts_by_pdb: Mapping[str, CenteredCalibrationFacts],
    scores_by_pdb: Mapping[str, np.ndarray],
    min_voxels: int,
    beta: float,
) -> dict[str, float]:
    """按全局实际分数降序增量搜索来源均值阈值.

    固定预过滤未通过的候选不进入实际分数轴. 其余候选只加入一次; semantic
    与 coverage 使用累计计数, one-to-one@0.3 对当前 PDB 的匹配执行一次增广,
    不会对每个阈值重新计算全部交集或 Hungarian.

    输入参数:
        - facts_by_pdb: 以小写 PDB 标识为键的 centered 校准事实; `prefilter_eligible` 已在参数搜索前固定.
        - scores_by_pdb: 以同一 PDB 标识为键的 float32 `(N_candidate,)` 分数; 候选轴与对应事实对齐.
        - min_voxels: int, 进入阈值扫描的来源 blob 最小体素数, 包含端点.
        - beta: float, 三项 micro F-beta 共同使用的 beta.

    返回字段:
        - objective: float, semantic, coverage@0.3 与 one-to-one@0.3 三项 micro F-beta 之和.
        - score_threshold: float, 最大目标值严格提升时对应的首个实际候选分数; 非空候选的最佳目标仍为 0 时, 返回刚好高于最高分的空选择阈值; 完全没有候选时为 0.0.
    """

    rows = [
        (float(score), pdb_id, index)
        for pdb_id, scores in scores_by_pdb.items()
        for index, score in enumerate(scores.tolist())
        if facts_by_pdb[pdb_id].prefilter_eligible[index]
        and facts_by_pdb[pdb_id].voxel_count[index] >= int(min_voxels)
    ]
    rows.sort(key=lambda item: (-item[0], item[1], item[2]))
    if not rows:
        return {"objective": 0.0, "score_threshold": 0.0}
    # 以下状态从没有候选入选开始, 再按相同分数组成的候选组递增更新.
    semantic_gt = sum(
        facts.evaluation.semantic_tp + facts.evaluation.semantic_fn
        for facts in facts_by_pdb.values()
    )
    gt_count = sum(facts.evaluation.gt_sizes.size for facts in facts_by_pdb.values())
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
        covered_gt_by_pdb[pdb_id] = np.zeros(evaluation.gt_sizes.size, dtype=np.bool_)
        matched_gt_by_pdb[pdb_id] = np.full(
            evaluation.gt_sizes.size, -1, dtype=np.int32
        )

    empty_threshold = float(np.nextafter(np.float32(rows[0][0]), np.float32(np.inf)))
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
            if _augment_one_to_one_match(
                candidate_index,
                adjacency,
                matched_gt,
                np.zeros(matched_gt.size, dtype=np.bool_),
            ):
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
    """计算一个冻结选择组合的 semantic, coverage@0.3 和 one-to-one@0.3 micro F-beta 之和.

    输入参数:
        - facts_by_pdb: 以小写 PDB 标识为键的 centered 校准事实; `prefilter_eligible` 已在参数搜索前固定.
        - scores_by_pdb: 以同一 PDB 标识为键的 float32 `(N_candidate,)` 分数; 候选轴与对应事实对齐.
        - score_threshold: float, 候选分数下限, 包含端点.
        - min_voxels: int, 来源 blob 最小体素数, 包含端点.
        - beta: float, 三项 micro F-beta 共同使用的 beta.

    返回值:
        - objective: float, 入选候选的 semantic, coverage@0.3 与 one-to-one@0.3 三项 micro F-beta 之和.
    """

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
            & facts.prefilter_eligible
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


# ================================================================================================


def calibrate_semantic_thresholds(
    probability_and_target: Iterable[tuple[np.ndarray, np.ndarray]],
    denominator: int,
    alpha: float,
) -> dict[str, object]:
    """按 calibration 全集的 micro TP/FP/FN 冻结一个 F-alpha 阈值.

    输入可以是一次性迭代器, 因而调用者能够逐 PDB 解压完整概率图而不在内存
    中保留整个 calibration 集.

    每个概率先按 `floor(clip(p, 0, 1) * denominator)` 量化到整数网格. 当前
    阈值的指标为
    `F_alpha = (1 + alpha^2) * TP / ((1 + alpha^2) * TP + alpha^2 * FN + FP)`;
    alpha 大于 1 时漏检 FN 的权重更高, 因而相对偏重召回.

    输入参数:
        - probability_and_target.probability: float32 `(D, H, W)`, 当前 PDB 的完整图 ZYX 配体概率.
        - probability_and_target.target: bool `(D, H, W)`, 当前 PDB 的真实配体并集; True 表示属于至少一个 ligand occurrence, False 表示背景.
        - denominator: int, 闭区间 `[0, 1]` 概率网格的分母; 扫描 `denominator + 1` 个阈值.
        - alpha: 正浮点数, 当前语义阈值使用的 F-alpha 参数.

    返回字段:
        - denominator: int, 阈值网格分母.
        - positive_voxel_count: int, calibration 全集真实配体体素数.
        - negative_voxel_count: int, calibration 全集真实背景体素数.
        - alpha: float, 当前 F-alpha 参数.
        - threshold_grid_index: int, 首个最优阈值的整数网格位置.
        - threshold_value: float, `threshold_grid_index / denominator` 得到的概率阈值.
        - micro_f_beta: float, calibration 全集最优 micro F-alpha.
        - tp: int, 最优阈值的 micro TP.
        - fp: int, 最优阈值的 micro FP.
        - fn: int, 最优阈值的 micro FN.
        - scan.denominator: int32 标量, 与顶层分母相同.
        - scan.alpha: float64 标量, 当前 F-alpha 参数.
        - scan.threshold_grid_index: int32 ``(denominator+1,)``, 阈值整数轴.
        - scan.f_beta_curve: float64 ``(denominator+1,)``, 完整 micro F-alpha 曲线.
        - scan.tp: int64 ``(denominator+1,)``, 每个阈值的 micro TP.
        - scan.fp: int64 ``(denominator+1,)``, 每个阈值的 micro FP.
        - scan.fn: int64 ``(denominator+1,)``, 每个阈值的 micro FN.

    JSON 摘要发布前把 ``scan`` 分离为独立 NPZ.
    """

    # 两个直方图按概率网格位置累计真实配体体素和真实背景体素.
    positive_histogram = np.zeros(int(denominator) + 1, dtype=np.int64)
    negative_histogram = np.zeros(int(denominator) + 1, dtype=np.int64)
    for probability, target in probability_and_target:
        values = np.asarray(probability, dtype=np.float64)
        truth = np.asarray(target, dtype=np.bool_)
        bins = np.floor(np.clip(values, 0.0, 1.0) * int(denominator)).astype(np.int64)
        positive_histogram += np.bincount(bins[truth], minlength=int(denominator) + 1)
        negative_histogram += np.bincount(bins[~truth], minlength=int(denominator) + 1)
    # 反向累积把每个网格位置解释为包含端点的概率下限.
    tp = np.cumsum(positive_histogram[::-1], dtype=np.int64)[::-1]
    fp = np.cumsum(negative_histogram[::-1], dtype=np.int64)[::-1]
    fn = int(positive_histogram.sum()) - tp
    alpha2 = float(alpha) ** 2
    numerator = (1.0 + alpha2) * tp.astype(np.float64)
    metric_denominator = numerator + alpha2 * fn + fp
    curve = np.divide(
        numerator,
        metric_denominator,
        out=np.zeros_like(numerator),
        where=metric_denominator > 0,
    )
    # np.argmax 在并列时返回首个网格位置, 因而固定选择最低的最优阈值.
    grid_index = int(np.argmax(curve))
    return {
        "alpha": float(alpha),
        "denominator": int(denominator),
        "positive_voxel_count": int(positive_histogram.sum()),
        "negative_voxel_count": int(negative_histogram.sum()),
        "threshold_grid_index": grid_index,
        "threshold_value": float(grid_index) / float(denominator),
        "micro_f_beta": float(curve[grid_index]),
        "tp": int(tp[grid_index]),
        "fp": int(fp[grid_index]),
        "fn": int(fn[grid_index]),
        "scan": {
            "denominator": np.asarray(denominator, dtype=np.int32),
            "alpha": np.asarray(alpha, dtype=np.float64),
            "threshold_grid_index": np.arange(int(denominator) + 1, dtype=np.int32),
            "f_beta_curve": curve,
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
    prefiltered_min_voxel: int,
    min_voxel_values: Sequence[int],
    objective_beta: float,
    coverage_thresholds: Sequence[float],
    topk_values: Sequence[int],
) -> dict[str, object]:
    """按冻结顺序搜索 centered 分数与最小体素数.

    两种模式先固定 ``prefiltered_min_voxel``, 小于该值的候选在全部参数尝试中
    保持未入选. basic 模式再在最小 ``min_voxels`` 下扫描实际出现的 float32 分数, 然后
    冻结分数阈值并扫描全部最小体素数. Gaussian 模式第一阶段扫描 tau,
    两个 lambda 和 ``gauss_score_min`` 的显式粗网格; 第二阶段固定 tau, 对
    第一阶段三个其余参数应用显式乘数; 第三阶段冻结 Gaussian 参数并只扫描
    ``min_voxels``. 目标是 semantic, coverage@0.3 和 one-to-one@0.3 三个
    micro F-beta 之和. basic 按分数降序扫描并只在目标严格提升时替换阈值;
    非空候选的最佳目标仍为 0 时保留高于最高分的空选择阈值. Gaussian 网格
    与两种模式的 `min_voxels` 扫描在完全并列时保留配置顺序中先出现的值.

    输入参数:
        - centered_items.pdb_id: 字符串, 当前小写 PDB 标识.
        - centered_items.centered.source_blob_index: int32 `(N_candidate,)`, 来源 blob 编号.
        - centered_items.centered.source_probability_mean: float32 `(N_candidate,)`, 来源 blob 平均概率.
        - centered_items.centered.voxel_offsets: int64 `(N_candidate + 1,)`, 以半开区间切分 `voxel_index_local_zyx`; 首值为 0, 末值为 L_voxel.
        - centered_items.centered.voxel_index_local_zyx: int16 `(L_voxel, 3)`, 候选 BOX 内 ZYX 体素索引.
        - centered_items.centered.box_start_zyx: int32 `(N_candidate, 3)`, 候选 BOX 在完整图中的 ZYX 起点.
        - centered_items.centered.A_offsets: Find Gaussian 专用 int64 `(N_candidate + 1,)`, 以半开区间同步切分 `A_coord_local_xyz` 与 `A_probability`; 首值为 0, 末值为 N_A.
        - centered_items.centered.A_coord_local_xyz: Find Gaussian 专用 float32 `(N_A, 3)`, BOX 局部 XYZ 原子坐标.
        - centered_items.centered.A_probability: Find Gaussian 专用 float32 `(N_A,)`, A 原子概率.
        - centered_items.centered.voxel_size_world: Find Gaussian 专用 float32 `(N_candidate, 3)`, 世界 XYZ 体素尺寸.
        - ground_truth_by_pdb.occurrence_id: int32 `(N_gt,)`, 当前 PDB 的 ligand occurrence 标识.
        - ground_truth_by_pdb.occurrence_voxel_zyx: 长度 N_gt 的 int32 `(K_i, 3)` 序列, 每项保存一个 occurrence 的完整图 ZYX 体素.
        - ground_truth_by_pdb.full_shape_zyx: 三个整数, 当前 PDB 的完整图 ZYX 形状.
        - score_mode: 字符串, `basic` 或 `gaussian`.
        - score_parameter_grid.tau_angstrom: Sequence[float] 或 None, Gaussian 距离标准差粗网格.
        - score_parameter_grid.lambda_positive: Sequence[float] 或 None, Gaussian 正项系数粗网格.
        - score_parameter_grid.lambda_negative: Sequence[float] 或 None, Gaussian 负项系数粗网格.
        - score_parameter_grid.gauss_score_min: Sequence[float] 或 None, Gaussian 分数下限粗网格.
        - refinement_multipliers.lambda: Sequence[float] 或 None, Gaussian 正负系数的细网格乘数.
        - refinement_multipliers.score_threshold: Sequence[float] 或 None, Gaussian 分数下限的细网格乘数.
        - prefiltered_min_voxel: int, tune 开始前固定的来源 blob 体素数下限, 包含端点; 不限制 `min_voxel_values`.
        - min_voxel_values: 整数序列, 来源 blob 最小体素数候选值, 每个阈值包含端点.
        - objective_beta: float, semantic, coverage@0.3 与 one-to-one@0.3 三项 micro F-beta 共同使用的 beta.
        - coverage_thresholds: 浮点数序列, 逐 PDB 事实保存的双向覆盖阈值轴; 校准目标使用 0.3.
        - topk_values: 整数序列, 逐 PDB 事实保存的 top-K 候选数量轴.

    返回字段:
        - objective: float, 最终最小体素数对应的三项 micro F-beta 之和.
        - objective_beta: float, tune 命令显式采用的三项 F-beta 参数.
        - score_mode: str, `basic` 或 `gaussian`.
        - score_parameters.tau_angstrom: float, Gaussian 距离标准差; 来源均值模式无此字段.
        - score_parameters.lambda_positive: float, Gaussian 正项系数; 来源均值模式无此字段.
        - score_parameters.lambda_negative: float, Gaussian 负项系数; 来源均值模式无此字段.
        - score_threshold: float, 冻结分数下限, 包含端点.
        - prefiltered_min_voxel: int, tune 前固定的来源 blob 体素数下限, 包含端点.
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

    # 每个 PDB 只保留交集矩阵, 来源均值, 体素数和可选 A 原子距离表; 大型 centered 特征不常驻校准内存.
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
        if score_mode == "gaussian":
            atom_table = build_gaussian_distance_table(centered)
            atom_offsets = atom_table["A_offsets"]
            atom_distance = atom_table["A_distance_to_source"]
            atom_probability = atom_table["A_probability"]
        else:
            atom_offsets = np.zeros(1, dtype=np.int64)
            atom_distance = np.empty(0, dtype=np.float32)
            atom_probability = np.empty(0, dtype=np.float32)
        voxel_count = np.diff(np.asarray(centered["voxel_offsets"], dtype=np.int64))
        facts_by_pdb[pdb_id] = CenteredCalibrationFacts(
            evaluation=evaluation,
            source_probability_mean=np.asarray(
                centered["source_probability_mean"], dtype=np.float32
            ),
            voxel_count=voxel_count,
            prefilter_eligible=voxel_count >= int(prefiltered_min_voxel),
            atom_offsets=atom_offsets,
            atom_distance=atom_distance,
            atom_probability=atom_probability,
        )

    # 第一和第二阶段统一使用最宽松的体素下限, 最终阶段才单独冻结 min_voxels.
    initial_min_voxels = min(int(value) for value in min_voxel_values)
    if score_mode == "basic":
        # basic 只扫描实际出现的来源 blob 平均概率, 不建立人为阈值网格.
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
        # Gaussian 粗网格同时搜索距离尺度, 两个符号项系数和分数下限.
        coarse_best: dict[str, object] | None = None
        if score_parameter_grid is None:
            raise ValueError("gaussian 缺少 score_parameter_grid.")
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
                        if coarse_best is None or objective > float(
                            coarse_best["objective"]
                        ):
                            coarse_best = {
                                "objective": objective,
                                "tau_angstrom": float(tau),
                                "lambda_positive": float(lambda_positive),
                                "lambda_negative": float(lambda_negative),
                                "score_threshold": float(score_min),
                            }
        if coarse_best is None or refinement_multipliers is None:
            raise ValueError("gaussian 粗网格或第二阶段乘数为空.")
        # 细网格固定粗搜索的 tau, 只对两个系数和分数下限应用显式乘数.
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
                    if refined_best is None or objective > float(
                        refined_best["objective"]
                    ):
                        refined_best = {
                            "objective": objective,
                            "tau_angstrom": tau,
                            "lambda_positive": lambda_positive,
                            "lambda_negative": lambda_negative,
                            "score_threshold": score_threshold,
                        }
        if refined_best is None:
            raise RuntimeError("gaussian 第二阶段没有产生参数组合.")
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

    # 分数定义冻结后只扫描来源 blob 最小体素数.
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
        "prefiltered_min_voxel": int(prefiltered_min_voxel),
        "min_voxels": int(minimum_best["min_voxels"]),
        "stages": stages,
    }
