# -*- coding: utf-8 -*-
"""计算 Stage1 完整图语义 PRAUC, 候选语义和实例指标.

主要入口 :func:`semantic_prauc_histogram` 和 :func:`aggregate_semantic_prauc`
生成完整图 PRAUC, :func:`evaluate_centered_pdb` 和
:func:`aggregate_stage1_metrics` 生成候选交集事实及跨 PDB 汇总.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch


# 语义 PRAUC 使用与 dense ligand 训练指标相同的均匀阈值数量.
SEMANTIC_PRAUC_THRESHOLD_COUNT = 1024
# float32, (1024,), 直接复用训练指标生成阈值的 PyTorch 数值路径.
_SEMANTIC_PRAUC_THRESHOLDS = torch.linspace(
    0.0,
    1.0,
    SEMANTIC_PRAUC_THRESHOLD_COUNT,
    dtype=torch.float32,
).numpy()
# 每次最多量化该数量的体素, 避免完整图分箱产生大型临时索引.
_PRAUC_VOXEL_CHUNK_SIZE = 8_000_000


@dataclass(frozen=True)
class PdbEvaluation:
    """保存一个 PDB 的候选, 真实 occurrence 和指标对齐事实.

    字段:
        - pdb_id: str, 小写 PDB 标识.
        - occurrence_id: int32 ``(N_gt,)``, 真实 occurrence 标识.
        - source_blob_index: int32 ``(N_pred,)``, 按最终 score 稳定降序的来源 blob 标识.
        - candidate_score: float32 ``(N_pred,)``, 与候选轴对齐的冻结分数.
        - candidate_selected: bool ``(N_pred,)``, 当前评估实际纳入指标的候选掩码.
        - intersections: int64 ``(N_pred, N_gt)``, 候选与 occurrence 的体素交集数.
        - pred_sizes: int64 ``(N_pred,)``, 每个候选的稀疏体素数.
        - gt_sizes: int64 ``(N_gt,)``, 每个真实 occurrence 的稀疏体素数.

        - candidate_semantic_tp: int64 ``(N_pred,)``, 每个候选与真实并集的交集数.
        - semantic_tp: int, 所有 ``selected=True`` 候选并集与真实并集的交集体素数.
        - semantic_fp: int, 已选候选并集落在真实并集外的体素数.
        - semantic_fn: int, 真实并集未被已选候选并集覆盖的体素数.
        - coverage_thresholds: float32 ``(N_threshold,)``, 评估阈值, 0.3/0.5/0.6.
        - topk_values: int32 ``(N_topk,)``, top-K 轴.

        - coverage_pred_hit_mask: bool ``(N_threshold, N_pred)``, 每个候选是否达到多对多覆盖阈值, 与 selected 无关.
        - coverage_gt_hit_mask: bool ``(N_threshold, N_gt)``, 是否至少被一个已选候选覆盖.
        - one_to_one_match_offsets: int64 ``(N_threshold+1,)``, 以半开区间同时切分 `one_to_one_match_pred_index` 与 `one_to_one_match_gt_index`; 首值为 0, 末值为 L_match.
        - coverage_pred_hit: int64 ``(N_threshold,)``, 每个阈值命中的已选候选数.
        - coverage_gt_hit: int64 ``(N_threshold,)``, 每个阈值被已选候选命中的 occurrence 数.
        - one_to_one_match_pred_index: int32 ``(L_match,)``, 各阈值最大匹配的候选轴下标.
        - one_to_one_match_gt_index: int32 ``(L_match,)``, 与前项对齐的 occurrence 轴下标.
        - one_to_one_tp: int64 ``(N_threshold,)``, 每个阈值下已选候选与 occurrence 的最大一对一匹配数.

        - topk_success: int64 ``(N_topk, N_threshold)``, 每组 top-K 与覆盖阈值是否至少命中一个 occurrence, 取值为 0 或 1.
        - topk_winning_candidate_rank: int32 ``(N_topk, N_threshold)``, 首个获胜候选在已选候选序列中从 0 开始的名次, 不是完整候选轴下标; 未命中为 ``-1``.
        - topk_winning_occurrence_index: int32 ``(N_topk, N_threshold)``, 对应 occurrence 轴下标, 未命中为 ``-1``.

    所有候选轴都与 ``source_blob_index`` 对齐;
    阈值轴与 ``coverage_thresholds`` 对齐, top-K 轴与 ``topk_values`` 对齐.
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

    文件字段:
        - grid_shape_zyx: 整数数组 `(3,)`, 完整图 ZYX 形状.
        - mask_<occurrence_id>: 整数数组 `(K_gt, 3)`, 一个真实 ligand occurrence 的完整图 ZYX 体素索引; 数字后缀是 occurrence 标识.

    返回值:
        - occurrence_id: int32 `(N_gt,)`, 按数字升序排列的 occurrence 标识.
        - voxel_rows: 长度 N_gt 的元组; 第 i 项是 int32 `(K_gt_i, 3)` 完整图 ZYX 体素索引, 与 `occurrence_id[i]` 对齐.
        - full_shape_zyx: 三个整数, 完整图 ZYX 形状.
    """
    path = Path(ligand_area_path)
    with np.load(path, allow_pickle=False) as archive:
        # shape: 三个整数, 完整图的 ZYX 体素数量.
        shape = tuple(int(value) for value in archive["grid_shape_zyx"])
        # names: 按数字后缀升序排列的 mask_<occurrence_id> 字段名.
        names = sorted(
            (
                name
                for name in archive.files
                if name.startswith("mask_") and name[5:].isdigit()
            ),
            key=lambda name: int(name[5:]),
        )
        # int32, (N_gt,), 每个真实 ligand occurrence 的数字标识.
        occurrence_id = np.asarray([int(name[5:]) for name in names], dtype=np.int32)
        # rows: 长度 N_gt 的元组; 第 i 项是 occurrence_id[i] 的 int32 (K_gt_i, 3) 完整图 ZYX 体素索引.
        rows = tuple(np.asarray(archive[name], dtype=np.int32) for name in names)
    return occurrence_id, rows, shape


def semantic_prauc_histogram(
    probability_map: np.ndarray,
    union_mask: np.ndarray,
) -> np.ndarray:
    """把一个 PDB 的完整图语义概率压缩为 PRAUC 正负体素计数.

    输入参数:
        - probability_map: float32 ``(D, H, W)``, 完整图 ZYX 配体区域概率; D, H, W 依次对应 Z, Y, X.
        - union_mask: bool ``(D, H, W)``, 同一完整图的 ligand occurrence 体素并集; True 表示配体体素, False 表示背景体素.

    返回值:
        - histogram: int64 ``(2, 1024)``, 第一轴依次是负体素和正体素; 第 j 个概率档保存最高通过阈值为 ``t_j`` 的体素数.

    ``t_j`` 是 ``torch.linspace(0, 1, 1024, dtype=torch.float32)[j]``, 包含
    0 和 1 两个端点; ``probability_map >= t_j`` 记为该阈值下的阳性预测.
    概率分块处理, 不在内存中构造 ``N_voxel x 1024`` 的阈值判定矩阵.
    """

    # probability: float32, (N_voxel,), 当前 PDB 的完整图概率.
    probability = np.asarray(probability_map, dtype=np.float32).reshape(-1)
    # target: bool, (N_voxel,), True 表示体素属于真实 ligand occurrence 并集.
    target = np.asarray(union_mask, dtype=np.bool_).reshape(-1)
    # int64, (2, 1024), 第一轴依次累计负体素和正体素, 第二维是与 _SEMANTIC_PRAUC_THRESHOLDS 对齐的升序阈值概率档.
    histogram = np.zeros((2, SEMANTIC_PRAUC_THRESHOLD_COUNT), dtype=np.int64)
    for begin in range(0, probability.size, _PRAUC_VOXEL_CHUNK_SIZE):
        # begin, end: 当前展平概率分块的半开区间端点.
        end = min(begin + _PRAUC_VOXEL_CHUNK_SIZE, probability.size)
        # int32, (N_chunk,), 每个概率在 _SEMANTIC_PRAUC_THRESHOLDS 中所能通过的最高包含端点阈值下标, 用于累计 histogram 第二维.
        threshold_index = (
            np.searchsorted(
                _SEMANTIC_PRAUC_THRESHOLDS,
                probability[begin:end],
                side="right",
            ).astype(np.int32, copy=False)
            - 1
        )
        # bool, (N_chunk,), 与 threshold_index 对齐的当前分块语义真值.
        target_chunk = target[begin:end]
        histogram[0] += np.bincount(
            threshold_index[~target_chunk],
            minlength=SEMANTIC_PRAUC_THRESHOLD_COUNT,
        )
        histogram[1] += np.bincount(
            threshold_index[target_chunk],
            minlength=SEMANTIC_PRAUC_THRESHOLD_COUNT,
        )
    return histogram


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

    输入参数:
        - pdb_id: 字符串, 当前 PDB 标识.
        - centered: centered 产物字段映射; 读取分数, 选择掩码, 稀疏体素, BOX 起点和来源 blob 编号.
        - occurrence_id: int32 `(N_gt,)`, 真实 occurrence 标识.
        - occurrence_voxel_zyx: 长度 N_gt 的稀疏坐标序列; 第 i 项是 occurrence_id[i] 的完整图 ZYX 体素索引.
        - full_shape_zyx: 三个整数, 完整图 ZYX 形状, 用于把三维坐标转成不跨 PDB 的线性体素编号.
        - coverage_thresholds: 覆盖阈值序列; 候选侧和 occurrence 侧都达到同一阈值才形成匹配边.
        - topk_values: 正整数序列; 每个值限制按分数排序后参与 top-K 评估的已选候选数.

    返回值:
        - evaluation: `PdbEvaluation`, 保存完整候选轴, 真实 occurrence 轴, 两轴交集, 语义计数, 多对多覆盖, 一对一匹配与 top-K 命中事实.
    """
    # float64, (N_pred,), centered 候选的冻结分数, 输入顺序与 source_blob_index 一致.
    score = np.asarray(centered["score"], dtype=np.float64)
    # int64, (N_pred,), 按分数稳定降序排列的原候选轴下标.
    order = np.argsort(-score, kind="stable")
    # bool, (N_pred,), 按分数降序对齐的最终候选选择掩码.
    candidate_selected = np.asarray(centered["selected"], dtype=np.bool_)[order]
    score = score[order]

    # int64, (N_pred + 1,), 以原候选轴顺序切分 voxel_index_local_zyx.
    voxel_offsets = np.asarray(centered["voxel_offsets"], dtype=np.int64)
    # int64, (L_voxel, 3), 每个候选稀疏体素在所属 BOX 内的局部 ZYX 索引.
    local_rows = np.asarray(centered["voxel_index_local_zyx"], dtype=np.int64)
    # int64, (N_pred, 3), 每个候选 80³ BOX 在完整图中的 ZYX 起点.
    box_starts = np.asarray(centered["box_start_zyx"], dtype=np.int64)
    # shape: 三个整数, 完整图的 ZYX 体素数量.
    shape = tuple(int(value) for value in full_shape_zyx)

    # 每个稀疏 ZYX 坐标集合转换成完整图 C-order 线性编号, 便于执行集合交并.
    pred_linear: list[np.ndarray] = []
    for entry_index in order.tolist():
        # begin, end: 当前原候选在 voxel_index_local_zyx 中的半开区间端点.
        begin = int(voxel_offsets[entry_index])
        end = int(voxel_offsets[entry_index + 1])
        # int64, (K_pred, 3), 当前候选在完整图中的 ZYX 体素索引.
        global_zyx = local_rows[begin:end] + box_starts[entry_index][None, :]
        pred_linear.append(np.unique(np.ravel_multi_index(global_zyx.T, shape)))

    # gt_linear: 长度 N_gt 的列表; 第 j 项是 occurrence_id[j] 的唯一 C-order 线性体素编号.
    gt_linear = [
        np.unique(np.ravel_multi_index(np.asarray(rows, dtype=np.int64).T, shape))
        for rows in occurrence_voxel_zyx
    ]
    # int64, (N_pred, N_gt), 两个轴分别由 score 排序候选和 occurrence_id 固定.
    intersections = np.asarray(
        [
            [np.intersect1d(pred, gt, assume_unique=True).size for gt in gt_linear]
            for pred in pred_linear
        ],
        dtype=np.int64,
    ).reshape(len(pred_linear), len(gt_linear))
    # int64, (N_pred,), 每个预测候选的唯一体素数, 候选轴按 score 降序.
    pred_sizes = np.asarray([rows.size for rows in pred_linear], dtype=np.int64)
    # int64, (N_gt,), 每个真实 occurrence 的唯一体素数.
    gt_sizes = np.asarray([rows.size for rows in gt_linear], dtype=np.int64)

    # 语义 TP/FP/FN 使用已选候选的体素并集; 逐候选事实仍保留全部候选.
    # int64, (N_selected,), 已选候选在分数降序候选轴中的下标.
    selected_rows = np.flatnonzero(candidate_selected)
    # selected_pred_linear: 按分数降序排列的已选候选线性体素集合.
    selected_pred_linear = [pred_linear[index] for index in selected_rows.tolist()]
    # int64, (K_pred_union,), 全部已选候选的唯一 C-order 线性体素并集.
    pred_union = (
        np.unique(np.concatenate(selected_pred_linear))
        if selected_pred_linear
        else np.empty(0, dtype=np.int64)
    )
    # int64, (K_gt_union,), 全部真实 occurrence 的唯一 C-order 线性体素并集.
    gt_union = (
        np.unique(np.concatenate(gt_linear))
        if gt_linear
        else np.empty(0, dtype=np.int64)
    )
    # int64, (N_pred,), 每个候选与真实 occurrence 总并集的交集体素数.
    candidate_semantic_tp = np.asarray(
        [
            np.intersect1d(pred, gt_union, assume_unique=True).size
            for pred in pred_linear
        ],
        dtype=np.int64,
    )
    semantic_tp = int(np.intersect1d(pred_union, gt_union, assume_unique=True).size)
    semantic_fp = int(pred_union.size - semantic_tp)
    semantic_fn = int(gt_union.size - semantic_tp)

    # float64, (N_pred, N_gt), 交集占预测候选体素数的比例.
    pred_cover = np.divide(
        intersections,
        pred_sizes[:, None],
        out=np.zeros(intersections.shape, dtype=np.float64),
        where=pred_sizes[:, None] > 0,
    )
    # float64, (N_pred, N_gt), 交集占真实 occurrence 体素数的比例.
    gt_cover = np.divide(
        intersections,
        gt_sizes[None, :],
        out=np.zeros(intersections.shape, dtype=np.float64),
        where=gt_sizes[None, :] > 0,
    )

    # thresholds: 长度 N_threshold 的双向覆盖阈值轴, 保持调用方顺序.
    thresholds = tuple(float(value) for value in coverage_thresholds)
    # bool, (N_threshold, N_pred), 每个候选是否与任一 occurrence 达到双向覆盖阈值.
    coverage_pred_hit_mask = np.zeros(
        (len(thresholds), len(pred_linear)), dtype=np.bool_
    )
    # bool, (N_threshold, N_gt), 每个 occurrence 是否被任一已选候选达到双向覆盖阈值.
    coverage_gt_hit_mask = np.zeros((len(thresholds), len(gt_linear)), dtype=np.bool_)
    # match_pred_rows: 长度 N_threshold 的列表; 每项保存该阈值一对一匹配的候选轴下标.
    match_pred_rows: list[np.ndarray] = []
    # match_gt_rows: 长度 N_threshold 的列表; 每项保存与 match_pred_rows 同步的 occurrence 轴下标.
    match_gt_rows: list[np.ndarray] = []
    for threshold_row, threshold in enumerate(thresholds):
        # bool, (N_pred, N_gt), 两个方向的覆盖率都达到同一阈值才形成有效边.
        valid = (pred_cover >= threshold) & (gt_cover >= threshold)
        coverage_pred_hit_mask[threshold_row] = valid.any(axis=1)
        # bool, (N_selected, N_gt), 只保留已选候选后的有效匹配边.
        selected_valid = valid[selected_rows]
        coverage_gt_hit_mask[threshold_row] = selected_valid.any(axis=0)
        if selected_valid.size:
            # matched_pred, matched_gt: int64, (N_assignment,), Hungarian 返回的已选候选轴与 occurrence 轴下标.
            matched_pred, matched_gt = linear_sum_assignment(
                -selected_valid.astype(np.int8)
            )
            # bool, (N_assignment,), True 表示 Hungarian 返回的候选-occurrence 配对确实达到双向覆盖阈值.
            keep = selected_valid[matched_pred, matched_gt]
            match_pred_rows.append(selected_rows[matched_pred[keep]].astype(np.int32))
            match_gt_rows.append(matched_gt[keep].astype(np.int32))
        else:
            match_pred_rows.append(np.empty(0, dtype=np.int32))
            match_gt_rows.append(np.empty(0, dtype=np.int32))

    # int64, (N_threshold,), 每个覆盖阈值下有效的一对一匹配数量.
    one_to_one_tp = np.asarray([rows.size for rows in match_pred_rows], dtype=np.int64)
    # int64, (N_threshold + 1,), 同步切分扁平化的一对一候选下标与 occurrence 下标.
    match_offsets = np.concatenate(
        (
            np.zeros(1, dtype=np.int64),
            np.cumsum(one_to_one_tp, dtype=np.int64),
        )
    )
    # int64, (N_topk, N_threshold), 每组 top-K 是否至少命中一个 occurrence, 取值为 0 或 1.
    topk_success = np.zeros((len(topk_values), len(thresholds)), dtype=np.int64)
    # int32, (N_topk, N_threshold), 首个获胜候选在已选候选序列中的名次; -1 表示未命中.
    topk_winning_candidate_rank = np.full(topk_success.shape, -1, dtype=np.int32)
    # int32, (N_topk, N_threshold), 首个获胜 occurrence 的轴下标; -1 表示未命中.
    topk_winning_occurrence_index = np.full(topk_success.shape, -1, dtype=np.int32)
    for topk_row, topk in enumerate(topk_values):
        # int64, (min(K, N_selected),), 前 K 个已选候选在完整分数降序候选轴中的下标.
        top_candidate_rows = selected_rows[: int(topk)]
        for threshold_row, threshold in enumerate(thresholds):
            # bool, (min(K, N_selected), N_gt), top-K 候选与 occurrence 的双向覆盖命中表.
            valid = (pred_cover[top_candidate_rows] >= threshold) & (
                gt_cover[top_candidate_rows] >= threshold
            )
            # int64, (N_winner, 2), 命中表中按 C-order 排列的候选名次与 occurrence 轴下标.
            winners = np.argwhere(valid)
            if winners.size:
                topk_success[topk_row, threshold_row] = 1
                topk_winning_candidate_rank[topk_row, threshold_row] = winners[0, 0]
                topk_winning_occurrence_index[topk_row, threshold_row] = winners[0, 1]
    return PdbEvaluation(
        pdb_id=str(pdb_id).lower(),
        occurrence_id=np.asarray(occurrence_id, dtype=np.int32),
        source_blob_index=np.asarray(centered["source_blob_index"], dtype=np.int32)[
            order
        ],
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
    """把逐 PDB 候选事实汇总为数据划分级 JSON 指标.

    输入参数:
        - evaluations: Sequence[PdbEvaluation], 同一 producer, 数据划分和 centered 角色的逐 PDB 事实; macro 指标给予每个 PDB 相同权重.
        - coverage_thresholds: Sequence[float], 双向覆盖阈值轴; 顺序与每个 PdbEvaluation.coverage_thresholds 一致.
        - topk_values: Sequence[int], top-K 候选数量轴; 顺序与每个 PdbEvaluation.topk_values 一致.

    固定返回字段:
        - pdb_count: int, evaluations 中的 PDB 数量.
        - semantic_tp: int, 全部 PDB 已选候选体素并集与真实体素并集的交集数.
        - semantic_fp: int, 全部 PDB 已选候选体素并集落在真实体素并集外的体素数.
        - semantic_fn: int, 全部 PDB 真实体素并集未被已选候选覆盖的体素数.
        - semantic_micro_f1: float, 先汇总全部 PDB 的 TP, FP, FN 再计算的语义 F1.
        - semantic_micro_f2: float, 先汇总全部 PDB 的 TP, FP, FN 再计算的语义 F2.
        - semantic_macro_f1: float, 逐 PDB 语义 F1 的算术平均.
        - semantic_macro_f2: float, 逐 PDB 语义 F2 的算术平均.
        - topk_eligible_pdb_count: int, 至少含一个真实 occurrence 的 PDB 数量; top-K 成功比例使用该值作分母.

    每个覆盖阈值 t 的动态字段:
        - coverage_micro_precision_<t>: float, 全部已选候选中的多对多覆盖命中比例.
        - coverage_micro_recall_<t>: float, 全部 occurrence 中的多对多覆盖命中比例.
        - coverage_micro_f1_<t>: float, 由全局 coverage precision 和 recall 计算的 F1.
        - coverage_micro_f2_<t>: float, 由全局 coverage precision 和 recall 计算的 F2.
        - coverage_macro_f1_<t>: float, 逐 PDB coverage F1 的算术平均.
        - coverage_macro_f2_<t>: float, 逐 PDB coverage F2 的算术平均.
        - one_to_one_micro_precision_<t>: float, 全部最大一对一匹配数除以已选候选数.
        - one_to_one_micro_recall_<t>: float, 全部最大一对一匹配数除以 occurrence 数.
        - one_to_one_micro_f1_<t>: float, 由全局 one-to-one precision 和 recall 计算的 F1.
        - one_to_one_micro_f2_<t>: float, 由全局 one-to-one precision 和 recall 计算的 F2.
        - one_to_one_macro_f1_<t>: float, 逐 PDB one-to-one F1 的算术平均.
        - one_to_one_macro_f2_<t>: float, 逐 PDB one-to-one F2 的算术平均.
        - top<K>_success_count_<t>: int, 前 K 个已选候选至少覆盖一个 occurrence 的 PDB 数量.
        - top<K>_success_ratio_<t>: float, 前述数量除以 topk_eligible_pdb_count.

    阈值字段后缀把小数点替换为 `p`, 例如 0.3 写成 `0p3`. 任一指标分母为零时保存 0.0.
    """
    # thresholds: 长度 N_threshold 的双向覆盖阈值轴, 与每个 PdbEvaluation 的首轴一致.
    thresholds = tuple(float(value) for value in coverage_thresholds)
    # int64, (N_pdb, 3), 每个 PDB 的语义 TP, FP, FN; 空数据划分保持 (0, 3).
    semantic = np.asarray(
        [
            [item.semantic_tp, item.semantic_fp, item.semantic_fn]
            for item in evaluations
        ],
        dtype=np.int64,
    )
    if semantic.size == 0:
        semantic = np.zeros((0, 3), dtype=np.int64)
    # total_tp, total_fp, total_fn: 跨 PDB 汇总的语义体素计数.
    total_tp, total_fp, total_fn = semantic.sum(axis=0, dtype=np.int64).tolist()
    semantic_precision = (
        total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    )
    semantic_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    # report: 最终写入数据划分级 global_metrics.json 的标量字段映射.
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
        # per_pdb: 长度 N_pdb 的语义 F-beta, 用于 PDB 等权 macro 平均.
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

    # pred_total: 全部 PDB 的已选候选总数, 是预测侧 micro precision 的分母.
    pred_total = sum(int(item.candidate_selected.sum()) for item in evaluations)
    # gt_total: 全部 PDB 的真实 occurrence 总数, 是真实侧 micro recall 的分母.
    gt_total = sum(int(item.gt_sizes.size) for item in evaluations)
    # int64, (N_threshold,), 各阈值下达到多对多覆盖条件的已选候选总数.
    coverage_pred = (
        np.sum(np.stack([item.coverage_pred_hit for item in evaluations]), axis=0)
        if evaluations
        else np.zeros(len(thresholds), dtype=np.int64)
    )
    # int64, (N_threshold,), 各阈值下被已选候选覆盖的真实 occurrence 总数.
    coverage_gt = (
        np.sum(np.stack([item.coverage_gt_hit for item in evaluations]), axis=0)
        if evaluations
        else np.zeros(len(thresholds), dtype=np.int64)
    )
    # int64, (N_threshold,), 各阈值下最大一对一匹配的总数.
    one_to_one = (
        np.sum(np.stack([item.one_to_one_tp for item in evaluations]), axis=0)
        if evaluations
        else np.zeros(len(thresholds), dtype=np.int64)
    )

    for row, threshold in enumerate(thresholds):
        # tag: 当前阈值的 JSON 字段后缀, 例如 0.3 转为 0p3.
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

            # per_pdb_f1: 长度 N_pdb 的当前指标 F1, 用于 PDB 等权 macro 平均.
            per_pdb_f1: list[float] = []
            # per_pdb_f2: 长度 N_pdb 的当前指标 F2, 用于 PDB 等权 macro 平均.
            per_pdb_f2: list[float] = []
            for item in evaluations:
                if name == "coverage":
                    local_pred_hit = int(item.coverage_pred_hit[row])
                    local_gt_hit = int(item.coverage_gt_hit[row])
                else:
                    local_pred_hit = local_gt_hit = int(item.one_to_one_tp[row])
                local_pred_count = int(item.candidate_selected.sum())
                local_precision = (
                    local_pred_hit / local_pred_count if local_pred_count else 0.0
                )
                local_recall = (
                    local_gt_hit / item.gt_sizes.size if item.gt_sizes.size else 0.0
                )
                for beta, target in ((1.0, per_pdb_f1), (2.0, per_pdb_f2)):
                    beta2 = beta * beta
                    denominator = beta2 * local_precision + local_recall
                    target.append(
                        (1.0 + beta2) * local_precision * local_recall / denominator
                        if denominator
                        else 0.0
                    )
            report[f"{name}_macro_f1_{tag}"] = (
                float(np.mean(per_pdb_f1)) if per_pdb_f1 else 0.0
            )
            report[f"{name}_macro_f2_{tag}"] = (
                float(np.mean(per_pdb_f2)) if per_pdb_f2 else 0.0
            )

    # 只有含真实 occurrence 的 PDB 进入 top-K 成功比例分母.
    eligible_pdb = sum(int(item.gt_sizes.size > 0) for item in evaluations)
    report["topk_eligible_pdb_count"] = eligible_pdb
    for topk_row, topk in enumerate(topk_values):
        # int64, (N_threshold,), 当前 K 下至少命中一个 occurrence 的 PDB 数量.
        successes = (
            np.sum(
                np.stack([item.topk_success[topk_row] for item in evaluations]), axis=0
            )
            if evaluations
            else np.zeros(len(thresholds), dtype=np.int64)
        )
        for threshold_row, threshold in enumerate(thresholds):
            tag = f"{threshold:.3f}".rstrip("0").rstrip(".").replace(".", "p")
            report[f"top{int(topk)}_success_count_{tag}"] = int(
                successes[threshold_row]
            )
            report[f"top{int(topk)}_success_ratio_{tag}"] = (
                float(successes[threshold_row]) / eligible_pdb if eligible_pdb else 0.0
            )
    return report


def aggregate_semantic_prauc(
    histograms: Sequence[np.ndarray],
) -> dict[str, float]:
    """从逐 PDB 体素直方图计算语义 micro 与 PDB 等权 macro PRAUC.

    输入参数:
        - histograms: Sequence[np.ndarray], 每项是 :func:`semantic_prauc_histogram` 返回的 int64 ``(2, 1024)`` 计数.

    返回字段:
        - semantic_micro_prauc: float, 先合并全部 PDB 体素计数, 再按 1024 阈值 ``BinaryAveragePrecision`` 公式计算的 AP.
        - semantic_macro_prauc: float, 逐 PDB 计算同口径 AP 后的算术平均, 每个 PDB 权重相同.

    对升序阈值 ``t_j`` 计算 ``precision_j`` 与 ``recall_j``, 再以
    ``sum((recall_j - recall_{j+1}) * precision_j)`` 得到 AP, 其中
    ``recall_1024 = 0``. precision 或 recall 分母为零时相应值记为 0;
    空数据划分和没有正体素的单个 PDB 都返回 0.0.
    """

    if not histograms:
        return {"semantic_micro_prauc": 0.0, "semantic_macro_prauc": 0.0}
    # int64, (N_pdb, 2, 1024), 三轴依次是 PDB, 负/正体素和升序阈值概率档.
    per_pdb_histogram = np.stack(histograms).astype(np.int64, copy=False)
    # int64, (N_pdb + 1, 2, 1024), 第 0 项为 micro 总计数, 后续各项为单个 PDB; 末两轴仍是负/正体素与升序阈值概率档.
    combined_histogram = np.concatenate(
        (
            per_pdb_histogram.sum(axis=0, dtype=np.int64)[None, ...],
            per_pdb_histogram,
        ),
        axis=0,
    )
    # int64, (N_pdb + 1, 1024), 每个阈值下概率大于等于该阈值的真阳性体素数.
    true_positive = np.cumsum(
        combined_histogram[:, 1, ::-1],
        axis=1,
        dtype=np.int64,
    )[:, ::-1]
    # int64, (N_pdb + 1, 1024), 每个阈值下概率大于等于该阈值的假阳性体素数.
    false_positive = np.cumsum(
        combined_histogram[:, 0, ::-1],
        axis=1,
        dtype=np.int64,
    )[:, ::-1]
    # float64, (N_pdb + 1, 1024), 按阈值升序排列的 precision.
    precision = np.divide(
        true_positive,
        true_positive + false_positive,
        out=np.zeros(true_positive.shape, dtype=np.float64),
        where=(true_positive + false_positive) > 0,
    )
    # int64, (N_pdb + 1, 1), micro 总体与每个 PDB 的正体素数.
    positive_count = combined_histogram[:, 1].sum(
        axis=1,
        dtype=np.int64,
    )[:, None]
    # float64, (N_pdb + 1, 1024), 按阈值升序排列的 recall.
    recall = np.divide(
        true_positive,
        positive_count,
        out=np.zeros(true_positive.shape, dtype=np.float64),
        where=positive_count > 0,
    )
    # float64, (N_pdb + 1, 1024), 相邻阈值之间的 recall 下降量, 末项与 TorchMetrics 追加的 recall=0 端点对齐.
    recall_drop = recall.copy()
    recall_drop[:, :-1] -= recall[:, 1:]
    # float64, (N_pdb + 1,), 第 0 项是 micro AP, 后续各项是逐 PDB AP.
    prauc = np.sum(recall_drop * precision, axis=1, dtype=np.float64)
    return {
        "semantic_micro_prauc": float(prauc[0]),
        "semantic_macro_prauc": float(np.mean(prauc[1:])),
    }
