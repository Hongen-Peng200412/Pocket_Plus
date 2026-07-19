"""以 32769 个整数 bin 扫描 calibration micro-Fα 阈值。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from src.artifacts.io import atomic_savez_compressed, atomic_write_json, load_npz_strict
from src.artifacts.paths import Stage1ArtifactPaths
from src.artifacts.states import is_role_complete
from src.component_lineage.forest import build_component_forest
from src.datasets.stage1_requests import centered_start_from_centroid_zyx

from .instance_metrics import (
    aggregate_instance_counts,
    evaluate_instance_overlap_counts,
    evaluate_topk_overlap_counts,
)
from .voxel_metrics import average_precision_full_grid


DEFAULT_ALPHA_VALUES: tuple[float, ...] = (
    1.0 / 2.0,
    2.0 / 3.0,
    4.0 / 5.0,
    1.0,
    5.0 / 4.0,
    3.0 / 2.0,
    2.0,
)


@dataclass
class ThresholdHistogram:
    """
    累加固定分母阈值网格所需的正、负 voxel 直方图。

    输入参数:
        - denominator: int, 阈值分母；正式值为 32768，因此 bin 数为 32769

    属性:
        - positive_histogram: np.ndarray, (denominator+1,), int64，按
          `floor(probability*denominator)` 分箱的正类 voxel 数
        - negative_histogram: np.ndarray, (denominator+1,), int64，同口径负类数
    """

    denominator: int

    def __post_init__(self) -> None:
        if int(self.denominator) <= 0:
            raise ValueError("denominator 必须为正整数")
        self.denominator = int(self.denominator)
        self.positive_histogram = np.zeros(self.denominator + 1, dtype=np.int64)
        self.negative_histogram = np.zeros(self.denominator + 1, dtype=np.int64)

    def update(self, probability_map: np.ndarray, gt_union_mask: np.ndarray) -> None:
        """
        把一张完整图加入 calibration 直方图，不物化 voxel×threshold 矩阵。

        输入参数:
            - probability_map: np.ndarray, (D,H,W), 连续概率，必须位于 [0,1]
            - gt_union_mask: np.ndarray, (D,H,W), 全部 occurrence ligand-area 并集

        输出:
            - None, 原地更新两个 int64 直方图
        """
        probability = np.asarray(probability_map)
        target = np.asarray(gt_union_mask, dtype=np.bool_)
        if probability.shape != target.shape or probability.ndim != 3:
            raise ValueError("probability_map 与 gt_union_mask 必须是同 shape 的三维数组")
        if not bool(np.all(np.isfinite(probability))):
            raise ValueError("probability_map 含非有限值")
        if float(probability.min()) < 0.0 or float(probability.max()) > 1.0:
            raise ValueError("probability_map 必须位于 [0,1]")
        grid_index = np.floor(
            probability.astype(np.float64, copy=False) * float(self.denominator)
        ).astype(np.int64)
        np.clip(grid_index, 0, self.denominator, out=grid_index)
        self.positive_histogram += np.bincount(
            grid_index[target], minlength=self.denominator + 1
        ).astype(np.int64, copy=False)
        self.negative_histogram += np.bincount(
            grid_index[~target], minlength=self.denominator + 1
        ).astype(np.int64, copy=False)

    def threshold_counts(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        返回全部 `j=0..denominator` 的 micro TP/FP/FN。

        输出:
            - tp: np.ndarray, (denominator+1,), int64，各阈值预测正中的 GT 正类数
            - fp: np.ndarray, (denominator+1,), int64，各阈值预测正中的 GT 负类数
            - fn: np.ndarray, (denominator+1,), int64，各阈值漏掉的 GT 正类数
        """
        tp = np.cumsum(self.positive_histogram[::-1], dtype=np.int64)[::-1]
        fp = np.cumsum(self.negative_histogram[::-1], dtype=np.int64)[::-1]
        fn = int(self.positive_histogram.sum()) - tp
        return tp, fp, fn


@dataclass(frozen=True)
class ThresholdCalibrationResult:
    """
    保存七个 Fα 的冻结阈值和完整扫描曲线。

    输入参数:
        - denominator: int, 阈值整数分母
        - alpha_values: np.ndarray, (7,), float64，固定 alpha 顺序
        - alpha_threshold_grid_index: np.ndarray, (7,), int32，各 alpha 首个最大值 j
        - t_alpha: np.ndarray, (7,), float32，各项严格等于 j/denominator
        - t_F1: float, alpha=1 对应阈值
        - f_alpha_curve: np.ndarray, (7,denominator+1), float64，micro-Fα 曲线
        - tp/fp/fn: np.ndarray, (denominator+1,), int64，全部阈值计数
    """

    denominator: int
    alpha_values: np.ndarray
    alpha_threshold_grid_index: np.ndarray
    t_alpha: np.ndarray
    t_F1: float
    f_alpha_curve: np.ndarray
    tp: np.ndarray
    fp: np.ndarray
    fn: np.ndarray

    def thresholds_payload(
        self,
        stage1_model_name: str,
        min_voxels: int,
        max_voxels: int,
    ) -> dict[str, object]:
        """
        构造 `thresholds.json` 的冷读字段。

        输入参数:
            - stage1_model_name: str, 当前 producer 正式名
            - min_voxels: int, 冻结最小组件体素数
            - max_voxels: int, 冻结最大组件体素数

        输出:
            - payload: dict[str,object], 与 BOX 契约逐字段对应的 JSON 内容
        """
        return {
            "stage1_model_name": str(stage1_model_name),
            "denominator": int(self.denominator),
            "alpha_values": [float(value) for value in self.alpha_values],
            "alpha_threshold_grid_index": [
                int(value) for value in self.alpha_threshold_grid_index
            ],
            "t_alpha": [float(value) for value in self.t_alpha],
            "t_F1": float(self.t_F1),
            "min_voxels": int(min_voxels),
            "max_voxels": int(max_voxels),
            "connectivity": 26,
        }


def calibrate_thresholds(
    probability_and_target: Iterable[tuple[np.ndarray, np.ndarray]],
    denominator: int,
    alpha_values: Sequence[float] = DEFAULT_ALPHA_VALUES,
) -> ThresholdCalibrationResult:
    """
    汇总 calibration 全体 PDB/voxel 并选择每个 micro-Fα 的首个最大阈值。

    输入参数:
        - probability_and_target: Iterable[(np.ndarray,np.ndarray)]，每项为同 shape 的
          完整图 probability 与 occurrence-union GT；只遍历一次
        - denominator: int, 固定阈值分母，正式值为 32768
        - alpha_values: Sequence[float], 固定七个 alpha，顺序写入阈值表

    输出:
        - result: ThresholdCalibrationResult, 使用 int64 直方图反向累积得到，
          并按 `j=0..denominator` 升序扫描的第一个最大值处理并列
    """
    histogram = ThresholdHistogram(denominator=int(denominator))
    sample_count = 0
    for probability_map, target in probability_and_target:
        histogram.update(probability_map, target)
        sample_count += 1
    if sample_count == 0:
        raise ValueError("calibration 至少需要一个 PDB")
    alphas = np.asarray(alpha_values, dtype=np.float64)
    if alphas.ndim != 1 or alphas.size == 0 or bool(np.any(alphas <= 0)):
        raise ValueError("alpha_values 必须是一维正数序列")
    alpha_one_rows = np.flatnonzero(np.isclose(alphas, 1.0, rtol=0.0, atol=1e-12))
    if alpha_one_rows.size != 1:
        raise ValueError("alpha_values 必须恰好包含一个 alpha=1")

    tp, fp, fn = histogram.threshold_counts()
    precision = np.divide(
        tp,
        tp + fp,
        out=np.zeros(tp.shape, dtype=np.float64),
        where=(tp + fp) > 0,
    )
    recall = np.divide(
        tp,
        tp + fn,
        out=np.zeros(tp.shape, dtype=np.float64),
        where=(tp + fn) > 0,
    )
    curves: list[np.ndarray] = []
    best_indices: list[int] = []
    for alpha in alphas:
        alpha_squared = float(alpha) ** 2
        numerator = (1.0 + alpha_squared) * precision * recall
        denominator_values = alpha_squared * precision + recall
        curve = np.divide(
            numerator,
            denominator_values,
            out=np.zeros(numerator.shape, dtype=np.float64),
            where=denominator_values > 0,
        )
        curves.append(curve)
        best_indices.append(int(np.argmax(curve)))
    best = np.asarray(best_indices, dtype=np.int32)
    thresholds = (best.astype(np.float64) / float(denominator)).astype(np.float32)
    f1_row = int(alpha_one_rows[0])
    return ThresholdCalibrationResult(
        denominator=int(denominator),
        alpha_values=alphas,
        alpha_threshold_grid_index=best,
        t_alpha=thresholds,
        t_F1=float(thresholds[f1_row]),
        f_alpha_curve=np.stack(curves, axis=0),
        tp=tp,
        fp=fp,
        fn=fn,
    )


def publish_threshold_calibration(
    paths: Stage1ArtifactPaths,
    result: ThresholdCalibrationResult,
    min_voxels: int,
    max_voxels: int,
    fitted_metrics: dict[str, object],
) -> None:
    """
    原子发布冻结阈值、完整扫描数组、calibration-fitted 指标与最终 `_COMPLETE`。

    输入参数:
        - paths: Stage1ArtifactPaths, 使用其 producer 级 calibration 路径；split/PDB
          字段不参与 calibration 寻址
        - result: ThresholdCalibrationResult, 七个首个最大值与完整扫描计数
        - min_voxels: int, 正式组件最小体素数
        - max_voxels: int, 用户冻结的 Q95×1.5 整数
        - fitted_metrics: dict[str,object], macro AP、Dice、coverage、one-to-one、top-K
          等同一 calibration 上的结果；必须由调用方明确标注其科学字段

    输出:
        - None, `thresholds.json`、`threshold_scan.npz`、`metrics.json` 全部关闭并
          校验后，最后写 producer calibration `_COMPLETE`
    """
    threshold_payload = result.thresholds_payload(
        stage1_model_name=paths.stage1_model_name,
        min_voxels=min_voxels,
        max_voxels=max_voxels,
    )
    atomic_write_json(paths.thresholds_json, threshold_payload)

    scan_arrays = {
        "denominator": np.asarray(result.denominator, dtype=np.int32),
        "alpha_values": np.asarray(result.alpha_values, dtype=np.float64),
        "f_alpha_curve": np.asarray(result.f_alpha_curve, dtype=np.float64),
        "tp": np.asarray(result.tp, dtype=np.int64),
        "fp": np.asarray(result.fp, dtype=np.int64),
        "fn": np.asarray(result.fn, dtype=np.int64),
    }

    def validate_scan(arrays: dict[str, np.ndarray]) -> None:
        expected_length = int(result.denominator) + 1
        if np.asarray(arrays["f_alpha_curve"]).shape != (
            int(result.alpha_values.size),
            expected_length,
        ):
            raise ValueError("f_alpha_curve shape 与 alpha/denominator 不一致")
        for field in ("tp", "fp", "fn"):
            if np.asarray(arrays[field]).shape != (expected_length,):
                raise ValueError(f"{field} 长度必须为 denominator+1")

    atomic_savez_compressed(paths.threshold_scan_npz, scan_arrays, validator=validate_scan)
    from .report import to_json_values

    metrics_payload = to_json_values({
        "stage1_model_name": paths.stage1_model_name,
        "result_scope": "calibration_fitted",
        "threshold_scan": paths.threshold_scan_npz.name,
        **fitted_metrics,
    })
    atomic_write_json(paths.calibration_metrics_json, metrics_payload)
    atomic_write_json(
        paths.calibration_complete_path,
        {
            "stage1_model_name": paths.stage1_model_name,
            "result_scope": "calibration_fitted",
        },
    )


OccurrenceVoxelLoader = Callable[[str, tuple[int, int, int]], Mapping[int, np.ndarray]]


def calibrate_published_full_maps_and_freeze_thresholds(
    output_root: str | Path,
    stage1_model_name: str,
    calibration_pdb_ids: Sequence[str],
    occurrence_voxel_loader: OccurrenceVoxelLoader,
    min_voxels: int,
    max_voxels: int,
    denominator: int = 32768,
    split: str = "calibration",
) -> tuple[ThresholdCalibrationResult, dict[str, object]]:
    """
    从已发布 calibration probability 端到端冻结阈值并发布完整 fitted 报告。

    输入参数:
        - output_root: str | Path，`stage1_outputs` 根目录
        - stage1_model_name: str，当前 producer 正式名
        - calibration_pdb_ids: Sequence[str]，固定 calibration 清单顺序
        - occurrence_voxel_loader: Callable，接收 `(pdb_id,full_shape_zyx)`，返回
          `occurrence_id -> 全图 C-order linear voxel index`；只允许非负唯一 index
        - min_voxels: int，正式组件最小体素数，当前为 32
        - max_voxels: int，已在代码外冻结的 Q95×1.5 上限
        - denominator: int，整数阈值分母，正式值为 32768
        - split: str，calibration probability 所在 split，正式值为 `calibration`

    输出:
        - result: ThresholdCalibrationResult，七个 first-maximum 阈值与完整扫描曲线
        - fitted_metrics: dict[str,object]，full-grid macro AP、t_F1 micro Dice、固定
          Hungarian coverage/one-to-one F1、top-3/4/5 success ratio 与样本计数

    每张概率图只在需要时从已 `_COMPLETE` 的正式 NPZ 读取。第一遍累计 32769-bin
    直方图，第二遍在冻结 `t_F1` 上临时构造单层 26-CCL；不发布 components，因而
    保持“calibration probability 与阈值先完成，阶段二再回填各 role”的时序。
    """
    pdb_ids = tuple(str(pdb_id) for pdb_id in calibration_pdb_ids)
    if len(pdb_ids) == 0 or len(set(pdb_ids)) != len(pdb_ids):
        raise ValueError("calibration_pdb_ids 必须是非空且唯一的固定清单")

    def threshold_inputs() -> Iterable[tuple[np.ndarray, np.ndarray]]:
        for pdb_id in pdb_ids:
            probability, occurrence_voxels = _load_published_calibration_sample(
                output_root=output_root,
                stage1_model_name=stage1_model_name,
                split=split,
                pdb_id=pdb_id,
                occurrence_voxel_loader=occurrence_voxel_loader,
            )
            yield probability, _union_mask(probability.shape, occurrence_voxels)

    result = calibrate_thresholds(
        probability_and_target=threshold_inputs(),
        denominator=int(denominator),
    )
    fitted_metrics = _evaluate_frozen_calibration(
        output_root=output_root,
        stage1_model_name=stage1_model_name,
        split=split,
        pdb_ids=pdb_ids,
        occurrence_voxel_loader=occurrence_voxel_loader,
        result=result,
        min_voxels=int(min_voxels),
        max_voxels=int(max_voxels),
    )
    calibration_paths = Stage1ArtifactPaths(
        output_root=Path(output_root),
        stage1_model_name=stage1_model_name,
        split=split,
        pdb_id=pdb_ids[0],
    )
    publish_threshold_calibration(
        paths=calibration_paths,
        result=result,
        min_voxels=int(min_voxels),
        max_voxels=int(max_voxels),
        fitted_metrics=fitted_metrics,
    )
    return result, fitted_metrics


def _load_published_calibration_sample(
    output_root: str | Path,
    stage1_model_name: str,
    split: str,
    pdb_id: str,
    occurrence_voxel_loader: OccurrenceVoxelLoader,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """读取一个已发布 probability，并规范化该 PDB 的 occurrence 稀疏体素集。"""
    paths = Stage1ArtifactPaths(
        output_root=Path(output_root),
        stage1_model_name=stage1_model_name,
        split=split,
        pdb_id=pdb_id,
    )
    if paths.running_dir.exists() or not is_role_complete(paths, "probability"):
        raise RuntimeError(f"calibration probability 尚不可消费: {pdb_id}")
    probability = np.asarray(
        load_npz_strict(paths.probability_npz)["probability_map"], dtype=np.float32
    )
    if probability.ndim != 3 or not bool(np.all(np.isfinite(probability))):
        raise ValueError(f"{pdb_id}: probability_map 必须是有限 float32 三维网格")
    voxel_count = int(probability.size)
    raw_occurrences = occurrence_voxel_loader(
        pdb_id, tuple(int(value) for value in probability.shape)
    )
    occurrences: dict[int, np.ndarray] = {}
    for occurrence_id, indices in raw_occurrences.items():
        identity = int(occurrence_id)
        values = np.asarray(indices, dtype=np.int64).reshape(-1)
        if values.size and (
            int(values.min()) < 0 or int(values.max()) >= voxel_count
        ):
            raise ValueError(f"{pdb_id}/{identity}: occurrence voxel index 越过完整图")
        unique = np.unique(values)
        if unique.size != values.size:
            raise ValueError(f"{pdb_id}/{identity}: occurrence voxel index 必须唯一")
        occurrences[identity] = unique
    return probability, dict(sorted(occurrences.items()))


def _union_mask(
    full_shape_zyx: tuple[int, int, int],
    occurrence_voxels: Mapping[int, np.ndarray],
) -> np.ndarray:
    """从 occurrence 稀疏线性体素构造完整图 union ligand-area bool mask。"""
    target = np.zeros(int(np.prod(full_shape_zyx)), dtype=np.bool_)
    for indices in occurrence_voxels.values():
        target[np.asarray(indices, dtype=np.int64)] = True
    return target.reshape(full_shape_zyx)


def _evaluate_frozen_calibration(
    output_root: str | Path,
    stage1_model_name: str,
    split: str,
    pdb_ids: Sequence[str],
    occurrence_voxel_loader: OccurrenceVoxelLoader,
    result: ThresholdCalibrationResult,
    min_voxels: int,
    max_voxels: int,
) -> dict[str, object]:
    """在冻结 `t_F1` 上汇总 calibration fitted voxel 与 instance 指标。"""
    alpha_one_row = int(
        np.flatnonzero(np.isclose(result.alpha_values, 1.0, rtol=0.0, atol=1e-12))[0]
    )
    f1_grid_index = int(result.alpha_threshold_grid_index[alpha_one_row])
    ap_values: list[float] = []
    instance_counts = []
    topk_totals: dict[str, int] = {}
    blob_exceed_count = 0

    for pdb_id in pdb_ids:
        probability, occurrence_voxels = _load_published_calibration_sample(
            output_root=output_root,
            stage1_model_name=stage1_model_name,
            split=split,
            pdb_id=pdb_id,
            occurrence_voxel_loader=occurrence_voxel_loader,
        )
        target = _union_mask(probability.shape, occurrence_voxels)
        ap = average_precision_full_grid(probability, target)
        if ap is not None:
            ap_values.append(float(ap))
        forest, _ = build_component_forest(
            probability_map=probability,
            threshold_grid_indices=(f1_grid_index,),
            denominator=int(result.denominator),
            min_voxels=int(min_voxels),
            max_voxels=int(max_voxels),
            resolve_box_start=centered_start_from_centroid_zyx,
        )
        predictions = [
            node
            for node in forest.nodes
            if node.candidate_eligible
            and node.threshold_grid_index == f1_grid_index
        ]
        if len(predictions) > 200:
            blob_exceed_count += 1
        occurrence_rows = tuple(occurrence_voxels.values())
        intersections = np.asarray(
            [
                [
                    np.intersect1d(
                        node.voxel_global_linear_index,
                        gt_indices,
                        assume_unique=True,
                    ).size
                    for gt_indices in occurrence_rows
                ]
                for node in predictions
            ],
            dtype=np.int64,
        ).reshape(len(predictions), len(occurrence_rows))
        pred_sizes = np.asarray(
            [node.voxel_global_linear_index.size for node in predictions],
            dtype=np.int64,
        )
        gt_sizes = np.asarray(
            [indices.size for indices in occurrence_rows], dtype=np.int64
        )
        instance_counts.append(
            evaluate_instance_overlap_counts(
                intersections=intersections,
                pred_sizes=pred_sizes,
                gt_sizes=gt_sizes,
                coverage_thresholds=(0.3, 0.5),
            )
        )
        topk = evaluate_topk_overlap_counts(
            intersections=intersections,
            pred_sizes=pred_sizes,
            gt_sizes=gt_sizes,
            candidate_scores=[node.probability_mean for node in predictions],
            topk_values=(3, 4, 5),
            coverage_thresholds=(0.3, 0.5),
        )
        for field, value in topk.items():
            topk_totals[field] = topk_totals.get(field, 0) + int(value)

    total_instance = aggregate_instance_counts(instance_counts)
    tp = int(result.tp[f1_grid_index])
    fp = int(result.fp[f1_grid_index])
    fn = int(result.fn[f1_grid_index])
    dice_denominator = 2 * tp + fp + fn
    topk_denominator = int(topk_totals.pop("n_topk_eligible_pdb", 0))
    topk_metrics: dict[str, object] = {
        "n_topk_eligible_pdb": topk_denominator,
    }
    for field, value in topk_totals.items():
        topk_metrics[field] = int(value)
        topk_metrics[field.replace("_success_", "_success_ratio_")] = (
            float(value) / float(topk_denominator) if topk_denominator else 0.0
        )
    return {
        "voxel_average_precision_macro": (
            float(np.mean(ap_values)) if ap_values else float("nan")
        ),
        "n_valid_voxel_ap_pdb": len(ap_values),
        "n_total_pdb": len(pdb_ids),
        "semantic_dice_t_F1": (
            float(2 * tp) / float(dice_denominator) if dice_denominator else 0.0
        ),
        "semantic_tp_t_F1": tp,
        "semantic_fp_t_F1": fp,
        "semantic_fn_t_F1": fn,
        "n_blob_exceed_pdb": int(blob_exceed_count),
        **total_instance.metrics(),
        **topk_metrics,
    }
