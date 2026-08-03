"""用固定整数直方图冻结 calibration micro-Fα 阈值并发布 fitted 指标。

主要入口:
    - `ThresholdHistogram`: 把每个完整图 voxel 的概率映射到 `j=floor(p*D_threshold)`，分别累加真实正类和负类直方图。
    - `calibrate_thresholds`: 反向累积全部 j 的 micro TP/FP/FN，按 j 升序选择每条 Fα 曲线的首个最大值。
    - `calibrate_published_full_maps_and_freeze_thresholds`: 两遍消费已完成的 calibration probability；第一遍冻结阈值，第二遍在 `t_F1` 上构造临时单层组件并计算 fitted 指标。
    - `publish_threshold_calibration`: 发布 `thresholds.json`、`threshold_scan.npz`、`metrics.json` 和 producer 级 `_COMPLETE`。

正式分母为 32768，因此共有 32769 个整数 bin。直方图方法避免物化 voxel×threshold 大矩阵；第二遍临时 forest 不发布 components，保持先完成 probability 与阈值、再由后续阶段补齐 PDB roles 的时序。
"""

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
from .voxel_metrics import average_precision_full_grid, semantic_dice


# 七个正式 Fα 权重，顺序同时决定 `thresholds.json` 和 `threshold_scan.npz` 的 alpha 第一维；其中恰有一个 alpha=1。
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
        - denominator: int, 概率阈值分母 D_threshold；整数网格 j 的范围为 `0..D_threshold`，正式值为 32768。

    属性:
        - positive_histogram: int64, (D_threshold + 1,), 按 `floor(probability*D_threshold)` 分箱的真实正 voxel 数。
        - negative_histogram: int64, (D_threshold + 1,), 使用相同概率分箱规则的真实负 voxel 数。
    """
    denominator: int

    def __post_init__(self) -> None:
        """
        校验阈值分母并初始化正负 voxel 直方图。

        输出:
            - None, 原地建立两个 `(D_threshold + 1,)` int64 零直方图。
        """
        if int(self.denominator) <= 0:
            raise ValueError("denominator 必须为正整数")
        self.denominator = int(self.denominator)
        self.positive_histogram = np.zeros(self.denominator + 1, dtype=np.int64)
        self.negative_histogram = np.zeros(self.denominator + 1, dtype=np.int64)

    def update(self, probability_map: np.ndarray, gt_union_mask: np.ndarray) -> None:
        """
        把一张完整图加入 calibration 直方图以进行更新。

        输入参数:
            - probability_map: numeric, (D, H, W), 完整 ZYX voxel 网格上的连续概率；所有值必须有限且位于 `[0, 1]`。
            - gt_union_mask: bool, (D, H, W), 同一完整网格上的真实 occurrence 配体区域并集。

        输出:
            - None, 原地更新两个 int64 直方图
        """
        # numeric, (D, H, W), 当前 calibration PDB 的连续完整图概率，轴顺序 ZYX。
        probability = np.asarray(probability_map)
        # bool, (D, H, W), 当前 PDB 全部真实 occurrence 配体区域的完整图并集标签。
        target = np.asarray(gt_union_mask, dtype=np.bool_)
        if probability.shape != target.shape or probability.ndim != 3:
            raise ValueError("probability_map 与 gt_union_mask 必须是同 shape 的三维数组")
        if not bool(np.all(np.isfinite(probability))):
            raise ValueError("probability_map 含非有限值")
        if float(probability.min()) < 0.0 or float(probability.max()) > 1.0:
            raise ValueError("probability_map 必须位于 [0,1]")
        # int64, (D, H, W), 每个 voxel 的整数概率 bin `j=floor(probability*D_threshold)`，随后裁到闭区间 `[0, D_threshold]`。
        grid_index = np.floor(probability.astype(np.float64, copy=False) * float(self.denominator)).astype(np.int64)
        np.clip(grid_index, 0, self.denominator, out=grid_index)
        self.positive_histogram += np.bincount(grid_index[target], minlength=self.denominator + 1).astype(np.int64, copy=False)
        self.negative_histogram += np.bincount(grid_index[~target], minlength=self.denominator + 1).astype(np.int64, copy=False)

    def threshold_counts(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        返回全部 `j=0..denominator` 的 micro TP/FP/FN。

        输出:
            - tp: int64, (D_threshold + 1,), 阈值 j 下满足 `probability >= j/D_threshold` 的真实正 voxel 数。
            - fp: int64, (D_threshold + 1,), 阈值 j 下满足 `probability >= j/D_threshold` 的真实负 voxel 数。
            - fn: int64, (D_threshold + 1,), 阈值 j 下概率低于阈值的真实正 voxel 数。
        """
        # int64, (D_threshold + 1,), 从高 bin 反向累积得到的真实正预测数 TP(j)。
        tp = np.cumsum(self.positive_histogram[::-1], dtype=np.int64)[::-1]
        # int64, (D_threshold + 1,), 从高 bin 反向累积得到的真实负预测数 FP(j)。
        fp = np.cumsum(self.negative_histogram[::-1], dtype=np.int64)[::-1]
        # int64, (D_threshold + 1,), 总真实正 voxel 数减去 TP(j) 得到的 FN(j)。
        fn = int(self.positive_histogram.sum()) - tp
        return tp, fp, fn

@dataclass(frozen=True)
class ThresholdCalibrationResult:
    """
    保存七个 Fα 的冻结阈值和完整扫描曲线。

    输入参数:
        - denominator: int, 概率阈值整数分母 D_threshold。
        - alpha_values: float64, (N_alpha,), 固定 Fα 权重顺序, 可重复。
        - alpha_threshold_grid_index: int32, (N_alpha,), 每个 Fα 曲线按 j 升序遇到的首个最大值网格整数, 可重复。
        - t_alpha: float32, (N_alpha,), 每个 alpha 的冻结概率阈值，逐项严格由 `j/D_threshold` 生成。
        - t_F1: float, 唯一 `alpha=1` 对应的冻结阈值。
        - f_alpha_curve: float64, (N_alpha, D_threshold + 1), 每个 alpha 在全部整数阈值 j 上的 micro-Fα 曲线。
        - tp: int64, (D_threshold + 1,), 全 calibration voxel 在每个阈值 j 上的 true-positive 计数。
        - fp: int64, (D_threshold + 1,), 全 calibration voxel 在每个阈值 j 上的 false-positive 计数。
        - fn: int64, (D_threshold + 1,), 全 calibration voxel 在每个阈值 j 上的 false-negative 计数。
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

        输出字段:
            - stage1_model_name: str, 当前冻结阈值所属的 producer。
            - denominator: int, 概率阈值整数分母 D_threshold。
            - alpha_values: float64, (N_alpha,), 固定 Fα 权重顺序。
            - alpha_threshold_grid_index: int32, (N_alpha,), 每个 Fα 曲线按 j 升序遇到的首个最大值网格整数。
            - t_alpha: float32, (N_alpha,), 每个 alpha 的冻结概率阈值，逐项严格由 `j/D_threshold` 生成。
            - t_F1: float, 唯一 `alpha=1` 对应的冻结阈值。
            - min_voxels: int, 正式候选组件最小 voxel 数。
            - max_voxels: int, 正式候选组件最大 voxel 数。
            - connectivity: int, 固定为 26 的三维连通邻域。
        """
        return {
            "stage1_model_name": str(stage1_model_name),
            "denominator": int(self.denominator),
            "alpha_values": [float(value) for value in self.alpha_values],
            "alpha_threshold_grid_index": [int(value) for value in self.alpha_threshold_grid_index],
            "t_alpha": [float(value) for value in self.t_alpha],
            "t_F1": float(self.t_F1),
            "min_voxels": int(min_voxels),
            "max_voxels": int(max_voxels),
            "connectivity": 26,
        }


########## 核心计算函数 ##########
def calibrate_thresholds(
    probability_and_target: Iterable[tuple[np.ndarray, np.ndarray]],
    denominator: int,
    alpha_values: Sequence[float] = DEFAULT_ALPHA_VALUES,
) -> ThresholdCalibrationResult:
    """
    汇总 calibration 全体 PDB/voxel 并选择每个 micro-Fα 的首个最大阈值。

    输入参数:
        - probability_and_target: Iterable[tuple[np.ndarray, np.ndarray]], 每项为同 shape 的完整 ZYX 概率图和 occurrence 配体区域并集；迭代器只消费一次。
        - denominator: int, 固定阈值分母 D_threshold，正式值为 32768。
        - alpha_values: Sequence[float], 长度 N_alpha 的正 Fα 权重；必须恰好包含一个 alpha=1，顺序写入全部产物。

    输出:
        - result: ThresholdCalibrationResult, 校准结果对象。
            - denominator: int, 概率阈值网格的固定分母 `D_threshold`。
            - alpha_values: float64, `(N_alpha,)`，按产物顺序排列的 Fα 权重，恰好包含一个 `alpha=1`。
            - alpha_threshold_grid_index: int32, `(N_alpha,)`，各 alpha 曲线并列最大值对应的首个网格位置 `j`。
            - t_alpha: float32, `(N_alpha,)`，各 alpha 的冻结概率阈值，逐项等于 `j/D_threshold`。
            - t_F1: float, `alpha=1` 对应的冻结概率阈值。
            - f_alpha_curve: float64, `(N_alpha, D_threshold + 1)`，各 alpha 在全部整数阈值 `j=0..D_threshold` 上的 micro-Fα 曲线。
            - tp: int64, `(D_threshold + 1,)`，各整数阈值下的真阳性体素计数。
            - fp: int64, `(D_threshold + 1,)`，各整数阈值下的假阳性体素计数。
            - fn: int64, `(D_threshold + 1,)`，各整数阈值下的假阴性体素计数。
    """
    # ThresholdHistogram, 汇总全部 calibration PDB 正/负 voxel 的整数概率分箱，不保存单 PDB 网格。
    histogram = ThresholdHistogram(denominator=int(denominator))
    sample_count = 0
    for probability_map, target in probability_and_target:
        histogram.update(probability_map, target)
        sample_count += 1
    if sample_count == 0:
        raise ValueError("calibration 至少需要一个 PDB")
    # float64, (N_alpha,), 固定顺序的 Fα 权重；该顺序贯穿阈值、曲线与发布字段。
    alphas = np.asarray(alpha_values, dtype=np.float64)
    if alphas.ndim != 1 or alphas.size == 0 or bool(np.any(alphas <= 0)):
        raise ValueError("alpha_values 必须是一维正数序列")
    alpha_one_rows = np.flatnonzero(np.isclose(alphas, 1.0, rtol=0.0, atol=1e-12))
    if alpha_one_rows.size != 1:
        raise ValueError("alpha_values 必须恰好包含一个 alpha=1")

    # 三个 int64 `(D_threshold + 1,)` 数组，全 calibration voxel 在每个整数阈值 j 上的 micro TP、FP、FN。
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
    # list[float64 array], 长度 N_alpha；每项 `(D_threshold + 1,)` 保存一个 alpha 的完整 micro-Fα 曲线。
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
    # int32, (N_alpha,), `np.argmax` 在 j 升序曲线上返回的首个最大值整数网格下标。
    best = np.asarray(best_indices, dtype=np.int32)
    # float32, (N_alpha,), 与 `best` 逐 alpha 对齐的冻结概率阈值 j/D_threshold。
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

# 打包保存
def publish_threshold_calibration(
    paths: Stage1ArtifactPaths,
    result: ThresholdCalibrationResult,
    min_voxels: int,
    max_voxels: int,
    fitted_metrics: dict[str, object],
) -> None:
    """
    不涉及任何计算的纯打包保存函数: 原子发布冻结阈值、完整扫描数组、calibration-fitted 指标与最终 `_COMPLETE`。

    输入参数:
        - paths: Stage1ArtifactPaths, 使用其 producer 级 calibration 路径；`split` 和 `pdb_id` 不参与 calibration 寻址。
        - result: ThresholdCalibrationResult, 七个首个最大值阈值、完整 Fα 扫描曲线及 TP/FP/FN 计数。
        - min_voxels: int, 冻结的正式候选组件最小 voxel 数。
        - max_voxels: int, 冻结的正式候选组件最大 voxel 数，为真实 occurrence 体积 Q95×3.0 的整数结果。
        - fitted_metrics: dict[str, object], 同一 calibration 数据上的 macro AP、Dice、coverage、one-to-one 与 top-K 指标；调用方负责提供明确科学字段名。

    落盘产物:
        - `thresholds.json`: JSON，保存 producer、分母、alpha、冻结网格下标、物理阈值、`t_F1`、组件体积上下限和 26-连通约定。
        - `threshold_scan.npz`: NPZ，保存 `denominator/alpha_values/f_alpha_curve/tp/fp/fn` 的完整扫描基础事实。
        - `metrics.json`: JSON，保存 `result_scope=calibration_fitted`、扫描文件名和 fitted 指标。
        - `calibration/_COMPLETE`: JSON，在前三个文件都原子发布后最后写入。
    """
    # dict[str, object], 可由冷读消费者直接恢复阈值和组件构造参数的 `thresholds.json` 顶层字段。
    threshold_payload = result.thresholds_payload(
        stage1_model_name=paths.stage1_model_name,
        min_voxels=min_voxels,
        max_voxels=max_voxels,
    )
    atomic_write_json(paths.thresholds_json, threshold_payload)

    # dict[str, np.ndarray], `threshold_scan.npz` 的标量分母、alpha 表、完整曲线和三张 micro 计数表。
    scan_arrays = {
        "denominator": np.asarray(result.denominator, dtype=np.int32),
        "alpha_values": np.asarray(result.alpha_values, dtype=np.float64),
        "f_alpha_curve": np.asarray(result.f_alpha_curve, dtype=np.float64),
        "tp": np.asarray(result.tp, dtype=np.int64),
        "fp": np.asarray(result.fp, dtype=np.int64),
        "fn": np.asarray(result.fn, dtype=np.int64),
    }

    def validate_scan(arrays: dict[str, np.ndarray]) -> None:
        """
        校验临时重读的 threshold scan 数组 shape。

        输入参数:
            - arrays: dict[str, np.ndarray], 临时重读的 `threshold_scan.npz` 字段；包含分母、alpha、Fα 曲线和 TP/FP/FN。

        输出:
            - None: 曲线与计数长度均为 `denominator+1` 时返回
        """
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

    # dict[str, Any], 已递归转换为标准 JSON 类型的 calibration fitted 指标顶层对象。
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








# ======================================================== 打包保存 & 关于 F1 的metric汇总 ========================================================
# 可注入的 GT occurrence 读取接口：输入 PDB 身份和完整图 ZYX 形状，返回每个真实配体 occurrence 的完整图 C-order 线性体素索引。
OccurrenceVoxelLoader = Callable[[str, tuple[int, int, int]], Mapping[int, np.ndarray]]

########## 核心打包函数 ##########
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
        - output_root: str | Path, `stage1_outputs` 根目录。
        - stage1_model_name: str, 当前冻结阈值所属的正式 producer。
        - calibration_pdb_ids: Sequence[str], 非空且唯一的固定 calibration PDB 顺序；两遍读取都严格复用该顺序。
        - occurrence_voxel_loader: OccurrenceVoxelLoader, 真实标注（ground truth，GT）配体 occurrence 的体素读取器；接收 `(pdb_id, full_shape_zyx)`，返回 `occurrence_id -> int64 (K_occ,)` 完整图 C-order 线性体素索引，每个索引必须非负、唯一且不越界。
        - min_voxels: int, 正式候选组件最小 voxel 数，当前为 10。
        - max_voxels: int, 已在本函数外冻结的真实 occurrence 体积 Q95×3.0 上限。
        - denominator: int, 概率阈值整数分母 D_threshold，正式值为 32768。
        - split: str, calibration probability 所在数据划分，正式值为 `calibration`。

    当前唯一正式装配位置是 `src.inference.cli.main` 的 `freeze-thresholds` 分支，它传入 `AGOccurrenceVoxelLoader(data_root)`，由该对象读取 `density/{pdb_id}/ligand_area.npz` 的 `mask_{occurrence_id}` GT 稀疏 ZYX 坐标；模型预测来自另行读取的 `probability_map.npz`，不由这个 loader 产生。校准先扫描阈值、再计算冻结阈值指标，所以每个 PDB 的 loader 在一次校准中会被调用两次。

    输出:
        - result: ThresholdCalibrationResult, 冻结阈值与完整阈值扫描结果。
            - denominator: int, 概率阈值网格的固定分母 `D_threshold`。
            - alpha_values: float64, `(N_alpha,)`，按产物顺序排列的 Fα 权重，恰好包含一个 `alpha=1`。
            - alpha_threshold_grid_index: int32, `(N_alpha,)`，各 alpha 曲线并列最大值对应的首个网格位置 `j`。
            - t_alpha: float32, `(N_alpha,)`，各 alpha 的冻结概率阈值，逐项等于 `j/D_threshold`。
            - t_F1: float, `alpha=1` 对应的冻结概率阈值。
            - f_alpha_curve: float64, `(N_alpha, D_threshold + 1)`，各 alpha 在全部整数阈值 `j=0..D_threshold` 上的 micro-Fα 曲线。
            - tp: int64, `(D_threshold + 1,)`，各整数阈值下的真阳性体素计数。
            - fp: int64, `(D_threshold + 1,)`，各整数阈值下的假阳性体素计数。
            - fn: int64, `(D_threshold + 1,)`，各整数阈值下的假阴性体素计数。

        - fitted_metrics: dict[str, object], 冻结 `t_F1` 后汇总的 voxel、semantic、instance 和 top-K 指标。
            - `voxel_average_precision_macro`: float, 有效 PDB 的完整图 voxel AP 等权平均值。
            - `n_valid_voxel_ap_pdb`: int, 参与 voxel AP 等权平均的有效 PDB 数。
            - `n_total_pdb`: int, calibration PDB 总数。
            - `semantic_dice_micro_t_F1`: float, 先汇总全部 calibration PDB 的 TP/FP/FN，再计算的 micro Dice。
            - `semantic_dice_macro_t_F1`: float, 每个 calibration PDB 分别计算 Dice 后的等权均值。
            - `semantic_tp_t_F1`: int, 全部 calibration voxel 在 `t_F1` 下的真阳性数。
            - `semantic_fp_t_F1`: int, 全部 calibration voxel 在 `t_F1` 下的假阳性数。
            - `semantic_fn_t_F1`: int, 全部 calibration voxel 在 `t_F1` 下的假阴性数。
            - `n_blob_exceed_pdb`: int, `t_F1` 下 eligible 预测组件数超过 200 的 PDB 数。
            - `n_pred_instances`: int, 全部 PDB 的 eligible 预测组件总数。
            - `n_gt_instances`: int, 全部 PDB 的真实 occurrence 总数。
            - `coverage_precision_{tag}`: float, `tag ∈ {0p3, 0p5}`，对应双向 coverage 阈值下的预测组件 precision。
            - `coverage_recall_{tag}`: float, `tag ∈ {0p3, 0p5}`，对应双向 coverage 阈值下的真实 occurrence recall。
            - `coverage_f1_{tag}`: float, `tag ∈ {0p3, 0p5}`，对应双向 coverage precision 与 recall 的 F1。
            - `one_to_one_precision_{tag}`: float, `tag ∈ {0p3, 0p5}`，固定 Hungarian 配对下的预测组件 precision。
            - `one_to_one_recall_{tag}`: float, `tag ∈ {0p3, 0p5}`，固定 Hungarian 配对下的真实 occurrence recall。
            - `one_to_one_f1_{tag}`: float, `tag ∈ {0p3, 0p5}`，固定 Hungarian 配对下的 one-to-one F1。
            - `n_topk_eligible_pdb`: int, 真实 occurrence 数大于 0、参与 top-K 汇总分母的 PDB 数。
            - `top{K}_success_{tag}`: int, `K ∈ {3, 4, 5}` 且 `tag ∈ {0p3, 0p5}`，前 K 个候选存在双向 coverage 达标配对的 PDB 数。
            - `top{K}_success_ratio_{tag}`: float, `K ∈ {3, 4, 5}` 且 `tag ∈ {0p3, 0p5}`，对应 top-K 成功数除以 `n_topk_eligible_pdb`。

    每张概率图只在需要时从已 `_COMPLETE` 的正式 NPZ 读取。第一遍累计 32769-bin 直方图，第二遍在冻结 `t_F1` 上临时构造单层 26-CCL；不发布 components，因而保持: “calibration probability 与阈值先完成，阶段二再回填各 role”的时序。
    """
    pdb_ids = tuple(str(pdb_id) for pdb_id in calibration_pdb_ids)
    if len(pdb_ids) == 0 or len(set(pdb_ids)) != len(pdb_ids):
        raise ValueError("calibration_pdb_ids 必须是非空且唯一的固定清单")

    def threshold_inputs() -> Iterable[tuple[np.ndarray, np.ndarray]]:
        """
        按固定 calibration PDB 顺序惰性读取 probability 与 union target。

        输出:
            - samples: Iterable[tuple[np.ndarray, np.ndarray]], 每项是同 shape 的 float32 完整 ZYX 概率图与 bool occurrence 配体区域并集。
        """
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
    """
    读取一个已发布 probability，并规范化该 PDB 的 occurrence 稀疏体素集。

    输入参数:
        - output_root: str | Path, `stage1_outputs` 根目录
        - stage1_model_name: str, 当前 producer 正式名
        - split: str, probability 所在 split, 例如 calibration、validation 或 train
        - pdb_id: str, 当前 PDB identity
        - occurrence_voxel_loader: OccurrenceVoxelLoader, 接收 PDB 与完整图 ZYX shape，返回真实配体 occurrence 到完整图 C-order 离散线性 voxel 索引的映射

    输出:
        - probability: float32, (D, H, W), 已发布且有限的完整 ZYX voxel 网格概率。
        - occurrences: dict[int, np.ndarray], 真实配体 occurrence_id 到去重升序的完整图 C-order 离散线性 voxel 索引映射；字典按 occurrence_id 排序。
    """
    paths = Stage1ArtifactPaths(
        output_root=Path(output_root),
        stage1_model_name=stage1_model_name,
        split=split,
        pdb_id=pdb_id,
    )
    if paths.running_dir.exists() or not is_role_complete(paths, "probability"):
        raise RuntimeError(f"calibration probability 尚不可消费: {pdb_id}")
    # float32, (D, H, W), 从已完成 `probability_map.npz` 冷读的完整 ZYX 概率图。
    probability = np.asarray(load_npz_strict(paths.probability_npz)["probability_map"], dtype=np.float32)
    if probability.ndim != 3 or not bool(np.all(np.isfinite(probability))):
        raise ValueError(f"{pdb_id}: probability_map 必须是有限 float32 三维网格")
    voxel_count = int(probability.size)
    raw_occurrences = occurrence_voxel_loader(
        pdb_id, tuple(int(value) for value in probability.shape)
    )
    # dict[int, int64 array], 逐 occurrence 校验并去重后的完整图 C-order voxel 索引，最后按 occurrence_id 排序。
    occurrences: dict[int, np.ndarray] = {}
    for occurrence_id, indices in raw_occurrences.items():
        identity = int(occurrence_id)
        values = np.asarray(indices, dtype=np.int64).reshape(-1)
        if values.size and (int(values.min()) < 0 or int(values.max()) >= voxel_count):
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
    """
    从 occurrence 稀疏线性 voxel indices 构造完整图 union ligand-area bool mask。

    输入参数:
        - full_shape_zyx: tuple[int, int, int], 完整图的 ZYX voxel 形状 `(D, H, W)`。
        - occurrence_voxels: Mapping[int, np.ndarray], 真实配体 occurrence_id 到完整图 C-order 离散线性 voxel 索引的映射。

    输出:
        - union_mask: bool, (D, H, W), 完整 ZYX voxel 网格上的真实 occurrence 配体区域并集。
    """
    # bool, (D*H*W,), 按完整 ZYX 网格 C-order 展平的 occurrence 并集，填充后恢复三维形状。
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
    """
    在冻结 `t_F1` 上(不是其它阈值！)汇总 calibration fitted voxel 与 instance 指标。

    输入参数:
        - output_root: str | Path, `stage1_outputs` 根目录
        - stage1_model_name: str, 当前 producer 正式名
        - split: str, calibration probability 所在 split
        - pdb_ids: Sequence[str], 固定 calibration PDB 顺序
        - occurrence_voxel_loader: OccurrenceVoxelLoader, 加载真实配体 occurrence 的完整图 C-order 离散线性 voxel 索引
        - result: ThresholdCalibrationResult, 已冻结的阈值与 TP/FP/FN 曲线
        - min_voxels: int, candidate 最小 voxel 数
        - max_voxels: int, candidate 最大 voxel 数

    输出(都是在 t_F1 阈值下):
        - metrics: dict[str, object], 冻结 `t_F1` 后汇总的 voxel、semantic、instance 和 top-K 指标。
            - `voxel_average_precision_macro`: float, 有效 PDB 的完整图 voxel AP 等权平均值。
            - `n_valid_voxel_ap_pdb`: int, 参与 voxel AP 等权平均的有效 PDB 数。
            - `n_total_pdb`: int, calibration PDB 总数。
            - `n_evaluated_pdb`: int, 未触发组件数上限、实际进入全部拟合指标的 PDB 数。
            - `semantic_dice_micro_t_F1`: float, 先汇总全部未超限 PDB 的 TP/FP/FN，再计算的 micro Dice。
            - `semantic_dice_macro_t_F1`: float, 每个未超限 PDB 分别计算 Dice 后的等权均值。
            - `semantic_tp_t_F1`: int, 全部未超限 calibration voxel 在 `t_F1` 下的真阳性数(micro)。
            - `semantic_fp_t_F1`: int, 全部未超限 calibration voxel 在 `t_F1` 下的假阳性数(micro)。
            - `semantic_fn_t_F1`: int, 全部未超限 calibration voxel 在 `t_F1` 下的假阴性数(micro)。
            - `n_blob_exceed_pdb`: int, `t_F1` 下 eligible 预测组件数超过 200、已从全部拟合指标排除的 PDB 数。
            - `n_pred_instances`: int, 全部未超限 PDB 的 eligible 预测组件总数。
            - `n_gt_instances`: int, 全部未超限 PDB 的真实 occurrence 总数。
            - `coverage_precision_{tag}`: float, `tag ∈ {0p3, 0p5}`，对应双向 coverage 阈值下的预测组件 precision。
            - `coverage_recall_{tag}`: float, `tag ∈ {0p3, 0p5}`，对应双向 coverage 阈值下的真实 occurrence recall。
            - `coverage_f1_{tag}`: float, `tag ∈ {0p3, 0p5}`，对应双向 coverage precision 与 recall 的 F1。
            - `one_to_one_precision_{tag}`: float, `tag ∈ {0p3, 0p5}`，固定 Hungarian 配对下的预测组件 precision。
            - `one_to_one_recall_{tag}`: float, `tag ∈ {0p3, 0p5}`，固定 Hungarian 配对下的真实 occurrence recall。
            - `one_to_one_f1_{tag}`: float, `tag ∈ {0p3, 0p5}`，固定 Hungarian 配对下的 one-to-one F1。
            - `n_topk_eligible_pdb`: int, 真实 occurrence 数大于 0、参与 top-K 汇总分母的 PDB 数。
            - `top{K}_success_{tag}`: int, `K ∈ {3, 4, 5}` 且 `tag ∈ {0p3, 0p5}`，前 K 个候选存在双向 coverage 达标配对的 PDB 数。
            - `top{K}_success_ratio_{tag}`: float, `K ∈ {3, 4, 5}` 且 `tag ∈ {0p3, 0p5}`，对应 top-K 成功数除以 `n_topk_eligible_pdb`。
    """
    # int, `result.alpha_values` 中唯一 `alpha=1` 的行号；用于读取与 alpha 第一维对齐的 F1 冻结阈值。
    alpha_one_row = int(np.flatnonzero(np.isclose(result.alpha_values, 1.0, rtol=0.0, atol=1e-12))[0])
    # int, F1 冻结阈值的网格整数 `j`；对应概率 `t_F1=j/denominator`，并索引全局 TP/FP/FN 曲线。
    f1_grid_index = int(result.alpha_threshold_grid_index[alpha_one_row])
    # float, 与整数直方图 TP/FP/FN 完全一致的 F1 二值化阈值；避免借由 float32 持久化值改变临界点归属。
    f1_threshold = float(f1_grid_index) / float(result.denominator)
    # list[float], 仅保存至少含一个真实正 voxel 的 PDB AP，用于等权 macro 平均。
    ap_values: list[float] = []
    # list[float], 未超限 PDB 的单 PDB semantic Dice；包括分母为 0 且按契约记为 0.0 的 PDB。
    semantic_dice_values: list[float] = []
    # list[InstanceCounts], 未超限 PDB 的实例计数(见 src\evaluation\instance_metrics.py)，循环后按分子和分母求和。
    instance_counts = []
    # dict[str, int], 跨 PDB 累加的 top-K eligible 分母和各 K/coverage 成功计数。
    topk_totals: dict[str, int] = {}
    # int, 在冻结 `t_F1` 下产生超过 200 个 eligible 预测组件的 PDB 数。
    blob_exceed_count = 0
    # 三个 int, 只累计没有触发 `_BLOB_EXCEED` 的 PDB，用于冻结阈值后的 semantic micro Dice。
    semantic_tp = 0
    semantic_fp = 0
    semantic_fn = 0

    # 第二遍按固定顺序逐 PDB 重读预测概率与 GT occurrence，在冻结 `t_F1` 上计算拟合后指标；本循环不发布组件产物。
    for pdb_id in pdb_ids:
        # tuple[np.ndarray, dict[int, np.ndarray]], 同一完整 ZYX 网格上的模型预测概率与真实配体 occurrence 线性体素索引映射。
        probability, occurrence_voxels = _load_published_calibration_sample(
            output_root=output_root,
            stage1_model_name=stage1_model_name,
            split=split,
            pdb_id=pdb_id,
            occurrence_voxel_loader=occurrence_voxel_loader,
        )
        # ComponentForest, 仅在冻结 `t_F1` 层从预测概率构建的 26 邻域组件森林；节点同时记录体积与 80³ BOX eligibility。
        forest, _ = build_component_forest(
            probability_map=probability,
            threshold_grid_indices=(f1_grid_index,),
            denominator=int(result.denominator),
            min_voxels=int(min_voxels),
            max_voxels=int(max_voxels),
            resolve_box_start=centered_start_from_centroid_zyx,
        )
        # list[ComponentNode], 当前 PDB 在冻结 `t_F1` 层满足体积和 80³ BOX 规则的预测组件。
        predictions = [
            node
            for node in forest.nodes
            if node.candidate_eligible
            and node.threshold_grid_index == f1_grid_index
        ]
        # `_BLOB_EXCEED` PDB 不属于可消费的正式样本，因此不得进入任何冻结阈值后的拟合评估指标。
        if len(predictions) > 200:
            blob_exceed_count += 1
            continue
        # bool, (D,H,W), 当前 PDB 全部真实配体 occurrence 的体素并集；与 `probability` 逐体素对齐。
        target = _union_mask(probability.shape, occurrence_voxels)
        # float | None, 当前完整图的连续概率 AP；没有任何真实正体素时为 None，不进入 macro 平均。
        ap = average_precision_full_grid(probability, target)
        if ap is not None:
            ap_values.append(float(ap))
        # dict[str, float | int], 当前 PDB 的 semantic Dice 与 TP/FP/FN；只累计未超限 PDB。
        semantic = semantic_dice(probability, target, f1_threshold)
        semantic_dice_values.append(float(semantic["dice"]))
        semantic_tp += int(semantic["tp"])
        semantic_fp += int(semantic["fp"])
        semantic_fn += int(semantic["fn"])
        # tuple[int64 array, ...], 按 occurrence_id 排序的真实配体 voxel 集，与交集矩阵第二维对齐。
        occurrence_rows = tuple(occurrence_voxels.values())
        # int64, (N_pred, N_gt), 每个 eligible `t_F1` 组件与真实 occurrence 的交集 voxel 数。
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
        # int64, (N_pred,), 每个 eligible `t_F1` 组件的 voxel 数。
        pred_sizes = np.asarray(
            [node.voxel_global_linear_index.size for node in predictions],
            dtype=np.int64,
        )
        # int64, (N_gt,), 每个真实 occurrence 的 voxel 数，与 `occurrence_rows` 对齐。
        gt_sizes = np.asarray(
            [indices.size for indices in occurrence_rows], dtype=np.int64
        )
        # InstanceCounts, 当前 PDB 在 0.3/0.5 双向 coverage 下的多对多命中数与固定 Hungarian 一对一命中数；循环后按计数做 global/micro 聚合。
        instance_counts.append(
            evaluate_instance_overlap_counts(
                intersections=intersections,
                pred_sizes=pred_sizes,
                gt_sizes=gt_sizes,
                coverage_thresholds=(0.3, 0.5),
            )
        )
        # dict[str, int], 当前 PDB 的 top-3/4/5 评估资格与 0/1 成功标志；预测组件按 `probability_mean` 稳定降序排名，不执行 Hungarian 配对。
        topk = evaluate_topk_overlap_counts(
            intersections=intersections,
            pred_sizes=pred_sizes,
            gt_sizes=gt_sizes,
            candidate_scores=[node.probability_mean for node in predictions],
            topk_values=(3, 4, 5),
            coverage_thresholds=(0.3, 0.5),
        )
        # 把当前 PDB 的 top-K 资格和成功标志逐字段相加，得到跨 PDB 的分母与成功次数。
        for field, value in topk.items():
            topk_totals[field] = topk_totals.get(field, 0) + int(value)

    if not instance_counts:
        raise ValueError("全部 calibration PDB 均触发 `_BLOB_EXCEED`，没有可评估样本")
    # InstanceCounts, 将全部未超限 PDB 的预测数、GT 数和各阈值命中数求和，供 global/micro precision、recall 与 F1 计算。
    total_instance = aggregate_instance_counts(instance_counts)
    # int, 未超限 PDB 的 semantic Dice 分母 `2*TP+FP+FN`；全空时最终 Dice 定义为 0.0。
    dice_denominator = 2 * semantic_tp + semantic_fp + semantic_fn
    # int, 至少含一个真实 occurrence 的 PDB 数；从累计字典移出后作为全部 top-K 成功比例的共同分母。
    topk_denominator = int(topk_totals.pop("n_topk_eligible_pdb", 0))
    # dict[str, object], 将包含 top-K eligible PDB 分母、成功计数及对应成功比例。
    topk_metrics: dict[str, object] = {"n_topk_eligible_pdb": topk_denominator, }
    # 每个累计成功字段同时保留整数成功次数，并生成同 K、同 coverage 阈值的成功比例字段。
    for field, value in topk_totals.items():
        topk_metrics[field] = int(value)
        topk_metrics[field.replace("_success_", "_success_ratio_")] = (float(value) / float(topk_denominator) if topk_denominator else 0.0)
    # dict[str, object], 合并 voxel macro AP、semantic micro/macro Dice、instance global/micro 指标和 top-K 跨 PDB 指标。
    return {
        "voxel_average_precision_macro": (float(np.mean(ap_values)) if ap_values else float("nan")),
        "n_valid_voxel_ap_pdb": len(ap_values),
        "n_total_pdb": len(pdb_ids),
        "n_evaluated_pdb": len(instance_counts),
        "semantic_dice_micro_t_F1": (float(2 * semantic_tp) / float(dice_denominator) if dice_denominator else 0.0),
        "semantic_dice_macro_t_F1": (
            float(np.mean(semantic_dice_values)) if semantic_dice_values else 0.0
        ),
        "semantic_tp_t_F1": semantic_tp,
        "semantic_fp_t_F1": semantic_fp,
        "semantic_fn_t_F1": semantic_fn,
        "n_blob_exceed_pdb": int(blob_exceed_count),
        **total_instance.metrics(),
        **topk_metrics,
    }
