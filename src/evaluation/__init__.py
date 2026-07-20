"""统一导出 AdaLigand Stage1 阈值冻结、voxel 评估和组件实例评估接口。

主要入口:
    - `calibrate_published_full_maps_and_freeze_thresholds`: 消费已发布 calibration 完整图，冻结七个 micro-Fα 阈值并发布 fitted 报告。
    - `average_precision_full_grid`、`semantic_dice`: 计算完整图 voxel 级连续与阈值指标。
    - `evaluate_instance_overlap_counts`、`evaluate_topk_overlap_counts`: 从组件与 occurrence 的交集计数计算实例指标。

本模块只汇总公共符号，不读取 probability、不构造 forest 也不写 calibration 文件。
"""

from .calibration import (
    DEFAULT_ALPHA_VALUES,
    ThresholdCalibrationResult,
    ThresholdHistogram,
    calibrate_published_full_maps_and_freeze_thresholds,
    calibrate_thresholds,
    publish_threshold_calibration,
)
from .instance_metrics import (
    DEFAULT_COVERAGE_THRESHOLDS,
    DEFAULT_TOPK_VALUES,
    InstanceCounts,
    aggregate_instance_counts,
    evaluate_instance_overlap_counts,
    evaluate_instance_masks,
    evaluate_topk_overlap_counts,
    evaluate_topk_success,
)
from .voxel_metrics import average_precision_full_grid, macro_average_precision, semantic_dice

__all__ = [
    "DEFAULT_ALPHA_VALUES",
    "DEFAULT_COVERAGE_THRESHOLDS",
    "DEFAULT_TOPK_VALUES",
    "InstanceCounts",
    "ThresholdCalibrationResult",
    "ThresholdHistogram",
    "aggregate_instance_counts",
    "average_precision_full_grid",
    "calibrate_published_full_maps_and_freeze_thresholds",
    "calibrate_thresholds",
    "evaluate_instance_overlap_counts",
    "evaluate_instance_masks",
    "evaluate_topk_overlap_counts",
    "evaluate_topk_success",
    "macro_average_precision",
    "publish_threshold_calibration",
    "semantic_dice",
]
