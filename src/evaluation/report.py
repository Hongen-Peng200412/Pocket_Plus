"""把 calibration 数值结果转换为可复核 JSON 报告。"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.artifacts.io import atomic_write_json


def to_json_values(value: Any) -> Any:
    """
    递归把 NumPy 标量/数组转换为标准 JSON 值。

    输入参数:
        - value: Any, 指标 dict、list、NumPy 数组或标量

    输出:
        - converted: Any, 不含 ndarray/NumPy scalar 的等价结构
    """
    if isinstance(value, Mapping):
        return {str(key): to_json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_values(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_calibration_report(
    path: str,
    stage1_model_name: str,
    voxel_metrics: Mapping[str, Any],
    instance_metrics: Mapping[str, Any],
    threshold_summary: Mapping[str, Any],
) -> None:
    """
    原子写入明确标注 calibration-fitted 的 producer 指标报告。

    输入参数:
        - path: str, 正式 `metrics.json` 路径
        - stage1_model_name: str, 当前 producer 名
        - voxel_metrics: Mapping[str,Any], macro AP、Dice 等 voxel 指标
        - instance_metrics: Mapping[str,Any], coverage/one-to-one/top-K 指标
        - threshold_summary: Mapping[str,Any], 七个阈值及曲线引用/摘要

    输出:
        - None, 通过原子 JSON IO 发布报告
    """
    atomic_write_json(
        path,
        to_json_values(
            {
                "stage1_model_name": stage1_model_name,
                "result_scope": "calibration_fitted",
                "voxel_metrics": dict(voxel_metrics),
                "instance_metrics": dict(instance_metrics),
                "threshold_summary": dict(threshold_summary),
            }
        ),
    )
