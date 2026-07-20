"""仅用于 calibration: 把 calibration 数值结果转换为标准 JSON 值并原子发布指标报告。

主要入口:
    - `to_json_values`: 递归转换 Mapping、list、tuple、NumPy 数组和 NumPy 标量。
    - `write_calibration_report`: 发布带有 producer 身份和 `calibration_fitted` 范围标记的 `metrics.json`。

本模块不计算任何指标，也不改变字段名称或数值；它只完成 JSON 类型转换和原子写入。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.artifacts.io import atomic_write_json


def to_json_values(value: Any) -> Any:
    """
    递归把 NumPy 标量/数组转换为标准 JSON 值。

    输入参数:
        - value: Any, 指标 Mapping、list、tuple、NumPy 数组、NumPy 标量或标准 JSON 值；嵌套结构递归处理。

    输出:
        - converted: Any, 不含 ndarray 或 NumPy scalar 的等价 JSON 可编码结构；Mapping key 统一转为字符串。
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
        - path: str, 正式 `metrics.json` 路径。
        - stage1_model_name: str, 当前 calibration fitted 指标所属的 producer。
        - voxel_metrics: Mapping[str, Any], macro AP、Dice、TP、FP、FN 与有效 PDB 计数。
        - instance_metrics: Mapping[str, Any], coverage、one-to-one、top-K 成功计数和比例。
        - threshold_summary: Mapping[str, Any], 七个冻结阈值、网格下标及扫描曲线文件引用或摘要。

    落盘产物:
        - `path`: JSON，顶层包含 `stage1_model_name/result_scope/voxel_metrics/instance_metrics/threshold_summary`；`result_scope` 固定为 `calibration_fitted`。
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
