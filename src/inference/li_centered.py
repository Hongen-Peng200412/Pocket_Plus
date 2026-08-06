"""从完整图概率计算逐图 Li 阈值，并构造不落盘 forest 的居中候选。"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.component_lineage import ComponentNode
from src.component_lineage.forest import build_component_forest
from src.artifacts import Stage1ArtifactPaths, atomic_savez_compressed, mark_role_complete
from src.artifacts.io import pack_centered_entries, validate_centered_archive


def li_threshold(probability_map: np.ndarray, tolerance: float = 1e-5) -> float:
    """用 Li 最小交叉熵迭代返回一张有限三维概率图的 map-specific 阈值。"""

    probability = np.asarray(probability_map, dtype=np.float64)
    if probability.ndim != 3 or not bool(np.all(np.isfinite(probability))):
        raise ValueError("Li threshold 要求有限三维 probability_map")
    if bool(np.any((probability < 0.0) | (probability > 1.0))):
        raise ValueError("Li threshold 要求 probability_map 位于 [0,1]")
    values = probability.reshape(-1)
    minimum = float(values.min())
    maximum = float(values.max())
    if minimum == maximum:
        return minimum

    threshold = float(values.mean())
    tiny = np.finfo(np.float64).tiny
    for _ in range(100):
        background = values[values <= threshold]
        foreground = values[values > threshold]
        if background.size == 0 or foreground.size == 0:
            break
        mean_background = max(float(background.mean()), tiny)
        mean_foreground = max(float(foreground.mean()), tiny)
        denominator = np.log(mean_background) - np.log(mean_foreground)
        if denominator == 0.0:
            break
        updated = (mean_background - mean_foreground) / denominator
        updated = float(np.clip(updated, minimum, maximum))
        if abs(updated - threshold) <= float(tolerance):
            threshold = updated
            break
        threshold = updated
    return threshold


def quantize_li_threshold(threshold: float, denominator: int) -> tuple[int, float]:
    """把 Li 浮点阈值向上量化到整数网格，避免纳入低于原值的体素。"""

    if int(denominator) <= 0:
        raise ValueError("denominator 必须为正整数")
    value = float(threshold)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("Li threshold 必须是 [0,1] 内的有限值")
    grid_index = int(np.clip(np.ceil(value * int(denominator)), 0, int(denominator)))
    return grid_index, float(grid_index) / float(denominator)


def build_li_nodes(
    probability_map: np.ndarray,
    denominator: int,
    min_voxels: int,
    max_voxels: int,
    resolve_box_start: Callable[[Sequence[float], Sequence[int]], Sequence[int]],
) -> tuple[list[ComponentNode], float, int, float]:
    """返回 Li 单阈值 eligible blob、原始阈值、网格编号和实际应用阈值。"""

    raw_threshold = li_threshold(probability_map)
    grid_index, applied_threshold = quantize_li_threshold(raw_threshold, denominator)
    forest, _ = build_component_forest(
        probability_map=np.asarray(probability_map, dtype=np.float32),
        threshold_grid_indices=(grid_index,),
        denominator=int(denominator),
        min_voxels=int(min_voxels),
        max_voxels=int(max_voxels),
        resolve_box_start=resolve_box_start,
    )
    nodes = [node for node in forest.nodes if node.candidate_eligible]
    nodes.sort(key=lambda node: (-node.probability_mean, node.tree_id, node.node_id))
    return nodes, raw_threshold, grid_index, applied_threshold


def publish_li_centered_entries(
    paths: Stage1ArtifactPaths,
    entries: Sequence[Mapping[str, Any]],
    raw_threshold: float,
    grid_index: int,
    applied_threshold: float,
    denominator: int,
) -> None:
    """原子发布一张图的 `Li_centered.npz` 与完成标记。"""

    metadata = {
        "li_threshold_raw": float(raw_threshold),
        "li_threshold_grid_index": int(grid_index),
        "li_threshold_applied": float(applied_threshold),
        "threshold_denominator": int(denominator),
    }
    arrays = pack_centered_entries(
        entries,
        "Li_centered",
        stage1_model_name=paths.stage1_model_name,
        role_metadata=metadata,
    )
    atomic_savez_compressed(
        paths.centered_npz("Li_centered"),
        arrays,
        validator=lambda value: validate_centered_archive(
            value, "Li_centered", stage1_model_name=paths.stage1_model_name
        ),
    )
    mark_role_complete(paths, "Li_centered")


__all__ = [
    "build_li_nodes",
    "li_threshold",
    "publish_li_centered_entries",
    "quantize_li_threshold",
]
