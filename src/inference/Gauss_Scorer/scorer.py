"""根据 F1-centered A 原子结合概率给来源组件节点计算高斯联合分数。

本模块不重新运行模型，也不改变组件资格、CLG 或 Selector 候选。它只为
`components/forest.npz` 增加逐节点 `gauss_score` 与 `gauss_selected` 两个字段。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.spatial import cKDTree

from src.artifacts import atomic_savez_compressed, load_npz_strict
from src.artifacts.io import validate_centered_archive
from src.component_lineage import ComponentForest


@dataclass(frozen=True)
class GaussScorerParameters:
    """保存高斯联合打分的四个待校准正参数与固定空间截断。"""

    lambda_positive: float
    lambda_negative: float
    tau_angstrom: float
    gauss_score_min: float
    distance_cutoff_angstrom: float = 5.0

    def __post_init__(self) -> None:
        """拒绝非有限值、非正调参值和非正截断距离。"""

        positive_values = {
            "lambda_positive": self.lambda_positive,
            "lambda_negative": self.lambda_negative,
            "tau_angstrom": self.tau_angstrom,
            "gauss_score_min": self.gauss_score_min,
            "distance_cutoff_angstrom": self.distance_cutoff_angstrom,
        }
        invalid = [
            name
            for name, value in positive_values.items()
            if not np.isfinite(float(value)) or float(value) <= 0.0
        ]
        if invalid:
            raise ValueError(f"Gauss scorer 参数必须是有限正数: {invalid}")


def _node_row_index(forest_arrays: Mapping[str, np.ndarray]) -> dict[tuple[int, int], int]:
    """建立 `(tree_id,node_id)` 到 forest 主表行号的一一映射。"""

    keys = zip(
        np.asarray(forest_arrays["tree_id"], dtype=np.int32).tolist(),
        np.asarray(forest_arrays["node_id"], dtype=np.int32).tolist(),
        strict=True,
    )
    result = {tuple(map(int, key)): row for row, key in enumerate(keys)}
    if len(result) != int(np.asarray(forest_arrays["tree_id"]).size):
        raise ValueError("forest 主表包含重复的 (tree_id,node_id)")
    return result


def _gauss_terms_for_one_entry(
    *,
    forest_arrays: Mapping[str, np.ndarray],
    forest_row: int,
    centered_arrays: Mapping[str, np.ndarray],
    entry_row: int,
    full_shape_zyx: tuple[int, int, int],
    tau_angstrom: float,
    distance_cutoff_angstrom: float,
) -> tuple[float, float]:
    """计算一个 F1 来源节点的正负 A 原子高斯加权和。"""

    voxel_offsets = np.asarray(forest_arrays["node_voxel_offsets"], dtype=np.int64)
    voxel_values = np.asarray(
        forest_arrays["node_voxel_global_linear_index"], dtype=np.int64
    )
    blob_linear = voxel_values[
        int(voxel_offsets[forest_row]) : int(voxel_offsets[forest_row + 1])
    ]
    if blob_linear.size == 0:
        raise ValueError("F1 来源节点不得是空组件")

    box_start_zyx = np.asarray(centered_arrays["box_start_zyx"], dtype=np.int64)[entry_row]
    voxel_size_xyz = np.asarray(centered_arrays["voxel_size_world"], dtype=np.float64)[entry_row]
    if not bool(np.all(np.isfinite(voxel_size_xyz))) or not bool(np.all(voxel_size_xyz > 0.0)):
        raise ValueError("F1-centered voxel_size_world 必须是有限正数")

    blob_global_zyx = np.column_stack(np.unravel_index(blob_linear, full_shape_zyx))
    blob_local_zyx = blob_global_zyx - box_start_zyx[None, :]
    blob_center_local_world = (
        blob_local_zyx[:, [2, 1, 0]].astype(np.float64) + 0.5
    ) * voxel_size_xyz[None, :]

    atom_offsets = np.asarray(centered_arrays["A_offsets"], dtype=np.int64)
    atom_slice = slice(int(atom_offsets[entry_row]), int(atom_offsets[entry_row + 1]))
    atom_local_xyz = np.asarray(
        centered_arrays["A_coord_local_xyz"], dtype=np.float64
    )[atom_slice]
    atom_probability = np.asarray(
        centered_arrays["A_probability"], dtype=np.float64
    )[atom_slice]
    if atom_probability.size == 0:
        return 0.0, 0.0
    if not bool(np.all(np.isfinite(atom_local_xyz))):
        raise ValueError("A_coord_local_xyz 包含非有限值")
    if not bool(np.all(np.isfinite(atom_probability))) or bool(
        np.any((atom_probability < 0.0) | (atom_probability > 1.0))
    ):
        raise ValueError("A_probability 必须是 [0,1] 内的有限值")

    atom_local_world = atom_local_xyz * voxel_size_xyz[None, :]
    nearest_distance, _ = cKDTree(blob_center_local_world).query(
        atom_local_world,
        k=1,
        distance_upper_bound=float(distance_cutoff_angstrom),
    )
    within = np.isfinite(nearest_distance)
    weights = np.zeros(atom_probability.shape, dtype=np.float64)
    weights[within] = np.exp(
        -(nearest_distance[within] ** 2) / (2.0 * float(tau_angstrom) ** 2)
    )
    positive_sum = float(np.sum(weights * atom_probability, dtype=np.float64))
    negative_sum = float(np.sum(weights * (1.0 - atom_probability), dtype=np.float64))
    return positive_sum, negative_sum


def compute_f1_centered_gauss_terms(
    *,
    forest_arrays: Mapping[str, np.ndarray],
    centered_arrays: Mapping[str, np.ndarray],
    full_shape_zyx: tuple[int, int, int],
    tau_angstrom: float,
    distance_cutoff_angstrom: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """按 forest 主表顺序返回正、负 A 原子高斯加权和。

    非 F1 来源节点的两项均为 NaN。该入口让 calibration 参数扫描按每个 tau
    只计算一次空间距离，随后可低成本组合正负权重与分数阈值。
    """

    for name, value in (
        ("tau_angstrom", tau_angstrom),
        ("distance_cutoff_angstrom", distance_cutoff_angstrom),
    ):
        if not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} 必须是有限正数")

    ComponentForest.from_arrays(forest_arrays)
    validate_centered_archive(centered_arrays, "F1_centered", stage1_model_name="Find_0")
    shape = tuple(int(value) for value in full_shape_zyx)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError("full_shape_zyx 必须包含三个正整数")

    n_node = int(np.asarray(forest_arrays["tree_id"]).size)
    positive_terms = np.full(n_node, np.nan, dtype=np.float64)
    negative_terms = np.full(n_node, np.nan, dtype=np.float64)
    row_by_key = _node_row_index(forest_arrays)
    source_tree_ids = np.asarray(centered_arrays["source_tree_id"], dtype=np.int32)
    source_node_ids = np.asarray(centered_arrays["source_node_id"], dtype=np.int32)
    source_grid_indices = np.asarray(
        centered_arrays["source_threshold_grid_index"], dtype=np.int32
    )
    source_thresholds = np.asarray(
        centered_arrays["source_threshold_value"], dtype=np.float32
    )
    seen: set[tuple[int, int]] = set()

    for entry_row, (tree_id, node_id) in enumerate(
        zip(source_tree_ids.tolist(), source_node_ids.tolist(), strict=True)
    ):
        key = (int(tree_id), int(node_id))
        if key in seen:
            raise ValueError(f"F1-centered 来源节点重复: {key}")
        seen.add(key)
        if key not in row_by_key:
            raise KeyError(f"F1-centered 来源节点不在 forest: {key}")
        forest_row = row_by_key[key]
        if not bool(np.asarray(forest_arrays["candidate_eligible"])[forest_row]):
            raise ValueError(f"F1-centered 来源节点不满足 candidate_eligible: {key}")
        if int(np.asarray(forest_arrays["threshold_grid_index"])[forest_row]) != int(
            source_grid_indices[entry_row]
        ):
            raise ValueError(f"F1-centered 来源节点阈值网格身份不一致: {key}")
        if not np.isclose(
            float(np.asarray(forest_arrays["threshold_value"])[forest_row]),
            float(source_thresholds[entry_row]),
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError(f"F1-centered 来源节点阈值数值不一致: {key}")

        positive_sum, negative_sum = _gauss_terms_for_one_entry(
            forest_arrays=forest_arrays,
            forest_row=forest_row,
            centered_arrays=centered_arrays,
            entry_row=entry_row,
            full_shape_zyx=shape,
            tau_angstrom=float(tau_angstrom),
            distance_cutoff_angstrom=float(distance_cutoff_angstrom),
        )
        positive_terms[forest_row] = positive_sum
        negative_terms[forest_row] = negative_sum

    return positive_terms, negative_terms


def score_f1_centered_nodes(
    *,
    forest_arrays: Mapping[str, np.ndarray],
    centered_arrays: Mapping[str, np.ndarray],
    full_shape_zyx: tuple[int, int, int],
    parameters: GaussScorerParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """按 forest 主表顺序返回高斯分数和独立保留决定。"""

    positive_terms, negative_terms = compute_f1_centered_gauss_terms(
        forest_arrays=forest_arrays,
        centered_arrays=centered_arrays,
        full_shape_zyx=full_shape_zyx,
        tau_angstrom=parameters.tau_angstrom,
        distance_cutoff_angstrom=parameters.distance_cutoff_angstrom,
    )
    probability_mean = np.asarray(forest_arrays["probability_mean"], dtype=np.float64)
    finite = np.isfinite(positive_terms) & np.isfinite(negative_terms)
    scores = np.full(probability_mean.shape, np.nan, dtype=np.float32)
    combined = (
        probability_mean[finite]
        + float(parameters.lambda_positive) * positive_terms[finite]
        - float(parameters.lambda_negative) * negative_terms[finite]
    )
    if not bool(np.all(np.isfinite(combined))):
        raise ValueError("Gauss scorer 产生非有限分数")
    scores[finite] = combined.astype(np.float32)
    selected = finite & (scores >= np.float32(parameters.gauss_score_min))

    return scores, selected


def add_gauss_fields(
    forest_arrays: Mapping[str, np.ndarray],
    gauss_score: np.ndarray,
    gauss_selected: np.ndarray,
) -> dict[str, np.ndarray]:
    """保留 forest 全部原字段，并增加或幂等确认两个高斯字段。"""

    score = np.asarray(gauss_score, dtype=np.float32)
    selected = np.asarray(gauss_selected, dtype=np.bool_)
    updated = {name: np.asarray(value) for name, value in forest_arrays.items()}
    if "gauss_score" in updated or "gauss_selected" in updated:
        if "gauss_score" not in updated or "gauss_selected" not in updated:
            raise KeyError("forest.npz 的高斯字段只存在一项，拒绝覆盖")
        if not np.array_equal(updated["gauss_score"], score, equal_nan=True) or not np.array_equal(
            updated["gauss_selected"], selected
        ):
            raise FileExistsError("forest.npz 已包含不同的高斯打分结果，拒绝覆盖")
        ComponentForest.from_arrays(updated)
        return updated
    updated["gauss_score"] = score
    updated["gauss_selected"] = selected
    ComponentForest.from_arrays(updated)
    return updated


def publish_gauss_fields(
    forest_path: str | Path,
    gauss_score: np.ndarray,
    gauss_selected: np.ndarray,
) -> None:
    """原子回填一个现有 forest，并在正式替换前后执行完整结构校验。"""

    path = Path(forest_path)
    original = load_npz_strict(path)
    updated = add_gauss_fields(original, gauss_score, gauss_selected)
    atomic_savez_compressed(path, updated, validator=ComponentForest.from_arrays)
